"""Pipeline orchestrator - coordinates all stages.

Stage order (single image):
    perception -> world state -> flat retrieval -> ranking -> planning
    -> context -> generation -> evaluation

For multi-image and collection runs the perception stage is identical; the
collection path additionally builds hierarchical collection memory and uses
`HierarchicalRetriever` + `CollectionContextBuilder` instead of the flat
retriever and `ContextBuilder`.
"""
import time
import uuid

from PIL import Image

from ..domain.schemas import (
    PipelineArtifacts,
    PipelineConfig,
    RetrievedEvidence,
    VisualObservations,
    WorldState,
)
from ..evaluation.evaluator import ComprehensiveEvaluator
from ..experiments.manifests import build_manifest
from ..generation.base import StoryGenerationPipeline, StoryGenerator, create_generator
from ..memory.collection import CollectionMemoryBuilder, create_collection_memory_builder
from ..memory.embeddings import EmbeddingModelInterface, create_embedding_model
from ..memory.faiss_store import FAISSVectorStore
from ..memory.hierarchical import CollectionMemory
from ..narrative.planner import CreativePlanner
from ..observability.logging import get_logger
from ..observability.timing import StageTimer
from ..retrieval.collection_context import (
    CollectionContextBuilder,
    create_collection_context_builder,
)
from ..retrieval.compression import ContextBuilder
from ..retrieval.flat import EvidenceRetriever
from ..retrieval.hierarchical import HierarchicalRetriever, create_hierarchical_retriever
from ..retrieval.ranking import EvidenceRanker
from ..retrieval.sequence import SequenceContextBuilder
from ..verification.name_guard import guard_invented_names
from ..verification.visual import VisualVerifier
from ..vision.florence import Florence2Model
from ..vision.grounding import GroundingDINOModel, create_grounding_dino
from ..vision.ocr import OCRModel, create_ocr
from ..world.world_state import WorldStateBuilder


class PipelineOrchestrator:
    """Coordinates every pipeline stage for one run.

    Components are created lazily by `_initialize_components()` so that loading a
    mode that does not use a model never pays for it. They are declared as
    annotations without values: the attribute does not exist until initialization,
    and `_ensure_ready()` guarantees it exists before any run method touches it.
    """

    # Declared here, assigned in _initialize_components. The optional models
    # (`_grounding`, `_ocr`) are declared in __init__ instead.
    _florence: Florence2Model
    _embedding_model: EmbeddingModelInterface
    _vector_store: FAISSVectorStore
    _retriever: EvidenceRetriever
    _world_builder: WorldStateBuilder
    _ranker: EvidenceRanker
    _context_builder: ContextBuilder
    _sequence_builder: SequenceContextBuilder
    _collection_memory_builder: CollectionMemoryBuilder
    _hierarchical_retriever: HierarchicalRetriever
    _collection_context_builder: CollectionContextBuilder
    _creative_planner: CreativePlanner
    _generator: StoryGenerator
    _gen_pipeline: StoryGenerationPipeline
    _evaluator: ComprehensiveEvaluator
    _verifier: VisualVerifier

    def __init__(self, config: PipelineConfig):
        self._config = config
        self._runtime: dict[str, float] = {}
        self._logs: list[str] = []
        self._timer = StageTimer()
        self.logger = get_logger("image_story.pipeline")
        self._initialized = False

        # Optional models: absent unless the config enables them. Declared here so
        # the attribute always exists; the rest are created in _initialize_components.
        self._grounding: GroundingDINOModel | None = None
        self._ocr: OCRModel | None = None

        # Collection memory from the most recent run.
        self._collection_memory: CollectionMemory | None = None

    def _log(self, message: str) -> None:
        entry = f"[{time.strftime('%H:%M:%S')}] {message}"
        self._logs.append(entry)
        self.logger.info(message)

    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        device = self._config.device

        # Vision models
        self._florence = Florence2Model(device=device)
        self._florence.load()

        if self._config.use_grounding_dino:
            self._grounding = create_grounding_dino(device=device, enabled=True)
            self._grounding.load()

        if self._config.use_ocr:
            self._ocr = create_ocr(device=device, enabled=True)
            self._ocr.load()

        # Memory
        self._embedding_model = create_embedding_model(device=device)
        self._embedding_model.load()
        self._vector_store = FAISSVectorStore(self._embedding_model, device=device)
        self._retriever = EvidenceRetriever(self._embedding_model, self._vector_store, device=device)

        # Hierarchical collection memory
        self._collection_memory_builder = create_collection_memory_builder(
            device=device,
            similarity_threshold=self._config.scene_similarity_threshold,
            top_k_scenes=self._config.top_k_scenes,
            top_k_evidence_per_scene=self._config.top_k_evidence_per_scene,
        )
        self._hierarchical_retriever = create_hierarchical_retriever(
            self._embedding_model,
            self._vector_store,
            top_k_scenes=self._config.top_k_scenes,
            top_k_evidence_per_scene=self._config.top_k_evidence_per_scene,
            max_total_evidence=self._config.faiss_top_k * 5,
        )
        # Retrieval thresholds are a property of the retriever's own config
        # (HierarchicalRetrievalConfig), so they are injected here rather than
        # passed per-call.
        self._hierarchical_retriever.config.scene_similarity_threshold = (
            self._config.scene_retrieval_threshold
        )
        self._hierarchical_retriever.config.evidence_similarity_threshold = (
            self._config.evidence_retrieval_threshold
        )
        self._collection_context_builder = create_collection_context_builder(
            max_words=self._config.context_max_words,
            max_scenes=self._config.max_scenes_in_context,
        )

        # Context
        self._world_builder = WorldStateBuilder()
        self._ranker = EvidenceRanker()
        self._context_builder = ContextBuilder(max_words=self._config.context_max_words)
        self._sequence_builder = SequenceContextBuilder(max_total_words=self._config.context_max_words)

        # Narrative
        self._creative_planner = CreativePlanner(
            seed=self._config.seed,
            creativity_config=self._config.creativity_config,
            genre=self._config.genre,
            tone=self._config.tone,
        )

        # Generation
        self._generator = create_generator("qwen", device=device)
        self._gen_pipeline = StoryGenerationPipeline(self._generator, self._config)

        # Verification
        self._verifier = VisualVerifier(
            florence_model=self._florence,
            grounding_model=self._grounding,
            device=device,
        )

        # Evaluation
        self._evaluator = ComprehensiveEvaluator(device=device)

    # --- shared stage helpers -------------------------------------------------

    def _ensure_ready(self) -> None:
        if not self._initialized:
            self._initialize_components()

    def _new_run(self, image_paths: list[str]) -> PipelineArtifacts:
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        manifest = build_manifest(
            self._config,
            run_id=run_id,
            image_paths=image_paths,
            vision_model="florence-community/Florence-2-base",
            language_model="Qwen/Qwen2.5-0.5B-Instruct",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        return PipelineArtifacts(run_id=run_id, manifest=manifest)

    def _perceive(self, image: Image.Image, frame_id: int) -> VisualObservations:
        """Run Florence-2, then GroundingDINO and OCR when enabled.

        The single perception path for all three entry points.
        """
        florence, grounding, ocr = self._florence, self._grounding, self._ocr
        observations = florence.analyze(image, frame_id=frame_id)

        if grounding and grounding.enabled:
            # Detected labels become the open-vocabulary grounding prompts.
            phrases = observations.od_labels[:10] + observations.characters[:5]
            if phrases:
                grounding_evidence = grounding.detect_with_phrases(
                    image, phrases, frame_id=frame_id
                )
                observations.evidence_records.extend(grounding_evidence)
                observations.grounding_detections = [e.to_dict() for e in grounding_evidence]

        if ocr and ocr.enabled:
            ocr_evidence = ocr.extract_text(image, frame_id=frame_id)
            observations.evidence_records.extend(ocr_evidence)
            if ocr_evidence:
                observations.ocr_text = " ".join(
                    e.evidence_text.replace("OCR text: ", "") for e in ocr_evidence
                )

        return observations

    @staticmethod
    def _build_query(observations: list[VisualObservations]) -> str:
        """Retrieval query derived from scenes, characters and detected labels."""
        return " ".join(
            [obs.scene for obs in observations if obs.scene]
            + [" ".join(obs.characters) for obs in observations]
            + [" ".join(obs.od_labels) for obs in observations]
        )

    def _entity_recurrence(self, world_state: WorldState) -> dict[str, int]:
        return {
            entity.label: len(entity.frames_present)
            for entity in world_state.get_all_entities()
            if entity.is_recurring
        }

    def _generate(self, artifacts: PipelineArtifacts, prompt: str) -> None:
        start = time.perf_counter()
        try:
            artifacts.story_draft = self._gen_pipeline.generate_story(
                prompt, artifacts.story_plan
            )
        finally:
            self._timer.record("generation_s", time.perf_counter() - start)
        artifacts.story_draft.generation_time_s = self._timer.stages["generation_s"]
        self._guard_names(artifacts)

    def _guard_names(self, artifacts: PipelineArtifacts) -> None:
        """Replace names the evidence context does not support."""
        draft = artifacts.story_draft
        if not draft or not draft.text or not artifacts.context:
            return
        text, removed = guard_invented_names(draft.text, artifacts.context)
        if removed:
            self.logger.info("Name guard replaced %d unsupported name(s): %s", len(removed), ", ".join(sorted(set(removed))))
            draft.text = text
            draft.word_count = len(text.split())

    def _finalize(
        self,
        artifacts: PipelineArtifacts,
        image: Image.Image | None,
        evaluate: bool,
        frames: list[Image.Image] | None = None,
    ) -> PipelineArtifacts:
        if evaluate:
            with self._timer.stage("evaluation_s"):
                if frames is None:
                    artifacts.evaluation = self._evaluator.evaluate(artifacts, image)
                else:
                    artifacts.evaluation = self._evaluator.evaluate(artifacts, image, frames=frames)

        artifacts.runtime = self._timer.to_dict()
        artifacts.logs = self._logs
        return artifacts

    # --- public entry points -------------------------------------------------

    def run_single_image(self, image_path: str, evaluate: bool = True) -> PipelineArtifacts:
        """Run the pipeline on one image."""
        artifacts = self._new_run([image_path])
        self._ensure_ready()
        image = Image.open(image_path).convert("RGB")

        # Stage 1: Visual perception
        with self._timer.stage("vision_s"):
            observations = self._perceive(image, 0)
        artifacts.observations = [observations]

        # Stage 2: World state
        with self._timer.stage("world_state_s"):
            world_state = self._world_builder.add_observations(observations)
        artifacts.world_state = world_state

        # Stage 3: Flat memory + retrieval
        query = self._build_query([observations])
        with self._timer.stage("retrieval_s"):
            if self._config.use_faiss:
                self._retriever.initialize()
                self._retriever.index_observations([observations])
                artifacts.retrieved_evidence = self._retriever.retrieve(
                    query, top_k=self._config.faiss_top_k
                )

        # Stage 4: Ranking
        with self._timer.stage("ranking_s"):
            ranked = self._ranker.rank(
                artifacts.retrieved_evidence,
                query=query,
                current_frame=0,
                entity_recurrence=self._entity_recurrence(world_state),
            )
            artifacts.ranked_evidence = self._ranker.select_for_context(
                ranked, max_tokens=int(self._config.context_max_words * 1.3)
            )

        # Stage 5: Creative planning
        with self._timer.stage("planning_s"):
            creative_plan = self._creative_planner.create_creative_plan(
                world_state, [observations], artifacts.ranked_evidence
            )
            artifacts.creative_plan = creative_plan
            artifacts.story_plan = self._creative_planner.create_story_plan(
                creative_plan, self._config.target_story_words
            )

        # Stage 6: Context + prompt
        with self._timer.stage("context_s"):
            artifacts.context = self._context_builder.build_context(
                [observations], world_state, artifacts.ranked_evidence, creative_plan
            )
            prompt = self._context_builder.build_story_prompt(
                artifacts.context, creative_plan, self._config.target_story_words
            )

        # Stage 7: Generation
        self._generate(artifacts, prompt)

        return self._finalize(artifacts, image, evaluate)

    def run_multi_image(self, image_paths: list[str], evaluate: bool = True) -> PipelineArtifacts:
        """Run the pipeline over an ordered image sequence (flat retrieval path)."""
        artifacts = self._new_run(image_paths)
        self._ensure_ready()
        images = [Image.open(p).convert("RGB") for p in image_paths]

        # Stage 1: Visual perception (all frames)
        with self._timer.stage("vision_s"):
            all_observations = [
                self._perceive(image, frame_id)
                for frame_id, image in enumerate(images)
            ]
        artifacts.observations = all_observations

        # Stage 2: World state
        with self._timer.stage("world_state_s"):
            world_state = self._world_builder.add_observations_batch(all_observations)
        artifacts.world_state = world_state

        # Stage 3: Flat memory + retrieval
        query = self._build_query(all_observations)
        with self._timer.stage("retrieval_s"):
            if self._config.use_faiss:
                self._retriever.initialize()
                self._retriever.index_observations(all_observations)
                artifacts.retrieved_evidence = self._retriever.retrieve_for_story_planning(
                    query, len(all_observations) - 1, top_k=self._config.faiss_top_k
                )

        # Stage 4: Ranking
        with self._timer.stage("ranking_s"):
            ranked = self._ranker.rank_for_story_planning(
                artifacts.retrieved_evidence,
                query,
                len(all_observations) - 1,
                self._entity_recurrence(world_state),
            )
            artifacts.ranked_evidence = self._ranker.select_for_context(
                ranked, max_tokens=int(self._config.context_max_words * 1.3)
            )

        # Stage 5: Creative planning
        with self._timer.stage("planning_s"):
            creative_plan = self._creative_planner.create_creative_plan(
                world_state, all_observations, artifacts.ranked_evidence
            )
            artifacts.creative_plan = creative_plan
            artifacts.story_plan = self._creative_planner.create_story_plan(
                creative_plan, self._config.target_story_words
            )

        # Stage 6: Sequence context + prompt
        with self._timer.stage("context_s"):
            artifacts.context = self._sequence_builder.build_sequence_context(
                all_observations, world_state, artifacts.ranked_evidence, creative_plan
            )
            prompt = self._sequence_builder.build_multi_image_prompt(
                artifacts.context,
                len(all_observations),
                creative_plan,
                self._config.target_story_words,
            )

        # Stage 7: Generation
        self._generate(artifacts, prompt)

        # Evaluation: the first frame drives claim verification; CLIP grounding scores all frames.
        return self._finalize(artifacts, images[0] if images else None, evaluate, frames=images)

    def run_collection(self, image_paths: list[str], evaluate: bool = True) -> PipelineArtifacts:
        """Run the pipeline over a collection using hierarchical memory.

        Uses:
        - `CollectionMemoryBuilder` for scene/entity/transition/narrative memory
        - `HierarchicalRetriever` for scene-aware retrieval
        - `CollectionContextBuilder` for compressed multi-scene context
        """
        artifacts = self._new_run(image_paths)
        self._ensure_ready()
        images = [Image.open(p).convert("RGB") for p in image_paths]

        # Stage 1: Visual perception (all frames)
        with self._timer.stage("vision_s"):
            all_observations = [
                self._perceive(image, frame_id)
                for frame_id, image in enumerate(images)
            ]
        artifacts.observations = all_observations

        # Stage 2: Hierarchical collection memory
        with self._timer.stage("collection_memory_s"):
            self._collection_memory = self._collection_memory_builder.build_from_observations(
                all_observations, self._config
            )
            artifacts.collection_memory = self._collection_memory
            # CollectionMemoryBuilder populates its own internal retriever; the
            # orchestrator queries a separately-configured HierarchicalRetriever,
            # so the built memory must be injected into it explicitly. Without
            # this the collection path retrieves nothing (empty RetrievalResult
            # -> no grounded evidence -> ungrounded story).
            self._hierarchical_retriever.set_collection_memory(self._collection_memory)

        world_state = self._world_builder.add_observations_batch(all_observations)
        artifacts.world_state = world_state

        # Stage 3: Hierarchical retrieval
        query = self._build_query(all_observations)
        with self._timer.stage("retrieval_s"):
            retrieval_result = self._hierarchical_retriever.retrieve_full(
                query,
                top_k_scenes=self._config.top_k_scenes,
                top_k_evidence_per_scene=self._config.top_k_evidence_per_scene,
                include_transitions=True,
                include_narrative=True,
            )
            artifacts.retrieval_result = retrieval_result
            artifacts.retrieved_evidence = [
                RetrievedEvidence(record=e, semantic_similarity=1.0)
                for e in retrieval_result.evidence
            ]

        # Stage 4: Ranking
        with self._timer.stage("ranking_s"):
            ranked = self._ranker.rank_for_story_planning(
                artifacts.retrieved_evidence,
                query,
                len(all_observations) - 1,
                self._entity_recurrence(world_state),
            )
            artifacts.ranked_evidence = self._ranker.select_for_context(
                ranked, max_tokens=int(self._config.context_max_words * 1.3)
            )

        # Stage 5: Creative planning
        with self._timer.stage("planning_s"):
            creative_plan = self._creative_planner.create_creative_plan(
                world_state,
                all_observations,
                artifacts.ranked_evidence,
                collection_memory=self._collection_memory,
                retrieval_result=retrieval_result,
            )
            artifacts.creative_plan = creative_plan
            artifacts.story_plan = self._creative_planner.create_story_plan(
                creative_plan, self._config.target_story_words
            )

        # Stage 6: Collection context + prompt
        with self._timer.stage("context_s"):
            current_scene_id = None
            if self._collection_memory and self._collection_memory.scene_summaries:
                current_scene_id = self._collection_memory.scene_summaries[-1].scene_id

            artifacts.context = self._collection_context_builder.build_collection_context(
                self._collection_memory,
                retrieval_result,
                creative_plan,
                current_scene_id=current_scene_id,
            )
            prompt = self._collection_context_builder.build_multi_scene_prompt(
                artifacts.context, creative_plan, self._config.target_story_words
            )

        # Stage 7: Generation
        self._generate(artifacts, prompt)

        return self._finalize(artifacts, images[0] if images else None, evaluate, frames=images)

    def cleanup(self) -> None:
        """Unload every loaded model."""
        for model in [
            self._florence,
            self._grounding,
            self._ocr,
            self._embedding_model,
            self._generator,
        ]:
            if model and hasattr(model, "unload"):
                model.unload()


def create_orchestrator(config: PipelineConfig) -> PipelineOrchestrator:
    """Factory function to create the pipeline orchestrator."""
    return PipelineOrchestrator(config)
