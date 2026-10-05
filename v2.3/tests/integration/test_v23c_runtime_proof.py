"""
V2.3C RUNTIME PROOF: Real end-to-end execution of PipelineOrchestrator.run_collection().

These tests do NOT mock the orchestration or any V2.3B component. They spy on
real component instances (wrapping them so the real implementation still runs)
and stub ONLY the expensive ML model boundaries:

  - Florence2Model.analyze        (vision inference)
  - EmbeddingModelInterface.encode (sentence-transformer inference)
  - StoryGenerator.generate       (Qwen inference)
  - GroundingEvaluator internals  (CLIP / NLI inference)

Everything between those boundaries -- CollectionMemoryBuilder,
HierarchicalRetriever, CollectionContextBuilder, CreativePlanner,
StoryGenerationPipeline, ClaimExtractor, ClaimVerifier,
ComprehensiveEvaluator -- is the REAL implementation.
"""
import os
import sys
import tempfile
import numpy as np
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from PIL import Image

from image_story.domain.schemas import (
    PipelineConfig,
    VisualObservations,
    EvidenceRecord,
    EvidenceType,
    SourceModel,
    InformationClass,
    RetrievedEvidence,
    StoryClaim,
)
from image_story.domain.enums import ClaimStatus
from image_story.memory.hierarchical import (
    CollectionMemory,
    RetrievalResult,
    SceneSummary,
)
from image_story.memory.collection_builder import CollectionMemoryBuilder
from image_story.memory.hierarchical_retriever import HierarchicalRetriever
from image_story.memory.faiss_store import FAISSVectorStore
from image_story.context.collection_builder import CollectionContextBuilder
from image_story.context.world_state import WorldStateBuilder
from image_story.context.ranker import EvidenceRanker
from image_story.narrative.planner import CreativePlanner
from image_story.generation.base import StoryGenerationPipeline, StoryGenerator
from image_story.evaluation.claims import ClaimExtractor, ClaimVerifier
from image_story.evaluation.evaluator import ComprehensiveEvaluator
from image_story.pipeline.orchestrator import PipelineOrchestrator


# ---------------------------------------------------------------------------
# MODEL BOUNDARY STUBS (the only things mocked)
# ---------------------------------------------------------------------------

class StubEmbeddingModel:
    """Deterministic embedding stub. Replaces sentence-transformer only."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self.encoding_calls = 0

    @property
    def embedding_dim(self) -> int:
        return self._dimension

    @property
    def is_loaded(self) -> bool:
        return True

    def load(self):
        pass

    def unload(self):
        pass

    def encode_single(self, text: str) -> np.ndarray:
        self.encoding_calls += 1
        np.random.seed(abs(hash(text)) % (2 ** 32))
        emb = np.random.randn(self._dimension).astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-8)

    def encode(self, texts) -> np.ndarray:
        return np.array([self.encode_single(t) for t in texts])


class StubVisionModel:
    """
    Deterministic vision stub. Produces scripted observations per frame so the
    downstream memory/retrieval/context code sees real VisualObservations and
    real EvidenceRecords.
    """

    def __init__(self, script: dict):
        # script: {frame_id: VisualObservations}
        self._script = script
        self.analyze_calls = []

    @property
    def enabled(self) -> bool:
        return True

    def load(self):
        pass

    def unload(self):
        pass

    def analyze(self, image, frame_id: int = 0) -> VisualObservations:
        self.analyze_calls.append(frame_id)
        if frame_id not in self._script:
            raise AssertionError(
                f"vision stub has no scripted observation for frame {frame_id}; "
                f"scripted frames: {sorted(self._script)}"
            )
        return self._script[frame_id]


class StubGenerator(StoryGenerator):
    """
    Deterministic story generator. Replaces Qwen inference only.
    The REAL StoryGenerationPipeline still wraps and processes this.
    """

    def __init__(self, story_text: str):
        self._story_text = story_text
        self.generate_calls = []

    @property
    def model_id(self) -> str:
        return "stub-qwen"

    @property
    def is_loaded(self) -> bool:
        return True

    def load(self):
        pass

    def unload(self):
        pass

    def generate(self, prompt, max_new_tokens=500, temperature=0.0,
                 repetition_penalty=1.05, seed=0) -> str:
        self.generate_calls.append(prompt)
        return self._story_text

    def get_model_info(self) -> dict:
        return {"model_id": "stub-qwen"}


# ---------------------------------------------------------------------------
# SPY: wraps a REAL object, records calls, delegates to real implementation
# ---------------------------------------------------------------------------

class Spy:
    """
    Transparent spy for a real object.

    Attribute access returns a bound wrapper that records the call and its
    arguments, then delegates to the real method. Non-callable attributes are
    returned unchanged so the real object keeps working.
    """

    def __init__(self, real_obj, label: str, calls: list):
        object.__setattr__(self, '_real', real_obj)
        object.__setattr__(self, '_label', label)
        object.__setattr__(self, '_calls', calls)

    def __getattr__(self, name):
        real = object.__getattribute__(self, '_real')
        attr = getattr(real, name)
        if not callable(attr):
            return attr

        label = object.__getattribute__(self, '_label')
        calls = object.__getattribute__(self, '_calls')

        def wrapper(*args, **kwargs):
            result = attr(*args, **kwargs)
            calls.append({
                "component": label,
                "method": name,
                "args": args,
                "kwargs": kwargs,
                "result": result,
            })
            return result

        return wrapper

    def calls_to(self, method_name: str) -> list:
        calls = object.__getattribute__(self, '_calls')
        return [c for c in calls if c["method"] == method_name]

    @property
    def call_count(self) -> int:
        return len(object.__getattribute__(self, '_calls'))


# ---------------------------------------------------------------------------
# SYNTHETIC COLLECTION FIXTURES (deterministic, no vision inference)
# ---------------------------------------------------------------------------

def _obs(frame_id, scene, characters, objects, actions=None, mood="",
         relation_text=None):
    """
    Build a VisualObservations.

    ``relation_text`` maps an entity label to a natural-language evidence
    sentence that contains both the subject and the object of the intended
    claim. This mirrors how real detectors emit `evidence_text`, and is what
    lets ClaimVerifier's Strategy-1 (pure logic) resolve supporting evidence
    without any model.
    """
    chars = characters or []
    objs = objects or []
    acts = actions or []
    relation_text = relation_text or {}
    ev = []
    for c in chars:
        ev.append(EvidenceRecord(
            entity=c, type=EvidenceType.PERSON, frame_id=frame_id,
            confidence=0.9, source=SourceModel.FLORENCE2,
            evidence_text=relation_text.get(c, f"Person detected: {c}"),
            information_class=InformationClass.HARD_FACT,
        ))
    for o in objs:
        ev.append(EvidenceRecord(
            entity=o, type=EvidenceType.OBJECT, frame_id=frame_id,
            confidence=0.8, source=SourceModel.GROUNDING_DINO,
            evidence_text=relation_text.get(o, f"Grounded detection: {o}"),
            information_class=InformationClass.HARD_FACT,
        ))
    for a in acts:
        ev.append(EvidenceRecord(
            entity=a, type=EvidenceType.ACTION, frame_id=frame_id,
            confidence=0.7, source=SourceModel.FLORENCE2,
            information_class=InformationClass.SOFT_INFERENCE,
        ))
    return VisualObservations(
        image_id=f"img_{frame_id}", frame_id=frame_id, scene=scene,
        detailed_caption=f"{scene} with {chars} and {objs}",
        characters=chars, od_labels=objs, actions=acts,
        style_or_mood=mood, evidence_records=ev,
    )


def build_two_image_script():
    """IMAGE_A -> scene_A/evidence_A/entity_A ; IMAGE_B -> scene_B/evidence_B/entity_B."""
    return {
        0: _obs(
            0, "park", ["girl"], ["red_balloon"], ["holding"],
            relation_text={
                "girl": "The girl holds a red balloon",
                "red_balloon": "The girl holds a red balloon",
            },
        ),
        1: _obs(
            1, "street", ["boy"], ["bicycle"], ["riding"],
            relation_text={
                "boy": "The boy rides a bicycle",
                "bicycle": "The boy rides a bicycle",
            },
        ),
    }


def build_eight_image_script():
    """8 frames: girl+balloon (recurring), boy+bicycle, girl+kite."""
    script = {}
    for i in range(8):
        if i < 3:
            script[i] = _obs(
                i, "park", ["girl"], ["red_balloon"], ["holding"],
                relation_text={
                    "girl": "The girl holds a red balloon",
                    "red_balloon": "The girl holds a red balloon",
                },
            )
        elif i < 6:
            script[i] = _obs(
                i, "street", ["boy"], ["bicycle"], ["riding"],
                relation_text={
                    "boy": "The boy rides a bicycle",
                    "bicycle": "The boy rides a bicycle",
                },
            )
        else:
            script[i] = _obs(
                i, "beach", ["girl"], ["kite"], ["flying"],
                relation_text={
                    "girl": "The girl flies a kite",
                    "kite": "The girl flies a kite",
                },
            )
    return script


def build_one_image_script():
    return {
        0: _obs(
            0, "beach", ["dog"], ["frisbee"], ["running"],
            relation_text={
                "dog": "The dog carries a frisbee",
                "frisbee": "The dog carries a frisbee",
            },
        ),
    }


GENERATED_STORY = (
    "The girl holds a red balloon. "
    "The boy rides a bicycle. "
    "The girl flies a kite. "
)


# ---------------------------------------------------------------------------
# ORCHESTRATOR BUILDER: real components, spied, with only models stubbed
# ---------------------------------------------------------------------------

def build_orchestrator(script: dict, story_text: str = GENERATED_STORY,
                       spy_calls: list | None = None) -> PipelineOrchestrator:
    """
    Construct a PipelineOrchestrator whose V2.3B + V2.2 + generation +
    evaluation components are all REAL implementations, then wrap each in a
    Spy so we can prove invocation.

    Only these are stubbed (model boundaries):
      - vision (Florence2.analyze)
      - embeddings (encode_single / encode)
      - story generation (StoryGenerator.generate)
      - CLIP/NLI inside GroundingEvaluator
    """
    spy_calls = spy_calls if spy_calls is not None else []

    cfg = PipelineConfig(
        mode="standard",
        use_grounding_dino=False,
        use_ocr=False,
        use_faiss=True,
        use_creative_planner=True,
        use_verification=True,
        faiss_top_k=5,
        context_max_words=500,
        target_story_words=120,
        seed=7,
        device="cpu",
    )

    orch = PipelineOrchestrator(cfg)

    # ---- model boundaries -------------------------------------------------
    embeddings = StubEmbeddingModel()
    orch._florence = StubVisionModel(script)
    orch._grounding = None            # disabled -> exercises OCR/grounding skip
    orch._ocr = None
    orch._embedding_model = embeddings

    # ---- REAL components (no mocking of behaviour) -----------------------
    vector_store = FAISSVectorStore(embeddings, index_type="flat_ip")

    real_collection_builder = CollectionMemoryBuilder(
        embedding_model=embeddings,
        vector_store=vector_store,
        device="cpu",
    )
    real_hier_retriever = HierarchicalRetriever(embeddings, vector_store)
    real_collection_ctx = CollectionContextBuilder(max_words=800)
    real_world_builder = WorldStateBuilder()
    real_ranker = EvidenceRanker()
    real_planner = CreativePlanner.from_config(cfg, seed=cfg.seed)
    real_evaluator = ComprehensiveEvaluator(device="cpu")
    real_generator = StubGenerator(story_text)
    real_gen_pipeline = StoryGenerationPipeline(real_generator, cfg)

    # evaluator's claim stages are real; stub only the CLIP/NLI model boundary.
    # `_load_clip` / `_load_nli` download HF weights, so they are the model
    # boundary. Stubbing them (plus the two compute kernels they feed) keeps the
    # real GroundingEvaluator.evaluate() logic -- sentence splitting,
    # attribute-conflict rules, score aggregation, length/repetition checks --
    # fully executable without network or GPU.
    import torch

    ge = real_evaluator._grounding_evaluator
    ge._load_clip = lambda: None
    ge._load_nli = lambda: None
    ge._clip_low = 0.0
    ge._clip_span = 1.0
    ge._compute_clip_sims = lambda image, texts: torch.tensor([0.5] * len(texts))
    ge._compute_nli_contradictions = lambda caption, sentences: torch.tensor([0.1] * len(sentences))

    # VisualVerifier's CLIP re-detection is likewise a model boundary. Its
    # Strategy-1 (deterministic evidence matching), Strategy-2 dispatch and the
    # confidence assembly in verify_claim() all remain fully real.
    vv = real_evaluator._visual_verifier
    vv._load_clip = lambda: None
    vv._verify_with_clip = lambda claim, image: 0.5

    # ---- wrap in spies ----------------------------------------------------
    orch._vector_store = vector_store
    orch._collection_memory_builder = Spy(
        real_collection_builder, "CollectionMemoryBuilder", spy_calls)
    orch._hierarchical_retriever = Spy(
        real_hier_retriever, "HierarchicalRetriever", spy_calls)
    orch._collection_context_builder = Spy(
        real_collection_ctx, "CollectionContextBuilder", spy_calls)
    orch._world_builder = Spy(real_world_builder, "WorldStateBuilder", spy_calls)
    orch._ranker = Spy(real_ranker, "EvidenceRanker", spy_calls)
    orch._creative_planner = Spy(real_planner, "CreativePlanner", spy_calls)
    orch._gen_pipeline = Spy(real_gen_pipeline, "StoryGenerationPipeline", spy_calls)
    orch._evaluator = Spy(real_evaluator, "ComprehensiveEvaluator", spy_calls)

    # Spy the evaluator's claim stages too
    real_evaluator._claim_extractor = Spy(
        real_evaluator._claim_extractor, "ClaimExtractor", spy_calls)
    real_evaluator._claim_verifier = Spy(
        real_evaluator._claim_verifier, "ClaimVerifier", spy_calls)

    # legacy flat retriever: kept for run_multi_image, guarded in tests
    orch._retriever = MagicMock()
    orch._retriever.index_observations.return_value = []
    orch._retriever.retrieve.return_value = []
    orch._retriever.retrieve_for_story_planning.return_value = []

    # legacy context builders (used only by run_single_image/run_multi_image)
    from image_story.context.builder import ContextBuilder
    from image_story.context.sequence import SequenceContextBuilder
    orch._context_builder = ContextBuilder(max_words=cfg.context_max_words)
    orch._sequence_builder = SequenceContextBuilder(max_total_words=cfg.context_max_words)

    orch._verifier = MagicMock()
    orch._spy_calls = spy_calls
    return orch


@pytest.fixture
def images_2():
    """Two tiny in-memory PNG files on disk (run_collection opens paths)."""
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i in range(2):
            p = os.path.join(d, f"img_{i}.png")
            Image.new("RGB", (16, 16), color=(i * 40, 60, 80)).save(p)
            paths.append(p)
        yield paths


@pytest.fixture
def images_8():
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i in range(8):
            p = os.path.join(d, f"img_{i}.png")
            Image.new("RGB", (16, 16), color=(i * 10, 60, 80)).save(p)
            paths.append(p)
        yield paths


@pytest.fixture
def images_1():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "img_0.png")
        Image.new("RGB", (16, 16), color=(10, 20, 30)).save(p)
        yield [p]


# ===========================================================================
# 1. REAL EXECUTION: every critical stage actually runs
# ===========================================================================

class TestRealExecutionAllStagesRun:
    """Prove the real run_collection() drives every stage."""

    @pytest.fixture
    def executed(self, images_2):
        spy_calls = []
        orch = build_orchestrator(build_two_image_script(), spy_calls=spy_calls)
        artifacts = orch.run_collection(images_2, evaluate=True)
        return {"artifacts": artifacts, "spy": spy_calls, "orch": orch}

    def test_collection_memory_builder_really_executed(self, executed):
        """build_from_observations() ran on the REAL implementation."""
        calls = [c for c in executed["spy"]
                 if c["method"] == "build_from_observations"]
        assert len(calls) == 1, "CollectionMemoryBuilder.build_from_observations not called once"

        call = calls[0]
        # data flow: real VisualObservations were passed in
        observations = call["args"][0]
        assert isinstance(observations, list)
        assert len(observations) == 2
        assert all(isinstance(o, VisualObservations) for o in observations)
        assert [o.frame_id for o in observations] == [0, 1]
        # and it returned a real CollectionMemory
        assert isinstance(call["result"], CollectionMemory)

    def test_hierarchical_retriever_really_executed(self, executed):
        """retrieve_full() ran and received the CollectionMemory."""
        calls = [c for c in executed["spy"]
                 if c["method"] == "retrieve_full"]
        assert len(calls) == 1, "HierarchicalRetriever.retrieve_full not called"

        call = calls[0]
        assert isinstance(call["args"][0], str) and call["args"][0]
        result = call["result"]
        assert isinstance(result, RetrievalResult)
        assert isinstance(result.scenes, list)
        assert isinstance(result.evidence, list)
        assert isinstance(result.entities, list)
        assert isinstance(result.transitions, list)
        assert isinstance(result.narrative_elements, list)

    def test_collection_context_builder_really_executed(self, executed):
        """build_collection_context() ran with real CollectionMemory+RetrievalResult."""
        calls = [c for c in executed["spy"]
                 if c["method"] == "build_collection_context"]
        assert len(calls) == 1, "CollectionContextBuilder.build_collection_context not called"

        call = calls[0]
        collection_arg = call["args"][0]
        retrieval_arg = call["args"][1]
        assert isinstance(collection_arg, CollectionMemory)
        assert isinstance(retrieval_arg, RetrievalResult)
        # produced a real non-empty context string
        context = call["result"]
        assert isinstance(context, str)
        assert len(context) > 0

    def test_creative_planner_both_methods_executed(self, executed):
        """create_creative_plan() AND create_story_plan() ran with real inputs."""
        plan_calls = [c for c in executed["spy"]
                      if c["method"] == "create_creative_plan"]
        story_plan_calls = [c for c in executed["spy"]
                            if c["method"] == "create_story_plan"]
        assert len(plan_calls) == 1, "CreativePlanner.create_creative_plan not called"
        assert len(story_plan_calls) == 1, "CreativePlanner.create_story_plan not called"

        # create_creative_plan received the real world state, the real
        # observations and the ranked evidence (not empty placeholders)
        args = plan_calls[0]["args"]
        assert len(args) >= 3
        observations = args[1]
        assert observations and all(
            isinstance(o, VisualObservations) for o in observations
        )
        assert len(observations) == 2
        ranked = args[2]
        assert ranked, "CreativePlanner received no ranked evidence to ground on"

        # A real plan carries narrative content produced by the sub-engines.
        # A skipped planner would return a default CreativePlan() with every
        # collection empty, so these assertions detect that regression.
        plan = plan_calls[0]["result"]
        assert plan.hard_facts, "planner produced no hard facts from evidence"
        assert plan.locked_facts, "planner produced no locked facts"
        assert plan.characters, "CharacterSystem produced no character profiles"
        assert plan.central_conflict, "ConflictEngine produced no conflict"
        assert plan.narrative_arc, "planner produced no narrative arc"
        assert len(plan.creative_budget), "creative budget missing from plan"

        # StoryPlan built on top of that CreativePlan
        sp = story_plan_calls[0]["result"]
        assert sp is not None
        assert len(sp.beats) == 7
        assert sp.creative_plan is not None

    def test_story_generation_really_executed(self, executed):
        """StoryGenerationPipeline.generate_story() ran with real prompt+plan."""
        calls = [c for c in executed["spy"]
                 if c["method"] == "generate_story"]
        assert len(calls) == 1, "StoryGenerationPipeline.generate_story not called"

        call = calls[0]
        prompt = call["args"][0]
        story_plan = call["args"][1]
        assert isinstance(prompt, str) and len(prompt) > 0
        assert story_plan is not None
        assert hasattr(story_plan, "beats")

        draft = call["result"]
        assert draft is not None
        assert draft.text

    def test_claim_extraction_and_verification_really_executed(self, executed):
        """ClaimExtractor and ClaimVerifier ran during evaluation."""
        extract_calls = [c for c in executed["spy"]
                         if c["method"] == "extract_claims"]
        assert len(extract_calls) == 1, "ClaimExtractor.extract_claims not called"

        story = extract_calls[0]["args"][0]
        assert story is not None and story.text

        claims = extract_calls[0]["result"]
        assert isinstance(claims, list)
        assert all(isinstance(c, StoryClaim) for c in claims)

        # verification actually happened (textual path or visual path)
        verify_calls = [
            c for c in executed["spy"]
            if c["method"] in ("verify_claims", "_textual_verification")
        ]
        assert len(verify_calls) >= 1, "ClaimVerifier never invoked"

    def test_evaluator_really_executed(self, executed):
        """ComprehensiveEvaluator.evaluate() ran and produced EvaluationResult."""
        calls = [c for c in executed["spy"] if c["method"] == "evaluate"]
        assert len(calls) == 1, "ComprehensiveEvaluator.evaluate not called"

        artifacts_arg = calls[0]["args"][0]
        # the artifacts passed in already had story + plan populated
        assert artifacts_arg.story_draft is not None
        assert artifacts_arg.creative_plan is not None
        assert artifacts_arg.story_plan is not None

        result = calls[0]["result"]
        assert result is not None
        assert result.eval_total_s >= 0.0

    def test_vision_boundary_was_stubbed_not_skipped(self, executed):
        """Vision model was called for each frame (stub boundary proven active)."""
        assert executed["orch"]._florence.analyze_calls == [0, 1]


# ===========================================================================
# 2. DATA FLOW BETWEEN STAGES (real objects, not attribute checks)
# ===========================================================================

class TestDataFlowBetweenStages:
    """Verify actual objects move stage to stage through artifacts."""

    @pytest.fixture
    def artifacts(self, images_2):
        orch = build_orchestrator(build_two_image_script())
        return orch.run_collection(images_2, evaluate=True)

    def test_observations_flow_into_artifacts(self, artifacts):
        assert len(artifacts.observations) == 2
        assert artifacts.observations[0].frame_id == 0
        assert artifacts.observations[0].evidence_records

    def test_collection_memory_stored_with_scenes_and_entities(self, artifacts):
        cm = artifacts.collection_memory
        assert isinstance(cm, CollectionMemory)
        assert cm.total_images == 2
        assert cm.total_evidence > 0
        assert len(cm.scene_summaries) >= 1
        assert all(isinstance(s, SceneSummary) for s in cm.scene_summaries)
        assert len(cm.global_entities) > 0

        # entities from BOTH frames survived into memory
        labels = set(cm.global_entities.keys())
        assert "girl" in labels
        assert "boy" in labels

    def test_retrieval_result_stored_and_matches_collection(self, artifacts):
        rr = artifacts.retrieval_result
        assert isinstance(rr, RetrievalResult)
        assert rr is not None

        scene_ids = {s.scene_id for s in artifacts.collection_memory.scene_summaries}
        for scene in rr.scenes:
            assert scene.scene_id in scene_ids, \
                "retrieved scene not present in CollectionMemory"

    def test_retrieved_evidence_contains_real_evidence_records(self, artifacts):
        assert len(artifacts.retrieved_evidence) > 0
        assert all(isinstance(r, RetrievedEvidence) for r in artifacts.retrieved_evidence)

        memory_evidence_ids = {
            eid
            for s in artifacts.collection_memory.scene_summaries
            for eid in s.key_evidence_ids
        }
        retrieved_ids = {r.record.id for r in artifacts.retrieved_evidence}
        assert retrieved_ids <= memory_evidence_ids, \
            "retrieved evidence IDs must be traceable to scene key_evidence_ids"

    def test_ranked_evidence_populated(self, artifacts):
        assert len(artifacts.ranked_evidence) > 0
        assert all(isinstance(r, RetrievedEvidence) for r in artifacts.ranked_evidence)

    def test_creative_plan_populated_from_memory(self, artifacts):
        """
        The plan must be substantive, not a default CreativePlan(). Each of
        these collections is empty on a default plan, so this detects a
        planner that was skipped or fed nothing to ground on.
        """
        plan = artifacts.creative_plan
        assert plan is not None
        assert plan.genre
        assert plan.tone

        assert plan.hard_facts, "no hard facts extracted from retrieved evidence"
        assert plan.locked_facts, "no locked facts derived"
        assert plan.characters, "no character profiles generated"
        assert plan.central_conflict, "no central conflict generated"
        assert plan.narrative_arc, "no narrative arc designed"
        assert plan.creative_budget, "creative budget absent"

        # hard facts must reference the entities actually observed
        observed = {
            e.entity
            for o in artifacts.observations
            for e in o.evidence_records
        }
        assert any(any(ent in fact for ent in observed) for fact in plan.hard_facts), \
            "hard facts are not grounded in the observed entities"

    def test_story_plan_built_from_creative_plan(self, artifacts):
        sp = artifacts.story_plan
        assert sp is not None
        assert len(sp.beats) == 7
        assert sp.creative_plan is artifacts.creative_plan or \
            sp.creative_plan.generre == artifacts.creative_plan.genre

    def test_story_draft_built_with_prompt_and_plan(self, artifacts):
        draft = artifacts.story_draft
        assert draft is not None
        assert draft.text
        assert draft.word_count == len(draft.text.split())
        assert draft.story_plan is artifacts.story_plan
        assert draft.prompt_used, "generation prompt was not recorded"

    def test_claims_and_verification_land_in_artifacts(self, artifacts):
        assert len(artifacts.claims) > 0
        assert all(isinstance(c, StoryClaim) for c in artifacts.claims)
        assert len(artifacts.verification_results) > 0

    def test_evaluation_result_is_final_artifact(self, artifacts):
        ev = artifacts.evaluation
        assert ev is not None
        assert ev.eval_total_s >= 0.0
        assert ev.supported_claims + ev.unsupported_claims + \
            ev.contradicted_claims == len(artifacts.verification_results)

    def test_artifacts_to_dict_serializes_with_memory(self, artifacts):
        d = artifacts.to_dict()
        assert "collection_memory" in d
        assert "retrieval_result" in d
        assert d["collection_memory"]["total_images"] == 2
        assert d["retrieval_result"]["query"] is not None


# ===========================================================================
# 3. FLAT FAISS BYPASS: run_collection must NOT use legacy flat retrieval
# ===========================================================================

class TestFlatRetrievalBypassed:
    """Guard: the V2.2 flat retrieval path must not serve collection runs."""

    def test_flat_story_planning_retrieval_not_called(self, images_2):
        orch = build_orchestrator(build_two_image_script())
        orch.run_collection(images_2, evaluate=True)

        assert orch._retriever.retrieve_for_story_planning.call_count == 0, \
            "run_collection must NOT use EvidenceRetriever.retrieve_for_story_planning"
        assert orch._retriever.retrieve.call_count == 0, \
            "run_collection must NOT use flat EvidenceRetriever.retrieve"
        assert orch._retriever.index_observations.call_count == 0, \
            "run_collection must NOT index via legacy flat retriever"

    def test_collection_memory_indexed_with_provenance(self, images_2):
        """
        The legacy retriever was bypassed, but the SAME FAISS store received
        provenance-aware records from CollectionMemoryBuilder -- proving
        evidence really is scene-linked rather than silently dropped.
        """
        orch = build_orchestrator(build_two_image_script())
        artifacts = orch.run_collection(images_2, evaluate=True)

        store = orch._vector_store
        assert store.count > 0, "no evidence was indexed at all"

        scene_ids = set(store.get_scene_ids())
        memory_scene_ids = {
            s.scene_id for s in artifacts.collection_memory.scene_summaries
        }
        assert scene_ids <= memory_scene_ids

        # provenance records expose scene + image + collection
        idx = store.get_indexed_evidence(0)
        assert idx is not None
        assert idx.scene_id in memory_scene_ids
        assert idx.image_id.startswith("img_")
        assert idx.collection_id

    def test_hierarchical_retriever_is_the_only_evidence_source(self, images_2):
        spy_calls = []
        orch = build_orchestrator(build_two_image_script(), spy_calls=spy_calls)
        artifacts = orch.run_collection(images_2, evaluate=True)

        retrieval_calls = [c for c in spy_calls if c["method"] == "retrieve_full"]
        assert len(retrieval_calls) == 1

        # artifacts' retrieved_evidence derives from the RetrievalResult
        rr_ids = {e.id for e in artifacts.retrieval_result.evidence}
        art_ids = {r.record.id for r in artifacts.retrieved_evidence}
        assert art_ids <= rr_ids


# ===========================================================================
# 4. END-TO-END PROVENANCE CHAIN
# ===========================================================================

class TestProvenanceChain:
    """
    Story -> claim -> verification -> evidence -> scene -> image -> observation.

    NOTE ON A KNOWN GAP: `StoryClaim.evidence_ids` is declared in the schema
    but no code path in the current architecture populates it (ClaimExtractor
    builds claims with the default empty list). The claim -> evidence link is
    therefore carried by `VerificationResult.claim` / `.supporting_evidence`,
    not by `StoryClaim.evidence_ids`. The gap is asserted explicitly by
    `test_claim_evidence_ids_field_is_unpopulated_known_gap` so it cannot be
    silently claimed as working.
    """

    @pytest.fixture
    def artifacts(self, images_2):
        orch = build_orchestrator(build_two_image_script())
        return orch.run_collection(images_2, evaluate=True)

    def test_verification_result_links_claim_and_evidence(self, artifacts):
        """VerificationResult carries the claim AND its supporting evidence."""
        assert artifacts.verification_results, "no verification results"
        assert artifacts.claims, "no claims extracted"

        claim_ids = {c.id for c in artifacts.claims}
        for vr in artifacts.verification_results:
            # claim <-> verification link
            assert vr.claim_id in claim_ids
            assert vr.claim is not None
            assert vr.claim.id == vr.claim_id
            assert vr.status in ("supported", "unsupported", "contradicted")
            # evidence is physically attached to the verification result
            for e in vr.supporting_evidence:
                assert e.id
            for e in vr.contradicting_evidence:
                assert e.id

    def test_claim_to_observation_chain_resolves(self, artifacts):
        """
        Trace the full supported chain:

        StoryClaim -> VerificationResult -> EvidenceRecord
                   -> VisualObservations -> SceneSummary -> frame index
        """
        traced = []
        for vr in artifacts.verification_results:
            for evidence in vr.supporting_evidence:
                # evidence -> observation
                matches = [
                    o for o in artifacts.observations
                    if any(e.id == evidence.id for e in o.evidence_records)
                ]
                assert matches, \
                    f"evidence {evidence.id} not found in any VisualObservations"
                obs = matches[0]

                # observation -> scene
                scene = next(
                    (s for s in artifacts.collection_memory.scene_summaries
                     if obs.image_id in s.image_ids),
                    None,
                )
                assert scene is not None, \
                    f"observation {obs.image_id} not linked to any SceneSummary"

                # observation -> frame index (ordering link)
                assert obs.frame_id in scene.frame_indices
                traced.append((vr.claim, evidence, obs, scene))

        assert traced, \
            "no claim resolved to supporting evidence; provenance chain untested"

    def test_claims_are_grounded_in_the_generated_story(self, artifacts):
        """Claims must originate from the StoryDraft, not be fabricated."""
        story_text = artifacts.story_draft.text
        for c in artifacts.claims:
            assert c.original_sentence
            core = c.original_sentence.strip().rstrip(".")
            assert core in story_text, \
                "claim sentence does not appear in the generated story"

    def test_retrieval_result_scene_to_image_link(self, artifacts):
        rr = artifacts.retrieval_result
        assert rr is not None
        assert rr.scenes, "hierarchical retrieval returned no scenes"
        for scene in rr.scenes:
            assert scene.scene_id
            assert isinstance(scene.frame_indices, list)
            for frame_idx in scene.frame_indices:
                assert any(o.frame_id == frame_idx for o in artifacts.observations)

    def test_retrieved_evidence_is_real_evidence_records(self, artifacts):
        """RetrievalResult.evidence are the same records held by observations."""
        rr = artifacts.retrieval_result
        observation_evidence = {
            e.id for o in artifacts.observations for e in o.evidence_records
        }
        assert rr.evidence, "hierarchical retrieval returned no evidence"
        for e in rr.evidence:
            assert isinstance(e, EvidenceRecord)
            assert e.id in observation_evidence, \
                "retrieved evidence is not traceable to a VisualObservation"

    def test_scene_key_evidence_ids_resolve_to_observations(self, artifacts):
        all_evidence_ids = {
            e.id for o in artifacts.observations for e in o.evidence_records
        }
        for scene in artifacts.collection_memory.scene_summaries:
            for eid in scene.key_evidence_ids:
                assert eid in all_evidence_ids, \
                    f"scene {scene.scene_id} references unknown evidence {eid}"

    def test_entity_memory_links_to_scenes(self, artifacts):
        cm = artifacts.collection_memory
        scene_ids = {s.scene_id for s in cm.scene_summaries}
        for label, entity in cm.global_entities.items():
            assert isinstance(entity.entity_id, str)
            for sid in entity.scene_ids:
                assert sid in scene_ids, \
                    f"entity {label} references unknown scene {sid}"

    def test_narrative_elements_link_to_scenes(self, artifacts):
        cm = artifacts.collection_memory
        scene_ids = {s.scene_id for s in cm.scene_summaries}
        assert cm.narrative_elements, "no narrative elements built"
        for elem in cm.narrative_elements:
            assert isinstance(elem.element_id, str)
            assert isinstance(elem.label, str)
            for sid in elem.scene_ids:
                if sid:
                    assert sid in scene_ids

    def test_state_transitions_have_entity_and_scene_links(self, artifacts):
        cm = artifacts.collection_memory
        entity_ids = {e.entity_id for e in cm.global_entities.values()}
        scene_ids = {s.scene_id for s in cm.scene_summaries}
        assert cm.state_transitions, "no state transitions detected"
        for t in cm.state_transitions:
            assert t.entity_label
            assert t.transition_type
            if t.entity_id:
                assert t.entity_id in entity_ids
            for sid in (t.from_scene, t.to_scene):
                if sid:
                    assert sid in scene_ids

    def test_claim_evidence_ids_field_is_unpopulated_known_gap(self, artifacts):
        """
        DOCUMENTED GAP, not a passing claim of correctness.

        `StoryClaim.evidence_ids` exists in the schema but ClaimExtractor never
        populates it, so the direct claim -> evidence_ids -> RetrievedEvidence
        hop is absent in the current architecture. This test records the actual
        behaviour so the gap is visible and cannot be mistaken for a working
        link; it will need updating when the field is wired up.
        """
        populated = [c for c in artifacts.claims if c.evidence_ids]
        assert populated == [], (
            "StoryClaim.evidence_ids is now populated -- the documented "
            "provenance gap has been closed; update this test and the report."
        )


# ===========================================================================
# 5. MULTI-IMAGE CONTINUITY (8 images)
# ===========================================================================

class TestEightImageContinuity:
    """8-image collection must retain full collection-level memory."""

    @pytest.fixture
    def artifacts(self, images_8):
        orch = build_orchestrator(build_eight_image_script())
        return orch.run_collection(images_8, evaluate=True)

    def test_all_eight_images_processed(self, artifacts):
        assert len(artifacts.observations) == 8
        assert artifacts.collection_memory.total_images == 8

    def test_memory_retains_image_ids_scene_ids_entity_ids(self, artifacts):
        cm = artifacts.collection_memory

        image_ids = {img for s in cm.scene_summaries for img in s.image_ids}
        assert len(image_ids) == 8, "not all image IDs retained in scene summaries"

        assert all(s.scene_id for s in cm.scene_summaries)

        entity_ids = [e.entity_id for e in cm.global_entities.values()]
        assert entity_ids and all(isinstance(e, str) for e in entity_ids)

    def test_recurring_entities_span_multiple_scenes(self, artifacts):
        cm = artifacts.collection_memory
        multi_scene = [e for e in cm.global_entities.values() if len(e.scene_ids) >= 2]
        assert multi_scene, "no recurring entity tracked across scenes in 8-image run"

    def test_state_transitions_present_and_linked(self, artifacts):
        transitions = artifacts.collection_memory.state_transitions
        assert transitions, "no state transitions detected in 8-image collection"
        assert all(t.entity_label for t in transitions)
        assert all(t.transition_type for t in transitions)

    def test_narrative_elements_present(self, artifacts):
        elements = artifacts.collection_memory.narrative_elements
        assert elements, "no narrative memory built in 8-image collection"
        assert all(e.element_type for e in elements)

    def test_context_contains_collection_level_information(self, artifacts):
        """
        Context must expose scene structure / entities / transitions --
        not merely a flat frame-by-frame dump.
        """
        ctx = artifacts.context
        assert ctx and len(ctx) > 0

        assert "COLLECTION OVERVIEW" in ctx
        assert "Total scenes:" in ctx
        assert "Total images: 8" in ctx

        # hierarchical sections surfaced
        assert any(marker in ctx for marker in (
            "RELEVANT SCENES", "KEY ENTITIES", "STATE TRANSITIONS",
            "NARRATIVE ELEMENTS", "HARD FACTS",
        )), "collection-level sections missing from context"

    def test_context_respects_word_budget(self, artifacts):
        ctx = artifacts.context
        assert len(ctx.split()) <= 800 + 50, "context exceeded configured budget"

    def test_context_is_not_flat_observation_dump(self, artifacts):
        """Context must be built by CollectionContextBuilder, not sequence dump."""
        assert "SEQUENCE" not in artifacts.context or \
            "COLLECTION OVERVIEW" in artifacts.context

    def test_retrieval_scoped_to_collection(self, artifacts):
        rr = artifacts.retrieval_result
        assert rr is not None
        assert len(rr.scenes) <= 5, "top_k_scenes budget not respected"
        assert len(rr.evidence) <= 50, "max_total_evidence budget not respected"


# ===========================================================================
# 6. SINGLE IMAGE through the real collection path
# ===========================================================================

class TestSingleImageCollectionPath:
    """run_collection() with one image must work without special-casing."""

    @pytest.fixture
    def artifacts(self, images_1):
        orch = build_orchestrator(build_one_image_script())
        return orch.run_collection(images_1, evaluate=True)

    def test_single_image_produces_one_scene(self, artifacts):
        cm = artifacts.collection_memory
        assert cm.total_images == 1
        assert len(cm.scene_summaries) == 1
        assert cm.scene_summaries[0].image_ids == ["img_0"]

    def test_single_image_retrieval_works(self, artifacts):
        rr = artifacts.retrieval_result
        assert isinstance(rr, RetrievalResult)
        assert len(rr.scenes) == 1

    def test_single_image_context_works(self, artifacts):
        assert artifacts.context
        assert "COLLECTION OVERVIEW" in artifacts.context
        assert "Total images: 1" in artifacts.context

    def test_single_image_planner_and_generation_work(self, artifacts):
        assert artifacts.creative_plan is not None
        assert artifacts.story_plan is not None
        assert len(artifacts.story_plan.beats) == 7
        assert artifacts.story_draft is not None
        assert artifacts.story_draft.text

    def test_single_image_evaluation_works(self, artifacts):
        assert artifacts.evaluation is not None
        assert artifacts.evaluation.eval_total_s >= 0.0

    def test_single_image_flat_retrieval_still_bypassed(self, images_1):
        orch = build_orchestrator(build_one_image_script())
        orch.run_collection(images_1, evaluate=True)
        assert orch._retriever.retrieve_for_story_planning.call_count == 0
        assert orch._retriever.retrieve.call_count == 0


# ===========================================================================
# 7. LEGACY PATH INDEPENDENCE
# ===========================================================================

class _WorkingFlatRetriever:
    """
    A functioning stand-in for the V2.2 flat retriever.

    Deliberately NOT a MagicMock: the legacy path must be able to execute
    indexing + retrieval for real, so run_single_image()/run_multi_image()
    are exercised rather than stubbed out.
    """

    def __init__(self):
        self.calls = []
        self._indexed = []

    def initialize(self):
        self.calls.append("init")

    def index_observations(self, observations):
        self.calls.append(f"index:{len(observations)}")
        self._indexed = [e for o in observations for e in o.evidence_records]

    def retrieve(self, query, top_k=10):
        self.calls.append("retrieve")
        return [RetrievedEvidence(record=e, semantic_similarity=0.9)
                for e in self._indexed][:top_k]

    def retrieve_for_story_planning(self, *args, **kwargs):
        self.calls.append("retrieve_for_story_planning")
        return []


class TestLegacyPathIndependent:
    """Legacy V2.2 multi-image path must remain usable and NOT auto-upgraded."""

    def test_run_multi_image_still_works(self, images_2):
        """
        Legacy path runs on its own components. Collection memory components
        are never invoked by run_multi_image().
        """
        spy_calls = []
        orch = build_orchestrator(build_two_image_script(), spy_calls=spy_calls)
        # force legacy path explicitly
        artifacts = orch.run_multi_image(images_2, evaluate=True)

        assert artifacts.story_draft is not None
        assert artifacts.story_draft.text

        # legacy path did NOT build collection memory
        collection_methods = {
            "build_from_observations", "retrieve_full",
            "build_collection_context", "build_multi_scene_prompt",
        }
        used = {c["method"] for c in spy_calls} & collection_methods
        assert not used, \
            f"legacy run_multi_image wrongly used V2.3 methods: {used}"

        # and it left collection_memory unset
        assert artifacts.collection_memory is None
        assert artifacts.retrieval_result is None

    def test_legacy_path_used_sequence_context(self, images_2):
        orch = build_orchestrator(build_two_image_script())
        artifacts = orch.run_multi_image(images_2, evaluate=True)
        assert "COLLECTION OVERVIEW" not in artifacts.context, \
            "legacy path must not emit collection context"

    def test_legacy_and_collection_paths_are_distinct_entry_points(self):
        orch = build_orchestrator(build_two_image_script())
        assert hasattr(orch, "run_multi_image")
        assert hasattr(orch, "run_collection")
        assert orch.run_multi_image is not orch.run_collection

    def test_run_single_image_executes_end_to_end(self, images_1):
        """
        The legacy single-image path must actually RUN.

        This previously carried a latent NameError: run_single_image() was
        passed `collection_memory=` / `retrieval_result=` kwargs that only
        exist in the collection path, and no test executed the method, so the
        suite stayed green while every real single-image run would crash.
        """
        orch = build_orchestrator(build_one_image_script())
        orch._retriever = _WorkingFlatRetriever()

        artifacts = orch.run_single_image(images_1[0], evaluate=True)

        # every legacy stage produced real output
        assert artifacts.story_draft is not None
        assert artifacts.story_draft.text
        assert artifacts.creative_plan is not None
        assert artifacts.creative_plan.locked_facts
        assert artifacts.creative_plan.characters
        assert artifacts.story_plan is not None
        assert len(artifacts.story_plan.beats) == 7
        assert artifacts.evaluation is not None
        assert artifacts.context

        # legacy path uses the flat retriever and no collection structures
        assert orch._retriever.calls[0] == "init"
        assert any(c.startswith("index:") for c in orch._retriever.calls)
        assert "retrieve" in orch._retriever.calls
        assert artifacts.collection_memory is None
        assert artifacts.retrieval_result is None
        assert "COLLECTION OVERVIEW" not in artifacts.context, \
            "single-image legacy path must not emit collection context"

    def test_single_image_does_not_build_collection_memory(self, images_1):
        """Selecting the legacy path must not silently upgrade to V2.3."""
        spy_calls = []
        orch = build_orchestrator(build_one_image_script(), spy_calls=spy_calls)
        orch._retriever = _WorkingFlatRetriever()
        orch.run_single_image(images_1[0], evaluate=True)

        v23_methods = {"build_from_observations", "retrieve_full",
                       "build_collection_context"}
        used = {c["method"] for c in spy_calls} & v23_methods
        assert not used, f"legacy single-image path used V2.3 methods: {used}"


# ===========================================================================
# 8. V2.2 GROUNDED CREATIVITY STILL ENFORCED IN COLLECTION RUN
# ===========================================================================

class TestOrchestratorWiringIntegrity:
    """
    Static + behavioural guards against the two integration defects that the
    earlier attribute-only suite could not see:

      1. run_single_image() referenced `retrieval_result` /
         `self._collection_memory`, which are only defined in the collection
         path -> NameError on every single-image run.
      2. run_collection() did NOT pass collection_memory/retrieval_result to
         the planner, leaving the planner's V2.3 memory awareness as dead code.
    """

    @staticmethod
    def _source() -> str:
        import pathlib
        return pathlib.Path(
            "src/image_story/pipeline/orchestrator.py"
        ).read_text(encoding="utf-8")

    def test_single_image_does_not_reference_collection_only_names(self):
        src = self._source()
        i = src.index("def run_single_image(")
        j = src.index("def run_multi_image(")
        body = src[i:j]
        assert "retrieval_result" not in body, \
            "run_single_image must not reference collection-only `retrieval_result`"
        assert "self._collection_memory" not in body, \
            "run_single_image must not reference collection-only `_collection_memory`"

    def test_single_image_planner_call_uses_legacy_signature(self):
        import inspect
        from image_story.narrative.planner import CreativePlanner
        src = self._source()
        i = src.index("def run_single_image(")
        j = src.index("def run_multi_image(")
        assert "collection_memory=" not in src[i:j]

    def test_collection_path_passes_collection_memory_to_planner(self):
        src = self._source()
        k = src.index("def run_collection(")
        body = src[k:]
        assert "collection_memory=self._collection_memory" in body, \
            "run_collection must give the planner the CollectionMemory"
        assert "retrieval_result=retrieval_result" in body, \
            "run_collection must give the planner the RetrievalResult"

    def test_planner_receives_memory_at_runtime(self, images_2):
        """Runtime proof that the planner actually receives both objects."""
        spy_calls = []
        orch = build_orchestrator(build_two_image_script(), spy_calls=spy_calls)
        orch.run_collection(images_2, evaluate=True)

        pc = [c for c in spy_calls if c["method"] == "create_creative_plan"]
        assert len(pc) == 1
        kwargs = pc[0]["kwargs"]
        assert "collection_memory" in kwargs, \
            "planner was not given collection_memory"
        assert isinstance(kwargs["collection_memory"], CollectionMemory)
        assert "retrieval_result" in kwargs, \
            "planner was not given retrieval_result"
        assert isinstance(kwargs["retrieval_result"], RetrievalResult)

    def test_planner_callback_plan_uses_collection_memory(self, images_8):
        """
        With memory supplied, the planner's V2.3 callback path is used
        (narrative elements + recurring entities), not the legacy world-state
        only path.

        Uses the 8-image collection because callbacks are derived from
        *recurrence*; a 2-image collection has no recurring entity and
        correctly yields no callbacks.
        """
        orch = build_orchestrator(build_eight_image_script())
        artifacts = orch.run_collection(images_8, evaluate=True)

        cm = artifacts.collection_memory
        recurring = [e for e in cm.global_entities.values()
                     if len(e.scene_ids) > 1]
        assert recurring, "fixture should contain recurring entities"

        callbacks = artifacts.creative_plan.callback_plan
        assert callbacks, "planner produced no callbacks despite recurring entities"
        # V2.3 callback entries carry scene-level provenance
        assert any("payoff_scene" in c for c in callbacks), \
            "planner did not use the V2.3 collection-memory callback path"

        # and the payoff scenes are real scenes
        scene_ids = {s.scene_id for s in cm.scene_summaries}
        for c in callbacks:
            ps = c.get("payoff_scene")
            if ps:
                assert ps in scene_ids

    def test_two_image_collection_correctly_has_no_callbacks(self, images_2):
        """
        Guards against over-claiming: with no recurring entity across scenes
        the V2.3 callback path must produce nothing rather than invent them.
        """
        orch = build_orchestrator(build_two_image_script())
        artifacts = orch.run_collection(images_2, evaluate=True)
        recurring = [e for e in artifacts.collection_memory.global_entities.values()
                     if len(e.scene_ids) > 1]
        assert not recurring, "2-image fixture unexpectedly has recurring entities"
        assert artifacts.creative_plan.callback_plan == []


class TestGroundedCreativityInCollectionRun:

    @pytest.fixture
    def artifacts(self, images_2):
        orch = build_orchestrator(build_two_image_script())
        return orch.run_collection(images_2, evaluate=True)

    def test_creative_budget_intact(self, artifacts):
        budget = artifacts.creative_plan.creative_budget
        assert budget["safe_creative"]["max"] == 8
        assert budget["risky_inferred"]["max"] == 3
        assert budget["forbidden_visual"]["max"] == 0

    def test_locked_facts_derived_from_evidence(self, artifacts):
        plan = artifacts.creative_plan
        assert isinstance(plan.locked_facts, list)
        # locked facts must come from real hard-fact evidence
        hard_fact_entities = {
            e.entity
            for o in artifacts.observations
            for e in o.evidence_records
            if e.information_class == InformationClass.HARD_FACT
        }
        for fact in plan.locked_facts:
            assert fact in hard_fact_entities

    def test_claims_classified_into_three_categories(self, artifacts):
        classifications = {c.claim_classification for c in artifacts.claims}
        assert classifications <= {"observed", "inferred", "creative"}
        assert classifications, "claims were not classified at all"

    def test_creative_claims_not_counted_as_contradictions(self, artifacts):
        """Minimal-repair semantics: creative claims must not be contradicted."""
        creative = [c for c in artifacts.claims
                    if c.claim_classification == "creative"]
        for c in creative:
            vr = next((r for r in artifacts.verification_results
                       if r.claim_id == c.id), None)
            if vr is not None:
                assert vr.status != ClaimStatus.CONTRADICTED.value, \
                    "creative claim was marked contradicted"

    def test_prompt_contains_creative_direction_and_locked_facts(self, artifacts):
        prompt = artifacts.story_draft.prompt_used
        plan = artifacts.creative_plan
        assert plan.genre in prompt
        assert plan.tone in prompt
        if plan.locked_facts:
            assert plan.locked_facts[0] in prompt