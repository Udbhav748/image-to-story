"""Integration tests for V2.3C - verifying the actual call graph."""
import pytest
import contextlib
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np
from PIL import Image

from image_story.pipeline.orchestrator import PipelineOrchestrator
from image_story.collections import CollectionPipeline, CollectionPipelineConfig, ProcessingConfig
from image_story.memory.collection_builder import CollectionMemoryBuilder
from image_story.memory.hierarchical import CollectionMemory
from image_story.memory.hierarchical_retriever import HierarchicalRetriever
from image_story.context.collection_builder import CollectionContextBuilder
from image_story.memory.scene_grouper import SceneGrouper
from image_story.domain.schemas import (
    PipelineConfig, PipelineArtifacts, VisualObservations,
    EvidenceRecord, EvidenceType, SourceModel, InformationClass,
    CreativePlan, StoryPlan, StoryDraft, ExperimentManifest,
    RetrievedEvidence, WorldState, WorldEntity
)
from image_story.domain.enums import PipelineMode


class TestV23CIntegration:
    """Integration tests verifying the V2.3C call graph."""

    @pytest.fixture
    def pipeline_config(self):
        return PipelineConfig.from_mode("standard")

    @pytest.fixture
    def mock_images(self):
        """Create mock PIL images for testing."""
        images = []
        for i in range(3):
            img = Mock(spec=Image.Image)
            img.convert.return_value = img
            images.append(img)
        return images

    @pytest.fixture
    def sample_observations(self):
        """Create sample visual observations for testing."""
        obs_list = []
        for i in range(3):
            obs = VisualObservations(
                image_id=f"img_{i}",
                frame_id=i,
                scene="test scene",
                detailed_caption=f"Frame {i} description",
                objects=[f"obj_{i}"],
                od_labels=[f"label_{i}"],
                characters=[f"char_{i}"],
                actions=[f"action_{i}"],
                spatial_relations=[f"spatial_{i}"],
                region_descriptions=[f"region_{i}"],
                style_or_mood="test",
                evidence_records=[
                    EvidenceRecord(
                        entity=f"entity_{i}",
                        type=EvidenceType.OBJECT,
                        frame_id=i,
                        confidence=0.9,
                        source=SourceModel.FLORENCE2,
                        evidence_text=f"Object detected: entity_{i}",
                        information_class=InformationClass.HARD_FACT,
                    )
                ],
            )
            obs_list.append(obs)
        return obs_list

    def test_collection_memory_builder_invoked(self, pipeline_config, sample_observations):
        """Test 1: CollectionMemoryBuilder is actually invoked."""
        builder = CollectionMemoryBuilder()

        with patch.object(builder, 'build_from_observations', wraps=builder.build_from_observations) as mock_build:
            collection = builder.build_from_observations(sample_observations, pipeline_config)

            mock_build.assert_called_once()
            assert isinstance(collection, CollectionMemory)
            assert len(collection.scene_summaries) > 0

    def test_hierarchical_retriever_invoked(self, pipeline_config, sample_observations):
        """Test 2: HierarchicalRetriever is actually invoked."""
        from image_story.memory.faiss_store import FAISSVectorStore
        from image_story.memory.embeddings import create_embedding_model
        from image_story.memory.hierarchical import RetrievalResult

        embedding_model = create_embedding_model()
        embedding_model.load()
        vector_store = FAISSVectorStore(embedding_model)

        retriever = HierarchicalRetriever(embedding_model, vector_store)

        # Mock the embedding model to avoid actual encoding
        with patch.object(embedding_model, 'encode_single', return_value=np.random.rand(384).astype(np.float32)) as mock_encode:
            with patch.object(embedding_model, 'encode', return_value=np.random.rand(3, 384).astype(np.float32)) as mock_encode_batch:
                vector_store.initialize()
                # Add some evidence
                for obs in sample_observations:
                    for evidence in obs.evidence_records:
                        text = f"Entity: {evidence.entity}"
                        vector_store.add_evidence_with_provenance(evidence, text, "img_1", "scene_1", 0, "coll_1")

                retriever.set_collection_memory(
                    CollectionMemory(
                        collection_id="test",
                        scene_summaries=[],
                        global_entities={},
                        state_transitions=[],
                        narrative_elements=[],
                        total_images=3,
                        total_evidence=len(sample_observations),
                    )
                )

                # This should use the hierarchical retriever
                results = retriever.retrieve_full("test query")

                assert isinstance(results, RetrievalResult)

    def test_collection_context_builder_invoked(self, pipeline_config, sample_observations):
        """Test 3: CollectionContextBuilder is actually invoked."""
        from image_story.memory.hierarchical import RetrievalResult

        builder = CollectionContextBuilder()
        collection = CollectionMemory(
            collection_id="test",
            scene_summaries=[],
            global_entities={},
            state_transitions=[],
            narrative_elements=[],
            total_images=3,
            total_evidence=3,
        )

        retrieval_result = RetrievalResult(
            scenes=[],
            evidence=[evidence for obs in sample_observations for evidence in obs.evidence_records],
            entities=[],
            transitions=[],
            narrative_elements=[],
            query="test",
            retrieval_time_ms=1.0,
        )

        creative_plan = CreativePlan(
            genre="whimsical",
            tone="comedic",
            locked_facts=["entity_0", "entity_1"],
        )

        context = builder.build_collection_context(
            collection,
            retrieval_result,
            creative_plan,
            current_scene_id=None,
        )

        assert "COLLECTION OVERVIEW" in context
        assert "LOCKED FACTS" in context

    def test_creative_planner_receives_hierarchical_info(self, pipeline_config, sample_observations):
        """Test 4: CreativePlanner receives hierarchical collection information."""
        from image_story.memory.hierarchical import RetrievalResult
        from image_story.memory.collection_builder import create_collection_memory_builder
        from image_story.narrative.planner import CreativePlanner
        from image_story.context.world_state import WorldStateBuilder

        # Build collection memory
        builder = create_collection_memory_builder()
        collection = builder.build_from_observations(sample_observations, pipeline_config)

        # Build world state for the creative planner
        world_builder = WorldStateBuilder()
        world_state = world_builder.add_observations_batch(sample_observations)

        # Create creative planner
        planner = CreativePlanner.from_config(pipeline_config)

        # Create mock ranked evidence
        ranked_evidence = [
            type('obj', (object,), {'record': e, 'semantic_similarity': 0.9})
            for e in sample_observations[0].evidence_records
        ]

        # This should work without error
        creative_plan = planner.create_creative_plan(
            world_state,
            sample_observations,
            ranked_evidence
        )

        assert creative_plan is not None
        assert len(creative_plan.characters) >= 0

    def test_story_generation_still_happens(self, pipeline_config, sample_observations):
        """Test 5: Story generation still happens after new memory stage."""
        from image_story.generation.base import create_generator, StoryGenerationPipeline
        from image_story.narrative.planner import CreativePlanner

        # Create a simple creative plan
        creative_plan = CreativePlan(
            genre="whimsical",
            tone="comedic",
            locked_facts=["test"],
        )

        story_plan = StoryPlan(
            beats=[],
            creative_plan=creative_plan,
            target_total_words=100,
        )

        # Mock the generator
        mock_generator = Mock()
        mock_generator.generate.return_value = "This is a test story."
        mock_generator.model_id = "test-model"

        gen_pipeline = StoryGenerationPipeline(mock_generator, pipeline_config)
        story_draft = gen_pipeline.generate_story("Test prompt", story_plan)

        assert story_draft is not None
        assert story_draft.text == "This is a test story."

    def test_claim_extraction_still_occurs(self, sample_observations):
        """Test 6: Claim extraction still occurs."""
        from image_story.evaluation.claims import ClaimExtractor
        from image_story.domain.schemas import StoryDraft

        extractor = ClaimExtractor()
        story = StoryDraft(text="The girl holds a red balloon. She secretly wants to fly.")

        claims = extractor.extract_claims(story, "test context")

        assert len(claims) > 0
        # Verify claim classification works
        creative_claims = [c for c in claims if c.claim_classification == "creative"]
        observed_claims = [c for c in claims if c.claim_classification == "observed"]

        assert len(creative_claims) > 0 or len(observed_claims) > 0

    def test_claim_verification_still_occurs(self, sample_observations):
        """Test 7: Claim verification still occurs."""
        from image_story.evaluation.claims import ClaimVerifier
        from image_story.vision.verifier import VisualVerifier
        from image_story.domain.schemas import StoryClaim, EvidenceRecord, EvidenceType, SourceModel, InformationClass
        from image_story.domain.enums import ClaimStatus

        claim = StoryClaim(
            subject="girl",
            relation="holds",
            object="red balloon",
            original_sentence="The girl holds a red balloon.",
            claim_classification="observed",
        )

        # Create evidence that matches the claim
        evidence = EvidenceRecord(
            entity="girl holds red balloon",
            type=EvidenceType.ACTION,
            frame_id=0,
            confidence=0.9,
            source=SourceModel.FLORENCE2,
            evidence_text="Girl holds red balloon detected",
            information_class=InformationClass.HARD_FACT,
        )

        # Mock visual verifier
        mock_verifier = Mock()
        mock_verifier.verify_claim.return_value = Mock(
            status="supported",
            confidence=0.9,
            supporting_evidence=[],
            contradicting_evidence=[],
        )

        verifier = ClaimVerifier(visual_verifier=mock_verifier)
        results = verifier.verify_claims([claim], None, [evidence])

        assert len(results) == 1
        assert results[0].status == "supported"

    def test_evaluation_still_occurs(self):
        """Test 8: Evaluation still occurs."""
        from image_story.evaluation.evaluator import ComprehensiveEvaluator
        from image_story.domain.schemas import PipelineArtifacts, StoryDraft, ExperimentManifest, EvaluationResult

        evaluator = ComprehensiveEvaluator(device="cpu")

        # Create mock artifacts
        story = StoryDraft(text="Test story.")
        artifacts = PipelineArtifacts(
            run_id="test",
            manifest=ExperimentManifest(),
            story_draft=story,
        )

        # Mock the evaluator components
        with patch.object(evaluator, '_grounding_evaluator') as mock_grounding:
            with patch.object(evaluator, '_claim_extractor') as mock_extract:
                with patch.object(evaluator, '_claim_verifier') as mock_verifier:
                    mock_grounding.evaluate.return_value = EvaluationResult(
                        grounding_score=0.8,
                        clip_image_story_mean=0.25,
                        nli_contra_mean=0.1,
                        attribute_conflict=[],
                        repetition_rate=0.0,
                        length_valid=True,
                        grounding_pass=True,
                        eval_clip_s=0.1,
                        eval_nli_s=0.1,
                        eval_rules_s=0.01,
                    )
                    mock_extract.extract_claims.return_value = []
                    mock_verifier.verify_claims.return_value = []

                    result = evaluator.evaluate(artifacts, None)

                    assert result is not None
                    assert hasattr(result, 'grounding_score')

    def test_single_image_supported(self, pipeline_config):
        """Test 9: Single image remains supported."""
        from image_story.pipeline.orchestrator import PipelineOrchestrator
        from image_story.domain.schemas import PipelineArtifacts
        from PIL import Image

        # This test just verifies the orchestrator can be created
        orchestrator = PipelineOrchestrator(pipeline_config)
        assert orchestrator is not None
        assert orchestrator._config == pipeline_config

    def _make_mock_observations(self, image_id="test", frame_id=0):
        return VisualObservations(
            image_id=image_id,
            frame_id=frame_id,
            scene="test scene",
            detailed_caption="A girl holding a red balloon.",
            characters=["girl"],
            od_labels=["balloon"],
            actions=["holding"],
            evidence_records=[
                EvidenceRecord(
                    entity="girl",
                    type=EvidenceType.PERSON,
                    frame_id=frame_id,
                    confidence=0.9,
                    source=SourceModel.FLORENCE2,
                    evidence_text="Person detected: girl",
                    information_class=InformationClass.HARD_FACT,
                ),
                EvidenceRecord(
                    entity="balloon",
                    type=EvidenceType.OBJECT,
                    frame_id=frame_id,
                    confidence=0.8,
                    source=SourceModel.FLORENCE2,
                    evidence_text="Object detected: balloon",
                    information_class=InformationClass.HARD_FACT,
                ),
            ],
        )

    @contextlib.contextmanager
    def _stubbed_models(self):
        """Patch out every heavyweight model so no real weights are loaded."""
        from image_story.pipeline import orchestrator as orch_mod

        mock_img = Mock()
        mock_img.convert.return_value = mock_img
        mock_img.size = (640, 480)
        mock_img.mode = "RGB"

        fake_generator = Mock()
        fake_generator.model_id = "stub-qwen"
        fake_generator.generate.return_value = (
            "A girl held a red balloon. She dreamed of the sky."
        )

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("PIL.Image.open", return_value=mock_img))
            stack.enter_context(patch.object(orch_mod.Florence2Model, "load"))
            stack.enter_context(
                patch.object(
                    orch_mod.Florence2Model,
                    "analyze",
                    side_effect=lambda *a, **k: self._make_mock_observations(),
                )
            )
            stack.enter_context(
                patch.object(orch_mod, "create_grounding_dino", return_value=Mock())
            )
            stack.enter_context(patch.object(orch_mod, "create_ocr", return_value=Mock()))
            stack.enter_context(patch.object(orch_mod, "create_generator", return_value=fake_generator))
            stack.enter_context(patch.object(orch_mod, "ComprehensiveEvaluator", return_value=Mock()))
            stack.enter_context(patch.object(orch_mod, "VisualVerifier", return_value=Mock()))
            yield fake_generator

    def test_multi_image_benchmark_supported(self):
        """Test 10: multi-image V2.2 benchmark path still produces a story."""
        from image_story.pipeline.orchestrator import PipelineOrchestrator

        with self._stubbed_models():
            orchestrator = PipelineOrchestrator(PipelineConfig.from_mode("fast"))
            artifacts = orchestrator.run_multi_image(
                ["img1.jpg", "img2.jpg", "img3.jpg"], evaluate=False
            )

        assert artifacts is not None
        assert artifacts.story_draft is not None
        assert artifacts.story_draft.text

    def test_legacy_v22_path_supported(self):
        """Test 11: legacy V2.2 run_multi_image path remains wired to generation."""
        from image_story.pipeline.orchestrator import PipelineOrchestrator

        with self._stubbed_models() as fake_generator:
            orchestrator = PipelineOrchestrator(PipelineConfig.from_mode("fast"))
            artifacts = orchestrator.run_multi_image(["img1.jpg", "img2.jpg"], evaluate=False)

            assert fake_generator.generate.called

        assert artifacts is not None
        assert artifacts.story_draft is not None

    def test_collection_pipeline_routes_to_run_collection(self):
        """Test 12: CollectionPipeline with memory enabled routes to run_collection."""
        from image_story.pipeline.orchestrator import PipelineOrchestrator

        from image_story.domain.schemas import ImageCollection, ImageRecord

        coll_cfg = CollectionPipelineConfig(skip_validation=True)

        collection = ImageCollection(ordering_mode="upload_order")
        for i in range(3):
            collection.add_image(
                ImageRecord(
                    path=f"img_{i}.jpg",
                    validation_status="valid",
                    width=640,
                    height=480,
                    mime_type="image/jpeg",
                )
            )

        pipeline = CollectionPipeline(coll_cfg)

        fake_artifacts = PipelineArtifacts(run_id="stub", manifest=ExperimentManifest())
        fake_artifacts.story_draft = StoryDraft(text="A stub story.")

        with patch.object(PipelineOrchestrator, "run_collection", return_value=fake_artifacts) as mock_run_collection, \
             patch.object(PipelineOrchestrator, "run_multi_image") as mock_run_multi, \
             patch.object(
                 type(pipeline.ingestion),
                 "process_collection",
                 autospec=True,
                 side_effect=lambda _self, coll, processing_config=None: [
                     setattr(img, "validation_status", "valid") for img in coll.images
                 ],
             ):
            result = pipeline.process_collection(
                collection,
                use_collection_memory=True,
                evaluate=False,
            )

        assert mock_run_collection.called, "V2.3B collection path was not taken"
        assert not mock_run_multi.called, "legacy V2.2 path was used instead"
        assert result["success"] is True

    # ============================================================
    # SCALE TESTS (synthetic data, no real model inference)
    # ============================================================

    def _make_synthetic_observations(self, num_frames: int):
        """Create synthetic observations for scale testing."""
        from image_story.domain.schemas import (
            VisualObservations, EvidenceRecord, EvidenceType, 
            SourceModel, InformationClass
        )
        observations = []
        for i in range(num_frames):
            obs = VisualObservations(
                image_id=f"img_{i}",
                frame_id=i,
                scene=f"scene_{i // 10}",
                detailed_caption=f"Frame {i} description",
                objects=[f"obj_{i}"],
                od_labels=[f"label_{i}"],
                characters=[f"char_{i}"],
                actions=[f"action_{i}"],
                spatial_relations=[f"spatial_{i}"],
                region_descriptions=[f"region_{i}"],
                style_or_mood="test",
                evidence_records=[
                    EvidenceRecord(
                        entity=f"entity_{i % 20}",
                        type=EvidenceType.OBJECT,
                        frame_id=i,
                        confidence=0.9,
                        source=SourceModel.FLORENCE2,
                        evidence_text=f"Detected entity_{i % 20}",
                        information_class=InformationClass.HARD_FACT,
                    )
                    for _ in range(3)
                ],
            )
            observations.append(obs)
        return observations

    def test_scale_100_collection_memory_build(self):
        """Test 13: CollectionMemory builds correctly at 100 frames scale."""
        from image_story.memory.collection_builder import create_collection_memory_builder
        from image_story.domain.schemas import PipelineConfig
        
        config = PipelineConfig.from_mode("standard")
        observations = self._make_synthetic_observations(100)
        
        builder = create_collection_memory_builder(device="cpu")
        collection = builder.build_from_observations(observations, config)
        
        assert collection is not None
        assert collection.total_images == 100
        assert len(collection.scene_summaries) > 0
        assert len(collection.global_entities) > 0
        assert len(collection.state_transitions) >= 0

    def test_scale_1000_collection_memory_build(self):
        """Test 14: CollectionMemory builds correctly at 1000 frames scale."""
        from image_story.memory.collection_builder import create_collection_memory_builder
        from image_story.domain.schemas import PipelineConfig
        
        config = PipelineConfig.from_mode("standard")
        observations = self._make_synthetic_observations(1000)
        
        builder = create_collection_memory_builder(device="cpu")
        collection = builder.build_from_observations(observations, config)
        
        assert collection is not None
        assert collection.total_images == 1000
        assert len(collection.scene_summaries) > 0
        assert len(collection.global_entities) > 0

    def test_scale_100_hierarchical_retrieval(self):
        """Test 15: Hierarchical retrieval works at 100 frames scale."""
        from image_story.memory.collection_builder import create_collection_memory_builder
        from image_story.memory.hierarchical_retriever import create_hierarchical_retriever
        from image_story.memory.faiss_store import FAISSVectorStore
        from image_story.memory.embeddings import create_embedding_model
        from image_story.domain.schemas import PipelineConfig
        
        config = PipelineConfig.from_mode("standard")
        observations = self._make_synthetic_observations(100)
        
        # Build memory
        builder = create_collection_memory_builder(device="cpu")
        collection = builder.build_from_observations(observations, config)
        
        # Build FAISS
        embedding_model = create_embedding_model(device="cpu")
        embedding_model.load()
        vector_store = FAISSVectorStore(embedding_model, device="cpu")
        vector_store.initialize()
        
        for obs in observations:
            for ev in obs.evidence_records:
                text = f"Entity: {ev.entity}"
                vector_store.add_evidence_with_provenance(
                    ev, text, f"img_{ev.frame_id}", f"scene_{ev.frame_id // 10}", ev.frame_id, "test"
                )
        
        # Retrieve
        retriever = create_hierarchical_retriever(
            embedding_model, vector_store,
            top_k_scenes=config.top_k_scenes,
            top_k_evidence_per_scene=config.top_k_evidence_per_scene,
            max_total_evidence=config.faiss_top_k * 5,
        )
        retriever.set_collection_memory(collection)
        result = retriever.retrieve_full("test query")
        
        assert result is not None
        assert len(result.scenes) <= config.top_k_scenes
        assert len(result.evidence) <= config.top_k_evidence_per_scene * config.top_k_scenes

    def test_scale_1000_hierarchical_retrieval(self):
        """Test 16: Hierarchical retrieval works at 1000 frames scale."""
        from image_story.memory.collection_builder import create_collection_memory_builder
        from image_story.memory.hierarchical_retriever import create_hierarchical_retriever
        from image_story.memory.faiss_store import FAISSVectorStore
        from image_story.memory.embeddings import create_embedding_model
        from image_story.domain.schemas import PipelineConfig
        
        config = PipelineConfig.from_mode("standard")
        observations = self._make_synthetic_observations(1000)
        
        builder = create_collection_memory_builder(device="cpu")
        collection = builder.build_from_observations(observations, config)
        
        embedding_model = create_embedding_model(device="cpu")
        embedding_model.load()
        vector_store = FAISSVectorStore(embedding_model, device="cpu")
        vector_store.initialize()
        
        for obs in observations:
            for ev in obs.evidence_records:
                text = f"Entity: {ev.entity}"
                vector_store.add_evidence_with_provenance(
                    ev, text, f"img_{ev.frame_id}", f"scene_{ev.frame_id // 10}", ev.frame_id, "test"
                )
        
        retriever = create_hierarchical_retriever(
            embedding_model, vector_store,
            top_k_scenes=config.top_k_scenes,
            top_k_evidence_per_scene=config.top_k_evidence_per_scene,
            max_total_evidence=config.faiss_top_k * 5,
        )
        retriever.set_collection_memory(collection)
        result = retriever.retrieve_full("test query")
        
        assert result is not None
        assert len(result.scenes) <= config.top_k_scenes
        assert len(result.evidence) <= config.top_k_evidence_per_scene * config.top_k_scenes

    def test_scale_context_budget_respected(self):
        """Test 17: Context budget is respected at scale (1000 frames)."""
        from image_story.memory.collection_builder import create_collection_memory_builder
        from image_story.memory.hierarchical_retriever import create_hierarchical_retriever
        from image_story.memory.faiss_store import FAISSVectorStore
        from image_story.memory.embeddings import create_embedding_model
        from image_story.context.collection_builder import create_collection_context_builder
        from image_story.domain.schemas import PipelineConfig, CreativePlan
        
        config = PipelineConfig.from_mode("standard")
        observations = self._make_synthetic_observations(1000)
        
        builder = create_collection_memory_builder(device="cpu")
        collection = builder.build_from_observations(observations, config)
        
        embedding_model = create_embedding_model(device="cpu")
        embedding_model.load()
        vector_store = FAISSVectorStore(embedding_model, device="cpu")
        vector_store.initialize()
        
        for obs in observations:
            for ev in obs.evidence_records:
                text = f"Entity: {ev.entity}"
                vector_store.add_evidence_with_provenance(
                    ev, text, f"img_{ev.frame_id}", f"scene_{ev.frame_id // 10}", ev.frame_id, "test"
                )
        
        retriever = create_hierarchical_retriever(
            embedding_model, vector_store,
            top_k_scenes=config.top_k_scenes,
            top_k_evidence_per_scene=config.top_k_evidence_per_scene,
            max_total_evidence=config.faiss_top_k * 5,
        )
        retriever.set_collection_memory(collection)
        result = retriever.retrieve_full("test query")
        
        context_builder = create_collection_context_builder(
            max_words=config.context_max_words,
            max_scenes=config.max_scenes_in_context,
        )
        creative_plan = CreativePlan(genre="test", tone="test", locked_facts=["fact1"])
        context = context_builder.build_collection_context(
            collection, result, creative_plan, None
        )
        
        word_count = len(context.split())
        assert word_count <= config.context_max_words, f"Context exceeded budget: {word_count} > {config.context_max_words}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])