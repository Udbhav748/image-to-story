"""Tests for narrative memory and collection context builder."""
import pytest
import numpy as np

from image_story.memory.narrative_memory import NarrativeMemory, NarrativeMemoryRetriever, create_narrative_memory
from image_story.context.collection_builder import CollectionContextBuilder, CollectionContextConfig, create_collection_context_builder
from image_story.memory.hierarchical import (
    SceneSummary, CollectionMemory, EntityMemory, StateTransition, NarrativeElement, RetrievalResult
)
from image_story.domain.schemas import CreativePlan, EvidenceRecord, EvidenceType, SourceModel, InformationClass
from image_story.domain.enums import StoryGenre, StoryTone


class TestNarrativeMemory:
    def setup_method(self):
        self.narrative_memory = NarrativeMemory()
    
    def test_build_recurring_elements(self):
        entity_memories = {
            "ent_1": EntityMemory(
                entity_id="ent_1",
                normalized_label="girl",
                entity_type="character",
                scene_ids=["scene_1", "scene_2", "scene_3"],
                evidence_ids=["ev1", "ev2", "ev3"],
            ),
            "ent_2": EntityMemory(
                entity_id="ent_2",
                normalized_label="balloon",
                entity_type="object",
                scene_ids=["scene_1"],
                evidence_ids=["ev4"],
            ),
        }
        
        scenes = [
            SceneSummary(scene_id="scene_1", start_frame=0, end_frame=0),
            SceneSummary(scene_id="scene_2", start_frame=1, end_frame=1),
            SceneSummary(scene_id="scene_3", start_frame=2, end_frame=2),
        ]
        
        elements = self.narrative_memory.build_narrative_elements(
            entity_memories, scenes, [], [], None
        )
        
        # Should create recurring element for girl (3 scenes)
        recurring = [e for e in elements if e.element_type == "recurring_entity" and e.label == "girl"]
        assert len(recurring) == 1
        assert recurring[0].recurrence_count == 3
        assert recurring[0].potential_callback is True
        
        # Balloon only in 1 scene, no recurring element
        balloon_recurring = [e for e in elements if e.element_type == "recurring_entity" and e.label == "balloon"]
        assert len(balloon_recurring) == 0
    
    def test_build_transition_elements(self):
        transitions = [
            StateTransition(
                entity_id="ent_1",
                entity_label="balloon",
                transition_type="disappeared",
                from_scene="scene_1",
                to_scene="",
                description="Balloon disappeared after scene 1",
                confidence=0.8,
            ),
            StateTransition(
                entity_id="ent_2",
                entity_label="girl",
                transition_type="moved",
                from_scene="scene_1",
                to_scene="scene_2",
                description="Girl moved significantly",
                confidence=0.7,
            ),
        ]
        
        scenes = [
            SceneSummary(scene_id="scene_1"),
            SceneSummary(scene_id="scene_2"),
        ]
        
        elements = self.narrative_memory.build_narrative_elements(
            {}, scenes, transitions, [], None
        )
        
        # Should create elements for disappearance and relationship/state changes
        disappearance = [e for e in elements if e.element_type == "object_disappearance"]
        assert len(disappearance) == 1
        assert "balloon" in disappearance[0].label
        
        # Movement doesn't create element by default (only specific types)
    
    def test_build_scene_elements(self):
        scenes = [
            SceneSummary(
                scene_id="scene_1",
                dominant_entities=["girl"],
                important_objects=["balloon", "kite"],
                start_frame=0, end_frame=0,
            )
        ]
        
        elements = self.narrative_memory.build_narrative_elements(
            {}, scenes, [], [], None
        )
        
        # Should create scene character and object elements
        scene_chars = [e for e in elements if e.element_type == "scene_character"]
        scene_objs = [e for e in elements if e.element_type == "scene_object"]
        
        assert len(scene_chars) == 1
        assert scene_chars[0].label == "girl"
        
        assert len(scene_objs) >= 1
        assert "balloon" in [e.label for e in scene_objs]
    
    def test_build_open_loop_elements(self):
        transitions = [
            StateTransition(
                entity_id="ent_1",
                entity_label="balloon",
                transition_type="disappeared",
                from_scene="scene_1",
                to_scene="",
                description="Balloon disappeared",
            ),
        ]
        
        scenes = [SceneSummary(scene_id="scene_1")]
        
        elements = self.narrative_memory.build_narrative_elements(
            {}, scenes, transitions, [], None
        )
        
        open_loops = [e for e in elements if e.element_type == "open_loop"]
        assert len(open_loops) == 1
        assert "balloon" in open_loops[0].label
        assert open_loops[0].potential_callback is True
    
    def test_score_narrative_importance(self):
        # Create elements with different characteristics
        elem1 = NarrativeElement(
            element_id="n1", element_type="recurring_entity",
            label="girl", description="Main character",
            first_scene="s1", scene_ids=["s1", "s2", "s3"],
            recurrence_count=3, narrative_importance=0.0,
            associated_entities=["girl"],
        )
        elem2 = NarrativeElement(
            element_id="n2", element_type="scene_object",
            label="kite", description="Kite in scene",
            first_scene="s1", scene_ids=["s1"],
            recurrence_count=1, narrative_importance=0.0,
            associated_entities=["kite"],
        )
        elem3 = NarrativeElement(
            element_id="n3", element_type="open_loop",
            label="missing balloon", description="Balloon gone",
            first_scene="s2", scene_ids=["s2"],
            recurrence_count=1, narrative_importance=0.0,
            associated_entities=["balloon"],
        )
        
        scenes = [
            SceneSummary(scene_id="s1"),
            SceneSummary(scene_id="s2"),
            SceneSummary(scene_id="s3"),
        ]
        
        elements = [elem1, elem2, elem3]
        self.narrative_memory._score_narrative_importance(elements, scenes, [])
        
        # Open loop should get bonus
        assert elem3.narrative_importance > 0.2
        # All elements should have some score
        assert elem1.narrative_importance > 0
        assert elem2.narrative_importance > 0
        assert elem3.narrative_importance > 0
    
    def test_mark_callback_potential(self):
        elem1 = NarrativeElement(
            element_id="n1", element_type="recurring_entity",
            label="girl", description="Main character",
            first_scene="s1", scene_ids=["s1", "s3"],
            recurrence_count=2, narrative_importance=0.8,
            potential_callback=False,
        )
        elem2 = NarrativeElement(
            element_id="n2", element_type="scene_object",
            label="kite", description="Kite in scene",
            first_scene="s1", scene_ids=["s1"],
            recurrence_count=1, narrative_importance=0.3,
            potential_callback=False,
        )
        
        scenes = [
            SceneSummary(scene_id="s1"),
            SceneSummary(scene_id="s2"),
            SceneSummary(scene_id="s3"),
        ]
        
        elements = [elem1, elem2]
        self.narrative_memory._mark_callback_potential(elements, scenes)
        
        # Girl appears in s1 and s3, so from s1 perspective, later appearance in s3
        assert elem1.potential_callback is True
        assert elem1.callback_payoff_scene == "s3"
        
        # Kite only in s1, no later appearance
        assert elem2.potential_callback is False


class TestNarrativeMemoryRetriever:
    def setup_method(self):
        self.elem1 = NarrativeElement(
            element_id="n1", element_type="recurring_entity",
            label="girl", description="Main character appears throughout",
            first_scene="s1", scene_ids=["s1", "s2", "s3"],
            narrative_importance=0.9,
            associated_entities=["girl"],
            potential_callback=True,
            callback_payoff_scene="s3",
        )
        self.elem2 = NarrativeElement(
            element_id="n2", element_type="open_loop",
            label="missing balloon", description="Balloon disappeared in s1",
            first_scene="s1", scene_ids=["s1"],
            narrative_importance=0.8,
            associated_entities=["balloon"],
            potential_callback=True,
            callback_payoff_scene=None,
        )
        self.elem3 = NarrativeElement(
            element_id="n3", element_type="scene_object",
            label="bicycle", description="Bicycle in s2",
            first_scene="s2", scene_ids=["s2"],
            narrative_importance=0.4,
            associated_entities=["bicycle"],
        )
        
        self.retriever = NarrativeMemoryRetriever([self.elem1, self.elem2, self.elem3])
    
    def test_retrieve_for_scene(self):
        elements = self.retriever.retrieve_for_scene("s2")
        
        # Should include elements from s2 and potential callbacks from earlier
        assert len(elements) >= 1
        # Should be sorted by importance
        assert elements[0].narrative_importance >= elements[-1].narrative_importance
    
    def test_retrieve_callback_candidates(self):
        candidates = self.retriever.retrieve_callback_candidates("s3", "balloon")
        
        # Should include open loop from s1 (balloon)
        labels = [c.label for c in candidates]
        assert "missing balloon" in labels
    
    def test_retrieve_by_type(self):
        recurring = self.retriever.retrieve_by_type("recurring_entity")
        assert len(recurring) == 1
        assert recurring[0].label == "girl"
        
        open_loops = self.retriever.retrieve_by_type("open_loop")
        assert len(open_loops) == 1
        assert open_loops[0].label == "missing balloon"
    
    def test_retrieve_by_entity(self):
        girl_elements = self.retriever.retrieve_by_entity("girl")
        assert len(girl_elements) == 1
        assert girl_elements[0].label == "girl"
        
        balloon_elements = self.retriever.retrieve_by_entity("balloon")
        assert len(balloon_elements) == 1
        assert balloon_elements[0].label == "missing balloon"
    
    def test_get_open_loops(self):
        open_loops = self.retriever.get_open_loops()
        
        assert len(open_loops) == 1
        assert open_loops[0].element_type == "open_loop"
        assert open_loops[0].callback_payoff_scene is None


class TestCollectionContextBuilder:
    def setup_method(self):
        self.config = CollectionContextConfig(
            max_scenes_in_context=3,
            max_evidence_per_scene=5,
            max_total_words=500,
        )
        self.builder = CollectionContextBuilder(max_words=500, config=self.config)
    
    def test_build_collection_context(self):
        from image_story.domain.schemas import EvidenceRecord, EvidenceType, SourceModel, InformationClass
        
        scene1 = SceneSummary(
            scene_id="scene_1",
            start_frame=0, end_frame=2,
            image_ids=["img_0", "img_1", "img_2"],
            dominant_entities=["girl"],
            important_objects=["balloon"],
            actions=["holding", "walking"],
            location="park",
            environment="peaceful",
            recurring_entities=["girl", "balloon"],
            key_evidence_ids=["ev1", "ev2"],
            summary_text="Girl with balloon in park",
        )
        scene2 = SceneSummary(
            scene_id="scene_2",
            start_frame=3, end_frame=3,
            image_ids=["img_3"],
            dominant_entities=["boy"],
            important_objects=["bicycle"],
            location="street",
            summary_text="Boy riding bicycle",
        )
        
        collection = CollectionMemory(
            collection_id="coll_1",
            scene_summaries=[scene1, scene2],
            total_images=4,
            total_evidence=10,
        )
        
        evidence = [
            EvidenceRecord(entity="girl", type=EvidenceType.PERSON, frame_id=0, confidence=0.9,
                         source=SourceModel.FLORENCE2, information_class=InformationClass.HARD_FACT),
            EvidenceRecord(entity="balloon", type=EvidenceType.OBJECT, frame_id=0, confidence=0.8,
                         source=SourceModel.GROUNDING_DINO, information_class=InformationClass.HARD_FACT),
        ]
        
        retrieval = RetrievalResult(
            scenes=[scene1, scene2],
            entities=[
                EntityMemory(entity_id="e1", normalized_label="girl", entity_type="character",
                           scene_ids=["scene_1", "scene_2"], confidence=0.9),
                EntityMemory(entity_id="e2", normalized_label="balloon", entity_type="object",
                           scene_ids=["scene_1"], confidence=0.8),
            ],
            transitions=[
                StateTransition(entity_id="e1", entity_label="girl", transition_type="appeared",
                              from_scene="", to_scene="scene_1", description="Girl appeared"),
            ],
            narrative_elements=[
                NarrativeElement(element_id="n1", element_type="recurring_entity",
                               label="girl", description="Main character",
                               first_scene="scene_1", scene_ids=["scene_1", "scene_2"],
                               narrative_importance=0.9, potential_callback=True),
            ],
            evidence=evidence,
        )
        
        creative_plan = CreativePlan(
            genre="whimsical",
            tone="comedic",
            locked_facts=["girl", "balloon", "park"],
        )
        
        context = self.builder.build_collection_context(
            collection, retrieval, creative_plan, current_scene_id="scene_1"
        )
        
        assert "COLLECTION OVERVIEW" in context
        assert "Total scenes: 2" in context
        assert "Total images: 4" in context
        assert "RELEVANT SCENES" in context
        assert "scene_1" in context
        assert "CURRENT SCENE DETAIL" in context
        assert "KEY ENTITIES" in context
        assert "STATE TRANSITIONS" in context
        assert "NARRATIVE ELEMENTS" in context
        assert "HARD FACTS" in context
        assert "CREATIVE DIRECTION" in context
        assert "LOCKED FACTS" in context
    
    def test_build_multi_scene_context(self):
        scenes = [
            SceneSummary(
                scene_id="scene_1", start_frame=0, end_frame=1,
                image_ids=["img_0", "img_1"],
                dominant_entities=["girl"],
                important_objects=["balloon"],
                location="park",
                summary_text="Girl with balloon",
            ),
            SceneSummary(
                scene_id="scene_2", start_frame=2, end_frame=2,
                image_ids=["img_2"],
                dominant_entities=["boy"],
                important_objects=["bicycle"],
                location="street",
                summary_text="Boy on bicycle",
            ),
        ]
        
        entities = [
            EntityMemory(entity_id="e1", normalized_label="girl", entity_type="character",
                       scene_ids=["scene_1"], confidence=0.9),
            EntityMemory(entity_id="e2", normalized_label="balloon", entity_type="object",
                       scene_ids=["scene_1"], confidence=0.8),
        ]
        
        evidence = [
            EvidenceRecord(entity="girl", type=EvidenceType.PERSON, frame_id=0, confidence=0.9,
                         source=SourceModel.FLORENCE2, information_class=InformationClass.HARD_FACT),
            EvidenceRecord(entity="balloon", type=EvidenceType.OBJECT, frame_id=0, confidence=0.8,
                         source=SourceModel.GROUNDING_DINO, information_class=InformationClass.HARD_FACT),
        ]
        
        context = self.builder.build_multi_scene_context(scenes, entities, evidence)
        
        assert "SCENE SEQUENCE" in context
        assert "SCENE 1" in context
        assert "SCENE 2" in context
        assert "CONTINUITY NOTES" in context
        assert "ENTITIES" in context
        assert "KEY EVIDENCE" in context
    
    def test_single_image_fallback(self):
        from image_story.domain.schemas import VisualObservations
        
        obs = VisualObservations(
            image_id="img_1",
            frame_id=0,
            scene="beach",
            characters=["dog"],
            od_labels=["frisbee"],
            actions=["running"],
            style_or_mood="sunny",
        )
        
        context = self.builder.build_single_image_fallback(obs)
        
        assert "Scene: beach" in context
        assert "Characters: dog" in context
        assert "Objects: frisbee" in context
        assert "Actions: running" in context
        assert "Style: sunny" in context


class TestCreateFunctions:
    def test_create_narrative_memory(self):
        nm = create_narrative_memory()
        assert isinstance(nm, NarrativeMemory)
    
    def test_create_collection_context_builder(self):
        builder = create_collection_context_builder(max_words=600, max_scenes=3)
        assert isinstance(builder, CollectionContextBuilder)
        assert builder._config.max_total_words == 600
        assert builder._config.max_scenes_in_context == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])