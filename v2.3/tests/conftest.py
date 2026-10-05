"""Pytest configuration and fixtures."""
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from image_story.domain.schemas import (
    VisualObservations,
    EvidenceRecord,
    EvidenceType,
    SourceModel,
    InformationClass,
    WorldEntity,
    WorldState,
    BoundingBox,
)


@pytest.fixture
def sample_observation():
    """Create a sample visual observation for testing."""
    return VisualObservations(
        image_id="test_001",
        frame_id=0,
        scene="A peaceful park with a girl sitting on a bench",
        detailed_caption="A young girl sits on a wooden bench in a sunny park, holding a red balloon.",
        objects=["girl", "bench", "balloon", "tree", "grass"],
        od_labels=["girl", "bench", "balloon", "tree"],
        characters=["girl"],
        actions=["sitting", "holding"],
        spatial_relations=["on bench", "holding balloon"],
        region_descriptions=["girl sitting on bench", "red balloon in hand"],
        style_or_mood="peaceful",
        evidence_records=[
            EvidenceRecord(
                entity="girl",
                type=EvidenceType.PERSON,
                frame_id=0,
                confidence=0.95,
                source=SourceModel.FLORENCE2,
                evidence_text="Person detected: girl",
                information_class=InformationClass.HARD_FACT,
            ),
            EvidenceRecord(
                entity="red balloon",
                type=EvidenceType.OBJECT,
                frame_id=0,
                confidence=0.9,
                source=SourceModel.GROUNDING_DINO,
                evidence_text="Grounded detection: red balloon",
                information_class=InformationClass.HARD_FACT,
            ),
        ],
    )


@pytest.fixture
def sample_world_state():
    """Create a sample world state for testing."""
    state = WorldState()
    state.characters = [
        WorldEntity(
            id="c1",
            label="girl",
            entity_type="character",
            first_frame=0,
            last_frame=2,
            frames_present=[0, 1, 2],
            is_recurring=True,
        ),
        WorldEntity(
            id="c2",
            label="boy",
            entity_type="character",
            first_frame=1,
            last_frame=1,
            frames_present=[1],
        ),
    ]
    state.objects = [
        WorldEntity(
            id="o1",
            label="red balloon",
            entity_type="object",
            first_frame=0,
            last_frame=1,
            frames_present=[0, 1],
            is_recurring=True,
        ),
        WorldEntity(
            id="o2",
            label="backpack",
            entity_type="object",
            first_frame=0,
            last_frame=0,
            frames_present=[0],
            disappearance_frame=0,
        ),
    ]
    state.locations = [
        WorldEntity(
            id="l1",
            label="park",
            entity_type="location",
            first_frame=0,
            last_frame=2,
            frames_present=[0, 1, 2],
        ),
    ]
    state.open_loops = [
        {
            "type": "disappearance",
            "entity_label": "backpack",
            "description": "backpack disappeared after frame 0",
        },
    ]
    return state


@pytest.fixture
def sample_evidence_records():
    """Create sample evidence records for testing."""
    return [
        EvidenceRecord(
            entity="girl",
            type=EvidenceType.PERSON,
            frame_id=0,
            confidence=0.95,
            source=SourceModel.FLORENCE2,
            evidence_text="Person detected: girl",
            information_class=InformationClass.HARD_FACT,
        ),
        EvidenceRecord(
            entity="red balloon",
            type=EvidenceType.OBJECT,
            frame_id=0,
            confidence=0.9,
            source=SourceModel.GROUNDING_DINO,
            bbox=BoundingBox(100, 100, 150, 150),
            evidence_text="Grounded detection: red balloon at (100,100,150,150)",
            information_class=InformationClass.HARD_FACT,
        ),
        EvidenceRecord(
            entity="sitting",
            type=EvidenceType.ACTION,
            frame_id=0,
            confidence=0.75,
            source=SourceModel.FLORENCE2,
            evidence_text="Action detected: sitting",
            information_class=InformationClass.SOFT_INFERENCE,
        ),
    ]


# Pytest markers
def pytest_configure(config):
    config.addinivalue_line("markers", "unit: Unit tests (no model inference)")
    config.addinivalue_line("markers", "integration: Integration tests (requires models)")
    config.addinivalue_line("markers", "slow: Slow tests")