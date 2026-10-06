"""Pytest configuration and shared fixtures.

`tests/conftest.py` sits one level below the repository root, so `src` is a
sibling of `tests`. This path insert lets the suite run against the working
tree without installing the package; `pip install -e .` also works.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from image_story.domain.schemas import (  # noqa: E402
    BoundingBox,
    EvidenceRecord,
    EvidenceType,
    InformationClass,
    SourceModel,
    VisualObservations,
    WorldEntity,
    WorldState,
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


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Repository root, for tests that need benchmark images or configs."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def challenge_images(repo_root: Path) -> list[Path]:
    """The eight Challenge V1 evaluation frames."""
    images = sorted((repo_root / "benchmarks" / "challenge-8-images").glob("*.jpg"))
    images += sorted((repo_root / "benchmarks" / "challenge-8-images").glob("*.png"))
    return sorted(images)


@pytest.fixture(scope="session")
def golden_cases() -> list[Path]:
    """Golden evaluation cases shipped with the suite."""
    return sorted((Path(__file__).parent / "fixtures").glob("golden_case_*.json"))


# Pytest markers
def pytest_configure(config):
    config.addinivalue_line("markers", "unit: Unit tests (no model inference)")
    config.addinivalue_line(
        "markers", "integration: Integration tests (require models or heavy fixtures)"
    )
    config.addinivalue_line(
        "markers", "regression: Regression tests pinning previously observed behaviour"
    )
    config.addinivalue_line("markers", "benchmark: Scale benchmarks (deselected by default)")
    config.addinivalue_line("markers", "slow: Slow tests")
