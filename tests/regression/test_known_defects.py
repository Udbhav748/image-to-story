"""Regressions found and fixed during the repository reorganisation.

Each test here corresponds to a defect that existed before the restructure. They
are cheap and specific: if one of them fails, a bug that was deliberately fixed
has come back.
"""
import ast
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from image_story.domain.schemas import CacheEntry, PipelineConfig
from image_story.ingestion import VisionCache
from image_story.ingestion.dedup import Deduplicator


class TestCacheEntryIsADataclass:
    """`CacheEntry` was missing `@dataclass`, so `CacheEntry(...)` raised
    `TypeError: CacheEntry() takes no arguments` and the vision cache could not
    be written to at all."""

    def test_constructor_accepts_all_fields(self):
        entry = CacheEntry(
            cache_key="k",
            content_hash="h",
            model_name="m",
            model_version="v1",
            result={"scene": "a park"},
        )
        assert entry.cache_key == "k"
        assert entry.result == {"scene": "a park"}
        assert entry.access_count == 0

    def test_timestamps_default(self):
        entry = CacheEntry(cache_key="k", content_hash="h", model_name="m", model_version="v", result={})
        datetime.fromisoformat(entry.created_at)
        datetime.fromisoformat(entry.accessed_at)


class TestVisionCacheLookup:
    """`VisionCache.get()` ignored its arguments and scanned every cache file
    comparing against empty strings, so it could only ever miss."""

    def test_get_uses_its_arguments(self, tmp_path):
        cache = VisionCache(str(tmp_path / "c"))
        cache.put_result("hash_a", "florence2", "v1", {"scene": "a"})
        assert cache.get("hash_a", "florence2", "v1") == {"scene": "a"}
        assert cache.get("hash_a", "florence2", "v2") is None

    def test_clear_old_entries_actually_removes(self, tmp_path):
        """Deleting during a `rglob` iteration failed on Windows (WinError 32)
        and the bare `except` hid it, so cleanup removed nothing."""
        cache = VisionCache(str(tmp_path / "c"))
        cache.put(
            CacheEntry(
                cache_key="k" * 32,
                content_hash="h",
                model_name="m",
                model_version="v1",
                result={"scene": "a"},
                created_at=(datetime.now() - timedelta(days=90)).isoformat(),
            )
        )
        assert cache.clear_old_entries(max_age_days=30) == 1
        assert cache.get_stats()["entry_count"] == 0

    def test_clear_keeps_recent_used_entries(self, tmp_path):
        cache = VisionCache(str(tmp_path / "c"))
        cache.put_result("hash_a", "florence2", "v1", {"scene": "a"})
        cache.get("hash_a", "florence2", "v1")  # access_count becomes 1
        assert cache.clear_old_entries(max_age_days=30) == 0
        assert cache.get("hash_a", "florence2", "v1") == {"scene": "a"}


class TestDeduplicatorRegistration:
    """`Deduplicator.register_image` was defined twice in `ingestion/hashing.py`
    and the first definition was shadowed."""

    def test_only_one_registration_method_exists(self):
        import inspect


        source = inspect.getsource(Deduplicator)
        assert source.count("def register_image") == 1
        assert source.count("def register(") == 0

    def test_exact_duplicate_is_reported(self):
        dedup = Deduplicator()
        dedup.register_image("img_1", "hash_a", "phash_a")
        duplicate, _ = dedup.register_image("img_2", "hash_a", "phash_a")
        assert duplicate == "img_1"

    def test_deduplicate_pipeline_counts_exact_duplicates(self, tmp_path):
        """`IngestionPipeline.deduplicate()` never incremented `exact_duplicates`
        (it relied on a `register_image` return value that never arrived)."""
        from PIL import Image

        from image_story.ingestion import IngestionPipeline

        a, b = tmp_path / "a.png", tmp_path / "b.png"
        Image.new("RGB", (64, 48), color=(10, 120, 200)).save(a)
        b.write_bytes(a.read_bytes())

        pipeline = IngestionPipeline()
        collection = pipeline.create_collection_from_paths([str(a), str(b)])
        pipeline.validate_images(collection)
        pipeline.compute_hashes(collection)
        exact, _ = pipeline.deduplicate(collection)
        assert exact == 1


class TestResumeActuallyResumes:
    """`process_collection_with_resume` marked images "skipped", then
    `validate_images` re-validated them and reset the status, so resume
    reprocessed everything it was supposed to skip."""

    def test_skipped_images_are_not_revalidated(self, tmp_path):
        from PIL import Image

        from image_story.ingestion import IngestionPipeline

        path = tmp_path / "frame.png"
        Image.new("RGB", (64, 48)).save(path)

        pipeline = IngestionPipeline()
        collection = pipeline.create_collection_from_paths([str(path)])
        image = collection.images[0]
        image.processing_status = "completed"
        image.validation_status = "valid"
        image.content_hash = "already"
        image.perceptual_hash = "already"

        pipeline.process_collection_with_resume(collection, session=None)
        assert image.processing_status == "skipped"
        assert image.content_hash == "already"


class TestRetryReportsItsCount:
    """`retry_failed_images` discarded the number of images it retried."""

    def test_retry_count_is_reported(self, tmp_path):
        from unittest.mock import patch

        from PIL import Image

        from image_story.domain.schemas import ExperimentManifest, PipelineArtifacts, StoryDraft
        from image_story.ingestion import CollectionPipeline
        from image_story.pipeline.orchestrator import PipelineOrchestrator

        path = tmp_path / "frame.png"
        Image.new("RGB", (64, 48)).save(path)

        artifacts = PipelineArtifacts(run_id="stub", manifest=ExperimentManifest())
        artifacts.story_draft = StoryDraft(text="A stub story.")

        pipeline = CollectionPipeline()
        collection = pipeline.create_collection_from_paths([str(path)])
        collection.images[0].processing_status = "failed"
        collection.images[0].error = "boom"

        with patch.object(PipelineOrchestrator, "run_multi_image", return_value=artifacts):
            result = pipeline.retry_failed_images(collection)
        assert result["retried"] == 1


class TestCollectionStatusFilter:
    """`IngestionPipeline.process_collection()` counted exact duplicates as
    `img.validation_status == "duplicate"`, which it happened to get right, but
    read a field the wrong way round next to `validator_status`; the consolidation
    pins the correct field name."""

    def test_duplicate_summary_uses_validation_status(self, tmp_path):
        from PIL import Image

        from image_story.ingestion import IngestionPipeline

        a, b = tmp_path / "a.png", tmp_path / "b.png"
        Image.new("RGB", (64, 48), color=(9, 9, 200)).save(a)
        b.write_bytes(a.read_bytes())

        pipeline = IngestionPipeline()
        collection = pipeline.create_collection_from_paths([str(a), str(b)])
        summary = pipeline.process_collection(collection)
        assert summary["duplicates_exact"] == 1
        assert collection.images[1].validation_status == "duplicate"


class TestModePresetsAreData:
    """`PipelineConfig.from_mode()` was a 130-line if/elif chain. It now reads
    `config.defaults.MODE_PRESETS`; these checks confirm the values survived."""

    @pytest.mark.parametrize(
        "mode,expected",
        [
            ("fast", {"use_faiss": False, "target_story_words": 100}),
            ("standard", {"use_faiss": True, "target_story_words": 250}),
            ("full", {"use_ocr": True, "target_story_words": 300}),
            ("baseline", {"use_creative_planner": False, "target_story_words": 100}),
            ("collection", {"top_k_scenes": 10, "target_story_words": 300}),
        ],
    )
    def test_preset_values_unchanged(self, mode, expected):
        config = PipelineConfig.from_mode(mode)
        for key, value in expected.items():
            assert getattr(config, key) == value

    def test_every_mode_forbids_new_visual_objects(self):
        for mode in ("fast", "standard", "full", "baseline", "collection"):
            assert PipelineConfig.from_mode(mode).creative_budget["forbidden_visual"]["max"] == 0


class TestEvidenceTextProjection:
    """`_evidence_to_text` existed four times with three formats."""

    def test_only_the_shared_helper_remains(self):
        from pathlib import Path

        import image_story

        offenders = [
            path.name
            for path in Path(image_story.__file__).parent.rglob("*.py")
            if "def _evidence_to_text" in path.read_text(encoding="utf-8")
        ]
        assert offenders == []


class TestNoDuplicateEnums:
    """The collection/ingestion enums were each defined five times in one module."""

    def test_enums_defined_once_each(self):
        from image_story.domain import enums

        tree = ast.parse(Path(enums.__file__).read_text(encoding="utf-8"))
        names = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
        duplicates = [name for name, count in Counter(names).items() if count > 1]
        assert duplicates == []


class TestArtefactWriterIsInThePackage:
    """`CollectionPipeline` imported `save_artifacts` from the top-level
    `main_v2.py` script, so it broke the moment the entry point moved."""

    def test_artifacts_module_exists(self):
        from image_story.experiments.artifacts import save_artifacts

        assert callable(save_artifacts)

    def test_jobs_no_longer_imports_a_script(self):
        from pathlib import Path

        import image_story

        text = (
            Path(image_story.__file__).parent / "ingestion" / "jobs.py"
        ).read_text(encoding="utf-8")
        assert "main_v2" not in text
        assert "from main" not in text


class TestLibraryDoesNotPrint:
    """`IngestionPipeline.process_collection` used bare `print()`; it now logs.

    The CLI is exempt: printing to stdout is its job.
    """

    #: Modules allowed to print: the command-line surface and the benchmark report.
    ALLOWED_TO_PRINT = {"cli.py", "__main__.py", "benchmarks.py"}

    def test_library_modules_do_not_print(self):
        import image_story

        offenders = []
        for path in Path(image_story.__file__).parent.rglob("*.py"):
            if path.name in self.ALLOWED_TO_PRINT:
                continue
            for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if line.strip().startswith("print("):
                    offenders.append(f"{path.name}:{number}")
        assert offenders == []
