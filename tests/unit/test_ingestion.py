"""Ingestion: validation, hashing, deduplication, cache and the collection pipeline.

Covers the consolidation: `validator.py` no longer carries private copies of
`ImageHasher`/`Deduplicator`, `Deduplicator.register_image` exists once and
returns its result, and `VisionCache` looks entries up by key instead of
scanning every cache file against empty strings.
"""
from datetime import datetime, timedelta

import pytest
from PIL import Image

from image_story.domain.schemas import CacheEntry, ImageCollection, ImageRecord, ProcessingConfig
from image_story.ingestion import (
    CollectionPipeline,
    Deduplicator,
    ImageHasher,
    ImageValidator,
    IngestionPipeline,
    VisionCache,
)
from image_story.ingestion.dedup import Deduplicator as CanonicalDeduplicator
from image_story.ingestion.hashing import ImageHasher as CanonicalHasher
from image_story.ingestion.validator import ImageValidator as CanonicalValidator


@pytest.fixture
def image_file(tmp_path):
    path = tmp_path / "frame.png"
    Image.new("RGB", (64, 48), color=(10, 120, 200)).save(path)
    return path


class TestSingleDefinition:
    def test_validator_has_no_private_hasher_or_deduplicator(self):
        import image_story.ingestion.validator as validator

        assert not hasattr(validator, "ImageHasher")
        assert not hasattr(validator, "Deduplicator")
        assert ImageValidator is CanonicalValidator
        assert ImageHasher is CanonicalHasher
        assert Deduplicator is CanonicalDeduplicator


class TestImageValidator:
    def test_accepts_a_valid_image(self, image_file):
        record = ImageRecord(path=str(image_file))
        status, error = ImageValidator().validate(record)
        assert status == "valid"
        assert error == ""
        assert record.width == 64
        assert record.height == 48

    def test_missing_file_is_invalid(self, tmp_path):
        status, error = ImageValidator().validate(ImageRecord(path=str(tmp_path / "nope.png")))
        assert status == "invalid"
        assert "not found" in error

    def test_unsupported_extension_is_invalid(self, tmp_path):
        path = tmp_path / "frame.txt"
        path.write_text("not an image")
        status, error = ImageValidator().validate(ImageRecord(path=str(path)))
        assert status == "invalid"
        assert "Unsupported format" in error

    def test_zero_byte_file_is_invalid(self, tmp_path):
        path = tmp_path / "frame.png"
        path.write_bytes(b"")
        status, error = ImageValidator().validate(ImageRecord(path=str(path)))
        assert status == "invalid"
        assert "Zero-byte" in error

    def test_too_small_is_invalid(self, tmp_path):
        path = tmp_path / "tiny.png"
        Image.new("RGB", (8, 8)).save(path)
        status, error = ImageValidator(min_dimension=32).validate(ImageRecord(path=str(path)))
        assert status == "invalid"
        assert "below minimum" in error

    def test_corrupt_file_is_invalid(self, tmp_path):
        path = tmp_path / "broken.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"garbage" * 40)
        status, error = ImageValidator().validate(ImageRecord(path=str(path)))
        assert status == "invalid"


class TestImageHasher:
    def test_content_hash_is_sha256_of_the_file(self, image_file):
        import hashlib

        expected = hashlib.sha256(image_file.read_bytes()).hexdigest()
        assert ImageHasher().compute_content_hash(str(image_file)) == expected

    def test_content_hash_differs_per_file(self, image_file, tmp_path):
        other = tmp_path / "other.png"
        Image.new("RGB", (64, 48), color=(200, 10, 10)).save(other)
        hasher = ImageHasher()
        assert hasher.compute_content_hash(str(image_file)) != hasher.compute_content_hash(str(other))

    def test_perceptual_hash_is_stable_for_identical_content(self, tmp_path):
        a, b = tmp_path / "a.png", tmp_path / "b.png"
        Image.new("RGB", (64, 48), color=(10, 120, 200)).save(a)
        Image.new("RGB", (64, 48), color=(10, 120, 200)).save(b)
        hasher = ImageHasher()
        assert hasher.compute_perceptual_hash(str(a)) == hasher.compute_perceptual_hash(str(b))

    def test_compute_hashes_returns_both(self, image_file):
        content_hash, perceptual_hash = ImageHasher().compute_hashes(
            ImageRecord(path=str(image_file))
        )
        assert len(content_hash) == 64
        assert perceptual_hash


class TestDeduplicator:
    def test_register_image_returns_duplicate_info(self):
        """The shadowed first definition used to discard this entirely."""
        dedup = Deduplicator()
        first, similar = dedup.register_image("img_1", "hash_a", "phash_a")
        assert first is None
        assert similar == []

        duplicate, _ = dedup.register_image("img_2", "hash_a", "phash_a")
        assert duplicate == "img_1"

    def test_registration_is_keyed_by_content_hash(self):
        dedup = Deduplicator()
        dedup.register_image("img_1", "hash_a", "phash_a")
        assert dedup.check_exact_duplicate("hash_a") == "img_1"
        assert dedup.check_exact_duplicate("hash_b") is None

    def test_near_duplicates_are_found_by_perceptual_hash(self):
        dedup = Deduplicator(perceptual_threshold=4)
        dedup.register_image("img_1", "hash_a", "0000" * 16)
        _, similar = dedup.register_image("img_2", "hash_b", "0001" + "0" * 62)
        assert [img_id for img_id, _ in similar] == ["img_1"]

    def test_distant_images_are_not_near_duplicates(self):
        dedup = Deduplicator(perceptual_threshold=2)
        dedup.register_image("img_1", "hash_a", "0" * 64)
        _, similar = dedup.register_image("img_2", "hash_b", "1" * 64)
        assert similar == []

    def test_an_image_is_not_its_own_near_duplicate(self):
        dedup = Deduplicator(perceptual_threshold=64)
        _, similar = dedup.register_image("img_1", "hash_a", "0" * 64)
        assert similar == []

    def test_remove_image_clears_both_maps(self):
        dedup = Deduplicator()
        dedup.register_image("img_1", "hash_a", "phash_a")
        dedup.remove_image("img_1")
        assert dedup.check_exact_duplicate("hash_a") is None
        assert dedup.check_near_duplicates("phash_a") == []


class TestIngestionPipeline:
    def test_creates_a_collection_in_order(self, image_file):
        pipeline = IngestionPipeline()
        collection = pipeline.create_collection_from_paths([str(image_file)])
        assert isinstance(collection, ImageCollection)
        assert collection.total_images == 1
        assert collection.images[0].sequence_index == 0

    def test_filename_ordering_is_applied(self, tmp_path):
        paths = []
        for name in ("c.png", "a.png", "b.png"):
            path = tmp_path / name
            Image.new("RGB", (64, 48)).save(path)
            paths.append(str(path))

        collection = IngestionPipeline().create_collection_from_paths(paths, "filename")
        assert [img.path for img in collection.images] == sorted(paths)
        assert [img.sequence_index for img in collection.images] == [0, 1, 2]

    def test_validate_marks_good_images_valid(self, image_file):
        pipeline = IngestionPipeline()
        collection = pipeline.create_collection_from_paths([str(image_file)])
        valid, invalid, skipped = pipeline.validate_images(collection)
        assert (valid, invalid, skipped) == (1, 0, 0)
        assert collection.images[0].validation_status == "valid"

    def test_validate_counts_an_invalid_image(self, tmp_path, image_file):
        pipeline = IngestionPipeline()
        collection = pipeline.create_collection_from_paths(
            [str(image_file), str(tmp_path / "missing.png")]
        )
        valid, invalid, _ = pipeline.validate_images(collection)
        assert valid == 1
        assert invalid == 1

    def test_hashes_are_computed_for_valid_images(self, image_file):
        pipeline = IngestionPipeline()
        collection = pipeline.create_collection_from_paths([str(image_file)])
        pipeline.validate_images(collection)
        computed, skipped = pipeline.compute_hashes(collection)
        assert computed == 1
        assert skipped == 0
        assert collection.images[0].content_hash

    def test_exact_duplicates_are_counted_and_marked(self, tmp_path):
        a, b = tmp_path / "a.png", tmp_path / "b.png"
        Image.new("RGB", (64, 48), color=(10, 120, 200)).save(a)
        a.read_bytes()
        b.write_bytes(a.read_bytes())

        pipeline = IngestionPipeline()
        collection = pipeline.create_collection_from_paths([str(a), str(b)])
        pipeline.validate_images(collection)
        pipeline.compute_hashes(collection)
        exact, near = pipeline.deduplicate(collection)

        assert exact == 1
        assert near == 0
        assert collection.images[1].validation_status == "duplicate"
        assert collection.images[1].duplicate_of == collection.images[0].image_id

    def test_process_collection_summarises(self, image_file):
        pipeline = IngestionPipeline()
        collection = pipeline.create_collection_from_paths([str(image_file)])
        summary = pipeline.process_collection(collection)
        assert summary["total_images"] == 1
        assert summary["valid"] == 1
        assert summary["duplicates_exact"] == 0
        assert "duration_seconds" in summary

    def test_resume_skips_completed_images(self, image_file):
        pipeline = IngestionPipeline(processing_config=ProcessingConfig(resume=True))
        collection = pipeline.create_collection_from_paths([str(image_file)])
        collection.images[0].processing_status = "completed"
        collection.images[0].validation_status = "valid"
        pipeline.process_collection_with_resume(collection, session=None)
        assert collection.images[0].processing_status == "skipped"


class TestVisionCache:
    def test_miss_returns_none(self, tmp_path):
        assert VisionCache(str(tmp_path / "c")).get("hash", "model", "v1") is None

    def test_put_then_get_round_trips(self, tmp_path):
        cache = VisionCache(str(tmp_path / "c"))
        cache.put_result("hash_a", "florence2", "v1", {"scene": "a park"})
        assert cache.get("hash_a", "florence2", "v1") == {"scene": "a park"}

    def test_different_model_version_is_a_different_entry(self, tmp_path):
        cache = VisionCache(str(tmp_path / "c"))
        cache.put_result("hash_a", "florence2", "v1", {"scene": "v1"})
        assert cache.get("hash_a", "florence2", "v2") is None

    def test_different_content_hash_is_a_different_entry(self, tmp_path):
        cache = VisionCache(str(tmp_path / "c"))
        cache.put_result("hash_a", "florence2", "v1", {"scene": "a"})
        assert cache.get("hash_b", "florence2", "v1") is None

    def test_get_by_content_hash_matches_get(self, tmp_path):
        cache = VisionCache(str(tmp_path / "c"))
        cache.put_result("hash_a", "florence2", "v1", {"scene": "a"})
        assert cache.get_by_content_hash("hash_a", "florence2", "v1") == {"scene": "a"}

    def test_access_count_increments_on_read(self, tmp_path):
        cache = VisionCache(str(tmp_path / "c"))
        cache.put_result("hash_a", "florence2", "v1", {"scene": "a"})
        key = cache._make_cache_key("hash_a", "florence2", "v1")
        assert cache.get_by_key(key).access_count == 1
        assert cache.get_by_key(key).access_count == 2

    def test_entries_are_sharded_by_key_prefix(self, tmp_path):
        cache = VisionCache(str(tmp_path / "c"))
        cache.put_result("hash_a", "florence2", "v1", {"scene": "a"})
        files = list((tmp_path / "c").rglob("*.json"))
        assert len(files) == 1
        assert files[0].parent.name == files[0].stem[:2]

    def test_corrupt_entry_is_treated_as_a_miss(self, tmp_path):
        cache = VisionCache(str(tmp_path / "c"))
        cache.put_result("hash_a", "florence2", "v1", {"scene": "a"})
        key = cache._make_cache_key("hash_a", "florence2", "v1")
        cache._get_cache_path(key).write_text("{not json", encoding="utf-8")
        assert cache.get("hash_a", "florence2", "v1") is None

    def test_clear_old_entries_removes_stale_entries(self, tmp_path):
        cache = VisionCache(str(tmp_path / "c"))
        entry = CacheEntry(
            cache_key="k" * 32,
            content_hash="hash_a",
            model_name="florence2",
            model_version="v1",
            result={"scene": "a"},
            created_at=(datetime.now() - timedelta(days=90)).isoformat(),
        )
        cache.put(entry)
        assert cache.clear_old_entries(max_age_days=30) == 1
        assert cache.get_stats()["entry_count"] == 0

    def test_stats_report_entries_and_size(self, tmp_path):
        cache = VisionCache(str(tmp_path / "c"))
        cache.put_result("hash_a", "florence2", "v1", {"scene": "a"})
        stats = cache.get_stats()
        assert stats["entry_count"] == 1
        assert stats["total_size_mb"] > 0
        assert stats["cache_dir"].endswith("c")


class TestCollectionPipeline:
    def test_defaults_are_wired(self):
        pipeline = CollectionPipeline()
        assert pipeline.collection_config is not None
        assert isinstance(pipeline.ingestion, IngestionPipeline)

    def test_no_valid_images_is_reported_as_failure(self, tmp_path):
        pipeline = CollectionPipeline()
        collection = pipeline.create_collection_from_paths([str(tmp_path / "missing.png")])
        result = pipeline.process_collection(collection, evaluate=False)
        assert result["success"] is False
        assert "No valid images" in result["error"]

    def test_retry_resets_failed_images(self, image_file):
        """Retry resets the failed image, then reprocesses the collection.

        The orchestrator is stubbed: this test is about the retry bookkeeping,
        not about running models.
        """
        from unittest.mock import patch

        from image_story.domain.schemas import ExperimentManifest, PipelineArtifacts, StoryDraft
        from image_story.pipeline.orchestrator import PipelineOrchestrator

        artifacts = PipelineArtifacts(run_id="stub", manifest=ExperimentManifest())
        artifacts.story_draft = StoryDraft(text="A stub story.")

        pipeline = CollectionPipeline()
        collection = pipeline.create_collection_from_paths([str(image_file)])
        collection.images[0].processing_status = "failed"
        collection.images[0].error = "boom"

        with patch.object(
            PipelineOrchestrator, "run_multi_image", return_value=artifacts
        ):
            result = pipeline.retry_failed_images(collection)

        assert result["success"] is True
        assert result["retried"] == 1
        assert collection.images[0].metadata["retry_count"] == 1

    def test_retry_respects_the_attempt_budget(self, image_file):
        from unittest.mock import patch

        from image_story.domain.schemas import ExperimentManifest, PipelineArtifacts, StoryDraft
        from image_story.pipeline.orchestrator import PipelineOrchestrator

        artifacts = PipelineArtifacts(run_id="stub", manifest=ExperimentManifest())
        artifacts.story_draft = StoryDraft(text="A stub story.")

        pipeline = CollectionPipeline()
        collection = pipeline.create_collection_from_paths([str(image_file)])
        image = collection.images[0]
        image.processing_status = "failed"
        image.error = "boom"
        image.metadata["retry_count"] = 3  # default max_retries

        with patch.object(
            PipelineOrchestrator, "run_multi_image", return_value=artifacts
        ) as run:
            result = pipeline.retry_failed_images(collection)

        assert result["retried"] == 0
        assert not run.called

    def test_retry_is_a_no_op_without_failures(self, image_file):
        pipeline = CollectionPipeline()
        collection = pipeline.create_collection_from_paths([str(image_file)])
        assert pipeline.retry_failed_images(collection)["retried"] == 0
