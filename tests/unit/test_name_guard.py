"""Unit tests for the invented-name guard."""

from image_story.verification.name_guard import REPLACEMENT, guard_invented_names


def test_unsupported_name_is_replaced():
    text = "Thomas walked to the farm. He smiled."
    context = "Scene: a man walks past a farm. Characters: man."
    out, removed = guard_invented_names(text, context)
    assert "Thomas" not in out
    assert removed == ["Thomas"]
    assert out.startswith("The figure walked")


def test_supported_name_is_kept():
    text = "Sarah waved at the car."
    context = "Characters: Sarah, car."
    out, removed = guard_invented_names(text, context)
    assert out == text
    assert removed == []


def test_common_words_are_never_replaced():
    text = "Suddenly, In the morning, The dog ran. Then It stopped."
    out, removed = guard_invented_names(text, "Objects: dog.")
    assert removed == []
    assert out == text


def test_mid_sentence_name_uses_lowercase_replacement():
    text = "She met Tom at noon."
    out, removed = guard_invented_names(text, "Characters: woman.")
    assert removed == ["Tom"]
    assert f"met {REPLACEMENT} at" in out


def test_empty_context_changes_nothing():
    text = "Thomas walked."
    assert guard_invented_names(text, "") == (text, [])
