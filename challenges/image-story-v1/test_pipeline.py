"""Tests for the Image -> Story pipeline. No machine-specific paths: image tests use selftest_image.jpg.

Usage:
  python test_pipeline.py            # unit tests (no model inference)
  python test_pipeline.py --models   # also run model-dependent tests (CLIP/NLI scoring, Florence-2 vision)

Importing `metric` loads CLIP and the NLI model from the local Hugging Face cache, so the cache must exist
(run download_models.py once). The unit tests themselves run no model inference.
"""
import os, sys, csv, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SELFTEST_IMAGE = os.path.join(HERE, "selftest_image.jpg")

import baseline
import seeing
import context_builder
import metric
import main


# ---------------------------------------------------------------- unit tests (no model inference)

def test_split_sentences():
    assert metric.split_sentences("Hello. World!") == ["Hello.", "World!"]
    assert metric.split_sentences("No punctuation") == ["No punctuation"]
    assert metric.split_sentences("") == []


def test_grounding_formula_production_helper():
    """Calls the production helper that metric.score() uses."""
    clip_n, g = metric.grounding_from_components(0.225, 0.1, False)
    assert abs(clip_n - 0.5) < 1e-9
    assert abs(g - (0.4 * 0.5 + 0.4 * 0.9 + 0.2)) < 1e-9
    _, g_conflict = metric.grounding_from_components(0.225, 0.1, True)
    assert abs((g - g_conflict) - 0.2) < 1e-9          # a conflict removes exactly the 0.2 attribute term
    assert metric.grounding_from_components(0.05, 0.0, False)[0] == 0.0   # clip_n clamps at 0
    assert metric.grounding_from_components(0.90, 0.0, False)[0] == 1.0   # ...and at 1


def test_grounding_pass_definition():
    t = metric.THRESHOLD
    assert metric.is_grounding_pass(t, [], True)
    assert not metric.is_grounding_pass(t - 0.001, [], True)   # below threshold
    assert not metric.is_grounding_pass(0.95, ["blue"], True)  # attribute conflict blocks a pass
    assert not metric.is_grounding_pass(0.95, [], False)       # wrong length blocks a pass


def test_word_count_boundaries():
    """The production length rule: 79 invalid, 80 valid, 120 valid, 121 invalid."""
    story = lambda n: " ".join(["word"] * n)
    for n, expected in ((79, False), (80, True), (120, True), (121, False)):
        assert len(story(n).split()) == n
        assert baseline.is_length_valid(len(story(n).split())) is expected, n


def test_grounding_pass_combines_all_three_conditions():
    t = metric.THRESHOLD
    assert metric.is_grounding_pass(0.9, [], baseline.is_length_valid(100))        # valid length, good score, no conflict
    assert not metric.is_grounding_pass(0.9, [], baseline.is_length_valid(79))     # invalid length
    assert not metric.is_grounding_pass(0.9, ["oak"], baseline.is_length_valid(100))  # attribute conflict
    assert not metric.is_grounding_pass(t - 0.01, [], baseline.is_length_valid(100))  # score below threshold


def test_repetition_detection():
    r1 = metric.repetition_score("The cat sat on the mat. It purred softly.")
    assert r1["repetition_rate"] == 0.0 and r1["distinct3"] == 1.0 and r1["repeated_sentences"] == 0
    assert metric.repetition_score("The cat sat. The cat sat. The dog ran.")["repeated_sentences"] == 1
    r3 = metric.repetition_score("a b c d a b c d")
    assert r3["repeated_trigrams"] >= 1 and r3["repetition_rate"] > 0


def test_attribute_conflict_challenge_example():
    """The baseline failure from the challenge text: 'white' became 'oak'."""
    assert metric.attribute_conflict("the cabinets are white", "the old oak cabinet stood tall") == ["oak"]


def test_attribute_conflict_colours_and_materials():
    ac = metric.attribute_conflict
    cap = "a red wooden car"
    assert ac(cap, "The red wooden car drove fast.") == []                 # matching colour and material
    assert ac(cap, "The blue wooden car drove fast.") == ["blue"]          # conflicting colour
    assert ac(cap, "The red metal car drove fast.") == ["metal"]           # conflicting material
    assert ac("a wooden table", "An oak table stood there.") == []         # wood, wooden and oak are one family
    assert ac("a brown table", "An oak table stood there.") == []          # oak implies brown: compatible
    assert ac("a grey cat", "The gray cat slept.") == []                   # spelling variants of one colour
    assert ac("a car", "The blue metal car drove fast.") == []             # caption names no colour or material: nothing checked
    assert ac("a red car", "The car drove fast.") == []                    # story names nothing: no conflict


def test_normalize_entity_is_generic():
    n = context_builder.normalize_entity
    assert n("People") == "person" and n("children") == "child" and n("Windows") == "window"
    assert n("  human   faces ") == "human face"
    assert n("glass") == "glass"  # words ending in 'ss' are not singularised


def test_recurring_means_two_or_more_frames():
    """frame 1: person, car | frame 2: person, tree | frame 3: dog  -> only 'person' recurs."""
    frames = [{"characters": ["person"], "od_labels": ["car"]},
              {"characters": ["people"], "od_labels": ["tree"]},
              {"characters": [], "od_labels": ["dog"]}]
    ents = [context_builder.frame_entities(f) for f in frames]
    assert context_builder.recurring_entities(ents) == ["person"]


def test_continuity_not_applicable_for_single_image():
    c = metric.continuity_score([{"characters": ["cat"], "od_labels": ["mat"]}], "The cat sat on the mat.")
    assert c["entity_consistency"] is None and c["adjacent_similarity"] is None and c["transition_markers"] is None
    assert c["recurring_entities"] == []


def test_continuity_multi_frame():
    frames = [{"characters": ["cat"], "od_labels": ["mat"]}, {"characters": ["cat"], "od_labels": ["bowl"]}]
    c = metric.continuity_score(frames, "The cat sat on the mat then ate from the bowl.")
    assert c["recurring_entities"] == ["cat"] and c["entity_consistency"] == 1.0
    assert c["transition_markers"] == 1                         # "then"
    assert 0 < c["adjacent_similarity"] < 1                     # {cat} shared of {cat, mat, bowl}
    c2 = metric.continuity_score(frames, "Something happened somewhere.")
    assert c2["entity_consistency"] == 0.0                      # recurring entity never mentioned
    c3 = metric.continuity_score([{"characters": ["cat"], "od_labels": []}, {"characters": ["dog"], "od_labels": []}], "x")
    assert c3["entity_consistency"] is None                     # nothing recurs: not a vacuous 1.0


def test_transition_words_use_whole_words():
    frames = [{"characters": ["cat"], "od_labels": []}, {"characters": ["cat"], "od_labels": []}]
    assert metric.continuity_score(frames, "It was a casual afternoon, as usual.")["transition_markers"] == 0


def test_context_builder_single_is_deterministic_and_generic():
    desc = {"scene": "A scene", "characters": ["robot"], "od_labels": ["teapot", "zebra", "lamp"],
            "objects": ["teapot", "zebra", "lamp", "a red teapot on a shelf"], "actions": ["standing"],
            "spatial_relations": [], "region_descriptions": ["a robot standing"], "style_or_mood": "calm", "ocr_text": ""}
    c1, c2 = context_builder.build_single_image_context(desc), context_builder.build_single_image_context(desc)
    assert c1 == c2
    assert "Objects: teapot, zebra, lamp, a red teapot on a shelf" in c1   # arbitrary labels kept, detection order preserved
    for key in ("Scene:", "Characters:", "Actions:", "Region:", "Style:"):
        assert key in c1


def test_context_builder_sequence_lists_only_recurring():
    descs = [{"image_id": "leakyname_a.png", "scene": "S1", "characters": ["person"], "od_labels": ["car"], "objects": ["car"]},
             {"image_id": "leakyname_b.png", "scene": "S2", "characters": ["person"], "od_labels": ["tree"], "objects": ["tree"]},
             {"image_id": "leakyname_c.png", "scene": "S3", "characters": [], "od_labels": ["dog"], "objects": ["dog"]}]
    seq = context_builder.build_sequence_context(descs)
    assert "SEQUENCE OF IMAGES" in seq and "IMAGE 3" in seq and "CONTINUITY NOTES" in seq
    assert "leakyname" not in seq   # file names must not reach the story model (they leaked a character name before)
    notes = seq.split("CONTINUITY NOTES")[1]
    assert "Recurring characters (in 2+ frames): person" in notes
    assert "car" not in notes and "tree" not in notes and "dog" not in notes   # present in only one frame
    assert "continuous story" in context_builder.build_story_prompt(seq, 3, target_words=250)


def test_find_images_sorted_by_name():
    with tempfile.TemporaryDirectory() as d:
        for n in ("b.png", "a.jpg", "c.txt", "A2.png"):
            open(os.path.join(d, n), "wb").close()
        names = [os.path.basename(p) for p in main.find_images([d])]
        assert names == ["a.jpg", "A2.png", "b.png"]   # global name order, not grouped by extension


def test_csv_writer_handles_mixed_ok_and_error_rows():
    rows = [main._error_row("x.png", "A_baseline", "boom"),
            {"image": "y.png", "variant": "A_baseline", "status": "ok", "error": "", "grounding_score": 0.5}]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "r.csv")
        main.write_rows_csv(rows, path)
        back = list(csv.DictReader(open(path, encoding="utf-8")))
    assert [r["status"] for r in back] == ["error", "ok"]
    assert back[0]["grounding_score"] == "" and back[0]["error"] == "boom"   # no invented metric for the failed row


def test_failed_images_are_recorded_not_scored():
    """A failing image gives status=error rows without metrics, and the means ignore it."""
    def boom(*a, **k): raise RuntimeError("simulated failure")
    real = (baseline.run, seeing.describe)
    baseline.run, seeing.describe = boom, boom
    try:
        with tempfile.TemporaryDirectory() as d:
            csv_path, vis_path = os.path.join(d, "r.csv"), os.path.join(d, "custom_vision.json")
            rows, vision = main.run_comparison(["missing.jpg"], False, csv_path, vis_path)
            assert os.path.isfile(csv_path) and os.path.isfile(vis_path)   # the requested --output / --vision-json paths are used
    finally:
        baseline.run, seeing.describe = real
    assert [r["status"] for r in rows] == ["error", "error"] and vision == {}
    assert all("grounding_score" not in r for r in rows)
    s = main.summarize(rows)
    assert s["A_baseline"]["successful"] == 0 and s["A_baseline"]["failed"] == 1 and "mean_grounding" not in s["A_baseline"]


def test_vision_json_cli_option():
    parser = main.build_parser()
    assert parser.parse_args(["images/"]).vision_json == "vision.json"                          # default unchanged
    assert parser.parse_args(["images/", "--vision-json", "out/v.json"]).vision_json == "out/v.json"
    assert parser.parse_args(["images/", "--output", "r.csv"]).output == "r.csv"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "v.json")
        context_builder.save_vision_json([{"image_id": "a.png", "objects": ["x"], "od_labels": ["x"]}], path)
        import json
        assert json.load(open(path, encoding="utf-8"))["a.png"]["od_labels"] == ["x"]


def test_error_handling_missing_and_malformed_input():
    try:
        metric.score(os.path.join(HERE, "no_such_image.png"), "a cat", "A cat sat.")
        assert False, "should have raised"
    except FileNotFoundError:
        pass
    assert main.find_images(["no_such_dir_xyz"]) == []                       # nonexistent path
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "notes.txt"), "w").close()
        assert main.find_images([d]) == []                                    # directory with no images
    assert isinstance(context_builder.build_single_image_context({}), str)   # empty vision output does not crash
    assert context_builder.build_sequence_context([]) == ""
    assert metric.continuity_score(None, "x")["entity_consistency"] is None


def test_summary_averages_only_successful_rows():
    ok = lambda v, g: {"variant": v, "status": "ok", "grounding_score": g, "clip_image_story_mean": 0.2, "nli_contra_mean": 0.1,
                       "repetition_rate": 0.0, "length_valid": True, "grounding_pass": True, "see_s": 1.0, "story_s": 1.0,
                       "generation_s": 2.0, "eval_total_s": 0.5}
    rows = [ok("A_baseline", 0.6), ok("A_baseline", 0.8), main._error_row("z.png", "A_baseline", "boom")]
    s = main.summarize(rows)["A_baseline"]
    assert s["successful"] == 2 and s["failed"] == 1 and s["total"] == 3
    assert abs(s["mean_grounding"] - 0.7) < 1e-9
    assert s["mean_generation_s"] == 2.0 and s["mean_eval_total_s"] == 0.5   # generation and evaluation time kept apart


def test_fit_length_drops_cut_off_sentence_and_respects_limit():
    assert baseline._fit_length("One two three. Four five six. Seven eight") == "One two three. Four five six."
    long_story = " ".join(["word " * 10 + "end."] * 20)           # 20 sentences of 11 words
    assert len(baseline._fit_length(long_story, hi=120).split()) <= 120


def test_configuration():
    assert baseline.SEED == 0 and baseline.QWEN_ID == "Qwen/Qwen2.5-0.5B-Instruct"
    assert seeing.MODEL_ID == "florence-community/Florence-2-base" and seeing.NUM_BEAMS == 1
    assert os.environ.get("HF_HUB_OFFLINE") == "1" and os.path.isfile(SELFTEST_IMAGE)


def test_no_machine_specific_paths_in_sources():
    bad = ("D:" + "\\", "C:" + "\\Users", "AI-" + "Models")   # concatenated so this file does not match itself
    for name in ("main.py", "baseline.py", "seeing.py", "context_builder.py", "metric.py", "test_pipeline.py"):
        text = open(os.path.join(HERE, name), encoding="utf-8").read()
        for b in bad:
            assert b not in text, f"{name} contains machine-specific path {b!r}"


# ---------------------------------------------------------------- model-dependent tests (--models)

def _need_models():
    """Model-dependent tests run only with `--models` or RUN_MODEL_TESTS=1; under pytest they are otherwise skipped."""
    if "--models" not in sys.argv and os.environ.get("RUN_MODEL_TESTS") != "1":
        import unittest
        raise unittest.SkipTest("model-dependent test (use --models or RUN_MODEL_TESTS=1)")


def test_model_score_end_to_end():
    """metric.score() on the repository's own test image with synthetic strings."""
    _need_models()
    cap = "a brown cat sitting inside a blue bag"
    good = " ".join(["The brown cat sat quietly inside the blue bag and looked around the room."] * 1 +
                    ["Its ears twitched when someone walked past, and it settled down again."] * 1 +
                    ["The bag was warm and the cat was calm and sleepy in the afternoon light."] * 1)
    bad = "A tall castle stood above a frozen lake while wolves howled at the stormy sea. The sailor lit a candle."
    r_good, r_bad = metric.score(SELFTEST_IMAGE, cap, good), metric.score(SELFTEST_IMAGE, cap, bad)
    assert r_good["grounding_score"] > r_bad["grounding_score"]
    assert r_good["nli_contra_mean"] < r_bad["nli_contra_mean"]
    for key in ("entity_consistency", "adjacent_similarity", "transition_markers"):
        assert key not in r_good           # continuity is multi-image only
    for key in ("eval_clip_s", "eval_nli_s", "eval_rules_s", "eval_total_s"):
        assert r_good[key] >= 0            # evaluation runtime is reported separately from generation runtime
    assert r_good["eval_total_s"] >= r_good["eval_clip_s"]
    clip_n, g = metric.grounding_from_components(r_good["clip_image_story_mean"], r_good["nli_contra_mean"], bool(r_good["attribute_conflict"]))
    assert abs(g - r_good["grounding_score"]) < 1e-9   # score() really uses the production formula
    with tempfile.TemporaryDirectory() as d:           # evaluate_run writes file names only, never absolute paths
        out = os.path.join(d, "eval.csv")
        metric.evaluate_run([{"image": SELFTEST_IMAGE, "caption": cap, "story": good}], out)
        assert list(csv.DictReader(open(out, encoding="utf-8")))[0]["image"] == "selftest_image.jpg"


def test_model_vision_json_schema():
    _need_models()
    desc = seeing.describe(SELFTEST_IMAGE)
    required = ["image_id", "scene", "description", "objects", "od_labels", "characters", "actions", "relationships",
                "ocr_text", "spatial_relations", "region_descriptions", "style_or_mood", "runtime_s", "model_load_s"]
    for k in required:
        assert k in desc, f"Missing key: {k}"
    for k in ("objects", "od_labels", "characters", "actions", "region_descriptions"):
        assert isinstance(desc[k], list)
    try:
        seeing.describe(os.path.join(HERE, "nonexistent_xyz.jpg"))
        assert False, "should have raised"
    except (FileNotFoundError, OSError):
        pass


UNIT = [v for k, v in sorted(globals().items()) if k.startswith("test_") and not k.startswith("test_model_")]
MODEL = [v for k, v in sorted(globals().items()) if k.startswith("test_model_")]


def _run(tests):
    passed = 0
    for t in tests:
        t()
        print("OK", t.__name__)
        passed += 1
    return passed


if __name__ == "__main__":
    n_unit = _run(UNIT)
    n_model = _run(MODEL) if "--models" in sys.argv else 0
    msg = f"\n{n_unit}/{len(UNIT)} unit tests passed (no model inference)"
    msg += f"; {n_model}/{len(MODEL)} model-dependent tests passed" if "--models" in sys.argv else f"; {len(MODEL)} model-dependent tests skipped (use --models)"
    print(msg)
