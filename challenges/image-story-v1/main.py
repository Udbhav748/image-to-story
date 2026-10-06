"""Main entry point: Image -> Story pipeline with baseline and improved modes.

Usage:
  python main.py images/                    # improved (Florence-2) on all images
  python main.py images/ --baseline         # baseline (BLIP) on all images
  python main.py images/ --both             # both pipelines, scored and compared (writes CSV + vision JSON)
  python main.py img1.jpg img2.jpg          # specific images
  python main.py images/ --multi            # one story across all images, in filename order
  python main.py images/ --both --fix-length  # length-controlled generation (see baseline.write_story)
"""
import os, time, json, csv, argparse
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# HF_HOME: use the environment variable if set, otherwise Hugging Face's default cache is used.

import torch

import baseline
import seeing
import context_builder
import metric

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
MULTI_TARGET_WORDS = 250     # total length requested for a multi-image story
MULTI_MAX_NEW_TOKENS = 400   # generation cap for a multi-image story


def find_images(paths):
    """Expand files/directories to image files, sorted by file name (the frame order used by --multi)."""
    found = {}
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files = [f for f in path.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            files = [path]
        else:
            files = []
        for f in files:
            found[str(f)] = f
    return [str(f) for f in sorted(found.values(), key=lambda f: f.name.lower())]


def run_baseline(image_path, fix_length=False):
    """Run baseline pipeline: BLIP caption -> Qwen story."""
    try:
        return baseline.run(image_path, fix_length)
    except Exception as e:
        return {"image": str(image_path), "error": f"baseline failed: {e}"}


def run_improved(image_path, fix_length=False):
    """Run improved pipeline: Florence-2 structured -> context -> Qwen story."""
    try:
        desc = seeing.describe(image_path)  # vision runtime excludes model loading
        ctx = context_builder.build_single_image_context(desc)
        baseline.warm_up(blip=False)         # Qwen is loaded before the story timer starts
        t1 = time.perf_counter()
        story = baseline.write_story(ctx, fix_length)
        story_s = time.perf_counter() - t1
        return {
            "image": str(image_path),
            "caption": ctx,  # context used as the caption for the metric
            "story": story,
            "word_count": len(story.split()),
            "length_valid": baseline.is_length_valid(len(story.split())),
            "caption_s": round(desc["runtime_s"], 2),
            "story_s": round(story_s, 2),
            "generation_s": round(desc["runtime_s"] + story_s, 2),
            "vision_json": desc,
        }
    except Exception as e:
        return {"image": str(image_path), "error": f"improved failed: {e}"}


def run_multi_image_story(image_paths):
    """One story across all images (filename order) plus the multi-image continuity proxy.

    Frames whose vision stage fails are skipped and reported in `frames_failed`. A final sentence cut off by the
    token limit is removed (`cut_off_tail_removed` says whether that happened). The --fix-length retry mechanism
    does not apply to this mode.
    """
    try:
        descs, failed, total_see_s = [], [], 0.0
        for p in image_paths:
            try:
                t0 = time.perf_counter()
                descs.append(seeing.describe(p))
                total_see_s += time.perf_counter() - t0
            except Exception as e:
                failed.append({"image": os.path.basename(p), "error": str(e)})
        if not descs:
            return {"images": [os.path.basename(p) for p in image_paths], "frames_failed": failed,
                    "error": "multi failed: no frame could be described"}

        seq_ctx = context_builder.build_sequence_context(descs)
        prompt = context_builder.build_story_prompt(seq_ctx, len(descs), target_words=MULTI_TARGET_WORDS)

        tok, model = baseline._qwen()  # loaded before the story timer starts
        t1 = time.perf_counter()
        msgs = [{"role": "system", "content": "You are a creative storyteller."},
                {"role": "user", "content": prompt}]
        prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt_text, return_tensors="pt")
        torch.manual_seed(baseline.SEED)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MULTI_MAX_NEW_TOKENS, do_sample=False, repetition_penalty=1.05)
        raw_story = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        story = baseline._fit_length(raw_story, hi=10 ** 6)  # keep every complete sentence, drop a cut-off tail
        story_s = time.perf_counter() - t1

        return {
            "images": [d["image_id"] for d in descs],
            "frames_failed": failed,
            "target_words": MULTI_TARGET_WORDS,
            "story": story,
            "cut_off_tail_removed": story != raw_story,
            "word_count": len(story.split()),
            "see_s": round(total_see_s, 2),
            "story_s": round(story_s, 2),
            "continuity": metric.continuity_score(descs, story),
            "vision_json": {d["image_id"]: {k: v for k, v in d.items() if k not in ("runtime_s", "model_load_s")} for d in descs},
        }
    except Exception as e:
        return {"images": [os.path.basename(p) for p in image_paths], "error": f"multi failed: {e}"}


def _ok_row(name, variant, path, context, story, see_s, story_s):
    """One successful run. Runtime columns: see_s + story_s = generation_s (the pipeline itself);
    eval_clip_s / eval_nli_s / eval_rules_s / eval_total_s are the metric cost, reported separately."""
    m = metric.score(path, context, story)
    m["attribute_conflict"] = " ".join(m["attribute_conflict"])
    return {"image": name, "variant": variant, "status": "ok", "error": "", "context": context, "story": story,
            "see_s": round(see_s, 2), "story_s": round(story_s, 2), "generation_s": round(see_s + story_s, 2), **m}


def _error_row(name, variant, err):
    """A failed run is recorded as a failure; no metric values are invented for it."""
    return {"image": name, "variant": variant, "status": "error", "error": str(err)}


def write_rows_csv(rows, path):
    """Write rows to CSV. Columns are the union of all row keys (failed rows simply have empty metric cells)."""
    fieldnames = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        w.writeheader()
        w.writerows(rows)


def summarize(rows, variants=("A_baseline", "B_improved")):
    """Per-variant means over SUCCESSFUL runs only, plus explicit success/failure counts."""
    out = {}
    for v in variants:
        mine = [r for r in rows if r["variant"] == v]
        ok = [r for r in mine if r.get("status") == "ok"]
        n = len(ok)
        s = {"successful": n, "failed": len(mine) - n, "total": len(mine)}
        if n:
            mean = lambda k: sum(float(x[k]) for x in ok) / n
            s.update({
                "mean_grounding": mean("grounding_score"), "mean_clip": mean("clip_image_story_mean"),
                "mean_nli_contradiction": mean("nli_contra_mean"), "mean_repetition_rate": mean("repetition_rate"),
                "length_valid": sum(bool(x["length_valid"]) for x in ok),
                "grounding_pass": sum(bool(x["grounding_pass"]) for x in ok),
                "mean_see_s": mean("see_s"), "mean_story_s": mean("story_s"),
                "mean_generation_s": mean("generation_s"), "mean_eval_total_s": mean("eval_total_s"),
            })
        out[v] = s
    return out


def run_comparison(image_paths, fix_length=False, output_csv="results.csv", vision_json="vision.json"):
    """Run both pipelines on all images and evaluate. One failing image never stops the others."""
    rows, vision = [], {}

    for p in image_paths:
        name = os.path.basename(p)
        print(f"\nProcessing {name}...")

        # A: baseline (BLIP -> Qwen)
        try:
            a = baseline.run(p, fix_length)
            rows.append(_ok_row(name, "A_baseline", p, a["caption"], a["story"], a["caption_s"], a["story_s"]))
        except Exception as e:
            rows.append(_error_row(name, "A_baseline", e))
            print(f"  A_baseline error: {e}")

        # B: improved (Florence-2 -> structured JSON -> context builder -> Qwen)
        try:
            desc = seeing.describe(p)
            ctx = context_builder.build_single_image_context(desc)
            baseline.warm_up(blip=False)  # Qwen is loaded before the story timer starts
            t = time.perf_counter()
            story_b = baseline.write_story(ctx, fix_length)
            story_s = time.perf_counter() - t
            vision[name] = desc
            rows.append(_ok_row(name, "B_improved", p, ctx, story_b, desc["runtime_s"], story_s))
        except Exception as e:
            rows.append(_error_row(name, "B_improved", e))
            print(f"  B_improved error: {e}")

        for r in rows[-2:]:
            if r["status"] == "ok":
                print(f"  {r['variant']}: grounding={r['grounding_score']:.3f} pass={r['grounding_pass']} "
                      f"words={r['word_count']} rep={r['repetition_rate']:.3f}")

    if rows:
        write_rows_csv(rows, output_csv)
        print(f"\nWrote {output_csv} with {len(rows)} rows")
    context_builder.save_vision_json(list(vision.values()), vision_json)
    print(f"Wrote {vision_json}")

    for v, s in summarize(rows).items():
        print(f"\n{v}: successful evaluations {s['successful']}/{s['total']}, failed {s['failed']}/{s['total']}")
        if s["successful"]:
            n = s["successful"]
            print(f"  mean grounding={s['mean_grounding']:.3f}  mean clip={s['mean_clip']:.3f}  mean nli_contradiction={s['mean_nli_contradiction']:.3f}")
            print(f"  grounding_pass={s['grounding_pass']}/{n}  length_valid={s['length_valid']}/{n}")
            print(f"  mean repetition_rate={s['mean_repetition_rate']:.4f}")
            print(f"  generation: mean see_s={s['mean_see_s']:.1f}  story_s={s['mean_story_s']:.1f}  total={s['mean_generation_s']:.1f}  (model loading excluded)")
            print(f"  evaluation: mean eval_total_s={s['mean_eval_total_s']:.2f}  (CLIP + NLI + rules, not part of generation)")
    return rows, vision


def _fmt(x):
    return "n/a" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))


def build_parser():
    parser = argparse.ArgumentParser(description="Image -> Story pipeline")
    parser.add_argument("paths", nargs="+", help="Image files or directories")
    parser.add_argument("--baseline", action="store_true", help="Run baseline (BLIP) only")
    parser.add_argument("--improved", action="store_true", help="Run improved (Florence-2) only")
    parser.add_argument("--both", action="store_true", help="Run both and compare (default)")
    parser.add_argument("--multi", action="store_true", help="Generate one story across all images (filename order)")
    parser.add_argument("--fix-length", action="store_true",
                        help="Length-controlled generation: up to 3 attempts with stricter length prompts, trimmed to whole "
                             "sentences; returns the first 80-120 word story or else the closest attempt (not used by --multi)")
    parser.add_argument("--output", default="results.csv", help="Output CSV file (--both)")
    parser.add_argument("--vision-json", default="vision.json", help="Output vision JSON file (--both)")
    parser.add_argument("--multi-output", default="combined_story.json", help="Output JSON file (--multi)")
    return parser


def main():
    args = build_parser().parse_args()

    if not (args.baseline or args.improved or args.both or args.multi):
        args.both = True

    images = find_images(args.paths)
    if not images:
        print("No images found")
        return

    print(f"Found {len(images)} image(s)")
    print(f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}")

    if args.multi:
        result = run_multi_image_story(images)
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        print(f"\nMulti-image story ({result['word_count']} words, {len(result['images'])} frames):")
        print(result["story"])
        cont = result["continuity"]
        print("\nContinuity proxy (entity overlap + transition words; not a measure of narrative coherence):")
        print(f"  recurring_entities ({cont['recurring_count']}): {', '.join(cont['recurring_entities']) or 'none'}")
        print(f"  entity_consistency: {_fmt(cont['entity_consistency'])}")
        print(f"  transition_markers: {_fmt(cont['transition_markers'])}")
        print(f"  adjacent_similarity: {_fmt(cont['adjacent_similarity'])}")
        with open(args.multi_output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {args.multi_output}")
        return

    if args.baseline:
        for p in images:
            r = run_baseline(p, args.fix_length)
            print(f"Error: {r['error']}" if "error" in r else json.dumps(r, indent=1))
        return

    if args.improved:
        for p in images:
            r = run_improved(p, args.fix_length)
            print(f"Error: {r['error']}" if "error" in r else json.dumps({k: v for k, v in r.items() if k != "vision_json"}, indent=1))
        return

    run_comparison(images, args.fix_length, args.output, args.vision_json)


if __name__ == "__main__":
    main()
