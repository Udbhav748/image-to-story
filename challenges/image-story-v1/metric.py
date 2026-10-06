"""Evaluation metrics for Image -> Story (Module 10: evaluation + runtime logging).

Per image-story pair (score):
  grounding_score = 0.4*clip_n + 0.4*(1 - nli_contra_mean) + 0.2*(0 if attribute_conflict else 1)
    clip_n          = clip((clip_image_story_mean - 0.15) / 0.15, 0, 1)
                      CLIP ViT-B/32 cosine between the IMAGE and each story sentence, averaged.
                      This is the only signal that compares the story with the image itself.
    nli_contra_mean = mean P(contradiction | premise = supplied context, hypothesis = story sentence).
                      NLI measures TEXTUAL consistency between the story and the supplied/generated visual
                      context (BLIP caption or the structured Florence-2 context). It is not independent
                      verification of the image: if the context is wrong, a story faithful to it still scores well.
    attribute_conflict = deterministic textual check of colour/material words (a different colour or material
                      than the context names, or a material whose usual colour the context's colour excludes,
                      e.g. caption "white cabinets" vs story "oak cabinet"). See attribute_conflict().
  grounding_score is a composite automatic proxy, not ground truth.

  length_valid   = 80 <= word_count <= 120. This is a benchmark requirement, not a quality measure: a story can be
                   well aligned with the image but too short, or the right length and visually wrong.
  grounding_pass = grounding_score >= 0.60 AND no attribute conflict AND length_valid.

Also reported (not part of grounding_score): truncated, repetition metrics, runtime.

continuity_score (multi-image only, see main.py --multi) is a lightweight structural proxy based on entity
overlap and transition words. It does not evaluate narrative coherence.
"""
import os, re, time, csv
from contextlib import ContextDecorator
from collections import Counter

# HF_HOME: use the environment variable if set, otherwise Hugging Face's default cache is used.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, AutoModelForSequenceClassification

from context_builder import frame_entities, recurring_entities, normalize_entity
from baseline import is_length_valid

THRESHOLD = 0.60
CLIP_LOW, CLIP_SPAN = 0.15, 0.15  # clip_n = clip((cos - 0.15) / 0.15, 0, 1); untuned heuristic range
TIMINGS = {}  # stage name -> list of seconds


class Timer(ContextDecorator):
    """Use as `with Timer('stage'):` or `@Timer('stage')`; appends seconds to TIMINGS."""
    def __init__(self, name): self.name = name
    def __enter__(self): self.t = time.perf_counter(); return self
    def __exit__(self, *a): TIMINGS.setdefault(self.name, []).append(time.perf_counter() - self.t)


with Timer("load_models"):
    _clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    _clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _nli_name = "cross-encoder/nli-MiniLM2-L6-H768"
    _nli_tok = AutoTokenizer.from_pretrained(_nli_name)
    _nli = AutoModelForSequenceClassification.from_pretrained(_nli_name).eval()
_CONTRA = [i for i, l in _nli.config.id2label.items() if l.lower() == "contradiction"][0]

COLORS = {"white", "black", "red", "blue", "green", "yellow", "brown", "grey", "pink", "orange", "purple"}
MATERIAL_FAMILIES = {"wood": {"wood", "wooden", "oak"}, "glass": {"glass"}, "metal": {"metal", "steel"},
                     "plastic": {"plastic"}, "stone": {"stone", "marble"}, "brick": {"brick"}, "leather": {"leather"}}
TYPICAL_COLOR = {"wood": {"brown"}}  # colour a material normally implies (heuristic; see attribute_conflict)
_MATERIAL_WORDS = {w: fam for fam, ws in MATERIAL_FAMILIES.items() for w in ws}


def split_sentences(text): return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
def words(text): return {"grey" if w == "gray" else w for w in re.findall(r"[a-z]+", text.lower())}


def attribute_conflict(caption, story):
    """Deterministic textual check of colour/material words; returns the offending story words.

    1. Colour: the caption names a colour and the story names a different colour -> conflict.
    2. Material: the caption names a material family (wood, wooden and oak are one family) and the story names
       another family -> conflict.
    3. Implied colour: the caption names a colour, names no word of the story's material family, and the
       material normally implies a different colour (oak/wood -> brown) -> conflict. A heuristic: a white-painted
       wooden cabinet is legitimate, so this can raise false alarms.
    Nothing is checked for a group the caption does not mention.
    """
    cw, sw, bad = words(caption), words(story), []
    cap_colors, story_mats = cw & COLORS, sorted(sw & set(_MATERIAL_WORDS))
    if cap_colors:
        bad += sorted((sw & COLORS) - cw)
    cap_fams = {_MATERIAL_WORDS[w] for w in cw if w in _MATERIAL_WORDS}
    for w in story_mats:
        fam = _MATERIAL_WORDS[w]
        if cap_fams and fam not in cap_fams:
            bad.append(w)
        elif not cap_fams and cap_colors and fam in TYPICAL_COLOR and not (TYPICAL_COLOR[fam] & cap_colors):
            bad.append(w)
    return bad


def grounding_from_components(clip_mean, nli_contra_mean, has_conflict):
    """The grounding formula in one place (used by score() and by the tests). Returns (clip_n, grounding_score)."""
    clip_n = min(max((clip_mean - CLIP_LOW) / CLIP_SPAN, 0.0), 1.0)
    g = 0.4 * clip_n + 0.4 * (1 - nli_contra_mean) + 0.2 * (0 if has_conflict else 1)
    return clip_n, g


def is_grounding_pass(grounding_score, conflicts, length_valid):
    """grounding_pass definition: score >= THRESHOLD AND no attribute conflict AND length_valid."""
    return bool(grounding_score >= THRESHOLD and not conflicts and length_valid)


@torch.no_grad()
def _clip_sims(image, texts):
    inp = _clip_proc(text=texts, images=image, return_tensors="pt", padding=True, truncation=True, max_length=77)
    out = _clip(**inp)
    return out.logits_per_image[0] / _clip.logit_scale.exp()  # cosine similarities


@torch.no_grad()
def _contradiction_probs(caption, sentences):
    inp = _nli_tok([caption] * len(sentences), sentences, return_tensors="pt", padding=True, truncation=True)
    return _nli(**inp).logits.softmax(-1)[:, _CONTRA]


def repetition_score(story: str) -> dict:
    """Lightweight deterministic repetition metrics (trigram based).

    repetition_rate = repeated_trigrams / total_trigrams ; distinct3 = 1 - repetition_rate.
    """
    sents = split_sentences(story)
    toks = re.findall(r"[a-z']+", story.lower())

    repeated_sentences = sum(c - 1 for c in Counter(sents).values() if c > 1)
    bigrams = list(zip(toks, toks[1:])) if len(toks) > 1 else []
    repeated_bigrams = sum(c - 1 for c in Counter(bigrams).values() if c > 1)
    trigrams = list(zip(toks, toks[1:], toks[2:])) if len(toks) > 2 else []
    repeated_trigrams = sum(c - 1 for c in Counter(trigrams).values() if c > 1)

    rate = repeated_trigrams / len(trigrams) if trigrams else 0.0
    return {
        "repeated_sentences": repeated_sentences,
        "repeated_bigrams": repeated_bigrams,
        "repeated_trigrams": repeated_trigrams,
        "repetition_rate": round(rate, 4),
        "distinct3": round(1.0 - rate if trigrams else 1.0, 4),
    }


TRANSITION_WORDS = {"then", "next", "after", "afterwards", "later", "suddenly", "meanwhile",
                    "before", "finally", "soon", "eventually", "following"}


def continuity_score(descs: list, story: str) -> dict:
    """Lightweight structural continuity proxy for a MULTI-image story. Needs >= 2 ordered frames.

    descs: ordered per-frame structured vision outputs (characters + od_labels are used).
    - recurring_entities: entities present in at least two different frames (normalised labels from the vision stage).
    - entity_consistency: fraction of recurring entities that the story mentions. None when no entity recurs.
    - transition_markers: number of transition words in the story (whole-word match).
    - adjacent_similarity: mean Jaccard overlap of entity sets of adjacent frames (frames with no entities skipped).
    This is an overlap-and-lexicon proxy; it does not measure narrative coherence. Generic recurring labels
    (for example "person") count like any other recurring entity, so read the listed entities alongside the score.
    """
    n = len(descs or [])
    result = {"num_frames": n, "recurring_entities": [], "recurring_count": 0,
              "entity_consistency": None, "transition_markers": None, "adjacent_similarity": None}
    if n < 2:
        return result  # not applicable for a single image

    frames = [frame_entities(d) for d in descs]
    recurring = recurring_entities(frames)
    story_norm = " " + " ".join(normalize_entity(w) for w in re.findall(r"[a-z']+", story.lower())) + " "
    mentioned = [e for e in recurring if f" {e} " in story_norm]

    overlaps = [len(a & b) / len(a | b) for a, b in zip(frames, frames[1:]) if (a | b)]
    result.update({
        "recurring_entities": recurring,
        "recurring_count": len(recurring),
        "entity_consistency": round(len(mentioned) / len(recurring), 4) if recurring else None,
        "transition_markers": sum(1 for t in re.findall(r"[a-z]+", story.lower()) if t in TRANSITION_WORDS),
        "adjacent_similarity": round(sum(overlaps) / len(overlaps), 4) if overlaps else None,
    })
    return result


def score(image_path, caption, story):
    """Score one image-story pair against the supplied context/caption. Single-image metrics only.

    Evaluation runtime is reported separately from generation runtime (perf_counter, models already loaded):
    eval_clip_s (CLIP), eval_nli_s (NLI), eval_rules_s (attribute check, repetition, formula), eval_total_s.
    """
    t0 = time.perf_counter()
    sents = split_sentences(story) or [story]
    wc = len(story.split())
    image = Image.open(image_path).convert("RGB")
    t = time.perf_counter()
    with Timer("clip"):
        s = _clip_sims(image, sents + [caption])
    clip_s = time.perf_counter() - t
    sent_sims, cap_sim = s[:-1], float(s[-1])
    t = time.perf_counter()
    with Timer("nli"):
        c = _contradiction_probs(caption, sents)
    nli_s = time.perf_counter() - t
    t = time.perf_counter()
    bad = attribute_conflict(caption, story)
    clip_mean = float(sent_sims.mean())
    _, g = grounding_from_components(clip_mean, float(c.mean()), bool(bad))
    ok_len = is_length_valid(wc)
    rep = repetition_score(story)
    rules_s = time.perf_counter() - t

    return {
        "word_count": wc, "length_valid": ok_len,
        "truncated": not story.strip().rstrip('"\'').endswith((".", "!", "?")),
        "clip_image_story_mean": clip_mean, "clip_image_story_min": float(sent_sims.min()),
        "clip_image_caption": cap_sim,
        "nli_contra_mean": float(c.mean()), "nli_contra_max": float(c.max()),
        "attribute_conflict": bad, "grounding_score": g,
        "grounding_pass": is_grounding_pass(g, bad, ok_len),
        **rep,
        "eval_clip_s": round(clip_s, 3), "eval_nli_s": round(nli_s, 3), "eval_rules_s": round(rules_s, 4),
        "eval_total_s": round(time.perf_counter() - t0, 3),
    }


def evaluate_run(rows, out_csv):
    """rows: list of {image, caption, story}. Writes CSV, prints means and failure counts."""
    res = [{**r, **score(r["image"], r["caption"], r["story"])} for r in rows]
    for r in res: r["attribute_conflict"] = " ".join(r["attribute_conflict"])
    out_rows = [{**r, "image": os.path.basename(r["image"])} for r in res]  # CSV keeps file names only, never absolute paths
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
    num = ["clip_image_story_mean", "clip_image_story_min", "nli_contra_mean", "nli_contra_max",
           "grounding_score", "repetition_rate", "eval_total_s"]
    print("n =", len(res), {k: round(sum(r[k] for r in res) / len(res), 3) for k in num})
    print("fail: length", sum(not r["length_valid"] for r in res), "| attr_conflict", sum(bool(r["attribute_conflict"]) for r in res),
          "| grounding", sum(not r["grounding_pass"] for r in res))
    return res


if __name__ == "__main__":  # self-test with SYNTHETIC strings (test only)
    here = os.path.dirname(os.path.abspath(__file__))
    img = os.path.join(here, "selftest_image.jpg")  # ginger cat in a blue bag
    print("Image:", img, "| load_models s:", round(TIMINGS["load_models"][0], 2))
    cap = "a brown cat sitting inside a blue bag"
    good = ("The little brown cat curled up inside the blue bag and watched the room with wide amber eyes. "
            "Every time footsteps passed, her ears twitched and she peeked out from the plastic folds. "
            "The bag crinkled softly whenever she shifted her paws, and the sound made her tilt her head. "
            "It was warm and safe in there, her favorite hiding place in the whole house. "
            "Soon the afternoon sun slid across the floor, and the cat settled down to nap, "
            "purring quietly inside her cozy blue shelter, content and completely unbothered.")
    bad = ("The old oak cabinet creaked in the dark kitchen of the grandmother's farmhouse. "
           "Inside it she kept a black iron key, hidden under red velvet cloth, and nobody had touched it for years. "
           "A storm roared over the mountain while the wolves howled outside the castle gates. "
           "She lit a candle and read the ancient map by its flickering light. "
           "Nobody knew that the treasure lay beneath the frozen lake, waiting for the brave sailor "
           "who dared to cross the stormy sea at night, alone and without any fear.")
    rows = [{"image": img, "caption": cap, "story": t} for t in (good, bad)]
    for name, r in zip(("consistent", "contradicting"), evaluate_run(rows, os.path.join(here, "selftest.csv"))):
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("image", "caption", "story")})

    print("\nRepetition test:")
    print("good:", repetition_score(good))
    print("bad:", repetition_score(bad))

    print("\nContinuity test (synthetic frames; only entities in 2+ frames recur):")
    frames = [{"characters": ["person"], "od_labels": ["car"]},
              {"characters": ["person"], "od_labels": ["tree"]},
              {"characters": [], "od_labels": ["dog"]}]
    print(continuity_score(frames, "A person drove a car, then walked past a tree."))
