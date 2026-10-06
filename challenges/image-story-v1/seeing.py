"""Seeing stage: Florence-2-base -> rich structured description. Offline, CPU, float32."""
import os
import re
import time
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# HF_HOME: use environment variable if set, otherwise let Hugging Face use its default cache
# os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

MODEL_ID = "florence-community/Florence-2-base"
NUM_BEAMS = 1
_STATE = {}
LOAD_TIME_S = None
LOAD_MODE = None


def _load():
    """Lazy, once. Native transformers class first, trust_remote_code fallback."""
    global LOAD_TIME_S, LOAD_MODE
    if "model" in _STATE:
        return _STATE["model"], _STATE["proc"]
    from transformers import AutoProcessor
    t0 = time.perf_counter()
    try:
        from transformers import Florence2ForConditionalGeneration
        proc = AutoProcessor.from_pretrained(MODEL_ID)
        model = Florence2ForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        LOAD_MODE = "native"
    except Exception as e:
        from transformers import AutoModelForCausalLM, AutoConfig
        cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
        for c in (cfg, getattr(cfg, "text_config", None)):
            if c is not None:
                for k in ("forced_bos_token_id", "forced_eos_token_id"):
                    if not hasattr(c, k):
                        setattr(c, k, None)
        proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, config=cfg, trust_remote_code=True, torch_dtype=torch.float32)
        LOAD_MODE = "remote_code (native failed: %s)" % str(e)[:120]
    model.eval()
    _STATE["model"], _STATE["proc"] = model, proc
    LOAD_TIME_S = round(time.perf_counter() - t0, 2)
    return model, proc


def _run(model, proc, image, task, max_new_tokens=256):
    inputs = proc(text=task, images=image, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"].to(torch.float32),
            max_new_tokens=max_new_tokens, num_beams=NUM_BEAMS, do_sample=False,
            use_cache=True,
        )
    text = proc.batch_decode(ids, skip_special_tokens=False)[0]
    return text, proc.post_process_generation(text, task=task, image_size=image.size)


def _clean(s):
    return re.sub(r"\s+", " ", re.sub(r"</?s>|<pad>", "", s or "")).strip()


_CHARACTER_KEYWORDS = {
    "person", "people", "man", "woman", "girl", "boy", "child", "children",
    "baby", "infant", "teenager", "adult", "elderly", "senior",
    "character", "figure", "human", "portrait",
    "dog", "cat", "bird", "horse", "cow", "pig", "sheep", "animal",
    "pet", "creature", "monster", "dragon", "spirit",
}

_ACTION_VERBS = {
    "walking", "running", "standing", "sitting", "lying", "sleeping",
    "holding", "carrying", "wearing", "eating", "drinking", "reading",
    "writing", "looking", "watching", "gazing", "staring", "smiling",
    "laughing", "crying", "talking", "speaking", "listening",
    "playing", "working", "cooking", "painting", "drawing",
    "driving", "riding", "flying", "swimming", "jumping", "climbing",
    "opening", "closing", "pushing", "pulling", "lifting", "carrying",
    "reaching", "pointing", "waving", "gesturing", "dancing", "singing",
}


def _extract_entities_from_dense(labels):
    """Parse DENSE_REGION_CAPTION labels into structured entities."""
    character_types = []
    action_verbs = []
    spatial_phrases = []
    region_descriptions = []
    object_descriptions = []
    
    # Spatial prepositions (require word boundaries)
    spatial_patterns = [
        r"\bnext to\b", r"\bbeside\b", r"\bbehind\b", r"\bin front of\b",
        r"\babove\b", r"\bbelow\b", r"\bunder\b", r"\bover\b",
        r"\bbetween\b", r"\bamong\b", r"\bnear\b", r"\bfar\b",
        r"\bleft\b", r"\bright\b", r"\bcenter\b", r"\bcorner\b", r"\bedge\b", r"\bside\b",
        r"\binside\b", r"\boutside\b", r"\bwithin\b",
    ]
    # "on", "at", "by" are too ambiguous, skip
    
    for label in labels:
        label_lower = label.lower()
        region_descriptions.append(label)
        
        # Character types (word boundary matching)
        for kw in _CHARACTER_KEYWORDS:
            if re.search(r"\b" + re.escape(kw) + r"\b", label_lower):
                if kw not in character_types:
                    character_types.append(kw)
        
        # Action verbs (word boundary matching)
        for verb in _ACTION_VERBS:
            if re.search(r"\b" + re.escape(verb) + r"\b", label_lower):
                if verb not in action_verbs:
                    action_verbs.append(verb)
        
        # Spatial phrases
        for pattern in spatial_patterns:
            if re.search(pattern, label_lower):
                # Extract the phrase around the spatial term
                match = re.search(pattern, label_lower)
                if match:
                    start = max(0, match.start() - 15)
                    end = min(len(label_lower), match.end() + 15)
                    phrase = label_lower[start:end].strip()
                    if phrase not in spatial_phrases:
                        spatial_phrases.append(phrase)
                break
        
        # If no character keyword, treat as object description
        has_char = any(re.search(r"\b" + re.escape(kw) + r"\b", label_lower) for kw in _CHARACTER_KEYWORDS)
        if not has_char:
            object_descriptions.append(label)
    
    return {
        "character_types": character_types,
        "action_verbs": action_verbs,
        "spatial_phrases": spatial_phrases,
        "region_descriptions": region_descriptions,
        "object_descriptions": object_descriptions,
    }


def describe(image_path):
    t0 = time.perf_counter()
    model, proc = _load()
    t1 = time.perf_counter()
    image = Image.open(image_path).convert("RGB")
    out = {}
    
    # Task sequence: detailed caption + object detection + dense region caption
    # OCR dropped (returns junk on anime frames), CAPTION redundant with detailed
    for key, task, mnt in [
        ("detailed_caption", "<MORE_DETAILED_CAPTION>", 160),
        ("od", "<OD>", 96),
        ("dense", "<DENSE_REGION_CAPTION>", 256),
    ]:
        _, parsed = _run(model, proc, image, task, mnt)
        out[key] = parsed.get(task) if isinstance(parsed, dict) else parsed
    
    # Extract object labels from OD
    labels, seen = [], set()
    for l in (out["od"] or {}).get("labels", []):
        l = _clean(l).lower()
        if l and l not in seen:
            seen.add(l)
            labels.append(l)
    
    # Extract structured info from dense region captions
    dense_labels = (out["dense"] or {}).get("labels", [])
    dense_entities = _extract_entities_from_dense(dense_labels)
    
    # Merge OD objects with dense object descriptions
    all_objects = labels + dense_entities["object_descriptions"]
    all_objects = list(dict.fromkeys(all_objects))  # dedupe preserving order
    
    # Infer scene from detailed caption
    detailed = _clean(out["detailed_caption"])
    scene = ""
    if detailed:
        first_sent = detailed.split(".")[0]
        if "scene" in first_sent.lower() or "street" in first_sent.lower() or "city" in first_sent.lower():
            scene = first_sent
        elif "room" in first_sent.lower() or "interior" in first_sent.lower():
            scene = first_sent
        elif "forest" in first_sent.lower() or "woods" in first_sent.lower():
            scene = first_sent
        elif "restaurant" in first_sent.lower() or "kitchen" in first_sent.lower():
            scene = first_sent
        else:
            scene = first_sent[:120]
    
    # Infer style/mood from detailed caption
    style_mood = ""
    mood_kws = {"whimsical", "playful", "serene", "peaceful", "cheerful", "joyful",
                "festive", "lively", "dark", "gloomy", "mysterious", "ominous",
                "cozy", "warm", "cold", "bright", "colorful", "vibrant", "dull",
                "anime", "illustration", "cartoon", "sketch", "painting", "photograph"}
    for kw in mood_kws:
        if kw in detailed.lower():
            style_mood = kw
            break
    
    # Also extract characters/actions from detailed caption
    detailed_lower = detailed.lower()
    detailed_chars = []
    for kw in _CHARACTER_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", detailed_lower):
            detailed_chars.append(kw)
    
    detailed_actions = []
    for verb in _ACTION_VERBS:
        if re.search(r"\b" + re.escape(verb) + r"\b", detailed_lower):
            detailed_actions.append(verb)
    
    # Merge dense + detailed caption extractions
    all_character_types = dense_entities["character_types"] + detailed_chars
    all_action_verbs = dense_entities["action_verbs"] + detailed_actions
    
    def dedup(lst):
        seen = set()
        out = []
        for x in lst:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
    
    return {
        "image_id": os.path.basename(image_path),
        "scene": scene,
        "description": detailed,
        "objects": all_objects,
        "od_labels": labels,  # short object-detection labels only (used for entity matching across frames)
        "characters": dedup(all_character_types),
        "actions": dedup(all_action_verbs),
        "relationships": dedup(dense_entities["spatial_phrases"]),
        "ocr_text": "",  # OCR disabled (junk on anime)
        "spatial_relations": dedup(dense_entities["spatial_phrases"]),
        "region_descriptions": dense_entities["region_descriptions"],
        "style_or_mood": style_mood,
        "runtime_s": round(time.perf_counter() - t1, 2),
        "model_load_s": round(t1 - t0, 2),
    }


def to_llm_context(desc, max_words=120):
    """Backward-compatible context builder (deprecated; use context_builder.py)."""
    parts = [desc.get("description") or desc.get("detailed_caption") or desc.get("short_caption", "")]
    if desc.get("objects"):
        parts.append("Objects visible: " + ", ".join(desc["objects"]) + ".")
    if desc.get("ocr_text"):
        parts.append("Text in image: " + desc["ocr_text"])
    words = " ".join(parts).split()
    if len(words) > max_words:
        words = words[:max_words]
    return " ".join(words)


if __name__ == "__main__":
    import sys, json
    for p in sys.argv[1:]:
        d = describe(p)
        print(p)
        print(json.dumps(d, indent=1))
        print("CONTEXT:", to_llm_context(d))
    print("load_mode:", LOAD_MODE, "load_s:", LOAD_TIME_S)