"""Stand-in baseline: image -> BLIP caption -> Qwen2.5-0.5B-Instruct story (80-120 words).
Usage: python baseline.py img1.jpg [img2.jpg ...]   |   from baseline import caption_blip, write_story, run
"""
import os, sys, time, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# HF_HOME: use environment variable if set, otherwise let Hugging Face use its default cache
# os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM

BLIP_ID = "Salesforce/blip-image-captioning-base"
QWEN_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 0
MIN_WORDS, MAX_WORDS = 80, 120  # benchmark length requirement
_m = {}
load_times = {}


def is_length_valid(word_count):
    """The benchmark length rule, defined once: 80 <= words <= 120 (a requirement, not a quality measure)."""
    return MIN_WORDS <= word_count <= MAX_WORDS


def warm_up(blip=True, qwen=True):
    """Load models now so that per-image timings never include model-load time (see load_times)."""
    if blip: _blip()
    if qwen: _qwen()

def _blip():
    if "blip" not in _m:
        t = time.perf_counter()
        _m["blip"] = (BlipProcessor.from_pretrained(BLIP_ID),
                      BlipForConditionalGeneration.from_pretrained(BLIP_ID).eval())
        load_times["blip"] = time.perf_counter() - t
    return _m["blip"]

def _qwen():
    if "qwen" not in _m:
        t = time.perf_counter()
        _m["qwen"] = (AutoTokenizer.from_pretrained(QWEN_ID),
                      AutoModelForCausalLM.from_pretrained(QWEN_ID, torch_dtype=torch.float32).eval())
        load_times["qwen"] = time.perf_counter() - t
    return _m["qwen"]

def caption_blip(image_path):
    proc, model = _blip()
    img = Image.open(image_path).convert("RGB")
    inputs = proc(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30)
    return proc.decode(out[0], skip_special_tokens=True).strip()

LENGTH_PROMPTS = [  # attempt 1 is the original prompt; later attempts are used only when fix_length=True
    "Write a short story of 80 to 120 words based only on this description of an image: {c}\nReturn only the story.",
    "Write a story of about 100 words (never fewer than 90) based only on this description of an image: {c}\nReturn only the story.",
    "Write a story of 100 to 110 words in 6 to 8 sentences based only on this description of an image: {c}\nReturn only the story.",
]

def _generate(context_text, template, max_new_tokens):
    tok, model = _qwen()
    msgs = [{"role": "system", "content": "You are a creative storyteller."},
            {"role": "user", "content": template.format(c=context_text)}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt")
    torch.manual_seed(SEED)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.05)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def _sentences(text):
    import re
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]

def _fit_length(story, hi=MAX_WORDS):
    """Keep whole sentences only (drops a cut-off tail), then stop adding sentences once past `hi` words."""
    sents = _sentences(story)
    if sents and not sents[-1].rstrip('"\'').endswith((".", "!", "?")):
        sents = sents[:-1]  # last sentence was cut off by the token limit
    kept, n = [], 0
    for s in sents:
        w = len(s.split())
        if n + w > hi: break  # adding this sentence would exceed the limit
        kept.append(s); n += w
    return " ".join(kept) if kept else story

def write_story(context_text, fix_length=False):
    """Write a story from a context string with Qwen.

    fix_length=False: one greedy generation with the original prompt (max 200 new tokens).
    fix_length=True ("length-controlled"): up to 3 greedy attempts, each with a stricter length prompt
    (LENGTH_PROMPTS) and a 230-token cap. Every attempt is cut to whole sentences (a cut-off last sentence
    is dropped) and to at most 120 words. The first attempt with 80-120 words is returned; if none qualifies,
    the attempt whose length is closest to 100 words is returned. This does NOT produce equal word counts.
    """
    if not fix_length:
        return _generate(context_text, LENGTH_PROMPTS[0], 200)
    best = None
    for template in LENGTH_PROMPTS:  # retry with a stricter length prompt until the story is 80-120 words
        story = _fit_length(_generate(context_text, template, 230))
        wc = len(story.split())
        if is_length_valid(wc): return story
        if best is None or abs(wc - 100) < abs(len(best.split()) - 100): best = story
    return best

def run(image_path, fix_length=False):
    """Baseline pipeline for one image. Timings (perf_counter) exclude model loading, which is done first."""
    warm_up()
    t = time.perf_counter(); cap = caption_blip(image_path); cs = time.perf_counter() - t
    t = time.perf_counter(); story = write_story(cap, fix_length); ss = time.perf_counter() - t
    wc = len(story.split())
    return {"image": str(image_path), "caption": cap, "story": story, "word_count": wc,
            "length_valid": is_length_valid(wc), "caption_s": round(cs, 2), "story_s": round(ss, 2),
            "generation_s": round(cs + ss, 2)}

if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(run(p), indent=1))
    print("load_times", load_times)
