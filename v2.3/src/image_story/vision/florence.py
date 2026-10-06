"""Florence-2 vision model adapter."""
import os
import re
import time
import uuid
from typing import Any
from PIL import Image
import torch

from ..domain.schemas import (
    VisualObservations,
    EvidenceRecord,
    EvidenceType,
    EvidenceConfidence,
    SourceModel,
    BoundingBox,
    InformationClass,
)
from ..domain.enums import VisionTask
from .base import VisionModel


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

_SPATIAL_PATTERNS = [
    r"\bnext to\b", r"\bbeside\b", r"\bbehind\b", r"\bin front of\b",
    r"\babove\b", r"\bbelow\b", r"\bunder\b", r"\bover\b",
    r"\bbetween\b", r"\bamong\b", r"\bnear\b", r"\bfar\b",
    r"\bleft\b", r"\bright\b", r"\bcenter\b", r"\bcorner\b", r"\bedge\b", r"\bside\b",
    r"\binside\b", r"\boutside\b", r"\bwithin\b",
]


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"</?s>|<pad>", "", text or "")).strip()


def _extract_entities_from_dense(labels: list[str]) -> dict[str, list[str]]:
    """Parse DENSE_REGION_CAPTION labels into structured entities."""
    character_types = []
    action_verbs = []
    spatial_phrases = []
    region_descriptions = []
    object_descriptions = []
    
    for label in labels:
        label_lower = label.lower()
        region_descriptions.append(label)
        
        for kw in _CHARACTER_KEYWORDS:
            if re.search(r"\b" + re.escape(kw) + r"\b", label_lower):
                if kw not in character_types:
                    character_types.append(kw)
        
        for verb in _ACTION_VERBS:
            if re.search(r"\b" + re.escape(verb) + r"\b", label_lower):
                if verb not in action_verbs:
                    action_verbs.append(verb)
        
        for pattern in _SPATIAL_PATTERNS:
            if re.search(pattern, label_lower):
                match = re.search(pattern, label_lower)
                if match:
                    start = max(0, match.start() - 15)
                    end = min(len(label_lower), match.end() + 15)
                    phrase = label_lower[start:end].strip()
                    if phrase not in spatial_phrases:
                        spatial_phrases.append(phrase)
                break
        
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


def _dedupe(lst: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in lst:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


class Florence2Model(VisionModel):
    """Florence-2-base vision model adapter."""
    
    def __init__(self, model_id: str = "florence-community/Florence-2-base", device: str = "cpu"):
        self._model_id = model_id
        self._device = device
        self._model = None
        self._processor = None
        self._load_mode = None
        self._load_time_s = 0.0
        self._num_beams = 1
    
    @property
    def model_id(self) -> str:
        return self._model_id
    
    @property
    def is_loaded(self) -> bool:
        return self._model is not None
    
    def load(self) -> None:
        if self.is_loaded:
            return
        
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        
        from transformers import AutoProcessor
        
        t0 = time.perf_counter()
        try:
            from transformers import Florence2ForConditionalGeneration
            self._processor = AutoProcessor.from_pretrained(self._model_id)
            self._model = Florence2ForConditionalGeneration.from_pretrained(
                self._model_id, torch_dtype=torch.float32
            )
            self._load_mode = "native"
        except Exception as e:
            from transformers import AutoModelForCausalLM, AutoConfig
            cfg = AutoConfig.from_pretrained(self._model_id, trust_remote_code=True)
            for c in (cfg, getattr(cfg, "text_config", None)):
                if c is not None:
                    for k in ("forced_bos_token_id", "forced_eos_token_id"):
                        if not hasattr(c, k):
                            setattr(c, k, None)
            self._processor = AutoProcessor.from_pretrained(self._model_id, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_id, config=cfg, trust_remote_code=True, torch_dtype=torch.float32
            )
            self._load_mode = f"remote_code (native failed: {str(e)[:120]})"
        
        self._model.eval()
        self._model.to(self._device)
        self._load_time_s = round(time.perf_counter() - t0, 2)
    
    def unload(self) -> None:
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _run_task(self, image: Image.Image, task: str, max_new_tokens: int = 256) -> tuple[str, Any]:
        inputs = self._processor(text=task, images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"].to(torch.float32),
                max_new_tokens=max_new_tokens,
                num_beams=self._num_beams,
                do_sample=False,
                use_cache=True,
            )
        
        text = self._processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(text, task=task, image_size=image.size)
        return text, parsed
    
    def analyze(self, image: Image.Image, frame_id: int = 0) -> VisualObservations:
        if not self.is_loaded:
            self.load()
        
        t0 = time.perf_counter()
        image_id = f"frame_{frame_id}_{uuid.uuid4().hex[:8]}"
        evidence_records = []
        models_used = ["florence2"]
        
        results = {}
        for key, task, mnt in [
            ("detailed_caption", VisionTask.DETAILED_CAPTION.value, 160),
            ("od", VisionTask.OBJECT_DETECTION.value, 96),
            ("dense", VisionTask.DENSE_REGION_CAPTION.value, 256),
        ]:
            _, parsed = self._run_task(image, task, mnt)
            results[key] = parsed.get(task) if isinstance(parsed, dict) else parsed
        
        detailed = _clean(results.get("detailed_caption", ""))
        
        labels, seen = [], set()
        for l in (results.get("od") or {}).get("labels", []):
            l = _clean(l).lower()
            if l and l not in seen:
                seen.add(l)
                labels.append(l)
                
                evidence_records.append(EvidenceRecord(
                    entity=l,
                    type=EvidenceType.OBJECT,
                    frame_id=frame_id,
                    confidence=0.8,
                    source=SourceModel.FLORENCE2,
                    evidence_text=f"Object detected: {l}",
                    provenance={"task": "object_detection", "model": "florence2"},
                    sequence_position=frame_id,
                    information_class=InformationClass.HARD_FACT,
                ))
        
        dense_labels = (results.get("dense") or {}).get("labels", [])
        dense_entities = _extract_entities_from_dense(dense_labels)
        
        for region_desc in dense_entities["region_descriptions"]:
            evidence_records.append(EvidenceRecord(
                entity=region_desc[:100],
                type=EvidenceType.ENTITY,
                frame_id=frame_id,
                confidence=0.75,
                source=SourceModel.FLORENCE2,
                evidence_text=region_desc,
                provenance={"task": "dense_region_caption", "model": "florence2"},
                sequence_position=frame_id,
                information_class=InformationClass.HARD_FACT,
            ))
        
        for char in dense_entities["character_types"]:
            evidence_records.append(EvidenceRecord(
                entity=char,
                type=EvidenceType.PERSON,
                frame_id=frame_id,
                confidence=0.75,
                source=SourceModel.FLORENCE2,
                evidence_text=f"Character detected: {char}",
                provenance={"task": "dense_region_caption", "model": "florence2"},
                sequence_position=frame_id,
                information_class=InformationClass.HARD_FACT,
            ))
        
        for action in dense_entities["action_verbs"]:
            evidence_records.append(EvidenceRecord(
                entity=action,
                type=EvidenceType.ACTION,
                frame_id=frame_id,
                confidence=0.7,
                source=SourceModel.FLORENCE2,
                evidence_text=f"Action detected: {action}",
                provenance={"task": "dense_region_caption", "model": "florence2"},
                sequence_position=frame_id,
                information_class=InformationClass.SOFT_INFERENCE,
            ))
        
        for spatial in dense_entities["spatial_phrases"]:
            evidence_records.append(EvidenceRecord(
                entity=spatial,
                type=EvidenceType.SPATIAL_FACT,
                frame_id=frame_id,
                confidence=0.7,
                source=SourceModel.FLORENCE2,
                evidence_text=f"Spatial relation: {spatial}",
                provenance={"task": "dense_region_caption", "model": "florence2"},
                sequence_position=frame_id,
                information_class=InformationClass.HARD_FACT,
            ))
        
        all_objects = labels + dense_entities["object_descriptions"]
        all_objects = _dedupe(all_objects)
        
        scene = ""
        if detailed:
            first_sent = detailed.split(".")[0]
            scene_keywords = ["scene", "street", "city", "room", "interior", "forest", "woods", "restaurant", "kitchen"]
            if any(kw in first_sent.lower() for kw in scene_keywords):
                scene = first_sent
            else:
                scene = first_sent[:120]
        
        style_mood = ""
        mood_kws = {"whimsical", "playful", "serene", "peaceful", "cheerful", "joyful",
                    "festive", "lively", "dark", "gloomy", "mysterious", "ominous",
                    "cozy", "warm", "cold", "bright", "colorful", "vibrant", "dull",
                    "anime", "illustration", "cartoon", "sketch", "painting", "photograph"}
        for kw in mood_kws:
            if kw in detailed.lower():
                style_mood = kw
                break
        
        detailed_lower = detailed.lower()
        detailed_chars = []
        for kw in _CHARACTER_KEYWORDS:
            if re.search(r"\b" + re.escape(kw) + r"\b", detailed_lower):
                detailed_chars.append(kw)
        
        detailed_actions = []
        for verb in _ACTION_VERBS:
            if re.search(r"\b" + re.escape(verb) + r"\b", detailed_lower):
                detailed_actions.append(verb)
        
        all_character_types = _dedupe(dense_entities["character_types"] + detailed_chars)
        all_action_verbs = _dedupe(dense_entities["action_verbs"] + detailed_actions)
        
        runtime_s = round(time.perf_counter() - t0, 2)
        
        return VisualObservations(
            image_id=image_id,
            frame_id=frame_id,
            detailed_caption=detailed,
            objects=all_objects,
            od_labels=labels,
            characters=all_character_types,
            actions=all_action_verbs,
            spatial_relations=dense_entities["spatial_phrases"],
            region_descriptions=dense_entities["region_descriptions"],
            style_or_mood=style_mood,
            ocr_text="",
            grounding_detections=[],
            evidence_records=evidence_records,
            runtime_s=runtime_s,
            model_load_s=self._load_time_s,
            models_used=models_used,
        )
    
    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_id": self._model_id,
            "load_mode": self._load_mode,
            "load_time_s": self._load_time_s,
            "device": self._device,
            "is_loaded": self.is_loaded,
        }