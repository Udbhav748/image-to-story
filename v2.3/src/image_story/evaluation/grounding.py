"""Grounding evaluation metrics (CLIP, NLI, attribute checks)."""
import os
import time
import re
from typing import Any

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, AutoModelForSequenceClassification

from ..domain.schemas import EvaluationResult, StoryDraft
from ..domain.enums import EvaluationMetric

from ..domain.exceptions import MetricComputationError


class GroundingEvaluator:
    """Evaluate story grounding using CLIP, NLI, and attribute checks."""
    
    def __init__(self, device: str = "cpu"):
        self._device = device
        self._clip_model = None
        self._clip_processor = None
        self._nli_model = None
        self._nli_tokenizer = None
        self._contra_idx = None
        
        # CLIP normalization parameters
        self._clip_low = 0.15
        self._clip_span = 0.15
        
        # Attribute check vocabularies
        self._colors = {"white", "black", "red", "blue", "green", "yellow", "brown", "grey", "pink", "orange", "purple", "gray"}
        self._material_families = {
            "wood": {"wood", "wooden", "oak"},
            "glass": {"glass"},
            "metal": {"metal", "steel"},
            "plastic": {"plastic"},
            "stone": {"stone", "marble"},
            "brick": {"brick"},
            "leather": {"leather"},
        }
        self._typical_color = {"wood": {"brown"}}
        self._material_words = {w: fam for fam, ws in self._material_families.items() for w in ws}
    
    def _load_clip(self):
        if self._clip_model is None:
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._clip_model.to(self._device)
    
    def _load_nli(self):
        if self._nli_model is None:
            nli_name = "cross-encoder/nli-MiniLM2-L6-H768"
            self._nli_tokenizer = AutoTokenizer.from_pretrained(nli_name)
            self._nli_model = AutoModelForSequenceClassification.from_pretrained(nli_name).eval()
            self._nli_model.to(self._device)
            self._contra_idx = [i for i, l in self._nli_model.config.id2label.items() if l.lower() == "contradiction"][0]
    
    def evaluate(
        self,
        image: Image.Image,
        caption: str,
        story: StoryDraft,
    ) -> EvaluationResult:
        """Full grounding evaluation."""
        self._load_clip()
        self._load_nli()
        
        t0 = time.perf_counter()
        
        # Split story into sentences
        sentences = self._split_sentences(story.text)
        if not sentences:
            sentences = [story.text]
        
        # CLIP evaluation
        clip_start = time.perf_counter()
        clip_sims = self._compute_clip_sims(image, sentences)
        clip_s = time.perf_counter() - clip_start
        
        clip_mean = float(clip_sims.mean()) if len(clip_sims) > 0 else 0.0
        
        # NLI evaluation
        nli_start = time.perf_counter()
        nli_contras = self._compute_nli_contradictions(caption, sentences)
        nli_s = time.perf_counter() - nli_start
        
        nli_mean = float(nli_contras.mean()) if len(nli_contras) > 0 else 0.0
        
        # Attribute conflict check
        rules_start = time.perf_counter()
        conflicts = self._attribute_conflict(caption, story.text)
        rules_s = time.perf_counter() - rules_start
        
        # Compute grounding score
        clip_n = min(max((clip_mean - self._clip_low) / self._clip_span, 0.0), 1.0)
        grounding_score = 0.4 * clip_n + 0.4 * (1 - nli_mean) + 0.2 * (0 if conflicts else 1)
        
        # Length validity
        word_count = len(story.text.split())
        length_valid = 80 <= word_count <= 120
        
        # Grounding pass
        grounding_pass = grounding_score >= 0.60 and not conflicts and length_valid
        
        # Repetition metrics
        repetition = self._repetition_score(story.text)
        
        total_s = time.perf_counter() - t0
        
        return EvaluationResult(
            grounding_score=grounding_score,
            clip_image_story_mean=clip_mean,
            clip_image_story_min=float(clip_sims.min()) if len(clip_sims) > 0 else 0.0,
            clip_image_caption=0.0,  # Not computed here
            nli_contra_mean=nli_mean,
            nli_contra_max=float(nli_contras.max()) if len(nli_contras) > 0 else 0.0,
            attribute_conflict=conflicts,
            repetition_rate=repetition["repetition_rate"],
            length_valid=length_valid,
            grounding_pass=grounding_pass,
            eval_clip_s=round(clip_s, 3),
            eval_nli_s=round(nli_s, 3),
            eval_rules_s=round(rules_s, 4),
            eval_total_s=round(total_s, 3),
        )
    
    def _split_sentences(self, text: str) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    
    def _compute_clip_sims(self, image: Image.Image, texts: list[str]) -> torch.Tensor:
        inputs = self._clip_processor(
            text=texts, images=image, return_tensors="pt", padding=True, truncation=True, max_length=77
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            sims = logits_per_image[0] / self._clip_model.logit_scale.exp()
        
        return sims.cpu()
    
    def _compute_nli_contradictions(self, caption: str, sentences: list[str]) -> torch.Tensor:
        inputs = self._nli_tokenizer(
            [caption] * len(sentences), sentences,
            return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self._nli_model(**inputs).logits
            probs = logits.softmax(-1)
        
        return probs[:, self._contra_idx].cpu()
    
    def _attribute_conflict(self, caption: str, story: str) -> list[str]:
        cw = self._words(caption)
        sw = self._words(story)
        bad = []
        
        cap_colors = cw & self._colors
        story_colors = sw & self._colors
        
        if cap_colors:
            bad += sorted(story_colors - cap_colors)
        
        cap_fams = {self._material_words[w] for w in cw if w in self._material_words}
        story_mats = sorted(sw & set(self._material_words))
        
        for w in story_mats:
            fam = self._material_words[w]
            if cap_fams and fam not in cap_fams:
                bad.append(w)
            elif not cap_fams and cap_colors and fam in self._typical_color and not (self._typical_color[fam] & cap_colors):
                bad.append(w)
        
        return bad
    
    def _words(self, text: str) -> set[str]:
        return {"grey" if w == "gray" else w for w in re.findall(r"[a-z]+", text.lower())}
    
    def _repetition_score(self, story: str) -> dict[str, float]:
        import re
        from collections import Counter
        
        sents = self._split_sentences(story)
        toks = re.findall(r"[a-z']+", story.lower())
        
        repeated_sentences = sum(c - 1 for c in Counter(sents).values() if c > 1)
        trigrams = list(zip(toks, toks[1:], toks[2:])) if len(toks) > 2 else []
        repeated_trigrams = sum(c - 1 for c in Counter(trigrams).values() if c > 1)
        
        rate = repeated_trigrams / len(trigrams) if trigrams else 0.0
        
        return {
            "repeated_sentences": repeated_sentences,
            "repeated_trigrams": repeated_trigrams,
            "repetition_rate": round(rate, 4),
            "distinct3": round(1.0 - rate if trigrams else 1.0, 4),
        }


def compute_grounding_score(clip_mean: float, nli_contra_mean: float, has_conflict: bool) -> float:
    """Compute grounding score from components."""
    clip_n = min(max((clip_mean - 0.15) / 0.15, 0.0), 1.0)
    return 0.4 * clip_n + 0.4 * (1 - nli_contra_mean) + 0.2 * (0 if has_conflict else 1)