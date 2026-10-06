"""Story generation module with model adapter pattern."""
import os
import time
import torch
from abc import ABC, abstractmethod
from typing import Any

from ..domain.schemas import StoryDraft, StoryPlan, PipelineConfig
from ..domain.exceptions import ModelGenerationError, ModelLoadError


class StoryGenerator(ABC):
    """Abstract base class for story generation models."""
    
    @property
    @abstractmethod
    def model_id(self) -> str:
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        pass
    
    @abstractmethod
    def load(self) -> None:
        pass
    
    @abstractmethod
    def unload(self) -> None:
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 500,
        temperature: float = 0.0,
        repetition_penalty: float = 1.05,
        seed: int = 0,
    ) -> str:
        pass
    
    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        pass


class QwenGenerator(StoryGenerator):
    """Qwen2.5-0.5B-Instruct story generator."""
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cpu",
    ):
        self._model_id = model_id
        self._device = device
        self._model = None
        self._tokenizer = None
        self._load_time_s = 0.0
    
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
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        t0 = time.perf_counter()
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                torch_dtype=torch.float32,
            )
            self._model.eval()
            self._model.to(self._device)
            self._load_time_s = round(time.perf_counter() - t0, 2)
        except Exception as e:
            raise ModelLoadError(f"Failed to load Qwen model: {e}")
    
    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 500,
        temperature: float = 0.0,
        repetition_penalty: float = 1.05,
        seed: int = 0,
    ) -> str:
        if not self.is_loaded:
            self.load()
        
        torch.manual_seed(seed)
        
        messages = [
            {"role": "system", "content": "You are a creative storyteller."},
            {"role": "user", "content": prompt},
        ]
        
        prompt_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self._tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            if temperature > 0:
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            else:
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
        
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        story = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        
        return story
    
    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_id": self._model_id,
            "load_time_s": self._load_time_s,
            "device": self._device,
            "is_loaded": self.is_loaded,
        }


class StoryGenerationPipeline:
    """High-level story generation pipeline with length control."""
    
    def __init__(
        self,
        generator: StoryGenerator,
        config: PipelineConfig | None = None,
    ):
        self._generator = generator
        self._config = config or PipelineConfig()
    
    def generate_story(
        self,
        prompt: str,
        story_plan: StoryPlan | None = None,
        fix_length: bool = False,
    ) -> StoryDraft:
        """Generate a story with optional length control."""
        
        target_words = self._config.target_story_words
        if story_plan:
            target_words = story_plan.target_total_words
        
        max_new_tokens = min(target_words * 2, 800)
        
        if not fix_length:
            story_text = self._generator.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                repetition_penalty=1.05,
                seed=self._config.seed,
            )
        else:
            story_text = self._generate_with_length_control(prompt, target_words, max_new_tokens)
        
        # Clean up story - ensure complete sentences
        story_text = self._clean_story(story_text, target_words)
        
        return StoryDraft(
            text=story_text,
            story_plan=story_plan,
            word_count=len(story_text.split()),
            generation_time_s=0.0,  # Set by caller
            model_used=self._generator.model_id,
            prompt_used=prompt[:500] + "..." if len(prompt) > 500 else prompt,
        )
    
    def _generate_with_length_control(
        self,
        prompt: str,
        target_words: int,
        max_new_tokens: int,
    ) -> str:
        """Generate with length control - multiple attempts."""
        
        # Progressive prompts for length control
        length_prompts = [
            prompt,
            prompt.replace(
                f"approximately {target_words} words",
                f"about {target_words} words (between {target_words-20} and {target_words+20} words)"
            ),
            prompt.replace(
                f"approximately {target_words} words",
                f"exactly {target_words} words in 6-8 sentences"
            ),
        ]
        
        best_story = ""
        best_diff = float('inf')
        
        for i, length_prompt in enumerate(length_prompts):
            story = self._generator.generate(
                length_prompt,
                max_new_tokens=max_new_tokens + i * 50,
                temperature=0.0,
                repetition_penalty=1.05,
                seed=self._config.seed + i,
            )
            
            story = self._clean_story(story, target_words + 20)
            word_count = len(story.split())
            diff = abs(word_count - target_words)
            
            if target_words - 20 <= word_count <= target_words + 20:
                return story
            
            if diff < best_diff:
                best_diff = diff
                best_story = story
        
        return best_story
    
    def _clean_story(self, story: str, max_words: int) -> str:
        """Clean story - keep complete sentences, respect word limit."""
        import re
        
        sentences = re.split(r"(?<=[.!?])\s+", story.strip())
        
        # Remove incomplete last sentence
        if sentences and not sentences[-1].rstrip('"\'').endswith((".", "!", "?")):
            sentences = sentences[:-1]
        
        kept = []
        word_count = 0
        for sent in sentences:
            sent_words = len(sent.split())
            if word_count + sent_words > max_words:
                break
            kept.append(sent)
            word_count += sent_words
        
        if not kept:
            return story[:max_words * 5]  # fallback
        
        return " ".join(kept)


def create_generator(
    model_type: str = "qwen",
    model_id: str | None = None,
    device: str = "cpu",
) -> StoryGenerator:
    """Factory function to create story generator."""
    if model_type == "qwen":
        return QwenGenerator(model_id=model_id or "Qwen/Qwen2.5-0.5B-Instruct", device=device)
    else:
        raise ValueError(f"Unknown generator type: {model_type}")