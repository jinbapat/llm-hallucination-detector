from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from llm_hallucination_detector.settings import ClaimExtractionSettings
from llm_hallucination_detector.utils.device import resolve_device
from llm_hallucination_detector.utils.text import parse_json_array

logger = logging.getLogger(__name__)


class ClaimExtractor:
    def __init__(self, settings: ClaimExtractionSettings) -> None:
        self.settings = settings
        self.device = resolve_device(settings.device)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            settings.model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ).to(self.device)
        self.max_new_tokens = settings.max_new_tokens
        self.temperature = settings.temperature
        self.prompt_template = Path(settings.prompt_path).read_text(encoding="utf-8")
        self._lock = Lock()

    def extract(self, question: str, answer: str) -> List[str]:
        prompt = self.prompt_template.format(
            question=question.strip(),
            answer=answer.strip(),
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0.0,
        }
        if self.temperature > 0.0:
            generation_kwargs["temperature"] = self.temperature

        with self._lock, torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        claims = parse_json_array(decoded)
        if not claims:
            logger.warning("Claim extraction produced no claims or invalid JSON")
        return claims
