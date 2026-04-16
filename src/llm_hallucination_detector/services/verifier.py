from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from llm_hallucination_detector.settings import VerifierSettings
from llm_hallucination_detector.utils.device import resolve_device

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    label: str
    score: float
    evidence: str | None


class NLIVerifier:
    def __init__(self, settings: VerifierSettings) -> None:
        self.settings = settings
        self.device = resolve_device(settings.device)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            settings.model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ).to(self.device)
        self.max_length = settings.max_length
        self.batch_size = settings.batch_size
        self._label_ids = self._resolve_label_ids()
        self._lock = Lock()

    def verify(self, claim: str, evidence: List[str]) -> VerificationResult:
        if not evidence:
            return VerificationResult(
                label="not_enough_evidence",
                score=0.0,
                evidence=None,
            )

        best: VerificationResult | None = None
        entail_id = self._label_ids["entailment"]

        for batch_start in range(0, len(evidence), self.batch_size):
            batch = evidence[batch_start : batch_start + self.batch_size]
            pairs = [(text, claim) for text in batch]
            inputs = self.tokenizer(
                pairs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            with self._lock, torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for idx, prob in enumerate(probs):
                label, score = self._select_label(prob)
                result = VerificationResult(
                    label=label,
                    score=float(score),
                    evidence=batch[idx],
                )
                if best is None:
                    best = result
                    continue
                if prob[entail_id] > best.score:
                    best = result

        return best or VerificationResult(label="not_enough_evidence", score=0.0, evidence=None)

    def _select_label(self, probabilities) -> tuple[str, float]:
        entail_id = self._label_ids["entailment"]
        contra_id = self._label_ids["contradiction"]
        neutral_id = self._label_ids["neutral"]
        labels = {
            "entailed": probabilities[entail_id],
            "contradicted": probabilities[contra_id],
            "neutral": probabilities[neutral_id],
        }
        best_label = max(labels, key=labels.get)
        return best_label, labels[best_label]

    def _resolve_label_ids(self) -> dict:
        label2id = {k.lower(): v for k, v in self.model.config.label2id.items()}

        def _find(keys: List[str], fallback: int) -> int:
            for key in keys:
                for label, idx in label2id.items():
                    if key in label:
                        return idx
            return fallback

        return {
            "contradiction": _find(["contradiction", "contradict"], 0),
            "neutral": _find(["neutral"], 1),
            "entailment": _find(["entail"], 2),
        }
