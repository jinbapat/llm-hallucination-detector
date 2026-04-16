from __future__ import annotations

from dataclasses import dataclass
from typing import List

from llm_hallucination_detector.settings import ScoringSettings


@dataclass
class ClaimScore:
    claim: str
    label: str
    score: float
    evidence: str | None


class AnswerScorer:
    def __init__(self, settings: ScoringSettings) -> None:
        self.settings = settings

    def aggregate(self, results: List[ClaimScore]) -> dict:
        total = len(results)
        if total == 0:
            return {"hallucination_score": 0.0, "claims": []}

        weighted = 0.0
        for result in results:
            if result.label == "contradicted":
                weighted += 1.0
            elif result.label == "neutral":
                weighted += self.settings.neutral_weight
            elif result.label == "not_enough_evidence":
                weighted += self.settings.insufficient_weight

        return {
            "hallucination_score": weighted / total,
            "claims": results,
        }
