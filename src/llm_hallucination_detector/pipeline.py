from __future__ import annotations

from typing import List, Sequence

from llm_hallucination_detector.scoring import AnswerScorer, ClaimScore
from llm_hallucination_detector.services.claim_extractor import ClaimExtractor
from llm_hallucination_detector.services.embedding_model import EmbeddingModel
from llm_hallucination_detector.services.retriever import EvidenceRetriever
from llm_hallucination_detector.services.verifier import NLIVerifier
from llm_hallucination_detector.settings import Settings
from llm_hallucination_detector.storage.vector_index import VectorIndex


class HallucinationDetector:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.extractor = ClaimExtractor(settings.models.claim_extraction)
        self.embedder = EmbeddingModel(settings.models.embeddings)
        self.vector_index = VectorIndex(self.embedder, settings.index)
        self.retriever = EvidenceRetriever(
            retrieval=settings.retrieval,
            sources=settings.sources,
            router=settings.router,
            cache=settings.cache,
            embedder=self.embedder,
            vector_index=self.vector_index,
        )
        self.verifier = NLIVerifier(settings.models.verifier)
        self.scorer = AnswerScorer(settings.scoring)

    def extract_claims(self, question: str, answer: str) -> List[str]:
        return self.extractor.extract(question, answer)

    def retrieve_evidence(
        self,
        claim: str,
        sources: Sequence[str] | None = None,
        top_k: int | None = None,
    ) -> List[str]:
        documents = self.retriever.retrieve(claim, sources, top_k)
        return [doc.text for doc in documents]

    def verify_claim(self, claim: str, evidence: List[str]) -> dict:
        result = self.verifier.verify(claim, evidence)
        return {
            "label": result.label,
            "score": result.score,
            "evidence": result.evidence,
        }

    def detect(
        self,
        question: str,
        answer: str,
        sources: Sequence[str] | None = None,
        top_k: int | None = None,
    ) -> dict:
        claims = self.extract_claims(question, answer)
        results: List[ClaimScore] = []
        for claim in claims:
            evidence_docs = self.retriever.retrieve(claim, sources, top_k)
            evidence_texts = [doc.text for doc in evidence_docs]
            verdict = self.verifier.verify(claim, evidence_texts)
            results.append(
                ClaimScore(
                    claim=claim,
                    label=verdict.label,
                    score=verdict.score,
                    evidence=verdict.evidence,
                )
            )
        return self.scorer.aggregate(results)
