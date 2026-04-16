from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI

from llm_hallucination_detector.logging_config import configure_logging
from llm_hallucination_detector.pipeline import HallucinationDetector
from llm_hallucination_detector.schemas import (
    ClaimsRequest,
    ClaimsResponse,
    DetectRequest,
    DetectResponse,
    EvidenceRequest,
    EvidenceResponse,
    VerifyRequest,
    VerifyResponse,
)
from llm_hallucination_detector.settings import Settings, load_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    configure_logging(settings)

    app = FastAPI(title="LLM Hallucination Detector", version="0.1.0")

    detector = _get_detector(settings)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/claims", response_model=ClaimsResponse)
    def claims(request: ClaimsRequest) -> ClaimsResponse:
        claims_list = detector.extract_claims(request.question, request.answer)
        return ClaimsResponse(claims=claims_list)

    @app.post("/evidence", response_model=EvidenceResponse)
    def evidence(request: EvidenceRequest) -> EvidenceResponse:
        documents = detector.retriever.retrieve(request.claim, request.sources, request.top_k)
        return EvidenceResponse(
            claim=request.claim,
            evidence=[
                {"text": doc.text, "source": doc.source, "metadata": doc.metadata}
                for doc in documents
            ],
        )

    @app.post("/verify", response_model=VerifyResponse)
    def verify(request: VerifyRequest) -> VerifyResponse:
        result = detector.verify_claim(request.claim, request.evidence)
        return VerifyResponse(
            claim=request.claim,
            label=result["label"],
            score=result["score"],
            evidence=result.get("evidence"),
        )

    @app.post("/detect", response_model=DetectResponse)
    def detect(request: DetectRequest) -> DetectResponse:
        result = detector.detect(
            request.question,
            request.answer,
            request.sources,
            request.top_k,
        )
        return DetectResponse(
            hallucination_score=result["hallucination_score"],
            claims=[
                {
                    "claim": item.claim,
                    "label": item.label,
                    "score": item.score,
                    "evidence": item.evidence,
                }
                for item in result["claims"]
            ],
        )

    return app


@lru_cache(maxsize=1)
def _get_detector(settings: Settings) -> HallucinationDetector:
    return HallucinationDetector(settings)
