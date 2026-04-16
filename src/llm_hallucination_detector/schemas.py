from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ClaimsRequest(BaseModel):
    question: str
    answer: str


class ClaimsResponse(BaseModel):
    claims: List[str] = Field(default_factory=list)


class EvidenceRequest(BaseModel):
    claim: str
    sources: Optional[List[str]] = None
    top_k: Optional[int] = None


class EvidenceItem(BaseModel):
    text: str
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidenceResponse(BaseModel):
    claim: str
    evidence: List[EvidenceItem] = Field(default_factory=list)


class VerifyRequest(BaseModel):
    claim: str
    evidence: List[str] = Field(default_factory=list)


class VerifyResponse(BaseModel):
    claim: str
    label: str
    score: float
    evidence: Optional[str] = None


class DetectRequest(BaseModel):
    question: str
    answer: str
    sources: Optional[List[str]] = None
    top_k: Optional[int] = None


class ClaimResult(BaseModel):
    claim: str
    label: str
    score: float
    evidence: Optional[str] = None


class DetectResponse(BaseModel):
    hallucination_score: float
    claims: List[ClaimResult] = Field(default_factory=list)
