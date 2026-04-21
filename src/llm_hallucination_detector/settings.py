from __future__ import annotations

import os
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"


class ClaimExtractionSettings(BaseModel):
    model_name: str = "google/flan-t5-large"
    device: str = "auto"
    max_new_tokens: int = 256
    temperature: float = 0.0
    prompt_path: str = "experiments/claim_extraction/prompts/extract_claims.txt"


class VerifierSettings(BaseModel):
    model_name: str = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    device: str = "auto"
    max_length: int = 512
    batch_size: int = 8


class EmbeddingsSettings(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"
    normalize: bool = True


class RetrievalSettings(BaseModel):
    bm25_top_k: int = 5
    vector_top_k: int = 5
    chunk_size: int = 300
    chunk_overlap: int = 50
    max_documents: int = 2000


class WikipediaSourceSettings(BaseModel):
    enabled: bool = True
    cache_dir: str = "data/wiki"
    search_top_k: int = 3


class NewsSourceSettings(BaseModel):
    enabled: bool = True
    days_back: int = 30
    max_records: int = 50
    cache_dir: str = "data/news"
    request_timeout: int = 10


class SourceSettings(BaseModel):
    wikipedia: WikipediaSourceSettings = WikipediaSourceSettings()
    news: NewsSourceSettings = NewsSourceSettings()


class TopicSettings(BaseModel):
    name: str
    keywords: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class RouterSettings(BaseModel):
    enabled: bool = True
    semantic: bool = True
    semantic_threshold: float = 0.35
    topics: List[TopicSettings] = Field(default_factory=list)


class IndexSettings(BaseModel):
    enabled: bool = True
    persist: bool = True
    path: str = "data/index"
    metric: str = "cosine"
    max_docs: int = 200000
    flush_interval: int = 100


class CacheSettings(BaseModel):
    mode: str = "none"
    clear_on_response: bool = False


class ScoringSettings(BaseModel):
    neutral_weight: float = 0.5
    insufficient_weight: float = 0.75


class ModelSettings(BaseModel):
    claim_extraction: ClaimExtractionSettings = ClaimExtractionSettings()
    verifier: VerifierSettings = VerifierSettings()
    embeddings: EmbeddingsSettings = EmbeddingsSettings()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__")

    service: ServiceSettings = ServiceSettings()
    models: ModelSettings = ModelSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    sources: SourceSettings = SourceSettings()
    router: RouterSettings = RouterSettings()
    index: IndexSettings = IndexSettings()
    cache: CacheSettings = CacheSettings()
    scoring: ScoringSettings = ScoringSettings()


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def load_settings(path: str | Path | None = None) -> Settings:
    config_path = Path(path or os.getenv("LHD_CONFIG_PATH", "config/models.yaml"))
    data = _load_yaml(config_path)
    return Settings(**data)
