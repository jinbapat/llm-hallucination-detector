from __future__ import annotations

import logging
from typing import Dict, List, Sequence

from rank_bm25 import BM25Okapi

from llm_hallucination_detector.routing import TopicRouter
from llm_hallucination_detector.settings import CacheSettings, RetrievalSettings, RouterSettings, SourceSettings
from llm_hallucination_detector.sources.base import Document, EvidenceSource
from llm_hallucination_detector.sources.gdelt import GDELTSource
from llm_hallucination_detector.sources.wikipedia import WikipediaSource
from llm_hallucination_detector.storage.vector_index import VectorIndex
from llm_hallucination_detector.utils.text import chunk_text
from llm_hallucination_detector.services.embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)


class EvidenceRetriever:
    def __init__(
        self,
        retrieval: RetrievalSettings,
        sources: SourceSettings,
        router: RouterSettings,
        cache: CacheSettings,
        embedder: EmbeddingModel,
        vector_index: VectorIndex,
    ) -> None:
        self.retrieval = retrieval
        self.embedder = embedder
        self.vector_index = vector_index
        self.sources: Dict[str, EvidenceSource] = {}

        if sources.wikipedia.enabled:
            self.sources["wikipedia"] = WikipediaSource(sources.wikipedia, cache)
        if sources.news.enabled:
            self.sources["news"] = GDELTSource(sources.news, cache)

        self.router = TopicRouter(router, embedder)

    def retrieve(self, claim: str, source_names: Sequence[str] | None, top_k: int | None) -> List[Document]:
        if not claim.strip():
            return []

        available_sources = list(self.sources.keys())
        if source_names:
            selected_sources = [s for s in source_names if s in available_sources]
        else:
            selected_sources = self.router.select_sources(claim, available_sources)

        documents: List[Document] = []
        for name in selected_sources:
            documents.extend(self.sources[name].fetch(claim))

        if not documents:
            return []

        chunked = self._chunk_documents(documents)
        chunked = chunked[: self.retrieval.max_documents]
        if not chunked:
            return []

        top_k = top_k or self.retrieval.vector_top_k

        bm25 = self._build_bm25(chunked)
        lexical_hits = self._bm25_search(bm25, claim, chunked, self.retrieval.bm25_top_k)

        self.vector_index.add_documents(chunked)
        semantic_hits = self.vector_index.search(claim, top_k)

        return self._merge_results(lexical_hits, semantic_hits, top_k)

    def clear_caches(self) -> None:
        for source in self.sources.values():
            clear_fn = getattr(source, "clear_cache", None)
            if callable(clear_fn):
                clear_fn()

    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        chunked: List[Document] = []
        for doc in documents:
            chunks = chunk_text(
                doc.text,
                chunk_size=self.retrieval.chunk_size,
                overlap=self.retrieval.chunk_overlap,
            )
            for idx, chunk in enumerate(chunks):
                metadata = dict(doc.metadata)
                metadata["chunk"] = idx
                chunked.append(Document(text=chunk, source=doc.source, metadata=metadata))
        return chunked

    @staticmethod
    def _build_bm25(documents: List[Document]) -> BM25Okapi:
        tokenized = [doc.text.split() for doc in documents]
        return BM25Okapi(tokenized)

    @staticmethod
    def _bm25_search(bm25: BM25Okapi, query: str, documents: List[Document], top_k: int) -> List[Document]:
        scores = bm25.get_scores(query.split())
        ranked = sorted(
            zip(scores, documents),
            key=lambda x: x[0],
            reverse=True,
        )
        return [doc for _, doc in ranked[:top_k]]

    @staticmethod
    def _merge_results(
        lexical: List[Document],
        semantic: List[Document],
        top_k: int,
    ) -> List[Document]:
        merged: List[Document] = []
        seen = set()
        for doc in lexical + semantic:
            key = (doc.source, doc.metadata.get("title"), doc.metadata.get("url"), doc.text[:200])
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
            if len(merged) >= top_k:
                break
        return merged
