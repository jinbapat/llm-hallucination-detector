from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import numpy as np

from llm_hallucination_detector.settings import IndexSettings
from llm_hallucination_detector.sources.base import Document

logger = logging.getLogger(__name__)


class VectorIndex:
    def __init__(self, embedder, settings: IndexSettings) -> None:
        self.embedder = embedder
        self.settings = settings
        self.enabled = settings.enabled
        self.persist = settings.persist
        self.path = Path(settings.path)
        self.metric = settings.metric
        self.max_docs = settings.max_docs
        self.flush_interval = settings.flush_interval

        self._index = None
        self._embeddings: np.ndarray | None = None
        self._documents: List[Document] = []
        self._doc_hashes: set[str] = set()
        self._added_since_flush = 0

        self._faiss = None
        try:
            import faiss  # type: ignore

            self._faiss = faiss
        except Exception:
            logger.info("FAISS not available; using numpy fallback")

        if self.enabled and self.persist:
            self._load()

    def add_documents(self, documents: List[Document]) -> None:
        if not self.enabled or not documents:
            return

        new_docs = []
        for doc in documents:
            doc_hash = self._hash_document(doc)
            if doc_hash in self._doc_hashes:
                continue
            self._doc_hashes.add(doc_hash)
            new_docs.append(doc)

        if not new_docs:
            return

        embeddings = self.embedder.encode([doc.text for doc in new_docs])
        embeddings = self._normalize(embeddings)
        self._documents.extend(new_docs)

        if self._faiss:
            if self._index is None:
                self._index = self._faiss.IndexFlatIP(embeddings.shape[1])
            self._index.add(embeddings)
        else:
            if self._embeddings is None:
                self._embeddings = embeddings
            else:
                self._embeddings = np.vstack([self._embeddings, embeddings])

        if len(self._documents) > self.max_docs:
            self._trim_to_max()

        self._added_since_flush += len(new_docs)
        if self.persist and self._added_since_flush >= self.flush_interval:
            self.save()
            self._added_since_flush = 0

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        if not self.enabled or not self._documents:
            return []
        embeddings = self.embedder.encode([query])
        embeddings = self._normalize(embeddings)

        if self._faiss and self._index is not None:
            scores, indices = self._index.search(embeddings, top_k)
            return [self._documents[i] for i in indices[0] if i < len(self._documents)]

        if self._embeddings is None:
            return []
        scores = np.dot(self._embeddings, embeddings[0])
        best_indices = np.argsort(scores)[::-1][:top_k]
        return [self._documents[i] for i in best_indices]

    def save(self) -> None:
        if not (self.enabled and self.persist):
            return
        self.path.mkdir(parents=True, exist_ok=True)

        docs_path = self.path / "documents.jsonl"
        with docs_path.open("w", encoding="utf-8") as handle:
            for doc in self._documents:
                handle.write(json.dumps({
                    "text": doc.text,
                    "source": doc.source,
                    "metadata": doc.metadata,
                }) + "\n")

        meta_path = self.path / "meta.json"
        meta_path.write_text(
            json.dumps({
                "metric": self.metric,
                "count": len(self._documents),
            }),
            encoding="utf-8",
        )

        if self._faiss and self._index is not None:
            self._faiss.write_index(self._index, str(self.path / "index.faiss"))
        elif self._embeddings is not None:
            np.save(self.path / "embeddings.npy", self._embeddings)

    def _load(self) -> None:
        if not self.path.exists():
            return

        docs_path = self.path / "documents.jsonl"
        if docs_path.exists():
            with docs_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    data = json.loads(line)
                    doc = Document(
                        text=data.get("text", ""),
                        source=data.get("source", ""),
                        metadata=data.get("metadata", {}),
                    )
                    self._documents.append(doc)
                    self._doc_hashes.add(self._hash_document(doc))

        if self._faiss and (self.path / "index.faiss").exists():
            self._index = self._faiss.read_index(str(self.path / "index.faiss"))
        elif (self.path / "embeddings.npy").exists():
            self._embeddings = np.load(self.path / "embeddings.npy")

        if self._documents and self._index is None and self._embeddings is None:
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        if not self._documents:
            return
        embeddings = self.embedder.encode([doc.text for doc in self._documents])
        embeddings = self._normalize(embeddings)

        if self._faiss:
            self._index = self._faiss.IndexFlatIP(embeddings.shape[1])
            self._index.add(embeddings)
            self._embeddings = None
        else:
            self._embeddings = embeddings

    def _trim_to_max(self) -> None:
        if len(self._documents) <= self.max_docs:
            return
        self._documents = self._documents[-self.max_docs :]
        self._doc_hashes = {self._hash_document(doc) for doc in self._documents}
        self._rebuild_index()

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        if self.metric != "cosine":
            return embeddings.astype("float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (embeddings / norms).astype("float32")

    @staticmethod
    def _hash_document(doc: Document) -> str:
        return str(hash((doc.text, doc.source)))
