from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from llm_hallucination_detector.settings import EmbeddingsSettings
from llm_hallucination_detector.utils.device import resolve_device


class EmbeddingModel:
    def __init__(self, settings: EmbeddingsSettings) -> None:
        self.settings = settings
        device = resolve_device(settings.device)
        self.model = SentenceTransformer(settings.model_name, device=device.type)

    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.settings.normalize,
        )
        return embeddings.astype("float32")
