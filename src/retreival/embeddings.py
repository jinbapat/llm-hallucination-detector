from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class EmbeddingIndex:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.corpus = []

    def build(self, documents: list[str]):
        self.corpus = documents
        embeddings = self.model.encode(documents, show_progress_bar=True)
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings.astype("float32"))

    def search(self, query: str, top_k: int = 5):
        q_emb = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(q_emb, top_k)
        return [self.corpus[i] for i in indices[0]]
