from src.retrieval.wiki_loader import fetch_wikipedia_page
from src.retrieval.news_loader import fetch_news_articles
from src.retrieval.bm25 import build_bm25, bm25_search
from src.retrieval.embeddings import EmbeddingIndex


class EvidenceRetriever:
    def __init__(self):
        self.embedding_index = EmbeddingIndex()

    def retrieve(self, claim: str, wiki_titles: list[str], top_k: int = 5):
        documents = []

        # Wikipedia
        for title in wiki_titles:
            text = fetch_wikipedia_page(title)
            if text:
                documents.append(text)

        # News
        news = fetch_news_articles(claim)
        for n in news:
            documents.append(n["url"])

        if not documents:
            return []

        # Build indices
        bm25 = build_bm25(documents)
        self.embedding_index.build(documents)

        lexical_hits = bm25_search(bm25, claim, documents, top_k)
        semantic_hits = self.embedding_index.search(claim, top_k)

        # Merge
        return list(dict.fromkeys(lexical_hits + semantic_hits))
