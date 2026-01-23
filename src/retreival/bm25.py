from rank_bm25 import BM25Okapi

def build_bm25(corpus: list[str]):
    tokenized = [doc.split() for doc in corpus]
    return BM25Okapi(tokenized)

def bm25_search(bm25, query: str, corpus: list[str], top_k: int = 5):
    scores = bm25.get_scores(query.split())
    ranked = sorted(
        zip(scores, corpus),
        key=lambda x: x[0],
        reverse=True
    )
    return [doc for _, doc in ranked[:top_k]]
