from __future__ import annotations

from typing import List, Sequence

from llm_hallucination_detector.settings import RouterSettings, TopicSettings


class TopicRouter:
    def __init__(self, settings: RouterSettings, embedder=None) -> None:
        self.settings = settings
        self.embedder = embedder
        self._topics = settings.topics
        self._topic_embeddings = None

        if self.settings.semantic and self.embedder and self._topics:
            texts = [self._topic_text(topic) for topic in self._topics]
            self._topic_embeddings = self.embedder.encode(texts)

    def select_sources(self, claim: str, available_sources: Sequence[str]) -> List[str]:
        if not self.settings.enabled or not self._topics:
            return list(available_sources)

        claim_lower = claim.lower()
        best_topic, best_score = self._keyword_match(claim_lower)
        if best_topic and best_score > 0:
            return self._filter_sources(best_topic, available_sources)

        if self.settings.semantic and self._topic_embeddings is not None and self.embedder:
            claim_emb = self.embedder.encode([claim])
            scores = (self._topic_embeddings @ claim_emb[0]).tolist()
            best_idx = max(range(len(scores)), key=scores.__getitem__)
            if scores[best_idx] >= self.settings.semantic_threshold:
                return self._filter_sources(self._topics[best_idx], available_sources)

        return list(available_sources)

    def _keyword_match(self, claim_lower: str) -> tuple[TopicSettings | None, int]:
        best_topic = None
        best_score = 0
        for topic in self._topics:
            score = sum(1 for kw in topic.keywords if kw.lower() in claim_lower)
            if score > best_score:
                best_topic = topic
                best_score = score
        return best_topic, best_score

    def _filter_sources(self, topic: TopicSettings, available_sources: Sequence[str]) -> List[str]:
        return [source for source in topic.sources if source in available_sources]

    @staticmethod
    def _topic_text(topic: TopicSettings) -> str:
        return " ".join([topic.name] + list(topic.keywords))
