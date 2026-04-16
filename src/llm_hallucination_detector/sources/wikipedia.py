from __future__ import annotations

import logging
from typing import List

import requests
import wikipediaapi

from llm_hallucination_detector.settings import CacheSettings, WikipediaSourceSettings
from llm_hallucination_detector.sources.base import Document
from llm_hallucination_detector.storage.cache import DiskCache

logger = logging.getLogger(__name__)


class WikipediaSource:
    name = "wikipedia"

    def __init__(self, settings: WikipediaSourceSettings, cache: CacheSettings) -> None:
        self.settings = settings
        self.cache = DiskCache(settings.cache_dir, cache.mode == "disk")
        self.wiki = wikipediaapi.Wikipedia(
            language="en",
            extract_format=wikipediaapi.ExtractFormat.WIKI,
        )

    def fetch(self, query: str) -> List[Document]:
        if not query.strip():
            return []
        titles = self._search_titles(query)
        documents: List[Document] = []
        for title in titles:
            cached = self.cache.get(title)
            if cached is not None:
                documents.append(Document(text=cached, source=self.name, metadata={"title": title}))
                continue
            page = self.wiki.page(title)
            if not page.exists() or not page.text:
                continue
            text = page.text
            self.cache.set(title, text)
            documents.append(Document(text=text, source=self.name, metadata={"title": page.title}))
        return documents

    def _search_titles(self, query: str) -> List[str]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
        }
        try:
            response = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params=params,
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            logger.warning("Wikipedia search failed: %s", exc)
            return []

        search_results = data.get("query", {}).get("search", [])
        titles = [item.get("title") for item in search_results if item.get("title")]
        return titles[: self.settings.search_top_k]
