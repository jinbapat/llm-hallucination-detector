from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List

import requests
import trafilatura
from gdelt import gdelt

from llm_hallucination_detector.settings import CacheSettings, NewsSourceSettings
from llm_hallucination_detector.sources.base import Document
from llm_hallucination_detector.storage.cache import DiskCache

logger = logging.getLogger(__name__)


class GDELTSource:
    name = "news"

    def __init__(self, settings: NewsSourceSettings, cache: CacheSettings) -> None:
        self.settings = settings
        self.cache = DiskCache(settings.cache_dir, cache.mode == "disk")
        self.client = gdelt.GDELT(version=2)

    def fetch(self, query: str) -> List[Document]:
        if not query.strip():
            return []
        end = datetime.utcnow()
        start = end - timedelta(days=self.settings.days_back)

        results = self.client.Search(
            query=query,
            table="events",
            start_date=start.strftime("%Y %m %d"),
            end_date=end.strftime("%Y %m %d"),
            maxrecords=self.settings.max_records,
        )
        if results is None or results.empty:
            return []

        documents: List[Document] = []
        for _, row in results.iterrows():
            url = row.get("SOURCEURL", "")
            if not url:
                continue
            text = self._fetch_article(url)
            if not text:
                continue
            documents.append(
                Document(
                    text=text,
                    source=self.name,
                    metadata={
                        "url": url,
                        "date": row.get("SQLDATE", ""),
                    },
                )
            )
        return documents

    def _fetch_article(self, url: str) -> str | None:
        cached = self.cache.get(url)
        if cached is not None:
            return cached
        try:
            response = requests.get(
                url,
                timeout=self.settings.request_timeout,
                headers={"User-Agent": "lhd/1.0"},
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("News fetch failed for %s: %s", url, exc)
            return None

        extracted = trafilatura.extract(response.text)
        if extracted:
            self.cache.set(url, extracted)
        return extracted
