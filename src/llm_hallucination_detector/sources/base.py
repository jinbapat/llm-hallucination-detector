from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Protocol


@dataclass
class Document:
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class EvidenceSource(Protocol):
    name: str

    def fetch(self, query: str) -> List[Document]:
        ...
