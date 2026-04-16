from __future__ import annotations

import hashlib
import re
from pathlib import Path


class DiskCache:
    def __init__(self, base_dir: str | Path, enabled: bool) -> None:
        self.enabled = enabled
        self.base_dir = Path(base_dir)
        if self.enabled:
            self.base_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", key.strip())
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
        return self.base_dir / f"{safe}_{digest}.txt"

    def get(self, key: str) -> str | None:
        if not self.enabled:
            return None
        path = self._key_to_path(key)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def set(self, key: str, value: str) -> None:
        if not self.enabled:
            return
        path = self._key_to_path(key)
        path.write_text(value, encoding="utf-8")
