from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any


class LRUCache:
    """Thread-safe LRU cache for embedding vectors and NLI pair scores."""

    def __init__(self, maxsize: int = 8192) -> None:
        self.maxsize = maxsize
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key not in self._data:
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return self._data[key]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "size": len(self._data),
                "hits": self.hits,
                "misses": self.misses,
            }


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pair_key(premise: str, hypothesis: str) -> str:
    return text_key(f"{premise}\x1f{hypothesis}")


EMBED_CACHE = LRUCache(maxsize=16384)
NLI_CACHE = LRUCache(maxsize=16384)
