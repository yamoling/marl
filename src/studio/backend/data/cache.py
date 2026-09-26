"""Small thread-safe LRU cache and file-stat keys used for mtime-based invalidation."""

import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from pathlib import Path

FileKey = tuple[str, int, int] | None
"""`(path, mtime_ns, size)`, or None when the file does not exist."""

_MISSING = object()


def file_key(path: Path | str) -> FileKey:
    """@ai-generated"""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (str(path), st.st_mtime_ns, st.st_size)


class LRU[K: Hashable, V]:
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._data = OrderedDict[K, V]()
        self._lock = threading.Lock()

    def get(self, key: K) -> V | None:
        """@ai-generated"""
        with self._lock:
            value = self._data.get(key, _MISSING)
            if value is _MISSING:
                return None
            self._data.move_to_end(key)
            return value  # type: ignore[return-value]

    def put(self, key: K, value: V):
        """@ai-generated"""
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def get_or_compute(self, key: K, compute: Callable[[], V]) -> V:
        """
        Return the cached value, computing and storing it on a miss. `compute` runs outside the lock.

        @ai-generated
        """
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
        value = compute()
        self.put(key, value)
        return value

    def discard_if(self, predicate: Callable[[K], bool]):
        """@ai-generated"""
        with self._lock:
            for key in [k for k in self._data if predicate(k)]:
                del self._data[key]

    def clear(self):
        with self._lock:
            self._data.clear()

    def __len__(self):
        return len(self._data)


class ByteLRU[K: Hashable, V](LRU[K, V]):
    """LRU bounded by the total size of its values (as measured by `sizeof`), not their number."""

    def __init__(self, max_bytes: int, sizeof: Callable[[V], int]):
        super().__init__(maxsize=2**62)
        self.max_bytes = max_bytes
        self.sizeof = sizeof
        self._sizes = dict[K, int]()
        self.nbytes = 0

    def put(self, key: K, value: V):
        """Store `value`, evicting the least recently used entries beyond `max_bytes`. Values larger than the bound are not stored. @ai-generated"""
        size = self.sizeof(value)
        if size > self.max_bytes:
            return
        with self._lock:
            if key in self._data:
                self.nbytes -= self._sizes.pop(key)
            self._data[key] = value
            self._data.move_to_end(key)
            self._sizes[key] = size
            self.nbytes += size
            while self.nbytes > self.max_bytes and self._data:
                old, _ = self._data.popitem(last=False)
                self.nbytes -= self._sizes.pop(old)

    def discard_if(self, predicate: Callable[[K], bool]):
        """@ai-generated"""
        with self._lock:
            for key in [k for k in self._data if predicate(k)]:
                del self._data[key]
                self.nbytes -= self._sizes.pop(key)

    def clear(self):
        with self._lock:
            self._data.clear()
            self._sizes.clear()
            self.nbytes = 0
