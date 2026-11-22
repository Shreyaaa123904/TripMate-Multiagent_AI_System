"""Request-level security helpers: rate limiting and thread-id handling."""

import hashlib
import re
import secrets
import threading
import time
from collections import deque

THREAD_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
_THREAD_ID_RE = re.compile(THREAD_ID_PATTERN)


def is_valid_thread_id(value: str) -> bool:
    return bool(_THREAD_ID_RE.fullmatch(value))


def new_thread_id() -> str:
    """Unguessable thread id (128 random bits), same shape as before."""
    return f"user_{secrets.token_hex(16)}"


def thread_ref(thread_id: str | None) -> str:
    """Short, non-reversible reference for logs. Never log full thread ids."""
    if not thread_id:
        return "-"
    return hashlib.sha256(thread_id.encode()).hexdigest()[:12]


class RateLimiter:
    """Thread-safe in-memory sliding-window limiter (per process)."""

    def __init__(self, limit: int, window_seconds: float = 60.0, clock=time.monotonic):
        self.limit = limit
        self.window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> tuple[bool, int]:
        """Record a request. Returns (allowed, retry_after_seconds)."""
        if self.limit <= 0:
            return True, 0
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                retry_after = max(1, int(hits[0] + self.window - now + 0.999))
                return False, retry_after
            hits.append(now)
            if len(self._hits) > 10_000:
                self._prune(now)
            return True, 0

    def _prune(self, now: float) -> None:
        for key in [k for k, v in self._hits.items() if not v or v[-1] <= now - self.window]:
            del self._hits[key]

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
