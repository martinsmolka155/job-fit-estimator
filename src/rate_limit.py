"""In-memory per-client rate limiting for the public HTTP API.

A single-process, in-memory sliding-window limiter. This is adequate for one
uvicorn worker behind a single nginx proxy. For a multi-worker or multi-host
deployment it would need a shared store (e.g. Redis); the constraint is
documented here so it is not mistaken for distributed enforcement.

The limiter is the public-facing complement to the daily budget guard in
src.cost_tracker: the budget caps *total* spend per day, while this caps how
fast any single client can consume it.
"""

from __future__ import annotations

import threading
import time
from collections import deque

# Above this many tracked keys, opportunistically drop keys whose newest hit
# has already aged out of the window, to bound memory under scattered traffic.
_PRUNE_THRESHOLD = 1024


class RateLimitExceeded(RuntimeError):
    """Raised when a client exceeds its request quota for the current window."""

    def __init__(self, retry_after_seconds: int) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"Rate limit exceeded; retry after {retry_after_seconds}s")


class SlidingWindowRateLimiter:
    """At most ``max_requests`` per ``window_seconds`` for each key (e.g. IP)."""

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        if max_requests < 1 or window_seconds < 1:
            raise ValueError("max_requests and window_seconds must both be >= 1")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, *, now: float | None = None) -> None:
        """Record a hit for ``key``; raise RateLimitExceeded if over quota.

        A blocked request is *not* recorded, so being rate-limited never pushes
        a caller's own retry window further into the future.
        """
        ts = time.monotonic() if now is None else now
        cutoff = ts - self.window_seconds
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                hits = deque()
                self._hits[key] = hits
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.max_requests:
                retry_after = int(hits[0] + self.window_seconds - ts) + 1
                raise RateLimitExceeded(max(retry_after, 1))
            hits.append(ts)
            self._maybe_prune(ts)

    def _maybe_prune(self, now: float) -> None:
        if len(self._hits) <= _PRUNE_THRESHOLD:
            return
        cutoff = now - self.window_seconds
        stale = [key for key, hits in self._hits.items() if not hits or hits[-1] <= cutoff]
        for key in stale:
            del self._hits[key]

    def reset(self) -> None:
        """Clear all recorded hits (used by tests for isolation)."""
        with self._lock:
            self._hits.clear()
