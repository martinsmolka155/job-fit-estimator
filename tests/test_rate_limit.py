"""Unit tests for the in-memory sliding-window rate limiter."""

from __future__ import annotations

import pytest

from src.rate_limit import RateLimitExceeded, SlidingWindowRateLimiter


def test_allows_up_to_limit() -> None:
    limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
    for i in range(3):
        limiter.check("ip", now=1000.0 + i)  # must not raise


def test_blocks_over_limit_with_retry_after() -> None:
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
    limiter.check("ip", now=1000.0)
    limiter.check("ip", now=1001.0)
    with pytest.raises(RateLimitExceeded) as exc:
        limiter.check("ip", now=1002.0)
    # First hit was at 1000.0; window frees up at 1060.0 → ~58s from now.
    assert 1 <= exc.value.retry_after_seconds <= 60


def test_window_expiry_allows_again() -> None:
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
    limiter.check("ip", now=1000.0)
    with pytest.raises(RateLimitExceeded):
        limiter.check("ip", now=1030.0)
    limiter.check("ip", now=1061.0)  # original hit aged out → allowed


def test_keys_are_isolated() -> None:
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
    limiter.check("client-a", now=1000.0)
    limiter.check("client-b", now=1000.0)  # different key → allowed


def test_blocked_request_does_not_extend_window() -> None:
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
    limiter.check("ip", now=1000.0)
    for t in (1010.0, 1020.0):  # repeated blocked attempts
        with pytest.raises(RateLimitExceeded):
            limiter.check("ip", now=t)
    limiter.check("ip", now=1061.0)  # still frees relative to the 1000.0 hit


def test_reset_clears_state() -> None:
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
    limiter.check("ip", now=1000.0)
    limiter.reset()
    limiter.check("ip", now=1000.0)  # allowed again after reset


@pytest.mark.parametrize(("max_requests", "window"), [(0, 60), (1, 0), (-1, -1)])
def test_invalid_config_rejected(max_requests: int, window: int) -> None:
    with pytest.raises(ValueError):
        SlidingWindowRateLimiter(max_requests=max_requests, window_seconds=window)
