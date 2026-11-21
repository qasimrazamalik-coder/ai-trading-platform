"""Retries with backoff and a circuit breaker for data and broker calls.

Policy: transient failures (timeouts, 429, 5xx) are retried with capped exponential backoff and
jitter; repeated failures open the breaker so the loop stops hammering a broken service and,
more importantly, stops opening new positions until the service is healthy again.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum


class TransientError(Exception):
    """Raise for failures worth retrying (timeouts, rate limits, 5xx)."""


class BreakerOpenError(Exception):
    """The circuit is open; the call was not attempted."""


def retry[T](
    fn: Callable[[], T],
    *,
    attempts: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retry_on: tuple[type[BaseException], ...] = (TransientError, TimeoutError, ConnectionError),
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
) -> T:
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except retry_on:
            if attempt == attempts:
                raise
            delay = min(max_delay, base_delay * 2 ** (attempt - 1))
            sleep(delay * (0.5 + 0.5 * jitter()))
    raise AssertionError("unreachable")


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(slots=True)
class CircuitBreaker:
    name: str
    failure_threshold: int = 3
    reset_after: float = 300.0  # seconds before one trial call is allowed
    clock: Callable[[], float] = time.monotonic
    _failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)

    @property
    def state(self) -> BreakerState:
        if self._opened_at is None:
            return BreakerState.CLOSED
        if self.clock() - self._opened_at >= self.reset_after:
            return BreakerState.HALF_OPEN
        return BreakerState.OPEN

    @property
    def healthy(self) -> bool:
        return self.state is BreakerState.CLOSED

    def call[R](self, fn: Callable[[], R]) -> R:
        if self.state is BreakerState.OPEN:
            raise BreakerOpenError(f"{self.name} circuit is open")
        try:
            result = fn()
        except Exception:
            self._failures += 1
            if self._failures >= self.failure_threshold or self._opened_at is not None:
                self._opened_at = self.clock()  # (re)open; a failed trial re-opens at once
            raise
        self._failures = 0
        self._opened_at = None
        return result
