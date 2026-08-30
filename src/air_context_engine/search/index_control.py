"""Cooperative cancellation, deadlines, and progress for repository indexing."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable


class IndexCancelled(RuntimeError):
    """Raised when a caller cancels an in-flight index operation."""


class IndexTimeout(TimeoutError):
    """Raised when an index operation exceeds its configured deadline."""


@dataclass(frozen=True)
class IndexProgress:
    phase: str
    completed: int
    total: int | None
    message: str


ProgressCallback = Callable[[IndexProgress], None]


class IndexControl:
    """Thread-safe cooperative stop signal with an optional monotonic deadline."""

    def __init__(self, timeout_seconds: float | None = None) -> None:
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds pozitif olmalıdır")
        self.started_at = time.monotonic()
        self.deadline = (
            self.started_at + timeout_seconds if timeout_seconds is not None else None
        )
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def remaining_seconds(self) -> float | None:
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time.monotonic())

    def checkpoint(self) -> None:
        if self.cancelled:
            raise IndexCancelled("İndeksleme kullanıcı tarafından iptal edildi.")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise IndexTimeout("İndeksleme zaman aşımına uğradı.")
