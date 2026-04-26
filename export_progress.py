"""Single-line export progress UI (bar, rate, concurrency cap)."""

from __future__ import annotations

import sys
import time
from collections import deque
from typing import TextIO


class ExportProgressLine:
    """Format and write a live-updating ``\\r`` progress line to a text stream."""

    __slots__ = ("_bar_width", "_stream", "_times", "_last_done", "_start_time")

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        bar_width: int = 36,
        rate_window_sec: float = 60.0,
    ) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._bar_width = max(1, bar_width)
        self._times: deque[float] = deque()
        self._last_done = 0
        self._start_time: float | None = None

    def format_line(
        self,
        done: int,
        total: int,
        rates_per_min: tuple[float, float, float],
        concurrent_limit: int,
        concurrent_cap: int,
    ) -> str:
        cap = f"{concurrent_limit}/{concurrent_cap}"
        if total <= 0:
            return f"\rExport 0/0  cap {cap}"
        r1, r3, r10 = rates_per_min
        w = self._bar_width
        frac = done / total
        n = min(w, int(frac * w))
        bar = "#" * n + "-" * (w - n)
        return (
            f"\rExport [{bar}] {done}/{total}  "
            f"{r1:.1f} {r3:.1f} {r10:.1f}/min  cap {cap}"
        )

    def _update_rates(self, done: int) -> tuple[float, float, float]:
        now = time.monotonic()
        delta = max(0, done - self._last_done)
        if delta:
            if self._start_time is None:
                self._start_time = now
            for _ in range(delta):
                self._times.append(now)
            self._last_done = done
        # Keep at most 10-minute history.
        while self._times and self._times[0] < now - 600.0:
            self._times.popleft()
        c1 = sum(1 for t in self._times if t >= now - 60.0)
        c3 = sum(1 for t in self._times if t >= now - 180.0)
        c10 = len(self._times)
        if self._start_time is None:
            return (0.0, 0.0, 0.0)
        elapsed_min = max((now - self._start_time) / 60.0, 1e-9)
        # Divide each window count by min(elapsed_minutes, window_minutes).
        d1 = min(elapsed_min, 1.0)
        d3 = min(elapsed_min, 3.0)
        d10 = min(elapsed_min, 10.0)
        r1 = c1 / d1
        r3 = c3 / d3
        r10 = c10 / d10
        return (r1, r3, r10)

    def emit(
        self,
        done: int,
        total: int,
        recent_count: int,
        concurrent_limit: int,
        concurrent_cap: int,
    ) -> None:
        """Write one progress line and flush (for use as *on_each_done*)."""
        rates = self._update_rates(done)
        self._stream.write(
            self.format_line(
                done,
                total,
                rates,
                concurrent_limit,
                concurrent_cap,
            )
        )
        self._stream.flush()

    def end_line(self) -> None:
        """Clear the ``\\r`` line by ending with a newline."""
        self._stream.write("\n")
        self._stream.flush()
