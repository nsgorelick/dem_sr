"""Single-line export progress UI (bar, rate, concurrency cap)."""

from __future__ import annotations

import sys
from typing import TextIO


class ExportProgressLine:
    """Format and write a live-updating ``\\r`` progress line to a text stream."""

    __slots__ = ("_bar_width", "_rate_label", "_stream")

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        bar_width: int = 36,
        rate_window_sec: float = 60.0,
    ) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._bar_width = max(1, bar_width)
        rw = float(rate_window_sec)
        w = int(rw) if rw.is_integer() else rw
        self._rate_label = f"last {w}s"

    def format_line(
        self,
        done: int,
        total: int,
        recent_count: int,
        concurrent_limit: int,
        concurrent_cap: int,
    ) -> str:
        cap = f"{concurrent_limit}/{concurrent_cap}"
        if total <= 0:
            return f"\rExport 0/0  cap {cap}"
        w = self._bar_width
        frac = done / total
        n = min(w, int(frac * w))
        bar = "#" * n + "-" * (w - n)
        return (
            f"\rExport [{bar}] {done}/{total}  "
            f"{recent_count}/min ({self._rate_label})  cap {cap}"
        )

    def emit(
        self,
        done: int,
        total: int,
        recent_count: int,
        concurrent_limit: int,
        concurrent_cap: int,
    ) -> None:
        """Write one progress line and flush (for use as *on_each_done*)."""
        self._stream.write(
            self.format_line(
                done,
                total,
                recent_count,
                concurrent_limit,
                concurrent_cap,
            )
        )
        self._stream.flush()

    def end_line(self) -> None:
        """Clear the ``\\r`` line by ending with a newline."""
        self._stream.write("\n")
        self._stream.flush()
