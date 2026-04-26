"""Thread-pool export orchestration: adaptive concurrency, retries, and completion order.

Image / Earth Engine logic stays in plain callables; this module only schedules work,
limits overlapping calls, backs off on errors, and shrinks or grows the limit based
on recent concurrency-related failures.
"""

from __future__ import annotations

import threading
import time
import urllib.error
import sys
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


class RollingCompletionWindow:
    """Append-only completion times; ``record()`` returns count in the last *window_sec* seconds."""

    __slots__ = ("_times", "_window_sec")

    def __init__(self, window_sec: float) -> None:
        self._window_sec = window_sec
        self._times: deque[float] = deque()

    def record(self) -> int:
        now = time.monotonic()
        self._times.append(now)
        while self._times and self._times[0] < now - self._window_sec:
            self._times.popleft()
        return len(self._times)


def ee_concurrency_error(exc: BaseException) -> bool:
    """Heuristic: treat as Earth Engine / HTTP overload (shrink concurrency)."""
    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        if isinstance(cur, urllib.error.HTTPError) and cur.code in (429, 503):
            return True
        # Some exception types carry status codes as attrs.
        for attr in ("code", "status", "status_code", "http_status"):
            v = getattr(cur, attr, None)
            try:
                vi = int(v)
            except (TypeError, ValueError):
                vi = None
            if vi in (429, 503):
                return True
        msg = str(cur).lower()
        # EE often wraps HTTP status inside EEException text rather than raising HTTPError directly.
        if "429" in msg or "503" in msg:
            return True
        # Traverse wrapped causes/contexts.
        if isinstance(getattr(cur, "__cause__", None), BaseException):
            stack.append(cur.__cause__)  # type: ignore[arg-type]
        if isinstance(getattr(cur, "__context__", None), BaseException):
            stack.append(cur.__context__)  # type: ignore[arg-type]
    needles = (
        "concurrent",
        "too many",
        "too many requests",
        "rate limit",
        "quota",
        "throttl",
        "resource exhausted",
        "deadline exceeded",
        "unavailable",
    )
    return any(n in msg for n in needles)


class _ConcurrencyGate:
    """Limit how many wrapped jobs run at once; limit can change while tasks are in flight."""

    def __init__(self, low: int, high: int, start: int) -> None:
        self._low = max(1, low)
        self._high = max(self._low, high)
        start = max(self._low, min(self._high, start))
        self._limit = start
        self._active = 0
        self._cond = threading.Condition()

    @property
    def limit(self) -> int:
        with self._cond:
            return self._limit

    def acquire(self) -> None:
        with self._cond:
            while self._active >= self._limit:
                self._cond.wait()
            self._active += 1

    def release(self) -> None:
        with self._cond:
            self._active -= 1
            self._cond.notify_all()

    def shrink_on_overload(self, step: int) -> None:
        """Lower the limit by *step* (floored at *low*)."""
        with self._cond:
            new_limit = max(self._low, self._limit - max(1, step))
            if new_limit < self._limit:
                self._limit = new_limit
            self._cond.notify_all()

    def try_grow(self, step: int) -> None:
        with self._cond:
            if self._limit >= self._high:
                return
            self._limit = min(self._high, self._limit + step)
            self._cond.notify_all()


class AdaptiveThreadExportRunner:
    """Run ``fn(item)`` across a thread pool with a sliding concurrency cap and retries."""

    def __init__(
        self,
        *,
        min_concurrent: int = 1,
        max_concurrent: int = 20,
        initial_concurrent: int | None = None,
        quiet_before_scale_up_sec: float = 45.0,
        scale_down_step: int = 1,
        scale_down_min_interval_sec: float = 1.0,
        scale_up_step: int = 1,
        max_tries: int = 10,
        retry_base_delay_sec: float = 1.0,
        retry_backoff: float = 2.0,
        is_concurrency_error: Callable[[BaseException], bool] | None = None,
    ) -> None:
        self._low = min_concurrent
        self._high = max_concurrent
        start = initial_concurrent if initial_concurrent is not None else max_concurrent
        self._gate = _ConcurrencyGate(self._low, self._high, start)
        self._quiet_sec = quiet_before_scale_up_sec
        self._scale_down_step = max(1, scale_down_step)
        self._scale_down_min_interval_sec = max(0.0, scale_down_min_interval_sec)
        self._scale_up_step = scale_up_step
        self._max_tries = max(1, max_tries)
        self._retry_delay0 = retry_base_delay_sec
        self._retry_backoff = retry_backoff
        self._is_concurrency_error = is_concurrency_error or ee_concurrency_error
        # No scale-up until ``quiet_before_scale_up_sec`` after startup (and after any overload).
        self._last_concurrency_error_at = time.monotonic()
        self._last_scale_bump_at = 0.0
        self._last_scale_down_at: float | None = None
        self._policy_lock = threading.Lock()

    @property
    def concurrent_limit(self) -> int:
        """Current adaptive cap on parallel ``fn`` invocations."""
        return self._gate.limit

    @property
    def concurrent_cap(self) -> int:
        """Configured maximum (``max_concurrent`` passed to the constructor)."""
        return self._high

    def _on_concurrency_error(self) -> None:
        now = time.monotonic()
        did_shrink = False
        with self._policy_lock:
            self._last_concurrency_error_at = now
            if self._scale_down_min_interval_sec > 0:
                if (
                    self._last_scale_down_at is not None
                    and now - self._last_scale_down_at < self._scale_down_min_interval_sec
                ):
                    return
                self._last_scale_down_at = now
            before = self._gate.limit
            self._gate.shrink_on_overload(self._scale_down_step)
            after = self._gate.limit
            did_shrink = after < before
        if did_shrink:
            print(
                f"\n[adaptive] overload detected; cap {before}->{after}/{self._high}",
                file=sys.stderr,
                flush=True,
            )

    def _on_success(self) -> None:
        now = time.monotonic()
        with self._policy_lock:
            if now - self._last_concurrency_error_at < self._quiet_sec:
                return
            if now - self._last_scale_bump_at < self._quiet_sec:
                return
            self._last_scale_bump_at = now
        self._gate.try_grow(self._scale_up_step)

    def _invoke_with_retries(self, fn: Callable[[T], R], item: T) -> R:
        delay = self._retry_delay0
        for attempt in range(self._max_tries):
            try:
                out = fn(item)
                self._on_success()
                return out
            except BaseException as exc:
                if self._is_concurrency_error(exc):
                    self._on_concurrency_error()
                if attempt + 1 >= self._max_tries:
                    raise
                time.sleep(delay)
                delay *= self._retry_backoff

    def _wrapped(self, fn: Callable[[T], R], item: T) -> R:
        self._gate.acquire()
        try:
            return self._invoke_with_retries(fn, item)
        finally:
            self._gate.release()

    def run_unordered(
        self,
        fn: Callable[[T], R],
        items: Sequence[T],
        *,
        on_each_done: Callable[[int, int, int, int, int], None] | None = None,
        rate_window_sec: float | None = None,
        continue_on_errors: bool = False,
        on_task_error: Callable[[Exception], None] | None = None,
    ) -> Iterator[R]:
        """Yield results as tasks finish (order not preserved).

        *on_each_done(done, total, recent_count, concurrent_limit, concurrent_cap)* runs after
        each completion (and once at start with *done=0*, *recent_count=0*). *concurrent_limit*
        is the adaptive gate; *concurrent_cap* is the configured ceiling. If *rate_window_sec* is
        set, *recent_count* counts completions in that rolling window; otherwise it is always 0.

        If *continue_on_errors* is true, a failed task does not stop the run: *on_task_error* is
        called with the exception (after *on_each_done*), nothing is yielded for that task, and
        the next items keep running. ``KeyboardInterrupt`` and other :class:`BaseException` types
        other than :class:`Exception` are never swallowed.
        """

        total = len(items)
        if total == 0:
            return
        rate_tracker = (
            RollingCompletionWindow(rate_window_sec) if rate_window_sec is not None else None
        )
        if on_each_done is not None:
            on_each_done(0, total, 0, self.concurrent_limit, self.concurrent_cap)

        wrapped = lambda item: self._wrapped(fn, item)  # noqa: E731

        # Cap in-flight futures so we do not enqueue millions of Future objects at once.
        max_workers = self._high
        max_pending = min(total, max(max_workers * 4, max_workers + 16))
        item_iter = iter(items)
        done_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            pending: set = set()

            def submit_next() -> bool:
                try:
                    item = next(item_iter)
                except StopIteration:
                    return False
                pending.add(ex.submit(wrapped, item))
                return True

            for _ in range(min(max_pending, total)):
                submit_next()

            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in finished:
                    pending.discard(fut)
                    err: Exception | None = None
                    try:
                        result = fut.result()
                    except Exception as exc:
                        err = exc
                    done_count += 1
                    recent = rate_tracker.record() if rate_tracker is not None else 0
                    if on_each_done is not None:
                        on_each_done(
                            done_count,
                            total,
                            recent,
                            self.concurrent_limit,
                            self.concurrent_cap,
                        )
                    if err is not None:
                        if on_task_error is not None:
                            on_task_error(err)
                        if not continue_on_errors:
                            raise err
                    else:
                        yield result
                    submit_next()
