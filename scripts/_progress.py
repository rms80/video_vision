"""Stream short status updates from a Python runner back to the dev server.

Lines printed via `progress("...")` are tagged with a magic prefix; the
dev server (vite.config.ts -> runPython) tails each subprocess's stdout,
detects those tagged lines, and surfaces the most recent one on the
running job's state (visible in the UI status bar as the script runs).
Untagged stdout/stderr still goes to the per-stage log file.

Wallclock timing is owned by the server (it knows when the whole
pipeline started + finished), so scripts don't need to print elapsed
themselves — but `step()` is provided for finer in-script timing if
useful.
"""
from __future__ import annotations

import sys
import time

PROGRESS_PREFIX = "[progress] "

# Start the clock when this module is first imported. Runner scripts
# import _progress near the top, so this is effectively script-startup.
_START = time.monotonic()


def _stamp() -> str:
    """`[mm:ss]` since module import, rolling to `[hh:mm:ss]` past 1h."""
    elapsed = int(time.monotonic() - _START)
    m, s = divmod(elapsed, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"[{h:02d}:{m:02d}:{s:02d}]"
    return f"[{m:02d}:{s:02d}]"


def progress(msg: str) -> None:
    """Emit a status line that the dev server surfaces to the UI."""
    print(f"{PROGRESS_PREFIX}{_stamp()} {msg}", flush=True)


class step:
    """Context manager that emits a `label` line on enter and a
    `label — Xs` line on exit (using the same progress channel)."""

    def __init__(self, label: str):
        self.label = label
        self._t0 = 0.0

    def __enter__(self) -> "step":
        progress(self.label)
        self._t0 = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed = time.monotonic() - self._t0
        # On exception, don't claim success — server will surface the error.
        if exc_type is None:
            progress(f"{self.label} — {format_elapsed(elapsed)}")


def format_elapsed(seconds: float) -> str:
    """Compact human-readable duration: `0.4s`, `12s`, `1m 23s`, `1h 4m`."""
    if seconds < 1:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    h, rem = divmod(int(seconds), 3600)
    m = rem // 60
    return f"{h}h {m}m"


# Make sure stdout is line-buffered so progress lines surface immediately
# even when not flushed explicitly. (Each progress() already flushes; this
# is belt-and-suspenders for any plain `print` calls in the same script.)
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
