"""Run 00_venv.py then every plugin_*.py script in setup/.

Convenience wrapper for a from-scratch install. Each child script is
idempotent and skips already-installed artifacts.

Run with the system Python (>= 3.11):

    python setup/EVERYTHING.py

Pass --force to forward `--force` to every child (wipes and reinstalls):

    python setup/EVERYTHING.py --force

One plugin failing does not abort the rest; a summary is printed at the
end and the exit code is non-zero if any plugin failed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SETUP_DIR = Path(__file__).resolve().parent


def discover_plugins() -> list[Path]:
    return sorted(SETUP_DIR.glob("plugin_*.py"))


def run_script(script: Path, force: bool) -> int:
    cmd = [sys.executable, str(script)]
    if force:
        cmd.append("--force")
    print(f"\n{'=' * 72}\n[EVERYTHING] $ {' '.join(cmd)}\n{'=' * 72}")
    return subprocess.run(cmd).returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 00_venv.py then all plugin_*.py scripts in setup/.",
    )
    parser.add_argument("--force", action="store_true",
                        help="Forward --force to every child script")
    args = parser.parse_args()

    scripts = [SETUP_DIR / "00_venv.py", *discover_plugins()]
    missing = [s for s in scripts if not s.exists()]
    if missing:
        sys.exit(f"[EVERYTHING] missing scripts: {missing}")

    print(f"[EVERYTHING] will run {len(scripts)} script(s):")
    for s in scripts:
        print(f"  - {s.name}")

    failures: list[tuple[Path, int]] = []
    for s in scripts:
        rc = run_script(s, force=args.force)
        if rc != 0:
            failures.append((s, rc))
            print(f"[EVERYTHING] {s.name} exited {rc}; continuing")

    print(f"\n{'=' * 72}")
    if failures:
        print(f"[EVERYTHING] {len(failures)} of {len(scripts)} script(s) failed:")
        for s, rc in failures:
            print(f"  - {s.name} (exit {rc})")
        print(f"{'=' * 72}")
        sys.exit(1)
    print(f"[EVERYTHING] all {len(scripts)} script(s) succeeded")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
