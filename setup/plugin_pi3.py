"""Install Pi3 from git and pre-cache its HF weights.

Run after 00_venv.py:

    python setup/plugin_pi3.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib  # noqa: E402

COMMIT = "b412c3bd236dfd7686f1e4b48004d5087f2fa093"
GIT_URL = f"git+https://github.com/yyfz/Pi3.git@{COMMIT}"
HF_REPO = "yyfz233/Pi3X"


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up Pi3.")
    parser.add_argument("--force", action="store_true",
                        help="Force pip reinstall; HF cache is content-addressed")
    parser.parse_args()

    _lib.assert_venv()

    print(f"[pi3] pip install {GIT_URL}")
    _lib.pip_install(GIT_URL)

    print(f"[pi3] caching {HF_REPO} via HuggingFace hub")
    _lib.hf_snapshot(HF_REPO)

    print("[pi3] done")


if __name__ == "__main__":
    main()
