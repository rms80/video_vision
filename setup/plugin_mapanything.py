"""Install MapAnything from git and pre-cache its HF weights.

Run after 00_venv.py:

    python setup/plugin_mapanything.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib  # noqa: E402

COMMIT = "f7ebafb4d8349776705aaa686cf928988d1bd7f4"
GIT_URL = f"git+https://github.com/facebookresearch/map-anything.git@{COMMIT}"
HF_REPO = "facebook/map-anything"


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up MapAnything.")
    parser.add_argument("--force", action="store_true",
                        help="Force pip reinstall; HF cache is content-addressed")
    parser.parse_args()

    _lib.assert_venv()

    print(f"[mapanything] pip install {GIT_URL}")
    _lib.pip_install(GIT_URL)

    print(f"[mapanything] caching {HF_REPO} via HuggingFace hub")
    _lib.hf_snapshot(HF_REPO)

    print("[mapanything] done")


if __name__ == "__main__":
    main()
