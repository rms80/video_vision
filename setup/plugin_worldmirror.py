"""Clone HunyuanWorld-Mirror and pre-cache its HF weights.

Notes:
    - We do not pip-install requirements.txt: it pins torch==2.3.1 (would
      clobber our CUDA build) and depends on `gsplat` which we don't ship
      a CUDA build of. run_worldmirror.py stubs `gsplat` at import time
      so the gaussian-splat head is unused.

Run after 00_venv.py:

    python setup/plugin_worldmirror.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib  # noqa: E402

REPO_URL = "https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror.git"
COMMIT = "b38bdd12e677f406788b1a56db5c3b4585f9ccd3"
HF_REPO = "tencent/HunyuanWorld-Mirror"


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up HunyuanWorld-Mirror.")
    parser.add_argument("--force", action="store_true",
                        help="Re-clone the checkout; HF cache is content-addressed")
    args = parser.parse_args()

    _lib.assert_venv()

    repo_dir = _lib.models_dir() / "external" / "hunyuanworld-mirror"
    _lib.clone_repo(REPO_URL, repo_dir, commit=COMMIT, force=args.force)

    print(f"[worldmirror] caching {HF_REPO} via HuggingFace hub")
    _lib.hf_snapshot(HF_REPO)

    print("[worldmirror] done")


if __name__ == "__main__":
    main()
