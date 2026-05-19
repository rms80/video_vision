"""Clone HY-World-2.0 (HunyuanWorld-Mirror v2) and pre-cache its HF weights.

Notes:
    - We do not pip-install requirements.txt: it pulls a Linux-only gsplat
      wheel and uses flash-attention paths. run_worldmirror2.py stubs
      both `gsplat` and `flash_attn` at import time.
    - The HF repo also ships HY-Pano-2.0 (panorama generation, ~170 GB)
      which the runner doesn't use. We restrict the snapshot to the
      ~5 GB `HY-WorldMirror-2.0/` subfolder.

Run after 00_venv.py:

    python setup/plugin_worldmirror2.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib  # noqa: E402

REPO_URL = "https://github.com/Tencent-Hunyuan/HY-World-2.0.git"
COMMIT = "ee5d5bc02c92671486ab6dd81f0cba577d0478c8"
HF_REPO = "tencent/HY-World-2.0"


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up HY-World-2.0.")
    parser.add_argument("--force", action="store_true",
                        help="Re-clone the checkout; HF cache is content-addressed")
    args = parser.parse_args()

    _lib.assert_venv()

    repo_dir = _lib.models_dir() / "external" / "hy-world-2.0"
    _lib.clone_repo(REPO_URL, repo_dir, commit=COMMIT, force=args.force)

    print(f"[worldmirror2] caching {HF_REPO} (HY-WorldMirror-2.0/ subfolder only) via HuggingFace hub")
    _lib.hf_snapshot(HF_REPO, allow_patterns=["HY-WorldMirror-2.0/*"])

    print("[worldmirror2] done")


if __name__ == "__main__":
    main()
