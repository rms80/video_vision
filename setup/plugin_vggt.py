"""Clone VGGT, install its Python deps, and pre-cache the VGGT-1B weights.

Run after 00_venv.py:

    python setup/plugin_vggt.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib  # noqa: E402

REPO_URL = "https://github.com/facebookresearch/vggt.git"
COMMIT = "44b3afbd1869d8bde4894dd8ea1e293112dd5eba"
HF_REPO = "facebook/VGGT-1B"


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up VGGT.")
    parser.add_argument("--force", action="store_true",
                        help="Re-clone and reinstall deps; HF cache is content-addressed")
    args = parser.parse_args()

    _lib.assert_venv()

    vggt_dir = _lib.models_dir() / "external" / "vggt"
    _lib.clone_repo(REPO_URL, vggt_dir, commit=COMMIT, force=args.force)

    req = vggt_dir / "requirements.txt"
    if req.exists():
        print(f"[vggt] installing {req.name} (skipping torch/torchvision/numpy pins)")
        _lib.install_requirements_filtered(req)
    else:
        print(f"[vggt] no requirements.txt at {req}; skipping pip install")

    print(f"[vggt] caching {HF_REPO} via HuggingFace hub")
    _lib.hf_snapshot(HF_REPO)

    print("[vggt] done")


if __name__ == "__main__":
    main()
