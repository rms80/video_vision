"""Clone Facebook's Boxer and download its HF checkpoints.

Used by scripts/run_boxer.py — lifts SAM2-tracked 2D bboxes to 3D using
per-frame depth from the active scene plugin.

Notes:
    - We do not pip-install requirements.txt: Boxer ships only a
      pyproject (uv-based) and lists `torch>=2.0` + opencv + tqdm,
      which our base venv already has. The one Boxer-only dep we need
      is `dill` (used by its checkpoint loader).

Run after 00_venv.py:

    python setup/plugin_boxer.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib  # noqa: E402

REPO_URL = "https://github.com/facebookresearch/boxer.git"
COMMIT = "df474128a76ba42b05bc81feca7ac1a53fab41af"
HF_REPO = "facebook/boxer"
CKPTS = [
    "boxernet_hw960in4x6d768-wssxpf9p.ckpt",
    "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    "owlv2-base-patch16-ensemble.pt",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up Facebook's Boxer.")
    parser.add_argument("--force", action="store_true",
                        help="Re-clone the checkout; HF cache is content-addressed")
    args = parser.parse_args()

    _lib.assert_venv()

    repo_dir = _lib.models_dir() / "external" / "boxer"
    _lib.clone_repo(REPO_URL, repo_dir, commit=COMMIT, force=args.force)

    print("[boxer] installing dill (used by Boxer's checkpoint loader)")
    _lib.pip_install("dill")

    ckpts_dir = repo_dir / "ckpts"
    ckpts_dir.mkdir(parents=True, exist_ok=True)
    print(f"[boxer] caching {HF_REPO} weights into {ckpts_dir}")
    _lib.hf_snapshot(HF_REPO, allow_patterns=CKPTS, local_dir=ckpts_dir)

    print("[boxer] done")


if __name__ == "__main__":
    main()
