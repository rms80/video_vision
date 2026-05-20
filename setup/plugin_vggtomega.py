"""Clone VGGT-Omega, install its Python deps, and pre-cache the 512 checkpoint.

VGGT-Omega is a successor to VGGT-1B (CVPR 2026 Oral). Unlike VGGT-1B it
ships plain `.pt` state dicts on a gated HuggingFace repo — the model
class itself has no `from_pretrained`. The setup downloads only the
non-text 512-resolution checkpoint (4.58 GB); the 256-text variant is
skipped because we only consume camera + depth here.

Requires HuggingFace access to facebook/VGGT-Omega (the user must have
accepted the gate and be logged in via `huggingface-cli login`).

Run after 00_venv.py:

    python setup/plugin_vggtomega.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib  # noqa: E402

REPO_URL = "https://github.com/facebookresearch/vggt-omega.git"
COMMIT = "39a0cb8af88554f15ddcb5354cd52bde588fa014"
HF_REPO = "facebook/VGGT-Omega"
CKPT_FILE = "vggt_omega_1b_512.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up VGGT-Omega.")
    parser.add_argument("--force", action="store_true",
                        help="Re-clone and reinstall deps; HF cache is content-addressed")
    args = parser.parse_args()

    _lib.assert_venv()

    repo_dir = _lib.models_dir() / "external" / "vggt-omega"
    _lib.clone_repo(REPO_URL, repo_dir, commit=COMMIT, force=args.force)

    req = repo_dir / "requirements.txt"
    if req.exists():
        print(f"[vggtomega] installing {req.name} (skipping torch/torchvision/numpy/pillow pins)")
        _lib.install_requirements_filtered(req)
    else:
        print(f"[vggtomega] no requirements.txt at {req}; skipping pip install")

    print(f"[vggtomega] caching {CKPT_FILE} from {HF_REPO}")
    _lib.hf_snapshot(HF_REPO, allow_patterns=[CKPT_FILE])

    print("[vggtomega] done")


if __name__ == "__main__":
    main()
