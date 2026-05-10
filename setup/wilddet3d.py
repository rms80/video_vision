"""Clone allenai/WildDet3D (with submodules), install its deps, and
download the WildDet3D checkpoint.

Used by scripts/run_wilddet3d.py and scripts/run_wilddet3d_scene.py —
neural 3D detector that lifts SAM2-tracked 2D bboxes to 3D in one pass,
optionally consuming camera intrinsics and/or depth.

Notes:
    - Cloned recursively: WildDet3D pulls in `third_party/sam3` and
      `third_party/lingbot_depth` as submodules. The runner adds those
      paths to sys.path so the `sam3` and `mdm` packages import without
      pip-install.
    - The runner builds the model with `skip_pretrained=True`, which
      means SAM3 / LingBot pretrained weights are *not* needed — the
      WildDet3D checkpoint already contains them. So we only fetch
      `wilddet3d_alldata_all_prompt_v1.0.pt`.
    - Pinned torch / torchvision / numpy / pillow are filtered out of
      the requirements install so we keep the venv's CUDA torch build.

Run after 00_venv.py:

    python setup/wilddet3d.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib  # noqa: E402

REPO_URL = "https://github.com/allenai/WildDet3D.git"
COMMIT = "1768ffcd4c5e9bb1856d3f1a5b0b5e0498b89c97"
HF_REPO = "allenai/WildDet3D"
CKPT_NAME = "wilddet3d_alldata_all_prompt_v1.0.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up WildDet3D.")
    parser.add_argument("--force", action="store_true",
                        help="Re-clone the checkout; HF cache is content-addressed")
    args = parser.parse_args()

    _lib.assert_venv()

    repo_dir = _lib.models_dir() / "external" / "wilddet3d"
    _lib.clone_repo(REPO_URL, repo_dir, commit=COMMIT,
                    recursive=True, force=args.force)

    req = repo_dir / "requirements.txt"
    if req.exists():
        print(f"[wilddet3d] installing {req.name} (skipping torch/torchvision/numpy/pillow pins)")
        _lib.install_requirements_filtered(req)
    else:
        print(f"[wilddet3d] no requirements.txt at {req}; skipping pip install")

    ckpt_dir = repo_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[wilddet3d] caching {CKPT_NAME} from {HF_REPO} into {ckpt_dir}")
    _lib.hf_snapshot(HF_REPO, allow_patterns=[CKPT_NAME], local_dir=ckpt_dir)

    print("[wilddet3d] done")


if __name__ == "__main__":
    main()
