"""Install InfiniDepth (CVPR 2026) as a depth refiner.

Run after 00_venv.py:

    python setup/plugin_infinidepth.py

This plugin consumes per-frame camera poses *and* per-frame depth from an
existing scene plugin (Pi3, DA3, VGGT, ...) and feeds them through
InfiniDepth's neural implicit field to produce a sharper / higher-res
depth map. MoGe-2 is no longer needed (we override the depth prior at
runtime), so its weights are not downloaded. The Gaussian-Splatting
branch (`inference_gs.py` + `gsplat`) is also skipped — we don't run it.

Weights end up under `models/weights/infinidepth/`:
  - `depth/infinidepth.ckpt`         (from ritianyu/InfiniDepth)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib  # noqa: E402

REPO_URL = "https://github.com/zju3dv/InfiniDepth.git"
COMMIT = "36c6e0c31887fafc210184ee43ca475230704095"

INFINIDEPTH_HF = "ritianyu/InfiniDepth"

# Skip packages whose CUDA-built / pinned-elsewhere versions we already control:
#   - torch/torchvision/numpy/pillow: pinned in 00_venv.py (CUDA wheels)
#   - xformers: upstream pins 0.0.33.post1 against torch 2.9; the venv has its
#     own torch and the wrong xformers will clobber it.
#   - gsplat: only needed by inference_gs.py (Gaussian Splatting), which we
#     don't run. Skip both the GH-install in INSTALL.md and any transitive ask.
#   - open3d: no wheel for our Python yet on some platforms; only viz/export
#     paths use it and inference does not (same skip rationale as DA3).
#   - moviepy: requirements.txt is unversioned; 2.x dropped moviepy.editor that
#     InfiniDepth imports. Pin to 1.0.3 explicitly below.
#   - spaces: HF Space SDK shim; only needed inside huggingface.co/spaces, not
#     for local CLI inference. Skip.
SKIP_DEPS = (
    "torch", "torchvision", "torchaudio",
    "numpy", "pillow",
    "xformers", "gsplat", "open3d", "moviepy", "spaces",
)
MOVIEPY_PIN = "moviepy==1.0.3"


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up InfiniDepth.")
    parser.add_argument("--force", action="store_true",
                        help="Re-clone, reinstall, and re-download weights")
    args = parser.parse_args()

    _lib.assert_venv()

    repo_dir = _lib.models_dir() / "external" / "infinidepth"
    _lib.clone_repo(REPO_URL, repo_dir, commit=COMMIT, force=args.force)

    req = repo_dir / "requirements.txt"
    if req.exists():
        print(f"[infinidepth] installing {req.name} (skipping {', '.join(SKIP_DEPS)})")
        _lib.install_requirements_filtered(req, skip_names=SKIP_DEPS)

    print(f"[infinidepth] installing {MOVIEPY_PIN} (InfiniDepth imports moviepy.editor, removed in 2.x)")
    _lib.pip_install(MOVIEPY_PIN)

    # NB: we do NOT install the `moge` python package nor download MoGe-2
    # weights. The runner always supplies override_gt_depth + intrinsics, so
    # `inference_utils.resolve_camera_intrinsics_for_inference` and
    # `prepare_metric_depth_inputs` never reach the lazy `from moge.model.v2
    # import MoGeModel` import inside `moge_utils._get_moge2_model`.

    weights_root = _lib.models_dir() / "weights" / "infinidepth"

    depth_dir = weights_root / "depth"
    depth_ckpt = depth_dir / "infinidepth.ckpt"
    if depth_ckpt.exists() and not args.force:
        print(f"[infinidepth] skip (already present): {depth_ckpt}")
    else:
        print(f"[infinidepth] downloading {INFINIDEPTH_HF} -> {depth_dir}")
        _lib.hf_snapshot(
            INFINIDEPTH_HF,
            allow_patterns=["infinidepth.ckpt"],
            local_dir=depth_dir,
        )

    print("[infinidepth] done")


if __name__ == "__main__":
    main()
