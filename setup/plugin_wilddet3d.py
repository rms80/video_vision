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
      skips *loading* the SAM3 / LingBot pretrained weights (the WildDet3D
      checkpoint already contains them). But LingBot's config is read
      from `model.pt` of `robbyant/lingbot-depth-postrain-dc-vitl14`, so
      that file still has to be cached locally — otherwise inference.py
      hits `hf_hub_download` at runtime, which crashes the runner once
      the dev server sets `HF_HUB_OFFLINE=1`. We pre-cache it here.
    - Pinned torch / torchvision / numpy / pillow are filtered out of
      the requirements install so we keep the venv's CUDA torch build.
    - `utils3d` in requirements.txt resolves to the wrong PyPI package
      (Kalash Jain's, no `.pt`/`.np` submodules). WildDet3D's depth
      backend calls `utils3d.pt.depth_map_to_point_map`, which is from
      EasternJournalist's `utils3d` — git-only. We filter the PyPI name
      out and install the git version separately.
    - `vis4d==1.0.0` pulls in `bdd100k` and `scalabel`, which transitively
      pin matplotlib 3.5.3 / Shapely 1.8 — neither has a Python 3.13
      wheel and both fail to build from sdist. The wilddet3d inference
      path doesn't touch any of those. We install vis4d with --no-deps
      and add only the runtime libs from WildDet3D's HF demo
      requirements (vetted by allenai for inference-only).

Run after 00_venv.py:

    python setup/plugin_wilddet3d.py
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib  # noqa: E402

REPO_URL = "https://github.com/allenai/WildDet3D.git"
COMMIT = "1768ffcd4c5e9bb1856d3f1a5b0b5e0498b89c97"
HF_REPO = "allenai/WildDet3D"
CKPT_NAME = "wilddet3d_alldata_all_prompt_v1.0.pt"

# LingBot-Depth backbone: even with skip_pretrained=True the runner reads
# `model_config` from this file's checkpoint, so it must be in the HF cache
# for the offline-mode runtime to find it.
LINGBOT_HF_REPO = "robbyant/lingbot-depth-postrain-dc-vitl14"
LINGBOT_FILE = "model.pt"

UTILS3D_URL = "git+https://github.com/EasternJournalist/utils3d.git@94d1037aabbce32dea9c07a7c4849525817a1615"

# vis4d runtime deps (from WildDet3D's demo/huggingface/requirements.txt)
VIS4D_RUNTIME_DEPS = (
    "lightning",
    "jsonargparse[signatures]",
    "pydantic>=2.0",
    "cloudpickle",
    "devtools",
    "h5py",
)

# Submodule runtime deps (sam3 imports ftfy/regex/iopath/open_clip_torch;
# lingbot_depth needs safetensors). Same set as the HF demo.
SUBMODULE_DEPS = (
    "ftfy",
    "regex",
    "iopath",
    "open_clip_torch",
    "safetensors",
)

# sam3.model.edt does `import triton`. The HF demo's Linux base ships triton
# with torch; on Windows we need triton-windows (registers itself as `triton`).
TRITON_WINDOWS = "triton-windows"


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
        print(f"[wilddet3d] installing {req.name} (skipping torch/torchvision/numpy/pillow/utils3d/vis4d)")
        _lib.install_requirements_filtered(
            req,
            skip_names=("torch", "torchvision", "numpy", "pillow", "utils3d", "vis4d"),
        )
    else:
        print(f"[wilddet3d] no requirements.txt at {req}; skipping pip install")

    print("[wilddet3d] installing vis4d==1.0.0 with --no-deps (avoids broken bdd100k/scalabel chain)")
    _lib.run_in_venv(["-m", "pip", "install", "--no-deps", "vis4d==1.0.0"])

    print(f"[wilddet3d] installing vis4d runtime deps: {', '.join(VIS4D_RUNTIME_DEPS)}")
    _lib.pip_install(*VIS4D_RUNTIME_DEPS)

    print(f"[wilddet3d] installing sam3/lingbot_depth deps: {', '.join(SUBMODULE_DEPS)}")
    _lib.pip_install(*SUBMODULE_DEPS)

    if platform.system() == "Windows":
        print(f"[wilddet3d] installing {TRITON_WINDOWS} (sam3 imports triton)")
        _lib.pip_install(TRITON_WINDOWS)

    print(f"[wilddet3d] installing utils3d from {UTILS3D_URL}")
    _lib.pip_install(UTILS3D_URL)

    ckpt_dir = repo_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[wilddet3d] caching {CKPT_NAME} from {HF_REPO} into {ckpt_dir}")
    _lib.hf_snapshot(HF_REPO, allow_patterns=[CKPT_NAME], local_dir=ckpt_dir)

    # Populate the hub cache (no local_dir) so the runtime
    # `hf_hub_download(repo_id=LINGBOT_HF_REPO, filename=LINGBOT_FILE)` call
    # resolves from cache under HF_HUB_OFFLINE=1.
    print(f"[wilddet3d] caching {LINGBOT_FILE} from {LINGBOT_HF_REPO} into HF cache")
    _lib.hf_snapshot(LINGBOT_HF_REPO, allow_patterns=[LINGBOT_FILE])

    print("[wilddet3d] done")


if __name__ == "__main__":
    main()
