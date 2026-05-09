# Model Tasks

Inventory of every model the original `segviewer` runs, what it produces,
and the files / external dependencies it needs. Pulled from
`D:\git\sam_experiments\segviewer` on 2026-05-09. Use this as a reference
when redoing the local-model setup in this repo.

External-checkout paths are written relative to the **original**
`segviewer` directory (i.e. `..` is `sam_experiments`, `..\..` is
`D:\git`). When repointing them for this repo, redirect each path to its
new home.

---

## Setup-script plan

**Goal:** automate the local-model setup from a clean checkout on both
Windows (CUDA) and macOS (MPS). Split into one script per concern so we
can re-run / re-do individual pieces without touching the rest. All
scripts live in `setup/` at the repo root, written in **Python** for
cross-platform support — invoked as `python setup/<name>.py`.

### Conventions

Everything model-related lives under a single `models/` directory at the
repo root, gitignored as a whole:

- `models/.venv/` — the project venv. Interpreter resolved via
  `sys.platform`: `models/.venv/Scripts/python.exe` on Windows,
  `models/.venv/bin/python` on macOS/Linux.
- `models/external/<repo>/` — third-party model repos cloned at fixed
  commits.
- `models/weights/` — model checkpoints downloaded from HF hub or
  direct URL.
- `models/tools/` — native binaries (e.g. COLMAP) when not installed via
  a package manager.

`/models/` itself is added to `.gitignore` so the repo stays code-only.

Other rules:

- Every setup script must be **idempotent**: detect existing artifacts
  and skip; only re-do work when explicitly forced (`--force`).
- Each script accepts `--force` (re-do) and is safe to invoke directly
  with the **system** Python — it shells out to the venv interpreter
  for any `pip install` or HF download work.
- `vite.config.ts` and the Python scripts will need their hard-coded
  `..\..\<repo>` paths repointed to `models/external/<repo>/` (and the
  `models/weights/` and `models/tools/` subdirs). That repointing is
  part of the switch-over, not the setup scripts themselves.

### Scripts

| Script | What it does |
| --- | --- |
| `setup/00_venv.py` | Create `models/.venv/` (Python 3.11) using stdlib `venv` only — no third-party deps, since this is the bootstrap. Then install base deps shared by every plugin: torch (CUDA 12.4 wheel index on Windows, default index on macOS for MPS build), opencv-python, numpy, pillow, transformers, huggingface_hub, tqdm. Prerequisite for every subsequent script. |
| `setup/colmap.py` | Windows: download the COLMAP release zip, extract to `models/tools/colmap/`. macOS: shell out to `brew install colmap` (or instruct the user if Homebrew missing). Both: `pip install pycolmap` into the venv and verify the binary runs. |
| `setup/depthanythingv2.py` | Pre-cache `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf` via `huggingface_hub.snapshot_download` so first scene run isn't slow. |
| `setup/cut3r.py` | `git clone` CUT3R into `models/external/cut3r/` at a pinned commit; download `cut3r_512_dpt_4_64.pth` into `models/external/cut3r/src/`; `pip install -r` its requirements. |
| `setup/vggt.py` | `git clone` VGGT into `models/external/vggt/`; install its deps; pre-cache `facebook/VGGT-1B`. |
| `setup/da3.py` | `pip install depth_anything_3` (or git-install if not on PyPI); pre-cache `depth-anything/DA3-LARGE-1.1` and `depth-anything/DA3METRIC-LARGE`. |
| `setup/pi3.py` | Install the `pi3` package (likely from git); pre-cache `yyfz233/Pi3X`. |
| `setup/mapanything.py` | Install `mapanything` package; pre-cache `facebook/map-anything`. |
| `setup/worldmirror.py` | `git clone` `hunyuanworld-mirror` into `models/external/`; pre-cache `tencent/HunyuanWorld-Mirror`. (No `gsplat` / `flash_attn` install — the script stubs both.) |
| `setup/worldmirror2.py` | `git clone` `hy-world-2.0` into `models/external/`; pre-cache `tencent/HY-World-2.0`. (Same stubbing as worldmirror.) |
| `setup/wilddet3d.py` | `git clone` WildDet3D into `models/external/wilddet3d/` (with submodules `third_party/sam3` and `third_party/lingbot_depth`); download `wilddet3d_alldata_all_prompt_v1.0.pt` to `models/external/wilddet3d/ckpt/` and `sam3_detector.pt` to `models/external/wilddet3d/pretrained/sam3/`; install its requirements. |
| `setup/boxer.py` | `git clone` Boxer into `models/external/boxer/`; download `boxernet_hw960in4x6d768-wssxpf9p.ckpt` into `models/external/boxer/ckpts/`; install its requirements. |
| `setup/sam.py` | Download `sam2.1_l.pt` and `sam3.pt` into `models/weights/`. Used by `track_object.py` and `detect_object.py`. |
| `setup/all.py` | Convenience: run `00_venv.py` then every other script in order. Each child script's idempotency means re-running `all.py` is safe. |

### Shared helpers

A `setup/_lib.py` module imported by each script for things every script
needs:

- `venv_python()` — returns the path to `.venv`'s python, platform-aware.
- `assert_venv()` — fail fast with a clear message if `.venv` is missing.
- `run_in_venv(args)` — `subprocess.run` wrapper around the venv python.
- `pip_install(*pkgs, index_url=None)` — `pip install` via the venv.
- `download(url, dest, sha256=None)` — resumable download with optional
  hash check; skip if `dest` exists and hash matches.
- `clone_repo(url, dest, commit=None, recursive=False)` — git clone (or
  fetch + reset) at a pinned commit; no-op if already at the right rev.
- `hf_snapshot(repo_id, allow_patterns=None)` — wrapper around
  `huggingface_hub.snapshot_download` (run inside the venv since
  `huggingface_hub` is a venv dep).

### Platform notes

- **Windows**: CUDA 12.4 torch wheels via `--index-url
  https://download.pytorch.org/whl/cu124`. COLMAP from the official
  Windows release.
- **macOS**: default torch wheels (MPS support, no CUDA). COLMAP via
  `brew install colmap`. Several plugins (any that hardcode
  `torch.cuda.is_available()` or use bf16/flash-attention paths) will
  degrade or be unusable on MPS — flag them in each script's docstring
  rather than blocking installation.

### Open questions to resolve before writing the scripts

- COLMAP: pin to an exact release tag for reproducibility on Windows
  (likely the CUDA build, since the original script passes
  `--use_gpu 1`); on macOS accept whatever Homebrew gives.
- For each external repo, decide whether to track `main` or pin to a
  specific commit. Pinning is safer; `main` is easier.
- Torch version: the original used CUDA 12.4 wheels. Some plugins may
  need different torch versions — flag any conflicts as we hit them.
- Decide whether macOS support is best-effort (install scripts succeed
  but some plugins won't run) or whether scripts should refuse to
  install plugins known to be CUDA-only.

---

## Pipeline orchestration

`vite.config.ts` is the dispatcher: its dev-server middleware shells out
to the Python scripts under `scripts/` using the project venv (CUDA
torch) at `..\.venv\Scripts\python.exe`.

`src/scenePlugins.ts` is the registry consumed by both the backend
middleware and the Solid frontend. Adding a scene method = one entry
there + one script under `scripts/`.

## Per-scene output layout

Every video gets an `analysis/<video_stem>/_scene/` directory. Each
scene-reconstruction plugin writes into its own subdirectory under
`_scene/`:

```
_scene/
  frames/NNNNNN.jpg            (extract_frames.py)
  frames.json                  (extract_frames.py)
  <plugin>/cameras.json        (per-plugin pose + intrinsics)
  <plugin>/depth/NNNNNN.npz    (per-plugin per-frame depth)
  <plugin>/pointmap/NNNNNN.npz (per-plugin per-frame camera-space pointmap, optional)
  <plugin>/scene_pointmap.npz  (global world-space pointmap, optional)
  <plugin>.log / prepare.log   (stdout+stderr log)
```

Per-object outputs (segmentation, 3D boxes) go under
`analysis/<video_stem>/<object_name>/` instead of `_scene/`.

---

## Scene reconstruction plugins

Each entry below is one row of `SCENE_PLUGINS` in `src/scenePlugins.ts`.
Inputs are always `_scene/frames/*.jpg` + `_scene/frames.json` (produced
by `extract_frames.py`) unless noted.

### COLMAP + DepthAnythingV2 — `id: colmap`
- **Scripts**: `extract_frames.py` → `run_colmap.py` → `run_depth.py`
- **External**: `..\..\tools\colmap\COLMAP.bat` (native COLMAP build);
  Python deps `pycolmap`, `opencv`, `transformers`.
- **HF model**: `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf`.
- **Output**: `colmap/cameras.json` + `depthanythingv2/NNNNNN.npz` +
  `depthanythingv2/meta.json`.
- **Notes**: COLMAP solves shared intrinsics + per-frame poses on
  downscaled frames (`--max-size 1920`, `--every 3`). `run_depth.py`
  RANSAC-fits a per-frame affine between DA2 depth and COLMAP sparse
  observations to recover a global metric scale, then rescales
  `cameras.json` translations into meters.

### CUT3R — `id: cut3r`
- **Script**: `run_cut3r.py`
- **External**: `..\..\cut3r` checkout; weights at
  `..\..\cut3r\src\cut3r_512_dpt_4_64.pth`.
- **Output**: `cut3r/cameras.json`, `cut3r/depth/`, `cut3r/pointmap/`.
- **Notes**: Single feed-forward pass produces both poses and depth.
  Adds `pointmap` feature.

### VGGT — `id: vggt`
- **Script**: `run_vggt.py` (run with `--anchors-only`)
- **External**: `..\..\vggt` checkout. **HF model**: `facebook/VGGT-1B`.
- **Output**: `vggt/cameras.json`, `vggt/depth/`, `vggt/pointmap/`.
- **Notes**: Two-phase strategy — phase 1 is anchors-only (every 10th
  frame); phase 2 fills in the in-between spans via similarity
  alignment. The plugin runs phase 1 only.

### Depth-Anything-3 Metric (Large) — `id: da3`
- **Script**: `run_da3.py`
- **HF models**: `depth-anything/DA3-LARGE-1.1` (pose) +
  `depth-anything/DA3METRIC-LARGE` (metric depth).
- **Output**: `da3/cameras.json`, `da3/depth/`, `da3/pointmap/`.
- **Notes**: Two-model pipeline. Pose model gives relative depth +
  poses; metric model gives metric depth. A scene-wide ratio reconciles
  the two so translations land in meters.

### Pi3 — `id: pi3`
- **Script**: `run_pi3.py`
- **HF model**: `yyfz233/Pi3X`.
- **Output**: `pi3/cameras.json`, `pi3/depth/`, `pi3/pointmap/`,
  `pi3/scene_pointmap.npz`.
- **Notes**: Single feed-forward pass. Multimodal conditioning
  branches disabled (`disable_multimodal()`) to save VRAM. Publishes
  both per-frame pointmaps and a global scene pointmap.

### MapAnything — `id: mapanything`
- **Script**: `run_mapanything.py`
- **HF model**: `facebook/map-anything`.
- **Output**: `mapanything/cameras.json`, `mapanything/depth/`,
  `mapanything/pointmap/`, `mapanything/scene_pointmap.npz`.
- **Notes**: Memory-efficient inference with `minibatch_size=1` for
  12 GB GPUs. Edge-aware masking applied to the scene pointmap.

### HunyuanWorld-Mirror — `id: worldmirror`
- **Script**: `run_worldmirror.py`
- **External**: `..\..\hunyuanworld-mirror` checkout.
  **HF model**: `tencent/HunyuanWorld-Mirror`.
- **Output**: `worldmirror/cameras.json`, `worldmirror/depth/`,
  `worldmirror/pointmap/`, `worldmirror/scene_pointmap.npz`.
- **Notes**: Image-only mode (no priors). The gaussian-splat head is
  stubbed (`gsplat` import shimmed) so the script doesn't need a CUDA
  build of `gsplat`.

### HunyuanWorld-Mirror 2.0 — `id: worldmirror2`
- **Script**: `run_worldmirror2.py`
- **External**: `..\..\hy-world-2.0` checkout.
  **HF model**: `tencent/HY-World-2.0` (subfolder `HY-WorldMirror-2.0`).
- **Output**: `worldmirror2/cameras.json`, `worldmirror2/depth/`,
  `worldmirror2/pointmap/`, `worldmirror2/scene_pointmap.npz`.
- **Notes**: Same gsplat stub as v1; additionally stubs `flash_attn`
  with an SDPA shim so we don't need the flash-attention build.

### WildDet3D scene — `id: wilddet3d`
- **Script**: `run_wilddet3d_scene.py`
- **External**: `..\..\wilddet3d` checkout. Weights at
  `..\..\wilddet3d\ckpt\wilddet3d_alldata_all_prompt_v1.0.pt`.
- **Output**: `wilddet3d/cameras.json`, `wilddet3d/depth/`.
- **Notes**: Produces depth + predicted intrinsics per frame but **no
  cross-frame pose solve** — every camera pose is identity. Useful as
  a depth/K signal, not as a real reconstruction.

---

## Object pipeline (per-object, not per-scene)

Driven by the SAM2 click-to-track flow in the UI. All outputs land in
`analysis/<video_stem>/<object_name>/`.

### `detect_object.py` — SAM3 click-to-segment
- **HF model**: `facebook/sam3` → `..\..\weights\sam3.pt` (auto-downloaded).
- **Inputs**: image path, click x/y, label string.
- **Output**: `detect.json` (bbox + base64 RGBA mask) + `frame0_mask.png`.

### `track_object.py` — SAM2 video tracking
- **Weights**: `..\..\weights\sam2.1_l.pt` (also a stale copy at
  `segviewer/sam2.1_l.pt` from earlier). Loaded via Ultralytics
  `SAM2VideoPredictor`.
- **Inputs**: video path + `detect.json` (uses bbox, not mask, due to a
  shape bug in Ultralytics' mask-prompt path).
- **Output**: `track.json` + `masks/NNNNNN.png` per frame.

### `run_boxer.py` — Facebook Boxer 3D-box lifter
- **External**: `..\..\boxer` checkout. Weights at
  `..\..\boxer\ckpts\boxernet_hw960in4x6d768-wssxpf9p.ckpt`.
- **Inputs**: `<scene>/<source>/cameras.json`, `<scene>/<depth_dir>/`
  (optionally pointmaps for cut3r/vggt/da3/pi3/mapanything/worldmirror*),
  `<analysis>/track.json`.
- **Output**: `<analysis>/boxer.json` (per-frame OBBs + optional fused
  static instances).
- **Notes**: Rotates the world into Boxer's gravity convention
  (gravity = `[0,0,-1]`) before inference, rotates results back.

### `run_wilddet3d.py` — WildDet3D 3D-box lifter
- **External**: same `..\..\wilddet3d` checkout as the scene plugin;
  also needs `..\..\wilddet3d\pretrained\sam3\sam3_detector.pt`.
- **Inputs / outputs**: like `run_boxer.py`, writing
  `<analysis>/wilddet3d.json`.
- **Notes**: Runs on every ~10th frame and propagates to neighboring
  frames using the nearest preceding keyframe.

---

## Scene utilities

### `extract_frames.py`
Decode every video frame to JPEG q=92 under `_scene/frames/` and write
`_scene/frames.json` (fps, count, source size). No model deps.

### `align_scene.py`
Rotate cameras + scene pointmap so a user-picked floor plane lies on
`y = 0`, then yaw so frame 0 looks toward viewer +Z and translate frame
0 to the origin. Marks `cameras.json` with `gravity_aligned: true`.
Pure NumPy / depth-map unproject — no model deps.

---

## Summary of external dependencies (to redirect)

| What | Original path | Used by |
| --- | --- | --- |
| Project venv (CUDA torch) | `..\.venv` | all scripts |
| COLMAP binary | `..\..\tools\colmap\COLMAP.bat` | `run_colmap.py` |
| CUT3R repo + ckpt | `..\..\cut3r` (`src/cut3r_512_dpt_4_64.pth`) | `run_cut3r.py` |
| VGGT repo | `..\..\vggt` | `run_vggt.py` |
| HunyuanWorld-Mirror repo | `..\..\hunyuanworld-mirror` | `run_worldmirror.py` |
| HY-World-2.0 repo | `..\..\hy-world-2.0` | `run_worldmirror2.py` |
| WildDet3D repo + ckpt | `..\..\wilddet3d` (`ckpt/wilddet3d_alldata_all_prompt_v1.0.pt`, `pretrained/sam3/sam3_detector.pt`) | `run_wilddet3d.py`, `run_wilddet3d_scene.py` |
| Boxer repo + ckpt | `..\..\boxer` (`ckpts/boxernet_hw960in4x6d768-wssxpf9p.ckpt`) | `run_boxer.py` |
| SAM3 weights (auto-DL) | `..\..\weights\sam3.pt` | `detect_object.py` |
| SAM2 weights | `..\..\weights\sam2.1_l.pt` (also stale copy at `segviewer/sam2.1_l.pt`) | `track_object.py` |

HuggingFace-downloaded models (no manual checkout — pulled on first
run, cached in HF cache): `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf`,
`depth-anything/DA3-LARGE-1.1`, `depth-anything/DA3METRIC-LARGE`,
`yyfz233/Pi3X`, `facebook/VGGT-1B`, `facebook/map-anything`,
`tencent/HunyuanWorld-Mirror`, `tencent/HY-World-2.0`, `facebook/sam3`.
