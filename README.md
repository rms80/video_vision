# video_vision

A browser-based workbench for experimenting with ML/Vision models for video processing. You can run various models on an uploaded video to:

* Generate camera pose + intrinsics + depth
* 2D object segmentation / tracking
* 3D bounding box 
* Scene & Object point clouds

Runs as a local backend server & frontend web app. The UI is a Solid + Three.js
app served by Vite; all heavy lifting is done by Python scripts that the
Vite dev server shells out to. A plugin architecture makes it easy to add additional models.

The repo bundles a Python `setup/` toolchain that installs every model
into a single, gitignored `models/` directory: one venv, one set of
pinned commits, one canonical location for weights.

All computed results are saved, and can be browsed from the frontend. 

This repo has been almost entirely created by Claude Code, under step-by-step human guidance. **DO NOT** deploy on the internet - code has not been hardened.

Questions / Comments? find me at rms@rms80.com, or [@rms80](http://twitter.com/rms80).



## Models Supported

See the [Model / method reference](#model--method-reference) section below for
pinned commits, custom patches, and per-plugin quirks.

### Scene Analysis (Cameras + Depth)

- **COLMAP + DepthAnythingV2** — classical SfM camera solve
  ([COLMAP 4.0.3](https://github.com/colmap/colmap)) paired with
  [DepthAnythingV2 Metric Indoor](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf)
  for per-frame depth, RANSAC-scaled to meters.
- **CUT3R** — [CUT3R/CUT3R](https://github.com/CUT3R/CUT3R), feed-forward
  poses + per-frame depth + pointmaps from a sliding context window.
- **VGGT** — [facebookresearch/vggt](https://github.com/facebookresearch/vggt),
  Meta's 1B transformer; anchors-only phase produces cameras + depth +
  pointmaps in one shot.
- **VGGT-Omega** *(gated)* —
  [facebookresearch/vggt-omega](https://github.com/facebookresearch/vggt-omega),
  CVPR 2026 successor to VGGT-1B. Requires HF access to
  [`facebook/VGGT-Omega`](https://huggingface.co/facebook/VGGT-Omega).
- **Depth-Anything-3 (Metric, Large)** —
  [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3),
  pairs the LARGE-1.1 (pose + relative depth) and DA3METRIC-LARGE
  (metric depth) heads for metric-scale reconstructions.
- **Pi3X** — [yyfz/Pi3](https://github.com/yyfz/Pi3), single feed-forward
  pass producing per-frame pointmaps + a global scene pointmap; fits in
  12 GB VRAM.
- **MapAnything** —
  [facebookresearch/map-anything](https://github.com/facebookresearch/map-anything),
  memory-efficient inference with edge-aware scene-pointmap masking.
- **HunyuanWorld-Mirror** —
  [Tencent-Hunyuan/HunyuanWorld-Mirror](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror),
  Tencent's multi-head model (we use the pointmap + depth + camera
  heads; gaussian-splat head stubbed out).
- **HunyuanWorld-Mirror 2.0** —
  [Tencent-Hunyuan/HY-World-2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0),
  v2 with a flash-attention-free SDPA shim.
- **WildDet3D (depth + K)** —
  [allenai/WildDet3D](https://github.com/allenai/WildDet3D), produces
  depth + predicted intrinsics per frame (no cross-frame pose solve —
  useful as a depth/K signal, not a real reconstruction).
- **InfiniDepth** *(depth refiner)* —
  [zju3dv/InfiniDepth](https://github.com/zju3dv/InfiniDepth). Not a
  standalone reconstruction — consumes another plugin's cameras +
  depth and sharpens the per-frame depth via a neural implicit field.

### Object Segmentation

- **SAM3 (detect)** *(gated)* —
  [`facebook/sam3`](https://huggingface.co/facebook/sam3) via
  Ultralytics. Click + text label → first-frame bbox & RGBA mask.
- **SAM2 (track)** —
  [`sam2.1_l.pt`](https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_l.pt)
  via Ultralytics' `SAM2VideoPredictor`. Propagates the SAM3 detection
  across all frames in the tracked range.

### 3D Box Generation

- **Boxer** *(default)* —
  [facebookresearch/boxer](https://github.com/facebookresearch/boxer),
  per-frame oriented bounding boxes from the masked depth/pointmap. The
  *Fuse* toggle merges all frames into one shared static box (more
  stable, slower).
- **WildDet3D** —
  [allenai/WildDet3D](https://github.com/allenai/WildDet3D), neural 3D
  detector with optional *Use Cameras* (intrinsics prior) and *Use
  Depth* toggles that pipe the active scene plugin's `K` / depth in as
  priors.

---


## Usage flow

The typical end-to-end run looks like this. Every step writes its outputs
to `analysis/<video_stem>/` and shows up in the right viewport without
needing a refresh.

1. **Upload a video.** Drag-drop or pick a previosly-uploaded video.
   The server re-encodes for smooth scrubbing 
   and pre-extracts every frame as a JPEG under
   `analysis/<video>/_scene/frames/`. 

2. **Scene analysis**. Pick a method and run. Produces camera poses + intrinsics + per-frame depth, and (for most plugins) per-frame and/or
   global pointmaps. **InfiniDepth** is also available as a depth
   *refiner*.

3. **World-up annotation** (optional). Click 3+
   points on horizontal surfaces (floor, table) across one or more
   frames, then *Align Scene*. This rotates the reconstruction so up is
   `+y`, yaws frame 0 to look down `+z`, and translates frame 0 to the
   origin. Stored per-video; reusable across analyses.

4. **Object segmentation.** Click any object in the source frame, and enter a text label. SAM3 is run to segment the first frame (writes `detect.json` (bbox + base64 RGBA mask) + a frame-0 PNG mask). Then SAM2 can be run to propagate tracking (writes `track.json` and per-frame `masks/NNNNNN.png`).  *Note: the click is used to disambiguate which segmentation result to track*

   Each detect-then-track pair lives in its own analysis folder `<label>_<N>` (e.g. `chair_1`), selectable from the UI. 

5. **Box solve** Lift the tracked 2D mask
   into a 3D oriented bounding box. Two solvers:
   - **Boxer** (default): per-frame OBB from masked depth/pointmap.
     The *Fuse* toggle merges all frames' point clouds into one shared
     static box (more stable, slower).
   - **WildDet3D**: neural 3D detector with optional camera-intrinsics-prior
     and depth-prior toggles.

   Output: `<analysis>/<solver>/boxes.json`. Solvers can co-exist on
   one analysis run (output dirs are keyed by solver id).

6. **Object point cloud** . Builds a dense
   object-only point cloud by unprojecting per-frame depth through the
   per-frame mask and concatenating across the tracked range. Streamed
   to the viewer as chunked `.npz` blobs.

---

## Right viewport: the five tabs

The right side of the UI is a tab strip + viewport. Tabs are
keyboard-navigable; arrow keys step frames within a tab (or jump to the nearest keyframe).

- **Source** — the raw video frame, overlaid with the current mask /
  bbox if an analysis is loaded. 

- **Depth** — the active scene plugin's per-frame depth map, colourised and aligned to the source frame's resolution.

- **3D (Per-Frame)** — Three.js viewport showing the current frame's
  depth lifted into a 3D mesh, plus camera path/frustums, solved boxes, etc

- **3D (Scene)** — global scene pointmap streamed in chunks (plus boxes / etc)

- **3D (Object)** — object-only point cloud built by unprojecting depth
  through the tracked masks. 

Status / log output appears in a fixed bar at the bottom of the
viewport and follows the latest pipeline run (scene prep, detect,
track, box, object cloud).

---

## Setup

### 1. Model Access (Hugging Face)

A few plugins pull weights from **gated** Hugging Face repos. Skip this
step if you don't need them — the rest of the setup still works. To request acess, open the links below and fill out the form (approved by Meta, often within minutes)

- [`facebook/sam3`](https://huggingface.co/facebook/sam3) — required
  for object detection / tracking.
- [`facebook/VGGT-Omega`](https://huggingface.co/facebook/VGGT-Omega) —
  required for the VGGT-Omega scene plugin.

While you are waiting, generate a token at <https://huggingface.co/settings/tokens> (*paste it into a text file until you have finished setup, as you only get to see it once on the website!*). Then either provide it in the text field in the GUI installer (see below) or run `hf auth login` (do `pip install -U huggingface_hub` to get the hf commands).

### 2. Windows + CUDA setup

**Prerequisites**

- **Python 3.11+** on `PATH`. Setup scripts use only the stdlib until
  the venv exists.
- **Node.js 18+** for the Vite dev server.
- **Git** (every external model is a pinned `git clone`).
- **NVIDIA GPU + CUDA 12.4 driver** for the included torch wheels.
- **Visual Studio 2019/2022 Build Tools** *(optional)* — only needed if
  you want CUT3R's `curope` CUDA extension to build. Skipping is fine;
  CUT3R falls back to a slower pure-Python RoPE.

**Install**

```
npm install
python setup/INSTALL.py        # GUI (recommended)
```

**Run the app**

```
run_server.bat
```

### 3. MacOS setup

**Prerequisites**

- **Python 3.11+** on `PATH`.
- **Node.js 18+**.
- **Git**.
- **Homebrew** — only used to install COLMAP.
- No CUDA: macOS uses the default MPS torch wheels.

**Install**

```
npm install
python setup/INSTALL.py        # GUI (recommended)
```

**Run the app**

```
bash run_server.sh
```

### 4. Linux + CUDA setup

**Prerequisites**

- **Python 3.11+** on `PATH`.
- **Node.js 18+**.
- **Git**.
- **NVIDIA GPU + CUDA 12.4 driver** for the included torch wheels.
- For the GUI installer: `python3-tk` on Debian/Ubuntu, or
  `python3-tkinter` on Fedora/RHEL. The headless installer doesn't need
  Tk.
- On Ubuntu 26.04, `plugin_colmap.py` auto-fixes a known `libposelib`
  packaging gap.

**Install**

```
npm install
python setup/INSTALL.py        # python GUI installer
```

**Run the app**

```
bash run_server.sh
```


### Install options

Installers run `00_venv.py` first and then the requested
`plugin_*.py` scripts. Every script is **idempotent** — it skips
already-present artifacts — and supports `--force` to wipe and
reinstall. Re-running either installer is safe.

**GUI (`setup/INSTALL.py`)** — a Tk window with one checkbox per setup step (venv, plugins). Optionally enter your HF auth token for gated models. Each script's output streams into the log pane.
The **--force** checkbox forwards `--force` to every selected script.

**Headless (`setup/EVERYTHING.py`)** — runs every plugin in order,
unconditionally. Use this when you can't run a GUI. Be aware this will consume approx 100GB of disk space.

**Per-component** — if you only need a subset, run scripts piecewise:

```
python setup/00_venv.py                    # always first
python setup/plugin_colmap.py              # if you want the COLMAP plugin
python setup/plugin_depthanythingv2.py     # paired with COLMAP
python setup/plugin_cut3r.py
python setup/plugin_vggt.py
python setup/plugin_vggtomega.py           # gated HF repo
python setup/plugin_da3.py
python setup/plugin_pi3.py
python setup/plugin_mapanything.py
python setup/plugin_worldmirror.py
python setup/plugin_worldmirror2.py
python setup/plugin_wilddet3d.py           # provides scene AND box solver
python setup/plugin_infinidepth.py         # depth refiner (post-process)
python setup/plugin_boxer.py
python setup/plugin_sam.py                 # required for detect/track (gated)
```

Everything except COLMAP (Windows zip / macOS Homebrew / Linux apt) and
the `models/.venv/` itself lands under `models/`, which is gitignored.

### Dev server notes

The Vite dev server listens on **port 4444**. `run_server.bat` /
`run_server.sh` kill any existing listener on that port before starting
— don't invoke `npm run dev` directly, since `strictPort: true` makes a
stale listener fatal. Change the port in the script if you need a
different one.


### Directory layout after install

```
models/
  .venv/                            project venv (CUDA torch on Win, MPS on Mac)
  external/
    boxer/
    cut3r/
    depth-anything-3/
    hunyuanworld-mirror/
    hy-world-2.0/
    infinidepth/
    vggt/
    vggt-omega/
    wilddet3d/                      cloned --recursive (sam3, lingbot_depth submodules)
  tools/
    colmap/                         Windows only; macOS uses Homebrew's binary
  weights/
    sam2.1_l.pt
    sam3.pt
    infinidepth/depth/infinidepth.ckpt
```

HuggingFace-distributed weights live in the standard HF cache
(`~/.cache/huggingface`), not under `models/`.

---

## Model / method reference

Pinned commits as of the last setup-script update. Every commit hash
below comes straight from `setup/<name>.py`.

### Scene plugins

#### COLMAP + DepthAnythingV2 — `id: colmap`
- **Native binary**: COLMAP **4.0.3**
  ([colmap-x64-windows-cuda.zip](https://github.com/colmap/colmap/releases/tag/4.0.3)
  on Windows; `brew install colmap` on macOS).
- **HF model**: `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf`.
- **Python**: `pycolmap` (latest pip wheel).
- **Notes**: COLMAP solves shared intrinsics + per-frame poses on
  downscaled frames (`--max-size 1920`, default `--every 3`).
  `run_depth.py` then RANSAC-fits a per-frame affine between DA2 depth
  and COLMAP sparse observations to recover a global metric scale and
  rescales `cameras.json` translations into meters.

#### CUT3R — `id: cut3r`
- **Repo**: [`CUT3R/CUT3R`](https://github.com/CUT3R/CUT3R) pinned to
  `8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf`.
- **Checkpoint**: `cut3r_512_dpt_4_64.pth` fetched from CUT3R's Google
  Drive via `gdown` (Drive throttling occasionally needs a manual
  download; the script prints the URL and bails gracefully if so).
- **Custom modifications applied by `setup/plugin_cut3r.py`**:
  - Filters CUT3R's `requirements.txt` to skip its pinned `torch /
    torchvision / numpy / pillow` (we keep the CUDA-built torch from
    the base venv).
  - Applies two local patches in `setup/patches/`:
    - `cut3r-fallback-rope-and-load.patch` — adds a Python-only RoPE2D
      fallback path so CUT3R imports even without the `curope` CUDA
      extension, and tolerates modern PyTorch's stricter `torch.load`.
    - `cut3r-curope-modern-pytorch.patch` — fixes the bundled `curope`
      CUDA extension's `setup.py` for current PyTorch / CUDA toolkit
      headers.
  - Builds the `curope` CUDA extension in-place if both `nvcc` and a
    Windows `vcvars64.bat` are reachable. If either is missing, the
    build is skipped and CUT3R falls back to the patched Python RoPE.

#### VGGT — `id: vggt`
- **Repo**: [`facebookresearch/vggt`](https://github.com/facebookresearch/vggt)
  pinned to `44b3afbd1869d8bde4894dd8ea1e293112dd5eba`.
- **HF model**: `facebook/VGGT-1B`.
- **Notes**: Two-phase strategy in the upstream code — phase 1 is
  anchors-only (every 10th frame); phase 2 fills in spans via similarity
  alignment. The plugin **runs phase 1 only** (`--anchors-only`). Repo
  `requirements.txt` is installed filtered (skip torch/torchvision/numpy/
  pillow).

#### VGGT-Omega — `id: vggtomega`
- **Repo**: [`facebookresearch/vggt-omega`](https://github.com/facebookresearch/vggt-omega)
  pinned to `39a0cb8af88554f15ddcb5354cd52bde588fa014`.
- **HF model**: [`facebook/VGGT-Omega`](https://huggingface.co/facebook/VGGT-Omega)
  — **gated** (request access, then `hf auth login`). The setup
  downloads only the non-text 512-resolution checkpoint
  (`vggt_omega_1b_512.pt`, ~4.58 GB); the 256-text variant is skipped
  since we only consume camera + depth here.
- **Notes**: Successor to VGGT-1B (CVPR 2026 Oral). Unlike VGGT-1B
  it ships plain `.pt` state dicts (no `from_pretrained`). Repo
  `requirements.txt` is installed filtered (skip torch/torchvision/
  numpy/pillow).

#### Depth-Anything-3 Metric (Large) — `id: da3`
- **Repo**: [`ByteDance-Seed/Depth-Anything-3`](https://github.com/ByteDance-Seed/Depth-Anything-3)
  pinned to `41736238f5bced4debf3f2a12375d2466874866d`.
- **HF models**: `depth-anything/DA3-LARGE-1.1` (pose + relative depth)
  and `depth-anything/DA3METRIC-LARGE` (metric depth). A scene-wide
  ratio reconciles the two so translations land in meters.
- **Custom modifications applied by `setup/plugin_da3.py`** (DA3's upstream
  requirements are not Python 3.13-friendly):
  - Skips `open3d` (no 3.13 wheel on PyPI; only viz paths use it),
    `xformers` (latest hard-requires torch >= 2.10 and would clobber
    our CUDA build), `moviepy` (DA3 imports `moviepy.editor`, removed
    in 2.x — explicitly pinned to `moviepy==1.0.3`), and
    `pre-commit` (dev tool only).
  - Installs `addict` manually — imported by `depth_anything_3.model.da3`
    but missing from upstream `pyproject` / `requirements.txt`.
  - DA3's `pyproject.toml` pins `requires-python = "<=3.13"`, which
    PEP 440 reads as excluding 3.13.x patch releases. We install with
    `--ignore-requires-python --no-deps`.

#### Pi3 (Pi3X Variant) — `id: pi3`
- **Source**: `pip install git+https://github.com/yyfz/Pi3.git@b412c3bd236dfd7686f1e4b48004d5087f2fa093`.
- **HF model**: `yyfz233/Pi3X`.
- **Notes**: Single feed-forward pass. The runner disables Pi3's
  multimodal conditioning branches (`disable_multimodal()`) to fit in
  12 GB VRAM. Publishes both per-frame pointmaps and a global scene
  pointmap.

#### MapAnything — `id: mapanything`
- **Source**: `pip install git+https://github.com/facebookresearch/map-anything.git@f7ebafb4d8349776705aaa686cf928988d1bd7f4`.
- **HF model**: `facebook/map-anything`.
- **Notes**: Memory-efficient inference path with `minibatch_size=1` for
  12 GB GPUs. Edge-aware masking applied to the scene pointmap.

#### HunyuanWorld-Mirror — `id: worldmirror`
- **Repo**: [`Tencent-Hunyuan/HunyuanWorld-Mirror`](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror)
  pinned to `b38bdd12e677f406788b1a56db5c3b4585f9ccd3`.
- **HF model**: `tencent/HunyuanWorld-Mirror`.
- **Custom modifications**:
  - The setup script does **not** install the upstream `requirements.txt`
    — it pins `torch==2.3.1` (would clobber our CUDA build) and depends
    on a CUDA build of `gsplat` that we do not ship.
  - `run_worldmirror.py` stubs `gsplat` at import time so the
    gaussian-splat head is unused. Pointmap + depth + cameras work
    without it.

#### HunyuanWorld-Mirror 2.0 — `id: worldmirror2`
- **Repo**: [`Tencent-Hunyuan/HY-World-2.0`](https://github.com/Tencent-Hunyuan/HY-World-2.0)
  pinned to `484e22020e7d7943eb199e31a00e10facf64c3d9`.
- **HF model**: `tencent/HY-World-2.0` (subfolder `HY-WorldMirror-2.0`).
- **Custom modifications**: same `gsplat` stub as v1, plus a
  `flash_attn` shim in `run_worldmirror2.py` that routes everything to
  PyTorch SDPA so the flash-attention build is unnecessary. Upstream
  `requirements.txt` is again skipped (Linux-only `gsplat` wheel +
  flash-attention).

#### WildDet3D (depth + K) — `id: wilddet3d`
- **Repo**: [`allenai/WildDet3D`](https://github.com/allenai/WildDet3D)
  pinned to `1768ffcd4c5e9bb1856d3f1a5b0b5e0498b89c97`, cloned
  recursively (submodules `third_party/sam3` and `third_party/lingbot_depth`).
- **Checkpoint**: `wilddet3d_alldata_all_prompt_v1.0.pt` from
  `allenai/WildDet3D` on HF.
- **Custom modifications applied by `setup/plugin_wilddet3d.py`** (this one's
  the worst — upstream requirements are heavily incompatible with
  Python 3.13):
  - `utils3d` in upstream `requirements.txt` resolves to the wrong PyPI
    package (Kalash Jain's `utils3d`, which has no `.pt` / `.np`
    submodules). WildDet3D's depth backend calls
    `utils3d.pt.depth_map_to_point_map`, which is from
    EasternJournalist's git-only `utils3d`. We filter the PyPI name out
    and install the git version pinned to commit
    `94d1037aabbce32dea9c07a7c4849525817a1615`.
  - `vis4d==1.0.0` is installed with `--no-deps`: its transitive deps
    (`bdd100k`, `scalabel`) pin `matplotlib==3.5.3` / `Shapely==1.8`,
    neither of which has a 3.13 wheel and both fail to build from
    sdist. The inference path doesn't touch any of that. We then
    install the actual runtime deps (`lightning`, `jsonargparse[signatures]`,
    `pydantic>=2.0`, `cloudpickle`, `devtools`, `h5py`) from WildDet3D's
    HF demo `requirements.txt`, which the upstream authors vetted as
    inference-only.
  - Submodule runtime deps installed explicitly: `ftfy`, `regex`,
    `iopath`, `open_clip_torch`, `safetensors`.
  - On Windows, installs `triton-windows` (registers itself as
    `triton`) since `sam3.model.edt` does a bare `import triton` and
    the Linux HF demo gets that for free with `torch`.
  - The runner builds the model with `skip_pretrained=True`, so SAM3 /
    LingBot pretrained weights are **not** needed — the WildDet3D
    checkpoint already contains them.
- **Notes**: Produces depth + predicted intrinsics per frame but **no
  cross-frame pose solve** (every camera pose is identity). Useful as
  a depth/K signal, not as a real reconstruction.

#### InfiniDepth (depth refiner) — `id: infinidepth`
- **Repo**: [`zju3dv/InfiniDepth`](https://github.com/zju3dv/InfiniDepth)
  pinned to `36c6e0c31887fafc210184ee43ca475230704095`.
- **HF model**: `ritianyu/InfiniDepth` →
  `models/weights/infinidepth/depth/infinidepth.ckpt`.
- **Notes**: Not a standalone reconstruction. The runner consumes an
  upstream plugin's `cameras.json` + per-frame depth and feeds them
  through InfiniDepth's neural implicit field to produce a sharper /
  higher-res depth map. Pick the upstream source in the UI when
  running the plugin.
- **Custom modifications applied by `setup/plugin_infinidepth.py`**:
  - Filters upstream `requirements.txt` to skip
    `torch/torchvision/torchaudio/numpy/pillow` (CUDA build in the
    base venv), `xformers` (pins torch 2.9 and would clobber it),
    `gsplat` (only used by the Gaussian-Splatting inference path,
    which we don't run), `open3d` (no 3.13 wheel; viz-only), and
    `spaces` (HF Space SDK shim).
  - Explicitly pins `moviepy==1.0.3` because InfiniDepth imports
    `moviepy.editor`, which 2.x dropped.
  - Skips MoGe-2 entirely: the runner always supplies
    `override_gt_depth` + intrinsics, so the lazy
    `from moge.model.v2 import MoGeModel` import inside
    `moge_utils._get_moge2_model` is never reached.

### Box solvers (per-object)

#### Boxer — `id: boxer`
- **Repo**: [`facebookresearch/boxer`](https://github.com/facebookresearch/boxer)
  pinned to `df474128a76ba42b05bc81feca7ac1a53fab41af`.
- **HF model**: `facebook/boxer` (we pull three checkpoints into
  `models/external/boxer/ckpts/`):
  - `boxernet_hw960in4x6d768-wssxpf9p.ckpt`
  - `dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth`
  - `owlv2-base-patch16-ensemble.pt`
- **Custom modifications**: skip Boxer's `pyproject` (uv-based, pins
  versions we already control). We install only `dill` (used by its
  checkpoint loader); torch + opencv + tqdm are already in the base
  venv.
- **Notes**: The runner rotates the world into Boxer's gravity
  convention (`gravity = [0, 0, -1]`) before inference and rotates
  results back. `--fuse` (UI: *Fuse* toggle) fuses all frames' masked
  pointclouds into one static box.

#### WildDet3D — `id: wilddet3d`
- Same `wilddet3d` checkout and checkpoint as the scene plugin.
- **Notes**: Runs on every ~10th frame and propagates to neighbouring
  frames using the nearest preceding keyframe. UI toggles *Use Cameras*
  (intrinsics prior) and *Use Depth* (depth prior) control whether the
  active scene plugin's `K` / depth are passed in.

### Object segmentation

#### SAM3 detect — `detect_object.py`
- **Source**: [`facebook/sam3`](https://huggingface.co/facebook/sam3)
  on HF → `models/weights/sam3.pt`. **Gated repo**: request access on
  the model page and `hf auth login` into the project venv before
  running `setup/plugin_sam.py`. See the [Setup](#setup) section.
- **Loaded via**: Ultralytics (`ultralytics>=8.4.37`).
- **Inputs**: frame image, click x/y, label.
- **Output**: `detect.json` (bbox + base64 RGBA mask) + `frame0_mask.png`.

#### SAM2 track — `track_object.py`
- **Source**: [`sam2.1_l.pt`](https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_l.pt)
  from the Ultralytics asset release →  `models/weights/sam2.1_l.pt`.
- **Loaded via**: `SAM2VideoPredictor` (Ultralytics).
- **Notes**: Uses the **bbox** from `detect.json`, not the mask, due to
  a shape bug in Ultralytics' mask-prompt path. Output is `track.json`
  + `masks/NNNNNN.png`.

---

## Per-video output layout

Every video gets an `analysis/<video_stem>/` directory:

```
analysis/<video>/
  _scene/
    frames/NNNNNN.jpg           (extract_frames.py)
    frames.json                 (fps, frame count, source size)
    <plugin>/cameras.json       per-plugin poses + intrinsics
    <plugin>/depth/NNNNNN.npz   per-plugin per-frame depth
    <plugin>/pointmap/NNNNNN.npz (optional, per-plugin)
    <plugin>/scene_pointmap.npz  (optional global pointmap)
    <plugin>.log / prepare.log
  <object_name>_<N>/            one per object analysis (e.g. chair_1)
    detect.json
    frame0_mask.png
    track.json
    masks/NNNNNN.png
    boxer/boxes.json            (optional)
    wilddet3d/boxes.json        (optional)
```

Adding a new scene method is a single entry in
`src/scenePlugins.ts` plus one runner script under `scripts/`. Adding
a new box solver is the same shape but in `src/boxSolverPlugins.ts`.

