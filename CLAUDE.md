# video_vision

A browser-based workbench (Solid + Three on Vite) that turns a single
video into 3D: camera poses, per-frame depth, per-object 2D masks, 3D
boxes, and dense point clouds. The UI shells out to Python scripts in
`scripts/` for all heavy lifting. See `README.md` for the full feature
tour, pinned model commits, and per-plugin quirks.

## Layout
- `src/` — Solid + Three frontend. `App.tsx`, `scenePlugins.ts`,
  `boxSolverPlugins.ts` are the main wiring points.
- `scripts/` — per-pipeline Python runners (`run_<plugin>.py`,
  `detect_object.py`, `track_object.py`, `extract_frames.py`,
  `align_scene.py`, `build_object_pointmap.py`). `_device.py` is the
  shared CUDA/MPS/CPU picker.
- `setup/` — install scripts (see below).
- `models/` — gitignored: venv, cloned external repos under
  `external/`, weights under `weights/`, tools (COLMAP) under `tools/`.
- `analysis/<video>/` — per-video outputs (frames, per-plugin scene
  artifacts, per-object detect/track/box dirs).

## Python venv

The venv lives at **`models/.venv/`**, NOT at the project root. Invoke
the project's Python as `models/.venv/bin/python` (or
`models/.venv/Scripts/python.exe` on Windows). `_lib.venv_python()` and
`_lib.venv_dir()` in `setup/_lib.py` resolve this for setup scripts.

## Setup scripts

All setup files live in `setup/` and follow the same shape:

- `00_venv.py` — **always runs first**. Creates the venv (uses `uv
  venv --seed` on Linux when available, stdlib `venv` otherwise),
  installs torch + torchvision (CUDA wheels on Win/Linux, MPS on Mac),
  installs base deps.
- `_lib.py` — shared helpers: `venv_python()`, `pip_install()`,
  `download()`, `hf_snapshot()`, `apply_patch()`, `find_nvcc()`,
  `find_vcvars()`, `install_requirements_filtered()` (skips
  torch/torchvision/numpy/pillow so per-plugin requirements don't
  clobber the CUDA torch).
- `patches/` — unified-diff patches applied to cloned upstream repos
  via `apply_patch()`.
- One script per model/tool:
  - **Scene plugins**: `plugin_colmap.py`, `plugin_depthanythingv2.py`,
    `plugin_cut3r.py`, `plugin_vggt.py`, `plugin_da3.py`,
    `plugin_pi3.py`, `plugin_mapanything.py`, `plugin_worldmirror.py`,
    `plugin_worldmirror2.py`, `plugin_wilddet3d.py`.
  - **Box solvers**: `plugin_boxer.py`, `plugin_wilddet3d.py` (same
    script provides both the scene + box-solver plugin).
  - **Object seg**: `plugin_sam.py` (SAM2 + SAM3 weights).
- `EVERYTHING.py` — runs `00_venv.py` then every `plugin_*.py` in
  order. Forwards `--force` to each.

Each script is **idempotent** (skips already-installed artifacts) and
supports `--force` to wipe and reinstall. `colmap.py` branches by OS:
Windows downloads the prebuilt CUDA zip, macOS uses `brew`, Linux uses
`apt-get` and auto-fixes known Ubuntu 26.04 packaging gaps (missing
`libposelib`).

## Dev server

To start or restart the Vite dev server, run `bash run_server.sh`.
It frees port 4444 (kills any existing listener) and runs `npm run dev`.
Don't `npm run dev` directly — `strictPort: true` means a stale listener
on 4444 will fail the start.
