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
  installs base deps (including `huggingface_hub`, so `hf` is available
  in the venv).
- `_lib.py` — shared helpers: `venv_python()`, `pip_install()`,
  `download()`, `hf_snapshot()`, `apply_patch()`, `find_nvcc()`,
  `find_vcvars()`, `install_requirements_filtered()` (skips
  torch/torchvision/numpy/pillow so per-plugin requirements don't
  clobber the CUDA torch).
- `patches/` — unified-diff patches applied to cloned upstream repos
  via `apply_patch()`.
- One script per model/tool:
  - **Scene plugins**: `plugin_colmap.py`, `plugin_depthanythingv2.py`,
    `plugin_cut3r.py`, `plugin_vggt.py`, `plugin_vggtomega.py` *(gated
    HF)*, `plugin_da3.py`, `plugin_pi3.py`, `plugin_mapanything.py`,
    `plugin_worldmirror.py`, `plugin_worldmirror2.py`,
    `plugin_wilddet3d.py`, `plugin_infinidepth.py` *(depth refiner —
    consumes another plugin's poses + depth, not a standalone
    reconstruction)*.
  - **Box solvers**: `plugin_boxer.py`, `plugin_wilddet3d.py` (same
    script provides both the scene + box-solver plugin).
  - **Object seg**: `plugin_sam.py` *(gated HF — `facebook/sam3`)*
    pulls both SAM2 + SAM3 weights.
- `EVERYTHING.py` — headless: runs `00_venv.py` then every
  `plugin_*.py` in order. Forwards `--force` to each.
- `INSTALL.py` — Tk-based GUI installer wrapping the same scripts:
  checkbox per step, masked HF-token field (enabled only when a gated
  plugin is checked) that runs `huggingface_hub.login()` in the venv
  after `00_venv.py`, and a **--force** toggle.

Each script is **idempotent** (skips already-installed artifacts) and
supports `--force` to wipe and reinstall. `plugin_colmap.py` branches
by OS: Windows downloads the prebuilt CUDA zip, macOS uses `brew`,
Linux uses `apt-get` and auto-fixes known Ubuntu 26.04 packaging gaps
(missing `libposelib`).

Gated HF plugins (`plugin_sam`, `plugin_vggtomega`) need a token at
`$HF_HOME/token` (per-user, shared across venvs) — paste it into the
`INSTALL.py` GUI field, or run `hf auth login` from any environment.

## Dev server

To start or restart the Vite dev server:
- **macOS / Linux**: `bash run_server.sh`
- **Windows**: `run_server.bat`

Both free port 4444 (kill any existing listener) and run `npm run dev`.
Don't `npm run dev` directly — `strictPort: true` means a stale listener
on 4444 will fail the start.

## Git commits

Commits go through `git` invoked from a POSIX shell, **not** PowerShell.
Do not use PowerShell here-string syntax (`@'...'@`) for commit messages —
bash treats the `@` characters literally and they end up in the message.
For multi-line messages use a bash heredoc or write the message to a temp
file and pass it with `git commit -F <file>`.
