# video_vision

## Origin

This project was extracted from `D:\git\sam_experiments\segviewer` on
2026-05-09. The original directory still exists locally and may be
useful as a reference for:

- Local model checkpoints (e.g. `sam2.1_l.pt`) that were not copied
  over — model setup is being reworked here.
- Generated analysis outputs under `analysis/` (frames, depth maps,
  masks, point clouds, logs) from prior runs.
- Sample uploads under `uploads/` used during earlier experiments.

When in doubt about prior context or intermediate data, check the
original path before regenerating.

## Dev server

To start or restart the Vite dev server, run `bash scripts/restart-dev.sh`.
It frees port 4444 (kills any existing listener) and runs `npm run dev`.
Don't `npm run dev` directly — `strictPort: true` means a stale listener
on 4444 will fail the start.
