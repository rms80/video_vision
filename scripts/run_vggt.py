"""Run VGGT to get camera poses + depth + pointmaps from video frames.

Usage:
    python run_vggt.py <scene_dir> [--subsample N] [--anchor-step M]
                                   [--batch-size 24] [--overlap 4]

Two-phase strategy to stabilize multi-window runs:

  Phase 1 — Run VGGT on every Mth frame (the "anchors"). These frames share
            a single VGGT forward pass (or windowed+stitched if the anchor
            set is larger than `batch_size`), giving globally consistent
            poses and metric scale across the whole clip.

  Phase 2 — For each gap between consecutive anchors, run VGGT on the full
            span (both bracketing anchors + all in-between frames). Fit a
            similarity transform using both endpoint anchors (matches both
            position and metric scale against phase 1) and emit the
            in-between frames.

Outputs:
    <scene_dir>/vggt/cameras.json
    <scene_dir>/vggt/depth/NNNNNN.npz
    <scene_dir>/vggt/pointmap/NNNNNN.npz
"""

import sys
import os
import json
import argparse
import glob
import time
import contextlib

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _device import pick_device  # noqa: E402
from _progress import progress  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
VGGT_ROOT = os.path.join(REPO_ROOT, "models", "external", "vggt")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def camera_center(R_cw: np.ndarray, t_cw: np.ndarray) -> np.ndarray:
    return -R_cw.T @ t_cw


def apply_similarity_to_extrinsic(R_cw: np.ndarray, t_cw: np.ndarray,
                                  s: float, R: np.ndarray, t: np.ndarray
                                  ) -> tuple[np.ndarray, np.ndarray]:
    """Given a similarity X_global = s·R·X_local + t on world points, return
    the cam-from-world extrinsic in the global frame. Cam-space points from
    this camera must be scaled by `s` to keep metric consistent."""
    R_cw_new = R_cw @ R.T
    t_cw_new = s * t_cw - R_cw_new @ t
    return R_cw_new, t_cw_new


def align_extrinsics_similarity(local_extr: list[np.ndarray],
                                global_extr: list[np.ndarray]
                                ) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate similarity (s, R, t) from matched cam-from-world extrinsics.
    Rotation from averaged camera orientations, scale from pairwise distance
    ratios, translation from centroids."""
    n = len(local_extr)
    R_sum = np.zeros((3, 3))
    for le, ge in zip(local_extr, global_extr):
        R_sum += ge[:3, :3].T @ le[:3, :3]
    U, _, Vt = np.linalg.svd(R_sum)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt

    local_C = np.stack([camera_center(e[:3, :3], e[:3, 3]) for e in local_extr])
    global_C = np.stack([camera_center(e[:3, :3], e[:3, 3]) for e in global_extr])
    ratios = []
    for i in range(n):
        for j in range(i + 1, n):
            d_local = np.linalg.norm(local_C[i] - local_C[j])
            if d_local > 1e-6:
                ratios.append(np.linalg.norm(global_C[i] - global_C[j]) / d_local)
    s = float(np.median(ratios)) if ratios else 1.0
    t = global_C.mean(axis=0) - s * (R @ local_C.mean(axis=0))
    return s, R, t


def compute_windows(n: int, batch_size: int, overlap: int) -> list[list[int]]:
    if n <= batch_size:
        return [list(range(n))]
    stride = batch_size - overlap
    assert stride > 0, "overlap must be smaller than batch_size"
    windows: list[list[int]] = []
    start = 0
    while start + batch_size < n:
        windows.append(list(range(start, start + batch_size)))
        start += stride
    windows.append(list(range(n - batch_size, n)))
    return windows


# ---------------------------------------------------------------------------
# VGGT inference
# ---------------------------------------------------------------------------

class VGGTRunner:
    """Thin wrapper around a loaded VGGT model that yields numpy outputs."""

    def __init__(self, checkpoint: str):
        sys.path.insert(0, VGGT_ROOT)
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        self._load_and_preprocess = load_and_preprocess_images
        self._pose_encoding_to_extri_intri = pose_encoding_to_extri_intri

        self.device = pick_device()
        if self.device == "cuda":
            cap_major = torch.cuda.get_device_capability()[0]
            self.dtype = torch.bfloat16 if cap_major >= 8 else torch.float16
        else:
            self.dtype = torch.float32
        progress(f"Loading VGGT on {self.device} ({self.dtype})...")
        print(f"[vggt] Loading {checkpoint} on {self.device} ({self.dtype})...")
        self.model = VGGT.from_pretrained(checkpoint).to(self.device)
        self.model.eval()

    def run(self, paths: list[str]) -> dict:
        images = self._load_and_preprocess(paths).to(self.device)
        Nw, _, H, W = images.shape
        t0 = time.time()
        with torch.no_grad():
            ctx = (torch.amp.autocast("cuda", dtype=self.dtype)
                   if self.device == "cuda" else contextlib.nullcontext())
            with ctx:
                batched = images[None]
                tokens, ps_idx = self.model.aggregator(batched)
                pose_enc = self.model.camera_head(tokens)[-1]
                extrinsic, intrinsic = self._pose_encoding_to_extri_intri(
                    pose_enc, batched.shape[-2:]
                )
                depth_map, depth_conf = self.model.depth_head(tokens, batched, ps_idx)
        elapsed = time.time() - t0

        extr = extrinsic.squeeze(0).cpu().float().numpy()            # (N, 3, 4)
        intr = intrinsic.squeeze(0).cpu().float().numpy()            # (N, 3, 3)
        dmap = depth_map.squeeze(0).squeeze(-1).cpu().float().numpy() # (N, H, W)
        dconf = depth_conf.squeeze(0).cpu().float().numpy()           # (N, H, W)

        del images, batched, tokens, pose_enc, extrinsic, intrinsic, depth_map, depth_conf
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return {"extr": extr, "intr": intr, "depth": dmap, "conf": dconf,
                "H": H, "W": W, "elapsed": elapsed}


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_windowed_phase1(runner: VGGTRunner, paths: list[str],
                        batch_size: int, overlap: int
                        ) -> tuple[list, list, list, list, int, int]:
    """Run VGGT over overlapping windows and stitch outputs (similarity) into
    a single global frame. Used only when the anchor set is larger than the
    VGGT batch budget — smaller anchor sets take the single-pass fast path."""
    N = len(paths)
    windows = compute_windows(N, batch_size, overlap)
    extr_out: list[np.ndarray | None] = [None] * N
    intr_out: list[np.ndarray | None] = [None] * N
    depth_out: list[np.ndarray | None] = [None] * N
    conf_out: list[np.ndarray | None] = [None] * N
    H_out = W_out = None

    for wi, window in enumerate(windows):
        window_paths = [paths[i] for i in window]
        progress(f"VGGT phase 1: anchor window {wi+1}/{len(windows)} ({len(window)} frames)")
        print(f"[vggt] phase1 window {wi+1}/{len(windows)}: anchors "
              f"{window[0]}..{window[-1]} (count={len(window)})")
        out = runner.run(window_paths)
        if H_out is None:
            H_out, W_out = out["H"], out["W"]
            print(f"[vggt] working resolution: {W_out}x{H_out}")
        extr, intr, dmap, dconf = out["extr"], out["intr"], out["depth"], out["conf"]
        print(f"[vggt]   inference {out['elapsed']:.1f}s "
              f"({out['elapsed']/len(window):.2f}s/frame)")

        if wi == 0:
            s_a, R_a, t_a = 1.0, np.eye(3), np.zeros(3)
        else:
            overlap_ids = [fi for fi in window if extr_out[fi] is not None]
            local_extrs = [extr[window.index(fi)].astype(np.float64) for fi in overlap_ids]
            global_extrs = [extr_out[fi] for fi in overlap_ids]
            s_a, R_a, t_a = align_extrinsics_similarity(local_extrs, global_extrs)
            print(f"[vggt]   aligned via {len(overlap_ids)} overlap frames (s={s_a:.3f})")

        for wpos, fi in enumerate(window):
            if extr_out[fi] is not None:
                continue
            R_cw_l = extr[wpos, :3, :3].astype(np.float64)
            t_cw_l = extr[wpos, :3, 3].astype(np.float64)
            R_cw_new, t_cw_new = apply_similarity_to_extrinsic(
                R_cw_l, t_cw_l, s_a, R_a, t_a
            )
            extr_out[fi] = np.concatenate([R_cw_new, t_cw_new[:, None]], axis=1)
            intr_out[fi] = intr[wpos].astype(np.float64)
            depth_out[fi] = dmap[wpos] * s_a
            conf_out[fi] = dconf[wpos]

    assert all(e is not None for e in extr_out), "phase1: some anchors never processed"
    return extr_out, intr_out, depth_out, conf_out, H_out, W_out


def main():
    parser = argparse.ArgumentParser(description="Run VGGT on video frames")
    parser.add_argument("scene_dir", help="Path to _scene directory")
    parser.add_argument("--anchor-step", type=int, default=10,
                        help="Phase-1 anchor density: every Mth frame is an "
                             "anchor (default 10). Ignored if --num-frames is set.")
    parser.add_argument("--num-frames", type=int, default=None,
                        help="Target total anchor frames. When set, overrides "
                             "--anchor-step and picks exactly N evenly-spaced "
                             "frames from 0 to last (inclusive).")
    parser.add_argument("--interior-step", type=int, default=1,
                        help="Phase-2 density: within each span, take every "
                             "Nth interior frame (default 1 = all interiors)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Max frames per VGGT forward pass (default 32)")
    parser.add_argument("--overlap", type=int, default=4,
                        help="Overlap for phase-1 windowing when anchors > batch (default 4)")
    parser.add_argument("--checkpoint", default="facebook/VGGT-1B",
                        help="HuggingFace model id")
    parser.add_argument("--no-span-align", action="store_true",
                        help="Debug: emit phase-2 interior frames in their span's "
                             "LOCAL frame (no alignment). Anchors keep phase-1 poses.")
    parser.add_argument("--anchors-only", action="store_true",
                        help="Skip phase 2 entirely; output only phase-1 anchor frames.")
    args = parser.parse_args()

    scene_dir = args.scene_dir
    frames_dir = os.path.join(scene_dir, "frames")
    out_dir = os.path.join(scene_dir, "vggt")
    depth_dir = os.path.join(out_dir, "depth")
    pointmap_dir = os.path.join(out_dir, "pointmap")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(pointmap_dir, exist_ok=True)

    frames_json = os.path.join(scene_dir, "frames.json")
    with open(frames_json) as f:
        meta = json.load(f)
    src_w, src_h = meta["width"], meta["height"]

    all_frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not all_frames:
        print("[vggt] ERROR: no frames found", file=sys.stderr)
        sys.exit(1)

    frames_to_use = all_frames
    frame_indices = [int(os.path.splitext(os.path.basename(f))[0]) for f in frames_to_use]
    N = len(frames_to_use)
    print(f"[vggt] {N} frames at {src_w}x{src_h}")

    # Anchor set: either exactly T evenly-spaced positions (when
    # --num-frames is set) or every Mth frame plus the last frame.
    if args.num_frames is not None and args.num_frames > 0:
        T = min(args.num_frames, N)
        if T <= 1:
            anchor_positions = [0]
        else:
            anchor_positions = sorted({
                round(i * (N - 1) / (T - 1)) for i in range(T)
            })
    else:
        anchor_positions = sorted(set(list(range(0, N, args.anchor_step)) + [N - 1]))
    print(f"[vggt] {len(anchor_positions)} anchors, "
          f"{len(anchor_positions) - 1} span(s) to fill")

    def span_interior_positions(a: int, b: int) -> list[int]:
        """Return the subsampled interior positions of span (a, b), i.e.
        every `--interior-step`th frame strictly between a and b."""
        return list(range(a + args.interior_step, b, args.interior_step))

    # Each phase-2 pass feeds VGGT all anchor frames plus one span's
    # (subsampled) interior frames. Check the largest such batch fits.
    # Skip entirely when --anchors-only: phase 2 never runs.
    if not args.anchors_only:
        max_interior = max(
            (len(span_interior_positions(anchor_positions[i], anchor_positions[i + 1]))
             for i in range(len(anchor_positions) - 1)),
            default=0,
        )
        phase2_total = len(anchor_positions) + max_interior
        if phase2_total > args.batch_size:
            print(f"[vggt] ERROR: phase-2 batch {phase2_total} (= {len(anchor_positions)} "
                  f"anchors + {max_interior} interior) > batch_size {args.batch_size}. "
                  f"Raise --batch-size, --anchor-step, or --interior-step.",
                  file=sys.stderr)
            sys.exit(1)
        print(f"[vggt] phase-2 batch size = {phase2_total} "
              f"(interior-step={args.interior_step})")

    runner = VGGTRunner(args.checkpoint)

    # Per-frame global outputs.
    frame_extr: list[np.ndarray | None] = [None] * N
    frame_intr: list[np.ndarray | None] = [None] * N
    frame_depth: list[np.ndarray | None] = [None] * N
    frame_conf: list[np.ndarray | None] = [None] * N

    # ----- Phase 1: anchors ------------------------------------------------
    anchor_paths = [frames_to_use[p] for p in anchor_positions]
    if len(anchor_paths) <= args.batch_size:
        progress(f"VGGT phase 1: single pass over {len(anchor_paths)} anchors")
        print(f"[vggt] phase1: single pass over {len(anchor_paths)} anchors")
        out = runner.run(anchor_paths)
        print(f"[vggt]   inference {out['elapsed']:.1f}s "
              f"({out['elapsed']/len(anchor_paths):.2f}s/frame)")
        work_H, work_W = out["H"], out["W"]
        print(f"[vggt] working resolution: {work_W}x{work_H}")
        for i, pos in enumerate(anchor_positions):
            frame_extr[pos] = np.concatenate(
                [out["extr"][i, :3, :3].astype(np.float64),
                 out["extr"][i, :3, 3:4].astype(np.float64)], axis=1
            )
            frame_intr[pos] = out["intr"][i].astype(np.float64)
            frame_depth[pos] = out["depth"][i]
            frame_conf[pos] = out["conf"][i]
    else:
        extr_a, intr_a, depth_a, conf_a, work_H, work_W = run_windowed_phase1(
            runner, anchor_paths, args.batch_size, args.overlap
        )
        for i, pos in enumerate(anchor_positions):
            frame_extr[pos] = extr_a[i]
            frame_intr[pos] = intr_a[i]
            frame_depth[pos] = depth_a[i]
            frame_conf[pos] = conf_a[i]

    # ----- Phase 2: spans --------------------------------------------------
    from vggt.utils.geometry import depth_to_cam_coords_points  # noqa: E402

    if args.anchors_only:
        print("[vggt] --anchors-only: skipping phase 2")
    for si in ([] if args.anchors_only else range(len(anchor_positions) - 1)):
        a_pos = anchor_positions[si]
        b_pos = anchor_positions[si + 1]
        if b_pos - a_pos < 2:
            continue  # no in-between frames

        interior_positions = span_interior_positions(a_pos, b_pos)
        if not interior_positions:
            continue
        # Send ALL phase-1 anchors + this span's interior frames so VGGT
        # reasons about the whole scene geometry and the interior frames land
        # in a coordinate frame shared with all anchors.
        batch_positions = sorted(set(anchor_positions) | set(interior_positions))
        batch_paths = [frames_to_use[p] for p in batch_positions]
        pos_to_batch_idx = {p: i for i, p in enumerate(batch_positions)}
        progress(f"VGGT phase 2: span {si+1}/{len(anchor_positions)-1} "
                 f"({len(interior_positions)} interior frames)")
        print(f"[vggt] phase2 span {si+1}/{len(anchor_positions)-1}: "
              f"interior {a_pos+1}..{b_pos-1} ({len(interior_positions)} frames), "
              f"batch size {len(batch_paths)}")

        out = runner.run(batch_paths)
        extr, intr, dmap, dconf = out["extr"], out["intr"], out["depth"], out["conf"]
        print(f"[vggt]   inference {out['elapsed']:.1f}s "
              f"({out['elapsed']/len(batch_paths):.2f}s/frame)")

        if args.no_span_align:
            s_align, R_align, t_align = 1.0, np.eye(3), np.zeros(3)
            print(f"[vggt]   NO-ALIGN mode: emitting interior frames in span-local frame")
        else:
            # Similarity alignment using every phase-1 anchor appearing in
            # this batch.
            local_extrs = [extr[pos_to_batch_idx[p]].astype(np.float64)
                           for p in anchor_positions]
            global_extrs = [frame_extr[p] for p in anchor_positions]
            s_align, R_align, t_align = align_extrinsics_similarity(
                local_extrs, global_extrs
            )
            print(f"[vggt]   aligned via {len(anchor_positions)} anchors "
                  f"(s={s_align:.3f})")

        # Emit only the interior frames; anchors keep their phase-1 poses.
        for pos in interior_positions:
            bi = pos_to_batch_idx[pos]
            R_cw_l = extr[bi, :3, :3].astype(np.float64)
            t_cw_l = extr[bi, :3, 3].astype(np.float64)
            R_cw_new, t_cw_new = apply_similarity_to_extrinsic(
                R_cw_l, t_cw_l, s_align, R_align, t_align
            )
            frame_extr[pos] = np.concatenate([R_cw_new, t_cw_new[:, None]], axis=1)
            frame_intr[pos] = intr[bi].astype(np.float64)
            frame_depth[pos] = dmap[bi] * s_align
            frame_conf[pos] = dconf[bi]

    # Positions we actually have data for: all anchors plus the sampled
    # interior frames from each span. Everything else is skipped.
    emitted_positions = sorted({p for p, e in enumerate(frame_extr) if e is not None})
    print(f"[vggt] Emitting {len(emitted_positions)} frames of {N} total "
          f"({len(anchor_positions)} anchors + "
          f"{len(emitted_positions) - len(anchor_positions)} interior)")

    # ----- Write outputs ---------------------------------------------------
    progress(f"Writing {len(emitted_positions)} depth maps + pointmaps...")
    K_stack = np.stack([frame_intr[p] for p in emitted_positions], axis=0)
    median_K = np.median(K_stack, axis=0)
    fx, fy = float(median_K[0, 0]), float(median_K[1, 1])
    cx, cy = float(median_K[0, 2]), float(median_K[1, 2])
    K_out = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    print(f"[vggt] Median intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    scale_factor = ((work_W / src_w) + (work_H / src_h)) / 2

    frames_out = []
    for pos in emitted_positions:
        idx = frame_indices[pos]
        E = frame_extr[pos]
        frames_out.append({
            "idx": idx,
            "name": f"{idx:06d}.jpg",
            "registered": True,
            "R": E[:3, :3].tolist(),
            "t": E[:3, 3].tolist(),
            "sparse_obs": [],
        })

        depth_i = frame_depth[pos]
        np.savez_compressed(
            os.path.join(depth_dir, f"{idx:06d}.npz"),
            depth=depth_i.astype(np.float16),
        )

        pts_cam = depth_to_cam_coords_points(depth_i, frame_intr[pos])
        np.savez_compressed(
            os.path.join(pointmap_dir, f"{idx:06d}.npz"),
            pts3d=pts_cam.astype(np.float16),
            conf=frame_conf[pos].astype(np.float16),
        )

    cameras = {
        "model": "VGGT",
        "width": work_W,
        "height": work_H,
        "source_width": src_w,
        "source_height": src_h,
        "scale_factor": scale_factor,
        "anchor_step": args.anchor_step,
        "interior_step": args.interior_step,
        "K": K_out,
        "k1": 0.0,
        "num_points": 0,
        "num_registered": len(emitted_positions),
        "frames": frames_out,
    }

    cam_path = os.path.join(out_dir, "cameras.json")
    with open(cam_path, "w") as f:
        json.dump(cameras, f, indent=2)
    print(f"[vggt] Wrote {cam_path}")
    print(f"[vggt] Wrote {len(emitted_positions)} depth maps to {depth_dir}")
    print(f"[vggt] Wrote {len(emitted_positions)} pointmaps to {pointmap_dir}")


if __name__ == "__main__":
    main()
