"""Run Depth-Anything-3 for camera poses + metric depth.

Two-model pipeline:
  1. DA3-LARGE-1.1 (any-view) -> joint pose, intrinsics, relative depth, conf
  2. DA3METRIC-LARGE (monocular) -> metric depth, converted via
     `metric = focal * net_output / 300`. The pose model alone has no metric
     scale, so we rescale its world (translations) to match the metric depth
     by fitting a single scene-wide ratio between the two depth predictions.

Usage:
    python run_da3.py <scene_dir> [--subsample N] [--process-res 504]

Outputs:
    <scene_dir>/da3/cameras.json
    <scene_dir>/da3/depth/NNNNNN.npz
    <scene_dir>/da3/pointmap/NNNNNN.npz
"""

import sys
import os
import json
import argparse
import gc
import glob
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _device import pick_device  # noqa: E402
from _progress import progress  # noqa: E402


POSE_CKPT = "depth-anything/DA3-LARGE-1.1"
METRIC_CKPT = "depth-anything/DA3METRIC-LARGE"


def depth_to_cam_points(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Back-project a depth map (H, W) into camera-space (H, W, 3) points."""
    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    x = (us - cx) * depth / fx
    y = (vs - cy) * depth / fy
    z = depth
    return np.stack([x, y, z], axis=-1)


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Run DA3 (pose + metric depth)")
    parser.add_argument("scene_dir", help="Path to _scene directory")
    parser.add_argument("--subsample", type=int, default=2,
                        help="Use every Nth frame (default 2)")
    parser.add_argument("--process-res", type=int, default=504,
                        help="DA3 processing resolution (default 504)")
    args = parser.parse_args()

    scene_dir = args.scene_dir
    frames_dir = os.path.join(scene_dir, "frames")
    out_dir = os.path.join(scene_dir, "da3")
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
        print("[da3] ERROR: no frames found", file=sys.stderr)
        sys.exit(1)

    frames_to_use = all_frames[::args.subsample]
    frame_indices = [int(os.path.splitext(os.path.basename(f))[0]) for f in frames_to_use]
    N = len(frames_to_use)
    print(f"[da3] {N} frames (subsample={args.subsample}) at {src_w}x{src_h}")

    from depth_anything_3.api import DepthAnything3

    device = pick_device()
    print(f"[da3] device={device}", flush=True)

    # ---- Pass 1: pose model ----------------------------------------------
    progress(f"Loading DA3 pose model on {device}...")
    print(f"[da3] Loading {POSE_CKPT} on {device}...")
    pose_model = DepthAnything3.from_pretrained(POSE_CKPT).to(device).eval()
    progress(f"DA3 pose+depth inference on {N} frames...")
    print("[da3] Running pose+depth inference...")
    t0 = time.time()
    pose_pred = pose_model.inference(image=frames_to_use, process_res=args.process_res)
    pose_elapsed = time.time() - t0
    progress(f"DA3 pose pass done in {pose_elapsed:.1f}s")
    print(f"[da3] Pose pass {pose_elapsed:.1f}s")

    if pose_pred.extrinsics is None or pose_pred.intrinsics is None:
        print("[da3] ERROR: pose model returned no extrinsics/intrinsics", file=sys.stderr)
        sys.exit(1)

    depth_rel = pose_pred.depth.astype(np.float32)               # (N, H, W) relative
    conf = (pose_pred.conf.astype(np.float32)
            if pose_pred.conf is not None else None)             # (N, H, W) or None
    intr = pose_pred.intrinsics.astype(np.float64)               # (N, 3, 3)
    extr = pose_pred.extrinsics                                  # (N, 3, 4) or (N, 4, 4) w2c
    if extr.shape[-2:] == (4, 4):
        extr = extr[:, :3, :]
    extr = extr.astype(np.float64)

    _, work_H, work_W = depth_rel.shape
    print(f"[da3] working resolution: {work_W}x{work_H}")

    del pose_model, pose_pred
    free_cuda()

    # ---- Pass 2: metric model --------------------------------------------
    progress(f"Loading DA3 metric model on {device}...")
    print(f"[da3] Loading {METRIC_CKPT} on {device}...")
    metric_model = DepthAnything3.from_pretrained(METRIC_CKPT).to(device).eval()
    progress(f"DA3 metric depth inference on {N} frames...")
    print("[da3] Running metric depth inference...")
    t0 = time.time()
    metric_pred = metric_model.inference(image=frames_to_use, process_res=args.process_res)
    metric_elapsed = time.time() - t0
    progress(f"DA3 metric pass done in {metric_elapsed:.1f}s")
    print(f"[da3] Metric pass {metric_elapsed:.1f}s")

    metric_raw = metric_pred.depth.astype(np.float32)            # (N, H, W) raw net output
    if metric_raw.shape != depth_rel.shape:
        print(f"[da3] ERROR: metric shape {metric_raw.shape} != pose shape {depth_rel.shape}",
              file=sys.stderr)
        sys.exit(1)

    del metric_model, metric_pred
    free_cuda()

    # Per-frame conversion: meters = focal_avg * net_output / 300.
    depth_metric = np.empty_like(metric_raw)
    for i in range(N):
        focal = 0.5 * (float(intr[i, 0, 0]) + float(intr[i, 1, 1]))
        depth_metric[i] = metric_raw[i] * (focal / 300.0)

    # ---- Reconcile world scale -------------------------------------------
    # Pose model's depth is in arbitrary (relative) units; metric model's is in
    # meters. Ratios across both give a single scene-wide scalar that
    # rescales pose translations into meters. Use only confident, positive,
    # finite pixels.
    valid = (depth_rel > 1e-3) & (depth_metric > 1e-3) & np.isfinite(depth_rel) & np.isfinite(depth_metric)
    if conf is not None:
        valid &= conf > np.quantile(conf, 0.5)
    ratios = (depth_metric[valid] / depth_rel[valid])
    if ratios.size < 100:
        print("[da3] WARNING: too few valid pixels for scale; defaulting s=1", file=sys.stderr)
        s = 1.0
    else:
        s = float(np.median(ratios))
    print(f"[da3] world scale (meters / pose-units) = {s:.4f} "
          f"(from {ratios.size:,} pixels)")

    # Apply to translations only — rotations are scale-invariant. Note
    # extrinsics are world->camera: t_cw = -R_cw @ C_world. Scaling C_world
    # by s is equivalent to scaling t_cw by s.
    extr[:, :3, 3] *= s

    # ---- Median K for cameras.json ---------------------------------------
    median_K = np.median(intr, axis=0)
    fx_med, fy_med = float(median_K[0, 0]), float(median_K[1, 1])
    cx_med, cy_med = float(median_K[0, 2]), float(median_K[1, 2])
    K_out = [[fx_med, 0.0, cx_med], [0.0, fy_med, cy_med], [0.0, 0.0, 1.0]]
    print(f"[da3] Median intrinsics: fx={fx_med:.1f} fy={fy_med:.1f} "
          f"cx={cx_med:.1f} cy={cy_med:.1f}")

    scale_factor = ((work_W / src_w) + (work_H / src_h)) / 2

    # ---- Write outputs ---------------------------------------------------
    progress(f"Writing {N} depth maps + pointmaps...")
    frames_out = []
    for i in range(N):
        idx = frame_indices[i]
        frames_out.append({
            "idx": idx,
            "name": f"{idx:06d}.jpg",
            "registered": True,
            "R": extr[i, :3, :3].tolist(),
            "t": extr[i, :3, 3].tolist(),
            "sparse_obs": [],
        })

        d = depth_metric[i]
        np.savez_compressed(
            os.path.join(depth_dir, f"{idx:06d}.npz"),
            depth=d.astype(np.float16),
        )

        pts_cam = depth_to_cam_points(d, intr[i])
        conf_i = (conf[i] if conf is not None
                  else np.ones((work_H, work_W), dtype=np.float32))
        np.savez_compressed(
            os.path.join(pointmap_dir, f"{idx:06d}.npz"),
            pts3d=pts_cam.astype(np.float16),
            conf=conf_i.astype(np.float16),
        )

    cameras = {
        "model": "DA3",
        "pose_checkpoint": POSE_CKPT,
        "metric_checkpoint": METRIC_CKPT,
        "width": work_W,
        "height": work_H,
        "source_width": src_w,
        "source_height": src_h,
        "scale_factor": scale_factor,
        "world_scale_meters_per_unit": s,
        "subsample_every": args.subsample,
        "process_res": args.process_res,
        "K": K_out,
        "k1": 0.0,
        "num_points": 0,
        "num_registered": N,
        "frames": frames_out,
    }

    cam_path = os.path.join(out_dir, "cameras.json")
    with open(cam_path, "w") as f:
        json.dump(cameras, f, indent=2)
    print(f"[da3] Wrote {cam_path}")
    print(f"[da3] Wrote {N} depth maps to {depth_dir}")
    print(f"[da3] Wrote {N} pointmaps to {pointmap_dir}")


if __name__ == "__main__":
    main()
