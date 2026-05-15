"""Run MapAnything for camera poses + depth + global pointmap from video frames.

Usage:
    python run_mapanything.py <scene_dir> [--subsample N]

Reads <scene_dir>/frames/NNNNNN.jpg (produced by extract_frames.py).
Runs MapAnything feed-forward inference to produce camera poses, metric depth,
per-frame pointmaps, and a global scene-level point cloud.

MapAnything is a universal metric 3D reconstruction model from Meta that
produces depth, intrinsics, poses, and metric scale in a single forward pass.

Outputs:
    <scene_dir>/mapanything/cameras.json         — same schema as other plugins
    <scene_dir>/mapanything/depth/NNNNNN.npz     — per-frame depth
    <scene_dir>/mapanything/pointmap/NNNNNN.npz  — per-frame camera-space pointmap + conf
    <scene_dir>/mapanything/scene_pointmap_chunks.json — global cloud manifest
    <scene_dir>/mapanything/scene_pointmap_NNN.npz     — chunked global cloud
"""

import sys
import os
import json
import argparse
import glob
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _device import pick_device  # noqa: E402
from _progress import progress  # noqa: E402
import torch.nn.functional as F

from _pointcloud_io import save_chunked_pointcloud


CONF_THRESHOLD = 0.1
DEPTH_EDGE_RTOL = 0.03


def depth_edge(depth, rtol=0.03, kernel_size=3):
    """Detect depth discontinuities."""
    shape = depth.shape
    depth = depth.reshape(-1, 1, *shape[-2:])
    pad = kernel_size // 2
    diff = (F.max_pool2d(depth, kernel_size, stride=1, padding=pad)
            + F.max_pool2d(-depth, kernel_size, stride=1, padding=pad))
    edge = (diff / depth).nan_to_num_() > rtol
    return edge.reshape(*shape)


def main():
    parser = argparse.ArgumentParser(description="Run MapAnything on video frames")
    parser.add_argument("scene_dir", help="Path to _scene directory")
    parser.add_argument("--subsample", type=int, default=3,
                        help="Use every Nth frame (default 3)")
    args = parser.parse_args()

    scene_dir = args.scene_dir
    frames_dir = os.path.join(scene_dir, "frames")
    out_dir = os.path.join(scene_dir, "mapanything")
    depth_dir = os.path.join(out_dir, "depth")
    pointmap_dir = os.path.join(out_dir, "pointmap")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(pointmap_dir, exist_ok=True)

    # Load frames metadata
    frames_json = os.path.join(scene_dir, "frames.json")
    with open(frames_json) as f:
        meta = json.load(f)
    src_w, src_h = meta["width"], meta["height"]

    # Collect frame paths
    all_frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not all_frames:
        print("[mapanything] ERROR: no frames found", file=sys.stderr)
        sys.exit(1)

    frames_to_use = all_frames[::args.subsample]
    frame_indices = [int(os.path.splitext(os.path.basename(f))[0]) for f in frames_to_use]
    N = len(frames_to_use)
    print(f"[mapanything] {N} frames (subsample={args.subsample}) at {src_w}x{src_h}")

    # Load images using MapAnything's utility
    from mapanything.utils.image import load_images

    progress(f"Preprocessing {N} frames...")
    print("[mapanything] Loading and preprocessing images...")
    views = load_images(
        frames_to_use,
        norm_type="dinov2",
    )
    print(f"[mapanything] Preprocessed {len(views)} views")

    # Keep original RGB for scene pointmap coloring
    rgb_list = []
    for path in frames_to_use:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_list.append(img)

    # Load model
    device = pick_device()
    progress(f"Loading MapAnything on {device}...")
    print(f"[mapanything] Loading MapAnything on {device}...")
    from mapanything.models import MapAnything
    model = MapAnything.from_pretrained("facebook/map-anything").to(device)

    # Run inference with memory-efficient settings for 12GB GPU
    progress(f"Running MapAnything inference on {N} frames...")
    print("[mapanything] Running inference...")
    t0 = time.time()

    # Check if bfloat16 flash attention works, fall back to fp16.
    # On non-CUDA we disable AMP entirely — MapAnything's amp path
    # assumes CUDA flash-attention kernels.
    use_amp = device == "cuda"
    amp_dtype = "bf16"
    if device == "cuda":
        cap = torch.cuda.get_device_capability()
        if cap[0] < 8:
            amp_dtype = "fp16"

    outputs = model.infer(
        views,
        memory_efficient_inference=True,
        minibatch_size=1,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        apply_mask=True,
        mask_edges=True,
        edge_depth_threshold=DEPTH_EDGE_RTOL,
    )
    elapsed = time.time() - t0
    progress(f"Inference done in {elapsed:.1f}s ({elapsed / N:.2f}s/frame)")
    print(f"[mapanything] Inference done in {elapsed:.1f}s ({elapsed / N:.2f}s/frame)")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Extract working resolution from first output
    first_depth = outputs[0]["depth_z"][0].squeeze(-1)  # (H, W)
    work_H, work_W = first_depth.shape
    print(f"[mapanything] Working resolution: {work_W}x{work_H}")

    # Collect per-frame intrinsics for median K
    all_K = []
    for pred in outputs:
        K_i = pred["intrinsics"][0].cpu().numpy()  # (3, 3)
        all_K.append(K_i)
    K_stack = np.stack(all_K, axis=0)
    K_median = np.median(K_stack, axis=0)
    fx = float(K_median[0, 0])
    fy = float(K_median[1, 1])
    cx = float(K_median[0, 2])
    cy = float(K_median[1, 2])
    K_out = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    print(f"[mapanything] Median intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    scale_factor = ((work_W / src_w) + (work_H / src_h)) / 2

    # ---- Write per-frame outputs ----
    progress(f"Writing {N} depth maps + pointmaps...")
    frames_out = []
    all_world_pts = []
    all_world_rgb = []
    all_world_conf = []

    for i in range(N):
        idx = frame_indices[i]
        pred = outputs[i]

        # Camera pose: cam-to-world (4x4), OpenCV convention
        c2w = pred["camera_poses"][0].cpu().numpy().astype(np.float64)

        # Invert cam-to-world -> world-to-cam for cameras.json
        R_wc = c2w[:3, :3]
        t_wc = c2w[:3, 3]
        R_cw = R_wc.T
        t_cw = -R_wc.T @ t_wc

        frames_out.append({
            "idx": idx,
            "name": f"{idx:06d}.jpg",
            "registered": True,
            "R": R_cw.tolist(),
            "t": t_cw.tolist(),
            "sparse_obs": [],
        })

        # Depth (Z-depth in camera frame)
        depth_np = pred["depth_z"][0].squeeze(-1).cpu().numpy()  # (H, W)
        np.savez_compressed(
            os.path.join(depth_dir, f"{idx:06d}.npz"),
            depth=depth_np.astype(np.float16),
        )

        # Per-frame pointmap (camera-space)
        pts_cam = pred["pts3d_cam"][0].cpu().numpy()  # (H, W, 3)
        conf_np = pred["conf"][0].cpu().numpy()       # (H, W)
        np.savez_compressed(
            os.path.join(pointmap_dir, f"{idx:06d}.npz"),
            pts3d=pts_cam.astype(np.float16),
            conf=conf_np.astype(np.float16),
        )

        # Collect for scene pointmap
        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)  # (H, W)
        pts_world = pred["pts3d"][0].cpu().numpy()  # (H, W, 3)

        # Resize RGB to match working resolution
        rgb_resized = cv2.resize(rgb_list[i], (work_W, work_H),
                                 interpolation=cv2.INTER_LINEAR)

        all_world_pts.append(pts_world[mask])
        all_world_rgb.append(rgb_resized[mask])
        all_world_conf.append(conf_np[mask])

    # ---- Write global scene pointmap ----
    scene_pts = np.concatenate(all_world_pts, axis=0).astype(np.float32)
    scene_rgb = np.concatenate(all_world_rgb, axis=0)
    scene_conf = np.concatenate(all_world_conf, axis=0).astype(np.float32)

    # Convert from OpenCV world coords to Three.js convention: (x, -y, -z)
    scene_pts[:, 1] *= -1
    scene_pts[:, 2] *= -1

    scene_manifest = save_chunked_pointcloud(
        out_dir, "scene_pointmap", scene_pts, scene_rgb, scene_conf
    )
    print(f"[mapanything] Scene pointmap: {scene_pts.shape[0]:,} points")

    # ---- Write cameras.json ----
    metric_scale = float(outputs[0]["metric_scaling_factor"][0].cpu())
    cameras = {
        "model": "MAPANYTHING",
        "checkpoint": "facebook/map-anything",
        "width": work_W,
        "height": work_H,
        "source_width": src_w,
        "source_height": src_h,
        "scale_factor": scale_factor,
        "subsample_every": args.subsample,
        "metric_scale": metric_scale,
        "K": K_out,
        "k1": 0.0,
        "num_points": 0,
        "num_registered": N,
        "frames": frames_out,
    }

    cam_path = os.path.join(out_dir, "cameras.json")
    with open(cam_path, "w") as f:
        json.dump(cameras, f, indent=2)

    print(f"[mapanything] Wrote {cam_path}")
    print(f"[mapanything] Wrote {N} depth maps to {depth_dir}")
    print(f"[mapanything] Wrote {N} pointmaps to {pointmap_dir}")
    print(f"[mapanything] Wrote scene pointmap manifest to {scene_manifest}")


if __name__ == "__main__":
    main()
