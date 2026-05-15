"""Run Pi3X to get camera poses + depth + global pointmap from video frames.

Usage:
    python run_pi3.py <scene_dir> [--subsample N]

Reads <scene_dir>/frames/NNNNNN.jpg (produced by extract_frames.py).
Runs Pi3X feed-forward inference to produce camera poses, metric depth,
per-frame pointmaps, and a global scene-level point cloud.

Pi3X is an improved version of Pi3 with smoother reconstructions and
approximate metric scale recovery. Conditioning inputs (depth, intrinsics,
poses) are optional and not used here — we run in image-only mode.

Outputs:
    <scene_dir>/pi3/cameras.json         — same schema as other plugins
    <scene_dir>/pi3/depth/NNNNNN.npz     — per-frame depth
    <scene_dir>/pi3/pointmap/NNNNNN.npz  — per-frame camera-space pointmap + conf
    <scene_dir>/pi3/scene_pointmap_chunks.json — manifest for the global cloud
    <scene_dir>/pi3/scene_pointmap_NNN.npz     — chunked global cloud (streamed)
"""

import sys
import os
import json
import argparse
import glob
import math
import time

import contextlib

# Make stdout/stderr unbuffered so prepare.log shows progress instead of
# only the final batch dump when the process exits.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _device import pick_device  # noqa: E402
from _progress import progress  # noqa: E402
from torchvision import transforms

from _pointcloud_io import save_chunked_pointcloud


PIXEL_LIMIT = 255_000
CONF_THRESHOLD = 0.1
DEPTH_EDGE_RTOL = 0.03


def compute_target_size(W_orig, H_orig, pixel_limit=PIXEL_LIMIT):
    """Compute target (W, H) divisible by 14, respecting aspect ratio and pixel limit."""
    scale = math.sqrt(pixel_limit / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    k = round(W_orig * scale / 14)
    m = round(H_orig * scale / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > W_orig / H_orig:
            k -= 1
        else:
            m -= 1
    return max(1, k) * 14, max(1, m) * 14


def depth_edge(depth, rtol=0.03, kernel_size=3):
    """Detect depth discontinuities (adapted from pi3.utils.geometry)."""
    shape = depth.shape
    depth = depth.reshape(-1, 1, *shape[-2:])
    pad = kernel_size // 2
    diff = (F.max_pool2d(depth, kernel_size, stride=1, padding=pad)
            + F.max_pool2d(-depth, kernel_size, stride=1, padding=pad))
    edge = (diff / depth).nan_to_num_() > rtol
    return edge.reshape(*shape)


def main():
    parser = argparse.ArgumentParser(description="Run Pi3X on video frames")
    parser.add_argument("scene_dir", help="Path to _scene directory")
    parser.add_argument("--subsample", type=int, default=2,
                        help="Use every Nth frame (default 2)")
    args = parser.parse_args()

    scene_dir = args.scene_dir
    frames_dir = os.path.join(scene_dir, "frames")
    out_dir = os.path.join(scene_dir, "pi3")
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
        print("[pi3] ERROR: no frames found", file=sys.stderr)
        sys.exit(1)

    frames_to_use = all_frames[::args.subsample]
    frame_indices = [int(os.path.splitext(os.path.basename(f))[0]) for f in frames_to_use]
    N = len(frames_to_use)
    print(f"[pi3] {N} frames (subsample={args.subsample}) at {src_w}x{src_h}")

    # Compute target size
    target_w, target_h = compute_target_size(src_w, src_h)
    print(f"[pi3] Target resolution: {target_w}x{target_h}")

    # Load and preprocess images
    progress(f"Preprocessing {N} frames to {target_w}x{target_h}...")
    to_tensor = transforms.ToTensor()
    imgs_list = []
    rgb_list = []  # Keep uint8 copies for scene_pointmap coloring
    for path in frames_to_use:
        img = Image.open(path).convert("RGB")
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        rgb_list.append(np.array(img))  # (H, W, 3) uint8
        imgs_list.append(to_tensor(img))  # (3, H, W) in [0, 1]

    imgs = torch.stack(imgs_list, dim=0)  # (N, 3, H, W)

    # Load model
    device = pick_device()
    progress(f"Loading Pi3X on {device}...")
    print(f"[pi3] Loading Pi3X on {device}...")
    from pi3.models.pi3x import Pi3X
    model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()
    model.disable_multimodal()  # no conditioning inputs, save memory
    print("[pi3] Multimodal branches disabled (image-only mode)")

    # Run inference
    progress(f"Running Pi3X inference on {N} frames...")
    print("[pi3] Running inference...")
    t0 = time.time()
    # Use float16 — Pi3's attention forces FLASH_ATTENTION for bfloat16 which
    # may not be available in all torch builds. Skip autocast on non-CUDA
    # (MPS/CPU run in float32 fine).
    amp_ctx = (torch.amp.autocast("cuda", dtype=torch.float16)
               if device == "cuda" else contextlib.nullcontext())
    with torch.no_grad(), amp_ctx:
        res = model(imgs[None].to(device))  # (1, N, 3, H, W) -> dict
    elapsed = time.time() - t0
    progress(f"Inference done in {elapsed:.1f}s ({elapsed / N:.2f}s/frame)")
    print(f"[pi3] Inference done in {elapsed:.1f}s ({elapsed / N:.2f}s/frame)")

    # Extract outputs (remove batch dim)
    # Pi3X applies metric scale internally: points and local_points are
    # already in metric units, camera translations are already scaled.
    metric_scale = float(res["metric"][0].cpu())
    print(f"[pi3] Metric scale factor: {metric_scale:.4f}")

    points = res["points"][0].cpu()              # (N, H, W, 3) world-space, metric
    local_points = res["local_points"][0].cpu()   # (N, H, W, 3) camera-space, metric
    conf_logits = res["conf"][0, ..., 0].cpu()    # (N, H, W)
    cam2worlds = res["camera_poses"][0].cpu()      # (N, 4, 4) cam-to-world, metric
    rays = res["rays"][0].cpu()                    # (N, H, W, 3) normalized ray dirs

    del model, res
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Derive intrinsics from ray directions using Pi3's utility
    from pi3.utils.geometry import recover_intrinsic_from_rays_d
    # rays shape: (N, H, W, 3) — recover K per frame, then take median
    K_per_frame = recover_intrinsic_from_rays_d(rays)  # (N, 3, 3)
    K_median = K_per_frame.median(dim=0).values.numpy().astype(np.float64)
    fx = float(K_median[0, 0])
    fy = float(K_median[1, 1])
    cx = float(K_median[0, 2])
    cy = float(K_median[1, 2])
    K_out = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    print(f"[pi3] Intrinsics from rays: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    # Compute confidence mask
    conf_prob = torch.sigmoid(conf_logits)  # (N, H, W)
    conf_mask = conf_prob > CONF_THRESHOLD
    non_edge = ~depth_edge(local_points[..., 2], rtol=DEPTH_EDGE_RTOL)
    valid_mask = conf_mask & non_edge  # (N, H, W)

    scale_factor = ((target_w / src_w) + (target_h / src_h)) / 2

    # ---- Write per-frame outputs ----
    progress(f"Writing {N} depth maps + pointmaps...")
    frames_out = []
    for i in range(N):
        idx = frame_indices[i]
        c2w = cam2worlds[i].numpy().astype(np.float64)

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

        # Depth = Z component of local_points (already in metric units)
        depth_np = local_points[i, :, :, 2].numpy()
        np.savez_compressed(
            os.path.join(depth_dir, f"{idx:06d}.npz"),
            depth=depth_np.astype(np.float16),
        )

        # Per-frame pointmap (camera-space, metric)
        pts_np = local_points[i].numpy()
        conf_np = conf_prob[i].numpy()
        np.savez_compressed(
            os.path.join(pointmap_dir, f"{idx:06d}.npz"),
            pts3d=pts_np.astype(np.float16),
            conf=conf_np.astype(np.float16),
        )

    # ---- Write global scene pointmap ----
    progress("Building scene-wide point cloud...")
    # Collect valid world-space points + RGB from all frames
    all_pts = []
    all_rgb = []
    all_conf = []
    for i in range(N):
        mask_i = valid_mask[i].numpy()  # (H, W) bool
        pts_i = points[i].numpy()       # (H, W, 3)
        rgb_i = rgb_list[i]             # (H, W, 3) uint8
        conf_i = conf_prob[i].numpy()   # (H, W)

        all_pts.append(pts_i[mask_i])   # (M_i, 3)
        all_rgb.append(rgb_i[mask_i])   # (M_i, 3)
        all_conf.append(conf_i[mask_i]) # (M_i,)

    scene_pts = np.concatenate(all_pts, axis=0).astype(np.float32)
    scene_rgb = np.concatenate(all_rgb, axis=0)
    scene_conf = np.concatenate(all_conf, axis=0).astype(np.float32)

    # Convert from OpenCV world coords to Three.js convention: (x, -y, -z)
    scene_pts[:, 1] *= -1
    scene_pts[:, 2] *= -1

    scene_manifest = save_chunked_pointcloud(
        out_dir, "scene_pointmap", scene_pts, scene_rgb, scene_conf
    )
    print(f"[pi3] Scene pointmap: {scene_pts.shape[0]:,} points")

    # ---- Write cameras.json ----
    cameras = {
        "model": "PI3X",
        "checkpoint": "yyfz233/Pi3X",
        "width": target_w,
        "height": target_h,
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

    print(f"[pi3] Wrote {cam_path}")
    print(f"[pi3] Wrote {N} depth maps to {depth_dir}")
    print(f"[pi3] Wrote {N} pointmaps to {pointmap_dir}")
    print(f"[pi3] Wrote scene pointmap manifest to {scene_manifest}")


if __name__ == "__main__":
    main()
