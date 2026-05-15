"""Run HunyuanWorld-Mirror for camera poses + depth + global pointmap.

Usage:
    python run_worldmirror.py <scene_dir> [--subsample N] [--target-size 518]

Reads <scene_dir>/frames/NNNNNN.jpg (produced by extract_frames.py).
Runs WorldMirror feed-forward inference in image-only mode (no conditioning)
to produce camera poses, depth, per-frame pointmaps, and a global scene-level
point cloud.

HunyuanWorld-Mirror is Tencent's universal 3D reconstruction model. It accepts
images plus optional priors (pose/intrinsics/depth) and outputs world-space
pointmaps, per-view depth, surface normals, camera poses/intrinsics, and 3D
Gaussians. This plugin uses only the pointmap/depth/pose outputs; the
gaussian-splat head is skipped to keep the dep footprint small.

Outputs:
    <scene_dir>/worldmirror/cameras.json         — same schema as other plugins
    <scene_dir>/worldmirror/depth/NNNNNN.npz     — per-frame depth
    <scene_dir>/worldmirror/pointmap/NNNNNN.npz  — per-frame camera-space pointmap + conf
    <scene_dir>/worldmirror/scene_pointmap_chunks.json — global cloud manifest
    <scene_dir>/worldmirror/scene_pointmap_NNN.npz     — chunked global cloud
"""

import sys
import os
import json
import argparse
import glob
import time
import types
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pointcloud_io import save_chunked_pointcloud  # noqa: E402
from _device import pick_device  # noqa: E402
from _progress import progress  # noqa: E402

# Make the HunyuanWorld-Mirror checkout importable.
WORLDMIRROR_ROOT = Path(__file__).resolve().parent.parent / "models" / "external" / "hunyuanworld-mirror"
sys.path.insert(0, str(WORLDMIRROR_ROOT))

# `gsplat` is a CUDA-compiled package used only by the gaussian-splatting head
# (`GaussianSplatRenderer`). We don't invoke that head, so stub it out to avoid
# needing an nvcc build on Windows.
if "gsplat" not in sys.modules:
    _gs = types.ModuleType("gsplat"); _gs.__path__ = []
    _gs_r = types.ModuleType("gsplat.rendering")
    _gs_s = types.ModuleType("gsplat.strategy")
    def _rast_stub(*a, **kw):
        raise NotImplementedError("gsplat is stubbed; rasterization not available")
    class _DefaultStrategyStub: ...
    _gs_r.rasterization = _rast_stub
    _gs_s.DefaultStrategy = _DefaultStrategyStub
    sys.modules["gsplat"] = _gs
    sys.modules["gsplat.rendering"] = _gs_r
    sys.modules["gsplat.strategy"] = _gs_s

import cv2
import numpy as np
import torch
import torch.nn.functional as F


CONF_PERCENTILE = 10.0
EDGE_NORMAL_DEG = 5.0
EDGE_DEPTH_RTOL = 0.03


def depth_edge(depth, rtol=0.03, kernel_size=3):
    """Detect depth discontinuities (tensor version, same shape in/out)."""
    shape = depth.shape
    d = depth.reshape(-1, 1, *shape[-2:]).float()
    pad = kernel_size // 2
    diff = (F.max_pool2d(d, kernel_size, stride=1, padding=pad)
            + F.max_pool2d(-d, kernel_size, stride=1, padding=pad))
    edge = (diff / d.clamp_min(1e-6)).nan_to_num_() > rtol
    return edge.reshape(*shape)


def normals_edge(normals, tol_deg=5.0, kernel_size=3):
    """Detect normal-direction discontinuities. normals: (H, W, 3) torch tensor."""
    # Convert to (1, 3, H, W)
    n = normals.permute(2, 0, 1).unsqueeze(0).float()
    n = F.normalize(n, dim=1)
    pad = kernel_size // 2
    # For each neighbour, dot with center; if min dot < cos(tol), it's an edge.
    # Quick approximation: use max-pool on 1 - dot with 0.5 offsets across shifts.
    cos_tol = float(np.cos(np.deg2rad(tol_deg)))
    H, W = n.shape[-2:]
    edge = torch.zeros(H, W, dtype=torch.bool, device=n.device)
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = torch.roll(n, shifts=(dy, dx), dims=(2, 3))
            dot = (n * shifted).sum(dim=1).squeeze(0)  # (H, W)
            edge |= dot < cos_tol
    return edge


def main():
    parser = argparse.ArgumentParser(description="Run HunyuanWorld-Mirror on video frames")
    parser.add_argument("scene_dir", help="Path to _scene directory")
    parser.add_argument("--subsample", type=int, default=3,
                        help="Use every Nth frame (default 3)")
    parser.add_argument("--target-size", type=int, default=518,
                        help="Target image size for model (default 518)")
    args = parser.parse_args()

    scene_dir = args.scene_dir
    frames_dir = os.path.join(scene_dir, "frames")
    out_dir = os.path.join(scene_dir, "worldmirror")
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
        print("[worldmirror] ERROR: no frames found", file=sys.stderr)
        sys.exit(1)

    frames_to_use = all_frames[::args.subsample]
    frame_indices = [int(os.path.splitext(os.path.basename(f))[0]) for f in frames_to_use]
    N = len(frames_to_use)
    print(f"[worldmirror] {N} frames (subsample={args.subsample}) at {src_w}x{src_h}")

    # Load + preprocess images using WorldMirror's own utility (crop to target_size
    # on the short-axis workflow: width -> target_size, height rounded to /14 then
    # center-cropped to target_size if larger).
    progress(f"Preprocessing {N} frames...")
    from src.utils.inference_utils import prepare_images_to_tensor  # noqa: E402
    imgs = prepare_images_to_tensor(
        frames_to_use,
        resize_strategy="crop",
        target_size=args.target_size,
    )  # (1, N, 3, H, W) in [0, 1]
    B, S, C, work_H, work_W = imgs.shape
    print(f"[worldmirror] Working resolution: {work_W}x{work_H}")

    # Load model
    device = pick_device()
    print(f"[worldmirror] device={device}", flush=True)
    progress(f"Loading WorldMirror on {device}...")
    print(f"[worldmirror] Loading WorldMirror on {device}...")
    from src.models.models.worldmirror import WorldMirror  # noqa: E402
    model = WorldMirror.from_pretrained("tencent/HunyuanWorld-Mirror").to(device).eval()

    # Run inference (image-only; no priors)
    views = {"img": imgs.to(device)}
    cond_flags = [0, 0, 0]

    use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    progress(f"Running WorldMirror inference on {N} frames...")
    print(f"[worldmirror] Running inference (amp={amp_dtype})...")
    t0 = time.time()
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=bool(use_amp), dtype=amp_dtype):
            predictions = model(views=views, cond_flags=cond_flags)
    elapsed = time.time() - t0
    progress(f"Inference done in {elapsed:.1f}s ({elapsed / N:.2f}s/frame)")
    print(f"[worldmirror] Inference done in {elapsed:.1f}s ({elapsed / N:.2f}s/frame)")

    # Extract tensors (drop batch dim, move to CPU float32 for post-processing)
    pts3d_world = predictions["pts3d"][0].detach().cpu().float()      # (S, H, W, 3)
    pts3d_conf = predictions["pts3d_conf"][0].detach().cpu().float()  # (S, H, W)
    depth = predictions["depth"][0, :, :, :, 0].detach().cpu().float()  # (S, H, W)
    normals = predictions["normals"][0].detach().cpu().float()       # (S, H, W, 3)
    cam_poses = predictions["camera_poses"][0].detach().cpu().float()  # (S, 4, 4) c2w
    cam_intrs = predictions["camera_intrs"][0].detach().cpu().float()  # (S, 3, 3)

    # Keep processed RGB for scene-pointmap coloring (same resolution as predictions).
    rgb_np = (imgs[0].permute(0, 2, 3, 1).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()  # (S, H, W, 3)

    del model, predictions, views
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Median intrinsics across views
    K_stack = cam_intrs.numpy().astype(np.float64)
    K_median = np.median(K_stack, axis=0)
    fx = float(K_median[0, 0])
    fy = float(K_median[1, 1])
    cx = float(K_median[0, 2])
    cy = float(K_median[1, 2])
    K_out = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    print(f"[worldmirror] Median intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    scale_factor = ((work_W / src_w) + (work_H / src_h)) / 2

    # ---- Build filter mask (confidence percentile + depth/normal edges) ----
    progress("Computing filter masks (depth + normal edges)...")
    print("[worldmirror] Computing filter masks...")
    valid_masks = []
    for i in range(N):
        conf_i = pts3d_conf[i].numpy()
        thresh = float(np.quantile(conf_i, CONF_PERCENTILE / 100.0))
        conf_mask = conf_i >= thresh

        d_edge = depth_edge(depth[i:i+1], rtol=EDGE_DEPTH_RTOL).squeeze(0).numpy()
        n_edge = normals_edge(normals[i], tol_deg=EDGE_NORMAL_DEG).numpy()
        edge_mask = ~(d_edge & n_edge)

        valid_masks.append(conf_mask & edge_mask)
    valid_mask = np.stack(valid_masks, axis=0)  # (S, H, W) bool

    # ---- Write per-frame outputs ----
    progress(f"Writing {N} depth maps + pointmaps...")
    frames_out = []
    all_world_pts = []
    all_world_rgb = []
    all_world_conf = []

    for i in range(N):
        idx = frame_indices[i]
        c2w = cam_poses[i].numpy().astype(np.float64)  # (4, 4)

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

        # Depth (Z-depth in camera frame, already metric-ish from model)
        depth_np = depth[i].numpy()
        np.savez_compressed(
            os.path.join(depth_dir, f"{idx:06d}.npz"),
            depth=depth_np.astype(np.float16),
        )

        # Per-frame pointmap stored in CAMERA space (match Pi3X/MapAnything
        # convention). Transform world points through w2c: X_cam = R_cw @ X_w + t_cw.
        pts_w = pts3d_world[i].numpy().astype(np.float64)  # (H, W, 3)
        pts_cam = pts_w @ R_cw.T + t_cw  # broadcasting over (H, W, 3)
        np.savez_compressed(
            os.path.join(pointmap_dir, f"{idx:06d}.npz"),
            pts3d=pts_cam.astype(np.float16),
            conf=pts3d_conf[i].numpy().astype(np.float16),
        )

        # Collect valid world-space points for scene pointmap
        m = valid_mask[i]
        all_world_pts.append(pts_w[m])
        all_world_rgb.append(rgb_np[i][m])
        all_world_conf.append(pts3d_conf[i].numpy()[m])

    # ---- Write global scene pointmap ----
    scene_pts = np.concatenate(all_world_pts, axis=0).astype(np.float32)
    scene_rgb = np.concatenate(all_world_rgb, axis=0)
    scene_conf = np.concatenate(all_world_conf, axis=0).astype(np.float32)

    # Convert from OpenCV world coords to Three.js convention: (x, -y, -z)
    scene_pts[:, 1] *= -1
    scene_pts[:, 2] *= -1

    save_chunked_pointcloud(out_dir, "scene_pointmap", scene_pts, scene_rgb, scene_conf)
    print(f"[worldmirror] Scene pointmap: {scene_pts.shape[0]:,} points")

    # ---- Write cameras.json ----
    cameras = {
        "model": "HUNYUAN_WORLD_MIRROR",
        "checkpoint": "tencent/HunyuanWorld-Mirror",
        "width": work_W,
        "height": work_H,
        "source_width": src_w,
        "source_height": src_h,
        "scale_factor": scale_factor,
        "subsample_every": args.subsample,
        "K": K_out,
        "k1": 0.0,
        "num_points": 0,
        "num_registered": N,
        "frames": frames_out,
    }

    cam_path = os.path.join(out_dir, "cameras.json")
    with open(cam_path, "w") as f:
        json.dump(cameras, f, indent=2)

    print(f"[worldmirror] Wrote {cam_path}")
    print(f"[worldmirror] Wrote {N} depth maps to {depth_dir}")
    print(f"[worldmirror] Wrote {N} pointmaps to {pointmap_dir}")
    print(f"[worldmirror] Wrote scene pointmap chunks to {out_dir}")


if __name__ == "__main__":
    main()
