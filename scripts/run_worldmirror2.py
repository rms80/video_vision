"""Run HunyuanWorld-Mirror 2.0 for camera poses + depth + global pointmap.

Usage:
    python run_worldmirror2.py <scene_dir> [--subsample N] [--target-size 518]

Reads <scene_dir>/frames/NNNNNN.jpg (produced by extract_frames.py).
Runs WorldMirror 2.0 feed-forward inference in image-only mode (no conditioning)
to produce camera poses, depth, per-frame pointmaps, and a global scene-level
point cloud.

HunyuanWorld-Mirror 2.0 is the 1.2B-parameter successor to WorldMirror 1.0,
released inside the HY-World-2.0 umbrella repo (HF subfolder
`HY-WorldMirror-2.0`). It exposes a diffusers-style Pipeline API; we use its
`_run_inference` directly to get raw predictions without their file-saving
layer. The gaussian-splat head is disabled via `disable_heads=['gs']` to save
~33 M params and avoid needing a working `gsplat` build.

Outputs:
    <scene_dir>/worldmirror2/cameras.json         — same schema as other plugins
    <scene_dir>/worldmirror2/depth/NNNNNN.npz     — per-frame depth
    <scene_dir>/worldmirror2/pointmap/NNNNNN.npz  — per-frame camera-space pointmap + conf
    <scene_dir>/worldmirror2/scene_pointmap_chunks.json — global cloud manifest
    <scene_dir>/worldmirror2/scene_pointmap_NNN.npz     — chunked global cloud
"""

import sys
import os
import json
import argparse
import glob
import time
import types
from pathlib import Path

from _pointcloud_io import save_chunked_pointcloud
from _progress import progress

# Make the HY-World-2.0 checkout importable.
HYWORLD2_ROOT = Path(__file__).resolve().parent.parent / "models" / "external" / "hy-world-2.0"
sys.path.insert(0, str(HYWORLD2_ROOT))

# Stub `gsplat` (CUDA-compiled; only used by GaussianSplatRenderer which we
# disable via disable_heads=['gs']) and `flash_attn` (only used when the
# attention modules run with fused_attn=False, which none do by default).
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

if "flash_attn" not in sys.modules:
    import torch as _torch
    import torch.nn.functional as _F
    _fa = types.ModuleType("flash_attn"); _fa.__path__ = []
    _fai = types.ModuleType("flash_attn.flash_attn_interface")

    def _flash_attn_sdpa_shim(q, k, v, dropout_p=0.0, **kw):
        # flash_attn layout: (B, seqlen, num_heads, head_dim)
        # SDPA layout:       (B, num_heads, seqlen, head_dim)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        out = _F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        return out.transpose(1, 2)

    _fai.flash_attn_func = _flash_attn_sdpa_shim
    sys.modules["flash_attn"] = _fa
    sys.modules["flash_attn.flash_attn_interface"] = _fai

import cv2
import numpy as np
import torch
import torch.nn.functional as F


CONF_PERCENTILE = 10.0
EDGE_NORMAL_DEG = 5.0
EDGE_DEPTH_RTOL = 0.03


def depth_edge(depth, rtol=0.03, kernel_size=3):
    shape = depth.shape
    d = depth.reshape(-1, 1, *shape[-2:]).float()
    pad = kernel_size // 2
    diff = (F.max_pool2d(d, kernel_size, stride=1, padding=pad)
            + F.max_pool2d(-d, kernel_size, stride=1, padding=pad))
    edge = (diff / d.clamp_min(1e-6)).nan_to_num_() > rtol
    return edge.reshape(*shape)


def normals_edge(normals, tol_deg=5.0, kernel_size=3):
    """normals: (H, W, 3) torch tensor; returns (H, W) bool edge mask."""
    n = normals.permute(2, 0, 1).unsqueeze(0).float()
    n = F.normalize(n, dim=1)
    pad = kernel_size // 2
    cos_tol = float(np.cos(np.deg2rad(tol_deg)))
    H, W = n.shape[-2:]
    edge = torch.zeros(H, W, dtype=torch.bool, device=n.device)
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = torch.roll(n, shifts=(dy, dx), dims=(2, 3))
            dot = (n * shifted).sum(dim=1).squeeze(0)
            edge |= dot < cos_tol
    return edge


def main():
    parser = argparse.ArgumentParser(description="Run HunyuanWorld-Mirror 2.0 on video frames")
    parser.add_argument("scene_dir", help="Path to _scene directory")
    parser.add_argument("--subsample", type=int, default=3,
                        help="Use every Nth frame (default 3)")
    parser.add_argument("--target-size", type=int, default=518,
                        help="Target image size for model (default 518; pipeline default is 952 but that needs a lot of VRAM)")
    args = parser.parse_args()

    scene_dir = args.scene_dir
    frames_dir = os.path.join(scene_dir, "frames")
    out_dir = os.path.join(scene_dir, "worldmirror2")
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
        print("[worldmirror2] ERROR: no frames found", file=sys.stderr)
        sys.exit(1)

    frames_to_use = all_frames[::args.subsample]
    frame_indices = [int(os.path.splitext(os.path.basename(f))[0]) for f in frames_to_use]
    N = len(frames_to_use)
    print(f"[worldmirror2] {N} frames (subsample={args.subsample}) at {src_w}x{src_h}")

    # Load pipeline — will download weights (~2.4 GB) on first use.
    from hyworld2.worldrecon.pipeline import WorldMirrorPipeline  # noqa: E402
    progress("Loading WorldMirror 2.0 pipeline (may download weights)...")
    print("[worldmirror2] Loading WorldMirror 2.0 pipeline...")
    pipeline = WorldMirrorPipeline.from_pretrained(
        "tencent/HY-World-2.0",
        disable_heads=["gs"],
    )

    # Run inference using the pipeline's internal helper (skips their file I/O).
    progress(f"Running WorldMirror 2.0 inference (target_size={args.target_size})...")
    print(f"[worldmirror2] Running inference at target_size={args.target_size}...")
    t0 = time.time()
    # `_run_inference` doesn't wrap itself in no_grad, so do it here (the
    # pipeline's `__call__` does this via a decorator on the public entry point).
    with torch.no_grad():
        predictions, imgs, infer_time = pipeline._run_inference(
            frames_to_use, args.target_size, None, None
        )
    elapsed = time.time() - t0
    progress(f"Inference done in {elapsed:.1f}s ({elapsed / N:.2f}s/frame)")
    print(f"[worldmirror2] Inference done in {elapsed:.1f}s ({elapsed / N:.2f}s/frame)")

    B, S, C, work_H, work_W = imgs.shape
    print(f"[worldmirror2] Working resolution: {work_W}x{work_H}")

    # Extract tensors (drop batch dim; move to CPU float32)
    pts3d_world = predictions["pts3d"][0].detach().cpu().float()      # (S, H, W, 3)
    pts3d_conf = predictions["pts3d_conf"][0].detach().cpu().float()  # (S, H, W)
    depth = predictions["depth"][0, :, :, :, 0].detach().cpu().float()  # (S, H, W)
    normals = predictions["normals"][0].detach().cpu().float()       # (S, H, W, 3)
    cam_poses = predictions["camera_poses"][0].detach().cpu().float()  # (S, 4, 4) c2w
    cam_intrs = predictions["camera_intrs"][0].detach().cpu().float()  # (S, 3, 3)

    # Keep processed RGB for scene-pointmap coloring.
    rgb_np = (imgs[0].permute(0, 2, 3, 1).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

    del pipeline, predictions, imgs
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
    print(f"[worldmirror2] Median intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    scale_factor = ((work_W / src_w) + (work_H / src_h)) / 2

    # ---- Build filter mask (confidence percentile + depth/normal edges) ----
    progress("Computing filter masks (depth + normal edges)...")
    print("[worldmirror2] Computing filter masks...")
    valid_masks = []
    for i in range(N):
        conf_i = pts3d_conf[i].numpy()
        thresh = float(np.quantile(conf_i, CONF_PERCENTILE / 100.0))
        conf_mask = conf_i >= thresh

        d_edge = depth_edge(depth[i:i+1], rtol=EDGE_DEPTH_RTOL).squeeze(0).numpy()
        n_edge = normals_edge(normals[i], tol_deg=EDGE_NORMAL_DEG).numpy()
        edge_mask = ~(d_edge & n_edge)

        valid_masks.append(conf_mask & edge_mask)
    valid_mask = np.stack(valid_masks, axis=0)

    # ---- Write per-frame outputs ----
    progress(f"Writing {N} depth maps + pointmaps...")
    frames_out = []
    all_world_pts = []
    all_world_rgb = []
    all_world_conf = []

    for i in range(N):
        idx = frame_indices[i]
        c2w = cam_poses[i].numpy().astype(np.float64)

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

        depth_np = depth[i].numpy()
        np.savez_compressed(
            os.path.join(depth_dir, f"{idx:06d}.npz"),
            depth=depth_np.astype(np.float16),
        )

        # Per-frame pointmap in camera space.
        pts_w = pts3d_world[i].numpy().astype(np.float64)
        pts_cam = pts_w @ R_cw.T + t_cw
        np.savez_compressed(
            os.path.join(pointmap_dir, f"{idx:06d}.npz"),
            pts3d=pts_cam.astype(np.float16),
            conf=pts3d_conf[i].numpy().astype(np.float16),
        )

        m = valid_mask[i]
        all_world_pts.append(pts_w[m])
        all_world_rgb.append(rgb_np[i][m])
        all_world_conf.append(pts3d_conf[i].numpy()[m])

    scene_pts = np.concatenate(all_world_pts, axis=0).astype(np.float32)
    scene_rgb = np.concatenate(all_world_rgb, axis=0)
    scene_conf = np.concatenate(all_world_conf, axis=0).astype(np.float32)

    scene_pts[:, 1] *= -1
    scene_pts[:, 2] *= -1

    save_chunked_pointcloud(out_dir, "scene_pointmap", scene_pts, scene_rgb, scene_conf)
    print(f"[worldmirror2] Scene pointmap: {scene_pts.shape[0]:,} points")

    cameras = {
        "model": "HUNYUAN_WORLD_MIRROR_2",
        "checkpoint": "tencent/HY-World-2.0",
        "subfolder": "HY-WorldMirror-2.0",
        "width": work_W,
        "height": work_H,
        "source_width": src_w,
        "source_height": src_h,
        "scale_factor": scale_factor,
        "subsample_every": args.subsample,
        "target_size": args.target_size,
        "K": K_out,
        "k1": 0.0,
        "num_points": 0,
        "num_registered": N,
        "frames": frames_out,
    }

    cam_path = os.path.join(out_dir, "cameras.json")
    with open(cam_path, "w") as f:
        json.dump(cameras, f, indent=2)

    print(f"[worldmirror2] Wrote {cam_path}")
    print(f"[worldmirror2] Wrote {N} depth maps to {depth_dir}")
    print(f"[worldmirror2] Wrote {N} pointmaps to {pointmap_dir}")
    print(f"[worldmirror2] Wrote scene pointmap chunks to {out_dir}")


if __name__ == "__main__":
    main()
