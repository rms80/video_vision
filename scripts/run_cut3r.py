"""Run CUT3R to get camera poses + depth maps from video frames.

Usage:
    python run_cut3r.py <scene_dir> [--size 512] [--subsample N]

Reads <scene_dir>/frames/NNNNNN.jpg (produced by extract_frames.py).
Runs CUT3R feed-forward inference to produce both camera poses and metric
depth in a single pass — replacing COLMAP + DepthAnythingV2.

Outputs:
    <scene_dir>/cut3r/cameras.json   — same schema as colmap/cameras.json
    <scene_dir>/cut3r/depth/NNNNNN.npz — per-frame depth (same schema as depthanythingv2/)
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
CUT3R_ROOT = os.path.join(REPO_ROOT, "models", "external", "cut3r")
CUT3R_SRC = os.path.join(CUT3R_ROOT, "src")
CUT3R_CKPT = os.path.join(CUT3R_SRC, "cut3r_512_dpt_4_64.pth")


def main():
    parser = argparse.ArgumentParser(description="Run CUT3R on video frames")
    parser.add_argument("scene_dir", help="Path to _scene directory")
    parser.add_argument("--size", type=int, default=512,
                        help="CUT3R working resolution (default 512)")
    parser.add_argument("--subsample", type=int, default=2,
                        help="Use every Nth frame (default 2)")
    args = parser.parse_args()

    scene_dir = args.scene_dir
    frames_dir = os.path.join(scene_dir, "frames")
    out_dir = os.path.join(scene_dir, "cut3r")
    depth_dir = os.path.join(out_dir, "depth")
    pointmap_dir = os.path.join(out_dir, "pointmap")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(pointmap_dir, exist_ok=True)

    # Load frames metadata
    frames_json = os.path.join(scene_dir, "frames.json")
    with open(frames_json) as f:
        meta = json.load(f)
    src_w, src_h = meta["width"], meta["height"]
    frame_count = meta["frame_count"]

    # Collect frame paths
    all_frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not all_frames:
        print("[cut3r] ERROR: no frames found", file=sys.stderr)
        sys.exit(1)

    # Subsample
    subsample = args.subsample
    frames_to_use = all_frames[::subsample]
    frame_indices = [int(os.path.splitext(os.path.basename(f))[0]) for f in frames_to_use]
    print(f"[cut3r] {len(frames_to_use)} frames (subsample={subsample}) at {src_w}x{src_h}")

    # Set up CUT3R imports
    sys.path.insert(0, CUT3R_ROOT)
    sys.path.insert(0, CUT3R_SRC)
    sys.path.insert(0, os.path.join(CUT3R_SRC, "croco"))

    from add_ckpt_path import add_path_to_dust3r
    add_path_to_dust3r(CUT3R_CKPT)

    from src.dust3r.inference import inference, inference_recurrent
    from src.dust3r.model import ARCroco3DStereo
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.utils.image import load_images

    # Load model
    device = pick_device()
    progress(f"Loading CUT3R on {device}...")
    print(f"[cut3r] Loading model on {device}...")
    model = ARCroco3DStereo.from_pretrained(CUT3R_CKPT).to(device)
    model.eval()

    # Prepare input views
    progress(f"Preprocessing {len(frames_to_use)} frames...")
    print("[cut3r] Preparing input views...")
    images = load_images(frames_to_use, size=args.size, square_ok=True)
    views = []
    for i in range(len(images)):
        view = {
            "img": images[i]["img"],
            "ray_map": torch.full(
                (images[i]["img"].shape[0], 6,
                 images[i]["img"].shape[-2], images[i]["img"].shape[-1]),
                torch.nan,
            ),
            "true_shape": torch.from_numpy(images[i]["true_shape"]),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.from_numpy(
                np.eye(4, dtype=np.float32)
            ).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        }
        views.append(view)

    # Run inference
    progress(f"Running CUT3R inference on {len(views)} frames...")
    print("[cut3r] Running inference...")
    t0 = time.time()
    outputs, _ = inference_recurrent(views, model, device)
    elapsed = time.time() - t0
    progress(f"Inference done in {elapsed:.1f}s ({elapsed/len(views):.2f}s/frame)")
    print(f"[cut3r] Inference done in {elapsed:.1f}s "
          f"({elapsed/len(views):.2f}s/frame)")

    # Extract outputs
    pts3ds_self = torch.cat(
        [o["pts3d_in_self_view"].cpu() for o in outputs["pred"]], 0
    )  # (B, H, W, 3)
    conf_self = torch.cat(
        [o["conf_self"].cpu() for o in outputs["pred"]], 0
    )  # (B, H, W)

    # Camera poses (cam-to-world 4x4)
    pr_poses = [
        pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
        for pred in outputs["pred"]
    ]
    cam2worlds = torch.cat(pr_poses, 0)  # (B, 4, 4)

    # Depth = Z component of self-view pointmap
    depths = pts3ds_self[..., 2]  # (B, H, W)

    # Estimate focal lengths
    B, H, W, _ = pts3ds_self.shape
    pp = torch.tensor([W // 2, H // 2], dtype=torch.float32).repeat(B, 1)
    focals = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

    # Use median focal as shared intrinsic
    median_focal = float(focals.median())
    cx, cy = W / 2.0, H / 2.0
    K = [[median_focal, 0.0, cx], [0.0, median_focal, cy], [0.0, 0.0, 1.0]]
    print(f"[cut3r] Working resolution: {W}x{H}, median focal: {median_focal:.1f}")

    # Scale factor from source resolution to CUT3R working resolution
    scale_x = W / src_w
    scale_y = H / src_h
    scale = (scale_x + scale_y) / 2  # approximate, CUT3R may not preserve aspect ratio

    # Convert to cameras.json format
    progress(f"Writing {B} depth maps + pointmaps...")
    frames_out = []
    for i in range(B):
        idx = frame_indices[i]
        c2w = cam2worlds[i].numpy().astype(np.float64)  # 4x4 cam-to-world

        # Convert to camera-from-world (R_cw, t_cw) for cameras.json
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

        # Save depth as .npz (same format as depthanythingv2)
        depth_np = depths[i].numpy()
        npz_path = os.path.join(depth_dir, f"{idx:06d}.npz")
        np.savez_compressed(npz_path, depth=depth_np.astype(np.float16))

        # Save full pointmap + confidence
        pts_np = pts3ds_self[i].numpy()
        conf_np = conf_self[i].numpy()
        pm_path = os.path.join(pointmap_dir, f"{idx:06d}.npz")
        np.savez_compressed(pm_path, pts3d=pts_np.astype(np.float16), conf=conf_np.astype(np.float16))

    cameras = {
        "model": "CUT3R",
        "width": W,
        "height": H,
        "source_width": src_w,
        "source_height": src_h,
        "scale_factor": scale,
        "subsample_every": subsample,
        "K": K,
        "k1": 0.0,
        "num_points": 0,
        "num_registered": B,
        "frames": frames_out,
    }

    cam_path = os.path.join(out_dir, "cameras.json")
    with open(cam_path, "w") as f:
        json.dump(cameras, f, indent=2)
    print(f"[cut3r] Wrote {cam_path}")
    print(f"[cut3r] Wrote {B} depth maps to {depth_dir}")
    print(f"[cut3r] Wrote {B} pointmaps to {pointmap_dir}")


if __name__ == "__main__":
    main()
