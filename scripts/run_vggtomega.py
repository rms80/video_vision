"""Run VGGT-Omega to get camera poses + depth + pointmaps from video frames.

Usage:
    python run_vggtomega.py <scene_dir> [--num-frames N] [--ckpt-file FILE]

Single-pass strategy: pick exactly N evenly-spaced frames across the
clip and run one VGGT-Omega forward — the model is designed to scale to
hundreds of frames in a single pass with global consistency, so no
windowing/stitching is needed.

Outputs:
    <scene_dir>/vggtomega/cameras.json
    <scene_dir>/vggtomega/depth/NNNNNN.npz
    <scene_dir>/vggtomega/pointmap/NNNNNN.npz
"""

import sys
import os
import json
import argparse
import glob
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _device import pick_device  # noqa: E402
from _progress import progress  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
VGGT_OMEGA_ROOT = os.path.join(REPO_ROOT, "models", "external", "vggt-omega")


def depth_to_cam_points(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Back-project a depth map (H, W) into camera-space XYZ points (H, W, 3)."""
    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32))
    z = depth.astype(np.float32)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VGGT-Omega on video frames")
    parser.add_argument("scene_dir", help="Path to _scene directory")
    parser.add_argument("--num-frames", type=int, default=50,
                        help="Target total frames to process. Picks exactly N "
                             "evenly-spaced frames from 0 to last (inclusive). "
                             "Default 50.")
    parser.add_argument("--image-resolution", type=int, default=512,
                        help="VGGT-Omega working resolution (default 512)")
    parser.add_argument("--ckpt-file", default="vggt_omega_1b_512.pt",
                        help="Checkpoint filename within the facebook/VGGT-Omega HF repo")
    parser.add_argument("--hf-repo", default="facebook/VGGT-Omega",
                        help="HuggingFace repo id holding the checkpoint")
    args = parser.parse_args()

    scene_dir = args.scene_dir
    frames_dir = os.path.join(scene_dir, "frames")
    out_dir = os.path.join(scene_dir, "vggtomega")
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
        print("[vggtomega] ERROR: no frames found", file=sys.stderr)
        sys.exit(1)

    frame_indices = [int(os.path.splitext(os.path.basename(f))[0]) for f in all_frames]
    N = len(all_frames)
    print(f"[vggtomega] {N} frames at {src_w}x{src_h}")

    T = max(1, min(args.num_frames, N))
    if T <= 1:
        positions = [0]
    else:
        positions = sorted({round(i * (N - 1) / (T - 1)) for i in range(T)})
    selected_paths = [all_frames[p] for p in positions]
    selected_indices = [frame_indices[p] for p in positions]
    print(f"[vggtomega] processing {len(positions)} evenly-spaced frames")

    sys.path.insert(0, VGGT_OMEGA_ROOT)
    from vggt_omega.models import VGGTOmega  # noqa: E402
    from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402
    from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402
    from huggingface_hub import hf_hub_download  # noqa: E402

    device = pick_device()
    progress(f"Loading VGGT-Omega on {device}...")
    print(f"[vggtomega] resolving checkpoint {args.ckpt_file} from {args.hf_repo}")
    ckpt_path = hf_hub_download(repo_id=args.hf_repo, filename=args.ckpt_file)
    print(f"[vggtomega] checkpoint: {ckpt_path}")

    model = VGGTOmega().to(device).eval()
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    del state

    progress(f"Loading and preprocessing {len(selected_paths)} images...")
    images = load_and_preprocess_images(
        selected_paths, image_resolution=args.image_resolution
    ).to(device)
    # load_and_preprocess_images returns (N, 3, H, W); model auto-batches.
    work_H, work_W = int(images.shape[-2]), int(images.shape[-1])
    print(f"[vggtomega] working resolution: {work_W}x{work_H}")

    progress(f"VGGT-Omega forward pass over {len(selected_paths)} frames...")
    t0 = time.time()
    with torch.inference_mode():
        preds = model(images)
    elapsed = time.time() - t0
    print(f"[vggtomega] forward {elapsed:.1f}s "
          f"({elapsed/len(selected_paths):.2f}s/frame)")

    extr_t, intr_t = encoding_to_camera(
        preds["pose_enc"], preds["images"].shape[-2:]
    )
    # Squeeze the auto-added batch dim. Shapes after squeeze:
    #   extr (Nf, 3, 4), intr (Nf, 3, 3)
    extr = extr_t.squeeze(0).cpu().float().numpy()
    intr = intr_t.squeeze(0).cpu().float().numpy()

    depth_t = preds["depth"]
    depth_conf_t = preds["depth_conf"]
    # depth comes back as (1, Nf, H, W) or (1, Nf, H, W, 1) depending on
    # the dense head — collapse any trailing singleton.
    if depth_t.dim() == 5 and depth_t.shape[-1] == 1:
        depth_t = depth_t.squeeze(-1)
    dmap = depth_t.squeeze(0).cpu().float().numpy()
    dconf = depth_conf_t.squeeze(0).cpu().float().numpy()

    del preds, depth_t, depth_conf_t, extr_t, intr_t, images, model
    if device == "cuda":
        torch.cuda.empty_cache()

    # Output median intrinsics across emitted frames (matches VGGT's
    # convention so downstream align_scene/run_boxer get a single K).
    median_K = np.median(intr, axis=0)
    fx, fy = float(median_K[0, 0]), float(median_K[1, 1])
    cx, cy = float(median_K[0, 2]), float(median_K[1, 2])
    K_out = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    print(f"[vggtomega] Median intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    scale_factor = ((work_W / src_w) + (work_H / src_h)) / 2

    progress(f"Writing {len(positions)} depth maps + pointmaps...")
    frames_out = []
    for i, (pos, idx) in enumerate(zip(positions, selected_indices)):
        E = extr[i]
        frames_out.append({
            "idx": idx,
            "name": f"{idx:06d}.jpg",
            "registered": True,
            "R": E[:3, :3].tolist(),
            "t": E[:3, 3].tolist(),
            "sparse_obs": [],
        })

        depth_i = dmap[i]
        np.savez_compressed(
            os.path.join(depth_dir, f"{idx:06d}.npz"),
            depth=depth_i.astype(np.float16),
        )

        pts_cam = depth_to_cam_points(depth_i, intr[i])
        np.savez_compressed(
            os.path.join(pointmap_dir, f"{idx:06d}.npz"),
            pts3d=pts_cam.astype(np.float16),
            conf=dconf[i].astype(np.float16),
        )

    cameras = {
        "model": "VGGT-Omega",
        "width": work_W,
        "height": work_H,
        "source_width": src_w,
        "source_height": src_h,
        "scale_factor": scale_factor,
        "num_frames": args.num_frames,
        "image_resolution": args.image_resolution,
        "K": K_out,
        "k1": 0.0,
        "num_points": 0,
        "num_registered": len(positions),
        "frames": frames_out,
    }

    cam_path = os.path.join(out_dir, "cameras.json")
    with open(cam_path, "w") as f:
        json.dump(cameras, f, indent=2)
    print(f"[vggtomega] Wrote {cam_path}")
    print(f"[vggtomega] Wrote {len(positions)} depth maps to {depth_dir}")
    print(f"[vggtomega] Wrote {len(positions)} pointmaps to {pointmap_dir}")


if __name__ == "__main__":
    main()
