"""Run WildDet3D as a scene analysis method — depth + predicted K per frame.

Unlike run_cut3r.py / run_colmap.py, WildDet3D does NOT produce camera
extrinsics (no cross-frame pose solve). We fake it by putting every camera
at the origin. The useful signal is:
    - per-frame metric depth map
    - per-frame predicted camera intrinsics (K_pred)

Usage:
    python run_wilddet3d_scene.py <scene_dir> [--subsample 10]

Reads:
    <scene_dir>/frames.json
    <scene_dir>/frames/*.jpg

Writes:
    <scene_dir>/wilddet3d/cameras.json — same schema as cut3r/cameras.json
    <scene_dir>/wilddet3d/depth/{idx:06d}.npz
"""

import sys
import os
import json
import glob
import argparse
import contextlib

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _device import pick_device  # noqa: E402
from _progress import progress  # noqa: E402

WILDDET3D_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "external", "wilddet3d")
)
sys.path.insert(0, WILDDET3D_ROOT)
sys.path.insert(0, os.path.join(WILDDET3D_ROOT, "third_party", "sam3"))
sys.path.insert(0, os.path.join(WILDDET3D_ROOT, "third_party", "lingbot_depth"))

from wilddet3d.inference import build_model
from wilddet3d.preprocessing import preprocess


def main():
    parser = argparse.ArgumentParser(description="WildDet3D scene analysis (depth + K)")
    parser.add_argument("scene_dir", help="Scene directory with frames/ and frames.json")
    parser.add_argument("--subsample", type=int, default=10,
                        help="Run on every Nth frame")
    args = parser.parse_args()

    scene_dir = args.scene_dir
    frames_dir = os.path.join(scene_dir, "frames")
    out_dir = os.path.join(scene_dir, "wilddet3d")
    depth_dir = os.path.join(out_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)

    with open(os.path.join(scene_dir, "frames.json")) as f:
        meta = json.load(f)
    src_w, src_h = meta["width"], meta["height"]

    all_frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not all_frames:
        print("[wilddet3d-scene] ERROR: no frames found", file=sys.stderr)
        sys.exit(1)

    subsample = args.subsample
    frames_to_use = all_frames[::subsample]
    frame_indices = [int(os.path.splitext(os.path.basename(f))[0]) for f in frames_to_use]
    print(f"[wilddet3d-scene] {len(frames_to_use)} frames (subsample={subsample}) at {src_w}x{src_h}")

    device = pick_device()
    print(f"[wilddet3d-scene] device={device}", flush=True)
    print(f"[wilddet3d-scene] Using device: {device}")

    ckpt_path = os.path.join(WILDDET3D_ROOT, "ckpt",
                             "wilddet3d_alldata_all_prompt_v1.0.pt")
    progress(f"Loading WildDet3D on {device}...")
    print(f"[wilddet3d-scene] Loading model from {ckpt_path}")
    model = build_model(
        checkpoint=ckpt_path,
        score_threshold=0.3,
        device=device,
        skip_pretrained=True,
        use_predicted_intrinsics=True,
    )

    # Depth maps at 1008x1008 padded space — crop out the content region
    # and resize to source resolution before saving so the viewer can
    # unproject through the stored K at source resolution.
    frames_out = []
    global_K = None

    N = len(frames_to_use)
    for k, (frame_path, frame_idx) in enumerate(zip(tqdm(frames_to_use, desc="WildDet3D-scene"),
                                                     frame_indices)):
        img_bgr = cv2.imread(frame_path)
        if img_bgr is None:
            continue
        if k % 5 == 0 or k == N - 1:
            progress(f"Per-frame depth + intrinsics: {k+1}/{N}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        orig_h, orig_w = img_rgb.shape[:2]

        data = preprocess(img_rgb, intrinsics=None)

        # Run with a dummy text prompt (needed to satisfy the API; we
        # throw away detections and only use depth + K_pred).
        amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
                   if device == "cuda" else contextlib.nullcontext())
        with torch.no_grad(), amp_ctx:
            outputs = model(
                images=data["images"].to(device),
                intrinsics=data["intrinsics"].unsqueeze(0).to(device),
                input_hw=[data["input_hw"]],
                original_hw=[data["original_hw"]],
                padding=[data["padding"]],
                input_texts=["object"],
                return_predicted_intrinsics=True,
            )

        # Unpack 9-tuple with predicted intrinsics
        (_, _, _, _, _, _, depth_maps, predicted_K, _) = outputs

        # Extract depth (1, 1, H=1008, W=1008)
        depth_pad = depth_maps[0].float().cpu().numpy()
        if depth_pad.ndim == 3:
            depth_pad = depth_pad[0]  # (1008, 1008)

        # Crop padding to get content region
        pad_left, pad_right, pad_top, pad_bottom = data["padding"]
        padded_h = 1008 - pad_top - pad_bottom
        padded_w = 1008 - pad_left - pad_right
        depth_content = depth_pad[pad_top:pad_top + padded_h,
                                   pad_left:pad_left + padded_w]

        # Resize to source resolution
        depth_src = cv2.resize(depth_content, (src_w, src_h),
                               interpolation=cv2.INTER_LINEAR)

        npz_path = os.path.join(depth_dir, f"{frame_idx:06d}.npz")
        np.savez_compressed(npz_path, depth=depth_src.astype(np.float16))

        # Extract predicted intrinsics (in padded 1008 space). Convert to
        # source resolution.
        K_pad = predicted_K[0].float().cpu().numpy()  # (3, 3)
        # Remove padding offset from principal point, then scale to source
        K_src = K_pad.copy()
        K_src[0, 2] -= pad_left
        K_src[1, 2] -= pad_top
        scale_x = src_w / padded_w
        scale_y = src_h / padded_h
        K_src[0, 0] *= scale_x  # fx
        K_src[1, 1] *= scale_y  # fy
        K_src[0, 2] *= scale_x  # cx
        K_src[1, 2] *= scale_y  # cy

        if global_K is None:
            global_K = K_src

        # Identity pose (no extrinsics available)
        R_cw = np.eye(3)
        t_cw = np.zeros(3)

        frames_out.append({
            "idx": frame_idx,
            "name": f"{frame_idx:06d}.jpg",
            "registered": True,
            "R": R_cw.tolist(),
            "t": t_cw.tolist(),
            "sparse_obs": [],
        })

    cameras = {
        "model": "WILDDET3D",
        "width": src_w,
        "height": src_h,
        "source_width": src_w,
        "source_height": src_h,
        "scale_factor": 1.0,
        "subsample_every": subsample,
        "K": global_K.tolist() if global_K is not None else None,
        "k1": 0.0,
        "num_points": 0,
        "num_registered": len(frames_out),
        "frames": frames_out,
    }

    cam_path = os.path.join(out_dir, "cameras.json")
    with open(cam_path, "w") as f:
        json.dump(cameras, f, indent=2)
    print(f"[wilddet3d-scene] Wrote {cam_path}")
    print(f"[wilddet3d-scene] Wrote {len(frames_out)} depth maps to {depth_dir}")


if __name__ == "__main__":
    main()
