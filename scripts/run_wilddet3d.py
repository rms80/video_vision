"""Lift per-frame 2D bounding boxes to 3D using WildDet3D.

Usage:
    python run_wilddet3d.py <scene_dir> <analysis_dir> --out-dir <out_dir> \
        [--label LABEL] [--cameras-dir DIR] [--depth-dir DIR]
        [--use-intrinsics] [--use-depth] [--thresh3d 0.3]

Reads:
    <scene_dir>/<cameras_dir>/cameras.json  — camera intrinsics + per-frame poses
    <scene_dir>/<depth_dir>/*.npz           — per-frame depth maps (if --use-depth)
    <analysis_dir>/track.json               — per-frame 2D bboxes from SAM2 tracking

Outputs:
    <out_dir>/boxes.json              — per-frame 3D oriented bounding boxes
"""

import sys
import os
import json
import argparse
import contextlib

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _device import pick_device  # noqa: E402

# Add WildDet3D repo to path
WILDDET3D_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "external", "wilddet3d")
)
sys.path.insert(0, WILDDET3D_ROOT)
# WildDet3D's submodules
sys.path.insert(0, os.path.join(WILDDET3D_ROOT, "third_party", "sam3"))
sys.path.insert(0, os.path.join(WILDDET3D_ROOT, "third_party", "lingbot_depth"))

from wilddet3d.inference import build_model
from wilddet3d.preprocessing import preprocess


def load_cameras(scene_dir: str, cameras_dir: str) -> dict:
    cam_path = os.path.join(scene_dir, cameras_dir, "cameras.json")
    with open(cam_path) as f:
        return json.load(f)


def load_track(analysis_dir: str) -> dict:
    track_path = os.path.join(analysis_dir, "track.json")
    with open(track_path) as f:
        return json.load(f)


def load_depth(scene_dir: str, depth_dir: str, frame_idx: int) -> np.ndarray | None:
    """Load a depth .npz, returning float32 (H, W) or None."""
    npz_path = os.path.join(scene_dir, depth_dir, f"{frame_idx:06d}.npz")
    if not os.path.exists(npz_path):
        return None
    with np.load(npz_path) as data:
        depth = data["depth"].astype(np.float32)
    return depth


def quat_to_rotmat(qw, qx, qy, qz):
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def box3d_to_corners(center, dims, R_obj):
    """Compute 8 corners of an oriented bounding box.

    Args:
        center: (3,) center in world coordinates
        dims: (3,) [W, L, H] in WildDet3D's Omni3D order — width, length, height.
        R_obj: (3, 3) object rotation matrix in world frame
    Returns:
        corners: (8, 3) corner positions in world coordinates

    Note: WildDet3D's local axes are permuted vs (W, L, H): local x = length,
    local y = height, local z = width. Without this remap the rotation R_obj
    spins the box around the wrong axes and a chair lays on its side. See
    boxes3d_to_corners in WildDet3D's vis3d_glb.py.
    """
    w, l, h = dims
    # Half-extents along the object's *local* axes (x=L, y=H, z=W).
    dx, dy, dz = l / 2, h / 2, w / 2
    # 8 corners in object-local frame
    local = np.array([
        [-dx, -dy, -dz],
        [+dx, -dy, -dz],
        [+dx, +dy, -dz],
        [-dx, +dy, -dz],
        [-dx, -dy, +dz],
        [+dx, -dy, +dz],
        [+dx, +dy, +dz],
        [-dx, +dy, +dz],
    ], dtype=np.float64)
    # Rotate to world and translate
    return (local @ R_obj.T) + center


def main():
    parser = argparse.ArgumentParser(description="WildDet3D 3D bbox lifting")
    parser.add_argument("scene_dir", help="Scene directory with frames/ and cameras")
    parser.add_argument("analysis_dir", help="Analysis directory with track.json")
    parser.add_argument("--out-dir", required=True,
                        help="Directory to write boxes.json into (created if missing)")
    parser.add_argument("--label", default="object", help="Object label")
    parser.add_argument("--cameras-dir", default="colmap",
                        help="Scene-relative dir holding cameras.json")
    parser.add_argument("--depth-dir", default="depthanythingv2",
                        help="Scene-relative dir holding per-frame NNNNNN.npz depth maps "
                             "(used only when --use-depth is set)")
    parser.add_argument("--pointmap-dir", default=None,
                        help="Unused; accepted for dispatch compatibility with run_boxer.")
    parser.add_argument("--use-intrinsics", action="store_true",
                        help="Pass camera intrinsics to the model")
    parser.add_argument("--use-depth", action="store_true",
                        help="Pass depth maps as input to the model")
    parser.add_argument("--thresh3d", type=float, default=0.3,
                        help="3D confidence threshold")
    args = parser.parse_args()

    scene_dir = args.scene_dir
    analysis_dir = args.analysis_dir
    cameras_dir = args.cameras_dir
    depth_dir = args.depth_dir

    # Load data
    print(f"[wilddet3d] Loading cameras from {scene_dir}/{cameras_dir}")
    cameras = load_cameras(scene_dir, cameras_dir)
    print(f"[wilddet3d] Loading track from {analysis_dir}")
    track = load_track(analysis_dir)

    K = np.array(cameras["K"], dtype=np.float64)
    cam_width = cameras["width"]
    cam_height = cameras["height"]
    source_width = cameras["source_width"]
    source_height = cameras["source_height"]

    # Build frame lookup: frame_idx -> frame data
    frame_lookup = {}
    for frame in cameras["frames"]:
        if frame["registered"] and frame["R"] is not None:
            frame_lookup[frame["idx"]] = frame

    # Build track bbox lookup: frame_idx -> [x1, y1, x2, y2] in source pixels
    track_bboxes = {}
    for tframe in track["frames"]:
        if tframe["bbox"] is not None:
            track_bboxes[tframe["frame"]] = tframe["bbox"]

    # Find frames that have both a camera pose and a tracking bbox.
    # cameras.json stores the source frame index directly in frame["idx"],
    # so frame_idx is the source frame index.
    subsample = cameras.get("subsample_every", 1)
    valid_indices = []
    for frame_idx in sorted(frame_lookup.keys()):
        if frame_idx in track_bboxes:
            valid_indices.append(frame_idx)

    print(f"[wilddet3d] {len(valid_indices)} frames with both pose and bbox "
          f"(of {len(frame_lookup)} registered, {len(track_bboxes)} tracked)")

    if not valid_indices:
        print("[wilddet3d] ERROR: No frames with both camera pose and tracking bbox")
        sys.exit(1)

    # Device setup
    device = pick_device()
    print(f"[wilddet3d] Using device: {device}")

    # Load WildDet3D model
    ckpt_path = os.path.join(WILDDET3D_ROOT, "ckpt",
                             "wilddet3d_alldata_all_prompt_v1.0.pt")
    print(f"[wilddet3d] Loading model from {ckpt_path}")
    model = build_model(
        checkpoint=ckpt_path,
        sam3_checkpoint=os.path.join(WILDDET3D_ROOT, "pretrained", "sam3", "sam3_detector.pt"),
        score_threshold=args.thresh3d,
        device=device,
        skip_pretrained=True,
        use_predicted_intrinsics=not args.use_intrinsics,
        use_depth_input_test=args.use_depth,
    )

    # Run on roughly every 10th source frame. If the scene source already
    # subsamples (e.g. wilddet3d scene subsample_every=10), scale step down.
    target_source_step = 10
    keyframe_step = max(1, target_source_step // max(subsample, 1))
    run_indices = valid_indices[::keyframe_step]
    print(f"[wilddet3d] Running on {len(run_indices)} keyframes "
          f"(every {keyframe_step} of {len(valid_indices)} registered; "
          f"subsample_every={subsample}), "
          f"propagating to {len(valid_indices)} total frames")

    # Process keyframes
    results = []
    keyframe_results = {}  # frame_idx -> boxes
    for frame_idx in tqdm(run_indices, desc="WildDet3D"):
        frame = frame_lookup[frame_idx]
        actual_idx = frame_idx

        # Get 2D bbox (in source image pixels)
        bbox_src = track_bboxes.get(actual_idx) or track_bboxes.get(frame_idx)
        if bbox_src is None:
            continue
        x1, y1, x2, y2 = bbox_src

        # Load image
        img_path = os.path.join(scene_dir, "frames", frame["name"])
        if not os.path.exists(img_path):
            continue
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        orig_h, orig_w = img_rgb.shape[:2]

        # Prepare intrinsics for preprocessing (at original image resolution)
        intrinsics_for_preprocess = None
        if args.use_intrinsics:
            # K is defined at cam_width x cam_height; scale to original image resolution
            scale_x = orig_w / cam_width
            scale_y = orig_h / cam_height
            K_orig = np.array([
                [K[0, 0] * scale_x, 0, K[0, 2] * scale_x],
                [0, K[1, 1] * scale_y, K[1, 2] * scale_y],
                [0, 0, 1],
            ], dtype=np.float32)
            intrinsics_for_preprocess = K_orig

        # Preprocess image (handles resize to 1008x1008, normalization, padding)
        data = preprocess(img_rgb, intrinsics=intrinsics_for_preprocess)

        # Box prompt: pixel xyxy in original image coordinates
        input_boxes = [[float(x1), float(y1), float(x2), float(y2)]]

        # Prepare depth input if requested
        depth_gt = None
        if args.use_depth:
            depth_np = load_depth(scene_dir, depth_dir, frame_idx)
            if depth_np is not None:
                # Resize depth to model input size (1008x1008)
                depth_resized = cv2.resize(depth_np, (1008, 1008),
                                           interpolation=cv2.INTER_NEAREST)
                depth_gt = torch.from_numpy(depth_resized).unsqueeze(0).unsqueeze(0).to(device)

        # Run inference
        # preprocess() returns images as (1, 3, H, W) and intrinsics as (3, 3)
        amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
                   if device == "cuda" else contextlib.nullcontext())
        with torch.no_grad(), amp_ctx:
            outputs = model(
                images=data["images"].to(device),
                intrinsics=data["intrinsics"].unsqueeze(0).to(device),
                input_hw=[data["input_hw"]],
                original_hw=[data["original_hw"]],
                padding=[data["padding"]],
                input_boxes=input_boxes,
                prompt_text=f"geometric: {args.label}",
                depth_gt=depth_gt,
            )

        # Unpack: boxes, boxes3d, scores, scores_2d, scores_3d, class_ids, depth_maps
        boxes_2d, boxes3d, scores, scores_2d, scores_3d, class_ids, depth_maps = outputs

        # boxes3d[0] is (N, 10): [center_x, center_y, center_z, W, L, H, qw, qx, qy, qz]
        # in OpenCV camera frame (X-right, Y-down, Z-forward)
        b3d = boxes3d[0].cpu().numpy()
        b2d = boxes_2d[0].cpu().numpy()
        sc = scores[0].cpu().numpy()

        if len(b3d) == 0:
            results.append({
                "frame": actual_idx,
                "colmap_frame": frame_idx,
                "boxes": [],
            })
            continue

        # Camera-to-world transform from cameras.json
        R_cw = np.array(frame["R"], dtype=np.float64)
        t_cw = np.array(frame["t"], dtype=np.float64)
        R_wc = R_cw.T
        t_wc = -R_cw.T @ t_cw

        boxes = []
        for i in range(len(b3d)):
            center_cam = b3d[i, :3].astype(np.float64)
            dims = b3d[i, 3:6].astype(np.float64)  # [W, L, H]
            qw, qx, qy, qz = b3d[i, 6:10].astype(np.float64)

            # Object rotation in camera frame
            R_obj_cam = quat_to_rotmat(qw, qx, qy, qz)

            # Transform to world frame
            center_world = (R_wc @ center_cam + t_wc)
            R_obj_world = R_wc @ R_obj_cam

            # Compute corners
            corners = box3d_to_corners(center_world, dims, R_obj_world)

            boxes.append({
                "center": center_world.tolist(),
                "size": dims.tolist(),
                "R": R_obj_world.tolist(),
                "t": t_wc.tolist(),
                "corners": corners.tolist(),
                "confidence": float(sc[i]),
            })

        keyframe_results[frame_idx] = boxes

    # Build full results: for each valid frame, use the nearest preceding keyframe
    for frame_idx in valid_indices:
        actual_idx = frame_idx
        if frame_idx in keyframe_results:
            boxes = keyframe_results[frame_idx]
        else:
            # Find nearest preceding keyframe
            best_kf = run_indices[0]
            for kf in run_indices:
                if kf <= frame_idx:
                    best_kf = kf
                else:
                    break
            boxes = keyframe_results.get(best_kf, [])
        results.append({
            "frame": actual_idx,
            "colmap_frame": frame_idx,
            "boxes": boxes,
        })

    # Write output (matching BoxerResult format)
    output = {
        "label": args.label,
        "thresh3d": args.thresh3d,
        "use_intrinsics": args.use_intrinsics,
        "use_depth": args.use_depth,
        "gravity": [0.0, 1.0, 0.0],  # placeholder — WildDet3D works in camera frame
        "num_frames": len(results),
        "num_frames_with_boxes": sum(1 for r in results if r["boxes"]),
        "frames": results,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "boxes.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[wilddet3d] Wrote {out_path}")
    total_boxes = sum(len(r["boxes"]) for r in results)
    print(f"[wilddet3d] {total_boxes} 3D boxes across {len(results)} frames "
          f"({output['num_frames_with_boxes']} with detections)")


if __name__ == "__main__":
    main()
