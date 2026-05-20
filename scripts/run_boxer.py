"""Lift per-frame 2D bounding boxes to 3D using Facebook's Boxer model.

Usage:
    python run_boxer.py <scene_dir> <analysis_dir> --out-dir <out_dir> \
        [--label LABEL] [--thresh3d 0.5]

Reads:
    <scene_dir>/<cameras_dir>/cameras.json   — camera intrinsics + per-frame poses
    <scene_dir>/<depth_dir>/*.npz            — per-frame depth maps
    <scene_dir>/<pointmap_dir>/*.npz         — per-frame pointmaps (optional)
    <analysis_dir>/track.json                — per-frame 2D bboxes from SAM2 tracking

Outputs:
    <out_dir>/boxes.json              — per-frame 3D oriented bounding boxes
"""

import os
import sys

# Ensure UTF-8 output on Windows (boxer's fuser uses unicode symbols)
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _device import pick_device  # noqa: E402

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add Boxer repo to path
BOXER_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models", "external", "boxer"))
sys.path.insert(0, BOXER_ROOT)

from boxernet.boxernet import BoxerNet
from loaders.base_loader import BaseLoader
import utils.gravity as gravity_module
from utils.tw.pose import PoseTW
from utils.tw.obb import ObbTW


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


def load_pointmap(scene_dir: str, pointmap_dir: str, frame_idx: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Load a camera-frame pointmap .npz, returning (pts3d (H,W,3), conf (H,W)) or None."""
    npz_path = os.path.join(scene_dir, pointmap_dir, f"{frame_idx:06d}.npz")
    if not os.path.exists(npz_path):
        return None
    with np.load(npz_path) as data:
        pts3d = data["pts3d"].astype(np.float32)
        conf = data["conf"].astype(np.float32)
    return pts3d, conf


def sdp_from_pointmap(
    pts3d: np.ndarray,
    conf: np.ndarray,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
    num_samples: int = 10000,
    conf_quantile: float = 0.5,
) -> torch.Tensor:
    """Build semi-dense world-space points from a camera-space pointmap.

    Same output format as BaseLoader.sdp_from_depth: (num_samples, 3) float32,
    NaN-padded if fewer valid points.
    """
    h, w, _ = pts3d.shape
    step = max(1, int(np.sqrt(h * w / (num_samples * 2))))
    pts_sub = pts3d[::step, ::step].reshape(-1, 3)
    conf_sub = conf[::step, ::step].ravel()

    # Adaptive threshold: keep the top (1 - conf_quantile) fraction of points.
    # Works regardless of confidence scale (CUT3R ~1-15, Pi3 ~0-1, etc.)
    finite_mask = np.isfinite(pts_sub).all(axis=1) & np.isfinite(conf_sub)
    if finite_mask.sum() > 0:
        conf_threshold = float(np.quantile(conf_sub[finite_mask], conf_quantile))
    else:
        conf_threshold = 0.0

    # Filter by confidence and finite values
    valid = (conf_sub >= conf_threshold) & finite_mask
    pts_valid = pts_sub[valid]

    if len(pts_valid) > num_samples:
        idx = np.random.choice(len(pts_valid), size=num_samples, replace=False)
        pts_valid = pts_valid[idx]

    if len(pts_valid) == 0:
        return torch.zeros(0, 3, dtype=torch.float32)

    # Transform camera-space points to world space (in Boxer convention)
    sdp_w_np = (pts_valid @ R_wc.T) + t_wc
    sdp_w = torch.from_numpy(sdp_w_np)

    if sdp_w.shape[0] < num_samples:
        num_pad = num_samples - sdp_w.shape[0]
        pad_vals = torch.full((num_pad, 3), float("nan"), dtype=torch.float32)
        sdp_w = torch.cat([sdp_w, pad_vals], dim=0)

    return sdp_w.float()


def estimate_gravity(cameras: dict) -> np.ndarray:
    """Estimate gravity direction from camera poses.

    Assumes handheld video where camera Y-down roughly aligns with gravity.
    For COLMAP poses (camera-from-world): the camera's down direction in world
    coords is the second row of R (transposed), i.e. R_wc[:,1] = R_cw[1,:].

    We average across all registered frames and negate to get the gravity
    (downward) direction.
    """
    up_vectors = []
    for frame in cameras["frames"]:
        if not frame["registered"] or frame["R"] is None:
            continue
        R_cw = np.array(frame["R"], dtype=np.float64)
        # Camera Y-axis in world coords (points down in image / roughly down in world)
        cam_y_world = R_cw[1, :]  # = R_wc[:,1] since R_wc = R_cw.T
        up_vectors.append(cam_y_world)

    avg_down = np.mean(up_vectors, axis=0)
    avg_down /= np.linalg.norm(avg_down)
    # Gravity direction = average camera down direction
    return avg_down.astype(np.float32)


def obb_to_dict(obb: ObbTW, R_from_boxer: np.ndarray) -> dict:
    """Convert a single ObbTW to a JSON-serializable dict.

    R_from_boxer rotates from Boxer's gravity convention back to our world.
    """
    T_wo = obb.T_world_object
    R_wo = T_wo.R.squeeze().numpy()  # (3,3) in Boxer world
    t_wo = T_wo.t.squeeze().numpy()  # (3,) in Boxer world
    bb3 = obb.bb3_object.squeeze().numpy()  # (6,) [xmin,xmax,ymin,ymax,zmin,zmax]
    size = [
        float(bb3[1] - bb3[0]),
        float(bb3[3] - bb3[2]),
        float(bb3[5] - bb3[4]),
    ]
    center_obj = [
        float((bb3[0] + bb3[1]) / 2),
        float((bb3[2] + bb3[3]) / 2),
        float((bb3[4] + bb3[5]) / 2),
    ]
    # Transform center to Boxer world, then rotate back to our world
    center_boxer = R_wo @ np.array(center_obj) + t_wo
    center_world = (R_from_boxer @ center_boxer).tolist()

    # Compute 8 corners: object → Boxer world → our world
    corners_obj = obb.bb3corners_object.squeeze().numpy()  # (8, 3)
    corners_boxer = corners_obj @ R_wo.T + t_wo
    corners_world = (corners_boxer @ R_from_boxer.T).tolist()

    # Rotate the object-to-world transform back to our world
    R_wo_ours = R_from_boxer @ R_wo
    t_wo_ours = R_from_boxer @ t_wo

    return {
        "center": center_world,
        "size": size,
        "R": R_wo_ours.tolist(),
        "t": t_wo_ours.tolist(),
        "corners": corners_world,
        "confidence": float(obb.prob.squeeze().item()),
    }


def main():
    parser = argparse.ArgumentParser(description="Lift 2D boxes to 3D with Boxer")
    parser.add_argument("scene_dir", help="Path to _scene directory")
    parser.add_argument("analysis_dir", help="Path to analysis run directory")
    parser.add_argument("--out-dir", required=True,
                        help="Directory to write boxes.json into (created if missing)")
    parser.add_argument("--label", default="object", help="Object label for detection")
    parser.add_argument("--thresh3d", type=float, default=0.3,
                        help="3D confidence threshold")
    parser.add_argument("--fuse", action="store_true",
                        help="Run post-hoc 3D box fusion across frames")
    parser.add_argument("--cameras-dir", default="colmap",
                        help="Scene-relative dir holding cameras.json")
    parser.add_argument("--depth-dir", default="depthanythingv2",
                        help="Scene-relative dir holding per-frame NNNNNN.npz depth maps")
    parser.add_argument("--pointmap-dir", default=None,
                        help="Scene-relative dir holding per-frame NNNNNN.npz camera-space "
                             "pointmaps. If unset, the solver only consumes depth.")
    parser.add_argument("--force-precision", choices=["float32", "bfloat16"],
                        default=None)
    args = parser.parse_args()

    scene_dir = args.scene_dir
    analysis_dir = args.analysis_dir
    cameras_dir = args.cameras_dir
    depth_dir = args.depth_dir
    pointmap_dir = args.pointmap_dir

    # Load data
    print(f"[boxer] Loading cameras from {scene_dir}/{cameras_dir}")
    cameras = load_cameras(scene_dir, cameras_dir)
    print(f"[boxer] Loading track from {analysis_dir}")
    track = load_track(analysis_dir)

    K = np.array(cameras["K"], dtype=np.float64)
    cam_width = cameras["width"]
    cam_height = cameras["height"]
    source_width = cameras["source_width"]
    source_height = cameras["source_height"]
    scale_factor = cameras["scale_factor"]

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
    # Frame indices in cameras.json are already video frame indices.
    valid_indices = []
    for frame_idx in sorted(frame_lookup.keys()):
        if frame_idx in track_bboxes:
            valid_indices.append(frame_idx)

    print(f"[boxer] {len(valid_indices)} frames with both pose and bbox "
          f"(of {len(frame_lookup)} registered, {len(track_bboxes)} tracked)")

    if not valid_indices:
        print("[boxer] ERROR: No frames with both camera pose and tracking bbox")
        sys.exit(1)

    # Determine gravity direction and build a rotation to map our world
    # into Boxer's expected convention (gravity = [0,0,-1]).
    # BoxerNet was trained with Aria data where gravity is -Z.
    if cameras.get("gravity_aligned"):
        gravity = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        print(f"[boxer] Scene is floor-aligned, using gravity = {gravity}")
    else:
        gravity = estimate_gravity(cameras)
        print(f"[boxer] Estimated gravity direction: {gravity}")

    # R_boxer rotates our gravity to [0,0,-1]: R_boxer @ gravity = [0,0,-1]
    boxer_gravity = np.array([0.0, 0.0, -1.0])
    v = np.cross(gravity, boxer_gravity)
    s = np.linalg.norm(v)
    c_val = np.dot(gravity, boxer_gravity)
    if s < 1e-8:
        R_to_boxer = np.eye(3) if c_val > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R_to_boxer = np.eye(3) + vx + vx @ vx * ((1 - c_val) / (s * s))
    R_to_boxer = R_to_boxer.astype(np.float64)
    R_from_boxer = R_to_boxer.T  # inverse rotation to map results back
    print(f"[boxer] World->Boxer rotation angle: {np.degrees(np.arccos(np.clip(c_val, -1, 1))):.1f} deg")

    # Device setup
    device = pick_device()
    print(f"[boxer] Using device: {device}")

    # Load BoxerNet
    ckpt_path = os.path.join(BOXER_ROOT, "ckpts",
                             "boxernet_hw960in4x6d768-wssxpf9p.ckpt")
    print(f"[boxer] Loading BoxerNet from {ckpt_path}")
    boxernet = BoxerNet.load_from_checkpoint(ckpt_path, device=device)
    model_hw = boxernet.hw  # target image size (e.g. 960)
    print(f"[boxer] Model input size: {model_hw}x{model_hw}")

    # Precision
    if args.force_precision is not None:
        precision_dtype = (torch.bfloat16 if args.force_precision == "bfloat16"
                           else torch.float32)
    elif device == "cuda" and torch.cuda.is_bf16_supported():
        precision_dtype = torch.bfloat16
    else:
        precision_dtype = torch.float32

    # Process frames
    results = []
    all_obbs_boxer = []  # collect OBBs in Boxer world for fusion
    for frame_idx in tqdm(valid_indices, desc="Boxer"):
        frame = frame_lookup[frame_idx]

        # Get 2D bbox (in source image pixels)
        bbox_src = track_bboxes.get(frame_idx)
        if bbox_src is None:
            continue
        x1, y1, x2, y2 = bbox_src  # source pixel coords

        # Load image (original extracted frames)
        img_path = os.path.join(scene_dir, "frames", frame["name"])
        if not os.path.exists(img_path):
            continue
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        # Resize image to model input size
        img_resized = cv2.resize(img_rgb, (model_hw, model_hw),
                                 interpolation=cv2.INTER_LINEAR)

        # Scale intrinsics from camera working resolution to model resolution
        # (K is defined at cam_width x cam_height, not source resolution)
        scale_to_model_x = model_hw / cam_width
        scale_to_model_y = model_hw / cam_height
        fx = K[0, 0] * scale_to_model_x
        fy = K[1, 1] * scale_to_model_y
        cx = K[0, 2] * scale_to_model_x
        cy = K[1, 2] * scale_to_model_y

        # Scale bbox from source pixels directly to model resolution
        src_to_model_x = model_hw / source_width
        src_to_model_y = model_hw / source_height
        bb_x1 = x1 * src_to_model_x
        bb_y1 = y1 * src_to_model_y
        bb_x2 = x2 * src_to_model_x
        bb_y2 = y2 * src_to_model_y

        # Boxer expects bb2d as [xmin, xmax, ymin, ymax]
        bb2d = torch.tensor([[bb_x1, bb_x2, bb_y1, bb_y2]], dtype=torch.float32)

        # Build camera (pinhole)
        cam = BaseLoader.pinhole_from_K(
            model_hw, model_hw, fx, fy, cx, cy,
            valid_radius=(model_hw, model_hw),
        ).float()

        # Camera pose: cameras.json has camera-from-world (R_cw, t_cw)
        R_cw = np.array(frame["R"], dtype=np.float64)
        t_cw = np.array(frame["t"], dtype=np.float64)
        # World-from-camera, rotated into Boxer's gravity convention
        R_wc = R_to_boxer @ R_cw.T
        t_wc = R_to_boxer @ (-R_cw.T @ t_cw)

        R_flat = R_wc.flatten().astype(np.float32)
        t_vec = t_wc.astype(np.float32)
        T_wr = PoseTW(torch.tensor([*R_flat, *t_vec], dtype=torch.float32))

        # Load semi-dense points: prefer pointmap (direct 3D) over depth (unproject through K)
        pm = load_pointmap(scene_dir, pointmap_dir, frame_idx) if pointmap_dir else None
        if pm is not None:
            pts3d, conf = pm
            sdp_w = sdp_from_pointmap(
                pts3d, conf,
                R_wc.astype(np.float32), t_wc.astype(np.float32),
                num_samples=10000,
            )
        else:
            depth_np = load_depth(scene_dir, depth_dir, frame_idx)
            if depth_np is not None:
                # Resize depth to model resolution for correct unprojection
                depth_model = cv2.resize(depth_np, (model_hw, model_hw),
                                         interpolation=cv2.INTER_NEAREST)
                sdp_w = BaseLoader.sdp_from_depth(
                    depth_model,
                    float(fx), float(fy), float(cx), float(cy),
                    R_wc.astype(np.float32), t_wc.astype(np.float32),
                    num_samples=10000,
                )
            else:
                sdp_w = torch.zeros(0, 3, dtype=torch.float32)

        # Build datum
        img_tensor = BaseLoader.img_to_tensor(img_resized)
        datum = {
            "img0": img_tensor,
            "cam0": cam,
            "T_world_rig0": T_wr,
            "sdp_w": sdp_w,
            "bb2d": bb2d,
            "rotated0": torch.tensor(False).reshape(1),
            "time_ns0": frame_idx,
        }

        # Run BoxerNet
        if device == "mps":
            outputs = boxernet.forward(datum)
        else:
            with torch.autocast(device_type=device, dtype=precision_dtype):
                outputs = boxernet.forward(datum)

        obb_pr_w = outputs["obbs_pr_w"].cpu()[0]

        # Filter by confidence
        keep = obb_pr_w.prob.squeeze(-1) >= args.thresh3d
        obb_pr_w = obb_pr_w[keep]

        if len(obb_pr_w) == 0:
            results.append({
                "frame": frame_idx,
                "colmap_frame": frame_idx,
                "boxes": [],
            })
            continue

        # Collect for fusion (in Boxer world space)
        all_obbs_boxer.append(obb_pr_w)

        # Convert OBBs to JSON-serializable dicts
        boxes = []
        for i in range(len(obb_pr_w)):
            boxes.append(obb_to_dict(obb_pr_w[i:i+1], R_from_boxer))

        results.append({
            "frame": frame_idx,
            "colmap_frame": frame_idx,
            "boxes": boxes,
        })

    # Fuse per-frame boxes into static instances
    fused_boxes = []
    if args.fuse and all_obbs_boxer:
        from utils.fuse_3d_boxes import BoundingBox3DFuser
        all_obbs = ObbTW(torch.cat([o._data for o in all_obbs_boxer], dim=0))
        print(f"[boxer] Fusing {len(all_obbs)} detections across {len(all_obbs_boxer)} frames")
        fuser = BoundingBox3DFuser(
            iou_threshold=0.3,
            min_detections=4,
            conf_threshold=0.5,
        )
        fused_instances = fuser.fuse(all_obbs)
        print(f"[boxer] Fused into {len(fused_instances)} instances")
        for inst in fused_instances:
            fused_boxes.append(obb_to_dict(inst.obb.unsqueeze(0), R_from_boxer))

    # Write output
    output = {
        "label": args.label,
        "thresh3d": args.thresh3d,
        "fused": args.fuse,
        "gravity": gravity.tolist(),
        "num_frames": len(results),
        "num_frames_with_boxes": sum(1 for r in results if r["boxes"]),
        "frames": results,
    }
    if fused_boxes:
        output["fused_boxes"] = fused_boxes

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "boxes.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[boxer] Wrote {out_path}")
    total_boxes = sum(len(r["boxes"]) for r in results)
    print(f"[boxer] {total_boxes} 3D boxes across {len(results)} frames "
          f"({output['num_frames_with_boxes']} with detections)")
    if fused_boxes:
        print(f"[boxer] {len(fused_boxes)} fused instances")


if __name__ == "__main__":
    main()
