"""Build a per-object world-space point cloud from per-frame depth + tracking masks.

For each registered camera in cameras.json, load the depth map and the
per-frame segmentation mask (resized to depth resolution). Filter depth
points to those inside the mask, unproject through K, and transform to
world space (R^T (X - t)). All points are concatenated and written in
the same Three.js convention as scene_pointmap.npz so the viewer can
reuse the existing scene-pointmap rendering path.

Usage:
    python build_object_pointmap.py <scene_dir> <analysis_dir> \
        --source <plugin_id> --depth-dir <plugin/depth> --out OUT.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_mask(path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    """Load a SAM2-style RGBA mask PNG and return a bool (H, W) at target_hw."""
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3:
        m = arr[..., -1] if arr.shape[-1] == 4 else arr.max(axis=-1)
    else:
        m = arr
    if m.shape != target_hw:
        m = np.array(Image.fromarray(m).resize((target_hw[1], target_hw[0]), Image.NEAREST))
    return m > 0


def load_rgb(path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    """Load an RGB jpg, resized (BILINEAR) to target_hw, returning (H, W, 3) uint8."""
    img = Image.open(path).convert("RGB")
    if img.size != (target_hw[1], target_hw[0]):
        img = img.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
    return np.array(img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("analysis_dir", type=Path)
    ap.add_argument("--source", required=True, help="Scene plugin id (matches camerasDir)")
    ap.add_argument("--depth-dir", required=True,
                    help="Depth subdir under scene_dir (e.g. 'pi3/depth')")
    ap.add_argument("--out", type=Path, required=True, help="Output .npz path")
    args = ap.parse_args()

    cameras_path = args.scene_dir / args.source / "cameras.json"
    if not cameras_path.exists():
        raise SystemExit(f"cameras.json not found: {cameras_path}")
    cams = json.loads(cameras_path.read_text())
    K = np.asarray(cams["K"], dtype=np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx_, cy_ = float(K[0, 2]), float(K[1, 2])

    depth_dir = args.scene_dir / args.depth_dir
    masks_dir = args.analysis_dir / "masks"
    frames_dir = args.scene_dir / "frames"

    all_pts: list[np.ndarray] = []
    all_rgb: list[np.ndarray] = []

    n_used = 0
    n_skipped = 0
    for f in cams["frames"]:
        if not f.get("registered") or f.get("R") is None or f.get("t") is None:
            continue
        idx = int(f["idx"])
        pad = f"{idx:06d}"
        depth_path = depth_dir / f"{pad}.npz"
        mask_path = masks_dir / f"{pad}.png"
        rgb_path = frames_dir / f"{pad}.jpg"
        if not (depth_path.exists() and mask_path.exists() and rgb_path.exists()):
            n_skipped += 1
            continue

        with np.load(depth_path) as data:
            depth = data["depth"].astype(np.float32)
        H, W = depth.shape
        mask = load_mask(mask_path, (H, W))
        valid = np.isfinite(depth) & (depth > 0) & mask
        if not valid.any():
            n_skipped += 1
            continue
        rgb = load_rgb(rgb_path, (H, W))

        vy, vx = np.where(valid)
        z = depth[vy, vx].astype(np.float64)
        # Unproject into camera space (X_c)
        xc = (vx.astype(np.float64) - cx_) * z / fx
        yc = (vy.astype(np.float64) - cy_) * z / fy
        zc = z
        X_cam = np.stack([xc, yc, zc], axis=1)  # (N, 3)

        R = np.asarray(f["R"], dtype=np.float64)  # 3x3 (camera-from-world)
        t = np.asarray(f["t"], dtype=np.float64)  # (3,)  (camera-from-world)
        # World point: X_w = R^T (X_c - t).  In row-vector form: (X_c - t) @ R.
        X_w = (X_cam - t) @ R
        # COLMAP/OpenCV (Y down, Z forward) → Three.js (Y up, Z back)
        X_w[:, 1] *= -1
        X_w[:, 2] *= -1

        all_pts.append(X_w.astype(np.float32))
        all_rgb.append(rgb[vy, vx].astype(np.uint8))
        n_used += 1

    if not all_pts:
        raise SystemExit(
            f"No frames produced any points (used=0, skipped={n_skipped}). "
            "Check that depth maps, frame jpgs, and per-frame masks all exist."
        )

    pts = np.concatenate(all_pts, axis=0)
    rgbs = np.concatenate(all_rgb, axis=0)
    conf = np.ones(len(pts), dtype=np.float16)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        pts3d=pts.astype(np.float16),
        rgb=rgbs,
        conf=conf,
    )
    print(f"[object-pointmap] frames used: {n_used} (skipped {n_skipped})")
    print(f"[object-pointmap] points: {len(pts):,}")
    print(f"[object-pointmap] wrote {args.out}")


if __name__ == "__main__":
    main()
