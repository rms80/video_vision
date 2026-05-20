"""Build a per-object world-space point cloud from per-frame depth + tracking masks.

For each registered camera in cameras.json, load the depth map and the
per-frame segmentation mask (resized to depth resolution). Filter depth
points to those inside the mask, unproject through K, and transform to
world space (R^T (X - t)). All points are concatenated and written in
the same Three.js convention as scene_pointmap.npz so the viewer can
reuse the existing scene-pointmap rendering path.

Output is sharded into <basename>_NNN.npz chunks plus a <basename>_chunks.json
manifest so the viewer can stream the cloud and render progressively.

Usage:
    python build_object_pointmap.py <scene_dir> <analysis_dir> \
        --cameras-dir <plugin/cameras> --depth-dir <plugin/depth> --out OUT.npz
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from _pointcloud_io import DEFAULT_CHUNK_SIZE, save_chunked_pointcloud
from _progress import progress


def erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Shrink a bool mask inward by `radius` pixels (square kernel).

    Used to peel off the uncertain silhouette pixels where the depth map's
    interpolated boundary value would unproject to a fly-away point behind
    the object. Applied at the mask's native resolution (i.e. the original
    SAM track resolution) so the unit is meaningful regardless of which
    depth plugin is downstream.
    """
    if radius <= 0 or not mask.any():
        return mask
    k = 2 * radius + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def load_mask(path: Path, target_hw: tuple[int, int], erode: int = 0) -> np.ndarray:
    """Load a SAM2-style RGBA mask PNG, optionally erode at native resolution,
    then downsample (nearest) to `target_hw`. Eroding before the resize keeps
    the erode radius measured in *mask* pixels regardless of the depth plugin's
    working resolution."""
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3:
        m = arr[..., -1] if arr.shape[-1] == 4 else arr.max(axis=-1)
    else:
        m = arr
    mask = m > 0
    if erode > 0:
        mask = erode_mask(mask, erode)
    if mask.shape != target_hw:
        m8 = (mask.astype(np.uint8) * 255)
        m8 = np.array(Image.fromarray(m8).resize(
            (target_hw[1], target_hw[0]), Image.NEAREST,
        ))
        mask = m8 > 0
    return mask


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
    ap.add_argument("--cameras-dir", required=True,
                    help="Scene-relative dir holding cameras.json (e.g. 'pi3')")
    ap.add_argument("--depth-dir", required=True,
                    help="Scene-relative dir holding per-frame depth NPZs (e.g. 'pi3/depth')")
    ap.add_argument("--out", type=Path, required=True, help="Output .npz path")
    ap.add_argument("--erode", type=int, default=0,
                    help="Erode the mask by N pixels at the mask's native "
                         "(source) resolution before downsampling to depth res. "
                         "Peels off silhouette boundary points where interpolated "
                         "depth produces fly-aways. Default 0 (off).")
    args = ap.parse_args()

    cameras_path = args.scene_dir / args.cameras_dir / "cameras.json"
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
    registered = [f for f in cams["frames"]
                  if f.get("registered") and f.get("R") is not None and f.get("t") is not None]
    total = len(registered)
    erode_note = f", erode={args.erode}px" if args.erode > 0 else ""
    progress(f"Fusing {total} masked frames into object cloud{erode_note}...")
    for k, f in enumerate(registered):
        if k % 10 == 0 or k == total - 1:
            progress(f"Object cloud: frame {k+1}/{total}")
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
        mask = load_mask(mask_path, (H, W), erode=args.erode)
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

    progress(f"Writing {len(pts):,} points as chunked manifest...")
    out_dir = args.out.parent
    basename = args.out.stem  # strips .npz
    mp = save_chunked_pointcloud(
        out_dir, basename, pts, rgbs, conf, chunk_size=DEFAULT_CHUNK_SIZE,
    )
    print(f"[object-pointmap] frames used: {n_used} (skipped {n_skipped})")
    print(f"[object-pointmap] points: {len(pts):,}")
    print(f"[object-pointmap] wrote {mp} ({len(pts) // DEFAULT_CHUNK_SIZE + 1} chunks)")


if __name__ == "__main__":
    main()
