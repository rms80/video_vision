"""Reconstruct a segmented object as a 3D mesh with Hunyuan3D-Omni.

Usage:
    python run_hunyuan_omni.py <scene_dir> <analysis_dir> --frame N --source pi3 --out OUT.glb

Reads:
    <scene_dir>/frames/NNNNNN.jpg                    — current-frame image (raw RGB)
    <analysis_dir>/masks/NNNNNN.png                  — per-frame segmentation mask (RGBA)
    <scene_dir>/<source>/pointmap/NNNNNN.npz         — per-frame camera-space pointmap

Writes:
    <out>                                            — generated .glb mesh
    <out>.points.ply                                 — segmented input point cloud (debug)

Runs in a separate venv at <repo>/hunyuan3d-omni/.venv (Python 3.10, torch 2.5.1+cu124),
not the project-level .venv. The dispatcher in vite.config.ts must invoke that venv's
python.exe explicitly.
"""

import argparse
import os
import sys
from pathlib import Path

# Hunyuan3D-Omni's inference.py loads libgcc_s.so.1 at import time which only
# exists on Linux. We bypass inference.py and import the pipeline directly.
HUNYUAN_ROOT = Path(__file__).resolve().parent.parent.parent / "hunyuan3d-omni"
sys.path.insert(0, str(HUNYUAN_ROOT))

import numpy as np
import torch
import trimesh
from PIL import Image

from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline
from hy3dshape.postprocessors import FloaterRemover, DegenerateFaceRemover


def load_mask(mask_path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    """Load a SAM2-style RGBA mask PNG and return a bool (H, W) at target_hw."""
    img = Image.open(mask_path)
    arr = np.array(img)
    if arr.ndim == 3:
        m = arr[..., -1] if arr.shape[-1] == 4 else arr.max(axis=-1)
    else:
        m = arr
    m = (m > 0).astype(np.uint8) * 255
    if m.shape != target_hw:
        m = np.array(Image.fromarray(m).resize((target_hw[1], target_hw[0]), Image.NEAREST))
    return m > 0


def save_ply(path: Path, points: np.ndarray) -> None:
    n = len(points)
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def normalize_points(pts: np.ndarray, scale: float = 0.98) -> np.ndarray:
    """Center and scale points to fit in [-scale, scale] cube (Hunyuan convention)."""
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = (lo + hi) / 2
    extent = (hi - lo).max()
    if extent < 1e-8:
        return pts - center
    return (pts - center) / extent * 2 * scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("analysis_dir", type=Path)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--source", default="pi3", help="Scene plugin id with pointmap")
    parser.add_argument("--out", type=Path, required=True, help="Output .glb path")
    parser.add_argument("--repo-id", default="tencent/Hunyuan3D-Omni")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--octree", type=int, default=384,
                        help="Octree resolution; 512 is default but uses more VRAM")
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-points", type=int, default=81920,
                        help="Subsample to this many points before feeding to model")
    parser.add_argument("--conf-quantile", type=float, default=0.0,
                        help="Drop points below this confidence quantile (0=keep all valid)")
    parser.add_argument("--flashvdm", action="store_true")
    args = parser.parse_args()

    frame_str = f"{args.frame:06d}"
    image_path = args.scene_dir / "frames" / f"{frame_str}.jpg"
    mask_path = args.analysis_dir / "masks" / f"{frame_str}.png"
    pointmap_path = args.scene_dir / args.source / "pointmap" / f"{frame_str}.npz"

    for p in (image_path, mask_path, pointmap_path):
        if not p.exists():
            sys.exit(f"[hunyuan-omni] Missing input: {p}")

    print(f"[hunyuan-omni] frame={args.frame} source={args.source}")
    print(f"[hunyuan-omni] image:    {image_path}")
    print(f"[hunyuan-omni] mask:     {mask_path}")
    print(f"[hunyuan-omni] pointmap: {pointmap_path}")

    # Load pointmap (camera space) and segmentation mask
    with np.load(pointmap_path) as data:
        pts3d = data["pts3d"].astype(np.float32)  # (H, W, 3)
        conf = data["conf"].astype(np.float32)    # (H, W)
    H, W, _ = pts3d.shape
    mask = load_mask(mask_path, (H, W))

    finite = np.isfinite(pts3d).all(axis=-1) & np.isfinite(conf)
    keep = mask & finite
    if args.conf_quantile > 0 and keep.sum() > 0:
        thresh = float(np.quantile(conf[keep], args.conf_quantile))
        keep &= conf >= thresh

    pts = pts3d[keep]
    print(f"[hunyuan-omni] segmented points: {len(pts)} "
          f"(of {mask.sum()} masked, {finite.sum()} finite)")
    if len(pts) < 100:
        sys.exit(f"[hunyuan-omni] Too few segmented points ({len(pts)})")

    # Normalize to [-0.98, 0.98] cube as the model expects
    pts = normalize_points(pts, scale=0.98)

    if len(pts) > args.max_points:
        idx = np.random.default_rng(args.seed).choice(len(pts), args.max_points, replace=False)
        pts = pts[idx]
        print(f"[hunyuan-omni] subsampled to {len(pts)}")

    # Save the seg pointcloud for debugging next to the output mesh
    args.out.parent.mkdir(parents=True, exist_ok=True)
    debug_ply = args.out.with_suffix(".points.ply")
    save_ply(debug_ply, pts)
    print(f"[hunyuan-omni] wrote debug pointcloud -> {debug_ply}")

    # Load the pipeline (downloads weights on first run, ~several GB)
    print(f"[hunyuan-omni] loading pipeline from {args.repo_id} (flashvdm={args.flashvdm})")
    pipeline = Hunyuan3DOmniSiTFlowMatchingPipeline.from_pretrained(
        args.repo_id, fast_decode=args.flashvdm,
    )

    surface = torch.from_numpy(pts).unsqueeze(0).to(pipeline.device).to(pipeline.dtype)
    print(f"[hunyuan-omni] running inference: steps={args.steps} octree={args.octree}")
    result = pipeline(
        image=str(image_path),
        point=surface,
        num_inference_steps=args.steps,
        octree_resolution=args.octree,
        mc_level=0,
        guidance_scale=args.guidance,
        generator=torch.Generator("cuda").manual_seed(args.seed),
    )
    mesh = result["shapes"][0][0]
    print(f"[hunyuan-omni] raw mesh: V={len(mesh.vertices)} F={len(mesh.faces)}")

    # FloaterRemover / DegenerateFaceRemover use pymeshlab via NamedTemporaryFile
    # without delete=False, which collides with Windows file-lock semantics.
    # Skip cleanly if it fails — the raw mesh is still usable.
    try:
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        print(f"[hunyuan-omni] cleaned: V={len(mesh.vertices)} F={len(mesh.faces)}")
    except (PermissionError, OSError) as e:
        print(f"[hunyuan-omni] postprocess skipped ({e.__class__.__name__}: {e})")

    mesh.export(args.out)
    print(f"[hunyuan-omni] wrote mesh -> {args.out}")


if __name__ == "__main__":
    main()
