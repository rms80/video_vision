"""Run DepthAnythingV2 on COLMAP working frames, align to COLMAP scale.

Usage:
    python run_depth.py <scene_dir>

Reads:
    <scene_dir>/colmap/cameras.json       (per-frame sparse observations)
    <scene_dir>/colmap/images/NNNNNN.jpg  (downscaled frames used by COLMAP)

Writes:
    <scene_dir>/depthanythingv2/NNNNNN.npz  (per registered frame)
        depth:     float16 (H, W) metric depth in meters (raw DA2 output)
        a, b:      float   affine coefficients from calibration (diagnostic only)
        inliers:   int     number of sparse points used in fit
        rmse:      float   fit rmse on inliers

    Also rescales <scene_dir>/colmap/cameras.json to metric units (meters)
    by dividing all translations and sparse z values by the median affine
    scale factor.  Adds a "metric_scale" field to cameras.json.
"""
import sys
import os
import json
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _device import pick_device  # noqa: E402
from _progress import progress  # noqa: E402


MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"


def load_model():
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    device = pick_device()
    progress(f"Loading DepthAnythingV2 on {device}...")
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
    print(f"[depth] device={device}", flush=True)
    model = model.to(device).eval()
    return proc, model, device


def infer_depth(proc, model, device, img_pil: Image.Image) -> np.ndarray:
    inputs = proc(images=img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    # post_process_depth_estimation returns a list of dicts with "predicted_depth" at target size
    post = proc.post_process_depth_estimation(
        out, target_sizes=[(img_pil.height, img_pil.width)],
    )
    depth = post[0]["predicted_depth"].detach().cpu().numpy().astype(np.float32)
    return depth  # shape (H, W), metric depth in meters (higher = farther)


def ransac_affine(z_raw: np.ndarray, z_gt: np.ndarray, iters: int = 200,
                  rel_thresh: float = 0.1, rng=None):
    """Fit z_gt ≈ a*z_raw + b by RANSAC + LSQ on inliers. Returns (a, b, inliers_mask, rmse)."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(z_raw)
    if n < 3:
        # underdetermined — fall back to global scale
        a = float((z_gt * z_raw).sum() / max(1e-9, (z_raw * z_raw).sum()))
        b = 0.0
        pred = a * z_raw + b
        rmse = float(np.sqrt(np.mean((pred - z_gt) ** 2))) if n else 0.0
        return a, b, np.ones(n, dtype=bool), rmse

    best_inliers = None
    best_ab = (1.0, 0.0)
    for _ in range(iters):
        i, j = rng.choice(n, size=2, replace=False)
        if abs(z_raw[i] - z_raw[j]) < 1e-6:
            continue
        a = (z_gt[i] - z_gt[j]) / (z_raw[i] - z_raw[j])
        b = z_gt[i] - a * z_raw[i]
        pred = a * z_raw + b
        err = np.abs(pred - z_gt)
        thr = rel_thresh * np.maximum(z_gt, 1e-6)
        inliers = err < thr
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_ab = (float(a), float(b))

    # LSQ refit on inliers
    m = best_inliers
    if m.sum() >= 2:
        A = np.stack([z_raw[m], np.ones(int(m.sum()))], axis=1)
        sol, *_ = np.linalg.lstsq(A, z_gt[m], rcond=None)
        a, b = float(sol[0]), float(sol[1])
    else:
        a, b = best_ab
    pred = a * z_raw + b
    rmse = float(np.sqrt(np.mean((pred[m] - z_gt[m]) ** 2))) if m.sum() else float("inf")
    return a, b, m, rmse


def main():
    if len(sys.argv) < 2:
        print("Usage: run_depth.py <scene_dir>", file=sys.stderr)
        sys.exit(1)
    scene_dir = sys.argv[1]
    colmap_dir = os.path.join(scene_dir, "colmap")
    cameras_path = os.path.join(colmap_dir, "cameras.json")
    images_dir = os.path.join(colmap_dir, "images")
    out_dir = os.path.join(scene_dir, "depthanythingv2")
    os.makedirs(out_dir, exist_ok=True)

    cameras = json.load(open(cameras_path))
    frames = [f for f in cameras["frames"] if f["registered"]]
    print(f"[run_depth] {len(frames)} registered frames", flush=True)

    proc, model, device = load_model()
    print(f"[run_depth] model loaded on {device}", flush=True)

    # --- Pass 1: infer DA2 depth, compute per-frame affine fits to find scale ---
    rng = np.random.default_rng(0)
    align_stats = []
    raw_depths = {}  # name → (depth_raw, H, W)
    for k, f in enumerate(frames):
        name = f["name"]
        img_path = os.path.join(images_dir, name)
        img = Image.open(img_path).convert("RGB")
        depth_raw = infer_depth(proc, model, device, img)  # (H, W) float32, meters
        H, W = depth_raw.shape
        raw_depths[name] = (depth_raw, H, W)

        obs = np.asarray(f["sparse_obs"], dtype=np.float32)  # (N, 3): u, v, z_cam
        if len(obs) == 0:
            print(f"[run_depth] {name}: no sparse obs, skipping alignment", flush=True)
            continue
        u = np.clip(np.round(obs[:, 0]).astype(int), 0, W - 1)
        v = np.clip(np.round(obs[:, 1]).astype(int), 0, H - 1)
        z_gt = obs[:, 2]
        z_raw = depth_raw[v, u]

        # Fit z_cam(colmap) ≈ a * z_da2(meters) + b to find the COLMAP→metric
        # scale factor.  Filter outliers to 5–95 percentile.
        mask = np.isfinite(z_raw) & np.isfinite(z_gt) & (z_gt > 0)
        zr, zg = z_raw[mask], z_gt[mask]
        if len(zg) >= 20:
            zg_lo, zg_hi = np.quantile(zg, [0.05, 0.95])
            zr_lo, zr_hi = np.quantile(zr, [0.05, 0.95])
            keep = (zg >= zg_lo) & (zg <= zg_hi) & (zr >= zr_lo) & (zr <= zr_hi)
            zr, zg = zr[keep], zg[keep]

        a, b, inl_mask, rmse = ransac_affine(zr, zg, iters=500, rel_thresh=0.15, rng=rng)
        if a <= 0 and len(zr) >= 2:
            a_med = float(np.median(zg)) / max(1e-6, float(np.median(zr)))
            b = 0.0
            a = a_med

        align_stats.append({
            "name": name, "a": a, "b": b,
            "inliers": int(inl_mask.sum()), "total": int(mask.sum()),
            "rmse": rmse,
        })
        if k % 10 == 0 or k == len(frames) - 1:
            progress(f"Depth + scale align: frame {k+1}/{len(frames)}")
            print(f"[run_depth] {k+1}/{len(frames)} {name} a={a:.4g} b={b:.4g} "
                  f"inl={int(inl_mask.sum())}/{int(mask.sum())} rmse={rmse:.4g}", flush=True)

    # --- Compute global scale and rescale COLMAP to metric (meters) ---
    a_values = [s["a"] for s in align_stats]
    scale = float(np.median(a_values))
    print(f"[run_depth] COLMAP->metric scale factor: {scale:.4f} "
          f"(min a={min(a_values):.4f}, max a={max(a_values):.4f})", flush=True)

    # Rescale cameras.json: t → t/scale, sparse_obs z → z/scale, R and K unchanged
    for f in cameras["frames"]:
        if f["t"] is not None:
            f["t"] = [v / scale for v in f["t"]]
        f["sparse_obs"] = [[u, v, z / scale] for u, v, z in f["sparse_obs"]]
    cameras["metric_scale"] = scale
    with open(cameras_path, "w") as fp:
        json.dump(cameras, fp, indent=2)
    print(f"[run_depth] rescaled cameras.json to metric units (meters)", flush=True)

    # --- Pass 2: save DA2 metric depth directly (no affine alignment needed) ---
    progress(f"Writing {len(frames)} depth maps...")
    for f in frames:
        name = f["name"]
        if name not in raw_depths:
            continue
        depth_raw, H, W = raw_depths[name]
        stat = next((s for s in align_stats if s["name"] == name), None)

        out_path = os.path.join(out_dir, os.path.splitext(name)[0] + ".npz")
        np.savez_compressed(
            out_path,
            depth=depth_raw.astype(np.float16),
            a=np.float32(stat["a"] if stat else 1.0),
            b=np.float32(stat["b"] if stat else 0.0),
            inliers=np.int32(stat["inliers"] if stat else 0),
            rmse=np.float32(stat["rmse"] if stat else 0.0),
        )

    meta = {
        "model": MODEL_ID,
        "count": len(align_stats),
        "metric_scale": scale,
        "stats": align_stats,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fp:
        json.dump(meta, fp, indent=2)
    print(f"[run_depth] done: {len(align_stats)} depth maps (metric, meters)", flush=True)


if __name__ == "__main__":
    main()
