"""Run InfiniDepth (depth refiner) for per-frame metric depth, reusing
camera poses + depth from another scene plugin (Pi3, VGGT, DA3, ...).

InfiniDepth is a *conditional* depth refiner: given an RGB image, known
intrinsics, and a coarse depth prior, it queries a neural implicit field
to produce a sharp depth map at arbitrary resolution. We feed it poses
*and* coarse depth from an upstream scene plugin via that plugin's
`cameras.json` + `depthDir`, skipping the bundled MoGe-2 step entirely —
the upstream depth is already scale-consistent across frames.

The encoder runs at a per-video processing size that matches the source
aspect ratio (the upstream `load_image()` does a non-aspect-preserving
stretch, so feeding e.g. a 1:1 video through the default 4:3 size visibly
distorts depth). We target ~512² total pixels rounded to multiples of 14
and 16 (the two encoder patch sizes), then optionally super-resolve the
output by `--upscale` (1, 1.5, 2x). `scale_factor` is recorded in
cameras.json so downstream consumers can map source-pixel coordinates
into depth.

Usage:
    python run_infinidepth.py <scene_dir> \\
        --source-cameras-json <abs path> \\
        --source-depth-dir <abs path> \\
        [--upscale 1.5]

Outputs:
    <scene_dir>/infinidepth/cameras.json
    <scene_dir>/infinidepth/depth/NNNNNN.npz
    <scene_dir>/infinidepth/pointmap/NNNNNN.npz
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _device import pick_device  # noqa: E402
from _progress import progress  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
INFINIDEPTH_REPO = REPO_ROOT / "models" / "external" / "infinidepth"
INFINIDEPTH_WEIGHTS = REPO_ROOT / "models" / "weights" / "infinidepth"
DEPTH_CKPT = INFINIDEPTH_WEIGHTS / "depth" / "infinidepth.ckpt"

# Target for the *smaller* source dimension at the encoder's input resolution.
# Actual (H, W) is picked per-video by `pick_input_size`: aspect ratio is
# preserved and both dims snap to multiples of SIZE_MULTIPLE.
TARGET_MIN_DIM = 512
# LCM(14, 16) — divisible by both InfiniDepth's main encoder patch size (16)
# and MoGe-2's (14, kept for safety even though MoGe-2 is no longer used at
# inference). Slightly coarser snapping than just 16 but avoids any internal
# resize on either side.
SIZE_MULTIPLE = 112


def pick_input_size(src_h: int, src_w: int,
                    target_min: int = TARGET_MIN_DIM,
                    multiple: int = SIZE_MULTIPLE) -> tuple[int, int]:
    """Pick (H, W) preserving src aspect ratio with both dims as multiples
    of `multiple`. Targets min(H, W) ≈ target_min for significant
    downscaling, but never upscales: if src is already smaller than the
    target, snaps to the largest valid multiple ≤ src_min."""
    src_min = min(src_h, src_w)
    src_max = max(src_h, src_w)
    if src_min <= target_min:
        # Don't upscale — floor to the largest multiple that fits.
        min_dim = max(multiple, (src_min // multiple) * multiple)
    else:
        min_dim = max(multiple, round(target_min / multiple) * multiple)
    scale = min_dim / src_min
    max_dim = max(multiple, round(src_max * scale / multiple) * multiple)
    return (min_dim, max_dim) if src_h <= src_w else (max_dim, min_dim)


def depth_to_cam_points(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    x = (us - cx) * depth / fx
    y = (vs - cy) * depth / fy
    z = depth
    return np.stack([x, y, z], axis=-1)


def scale_K(K: np.ndarray, src_h: int, src_w: int, dst_h: int, dst_w: int) -> np.ndarray:
    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)
    out = K.astype(np.float64).copy()
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Run InfiniDepth with poses + depth from a source plugin")
    parser.add_argument("scene_dir", help="Path to _scene directory")
    parser.add_argument(
        "--source-cameras-json", required=True,
        help="Absolute path to the cameras.json of the upstream scene plugin "
             "supplying per-frame poses (e.g. .../pi3/cameras.json).",
    )
    parser.add_argument(
        "--source-depth-dir", required=True,
        help="Absolute path to the upstream plugin's depth directory "
             "(e.g. .../pi3/depth). Per-frame npz files keyed by 6-digit idx "
             "are used as the depth prior fed to InfiniDepth's implicit field.",
    )
    parser.add_argument(
        "--upscale", type=float, default=1.0,
        help="Output depth multiplier vs. the encoder's input resolution "
             "(1.0 = native, 1.5 / 2.0 = super-resolved).",
    )
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    frames_dir = scene_dir / "frames"
    out_dir = scene_dir / "infinidepth"
    depth_dir = out_dir / "depth"
    pointmap_dir = out_dir / "pointmap"
    depth_dir.mkdir(parents=True, exist_ok=True)
    pointmap_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("[infinidepth] ERROR: CUDA is required", file=sys.stderr)
        sys.exit(1)
    if not DEPTH_CKPT.exists():
        print(f"[infinidepth] ERROR: depth ckpt missing: {DEPTH_CKPT}", file=sys.stderr)
        sys.exit(1)
    if not INFINIDEPTH_REPO.exists():
        print(f"[infinidepth] ERROR: InfiniDepth repo missing: {INFINIDEPTH_REPO}", file=sys.stderr)
        sys.exit(1)
    src_depth_dir = Path(args.source_depth_dir)
    if not src_depth_dir.exists():
        print(f"[infinidepth] ERROR: source depth dir missing: {src_depth_dir}", file=sys.stderr)
        sys.exit(1)
    if args.upscale <= 0:
        print(f"[infinidepth] ERROR: --upscale must be > 0 (got {args.upscale})", file=sys.stderr)
        sys.exit(1)

    # Source poses (from an upstream scene plugin) -----------------------
    src_cams_path = Path(args.source_cameras_json)
    if not src_cams_path.exists():
        print(f"[infinidepth] ERROR: source cameras.json not found: {src_cams_path}",
              file=sys.stderr)
        sys.exit(1)
    with open(src_cams_path) as f:
        src_cams = json.load(f)
    src_cam_w = int(src_cams["width"])
    src_cam_h = int(src_cams["height"])
    src_K = np.array(src_cams["K"], dtype=np.float64)
    src_frames = [fr for fr in src_cams["frames"] if fr.get("registered")]
    if not src_frames:
        print("[infinidepth] ERROR: source cameras.json has no registered frames",
              file=sys.stderr)
        sys.exit(1)

    # Frame resolution (the .jpg under <scene>/frames/) ------------------
    frames_json = scene_dir / "frames.json"
    if not frames_json.exists():
        print(f"[infinidepth] ERROR: frames.json missing: {frames_json}", file=sys.stderr)
        sys.exit(1)
    with open(frames_json) as f:
        frames_meta = json.load(f)
    frame_w = int(frames_meta["width"])
    frame_h = int(frames_meta["height"])

    # Source K is in src_cam_w x src_cam_h; rescale to frame .jpg resolution
    # (the "original" image space InfiniDepth's fx_org/fy_org/cx_org/cy_org refer to).
    frame_K = scale_K(src_K, src_cam_h, src_cam_w, frame_h, frame_w)
    print(f"[infinidepth] source K @ {src_cam_w}x{src_cam_h} -> frame K @ {frame_w}x{frame_h}")
    print(f"[infinidepth] frame fx={frame_K[0,0]:.1f} fy={frame_K[1,1]:.1f} "
          f"cx={frame_K[0,2]:.1f} cy={frame_K[1,2]:.1f}")

    input_size = pick_input_size(frame_h, frame_w)
    print(f"[infinidepth] picked input_size={input_size[1]}x{input_size[0]} "
          f"(min dim ~{TARGET_MIN_DIM}, multiple of {SIZE_MULTIPLE}, "
          f"aspect-matched to {frame_w}x{frame_h})")

    # Output (decoded depth) size = input_size * upscale, snapped to a
    # multiple of 16 so the model's coordinate sampler doesn't have to
    # round on a non-integer-pixel grid.
    out_H = max(16, int(round(input_size[0] * args.upscale / 16)) * 16)
    out_W = max(16, int(round(input_size[1] * args.upscale / 16)) * 16)
    output_size = (out_H, out_W)
    print(f"[infinidepth] upscale={args.upscale} -> output {out_W}x{out_H}")

    # open3d has no Python 3.13 wheel, but several InfiniDepth utils import
    # it at module load — including type annotations (`pcd: o3d.geometry.PointCloud`)
    # that get evaluated when the file is imported. The depth-only path never
    # actually *calls* into o3d (only the unused PLY save/load helpers do),
    # so we install a stub that vivifies any attribute access into another
    # stub. Real calls will fail with `TypeError` rather than silently NOP.
    if "open3d" not in sys.modules:
        import types

        class _O3DStub(types.ModuleType):
            def __getattr__(self, name):
                # Let real dunder lookups (e.g. __file__, __spec__) fall
                # through to AttributeError — Python's import / frame
                # inspection machinery treats their absence correctly,
                # but a stub object in their place crashes things like
                # inspect.getsourcefile().
                if name.startswith("__") and name.endswith("__"):
                    raise AttributeError(name)
                child = _O3DStub(f"{self.__name__}.{name}")
                setattr(self, name, child)
                return child

        sys.modules["open3d"] = _O3DStub("open3d")

    # Make the InfiniDepth package importable from the cloned repo.
    sys.path.insert(0, str(INFINIDEPTH_REPO))
    progress("Loading InfiniDepth model...")
    from inference_depth import DepthInferenceArgs, run_depth_inference, load_depth_model  # noqa: E402

    inf_args = DepthInferenceArgs(
        input_image_path="",  # set per-frame below
        input_depth_path=None,
        save_pcd=False,
        model_type="InfiniDepth",
        depth_model_path=str(DEPTH_CKPT),
        # MoGe-2 is unused at inference: we always pass override_gt_depth from
        # the upstream plugin, and intrinsics are always supplied — neither
        # code path loads the MoGe-2 checkpoint. Placeholder string keeps the
        # dataclass happy without touching disk.
        moge2_pretrained="",
        input_size=input_size,
        output_size=output_size,
        output_resolution_mode="specific",
        upsample_ratio=1,  # unused in "specific" mode
        enable_skyseg_model=False,
    )

    device = pick_device()
    print(f"[infinidepth] device={device}", flush=True)
    t0 = time.time()
    model, dev = load_depth_model(inf_args)
    progress(f"Model loaded in {time.time() - t0:.1f}s")

    scale_factor = ((out_W / frame_w) + (out_H / frame_h)) / 2

    # The output K corresponds to the depth map's pixel space.
    out_K = scale_K(frame_K, frame_h, frame_w, out_H, out_W)
    K_out_list = out_K.tolist()
    print(f"[infinidepth] output {out_W}x{out_H} fx={out_K[0,0]:.1f} fy={out_K[1,1]:.1f} "
          f"cx={out_K[0,2]:.1f} cy={out_K[1,2]:.1f}")

    frames_out = []
    N = len(src_frames)
    for i, src_fr in enumerate(src_frames):
        idx = int(src_fr["idx"])
        name = src_fr.get("name") or f"{idx:06d}.jpg"
        frame_path = frames_dir / name
        if not frame_path.exists():
            print(f"[infinidepth] WARNING: frame {frame_path} missing — skipping", file=sys.stderr)
            continue

        src_depth_path = src_depth_dir / f"{idx:06d}.npz"
        if not src_depth_path.exists():
            print(f"[infinidepth] WARNING: source depth {src_depth_path} missing — skipping",
                  file=sys.stderr)
            continue
        with np.load(src_depth_path) as npz:
            # Upstream depth dirs store the array under either "depth" or
            # (older runs) the first key — match whichever is present.
            key = "depth" if "depth" in npz.files else npz.files[0]
            src_depth = np.asarray(npz[key], dtype=np.float32)
        prior = torch.from_numpy(src_depth).to(dev)

        progress(f"Inferring depth {i + 1}/{N} ({name})")
        result = run_depth_inference(
            inf_args,
            model=model,
            device=dev,
            input_image_path=str(frame_path),
            fx_org=float(frame_K[0, 0]),
            fy_org=float(frame_K[1, 1]),
            cx_org=float(frame_K[0, 2]),
            cy_org=float(frame_K[1, 2]),
            override_gt_depth=prior,
        )

        # pred_depthmap is (1, 1, H, W) float32 on GPU.
        depth = result.pred_depthmap[0, 0].detach().cpu().numpy().astype(np.float32)
        if depth.shape != (out_H, out_W):
            # Defensive: model may resolve a slightly different output size
            # if input_size doesn't divide cleanly. Pick up actual shape.
            ah, aw = depth.shape
            print(f"[infinidepth] WARNING: depth shape {ah}x{aw} != expected {out_H}x{out_W}")
        np.savez_compressed(
            depth_dir / f"{idx:06d}.npz",
            depth=depth.astype(np.float16),
        )

        # Camera-space pointmap via the output-resolution K.
        actual_h, actual_w = depth.shape
        if (actual_h, actual_w) != (out_H, out_W):
            pts_K = scale_K(out_K, out_H, out_W, actual_h, actual_w)
        else:
            pts_K = out_K
        pts_cam = depth_to_cam_points(depth, pts_K)
        np.savez_compressed(
            pointmap_dir / f"{idx:06d}.npz",
            pts3d=pts_cam.astype(np.float16),
            conf=np.ones(depth.shape, dtype=np.float16),
        )

        frames_out.append({
            "idx": idx,
            "name": name,
            "registered": True,
            "R": src_fr["R"],
            "t": src_fr["t"],
            "sparse_obs": [],
        })

        del result
        # Each frame allocates fresh tensors on cuda; explicit empty_cache
        # keeps peak usage bounded across long sequences.
        if (i + 1) % 8 == 0:
            free_cuda()

    cameras = {
        "model": "InfiniDepth",
        "depth_checkpoint": str(DEPTH_CKPT.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_cameras": str(src_cams_path),
        "source_depth_dir": str(src_depth_dir),
        "source_model": src_cams.get("model"),
        "upscale": float(args.upscale),
        "input_width": input_size[1],
        "input_height": input_size[0],
        "width": out_W,
        "height": out_H,
        "source_width": frame_w,
        "source_height": frame_h,
        "scale_factor": scale_factor,
        "K": K_out_list,
        "k1": 0.0,
        "num_points": 0,
        "num_registered": len(frames_out),
        "frames": frames_out,
    }
    cam_path = out_dir / "cameras.json"
    with open(cam_path, "w") as f:
        json.dump(cameras, f, indent=2)
    print(f"[infinidepth] Wrote {cam_path}")
    print(f"[infinidepth] Wrote {len(frames_out)} depth maps to {depth_dir}")
    print(f"[infinidepth] Wrote {len(frames_out)} pointmaps to {pointmap_dir}")


if __name__ == "__main__":
    main()
