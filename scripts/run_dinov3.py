"""Run DINOv3 dense patch features over video frames.

Stores per-frame fp16 patch grids under <scene_dir>/dinov3/ so the
frontend can hover a pixel and compute patch-level cosine similarity
on the fly. This is a video-level analysis that does NOT fit the scene
plugin shape (no cameras / depth / pointmap) — it lives in its own
output directory alongside the existing scene plugins.

Usage:
    python run_dinov3.py <scene_dir> [--subsample N] [--scaling F]
        [--model facebook/dinov3-vitl16-pretrain-lvd1689m]

`--scaling` is a multiplier on the source resolution (1.0 = native, 0.5 =
half on each axis, etc.). Default 0.5.

Reads <scene_dir>/frames/NNNNNN.jpg (produced by extract_frames.py).

Outputs:
    <scene_dir>/dinov3/meta.json     — model, grid, indices, dtype
    <scene_dir>/dinov3/NNNNNN.npz    — { patches: (Hg, Wg, D) fp16 }
"""

import sys
import os
import json
import argparse
import glob
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _device import pick_device  # noqa: E402
from _progress import progress  # noqa: E402

# Add the local dinov3 clone to sys.path so the architecture is importable.
# Only used with --random-weights for development smoke-testing; the normal
# path loads weights via transformers from the HF cache.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DINOV3_SRC = os.path.join(_REPO_ROOT, "models", "external", "dinov3")
if os.path.isdir(_DINOV3_SRC):
    sys.path.insert(0, _DINOV3_SRC)

# DINOv3 uses ImageNet normalization.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Patch size of every ViT-*/16 backbone. Other DINOv3 backbones (ConvNeXt,
# ViT-7B) would need adjustment; the runner is ViT-/16 only for now.
PATCH_SIZE = 16


def make_target_size(src_w: int, src_h: int, scaling: float, patch: int) -> tuple[int, int]:
    """Return (W, H) = round(source * scaling) snapped to a multiple of
    `patch` on each axis. Aspect ratio is preserved up to integer rounding."""
    out_w = max(patch, round(src_w * scaling / patch) * patch)
    out_h = max(patch, round(src_h * scaling / patch) * patch)
    return out_w, out_h


def load_hf_model(model_id: str, device: str):
    """Load a DINOv3 backbone from the HF cache via transformers. Returns
    (model, forward_fn) where forward_fn(x) -> patch_tokens (B, n_patches, D)
    in the same order as flatten(grid_h, grid_w)."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_id).to(device).eval()

    def forward(x: torch.Tensor, n_patches: int) -> torch.Tensor:
        out = model(pixel_values=x)
        # last_hidden_state: (B, 1 + n_storage + n_patches, D). Storage
        # tokens go right after CLS; patches are always at the tail.
        return out.last_hidden_state[:, -n_patches:, :]

    return model, forward


def load_local_random_model(device: str):
    """Build a randomly-initialized DINOv3 ViT-L/16 from the local clone.
    For smoke-testing the data path without the gated HF weights."""
    from dinov3.hub.backbones import dinov3_vitl16
    model = dinov3_vitl16(pretrained=False).to(device).eval()

    def forward(x: torch.Tensor, n_patches: int) -> torch.Tensor:
        out = model.forward_features(x)
        return out["x_norm_patchtokens"]

    return model, forward


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Compute DINOv3 dense patch features on video frames")
    parser.add_argument("scene_dir", help="Path to _scene directory")
    parser.add_argument("--subsample", type=int, default=2,
                        help="Use every Nth frame (default 2)")
    parser.add_argument("--scaling", type=float, default=0.5,
                        help="Scale factor on the source resolution before patching "
                             "(1.0 = full source size); each axis is rounded to a "
                             "multiple of patch size (default 0.5)")
    parser.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m",
                        help="HuggingFace model ID (default ViT-L/16 LVD-1689M)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Frames per forward pass (default 4)")
    parser.add_argument("--random-weights", action="store_true",
                        help="Use a randomly-initialized ViT-L/16 instead of "
                             "downloading weights; for development smoke tests only")
    args = parser.parse_args()

    scene_dir = args.scene_dir
    frames_dir = os.path.join(scene_dir, "frames")
    out_dir = os.path.join(scene_dir, "dinov3")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(scene_dir, "frames.json")) as f:
        meta_in = json.load(f)
    src_w, src_h = meta_in["width"], meta_in["height"]

    all_frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not all_frames:
        print("[dinov3] ERROR: no frames found", file=sys.stderr)
        sys.exit(1)

    frames_to_use = all_frames[::args.subsample]
    frame_indices = [int(os.path.splitext(os.path.basename(f))[0]) for f in frames_to_use]
    N = len(frames_to_use)
    print(f"[dinov3] {N} frames (subsample={args.subsample}) at {src_w}x{src_h}")

    target_w, target_h = make_target_size(src_w, src_h, args.scaling, PATCH_SIZE)
    grid_w, grid_h = target_w // PATCH_SIZE, target_h // PATCH_SIZE
    n_patches = grid_w * grid_h
    print(f"[dinov3] Target resolution: {target_w}x{target_h} "
          f"(scaling={args.scaling}, grid {grid_w}x{grid_h}, {n_patches} patches/frame)")

    device = pick_device()
    progress(f"Loading {'random ViT-L/16' if args.random_weights else args.model} on {device}...")
    print(f"[dinov3] Loading model on {device}...")
    if args.random_weights:
        model, forward_fn = load_local_random_model(device)
        model_id = "random-init-vitl16"
    else:
        model, forward_fn = load_hf_model(args.model, device)
        model_id = args.model

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    progress(f"Preprocessing {N} frames to {target_w}x{target_h}...")
    images = np.empty((N, 3, target_h, target_w), dtype=np.float32)
    for i, p in enumerate(frames_to_use):
        img = Image.open(p).convert("RGB").resize((target_w, target_h), Image.Resampling.BILINEAR)
        images[i] = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    imgs_tensor = torch.from_numpy(images)

    progress(f"Running DINOv3 on {N} frames (batch {args.batch_size})...")
    print("[dinov3] Running inference...")
    t0 = time.time()

    embed_dim: int | None = None
    feats = None  # allocated lazily once we know D
    for start in range(0, N, args.batch_size):
        end = min(start + args.batch_size, N)
        x = imgs_tensor[start:end].to(device, non_blocking=True)
        x = (x - mean) / std
        tokens = forward_fn(x, n_patches)  # (B, n_patches, D)
        if embed_dim is None:
            embed_dim = tokens.shape[-1]
            print(f"[dinov3] embed_dim={embed_dim}")
            feats = np.empty((N, grid_h, grid_w, embed_dim), dtype=np.float16)
        tokens = tokens.reshape(-1, grid_h, grid_w, embed_dim)
        feats[start:end] = tokens.to(torch.float16).cpu().numpy()
        progress(f"DINOv3 forward {end}/{N}")

    elapsed = time.time() - t0
    progress(f"DINOv3 inference done in {elapsed:.1f}s ({elapsed/N:.2f}s/frame)")
    print(f"[dinov3] Inference done in {elapsed:.1f}s ({elapsed/N:.2f}s/frame)")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    progress(f"Writing {N} feature maps...")
    for i, idx in enumerate(frame_indices):
        np.savez_compressed(
            os.path.join(out_dir, f"{idx:06d}.npz"),
            patches=feats[i],
        )

    meta_out = {
        "model": model_id,
        "patch_size": PATCH_SIZE,
        "scaling": args.scaling,
        "input_width": target_w,
        "input_height": target_h,
        "grid_width": grid_w,
        "grid_height": grid_h,
        "embed_dim": int(embed_dim),
        "subsample_every": args.subsample,
        "frame_indices": frame_indices,
        "source_width": src_w,
        "source_height": src_h,
        "dtype": "float16",
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"[dinov3] Wrote meta.json + {N} feature maps to {out_dir}")


if __name__ == "__main__":
    main()
