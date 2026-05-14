"""Track an object through a video using SAM2, seeded from a previous detection.

Usage:
    python track_object.py <video_path> <detect_json_path> <output_dir>

Reads detect.json (produced by detect_object.py) to get the frame-0 bbox, then
runs SAM2VideoPredictor over the whole video and writes:

    <output_dir>/track.json   — per-frame bbox + metadata
    <output_dir>/masks/<N>.png — per-frame binary mask PNGs

Note: we use the detection's bbox (not its mask) as the SAM2 prompt because the
ultralytics SAM2 video predictor's mask-input path has a 4D/3D tensor-shape bug
(the interpolate call inside `_use_mask_as_output` expects (N,C,H,W) but
`_prepare_prompts` squeezes masks down to (N,H,W)). The bbox path funnels
through the well-tested points-with-box-labels encoding and works correctly.
"""
import sys
import os
import json
from PIL import Image
import numpy as np
import torch
from ultralytics.models.sam import SAM2VideoPredictor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _device import pick_device  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "models", "weights"))
MODEL_NAME = "sam2.1_l.pt"
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, MODEL_NAME)


def load_seed_bbox(detect_json_path: str):
    """Return [x1, y1, x2, y2] from detect.json."""
    with open(detect_json_path, "r") as f:
        data = json.load(f)
    if "bbox" not in data:
        raise ValueError("detect.json has no bbox")
    return list(map(int, data["bbox"]))


def bbox_from_mask(mask: np.ndarray):
    """Return [x1, y1, x2, y2] bbox of nonzero pixels, or None if mask is empty."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def main():
    if len(sys.argv) < 4:
        print("Usage: track_object.py <video> <detect_json> <output_dir>", file=sys.stderr)
        sys.exit(1)

    video_path = sys.argv[1]
    detect_json_path = sys.argv[2]
    output_dir = sys.argv[3]

    os.makedirs(output_dir, exist_ok=True)
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    seed_bbox = load_seed_bbox(detect_json_path)
    print(f"[track] seed bbox={seed_bbox}", flush=True)

    device = pick_device()
    print(f"[track] device={device} model={MODEL_NAME}", flush=True)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    # Ultralytics resolves bare filenames against its model hub and caches locally;
    # passing an absolute path lets us keep the weights under models/weights/.
    model_arg = WEIGHTS_PATH if os.path.exists(WEIGHTS_PATH) else MODEL_NAME

    predictor = SAM2VideoPredictor(overrides=dict(
        conf=0.25,
        task="segment",
        mode="predict",
        imgsz=1024,
        model=model_arg,
        save=False,
        verbose=False,
        device=device,
    ))

    # SAM2VideoPredictor iterates frames and yields a Results per frame.
    # The seed bbox is consumed on frame 0; subsequent frames are propagated.
    results = predictor(source=video_path, bboxes=np.array([seed_bbox], dtype=np.float32), stream=True)

    frames_info = []
    video_w = None
    video_h = None

    for frame_idx, r in enumerate(results):
        if video_w is None:
            video_h, video_w = r.orig_shape[:2]

        entry = {"frame": frame_idx, "bbox": None, "has_mask": False}
        if r.masks is not None and len(r.masks.data) > 0:
            m = r.masks.data[0].cpu().numpy()
            if m.shape != (video_h, video_w):
                m_img = Image.fromarray((m * 255).astype(np.uint8), mode="L").resize(
                    (video_w, video_h), Image.NEAREST
                )
                m = (np.array(m_img) > 127).astype(np.uint8)
            else:
                m = (m > 0.5).astype(np.uint8)

            bbox = bbox_from_mask(m)
            entry["bbox"] = bbox
            entry["has_mask"] = bbox is not None

            if bbox is not None:
                rgba = np.zeros((video_h, video_w, 4), dtype=np.uint8)
                binary = m > 0
                rgba[binary, 0] = 46
                rgba[binary, 1] = 204
                rgba[binary, 2] = 113
                rgba[binary, 3] = 128
                Image.fromarray(rgba, mode="RGBA").save(
                    os.path.join(masks_dir, f"{frame_idx:06d}.png"), format="PNG"
                )

        frames_info.append(entry)
        if frame_idx % 30 == 0:
            print(f"[track] frame {frame_idx} bbox={entry['bbox']}", flush=True)

    out = {
        "video": os.path.basename(video_path),
        "image_width": int(video_w) if video_w else None,
        "image_height": int(video_h) if video_h else None,
        "frame_count": len(frames_info),
        "model": MODEL_NAME,
        "frames": frames_info,
    }
    track_json_path = os.path.join(output_dir, "track.json")
    with open(track_json_path, "w") as f:
        json.dump(out, f)
    print(f"[track] wrote {track_json_path} ({len(frames_info)} frames)", flush=True)


if __name__ == "__main__":
    main()
