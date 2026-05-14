"""Detect an object containing a click point in an image using SAM3.

Usage:
    python detect_object.py <image_path> <click_x> <click_y> <label> <output_json_path>

Writes a JSON file with: { bbox, mask_png_base64, image_width, image_height, confidence }
"""
import sys
import os
import json
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from ultralytics.models.sam import SAM3SemanticPredictor
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _device import pick_device  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "models", "weights"))
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "sam3.pt")


def main():
    if len(sys.argv) < 6:
        print("Usage: detect_object.py <image> <x> <y> <label> <output_json>", file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[1]
    click_x = int(sys.argv[2])
    click_y = int(sys.argv[3])
    label = sys.argv[4]
    output_json = sys.argv[5]

    if not os.path.exists(WEIGHTS_PATH):
        print(f"Downloading sam3.pt to {WEIGHTS_PATH}...", file=sys.stderr)
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        hf_hub_download(repo_id="facebook/sam3", filename="sam3.pt", local_dir=WEIGHTS_DIR)

    device = pick_device()
    print(f"[detect] device={device}", file=sys.stderr, flush=True)

    predictor = SAM3SemanticPredictor(overrides=dict(
        conf=0.1,
        imgsz=1008,
        task="segment",
        mode="predict",
        model=WEIGHTS_PATH,
        save=False,
        verbose=False,
        device=device,
    ))

    predictor.set_image(image_path)
    results = predictor(text=[label])
    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        with open(output_json, "w") as f:
            json.dump({"error": f"No '{label}' detections found"}, f)
        return

    img_w, img_h = Image.open(image_path).size

    # Find the mask whose region contains the click point.
    # Prefer the smallest containing mask (most specific) if multiple match.
    candidates = []
    for i in range(len(r.boxes)):
        mask = r.masks.data[i]
        mh, mw = mask.shape
        # Scale click coords to mask coordinates
        mx = int(click_x * mw / img_w)
        my = int(click_y * mh / img_h)
        if 0 <= mx < mw and 0 <= my < mh and mask[my, mx] > 0:
            area = int(mask.sum().item())
            candidates.append((area, i))

    if not candidates:
        with open(output_json, "w") as f:
            json.dump({"error": f"No '{label}' mask contains click point ({click_x}, {click_y})"}, f)
        return

    # Pick smallest containing mask
    candidates.sort(key=lambda c: c[0])
    chosen_idx = candidates[0][1]

    # Bbox in original image pixels
    box = r.boxes.xyxy[chosen_idx]
    x1, y1, x2, y2 = [int(v.item()) for v in box]
    conf = float(r.boxes.conf[chosen_idx].item())

    # Build the mask at original image resolution
    mask_np = r.masks.data[chosen_idx].cpu().numpy()
    mask_img = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
    if mask_img.size != (img_w, img_h):
        mask_img = mask_img.resize((img_w, img_h), Image.NEAREST)
    mask_arr = np.array(mask_img)

    # RGBA overlay: semi-transparent green where mask is set, fully transparent elsewhere
    rgba_arr = np.zeros((img_h, img_w, 4), dtype=np.uint8)
    binary = mask_arr > 127
    rgba_arr[binary, 0] = 46
    rgba_arr[binary, 1] = 204
    rgba_arr[binary, 2] = 113
    rgba_arr[binary, 3] = 128
    rgba = Image.fromarray(rgba_arr, mode="RGBA")

    buf = BytesIO()
    rgba.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Save mask as standalone PNG next to the output JSON
    mask_png_path = os.path.join(os.path.dirname(output_json), "frame0_mask.png")
    rgba.save(mask_png_path, format="PNG")

    result = {
        "bbox": [x1, y1, x2, y2],
        "mask_png_base64": mask_b64,
        "image_width": img_w,
        "image_height": img_h,
        "confidence": conf,
        "label": label,
        "seed_x": click_x,
        "seed_y": click_y,
    }
    with open(output_json, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
