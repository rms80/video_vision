"""Extract all frames of a video to JPG for COLMAP / depth pipelines.

Usage:
    python extract_frames.py <video_path> <output_scene_dir>

Writes:
    <output_scene_dir>/frames/NNNNNN.jpg   (6-digit zero-padded, JPEG q=2)
    <output_scene_dir>/frames.json         (fps, frame_count, width, height, source)
"""
import sys
import os
import json
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _progress import progress  # noqa: E402


def main():
    if len(sys.argv) < 3:
        print("Usage: extract_frames.py <video_path> <output_scene_dir>", file=sys.stderr)
        sys.exit(1)

    video_path = sys.argv[1]
    scene_dir = sys.argv[2]
    frames_dir = os.path.join(scene_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    progress(f"Opening {os.path.basename(video_path)}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}", file=sys.stderr)
        sys.exit(2)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    idx = 0
    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 92]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out_path = os.path.join(frames_dir, f"{idx:06d}.jpg")
        cv2.imwrite(out_path, frame, jpeg_params)
        if idx % 50 == 0:
            print(f"[extract_frames] wrote frame {idx}", flush=True)
            if total > 0:
                progress(f"Extracting frames: {idx}/{total}")
            else:
                progress(f"Extracting frames: {idx}")
        idx += 1
    cap.release()

    meta = {
        "source": os.path.basename(video_path),
        "fps": fps,
        "frame_count": idx,
        "width": width,
        "height": height,
    }
    with open(os.path.join(scene_dir, "frames.json"), "w") as f:
        json.dump(meta, f, indent=2)

    progress(f"Extracted {idx} frames at {width}x{height}")
    print(f"[extract_frames] done: {idx} frames, {width}x{height} @ {fps:.2f}fps", flush=True)


if __name__ == "__main__":
    main()
