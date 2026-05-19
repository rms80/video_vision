"""Run COLMAP on extracted video frames to get shared intrinsics + per-frame poses.

Usage:
    python run_colmap.py <scene_dir> [--every N] [--max-size PX]

Reads <scene_dir>/frames/NNNNNN.jpg (produced by extract_frames.py).
Stages downscaled/subsampled images at <scene_dir>/colmap/images/ and runs:
    feature_extractor (SIMPLE_PINHOLE, shared intrinsics, GPU SIFT)
    sequential_matcher (video-friendly)
    mapper

Outputs <scene_dir>/colmap/cameras.json:
    {
      model, width, height,             # working (downscaled) resolution
      source_width, source_height,      # native frame resolution
      K: 3x3,
      frames: [
        {idx, name, registered, R (3x3) or null, t (3) or null,
         sparse_obs: [[u, v, z_cam], ...]}  # only for registered frames
      ],
      num_points: int,
      scale_factor: float,              # working / source
      subsample_every: int,
    }
"""
import sys
import os
import json
import shutil
import argparse
import shutil
import subprocess
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from _progress import progress  # noqa: E402
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
# Windows uses the bundled standalone build; macOS/Linux use `colmap` on PATH
# (installed via `brew install colmap` on macOS, or distro package on Linux).
if sys.platform == "win32":
    COLMAP_BIN = os.path.join(REPO_ROOT, "models", "tools", "colmap", "COLMAP.bat")
else:
    COLMAP_BIN = shutil.which("colmap") or "colmap"


def run(cmd, cwd=None):
    print(f"[run_colmap] $ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}")


def _colmap_capabilities(binary: str):
    """Detect COLMAP's option namespace and CUDA support.

    COLMAP 4.x renamed `SiftExtraction`/`SiftMatching` to
    `FeatureExtraction`/`FeatureMatching`. Ubuntu 26.04 ships 3.12.x
    (Sift* names) built without CUDA; the Windows zip is 4.0.3 with
    CUDA. Returns dict with keys: extract_ns, match_ns, has_cuda.
    """
    r = subprocess.run(
        [binary, "feature_extractor", "--help"],
        capture_output=True, text=True,
    )
    help_text = (r.stdout or "") + (r.stderr or "")
    if "--FeatureExtraction.use_gpu" in help_text:
        extract_ns, match_ns = "FeatureExtraction", "FeatureMatching"
    else:
        extract_ns, match_ns = "SiftExtraction", "SiftMatching"

    # Bundle-adjuster namespace also changed between 3.x and 4.x
    # (BundleAdjustment.* -> BundleAdjustmentCeres.*).
    r2 = subprocess.run(
        [binary, "bundle_adjuster", "--help"],
        capture_output=True, text=True,
    )
    ba_help = (r2.stdout or "") + (r2.stderr or "")
    if "--BundleAdjustmentCeres." in ba_help:
        ba_ns = "BundleAdjustmentCeres"
    else:
        ba_ns = "BundleAdjustment"

    if sys.platform == "win32":
        # Our bundled Windows build is the CUDA zip.
        has_cuda = True
    else:
        has_cuda = False
        try:
            ldd = subprocess.run(["ldd", binary], capture_output=True, text=True)
            has_cuda = "libcudart" in (ldd.stdout or "")
        except FileNotFoundError:
            pass
    return {"extract_ns": extract_ns, "match_ns": match_ns,
            "ba_ns": ba_ns, "has_cuda": has_cuda}


def stage_images(scene_dir: str, every: int, max_size: int):
    src_dir = os.path.join(scene_dir, "frames")
    dst_dir = os.path.join(scene_dir, "colmap", "images")
    os.makedirs(dst_dir, exist_ok=True)
    for f in os.listdir(dst_dir):
        os.remove(os.path.join(dst_dir, f))

    meta = json.load(open(os.path.join(scene_dir, "frames.json")))
    src_w, src_h = meta["width"], meta["height"]
    long_edge = max(src_w, src_h)
    if max_size and long_edge > max_size:
        scale = max_size / long_edge
    else:
        scale = 1.0
    work_w = int(round(src_w * scale))
    work_h = int(round(src_h * scale))

    total_to_stage = len(range(0, meta["frame_count"], every))
    progress(f"Staging {total_to_stage} images at {work_w}x{work_h}...")
    staged = []
    for idx in range(0, meta["frame_count"], every):
        name = f"{idx:06d}.jpg"
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        img = cv2.imread(src)
        if scale != 1.0:
            img = cv2.resize(img, (work_w, work_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        staged.append((idx, name))
    print(f"[run_colmap] staged {len(staged)} images at {work_w}x{work_h} (scale={scale:.3f})", flush=True)
    return staged, work_w, work_h, scale, src_w, src_h


def run_colmap_pipeline(scene_dir: str):
    colmap_dir = os.path.join(scene_dir, "colmap")
    image_dir = os.path.join(colmap_dir, "images")
    db_path = os.path.join(colmap_dir, "database.db")
    sparse_dir = os.path.join(colmap_dir, "sparse")
    if os.path.exists(db_path):
        os.remove(db_path)
    if os.path.exists(sparse_dir):
        shutil.rmtree(sparse_dir)
    os.makedirs(sparse_dir, exist_ok=True)

    caps = _colmap_capabilities(COLMAP_BIN)
    gpu_flag = "1" if caps["has_cuda"] else "0"
    if not caps["has_cuda"]:
        print("[run_colmap] note: this COLMAP build has no CUDA support; "
              "SIFT extraction + matching will run on CPU", flush=True)

    progress(f"COLMAP feature extraction ({'GPU' if caps['has_cuda'] else 'CPU'} SIFT)...")
    run([COLMAP_BIN, "feature_extractor",
         "--image_path", image_dir,
         "--database_path", db_path,
         "--ImageReader.single_camera", "1",
         "--ImageReader.camera_model", "SIMPLE_RADIAL",
         f"--{caps['extract_ns']}.use_gpu", gpu_flag])

    progress("COLMAP sequential matching...")
    run([COLMAP_BIN, "sequential_matcher",
         "--database_path", db_path,
         f"--{caps['match_ns']}.use_gpu", gpu_flag,
         "--SequentialMatching.overlap", "30"])

    progress("COLMAP mapping (incremental SfM)...")
    run([COLMAP_BIN, "mapper",
         "--database_path", db_path,
         "--image_path", image_dir,
         "--output_path", sparse_dir])

    # Find the best reconstruction for final bundle adjustment
    sub = [d for d in os.listdir(sparse_dir)
           if os.path.isdir(os.path.join(sparse_dir, d))]
    if sub:
        import pycolmap
        best_dir, best_count = None, -1
        for s in sub:
            candidate = os.path.join(sparse_dir, s)
            try:
                r = pycolmap.Reconstruction(candidate)
                n = r.num_reg_images()
                if n > best_count:
                    best_count = n
                    best_dir = candidate
            except Exception:
                pass
        if best_dir:
            progress(f"COLMAP bundle adjustment ({best_count} images)...")
            print(f"[run_colmap] running final bundle adjustment on {best_dir} "
                  f"({best_count} images)", flush=True)
            run([COLMAP_BIN, "bundle_adjuster",
                 "--input_path", best_dir,
                 "--output_path", best_dir,
                 f"--{caps['ba_ns']}.max_num_iterations", "200",
                 f"--{caps['ba_ns']}.function_tolerance", "0"])


def export_cameras_json(scene_dir: str, staged, work_w, work_h, scale, src_w, src_h, every):
    import pycolmap
    colmap_dir = os.path.join(scene_dir, "colmap")
    sparse_dir = os.path.join(colmap_dir, "sparse")
    sub = [d for d in os.listdir(sparse_dir) if os.path.isdir(os.path.join(sparse_dir, d))]
    if not sub:
        raise RuntimeError("COLMAP mapper produced no reconstruction (empty sparse/). "
                           "Video may lack texture, parallax, or overlap.")
    # Pick the reconstruction with the most registered images
    best_dir, best_count = None, -1
    for s in sub:
        candidate = os.path.join(sparse_dir, s)
        r = pycolmap.Reconstruction(candidate)
        n = len(r.reg_image_ids())
        if n > best_count:
            best_count = n
            best_dir = candidate
    model_dir = best_dir
    rec = pycolmap.Reconstruction(model_dir)
    reg_ids = set(rec.reg_image_ids())
    num_reg = len(reg_ids)
    num_pts = len(rec.points3D)
    progress(f"Reconstruction: {num_reg}/{len(staged)} registered, {num_pts:,} points")
    print(f"[run_colmap] reconstruction: {num_reg}/{len(staged)} registered, "
          f"{num_pts} points", flush=True)

    if num_reg == 0:
        raise RuntimeError("COLMAP registered 0 images.")

    cams = list(rec.cameras.values())
    cam = cams[0]
    model_name = cam.model_name
    params = cam.params
    if model_name == "SIMPLE_RADIAL":
        f, cx, cy, k1 = float(params[0]), float(params[1]), float(params[2]), float(params[3])
    else:  # SIMPLE_PINHOLE fallback
        f, cx, cy = float(params[0]), float(params[1]), float(params[2])
        k1 = 0.0
    K = [[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]]

    name_to_image = {img.name: img for img in rec.images.values()}

    frames_out = []
    for idx, name in staged:
        img = name_to_image.get(name)
        if img is None or img.image_id not in reg_ids:
            frames_out.append({"idx": idx, "name": name, "registered": False,
                               "R": None, "t": None, "sparse_obs": []})
            continue
        # world→camera transform (cam_from_world is a method in pycolmap 4.x)
        rig = img.cam_from_world()
        R_np = np.asarray(rig.rotation.matrix())
        t_np = np.asarray(rig.translation)
        R = R_np.tolist()
        t = t_np.tolist()

        # Sparse observations visible in this frame (u, v, z_cam)
        sparse_obs = []
        for p2d in img.points2D:
            if not p2d.has_point3D():
                continue
            pt3 = rec.points3D[p2d.point3D_id]
            xyz_world = np.asarray(pt3.xyz)
            xyz_cam = R_np @ xyz_world + t_np
            z = float(xyz_cam[2])
            if z <= 0:
                continue
            u, v = float(p2d.xy[0]), float(p2d.xy[1])
            sparse_obs.append([u, v, z])

        frames_out.append({
            "idx": idx, "name": name, "registered": True,
            "R": R, "t": t, "sparse_obs": sparse_obs,
        })

    out = {
        "model": model_name,
        "width": work_w, "height": work_h,
        "source_width": src_w, "source_height": src_h,
        "scale_factor": scale,
        "subsample_every": every,
        "K": K,
        "k1": k1,
        "num_points": num_pts,
        "num_registered": num_reg,
        "frames": frames_out,
    }
    out_path = os.path.join(colmap_dir, "cameras.json")
    with open(out_path, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"[run_colmap] wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--max-size", type=int, default=1920)
    ap.add_argument("--export-only", action="store_true",
                    help="Skip staging + COLMAP run; only re-export cameras.json from existing sparse/")
    args = ap.parse_args()

    if not COLMAP_BIN or not os.path.exists(COLMAP_BIN):
        hint = ("Install via `python setup/plugin_colmap.py` (brew install colmap on macOS, "
                "standalone build on Windows)." )
        print(f"COLMAP binary not found at {COLMAP_BIN!r}. {hint}", file=sys.stderr)
        sys.exit(1)

    if args.export_only:
        # Rebuild staged list from existing colmap/images/ directory
        image_dir = os.path.join(args.scene_dir, "colmap", "images")
        names = sorted(os.listdir(image_dir))
        staged = [(int(os.path.splitext(n)[0]), n) for n in names]
        # Recover working/src resolution from a staged image + frames.json
        meta = json.load(open(os.path.join(args.scene_dir, "frames.json")))
        src_w, src_h = meta["width"], meta["height"]
        first = cv2.imread(os.path.join(image_dir, names[0]))
        work_h, work_w = first.shape[:2]
        scale = work_w / src_w
    else:
        staged, work_w, work_h, scale, src_w, src_h = stage_images(
            args.scene_dir, args.every, args.max_size)
        run_colmap_pipeline(args.scene_dir)
    export_cameras_json(args.scene_dir, staged, work_w, work_h, scale, src_w, src_h, args.every)


if __name__ == "__main__":
    main()
