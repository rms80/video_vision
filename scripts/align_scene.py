"""Align COLMAP scene to a user-defined floor plane.

Usage:
    python align_scene.py <scene_dir> <points_json>

points_json is a JSON string (or @file) with:
    [{"x": px, "y": py, "frame": frame_idx}, ...]

For each point, unprojects the pixel to 3D using the depth map and camera
pose, fits a plane to the 3D points, and rotates cameras.json so the plane
normal aligns with +Y (gravity down).
"""

import sys
import os
import json
import argparse
from pathlib import Path

import numpy as np

from _pointcloud_io import (
    chunked_pointcloud_exists,
    iter_chunked_pointcloud,
    manifest_path,
    chunk_path,
)


def main():
    parser = argparse.ArgumentParser(description="Align scene to floor plane")
    parser.add_argument("scene_dir", help="Path to _scene directory")
    parser.add_argument("points_json", help="JSON array of {x, y, frame} or @filename")
    parser.add_argument("--cameras-dir", default="colmap",
                        help="Scene-relative dir holding cameras.json (also where "
                             "scene-level outputs like scene_pointmap chunks live)")
    parser.add_argument("--depth-dir", default="depthanythingv2",
                        help="Scene-relative dir holding per-frame NNNNNN.npz depth maps")
    parser.add_argument("--worldup-id", default="",
                        help="Worldup point set ID to stamp into cameras.json")
    args = parser.parse_args()

    scene_dir = args.scene_dir
    cameras_dir = args.cameras_dir
    depth_dir = args.depth_dir

    # Parse points
    pj = args.points_json
    if pj.startswith("@"):
        with open(pj[1:]) as f:
            points = json.load(f)
    else:
        points = json.loads(pj)

    if len(points) < 3:
        print("[align] ERROR: need at least 3 floor points", file=sys.stderr)
        sys.exit(1)

    # Load cameras.json
    cameras_path = os.path.join(scene_dir, cameras_dir, "cameras.json")
    with open(cameras_path) as f:
        cameras = json.load(f)

    K = np.array(cameras["K"], dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    cam_w = cameras["width"]
    cam_h = cameras["height"]
    source_w = cameras["source_width"]
    source_h = cameras["source_height"]
    scale_factor = cameras["scale_factor"]
    subsample = cameras.get("subsample_every", 1)

    # Build frame lookup: frame_idx -> frame data
    frame_lookup = {}
    for frame in cameras["frames"]:
        if frame["registered"] and frame["R"] is not None:
            frame_lookup[frame["idx"]] = frame

    # Unproject each floor point to 3D world coords
    world_pts = []
    for pt in points:
        px, py = pt["x"], pt["y"]  # in source (video) pixel coords
        frame_idx = pt["frame"]

        # Find nearest registered frame (frame indices in cameras.json
        # are already video frame indices)
        best_idx = None
        best_dist = float("inf")
        for idx in frame_lookup:
            d = abs(idx - frame_idx)
            if d < best_dist:
                best_dist = d
                best_idx = idx
        if best_idx is None:
            print(f"[align] WARNING: no registered frame near {frame_idx}, skipping")
            continue

        frame = frame_lookup[best_idx]
        R_cw = np.array(frame["R"], dtype=np.float64)
        t_cw = np.array(frame["t"], dtype=np.float64)

        # Load depth map
        depth_path = os.path.join(scene_dir, depth_dir, f"{best_idx:06d}.npz")
        if not os.path.exists(depth_path):
            print(f"[align] WARNING: no depth for frame {best_idx}, skipping")
            continue
        with np.load(depth_path) as data:
            depth = data["depth"].astype(np.float32)

        dh, dw = depth.shape

        # Scale source pixel coords to depth map coords
        u = px * scale_factor
        v = py * scale_factor

        # Clamp to valid range
        ui = int(round(np.clip(u, 0, dw - 1)))
        vi = int(round(np.clip(v, 0, dh - 1)))
        z = float(depth[vi, ui])
        if z <= 0:
            print(f"[align] WARNING: zero depth at ({px},{py}) frame {frame_idx}, skipping")
            continue

        # Unproject to camera space
        x_cam = z * (u - cx) / fx
        y_cam = z * (v - cy) / fy
        z_cam = z
        p_cam = np.array([x_cam, y_cam, z_cam])

        # Transform to world space: X_world = R_cw^T @ (X_cam - t_cw)
        p_world = R_cw.T @ (p_cam - t_cw)
        world_pts.append(p_world)
        print(f"[align] point ({px},{py}) frame {frame_idx} -> "
              f"colmap {best_idx} -> world {p_world}")

    if len(world_pts) < 3:
        print("[align] ERROR: fewer than 3 valid 3D points", file=sys.stderr)
        sys.exit(1)

    world_pts = np.array(world_pts)  # (N, 3)

    # Fit plane via SVD: center the points, find normal as smallest singular vector
    centroid = world_pts.mean(axis=0)
    centered = world_pts - centroid
    _, S, Vt = np.linalg.svd(centered)
    normal = Vt[-1]  # last row = direction of least variance = plane normal

    # Ensure normal points "up" (positive Y component in current frame)
    # Convention: camera Y-down ≈ gravity, so average camera Y in world should
    # have positive dot with the floor normal if normal points down.
    # We want normal to point in +Y direction (down = gravity direction).
    # Use average camera Y-down to decide sign.
    cam_y_dirs = []
    for f in cameras["frames"]:
        if f["registered"] and f["R"] is not None:
            R_cw = np.array(f["R"])
            cam_y_dirs.append(R_cw[1, :])
    avg_cam_y = np.mean(cam_y_dirs, axis=0)
    if np.dot(normal, avg_cam_y) < 0:
        normal = -normal

    print(f"[align] floor normal: {normal}")
    print(f"[align] plane fit singular values: {S}")

    # Build rotation from floor normal → +Y using Rodrigues
    target = np.array([0.0, 1.0, 0.0])
    v = np.cross(normal, target)
    s = np.linalg.norm(v)
    c = np.dot(normal, target)

    if s < 1e-8:
        R_align = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R_align = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))

    angle_deg = np.degrees(np.arccos(np.clip(c, -1, 1)))
    print(f"[align] floor alignment rotation: {angle_deg:.1f} degrees")

    # Compute yaw rotation around Y to align frame 0's forward with viewer +Z
    # Frame 0's forward in COLMAP world = R_cw[2,:] (camera +Z axis)
    # After floor alignment: forward_aligned = R_align @ forward_old
    # Viewer +Z = COLMAP -Z, so target is (0, 0, -1) in COLMAP world
    frame0 = None
    for f in cameras["frames"]:
        if f["registered"] and f["R"] is not None:
            frame0 = f
            break

    if frame0 is not None:
        R_cw_0 = np.array(frame0["R"], dtype=np.float64)
        forward_old = R_cw_0[2, :]  # camera Z axis in old world
        forward_aligned = R_align @ forward_old
        # Project onto XZ plane (drop Y component)
        fx, fz = forward_aligned[0], forward_aligned[2]
        # Current angle from COLMAP -Z direction
        # Target: (0, -1) in XZ, i.e. atan2(0, -1) = pi
        theta_current = np.arctan2(fx, fz)
        theta_target = np.arctan2(0.0, -1.0)  # = pi
        delta = theta_target - theta_current
        # Rotation around Y axis by delta
        cd, sd = np.cos(delta), np.sin(delta)
        R_yaw = np.array([
            [cd,  0, sd],
            [0,   1,  0],
            [-sd, 0, cd],
        ])
        R_total = R_yaw @ R_align
        yaw_deg = np.degrees(delta)
        print(f"[align] yaw correction: {yaw_deg:.1f} degrees (frame 0 forward -> viewer +Z)")
    else:
        R_total = R_align
        print("[align] WARNING: no registered frame 0, skipping yaw correction")

    # Compute translation offset so frame 0 is at the origin
    # Camera center in old world: c = -R_cw^T @ t_cw
    # In new (rotated) world: c_new = R_total @ c_old
    if frame0 is not None:
        R_cw_0 = np.array(frame0["R"], dtype=np.float64)
        t_cw_0 = np.array(frame0["t"], dtype=np.float64)
        c_old = -R_cw_0.T @ t_cw_0
        offset = R_total @ c_old
        print(f"[align] translating origin to frame 0 (offset: {offset})")
    else:
        offset = np.zeros(3)

    # Apply rotation + translation to all camera poses
    # New world: X_final = R_total @ X_old - offset
    # New R_cw = R_cw_old @ R_total^T
    # New t_cw = R_cw_new @ offset + t_cw_old
    R_total_T = R_total.T
    for f in cameras["frames"]:
        if not f["registered"] or f["R"] is None:
            continue
        R_cw = np.array(f["R"], dtype=np.float64)
        t_cw = np.array(f["t"], dtype=np.float64)
        R_new = R_cw @ R_total_T
        t_new = R_new @ offset + t_cw
        f["R"] = R_new.tolist()
        f["t"] = t_new.tolist()

    # Mark as gravity-aligned and write
    cameras["gravity_aligned"] = True
    if args.worldup_id:
        cameras["worldup_id"] = args.worldup_id
    with open(cameras_path, "w") as fp:
        json.dump(cameras, fp, indent=2)
    print(f"[align] wrote {cameras_path}")

    # Transform the scene point cloud chunks in place.
    # Points are stored in Three.js convention (x, -y, -z). For each chunk
    # we un-flip to OpenCV world, apply R_total, then re-flip — keeping the
    # working set bounded so we don't materialize a 100M-point array.
    src_dir = Path(scene_dir) / cameras_dir
    if chunked_pointcloud_exists(src_dir, "scene_pointmap"):
        total = 0
        for ci, pts_in, rgb, conf in iter_chunked_pointcloud(src_dir, "scene_pointmap"):
            pts = pts_in.astype(np.float32, copy=True)
            pts[:, 1] *= -1
            pts[:, 2] *= -1
            pts = np.ascontiguousarray((R_total @ pts.T).T)
            pts[:, 1] *= -1
            pts[:, 2] *= -1
            cp = chunk_path(src_dir, "scene_pointmap", ci)
            np.savez_compressed(
                cp,
                pts3d=pts.astype(np.float16),
                rgb=rgb,
                conf=conf,
            )
            total += pts.shape[0]
        # Refresh chunk byte sizes in the manifest so the client's progress
        # bar stays accurate after the rewrite.
        mp = manifest_path(src_dir, "scene_pointmap")
        manifest = json.loads(mp.read_text())
        for c in manifest["chunks"]:
            c["bytes"] = int(os.path.getsize(src_dir / c["file"]))
        mp.write_text(json.dumps(manifest))
        print(f"[align] transformed {mp} ({total:,} points across {len(manifest['chunks'])} chunks)")


if __name__ == "__main__":
    main()
