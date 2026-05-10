"""Chunked write/read for the big concatenated point clouds.

The viewer streams these clouds (scene_pointmap.npz, object_pointmap/*.npz)
and creates one THREE.Points per chunk so it can render progressively as
each chunk arrives. We split at write time into a manifest + N small npz
files; each chunk has the same schema as the original single-file form
(`pts3d` float16 (M,3), `rgb` uint8 (M,3), `conf` float16 (M,)).

Manifest layout (`<basename>_chunks.json`):
    {
      "version": 1,
      "totalPoints": <int>,
      "chunkSize": <int>,         # nominal chunk size in points
      "chunks": [
        {"file": "<basename>_000.npz", "points": <int>, "bytes": <int>},
        ...
      ]
    }

Chunk files live next to the manifest (`<basename>_NNN.npz`).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np


DEFAULT_CHUNK_SIZE = 500_000


def manifest_path(out_dir: Path, basename: str) -> Path:
    return out_dir / f"{basename}_chunks.json"


def chunk_path(out_dir: Path, basename: str, idx: int) -> Path:
    return out_dir / f"{basename}_{idx:03d}.npz"


def delete_chunked_pointcloud(out_dir: Path, basename: str) -> None:
    """Remove the manifest and any matching <basename>_NNN.npz chunks."""
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return
    mp = manifest_path(out_dir, basename)
    if mp.exists():
        mp.unlink()
    pat = re.compile(rf"^{re.escape(basename)}_\d{{3}}\.npz$")
    for entry in out_dir.iterdir():
        if pat.match(entry.name):
            entry.unlink()


def save_chunked_pointcloud(
    out_dir: Path,
    basename: str,
    pts: np.ndarray,
    rgb: np.ndarray,
    conf: np.ndarray,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Path:
    """Write `pts/rgb/conf` to `<out_dir>/<basename>_{NNN}.npz` chunks plus a manifest.

    Returns the manifest path.

    Casts pts/conf to float16 and rgb to uint8 (matches the prior single-file
    layout). Wipes any existing chunks/manifest with the same basename first
    so a smaller second write can't leave stale chunks behind.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    delete_chunked_pointcloud(out_dir, basename)

    n = int(pts.shape[0])
    if n == 0:
        # Still write an empty manifest so the loader can report 0 points
        # without 404ing.
        manifest = {"version": 1, "totalPoints": 0, "chunkSize": chunk_size, "chunks": []}
        mp = manifest_path(out_dir, basename)
        mp.write_text(json.dumps(manifest))
        return mp

    if rgb.shape[0] != n or conf.shape[0] != n:
        raise ValueError(
            f"pts/rgb/conf length mismatch: {n}/{rgb.shape[0]}/{conf.shape[0]}"
        )

    chunks_meta = []
    for ci, start in enumerate(range(0, n, chunk_size)):
        end = min(start + chunk_size, n)
        cp = chunk_path(out_dir, basename, ci)
        np.savez_compressed(
            cp,
            pts3d=pts[start:end].astype(np.float16, copy=False),
            rgb=rgb[start:end].astype(np.uint8, copy=False),
            conf=conf[start:end].astype(np.float16, copy=False),
        )
        chunks_meta.append({
            "file": cp.name,
            "points": int(end - start),
            "bytes": int(os.path.getsize(cp)),
        })

    manifest = {
        "version": 1,
        "totalPoints": n,
        "chunkSize": int(chunk_size),
        "chunks": chunks_meta,
    }
    mp = manifest_path(out_dir, basename)
    mp.write_text(json.dumps(manifest))
    return mp


def load_chunked_pointcloud(
    out_dir: Path,
    basename: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all chunks back into a single (pts, rgb, conf) tuple.

    For tools that need the full cloud (e.g. align_scene re-transforms);
    streaming consumers should iterate `iter_chunked_pointcloud` instead.
    """
    out_dir = Path(out_dir)
    mp = manifest_path(out_dir, basename)
    if not mp.exists():
        raise FileNotFoundError(f"manifest not found: {mp}")
    manifest = json.loads(mp.read_text())
    pts_parts, rgb_parts, conf_parts = [], [], []
    for c in manifest["chunks"]:
        with np.load(out_dir / c["file"]) as data:
            pts_parts.append(data["pts3d"])
            rgb_parts.append(data["rgb"])
            conf_parts.append(data["conf"])
    if not pts_parts:
        empty_pts = np.zeros((0, 3), dtype=np.float16)
        empty_rgb = np.zeros((0, 3), dtype=np.uint8)
        empty_conf = np.zeros((0,), dtype=np.float16)
        return empty_pts, empty_rgb, empty_conf
    return (
        np.concatenate(pts_parts, axis=0),
        np.concatenate(rgb_parts, axis=0),
        np.concatenate(conf_parts, axis=0),
    )


def iter_chunked_pointcloud(out_dir: Path, basename: str):
    """Yield (chunk_index, pts, rgb, conf) for each chunk on disk."""
    out_dir = Path(out_dir)
    mp = manifest_path(out_dir, basename)
    manifest = json.loads(mp.read_text())
    for ci, c in enumerate(manifest["chunks"]):
        with np.load(out_dir / c["file"]) as data:
            yield ci, data["pts3d"], data["rgb"], data["conf"]


def chunked_pointcloud_exists(out_dir: Path, basename: str) -> bool:
    return manifest_path(Path(out_dir), basename).exists()
