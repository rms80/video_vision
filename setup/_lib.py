"""Shared helpers for setup/ scripts.

Stdlib-only so it can be imported by 00_venv.py before the venv exists.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def models_dir() -> Path:
    return repo_root() / "models"


def venv_dir() -> Path:
    return models_dir() / ".venv"


def venv_python() -> Path:
    if sys.platform == "win32":
        return venv_dir() / "Scripts" / "python.exe"
    return venv_dir() / "bin" / "python"


def assert_venv() -> None:
    py = venv_python()
    if not py.exists():
        sys.exit(
            f"[setup] venv not found at {py}\n"
            f"        Run: python setup/00_venv.py"
        )


def run_in_venv(args: list[str], check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run([str(venv_python()), *args], check=check, **kwargs)


def pip_install(*packages: str, index_url: str | None = None,
                extra_index_url: str | None = None) -> None:
    args = ["-m", "pip", "install"]
    if index_url:
        args.extend(["--index-url", index_url])
    if extra_index_url:
        args.extend(["--extra-index-url", extra_index_url])
    args.extend(packages)
    run_in_venv(args)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, sha256: str | None = None,
             force: bool = False) -> Path:
    dest = Path(dest)
    if dest.exists() and not force:
        if sha256 is None or _sha256(dest) == sha256:
            print(f"[download] skip (already present): {dest}")
            return dest
        print(f"[download] hash mismatch, re-downloading: {dest}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[download] {url} -> {dest}")

    req = urllib.request.Request(url, headers={"User-Agent": "video_vision-setup/0"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1 << 20
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 / total
                print(f"\r[download]   {downloaded / (1 << 20):.1f} / "
                      f"{total / (1 << 20):.1f} MiB ({pct:.1f}%)",
                      end="", flush=True)
        if total:
            print()

    if sha256 is not None:
        actual = _sha256(tmp)
        if actual != sha256:
            tmp.unlink()
            sys.exit(f"[download] hash mismatch: expected {sha256}, got {actual}")

    tmp.replace(dest)
    return dest


def clone_repo(url: str, dest: Path, commit: str | None = None,
               recursive: bool = False, force: bool = False) -> Path:
    dest = Path(dest)
    if force and dest.exists():
        print(f"[git] removing {dest}")
        shutil.rmtree(dest)

    if not dest.exists():
        print(f"[git] clone {url} -> {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["git", "clone"]
        if recursive:
            cmd.append("--recursive")
        cmd.extend([url, str(dest)])
        subprocess.run(cmd, check=True)
    else:
        print(f"[git] {dest} exists, fetching")
        subprocess.run(["git", "-C", str(dest), "fetch", "--all", "--tags"],
                       check=True)

    if commit:
        current = subprocess.run(
            ["git", "-C", str(dest), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if current == commit or current.startswith(commit):
            print(f"[git] already at {commit}")
        else:
            print(f"[git] checkout {commit}")
            subprocess.run(["git", "-C", str(dest), "checkout", commit], check=True)
            if recursive:
                subprocess.run(
                    ["git", "-C", str(dest), "submodule", "update", "--init", "--recursive"],
                    check=True,
                )

    return dest


def hf_snapshot(repo_id: str, allow_patterns: list[str] | None = None,
                local_dir: Path | None = None) -> None:
    """Pre-cache a HuggingFace repo. Requires huggingface_hub in the venv."""
    assert_venv()
    code = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id={repo_id!r}, "
        f"allow_patterns={allow_patterns!r}, "
        f"local_dir={(str(local_dir) if local_dir else None)!r})"
    )
    run_in_venv(["-c", code])
