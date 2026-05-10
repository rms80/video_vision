"""Shared helpers for setup/ scripts.

Stdlib-only so it can be imported by 00_venv.py before the venv exists.
"""

from __future__ import annotations

import hashlib
import os
import re
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
            # -q suppresses git's detached-HEAD advice; we always pin a commit.
            subprocess.run(["git", "-C", str(dest), "checkout", "-q", commit], check=True)
            if recursive:
                subprocess.run(
                    ["git", "-C", str(dest), "submodule", "update", "--init", "--recursive"],
                    check=True,
                )

    return dest


def install_requirements_filtered(
    req_path: Path,
    skip_names: tuple[str, ...] = ("torch", "torchvision", "numpy", "pillow"),
) -> None:
    """Install a requirements.txt into the venv, skipping packages whose
    canonical name appears in skip_names. Use this for repo requirements
    that pin versions of packages we already control in the base venv
    (e.g. CUDA-built torch). URL/VCS entries (`pkg @ url`) are filtered
    by the leading name too."""
    pkgs: list[str] = []
    skipped: list[str] = []
    for raw in Path(req_path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name = re.split(r"[\s<>=!~\[@]", line, 1)[0].strip().lower()
        if name in skip_names:
            skipped.append(line)
            continue
        pkgs.append(line)
    if skipped:
        print(f"[setup] skipping pinned deps: {', '.join(skipped)}")
    if pkgs:
        pip_install(*pkgs)


def find_nvcc() -> Path | None:
    """Locate nvcc. Tries PATH, then CUDA_PATH/CUDA_HOME, then default
    Windows install paths. Returns None if not found."""
    found = shutil.which("nvcc")
    if found:
        return Path(found)
    for var in ("CUDA_PATH", "CUDA_HOME"):
        cuda = os.environ.get(var)
        if cuda:
            nvcc = Path(cuda) / "bin" / ("nvcc.exe" if sys.platform == "win32" else "nvcc")
            if nvcc.exists():
                return nvcc
    if sys.platform == "win32":
        pf = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        cuda_root = pf / "NVIDIA GPU Computing Toolkit" / "CUDA"
        if cuda_root.is_dir():
            for v in sorted(cuda_root.glob("v*"), reverse=True):
                nvcc = v / "bin" / "nvcc.exe"
                if nvcc.exists():
                    return nvcc
    return None


def find_vcvars() -> Path | None:
    """Locate vcvars64.bat on Windows. Tries vswhere.exe first, then
    standard install paths. Returns None on non-Windows or if not found."""
    if sys.platform != "win32":
        return None
    pfx86 = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    vswhere = pfx86 / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.is_file():
        try:
            r = subprocess.run(
                [str(vswhere), "-latest", "-products", "*",
                 "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                 "-property", "installationPath"],
                capture_output=True, text=True, check=True,
            )
            install = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
            if install:
                vcvars = Path(install) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                if vcvars.is_file():
                    return vcvars
        except subprocess.CalledProcessError:
            pass
    pf = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    for prog in (pf, pfx86):
        for year in ("2022", "2019"):
            for edition in ("Community", "Professional", "Enterprise", "BuildTools"):
                p = prog / "Microsoft Visual Studio" / year / edition / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                if p.is_file():
                    return p
    return None


def apply_patch(patch_path: Path, repo_dir: Path) -> None:
    """Apply a unified-diff patch to a local git checkout. Idempotent: if
    the patch already applies cleanly in reverse, treat it as already
    applied and skip."""
    patch_path = Path(patch_path)
    repo_dir = Path(repo_dir)
    reverse = subprocess.run(
        ["git", "-C", str(repo_dir), "apply", "--reverse", "--check", str(patch_path)],
        capture_output=True,
    )
    if reverse.returncode == 0:
        print(f"[patch] {patch_path.name} already applied to {repo_dir.name}")
        return
    print(f"[patch] applying {patch_path.name} to {repo_dir.name}")
    subprocess.run(
        ["git", "-C", str(repo_dir), "apply", str(patch_path)],
        check=True,
    )


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
