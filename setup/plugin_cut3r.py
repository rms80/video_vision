"""Clone CUT3R, download its checkpoint, and install its Python deps.

Run after 00_venv.py:

    python setup/plugin_cut3r.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib  # noqa: E402

REPO_URL = "https://github.com/CUT3R/CUT3R.git"
COMMIT = "8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf"
CKPT_NAME = "cut3r_512_dpt_4_64.pth"
CKPT_DRIVE_URL = "https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link"
PATCHES_DIR = Path(__file__).resolve().parent / "patches"
PATCHES = [
    PATCHES_DIR / "cut3r-fallback-rope-and-load.patch",
    PATCHES_DIR / "cut3r-curope-modern-pytorch.patch",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up CUT3R.")
    parser.add_argument("--force", action="store_true",
                        help="Re-clone, re-download the checkpoint, and reinstall deps")
    args = parser.parse_args()

    _lib.assert_venv()

    cut3r_dir = _lib.models_dir() / "external" / "cut3r"
    _lib.clone_repo(REPO_URL, cut3r_dir, commit=COMMIT, force=args.force)
    for patch in PATCHES:
        _lib.apply_patch(patch, cut3r_dir)

    ckpt_path = cut3r_dir / "src" / CKPT_NAME
    if ckpt_path.exists() and not args.force:
        print(f"[cut3r] checkpoint already present: {ckpt_path}")
    else:
        if args.force and ckpt_path.exists():
            ckpt_path.unlink()
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[cut3r] installing gdown to fetch {CKPT_NAME} from Google Drive")
        _lib.pip_install("gdown")
        print(f"[cut3r] downloading {CKPT_NAME} -> {ckpt_path}")
        try:
            _lib.run_in_venv([
                "-m", "gdown", CKPT_DRIVE_URL, "-O", str(ckpt_path),
            ])
        except subprocess.CalledProcessError:
            sys.exit(
                "\n[cut3r] gdown failed — Google Drive may be throttling this file.\n"
                f"        Download {CKPT_NAME} manually from:\n"
                f"          {CKPT_DRIVE_URL}\n"
                f"        and save it to:\n"
                f"          {ckpt_path}\n"
                "        Then re-run `python setup/plugin_cut3r.py` to finish setup."
            )

    req = cut3r_dir / "requirements.txt"
    if req.exists():
        print(f"[cut3r] installing {req.name} (skipping torch/torchvision/numpy pins)")
        _lib.install_requirements_filtered(req)
    else:
        print(f"[cut3r] no requirements.txt at {req}; skipping pip install")

    build_curope(cut3r_dir, force=args.force)

    print("[cut3r] done")


def build_curope(cut3r_dir: Path, force: bool) -> None:
    """Build the curope CUDA extension in-place. Falls back gracefully
    (CUT3R's slow Python RoPE2D path) if nvcc or the host C++ toolchain
    is unavailable, or if the build itself fails."""
    curope_dir = cut3r_dir / "src" / "croco" / "models" / "curope"
    existing = list(curope_dir.glob("curope.*.pyd")) + list(curope_dir.glob("curope.*.so"))
    if existing and not force:
        print(f"[curope] already built: {existing[0].name}")
        return
    if force:
        for f in existing:
            f.unlink()

    nvcc = _lib.find_nvcc()
    if nvcc is None:
        print("[curope] nvcc not found — skipping CUDA build; "
              "CUT3R will use the slow Python RoPE2D fallback")
        return

    if sys.platform == "win32":
        vcvars = _lib.find_vcvars()
        if vcvars is None:
            print("[curope] Visual Studio (vcvars64.bat) not found — skipping CUDA build; "
                  "CUT3R will use the slow Python RoPE2D fallback")
            return
        print(f"[curope] building CUDA extension via {vcvars.parent.name}/vcvars64.bat "
              "(this can take a few minutes)")
        cmd = (
            f'"{vcvars}" && set DISTUTILS_USE_SDK=1 && '
            f'cd /d "{curope_dir}" && '
            f'"{_lib.venv_python()}" setup.py build_ext --inplace'
        )
        result = subprocess.run(["cmd", "/c", cmd])
    else:
        print("[curope] building CUDA extension (this can take a few minutes)")
        result = subprocess.run(
            [str(_lib.venv_python()), "setup.py", "build_ext", "--inplace"],
            cwd=str(curope_dir),
        )

    if result.returncode != 0:
        print(f"[curope] build failed (exit {result.returncode}); "
              "CUT3R will use the slow Python RoPE2D fallback")
        return

    built = list(curope_dir.glob("curope.*.pyd")) + list(curope_dir.glob("curope.*.so"))
    if built:
        print(f"[curope] built: {built[0].name}")
    else:
        print("[curope] build reported success but no .pyd/.so found; "
              "CUT3R will use the slow Python RoPE2D fallback")


if __name__ == "__main__":
    main()
