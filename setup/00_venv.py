"""Bootstrap the project venv at models/.venv with shared base deps.

Idempotent: skips work if the venv already exists. Pass --force to wipe
and recreate.

Run with the system Python (>= 3.11):

    python setup/00_venv.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import venv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib  # noqa: E402

MIN_PYTHON = (3, 11)
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu124"
BASE_DEPS = [
    "opencv-python",
    "numpy",
    "pillow",
    "transformers",
    "huggingface_hub",
    "tqdm",
    "ultralytics>=8.4.37",
]


def maybe_install_cuda_toolkit_linux() -> None:
    """On Linux with an NVIDIA GPU but no nvcc, apt-get install
    nvidia-cuda-toolkit + build-essential so CUT3R's curope CUDA
    extension can build on a later `python setup/plugin_cut3r.py --force`.
    No-op on Windows/macOS, no NVIDIA GPU, nvcc already present, or
    if apt-get isn't the package manager. Best-effort: prints and
    continues on failure rather than aborting venv setup."""
    if sys.platform != "linux":
        return
    if shutil.which("nvidia-smi") is None:
        print("[venv] no nvidia-smi on PATH; skipping CUDA toolkit install")
        return
    if shutil.which("nvcc") is not None:
        print(f"[venv] nvcc already on PATH ({shutil.which('nvcc')}); "
              "skipping CUDA toolkit install")
        return
    if shutil.which("apt-get") is None:
        print("[venv] NVIDIA GPU detected but apt-get unavailable; install a "
              "CUDA toolkit manually if you want curope-style extensions to build")
        return
    sudo = ["sudo"] if shutil.which("sudo") else []
    print("[venv] NVIDIA GPU detected without nvcc; installing "
          "nvidia-cuda-toolkit + build-essential (may prompt for sudo password)")
    try:
        subprocess.run([*sudo, "apt-get", "update"], check=True)
        subprocess.run(
            [*sudo, "apt-get", "install", "-y",
             "nvidia-cuda-toolkit", "build-essential"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[venv] CUDA toolkit install failed ({e}); continuing. "
              "CUT3R will fall back to slow Python RoPE2D.")
        return
    nvcc = shutil.which("nvcc")
    if nvcc:
        print(f"[venv] nvcc installed: {nvcc}")
    else:
        print("[venv] WARNING: apt-get succeeded but nvcc still not on PATH")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the project venv.")
    parser.add_argument("--force", action="store_true",
                        help="Delete and recreate the venv from scratch")
    args = parser.parse_args()

    if sys.version_info < MIN_PYTHON:
        sys.exit(f"[venv] Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required, "
                 f"got {sys.version_info.major}.{sys.version_info.minor}")

    venv_path = _lib.venv_dir()
    if args.force and venv_path.exists():
        print(f"[venv] removing existing {venv_path}")
        shutil.rmtree(venv_path)

    if not venv_path.exists():
        print(f"[venv] creating {venv_path} "
              f"(python {sys.version_info.major}.{sys.version_info.minor})")
        venv_path.parent.mkdir(parents=True, exist_ok=True)
        # On Linux, stdlib `venv` often fails: Debian/Ubuntu split out
        # python3-venv, and uv's standalone Pythons omit the bundled pip
        # wheel that `ensurepip` needs. If `uv` is on PATH, use it to
        # create a seeded venv (includes pip) using the current interpreter.
        uv = shutil.which("uv") if sys.platform == "linux" else None
        if uv:
            print(f"[venv] using uv ({uv}) to create seeded venv")
            subprocess.run(
                [uv, "venv", "--seed", "--python", sys.executable, str(venv_path)],
                check=True,
            )
        else:
            venv.EnvBuilder(with_pip=True, upgrade_deps=True).create(venv_path)
    else:
        print(f"[venv] reusing existing {venv_path}")

    print("[venv] upgrading pip")
    _lib.run_in_venv(["-m", "pip", "install", "--upgrade", "pip"])

    if sys.platform == "win32":
        print("[venv] installing torch + torchvision (CUDA 12.4 wheels)")
        _lib.pip_install("torch", "torchvision", index_url=TORCH_CUDA_INDEX)
    else:
        print(f"[venv] installing torch + torchvision (default index, "
              f"platform={sys.platform})")
        _lib.pip_install("torch", "torchvision")

    print(f"[venv] installing base deps: {', '.join(BASE_DEPS)}")
    _lib.pip_install(*BASE_DEPS)

    maybe_install_cuda_toolkit_linux()

    print(f"[venv] done. venv python: {_lib.venv_python()}")

    if sys.platform == "darwin":
        result = _lib.run_in_venv(
            ["-c", "import torch, sys; "
                   "sys.exit(0 if torch.backends.mps.is_available() else 1)"],
            check=False, capture_output=True,
        )
        if result.returncode != 0:
            print()
            print("[venv] WARNING: PyTorch MPS (Metal GPU) is not available.")
            print("[venv]   Models will fall back to CPU. On Apple Silicon this usually means")
            print("[venv]   torch was installed as x86_64 (Rosetta) instead of arm64, or you're")
            print("[venv]   on macOS < 12.3.")


if __name__ == "__main__":
    main()
