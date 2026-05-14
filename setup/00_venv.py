"""Bootstrap the project venv at models/.venv with shared base deps.

Idempotent: skips work if the venv already exists. Pass --force to wipe
and recreate.

Run with the system Python (>= 3.11):

    python setup/00_venv.py
"""

from __future__ import annotations

import argparse
import shutil
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
