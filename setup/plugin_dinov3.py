"""Clone DINOv3, install its Python deps, and pre-cache the ViT-L/16 weights.

DINOv3 has no installable pyproject — it is intended to be used either via
`torch.hub.load(repo_dir, ..., source='local')` or by adding the clone to
sys.path and importing `dinov3.*` directly. We clone to
`models/external/dinov3` so runners can do either.

The HuggingFace weight mirrors (`facebook/dinov3-*-pretrain-lvd1689m`) are
gated, so this script reuses the same token-check / help pattern as
`plugin_sam.py` and `plugin_vggtomega.py`.

Run after 00_venv.py:

    python setup/plugin_dinov3.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib  # noqa: E402

REPO_URL = "https://github.com/facebookresearch/dinov3.git"
COMMIT = "31703e4cbf1ccb7c4a72daa1350405f86754b6d1"
HF_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def _hf_token_present() -> bool:
    result = _lib.run_in_venv(
        ["-c", "import sys; from huggingface_hub import get_token; "
               "sys.exit(0 if get_token() else 1)"],
        check=False, capture_output=True,
    )
    return result.returncode == 0


def _print_auth_help() -> None:
    bin_dir = "Scripts" if sys.platform == "win32" else "bin"
    venv_hf = _lib.venv_dir() / bin_dir / "hf"
    bar = "=" * 72
    print()
    print(bar)
    print("[dinov3] HuggingFace weights are gated; authentication required")
    print(bar)
    print()
    print(f"  {HF_REPO} is a gated repository. To download it you need:")
    print()
    print("  1. Request access (one-time, manual approval by Meta):")
    print(f"       https://huggingface.co/{HF_REPO}")
    print("     Open the page while signed in to Hugging Face, accept the")
    print("     DINOv3 license, and wait for approval.")
    print()
    print("  2. Create a Hugging Face access token (read scope is enough):")
    print("       https://huggingface.co/settings/tokens")
    print()
    print("  3. Authenticate this project's venv with that token:")
    print(f"       {venv_hf} auth login")
    print()
    print("  4. Re-run this script:")
    print("       python setup/plugin_dinov3.py")
    print()
    print(bar)


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up DINOv3.")
    parser.add_argument("--force", action="store_true",
                        help="Re-clone and reinstall deps; HF cache is content-addressed")
    args = parser.parse_args()

    _lib.assert_venv()

    repo_dir = _lib.models_dir() / "external" / "dinov3"
    _lib.clone_repo(REPO_URL, repo_dir, commit=COMMIT, force=args.force)

    req = repo_dir / "requirements.txt"
    if req.exists():
        print(f"[dinov3] installing {req.name} (skipping torch/torchvision/numpy/pillow pins)")
        _lib.install_requirements_filtered(req)
    else:
        print(f"[dinov3] no requirements.txt at {req}; skipping pip install")

    print(f"[dinov3] caching {HF_REPO} via HuggingFace hub")
    if not _hf_token_present():
        print(f"[dinov3] no Hugging Face token configured in {_lib.venv_dir()}")
        _print_auth_help()
        sys.exit(1)

    try:
        _lib.hf_snapshot(HF_REPO)
    except subprocess.CalledProcessError:
        print(f"[dinov3] failed to download {HF_REPO}")
        print("[dinov3] this usually means your Hugging Face account has not")
        print(f"[dinov3] been granted access to {HF_REPO} yet, or the saved")
        print("[dinov3] token is stale.")
        _print_auth_help()
        sys.exit(1)

    print("[dinov3] done")


if __name__ == "__main__":
    main()
