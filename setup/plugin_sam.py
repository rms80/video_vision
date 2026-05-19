"""Download SAM2 and SAM3 weights into models/weights/.

SAM2 (sam2.1_l.pt) — Ultralytics GitHub release.
SAM3 (sam3.pt)    — facebook/sam3 HuggingFace repo (gated; see README).

Used by scripts/track_object.py (SAM2) and scripts/detect_object.py (SAM3).

Run after 00_venv.py:

    python setup/plugin_sam.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib  # noqa: E402

SAM2_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_l.pt"
SAM2_NAME = "sam2.1_l.pt"
SAM3_REPO = "facebook/sam3"
SAM3_NAME = "sam3.pt"


def _hf_token_present() -> bool:
    """Return True if the venv has a Hugging Face token available
    (via `HF_TOKEN` env var or a prior `huggingface-cli login`)."""
    result = _lib.run_in_venv(
        ["-c", "import sys; from huggingface_hub import get_token; "
               "sys.exit(0 if get_token() else 1)"],
        check=False, capture_output=True,
    )
    return result.returncode == 0


def _print_sam3_auth_help() -> None:
    bin_dir = "Scripts" if sys.platform == "win32" else "bin"
    venv_hf = _lib.venv_dir() / bin_dir / "hf"
    bar = "=" * 72
    print()
    print(bar)
    print("[sam] SAM3 download requires Hugging Face authentication")
    print(bar)
    print()
    print(f"  {SAM3_REPO} is a gated repository. To download {SAM3_NAME} you need:")
    print()
    print("  1. Request access (one-time, manual approval by Meta):")
    print(f"       https://huggingface.co/{SAM3_REPO}")
    print("     Open the page while signed in to Hugging Face, fill out the")
    print("     access form, and wait for approval.")
    print()
    print("  2. Create a Hugging Face access token (read scope is enough):")
    print("       https://huggingface.co/settings/tokens")
    print()
    print("  3. Authenticate this project's venv with that token:")
    print(f"       {venv_hf} auth login")
    print("     (Paste the token at the prompt. The old `huggingface-cli login`")
    print("      command is deprecated — use `hf auth login` instead.)")
    print()
    print("  4. Re-run this script:")
    print("       python setup/plugin_sam.py")
    print()
    print(bar)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SAM2 + SAM3 weights.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if the weights are already present")
    args = parser.parse_args()

    _lib.assert_venv()

    print("[sam] installing ultralytics into venv")
    _lib.pip_install("ultralytics>=8.4.37")

    weights_dir = _lib.models_dir() / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sam] downloading {SAM2_NAME} from Ultralytics release")
    _lib.download(SAM2_URL, weights_dir / SAM2_NAME, force=args.force)

    sam3_path = weights_dir / SAM3_NAME
    if args.force and sam3_path.exists():
        print(f"[sam] removing existing {sam3_path}")
        sam3_path.unlink()
    if sam3_path.exists():
        print(f"[sam] {sam3_path} already present, skipping SAM3 download")
        print("[sam] done")
        return

    print(f"[sam] downloading {SAM3_NAME} from {SAM3_REPO}")
    if not _hf_token_present():
        print(f"[sam] no Hugging Face token configured in {_lib.venv_dir()}")
        _print_sam3_auth_help()
        sys.exit(1)

    try:
        _lib.hf_snapshot(SAM3_REPO, allow_patterns=[SAM3_NAME], local_dir=weights_dir)
    except subprocess.CalledProcessError:
        print(f"[sam] failed to download {SAM3_NAME} from {SAM3_REPO}")
        print("[sam] this usually means your Hugging Face account has not been")
        print(f"[sam] granted access to {SAM3_REPO} yet, or the saved token is stale.")
        _print_sam3_auth_help()
        sys.exit(1)

    print("[sam] done")


if __name__ == "__main__":
    main()
