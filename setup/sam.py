"""Download SAM2 and SAM3 weights into models/weights/.

SAM2 (sam2.1_l.pt) — Ultralytics GitHub release.
SAM3 (sam3.pt)    — facebook/sam3 HuggingFace repo.

Used by scripts/track_object.py (SAM2) and scripts/detect_object.py (SAM3).

Run after 00_venv.py:

    python setup/sam.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib  # noqa: E402

SAM2_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_l.pt"
SAM2_NAME = "sam2.1_l.pt"
SAM3_REPO = "facebook/sam3"
SAM3_NAME = "sam3.pt"


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
    print(f"[sam] downloading {SAM3_NAME} from {SAM3_REPO}")
    _lib.hf_snapshot(SAM3_REPO, allow_patterns=[SAM3_NAME], local_dir=weights_dir)

    print("[sam] done")


if __name__ == "__main__":
    main()
