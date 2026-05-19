"""Pre-cache the DepthAnythingV2 metric indoor model used by run_depth.py.

Run after 00_venv.py:

    python setup/plugin_depthanythingv2.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib  # noqa: E402

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Pre-cache {MODEL_ID}.")
    parser.add_argument("--force", action="store_true",
                        help="(no-op; HF cache is content-addressed)")
    parser.parse_args()

    _lib.assert_venv()
    print(f"[depthanythingv2] caching {MODEL_ID} via HuggingFace hub")
    _lib.hf_snapshot(MODEL_ID)
    print("[depthanythingv2] done")


if __name__ == "__main__":
    main()
