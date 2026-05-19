"""Install Depth-Anything-3 from git and pre-cache its HF weights.

Run after 00_venv.py:

    python setup/plugin_da3.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib  # noqa: E402

REPO_URL = "https://github.com/ByteDance-Seed/Depth-Anything-3.git"
COMMIT = "41736238f5bced4debf3f2a12375d2466874866d"
HF_REPOS = [
    "depth-anything/DA3-LARGE-1.1",
    "depth-anything/DA3METRIC-LARGE",
]

# Extra skips beyond the install_requirements_filtered defaults:
#   - open3d: no Python 3.13 wheels on PyPI (only viz/export paths use it)
#   - xformers: latest hard-requires torch>=2.10, would clobber our CUDA build
#   - moviepy: requirements.txt is unversioned and pulls 2.x, but DA3 imports
#     moviepy.editor (removed in 2.x). Pinned explicitly below.
#   - pre-commit: dev tool, not needed at runtime
SKIP_DEPS = ("torch", "torchvision", "numpy", "pillow",
             "open3d", "xformers", "moviepy", "pre-commit")
MOVIEPY_PIN = "moviepy==1.0.3"


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up Depth-Anything-3.")
    parser.add_argument("--force", action="store_true",
                        help="Re-clone and reinstall; HF cache is content-addressed")
    args = parser.parse_args()

    _lib.assert_venv()

    repo_dir = _lib.models_dir() / "external" / "depth-anything-3"
    _lib.clone_repo(REPO_URL, repo_dir, commit=COMMIT, force=args.force)

    req = repo_dir / "requirements.txt"
    if req.exists():
        print(f"[da3] installing {req.name} (skipping {', '.join(SKIP_DEPS)})")
        _lib.install_requirements_filtered(req, skip_names=SKIP_DEPS)

    print(f"[da3] installing {MOVIEPY_PIN} (DA3 imports moviepy.editor, removed in 2.x)")
    _lib.pip_install(MOVIEPY_PIN)

    # addict is imported by depth_anything_3.model.da3 but missing from
    # upstream pyproject/requirements.txt.
    print("[da3] installing addict (missing from upstream requirements)")
    _lib.pip_install("addict")

    # DA3 pins requires-python="<=3.13" which excludes 3.13.x patch releases
    # under PEP 440; our venv is 3.13.3. --ignore-requires-python overrides it.
    # --no-deps because we already installed (filtered) requirements above.
    print("[da3] installing depth_anything_3 package (--no-deps)")
    _lib.run_in_venv([
        "-m", "pip", "install",
        "--ignore-requires-python", "--no-deps",
        str(repo_dir),
    ])

    for repo in HF_REPOS:
        print(f"[da3] caching {repo} via HuggingFace hub")
        _lib.hf_snapshot(repo)

    print("[da3] done")


if __name__ == "__main__":
    main()
