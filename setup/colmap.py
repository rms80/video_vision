"""Install COLMAP and pycolmap.

Windows: download the official CUDA build to models/tools/colmap/.
macOS:   brew install colmap (Homebrew required).

Both: pip install pycolmap into the venv and verify the binary runs.

Run after 00_venv.py:

    python setup/colmap.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib  # noqa: E402

COLMAP_VERSION = "4.0.3"
COLMAP_WIN_URL = (
    f"https://github.com/colmap/colmap/releases/download/"
    f"{COLMAP_VERSION}/colmap-x64-windows-cuda.zip"
)


def install_windows(force: bool) -> Path:
    target = _lib.models_dir() / "tools" / "colmap"
    bat = target / "COLMAP.bat"
    if bat.exists() and not force:
        print(f"[colmap] already installed at {target}")
        return bat

    if force and target.exists():
        shutil.rmtree(target)

    zip_dest = (_lib.models_dir() / "tools"
                / f"colmap-{COLMAP_VERSION}-windows-cuda.zip")
    _lib.download(COLMAP_WIN_URL, zip_dest, force=force)

    print(f"[colmap] extracting -> {target}")
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_dest) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        # Strip the top-level wrapper dir if all entries share one.
        first = members[0].filename if members else ""
        common = first.split("/", 1)[0] if "/" in first else ""
        if common and not all(m.filename.startswith(common + "/") for m in members):
            common = ""
        for m in members:
            rel = m.filename
            if common and rel.startswith(common + "/"):
                rel = rel[len(common) + 1:]
            if not rel:
                continue
            out = target / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)

    if not bat.exists():
        sys.exit(f"[colmap] expected {bat} after extracting; release layout may have changed")

    zip_dest.unlink()
    return bat


def install_macos(force: bool) -> Path:
    if shutil.which("brew") is None:
        sys.exit("[colmap] Homebrew not installed. Install from https://brew.sh and re-run.")

    existing = shutil.which("colmap")
    if existing and not force:
        print(f"[colmap] colmap already on PATH: {existing}")
        return Path(existing)

    print("[colmap] brew install colmap (this may take several minutes)")
    subprocess.run(["brew", "install", "colmap"], check=True)
    found = shutil.which("colmap")
    if found is None:
        sys.exit("[colmap] colmap not on PATH after brew install")
    return Path(found)


def verify(binary: Path) -> None:
    print(f"[colmap] verifying: {binary} help")
    result = subprocess.run(
        [str(binary), "help"], capture_output=True, text=True,
    )
    combined = (result.stdout or "") + (result.stderr or "")
    if "COLMAP" not in combined:
        sys.exit(f"[colmap] unexpected output from `{binary} help`:\n{combined}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Install COLMAP + pycolmap.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download / reinstall even if present")
    args = parser.parse_args()

    _lib.assert_venv()

    if sys.platform == "win32":
        binary = install_windows(args.force)
    elif sys.platform == "darwin":
        binary = install_macos(args.force)
    else:
        sys.exit(f"[colmap] unsupported platform: {sys.platform}")

    verify(binary)

    print("[colmap] installing pycolmap into venv")
    _lib.pip_install("pycolmap")

    print("[colmap] done")


if __name__ == "__main__":
    main()
