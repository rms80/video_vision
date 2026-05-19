"""Install COLMAP and pycolmap.

Windows: download the official CUDA build to models/tools/colmap/.
macOS:   brew install colmap (Homebrew required).
Linux:   apt-get install colmap (Debian/Ubuntu; needs sudo).

All: pip install pycolmap into the venv and verify the binary runs.

Run after 00_venv.py:

    python setup/plugin_colmap.py
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


def install_linux(force: bool) -> Path:
    have_apt = shutil.which("apt-get") is not None
    existing = shutil.which("colmap")
    if existing and not force:
        print(f"[colmap] colmap already on PATH: {existing}")
        binary = Path(existing)
    else:
        if not have_apt:
            sys.exit("[colmap] no apt-get found. Install colmap via your distro's "
                     "package manager and re-run.")
        print("[colmap] apt-get install colmap (may prompt for sudo password)")
        _apt_install("colmap", update=True)
        found = shutil.which("colmap")
        if found is None:
            sys.exit("[colmap] colmap not on PATH after apt-get install")
        binary = Path(found)

    if have_apt:
        _fixup_linux_shared_libs(binary)
    return binary


# Ubuntu 26.04 ships colmap 3.12.6-4 linked against libPoseLib.so but
# doesn't pull libposelib in as a hard dep. Map known missing soname
# -> apt package so we can self-heal.
_LINUX_LIB_TO_APT_PACKAGE = {
    "libPoseLib.so": "libposelib",
}


def _fixup_linux_shared_libs(binary: Path) -> None:
    """If `binary` reports missing shared libs that we know how to install
    via apt, install them. Silent no-op if ldd is missing or nothing is
    broken."""
    if shutil.which("ldd") is None:
        return
    result = subprocess.run(
        ["ldd", str(binary)], capture_output=True, text=True,
    )
    missing = []
    for line in (result.stdout or "").splitlines():
        # ldd format:  "\tlibFoo.so => not found"
        line = line.strip()
        if "=> not found" not in line:
            continue
        name = line.split("=>", 1)[0].strip()
        if name:
            missing.append(name)
    if not missing:
        return

    fixable = [n for n in missing if n in _LINUX_LIB_TO_APT_PACKAGE]
    unfixable = [n for n in missing if n not in _LINUX_LIB_TO_APT_PACKAGE]
    if fixable:
        pkgs = sorted({_LINUX_LIB_TO_APT_PACKAGE[n] for n in fixable})
        print(f"[colmap] colmap is missing shared libs ({', '.join(fixable)}); "
              f"installing: {', '.join(pkgs)}")
        _apt_install(*pkgs)
    if unfixable:
        print(f"[colmap] WARNING: colmap is missing shared libs with no known "
              f"apt fix: {', '.join(unfixable)}")


def _apt_install(*packages: str, update: bool = False) -> None:
    sudo = ["sudo"] if shutil.which("sudo") else []
    if update:
        subprocess.run([*sudo, "apt-get", "update"], check=True)
    subprocess.run([*sudo, "apt-get", "install", "-y", *packages], check=True)


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
    elif sys.platform == "linux":
        binary = install_linux(args.force)
    else:
        sys.exit(f"[colmap] unsupported platform: {sys.platform}")

    verify(binary)

    print("[colmap] installing pycolmap into venv")
    _lib.pip_install("pycolmap")

    print("[colmap] done")


if __name__ == "__main__":
    main()
