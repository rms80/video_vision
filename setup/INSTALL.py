"""Tk-based installer GUI: pick which setup scripts to run.

Same machinery as EVERYTHING.py (subprocess each setup/00_venv.py and
setup/plugin_*.py script in order), but with a checkbox UI so you can
choose exactly which steps to run instead of running them all.

Run with the system Python (>= 3.11), the same way you'd run EVERYTHING.py:

    python setup/INSTALL.py

Tkinter ships with Python on Windows and macOS. On Debian/Ubuntu install
the `python3-tk` package if `import tkinter` fails.
"""

from __future__ import annotations

import queue
import re
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import scrolledtext, ttk
except ImportError:
    if sys.platform.startswith("linux"):
        hint = (
            "Debian/Ubuntu:  sudo apt install python3-tk\n"
            "  Fedora/RHEL:    sudo dnf install python3-tkinter\n"
            "  Arch:           sudo pacman -S tk"
        )
    elif sys.platform == "darwin":
        hint = (
            "Reinstall Python from python.org, or `brew install python-tk`\n"
            "  if you are on Homebrew Python."
        )
    else:
        hint = "Reinstall Python from python.org — the official installer bundles Tk."
    sys.stderr.write(
        "setup/INSTALL.py needs Tkinter, which is not available in this Python.\n"
        f"  {hint}\n"
        "Or fall back to the headless installer: python setup/EVERYTHING.py\n"
    )
    sys.exit(1)

SETUP_DIR = Path(__file__).resolve().parent
VENV_SCRIPT = SETUP_DIR / "00_venv.py"

DEFAULT_CHECKED = {"00_venv", "plugin_colmap", "plugin_depthanythingv2", "plugin_pi3"}
GATED: dict[str, str] = {
    "plugin_sam": "https://huggingface.co/facebook/sam3",
    "plugin_vggtomega": "https://huggingface.co/facebook/VGGT-Omega",
}


def discover_plugins() -> list[Path]:
    return sorted(SETUP_DIR.glob("plugin_*.py"))


def short_doc(script: Path) -> str:
    try:
        text = script.read_text(encoding="utf-8")
    except OSError:
        return ""
    m = re.match(r'\s*"""(.*?)"""', text, flags=re.DOTALL)
    if not m:
        return ""
    body = m.group(1).strip()
    return body.splitlines()[0].strip() if body else ""


class InstallerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("video_vision installer")
        self.root.geometry("820x680")
        self.root.minsize(640, 480)

        self.scripts: list[Path] = [VENV_SCRIPT, *discover_plugins()]
        self.vars: dict[Path, tk.BooleanVar] = {}
        self.force = tk.BooleanVar(value=False)
        self.proc: subprocess.Popen[str] | None = None
        self.worker: threading.Thread | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._drain_log()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="Select setup scripts to run (in order, top to bottom):",
            font=("", 10, "bold"),
        ).pack(anchor="w")

        list_outer = ttk.Frame(outer, relief="sunken", borderwidth=1)
        list_outer.pack(fill="x", pady=(6, 6))

        for script in self.scripts:
            var = tk.BooleanVar(value=script.stem in DEFAULT_CHECKED)
            self.vars[script] = var
            row = ttk.Frame(list_outer, padding=(6, 2))
            row.pack(fill="x", anchor="w")
            ttk.Checkbutton(row, variable=var, text=script.stem, width=26).pack(
                side="left"
            )
            gate_url = GATED.get(script.stem)
            if gate_url:
                link = ttk.Label(
                    row,
                    text="Request Approval",
                    foreground="#0366d6",
                    cursor="hand2",
                    font=("TkDefaultFont", 9, "underline"),
                )
                link.pack(side="right", padx=(8, 6))
                link.bind(
                    "<Button-1>",
                    lambda _e, u=gate_url: webbrowser.open_new_tab(u),
                )
            ttk.Label(
                row,
                text=short_doc(script),
                foreground="#555",
                wraplength=440,
                justify="left",
            ).pack(side="left", padx=(8, 0), fill="x", expand=True)

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(0, 8))
        ttk.Button(controls, text="Select all", command=self._select_all).pack(
            side="left"
        )
        ttk.Button(controls, text="Select none", command=self._select_none).pack(
            side="left", padx=(6, 0)
        )
        ttk.Checkbutton(
            controls,
            text="--force (wipe and reinstall each step)",
            variable=self.force,
        ).pack(side="right")

        ttk.Label(outer, text="Log:").pack(anchor="w")
        self.log = scrolledtext.ScrolledText(
            outer, height=20, wrap="word", font=("Consolas", 9)
        )
        self.log.pack(fill="both", expand=True)
        self.log.configure(state="disabled")

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=(8, 0))
        self.install_btn = ttk.Button(
            actions, text="Install selected", command=self._on_install
        )
        self.install_btn.pack(side="left")
        self.cancel_btn = ttk.Button(
            actions, text="Cancel run", command=self._on_cancel, state="disabled"
        )
        self.cancel_btn.pack(side="left", padx=(6, 0))
        ttk.Button(actions, text="Close", command=self._on_close).pack(side="right")

    def _select_all(self) -> None:
        for v in self.vars.values():
            v.set(True)

    def _select_none(self) -> None:
        for v in self.vars.values():
            v.set(False)

    def _selected(self) -> list[Path]:
        return [s for s, v in self.vars.items() if v.get()]

    def _running(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def _append(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _drain_log(self) -> None:
        try:
            while True:
                self._append(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        self.root.after(80, self._drain_log)

    def _on_install(self) -> None:
        if self._running():
            return
        scripts = self._selected()
        if not scripts:
            self._append("[INSTALL] nothing selected\n")
            return
        self.install_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.worker = threading.Thread(
            target=self._run_all, args=(scripts, self.force.get()), daemon=True
        )
        self.worker.start()

    def _on_cancel(self) -> None:
        proc = self.proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass

    def _on_close(self) -> None:
        proc = self.proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass
        self.root.destroy()

    def _run_all(self, scripts: list[Path], force: bool) -> None:
        failures: list[tuple[Path, int]] = []
        for s in scripts:
            cmd = [sys.executable, str(s)]
            if force:
                cmd.append("--force")
            self.log_queue.put(
                f"\n{'=' * 72}\n[INSTALL] $ {' '.join(cmd)}\n{'=' * 72}\n"
            )
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            except OSError as e:
                self.log_queue.put(f"[INSTALL] failed to start {s.name}: {e}\n")
                failures.append((s, -1))
                continue
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self.log_queue.put(line)
            rc = self.proc.wait()
            self.proc = None
            if rc != 0:
                failures.append((s, rc))
                self.log_queue.put(
                    f"[INSTALL] {s.name} exited {rc}; continuing\n"
                )

        self.log_queue.put(f"\n{'=' * 72}\n")
        if failures:
            self.log_queue.put(
                f"[INSTALL] {len(failures)} of {len(scripts)} script(s) failed:\n"
            )
            for s, rc in failures:
                self.log_queue.put(f"  - {s.name} (exit {rc})\n")
        else:
            self.log_queue.put(
                f"[INSTALL] all {len(scripts)} script(s) succeeded\n"
            )
        self.log_queue.put(f"{'=' * 72}\n")

        self.root.after(0, self._on_done)

    def _on_done(self) -> None:
        self.install_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    InstallerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
