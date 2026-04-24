#!/usr/bin/env python3
"""
ICD-10 Code Prediction — One-Click Streamlit Launcher
======================================================

Lightweight alternative to `demo.py`: installs a smaller dependency set and
opens the same predict-only Streamlit app. It does **not** create a venv or
pin Python; prefer `python demo.py` if you want that bootstrap behavior.

Usage
-----
    python run_demo.py

What happens
------------
1. Installs any missing packages into the **current** interpreter (quiet pip).
2. Runs `streamlit run demo/streamlit_app.py` with headless-friendly flags,
   including disabling Streamlit's file watcher so optional `torchvision` is
   not pulled in during startup.

Requirements
------------
- Python that matches your project (3.10+ recommended; same as training).
- Trained weights under `data/models/` and pickles under `datasets/processed/`.

CS6120 NLP — Final Project
"""

import importlib
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEMO_SCRIPT = PROJECT_ROOT / "demo" / "streamlit_app.py"

REQUIRED_PACKAGES = [
    ("streamlit",    "streamlit"),
    ("pandas",       "pandas"),
    ("numpy",        "numpy"),
    ("sklearn",      "scikit-learn"),
    ("matplotlib",   "matplotlib"),
    ("seaborn",      "seaborn"),
    ("scipy",        "scipy"),
    ("PIL",          "Pillow"),
]


def ensure_packages():
    """Install any packages that are not already available."""
    missing = []
    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print(f"Installing {len(missing)} missing package(s): {', '.join(missing)}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q"] + missing,
            stdout=subprocess.DEVNULL,
        )
        print("Done.\n")


def main():
    print("=" * 60)
    print("  ICD-10 Code Prediction — Interactive Demo")
    print("  CS6120 NLP Final Project")
    print("=" * 60)
    print()

    ensure_packages()

    if not DEMO_SCRIPT.exists():
        print(f"ERROR: Dashboard script not found at {DEMO_SCRIPT}")
        sys.exit(1)

    print("Launching Streamlit dashboard …")
    print("(Press Ctrl+C to stop)\n")

    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run",
            str(DEMO_SCRIPT),
            "--server.headless", "true",
            # Same rationale as `demo.py`: watcher can import vision stacks unnecessarily.
            "--server.fileWatcherType", "none",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=str(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
