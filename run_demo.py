#!/usr/bin/env python3
"""
ICD-10 Code Prediction — One-Click Demo Launcher
=================================================

Run this file to install all required packages and launch the interactive
dashboard.  No other setup is necessary.

Usage
-----
    python run_demo.py

What happens
------------
1. Missing Python packages are installed automatically (into the current
   environment or virtualenv).
2. A Streamlit dashboard opens in your default browser showing:
   - Live ICD-10 code prediction from discharge summaries  (if model
     artifacts are present in datasets/processed/)
   - Side-by-side comparison of all 5 models + 5 ensemble variants
   - Training curves, head / torso / tail F1 breakdowns
   - Confusion matrices (once generated from notebooks)
   - Exploratory Data Analysis plots

Requirements
------------
- Python >= 3.9
- ~200 MB disk for packages on first run (cached afterwards)

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
            "--browser.gatherUsageStats", "false",
        ],
        cwd=str(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
