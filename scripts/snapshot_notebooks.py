"""
snapshot_notebooks.py
---------------------
Saves a timestamped copy of every notebook in notebooks_local/ (with cell
outputs intact) to notebooks_local/archive/<YYYY-MM-DD_HHMMSS>/.

Run this BEFORE rerunning any notebook so you can recover the previous outputs.

Usage:
    python scripts/snapshot_notebooks.py
"""

import shutil
from datetime import datetime
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks_local"
ARCHIVE_DIR   = NOTEBOOKS_DIR / "archive"

def main():
    timestamp  = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    target_dir = ARCHIVE_DIR / timestamp
    target_dir.mkdir(parents=True, exist_ok=True)

    notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    if not notebooks:
        print("No notebooks found in", NOTEBOOKS_DIR)
        return

    for nb in notebooks:
        dest = target_dir / nb.name
        shutil.copy2(nb, dest)
        print(f"  Saved: {dest.relative_to(NOTEBOOKS_DIR.parent)}")

    print(f"\nSnapshot complete -> notebooks_local/archive/{timestamp}/")
    print("These copies retain all cell outputs and can be opened in VS Code.")

if __name__ == "__main__":
    main()
