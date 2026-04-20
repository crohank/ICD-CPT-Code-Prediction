#!/usr/bin/env python3
"""
Extract the three EDA figures embedded in notebooks_local/01_data_extraction_local.ipynb
(cell that saves note_length_dist / label_cardinality / code_freq_tail).

Writes PNGs into data/ for use by build_final_presentation.py.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    nb_path = root / "notebooks_local" / "01_data_extraction_local.ipynb"
    out_dir = root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not nb_path.exists():
        print(f"Missing notebook: {nb_path}", file=sys.stderr)
        return 1

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    idx = None
    for i, cell in enumerate(nb.get("cells", [])):
        src = "".join(cell.get("source", []))
        if "note_length_dist.png" in src and "label_cardinality.png" in src:
            idx = i
            break
    if idx is None:
        print("Could not find EDA cell with savefig paths.", file=sys.stderr)
        return 1

    pngs: list[str] = []
    for o in nb["cells"][idx].get("outputs", []):
        if o.get("output_type") != "display_data":
            continue
        data = o.get("data", {})
        if "image/png" in data:
            raw = data["image/png"]
            if isinstance(raw, list):
                raw = "".join(raw)
            pngs.append(raw)

    names = ["note_length_dist.png", "label_cardinality.png", "code_freq_tail.png"]
    if len(pngs) < 3:
        print(f"Expected 3 embedded PNG outputs, found {len(pngs)}.", file=sys.stderr)
        return 1

    for name, b64 in zip(names, pngs):
        dest = out_dir / name
        dest.write_bytes(base64.b64decode(b64))
        print(f"Wrote {dest} ({dest.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
