#!/usr/bin/env python3
"""
Execute only confusion-matrix code cells in each notebook,
saving outputs back to the notebook file.
Uses nbformat + nbclient to execute cells in isolation.
"""
import nbformat
from nbclient import NotebookClient
from pathlib import Path
import sys
import os

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks_local"

NOTEBOOKS = {
    "03_model_a_tfidf_baseline_local.ipynb": [20, 22],
    "05_evaluation_demo_local.ipynb":        [17, 19],
}


def execute_cells(nb_name, cell_indices):
    nb_path = NB_DIR / nb_name
    print(f"\n{'='*60}")
    print(f"  {nb_name}")
    print(f"  Cells to execute: {cell_indices}")
    print(f"{'='*60}")

    nb = nbformat.read(nb_path, as_version=4)

    temp_nb = nbformat.v4.new_notebook()
    temp_nb.metadata = nb.metadata

    cell_map = {}
    for ci in cell_indices:
        cell = nb.cells[ci].copy()
        cell['outputs'] = []
        cell['execution_count'] = None
        temp_nb.cells.append(cell)
        cell_map[len(temp_nb.cells) - 1] = ci

    client = NotebookClient(
        temp_nb,
        timeout=300,
        kernel_name="nlp_final",
        resources={"metadata": {"path": str(NB_DIR)}},
    )

    try:
        client.execute()
        print(f"  Execution succeeded!")
    except Exception as e:
        print(f"  Execution error: {e}")
        for i, cell in enumerate(temp_nb.cells):
            if cell.get('outputs'):
                for out in cell['outputs']:
                    if out.get('output_type') == 'error':
                        print(f"    Cell {cell_indices[i]} error:")
                        print(f"      {''.join(out.get('traceback', []))[:500]}")
        return False

    for temp_idx, orig_idx in cell_map.items():
        nb.cells[orig_idx]['outputs'] = temp_nb.cells[temp_idx].get('outputs', [])
        nb.cells[orig_idx]['execution_count'] = temp_nb.cells[temp_idx].get('execution_count')

    nbformat.write(nb, nb_path)
    print(f"  Saved outputs to {nb_name}")

    for temp_idx, orig_idx in cell_map.items():
        outputs = nb.cells[orig_idx].get('outputs', [])
        n_out = len(outputs)
        has_img = any(
            'image/png' in (o.get('data', {}) if o.get('output_type') == 'display_data' else {})
            for o in outputs
        )
        print(f"    Cell {orig_idx}: {n_out} outputs, has_image={has_img}")

    return True


def main():
    os.chdir(NB_DIR)
    
    success = []
    failed = []
    for nb_name, cells in NOTEBOOKS.items():
        ok = execute_cells(nb_name, cells)
        if ok:
            success.append(nb_name)
        else:
            failed.append(nb_name)

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  Success: {len(success)}/{len(NOTEBOOKS)}")
    for s in success:
        print(f"    ✓ {s}")
    if failed:
        print(f"  Failed: {len(failed)}/{len(NOTEBOOKS)}")
        for f in failed:
            print(f"    ✗ {f}")


if __name__ == "__main__":
    main()
