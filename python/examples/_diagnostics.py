"""exokal's mesh diagnostics, written to a folder.

Nothing here computes anything about the mesh: one call to exokal's
diagnose_vtu returns the formal findings, the shape-regularity classification
at the library constants, and the degenerate-cell witnesses -- the latter as
numbers rather than a message, so the hundred of them a real mesh produces can
be sorted, joined back onto the mesh, or opened in a spreadsheet.

The cell ids in the CSV are the ids OF THE INPUT FILE -- the ones ParaView
selects -- not the ambient complex's, which are what the reader happens to
assign after it reorders the cells top-first.

percent is 100 |s| divided by the MEAN measure over the cell's node star, so a
uniform mesh reports 100 at every cell whatever its valence. The star's total is
carried alongside as a column; it is not the denominator.
"""

import csv
import os

import mimetika_cxx as mk


def write_report(mesh_path, out_dir, degeneracy_percent=0.1):
    """Write diagnostics.txt and degenerate_cells.csv into `out_dir`.

    Returns (n_violations, n_warnings, n_degenerate).
    """
    os.makedirs(out_dir, exist_ok=True)

    d = mk.diagnose_vtu(mesh_path, degeneracy_percent)
    text = d["report"] + "\n" + d["quality"]
    with open(os.path.join(out_dir, "diagnostics.txt"), "w") as f:
        f.write(text)

    rows = d["degenerate"]
    with open(os.path.join(out_dir, "degenerate_cells.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "vtk_cell_id",
                "dim",
                "measure",
                "neighborhood_mean",
                "neighborhood_total",
                "percent",
                "n_neighbors",
            ]
        )
        for r in sorted(rows, key=lambda r: r["percent"]):
            w.writerow(
                [
                    r["vtk_cell_id"],
                    r["dim"],
                    f"{r['measure']:.6e}",
                    f"{r['neighborhood_mean']:.6e}",
                    f"{r['neighborhood_total']:.6e}",
                    f"{r['percent']:.6f}",
                    r["n_neighbors"],
                ]
            )

    violations = text.count("[VIOLATION]")
    warnings = text.count("[warning]")
    return violations, warnings, len(rows)
