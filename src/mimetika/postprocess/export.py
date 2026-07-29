"""Postprocessing: export to VTK and compute error norms.

The VTK writer uses the legacy UNSTRUCTURED_GRID format with ``VTK_POLYHEDRON``
(cell type 42) face streams, so it handles arbitrary polyhedral cells, not just
hexes.  0-form fields are written as point data, 3-form fields as cell data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mimetika.mesh.mesh import Mesh

_VTK_POLYHEDRON = 42


def export_vtk(
    path: str | Path,
    mesh: Mesh,
    point_data: dict[str, np.ndarray] | None = None,
    cell_data: dict[str, np.ndarray] | None = None,
) -> Path:
    """Write the mesh (and optional fields) to a legacy ``.vtk`` file."""
    path = Path(path)
    cx = mesh.complex
    pts = mesh.geometry.points

    lines = [
        "# vtk DataFile Version 3.0",
        "mimetika export",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        f"POINTS {len(pts)} double",
    ]
    lines += [f"{x:.17g} {y:.17g} {z:.17g}" for x, y, z in pts]

    # Polyhedron face streams: per cell -> [nFaces, (nPts, ids...), ...].
    rows, total = [], 0
    for entry in cx.cell_facets:
        stream = [len(entry)]
        for fid, _sgn in entry:
            loop = cx.facet_vertices[fid]
            stream += [len(loop), *loop]
        row = [len(stream), *stream]
        rows.append(row)
        total += len(row)

    lines.append(f"CELLS {len(rows)} {total}")
    lines += [" ".join(map(str, r)) for r in rows]
    lines.append(f"CELL_TYPES {len(rows)}")
    lines += [str(_VTK_POLYHEDRON)] * len(rows)

    if point_data:
        lines.append(f"POINT_DATA {len(pts)}")
        for name, arr in point_data.items():
            lines += _scalar_block(name, arr)
    if cell_data:
        lines.append(f"CELL_DATA {len(rows)}")
        for name, arr in cell_data.items():
            lines += _scalar_block(name, arr)

    path.write_text("\n".join(lines) + "\n")
    return path


def _scalar_block(name: str, arr: np.ndarray) -> list[str]:
    return [
        f"SCALARS {name} double 1",
        "LOOKUP_TABLE default",
        *[f"{v:.17g}" for v in np.asarray(arr).ravel()],
    ]


def l2_error(
    mesh: Mesh, degree: int, discrete: np.ndarray, exact_at_centroid
) -> float:
    """Measure-weighted L2 error of a k-form against an exact field.

    ``discrete`` holds the cochain values (one per k-cell); ``exact_at_centroid``
    is a callable ``(N,3) -> (N,)`` giving the exact k-form density sampled at
    the k-cell centroids.  This is a first-order midpoint estimate.
    """
    measure = mesh.geometry.measure(degree)
    centroids = mesh.geometry.centroids(degree)
    exact = np.asarray(exact_at_centroid(centroids)).ravel()
    diff = np.asarray(discrete).ravel() - exact
    return float(np.sqrt(np.sum(measure * diff**2)))
