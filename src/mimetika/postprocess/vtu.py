"""Write results to the VTK XML unstructured-grid format (``.vtu``).

Polyhedral cells are written as ``VTK_POLYHEDRON`` face streams, so the
original polytopal geometry is preserved exactly -- ParaView shows the real
cells, not a tetrahedralisation of them.

Field shapes are mapped to VTK component counts automatically:

    ``(n,)``      -> scalar
    ``(n, 3)``    -> vector   (glyphable in ParaView)
    ``(n, 3, 3)`` -> tensor   (9 components)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mimetika.mesh.mesh import Mesh

_VTK_POLYHEDRON = 42


def export_vtu(
    path: str | Path,
    mesh: Mesh,
    cell_data: dict[str, np.ndarray] | None = None,
    point_data: dict[str, np.ndarray] | None = None,
) -> Path:
    """Write a mesh and its fields to an ASCII ``.vtu`` file."""
    path = Path(path)
    cx, pts = mesh.complex, mesh.geometry.points
    n_cells = mesh.num_cells(3)

    connectivity, offsets, faces, face_offsets = [], [], [], []
    for c in range(n_cells):
        entry = cx.facets_of(3, c)
        connectivity += sorted(cx.cell_vertices(c))
        offsets.append(len(connectivity))
        stream = [len(entry)]
        for fid, sign in entry:
            loop = cx.polygon_loops[fid]
            if sign < 0:  # emit outward-oriented loops
                loop = tuple(reversed(loop))
            stream += [len(loop), *loop]
        faces += stream
        face_offsets.append(len(faces))

    out = [
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">',
        "<UnstructuredGrid>",
        f'<Piece NumberOfPoints="{len(pts)}" NumberOfCells="{n_cells}">',
        "<Points>",
        _array("Points", pts.reshape(-1, 3), "Float64"),
        "</Points>",
        "<Cells>",
        _array("connectivity", np.asarray(connectivity), "Int64"),
        _array("offsets", np.asarray(offsets), "Int64"),
        _array("types", np.full(n_cells, _VTK_POLYHEDRON), "UInt8"),
        _array("faces", np.asarray(faces), "Int64"),
        _array("faceoffsets", np.asarray(face_offsets), "Int64"),
        "</Cells>",
    ]

    if cell_data:
        out += ["<CellData>", *[_field(k, v) for k, v in cell_data.items()], "</CellData>"]
    if point_data:
        out += [
            "<PointData>",
            *[_field(k, v) for k, v in point_data.items()],
            "</PointData>",
        ]

    out += ["</Piece>", "</UnstructuredGrid>", "</VTKFile>", ""]
    path.write_text("\n".join(out))
    return path


def _field(name: str, values: np.ndarray) -> str:
    v = np.asarray(values, dtype=float)
    if v.ndim == 3:  # tensor
        v = v.reshape(len(v), -1)
    elif v.ndim == 1:
        v = v.reshape(-1, 1)
    return _array(name, v, "Float64")


def _array(name: str, values: np.ndarray, dtype: str) -> str:
    v = np.asarray(values)
    if v.ndim == 1:  # a flat array is n rows of one component, not one row of n
        v = v.reshape(-1, 1)
    ncomp = v.shape[1]
    fmt = "%d" if dtype.startswith(("Int", "UInt")) else "%.17g"
    body = "\n".join(" ".join(fmt % x for x in row) for row in v)
    comp = f' NumberOfComponents="{ncomp}"' if ncomp > 1 else ""
    return (
        f'<DataArray type="{dtype}" Name="{name}"{comp} format="ascii">\n'
        f"{body}\n</DataArray>"
    )
