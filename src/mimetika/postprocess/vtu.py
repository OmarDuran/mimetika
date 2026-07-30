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
_VTK_POLYGON = 7
_VTK_LINE = 3


def export_vtu(
    path: str | Path,
    mesh: Mesh,
    cell_data: dict[str, np.ndarray] | None = None,
    point_data: dict[str, np.ndarray] | None = None,
) -> Path:
    """Write a mesh and its fields to an ASCII ``.vtu`` file.

    Works for 3D (polyhedra, written as ``VTK_POLYHEDRON`` face streams so the
    polytopal geometry survives intact) and for 2D (polygons).  A 2D mesh is not
    a special case to be flattened -- it is what a fracture is, and what the
    lower-dimensional half of a mixed-dimensional problem lives on.
    """
    path = Path(path)
    cx, pts = mesh.complex, mesh.geometry.points
    d = mesh.dim
    n_cells = mesh.num_cells(d)

    if d == 2:  # polygons: connectivity is the loop, no face stream
        connectivity, offsets = [], []
        for c in range(n_cells):
            connectivity += list(cx.polygon_loops[c])
            offsets.append(len(connectivity))
        cells = [
            _array("connectivity", np.asarray(connectivity), "Int64"),
            _array("offsets", np.asarray(offsets), "Int64"),
            _array("types", np.full(n_cells, _VTK_POLYGON), "UInt8"),
        ]
        return _write(path, pts, n_cells, cells, cell_data, point_data)

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


def export_facets(
    path: str | Path,
    mesh: Mesh,
    facets,
    cell_data: dict[str, np.ndarray] | None = None,
) -> Path:
    """Write a **subset of facets** as a standalone lower-dimensional ``.vtu``.

    This is how a fracture is plotted: it has no mesh of its own in the ambient
    problem, it *is* a tagged set of ``(d-1)``-facets.  Facets of a 3D mesh become
    ``VTK_POLYGON``, facets of a 2D mesh become ``VTK_LINE``.  ``cell_data`` is
    indexed by position in ``facets``, not by global facet id, so callers pass the
    per-fracture arrays they already have.

    Only the vertices actually used are written, so a fracture file stays small
    even when the ambient mesh is large.
    """
    path = Path(path)
    facets = np.asarray(facets, dtype=np.int64)
    cx = mesh.complex
    loops = [
        tuple(cx.polygon_loops[int(f)]) if mesh.dim == 3
        else tuple(int(v) for v in cx.edge_vertices[int(f)])
        for f in facets
    ]

    used = sorted({v for loop in loops for v in loop})
    renumber = {v: i for i, v in enumerate(used)}
    points = mesh.geometry.points[used] if used else np.zeros((0, 3))

    connectivity, offsets = [], []
    for loop in loops:
        connectivity += [renumber[v] for v in loop]
        offsets.append(len(connectivity))
    kind = _VTK_POLYGON if mesh.dim == 3 else _VTK_LINE
    cells = [
        _array("connectivity", np.asarray(connectivity), "Int64"),
        _array("offsets", np.asarray(offsets), "Int64"),
        _array("types", np.full(len(facets), kind), "UInt8"),
    ]
    return _write(path, points, len(facets), cells, cell_data, None)


def _write(path, points, n_cells, cells, cell_data, point_data) -> Path:
    out = [
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">',
        "<UnstructuredGrid>",
        f'<Piece NumberOfPoints="{len(points)}" NumberOfCells="{n_cells}">',
        "<Points>",
        _array("Points", np.asarray(points).reshape(-1, 3), "Float64"),
        "</Points>",
        "<Cells>", *cells, "</Cells>",
    ]
    if cell_data:
        out += ["<CellData>", *[_field(k, v) for k, v in cell_data.items()],
                "</CellData>"]
    if point_data:
        out += ["<PointData>", *[_field(k, v) for k, v in point_data.items()],
                "</PointData>"]
    out += ["</Piece>", "</UnstructuredGrid>", "</VTKFile>", ""]
    Path(path).write_text("\n".join(out))
    return Path(path)


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
