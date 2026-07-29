"""Mesh readers.

Currently: the VTK XML unstructured-grid format (``.vtu``), in ASCII.

``VTK_POLYHEDRON`` (type 42) cells carry an explicit face stream, which is
exactly mimetika's native "cell = list of face loops" description, so genuinely
polytopal meshes load without any conversion.  The standard linear cell types
(tetrahedron, hexahedron, wedge, pyramid) are also supported by expanding them
into their face loops.

Face orientation is normalised on load, **per face**: VTK does not require a
polyhedron's face loops to be oriented outward, and in practice writers emit
them in arbitrary directions.  Each loop is therefore reversed individually
when its area vector points into the cell.  Consistent outward orientation is
what makes the signed incidence -- and therefore ``dd = 0`` -- come out right;
without it, interior facets fail to cancel between their two cells.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from mimetika.mesh.mesh import Mesh

VTK_TETRA = 10
VTK_HEXAHEDRON = 12
VTK_WEDGE = 13
VTK_PYRAMID = 14
VTK_POLYHEDRON = 42

# Face loops of the standard linear cells, in VTK corner ordering.
_LINEAR_CELL_FACES = {
    VTK_TETRA: ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)),
    VTK_HEXAHEDRON: (
        (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (0, 4, 7, 3),
    ),
    VTK_WEDGE: ((0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)),
    VTK_PYRAMID: ((0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)),
}


def read_vtu(path: str | Path) -> Mesh:
    """Read an ASCII ``.vtu`` unstructured grid into a :class:`Mesh`."""
    text = Path(path).read_text()
    points = _data_array(text, index=0).reshape(-1, 3)
    types = _data_array(text, name="types").astype(np.int64)
    offsets = _data_array(text, name="offsets").astype(np.int64)
    connectivity = _data_array(text, name="connectivity").astype(np.int64)

    faces = _optional_array(text, "faces")
    face_offsets = _optional_array(text, "faceoffsets")

    cells = _build_cells(types, offsets, connectivity, faces, face_offsets)
    cells = [_orient_outward(points, cell) for cell in cells]
    return Mesh.from_cells(points, cells)


# -- parsing ------------------------------------------------------------------


def _data_array(text: str, name: str | None = None, index: int | None = None):
    if name is not None:
        m = re.search(
            r'<DataArray[^>]*Name="%s"[^>]*>(.*?)</DataArray>' % re.escape(name),
            text,
            re.S,
        )
        if m is None:
            raise KeyError(f"no DataArray named {name!r}")
    else:
        m = list(re.finditer(r"<DataArray[^>]*>(.*?)</DataArray>", text, re.S))[index]
    return np.fromstring(m.group(1), sep=" ")


def _optional_array(text: str, name: str):
    try:
        return _data_array(text, name=name).astype(np.int64)
    except KeyError:
        return None


def _build_cells(types, offsets, connectivity, faces, face_offsets):
    """Turn the VTK arrays into one list of face loops per cell."""
    cells: list[list[list[int]]] = []
    conn_start = 0
    face_start = 0
    for c, (ctype, conn_end) in enumerate(zip(types, offsets)):
        corners = connectivity[conn_start:conn_end]
        conn_start = conn_end

        if ctype == VTK_POLYHEDRON:
            if faces is None or face_offsets is None:
                raise ValueError("polyhedral cells need 'faces' and 'faceoffsets'")
            stream = faces[face_start : face_offsets[c]]
            face_start = face_offsets[c]
            cells.append(_decode_face_stream(stream))
        elif ctype in _LINEAR_CELL_FACES:
            cells.append(
                [[int(corners[i]) for i in f] for f in _LINEAR_CELL_FACES[ctype]]
            )
        else:
            raise NotImplementedError(f"unsupported VTK cell type {int(ctype)}")
    return cells


def _decode_face_stream(stream: np.ndarray) -> list[list[int]]:
    """``[nFaces, (nPts, ids...), ...]`` -> a list of vertex loops."""
    n_faces = int(stream[0])
    loops, i = [], 1
    for _ in range(n_faces):
        n = int(stream[i])
        loops.append([int(v) for v in stream[i + 1 : i + 1 + n]])
        i += 1 + n
    return loops


# -- orientation --------------------------------------------------------------


def _loop_area_vector(points: np.ndarray, loop: list[int]) -> np.ndarray:
    p = points[loop]
    c = p.mean(0)
    return 0.5 * np.cross(p - c, np.roll(p, -1, axis=0) - c).sum(0)


def _signed_volume(points: np.ndarray, cell: list[list[int]]) -> float:
    """``(1/3) sum_f x_f . A_f`` -- positive when every loop faces outward."""
    total = 0.0
    for loop in cell:
        a = _loop_area_vector(points, loop)
        total += float(np.dot(points[loop].mean(0), a))
    return total / 3.0


def _orient_outward(points: np.ndarray, cell: list[list[int]]) -> list[list[int]]:
    """Orient each face loop of a cell outward, individually.

    A loop points outward when its area vector agrees with the direction from an
    interior reference point (the mean of the face centroids) to the face.  This
    is the star-shapedness rule already used elsewhere in the library, and it
    fixes cells whose loops arrive in mixed directions -- reversing whole cells
    cannot.
    """
    centre = np.mean([points[loop].mean(0) for loop in cell], axis=0)
    oriented = []
    for loop in cell:
        a = _loop_area_vector(points, loop)
        outward = np.dot(a, points[loop].mean(0) - centre) >= 0.0
        oriented.append(list(loop) if outward else list(reversed(loop)))
    return oriented


def check_orientation(mesh: Mesh, atol: float = 1e-9) -> None:
    """Raise if any cell's oriented faces fail to form a closed surface.

    ``sum_f |f| n_f = 0`` over a cell is the discrete divergence theorem applied
    to a constant field; it fails exactly when a face loop is mis-oriented.
    """
    g = mesh.geometry
    normals, areas = g.facet_normals(), g.measure(2)
    worst, worst_cell = 0.0, -1
    for c in range(mesh.num_cells(3)):
        total = sum(
            s * areas[f] * normals[f] for f, s in mesh.complex.facets_of(3, c)
        )
        scale = sum(areas[f] for f, _ in mesh.complex.facets_of(3, c))
        rel = float(np.linalg.norm(total) / scale)
        if rel > worst:
            worst, worst_cell = rel, c
    if worst > atol:
        raise ValueError(
            f"cell {worst_cell} is not closed under its oriented faces "
            f"(relative residual {worst:.3e}); the mesh may be non-star-shaped"
        )
