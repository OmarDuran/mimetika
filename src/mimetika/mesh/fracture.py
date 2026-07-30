r"""Fracture tagging and extraction of the lower-dimensional subdomain.

A fracture is a **set of tagged facets** of the existing mesh -- no geometry is
created or modified.  From that tag set two things are derived:

* the ``d-1`` dimensional fracture mesh, whose cells *are* the tagged facets
  (conformal by construction, so no projection is ever needed), and
* the DOF duplication used by the flow problem
  (:class:`~mimetika.dof.facet_dofs.FacetDofMap`).

Tags travel in the ``.vtu`` file as **vertex loops**, not facet indices: facet
numbering is derived from the cell complex and would not survive a round trip,
whereas a vertex set identifies a face unambiguously -- it is exactly the key
the complex itself uses to merge shared faces.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from mimetika.mesh.mesh import Mesh


# -- tagging ------------------------------------------------------------------


def facets_on_plane(
    mesh: Mesh, point, normal, atol: float = 1e-9, interior_only: bool = True
) -> np.ndarray:
    """Facets whose vertices all lie on the plane through ``point``.

    The usual way to tag a planar fracture in a synthetic test.  By default only
    interior facets are returned -- a boundary facet has a single side and
    cannot carry a fracture.
    """
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    p0 = np.asarray(point, dtype=float)
    pts = mesh.geometry.points

    out = []
    for f in range(mesh.num_cells(mesh.dim - 1)):
        verts = facet_vertices(mesh, f)
        if np.abs((pts[list(verts)] - p0) @ n).max() > atol:
            continue
        if interior_only and len(_incident_cells(mesh, f)) < 2:
            continue
        out.append(f)
    return np.array(out, dtype=np.int64)


def facet_vertices(mesh: Mesh, facet: int):
    """Vertices of a facet, whatever the mesh dimension (edge in 2D, face in 3D)."""
    if mesh.dim == 3:
        return mesh.complex.polygon_loops[int(facet)]
    if mesh.dim == 2:
        return tuple(int(v) for v in mesh.complex.edge_vertices[int(facet)])
    raise NotImplementedError(f"facet tagging needs dim 2 or 3, got {mesh.dim}")


def _incident_cells(mesh: Mesh, facet: int) -> list[int]:
    row = mesh.complex.boundary_matrix(mesh.dim).tocsr()[facet]
    return sorted(int(c) for c in row.indices)


def interior_facets(mesh: Mesh) -> np.ndarray:
    """Facets with two incident cells -- the only ones a fracture can occupy."""
    inc = abs(mesh.complex.boundary_matrix(mesh.dim))
    counts = np.asarray(inc.sum(axis=1)).ravel()
    return np.where(counts == 2)[0]


# -- extraction ---------------------------------------------------------------


def fracture_mesh(mesh: Mesh, tagged) -> tuple[Mesh, np.ndarray]:
    """Build the ``d-1`` dimensional mesh whose cells are the tagged facets.

    Returns ``(fracture, facet_of_cell)`` where ``facet_of_cell[i]`` is the
    ambient facet id that became fracture cell ``i``.  Vertices are renumbered
    to those actually used, so the fracture mesh is standalone.
    """
    tagged = np.asarray(sorted(int(f) for f in tagged), dtype=np.int64)
    if mesh.dim != 3:
        raise NotImplementedError("fracture extraction is implemented for 3D meshes")

    used: dict[int, int] = {}
    polygons = []
    for f in tagged:
        loop = mesh.complex.polygon_loops[int(f)]
        polygons.append([used.setdefault(int(v), len(used)) for v in loop])

    points = np.empty((len(used), 3))
    for old, new in used.items():
        points[new] = mesh.geometry.points[old]
    return Mesh.from_polygons(points, polygons), tagged


# -- .vtu round trip ----------------------------------------------------------


def write_fracture_tags(path: str | Path, mesh: Mesh, tagged) -> Path:
    """Append fracture tags to a ``.vtu`` file as vertex loops in ``FieldData``.

    Stored as a flat vertex list plus offsets, so the tag survives any change of
    facet numbering.
    """
    path = Path(path)
    text = path.read_text()
    flat, offsets = [], []
    for f in sorted(int(x) for x in tagged):
        flat += list(facet_vertices(mesh, f))
        offsets.append(len(flat))

    block = (
        "<FieldData>\n"
        + _int_array("fracture_faces", flat)
        + _int_array("fracture_face_offsets", offsets)
        + "</FieldData>\n"
    )
    marker = "<UnstructuredGrid>"
    return _write(path, text.replace(marker, marker + "\n" + block, 1))


def read_fracture_tags(path: str | Path, mesh: Mesh) -> np.ndarray:
    """Read fracture tags from a ``.vtu`` and resolve them to facet ids.

    Each stored loop is matched to a facet by its **vertex set**, the same key
    the cell complex uses to identify shared faces.
    """
    text = Path(path).read_text()
    flat = _read_int_array(text, "fracture_faces")
    offsets = _read_int_array(text, "fracture_face_offsets")
    if flat is None or offsets is None:
        return np.zeros(0, dtype=np.int64)

    lookup = {
        frozenset(facet_vertices(mesh, f)): f
        for f in range(mesh.num_cells(mesh.dim - 1))
    }
    out, start = [], 0
    for end in offsets:
        key = frozenset(int(v) for v in flat[start:end])
        start = end
        if key not in lookup:
            raise KeyError(f"tagged face {sorted(key)} is not a facet of this mesh")
        out.append(lookup[key])
    return np.array(sorted(out), dtype=np.int64)


def _int_array(name: str, values) -> str:
    body = " ".join(str(int(v)) for v in values)
    return (
        f'<DataArray type="Int64" Name="{name}" format="ascii">\n{body}\n</DataArray>\n'
    )


def _read_int_array(text: str, name: str):
    m = re.search(
        r'<DataArray[^>]*Name="%s"[^>]*>(.*?)</DataArray>' % re.escape(name), text, re.S
    )
    if m is None:
        return None
    return np.fromstring(m.group(1), sep=" ", dtype=float).astype(np.int64)


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path
