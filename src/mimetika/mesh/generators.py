"""Mesh generators.

Currently: a structured hexahedral box.  Cells are emitted in the polytopal
cells-as-face-loops format, so they flow through the same topology construction
as arbitrary polyhedra -- the structured grid is just a convenient source.
"""

from __future__ import annotations

import numpy as np

from mimetika.mesh.mesh import Mesh

# Outward-oriented faces of a hexahedron given its 8 corners
# (VTK-style corner order: bottom 0-3 CCW, top 4-7 CCW above them).
_HEX_FACES = (
    (0, 3, 2, 1),  # z- (bottom)
    (4, 5, 6, 7),  # z+ (top)
    (0, 1, 5, 4),  # y- (front)
    (1, 2, 6, 5),  # x+ (right)
    (2, 3, 7, 6),  # y+ (back)
    (0, 4, 7, 3),  # x- (left)
)


def structured_box(
    nx: int,
    ny: int,
    nz: int,
    lengths: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Mesh:
    """A structured hex mesh of ``nx x ny x nz`` cells over a box."""
    lx, ly, lz = lengths
    ox, oy, oz = origin
    xs = ox + np.linspace(0, lx, nx + 1)
    ys = oy + np.linspace(0, ly, ny + 1)
    zs = oz + np.linspace(0, lz, nz + 1)

    def vid(i: int, j: int, k: int) -> int:
        return i + (nx + 1) * (j + (ny + 1) * k)

    points = np.empty(((nx + 1) * (ny + 1) * (nz + 1), 3))
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                points[vid(i, j, k)] = (xs[i], ys[j], zs[k])

    cells: list[list[list[int]]] = []
    for ck in range(nz):
        for cj in range(ny):
            for ci in range(nx):
                corners = [
                    vid(ci, cj, ck),
                    vid(ci + 1, cj, ck),
                    vid(ci + 1, cj + 1, ck),
                    vid(ci, cj + 1, ck),
                    vid(ci, cj, ck + 1),
                    vid(ci + 1, cj, ck + 1),
                    vid(ci + 1, cj + 1, ck + 1),
                    vid(ci, cj + 1, ck + 1),
                ]
                cells.append([[corners[n] for n in face] for face in _HEX_FACES])

    return Mesh.from_cells(points, cells)
