"""The mesh families the fixed-dimensional tests run on.

Ported cell-for-cell from tests/model/test_confined_compression.cpp so that the Python
suite meshes the same domains the C++ suite does. A different subdivision here
would make the two sets of numbers incomparable, which is the whole point of
having both.
"""

import mimetika_cxx as mk


def square(n: int, simplex: bool) -> mk.Mesh:
    """n x n cells over the unit square, as quadrilaterals or as triangles."""
    pts = [[i / n, j / n, 0.0] for j in range(n + 1) for i in range(n + 1)]

    def vid(i, j):
        return j * (n + 1) + i

    cells = []
    for j in range(n):
        for i in range(n):
            if simplex:
                cells.append([vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)])
                cells.append([vid(i, j), vid(i + 1, j + 1), vid(i, j + 1)])
            else:
                cells.append([vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)])
    return mk.Mesh.from_simplices(2, pts, cells) if simplex else mk.Mesh.from_polygons(pts, cells)


# the six-tetrahedron (Kuhn) subdivision of each cube: it tiles, and every
# tetrahedron is positively oriented
_KUHN = [[0, 1, 3, 7], [0, 1, 5, 7], [0, 4, 5, 7], [0, 4, 6, 7], [0, 2, 6, 7], [0, 2, 3, 7]]
_FACES = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [1, 3, 7, 5], [3, 2, 6, 7], [2, 0, 4, 6]]


def cube(n: int, simplex: bool) -> mk.Mesh:
    """n x n x n cells over the unit cube, as hexahedra or as tetrahedra."""
    pts = [
        [i / n, j / n, k / n]
        for k in range(n + 1)
        for j in range(n + 1)
        for i in range(n + 1)
    ]

    def vid(i, j, k):
        return (k * (n + 1) + j) * (n + 1) + i

    def corner(i, j, k, l):
        return vid(i + (l & 1), j + ((l >> 1) & 1), k + ((l >> 2) & 1))

    cells = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                if simplex:
                    for t in _KUHN:
                        cells.append([corner(i, j, k, l) for l in t])
                else:
                    cells.append([[corner(i, j, k, l) for l in f] for f in _FACES])
    return (
        mk.Mesh.from_simplices(3, pts, cells) if simplex else mk.Mesh.from_polyhedra(pts, cells)
    )
