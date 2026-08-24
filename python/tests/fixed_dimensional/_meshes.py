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


# THE POLYTOPE THAT IS NOT A DEGENERATE HEXAHEDRON.
#
# A honeycomb of hexagonal prisms: eight facets a cell -- two hexagons and six
# quadrilaterals -- so the lowest-order flux space carries eight unknowns on a
# cell where RT_0 spans four modes. That surplus is what a mimetic product has
# to close, and what makes this the mesh the polytopal claim is actually about;
# a cartesian hexahedron has six facets and a tetrahedron four.
#
# Corners are deduplicated by rounded coordinate, which is what makes the
# neighbouring cells share them: a hexagonal lattice's corners coincide exactly
# in the formulas below, so the tolerance never has to decide anything.
def honeycomb(nq: int, nr: int, nz: int, s: float = 1.0, h: float = 1.0) -> mk.Mesh:
    """nq x nr hexagonal prisms in the plane, nz layers deep, circumradius s."""
    import math

    pts, index = [], {}

    def vid(x, y, z):
        key = (round(x, 9), round(y, 9), round(z, 9))
        if key not in index:
            index[key] = len(pts)
            pts.append([key[0], key[1], key[2]])
        return index[key]

    cells = []
    for q in range(nq):
        for r in range(nr):
            cx = math.sqrt(3.0) * (q + 0.5 * r) * s
            cy = 1.5 * r * s
            corners = [
                (cx + s * math.cos(math.radians(60 * k + 30)),
                 cy + s * math.sin(math.radians(60 * k + 30)))
                for k in range(6)
            ]
            for k in range(nz):
                lo = [vid(x, y, k * h) for x, y in corners]
                hi = [vid(x, y, (k + 1) * h) for x, y in corners]
                faces = [lo[::-1], hi]
                for a in range(6):
                    b = (a + 1) % 6
                    faces.append([lo[a], lo[b], hi[b], hi[a]])
                cells.append(faces)
    return mk.Mesh.from_polyhedra(pts, cells)
