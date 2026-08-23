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


# Outward-oriented triangular faces of a tetrahedron with corners 0,1,2,3.
_TET_FACES = (
    (0, 2, 1),  # opposite vertex 3
    (0, 1, 3),  # opposite vertex 2
    (0, 3, 2),  # opposite vertex 1
    (1, 2, 3),  # opposite vertex 0
)


def single_tetrahedron(
    points: np.ndarray | None = None,
) -> Mesh:
    """A single tetrahedron. Defaults to the unit reference tet.

    ``points`` is a ``(4, 3)`` array of the four vertices, ordered so that the
    signed volume of (p1-p0, p2-p0, p3-p0) is positive.
    """
    if points is None:
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
    points = np.asarray(points, dtype=float)
    if points.shape != (4, 3):
        raise ValueError("a tetrahedron needs exactly 4 vertices in 3D")
    cell = [[int(points_idx) for points_idx in face] for face in _TET_FACES]
    return Mesh.from_cells(points, [cell])


def single_hexahedron(points: np.ndarray) -> Mesh:
    """A single (possibly distorted) hexahedron from 8 corner points.

    Corners follow the VTK ordering used by :func:`structured_box`:
    bottom face 0-3 counter-clockwise, top face 4-7 directly above.
    """
    points = np.asarray(points, dtype=float)
    if points.shape != (8, 3):
        raise ValueError("a hexahedron needs exactly 8 vertices in 3D")
    cell = [[int(n) for n in face] for face in _HEX_FACES]
    return Mesh.from_cells(points, [cell])


# Kuhn/Freudenthal split of a hexahedron into 6 tetrahedra (corner indices
# follow the VTK ordering used above).  The split is conforming across cells.
_HEX_TO_TETS = (
    (0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
    (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6),
)


def structured_tets(
    nx: int,
    ny: int,
    nz: int,
    lengths: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Mesh:
    """A conforming tetrahedral mesh of a box (each hex split into 6 tets).

    Simplicial meshes are the case where the mimetic stabilization vanishes, so
    this is the mesh family on which the schemes coincide with RT0 / AFW.
    """
    hexes = structured_box(nx, ny, nz, lengths, origin)
    pts = hexes.geometry.points

    cells = []
    for cid in range(hexes.num_cells(3)):
        corners = _hex_corner_order(hexes, cid)
        for tet in _HEX_TO_TETS:
            v = [corners[i] for i in tet]
            cells.append(_oriented_tet_faces(pts, v))
    return Mesh.from_cells(pts, cells)


def _hex_corner_order(mesh: Mesh, cid: int) -> list[int]:
    """Recover the VTK corner ordering of a structured-box hexahedron."""
    verts = np.array(sorted(mesh.complex.cell_vertices(cid)))
    p = mesh.geometry.points[verts]
    # bit-encode each corner by which side of the cell it sits on
    mid = p.mean(0)
    key = ((p > mid) * np.array([1, 2, 4])).sum(1)
    order = [0, 1, 3, 2, 4, 5, 7, 6]  # bit code -> VTK corner slot
    corners = [None] * 8
    for v, k in zip(verts, key):
        corners[order[k]] = int(v)
    return corners


def _oriented_tet_faces(points: np.ndarray, v: list[int]) -> list[list[int]]:
    """Faces of a tetrahedron, each oriented with an outward normal."""
    p = points[v]
    if np.linalg.det(p[1:] - p[0]) < 0:  # make the corner ordering positive
        v = [v[0], v[2], v[1], v[3]]
    return [[v[a] for a in f] for f in _TET_FACES]


def structured_quads(
    nx: int,
    ny: int,
    lengths: tuple[float, float] = (1.0, 1.0),
    origin: tuple[float, float] = (0.0, 0.0),
) -> Mesh:
    """A structured quadrilateral mesh of a rectangle, in the ``z = 0`` plane."""
    lx, ly = lengths
    xs = origin[0] + np.linspace(0, lx, nx + 1)
    ys = origin[1] + np.linspace(0, ly, ny + 1)
    points = np.array([[x, y, 0.0] for y in ys for x in xs])

    def vid(i, j):
        return j * (nx + 1) + i

    quads = [
        [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)]
        for j in range(ny)
        for i in range(nx)
    ]
    return Mesh.from_polygons(points, quads)


def graded_quads(xs, ys) -> Mesh:
    """A quadrilateral mesh from **explicit** node coordinates, in ``z = 0``.

    ``structured_quads`` spaces nodes uniformly, which is the wrong tool when the
    solution has features far smaller than the domain: resolving a 150 m
    reservoir inside a 4500 m block costs thousands of cells that do nothing.
    Passing the coordinates directly lets the mesh be fine where the physics is
    and coarse where it is not, and lets material and loading interfaces be
    placed exactly on cell faces.
    """
    xs = np.unique(np.asarray(xs, dtype=float))
    ys = np.unique(np.asarray(ys, dtype=float))
    nx, ny = len(xs) - 1, len(ys) - 1
    if nx < 1 or ny < 1:
        raise ValueError("need at least two distinct coordinates in each direction")
    points = np.array([[x, y, 0.0] for y in ys for x in xs])

    def vid(i, j):
        return j * (nx + 1) + i

    quads = [
        [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)]
        for j in range(ny)
        for i in range(nx)
    ]
    return Mesh.from_polygons(points, quads)


def _two_sided(left: float, right: float, spacing: float, growth: float):
    """Cell sizes across ``[left, right]``: smallest at both ends, largest mid-span.

    A one-sided geometric fan would refine one interface and starve the other.
    The weights are ``growth ** min(i, n-1-i)``, which is symmetric, so both
    bounding interfaces get the fine cells.
    """
    length = right - left
    for count in range(1, 4096):
        weights = growth ** np.minimum(np.arange(count), np.arange(count)[::-1])
        sizes = length * weights / weights.sum()
        if sizes.min() <= spacing:
            return left + np.concatenate([[0.0], np.cumsum(sizes)])
    raise ValueError("could not reach the requested spacing")  # pragma: no cover


def graded_triangles(xs, ys) -> Mesh:
    """:func:`graded_quads` with every cell split into two triangles.

    Triangles are the 2D simplex, and on a simplex the Mimetic-AFW stabilisation
    vanishes identically (``3 edges x 4 DOFs = 12 = d^2(d+1) = m``, so
    ``ker(N^T) = {0}``): the scheme reduces to the pure Arnold--Falk--Winther
    mixed element.  A quadrilateral carries ``16 > 12`` DOFs and therefore always
    carries a stabilisation term -- a genuine discretisation difference.
    """
    quads = graded_quads(xs, ys)
    points = quads.geometry.points
    tris = []
    for loop in quads.complex.polygon_loops:
        a, b, c, d = loop
        tris += [[a, b, c], [a, c, d]]
    return Mesh.from_polygons(points, tris)


def graded_coordinates(interfaces, extent, spacing, growth: float = 1.35,
                       max_spacing: float | None = None,
                       window: tuple[float, float] | None = None,
                       window_spacing: float | None = None):
    """Node coordinates that **honour** ``interfaces`` and cluster elements at them.

    Every value in ``interfaces`` becomes a node, so a discontinuity in material
    or loading lands on a cell face rather than bisecting a cell -- where a
    cell-centred test would put it on the wrong side and shift the answer by half
    a cell.

    Resolution is concentrated **at** the interfaces, not spread evenly between
    them.  That is where it is needed: the fields are non-smooth across a
    material or loading jump (in the displaced-fault benchmark the analytic
    Coulomb stress is logarithmically singular at the reservoir edges), so the
    discretisation error is dominated by the few cells nearest each interface
    and is negligible a handful of cells away.  ``spacing`` is therefore the size
    *at* an interface; cells grow by ``growth`` away from it -- towards mid-span
    between two interfaces, and outwards to ``extent`` beyond the outermost.

    ``window`` overrides that between two bounds: the range is meshed **uniformly**
    at ``window_spacing`` (default ``spacing``) and the geometric coarsening starts
    from its edges instead.  Clustering at interfaces is the right default when the
    error is concentrated *at* them, but it thins out in between -- and a solution
    that is smooth yet not small over a whole neighbourhood (the near field of a
    slipping fault, say) is then under-resolved there.
    The interfaces are still forced in, so nothing is lost if they fall off the
    uniform grid; they simply split one cell.
    """
    interfaces = np.unique(np.asarray(interfaces, dtype=float))
    lo, hi = float(interfaces[0]), float(interfaces[-1])
    if not (extent[0] <= lo and hi <= extent[1]):
        raise ValueError("interfaces must lie inside the extent")

    if window is None:
        outward = spacing
        nodes = [lo]
        for left, right in zip(interfaces[:-1], interfaces[1:]):
            nodes.extend(_two_sided(left, right, spacing, growth)[1:])
    else:
        lo, hi = float(min(window)), float(max(window))
        if not (extent[0] <= lo and hi <= extent[1]):
            raise ValueError("window must lie inside the extent")
        if lo > interfaces[0] or hi < interfaces[-1]:
            raise ValueError("window must contain every interface")
        outward = float(window_spacing or spacing)
        count = max(1, int(round((hi - lo) / outward)))
        nodes = list(np.linspace(lo, hi, count + 1))
        nodes.extend(interfaces.tolist())

    for start, stop, step in ((lo, extent[0], -1.0), (hi, extent[1], 1.0)):
        position, size = start, outward
        while (stop - position) * step > 1e-9:
            size *= growth
            if max_spacing is not None:
                size = min(size, max_spacing)
            position += step * size
            nodes.append(position if (stop - position) * step > 0 else stop)
        nodes.append(stop)
    return np.unique(np.asarray(nodes, dtype=float))


def structured_triangles(
    nx: int,
    ny: int,
    lengths: tuple[float, float] = (1.0, 1.0),
    origin: tuple[float, float] = (0.0, 0.0),
) -> Mesh:
    """A conforming triangular mesh of a rectangle (each quad split in two).

    Triangles are the 2D simplex, where the elasticity stabilization vanishes
    (``3 edges x 4 DOFs = 12 = d^2(d+1)``).
    """
    quads = structured_quads(nx, ny, lengths, origin)
    points = quads.geometry.points
    tris = []
    for loop in quads.complex.polygon_loops:
        a, b, c, d = loop
        tris += [[a, b, c], [a, c, d]]
    return Mesh.from_polygons(points, tris)
