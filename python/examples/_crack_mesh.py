r"""A conformal mesh for a straight inclined crack in a square domain.

WHY A GENERATOR AND NOT A BOX. An axis-aligned crack in a structured box is the
cheap way to a conforming fracture, and it is the wrong test twice over. It ties
the crack's resolution to the domain's -- a crack a tenth of the domain across
gets a tenth of the cells, so the paper's 108 to 432 interface elements would
cost a million cells -- and it is MIRROR SYMMETRIC about the crack plane, which
is exactly the configuration that has hidden contact bugs here before: the
facet normals do not alternate and the per-facet right-hand-side coefficients
cancel in pairs. An inclined crack through an unstructured, graded triangulation
has neither property.

THE MESH. gmsh meshes the square with the crack EMBEDDED as an internal curve,
so the triangulation is conformal across it -- the crack is a set of triangle
edges, with a triangle on each side, not a geometric line the mesh ignores. A
distance-graded size field puts the elements where the solution is.

THE EXTRUSION is done here rather than by gmsh, in one layer of prisms: a
conformal 2D triangulation extrudes to a conformal 3D mesh by construction, and
each crack EDGE becomes a crack FACE -- a quadrilateral with a prism on each
side. That is what the 3D strong-symmetry construction needs, and it is the same
device the paper uses to reach a plane-strain state.

THE TAG is the convention mimetika reads: one PointData array `f<i>_node_set`
per fracture, a FLAG PER POINT, nonzero on the nodes of that fracture
(include/mimetika/mesh/fracture.hpp:80). A facet joins the fracture when all of
its vertices carry the flag -- so the crack's quadrilaterals join, the triangular
prism caps never do, and the edges radiating from a tip node do not either,
because their far vertex is off the crack.
"""

import math

import numpy as np


def inclined_crack(path, *, half_length=1.0, domain=20.0, psi=20.0,
                   size_crack=0.05, size_far=1.0, grading=None, thickness=None,
                   refine_radius=0.0, order_quality=True):
    """Write a .vtu of a square domain with an embedded crack at `psi` degrees.

    Returns (n_cells, n_points, n_crack_nodes, n_crack_edges).

    `size_crack` is the target element size within `refine_radius` of the crack,
    blended to `size_far` over the next `grading` of distance. The crack carries
    about 2 * half_length / size_crack faces, which is the quantity the benchmark
    is read at -- it is set directly here rather than inherited from the domain,
    which is the whole point of meshing.

    REFINE A TUBE, NOT A LINE. With refine_radius = 0 the size field collapses
    from size_crack to size_far immediately off the crack, so halving size_crack
    adds elements only ON the crack and leaves the elastic field around it at the
    same resolution. The crack's facet count then rises while the bulk does not,
    and no bulk convergence rate can be observed: measured 23 -> 108 crack faces
    for 1114 -> 1646 cells, with the error flat. A positive refine_radius
    refines a neighbourhood, so the cell count grows like h^-2 and the ladder is
    a genuine refinement of the problem rather than of its boundary.
    """
    import gmsh

    b, L = half_length, domain
    psi_r = math.radians(psi)
    grading = grading if grading is not None else 2.0 * b
    thickness = thickness if thickness is not None else size_crack

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("crack")
        geo = gmsh.model.geo

        h = 0.5 * L
        corners = [geo.addPoint(x, y, 0.0, size_far)
                   for x, y in ((-h, -h), (h, -h), (h, h), (-h, h))]
        edges = [geo.addLine(corners[i], corners[(i + 1) % 4]) for i in range(4)]
        surface = geo.addPlaneSurface([geo.addCurveLoop(edges)])

        # the crack, as its own curve -- EMBEDDED, so the triangulation carries
        # it as edges instead of meshing straight through it
        tip_a = geo.addPoint(-b * math.cos(psi_r), -b * math.sin(psi_r), 0.0, size_crack)
        tip_b = geo.addPoint(b * math.cos(psi_r), b * math.sin(psi_r), 0.0, size_crack)
        crack = geo.addLine(tip_a, tip_b)
        geo.synchronize()
        gmsh.model.mesh.embed(1, [crack], 2, surface)

        # size_crack on the crack, growing to size_far over `grading`
        dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist, "CurvesList", [crack])
        thr = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thr, "InField", dist)
        gmsh.model.mesh.field.setNumber(thr, "SizeMin", size_crack)
        gmsh.model.mesh.field.setNumber(thr, "SizeMax", size_far)
        gmsh.model.mesh.field.setNumber(thr, "DistMin", refine_radius)
        gmsh.model.mesh.field.setNumber(thr, "DistMax", refine_radius + grading)
        gmsh.model.mesh.field.setAsBackgroundMesh(thr)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.model.mesh.generate(2)
        if order_quality:
            gmsh.model.mesh.optimize("Netgen")

        tags, coords, _ = gmsh.model.mesh.getNodes()
        xy = np.asarray(coords, float).reshape(-1, 3)
        index = np.full(int(tags.max()) + 1, -1, np.int64)
        index[np.asarray(tags, np.int64)] = np.arange(len(tags))
        tri = np.asarray(gmsh.model.mesh.getElementsByType(2)[1], np.int64).reshape(-1, 3)
        tri = index[tri]
        # the crack's own 2-node line elements, which the embedding guarantees
        # are edges of the triangulation
        line = np.asarray(gmsh.model.mesh.getElementsByType(1)[1], np.int64).reshape(-1, 2)
        line = index[line]
    finally:
        gmsh.finalize()

    # the crack nodes, by geometry: on the segment, to a tolerance well under
    # the element size. Taken from coordinates rather than from the line
    # elements because the boundary edges are line elements too.
    d = np.array([math.cos(psi_r), math.sin(psi_r)])
    n = np.array([-math.sin(psi_r), math.cos(psi_r)])
    s, off = xy[:, :2] @ d, xy[:, :2] @ n
    tol = 1e-9 * max(1.0, L)
    on_crack = (np.abs(off) < tol) & (np.abs(s) <= b + tol)

    # one layer of prisms: node v at z = 0 and node v + N at z = thickness
    N = len(xy)
    pts = np.vstack([np.column_stack([xy[:, 0], xy[:, 1], np.zeros(N)]),
                     np.column_stack([xy[:, 0], xy[:, 1], np.full(N, thickness)])])
    wedge = np.hstack([tri, tri + N])
    mask = np.concatenate([on_crack, on_crack]).astype(np.int32)

    import meshio
    meshio.write(path, meshio.Mesh(pts, [("wedge", wedge)],
                                   point_data={"f1_node_set": mask}))

    n_edges = int(((np.abs(off[line]) < tol).all(axis=1) &
                   (np.abs(s[line]) <= b + tol).all(axis=1)).sum())
    return len(wedge), len(pts), int(mask.sum()), n_edges
