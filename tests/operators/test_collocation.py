r"""Circumcentric collocation: making ``d || n`` hold on a simplicial mesh.

:class:`LumpedDeviatoricStress` is consistent only where the offset from the cell
collocation point to a facet centroid is parallel to that facet's normal.  The
centroid does not give that on a simplex -- but the collocation point is a *free*
parameter of the consistency derivation (``u(x_i) - u(x_c) = eps (x_i - x_c)`` holds
for any ``x_c``), so it can be chosen to make the condition true.

The object being asked for is an **orthogonal complex**: a polytopal mesh carrying one
point per cell such that every facet is orthogonal to the segment joining the two cell
points it separates.  Voronoi/PEBI generators are the general construction.
Circumcentres are the simplicial special case via Delaunay duality -- exact in 2D, and
in 3D not merely inexact but sometimes worse than the centroid, see
:func:`test_in_3d_the_circumcentre_is_not_enough`.

Every test checks against something independent of the code under test: equidistance
is measured from the vertices, and wherever the circumcentre is asserted to succeed
the centroid is asserted to *fail*, so a vacuously-true property cannot masquerade
as a result.
"""

import numpy as np
import pytest

from mimetika.mesh import Mesh, single_tetrahedron, structured_quads
from mimetika.operators import LumpedDeviatoricStress, circumcentres

MU = 2.5

#: Acute, so the circumcentre lies strictly inside and every ``d_n > 0``.  A *right*
#: triangle would be a bad choice: its circumcentre is the hypotenuse midpoint, which
#: sits on a facet and gives ``d_n = 0``.
ACUTE = [(0.0, 0.0), (1.0, 0.0), (0.5, 0.9)]

#: Obtuse: the circumcentre falls outside the cell, which must be caught.
OBTUSE = [(0.0, 0.0), (1.0, 0.0), (0.15, 0.25)]


def triangle(vertices) -> Mesh:
    return Mesh.from_polygons(
        np.array([[x, y, 0.0] for x, y in vertices]), [[0, 1, 2]]
    )


def tetrahedron() -> Mesh:
    return single_tetrahedron(
        # deliberately NOT near-regular: on an equilateral face the centroid and
        # the circumcentre coincide, which would hide the very effect being tested
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                  [0.25, 0.9, 0.0], [0.45, 0.3, 1.1]])
    )


def defects(mesh, collocation=None) -> np.ndarray:
    """Orthogonality defects with the guard disarmed.

    The constructor calls ``check_orthogonality`` itself, so a non-orthogonal cell
    raises before it can be measured.  Here we deliberately want to *measure* the
    failure, so the tolerance is opened up.
    """
    return LumpedDeviatoricStress(
        mesh, mu=MU, collocation=collocation, orthogonality_tol=1e9
    ).orthogonality_defects()


# -- the circumcentre is what it claims to be ------------------------------------------


@pytest.mark.parametrize("mesh_of", [lambda: triangle(ACUTE), tetrahedron])
def test_the_circumcentre_is_equidistant_from_every_vertex(mesh_of):
    """The definition, measured from the vertices -- not the formula re-run."""
    mesh = mesh_of()
    centre = circumcentres(mesh)[0]
    verts = sorted(mesh.complex.cell_vertices(0, mesh.dim))
    radii = np.linalg.norm(mesh.geometry.points[verts] - centre, axis=1)
    assert np.allclose(radii, radii[0], atol=1e-12), radii


def test_circumcentres_refuse_a_non_simplex():
    with pytest.raises(ValueError, match="simplices"):
        circumcentres(structured_quads(1, 1))


# -- 2D: it delivers orthogonality, and the centroid does not --------------------------


def test_circumcentric_collocation_gives_orthogonality_in_2d():
    """Both directions, so the test cannot pass by the property being trivial."""
    mesh = triangle(ACUTE)
    assert np.max(defects(mesh, circumcentres(mesh))) < 1e-12
    assert np.max(defects(mesh)) > 1e-3  # the centroid is genuinely not orthogonal


def test_the_guard_fires_for_centroids_on_a_triangle_and_not_for_circumcentres():
    """Construction itself is the guard -- a non-orthogonal cell never assembles."""
    mesh = triangle(ACUTE)
    LumpedDeviatoricStress(mesh, mu=MU, collocation=circumcentres(mesh))
    with pytest.raises(ValueError, match="not orthogonal"):
        LumpedDeviatoricStress(mesh, mu=MU)


# -- 3D: it is not enough, and this is why ----------------------------------------------


def test_in_3d_the_circumcentre_is_not_enough():
    r"""Recorded as a positive test because it is a real limitation, not a bug.

    The foot of the perpendicular from a simplex circumcentre to a face is that
    **face's circumcentre**.  In 2D a facet is an edge, whose circumcentre is its
    midpoint *is* its centroid, so ``d || n`` holds exactly.  In 3D a facet is a
    triangle, whose circumcentre differs from its centroid -- and the facet points
    used throughout the library are centroids.  So the offset misses the normal by a
    few percent no matter how good the tetrahedralisation is.

    Fixing it needs circumcentric *facet* points too -- the full circumcentric dual,
    not merely a different cell point.  Until then the lumped operator is restricted
    to 2D simplices and to Cartesian or Voronoi cells in 3D.
    """
    mesh = tetrahedron()
    circum = np.max(defects(mesh, circumcentres(mesh)))
    centroid = np.max(defects(mesh))
    # far above round-off -- in 2D the same measurement is ~1e-16
    assert circum > 1e-8, "if this now passes, 3D circumcentric collocation works"
    # and it is not even reliably an improvement: on a distorted tetrahedron the
    # circumcentre is measurably WORSE than the centroid, which is the clearest
    # evidence that simplex-circumcentre thinking is the wrong frame here
    assert circum > centroid, (circum, centroid)


# -- the failure mode that has to be caught ---------------------------------------------


def test_an_obtuse_simplex_is_rejected_rather_than_silently_accepted():
    """The circumcentre leaves the cell, ``d_n`` goes non-positive and positive
    definiteness is lost.  This is TPFA's negative-transmissibility failure; silently
    accepting it is the worst outcome, since the operator would assemble and return a
    plausible wrong answer."""
    mesh = triangle(OBTUSE)
    centre = circumcentres(mesh)[0]

    # independent confirmation that the circumcentre really is outside the triangle
    a, b, c = mesh.geometry.points[sorted(mesh.complex.cell_vertices(0, 2))][:, :2]
    def side(p, q, r):
        return np.sign((q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0]))
    turns = {side(a, b, centre[:2]), side(b, c, centre[:2]), side(c, a, centre[:2])}
    assert len(turns) > 1, "test premise broken: this triangle is not obtuse"

    with pytest.raises(ValueError):
        LumpedDeviatoricStress(mesh, mu=MU, collocation=circumcentres(mesh))


# -- the default must not have moved ----------------------------------------------------


def test_omitting_collocation_is_exactly_the_centroid():
    """Bit-identical when the argument is not passed.

    Checked on a quadrilateral, which is orthogonal about its centroid -- on a
    triangle neither operator could be built at all.
    """
    mesh = structured_quads(2, 2)
    default = LumpedDeviatoricStress(mesh, mu=MU)
    explicit = LumpedDeviatoricStress(
        mesh, mu=MU, collocation=mesh.geometry.centroids(2)
    )
    assert np.array_equal(default.collocation, explicit.collocation)
    assert np.array_equal(
        default.assemble().toarray(), explicit.assemble().toarray()
    )
