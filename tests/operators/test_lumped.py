"""The lumped (diagonal) deviatoric stress inner product.

Two claims carry this operator, and this file exists to hold them apart.

The first is that on an **orthogonal** cell -- one where the offset
``d = x_facet - x_collocation`` is parallel to the facet normal -- the diagonal
matrix ``d_n/(2 mu |e|) I`` is not an approximation but the *exact* consistent
inner product for constant stresses: ``M N = R`` to round-off.  That is the
whole point, so it is checked directly against the continuum identity
``M (sigma n) = sigma d / (2 mu |e|)`` and against the closed-form compliance,
never against a stored number.

The second is that this fails everywhere else, irrecoverably.  So the guard is
tested as a feature: it must fire on skewed cells, must name the offender, and
must stay quiet on rectangles, on a regular hexagon (orthogonal but not
Cartesian) and on a triangle collocated at its circumcentre.  The ``(n, t)``
block test pins *why*: the unique consistent facet block is
``[[d_n, d_t], [-d_t, d_n]]``, whose off-diagonal is exactly the defect the
guard measures.

Finally, the Poisson ratio must be invisible to ``M``.  Splitting the trace off
removed the material condition; all of the ``nu`` dependence has to live in the
rank-one volumetric term, and the two are checked to recombine into the exact
compliance.
"""

import numpy as np
import pytest

from mimetika.geometry.local_cell import LocalCell
from mimetika.materials import Material
from mimetika.mesh import (
    graded_quads,
    structured_box,
    structured_quads,
    structured_triangles,
)
from mimetika.mesh.mesh import Mesh
from mimetika.mesh.reference import reference_cells
from mimetika.operators.elasticity import (
    ElasticityInnerProduct,
    compliance_contraction,
)
from mimetika.operators.inner_product import consistency_residual
from mimetika.operators.lumped import LumpedDeviatoricStress

MU, LAM = 1.3, 2.7

# Cells on which ``x_facet - x_centroid`` is parallel to the facet normal.  The
# hexagon and the tilted square are in deliberately: an operator that only
# worked on axis-aligned boxes would still pass a Cartesian-only list.
ORTHOGONAL = [
    "segment-unit",
    "segment-oblique",
    "segment-diagonal",
    "square-unit",
    "square-tilted",
    "hexagon-regular",
    "cube-unit",
]
# ... and cells on which it is not, in every dimension and both convexities.
SKEWED = [
    "triangle-unit",
    "quad-irregular",
    "lshape-nonconvex",
    "lshape-tilted",
    "tet-reference",
    "hex-sheared",
    "prism-triangular",
    "pyramid-square",
    "cube-dented",
]

CARTESIAN = [
    ("graded-quads", graded_quads([0.0, 0.3, 1.0, 2.5], [0.0, 0.7, 1.0])),
    ("box", structured_box(2, 2, 2)),
]
CARTESIAN_IDS = [name for name, _ in CARTESIAN]
CARTESIAN_MESHES = [mesh for _, mesh in CARTESIAN]


def _mesh(name: str) -> Mesh:
    return next(c for c in reference_cells() if c.name == name).mesh


def _skewed(name: str) -> LumpedDeviatoricStress:
    """The operator on a cell it is *not* valid for -- the guard turned off.

    ``1.0`` disables it because the defect is a sine and cannot exceed one.
    """
    return LumpedDeviatoricStress(_mesh(name), mu=MU, lam=LAM, orthogonality_tol=1.0)


def _deviatoric(rng, d: int) -> np.ndarray:
    """A random symmetric trace-free tensor -- the stresses the operator owns."""
    A = rng.standard_normal((d, d))
    S = A + A.T
    return S - np.trace(S) * np.eye(d) / d


def _circumcentre(p: np.ndarray) -> np.ndarray:
    """Circumcentre of a triangle given as three rows in ``R^3``."""
    a, b = p[0] - p[2], p[1] - p[2]
    n = np.cross(a, b)
    return p[2] + np.cross((a @ a) * b - (b @ b) * a, n) / (2.0 * (n @ n))


# -- the reduced stress space -------------------------------------------------


def test_the_facet_space_is_one_traction_vector_not_the_afw_block():
    """``d`` DOFs per facet and ``d^2`` modes -- a different space, not a rescaling."""
    for mesh in CARTESIAN_MESHES:
        d = mesh.dim
        lumped = LumpedDeviatoricStress(mesh, mu=MU, lam=LAM)
        full = ElasticityInnerProduct(mesh, mu=MU, lam=LAM)
        assert lumped.dofs_per_facet(d) == d
        assert lumped.n_modes(d) == d * d
        assert full.dofs_per_facet(d) == d * d  # the block that was dropped
        N, R, Kbar, _, lc = lumped.local_matrices(0)
        assert N.shape == (lc.n_facets * d, d * d)
        assert R.shape == N.shape
        assert Kbar.shape == (d * d, d * d)
        assert lumped.assemble().shape == (
            d * mesh.num_cells(d - 1),
            d * mesh.num_cells(d - 1),
        )


@pytest.mark.parametrize("name", ORTHOGONAL + SKEWED)
def test_the_gram_identity_holds_whether_or_not_the_cell_is_orthogonal(name):
    """``N^T R = |E| Kbar`` is the divergence theorem, so geometry cannot break it.

    Separating this from ``M N = R`` is the point: the energy identity survives
    on every cell, and orthogonality is needed for *strong* consistency alone.
    """
    ip = _skewed(name)
    N, R, Kbar, vol, _ = ip.local_matrices(0)
    assert np.abs(N.T @ R - vol * Kbar).max() < 1e-14


def test_the_outward_normals_agree_with_the_local_cell_frame():
    """The guard reads geometry its own way; it must not drift from LocalCell.

    ``LocalCell`` orients facets by star-shapedness about the centroid, which is
    unusable once the collocation point may be a circumcentre, so this class
    derives outward normals from the incidence signs instead.  The two routes
    have to agree wherever both are valid.
    """
    for name in ORTHOGONAL + SKEWED:
        ip = _skewed(name)
        lc = LocalCell.build(ip.mesh.geometry, 0, ip.frame)
        facet_ids, signs, cells = next(iter(ip.cell_groups()))
        _, n_out, _ = ip._facet_geometry(facet_ids, signs, cells)
        assert list(facet_ids[0]) == list(lc.facet_ids), name
        assert np.abs(n_out[0] @ ip.frame - lc.facet_normals).max() < 1e-14, name


# -- consistency: the whole point ---------------------------------------------


@pytest.mark.parametrize("mesh", CARTESIAN_MESHES, ids=CARTESIAN_IDS)
def test_the_lumped_matrix_reproduces_the_traction_of_a_constant_deviatoric_stress(
    mesh,
):
    """``M g = S d / (2 mu)`` facet by facet, for constant trace-free ``S``.

    The continuum statement is ``M_i (S n_i) = S d_i / (2 mu |e_i|)``; with the
    integrated DOF ``g_i = |e_i| S n_i`` that is what is checked here, against
    ``S d_i`` rebuilt from the mesh rather than from the operator.
    """
    d = mesh.dim
    ip = LumpedDeviatoricStress(mesh, mu=MU, lam=LAM)
    rng = np.random.default_rng(0)
    for cell in (0, mesh.num_cells(d) // 2, mesh.num_cells(d) - 1):
        N, _, _, _, lc = ip.local_matrices(cell)
        M, _ = ip.local(cell)
        for _ in range(3):
            S = _deviatoric(rng, d)
            got = (M @ (N @ S.reshape(-1))).reshape(lc.n_facets, d)
            exact = (lc.facet_centroids @ S.T) / (2.0 * MU)
            assert np.abs(exact).max() > 1e-3  # would pass trivially on zero
            assert np.abs(got - exact).max() < 1e-14


@pytest.mark.parametrize("name", ORTHOGONAL)
def test_strong_consistency_holds_on_every_orthogonal_cell(name):
    """``M N = R`` -- exactness for constant stresses, in every dimension."""
    ip = LumpedDeviatoricStress(_mesh(name), mu=MU, lam=LAM)
    N, R, _, _, _ = ip.local_matrices(0)
    M, _ = ip.local(0)
    assert consistency_residual(M, N, R) < 1e-14


def test_the_consistent_facet_block_is_diagonal_exactly_when_d_is_parallel_to_n():
    """In the ``(n, t)`` basis the unique block is ``[[d_n, d_t], [-d_t, d_n]]``.

    Solved here from the deviatoric modes alone, with no diagonality assumed, so
    it says what the derivation says: the off-diagonal *is* ``d_t``.  On a
    rectangle it vanishes and the block collapses onto the operator's; on a
    skewed quad it does not, and the block is not even symmetric -- which is the
    part the rotation multiplier cannot absorb.
    """
    dev = [np.array([[1.0, 0.0], [0.0, -1.0]]), np.array([[0.0, 1.0], [1.0, 0.0]])]

    def blocks(ip):
        lc = LocalCell.build(ip.mesh.geometry, 0, ip.frame)
        offset = ip._local_offset(lc, 0)
        out = []
        for i in range(lc.n_facets):
            n = lc.facet_normals[i]
            t = np.array([-n[1], n[0]])
            offs = lc.facet_centroids[i] - offset
            scale = 2.0 * MU * lc.facet_measures[i]
            # M (S n) = S d / (2 mu |e|), two modes -> four equations, four unknowns
            M = np.linalg.solve(
                np.array([S @ n for S in dev]), np.array([S @ offs for S in dev]) / scale
            ).T
            frame = np.column_stack([n, t])
            expected = np.array([[offs @ n, offs @ t], [-(offs @ t), offs @ n]]) / scale
            out.append((frame.T @ M @ frame, expected, (offs @ t) / np.linalg.norm(offs)))
        return out

    for M_nt, expected, defect in blocks(LumpedDeviatoricStress(_mesh("square-unit"))):
        assert np.abs(M_nt - expected).max() < 1e-14
        assert abs(defect) < 1e-14
        assert abs(M_nt[0, 1]) < 1e-14  # diagonal, and so lumpable

    skewed = blocks(_skewed("quad-irregular"))
    assert max(abs(defect) for _, _, defect in skewed) > 0.3  # genuinely skewed
    for M_nt, expected, defect in skewed:
        assert np.abs(M_nt - expected).max() < 1e-14
        assert abs(M_nt[0, 1]) > 1e-2  # not diagonal ...
        assert np.abs(M_nt - M_nt.T).max() > 1e-2  # ... and not symmetric either


def test_a_skewed_cell_admits_no_consistent_lumped_operator_at_all():
    """Pins *why* the guard is an error and not a knob.

    The diagonal the formula produces is the least-squares solution of ``M N =
    R`` -- there is no better diagonal -- and its residual is a finite fraction
    of ``R`` on a skewed cell.  Refusing to assemble is the only sound response.
    """
    for name in ("quad-irregular", "hex-sheared", "tet-reference"):
        ip = _skewed(name)
        N, R, _, _, _ = ip.local_matrices(0)
        M, _ = ip.local(0)
        assert consistency_residual(M, N, R) > 0.1 * np.abs(R).max(), name


# -- structure of the local matrix --------------------------------------------


@pytest.mark.parametrize("name", ORTHOGONAL)
def test_the_local_matrix_is_diagonal_positive_definite_and_two_point(name):
    """``M = diag(d_n / (2 mu |e|))``, rebuilt from geometry to compare against."""
    mesh = _mesh(name)
    d = mesh.dim
    ip = LumpedDeviatoricStress(mesh, mu=MU, lam=LAM)
    M, facet_ids = ip.local(0)
    lc = LocalCell.build(mesh.geometry, 0, ip.frame)
    expected = np.einsum("ic,ic->i", lc.facet_centroids, lc.facet_normals) / (
        2.0 * MU * lc.facet_measures
    )
    assert np.abs(M - np.diag(np.diag(M))).max() == 0.0
    assert np.linalg.eigvalsh(M).min() > 0.0
    assert np.abs(np.diag(M) - np.repeat(expected, d)).max() < 1e-14
    assert list(facet_ids) == list(lc.facet_ids)


@pytest.mark.parametrize("name", ORTHOGONAL)
def test_the_stabilization_space_is_what_the_reduced_dof_count_leaves_over(name):
    """``dim ker(N^T) = d (n_facets - d)``: non-zero even on a simplex.

    Unlike the AFW operator, whose stabilization vanishes on simplices, here the
    lumping -- not an ``s (I - Q Q^T)`` term -- is what fixes these directions.
    """
    ip = LumpedDeviatoricStress(_mesh(name), mu=MU, lam=LAM)
    lc = LocalCell.build(ip.mesh.geometry, 0, ip.frame)
    d = lc.dim
    assert ip.stabilization_dim(0) == d * (lc.n_facets - d)


# -- the orthogonality guard ---------------------------------------------------


@pytest.mark.parametrize("name", SKEWED)
def test_the_guard_fires_on_every_non_orthogonal_cell(name):
    with pytest.raises(ValueError, match="not orthogonal"):
        LumpedDeviatoricStress(_mesh(name), mu=MU, lam=LAM)


@pytest.mark.parametrize("name", ORTHOGONAL)
def test_the_guard_stays_quiet_on_orthogonal_cells(name):
    ip = LumpedDeviatoricStress(_mesh(name), mu=MU, lam=LAM)
    assert ip.orthogonality_defects().max() < 1e-14


def test_the_guard_stays_quiet_on_rectangles_and_fires_when_one_node_moves():
    """The same mesh, one interior node displaced: quiet before, loud after.

    Comparing two meshes that differ in a single vertex is what makes this a
    test of the geometric condition rather than of the mesh generator.
    """
    mesh = structured_quads(2, 2)
    LumpedDeviatoricStress(mesh, mu=MU, lam=LAM)  # rectangles: no complaint

    points = np.array(mesh.geometry.points)
    moved = int(np.argmin(np.linalg.norm(points - np.array([0.5, 0.5, 0.0]), axis=1)))
    points[moved] += np.array([0.17, -0.11, 0.0])
    loops = [list(loop) for loop in mesh.complex.polygon_loops]
    skew = Mesh.from_polygons(points, loops)

    defects = LumpedDeviatoricStress(skew, orthogonality_tol=1.0).orthogonality_defects()
    with pytest.raises(ValueError, match="not orthogonal") as excinfo:
        LumpedDeviatoricStress(skew, mu=MU, lam=LAM)
    assert f"cell {int(np.argmax(defects))}" in str(excinfo.value)
    assert f"{defects.max():.3e}" in str(excinfo.value)  # the measured defect


def test_the_guard_rejects_a_collocation_point_lying_on_a_facet():
    """A right triangle circumscribes about its hypotenuse midpoint: ``d_n = 0``.

    Parallel to ``n`` but at zero distance, so the block vanishes and ``M`` is
    only semi-definite -- a different failure from skewness, and it needs its
    own message.
    """
    mesh = structured_triangles(1, 1)
    loops = mesh.complex.polygon_loops
    centres = np.array(
        [_circumcentre(mesh.geometry.points[list(loop)]) for loop in loops]
    )
    with pytest.raises(ValueError, match="non-positive collocation distance"):
        LumpedDeviatoricStress(mesh, mu=MU, lam=LAM, collocation=centres)


def test_moving_the_collocation_point_to_the_circumcentre_admits_a_simplex():
    """The collocation point is free, and using it recovers a whole mesh family.

    An acute triangle is inadmissible at its centroid and exact at its
    circumcentre -- the Delaunay half of TPFA's admissible family, and the
    reason ``collocation`` exists at all.
    """
    mesh = _mesh("triangle-irregular")
    with pytest.raises(ValueError, match="not orthogonal"):
        LumpedDeviatoricStress(mesh, mu=MU, lam=LAM)

    centre = _circumcentre(mesh.geometry.points[list(mesh.complex.polygon_loops[0])])
    ip = LumpedDeviatoricStress(mesh, mu=MU, lam=LAM, collocation=centre[None, :])
    N, R, _, _, _ = ip.local_matrices(0)
    M, _ = ip.local(0)
    assert ip.orthogonality_defects().max() < 1e-14
    assert consistency_residual(M, N, R) < 1e-14
    assert np.linalg.eigvalsh(M).min() > 0.0


# -- independence from the Poisson ratio --------------------------------------


@pytest.mark.parametrize("mesh", CARTESIAN_MESHES, ids=CARTESIAN_IDS)
def test_the_lumped_matrix_does_not_depend_on_the_poisson_ratio(mesh):
    """Identical for ``nu = 0`` and ``nu = 0.3`` -- bit for bit, not merely close.

    Splitting the trace off removed the material condition from the consistency
    derivation, leaving only the geometric one.  The anti-triviality half of the
    test is that the *volumetric* coefficient does move: at ``nu = 0`` there is
    no volumetric coupling at all, at ``nu = 0.3`` there is, so the operator is
    not simply ignoring the material.
    """
    incompressible = Material(shear_modulus=MU, poisson=0.0)
    poisson = Material(shear_modulus=MU, poisson=0.3)
    a = LumpedDeviatoricStress(mesh, material=incompressible)
    b = LumpedDeviatoricStress(mesh, material=poisson)

    assert np.abs(a.diagonal() - b.diagonal()).max() == 0.0
    assert (abs(a.assemble() - b.assemble()) > 0.0).nnz == 0
    for cell in (0, mesh.num_cells(mesh.dim) - 1):
        for x, y in zip(a.local_matrices(cell)[:3], b.local_matrices(cell)[:3]):
            assert np.abs(x - y).max() == 0.0

    _, ca = a.volumetric_operator()
    _, cb = b.volumetric_operator()
    assert np.all(ca == 0.0)  # nu = 0: C^{-1} really is I/(2 mu)
    assert np.all(cb < 0.0)


# -- the rank-one volumetric coupling ------------------------------------------


@pytest.mark.parametrize("mesh", CARTESIAN_MESHES, ids=CARTESIAN_IDS)
def test_the_volumetric_update_is_rank_one_on_every_cell(mesh):
    """One singular value per cell, not two -- what makes Woodbury applicable."""
    ip = LumpedDeviatoricStress(mesh, mu=MU, lam=LAM)
    for cell in range(mesh.num_cells(mesh.dim)):
        w, _, c = ip.volumetric_coupling(cell)
        assert np.linalg.norm(w) > 1e-6
        singular = np.linalg.svd(c * np.outer(w, w), compute_uv=False)
        assert singular[1] < 1e-14 * singular[0]
    W, coefficients = ip.volumetric_operator()
    assert W.shape == (mesh.num_cells(mesh.dim), len(ip.diagonal()))
    assert np.linalg.matrix_rank(W.toarray()) == mesh.num_cells(mesh.dim)
    assert coefficients.shape == (mesh.num_cells(mesh.dim),)


@pytest.mark.parametrize("name", ORTHOGONAL + SKEWED)
def test_the_volumetric_vector_is_the_discrete_trace(name):
    """``w . g = |E| tr(sigma) / (2 mu)`` -- the functional the pressure pairs with.

    Another divergence-theorem identity, so it holds on skewed cells too: it is
    the *lumping*, not the trace, that needs orthogonality.
    """
    ip = _skewed(name)
    d = ip.mesh.dim
    N, _, _, vol, lc = ip.local_matrices(0)
    w, _, _ = ip.volumetric_coupling(0)
    w_outward = w * np.repeat(lc.signs, d)  # local_matrices uses outward normals
    rng = np.random.default_rng(4)
    for _ in range(3):
        T = rng.standard_normal((d, d))
        got = w_outward @ (N @ T.reshape(-1))
        assert np.isclose(got, vol * np.trace(T) / (2.0 * MU), atol=1e-13)


@pytest.mark.parametrize("name", ORTHOGONAL)
def test_the_lumped_matrix_plus_the_rank_one_term_is_the_exact_compliance(name):
    """``g^T (M + c w w^T) g = int C^{-1} T : T`` for constant ``T``.

    The two halves have to add back up to the continuum energy, and the
    reference is the closed-form contraction, not a stored value.  Restricted to
    constant fields because that is the reconstruction space -- see
    :meth:`n_modes`.
    """
    ip = LumpedDeviatoricStress(_mesh(name), mu=MU, lam=LAM)
    d = ip.mesh.dim
    N, _, _, vol, lc = ip.local_matrices(0)
    M, _ = ip.local(0)
    w, _, c = ip.volumetric_coupling(0)
    w_outward = w * np.repeat(lc.signs, d)
    rng = np.random.default_rng(5)
    for _ in range(4):
        T = rng.standard_normal((d, d))
        g = N @ T.reshape(-1)
        energy = g @ M @ g + c * (w_outward @ g) ** 2
        assert np.isclose(energy, vol * compliance_contraction(T, T, MU, LAM))


# -- global assembly and the batched path --------------------------------------


@pytest.mark.parametrize("mesh", CARTESIAN_MESHES, ids=CARTESIAN_IDS)
def test_an_interior_facet_receives_both_half_compliances(mesh):
    """The TPFA structure: two half-compliances in series across a facet."""
    d = mesh.dim
    ip = LumpedDeviatoricStress(mesh, mu=MU, lam=LAM)
    contributions: dict[int, list[float]] = {}
    expected = np.zeros(d * mesh.num_cells(d - 1))
    for cell in range(mesh.num_cells(d)):
        values, facet_ids = ip.facet_compliances(cell)
        for value, facet in zip(values, facet_ids):
            expected[facet * d : (facet + 1) * d] += value
            contributions.setdefault(int(facet), []).append(float(value))
    assert np.abs(ip.diagonal() - expected).max() == 0.0

    interior = [f for f, halves in contributions.items() if len(halves) == 2]
    assert len(interior) > 0
    for facet in interior:
        halves = contributions[facet]
        assert min(halves) > 0.0  # each side contributes, neither cancels
        assert np.isclose(ip.diagonal()[facet * d], sum(halves))


@pytest.mark.parametrize("name", ORTHOGONAL)
def test_the_batched_path_reproduces_the_per_cell_one(name):
    ip = LumpedDeviatoricStress(_mesh(name), mu=MU, lam=LAM)
    for facet_ids, signs, cells in ip.cell_groups():
        Nb, Rb, Kb, volb, Xb = ip.local_matrices_batched(facet_ids, signs, cells)
        Mb, deficient = ip.local_inner_products_batched(Nb, Rb, Kb, volb)
        assert not deficient.any()
        for b, cell in enumerate(cells):
            N, R, Kbar, vol, _, X = ip.local_matrices(int(cell), with_facet_data=True)
            assert np.allclose(Nb[b], N, atol=1e-14)
            assert np.allclose(Rb[b], R, atol=1e-14)
            assert np.allclose(Kb[b], Kbar, atol=1e-14)
            assert np.allclose(Xb[b], X, atol=1e-14)
            assert np.isclose(volb[b], vol)
            # the batched inner product is the row-wise least-squares solution of
            # M N = R; on an orthogonal cell that is the closed-form diagonal
            assert np.allclose(Mb[b], ip.local(int(cell))[0], atol=1e-14)
