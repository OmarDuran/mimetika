import numpy as np
import pytest

from mimetika.mesh import single_tetrahedron, structured_box
from mimetika.mesh.reference import reference_cells
from mimetika.operators.elasticity import (
    ElasticityInnerProduct,
    compliance,
    compliance_contraction,
)
from mimetika.operators.inner_product import (
    assemble_local_inner_product,
    consistency_residual,
)

CELLS = [c for c in reference_cells() if c.dim >= 1]
IDS = [c.name for c in CELLS]


def _local(rc, mu=1.3, lam=2.7):
    ip = ElasticityInnerProduct(rc.mesh, mu=mu, lam=lam)
    N, R, Kbar, vol, lc = ip.local_matrices(0)
    return ip, N, R, Kbar, vol, lc


# -- the compliance tensor ----------------------------------------------------


def test_compliance_inverts_the_elastic_tensor():
    rng = np.random.default_rng(0)
    mu, lam = 1.3, 2.7
    for d in (1, 2, 3):
        T = rng.standard_normal((d, d))
        stiff = 2 * mu * T + lam * np.trace(T) * np.eye(d)
        assert np.allclose(compliance(stiff, mu, lam), T)


def test_compliance_matches_the_paper_formula_in_3d():
    mu, lam = 0.9, 1.7
    rng = np.random.default_rng(1)
    T = rng.standard_normal((3, 3))
    expected = T / (2 * mu) - lam / (2 * mu * (2 * mu + 3 * lam)) * np.trace(T) * np.eye(3)
    assert np.allclose(compliance(T, mu, lam), expected)


# -- structure of the local inner product -------------------------------------


@pytest.mark.parametrize("rc", CELLS, ids=IDS)
def test_local_matrix_is_spd(rc):
    ip, N, R, Kbar, vol, _ = _local(rc)
    M = assemble_local_inner_product(N, R, Kbar, vol)
    assert np.allclose(M, M.T)
    assert np.linalg.eigvalsh(M).min() > 0


@pytest.mark.parametrize("rc", CELLS, ids=IDS)
def test_strong_consistency(rc):
    ip, N, R, Kbar, vol, _ = _local(rc)
    M = assemble_local_inner_product(N, R, Kbar, vol)
    assert consistency_residual(M, N, R) < 1e-9


@pytest.mark.parametrize("rc", CELLS, ids=IDS)
def test_canonical_moments_satisfy_the_gram_identity(rc):
    """``N^T R = |E| Kbar`` on the constant-stress block, which is canonical."""
    ip, N, R, Kbar, vol, lc = _local(rc)
    nc = lc.dim**2
    assert np.abs(N.T @ R[:, :nc] - vol * Kbar[:, :nc]).max() < 1e-9


@pytest.mark.parametrize("rc", CELLS, ids=IDS)
def test_dof_and_mode_counts(rc):
    ip, N, _, _, _, lc = _local(rc)
    d = lc.dim
    assert N.shape == (d * d * lc.n_facets, d * d * (d + 1))


def test_stabilization_vanishes_on_every_simplex():
    """The design requirement, now across all dimensions."""
    for name in ("segment-unit", "triangle-unit", "triangle-irregular", "tet-reference", "tet-irregular"):
        rc = next(c for c in CELLS if c.name == name)
        assert ElasticityInnerProduct(rc.mesh).stabilization_dim(0) == 0, name


def test_stabilization_active_on_polytopes():
    expected = {
        "square-unit": 16 - 12,
        "hexagon-regular": 24 - 12,
        "cube-unit": 54 - 36,
        "prism-triangular": 45 - 36,
        "pyramid-square": 45 - 36,
    }
    for name, dim_stab in expected.items():
        rc = next(c for c in CELLS if c.name == name)
        assert ElasticityInnerProduct(rc.mesh).stabilization_dim(0) == dim_stab, name


# -- exactness on the reconstruction space ------------------------------------


@pytest.mark.parametrize("rc", CELLS, ids=IDS)
def test_inner_product_is_exact_on_constant_stresses(rc):
    """``g^T M g`` reproduces ``int C^{-1} T : T`` for constant stress fields."""
    mu, lam = 1.3, 2.7
    ip, N, R, Kbar, vol, lc = _local(rc, mu, lam)
    M = assemble_local_inner_product(N, R, Kbar, vol)
    d = lc.dim
    rng = np.random.default_rng(2)
    for _ in range(3):
        c = np.zeros(N.shape[1])
        c[: d * d] = rng.standard_normal(d * d)  # constant modes only
        g = N @ c
        T = c[: d * d].reshape(d, d)
        exact = vol * compliance_contraction(T, T, mu, lam)
        assert np.isclose(g @ M @ g, exact)


def test_incompressible_limit_stays_spd():
    for lam in (1.0, 1e3, 1e6):
        for mesh in (single_tetrahedron(), structured_box(1, 1, 1)):
            M, _ = ElasticityInnerProduct(mesh, mu=1.0, lam=lam).local(0)
            assert np.linalg.eigvalsh(M).min() > 0


def test_global_assembly_shape_and_symmetry():
    mesh = structured_box(1, 1, 1)
    A = ElasticityInnerProduct(mesh).assemble()
    n = 9 * mesh.num_cells(2)
    assert A.shape == (n, n)
    assert (abs(A - A.T) > 1e-9).nnz == 0


# -- closed-form facet data ---------------------------------------------------


@pytest.mark.parametrize("rc", [c for c in CELLS if c.dim == 3], ids=[c.name for c in CELLS if c.dim == 3])
def test_closed_form_facet_data_matches_quadrature(rc):
    """The 3D fast path uses closed forms; it must equal the quadrature route."""
    from mimetika.geometry.local_cell import LocalCell

    ip = ElasticityInnerProduct(rc.mesh)
    lc = LocalCell.build(rc.mesh.geometry, 0, ip.frame)
    scale = ip._scale(lc)
    m_fast, x_fast = ip.facet_data(lc, scale)
    m_ref, x_ref = ip._facet_data_quadrature(lc, scale)
    assert np.allclose(m_fast, m_ref, atol=1e-13)
    assert np.allclose(x_fast, x_ref, atol=1e-13)


@pytest.mark.parametrize("rc", [c for c in CELLS if c.dim == 3], ids=[c.name for c in CELLS if c.dim == 3])
def test_facet_second_moments_match_quadrature(rc):
    g = rc.mesh.geometry
    S = g.facet_second_moments()
    for f in range(rc.mesh.num_cells(2)):
        qp, qw = g.quadrature(2, f)
        rel = qp - g.centroids(2)[f]
        assert np.allclose(S[f], np.einsum("q,qi,qj->ij", qw, rel, rel), atol=1e-13)


# -- batched vs per-cell ------------------------------------------------------


@pytest.mark.parametrize(
    "mesh",
    [structured_box(2, 2, 2), reference_cells(dim=3)[6].mesh],
    ids=["hex", "dented"],
)
def test_batched_local_matrices_match_per_cell(mesh):
    """The grouped/batched path must reproduce the scalar path exactly."""
    ip = ElasticityInnerProduct(mesh, mu=1.3, lam=2.7)
    for facet_ids, signs, cells in ip.cell_groups():
        Nb, Rb, Kb, volb, Xb = ip.local_matrices_batched(facet_ids, signs, cells)
        Mb, deficient = ip.local_inner_products_batched(Nb, Rb, Kb, volb)
        assert not deficient.any()
        for b, c in enumerate(cells):
            N, R, K, vol, lc = ip.local_matrices(int(c))
            assert np.allclose(Nb[b], N, atol=1e-12)
            assert np.allclose(Rb[b], R, atol=1e-12)
            assert np.allclose(Kb[b], K, atol=1e-12)
            assert np.isclose(volb[b], vol)
            assert np.allclose(
                Mb[b], assemble_local_inner_product(N, R, K, vol), atol=1e-10
            )


def test_cell_groups_cover_every_cell_once():
    mesh = reference_cells(dim=3)[6].mesh
    ip = ElasticityInnerProduct(mesh)
    seen = np.concatenate([c for _, _, c in ip.cell_groups()])
    assert sorted(seen) == list(range(mesh.num_cells(3)))
    for facet_ids, signs, cells in ip.cell_groups():
        for b, c in enumerate(cells):
            expected = mesh.complex.facets_of(3, int(c))
            assert list(facet_ids[b]) == [f for f, _ in expected]
            assert list(signs[b]) == [s for _, s in expected]
