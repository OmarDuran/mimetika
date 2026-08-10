"""Consistency-only (BDM-type) scalar inner product.

What must hold, cell by cell: the selected basis is unisolvent (square
invertible ``N`` -- no stabilization has room to exist), ``M`` is SPD, and the
(S2) identity ``M N e_c = R e_c`` holds for the constant modes.  On simplices
the space is ``BDM_1`` with no enrichment; globally, a mixed solve reproduces
linear pressure fields exactly (the patch test).
"""

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from mimetika.geometry.local_cell import LocalCell
from mimetika.mesh import structured_box, structured_quads
from mimetika.mesh.reference import reference_cells
from mimetika.operators.derham import DeRhamDiffusionInnerProduct

ANISO_K = np.array([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]])
CELLS = [c for c in reference_cells() if c.dim >= 2]
IDS = [c.name for c in CELLS]


def _local(rc, K=ANISO_K):
    ip = DeRhamDiffusionInnerProduct(rc.mesh, K=K)
    N, G, lc, degree = ip.local_matrices(0)
    M = np.linalg.solve(N.T, np.linalg.solve(N.T, G).T).T
    return ip, N, G, lc, degree, 0.5 * (M + M.T)


@pytest.mark.parametrize("rc", CELLS, ids=IDS)
def test_unisolvent_and_spd(rc):
    """N square invertible (pure consistency, no stabilization space), M SPD."""
    _, N, _, lc, _, M = _local(rc)
    n_dof = lc.dim * lc.n_facets
    assert N.shape == (n_dof, n_dof)
    assert np.linalg.matrix_rank(N, tol=1e-10) == n_dof
    assert np.linalg.eigvalsh(M).min() > 0


@pytest.mark.parametrize("rc", CELLS, ids=IDS)
def test_constant_consistency(rc):
    """(S2): ``M N e_c = R e_c`` for the constant modes, any cell shape."""
    ip, N, _, lc, _, M = _local(rc)
    R = ip.canonical_constant_moments(0)
    scale = np.abs(R).max()
    assert np.abs(M @ N[:, : lc.dim] - R).max() < 1e-9 * max(scale, 1.0)


@pytest.mark.parametrize("rc", CELLS, ids=IDS)
def test_simplices_need_no_enrichment(rc):
    """On simplices the space is BDM_1 (degree 2 = plain P1); polytopes enrich."""
    _, _, _, lc, degree, _ = _local(rc)
    if lc.n_facets == lc.dim + 1:
        assert degree == 2
    else:
        assert degree >= 3


@pytest.mark.parametrize("rc", CELLS, ids=IDS)
def test_galerkin_energy_identity(rc):
    """``N^T M N = G``: M is the exact mass matrix of the reconstruction."""
    _, N, G, _, _, M = _local(rc)
    assert np.abs(N.T @ M @ N - G).max() < 1e-9 * np.abs(G).max()


def test_no_fallback_raises():
    """Insufficient max_degree is an error, never a silently stabilized cell."""
    rc = next(c for c in CELLS if c.name == "cube-unit")
    ip = DeRhamDiffusionInnerProduct(rc.mesh, max_degree=2)
    with pytest.raises(RuntimeError, match="no stabilization"):
        ip.local_matrices(0)


# -- patch test: a mixed solve reproduces linear pressures exactly ------------


def _mixed_patch(mesh, K, grad_p):
    """Solve K^{-1} u + grad p = 0, div u = 0 with exact Dirichlet pressure."""
    d = mesh.dim
    ip = DeRhamDiffusionInnerProduct(mesh, K=K)
    M = ip.assemble()
    n_cells, n_facets = mesh.num_cells(d), mesh.num_cells(d - 1)
    ndofs = d * n_facets

    p_exact = lambda x: float(np.asarray(x[:3]) @ grad_p)
    u_exact = -(K[:3, :3] @ grad_p)

    rows, cols, vals = [], [], []
    g = np.zeros(ndofs)
    u_interp = np.zeros(ndofs)
    n_sides = np.zeros(n_facets)
    for cid in range(n_cells):
        lc = LocalCell.build(mesh.geometry, cid, ip.frame)
        u_loc = lc.frame.T @ u_exact
        for i, fid in enumerate(lc.facet_ids):
            n_sides[fid] += 1
            rows.append(cid)
            cols.append(d * fid)  # the constant (average-flux) moment
            vals.append(lc.signs[i])
            # canonical interpolant: only the constant moment is nonzero
            u_interp[d * fid] = (
                lc.signs[i] * lc.facet_measures[i] * (u_loc @ lc.facet_normals[i])
            )
    D = sp.csr_matrix((vals, (rows, cols)), shape=(n_cells, ndofs))

    # boundary term: moments of the exact pressure on single-sided facets
    for cid in range(n_cells):
        lc = LocalCell.build(mesh.geometry, cid, ip.frame)
        for i, fid in enumerate(lc.facet_ids):
            if n_sides[fid] != 1:
                continue
            qp, _ = lc.facet_quadrature[i]
            pvals = np.array([p_exact(lc.to_ambient(x)[0]) for x in qp])
            coef = lc.expand_on_facet(i, pvals)
            g[d * fid : d * (fid + 1)] = lc.signs[i] * coef

    A = sp.bmat([[M, -D.T], [D, None]], format="csr")
    sol = spla.spsolve(A, np.concatenate([-g, np.zeros(n_cells)]))
    u_h, p_h = sol[:ndofs], sol[ndofs:]

    p_cells = np.array(
        [p_exact(x) for x in mesh.geometry.centroids(d)[:, :3]]
    )
    scale = max(np.abs(u_interp).max(), 1.0)
    return np.abs(u_h - u_interp).max() / scale, np.abs(p_h - p_cells).max()


def test_patch_2d_quads():
    err_u, err_p = _mixed_patch(
        structured_quads(3, 3), ANISO_K, np.array([1.0, -2.0, 0.0])
    )
    assert err_u < 1e-9
    assert err_p < 1e-9


def test_patch_3d_hexes():
    err_u, err_p = _mixed_patch(
        structured_box(2, 2, 2), np.diag([2.0, 1.0, 0.5]),
        np.array([1.0, 0.5, -1.0]),
    )
    assert err_u < 1e-9
    assert err_p < 1e-9
