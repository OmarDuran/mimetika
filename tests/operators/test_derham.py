"""Row-wise enriched stress space in the three- and four-field formulations.

The load-bearing facts: constant stress states are reproduced exactly on
polytopal meshes (patch test, both formulations); the three- and four-field
solutions coincide (congruence with the volumetric pair); with the
**constant** solid pressure the folded operator differs from AFW on
simplices by the covariance of the linear trace (PSD, vanishing on constant
stresses); and with the **linear** solid pressure (the P1 trace moments,
``solid_pressure="linear"``) that gap closes exactly -- three copies of
BDM_1 plus the algebraic rotation and volumetric couplings *is* the AFW
element, degree of freedom for degree of freedom.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from mimetika.assembly.four_field import FourFieldElasticity
from mimetika.assembly.mixed import MixedElasticity
from mimetika.mesh import structured_box, structured_quads, structured_triangles
from mimetika.operators.elasticity import ElasticityInnerProduct
from mimetika.operators.derham import (
    DeRhamDeviatoricStress,
    DeRhamElasticityInnerProduct,
)

MU, LAM = 1.0, 1.0


def _linear_state(d):
    """A displacement gradient and its exact stress, ambient 3x3."""
    A = np.zeros((3, 3))
    if d == 3:
        A[:] = np.array([[0.3, -0.2, 0.1], [0.5, 0.1, -0.4], [0.2, 0.3, -0.1]])
    else:
        A[:2, :2] = np.array([[0.3, -0.2], [0.5, 0.1]])
    eps = 0.5 * (A + A.T)
    sigma = 2 * MU * eps
    sigma[:d, :d] += LAM * np.trace(eps[:d, :d]) * np.eye(d)
    return A, sigma


def _patch_errors(problem, d):
    A, sigma = _linear_state(d)
    sol = problem.solve(dirichlet=lambda x: np.asarray(x) @ A.T)
    sig_I = problem.interpolate_stress(lambda x: np.broadcast_to(sigma, (len(x), 3, 3)))
    u_I = problem.interpolate_displacement(lambda x: np.asarray(x) @ A.T)
    scale = max(np.abs(sig_I).max(), 1.0)
    return (
        np.abs(sol["stress"] - sig_I).max() / scale,
        np.abs(sol["displacement"] - u_I).max(),
    )


@pytest.mark.parametrize("formulation", [MixedElasticity, FourFieldElasticity])
def test_patch_2d_quads(formulation):
    mesh = structured_quads(3, 3)
    problem = formulation(
        mesh, inner=DeRhamDeviatoricStress(mesh, mu=MU, lam=LAM)
    )
    err_s, err_u = _patch_errors(problem, 2)
    assert err_s < 1e-9
    assert err_u < 1e-9


@pytest.mark.parametrize("formulation", [MixedElasticity, FourFieldElasticity])
def test_patch_3d_hexes(formulation):
    mesh = structured_box(2, 2, 2)
    problem = formulation(
        mesh, inner=DeRhamDeviatoricStress(mesh, mu=MU, lam=LAM)
    )
    err_s, err_u = _patch_errors(problem, 3)
    assert err_s < 1e-9
    assert err_u < 1e-9


def test_three_and_four_field_solutions_coincide():
    """The congruence: same space, same solution, in either formulation."""
    mesh = structured_quads(3, 2)
    space = DeRhamDeviatoricStress(mesh, mu=MU, lam=LAM)
    bend = lambda x: np.column_stack(
        [0.1 * x[:, 0] ** 2, -0.2 * x[:, 0] * x[:, 1], np.zeros(len(x))]
    )
    three = MixedElasticity(mesh, inner=space).solve(dirichlet=bend)
    four = FourFieldElasticity(mesh, inner=space).solve(dirichlet=bend)
    for field in ("stress", "displacement", "rotation"):
        ref = np.abs(three[field]).max()
        assert np.abs(three[field] - four[field]).max() < 1e-8 * max(ref, 1.0)


@pytest.mark.parametrize(
    "mesh_fn", [lambda: structured_triangles(2, 2), lambda: structured_box(1, 1, 1)],
    ids=["triangles", "hex"],
)
def test_folded_operator_vs_afw(mesh_fn):
    """``M_row + W^T diag(c) W`` vs the AFW product.

    On simplices the difference is the linear-trace covariance: PSD and zero
    on constant stresses (the two methods share the patch test but are not
    the same matrix).  On non-simplices they differ already through the
    stabilization/enrichment complement -- only the constant-stress
    consistency is shared, which is what is asserted.
    """
    mesh = mesh_fn()
    space = DeRhamDeviatoricStress(mesh, mu=MU, lam=LAM)
    W, c = space.volumetric_operator()
    folded = space.assemble() + W.T @ sp.diags(c) @ W
    afw = ElasticityInnerProduct(mesh, mu=MU, lam=LAM).assemble()
    diff = (folded - afw).toarray()
    assert np.abs(diff).max() > 1e-8  # genuinely different members

    problem = MixedElasticity(mesh, inner=space)  # for the interpolant only
    d = mesh.dim
    _, sigma = _linear_state(d)
    dofs = problem.interpolate_stress(
        lambda x: np.broadcast_to(sigma, (len(x), 3, 3))
    )
    assert np.abs(diff @ dofs).max() < 1e-9 * max(np.abs(dofs).max(), 1.0)

    if all(
        len(mesh.complex.facets_of(d, c)) == d + 1
        for c in range(mesh.num_cells(d))
    ):  # simplicial: the difference is exactly the trace covariance, PSD
        assert np.linalg.eigvalsh(0.5 * (diff + diff.T)).min() > -1e-9


def test_linear_solid_pressure_is_afw_on_simplices():
    """With p_s in P1 the row-wise four-field solution equals AFW's exactly."""
    mesh = structured_triangles(2, 2)
    space = DeRhamDeviatoricStress(mesh, mu=MU, lam=LAM)
    bend = lambda x: np.column_stack(
        [0.1 * x[:, 0] ** 2, -0.2 * x[:, 0] * x[:, 1], np.zeros(len(x))]
    )
    four = FourFieldElasticity(mesh, inner=space, solid_pressure="linear").solve(
        dirichlet=bend
    )
    afw = MixedElasticity(
        mesh, inner=ElasticityInnerProduct(mesh, mu=MU, lam=LAM)
    ).solve(dirichlet=bend)
    for field in ("stress", "displacement", "rotation"):
        ref = max(np.abs(afw[field]).max(), 1.0)
        assert np.abs(four[field] - afw[field]).max() < 1e-8 * ref

    # matrix-level identity: the fold reproduces the AFW product exactly
    P, Gvol = space.trace_moment_operator()
    a = LAM / (2 * MU + 2 * LAM)
    folded = space.assemble() - (a / (2 * MU)) * (P.T @ Gvol @ P)
    m_afw = ElasticityInnerProduct(mesh, mu=MU, lam=LAM).assemble()
    scale = np.abs(m_afw.toarray()).max()
    assert np.abs((folded - m_afw).toarray()).max() < 1e-10 * scale


def test_no_stabilization_anywhere():
    """Every cell is unisolvent; the dimension of the stabilization is zero."""
    mesh = structured_quads(2, 2)
    space = DeRhamDeviatoricStress(mesh, mu=MU, lam=LAM)
    for c in range(mesh.num_cells(2)):
        assert space.stabilization_dim(c) == 0
        N, R, Kbar, vol, lc = space.local_matrices(c)
        n = lc.dim * lc.dim * lc.n_facets
        assert N.shape == (n, n)
        assert np.linalg.matrix_rank(N, tol=1e-10) == n
        # the mimetic identity, with R uniquely determined
        assert np.abs(N.T @ R - vol * Kbar).max() < 1e-9 * np.abs(vol * Kbar).max()


def test_full_derham_product_is_afw_on_simplices():
    """The three-field AFW product, assembled as d copies of BDM + trace term."""
    mesh = structured_triangles(2, 2)
    rowfull = DeRhamElasticityInnerProduct(mesh, mu=MU, lam=LAM)
    afw = ElasticityInnerProduct(mesh, mu=MU, lam=LAM)
    diff = (rowfull.assemble() - afw.assemble()).toarray()
    assert np.abs(diff).max() < 1e-10 * np.abs(afw.assemble().toarray()).max()

    bend = lambda x: np.column_stack(
        [0.1 * x[:, 0] ** 2, -0.2 * x[:, 0] * x[:, 1], np.zeros(len(x))]
    )
    ours = MixedElasticity(mesh, inner=rowfull).solve(dirichlet=bend)
    ref = MixedElasticity(
        mesh, inner=ElasticityInnerProduct(mesh, mu=MU, lam=LAM)
    ).solve(dirichlet=bend)
    for field in ("stress", "displacement", "rotation"):
        scale = max(np.abs(ref[field]).max(), 1.0)
        assert np.abs(ours[field] - ref[field]).max() < 1e-8 * scale


def test_full_derham_product_patch_on_quads():
    """Off simplices the full row-wise member still passes the patch test."""
    mesh = structured_quads(3, 2)
    problem = MixedElasticity(
        mesh, inner=DeRhamElasticityInnerProduct(mesh, mu=MU, lam=LAM)
    )
    err_s, err_u = _patch_errors(problem, 2)
    assert err_s < 1e-9
    assert err_u < 1e-9


def test_full_derham_product_patch_on_hexes():
    """Linear displacement patch for the full 3F product, 3D."""
    mesh = structured_box(2, 2, 2)
    problem = MixedElasticity(
        mesh, inner=DeRhamElasticityInnerProduct(mesh, mu=MU, lam=LAM)
    )
    err_s, err_u = _patch_errors(problem, 3)
    assert err_s < 1e-9
    assert err_u < 1e-9


@pytest.mark.parametrize(
    "mesh_fn, d",
    [(lambda: structured_quads(3, 2), 2), (lambda: structured_box(2, 2, 2), 3)],
    ids=["quads", "hexes"],
)
def test_four_field_linear_pressure_patch(mesh_fn, d):
    """Linear displacement patch for the 4F formulation with p_s in P1."""
    mesh = mesh_fn()
    problem = FourFieldElasticity(
        mesh,
        inner=DeRhamDeviatoricStress(mesh, mu=MU, lam=LAM),
        solid_pressure="linear",
    )
    err_s, err_u = _patch_errors(problem, d)
    assert err_s < 1e-9
    assert err_u < 1e-9
