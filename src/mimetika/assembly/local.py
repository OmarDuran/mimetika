r"""Local (single-element) mixed saddle-point problems.

These are the element-level counterparts of the global mixed systems, and they
are the sharpest available test of a mimetic inner product: an inner product
satisfying strong consistency ``M N = R`` reproduces the exact solution of the
local problem whenever that solution lies in the reconstruction space.  This is
the mimetic *patch test*.

Diffusion (mixed Poisson with Dirichlet data)::

    [ M   -a ] [ F  ]   [ -g ]
    [ a^T  0 ] [ p_E] = [ |E| f_mean ]

with ``a_e = |e|`` (so ``a^T F = |E| div_h F``) and ``g_e = \int_e p_D``.
A linear potential gives a constant flux, which the reconstruction space
contains, so ``F`` and ``p_E`` come out exact.

Elasticity (Hellinger--Reissner with weakly imposed symmetry, eq. (2.27))::

    [ M        |E| Dv^T  |E| As^T ] [ sigma ]   [ G ]
    [ |E| Dv    0          0      ] [ u_E   ] = [ |E| f_mean ]
    [ |E| As    0          0      ] [ s     ]   [ 0 ]

with ``Dv`` the discrete divergence, ``As`` the discrete anti-symmetry and
``G`` the boundary-displacement data.  A linear displacement gives a constant
stress, recovered exactly together with ``u_E`` (the element mean displacement)
and ``s`` (the skew part of the displacement gradient).
"""

from __future__ import annotations

import numpy as np

from mimetika.geometry.local_cell import LocalCell
from mimetika.operators.diffusion import DiffusionInnerProduct
from mimetika.operators.elasticity import ElasticityInnerProduct
from mimetika.operators.inner_product import assemble_local_inner_product


def skew_generators(d: int) -> np.ndarray:
    """``(d(d-1)/2, d, d)`` basis of skew matrices; ``S_p : tau = tau_ij - tau_ji``."""
    gens = []
    for i in range(d):
        for j in range(i + 1, d):
            S = np.zeros((d, d))
            S[i, j], S[j, i] = 1.0, -1.0
            gens.append(S)
    return np.array(gens).reshape(-1, d, d)


# -- diffusion ----------------------------------------------------------------


def solve_local_diffusion(
    inner: DiffusionInnerProduct,
    cell_id: int,
    potential,
    source_mean: float = 0.0,
):
    """Solve the local mixed Poisson problem on one cell.

    Parameters
    ----------
    inner
        The flux inner product supplying ``M_E``.
    cell_id
        Index of the cell.
    potential
        Dirichlet data: callable mapping ambient points ``(N,3)`` to ``(N,)``.
    source_mean
        Element mean of the source ``div F``.

    Returns
    -------
    ``(flux_dofs, p_E)`` -- outward normal fluxes per facet and the element
    pressure, in the cell's local (outward) orientation convention.
    """
    N, R, Kbar, vol, lc = inner.local_matrices(cell_id)
    M = assemble_local_inner_product(N, R, Kbar, vol)

    a = lc.facet_measures.astype(float)
    g = np.array(
        [
            qw @ np.asarray(potential(lc.to_ambient(qp))).ravel()
            for qp, qw in lc.facet_quadrature
        ]
    )

    n = len(a)
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = M
    A[:n, n] = -a
    A[n, :n] = a
    rhs = np.concatenate([-g, [vol * source_mean]])

    sol = np.linalg.solve(A, rhs)
    return sol[:n], float(sol[n])


def exact_diffusion_dofs(inner: DiffusionInnerProduct, cell_id: int, grad_p):
    """Exact flux DOFs and element pressure for a linear potential.

    ``grad_p`` is the (constant, ambient) gradient ``b`` of ``p = a + b.x``;
    the flux is ``F = -K grad p``.
    """
    lc = LocalCell.build(inner.mesh.geometry, cell_id)
    Kloc = lc.project_tensor(inner.K)
    grad_local = np.asarray(grad_p, dtype=float) @ lc.frame
    F = -Kloc @ grad_local
    return lc.facet_normals @ F


# -- elasticity ---------------------------------------------------------------


def elasticity_local_operators(inner: ElasticityInnerProduct, cell_id: int):
    """Return ``(M, Dv, As, lc)``: inner product, discrete div and asymmetry.

    ``Dv`` is ``(d, D)`` with ``Dv @ dofs = div_h tau``; ``As`` is
    ``(d(d-1)/2, D)`` with ``As @ dofs = as_h(tau)``.  Both are exact for any
    stress field with constant divergence (Beirao eqs. (2.12), (2.19)).
    """
    N, R, Kbar, vol, lc = inner.local_matrices(cell_id)
    M = assemble_local_inner_product(N, R, Kbar, vol)
    d, ndf = lc.dim, inner.dofs_per_facet(lc.dim)
    D = M.shape[0]

    # div_h: pair the tractions with the constant fields phi_k(xi) = e_k.
    #
    # This is deliberately the same construction as as_h below -- the two differ
    # only in the test space, constants for the divergence and rigid rotations for
    # the asymmetry, which is what they are mathematically.  Writing it as a
    # pairing through the facet basis is what makes this function agnostic to the
    # stress space: nothing here knows how many DOFs a facet carries or how they
    # are ordered.
    #
    # It replaces `Dv[k, i*ndf + k*d + 0] = 1/vol`, which reached into the DOF
    # vector by index and so hardcoded the AFW layout (`ndf = d*d`, constant
    # moment first).  That form overran the facet block for any space with fewer
    # DOFs per facet -- LumpedDeviatoricStress carries `ndf = d` -- and was the
    # only thing tying the assembly to one stress space.  The two agree exactly
    # because the facet P_1 basis has the constant as its first function, so the
    # expansion of `phi_k` is `delta_ck` on basis 0 and zero elsewhere.
    Dv = np.zeros((d, D))
    for i in range(lc.n_facets):
        nq = len(lc.facet_quadrature[i][0])
        phi = np.broadcast_to(np.eye(d), (nq, d, d))  # phi[q, c, k] = delta_ck
        coeff = lc.expand_on_facet(i, phi)  # (nb, d_components, d_tests)
        block = np.einsum("bck->kcb", coeff).reshape(d, ndf)
        Dv[:, i * ndf : (i + 1) * ndf] = block / vol

    # as_h: pair the tractions with the rigid rotations psi_p(xi) = S_p xi.
    gens = skew_generators(d)
    As = np.zeros((len(gens), D))
    for i in range(lc.n_facets):
        qp = lc.facet_quadrature[i][0]
        psi = np.einsum("pkc,qc->qkp", gens, qp)  # (nq, d, nskew)
        coeff = lc.expand_on_facet(i, psi)  # (nb, d, nskew)
        block = np.einsum("bkp->pkb", coeff).reshape(len(gens), ndf)
        As[:, i * ndf : (i + 1) * ndf] = block / vol

    return M, Dv, As, lc


def solve_local_elasticity(
    inner: ElasticityInnerProduct,
    cell_id: int,
    displacement,
    body_force_mean: np.ndarray | None = None,
):
    """Solve the local weakly-symmetric Hellinger--Reissner problem on one cell.

    Parameters
    ----------
    displacement
        Dirichlet data: callable mapping ambient points ``(N,3)`` to ``(N,3)``.
    body_force_mean
        Element mean of ``div sigma`` (ambient); defaults to zero.

    Returns
    -------
    ``(stress_dofs, u_E, s)`` in the cell's local frame; ``u_E`` is the element
    mean displacement and ``s`` the skew part of the displacement gradient.
    """
    M, Dv, As, lc = elasticity_local_operators(inner, cell_id)
    d, vol, D = lc.dim, lc.volume, M.shape[0]
    ndf = inner.dofs_per_facet(d)
    nsk = As.shape[0]

    # Boundary data: expand the displacement on each facet in the P1 basis.
    G = np.zeros(D)
    for i in range(lc.n_facets):
        qp = lc.facet_quadrature[i][0]
        u = np.asarray(displacement(lc.to_ambient(qp)), dtype=float) @ lc.frame
        G[i * ndf : (i + 1) * ndf] = lc.expand_on_facet(i, u).T.ravel()

    f = np.zeros(d) if body_force_mean is None else np.asarray(body_force_mean)
    f = np.atleast_1d(f) @ lc.frame if f.shape[-1] == 3 else f

    n = D + d + nsk
    A = np.zeros((n, n))
    A[:D, :D] = M
    A[:D, D : D + d] = vol * Dv.T
    A[D : D + d, :D] = vol * Dv
    if nsk:
        A[:D, D + d :] = vol * As.T
        A[D + d :, :D] = vol * As
    rhs = np.concatenate([G, vol * f, np.zeros(nsk)])

    sol = np.linalg.solve(A, rhs)
    return sol[:D], sol[D : D + d], sol[D + d :]


def stress_dofs(inner: ElasticityInnerProduct, cell_id: int, sigma_local):
    """DOFs of a constant stress tensor given in the cell's **local** frame."""
    lc = LocalCell.build(inner.mesh.geometry, cell_id)
    d, ndf = lc.dim, inner.dofs_per_facet(lc.dim)
    S = np.asarray(sigma_local, dtype=float)
    dofs = np.zeros(lc.n_facets * ndf)
    for i in range(lc.n_facets):
        qp, qw = lc.facet_quadrature[i]
        B, _ = lc.facet_scalar_basis(i)
        Tn = np.broadcast_to(S @ lc.facet_normals[i], (len(qp), d))
        dofs[i * ndf : (i + 1) * ndf] = np.einsum("q,qb,qk->kb", qw, B, Tn).ravel()
    return dofs


def linear_displacement_reference(
    inner: ElasticityInnerProduct, cell_id: int, a: np.ndarray, B: np.ndarray
):
    """Exact local solution for the ambient linear displacement ``u = a + B x``.

    The elasticity problem on a ``d``-cell is intrinsically ``d``-dimensional,
    so the reference stress is the ``d``-dimensional elastic response to the
    in-hull strain: ``sigma = 2 mu eps + lambda tr(eps) I`` with
    ``eps = sym(Q^T B Q)``.

    Returns ``(stress_dofs, u_E, s)`` matching the output of
    :func:`solve_local_elasticity`.
    """
    lc = LocalCell.build(inner.mesh.geometry, cell_id)
    d = lc.dim
    a, B = np.asarray(a, dtype=float), np.asarray(B, dtype=float)

    B_loc = lc.frame.T @ B @ lc.frame
    eps = 0.5 * (B_loc + B_loc.T)
    sigma = 2 * inner.mu * eps + inner.lam * np.trace(eps) * np.eye(d)

    u_E = (a + B @ lc.origin) @ lc.frame  # element mean = value at the centroid
    skew = 0.5 * (B_loc - B_loc.T)
    s = np.array([0.5 * np.sum(S * skew) for S in skew_generators(d)])
    return stress_dofs(inner, cell_id, sigma), u_E, s
