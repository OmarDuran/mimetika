#!/usr/bin/env python3
r"""Numerical evidence for the inf-sup stability of the mixed-dimensional
linear contact formulation.

WHAT IS MEASURED. Grouped as x = (sigma, t_c) and y = (u, gamma, u_c), the
system is a two-field saddle point

    A = [ A   0  ]     B = [ B   0  ]      K = [ A  B^T ]
        [ 0  A_c ]         [ C   0  ]          [ B   0  ]
                           [ T  J^T ]

and Brezzi asks for two constants, BOTH uniform in h:

  (C1)  inf_y sup_x  (Bx, y) / (|x|_X |y|_Y)  >=  beta  >  0
  (C2)  A(x,x) >= alpha |x|_X^2   for x in ker B

For fixed y the supremum is a dual norm, so

    beta_h^2 = lambda_min( B M_X^-1 B^T , M_Y )

the smallest generalized eigenvalue of the mass-Schur complement against the
multiplier Gram. Equivalently, with M_Y = L L^T, of the SPD matrix
L^-1 B M_X^-1 B^T L^-T -- which is of size dim Y, the SMALL block, and that is
what makes this computable: dim Y solves against M_X and one dense
eigendecomposition. The same spectrum answers the prior question for free: a
zero eigenvalue means B is not surjective, and the dimension of its eigenspace
counts the spurious multiplier modes. A count that grows under refinement is
fatal and needs no rate fitting.

(C2) IS THE ONE A_c = 0 THREATENS. At D_c = 0, A = diag(A, 0) is only semi-definite
and t_c has no mass of its own; it is controlled only through the constraint. On
ker B we have B sigma = 0 and J^T t = -T sigma with J^T injective, so
t = -(J J^T)^-1 J T sigma, and uniform coercivity on the kernel holds exactly
when that map is uniformly bounded:

    C_h = sup  |t|_T / |sigma|_Sigma  ,    alpha_h >= alpha_A / (1 + C_h^2)

which is a LARGEST singular value -- far cheaper than a smallest eigenvalue over
a kernel of dimension dim X - dim Y. Reported unrestricted (over all sigma, not
only the divergence-free ones), which is an upper bound: if it is flat, (C2)
is settled; only if it grows does the restriction have to be put back.

THE NORMS ARE NOT A DETAIL, and there is a trap. The blocks of B carry different
powers of h -- B is a divergence, T is O(1), J is a Gram -- so with |sigma| taken
as plain L^2, beta_h decays from that scaling alone and reads as instability. The
sigma norm must be H(div; Omega \ Gamma):

    |sigma|_X^2 = sigma^T A sigma + (B sigma)^T M_u^-1 (B sigma)

Then |u| and |gamma| are L^2 on cells and |t_c|, |u_c| are L^2 on Gamma, the
latter pair weighted by h^{2s} and h^{-2s}. s IS A PARAMETER OF THE STUDY:

    s_t = 0    L^2(Gamma) for t_c. The right norm while D_c is bounded and
               invertible, since A_c then gives genuine L^2 control.
    s_t = 1/2  the mesh-weighted surrogate for H^-1/2(Gamma), which is what t_c
               becomes as D_c -> 0, where that control vanishes and it is a true
               multiplier for [[u_c]] = 0.

and s_u likewise for u_c, weighted h^{-2 s_u}. THE TWO ARE SEPARATE EXPONENTS
because the norms on t_c (in X) and on u_c (in Y) are not required to be dual to
each other: t_c is paired with the stress facet moments through T, and with u_c
through J, and those two pairings do not have to be measured alike. Which pair
is flat is the answer this study is for -- and a decay under one choice is not a
failure of the formulation, only of that choice.

THE CASES walk at what the construction gives up: there are no unknowns on
fracture intersections, so one, two and three mutually intersecting fractures
is the sequence where a loss of uniformity should appear first.

Run it:

    PYTHONNOUSERSITE=1 python contact_inf_sup.py

which is the whole study -- 3D tetrahedra, one two and three mutually
intersecting fractures, both stabilized families, n = 2, 4, 6, 8 -- and writes
contact_inf_sup.png beside contact_inf_sup.json. Around ten minutes; dim Y
reaches 25344 at the top level, which only the iterative eigensolver reaches.

FOUR LEVELS RATHER THAN THREE because three cannot tell a converging constant
from a power law. A power law holds the per-step exponent FIXED; a constant
still settling has it shrink. With three points there are two exponents and the
distinction is a judgement call -- the 3-fracture beta read as "undecided" at
three levels and converges to 0.77 once the fourth pins it.

    PYTHONNOUSERSITE=1 python contact_inf_sup.py --refinements 2      # a quick look
    PYTHONNOUSERSITE=1 python contact_inf_sup.py --mesh-type cartesian --levels 2,4,6,8,10
    PYTHONNOUSERSITE=1 python contact_inf_sup.py --c-n 1e2 --c-t 5   # a compliant interface
    PYTHONNOUSERSITE=1 python contact_inf_sup.py --nu 0.49           # near-incompressible
    PYTHONNOUSERSITE=1 python contact_inf_sup.py --s 0:0,0.5:0.5,0.5:0   # the norm study

THE LADDER REACHES a factor of 4 in h on tetrahedra and 5 on hexahedra, which
are cheaper -- 512 cells at n = 8 against 3072. That separates a uniform
constant from an h^(1/2) loss, which would be a factor of 2 across it, and is
still not enough to claim an exponent of exactly zero.

The json is the expensive part, and the figure is redrawn from it for free:

    PYTHONNOUSERSITE=1 python contact_inf_sup.py --from contact_inf_sup.json --plot new.png

"""

import argparse
import json
import warnings
from pathlib import Path

import _bootstrap  # noqa: F401  -- BEFORE mimetika_cxx; see python/_shadowing.py
import mimetika_cxx as mk

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

MU = 1.0

# THREE DIMENSIONS ONLY. The wrench family exists nowhere else, and only in
# space do three fractures meet in a line and a triple point -- which is the
# configuration this formulation carries no unknowns on, and so the one worth
# testing.
DIM = 3

# x: the fields carrying a mass term. y: the multipliers, whose diagonal block
# is zero -- asserted below rather than assumed.
X_FIELDS = ("s_0", "t_1")
Y_FIELDS = ("u_0", "g_0", "uc_1")

PRODUCTS = {
    "derham_bdm": (mk.StressRealization.derham_bdm, mk.StressFormulation.weak_symmetry),
    "stabilized_bdm": (mk.StressRealization.stabilized_bdm, mk.StressFormulation.weak_symmetry),
    "stabilized_vem": (mk.StressRealization.stabilized_vem, mk.StressFormulation.strong_symmetry),
}


# ---- the cases ---------------------------------------------------------------
#
# Axis-aligned planes at the midpoint, so they stay node-conforming at every
# level of an even ladder and the geometry does not move with h. In 2D the third
# plane would have to be a diagonal, which is conforming only if the
# triangulation happens to carry those edges; the case is skipped when the node
# set induces no facets rather than silently solving something else.
def planes(count):
    """(axis, offset) for each fracture: one plane per axis at the midpoint,
    then parallel families at the quarters.

    MORE FRACTURES THAN AXES means parallel planes, which is the only way to
    push the network past three mutually intersecting ones and ask the question the
    h-ladder cannot: does beta survive COMPLEXITY, not just refinement. The
    quarter planes need n divisible by 4 to stay node-conforming; a set that
    induces no facets is reported rather than silently solved.
    """
    offsets = [0.5, 0.25, 0.75]
    if count > DIM * len(offsets):
        raise SystemExit(f"{count} fractures exceeds the {DIM * len(offsets)} this family of "
                         "axis-aligned planes offers")
    return [(i % DIM, offsets[i // DIM]) for i in range(count)]


CELLS = {"simplex": "tetrahedra", "cartesian": "hexahedra"}


def case(dim, n, count, mesh_type="simplex"):
    mesh = mk.box([n] * 3, dim, getattr(mk.Family, mesh_type))
    pts = np.array([mesh.point(v) for v in range(mesh.count(0))])
    sets = []
    for ax, off in planes(count):
        sets.append([int(v) for v in np.flatnonzero(np.abs(pts[:, ax] - off) < 1e-12)])
    return mesh, sets


# ---- assembly and the block split -------------------------------------------
def assemble(mesh, dim, sets, product, c_n, c_t, lame=None):
    how, form = PRODUCTS[product]
    model = mk.CauchyMechanicsModel(
        mesh, dim, mk.ElasticMaterial(MU, args_lame if lame is None else lame),
        how, form)
    model.fracture(sets, list(range(1, len(sets) + 1)))
    if c_n > 0.0:
        model.linear_contact(c_n, c_t if c_t > 0.0 else c_n)
    else:
        model.bond()

    # A FIXED, WELL-POSED CONFIGURATION across the whole ladder: confined
    # compression, traction on the far face and rollers elsewhere. Some essential
    # data is needed or rigid motions sit in ker B^T and every spectrum below
    # carries their zeros; this set is the one the solver is known to be
    # nonsingular on.
    loaded, confined = [], []
    for f in model.outer_boundary_facets():
        x = mk.centroid(mesh, dim - 1, f)
        (loaded if abs(x[dim - 1] - 1.0) < 1e-9 else confined).append(f)
    applied = [0.0] * 9
    applied[(dim - 1) * 3 + (dim - 1)] = -1.0
    model.add_traction(loaded, applied)
    model.add_free_slip(confined)
    model.build(assemble_jacobian=True)

    row, col, val, n_dofs = mk.system_triplets(model)
    K = sp.csr_matrix((val, (row, col)), shape=(n_dofs, n_dofs))
    blocks = {b["name"]: [tuple(r) for r in b["ranges"]] for b in mk.field_blocks(model)}
    pinned = np.zeros(n_dofs, bool)
    pinned[mk.constrained_dofs(model)] = True
    return model, K, blocks, pinned


def indices(blocks, names, pinned):
    """The free dofs of these fields, and where each field landed inside them."""
    take, where, at = [], {}, 0
    for name in names:
        if name not in blocks:
            continue
        got = []
        for begin, end in blocks[name]:
            idx = np.arange(begin, end)
            got.append(idx[~pinned[idx]])
        got = np.concatenate(got) if got else np.empty(0, int)
        where[name] = (at, at + len(got))
        at += len(got)
        take.append(got)
    return (np.concatenate(take) if take else np.empty(0, int)), where


# ---- the Gram matrices -------------------------------------------------------
def cell_weights(mesh, dim, blocks, name, pinned):
    r"""L^2 on cells for a field whose dof is the CELL INTEGRAL.

    u_E = int_E u, so |u|^2_{L2} = sum |E| ubar^2 = sum u_E^2 / |E|: the Gram is
    the INVERSE measure, not the measure. The same convention is what makes
    |div sigma|^2 = (B sigma)^T M_u^-1 (B sigma) with B the row the model
    assembles, which is the divergence already divided by |E|.
    """
    begin, end = blocks[name][0]
    per = (end - begin) // mesh.count(dim)
    w = np.empty(end - begin)
    for e in range(mesh.count(dim)):
        w[e * per : (e + 1) * per] = 1.0 / mk.measure(mesh, dim, e)
    return w[~pinned[np.arange(begin, end)]]


def face_grams(K, blocks, n_faces):
    r"""G on each fracture face, read off the assembled J rather than rebuilt.

    The jump term writes (t, u_c^+) = -G and (t, u_c^-) = +G on that face's own
    rows and columns, so the first half of each face's J block IS -G. Taking it
    from the matrix keeps the norm and the operator consistent by construction:
    no second implementation of the chart to drift from the first.

    PER FACE, not once. G = int_f psi_i . psi_j carries the face's measure, and
    the faces of an unstructured fracture differ by orders of magnitude -- one
    G for all of them would be a different norm on every face but the one it
    came from.
    """
    tb, te = blocks["t_1"][0]
    ub, _ = blocks["uc_1"][0]
    kf = (te - tb) // n_faces
    out = []
    for f in range(n_faces):
        r0 = tb + f * kf
        c0 = ub + f * 2 * kf
        G = -K[r0 : r0 + kf, c0 : c0 + kf].toarray()
        out.append(0.5 * (G + G.T))
    return out


def norms(mesh, dim, K, blocks, pinned, ix, iy, ixw, iyw, grams, s, h, a_sigma=None):
    """M_X and M_Y on the free dofs, as block diagonals in field order."""
    A = K[np.ix_(ix, ix)]
    B = K[np.ix_(iy, ix)]

    # ---- M_X: H(div; Omega \ Gamma) for sigma, h^{2s} L^2(Gamma) for t_c
    sb, se = ixw["s_0"]
    mu_w = cell_weights(mesh, dim, blocks, "u_0", pinned)
    ub, ue = iyw["u_0"]
    Bu = B[ub:ue, sb:se]
    # THE STRESS NORM, AND WHERE lambda ENTERS. A is the COMPLIANCE, so as
    # lambda grows it stops seeing the volumetric part of sigma: a near-
    # volumetric divergence-free sigma then has a vanishing |sigma|_Sigma while
    # its traction on Gamma does not vanish, and C_h inflates. `a_sigma` swaps
    # in the plain L^2 mass instead, which carries no lambda at all -- the test
    # of whether the lambda-dependence is the formulation's or the norm's.
    A_ss = A[sb:se, sb:se] if a_sigma is None else a_sigma
    M_sigma = (A_ss + Bu.T @ sp.diags(1.0 / mu_w) @ Bu).tocsc()
    parts = [0.5 * (M_sigma + M_sigma.T)]
    if "t_1" in ixw:
        # the t_c rows of A are A_c, which is ZERO at D_c = 0: the norm has to
        # come from the chart and not from the form, or M_X is singular exactly
        # in the case the study is about
        parts.append(sp.block_diag([G * h ** (2.0 * s[0]) for G in grams]).tocsc())
    M_X = sp.block_diag(parts).tocsc()

    # ---- M_Y: L^2 on cells for u and gamma, h^{-2s} L^2(Gamma) for u_c
    # M_Y AND ITS INVERSE FACTOR, side by side. M_Y is block diagonal -- diagonal
    # on the cell fields, one small dense block per fracture face -- so L with
    # M_Y = L L^T is block diagonal too and L^-1 is assembled from k x k inverses
    # rather than from a dense Cholesky of the whole thing. At n = 8 that
    # factorization alone would have been 5 GB.
    ypart, yinv = [], []
    for name in ("u_0", "g_0"):
        if name in iyw:
            w = cell_weights(mesh, dim, blocks, name, pinned)
            ypart.append(sp.diags(w))
            yinv.append(sp.diags(1.0 / np.sqrt(w)))
    if "uc_1" in iyw:
        # both sides against the same chart: the jump is their difference, so
        # they have to be measured in one norm
        w = h ** (-2.0 * s[1])
        faces, inv = [], []
        for G in grams:
            Gw = G * w
            faces.append(sp.block_diag([Gw, Gw]))
            Li = sla.solve_triangular(sla.cholesky(Gw, lower=True),
                                      np.eye(Gw.shape[0]), lower=True)
            inv.append(sp.block_diag([Li, Li]))
        ypart.append(sp.block_diag(faces))
        yinv.append(sp.block_diag(inv))
    M_Y = sp.block_diag(ypart).tocsc()
    L_inv = sp.block_diag(yinv).tocsc()
    if M_X.shape[0] != len(ix) or M_Y.shape[0] != len(iy):
        raise SystemExit(f"Gram sizes {M_X.shape[0]}, {M_Y.shape[0]} against "
                         f"dim X {len(ix)}, dim Y {len(iy)}")
    return M_X, M_Y, A, B, L_inv


# ---- the two constants -------------------------------------------------------
def inf_sup(M_X, B, L_inv, tol=1e-10, chunk=512):
    r"""beta_h^2 = lambda_min(B M_X^-1 B^T, M_Y), and the kernel count.

    WITH L^-1 FOLDED INTO B. Since M_Y = L L^T, the pencil's eigenvalues are
    those of L^-1 B M_X^-1 B^T L^-T, and L^-1 B is a SPARSE product -- so the
    whole thing is Bh M_X^-1 Bh^T with Bh = L^-1 B, and neither M_Y nor the
    unweighted S is ever formed densely.

    AND BUILT IN COLUMN CHUNKS. The right-hand sides are the columns of Bh^T,
    dim X by dim Y; densifying them all at once was 2.5 GB at n = 6 and would
    have been 12.4 GB at n = 8. A few hundred columns at a time costs nothing
    and bounds the peak by dim X * chunk.

    What remains dense is Sbar itself, dim Y squared, which the eigensolver
    needs whole.
    """
    Bh = (L_inv @ B).tocsc()
    Bt = Bh.T.tocsc()
    lu = spla.splu(M_X.tocsc())
    n_y = Bh.shape[0]
    Sbar = np.empty((n_y, n_y))
    for j0 in range(0, n_y, chunk):
        j1 = min(j0 + chunk, n_y)
        Sbar[:, j0:j1] = Bh @ lu.solve(np.asarray(Bt[:, j0:j1].todense()))
    Sbar = 0.5 * (Sbar + Sbar.T)
    ev = sla.eigvalsh(Sbar)
    top = max(ev[-1], 1e-300)
    kernel = int(np.sum(ev < tol * top))
    positive = ev[ev >= tol * top]
    return (np.sqrt(positive[0]) if positive.size else 0.0), kernel, np.sqrt(top)


def inf_sup_lobpcg(M_X, B, L_inv, block=8, tol=1e-10, maxiter=800, seed=0):
    r"""The same beta_h, without ever forming Sbar.

    Sbar = Bh M_X^-1 Bh^T with Bh = L^-1 B is only ever APPLIED: one sparse
    solve with M_X per column of the block, so the memory is dim X by `block`
    rather than dim Y squared. At tetrahedral n = 8 that is the difference
    between 5.1 GB and a few megabytes.

    WHY THE SMALLEST EIGENVALUE IS REACHABLE HERE. Krylov methods converge at
    the outside of a spectrum, and lambda_min of an SPD matrix is the hard end
    -- which is why C_h, a LARGEST singular value, went sparse without trouble
    and this did not. The saving grace is that L^-1 has already whitened M_Y
    out, so the spectrum of Sbar is exactly

        [beta_h^2, beta_max^2] = beta_h^2 [1, kappa_B^2]

    and kappa_B = beta_max/beta_h is the column this study already reports. It
    runs 11 to 17, so the effective condition number is 120 to 290 and LOBPCG
    converges in O(100) iterations with no preconditioner at all. If kappa_B
    ever grew, the cost would rise visibly in a number already on the table.

    lambda_max comes from ARPACK, which is the easy direction.
    """
    Bh = (L_inv @ B).tocsc()
    Bht = Bh.T.tocsc()
    lu = spla.splu(M_X.tocsc())
    n_y = Bh.shape[0]

    def apply(X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        return Bh @ lu.solve(np.asarray(Bht @ X))

    op = spla.LinearOperator((n_y, n_y), matvec=lambda v: apply(v).ravel(),
                             matmat=apply, dtype=float)
    top = float(spla.eigsh(op, k=1, which="LA", tol=1e-8,
                           return_eigenvectors=False)[0])

    k = min(block, max(2, n_y // 6))
    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((n_y, k))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev = spla.lobpcg(op, X0, largest=False, tol=tol * max(top, 1.0),
                         maxiter=maxiter)[0]
    ev = np.sort(np.asarray(ev, dtype=float))
    kernel = int(np.sum(ev < tol * top))
    if kernel == ev.size:
        raise SystemExit(f"every one of the {ev.size} eigenvalues found is at zero; "
                         "the kernel is larger than the block, so raise --block")
    positive = ev[ev >= tol * top]
    return float(np.sqrt(positive[0])), kernel, float(np.sqrt(max(top, 0.0)))


def kernel_coercivity(M_X, B, ixw, iyw, grams, s, h):
    r"""C_h = sup_sigma |P sigma|_T / |sigma|_Sigma  with  P = (J J^T)^-1 J T.

    On ker B we have J^T t = -T sigma with J^T injective, so t = -P sigma and
    the kernel is a graph over sigma. Then alpha_h >= 1/(1 + C_h^2), which is the
    whole of (C2) -- see the header.

    THE SUPREMUM IS TAKEN UNRESTRICTED, over every sigma rather than only the
    divergence-free ones (C2) asks about. That is an UPPER bound, so a bounded
    C_h settles the condition and only a growing one would need the restriction
    put back.

    EVERYTHING HERE IS SPARSE, and it has to be. J is block diagonal face by
    face, so J J^T is too and its inverse is assembled block by block; P is then
    sparse, and

        C_h^2 = lambda_max( P^T M_t P ,  M_sigma )

    is a largest generalized eigenvalue -- ARPACK's easy direction. Done densely
    instead, this was 69% of the study's runtime: it factored a dense copy of
    the sparse M_sigma and formed an explicit inverse, 4.8 GB twice over and
    O(dim sigma^3) at the top level. The sparse form agrees to 1e-14 and is 112x
    faster at n = 4, the gap growing as the cube.
    """
    if "t_1" not in ixw:
        return float("nan")
    tb, te = ixw["t_1"]
    sb, se = ixw["s_0"]
    ub, ue = iyw["uc_1"]
    T = B[ub:ue, sb:se].tocsc()      # the u_c rows against sigma
    JT = B[ub:ue, tb:te].tocsc()     # ... and against t, which is J^T
    J = JT.T.tocsc()

    JJT = (J @ JT).tocsc()
    kf = (te - tb) // len(grams)
    inverses = [np.linalg.inv(JJT[i * kf:(i + 1) * kf, i * kf:(i + 1) * kf].toarray())
                for i in range(len(grams))]
    # the sign of P does not survive a norm, so it is dropped
    P = (sp.block_diag(inverses).tocsc() @ (J @ T)).tocsc()

    M_t = sp.block_diag([G * h ** (2.0 * s[0]) for G in grams]).tocsc()
    M_sigma = M_X[sb:se, sb:se].tocsc()
    lu = spla.splu(M_sigma)
    Minv = spla.LinearOperator(M_sigma.shape, matvec=lu.solve, dtype=float)
    lam = spla.eigsh((P.T @ M_t @ P).tocsc(), k=1, M=M_sigma, Minv=Minv,
                     which="LA", tol=1e-10)[0][0]
    return float(np.sqrt(max(lam, 0.0)))


# ---- the study ---------------------------------------------------------------
def rate(hs, vals):
    """Least-squares exponent of val ~ h^p; p ~ 0 is uniform."""
    good = [(h, v) for h, v in zip(hs, vals) if v > 0]
    if len(good) < 2:
        return float("nan")
    h = np.log([g[0] for g in good])
    v = np.log([g[1] for g in good])
    return float(np.polyfit(h, v, 1)[0])


def measure(dim, n, count, product, c_n, c_t, exps, c_exps, max_dofs,
            mesh_type="simplex", sigma_norm="compliance",
            eigensolver="dense", block=8):
    """One mesh: the constants, the kernel count, and the sizes."""
    mesh, sets = case(dim, n, count, mesh_type)
    model, K, blocks, pinned = assemble(mesh, dim, sets, product, c_n, c_t)
    if model.n_fracture_faces == 0:
        return None, "no facets induced -- these planes are not conforming at this n"
    ix, ixw = indices(blocks, X_FIELDS, pinned)
    iy, iyw = indices(blocks, Y_FIELDS, pinned)
    if len(iy) > max_dofs:
        return None, f"dim Y = {len(iy)} over --max-dofs"

    # THE GROUPING, CHECKED: the multiplier block must be exactly zero, or x and
    # y have been split wrongly and every number here is about another problem.
    yy = K[np.ix_(iy, iy)]
    if yy.nnz and np.abs(yy.data).max() > 1e-12 * np.abs(K.data).max():
        raise SystemExit(f"the (y,y) block is not zero: {np.abs(yy.data).max():.3e}")

    h = 1.0 / n
    # 2 mu A(lambda = 0) IS the L^2 mass: at lambda = 0 the compliance is
    # I/(2 mu). Same mesh, same space, same numbering, so the index sets above
    # apply unchanged -- only the material differs.
    a_sigma = None
    if sigma_norm == "l2":
        sb, se = ixw["s_0"]
        _, K0, _, _ = assemble(mesh, dim, sets, product, c_n, c_t, lame=0.0)
        a_sigma = (2.0 * MU * K0[np.ix_(ix, ix)][sb:se, sb:se]).tocsc()
    grams = face_grams(K, blocks, model.n_fracture_faces)
    betas, tops, ker, mx = [], [], None, {}
    for e in exps:
        M_X, _, _, B, L_inv = norms(mesh, dim, K, blocks, pinned, ix, iy, ixw, iyw,
                                    grams, e, h, a_sigma)
        b, ker, top = (inf_sup_lobpcg(M_X, B, L_inv, block) if eigensolver == "lobpcg"
                       else inf_sup(M_X, B, L_inv))
        betas.append(b)
        tops.append(top)
        mx[e[0]] = (M_X, B)
    chs = [kernel_coercivity(mx[st][0], mx[st][1], ixw, iyw, grams, (st, st), h)
           for st in c_exps]
    # kappa_B = beta_max / beta_h: how the constraint operator is conditioned IN
    # THE STUDY'S OWN NORMS, from the same spectrum beta_h came out of, so it
    # costs nothing and exists at every level.
    #
    # It replaces cond(K), which was densifying the whole saddle point and so
    # was gated at 4000 dofs -- above that the column printed a bare nan that
    # read as a failure. It was also the wrong quantity: cond(K) moves with the
    # arbitrary relative scaling of the blocks, and this does not.
    kappa = tops[0] / betas[0] if betas[0] > 0 else float("inf")
    return dict(h=h, n=n, dim_x=len(ix), dim_y=len(iy), ker=ker,
                betas=betas, chs=chs, kappa=kappa, faces=model.n_fracture_faces), None


def draw(path, collected, dim, law, cells, nu, products):
    r"""The two Brezzi constants against h, side by side.

    THE KERNEL COUNT IS NOT A CURVE. dim ker B^T is 0 or it is not: 0 means B is
    surjective and a constant exists, anything else means spurious multiplier
    modes and there is no constant to plot. Summed over the ladder it is one
    number per curve, so it rides in the legend.

    THE TWO PANELS FAIL IN OPPOSITE DIRECTIONS. beta_h fails by DECAYING --
    h^(1/2) is the classic loss -- and C_h fails by GROWING, since
    alpha_h >= 1/(1 + C_h^2). Both are log-log against h, where a uniform
    constant reads as a flat line and a decaying one as a slope.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colour = {1: "#1f77b4", 2: "#d62728", 3: "#2ca02c"}
    style = {p: st for p, st in zip(products, ["-o", "--s", ":^"])}
    fig, (bx, cx) = plt.subplots(1, 2, figsize=(12.0, 7.2))

    for (prod, count), rows in sorted(collected.items()):
        hs = [r["h"] for r in rows]
        ker = sum(r["ker"] for r in rows)
        # the expectation is stated, so a nonzero one reads as wrong rather than
        # as just another number
        flag = "0" if ker == 0 else f"{ker} — EXPECTED 0"
        # SHORT ENOUGH TO TILE. Each entry also states its kernel sum, so the
        # case is a count rather than a description: n_f fractures, and the
        # product by its family alone -- both are stabilized, which the title
        # can say and the curve need not.
        name = rf"$n_f$ = {count}"
        if len(products) > 1:
            name += ", " + prod.replace("stabilized_", "").replace("derham_", "dR-")
        label = rf"{name}:  $\sum_h\dim\ker\mathfrak{{B}}_h^T$ = {flag}"
        st_ = style.get(prod, "-o")
        bx.plot(hs, [r["betas"][0] for r in rows], st_, color=colour.get(count, "0.3"),
                label=label)
        cx.plot(hs, [r["chs"][0] for r in rows], st_, color=colour.get(count, "0.3"))

    # THE DEFINITIONS, on the axis each belongs to.
    #
    # C_h NAMES t_c AND THEN SAYS WHAT IT IS, rather than substituting it away.
    # On ker B the row T sigma + J^T t_c = 0 with J^T injective DETERMINES t_c
    # from sigma, so the kernel is a graph over sigma and not a product: there
    # is no direction (0, t_c) in it. That is the whole reason A = diag(A, A_c)
    # can be coercive there while A_c = 0 -- the form is blind to t_c, and on
    # the kernel t_c has no independent existence to be blind to. Writing the
    # map out puts that on the axis instead of hiding it in a substitution.
    #
    # The supremum runs over every sigma, which is what is computed: an upper
    # bound for the same supremum over ker B, which also asks B sigma = 0.
    beta_def = (r"$\beta_h = \inf_y \sup_x \, \frac{(\mathfrak{B}x,\, y)}"
                r"{\|x\|_X \, \|y\|_Y}$")
    ch_def = (r"$C_h = \sup_\sigma \, \frac{\|t_c(\sigma)\|_T}{\|\sigma\|_\Sigma}$"
              "\n"
              r"$t_c(\sigma) = (JJ^T)^{-1} J T \sigma$")
    for ax, ylab, title in ((bx, beta_def, r"inf-sup constant"),
                            (cx, ch_def, r"coercivity on $\ker\mathfrak{B}$:  "
                                         r"$\alpha_h \geq 1/(1 + C_h^2)$")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("$h$")
        ax.set_ylabel(ylab, fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", alpha=0.25)

    # THE BLOCK STRUCTURE, so the figure says what was measured. matplotlib's
    # mathtext has no bmatrix -- \genfrac is the one bracketed stack it parses,
    # and its rule size must be a bare 0 rather than 0pt.
    blocks = (r"$\mathfrak{A} = \genfrac{[}{]}{0}{0}{A\ \ \ \ 0\ }{0\ \ A_c}$"
              r"$\qquad$"
              r"$\mathfrak{B} = \genfrac{[}{]}{0}{0}{B\ \ \ 0\ }{T\ \ J^T}$"
              r"$\qquad$"
              r"$\genfrac{[}{]}{0}{0}{\mathfrak{A}\ \ \mathfrak{B}^T}"
              r"{\mathfrak{B}\ \ \ 0\ }"
              r"\genfrac{[}{]}{0}{0}{x}{y} = \genfrac{[}{]}{0}{0}{f}{g}$")
    # WHAT x AND y ARE MADE OF. Without this the block row names x and y and the
    # axes then use sigma and t_c, which appear nowhere -- and it is the same
    # line that explains why dim Y differs between the families: the wrench
    # carries no rotation multiplier.
    parts = (r"$x = (\sigma,\; t_c)$" r"$\qquad$"
             r"$y = (u,\; \gamma,\; u_c)$, $\ \gamma$ only where the symmetry is weak")
    fig.suptitle(rf"{dim}D {cells}, $\nu$ = {nu:g}, {law}", fontsize=11, y=0.985)
    fig.text(0.5, 0.930, blocks, ha="center", va="top", fontsize=11)
    fig.text(0.5, 0.838, parts, ha="center", va="top", fontsize=10)
    handles, labels = bx.get_legend_handles_labels()
    # WIDE RATHER THAN TALL: three columns turns seven entries into three rows,
    # which is what buys back the vertical space the panels want.
    fig.legend(handles, labels, fontsize=8.5, loc="lower center", ncol=3,
               frameon=False, columnspacing=2.0, handlelength=2.4)
    fig.tight_layout(rect=(0, 0.13, 1, 0.800))
    fig.savefig(path, dpi=150)
    return path


def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # ONE PATH FOR EVERY SIZE. lobpcg is marginally slower than dense wherever
    # dense fits -- 3.3 s against 2.9 s at dim Y = 3240 -- and that is a price
    # worth paying: a threshold that switched between them would mean two runs
    # of one study came from different code with nothing in the output saying
    # which. It agrees with dense to 1e-14 where both are computable, and it is
    # the only path past dim Y ~ 12000.
    ap.add_argument("--eigensolver", default="lobpcg", choices=("dense", "lobpcg"),
                    help="lobpcg (the default) only APPLIES Sbar, so it costs dim X by --block "
                         "rather than dim Y squared; dense forms the whole spectrum, which is "
                         "exact and is what lobpcg is validated against")
    ap.add_argument("--block", type=int, default=8,
                    help="lobpcg block size; it must exceed the kernel dimension")
    ap.add_argument("--sigma-norm", default="compliance", choices=("compliance", "l2"),
                    help="the stress norm: the compliance energy sigma^T A sigma, which carries "
                         "lambda, or the plain L^2 mass, which does not. Both add |div sigma|^2")
    ap.add_argument("--mesh-type", default="simplex", choices=("simplex", "cartesian"),
                    help="simplex for tetrahedra, cartesian for hexahedra")
    ap.add_argument("--levels", default="2,4,6,8",
                    help="the mesh ladder as cells per side, comma separated. Each entry is one "
                         "refinement; h = 1/n")
    ap.add_argument("--refinements", type=int, metavar="N",
                    help="use only the first N of --levels. The cheap way to cut a run short: "
                         "the cost is dominated by the finest level")
    ap.add_argument("--fractures", default="1,2,3", help="how many intersecting fractures")
    ap.add_argument("--product", default="stabilized_bdm,stabilized_vem",
                    help="one product, or several comma separated onto one figure. The default "
                         "is the two stabilized families, one per symmetry. Available: "
                         + ", ".join(sorted(PRODUCTS)))
    ap.add_argument("--plot", metavar="PATH",
                    help="the figure: beta_h and C_h against h, one curve per case. This is the "
                         "output; the table is how to read it. Defaults to a name describing the "
                         "run, so configurations do not overwrite each other")
    # A LADDER COSTS AN HOUR at the top levels -- a dense eigendecomposition of
    # dim Y per mesh -- so the numbers are kept and the picture can be redrawn,
    # recoloured or combined across runs without paying for them twice.
    ap.add_argument("--save", metavar="PATH",
                    help="write the measurements as json, so the ladder is paid for once. "
                         "Defaults beside --plot")
    ap.add_argument("--from", dest="read", metavar="PATH",
                    help="draw from saved json instead of measuring; several comma separated "
                         "are merged, which is how two products end up on one figure")
    ap.add_argument("--c-n", type=float, default=0.0,
                    help="normal stiffness. 0, the default, is D_c = 0")
    ap.add_argument("--c-t", type=float, default=0.0, help="defaults to --c-n")
    ap.add_argument("--nu", type=float, default=0.25, help="Poisson ratio, through lambda")
    ap.add_argument("--s", default="0.5:0",
                    help="norm exponents as s_t:s_u, comma separated. s_t weights the t_c "
                         "Gram by h^(2 s_t) and s_u weights u_c by h^(-2 s_u)")
    # 12000 WAS THE DENSE LIMIT -- Sbar is dim Y squared and at 12000 that is
    # already 1 GB. lobpcg never forms it, so the guard is now about time and
    # the sparse factorization of M_X rather than about memory, and 30000
    # admits n = 8 on tetrahedra, where dim Y reaches 25344.
    ap.add_argument("--max-dofs", type=int, default=30000,
                    help="skip a level whose multiplier block exceeds this. A guard on cost, "
                         "not correctness; lower it to cut a run short, and note that "
                         "--eigensolver dense needs dim Y^2 x 8 bytes on top")
    args = ap.parse_args()

    global args_lame
    args_lame = 2.0 * MU * args.nu / (1.0 - 2.0 * args.nu)
    levels = [int(v) for v in args.levels.split(",")]
    if args.refinements is not None:
        if args.refinements < 1:
            raise SystemExit("--refinements needs at least one level")
        levels = levels[:args.refinements]
    counts = [int(v) for v in args.fractures.split(",")]
    exps = []
    for tok in args.s.split(","):
        st, _, su = tok.partition(":")
        exps.append((float(st), float(su or st)))
    c_exps = sorted({e[0] for e in exps})
    products = [v.strip() for v in args.product.split(",") if v.strip()]
    for prod in products:
        if prod not in PRODUCTS:
            raise SystemExit(f"unknown product {prod}; one of {', '.join(sorted(PRODUCTS))}")

    # THE COMPLIANCE, WRITTEN OUT. D_c = 0 for the rigid interface, and the
    # tensor with its numbers substituted otherwise -- a reader should not have
    # to know what a word means to know what was solved.
    # ONE SHAPE FOR BOTH ENDS, so the rigid case reads as a member of the family
    # rather than as a different statement: the coefficients go to zero, the
    # tensor does not go away.
    if args.c_n <= 0:
        normal = tangential = "(0)"
        spoken_n = spoken_t = "(0)"
        tag = "Dc0"
    else:
        cn, ct = args.c_n, (args.c_t or args.c_n)
        normal = rf"\left(\frac{{1}}{{{cn:g}}}\right)"
        tangential = rf"\left(\frac{{1}}{{{ct:g}}}\right)"
        spoken_n, spoken_t = f"(1/{cn:g})", f"(1/{ct:g})"
        tag = f"cn{cn:g}_cT{ct:g}"
    law = (rf"$\mathbb{{D}}_c = {normal}\, n \otimes n"
           rf" + {tangential}\, (I - n \otimes n)$")
    spoken = f"D_c = {spoken_n} n(x)n + {spoken_t} (I - n(x)n)"
    stem = f"contact_inf_sup_{args.mesh_type}_nu{args.nu:g}_{tag}"
    if args.sigma_norm != "compliance":
        stem += f"_{args.sigma_norm}"
        law += r",  $\|\sigma\|_\Sigma$ in $L^2$"
        spoken += ",  sigma-norm = L2"
    cells, nu = CELLS[args.mesh_type], args.nu
    print(f"{DIM}D {cells}, {', '.join(products)}, nu = {args.nu:g} "
          f"(lambda = {args_lame:g}), {spoken}")
    print()
    print("  x = (sigma, t_c)      the unknowns carrying a mass term;  dim X counts them")
    print("  y = (u, gamma, u_c)   the multipliers, whose diagonal block is zero;  dim Y counts")
    print("                        them.  gamma is absent where the symmetry is strong")
    print()
    print("  beta_h    the inf-sup constant.  Uniform in h is what stability asks for")
    print("  ker       dim ker B^T.  Anything but 0 is spurious multiplier modes")
    print("  C_h       the (C2) bound, with alpha_h >= 1/(1 + C_h^2).  Uniform, likewise")
    print("  kappa_B   beta_max/beta_h, the constraint's conditioning in these norms")
    print("  norms     " + ", ".join(f"s_t = {st:g}, s_u = {su:g}" for st, su in exps))
    print()

    collected = {}
    if args.read:
        for path in args.read.split(","):
            with open(path.strip()) as fh:
                saved = json.load(fh)
            for key, rows in saved["rows"].items():
                prod, count = key.rsplit("/", 1)
                collected[(prod, int(count))] = rows
            products = sorted({k[0] for k in collected})
            law = saved.get("law", law)
            cells = saved.get("cells", cells)
            nu = saved.get("nu", nu)
            stem = Path(path.strip()).with_suffix("").name
            # the norms are a property of the MEASUREMENT, not of this invocation
            exps = [tuple(e) for e in saved.get("exps", exps)]
            c_exps = saved.get("c_exps", c_exps)
        print(f"read {len(collected)} curve(s) from {args.read}")
    for prod in ([] if args.read else products):
        for count in counts:
            head = f"{'n':>4s}{'h':>8s}{'dim X':>8s}{'dim Y':>8s}{'ker':>5s}"
            for st, su in exps:
                head += f"{'beta ' + f'{st:g}:{su:g}':>14s}"
            for st in c_exps:
                head += f"{'C_h ' + f'{st:g}':>12s}"
            label = f"{count} fracture(s)" + (f", {prod}" if len(products) > 1 else "")
            print(f"-- {label} " + "-" * max(4, 34 - len(label)))
            print(head + f"{'kappa_B':>11s}")
            rows = []
            for n in levels:
                got, why = measure(DIM, n, count, prod, args.c_n, args.c_t,
                                   exps, c_exps, args.max_dofs, args.mesh_type,
                                   args.sigma_norm, args.eigensolver, args.block)
                if got is None:
                    print(f"{n:>4d}   {why}")
                    continue
                line = f"{got['n']:>4d}{got['h']:>8.4f}{got['dim_x']:>8d}{got['dim_y']:>8d}"
                line += f"{got['ker']:>5d}"
                for b in got["betas"]:
                    line += f"{b:>14.4e}"
                for c in got["chs"]:
                    line += f"{c:>12.4e}"
                print(line + f"{got['kappa']:>11.2f}")
                rows.append(got)
            if len(rows) >= 2:
                hs = [r["h"] for r in rows]
                many = len(exps) > 1
                bits = []
                for i, (st, su) in enumerate(exps):
                    tag = f" [{st:g}:{su:g}]" if many else ""
                    bits.append(f"beta{tag} ~ h^({rate(hs, [r['betas'][i] for r in rows]):+.3f})")
                for i, st in enumerate(c_exps):
                    tag = f" [{st:g}]" if many else ""
                    bits.append(f"C_h{tag} ~ h^({rate(hs, [r['chs'][i] for r in rows]):+.3f})")
                print("     least-squares fit over the ladder:  " + ",  ".join(bits))
                print("     an exponent near 0 is uniform; positive means the constant "
                      "DECAYS as h -> 0")
            if rows:
                collected[(prod, count)] = rows
            print()

    # nothing new was measured on a redraw, so there is nothing to save
    save_to = args.save or (None if args.read else stem + ".json")
    if save_to:
        with open(save_to, "w") as fh:
            json.dump({"dim": DIM, "law": law, "nu": args.nu, "cells": cells,
                       "exps": [list(e) for e in exps], "c_exps": c_exps,
                       "rows": {f"{p_}/{c}": r for (p_, c), r in collected.items()}}, fh)
        print(f"wrote {save_to}")

    plot_to = args.plot or stem + ".png"
    if plot_to:
        if not args.read and len(levels) < 2:
            print("--plot wants at least two levels: a single h draws no curve against h")
        else:
            out = draw(plot_to, collected, DIM, law, cells, nu, products)
            print(f"wrote {out}" if out else "nothing to plot")


if __name__ == "__main__":
    main()
