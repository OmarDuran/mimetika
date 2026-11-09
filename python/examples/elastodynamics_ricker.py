"""Elastic wave propagation: the Ricker pulse in a two-layer medium, marched
with a diagonally implicit Runge--Kutta rule.

THE POINT OF THE EXERCISE is that mimetika's static mixed elasticity operator
is already the whole spatial side of the elastodynamic problem. The first-order
velocity--stress system

    A^-1 d_t s - eps(v) = 0 ,     rho d_t v - div s = f ,

is what you get by differentiating the Hellinger--Reissner rows in time and
reading v = d_t u, w = d_t gamma. Nothing in the stress operators changes: the
assembled saddle point

    K = [  A   B^T  C^T ]            M = diag( A , M_v , 0 )
        [ -B    0    0  ]
        [ -C    0    0  ]

becomes M d_t U + K U = F, and the only new object is the velocity mass M_v.
The off-diagonal blocks are exactly anti-transposed -- verified, not assumed --
so

    E = 1/2 s^T A s + 1/2 v^T M_v v

is a conserved quantity of the semi-discrete system for ANY symmetric M_v:
d_t E = -s^T B^T v + v^T B s - (C s)^T w = 0 on the constraint C s = 0. The
implicit midpoint rule conserves every quadratic invariant of a linear system
exactly, so the measured energy drift is a round-off quantity. That is the
sharp correctness check here, and it needs no reference solution.

THE TWO FAMILIES DIFFER IN KIND, not in accuracy:

  * stabilized_bdm (weak symmetry) carries the rotation multiplier gamma as a
    cell field with NO mass, so M is singular and the system is an index-1
    DAE. The stage solve handles it -- M + dt a_ii K is nonsingular -- and the
    update never needs gamma from the previous step, only from the stage.
  * stabilized_vem (strong symmetry) has no multiplier at all: symmetry is
    built into the stress space, the cell unknown is the six-dimensional space
    of rigid motions, and M is nonsingular. A genuine ODE.

so the same marcher drives a DAE and an ODE with no change of route.

THE MEDIUM is the two-layer Ricker test: equal densities and a factor-two
contrast in both wave speeds, which is a contrast in mu, not in lambda alone --
c_S = sqrt(mu/rho). That is what set_shear_per_cell is for.

Homogeneous Dirichlet is NATURAL in the mixed realization. The datum enters the
stress row as int_G v_D . (tau n), which vanishes for v_D = 0, so no constraint
is imposed and no traction dof is pinned: nothing to call.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- BEFORE mimetika_cxx; see python/_shadowing.py
import mimetika_cxx as mk

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mimetika.simulation.timestepping import TABLEAUS, RKTimeStepping  # noqa: E402

DIM = 3
LENGTH = 3.0          # the paper's square is (-3/2, 3/2)^2, in the x-y plane
ORIGIN = -1.5
INTERFACE = 1         # the layers meet on y = 0, the axis the paper splits

PRODUCTS = {
    "stabilized_bdm": (mk.StressRealization.stabilized_bdm,
                       mk.StressFormulation.weak_symmetry),
    "stabilized_vem": (mk.StressRealization.stabilized_vem,
                       mk.StressFormulation.strong_symmetry),
}

# ---- the slab ----------------------------------------------------------------
# A SINGLE LAYER OF CELLS IN Z, refined only in x and y: the paper's test is two
# dimensional and this is how a three-dimensional code runs it. mk.box defaults
# to the UNIT box and takes `lengths` and `origin`, so the extent has to be
# asked for -- it is not implied by anything else in the script.
#
# One cell through the thickness is plane strain only if the z faces are
# SYMMETRY planes rather than clamped ones:
#
#     u_z = 0 ,   sigma_zx = sigma_zy = 0 ,   sigma_zz = lambda (e_xx + e_yy)
#
# which is free slip: the normal traction stays free, as plane strain needs.
# Clamping those faces would pin u_x and u_y through the thickness and the
# answer would not be the paper's. The four x and y faces get no call at all,
# which in the mixed realization is the homogeneous Dirichlet the paper imposes.
def _at(args, n):
    """The same settings at a different in-plane refinement. brick() reads the
    side lengths from args.n, so a mesh and the args it is measured with must
    never disagree -- every helper below builds its own mesh from its own args
    rather than being handed one."""
    return argparse.Namespace(**{**vars(args), "n": n})


def mesh_of(args):
    tz = args.thickness if args.thickness > 0.0 else LENGTH / args.n
    return mk.box([args.n, args.n, 1], DIM, mk.Family.cartesian,
                  lengths=[LENGTH, LENGTH, tz],
                  origin=[ORIGIN, ORIGIN, -0.5 * tz])


def plane_strain(model, mesh):
    """Free slip on the two faces normal to z, and nothing anywhere else."""
    # the outward normal comes back AREA WEIGHTED, so it has to be normalized
    # before its components mean anything: on a fine slab the z face's area is
    # far below one and a raw component test silently selects nothing
    faces = []
    for f in model.outer_boundary_facets():
        v = np.asarray(mk.boundary_outward_normal(mesh, DIM, f))
        if abs(v[2]) / np.linalg.norm(v) > 0.5:
            faces.append(f)
    if not faces:
        raise SystemExit("the slab has no z-normal boundary facets")
    model.add_free_slip(faces)


# ---- the medium --------------------------------------------------------------
# rho = 1 either side; the lower layer carries (c_S, c_P) = (1, sqrt 3) and the
# upper twice both. mu = rho c_S^2 and lambda = rho (c_P^2 - 2 c_S^2), so the
# pair is (1, 1) below and (4, 4) above: BOTH moduli jump, and a layered lambda
# alone could not express a shear-speed contrast.
LAYERS = {
    "lower": dict(rho=1.0, cs=1.0, cp=np.sqrt(3.0)),
    "upper": dict(rho=1.0, cs=2.0, cp=2.0 * np.sqrt(3.0)),
}


def moduli(layer):
    rho, cs, cp = layer["rho"], layer["cs"], layer["cp"]
    return rho * cs * cs, rho * (cp * cp - 2.0 * cs * cs)


def medium(mesh, homogeneous):
    """Per-cell (mu, lambda, rho), split on y = 0 as the paper splits it."""
    n = mesh.count(DIM)
    mu = np.empty(n)
    lam = np.empty(n)
    rho = np.empty(n)
    for e in range(n):
        y = mk.centroid(mesh, DIM, e)[INTERFACE]
        key = "lower" if (homogeneous or y < 0.0) else "upper"
        mu[e], lam[e] = moduli(LAYERS[key])
        rho[e] = LAYERS[key]["rho"]
    return mu, lam, rho


# ---- assembly ----------------------------------------------------------------
def assemble(mesh, product, mu, lam, bc=None):
    """The static operator, untouched.

    The four x and y faces get no call, and in the mixed realization that IS
    homogeneous Dirichlet: the datum reaches the stress row as
    int_G v_D . (tau n), which vanishes for v_D = 0, so nothing is imposed
    there. `bc` is the plane-strain free slip on the z faces, which DOES pin --
    the tangential traction is essential -- and those rows are reduced out in
    `setup`."""
    how, form = PRODUCTS[product]
    model = mk.CauchyMechanicsModel(
        mesh, DIM, mk.ElasticMaterial(float(mu[0]), float(lam[0])), how, form)
    if np.ptp(mu) > 0.0:
        model.set_shear_per_cell([float(v) for v in mu])
    if np.ptp(lam) > 0.0:
        model.set_lame_per_cell([float(v) for v in lam])
    if bc is not None:
        bc(model, mesh)
    model.build(assemble_jacobian=True)

    row, col, val, n_dofs = mk.system_triplets(model)
    K = sp.csr_matrix((val, (row, col)), shape=(n_dofs, n_dofs))
    blocks = {b["name"]: [tuple(r) for r in b["ranges"]]
              for b in mk.field_blocks(model)}
    free = np.ones(n_dofs, bool)
    free[mk.constrained_dofs(model)] = False
    if bc is None and not free.all():
        raise SystemExit("dofs were pinned with no boundary condition asked for")
    if bc is not None and free.all():
        raise SystemExit("the plane-strain condition pinned nothing; the z-face "
                         "selection is wrong")
    return model, K, blocks, free


def brick(mesh, args):
    """The cell side lengths, per axis, and the volumes.

    A SLAB IS NOT A CUBE, so the closed forms below -- the inertia, the
    quadrature -- need all three sides rather than one. Still a structured box
    of identical bricks, which is what makes them closed forms at all, so a mesh
    that is not one is refused."""
    h = np.array([LENGTH / args.n, LENGTH / args.n,
                  args.thickness if args.thickness > 0.0 else LENGTH / args.n])
    vols = np.array([mk.measure(mesh, DIM, e) for e in range(mesh.count(DIM))])
    if not np.allclose(vols, h.prod(), rtol=1e-10):
        raise SystemExit("the closed-form cell inertia assumes identical bricks")
    return h, vols


# ---- the velocity mass -------------------------------------------------------
# WHAT THE CELL DOF IS fixes M_v, and the model states it: displacement(e, k) is
# -U[k] / |E|, so the dof is -int_E u, the cell moment with the multiplier's
# sign. For the wrench the six coefficients are the L^2 projection of u onto the
# rigid motions and the last three are read the same way, -|E| omega.
#
#   v|_E = a + omega x xi ,   a = -d_t / |E| ,   omega = -d_r / |E|
#   int_E rho |v|^2 = rho ( |E| |a|^2 + omega^T J_E omega )
#                   = rho ( |d_t|^2 / |E| + d_r^T J_E d_r / |E|^2 )
#
# with J_E = int_E (|xi|^2 I - xi (x) xi) = tr(M_2) I - M_2, the same J the vem
# stress operator builds for its rotation reconstruction. On a BRICK of sides h
# the centred second moment is M_2 = |E| diag(h_x^2, h_y^2, h_z^2) / 12, so
#
#   J_E = (|E| / 12) diag( h_y^2 + h_z^2 , h_x^2 + h_z^2 , h_x^2 + h_y^2 ) ,
#
# diagonal but NOT isotropic once the cell stops being a cube -- which is the
# whole point of a slab. It stays diagonal, so M_v stays diagonal and the
# velocity still eliminates cell by cell. The sign never reaches M_v, which is
# quadratic.
def velocity_mass(mesh, blocks, rho, h, vols):
    per = (blocks["u_0"][0][1] - blocks["u_0"][0][0]) // mesh.count(DIM)
    if per not in (DIM, 2 * DIM):
        raise SystemExit(f"{per} cell velocity dofs is neither d nor 2d")
    w = np.empty(per * mesh.count(DIM))
    w.reshape(-1, per)[:, :DIM] = (rho / vols)[:, None]
    if per == 2 * DIM:
        h2 = h ** 2
        inertia = np.array([h2[1] + h2[2], h2[0] + h2[2], h2[0] + h2[1]]) / 12.0
        w.reshape(-1, per)[:, DIM:] = (rho / vols)[:, None] * inertia[None, :]
    return w


# ---- the initial pulse -------------------------------------------------------
def gauss(order=4):
    """Tensor Gauss--Legendre on the reference cube, for the cell moments of a
    field that is not a polynomial."""
    x, wx = np.polynomial.legendre.leggauss(order)
    g = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    w = np.einsum("i,j,k->ijk", wx, wx, wx).ravel() / 8.0   # weights sum to 1
    return g / 2.0, w                                       # points in (-1/2, 1/2)


def ricker(mesh, blocks, per, h, vols, theta, fc, centre):
    """The dofs of the paper's (81),

        v_0 = theta exp(-pi^2 r^2 / lam^2) (x - x_c, y - y_c, 0) ,

    with r the IN-PLANE radius and lam = v_P / f_c read from the layer the
    source sits in. Both the radius and the vector drop z: the test is two
    dimensional, and a radius that counted the thickness would make the pulse
    depend on how thin the slab is.

    The field is curl free in the plane -- grad r is parallel to (x - x_c) --
    so it is a PURE P SOURCE. Any shear arrival at the lower sensor is
    conversion at y = 0, not a feature of the source.

    Translations are the cell moment, -int_E v_0; rotations are -|E| omega with
    omega = J_E^-1 int_E (e_a x xi) . v_0, the moment against the rotation
    generators, which is what the wrench's extra three slots hold. J_E is
    diagonal but anisotropic on a brick, so it is inverted component by
    component rather than by one scalar.
    """
    layer = "lower" if centre[INTERFACE] < 0.0 else "upper"
    lam = LAYERS[layer]["cp"] / fc
    pts, wts = gauss()
    h2 = h ** 2
    jacobi = (vols[:, None] *
              np.array([h2[1] + h2[2], h2[0] + h2[2], h2[0] + h2[1]]) / 12.0)
    out = np.zeros(per * mesh.count(DIM))
    d = out.reshape(-1, per)
    for e in range(mesh.count(DIM)):
        xE = np.asarray(mk.centroid(mesh, DIM, e))
        x = xE + h * pts                       # the brick's quadrature points
        zeta = x - centre
        zeta[:, 2] = 0.0                       # in-plane offset only
        r2 = np.einsum("ij,ij->i", zeta, zeta)
        v = theta * np.exp(-np.pi ** 2 * r2 / lam ** 2)[:, None] * zeta
        vol = vols[e]
        d[e, :DIM] = -vol * (wts[:, None] * v).sum(0)
        if per == 2 * DIM:
            xi = x - xE
            mom = (wts[:, None] * np.cross(xi, v)).sum(0) * vol
            d[e, DIM:] = -vol * mom / jacobi[e]
    return out, lam


# ---- the stage solve --------------------------------------------------------
# The stage matrix M + dt a_ii K is 61k dofs at n = 12 and SuperLU puts 359M
# nonzeros into L + U -- 224 s and some 4 GB, against 33 s for the 200 steps it
# then serves. So eliminate instead.
#
# M_v IS DIAGONAL by construction (rho/|E|, and rho h^2 / 6|E| on the wrench's
# rotation slots), so the velocity leaves cell by cell with no solve at all.
# Writing g = dt a_ii and reading the stage rows
#
#   [ A     g B^T   g C^T ] [Zs]   [rs]        Zv = D^-1 ( rv + g B Zs )
#   [ -g B  M_v     0     ] [Zv] = [rv]  ==>   C Zs = -rg / g
#   [ -g C  0       0     ] [Zg]   [rg]
#
# substitution leaves the stress and the multiplier alone:
#
#   ( A + g^2 B^T D^-1 B ) Zs + g C^T Zg = rs - g B^T D^-1 rv ,   g C Zs = -rg
#
# which is SYMMETRIC, and SPD on the strong family where there is no C. That is
# the whole gain: a non-symmetric saddle point of 44k/61k becomes a 34k conjugate
# gradient, or a 56k symmetric-indefinite MINRES.
#
# WHY IT CONVERGES FAST. The shift is small. With dt set by accuracy rather than
# stability -- the point of an A-stable rule -- the CFL number dt c_P / h is
# 0.07 here, and the largest discrete frequency is omega_max ~ pi c_P / h, so
#
#     ( g omega_max )^2 ~ 0.012 :
#
# the operator is A perturbed by about one percent. A is the stress compliance,
# a mass-like facet matrix whose conditioning does not depend on h, so a
# diagonal preconditioner is already enough and the iteration count does not
# grow under refinement. The shift only matters once dt reaches the explicit
# stability limit, which is not where an implicit scheme is run.
#
# CONSERVATION SURVIVES, and not for the reason first expected. The energy
# identity reduces to d_t E = -(C sigma)^T omega, so what matters is not how
# accurately the stage is solved but whether the symmetry constraint holds. A
# saddle-point solve satisfies it to rtol and the drift followed, at 1e-12; the
# projected iteration below holds C sigma = 0 to ROUND-OFF at any tolerance, and
# the drift then sits at 1e-16 even for rtol = 1e-6 -- cleaner than the direct
# factorization, whose own round-off leaves a larger constraint residual.
class Direct:
    """One factorization, reused for every stage of every step."""

    name = "direct"

    def __init__(self, M, K, shift, parts, rtol, progress):
        clock = time.perf_counter()
        self.lu = spla.splu((M + shift * K).tocsc())
        if progress:
            print(f"  factorized M + dt a11 K: {K.shape[0]} dofs, "
                  f"{(self.lu.L.nnz + self.lu.U.nnz) / 1e6:.1f}M nonzeros in "
                  f"L+U, {time.perf_counter() - clock:.1f}s", flush=True)
        self.iters = []

    def solve(self, rhs):
        return self.lu.solve(rhs)


def projected_pcg(S, apply_P, project, b, rtol, maxiter):
    """Preconditioned CG on S x = b, confined to the null space of the
    constraint. `apply_P` is the CONSTRAINT PRECONDITIONER, returning z = P r
    from

        [ D   C^T ] [ z ]   [ r ]
        [ C    0  ] [ w ] = [ 0 ]   so that  C z = 0 ,

    and `project` removes the range(C^T) component of a vector in the D^-1
    metric. Starting from x = 0, feasible because the constraint is homogeneous
    here, every direction lies in the null space and so does every iterate.

    WHY THE PROJECTION IS APPLIED TWICE. In exact arithmetic d = z + beta d
    stays in the null space, every z being there already. In floating point the
    beta d recursion accumulates a component outside it: measured here, |C x|
    held at 1e-17 for twenty-five iterations and then ran away to 1e-3, taking
    the residual with it -- the method broke down rather than converged. So the
    direction is re-projected each iteration, which costs one solve against the
    small constraint operator and nothing else.

    The tolerance is on the P-norm of the residual and is NOT pushed to
    round-off: the floor here is near 1e-9 relative, so asking for 1e-12 is
    asking for breakdown. The best iterate is kept and returned, so a stall
    costs accuracy rather than correctness.
    """
    x = np.zeros_like(b)
    r = b.copy()
    z = apply_P(r)
    rz = float(r @ z)
    if not rz > 0.0:
        return x, 0, 0
    rz0 = rz
    best, best_rz = x.copy(), rz
    d = z.copy()
    for k in range(1, maxiter + 1):
        Sd = S @ d
        denom = float(d @ Sd)
        if denom <= 0.0:
            return best, -1, k
        alpha = rz / denom
        x += alpha * d
        r -= alpha * Sd
        z = apply_P(r)
        rz_new = float(r @ z)
        if rz_new < best_rz:
            best, best_rz = x.copy(), rz_new
        if rz_new <= rz0 * rtol * rtol:
            return x, 0, k
        if rz_new > 4.0 * best_rz and k > 4:
            return best, 0, k            # breakdown: keep what was best
        d = project(z + (rz_new / rz) * d)
        rz = rz_new
    return best, 1, maxiter


class Reduced:
    """The velocity eliminated exactly, then CG on what is left."""

    name = "iterative"

    def __init__(self, M, K, shift, parts, rtol, progress):
        clock = time.perf_counter()
        self.n = K.shape[0]
        self.shift = shift
        self.rtol = rtol
        self.S_i, self.V_i, self.G_i = parts
        M, K = M.tocsr(), K.tocsr()

        A = M[np.ix_(self.S_i, self.S_i)]
        self.Dinv = 1.0 / M[np.ix_(self.V_i, self.V_i)].diagonal()
        self.Bt = K[np.ix_(self.S_i, self.V_i)].tocsr()   # K[s, u] IS B^T
        self.S = (A + shift ** 2 *
                  (self.Bt @ sp.diags(self.Dinv) @ self.Bt.T)).tocsr()
        dinv = 1.0 / self.S.diagonal()

        if len(self.G_i):
            # the constraint preconditioner's Schur complement, formed ONCE and
            # factored. C couples a cell's rotation rows to its own facets, so
            # W is a cell-graph operator of 3 dofs per cell -- small, sparse,
            # and nothing like the stage matrix it replaces
            Ct = K[np.ix_(self.S_i, self.G_i)].tocsr()
            W = (shift ** 2 * (Ct.T @ sp.diags(dinv) @ Ct)).tocsc()
            luW = spla.splu(W)
            self.method = f"projected cg ({W.shape[0]} constraints)"

            def apply_P(r):
                y = luW.solve(shift * (Ct.T @ (dinv * r)))
                return dinv * (r - shift * (Ct @ y))

            def project(v):
                # v - D^-1 C^T (C D^-1 C^T)^-1 C v, the component that would
                # take the iterate off the constraint
                mu = shift ** 2 * luW.solve(Ct.T @ v)
                return v - dinv * (Ct @ mu)
        else:
            self.method = "cg"

            def apply_P(r):
                return dinv * r

            def project(v):
                return v

        self.apply_P = apply_P
        self.project = project
        self.iters = []
        if progress:
            print(f"  eliminated the velocity: {self.S.shape[0]} dofs for "
                  f"{self.method}, {self.S.nnz / 1e6:.1f}M nonzeros, "
                  f"{time.perf_counter() - clock:.1f}s -- no factorization of "
                  f"the stage matrix", flush=True)

    def solve(self, rhs):
        r_v = rhs[self.V_i]
        b = rhs[self.S_i] - self.shift * (self.Bt @ (self.Dinv * r_v))
        x, info, its = projected_pcg(self.S, self.apply_P, self.project, b,
                                     self.rtol, 2000)
        if info != 0:
            raise SystemExit(f"{self.method} did not converge: info={info} "
                             f"after {its} iterations")
        self.iters.append(its)
        # THE MULTIPLIER IS NEVER NEEDED. The march reads Z only through
        # (p - Z)/a_ii and through M @ p, whose rotation rows are zero, so the
        # stage value of gamma affects nothing and is not computed.
        Z = np.zeros(self.n)
        Z[self.S_i] = x
        Z[self.V_i] = self.Dinv * (r_v + self.shift * (self.Bt.T @ x))
        return Z


SOLVERS = {"direct": Direct, "cg": Reduced}


# ---- the march ---------------------------------------------------------------
def march(K, M, idx, dt, n_steps, tableau, U0, report=None, progress=False,
          parts=None, solver="direct", rtol=1e-12):
    """A diagonally implicit Runge--Kutta step on M U' + K U = 0.

    The stage system is (M + dt a_ii K) Z_i = M U^n-1 - dt sum_{j<i} a_ij K Z_j
    and the update M U^n = M U^n-1 - dt sum_i b_i K Z_i. Both are written
    WITHOUT ever applying M^-1, which matters because M is singular on the weak
    family. Writing dt K Z_i = M g_i, the stage equation gives

        g_i = ( p_i - Z_i ) / a_ii ,   p_i = U^n-1 - sum_{j<i} a_ij g_j

    so the stage right-hand side is M p_i and the update is
    U^n = U^n-1 - sum_i b_i g_i, with every term an image of M. The massless
    rows of g are meaningless and never read: M p zeroes them, and the update
    re-zeroes them. For the midpoint rule this collapses to U^n = 2 Z - U^n-1.

    Singly diagonal: one a_ii for every stage, so with a constant dt the stage
    matrix is factorized ONCE for the whole march.
    """
    A, b, _ = TABLEAUS[tableau]
    diag = np.unique(np.round(np.diag(A), 15))
    if diag.size != 1:
        raise SystemExit(f"{tableau} is not singly diagonal; one factorization "
                         "per stage would be needed")
    stage = SOLVERS[solver](M, K, dt * diag[0], parts, rtol, progress)

    U = U0.copy()
    energy = np.empty(n_steps + 1)
    energy[0] = 0.5 * U @ (M @ U)
    traces = [] if report is None else [report(0.0, U)]
    clock = time.perf_counter()
    tick = max(1, n_steps // 10)
    for n in range(n_steps):
        g = []
        for i in range(len(b)):
            p = U - sum(A[i, j] * g[j] for j in range(i))
            Z = stage.solve(M @ p)
            g.append((p - Z) / A[i, i])
        U = U - sum(b[i] * g[i] for i in range(len(b)))
        U[~idx] = 0.0                      # the massless rows carry no state
        energy[n + 1] = 0.5 * U @ (M @ U)
        if report is not None:
            traces.append(report((n + 1) * dt, U))
        if progress and (n + 1) % tick == 0:
            spent = time.perf_counter() - clock
            drift = abs(energy[n + 1] - energy[0]) / energy[0]
            it = (f"   {np.mean(stage.iters[-tick:]):5.1f} its"
                  if stage.iters else "")
            print(f"  step {n + 1:5d}/{n_steps}  t={(n + 1) * dt:7.4f}  "
                  f"{spent:6.1f}s, ~{spent / (n + 1) * (n_steps - n - 1):5.1f}s "
                  f"left   drift {drift:.2e}{it}", flush=True)
    return U, energy, traces


# ---- the problem -------------------------------------------------------------
def setup(mesh, product, args, homogeneous=False, bc=None):
    mu, lam, rho = medium(mesh, homogeneous)
    model, K, blocks, free = assemble(mesh, product, mu, lam, bc)
    h, vols = brick(mesh, args)

    sb, se = blocks["u_0"][0]
    per = (se - sb) // mesh.count(DIM)
    w = velocity_mass(mesh, blocks, rho, h, vols)

    # A MOVES FROM THE STIFFNESS TO THE MASS, and this is the whole content of
    # going transient. The static stress row is the constitutive law
    #
    #     A s + B^T u + C^T g = 0 ,
    #
    # and its time derivative -- which is what A^-1 d_t s = eps(v) is -- reads
    #
    #     A d_t s + B^T v + C^T w = 0 :
    #
    # the same A, now multiplying the RATE. So the transient operator is not
    # (M, K) with the assembled K but (M, K - A), and that matters because
    #
    #     U'^T M U' - U^T M U = -2 dt Z^T K Z
    #
    # for the midpoint rule, which vanishes exactly when the stiffness is SKEW.
    # Stripping A leaves [0 B^T C^T; -B 0 0; -C 0 0], whose blocks are
    # anti-transposed, so it is -- and the energy is then conserved to round-off
    # rather than merely bounded. Leaving A in the stiffness is a silent
    # dissipation that looks like physics.
    n = K.shape[0]
    S = np.arange(*blocks["s_0"][0])
    A = _pad(K[np.ix_(S, S)], blocks["s_0"][0][0], n)
    mass = np.zeros(n)
    mass[sb:se] = w
    M = sp.csr_matrix(sp.diags(mass)) + A
    K_static = K                 # kept whole: S^-1 in eigen_min is a static solve
    K = K - A

    carried = np.zeros(n, bool)                 # the rows M is invertible on
    carried[slice(*blocks["s_0"][0])] = True
    carried[sb:se] = True

    # THE FREE-DOF REDUCTION. Free slip pins stress dofs only, so the cell block
    # survives whole and `u_at` is where each cell's slots land after the
    # squeeze; `parts` is the same three index sets in reduced numbering, which
    # is what the stage solve eliminates against.
    if not free[sb:se].all():
        raise SystemExit("a boundary condition pinned a cell velocity dof")
    keep = np.flatnonzero(free)
    where = np.full(n, -1)
    where[keep] = np.arange(len(keep))
    K = K[np.ix_(keep, keep)]
    K_static = K_static[np.ix_(keep, keep)]
    M = M[np.ix_(keep, keep)]
    carried = carried[keep]

    s_i = where[S[free[S]]]
    v_i = where[np.arange(sb, se)]
    g_i = (where[np.arange(*blocks["g_0"][0])] if "g_0" in blocks
           else np.arange(0, 0))

    skew = abs(K + K.T).max()
    if skew > 1e-9 * abs(K).max():
        raise SystemExit(f"the stripped stiffness is not skew: |K+K^T| = {skew:.3e}")

    return dict(parts=(s_i, v_i, g_i),
                model=model, K=K.tocsc(), M=M.tocsc(), K_static=K_static.tocsc(),
                blocks=blocks, per=per, carried=carried, h=h, vols=vols,
                mu=mu, rho=rho, n_free=len(keep),
                u_at=v_i.reshape(-1, per))


def pulse(p, mesh, args):
    """The Ricker initial state: u_0 = 0, so s(0) = A eps(u_0) = 0."""
    v0, wavelength = ricker(mesh, p["blocks"], p["per"], p["h"], p["vols"],
                            args.theta, args.fc, np.asarray(args.centre))
    U0 = np.zeros(p["n_free"])
    U0[p["u_at"].ravel()] = v0
    return U0, wavelength


def _pad(block, offset, n):
    """A block placed at (offset, offset) inside an n x n zero matrix."""
    b = block.tocoo()
    return sp.csr_matrix((b.data, (b.row + offset, b.col + offset)), shape=(n, n))


# ---- reading the state -------------------------------------------------------
def cell_velocity(U, u_at, vols):
    """The cell-mean velocity, v|_E = -d_t / |E|: the translation slots, which
    is what both families share. The wrench's three rotation slots are its own."""
    return -U[u_at[:, :DIM]] / vols[:, None]


def nearest(mesh, point):
    x = np.array([mk.centroid(mesh, DIM, e) for e in range(mesh.count(DIM))])
    return int(np.argmin(((x - np.asarray(point)) ** 2).sum(1)))


# ---- validation --------------------------------------------------------------
def mass_identity(product, args):
    """d(v)^T M_v d(v) == int_O rho |v|^2, exactly, for v IN the velocity space.

    This is what validates M_v, and it is sharper than it looks: it tests the
    dof CONVENTION and the weights together. The dof is -int_E u, not -|E| u, so
    the weight is 1/|E| rather than |E| -- the two differ by |E|^2 and the test
    separates them. A cellwise constant field lives in both families' velocity
    space; a global rigid rotation lives only in the wrench's, the weak family's
    velocity being piecewise constant, so that half of the check is reported
    only where it means something."""
    mesh = mesh_of(args)
    mu, lam, rho = medium(mesh, False)
    _, _, blocks, _ = assemble(mesh, product, mu, lam, plane_strain)
    h, vols = brick(mesh, args)
    per = (blocks["u_0"][0][1] - blocks["u_0"][0][0]) // mesh.count(DIM)
    w = velocity_mass(mesh, blocks, rho, h, vols).reshape(-1, per)
    xc = np.array([mk.centroid(mesh, DIM, e) for e in range(mesh.count(DIM))])
    pts, wts = gauss(order=5)
    out = {}

    rng = np.random.default_rng(0)
    c = rng.normal(size=(mesh.count(DIM), DIM))        # cellwise constant
    d = np.zeros((mesh.count(DIM), per))
    d[:, :DIM] = -vols[:, None] * c
    out["cellwise constant"] = (
        float((w * d * d).sum()),
        float((rho * vols * (c * c).sum(1)).sum()))

    if per == 2 * DIM:
        om = np.array([0.3, -0.5, 0.7])               # a global rigid rotation
        d = np.zeros((mesh.count(DIM), per))
        d[:, :DIM] = -vols[:, None] * np.cross(om, xc)
        d[:, DIM:] = -vols[:, None] * om
        exact = 0.0
        for e in range(mesh.count(DIM)):
            x = xc[e] + h * pts
            v = np.cross(om, x)
            exact += rho[e] * vols[e] * (wts * (v * v).sum(1)).sum()
        out["rigid rotation"] = (float((w * d * d).sum()), float(exact))
    return out


def eigen_min(product, args):
    """The lowest frequency of the semi-discrete system.

    Eliminating the stress gives M_v v'' = -S v with S = B P B^T, and S^-1 IS a
    static solve: the static u-row is S u = f_u, so applying S^-1 is one
    back-substitution against the factored static operator -- which also carries
    the symmetry constraint, so the weak and strong families need no separate
    treatment. With M_v diagonal,

        T = M_v^(1/2) S^-1 M_v^(1/2)

    is symmetric and its LARGEST eigenvalue is 1 / omega_min^2, which is the
    well-conditioned end to ask for."""
    p = setup(mesh_of(args), product, args, homogeneous=True, bc=plane_strain)
    U = p["u_at"].ravel()
    n = p["n_free"]
    lu = spla.splu(p["K_static"])
    root = np.sqrt(p["M"].tocsr()[np.ix_(U, U)].diagonal())

    def apply(y):
        f = np.zeros(n)
        f[U] = root * y
        return root * lu.solve(f)[U]

    T = spla.LinearOperator((len(U), len(U)), matvec=apply, dtype=float)
    top = spla.eigsh(T, k=1, which="LA", tol=1e-10,
                     return_eigenvectors=False)[0]
    return float(1.0 / np.sqrt(top)) if top > 0.0 else float("nan")


def agreement_check(args, products):
    """omega_min of the two families, refined, in a HOMOGENEOUS medium.

    WHY THIS AND NOT AN ANALYTIC FREQUENCY. The cube with v = 0 on all six faces
    -- the condition this example actually imposes -- has no closed-form elastic
    mode: neither the divergence-free nor the curl-free construction that gives
    sin.sin.sin modes can vanish on every face at once. Imposing free slip on
    four sides would buy one, u = (0,0,sin(m pi Z)) with omega = c_P m pi / L,
    but that is a different problem from the one being marched.

    So compare the two discretizations instead. They are different spaces with
    different dof counts and different symmetry enforcement -- weak, with a
    rotation multiplier, against strong, built into the stress -- so their
    omega_min approaching ONE limit from opposite sides is a statement about the
    continuous problem rather than about either scheme. Together with the mass
    identity, which fixes M_v on its own, and the patch-tested static operator,
    that is what pins the wave speed: there is no third unknown.
    """
    rows = []
    for n in args.agree:
        got = {}
        for q in products:
            clock = time.perf_counter()
            got[q] = eigen_min(q, _at(args, n))
            print(f"      n={n}, {q}: omega_min = {got[q]:.6f}  "
                  f"({time.perf_counter() - clock:.1f}s)", flush=True)
        gap = (abs(got[products[0]] - got[products[1]]) /
               max(got.values())) if len(products) > 1 else float("nan")
        rows.append(dict(n=n, h=LENGTH / n, omega=got, gap=gap))
    return rows


def order_check(args, product):
    """Temporal order by Richardson against the finest march on a FIXED mesh:
    the spatial error cancels, so what is measured is the integrator alone. The
    level immediately below the reference is dropped -- its rate is contaminated
    by being only one refinement away -- and the error is taken in the energy
    norm, which ignores the massless rows the weak family carries."""
    fine = _at(args, args.order_n)
    mesh = mesh_of(fine)
    p = setup(mesh, product, fine, bc=plane_strain)
    U0, _ = pulse(p, mesh, fine)
    ladder = sorted(args.order_steps)
    runs = []
    for steps in ladder:
        U, energy, _ = march(p["K"], p["M"], p["carried"],
                             args.t_final / steps, steps, args.tableau, U0,
                             parts=p["parts"], solver=args.solver,
                             rtol=args.rtol)
        runs.append((steps, U,
                     float(np.abs(energy - energy[0]).max() / energy[0])))
    ref = runs[-1][1]
    norm = lambda d: float(np.sqrt(abs(d @ (p["M"] @ d))))
    rows = [dict(steps=k, dt=args.t_final / k, error=norm(U - ref), drift=g)
            for k, U, g in runs[:-2]]
    for i in range(1, len(rows)):
        rows[i]["rate"] = float(np.log2(rows[i - 1]["error"] / rows[i]["error"]))
    return rows


# ---- the run -----------------------------------------------------------------
def run(args, product):
    mesh = mesh_of(args)
    clock = time.perf_counter()
    p = setup(mesh, product, args, bc=plane_strain)
    print(f"  assembled {p['n_free']} dofs in "
          f"{time.perf_counter() - clock:.1f}s", flush=True)
    clock = time.perf_counter()
    U0, wavelength = pulse(p, mesh, args)
    print(f"  projected the initial pulse in "
          f"{time.perf_counter() - clock:.1f}s", flush=True)
    n_cells = mesh.count(DIM)
    sensors = {"S1 (lower)": nearest(mesh, args.s1),
               "S2 (upper)": nearest(mesh, args.s2)}
    rows = list(sensors.values())

    xc = np.array([mk.centroid(mesh, DIM, e) for e in range(n_cells)])
    slab = np.arange(n_cells)        # one cell thick: the mesh IS the plane
    snaps = {}
    want = {int(round(f * args.t_final / args.dt)) for f in args.snapshots}

    steps = int(round(args.t_final / args.dt))
    if args.vtu_every > 0:
        want |= set(range(0, steps + 1, args.vtu_every))
    want.add(steps)
    stem = Path(args.vtu).with_suffix("") if args.vtu else None
    if stem is not None:
        stem.parent.mkdir(parents=True, exist_ok=True)
    written = []

    def report(t, U):
        v = cell_velocity(U, p["u_at"], p["vols"])
        k = int(round(t / args.dt))
        if k in want:
            snaps[k] = v[slab].copy()          # components, not magnitude
        if stem is not None and k in want:
            # the index, zero padded, is what makes ParaView read the files as
            # ONE series rather than as unrelated meshes
            path = f"{stem}_{product}_{k:04d}.vtu"
            mk.write_vtu(mesh, path,
                         {"velocity": np.ascontiguousarray(v),
                          "speed": np.linalg.norm(v, axis=1),
                          "mu": p["mu"]}, True)
            written.append((t, path))
        return v[rows]

    U, energy, traces = march(p["K"], p["M"], p["carried"], args.dt, steps,
                              args.tableau, U0, report, progress=True,
                              parts=p["parts"], solver=args.solver,
                              rtol=args.rtol)
    if written:
        series = "\n".join(
            f'    <DataSet timestep="{t:.10g}" group="" part="0" '
            f'file="{Path(f).name}"/>' for t, f in written)
        collection = Path(f"{stem}_{product}.pvd")
        collection.write_text(
            '<?xml version="1.0"?>\n'
            '<VTKFile type="Collection" version="0.1" '
            'byte_order="LittleEndian">\n  <Collection>\n'
            f"{series}\n  </Collection>\n</VTKFile>\n")
        written.append((float("nan"), str(collection)))
    return dict(mesh=mesh, p=p, energy=energy, traces=np.array(traces),
                written=written,
                sensors=list(sensors), t=np.arange(steps + 1) * args.dt,
                snaps=snaps, slab=slab, xc=xc, U=U, n_dofs=p["n_free"],
                wavelength=wavelength)


# ---- the figure --------------------------------------------------------------
def draw(path, results, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    products = list(results)
    fig, ax = plt.subplots(2, 2 + len(args.snapshots), squeeze=False,
                           figsize=(4.0 * (2 + len(args.snapshots)), 7.4))

    # energy: the headline. A conserved quadratic invariant, so these are flat
    # at the level of round-off and the y axis is a RELATIVE drift.
    a = ax[0][0]
    for name in products:
        r = results[name]
        e = r["energy"]
        a.plot(r["t"], (e - e[0]) / e[0], label=name, lw=1.4)
    a.set(xlabel="t", ylabel=r"$(E^n - E^0)\,/\,E^0$",
          title="energy drift\n" + r"$E=\frac{1}{2}\sigma^TA\sigma+\frac{1}{2}v^TM_vv$")
    a.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    a.axhline(0.0, color="0.7", lw=0.8, zorder=0)
    a.legend(fontsize=7)
    a.grid(alpha=0.3)

    a = ax[1][0]
    for name in products:
        r = results[name]
        e = r["energy"]
        a.semilogy(r["t"][1:], np.abs(e[1:] - e[0]) / e[0] + 1e-18, lw=1.2,
                   label=name)
    a.axhline(np.finfo(float).eps, color="crimson", ls=":", lw=1.0,
              label="machine epsilon")
    a.set(xlabel="t", ylabel="relative drift", title="the same, on a log axis")
    a.legend(fontsize=7)
    a.grid(alpha=0.3, which="both")

    # the seismograms, normal and tangential to the interface
    for j, comp in enumerate((1, 0)):
        a = ax[j][1]
        for name in products:
            r = results[name]
            for s, label in enumerate(r["sensors"]):
                a.plot(r["t"], r["traces"][:, s, comp], lw=1.2,
                       ls="-" if s == 0 else "--",
                       label=f"{label}, {name.split('_')[1]}")
        a.set(xlabel="t", ylabel=f"$v_{'xyz'[comp]}$",
              title=("normal to the interface ($v_y$)" if comp == 1
                     else "along it ($v_x$)"))
        a.legend(fontsize=6)
        a.grid(alpha=0.3)

    # the wavefield on y = 0
    xc = results[products[0]]["xc"]
    slab = results[products[0]]["slab"]
    for col, frac in enumerate(args.snapshots):
        k = int(round(frac * args.t_final / args.dt))
        for row, name in enumerate(products[:2]):
            a = ax[row][2 + col]
            r = results[name]
            if k not in r["snaps"]:
                a.axis("off")
                continue
            m = np.linalg.norm(r["snaps"][k], axis=1)
            a.tricontourf(xc[slab, 0], xc[slab, 1], m, levels=24)
            a.axhline(0.0, color="w", lw=1.0)       # the material interface
            a.plot(args.centre[0], args.centre[1], "w*", ms=8)
            for s_ in (args.s1, args.s2):
                a.plot(s_[0], s_[1], "wv", ms=5)
            a.set(title=f"{name}  $|v|$ at t={k * args.dt:.2f}",
                  xlabel="x", ylabel="y", aspect="equal")
    for row, name in enumerate(products[:2]):
        if len(products) == 1 and row == 1:
            for col in range(len(args.snapshots)):
                ax[1][2 + col].axis("off")

    fig.suptitle(
        f"Ricker pulse in a two-layer medium, {DIM}D  |  "
        f"{args.tableau}, dt={args.dt:g}, {args.n}$^3$ hexahedra  |  "
        r"$c_{S}:1\to2$,  $c_{P}:\sqrt{3}\to2\sqrt{3}$", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def draw_profiles(path, r, args, product):
    """The paper's Fig. 5: v_x on the top row, v_y on the bottom, at the four
    snapshot times from left to right.

    The slab is a structured n x n grid of identical bricks one cell thick, so
    the cell values ARE an image: sorting the centroids by (y, x) and reshaping
    gives the pixels directly, with no interpolation to blur what the scheme
    actually produced.

    The map is diverging and centred on zero -- white is no motion, the two
    signs are the two colours -- which is what makes the reflected and converted
    fronts legible. The limits are shared by the four panels of a row, so the
    columns are comparable as a time sequence rather than each being
    renormalized; `TwoSlopeNorm` keeps zero at the white point even when the
    extremes are not symmetric, and the colourbar is labelled min / 0 / max as
    the paper labels it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    # ONLY the --snapshots times. The snapshot set also carries whatever
    # --vtu-every asked for, and plotting all of it turns a four-column figure
    # into a contact sheet.
    want = [int(round(f / args.dt)) for f in args.snapshots]
    times = [k for k in want if k in r["snaps"]]
    if not times:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    xc = r["xc"]
    order = np.lexsort((xc[:, 0], xc[:, 1]))     # rows of constant y
    n = args.n
    extent = (ORIGIN, ORIGIN + LENGTH, ORIGIN, ORIGIN + LENGTH)

    fig, ax = plt.subplots(2, len(times), squeeze=False,
                           figsize=(3.1 * len(times), 6.6))
    for row, comp in enumerate((0, 1)):
        field = [r["snaps"][k][order, comp].reshape(n, n) for k in times]
        shared = args.profile_scale == "row"
        lo = min(float(f.min()) for f in field)
        hi = max(float(f.max()) for f in field)
        print(f"    {'xy'[comp]}: data range [{lo:.3g}, {hi:.3g}]", flush=True)
        if args.profile_vlim:
            # the paper's own limits, imposed rather than measured
            lo, hi = args.profile_vlim[2 * comp], args.profile_vlim[2 * comp + 1]
            shared = True
        if not lo < 0.0 < hi:                    # a one-signed row still needs 0
            lo, hi = min(lo, -abs(hi) * 1e-3), max(hi, abs(lo) * 1e-3)
        for col, k in enumerate(times):
            a = ax[row][col]
            if not shared:
                lo, hi = float(field[col].min()), float(field[col].max())
                if not lo < 0.0 < hi:
                    lo, hi = min(lo, -abs(hi) * 1e-3), max(hi, abs(lo) * 1e-3)
            norm = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)
            im = a.imshow(field[col], origin="lower", extent=extent,
                          cmap="RdBu_r", norm=norm, interpolation="nearest")
            a.axhline(0.0, color="0.35", lw=0.6)   # the material interface
            a.set_xticks([])
            a.set_yticks([])
            if row == 0:
                a.set_title(f"$t = {k * args.dt:g}$", fontsize=10)
            if col == 0:
                a.set_ylabel(f"$v_{'xy'[comp]}$", fontsize=12)
            bar = fig.colorbar(im, ax=a, orientation="horizontal",
                               fraction=0.05, pad=0.03,
                               ticks=[lo, 0.0, hi])
            bar.ax.set_xticklabels([f"{lo:.1e}", "0", f"{hi:.1e}"], fontsize=7)

    fig.suptitle(f"{product}: velocity profiles at "
                 f"$t \\in \\{{{', '.join(f'{k * args.dt:g}' for k in times)}\\}}$"
                 f"   ({args.n}$\\times${args.n} slab, $f_c={args.fc:g}$, "
                 f"{args.tableau})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  wrote {path}")


# ---- main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--product", default="stabilized_bdm,stabilized_vem",
                    help="comma separated; both families by default")
    ap.add_argument("--n", type=int, default=48,
                    help="cells per side IN PLANE; the slab is always one cell "
                         "thick. No longer bounded by a factorization")
    ap.add_argument("--thickness", type=float, default=0.0,
                    help="the slab's z extent; 0 keeps the cells cubic at L/n")
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--t-final", type=float, default=1.0)
    ap.add_argument("--solver", default="cg", choices=sorted(SOLVERS),
                    help="'cg' eliminates the velocity exactly and iterates on "
                         "what is left -- CG for the strong family, MINRES for "
                         "the weak one. 'direct' factorizes the whole stage "
                         "matrix: 224s and 4GB at n=12, but exact")
    ap.add_argument("--rtol", type=float, default=1e-9,
                    help="the stage tolerance. Conservation does NOT track it: "
                         "the projection keeps C sigma = 0 to round-off at any "
                         "tolerance, so the drift stays at 1e-16 even at 1e-6. "
                         "What it buys is solution accuracy, and 1e-9 is "
                         "already far below the discretization error")
    ap.add_argument("--tableau", default="implicit-midpoint",
                    choices=sorted(TABLEAUS))
    ap.add_argument("--fc", type=float, default=10.0,
                    help="the pulse's central frequency, 10 as the paper. The "
                         "slab makes that affordable: refinement is in plane "
                         "only, so n=48 already resolves c_P/f_c")
    ap.add_argument("--theta", type=float, default=1e-2)
    ap.add_argument("--centre", type=float, nargs=3,
                    default=[0.0, 2.0 / 3.0, 0.0],
                    help="the source (x_c, y_c, 0), in the upper layer as in "
                         "the paper")
    ap.add_argument("--s1", type=float, nargs=3, default=[1 / 3, -1 / 3, 0.0],
                    help="the sensor in the lower layer, the paper's S_1")
    ap.add_argument("--s2", type=float, nargs=3, default=[1 / 3, 1 / 3, 0.0],
                    help="and the one in the upper layer, S_2")
    ap.add_argument("--order-n", type=int, default=16,
                    help="the mesh the order and mass checks are measured on")
    ap.add_argument("--agree", type=int, nargs="*", default=[8, 16, 24, 32],
                    help="the mesh ladder the two families are compared on")
    ap.add_argument("--order-steps", type=int, nargs="*",
                    default=[100, 200, 400, 800, 1600],
                    help="the temporal ladder. The weak family needs the finer "
                         "end to reach its asymptotic order; the two coarsest "
                         "rates are the interesting ones")
    ap.add_argument("--profile-vlim", type=float, nargs=4, default=None,
                    metavar=("VX_LO", "VX_HI", "VY_LO", "VY_HI"),
                    help="impose the colour limits instead of measuring them. "
                         "The paper's Fig. 5 uses -6.7e-2 6.7e-2 -7.6e-2 1.2e-1")
    ap.add_argument("--profile-scale", default="row", choices=("row", "panel"),
                    help="'row' shares one colour scale across the four times, "
                         "so the columns read as a sequence; 'panel' rescales "
                         "each, which exposes the late, weaker field")
    ap.add_argument("--snapshots", type=float, nargs="*",
                    default=[0.125, 0.25, 0.5, 1.0],
                    help="the paper's Fig. 5 times; they also drive the vtu "
                         "series and must land on whole steps")
    ap.add_argument("--skip-checks", action="store_true")
    ap.add_argument("--vtu", default=None,
                    help="write the velocity field as a ParaView series, at the "
                         "--snapshots times and the final time")
    ap.add_argument("--vtu-every", type=int, default=0,
                    help="also write every N steps, for an animation rather "
                         "than a few stills")
    ap.add_argument("--plot", default="elastodynamics_ricker.png")
    ap.add_argument("--save", default="elastodynamics_ricker.json")
    args = ap.parse_args()

    products = [p.strip() for p in args.product.split(",") if p.strip()]
    for p in products:
        if p not in PRODUCTS:
            raise SystemExit(f"unknown product {p!r}; have {sorted(PRODUCTS)}")

    h = LENGTH / args.n
    tz = args.thickness if args.thickness > 0.0 else h
    lam = LAYERS["upper"]["cp"] / args.fc
    mu1, la1 = moduli(LAYERS["lower"])
    mu2, la2 = moduli(LAYERS["upper"])
    print(f"slab ({ORIGIN:g}, {ORIGIN + LENGTH:g})^2 x {tz:g}, "
          f"{args.n} x {args.n} x 1 = {args.n ** 2} hexahedra, "
          f"h = {h:g}, plane strain on the z faces")
    print(f"media  y<0: (mu, lambda) = ({mu1:g}, {la1:g}), c_S = "
          f"{LAYERS['lower']['cs']:g}, c_P = {LAYERS['lower']['cp']:.4g}")
    print(f"       y>0: (mu, lambda) = ({mu2:g}, {la2:g}), c_S = "
          f"{LAYERS['upper']['cs']:g}, c_P = {LAYERS['upper']['cp']:.4g}")
    print(f"source (x_c, y_c) = ({args.centre[0]:g}, {args.centre[1]:g}), "
          f"f_c = {args.fc:g}, lambda = c_P/f_c = {lam:.4g} -> "
          f"{lam / h:.1f} cells per wavelength "
          f"({lam / 2 / h:.1f} once transmitted into the slower layer)")
    if lam / 2 / h < 3.0:
        print("  NOTE: under three cells per transmitted wavelength. The energy "
              "and order checks are exact regardless -- they are algebraic -- "
              "but the waveform is not resolved. Raise --n or lower --fc.")
    for f in args.snapshots:
        k = f / args.dt
        if abs(k - round(k)) > 1e-6:
            raise SystemExit(f"snapshot t={f:g} is {k:.3f} steps at dt={args.dt:g}; "
                             "it would be silently rounded. Pick a dt that "
                             "divides every snapshot time")
    print(f"march  {args.tableau}, dt = {args.dt:g}, "
          f"{int(round(args.t_final / args.dt))} steps to T = {args.t_final:g}, "
          f"{args.solver} stage solve at rtol {args.rtol:g}")

    out = {"settings": vars(args), "products": {}}
    results = {}

    if not args.skip_checks and len(products) > 1:
        print("\nomega_min of the two families, refined (homogeneous medium).")
        print("No analytic value is used: the slab with v = 0 on its four in-")
        print("plane faces has no closed-form elastic mode, so the evidence is")
        print("two different spaces approaching one limit from opposite sides.")
        print("        n       h    " + "".join(f"{q:>18s}" for q in products)
              + "        gap")
        ladder = agreement_check(args, products)
        for r in ladder:
            print(f"      {r['n']:3d}  {r['h']:6.4f}  "
                  + "".join(f"{r['omega'][q]:18.6f}" for q in products)
                  + f"  {r['gap']:9.2%}")
        out["agreement"] = ladder

    for name in products:
        print(f"\n=== {name} " + "=" * (56 - len(name)), flush=True)
        if not args.skip_checks:
            for what, (got, exact) in mass_identity(
                    name, _at(args, args.order_n)).items():
                rel = abs(got - exact) / exact
                print(f"  velocity mass, {what:18s}: "
                      f"d^T M_v d = {got:.10g} vs {exact:.10g}  "
                      f"(rel {rel:.2e})")
            rows = order_check(args, name)
            print("  temporal order on a fixed mesh (energy norm vs the finest):")
            print("      steps      dt        error      rate     drift")
            for r in rows:
                print(f"      {r['steps']:5d}  {r['dt']:8.3e}  {r['error']:9.3e}"
                      f"   {r.get('rate', float('nan')):6.3f}  {r['drift']:9.2e}")
            out["products"][name] = dict(order=rows)

        r = run(args, name)
        results[name] = r
        drift = float(np.abs(r["energy"] - r["energy"][0]).max() / r["energy"][0])
        print(f"  {r['n_dofs']} dofs, energy drift over the march: {drift:.3e}"
              f"  ({drift / np.finfo(float).eps:.1f} eps)")
        out["products"].setdefault(name, {})["run"] = dict(
            n_dofs=int(r["n_dofs"]), drift=drift,
            energy=[float(v) for v in r["energy"]],
            t=[float(v) for v in r["t"]],
            traces=r["traces"].tolist(), sensors=r["sensors"])

        if args.plot:
            stem = Path(args.plot).with_suffix("")
            draw_profiles(f"{stem}_{name}_profiles.png", r, args, name)
        if r["written"]:
            vtus = [f for t, f in r["written"] if f.endswith(".vtu")]
            pvd = [f for t, f in r["written"] if f.endswith(".pvd")]
            print(f"  wrote {len(vtus)} snapshots, {Path(vtus[0]).name} .. "
                  f"{Path(vtus[-1]).name}")
            if pvd:
                print(f"  open {pvd[0]} in ParaView: the collection carries the "
                      "physical times")

    if args.plot:
        draw(args.plot, results, args)
    if args.save:
        Path(args.save).write_text(json.dumps(out, indent=1))
        print(f"wrote {args.save}")


if __name__ == "__main__":
    main()
