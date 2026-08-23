#!/usr/bin/env python3
"""Cauchy mechanics on a quarter annulus: Lame's thick-walled tube.

A tube from a to b under internal pressure p_a and external p_b, rollers on the
symmetry planes. With

    A = (a^2 p_a - b^2 p_b) / (b^2 - a^2)
    B = (p_a - p_b) a^2 b^2 / (b^2 - a^2)

the closed form is

    sigma_rr(r) = A - B/r^2
    u_r(r)      = A r / (2(lam + mu))  +  B / (2 mu r)

Neither term is polynomial, so neither is in the reconstruction: as with the
Dupuit profile, the error is a resolution and the refinement table measures it.

The boundary datum: the exact stress at r = a is not -p_a I, but its traction
is -p_a n, and a uniform -p_a I delivers exactly that -- so the condition is a
constant tensor even though the solution is curved.

Run it:

    PYTHONPATH=.. python cauchy_mechanics.py
    PYTHONPATH=.. python cauchy_mechanics.py --dim 3 --product stabilized_bdm
"""

import argparse
import os
import sys
import math

import mimetika_cxx as mk
import numpy as np

# geometry and data
A_IN, B_OUT, HZ = 1.0, 4.0, 1.0
P_INNER, P_OUTER = 1.0, 0.25
MU, LAM = 1.0, 1.0

FAMILIES = {
    "cartesian": mk.Family.cartesian,
    "simplex": mk.Family.simplex,
    "prism": mk.Family.prism,
}


def solvers(rtol):
    """The linear solvers, at the residual tolerance asked for.

    "riesz" is the Riesz map of the space the operator is an isomorphism on --
    P is the Gram matrix of its norm -- so its iteration count does not grow
    with the mesh. "direct" is a full factorization: exact, and the wrong
    instrument past a few hundred thousand unknowns.

    The tolerance is on the residual, not on the answer. An iterative solve
    cannot show the round-off floor a direct one leaves, so a patch test read
    through it is bounded by this number rather than by the method.
    """
    return {
        "riesz": mk.SolverOptions(
            method="gmres", preconditioner="riesz", rtol=rtol, max_iterations=2000
        ),
        # The same map, with the stress block inverted by an auxiliary space.
        # ADS splits an H(div) operator along the de Rham complex instead of
        # factorizing it, so it costs no fill and is linear in the unknowns.
        # It is a 3D construction and it wants one unknown per facet; the AFW
        # facet carries d moments in each of d components, and reaches it
        # through the facet-constant subspace, as a two-level cycle.
        "ads": mk.SolverOptions(
            method="gmres", preconditioner="riesz", rtol=rtol, max_iterations=2000,
            riesz_block_pc="ads",
        ),
        # the block solved to a tolerance rather than approximated by one
        # cycle: more work per iteration, and a count that stops drifting
        "ads-cg": mk.SolverOptions(
            method="gmres", preconditioner="riesz", rtol=rtol, max_iterations=2000,
            riesz_block_pc="ads", riesz_block_its=50, riesz_block_rtol=1e-2,
        ),
        "direct": mk.SolverOptions(),
    }


SOLVER_NAMES = ("direct", "riesz", "ads", "ads-cg")
DEFAULT_RTOL = 1e-9

# ADS is a three-dimensional construction: its auxiliary spaces are built from
# the discrete gradient and curl, d_1 and d_2, and in two dimensions the
# H(div) unknowns sit on edges rather than faces -- the maps do not address
# them, and hypre has no ADS for that case.
def require_three_dimensions(solver, dim):
    if solver.startswith("ads") and dim != 3:
        raise SystemExit(
            f"--solver {solver} is a 3D construction (it needs the discrete "
            f"gradient and curl of a 3-complex); this problem is {dim}D. "
            "Use --solver riesz, or run in 3D."
        )



PRODUCTS = {
    "derham_bdm": mk.StressRealization.derham_bdm,
    "stabilized_bdm": mk.StressRealization.stabilized_bdm,
    # No reconstruction: d per facet, one constant traction vector, and M the
    # diagonal primal-dual star -- the two-point stress. Half the unknowns and
    # an eighth of the matrix entries of the products above, on any mesh, and
    # consistent only where the mesh is face-orthogonal. The Lame annulus is
    # not, so run it here to see what that costs rather than to be exact.
    #
    # It exists in four fields only, which formulation_for enforces.
    "diagonal_afw": mk.StressRealization.diagonal_afw,
}

FORMULATIONS = {
    "weak_symmetry": mk.StressFormulation.weak_symmetry,
    "weak_symmetry_total": mk.StressFormulation.weak_symmetry_total,
}


def formulation_for(product, asked):
    """Three fields, or four with the total pressure p = lambda div u.

    diagonal_afw is diagonal only when the compliance is (2 mu)^-1, which is
    what the total-pressure form gives; in three fields the trace couples the
    traction components and the product does not exist.
    """
    # Omitted is not the same as asked for. `asked` is None when the caller
    # said nothing, and then the product decides: three fields for the BDM
    # ones, four for diagonal_afw, which has no other form. Only an explicit
    # --formulation weak_symmetry with diagonal_afw is a contradiction, and it
    # is the only case refused.
    if product == "diagonal_afw":
        if asked == "weak_symmetry":
            raise SystemExit(
                "--product diagonal_afw exists only in the four-field form: drop "
                "--formulation weak_symmetry, or pass weak_symmetry_total"
            )
        return mk.StressFormulation.weak_symmetry_total
    return FORMULATIONS[asked or "weak_symmetry"]


class Lame:
    """The closed form, from the two Lame constants of the loading."""

    def __init__(self, mat):
        a2, b2 = A_IN * A_IN, B_OUT * B_OUT
        self.A = (a2 * P_INNER - b2 * P_OUTER) / (b2 - a2)
        self.B = (P_INNER - P_OUTER) * a2 * b2 / (b2 - a2)
        self.mat = mat

    def u_r(self, r):
        return self.A * r / (2.0 * (self.mat.lame + self.mat.shear)) + self.B / (
            2.0 * self.mat.shear * r
        )

    def sigma_rr(self, r):
        return self.A - self.B / (r * r)


def isotropic(value):
    """The constant stress tensor whose traction is `value` times the normal."""
    s = [0.0] * 9
    for k in range(3):
        s[k * 3 + k] = value
    return s


def solve(nr, nt, dim, family, how, mat, form=None, solver="riesz", rtol=DEFAULT_RTOL):
    """Build the annulus, impose the three conditions, solve, measure."""
    form = mk.StressFormulation.weak_symmetry if form is None else form
    mesh = mk.annulus(nr, nt, dim, family, A_IN, B_OUT, HZ)

    rmid = math.sqrt(A_IN * B_OUT)
    inner, outer, symmetry = [], [], []
    for f in mk.boundary_facets(mesh, dim):
        x = mk.centroid(mesh, dim - 1, f)
        on_symmetry = (
            abs(x[0]) < 1e-8
            or abs(x[1]) < 1e-8
            or (dim == 3 and (abs(x[2]) < 1e-8 or abs(x[2] - HZ) < 1e-8))
        )
        if on_symmetry:
            symmetry.append(f)
        elif math.hypot(x[0], x[1]) < rmid:
            inner.append(f)
        else:
            outer.append(f)

    model = mk.CauchyMechanicsModel(mesh, dim, mat, how, form)
    model.add_traction(inner, isotropic(-P_INNER))
    model.add_traction(outer, isotropic(-P_OUTER))
    model.add_free_slip(symmetry)  # rollers: no normal displacement, no shear
    model.solve(options=solvers(rtol)[solver])

    ex = Lame(mat)
    worst = rms = 0.0
    for e in range(model.n_cells):
        x = mk.centroid(mesh, dim, e)
        r = math.hypot(x[0], x[1])
        # the radial displacement, projected out of the cell's vector unknown
        u_r = (model.displacement(e, 0) * x[0] + model.displacement(e, 1) * x[1]) / r
        d = u_r - ex.u_r(r)
        worst = max(worst, abs(d))
        rms += d * d
    return model, mesh, ex, worst, math.sqrt(rms / model.n_cells)


# One process speaks and writes. Under mpirun every rank runs this file and
# solves the same problem -- the algebra is shared out, the script is not -- so
# without this the report appears N times and N processes race to write the
# same .vtu. The solve itself is unaffected: every rank takes part in it, and
# every rank ends up with the whole answer.
def only_root():
    if mk.mpi_rank() == 0:
        return True
    sys.stdout = open(os.devnull, "w")
    return False


def main():
    root = only_root()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dim", type=int, default=2, choices=(2, 3))
    ap.add_argument("--family", default="simplex", choices=sorted(FAMILIES))
    ap.add_argument("--product", default="stabilized_bdm", choices=sorted(PRODUCTS))
    ap.add_argument("--formulation", default=None, choices=sorted(FORMULATIONS),
                    help="three fields, or four with the total pressure p = lambda div u; "
                         "the default follows the product")
    ap.add_argument("--nr", type=int, default=6, help="radial divisions of the coarse mesh")
    ap.add_argument("--nu", type=float, default=None, help="Poisson ratio (default lam = mu = 1)")
    ap.add_argument("--vtu", help="write the coarse solution to this .vtu")
    ap.add_argument("--solver", default="riesz", choices=sorted(SOLVER_NAMES))
    ap.add_argument("--rtol", type=float, default=DEFAULT_RTOL,
                    help="residual tolerance of the iterative solver")
    args = ap.parse_args()

    if args.nu is None:
        mat = mk.ElasticMaterial(MU, LAM)
    else:
        mat = mk.ElasticMaterial(MU, 2.0 * MU * args.nu / (1.0 - 2.0 * args.nu))

    family, how = FAMILIES[args.family], PRODUCTS[args.product]
    require_three_dimensions(args.solver, args.dim)
    form = formulation_for(args.product, args.formulation)
    print(f"Lame tube, {args.dim}D {args.family}, {mk.stress_realization_name(how)}, "
          f"{mk.stress_formulation_name(form)}")
    print(f"  a = {A_IN}, b = {B_OUT},  p(a) = {P_INNER}, p(b) = {P_OUTER}")
    print(f"  mu = {mat.shear}, lam = {mat.lame}  (nu = {mat.poisson():.4f})\n")

    # ---- the profile on one mesh ------------------------------------------
    model, mesh, ex, worst, rms = solve(args.nr, args.nr // 2, args.dim, family, how, mat, form, args.solver, args.rtol)
    print(f"  {model.n_cells} cells, {model.n_dofs} dofs, {model.n_stabilized} stabilized\n")
    print(f"  {'r':>8}  {'u_r (computed)':>16}  {'u_r (Lame)':>12}  {'error':>10}"
          f"  {'sigma_rr (Lame)':>16}")
    rows = []
    for e in range(model.n_cells):
        x = mk.centroid(mesh, args.dim, e)
        r = math.hypot(x[0], x[1])
        u_r = (model.displacement(e, 0) * x[0] + model.displacement(e, 1) * x[1]) / r
        rows.append((r, u_r, ex.u_r(r), ex.sigma_rr(r)))
    rows.sort()
    for r, got, want, srr in rows[:: max(1, len(rows) // 10)]:
        print(f"  {r:8.4f}  {got:16.6f}  {want:12.6f}  {got - want:+10.2e}  {srr:16.6f}")

    # ---- the same solution as a field on the cells --------------------------
    if args.vtu and root:
        n = model.n_cells
        u = np.zeros((n, 3))  # VTK reads a 3-vector; a 2D run leaves u_z at zero
        r = np.empty(n)
        u_r = np.empty(n)
        for e in range(n):
            x = mk.centroid(mesh, args.dim, e)
            for k in range(args.dim):
                u[e, k] = model.displacement(e, k)
            r[e] = math.hypot(x[0], x[1])
            u_r[e] = (u[e, 0] * x[0] + u[e, 1] * x[1]) / r[e]
        mk.write_vtu(
            mesh,
            args.vtu,
            {
                "displacement": u,
                "u_r": u_r,
                "u_r_lame": np.array([ex.u_r(v) for v in r]),
                "sigma_rr_lame": np.array([ex.sigma_rr(v) for v in r]),
            },
        )
        print(f"\n  wrote {args.vtu}")

    # ---- and what refinement does to it ------------------------------------
    print(f"\n  {'cells':>8}  {'max error':>11}  {'rms error':>11}  {'rate':>6}")
    previous = None
    for nr in (args.nr, 2 * args.nr, 4 * args.nr):
        m, _, _, worst, rms = solve(nr, nr // 2, args.dim, family, how, mat, form, args.solver, args.rtol)
        rate = "" if previous is None else f"{math.log2(previous / rms):6.2f}"
        print(f"  {m.n_cells:8d}  {worst:11.3e}  {rms:11.3e}  {rate:>6}")
        previous = rms


if __name__ == "__main__":
    main()
