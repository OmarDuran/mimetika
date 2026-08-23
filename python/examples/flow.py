#!/usr/bin/env python3
"""Flow, of Darcy type, on a quarter annulus: the Dupuit profile.

Steady radial flow between two concentric boundaries, pressure prescribed on
both radii and the symmetry planes sealed. The closed form is logarithmic,

    p(r) = p_a + (p_b - p_a) log(r/a) / log(b/a)

which is not in the discrete space -- no polynomial reconstruction contains a
radial harmonic -- so the error here is a resolution. The refinement table must
fall, and at first order.

Run it:

    PYTHONPATH=.. python flow.py
    PYTHONPATH=.. python flow.py --dim 3 --family simplex
    PYTHONPATH=.. python flow.py --vtu dupuit.vtu
"""

import argparse
import os
import sys
import math

import mimetika_cxx as mk
import numpy as np

# geometry and data
A, B, HZ = 1.0, 10.0, 1.0
P_INNER, P_OUTER = 2.0, 1.0

FAMILIES = {
    "cartesian": mk.Family.cartesian,
    "simplex": mk.Family.simplex,
    "prism": mk.Family.prism,
}


def solvers(rtol):
    """The linear solvers, at the residual tolerance asked for.

    "riesz" is the Riesz map of H(div) x L^2 -- P is the Gram matrix of
    ||q||^2 = (K^-1 q,q) + ||div q||^2 and ||p||^2 = ||p||_L2^2 -- so its
    iteration count does not grow with the mesh. "direct" is a full
    factorization: exact, and the wrong instrument past a few hundred thousand
    unknowns.

    The tolerance is on the residual, not on the answer. An iterative solve
    cannot show the round-off floor a direct one leaves, so a patch test read
    through it is bounded by this number rather than by the method.
    """
    return {
        "riesz": mk.SolverOptions(
            method="gmres", preconditioner="riesz", rtol=rtol, max_iterations=2000
        ),
        # The same map, with the flux block inverted by an auxiliary space.
        # ADS splits an H(div) operator along the de Rham complex instead of
        # factorizing it, so it costs no fill and is linear in the unknowns.
        # It is a 3D construction and it wants one unknown per facet; a facet
        # carrying several moments (derham_bdm) reaches it through the
        # facet-constant subspace, as a two-level cycle.
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
    "derham_bdm": mk.FluxRealization.derham_bdm,
    "derham_rt": mk.FluxRealization.derham_rt,
    "stabilized_rt": mk.FluxRealization.stabilized_rt,
    # One flux per facet and no reconstruction: M is the diagonal primal-dual
    # star, M_ff = (|sigma*|/|sigma|) / (n.K n), which is the two-point flux
    # approximation. The same space as derham_rt and stabilized_rt -- one
    # unknown per facet -- and a different operator: it is strongly consistent
    # only where the mesh is K-orthogonal, so on a box it reproduces a linear
    # pressure to round-off and on a curved or polytopal one it does not.
    "diagonal_tpfa": mk.FluxRealization.diagonal_tpfa,
    # The per-cell selection between the two above, carried as eta in {0, 1}:
    # the stabilized product everywhere (eta = 1) and the diagonal star on the
    # cells the metric-degeneracy scan flags (eta = 0), which avoids a
    # reconstruction over a collapsed cell. The scan runs at exokal's default
    # threshold; --degeneracy-percent names another. The model reports the
    # selection as built, and that is the eta cell data written to the .vtu.
    "adaptive_rt": mk.FluxRealization.adaptive_rt,
}


def exact(x):
    r = math.hypot(x[0], x[1])
    return P_INNER + (P_OUTER - P_INNER) * math.log(r / A) / math.log(B / A)


def solve(nr, nt, dim, family, how, solver="riesz", rtol=DEFAULT_RTOL, degeneracy=None):
    """Build the annulus, impose the three conditions, solve, measure."""
    mesh = mk.annulus(nr, nt, dim, family, A, B, HZ)

    # Sort the boundary into three sets: a facet is on a symmetry plane, on the
    # inner radius, or on the outer one -- decided from its centroid, so the
    # same code works whatever the family meshed it with.
    rmid = math.sqrt(A * B)
    inner, outer, sealed = [], [], []
    for f in mk.boundary_facets(mesh, dim):
        x = mk.centroid(mesh, dim - 1, f)
        on_symmetry = (
            abs(x[0]) < 1e-8
            or abs(x[1]) < 1e-8
            or (dim == 3 and (abs(x[2]) < 1e-8 or abs(x[2] - HZ) < 1e-8))
        )
        if on_symmetry:
            sealed.append(f)
        elif math.hypot(x[0], x[1]) < rmid:
            inner.append(f)
        else:
            outer.append(f)

    model = mk.FlowModel(mesh, dim, 1.0, how)
    model.add_normal_flux(sealed)  # no flow through the symmetry planes
    model.add_pressure(inner, P_INNER)
    model.add_pressure(outer, P_OUTER)
    if degeneracy is not None:
        model.set_degeneracy_percent(degeneracy)
    model.solve(options=solvers(rtol)[solver])

    worst = rms = 0.0
    for e in range(model.n_cells):
        d = model.cell_pressure(e) - exact(mk.centroid(mesh, dim, e))
        worst = max(worst, abs(d))
        rms += d * d
    return model, mesh, worst, math.sqrt(rms / model.n_cells)


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
    ap.add_argument("--product", default="stabilized_rt", choices=sorted(PRODUCTS))
    ap.add_argument(
        "--nr", type=int, default=8, help="radial divisions of the coarse mesh"
    )
    ap.add_argument("--vtu", help="write the coarse solution to this .vtu")
    ap.add_argument("--solver", default="riesz", choices=sorted(SOLVER_NAMES))
    ap.add_argument("--rtol", type=float, default=DEFAULT_RTOL,
                    help="residual tolerance of the iterative solver")
    ap.add_argument("--degeneracy-percent", type=float, default=None,
                    help="adaptive_rt's scan threshold: cells below this "
                         "percentage of their node-star mean take the "
                         "two-point star instead of the stabilized product")
    args = ap.parse_args()

    family, how = FAMILIES[args.family], PRODUCTS[args.product]
    if args.degeneracy_percent is not None and args.product != "adaptive_rt":
        raise SystemExit("--degeneracy-percent is adaptive_rt's cell selection; "
                         "use --product adaptive_rt")
    require_three_dimensions(args.solver, args.dim)
    print(f"Dupuit annulus, {args.dim}D {args.family}, {mk.flux_realization_name(how)}")
    print(f"  a = {A}, b = {B},  p(a) = {P_INNER}, p(b) = {P_OUTER}\n")

    # ---- the profile on one mesh ------------------------------------------
    model, mesh, worst, rms = solve(
        args.nr, args.nr // 2, args.dim, family, how, args.solver,
        degeneracy=args.degeneracy_percent,
    )
    print(f"  {model.n_cells} cells, {model.n_dofs} dofs\n")
    print(f"  {'r':>8}  {'p (computed)':>14}  {'p (Dupuit)':>12}  {'error':>10}")
    rows = []
    for e in range(model.n_cells):
        x = mk.centroid(mesh, args.dim, e)
        rows.append((math.hypot(x[0], x[1]), model.cell_pressure(e), exact(x)))
    rows.sort()
    for r, got, want in rows[:: max(1, len(rows) // 10)]:
        print(f"  {r:8.4f}  {got:14.6f}  {want:12.6f}  {got - want:+10.2e}")

    # ---- the same solution as a field on the cells --------------------------
    if args.vtu and root:
        n = model.n_cells
        p = np.array([model.cell_pressure(e) for e in range(n)])
        q = np.array([exact(mk.centroid(mesh, args.dim, e)) for e in range(n)])
        fields = {"pressure": p, "dupuit": q, "error": p - q}
        # the blend as built -- with the forced zeros -- next to the solution
        # it produced, so a wrong cell is attributable
        if args.product == "adaptive_rt":
            fields["eta"] = model.eta
        mk.write_vtu(mesh, args.vtu, fields)
        print(f"\n  wrote {args.vtu}")

    # ---- and what refinement does to it ------------------------------------
    print(f"\n  {'cells':>8}  {'max error':>11}  {'rms error':>11}  {'rate':>6}")
    previous = None
    for nr in (args.nr, 2 * args.nr, 4 * args.nr):
        m, _, worst, rms = solve(nr, nr // 2, args.dim, family, how, args.solver, args.rtol,
                                 degeneracy=args.degeneracy_percent)
        rate = "" if previous is None else f"{math.log2(previous / rms):6.2f}"
        print(f"  {m.n_cells:8d}  {worst:11.3e}  {rms:11.3e}  {rate:>6}")
        previous = rms


if __name__ == "__main__":
    main()
