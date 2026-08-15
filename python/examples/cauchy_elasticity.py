#!/usr/bin/env python3
"""Cauchy elasticity on a quarter annulus: Lame's thick-walled tube.

A tube from a to b under internal pressure p_a and external p_b, rollers on the
symmetry planes. With

    A = (a^2 p_a - b^2 p_b) / (b^2 - a^2)
    B = (p_a - p_b) a^2 b^2 / (b^2 - a^2)

the closed form is

    sigma_rr(r) = A - B/r^2
    u_r(r)      = A r / (2(lam + mu))  +  B / (2 mu r)

Neither term is polynomial, so neither is in the reconstruction: as with the
Dupuit profile, the error is a resolution and the refinement table is the point.

Note the boundary datum. The exact stress at r = a is NOT -p_a I, but its
TRACTION is -p_a n, and a uniform -p_a I delivers exactly that -- so the
condition is a constant tensor even though the solution is curved.

Run it:

    PYTHONPATH=.. python cauchy_elasticity.py
    PYTHONPATH=.. python cauchy_elasticity.py --dim 3 --product stabilized_afw
"""

import argparse
import math

import mimetika_cxx as mk

# geometry and data
A_IN, B_OUT, HZ = 1.0, 4.0, 1.0
P_INNER, P_OUTER = 1.0, 0.25
MU, LAM = 1.0, 1.0

FAMILIES = {
    "cartesian": mk.Family.cartesian,
    "simplex": mk.Family.simplex,
    "prism": mk.Family.prism,
}
PRODUCTS = {
    "derham_afw": mk.StressRealization.derham_afw,
    "stabilized_afw": mk.StressRealization.stabilized_afw,
}


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


def solve(nr, nt, dim, family, how, mat):
    """Build the annulus, impose the three conditions, solve, measure."""
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

    model = mk.CauchyElasticityModel(mesh, dim, mat, how)
    model.add_traction(inner, isotropic(-P_INNER))
    model.add_traction(outer, isotropic(-P_OUTER))
    model.add_free_slip(symmetry)  # rollers: no normal displacement, no shear
    model.solve()

    ex = Lame(mat)
    worst = rms = 0.0
    for e in range(model.n_cells):
        x = mk.centroid(mesh, dim, e)
        r = math.hypot(x[0], x[1])
        # the RADIAL displacement, projected out of the cell's vector unknown
        u_r = (model.displacement(e, 0) * x[0] + model.displacement(e, 1) * x[1]) / r
        d = u_r - ex.u_r(r)
        worst = max(worst, abs(d))
        rms += d * d
    return model, mesh, ex, worst, math.sqrt(rms / model.n_cells)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dim", type=int, default=2, choices=(2, 3))
    ap.add_argument("--family", default="simplex", choices=sorted(FAMILIES))
    ap.add_argument("--product", default="derham_afw", choices=sorted(PRODUCTS))
    ap.add_argument("--nr", type=int, default=6, help="radial divisions of the coarse mesh")
    ap.add_argument("--nu", type=float, default=None, help="Poisson ratio (default lam = mu = 1)")
    args = ap.parse_args()

    if args.nu is None:
        mat = mk.ElasticMaterial(MU, LAM)
    else:
        mat = mk.ElasticMaterial(MU, 2.0 * MU * args.nu / (1.0 - 2.0 * args.nu))

    family, how = FAMILIES[args.family], PRODUCTS[args.product]
    print(f"Lame tube, {args.dim}D {args.family}, {mk.stress_realization_name(how)}")
    print(f"  a = {A_IN}, b = {B_OUT},  p(a) = {P_INNER}, p(b) = {P_OUTER}")
    print(f"  mu = {mat.shear}, lam = {mat.lame}  (nu = {mat.poisson():.4f})\n")

    # ---- the profile on one mesh ------------------------------------------
    model, mesh, ex, worst, rms = solve(args.nr, args.nr // 2, args.dim, family, how, mat)
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

    # ---- and what refinement does to it ------------------------------------
    print(f"\n  {'cells':>8}  {'max error':>11}  {'rms error':>11}  {'rate':>6}")
    previous = None
    for nr in (args.nr, 2 * args.nr, 4 * args.nr):
        m, _, _, worst, rms = solve(nr, nr // 2, args.dim, family, how, mat)
        rate = "" if previous is None else f"{math.log2(previous / rms):6.2f}"
        print(f"  {m.n_cells:8d}  {worst:11.3e}  {rms:11.3e}  {rate:>6}")
        previous = rms


if __name__ == "__main__":
    main()
