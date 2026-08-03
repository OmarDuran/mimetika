r"""Solver-efficiency benchmark: exact stress condensation on the lumped forms.

Five formulations of the same elasticity problem on a structured box, one
table.  The point being measured is structural, not incidental:

* the **AFW** inner product couples the facets of every cell, so the full
  saddle system must be factorised -- three- or four-field alike;
* the **three-field lumped** assembly folds the volumetric rank-one term back
  into ``M``, re-coupling the facets: smaller blocks (``d`` vs ``d^2`` per
  facet) buy a modest factor, and that is the ceiling;
* the **four-field lumped** arrangements (multiplier and kinematic-rotation)
  keep ``M`` exactly diagonal in the assembled system, so the stress -- about
  three quarters of the unknowns -- is eliminated by a diagonal scaling
  (:mod:`mimetika.solver.condense`) and only a cell-centred system of
  ``1 + d + d(d-1)/2`` unknowns per cell is factorised.

The two condensed rows close the rotation differently and condense
identically -- which is the point: the speedup comes from the four-field
diagonality, not from the rotation closure.  Their roles differ.  The
**multiplier** form (``lumped 4-field``) enforces ``skw sigma = 0`` exactly
and is congruent to the classic three-field system, which makes it the
*verification anchor*; but its rotation multiplier carries an O(h^2)-decaying
inf-sup -- harmless to ``sigma`` and ``u``, yet the multiplier conditioning
grows like ``h^-2``, which iterative solvers feel.  The **kinematic-rotation**
form (``two-point 4-field``) has no zero block and no inf-sup question in any
field, and is the *recommended* scheme for production and for iterative
solving at scale.

Timings are sparse-LU factor + solve (best of ``--repeats``); ``LU fill`` is
the factor storage, which is also the memory story.  Assembly is excluded: it
is comparable across formulations and amortised over time steps.

Run with ``python -m benchmarks.elasticity.condensation`` (default 5x5x5).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import scipy.sparse.linalg as spla

from mimetika.assembly.four_field import FourFieldElasticity
from mimetika.assembly.kinematic import TwoPointFourField
from mimetika.assembly.mixed import MixedElasticity
from mimetika.mesh import structured_box
from mimetika.operators.lumped import LumpedDeviatoricStress
from mimetika.solver.condense import eliminate_leading_diagonal

MU, LAM = 1.3, 2.7
U_B = np.array([[0.5, -0.3, 0.2], [0.15, 0.4, -0.25], [-0.1, 0.35, 0.6]])


def displacement(x):
    return np.atleast_2d(x) @ U_B.T


def _time_full(S, rhs, repeats):
    best = np.inf
    for _ in range(repeats):
        start = time.perf_counter()
        lu = spla.splu(S.tocsc())
        x = lu.solve(rhs)
        best = min(best, time.perf_counter() - start)
    return best, lu.L.nnz + lu.U.nnz, x, None


def _time_condensed(S, rhs, n0, repeats):
    best = np.inf
    for _ in range(repeats):
        start = time.perf_counter()
        reduced, reduced_rhs, recover = eliminate_leading_diagonal(S, rhs, n0)
        lu = spla.splu(reduced.tocsc())
        x = recover(lu.solve(reduced_rhs))
        best = min(best, time.perf_counter() - start)
    return best, lu.L.nnz + lu.U.nnz, x, reduced.shape[0]


def measure(n: int = 5, repeats: int = 3):
    """One row per formulation: ``(name, dofs, reduced, nnz, fill, time, x)``."""
    mesh = structured_box(n, n, n)
    lumped = lambda: LumpedDeviatoricStress(mesh, mu=MU, lam=LAM)  # noqa: E731
    cases = [
        ("AFW 3-field (full)", MixedElasticity(mesh, MU, LAM), False),
        ("lumped 3-field (full)", MixedElasticity(mesh, inner=lumped()), False),
        ("AFW 4-field (full)", FourFieldElasticity(mesh, MU, LAM), False),
        (
            "lumped 4-field (condensed)",
            FourFieldElasticity(mesh, inner=lumped()),
            True,
        ),
        ("two-point 4-field (condensed)", TwoPointFourField(mesh, MU, LAM), True),
    ]
    rows = []
    for name, problem, condensed in cases:
        S, rhs = problem.assemble(dirichlet=displacement)
        if condensed:
            t, fill, x, reduced = _time_condensed(S, rhs, problem.n_stress, repeats)
        else:
            t, fill, x, reduced = _time_full(S, rhs, repeats)
        rows.append(
            {
                "name": name,
                "dofs": S.shape[0],
                "reduced": reduced,
                "nnz": S.nnz,
                "fill": fill,
                "time": t,
                "solution": x,
                "problem": problem,
            }
        )
    return rows


def print_table(rows) -> None:
    reference = {
        "3-field": next(r["time"] for r in rows if r["name"].startswith("AFW 3")),
        "4-field": next(r["time"] for r in rows if r["name"].startswith("AFW 4")),
    }
    print(
        f"{'formulation':<30} {'dofs':>7} {'reduced':>8} {'nnz(A)':>9} "
        f"{'LU fill':>10} {'time [s]':>9} {'speedup':>8}"
    )
    for r in rows:
        ref = reference["3-field" if "3-field" in r["name"] else "4-field"]
        reduced = "-" if r["reduced"] is None else str(r["reduced"])
        print(
            f"{r['name']:<30} {r['dofs']:>7} {reduced:>8} {r['nnz']:>9} "
            f"{r['fill']:>10} {r['time']:>9.4f} {ref / r['time']:>7.1f}x"
        )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=5, help="cells per direction")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    mesh_cells = args.n**3
    print(f"3D box {args.n}x{args.n}x{args.n} ({mesh_cells} cells)")
    print_table(measure(args.n, args.repeats))


if __name__ == "__main__":
    main()
