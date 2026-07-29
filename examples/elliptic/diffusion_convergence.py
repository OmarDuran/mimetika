r"""Convergence of the mimetic mixed Poisson solver.

Solves, on the unit cube with Dirichlet data,

    div F = f ,    F = -K grad p ,    p = p_D  on the boundary,

for the manufactured solution

    p(x, y, z) = sin(pi x) sin(pi y) sin(pi z) ,   f = -div(K grad p).

Errors are measured against the *interpolant* of the exact solution (the
natural mimetic error): the element means for the pressure, and the average
normal fluxes for the flux, in the mesh-dependent ``M``-norm.

Two mesh families are compared:

* **hexahedra** -- a genuine polytopal case, where the inner product carries an
  active stabilization;
* **tetrahedra** -- the simplicial case, where the stabilization vanishes and
  the ``rt0`` scheme coincides with RT0 mixed FE.

Run with::

    python examples/elliptic/diffusion_convergence.py
"""

from __future__ import annotations

import numpy as np

from common import (  # noqa: E402  (path bootstrap lives in common)
    cell_l2_norm,
    energy_norm,
    observed_rates,
    print_table,
)

from mimetika.assembly.mixed import MixedPoisson  # noqa: E402
from mimetika.mesh import structured_box, structured_tets  # noqa: E402

K = np.diag([1.0, 1.0, 1.0])
PI = np.pi


def pressure(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    return np.prod(np.sin(PI * x), axis=1)


def flux(x: np.ndarray) -> np.ndarray:
    """``F = -K grad p``."""
    x = np.atleast_2d(x)
    s, c = np.sin(PI * x), np.cos(PI * x)
    grad = PI * np.column_stack(
        [
            c[:, 0] * s[:, 1] * s[:, 2],
            s[:, 0] * c[:, 1] * s[:, 2],
            s[:, 0] * s[:, 1] * c[:, 2],
        ]
    )
    return -grad @ K.T


def source(x: np.ndarray) -> np.ndarray:
    """``f = div F = -div(K grad p)``; for ``K = I`` this is ``3 pi^2 p``."""
    return 3.0 * PI**2 * np.trace(K) / 3.0 * pressure(x)


def run(family: str, resolutions, basis: str):
    make = structured_box if family == "hex" else structured_tets
    rows = []
    for n in resolutions:
        mesh = make(n, n, n)
        problem = MixedPoisson(mesh, K=K, basis=basis)
        sol = problem.solve(source=source, dirichlet=pressure)

        p_err = sol["pressure"] - problem.interpolate_pressure(pressure)
        f_err = sol["flux"] - problem.interpolate_flux(flux)

        rows.append(
            {
                "h": 1.0 / n,
                "cells": mesh.num_cells(3),
                "ndof": problem.n_flux + problem.n_pressure,
                "stab": problem.inner.stabilization_dim(0),
                "err_p": cell_l2_norm(mesh.geometry.measure(3), p_err),
                "err_F": energy_norm(problem.inner.assemble(), f_err),
            }
        )

    h = [r["h"] for r in rows]
    for key in ("err_p", "err_F"):
        for row, rate in zip(rows, observed_rates(h, [r[key] for r in rows])):
            row[key + "_rate"] = rate
    return rows


COLUMNS = [
    ("h", "h"),
    ("cells", "cells"),
    ("ndof", "ndof"),
    ("stab", "stab dim"),
    ("err_p", "err p"),
    ("err_p_rate", "rate"),
    ("err_F", "err F"),
    ("err_F_rate", "rate"),
]


def main() -> None:
    print(__doc__.split("Run with")[0].strip())
    for family, resolutions in (("hex", (2, 4, 8, 16)), ("tet", (2, 4, 8))):
        for basis in ("const", "rt0"):
            rows = run(family, resolutions, basis)
            print_table(
                f"mixed Poisson  |  {family} mesh  |  basis = {basis}", rows, COLUMNS
            )
    print(
        "\nExpected: first-order convergence in both variables for the lowest-order"
        "\nscheme; the pressure often superconverges. 'stab dim' is the dimension of"
        "\nthe local stabilization space (0 => the scheme is stabilization-free)."
    )


if __name__ == "__main__":
    main()
