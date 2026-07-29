r"""Convergence of the mimetic mixed Poisson solver.

Solves on the unit cube, with Dirichlet data,

    div F = f ,    F = -K grad p ,    p = p_D  on the boundary,

for the manufactured solution ``p = sin(pi x) sin(pi y) sin(pi z)``.

Errors are measured against the *interpolant* of the exact solution (the
natural mimetic error): element means for the pressure, average normal fluxes
for the flux, in the mesh-dependent ``M``-norm.

Four cases are run:

* **hex / isotropic** -- a grid-aligned, isotropic problem.  The symmetry makes
  the leading error terms cancel and the flux *superconverges* far past the
  guaranteed rate; this is a property of the special configuration, not of the
  method, which is why the anisotropic case is shown next to it.
* **hex / anisotropic** -- full tensor ``K`` with off-diagonal coupling breaks
  that symmetry and exposes the genuine rate.
* **tet / anisotropic**, with the ``const`` and ``rt0`` reconstruction spaces.
  On simplices ``rt0`` is stabilization-free (``stab dim = 0``) and coincides
  with RT0 mixed FE.

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

PI = np.pi
K_ISO = np.eye(3)
K_ANISO = np.array([[2.0, 0.6, 0.2], [0.6, 1.5, 0.3], [0.2, 0.3, 1.0]])


def pressure(x: np.ndarray) -> np.ndarray:
    return np.prod(np.sin(PI * np.atleast_2d(x)), axis=1)


def _grad(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    s, c = np.sin(PI * x), np.cos(PI * x)
    return PI * np.column_stack(
        [
            c[:, 0] * s[:, 1] * s[:, 2],
            s[:, 0] * c[:, 1] * s[:, 2],
            s[:, 0] * s[:, 1] * c[:, 2],
        ]
    )


def _hessian(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    s, c = np.sin(PI * x), np.cos(PI * x)
    p = s[:, 0] * s[:, 1] * s[:, 2]
    H = np.zeros((len(x), 3, 3))
    for i in range(3):
        H[:, i, i] = -(PI**2) * p
    H[:, 0, 1] = H[:, 1, 0] = PI**2 * c[:, 0] * c[:, 1] * s[:, 2]
    H[:, 0, 2] = H[:, 2, 0] = PI**2 * c[:, 0] * s[:, 1] * c[:, 2]
    H[:, 1, 2] = H[:, 2, 1] = PI**2 * s[:, 0] * c[:, 1] * c[:, 2]
    return H


def make_data(K: np.ndarray):
    """Return ``(flux, source)`` consistent with the manufactured pressure."""

    def flux(x):
        return -_grad(x) @ K.T

    def source(x):  # f = div F = -div(K grad p) = -K : Hess(p)
        return -np.einsum("ij,qij->q", K, _hessian(x))

    return flux, source


def run(family: str, resolutions, basis: str, K: np.ndarray):
    make = structured_box if family == "hex" else structured_tets
    flux, source = make_data(K)
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

CASES = [
    ("hex", (2, 4, 8, 16), "const", K_ISO, "isotropic K (grid-aligned)"),
    ("hex", (2, 4, 8, 16), "const", K_ANISO, "anisotropic K"),
    ("tet", (2, 4, 8), "const", K_ANISO, "anisotropic K"),
    ("tet", (2, 4, 8), "rt0", K_ANISO, "anisotropic K"),
]


def main() -> None:
    print(__doc__.split("Run with")[0].strip())
    for family, resolutions, basis, K, label in CASES:
        rows = run(family, resolutions, basis, K)
        print_table(
            f"mixed Poisson  |  {family} mesh  |  basis = {basis}  |  {label}",
            rows,
            COLUMNS,
        )
    print(
        "\nThe lowest-order scheme guarantees first order; the cell pressure"
        "\nsuperconverges to second order.  The isotropic grid-aligned flux rate is"
        "\nan artifact of that configuration's symmetry -- compare the anisotropic"
        "\nrow, which shows the genuine behaviour.  'stab dim' is the dimension of"
        "\nthe local stabilization space (0 => stabilization-free)."
    )


if __name__ == "__main__":
    main()
