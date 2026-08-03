r"""Convergence of the mimetic mixed elasticity solver (Mimetic-AFW).

Solves on the unit cube, in Hellinger--Reissner form with weakly imposed
symmetry,

    div sigma = f ,   sigma = 2 mu eps(u) + lambda tr(eps(u)) I ,   u = u_D  on
    the boundary,

for the manufactured displacement

    u = ( sin(pi x) + 0.3 sin(pi y),
          sin(pi y) + 0.2 sin(pi z),
          sin(pi z) + 0.25 sin(pi x) )

whose body force is the Navier operator
``f = div sigma = mu Laplace(u) + (mu + lambda) grad(div u)``.  The off-diagonal
terms give a genuinely non-symmetric strain, so the rates are not flattered by
grid alignment.

The method solves for **all three** unknowns of the weakly-symmetric
formulation, and all three are reported: the stress ``sigma``, the displacement
``u``, and the rotation multiplier ``s = skw(grad u)`` that enforces the
symmetry of ``sigma`` weakly.  Errors are measured against the interpolant of
the exact solution: element means for ``u`` and ``s``, traction moments for
``sigma`` (in the mesh-dependent ``M``-norm).

Two mesh families are compared:

* **hexahedra** -- a polytopal case, ``stab dim = 18`` per cell;
* **tetrahedra** -- simplicial, ``stab dim = 0``: the stabilization vanishes and
  the scheme reduces to the AFW (BDM_1-based) mixed element.

The last table repeats the hex case at ``lambda = 1e4`` to show that the rates
survive the near-incompressible limit.

Run with::

    python examples/elliptic/elasticity_convergence.py
"""

from __future__ import annotations

import numpy as np

from common import (  # noqa: E402  (path bootstrap lives in common)
    cell_l2_norm,
    energy_norm,
    observed_rates,
    print_table,
)

# the *classic* three-field AFW formulation, kept as the reference example;
# the standard four-field split lives in mimetika.assembly.four_field
from mimetika.assembly.mixed import MixedElasticity  # noqa: E402
from mimetika.mesh import structured_box, structured_tets  # noqa: E402

PI = np.pi
# u_i = sin(pi x_i) + COUPLING[i] * sin(pi x_{i+1 mod 3})
COUPLING = np.array([0.3, 0.2, 0.25])


def displacement(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    s = np.sin(PI * x)
    return s + COUPLING * s[:, [1, 2, 0]]


def _grad_u(x: np.ndarray) -> np.ndarray:
    """``(nq, 3, 3)`` with ``[q, i, j] = du_i / dx_j``."""
    x = np.atleast_2d(x)
    c = PI * np.cos(PI * x)
    G = np.zeros((len(x), 3, 3))
    for i in range(3):
        G[:, i, i] = c[:, i]
        G[:, i, (i + 1) % 3] = COUPLING[i] * c[:, (i + 1) % 3]
    return G


def grad_displacement(x: np.ndarray) -> np.ndarray:
    """``(nq, 3, 3)`` with ``[q, i, j] = du_i / dx_j`` (public: used for s)."""
    return _grad_u(x)


def strain(x: np.ndarray) -> np.ndarray:
    G = _grad_u(x)
    return 0.5 * (G + np.swapaxes(G, 1, 2))


def make_data(mu: float, lam: float):
    """Return ``(stress, body_force)`` consistent with the displacement."""

    def stress(x):
        eps = strain(x)
        tr = np.einsum("qii->q", eps)
        return 2 * mu * eps + lam * tr[:, None, None] * np.eye(3)

    def body_force(x):
        # Laplace(u) = -pi^2 u   and   grad(div u) = -pi^2 (sin pi x_i)_i
        x = np.atleast_2d(x)
        lap = -(PI**2) * displacement(x)
        grad_div = -(PI**2) * np.sin(PI * x)
        return mu * lap + (mu + lam) * grad_div

    return stress, body_force


def run(family: str, resolutions, mu: float, lam: float):
    make = structured_box if family == "hex" else structured_tets
    stress, body_force = make_data(mu, lam)
    rows = []
    for n in resolutions:
        mesh = make(n, n, n)
        problem = MixedElasticity(mesh, mu=mu, lam=lam)
        sol = problem.solve(body_force=body_force, dirichlet=displacement)

        u_err = sol["displacement"] - problem.interpolate_displacement(displacement)
        s_err = sol["stress"] - problem.interpolate_stress(stress)
        r_err = sol["rotation"] - problem.interpolate_rotation(grad_displacement)

        rows.append(
            {
                "h": 1.0 / n,
                "cells": mesh.num_cells(3),
                "ndof": problem.n_stress
                + (problem.d + problem.n_skew) * problem.n_cells,
                "stab": problem.inner.stabilization_dim(0),
                "err_u": cell_l2_norm(mesh.geometry.measure(3), u_err, problem.d),
                "err_s": energy_norm(problem.inner.assemble(), s_err),
                "err_r": cell_l2_norm(
                    mesh.geometry.measure(3), r_err, problem.n_skew
                ),
            }
        )

    h = [r["h"] for r in rows]
    for key in ("err_u", "err_s", "err_r"):
        for row, rate in zip(rows, observed_rates(h, [r[key] for r in rows])):
            row[key + "_rate"] = rate
    return rows


COLUMNS = [
    ("h", "h"),
    ("cells", "cells"),
    ("ndof", "ndof"),
    ("stab", "stab dim"),
    ("err_u", "err u"),
    ("err_u_rate", "rate"),
    ("err_s", "err sigma"),
    ("err_s_rate", "rate"),
    ("err_r", "err rot"),
    ("err_r_rate", "rate"),
]

CASES = [
    ("hex", (1, 2, 4, 8), 1.0, 1.0, "mu = 1, lambda = 1"),
    ("tet", (1, 2, 4), 1.0, 1.0, "mu = 1, lambda = 1"),
    ("hex", (1, 2, 4, 8), 1.0, 1e4, "mu = 1, lambda = 1e4 (near incompressible)"),
]


def main() -> None:
    print(__doc__.split("Run with")[0].strip())
    for family, resolutions, mu, lam, label in CASES:
        rows = run(family, resolutions, mu, lam)
        print_table(
            f"mixed elasticity  |  {family} mesh  |  {label}", rows, COLUMNS
        )
    print(
        "\nTheory guarantees first order in all three variables, uniformly in lambda"
        "\n(Beirao, Prop. 3.2-3.3).  On these structured meshes the observed rates"
        "\nrun closer to two -- superconvergence; the coarsest one or two rows are"
        "\npreasymptotic.  Note the lambda = 1e4 stress errors are large in absolute"
        "\nterms only because sigma itself scales with lambda: the *rate* is"
        "\nunaffected, which is the point of the mixed formulation."
        "\n'stab dim' is the dimension of the local stabilization space: 0 on the"
        "\ntetrahedral mesh, where the method coincides with the AFW mixed element."
    )


if __name__ == "__main__":
    main()
