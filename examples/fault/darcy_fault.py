r"""Darcy flow on the polyhedral fault mesh.

Solves the mixed (velocity--pressure) form of Darcy's law

    div u = f ,     u = -(K/mu) grad p ,     p = p_D  on the whole boundary,

on ``examples/meshes/fault_mesh.vtu`` -- a genuinely polytopal grid of 22 056
cells with 6 to 24 planar faces each, the kind of mesh produced by cutting a
corner-point grid with a fault.  This is exactly the setting mimetic methods
are built for: no reference element, no shape functions, just facet fluxes and
cell pressures.

Unknowns: one normal flux per facet, one pressure per cell::

    [  M   -B^T ] [ u ]   [ -g_D ]
    [ -B    0   ] [ p ] = [  -b  ]

``M`` is the mimetic flux inner product; ``B`` is the purely topological
discrete divergence (signed incidence times facet measures), so mass is
conserved on every cell to round-off regardless of cell shape.

**Boundary condition.**  A non-trivial Dirichlet pressure is prescribed on the
entire boundary: a regional gradient along ``x`` plus a lateral undulation and
a depth trend,

    p_D(x, y, z) = 1 - x + 0.25 sin(2 pi y) + 0.4 (1 - z / 0.4) ,

so the flow is genuinely three-dimensional rather than a one-dimensional
pressure drop.

Run with::

    python examples/fault/darcy_fault.py                 # full mesh, direct solve
    python examples/fault/darcy_fault.py --method minres # iterative
    python examples/fault/darcy_fault.py --vtk out.vtk   # write results
"""

from __future__ import annotations

import argparse

import numpy as np

from common import (  # noqa: E402  (path bootstrap lives in common)
    add_common_args,
    announce_backend,
    load_mesh,
    report_mesh,
    step,
    summarise,
)

from mimetika.assembly.mixed import MixedPoisson, discrete_divergence  # noqa: E402

# Permeability over viscosity, K/mu.  Anisotropic: layered media are far more
# permeable along bedding (x, y) than across it (z).
MOBILITY = np.array([[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 0.1]])

Z_TOP = 0.4


def pressure_bc(x: np.ndarray) -> np.ndarray:
    """Non-trivial Dirichlet pressure on the boundary."""
    x = np.atleast_2d(x)
    return (
        1.0
        - x[:, 0]
        + 0.25 * np.sin(2.0 * np.pi * x[:, 1])
        + 0.4 * (1.0 - x[:, 2] / Z_TOP)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("Run with")[0])
    add_common_args(parser)
    parser.add_argument(
        "--basis",
        choices=["const", "rt0"],
        default="const",
        help="flux reconstruction space",
    )
    args = parser.parse_args()

    print(__doc__.split("Run with")[0].strip())
    print("\n== mesh ==")
    mesh = load_mesh()
    report_mesh(mesh)

    print("\n== solve ==")
    backend = announce_backend(args.backend)
    problem = MixedPoisson(mesh, K=MOBILITY, basis=args.basis)
    print(
        f"  unknowns: {problem.n_flux} fluxes + {problem.n_pressure} pressures"
        f" = {problem.n_flux + problem.n_pressure}"
    )
    print(f"  stabilization dim on cell 0: {problem.inner.stabilization_dim(0)}")

    with step("assembling flux inner product"):
        M = problem.inner_product()
    with step("assembling saddle-point system"):
        A, rhs = problem.assemble(source=None, dirichlet=pressure_bc)
    print(f"    system {A.shape[0]} x {A.shape[1]}, nnz = {A.nnz}")

    with step(f"solving ({args.method})"):
        sol = problem.solve(
            source=None,
            dirichlet=pressure_bc,
            backend=backend,
            method=args.method,
            rtol=args.rtol,
            verbose=True,
        )

    flux, p = sol["flux"], sol["pressure"]

    print("\n== solution ==")
    summarise("pressure (cell)", p)
    summarise("normal flux (facet)", flux)

    print("\n== verification ==")
    # 1) local mass conservation: div_h u = 0 on every cell, to round-off
    B = discrete_divergence(mesh)
    residual = B @ flux
    vol = mesh.geometry.measure(3)
    print(
        f"    max |div_h u| per cell        {np.abs(residual / vol).max():.3e}"
        "   (source-free => exactly zero)"
    )
    # 2) the discrete maximum principle: no interior pressure outside the data range
    lo, hi = _boundary_pressure_range(mesh)
    print(f"    boundary data range           [{lo:.4f}, {hi:.4f}]")
    print(f"    computed pressure range       [{p.min():.4f}, {p.max():.4f}]")
    print(
        "    within data range             "
        f"{'yes' if p.min() >= lo - 1e-9 and p.max() <= hi + 1e-9 else 'NO (see note)'}"
    )
    # 3) global balance: total flux through the boundary must vanish
    print(f"    net flux through boundary     {residual.sum():+.3e}")

    # 4) patch test on this very mesh: a linear pressure must be reproduced
    #    exactly, because the local inner products satisfy M N = R
    grad = np.array([0.7, -1.1, 0.45])
    lin_p = lambda z: 0.3 + np.atleast_2d(z) @ grad          # noqa: E731
    lin_u = lambda z: np.broadcast_to(                        # noqa: E731
        -MOBILITY @ grad, (len(np.atleast_2d(z)), 3)
    )
    with step("patch test (linear pressure)"):
        exact = problem.solve(
            source=None, dirichlet=lin_p, backend=backend,
            method=args.method, rtol=args.rtol,
        )
    dp = np.abs(exact["pressure"] - problem.interpolate_pressure(lin_p)).max()
    du = np.abs(exact["flux"] - problem.interpolate_flux(lin_u)).max()
    print(f"    pressure error                {dp:.3e}")
    print(f"    flux error                    {du:.3e}")
    print(
        "    (limited only by the solver tolerance => the scheme is exact for"
        "\n     linear fields on this polytopal mesh)"
        if args.method != "direct"
        else "    (exact to round-off => the scheme is exact for linear fields here)"
    )

    # Facet fluxes are not plottable on their own: rebuild a cell-centred
    # velocity vector, exactly (the reconstruction reproduces constant fields).
    print("\n== postprocessing ==")
    from mimetika.postprocess import export_vtu, reconstruct_flux

    with step("reconstructing cell velocity"):
        velocity = reconstruct_flux(mesh, flux)
    speed = np.linalg.norm(velocity, axis=1)
    summarise("velocity magnitude", speed)

    if args.vtu:
        with step(f"writing {args.vtu}"):
            export_vtu(
                args.vtu,
                mesh,
                cell_data={
                    "pressure": p,
                    "velocity": velocity,       # vector -> Glyph in ParaView
                    "speed": speed,
                    "cell_volume": vol,
                },
            )
        print(f"    ParaView: colour by 'pressure', then Glyph by 'velocity'")


def _boundary_pressure_range(mesh):
    from mimetika.assembly.mixed import boundary_facets

    vals = []
    for f in boundary_facets(mesh):
        qp, _ = mesh.geometry.quadrature(2, f)
        vals.append(pressure_bc(qp))
    v = np.concatenate(vals)
    return float(v.min()), float(v.max())


if __name__ == "__main__":
    main()
