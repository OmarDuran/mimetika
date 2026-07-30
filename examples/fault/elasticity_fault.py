r"""Linear elasticity on the polyhedral fault mesh (Mimetic-AFW).

Solves the Hellinger--Reissner mixed form with **weakly imposed symmetry**

    div sigma = f ,   sigma = 2 mu eps(u) + lambda tr(eps(u)) I ,   u = u_D  on
    the boundary,

on ``examples/meshes/fault_mesh.vtu`` -- a polytopal grid whose cells carry 6 to
24 planar faces.  Three fields are solved simultaneously::

    [  M    D^T   A^T ] [ sigma ]   [ g_D ]    sigma : traction moments per facet
    [  D     0     0  ] [ u     ] = [  f  ]    u     : displacement per cell
    [  A     0     0  ] [ s     ]   [  0  ]    s     : rotation per cell

The third block equation is ``as_h(sigma) = 0``: the symmetry of the stress is
enforced weakly, and ``s`` is its Lagrange multiplier (discretely, the rotation
``skw(grad u)``).

**Boundary condition.**  A non-trivial Dirichlet displacement is prescribed on
the whole boundary -- a depth-dependent shear in two directions plus
compaction toward the top surface, so the response is fully three-dimensional:

    u_D = (  A_s (z/Z) sin(pi y),
             A_s (z/Z) sin(pi x) / 2,
            -A_c (1 - z/Z)          ).

**Size.**  The full mesh gives ~690 000 stress unknowns and ~89 M nonzeros,
which is beyond a direct factorisation; the example therefore runs on a
subregion by default.  Use ``--full`` (with ``--method minres``) for the whole
mesh.

Run with::

    python examples/fault/elasticity_fault.py                    # subregion
    python examples/fault/elasticity_fault.py --box 0.3 0.3 0 0.6 0.6 0.4
    python examples/fault/elasticity_fault.py --full --method minres
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

from mimetika.assembly.mixed import MixedElasticity  # noqa: E402

MU, LAM = 1.0, 1.0  # Lame parameters (normalised)
Z_TOP = 0.4
SHEAR, COMPACTION = 0.02, 0.01

# A modest central block; ~1000 cells, which solves directly in seconds.
DEFAULT_BOX = ([0.35, 0.35, 0.0], [0.55, 0.55, 0.4])


def displacement_bc(x: np.ndarray) -> np.ndarray:
    """Non-trivial Dirichlet displacement on the boundary."""
    x = np.atleast_2d(x)
    zeta = x[:, 2] / Z_TOP
    return np.column_stack(
        [
            SHEAR * zeta * np.sin(np.pi * x[:, 1]),
            0.5 * SHEAR * zeta * np.sin(np.pi * x[:, 0]),
            -COMPACTION * (1.0 - zeta),
        ]
    )


# A linear displacement, used for the patch test below.
LIN_A = np.array([0.013, -0.007, 0.004])
LIN_B = np.array([[0.02, -0.01, 0.006], [0.005, 0.015, -0.008], [-0.003, 0.011, 0.02]])


def linear_displacement(x):
    return LIN_A + np.atleast_2d(x) @ LIN_B.T


def linear_grad(x):
    return np.broadcast_to(LIN_B, (len(np.atleast_2d(x)), 3, 3))


def linear_stress(x):
    eps = 0.5 * (LIN_B + LIN_B.T)
    sig = 2 * MU * eps + LAM * np.trace(eps) * np.eye(3)
    return np.broadcast_to(sig, (len(np.atleast_2d(x)), 3, 3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("Run with")[0])
    add_common_args(parser, default_vtu="elasticity_fault.vtu")
    parser.add_argument(
        "--full", action="store_true", help="use the whole mesh (large; use --method minres)"
    )
    parser.add_argument(
        "--box",
        type=float,
        nargs=6,
        metavar=("XLO", "YLO", "ZLO", "XHI", "YHI", "ZHI"),
        default=None,
        help="centroid bounding box of the subregion to solve on",
    )
    parser.add_argument("--mu", type=float, default=MU)
    parser.add_argument("--lam", type=float, default=LAM)
    args = parser.parse_args()

    if args.full:
        box = None
    elif args.box:
        box = (args.box[:3], args.box[3:])
    else:
        box = DEFAULT_BOX

    print(__doc__.split("Run with")[0].strip())
    print("\n== mesh ==")
    if box is None:
        print("  using the FULL mesh -- a direct solve will not fit; use --method minres")
    else:
        print(f"  subregion: centroids in {list(box[0])} .. {list(box[1])}")
    mesh = load_mesh(box=box)
    report_mesh(mesh)

    print("\n== solve ==")
    backend = announce_backend(args.backend)
    problem = MixedElasticity(mesh, mu=args.mu, lam=args.lam)
    n_total = problem.n_stress + (problem.d + problem.n_skew) * problem.n_cells
    print(
        f"  unknowns: {problem.n_stress} stress + {problem.d * problem.n_cells}"
        f" displacement + {problem.n_skew * problem.n_cells} rotation = {n_total}"
    )
    print(f"  stabilization dim on cell 0: {problem.inner.stabilization_dim(0)}")

    with step("assembling M, div_h, as_h"):
        M, D, A = problem.assemble_operators()
    print(f"    nnz(M) = {M.nnz}")

    with step(f"solving ({args.method})"):
        sol = problem.solve(
            body_force=None,
            dirichlet=displacement_bc,
            backend=backend,
            method=args.method,
            rtol=args.rtol,
            verbose=True,
            options=args.petsc_opts,
            preconditioner=args.pc,
        )
    sigma, u, s = sol["stress"], sol["displacement"], sol["rotation"]

    print("\n== solution ==")
    summarise("displacement (cell)", u, per=3)
    summarise("rotation (cell)", s, per=3)
    summarise("stress moments (facet)", sigma)

    print("\n== verification ==")
    vol = mesh.geometry.measure(3)
    scale = np.abs(sigma).max()
    eq = np.abs(D @ sigma).max() / scale
    sym = np.abs(A @ sigma).max() / scale
    print(f"    equilibrium   |div_h sigma| / |sigma|   {eq:.3e}   (load-free => 0)")
    print(f"    weak symmetry |as_h sigma|  / |sigma|   {sym:.3e}   (constraint => 0)")

    # Patch test on this very mesh: a linear displacement must be reproduced
    # exactly, because the local inner products satisfy M N = R.
    with step("patch test (linear displacement)"):
        exact = problem.solve(
            body_force=None,
            dirichlet=linear_displacement,
            backend=backend,
            method=args.method,
            rtol=args.rtol,
        )
    du = np.abs(exact["displacement"] - problem.interpolate_displacement(linear_displacement)).max()
    ds = np.abs(exact["stress"] - problem.interpolate_stress(linear_stress)).max()
    dr = np.abs(exact["rotation"] - problem.interpolate_rotation(linear_grad)).max()
    print(f"    displacement error            {du:.3e}")
    print(f"    stress error                  {ds:.3e}")
    print(f"    rotation error                {dr:.3e}")
    print(
        "    (limited only by the solver tolerance => the scheme is exact for"
        "\n     linear fields on this polytopal mesh)"
        if args.method != "direct"
        else "    (exact to round-off => the scheme is exact for linear fields here)"
    )

    # Facet traction moments are not plottable on their own: rebuild a
    # cell-centred stress tensor, exactly (constant stresses are reproduced).
    print("\n== postprocessing ==")
    from mimetika.postprocess import (
        export_vtu,
        mean_stress,
        principal_stresses,
        reconstruct_stress,
        von_mises,
    )

    with step("reconstructing cell stress tensor"):
        sig_cell = reconstruct_stress(mesh, sigma)
    vm = von_mises(sig_cell)
    pmean = mean_stress(sig_cell)
    principal = principal_stresses(sig_cell)
    asym = np.abs(sig_cell - np.swapaxes(sig_cell, 1, 2)).max() / np.abs(sig_cell).max()
    summarise("von Mises stress", vm)
    summarise("mean stress", pmean)
    print(
        f"    reconstructed asymmetry       {asym:.3e}"
        "\n      (symmetry is imposed *weakly*: as_h(sigma) = 0 holds exactly above,"
        "\n       so the reconstructed tensor is symmetric only to discretisation"
        "\n       error -- this number is itself an error indicator.  von Mises and"
        "\n       the principal stresses use the symmetric part.)"
    )

    if args.vtu:
        with step(f"writing {args.vtu}"):
            uu = u.reshape(-1, 3)
            ss = s.reshape(-1, 3)
            export_vtu(
                args.vtu,
                mesh,
                cell_data={
                    "displacement": uu,          # vector -> Glyph / Warp
                    "|displacement|": np.linalg.norm(uu, axis=1),
                    "rotation": ss,              # vector (axial form of skw grad u)
                    "stress": sig_cell,          # 9-component tensor
                    "von_mises": vm,
                    "mean_stress": pmean,
                    "sigma_min": principal[:, 0],
                    "sigma_max": principal[:, 2],
                },
            )
        print("    ParaView: colour by 'von_mises'; Glyph by 'displacement';")
        print("              'stress' is a full tensor (Tensor Glyph / components)")


if __name__ == "__main__":
    main()
