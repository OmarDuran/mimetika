#!/usr/bin/env python3
"""Mesh diagnostics, before anything is solved on it.

Three questions, answered in order:

  1. What the mesh is. exokal's diagnose_vtu: the formal findings on the
     complex, the shape-regularity classification at the library constants,
     and the metric-degeneracy witnesses -- cells whose measure falls below a
     percentage of the mean over their vertex star, truly collapsed rather
     than merely small in a refined region.

  2. What a product looks like on it, cell by cell and nothing assembled. Any
     flux or stress inner product is built as the model would build it, and
     every cell's block is sent to an eigensolver: its extreme eigenvalues and
     their ratio, the conditioning of that cell's flux (or stress) block. The
     statistics over the mesh say where a solver's trouble will come from; the
     worst cells say which ones. For the adaptive products the selection eta
     rides along, and for the two-point members the star's validity: a facet
     the centroid does not see squarely has a non-positive weight, and the
     spectrum shows it as a negative eigenvalue.

  3. Named cells. --vtk-id takes the ids ParaView shows -- the file's own --
     and reports each such cell in every light at once: collapse percent,
     eta, star validity, and its block's spectrum.

Run it:

    PYTHONPATH=.. python mesh_diagnostics.py --mesh domain.vtu
    PYTHONPATH=.. python mesh_diagnostics.py --mesh domain.vtu --product stabilized_rt
    PYTHONPATH=.. python mesh_diagnostics.py --mesh domain.vtu --physics elasticity \\
        --product stabilized_vem --formulation strong_symmetry_total
    PYTHONPATH=.. python mesh_diagnostics.py --mesh domain.vtu --vtk-id 47788 47789 65113
    PYTHONPATH=.. python mesh_diagnostics.py --mesh domain.vtu --vtu flags.vtu --output diag/
"""

import argparse
import os

import mimetika_cxx as mk
import numpy as np

from _diagnostics import write_report
from _stages import stage

FLUX = {
    "derham_bdm": mk.FluxRealization.derham_bdm,
    "derham_rt": mk.FluxRealization.derham_rt,
    "stabilized_rt": mk.FluxRealization.stabilized_rt,
    "diagonal_tpfa": mk.FluxRealization.diagonal_tpfa,
    "adaptive_rt": mk.FluxRealization.adaptive_rt,
}
STRESS = {name: getattr(mk.StressRealization, name)
          for name in mk.StressRealization.__members__}
FORMULATIONS = {name: getattr(mk.StressFormulation, name)
                for name in mk.StressFormulation.__members__}
TWO_POINT = {"diagonal_tpfa", "adaptive_rt", "diagonal_vem", "adaptive_vem", "diagonal_afw"}


def percentile_row(name, values, fmt="{:.3e}"):
    v = np.asarray(values, dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return f"  {name:14s} (no finite values)"
    q = np.percentile(finite, [0, 50, 99, 100])
    return (f"  {name:14s} min {fmt.format(q[0])}   median {fmt.format(q[1])}   "
            f"p99 {fmt.format(q[2])}   max {fmt.format(q[3])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mesh", required=True, help="the .vtu to diagnose")
    ap.add_argument("--degeneracy-percent", type=float, default=None,
                    help="a cell is degenerate below this percent of its node-star mean "
                         "measure; defaults to exokal's default_degeneracy_percent")
    ap.add_argument("--cond-threshold", type=float, default=None,
                    help="for an adaptive product: select eta from CONDITIONING as well -- "
                         "a cell whose stabilized block has lambda_max/lambda_min above this "
                         "takes the diagonal star (eta = 0); composes with the collapse scan")
    ap.add_argument("--physics", default="flow", choices=("flow", "elasticity"),
                    help="which family --product names (derham_bdm exists in both)")
    ap.add_argument("--product", default="adaptive_rt",
                    help="the inner product evaluated cell by cell: a flux realization "
                         f"({', '.join(FLUX)}) or, with --physics elasticity, a stress one "
                         f"({', '.join(STRESS)})")
    ap.add_argument("--formulation", default=None, choices=sorted(FORMULATIONS),
                    help="for a stress product: the formulation its block is built under")
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--vtk-id", type=int, nargs="+", default=[],
                    help="report these cells, by the file's own (ParaView) cell ids")
    ap.add_argument("--worst", type=int, default=10,
                    help="how many worst-conditioned cells to list")
    ap.add_argument("--vtu", help="write the table as cell fields (n_dofs, lambda_min, "
                                  "lambda_max, cond, eta, star_invalid, n_neighbours, "
                                  "collapse_percent) here; counts and flags as Int32")
    ap.add_argument("--output", help="folder for diagnostics.txt and degenerate_cells.csv")
    ap.add_argument("--quiet", action="store_true", help="counts only, no full report")
    args = ap.parse_args()

    pct = args.degeneracy_percent
    if pct is None:
        pct = mk.default_degeneracy_percent

    # ---- 1. the mesh -------------------------------------------------------
    with stage(f"diagnosing {args.mesh}"):
        d = mk.diagnose_vtu(args.mesh, pct)
    if not args.quiet:
        print(d["report"])
        print(d["quality"])
    degenerate = sorted(d["degenerate"], key=lambda r: r["percent"])
    print(f"classification: {d['classification']}  "
          f"(gamma {d['gamma_all']:.3e}, eps set aside {d['gamma_minus_eps']:.3e}, "
          f"median {d['gamma_median']:.3e})")
    print(f"degenerate cells at {pct}%: {len(degenerate)}")
    for r in degenerate[:10]:
        print(f"  vtk id {r['vtk_cell_id']:8d}  measure {r['measure']:.3e}  "
              f"{r['percent']:.5f}% of its star mean ({r['n_neighbors']} neighbours)")
    if len(degenerate) > 10:
        print(f"  ... {len(degenerate) - 10} more")
    degenerate_pct = {int(r["cell"]): r["percent"] for r in degenerate}

    # ---- 2. the product, cell by cell ---------------------------------------
    with stage("reading the mesh"):
        mesh = mk.read_vtu(args.mesh)
        vtk_ids = np.asarray(mk.vtu_cell_ids(args.mesh))
    dim = mesh.dim
    n = mesh.count(dim)
    cell_of_vtk = {int(v): c for c, v in enumerate(vtk_ids)}
    # the collapse percent of every cell, so the tables never show a blank for
    # a cell the scan merely did not flag: flagged means below the threshold,
    # and the number itself says how far above it the others sit
    with stage("measuring every cell against its node star"):
        collapse = np.asarray(mk.cell_collapse_percent(mesh, dim))
        neighbours = np.asarray(mk.cell_star_neighbours(mesh, dim))

    if args.physics == "flow":
        if args.product not in FLUX:
            raise SystemExit(f"--product {args.product} is not a flux realization; "
                             f"use --physics elasticity for a stress one")
        with stage(f"evaluating {args.product} on every cell"):
            sp = mk.flux_cell_spectra(mesh, dim, FLUX[args.product], 1.0,
                                      args.degeneracy_percent, args.cond_threshold)
        label = args.product
    else:
        if args.product not in STRESS:
            raise SystemExit(f"--product {args.product} is not a stress realization")
        form = args.formulation
        if form is None:
            strong = args.product in ("stabilized_vem", "diagonal_vem", "adaptive_vem")
            total = args.product in ("diagonal_vem", "adaptive_vem", "diagonal_afw")
            form = ("strong_symmetry_total" if strong and total else
                    "strong_symmetry" if strong else
                    "weak_symmetry_total" if total else "weak_symmetry")
        with stage(f"evaluating {args.product} ({form}) on every cell"):
            sp = mk.stress_cell_spectra(mesh, dim, STRESS[args.product], FORMULATIONS[form],
                                        args.mu, args.lam, args.degeneracy_percent,
                                        args.cond_threshold)
        label = f"{args.product} ({form})"

    cond = np.asarray(sp["cond"])
    lmin = np.asarray(sp["lambda_min"])
    lmax = np.asarray(sp["lambda_max"])
    eta = np.asarray(sp["eta"])
    star = np.asarray(sp["star_invalid"])
    ndof = np.asarray(sp["n_dofs"])

    print(f"\n{label}: the cell block, over {n} cells "
          f"({int(ndof.min())}..{int(ndof.max())} dofs per cell)")
    print(percentile_row("lambda_min", lmin))
    print(percentile_row("lambda_max", lmax))
    print(percentile_row("cond", cond))
    n_neg = int((lmin <= 0.0).sum())
    if n_neg:
        print(f"  *** {n_neg} cell(s) have a NON-POSITIVE eigenvalue: the block is not positive")
        print("  *** definite there and nothing built on it is meaningful.")
    if args.cond_threshold is not None and not args.product.startswith("adaptive"):
        raise SystemExit("--cond-threshold selects eta, which only an adaptive product has")
    if args.product in TWO_POINT:
        on_star = int((eta == 0.0).sum())
        by_cond = int(sp.get("n_ill_conditioned", 0))
        print(f"  two-point star on {on_star} cell(s), {int(star.sum())} of them not star-shaped")
        if args.cond_threshold is not None:
            print(f"  {by_cond} of them switched by conditioning (cond > {args.cond_threshold:g}), "
                  f"{on_star - by_cond} by the collapse scan")
        if (star != 0).any():
            print("  *** the star is INVALID on those cells (a facet the centroid does not see)")

    # eta exists only for the adaptive products, the star only for the two-point
    # members; elsewhere the columns print "-" rather than a number that would
    # claim a selection the product never made
    has_eta = args.product.startswith("adaptive")
    has_star = args.product in TWO_POINT

    def row(c):
        e = f"{eta[c]:4.0f}" if has_eta else f"{'-':>4s}"
        st = f"{int(star[c]):12d}" if has_star else f"{'-':>12s}"
        return (f"{c:9d} {int(ndof[c]):5d} {lmin[c]:12.3e} {lmax[c]:12.3e} "
                f"{cond[c]:11.3e} {e} {st} {collapse[c]:11.5f}")

    order = np.argsort(-np.where(np.isfinite(cond), cond, np.inf))
    print(f"\n  worst {args.worst} cells by conditioning:")
    print(f"  {'vtk id':>9s} {'cell':>9s} {'dofs':>5s} {'lambda_min':>12s} {'lambda_max':>12s} "
          f"{'cond':>11s} {'eta':>4s} {'star_invalid':>12s} {'collapse %':>11s}")
    for c in order[:args.worst]:
        c = int(c)
        print(f"  {int(vtk_ids[c]):9d} {row(c)}")

    # ---- 3. the named cells ---------------------------------------------------
    if args.vtk_id:
        print(f"\n  named cells ({label}):")
        print(f"  {'vtk id':>9s} {'cell':>9s} {'dofs':>5s} {'lambda_min':>12s} {'lambda_max':>12s} "
              f"{'cond':>11s} {'eta':>4s} {'star_invalid':>12s} {'collapse %':>11s}")
        for v in args.vtk_id:
            c = cell_of_vtk.get(int(v))
            if c is None:
                print(f"  {v:9d}  not a top cell of this file")
                continue
            print(f"  {v:9d} {row(c)}")

    # ---- outputs --------------------------------------------------------------
    if args.output:
        out = os.path.join(args.output, os.path.splitext(os.path.basename(args.mesh))[0])
        with stage(f"writing the report into {out}"):
            write_report(args.mesh, out, pct)
    if args.vtu:
        with stage(f"writing {args.vtu}"):
            # the table's columns, one cell field each, counts and flags as
            # integers. Where a column has no meaning for the product -- eta
            # off the adaptive members, the star off the two-point ones --
            # the field carries -1 rather than a number that would read as a
            # selection or a verdict; cond is -1 where the block is not
            # positive (the eigenvalues say why), and collapse_percent is nan
            # on a cell with no star neighbour.
            fields = {
                "n_dofs": ndof.astype(np.int32),
                "lambda_min": lmin,
                "lambda_max": lmax,
                "cond": np.where(np.isfinite(cond), cond, -1.0),
                "eta": eta.astype(np.int32) if has_eta else np.full(n, -1, dtype=np.int32),
                "star_invalid": star.astype(np.int32) if has_star else np.full(n, -1, dtype=np.int32),
                "n_neighbours": neighbours.astype(np.int32),
                "collapse_percent": collapse,
            }
            mk.write_vtu(mesh, args.vtu, fields)


if __name__ == "__main__":
    main()
