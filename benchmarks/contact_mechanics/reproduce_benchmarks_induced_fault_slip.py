#!/usr/bin/env python
"""Reproduce the induced-fault-slip benchmark suite of Novikov et al. (2024).

Runs the four benchmarks against the paper's semi-analytical reference
(4TU dataset, doi 10.4121/d77f1a2c-29ea-4572-ad72-e33ed8dc8d22) on the
paper's own DARTS grids and drops every figure into ``benchmark_figures/``:

========================  ==================================================
benchmark                 paper figures reproduced
========================  ==================================================
benchmark_0               Fig. 4   (unfaulted reservoir, combined stresses)
benchmark_1               Fig. 6   (vertical frictionless displaced fault)
benchmark_2               Figs. 3, 8, 9, 10, 12  (inclined fault, constant
                          friction: initial state, pre-slip stresses,
                          post-slip states on both domains, patch
                          boundaries and merging)
benchmark_3               Fig. 14  (slip-weakening: pre-nucleation state and
                          the nucleation pressure ``p*``)
========================  ==================================================

See ``benchmark_figures/README.md`` for the mathematical formulation.

Run with ``python -m benchmarks.contact_mechanics.reproduce_benchmarks_induced_fault_slip``
or directly as a script from anywhere in the repository.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = HERE / "benchmark_figures"

RUNS = [
    ("benchmarks.contact_mechanics.benchmark_0", "benchmark_0.png"),
    ("benchmarks.contact_mechanics.benchmark_1", "benchmark_1.png"),
    ("benchmarks.contact_mechanics.benchmark_2", "benchmark_2.png"),
    ("benchmarks.contact_mechanics.benchmark_3", "benchmark_3.png"),
]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    for module, figure in RUNS:
        print(f"\n=== {module} ===", flush=True)
        t1 = time.time()
        subprocess.run(
            [sys.executable, "-m", module, "--figure", str(OUT / figure)],
            check=True,
            cwd=REPO,
        )
        print(f"    [{module.rsplit('.', 1)[-1]}: {time.time() - t1:.0f} s]",
              flush=True)
    figures = sorted(p.name for p in OUT.glob("*.png"))
    print(f"\nall benchmarks reproduced in {time.time() - t0:.0f} s")
    print(f"figures in {OUT}:")
    for name in figures:
        print(f"  {name}")


if __name__ == "__main__":
    main()
