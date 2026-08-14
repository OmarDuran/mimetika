"""The C++ consolidation column against Coussy Sect. 5.2.2, Eq. 5.87.

Run:  python python/check_terzaghi.py /tmp/terz20.txt

The series is the answer, not another code: this compares the assembled,
stepped C++ solution against the closed form directly, so nothing about the
Python discretization enters. Two errors are reported per time factor -- the
worst cell and the root mean square over the column -- because a mixed method
that is wrong in one cell and a mixed method that is wrong everywhere fail the
same maximum test while meaning entirely different things.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "benchmarks/poroelasticity")
from consolidation_soil import analytic_pressure  # noqa: E402


def main(path: str) -> int:
    rows, head = [], ""
    for line in open(path):
        if line.startswith("#"):
            head += line
            continue
        if line.strip():
            rows.append([float(x) for x in line.split()])
    data = np.array(rows)
    print(head.strip())

    print(f"{'T':>8} {'cells':>6} {'max |err|':>11} {'rms |err|':>11} {'p(base)':>10} "
          f"{'exact':>10}")
    worst = 0.0
    for T in sorted(set(data[:, 0])):
        sel = data[data[:, 0] == T]
        zbar, pbar = sel[:, 1], sel[:, 2]
        # the driver writes zbar as depth BELOW the drained surface; the series
        # is anchored there, and takes elevation above the sealed base
        exact = analytic_pressure(1.0 - zbar, T)
        err = np.abs(pbar - exact)
        worst = max(worst, err.max())
        deep = int(np.argmax(zbar))
        print(f"{T:>8g} {len(sel):>6d} {err.max():>11.3e} "
              f"{np.sqrt((err**2).mean()):>11.3e} {pbar[deep]:>10.6f} {exact[deep]:>10.6f}")
    print(f"\nworst over every sampled time and cell: {worst:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/terz20.txt"))
