"""Solve an exported system here, and check what only a solve can show.

Run:  python3 python/check_system.py /tmp/elast_sys.bin
"""

from __future__ import annotations

import sys

import numpy as np

from read_system import read_system, report, structure


def main(path: str) -> int:
    s = read_system(path)
    print(report(s))

    # the pins really were substituted: an identity row and nothing else
    bad = [d for d in np.flatnonzero(s.pinned)
           if s.A.getrow(int(d)).nnz != 1 or s.A[int(d), int(d)] != 1.0]
    print(f"  pinned rows are the identity: {'ok' if not bad else f'{len(bad)} wrong'}")

    # THE CONVENTION, visible only on the free block: a constitutive block is
    # symmetric and every off-diagonal pair is the negative transpose, so the
    # operator is neither symmetric nor antisymmetric but IS exactly adjoint
    # in magnitude.  A mismatch here means two physics disagreed about signs.
    st = structure(s)
    print(f"  adjoint in magnitude: {st['adjoint']:.2e}"
          f"  {'ok' if st['adjoint'] < 1e-10 else 'MISMATCHED CONVENTIONS'}")

    x = s.solve()
    residual = s.A @ (x - s.state) + s.residual
    print(f"  solved {s.n} dofs, linear residual {np.abs(residual).max():.2e}")
    for name in s.fields:
        print(f"    {name:8s} |.|inf = {np.abs(s.block(name, x)).max():.4e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "system.bin"))
