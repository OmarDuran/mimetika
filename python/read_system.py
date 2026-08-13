"""Read a system exported by the C++ assembly, and solve it here.

The port's claim is that the C++ assembles the SAME operator as the Python.
Checking that needs no C++ solver -- and building one would put a second
unvalidated component between the assembly and the answer.  So the system
comes across and the linear algebra happens where a solver already exists and
is trusted.

What arrives is everything needed to rebuild the problem: the matrix, the
residual, the field layout and the constraint mask with its values.  A
comparison holding only the matrix could not tell a difference in the operator
from a difference in what was pinned, and those fail in completely different
ways.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class System:
    """An assembled system: ``A``, ``residual``, the fields, and the pins."""

    def __init__(self, A, residual, state, fields, pinned, pin_value):
        self.A = A
        self.residual = residual
        self.state = state
        self.fields = fields          # name -> (offset, size)
        self.pinned = pinned          # bool mask over the dofs
        self.pin_value = pin_value

    @property
    def n(self) -> int:
        return self.A.shape[0]

    def block(self, name: str, x=None):
        """One field's slice of a vector, defaulting to the state."""
        off, size = self.fields[name]
        v = self.state if x is None else x
        return v[off : off + size]

    def solve(self):
        """Newton step: ``A dx = -residual``, with the pins already substituted.

        The exported rows of pinned dofs are the identity and their residual is
        the discrepancy, so nothing here needs to know about the constraints --
        which is the point of substituting rather than penalising.
        """
        import scipy.sparse.linalg as spla

        dx = spla.spsolve(self.A.tocsc(), -self.residual)
        return self.state + dx


def read_system(path: str) -> System:
    with open(path, "rb") as f:
        if f.read(8) != b"MIMSYS01":
            raise ValueError(f"{path}: not a mimetika system export")
        n, nnz, n_fields = np.fromfile(f, dtype=np.int64, count=3)

        fields = {}
        for _ in range(int(n_fields)):
            (ln,) = np.fromfile(f, dtype=np.int64, count=1)
            name = f.read(int(ln)).decode()
            off, size = np.fromfile(f, dtype=np.int64, count=2)
            fields[name] = (int(off), int(size))

        rows = np.fromfile(f, dtype=np.int64, count=int(nnz))
        cols = np.fromfile(f, dtype=np.int64, count=int(nnz))
        vals = np.fromfile(f, dtype=np.float64, count=int(nnz))
        residual = np.fromfile(f, dtype=np.float64, count=int(n))
        state = np.fromfile(f, dtype=np.float64, count=int(n))
        pinned = np.fromfile(f, dtype=np.int8, count=int(n)).astype(bool)
        pin_value = np.fromfile(f, dtype=np.float64, count=int(n))

    A = sp.coo_matrix((vals, (rows, cols)), shape=(int(n), int(n))).tocsr()
    return System(A, residual, state, fields, pinned, pin_value)


def structure(s: System) -> dict:
    """Structural facts about the FREE block, where the convention is visible.

    A constrained row is the identity, so it is neither symmetric nor
    antisymmetric with its column -- that is what substitution means.  Asking
    about the structure of the assembled operator therefore means asking about
    the rows that were not replaced.
    """
    free = np.flatnonzero(~s.pinned)
    B = s.A[free][:, free]
    return {
        "n_free": len(free),
        "sym": float(abs(B - B.T).max()) if B.nnz else 0.0,
        "anti": float(abs(B + B.T).max()) if B.nnz else 0.0,
        "adjoint": float(abs(abs(B) - abs(B.T)).max()) if B.nnz else 0.0,
    }


def report(s: System) -> str:
    lines = [f"{s.n} dofs, {s.A.nnz} nonzeros, {int(s.pinned.sum())} pinned"]
    for name, (off, size) in s.fields.items():
        lines.append(f"  {name:8s} offset {off:8d}  size {size:8d}")
    st = structure(s)
    lines.append(
        f"  free block {st['n_free']}: |A - A^T| {st['sym']:.2e}"
        f"   |A + A^T| {st['anti']:.2e}   ||A| - |A^T|| {st['adjoint']:.2e}"
    )
    return "\n".join(lines)
