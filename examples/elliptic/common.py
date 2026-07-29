"""Shared helpers for the elliptic convergence studies."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow running the examples straight from a checkout, without installing.
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def observed_rates(
    h: list[float], err: list[float], floor: float = 1e-12
) -> list[float | None]:
    """Rates ``log(e_i/e_{i-1}) / log(h_i/h_{i-1})``; ``None`` for the first entry.

    A rate measured against an error at round-off level carries no information
    (the exact solution happened to be reproduced), so those are reported as
    ``None`` rather than as a spurious large number.
    """
    rates: list[float | None] = [None]
    for i in range(1, len(h)):
        if min(err[i], err[i - 1]) <= floor:
            rates.append(None)
        else:
            rates.append(np.log(err[i] / err[i - 1]) / np.log(h[i] / h[i - 1]))
    return rates


def print_table(title: str, rows: list[dict], columns: list[tuple[str, str]]) -> None:
    """Print a convergence table.

    ``columns`` is a list of ``(key, header)`` pairs; keys ending in ``rate``
    are printed as ``--`` when ``None``.
    """
    print(f"\n{title}")
    print("-" * len(title))
    widths = [max(len(h), 11) for _, h in columns]
    print("  ".join(h.rjust(w) for (_, h), w in zip(columns, widths)))
    for row in rows:
        cells = []
        for (key, _), w in zip(columns, widths):
            v = row.get(key)
            if v is None:
                s = "--"
            elif isinstance(v, (int, np.integer)):
                s = str(v)
            elif "rate" in key:
                s = f"{v:.2f}"
            else:
                s = f"{v:.4e}"
            cells.append(s.rjust(w))
        print("  ".join(cells))


def energy_norm(M, e: np.ndarray) -> float:
    """``sqrt(e^T M e)`` -- the natural mesh-dependent norm for facet unknowns."""
    return float(np.sqrt(max(e @ (M @ e), 0.0)))


def cell_l2_norm(measures: np.ndarray, e: np.ndarray, ncomp: int = 1) -> float:
    """``sqrt(sum_E |E| |e_E|^2)`` for cell-wise (piecewise constant) unknowns."""
    v = e.reshape(-1, ncomp)
    return float(np.sqrt(np.sum(measures[:, None] * v**2)))
