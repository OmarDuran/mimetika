"""Time stepping as its own abstraction: the Runge--Kutta family.

The Biot system is marched with diagonally implicit Runge--Kutta schemes; the
one member implemented is backward Euler -- DIRK with a single stage, Butcher
tableau ``(A, b, c) = ([[1]], [1], [1])`` -- which the mixed formulation's
``previous``-state right-hand side realises.  Step sizes are held here,
independently of the scheme: a constant ``dt``, or an explicit ``schedule`` of
times (geometric refinement through a transient, coarsening in the tail),
yielded as ``(t, dt)`` pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np

#: Butcher tableaus of the implemented members, ``(A, b, c)``.
TABLEAUS = {
    "backward-euler": (np.array([[1.0]]), np.array([1.0]), np.array([1.0])),
}


@dataclass
class RKTimeStepping:
    """Step sizes and scheme of a transient march.

    Exactly one of ``dt`` (constant stepping, with ``n_steps`` or a final time
    given to :meth:`steps`) and ``schedule`` (explicit times) must be provided.
    ``tableau`` names the RK member; only ``"backward-euler"`` is in
    :data:`TABLEAUS`, and any other name raises.
    """

    dt: float | None = None
    schedule: Sequence[float] | None = None
    t0: float = 0.0
    tableau: str = "backward-euler"

    def __post_init__(self) -> None:
        if (self.dt is None) == (self.schedule is None):
            raise ValueError("provide exactly one of dt and schedule")
        if self.tableau not in TABLEAUS:
            raise NotImplementedError(
                f"tableau {self.tableau!r} not implemented; available: "
                f"{sorted(TABLEAUS)}"
            )
        if self.schedule is not None:
            times = np.asarray(self.schedule, dtype=float)
            if len(times) == 0 or np.any(np.diff(times) <= 0.0) \
                    or times[0] <= self.t0:
                raise ValueError(
                    "schedule must be strictly increasing and start after t0"
                )
            self.schedule = times

    @property
    def butcher(self):
        """The scheme's Butcher tableau ``(A, b, c)``."""
        return TABLEAUS[self.tableau]

    @property
    def constant(self) -> bool:
        """True when every step has the same size (the fast-path condition:
        a constant ``dt`` keeps the transient system matrix fixed, so one
        factorization serves the whole march)."""
        return self.dt is not None

    def steps(self, n_steps: int | None = None,
              t_end: float | None = None) -> Iterator[tuple[float, float]]:
        """Yield ``(t, dt)`` for each step of the march.

        Constant mode needs ``n_steps`` or ``t_end``; ``n_steps`` takes
        precedence when both are given, otherwise the count is
        ``round((t_end - t0) / dt)``.  Schedule mode ignores both and walks
        the given times.
        """
        if self.schedule is not None:
            t = self.t0
            for target in self.schedule:
                yield float(target), float(target - t)
                t = float(target)
            return
        if n_steps is None:
            if t_end is None:
                raise ValueError("constant stepping needs n_steps or t_end")
            n_steps = int(round((t_end - self.t0) / self.dt))
        t = self.t0
        for _ in range(n_steps):
            t += self.dt
            yield t, self.dt
