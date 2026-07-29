"""DOF layer: numbering of discrete differential-form degrees of freedom.

For the lowest-order mimetic (differential-forms) discretisation, a k-form has
exactly one degree of freedom per k-cell -- its integral over that cell:

    vertex (0-form)  -> point value
    edge   (1-form)  -> circulation along the edge
    facet  (2-form)  -> flux through the facet
    cell   (3-form)  -> integral over the cell

This layer maps ``(form_degree, local_cell_index)`` to a global row/column
index.  :class:`MixedDofHandler` stacks several form spaces into one blocked
system (e.g. flux + potential for a mixed Poisson problem).

The design is order-agnostic: a higher-order space just overrides
``n_dofs_per_cell`` and the local-to-global map without touching callers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mimetika.mesh.mesh import Mesh


@dataclass
class DofHandler:
    """Global numbering for a single k-form space (lowest order)."""

    mesh: Mesh
    degree: int  # form degree k

    @property
    def n_dofs_per_cell(self) -> int:
        return 1  # lowest order: one DOF per k-cell

    @property
    def n_dofs(self) -> int:
        return self.mesh.num_cells(self.degree) * self.n_dofs_per_cell

    def local_to_global(self, cell_indices: np.ndarray) -> np.ndarray:
        """Map k-cell indices to global DOF indices (identity at lowest order)."""
        return np.asarray(cell_indices, dtype=np.int64)


@dataclass
class MixedDofHandler:
    """Block numbering stacking several form spaces into one system."""

    blocks: list[DofHandler]

    @property
    def offsets(self) -> list[int]:
        off, acc = [], 0
        for b in self.blocks:
            off.append(acc)
            acc += b.n_dofs
        return off

    @property
    def n_dofs(self) -> int:
        return sum(b.n_dofs for b in self.blocks)

    def global_range(self, block: int) -> tuple[int, int]:
        off = self.offsets
        return off[block], off[block] + self.blocks[block].n_dofs
