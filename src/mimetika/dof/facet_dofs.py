r"""Numbering of facet degrees of freedom, with optional per-side duplication.

A fracture makes the normal trace of the flux **discontinuous**: the flow
problem lives in ``H(div)`` on the cut domain ``Omega \ Gamma``.  That is a
property of the discrete *space*, not of the geometry -- the mesh, the cell
complex and ``dd = 0`` are all untouched.  This module expresses exactly that:
the same :class:`~mimetika.mesh.mesh.Mesh` carries different DOF layouts for
different physics.

* An **untagged** facet gives one block of DOFs shared by its two cells -- the
  usual conforming space, in which ``sum_E s_{E,f} u_f = 0`` structurally, i.e.
  flux continuity.
* A **tagged** (fracture) facet gives each incident cell its *own* block, so
  ``un+`` and ``un-`` are independent and their sum -- the mass exchanged with
  the fracture -- is free rather than identically zero.

Mechanics uses the untagged layout even on fracture facets: a massless contact
interface satisfies ``t+ + t- = 0`` by equilibrium, so a single traction block
is the correct space.

Numbering is chosen so that **with no tags the map is the identity** (facet
``f`` owns dofs ``f*ndf ... f*ndf+ndf-1``), which keeps the un-fractured
assembly bit-identical to the version that indexed facets directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mimetika.mesh.mesh import Mesh


@dataclass
class FacetDofMap:
    """Maps ``(cell, facet)`` to global DOF indices on facets."""

    mesh: Mesh
    dofs_per_facet: int = 1
    duplicated: frozenset[int] = field(default_factory=frozenset)

    _second_block: dict[int, int] = field(default_factory=dict, repr=False)
    _second_owner: dict[int, int] = field(default_factory=dict, repr=False)
    n_blocks: int = 0

    def __post_init__(self) -> None:
        cx = self.mesh.complex
        n_facets = cx.num_cells(cx.dim - 1)
        self.duplicated = frozenset(int(f) for f in self.duplicated)

        # Blocks 0..n_facets-1 are the facets themselves; a duplicated facet
        # gets one extra block, owned by the *second* of its incident cells.
        self.n_blocks = n_facets
        for f in sorted(self.duplicated):
            cells = self.facet_cells(f)
            if len(cells) < 2:
                continue  # a boundary facet is already one-sided
            self._second_block[f] = self.n_blocks
            self._second_owner[f] = cells[1]
            self.n_blocks += 1

    # -- queries -------------------------------------------------------------

    def facet_cells(self, facet: int) -> list[int]:
        """Cells incident to a facet, in ascending order (deterministic)."""
        row = self.mesh.complex.boundary_matrix(self.mesh.dim).tocsr()[facet]
        return sorted(int(c) for c in row.indices)

    @property
    def n_dofs(self) -> int:
        return self.n_blocks * self.dofs_per_facet

    @property
    def n_duplicated(self) -> int:
        """Number of facets that actually carry two blocks."""
        return len(self._second_block)

    def block(self, cell: int, facet: int) -> int:
        """Index of the DOF block that ``cell`` sees on ``facet``."""
        if self._second_owner.get(facet) == cell:
            return self._second_block[facet]
        return facet

    def dofs(self, cell: int, facet: int) -> np.ndarray:
        """Global DOF indices that ``cell`` sees on ``facet``."""
        b = self.block(cell, facet)
        return self.dofs_per_facet * b + np.arange(self.dofs_per_facet)

    def cell_dofs(self, cell: int, facet_ids=None) -> np.ndarray:
        """All facet DOFs of one cell, concatenated in facet order."""
        if facet_ids is None:
            facet_ids = [f for f, _ in self.mesh.complex.facets_of(self.mesh.dim, cell)]
        blocks = np.array([self.block(cell, int(f)) for f in facet_ids])
        return (
            self.dofs_per_facet * blocks[:, None] + np.arange(self.dofs_per_facet)
        ).ravel()

    def sides(self, facet: int) -> list[tuple[int, int]]:
        """``(cell, block)`` for each side of a facet -- the fracture's two sides."""
        return [(c, self.block(c, facet)) for c in self.facet_cells(facet)]

    def is_conforming(self) -> bool:
        """True when no facet is duplicated (the standard space)."""
        return self.n_duplicated == 0
