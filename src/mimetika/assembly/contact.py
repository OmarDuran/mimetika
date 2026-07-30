r"""Linear contact law on fracture facets, for the mixed elasticity problem.

A fracture that is a **compliant interface** -- normal and shear springs, no
in-plane elasticity -- obeys

    ``sigma n = C_f [[u]]``,        ``A_f = C_f^{-1}``  (the compliance)

and since the traction ``sigma n`` *is* a degree of freedom of the 3D mixed
problem, nothing new has to be introduced:

    ``A_f sigma n = [[u]]`` .

**No duplication.**  A massless interface satisfies ``t+ + t- = 0`` by
equilibrium, so the two sides do not carry independent tractions and the
ordinary (shared) traction block is the correct space.  This is the mirror
image of the flow problem, where the fracture is a genuine mass sink and the
two normal fluxes *must* be independent.

**The jump operator.**  In the local Hellinger--Reissner relation

    ``M_E sigma_E + |E| Dv_E^T u_E + |E| As_E^T s_E = G_E``

the datum ``G_E`` *is* the trace of the displacement on the facets of ``E``,
in the facet ``P_1`` coefficient basis.  So the trace recovered from a cell is

    ``tr(u_E)|_f = ( M_E sigma_E + |E| Dv_E^T u_E + |E| As_E^T s_E )_f``

and ``[[u]]_f = tr(u+)|_f - tr(u-)|_f``.  All three terms matter: the second is
the cell translation, the third is **the rotation times the lever arm**, and the
first the deformation inside the cell.  Dropping the rotation term -- taking the
naive difference of cell displacements -- would make a rigid rotation of the
whole body produce a spurious jump, and hence spurious traction.

On an interior facet the two ``G`` contributions enter with opposite incidence
signs and cancel, which is why traces never appear in the unfractured problem.
With a fracture they leave ``[[u]]``, and substituting the contact law turns it
into a **compliance block added to the traction DOFs** -- structurally the same
move as the ``|f|/kappa`` stiffness in the flow problem.

Two conversions are needed to write that block:

* ``A_f`` is diagonal in the facet frame ``(n, t1, t2)`` and must be rotated
  into global components, ``A = Q diag(1/k_n, 1/k_s, 1/k_s) Q^T``;
* the contact law is *pointwise* while the DOFs are *moments*, so the block
  carries the inverse facet Gram matrix: ``A_f (x) Gram^{-1}``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from mimetika.mesh.mesh import Mesh


@dataclass
class FractureContact:
    """A linear contact law on a set of tagged facets."""

    mesh: Mesh
    facets: np.ndarray
    normal_stiffness: float = 1.0
    shear_stiffness: float = 1.0
    dofs_per_facet: int | None = None  # defaults to d^2
    #: optional ``(d, d)`` compliance in the **facet** frame, overriding the
    #: stiffnesses above (the contact driver supplies ``(1/r) I`` through it)
    facet_compliance: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.facets = np.asarray(sorted(int(f) for f in self.facets), dtype=np.int64)
        self.dim = self.mesh.dim
        if self.dofs_per_facet is None:
            self.dofs_per_facet = self.dim * self.dim

    # -- ingredients -----------------------------------------------------------

    def rotation(self, facet: int) -> np.ndarray:
        """``(d, d)`` rotation taking **facet-frame** components to mesh ones."""
        from mimetika.geometry.local_cell import mesh_frame

        rows = self.mesh.geometry.facet_frame(int(facet))  # (d, 3) ambient
        if self.dim == 3:
            return rows.T
        return (rows @ mesh_frame(self.mesh.geometry)).T  # in-plane components

    def local_compliance(self, facet: int) -> np.ndarray:
        """``A_f`` in **mesh** components: ``R A_facet R^T``, shape ``(d, d)``."""
        d = self.dim
        A = (
            np.diag(
                [1.0 / self.normal_stiffness]
                + [1.0 / self.shear_stiffness] * (d - 1)
            )
            if self.facet_compliance is None
            else np.asarray(self.facet_compliance, dtype=float)
        )
        R = self.rotation(int(facet))
        return R @ A @ R.T

    def facet_gram(self, facet: int) -> np.ndarray:
        """``Gram_ab = int_f b_a b_b`` for the facet ``P_1`` basis, in closed form.

        The basis is centred on the facet centroid, so the constant decouples:
        ``Gram = diag(|f|, (t_a S t_b)/|f|)`` with ``S`` the second moment.
        """
        d = self.dim
        g = self.mesh.geometry
        area = g.measure(d - 1)[facet]
        S = g.second_moments(d - 1)[facet]
        t = g.facet_frame(int(facet))[1:]  # (d-1, 3) tangents
        gram = np.zeros((d, d))
        gram[0, 0] = area
        gram[1:, 1:] = np.einsum("ad,dc,bc->ab", t, S, t) / area
        return gram

    def block(self, facet: int) -> np.ndarray:
        """``A_f (x) Gram^{-1}`` -- the compliance in the traction DOF basis.

        Both factors are ``d x d``, so the block is ``d^2 x d^2``: 4x4 in 2D,
        9x9 in 3D, matching the traction DOFs of a facet.
        """
        return np.kron(
            self.local_compliance(facet), np.linalg.inv(self.facet_gram(facet))
        )

    # -- assembly ---------------------------------------------------------------

    def assemble(self, n_dofs: int | None = None) -> sp.csr_matrix:
        """The global compliance contribution, to be **added** to ``M``."""
        ndf = self.dofs_per_facet
        n = (
            ndf * self.mesh.num_cells(self.dim - 1) if n_dofs is None else n_dofs
        )
        rows, cols, vals = [], [], []
        for f in self.facets:
            dofs = ndf * int(f) + np.arange(ndf)
            blk = self.block(int(f))
            rows.append(np.repeat(dofs, ndf))
            cols.append(np.tile(dofs, ndf))
            vals.append(blk.ravel())
        if not rows:
            return sp.csr_matrix((n, n))
        return sp.csr_matrix(
            (
                np.concatenate(vals),
                (np.concatenate(rows), np.concatenate(cols)),
            ),
            shape=(n, n),
        )

    def opening(self, stress: np.ndarray, facet: int) -> np.ndarray:
        """The displacement jump on a facet: ``[[u]] = A_f sigma n``.

        Returned as the constant part of the jump vector -- the mean opening
        (normal component) and slip (tangential components) of that facet.
        """
        d, ndf = self.dim, self.dofs_per_facet
        moments = np.asarray(stress)[ndf * int(facet) : ndf * (int(facet) + 1)]
        coeffs = np.linalg.solve(
            self.facet_gram(int(facet)), moments.reshape(d, d).T
        ).T
        return self.local_compliance(int(facet)) @ coeffs[:, 0]
