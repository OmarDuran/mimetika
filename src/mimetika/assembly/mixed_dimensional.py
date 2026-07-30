r"""Mixed-dimensional Darcy flow: a 3D matrix coupled to 2D fractures.

The fracture is a set of **tagged facets** of the ambient mesh.  Dimensional
reduction of a thin feature of aperture ``eps`` leaves two things behind: a
tangential Darcy problem on the ``d-1`` dimensional surface, and the flow
*across* the feature, which survives as an interface law.

Each subdomain keeps its own mixed saddle point.  For a fracture facet ``f`` of
cell ``E``, writing the trace of the matrix pressure as

    ``tr(p_E)|_f = kappa_E^{-1} un_E + p_2``,      ``kappa = 2 k_n / eps``

and substituting it into the local relation ``M_E F_E - p_E d_E + Pi_E = 0``
(where ``Pi_f = |f| tr(p_E)`` is exactly the slot a Dirichlet datum occupies)
gives

    ``(M_3 u)_f + (|f|/kappa_E) u_f - s_E |f| p_E + s_E |f| p_2 = 0`` .

So the Robin coupling is a **diagonal stiffness on the fracture flux DOFs**
plus a ``B``-like pairing with the fracture pressure -- no mortar unknowns, and
no trace unknowns either, because ``tr(p_E)`` was eliminated on the way in.

The two sides must be independent for the fracture to exchange mass: with a
single shared DOF, ``un+ + un- ≡ 0`` identically and the fracture receives
nothing.  That independence is supplied by the DOF map
(:class:`~mimetika.dof.facet_dofs.FacetDofMap`), **not** by modifying the mesh:
the geometry, the cell complex and ``dd = 0`` are untouched, and only the
discrete space changes.

The assembled system, symmetric with ``A`` block-diagonal::

    [ M_3 + S     0    | -B_3^T    C^T  ] [u_3]   [-g_3]
    [    0      M_2    |   0     -B_2^T ] [u_2] = [-g_2]
    [ -B_3        0    |   0        0   ] [p_3]   [-b_3]
    [    C      -B_2   |   0        0   ] [p_2]   [-b_2]

``C u_3`` is the mass entering each fracture cell, and it is the adjoint of the
``p_2`` pairing above -- the same matrix, so the system stays symmetric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from mimetika.assembly.mixed import (
    MixedSolution,
    boundary_facets,
    discrete_divergence,
    facet_cell_signs,
)
from mimetika.dof import FacetDofMap
from mimetika.mesh.fracture import fracture_mesh
from mimetika.mesh.mesh import Mesh
from mimetika.operators.diffusion import DiffusionInnerProduct
from mimetika.solver.saddle import solve_saddle


@dataclass
class Fracture:
    """A fracture: tagged facets plus its (constant) material parameters."""

    facets: np.ndarray
    aperture: float = 1e-3
    normal_permeability: float = 1.0
    tangential_permeability: float = 1.0

    def __post_init__(self) -> None:
        self.facets = np.asarray(sorted(int(f) for f in self.facets), dtype=np.int64)

    @property
    def kappa(self) -> float:
        """Normal conductivity across a half-aperture, ``2 k_n / eps``."""
        return 2.0 * self.normal_permeability / self.aperture

    @property
    def transmissivity(self) -> float:
        """Tangential conductivity integrated across the aperture, ``eps k_t``."""
        return self.aperture * self.tangential_permeability


class MixedDimensionalDarcy:
    """Mixed Darcy on a 3D matrix coupled to a 2D fracture by a Robin law."""

    def __init__(
        self, mesh: Mesh, fracture: Fracture, K: np.ndarray | None = None
    ) -> None:
        self.mesh = mesh
        self.fracture = fracture
        self.dofmap = FacetDofMap(mesh, 1, frozenset(fracture.facets))
        self.frac_mesh, self.facet_of_cell = fracture_mesh(mesh, fracture.facets)

        self.inner3 = DiffusionInnerProduct(mesh, K=K)
        self.inner2 = DiffusionInnerProduct(
            self.frac_mesh, K=fracture.transmissivity * np.eye(3)
        )
        self._signs = {
            (c, f): s
            for c in range(mesh.num_cells(3))
            for f, s in mesh.complex.facets_of(3, c)
        }
        # fracture vertex -> ambient vertex (fracture_mesh preserves loop order)
        self._ambient_vertex = {}
        for fc, f in enumerate(self.facet_of_cell):
            amb = mesh.complex.polygon_loops[int(f)]
            loc = self.frac_mesh.complex.polygon_loops[fc]
            self._ambient_vertex.update(zip(loc, amb))

    # -- sizes ---------------------------------------------------------------

    @property
    def n_flux3(self) -> int:
        return self.dofmap.n_dofs

    @property
    def n_flux2(self) -> int:
        return self.frac_mesh.num_cells(1)

    @property
    def n_p3(self) -> int:
        return self.mesh.num_cells(3)

    @property
    def n_p2(self) -> int:
        return self.frac_mesh.num_cells(2)

    # -- blocks ---------------------------------------------------------------

    def coupling(self) -> sp.csr_matrix:
        """``C`` with ``(C u_3)_F`` = mass entering fracture cell ``F``.

        Row ``F`` holds ``s_E |f|`` for each side of its facet; with duplicated
        DOFs those are two distinct columns, so the row does not cancel.
        """
        area = self.mesh.geometry.measure(2)
        rows, cols, vals = [], [], []
        for fc, f in enumerate(self.facet_of_cell):
            for cell, _ in self.dofmap.sides(int(f)):
                rows.append(fc)
                cols.append(int(self.dofmap.dofs(cell, int(f))[0]))
                # pairs with the *integrated* flux DOF, so no area factor
                vals.append(float(self._signs[(cell, int(f))]))
        return sp.csr_matrix(
            (vals, (rows, cols)), shape=(self.n_p2, self.n_flux3)
        )

    def interface_stiffness(self) -> sp.csr_matrix:
        """``S = diag(1 / (kappa |f|))`` on the fracture flux DOFs (both sides).

        The Robin resistance is ``|f| / kappa`` against the facet-*average* flux;
        with the integrated DOF ``F = |f| F_avg`` the flux-flux pairing picks up
        ``1/|f|`` on each side, leaving ``1 / (kappa |f|)``.
        """
        area = self.mesh.geometry.measure(2)
        diag = np.zeros(self.n_flux3)
        for f in self.fracture.facets:
            for cell, _ in self.dofmap.sides(int(f)):
                diag[self.dofmap.dofs(cell, int(f))[0]] += 1.0 / (
                    self.fracture.kappa * area[f]
                )
        return sp.diags(diag, format="csr")

    def fracture_no_flow(self, dirichlet_facets) -> np.ndarray:
        """Fracture rim edges that carry no-flow rather than pressure data.

        A rim edge takes Dirichlet data only where the fracture actually reaches
        a Dirichlet part of the domain boundary.  Everything else -- an immersed
        **tip**, or a rim lying on a no-flow wall -- is a zero-flux condition,
        which is *essential* in the mixed form and so must be constrained.
        """
        on_dirichlet = set()
        for f in dirichlet_facets:
            on_dirichlet.update(self.mesh.complex.polygon_loops[int(f)])

        out = []
        for e in boundary_facets(self.frac_mesh):
            verts = self.frac_mesh.complex.edge_vertices[e]
            amb = [self._ambient_vertex[int(v)] for v in verts]
            if not all(v in on_dirichlet for v in amb):
                out.append(int(e))
        return np.array(out, dtype=np.int64)

    def dirichlet_facets(self, no_flow) -> np.ndarray:
        """Domain-boundary facets that carry pressure data."""
        return np.array(
            sorted(set(int(f) for f in boundary_facets(self.mesh)) - set(no_flow)),
            dtype=np.int64,
        )

    # -- right-hand sides ------------------------------------------------------

    def _boundary_data(self, mesh, dim, potential, n_dofs, only=None):
        """``g_e = s_e mean_e p_D`` on the facets carrying pressure data.

        The facet **mean**, not the integral: this pairs with the *integrated*
        flux DOF ``int_e F.n``, the convention that lets the discrete divergence
        be the bare signed incidence.
        """
        g = np.zeros(n_dofs)
        if potential is None:
            return g
        allowed = None if only is None else set(int(f) for f in only)
        for f, s in facet_cell_signs(mesh).items():
            if allowed is not None and f not in allowed:
                continue
            qp, qw = mesh.geometry.quadrature(dim - 1, f)
            g[f] = s * (qw @ np.asarray(potential(qp)).ravel()) / qw.sum()
        return g

    def _source(self, mesh, dim, source, scale=1.0):
        b = np.zeros(mesh.num_cells(dim))
        if source is None:
            return b
        for c in range(len(b)):
            qp, qw = mesh.geometry.quadrature(dim, c)
            b[c] = scale * (qw @ np.asarray(source(qp)).ravel())
        return b

    # -- assembly ---------------------------------------------------------------

    def assemble(
        self, dirichlet=None, no_flow=(), source=None, fracture_source=None
    ):
        """Return ``(A, rhs, constrained)`` for the mixed-dimensional system.

        ``no_flow`` lists domain-boundary facets carrying zero flux; the rest of
        the boundary takes ``dirichlet`` data.  Zero flux is *essential* in the
        mixed form, so those DOFs are constrained, not moved to the right-hand
        side.
        """
        M3 = self.inner3.assemble(self.dofmap) + self.interface_stiffness()
        B3 = discrete_divergence(self.mesh, self.dofmap)
        M2 = self.inner2.assemble()
        B2 = discrete_divergence(self.frac_mesh)
        C = self.coupling()

        A = sp.bmat(
            [
                [M3, None, -B3.T, C.T],
                [None, M2, None, -B2.T],
                [-B3, None, None, None],
                [C, -B2, None, None],
            ],
            format="csr",
        )

        dir_facets = self.dirichlet_facets(no_flow)
        g3 = self._boundary_data(self.mesh, 3, dirichlet, self.n_flux3, dir_facets)
        g2 = self._boundary_data(
            self.frac_mesh, 2, dirichlet, self.n_flux2,
            set(range(self.n_flux2)) - set(self.fracture_no_flow(dir_facets)),
        )
        b3 = self._source(self.mesh, 3, source)
        b2 = self._source(
            self.frac_mesh, 2, fracture_source, scale=self.fracture.aperture
        )
        rhs = np.concatenate([-g3, -g2, -b3, -b2])

        constrained = np.concatenate(
            [
                np.array([int(f) for f in no_flow], dtype=np.int64),
                self.n_flux3 + self.fracture_no_flow(dir_facets),
            ]
        )
        return A, rhs, constrained

    def solve(
        self, dirichlet=None, no_flow=(), source=None, fracture_source=None, **kwargs
    ):
        A, rhs, constrained = self.assemble(
            dirichlet, no_flow, source, fracture_source
        )
        if len(constrained):
            A, rhs = _constrain_to_zero(A, rhs, constrained)

        n1, n2 = self.n_flux3, self.n_flux2
        n3 = self.n_p3
        x = solve_saddle(A, rhs, (n1 + n2, n3 + self.n_p2), **kwargs)
        return MixedSolution(
            {
                "flux": x[:n1],
                "fracture_flux": x[n1 : n1 + n2],
                "pressure": x[n1 + n2 : n1 + n2 + n3],
                "fracture_pressure": x[n1 + n2 + n3 :],
            }
        )

    # -- diagnostics -----------------------------------------------------------

    def exchange(self, solution) -> np.ndarray:
        """Mass entering each fracture cell, ``Q = un+ + un-``."""
        return self.coupling() @ solution["flux"]

    def side_fluxes(self, solution, facet: int) -> tuple[float, float]:
        """The two outward normal fluxes on one fracture facet."""
        u = solution["flux"]
        return tuple(
            self._signs[(cell, int(facet))] * u[self.dofmap.dofs(cell, int(facet))[0]]
            for cell, _ in self.dofmap.sides(int(facet))
        )


def _constrain_to_zero(A: sp.csr_matrix, rhs: np.ndarray, dofs: np.ndarray):
    """Pin the given DOFs to zero by symmetric row/column elimination."""
    A = A.tolil(copy=True)
    rhs = rhs.copy()
    for d in dofs:
        A[d, :] = 0.0
        A[:, d] = 0.0
        A[d, d] = 1.0
        rhs[d] = 0.0
    return A.tocsr(), rhs
