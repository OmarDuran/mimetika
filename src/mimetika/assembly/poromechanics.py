r"""Quasi-static Biot poromechanics in fully mixed form.

Unknowns -- five fields, all of them primary:

    ``sigma``  total stress (traction moments per facet)
    ``u``      displacement (per cell)
    ``s``      rotation multiplier (per cell, weak symmetry)
    ``q``      Darcy flux (per facet)
    ``p``      pore pressure (per cell)

Governing equations (tension-positive, ``Sigma = C eps(u) - alpha p I``):

    ``div Sigma = f`` ,   ``d/dt ( alpha div u + p/M ) + div q = r`` ,
    ``q = -(k/mu_f) grad p`` .

Why the mixed form is robust where a displacement formulation is not
---------------------------------------------------------------------
Inverting the constitutive law gives ``eps(u) = C^{-1}(Sigma + alpha p I)``, so
the operator the scheme actually inverts is ``C^{-1}``, **not** ``C``.  Three
consequences, and they are exactly the three regimes that break naive schemes:

* **Incompressible solid** (``nu -> 1/2``).  ``C^{-1}`` stays bounded -- it tends
  to the deviatoric projector -- so there is no volumetric locking and nothing
  to stabilise.  ``lambda`` never appears; the compliance coefficient is
  ``a = nu/(1-2nu+d nu)``, which is ``1/d`` at ``nu = 1/2``.
* **Incompressible fluid** (``1/M = 0``).  The storage coefficient
  ``S = alpha^2/K + 1/M`` merely vanishes, turning the pressure row into a
  constraint.  The system is still a well-posed saddle point.
* **High contrast.**  Moduli and permeabilities are per cell and enter only
  through local inner products, so a jump of many orders of magnitude changes
  the scaling of individual blocks but not the structure.

The trace coupling
------------------
``C^{-1}(alpha p I) = (alpha/(dK)) p I``, so the pressure enters the stress row
through the **discrete trace** operator ``T`` with ``(T tau)_E = int_E tr(tau)``.
It is built from the same per-facet expansions as ``div_h`` and ``as_h`` -- the
identity plays the role the rigid rotations play for the asymmetry.

Assembled system (backward Euler over ``dt``; the flux row is scaled by ``-dt``
and the pressure row by ``dt`` so the whole thing stays **symmetric**)::

    [  M      D^T   A^T    0        Tc^T   ] [sigma]   [ g_u  ]
    [  D       0     0     0         0     ] [ u   ]   [ f    ]
    [  A       0     0     0         0     ] [ s   ] = [ 0    ]
    [  0       0     0  -dt M_q   dt B^T   ] [ q   ]   [-dt g_p]
    [  Tc      0     0   dt B      S|E|    ] [ p   ]   [ rhs  ]

with ``Tc = diag(alpha/(dK)) T``.  Setting ``dt = None`` drops the flow rows and
treats the pressure as **given data** -- the quasi-steady regime the fault
benchmarks use, where the pressure field is prescribed cell by cell.

``Tc`` couples each cell pressure to *every* facet DOF of that cell.  The
four-field variant (:class:`.four_field.FourFieldPoroMechanics`) carries the
solid pressure ``p_s = tr_h(sigma)/d`` as an explicit unknown, and there the
same physics enters through a **diagonal** cell--cell block instead -- ``Tc``
disappears from the matrix entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from mimetika.assembly.local import skew_generators
from mimetika.assembly.mixed import (
    MixedElasticity,
    MixedSolution,
    discrete_divergence,
    facet_cell_signs,
)
from mimetika.materials import Material
from mimetika.mesh.mesh import Mesh
from mimetika.operators.diffusion import DiffusionInnerProduct
from mimetika.solver.saddle import solve_saddle


def _rot_stab(inner):
    """Negated rotation stabilisation for the (s, s) block, or None."""
    hook = getattr(inner, "rotation_stabilization", None)
    return None if hook is None else -hook()


@dataclass
class PoroMechanics:
    """Biot poromechanics on a mesh with (possibly) per-cell materials."""

    mesh: Mesh
    material: Material
    contact: object = None  # optional FractureContact
    #: optional stress inner product (e.g. ``LumpedDeviatoricStress`` built with
    #: the same ``material``); ``None`` means AFW.  The caller owns the
    #: material consistency between the two.
    stress_inner: object = None

    #: which mixed-elasticity assembly to build the mechanics rows with; the
    #: four-field variant overrides this single hook.
    mechanics_class = MixedElasticity

    def __post_init__(self) -> None:
        d = self.mesh.dim
        self.d = d
        self.material = self.material.expand(self.mesh.num_cells(d))
        inner = (
            self.stress_inner
            if self.stress_inner is not None
            else _elasticity_with(self.mesh, self.material)
        )
        self.mechanics = self.mechanics_class(
            self.mesh, contact=self.contact, inner=inner
        )
        # the flow block must see the *per-cell* mobility k/mu_f, or a
        # permeability contrast would silently vanish from the system
        self.flow = DiffusionInnerProduct(
            self.mesh,
            K=self.material.mobility[:, None, None] * np.eye(3)[None],
        )
        self.n_skew = len(skew_generators(d))

    # -- sizes -----------------------------------------------------------------

    @property
    def n_stress(self) -> int:
        return self.mechanics.n_stress

    @property
    def n_cells(self) -> int:
        return self.mesh.num_cells(self.d)

    @property
    def n_flux(self) -> int:
        return self.mesh.num_cells(self.d - 1)

    # -- the trace operator ------------------------------------------------------

    def trace_operator(self) -> sp.csr_matrix:
        """``T`` with ``(T tau)_E = int_E tr(tau)``.

        Built exactly like ``as_h``, with the identity in place of the rigid
        rotations: ``tr_h(tau)_E = (1/|E|) sum_e int_e (tau n_e) . (x - x_E)``.

        Cached: it depends only on the geometry, not on the material or the time
        step, but ``assemble`` needs it every call and it loops over every cell.
        """
        if getattr(self, "_trace", None) is not None:
            return self._trace
        self._trace = self._build_trace_operator()
        return self._trace

    def _build_trace_operator(self) -> sp.csr_matrix:
        # T = diag(2 mu) W: the volumetric coupling every stress space exposes
        # *is* the trace, scaled by the compliance 1/2mu -- one builder, both
        # spaces, and the four-field split reuses the same object.
        ip = self.mechanics.inner
        W, _ = ip.volumetric_operator()
        return (sp.diags(2.0 * ip._mu) @ W).tocsr()

    # -- assembly ------------------------------------------------------------------

    def assemble(
        self,
        dt: float | None = None,
        dirichlet=None,
        pressure=None,
        body_force=None,
        pressure_bc=None,
        source=None,
        traction=None,
        traction_facets=(),
        roller_facets=(),
        no_flow=(),
        previous: MixedSolution | None = None,
    ):
        """Assemble ``(A, rhs)``.

        ``dt=None`` gives the quasi-steady problem with ``pressure`` supplied as
        data (a callable or one value per cell); otherwise the flow is solved
        along with the mechanics.

        Boundary conditions follow the mixed convention, which is the opposite
        of the primal one: a prescribed **pressure** is *natural* (it enters
        ``pressure_bc`` on the right-hand side) and a prescribed **flux** is
        *essential*.  So a facet left out of ``no_flow`` and given no
        ``pressure_bc`` is drained at ``p = 0``, not sealed; ``no_flow`` lists
        the facets whose flux DOF is pinned to zero.
        """
        M, D, A = self.mechanics.assemble_operators()
        T = self.trace_operator()
        coupling = sp.diags(self.material.pressure_coupling(self.d)) @ T

        g_u = self.mechanics.dirichlet_vector(dirichlet)
        f = self.mechanics.source_vector(body_force)
        zero_s = np.zeros(self.n_skew * self.n_cells)

        if dt is None:
            p = self._pressure_data(pressure)
            # the pressure is data: its coupling moves to the right-hand side
            S = sp.bmat([[M, D.T, A.T], [D, None, None], [A, None, None]], "csr")
            rhs = np.concatenate([g_u - coupling.T @ p, f, zero_s])
            S, rhs = self._apply_essential(
                S, rhs, traction, traction_facets, roller_facets, ()
            )
            return S, rhs, p

        vol = self.mesh.geometry.measure(self.d)
        Mq = self.flow.assemble()
        Bq = discrete_divergence(self.mesh)
        storage = sp.diags(self.material.storage(self.d) * vol)

        S = sp.bmat(
            [
                [M, D.T, A.T, None, coupling.T],
                [D, None, None, None, None],
                [A, None, _rot_stab(self.mechanics.inner), None, None],
                [None, None, None, -dt * Mq, dt * Bq.T],
                [coupling, None, None, dt * Bq, storage],
            ],
            format="csr",
        )
        prev = self._previous_terms(previous, coupling, storage)
        rhs = np.concatenate(
            [
                g_u,
                f,
                zero_s,
                -dt * self._flow_dirichlet(pressure_bc),
                prev + dt * self._source(source),
            ]
        )
        S, rhs = self._apply_essential(
            S, rhs, traction, traction_facets, roller_facets, no_flow
        )
        return S, rhs, None

    def _apply_essential(
        self, S, rhs, traction, traction_facets, roller_facets, no_flow
    ):
        """Pin the essential DOFs: tractions, roller shear, and sealed facets."""
        from mimetika.assembly.mixed import _constrain

        if len(traction_facets):
            dofs, values = self.mechanics.traction_moments(traction_facets, traction)
            S, rhs = _constrain(S, rhs, dofs, values)
        if len(roller_facets):
            dofs = self.mechanics.roller_dofs(roller_facets)
            S, rhs = _constrain(S, rhs, dofs, np.zeros(len(dofs)))
        if len(no_flow):
            sealed = self._flux_offset + np.asarray(sorted({int(f) for f in no_flow}))
            S, rhs = _constrain(S, rhs, sealed, np.zeros(len(sealed)))
        return S, rhs

    @property
    def _flux_offset(self) -> int:
        """Index of the first flux DOF in the assembled vector."""
        return self.n_stress + (self.d + self.n_skew) * self.n_cells

    def _pressure_data(self, pressure) -> np.ndarray:
        if pressure is None:
            return np.zeros(self.n_cells)
        if callable(pressure):
            centroids = self.mesh.geometry.centroids(self.d)
            return np.asarray(pressure(centroids), dtype=float).ravel()
        return np.broadcast_to(
            np.asarray(pressure, dtype=float), (self.n_cells,)
        ).copy()

    def _previous_terms(self, previous, coupling, storage) -> np.ndarray:
        if previous is None:
            return np.zeros(self.n_cells)
        return coupling @ previous["stress"] + storage @ previous["pressure"]

    def _flow_dirichlet(self, pressure_bc) -> np.ndarray:
        g = np.zeros(self.n_flux)
        if pressure_bc is None:
            return g
        for f, s in facet_cell_signs(self.mesh).items():
            qp, qw = self.mesh.geometry.quadrature(self.d - 1, f)
            g[f] = s * (qw @ np.asarray(pressure_bc(qp)).ravel())
        return g

    def _source(self, source) -> np.ndarray:
        b = np.zeros(self.n_cells)
        if source is None:
            return b
        for c in range(self.n_cells):
            qp, qw = self.mesh.geometry.quadrature(self.d, c)
            b[c] = qw @ np.asarray(source(qp)).ravel()
        return b

    # -- solve ------------------------------------------------------------------------

    def solve(self, dt: float | None = None, **kwargs) -> MixedSolution:
        solver = {
            k: kwargs.pop(k)
            for k in ("backend", "method", "rtol", "options", "preconditioner", "verbose")
            if k in kwargs
        }
        solver.setdefault("method", "direct")
        A, rhs, given_p = self.assemble(dt=dt, **kwargs)

        n1 = self.n_stress
        n2 = n1 + self.d * self.n_cells
        n3 = n2 + self.n_skew * self.n_cells
        if dt is None:
            x = solve_saddle(A, rhs, (n1, n3 - n1), **solver)
            return MixedSolution(
                {
                    "stress": x[:n1],
                    "displacement": x[n1:n2],
                    "rotation": x[n2:n3],
                    "pressure": given_p,
                    "flux": np.zeros(self.n_flux),
                }
            )
        n4 = n3 + self.n_flux
        # CPR split: everything against the **pressure**, which is the elliptic
        # field and is last and contiguous in [sigma, u, s, q, p].  The Schur
        # complement of this split is `B diag(M)^-1 B^T + S|E|` -- the cell-centred
        # pressure operator -- so AMG on it is exactly the CPR idea.
        #
        # The previous split, `(n_stress + n_flux, ...)`, named no field at all:
        # entries n1..n1+n_flux are displacement and rotation, not flux, because
        # q sits at offset n3.  That silently mis-targeted both the fieldsplit
        # preconditioner and the block scaling.
        x = solve_saddle(A, rhs, (n4, self.n_cells), **solver)
        return MixedSolution(
            {
                "stress": x[:n1],
                "displacement": x[n1:n2],
                "rotation": x[n2:n3],
                "flux": x[n3:n4],
                "pressure": x[n4:],
            }
        )

    # -- diagnostics ---------------------------------------------------------------------

    def volumetric_strain(self, solution) -> np.ndarray:
        """``div u = inv_modulus * ( tr(Sigma) + d alpha p )``.

        Read straight off ``eps(u) = C^{-1}(Sigma + alpha p I)``; it vanishes
        identically at ``nu = 1/2``, which *is* the incompressibility constraint.
        """
        vol = self.mesh.geometry.measure(self.d)
        tr = (self.trace_operator() @ solution["stress"]) / vol
        inv = self.material.inverse_modulus(self.d)
        return inv * (tr + self.d * self.material.biot * solution["pressure"])


def _elasticity_with(mesh: Mesh, material: Material):
    from mimetika.operators.elasticity import ElasticityInnerProduct

    return ElasticityInnerProduct(mesh, material=material)
