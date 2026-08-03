r"""Four-field mimetic elasticity and poromechanics: the solid pressure as an
explicit unknown.

This is the **standard** formulation of the library -- four fields for
elasticity, six when coupled with flow.  The classic three-field system
(:class:`~mimetika.assembly.mixed.MixedElasticity`, five-field with flow) is
kept as the reference and example; the two are solution-identical by exact
congruence, which is what the tests in ``test_four_field.py`` pin.

The deviatoric/hydrostatic split of the complementary energy (Cockburn),

    ``sigma : C^{-1} sigma = (1/2mu) |sigma_dev|^2 + (d (1-ad)/2mu) p_s^2`` ,
    ``p_s = tr(sigma)/d`` ,   ``a = nu/(1-2nu+d nu)`` ,

suggests carrying ``p_s`` as a cell unknown next to the facet stress.  Two
things are bought by doing so:

* the **lumped** stress inner product stays diagonal in the assembled system --
  the three-field assembly must fold the volumetric rank-one term back into
  ``M``, filling in every facet pair of each cell;
* in Biot poromechanics the pore pressure couples to the mechanics through a
  **diagonal cell--cell block** instead of the trace operator, which couples a
  cell pressure to every facet DOF of that cell.  The ``(p_s, p_f)`` sub-block
  is ``2 x 2`` per cell -- what a CPR-type pressure Schur complement wants.

Why not the textbook arrangement
--------------------------------
The idealised four-field system pairs a *trace-free* stress ``sigma_dev`` with
``p_s`` through a hydrostatic embedding ``E`` and a zero ``(sigma_dev, p_s)``
block.  That arrangement needs ``M_dev E = 0`` and ``A E = 0`` **globally**,
and neither survives a conforming facet-DOF space:

* an interior facet is shared, so the embedded tractions of two neighbouring
  ``p_s`` values mix on it -- ``M_dev E`` and ``A E`` vanish cell by cell but
  not as matrix products;
* a *diagonal* ``M`` cannot annihilate the hydrostatic mode at all (its rows
  are positive multiples of the identity), and dropping the coupling anyway
  loses the patch test: the hydrostatic shadow of the constitutive row would
  enforce ``div u = 0`` instead of the hydrostatic law.

What is implemented instead is the exact congruent arrangement -- in the notes,
"eliminating ``p_s`` *before* the change of variables".  The facet unknown
remains the **total** stress, so ``D`` and ``A`` keep their meaning (and ``D``
stays purely topological); the split lives entirely in the compliance rows.

The exact four-field system
---------------------------
Every stress space here satisfies, cell by cell,

    ``M_full = M_dev + W^T diag(c) W`` ,  ``c_E = -2 mu a / |E| < 0`` ,

with ``w_E = R vec(I)`` the moment column of the hydrostatic mode -- i.e.
``W sigma = |E| tr_h(sigma) / 2mu``, the discrete trace scaled by the
compliance (:meth:`~mimetika.operators.lumped.LumpedDeviatoricStress.volumetric_operator`
and its AFW counterpart).  Introducing ``p_s`` through the *slack* identity
``p_s = (2mu/(|E| d)) W sigma`` and symmetrising gives, with

    ``Gamma = diag(-a d) W`` ,   ``B = diag(a d^2 |E| / 2mu)`` ,

the system

    [ M_dev   Gamma^T  D^T  A^T ] [sigma]   [g]
    [ Gamma   B        0    0   ] [p_s  ]   [0]
    [ D       0        0    0   ] [u    ] = [f]
    [ A       0        0    0   ] [s    ]   [0]

Eliminating ``p_s`` gives ``M_dev - Gamma^T B^{-1} Gamma = M_full`` **exactly**,
so the four-field solution coincides with the three-field one and the second
row evaluates ``p_s = tr_h(sigma)/d`` -- the first stress invariant is a primary
unknown (``I_1 = d p_s``).  The top-left ``2 x 2`` block is positive definite
whenever ``M_full`` is: its Schur complement *is* ``M_full``.

The construction degenerates only at ``a = 0`` (``nu = 0``): there the
volumetric compliance vanishes and ``p_s`` carries no energy, so the second row
would be identically zero -- such cells are rejected; use the three-field form.
A negative ``a`` (auxetic) is admissible: the congruence holds for any
``a != 0``, with the ``p_s`` diagonal turning negative.

Poromechanics: the diagonal Biot coupling
-----------------------------------------
In the three-field form the pore pressure enters the stress row through
``Tc^T = alpha (1-ad) W^T``.  Absorbing it into the slack variable,

    ``p_tilde = p_s - eta p_f`` ,   ``eta = 2 mu alpha (1/(2mu)) ... = 2 mu alpha inv_modulus / (a d)`` ,

removes ``p_f`` from the stress row altogether; the coupling reappears as the
**diagonal** block ``delta = alpha d |E| inv_modulus = alpha |E| / K`` between
``p_tilde`` and ``p_f``, plus a positive storage augmentation
``Delta = 2 mu alpha^2 inv_modulus^2 |E| / a = delta * eta``:

    [ M_dev  Gamma^T  D^T  A^T   0        0          ] [sigma  ]   [ g      ]
    [ Gamma  B        0    0     0        diag(delta)] [p_tilde]   [ 0      ]
    [ D      0        0    0     0        0          ] [u      ]   [ f      ]
    [ A      0        0    S_s   0        0          ] [s      ] = [ 0      ]
    [ 0      0        0    0    -dt M_q   dt B_q^T   ] [q      ]   [-dt g_p ]
    [ 0   diag(delta) 0    0     dt B_q   St+diag(Delta)] [p_f ]   [ rhs    ]

Eliminating ``p_tilde`` reproduces the five-field system verbatim (the
``Delta`` augmentation cancels against the elimination), so the two are
solution-identical; the honest solid pressure is recovered as
``p_s = p_tilde + eta p_f`` and reported in the solution.  In the quasi-steady
regime (``dt = None``) a prescribed pressure enters the right-hand side only
through ``-delta p`` on the ``p_tilde`` row -- one diagonal entry per cell,
instead of a trace-operator column against every facet DOF.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from mimetika.assembly.mixed import MixedElasticity, MixedSolution
from mimetika.assembly.poromechanics import PoroMechanics, _rot_stab
from mimetika.assembly.mixed import discrete_divergence
from mimetika.solver.saddle import solve_saddle


def _require_volumetric_energy(a: np.ndarray) -> None:
    """The split stores the hydrostatic energy on ``p_s``; ``a = 0`` has none.

    Only the *zero* is degenerate: at ``a = 0`` (``nu = 0``) the solid-pressure
    row would be identically zero.  A **negative** ``a`` (auxetic material) is
    fine -- the congruence with the three-field system is exact for any
    ``a != 0`` -- the ``p_s`` diagonal merely changes sign, so the ``(sigma,
    p_s)`` block is then indefinite rather than definite.  Direct solvers do
    not care; a block preconditioner assuming a definite leading block would.
    """
    a = np.asarray(a)
    if np.any(np.abs(a) < 1e-12):
        worst = int(np.argmin(np.abs(a)))
        raise ValueError(
            f"the four-field split needs a nonzero compliance coefficient "
            f"a = nu/(1-2nu+d nu) in every cell, but cell {worst} has "
            f"a = {float(a[worst]):.3e} (nu = 0).  With no volumetric "
            f"compliance the solid-pressure row is identically zero; use the "
            f"three-field formulation instead."
        )


class FourFieldElasticity(MixedElasticity):
    """Weakly-symmetric Hellinger--Reissner with an explicit solid pressure.

    Same interface, boundary handling and solution fields as
    :class:`~mimetika.assembly.mixed.MixedElasticity`, plus the cell unknown
    ``solid_pressure`` (``p_s = tr_h(sigma)/d``).  The solution is identical to
    the three-field one by exact congruence; what changes is the sparsity: the
    volumetric compliance is never folded into ``M``, so a lumped inner product
    stays diagonal.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pressure_blocks = None

    def assemble_operators(self):
        """``(M_dev, D, A)`` -- the inner product *without* the volumetric term.

        For the lumped space that is its own operator; for AFW the constant-
        hydrostatic compliance is stripped by adding back ``-W^T diag(c) W``
        (``c < 0``).  Either way ``M_full = M_dev + W^T diag(c) W`` holds
        exactly, which is what makes the four-field system congruent to the
        three-field one.
        """
        if self._ops is not None:
            return self._ops
        M, D, A = self._space_operators()
        if getattr(self.inner, "volumetric_included", True):
            W, cvec = self.inner.volumetric_operator()
            M = (M - W.T @ sp.diags(cvec) @ W).tocsr()
        self._ops = (M, D, A)
        return self._ops

    def solid_pressure_blocks(self):
        """``(Gamma, B)``: the ``p_s`` coupling ``diag(-a d) W`` and its diagonal.

        ``B = diag(a d^2 |E| / 2mu)``; the scaling is fixed by requiring both
        exactness (``Gamma^T B^{-1} Gamma = -W^T diag(c) W``) and an honest
        unknown (row two evaluating ``p_s = tr_h(sigma)/d``).
        """
        if self._pressure_blocks is not None:
            return self._pressure_blocks
        d = self.d
        a, mu = self.inner._a, self.inner._mu
        _require_volumetric_energy(a)
        vol = self.mesh.geometry.measure(d)
        W, _ = self.inner.volumetric_operator()
        Gamma = (sp.diags(-a * d) @ W).tocsr()
        B = sp.diags(a * d * d * vol / (2.0 * mu), format="csr")
        self._pressure_blocks = (Gamma, B)
        return self._pressure_blocks

    def assemble(self, body_force=None, dirichlet=None, extra_rhs=None):
        """``(A, rhs)`` of the four-field saddle-point system."""
        M, D, A = self.assemble_operators()
        Gamma, B = self.solid_pressure_blocks()
        blocks = [
            [M, Gamma.T, D.T, A.T],
            [Gamma, B, None, None],
            [D, None, None, None],
            [A, None, None, None],
        ]
        if self.n_skew == 0:
            blocks = [row[:-1] for row in blocks[:-1]]
        S = sp.bmat(blocks, format="csr")
        traction_rhs = self.dirichlet_vector(dirichlet)
        if extra_rhs is not None:
            traction_rhs = traction_rhs + np.asarray(extra_rhs, dtype=float)
        rhs = np.concatenate(
            [
                traction_rhs,
                np.zeros(self.n_cells),
                self.source_vector(body_force),
                np.zeros(self.n_skew * self.n_cells),
            ]
        )
        return S, rhs

    def constitutive_rows(self, contact: bool = True) -> sp.csr_matrix:
        """``[M_dev | Gamma^T | D^T | A^T]``, matching :meth:`split`.

        At any solution the solid-pressure row gives ``Gamma^T p_s =
        -Gamma^T B^{-1} Gamma sigma``, so this evaluates to the three-field
        ``M_full sigma + D^T u + A^T s`` -- the fracture jump the contact driver
        extracts is formulation-independent, as it must be.  ``contact=False``
        strips the fracture compliance, exactly as in the base class.
        """
        M, D, A = self.assemble_operators()
        if not contact and self.contact is not None:
            M = (M - self.contact.assemble(self.n_stress)).tocsr()
        Gamma, _ = self.solid_pressure_blocks()
        return sp.hstack([M, Gamma.T, D.T, A.T], format="csr")

    @property
    def block_sizes(self) -> tuple[int, int]:
        # (sigma, p_s) form the definite block: its Schur complement is M_full
        return (
            self.n_stress + self.n_cells,
            (self.d + self.n_skew) * self.n_cells,
        )

    def split(self, x: np.ndarray) -> MixedSolution:
        n1 = self.n_stress
        n1p = n1 + self.n_cells
        n2 = n1p + self.d * self.n_cells
        return MixedSolution(
            {
                "stress": x[:n1],
                "solid_pressure": x[n1:n1p],
                "displacement": x[n1p:n2],
                "rotation": x[n2:],
            }
        )

    def stress_invariants(self, solution) -> tuple[np.ndarray, np.ndarray]:
        """``(I_1, J_2)`` per cell, with ``I_1 = d p_s`` read off the unknowns.

        ``J_2 = (1/2) |dev sigma|^2`` still needs the cell-mean stress, but no
        trace reconstruction: the hydrostatic part is subtracted as ``p_s I``.
        """
        p = np.asarray(solution["solid_pressure"], dtype=float)
        S = self.cell_stress(solution["stress"])
        dev = S - p[:, None, None] * np.eye(self.d)
        return self.d * p, 0.5 * np.einsum("cij,cij->c", dev, dev)


class FourFieldPoroMechanics(PoroMechanics):
    """Biot poromechanics with the solid pressure explicit -- six fields.

    Solution-identical to :class:`~mimetika.assembly.poromechanics.PoroMechanics`
    by exact congruence; the matrix differs.  The pore pressure couples to the
    mechanics only through the diagonal ``(p_tilde, p_f)`` block, and the trace
    operator never enters the system (it is still used to evaluate the
    previous-step right-hand side).  The reported ``solid_pressure`` is the
    honest ``p_s = tr_h(sigma)/d = p_tilde + eta p_f``.
    """

    mechanics_class = FourFieldElasticity

    def _coupling_coefficients(self):
        """``(delta, eta, Delta)`` per cell (module docstring).

        ``delta = alpha |E| / K`` is the diagonal Biot block, ``eta`` the shift
        making ``p_tilde = p_s - eta p_f``, and ``Delta = delta * eta`` the
        storage augmentation that cancels on eliminating ``p_tilde``.
        """
        inner = self.mechanics.inner
        a, mu = inner._a, inner._mu
        _require_volumetric_energy(a)
        vol = self.mesh.geometry.measure(self.d)
        invm = self.material.inverse_modulus(self.d)
        alpha = self.material.biot
        delta = self.d * vol * alpha * invm
        eta = 2.0 * mu * alpha * invm / (a * self.d)
        return delta, eta, delta * eta

    @property
    def _flux_offset(self) -> int:
        # the solid-pressure block sits between the stress and the displacement
        return (
            self.n_stress
            + (1 + self.d + self.n_skew) * self.n_cells
        )

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
        """Assemble ``(A, rhs)``; same contract as the five-field version."""
        M, D, A = self.mechanics.assemble_operators()
        Gamma, B = self.mechanics.solid_pressure_blocks()
        delta, _, Delta = self._coupling_coefficients()

        g_u = self.mechanics.dirichlet_vector(dirichlet)
        f = self.mechanics.source_vector(body_force)
        zero_s = np.zeros(self.n_skew * self.n_cells)

        if dt is None:
            p = self._pressure_data(pressure)
            # the pressure is data: it reaches the mechanics only through the
            # diagonal delta on the solid-pressure row
            S = sp.bmat(
                [
                    [M, Gamma.T, D.T, A.T],
                    [Gamma, B, None, None],
                    [D, None, None, None],
                    [A, None, None, None],
                ],
                format="csr",
            )
            rhs = np.concatenate([g_u, -delta * p, f, zero_s])
            S, rhs = self._apply_essential(
                S, rhs, traction, traction_facets, roller_facets, ()
            )
            return S, rhs, p

        vol = self.mesh.geometry.measure(self.d)
        Mq = self.flow.assemble()
        Bq = discrete_divergence(self.mesh)
        storage = sp.diags(self.material.storage(self.d) * vol)
        Dd = sp.diags(delta)

        S = sp.bmat(
            [
                [M, Gamma.T, D.T, A.T, None, None],
                [Gamma, B, None, None, None, Dd],
                [D, None, None, None, None, None],
                [A, None, None, _rot_stab(self.mechanics.inner), None, None],
                [None, None, None, None, -dt * Mq, dt * Bq.T],
                [None, Dd, None, None, dt * Bq, storage + sp.diags(Delta)],
            ],
            format="csr",
        )
        # the previous-step functional is unchanged by the congruence, so it is
        # evaluated exactly as in the five-field form
        coupling = sp.diags(self.material.pressure_coupling(self.d)) @ self.trace_operator()
        prev = self._previous_terms(previous, coupling, storage)
        rhs = np.concatenate(
            [
                g_u,
                np.zeros(self.n_cells),
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

    def solve(self, dt: float | None = None, **kwargs) -> MixedSolution:
        solver = {
            k: kwargs.pop(k)
            for k in ("backend", "method", "rtol", "options", "preconditioner", "verbose")
            if k in kwargs
        }
        solver.setdefault("method", "direct")
        A, rhs, given_p = self.assemble(dt=dt, **kwargs)
        _, eta, _ = self._coupling_coefficients()

        n1 = self.n_stress
        n1p = n1 + self.n_cells
        n2 = n1p + self.d * self.n_cells
        n3 = n2 + self.n_skew * self.n_cells
        if dt is None:
            x = solve_saddle(A, rhs, (n1p, n3 - n1p), **solver)
            return MixedSolution(
                {
                    "stress": x[:n1],
                    "solid_pressure": x[n1:n1p] + eta * given_p,
                    "displacement": x[n1p:n2],
                    "rotation": x[n2:n3],
                    "pressure": given_p,
                    "flux": np.zeros(self.n_flux),
                }
            )
        n4 = n3 + self.n_flux
        # CPR split: everything against the pore pressure, which stays last
        x = solve_saddle(A, rhs, (n4, self.n_cells), **solver)
        p_f = x[n4:]
        return MixedSolution(
            {
                "stress": x[:n1],
                "solid_pressure": x[n1:n1p] + eta * p_f,
                "displacement": x[n1p:n2],
                "rotation": x[n2:n3],
                "flux": x[n3:n4],
                "pressure": p_f,
            }
        )
