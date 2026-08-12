"""One interface for a poromechanics PDE solve with frictional fractures.

Everything a simulation is made of is stated once, up front:

* the **mesh** with its **fractures**: a ``{fracture set: ContactDriver}``
  mapping.  A key names a *fracture set* -- one or several fractures
  treated as a single contact unit -- and fractures *imply* contact, so
  every set carries its own driver: its facets and its own contact law
  (constitutive map).  Different sets may use different laws.  The contact
  algebra (:class:`~mimetika.contact.driver.ContactDriver`) stays cleanly
  separated from the PDE side.  No fractures means a plain unfractured
  poromechanics simulation, and no driver is needed;
* the **boundary conditions**: :class:`MechanicsBC` for the mechanics and
  :class:`FlowBC` for the flow;
* the **flow mode**: ``flow="prescribed"`` (the quasi-steady one-way
  coupling -- the pore pressure is per-step *data* handed to
  :meth:`PoromechanicsSolver.step`, and the flow BC is empty) or
  ``flow="solved"`` (a Darcy step coupled to the mechanics; reserved, not
  implemented yet);
* the **initial conditions** (:class:`PoromechanicsIC`): the pore-pressure
  field, the in-situ fault prestress, and optionally a starting contact
  state;
* the **time step** ``dt`` (constant, for now) and the **linear solver**
  type (``"direct"``, for now).

The solver owns everything derived from those inputs: the assembled
four-field system and its cache, the pressure-dependent right-hand side
(the Biot coupling is the only place the flow enters, additively on the
stress block), the condensed contact map and its factorization reuse, the
locked (pre-slip) solve and the warm start built from it, and the fault
readbacks.  A time/loading program -- which pressures to apply, in which
order, with which continuation policy -- belongs to the caller; the solver
advances one step at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import scipy.sparse as sp

from mimetika.assembly.four_field import FourFieldElasticity
from mimetika.assembly.poromechanics import PoroMechanics
from mimetika.contact import ContactState
from mimetika.solver.saddle import solve_saddle


@dataclass
class MechanicsBC:
    """Boundary data of the mechanics problem.

    ``traction`` is essential in the Hellinger--Reissner form (the traction
    is a degree of freedom) and may return either the stress tensor
    ``(nq, 3, 3)`` -- the safe choice -- or the traction vector against the
    canonical facet normal.  ``dirichlet`` is the boundary displacement
    datum, entering naturally.  ``pins`` are discrete solution DOFs to
    constrain to zero (rigid-mode removal for all-traction problems).
    """

    traction: Callable | None = None
    traction_facets: Sequence[int] = ()
    roller_facets: Sequence[int] = ()
    dirichlet: Callable | None = None
    body_force: Callable | None = None
    pins: Sequence[int] = ()


@dataclass
class FlowBC:
    """Boundary data of the flow problem.

    In the mixed (flux-pressure) form the flux is the degree of freedom, so
    a prescribed **flux** is essential and a prescribed **pressure** enters
    naturally -- the mirror image of the mechanics.  The current solver
    takes the pore pressure as *prescribed data* per step (the
    quasi-steady one-way coupling of the benchmark model), so a non-trivial
    ``FlowBC`` marks a solved-flow simulation, which is not implemented
    yet; the field exists so the interface states the full problem.
    """

    flux: Callable | None = None
    flux_facets: Sequence[int] = ()
    pressure: Callable | None = None
    pressure_facets: Sequence[int] = ()

    def is_trivial(self) -> bool:
        return (self.flux is None and self.pressure is None
                and not len(self.flux_facets)
                and not len(self.pressure_facets))


@dataclass
class PoromechanicsIC:
    """Initial data of the simulation.

    ``pressure``: the initial pore-pressure field (per-cell array, or a
    callable of the cell centroids).  ``prestress``: the in-situ fault
    traction in the driver's facet frame -- an ``(nf, d)`` array or a
    callable ``(mesh, facets, pressure_cells) -> (nf, d)`` when it depends
    on the current pressure (a Biot term on the normal).  ``state``: a
    :class:`~mimetika.contact.driver.ContactState` to continue from.
    """

    pressure: np.ndarray | Callable | None = None
    prestress: np.ndarray | Callable | None = None
    state: object = None


class PoromechanicsSolver:
    """Poromechanics with contact on the active fractures, one step at a time.

    Parameters
    ----------
    mesh
        The (conforming) mesh; fracture facets are ordinary facets of it.
    material
        The :class:`~mimetika.materials.Material` (drained moduli, Biot).
    fractures
        ``{fracture set: ContactDriver}`` -- each key names a *fracture
        set* (one or several fractures treated as a single contact unit;
        any hashable label, e.g. a string or a tuple of fracture names).
        Fractures imply contact, so every set carries its own driver over
        the concatenation of its facets, with its own contact law -- sets
        may differ in their constitutive maps.  An empty mapping (or
        ``None``) means a plain unfractured poromechanics simulation, which
        needs no driver at all.
    bc, ic
        :class:`MechanicsBC` and :class:`PoromechanicsIC`.
    dt
        Constant time step; ``0`` means quasi-static (no rate effects).
    linear_solver
        ``"direct"`` (the only option, for now).
    inner
        The stress inner product; defaults to the consistency-only
        :class:`~mimetika.operators.derham.DeRhamDeviatoricStress`.
    cache
        A mutable dict carrying the per-mesh state (assembled system,
        pressure coupling, condensed contact map).  Passing the same dict
        across solvers on the same mesh shares the factorization.
    """

    def __init__(
        self,
        mesh,
        material,
        fractures: Mapping[str, object] | None = None,
        bc: MechanicsBC | None = None,
        flow: str = "prescribed",
        flow_bc: FlowBC | None = None,
        ic: PoromechanicsIC | None = None,
        dt: float = 0.0,
        linear_solver: str = "direct",
        inner=None,
        cache: dict | None = None,
    ) -> None:
        if linear_solver != "direct":
            raise ValueError("only linear_solver='direct' is supported")
        if flow not in ("prescribed", "solved"):
            raise ValueError("flow must be 'prescribed' or 'solved'")
        if flow == "solved":
            raise NotImplementedError(
                "flow='solved' (a Darcy step coupled to the mechanics) is "
                "reserved but not implemented yet"
            )
        self.flow = flow
        self.flow_bc = FlowBC() if flow_bc is None else flow_bc
        if flow == "prescribed" and not self.flow_bc.is_trivial():
            raise ValueError(
                "flow='prescribed' takes the pore pressure as per-step data "
                "and admits no flow boundary conditions; use flow='solved' "
                "(once available) for a flow boundary-value problem"
            )
        self.mesh = mesh
        self.material = material
        self.fractures = {} if fractures is None else dict(fractures)
        self.bc = MechanicsBC() if bc is None else bc
        self.ic = PoromechanicsIC() if ic is None else ic
        self.dt = float(dt)
        self.linear_solver = linear_solver
        self.cache = {} if cache is None else cache
        if inner is None:
            from mimetika.operators.derham import DeRhamDeviatoricStress

            inner = DeRhamDeviatoricStress(mesh, material=material)
        self.inner = inner

        for key, driver in self.fractures.items():
            if not hasattr(driver, "solve_step"):
                raise TypeError(
                    f"fracture set {key!r} must map to a ContactDriver -- "
                    "fractures imply contact mechanics"
                )
        if len(self.fractures) > 1:
            raise NotImplementedError(
                "one fracture set per solve for now: group fractures that "
                "share a law into one set (one driver over their "
                "concatenated facets); the multi-set coupled step is the "
                "composite-driver extension"
            )
        self.driver = next(iter(self.fractures.values()), None)

    # -- inputs -----------------------------------------------------------------

    @property
    def active_facets(self) -> np.ndarray:
        if not self.fractures:
            return np.empty(0, dtype=int)
        return np.concatenate(
            [np.asarray(d.facets, dtype=int) for d in self.fractures.values()]
        )

    def pressure_cells(self, pressure) -> np.ndarray:
        """Normalize a pressure input to one value per cell."""
        if callable(pressure):
            d = self.mesh.dim
            return np.asarray(
                pressure(self.mesh.geometry.centroids(d)), dtype=float
            ).ravel()
        return np.asarray(pressure, dtype=float).ravel()

    def prestress_values(self, pressure_cells, prestress=None) -> np.ndarray | None:
        """The in-situ fault traction for this step, ``(nf, d)`` or ``None``."""
        pre = self.ic.prestress if prestress is None else prestress
        if pre is None:
            return None
        if callable(pre):
            pre = pre(self.mesh, self.active_facets, pressure_cells)
        return np.asarray(pre, dtype=float)

    # -- the assembled mechanics ------------------------------------------------

    def coupling(self) -> sp.spmatrix:
        """The Biot pressure coupling ``C`` (cached): ``rhs -= C^T p``."""
        if "coupling" not in self.cache:
            poro = PoroMechanics(self.mesh, self.material)
            self.cache["coupling"] = (
                sp.diags(poro.material.pressure_coupling(self.mesh.dim))
                @ poro.trace_operator()
            )
        return self.cache["coupling"]

    def mechanics(self, pressure_cells, contact=None):
        """``(problem, matrix, rhs)`` at the given pore pressure.

        The system matrix never depends on the pressure: on a cache hit only
        the rhs is rebuilt, additively through ``extra = -(C^T p)`` on the
        stress block, with the boundary-pinned rows left untouched (they
        carry BC data, not load).
        """
        extra = -(self.coupling().T @ np.asarray(pressure_cells, dtype=float))

        if contact is None and "system" in self.cache:
            problem, matrix, rhs0, extra0, pinned = self.cache["system"]
            delta = np.zeros_like(rhs0)
            delta[: len(extra)] = np.asarray(extra - extra0).ravel()
            delta[pinned] = 0.0
            return problem, matrix, rhs0 + delta

        bc = self.bc
        problem = FourFieldElasticity(self.mesh, contact=contact,
                                      inner=self.inner)
        matrix, rhs = problem.assemble_constrained(
            body_force=bc.body_force,
            dirichlet=bc.dirichlet,
            extra_rhs=extra,
            traction=bc.traction,
            traction_facets=bc.traction_facets,
            roller_facets=bc.roller_facets,
        )
        if len(bc.pins):
            from mimetika.assembly.mixed import _constrain

            matrix, rhs = _constrain(matrix.tocsr(), rhs,
                                     np.asarray(bc.pins, dtype=np.int64),
                                     np.zeros(len(bc.pins)))
        if contact is None:
            pinned = []
            if len(bc.traction_facets):
                pins, _ = problem.traction_moments(bc.traction_facets,
                                                   bc.traction)
                pinned.append(np.asarray(pins, dtype=np.int64))
            if len(bc.roller_facets):
                pinned.append(problem.roller_dofs(bc.roller_facets))
            if len(bc.pins):
                pinned.append(np.asarray(bc.pins, dtype=np.int64))
            pinned = (np.concatenate(pinned) if pinned
                      else np.empty(0, dtype=np.int64))
            self.cache["system"] = (problem, matrix, rhs, extra, pinned)
        return problem, matrix, rhs

    # -- solves -----------------------------------------------------------------

    def locked_solution(self, pressure_cells):
        """The **locked** (no-contact) solve: ``(problem, solution, rhs)``.

        The plain continuum under the load -- the pre-slip state, and the
        basis of the warm start.
        """
        problem, matrix, rhs = self.mechanics(pressure_cells)
        solution = problem.split(
            solve_saddle(matrix, rhs, problem.block_sizes,
                         method=self.linear_solver)
        )
        return problem, solution, rhs

    def warm_state_from_locked(self, pressure_cells) -> ContactState:
        """A contact state whose multiplier is the locked fault tractions.

        On stiff, confined domains the contact Newton diverges from a zero
        multiplier (the first trial has the fault carrying nothing); the
        locked tractions sit next to the contact solution and keep the
        iteration in the physical basin.
        """
        driver = self._require_driver()
        _, sol0, _ = self.locked_solution(pressure_cells)
        xstar = driver.tractions(sol0["stress"])
        mult = (driver.moment_operator() @ xstar.ravel()).reshape(
            len(driver.facets), driver.ndf)
        s0 = driver.initial_state()
        return ContactState(multiplier=mult, internal=s0.internal,
                            jump=s0.jump)

    def step(
        self,
        pressure,
        state: ContactState | None = None,
        prestress=None,
        warm_from_locked: bool = False,
        recover: bool = True,
        solver: str = "newton",
    ) -> ContactState:
        """Advance one (quasi-)static step at the given pore pressure.

        ``state`` continues from a previous step; ``prestress`` overrides
        the IC's in-situ fault traction for this step;
        ``warm_from_locked`` builds the initial multiplier from the locked
        solve when no state is given; ``recover=False`` skips the field
        recovery for loops that only read the jump.  The condensed contact
        map is cached and reused across steps -- the factorization is paid
        once per mesh.

        Without active fractures there is no contact: the step is the plain
        poromechanics solve, returned as a state with empty contact fields.
        """
        if self.driver is None:
            p = self.pressure_cells(pressure)
            problem, solution, _ = self.locked_solution(p)
            return ContactState(
                multiplier=np.zeros((0, 0)),
                internal=np.zeros((0, 0)),
                jump=np.zeros((0, self.mesh.dim)),
                solution=solution,
                problem=problem,
                iterations=1,
                converged=True,
            )
        driver = self._require_driver()
        p = self.pressure_cells(pressure)
        pre = self.prestress_values(p, prestress)
        if pre is not None:
            driver.prestress = driver.expand_to_points(pre)
        if state is None:
            state = self.ic.state
        if state is None and warm_from_locked:
            state = self.warm_state_from_locked(p)

        def mech(contact=None):
            return self.mechanics(p, contact=contact)

        state = driver.solve_step(
            mech, state=state, solver=solver,
            dt=self.dt if self.dt > 0.0 else None,
            reuse=self.cache.get("condensed"),
            recover=recover,
            method=self.linear_solver,
        )
        if state.condensed is not None:
            self.cache["condensed"] = state.condensed
        return state

    # -- readbacks --------------------------------------------------------------

    def fracture_centroids(self, key=None) -> np.ndarray:
        """Ambient centroids of fracture facets, ``(nf, 3)``.

        ``key`` selects one fracture set; ``None`` gives all active facets
        in driver order.  What coordinate to read off is the caller's
        business -- the interface knows nothing about the geometry's
        orientation.
        """
        facets = (np.asarray(self.fractures[key].facets, dtype=int)
                  if key is not None else self.active_facets)
        return self.mesh.geometry.centroids(self.mesh.dim - 1)[facets]

    def fault_tractions(self, state) -> np.ndarray:
        """Total facet-frame traction per facet: prestress + solved increment."""
        driver = self._require_driver()
        return driver.per_facet(
            driver.prestress + driver.tractions(state.solution["stress"])
        )

    def fault_jump(self, state) -> np.ndarray:
        """Facet-frame jump per facet (facet means)."""
        return self._require_driver().per_facet(state.jump)

    def _require_driver(self):
        if self.driver is None:
            raise ValueError(
                "this is a fracture readback and the solver has no contact "
                "driver -- it was built without active fractures, so the "
                "simulation is plain unfractured poromechanics"
            )
        return self.driver
