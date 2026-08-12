"""PoromechanicsSolver: the one interface for poromechanics with fractures.

The contract under test:

* fractures are a ``{set: ContactDriver}`` mapping -- a fracture implies
  contact, an empty mapping is plain unfractured poromechanics;
* ``flow="prescribed"`` takes the pore pressure as per-step data and its
  step must equal the hand-rolled driver path it centralizes;
* ``flow="solved"`` is the transient coupled Biot step, and its constant-dt
  fast path must agree with the per-step assembly it accelerates.
"""

import numpy as np
import pytest

from mimetika.contact import ContactDriver, SignoriniCoulomb
from mimetika.materials import Material
from mimetika.mesh import structured_box, structured_quads
from mimetika.mesh.fracture import facets_on_plane
from mimetika.simulation import (
    FlowBC,
    MechanicsBC,
    PoromechanicsIC,
    PoromechanicsSolver,
    RKTimeStepping,
)

MU, LAM = 1.0, 1.0
NU = LAM / (2.0 * (LAM + MU))


def material():
    return Material(shear_modulus=MU, poisson=NU, biot=0.9,
                    inverse_biot_modulus=0.1, permeability=1.0, viscosity=1.0)


def squeeze(amount=-0.05):
    """Boundary displacement compressing across the fracture plane."""

    def u(x):
        x = np.atleast_2d(x)
        out = np.zeros((len(x), 3))
        out[:, 0] = amount * x[:, 0]
        return out

    return u


# -- construction contract --------------------------------------------------------


def test_flow_mode_is_validated():
    mesh = structured_quads(2, 2)
    with pytest.raises(ValueError):
        PoromechanicsSolver(mesh, material(), flow="telepathic")


def test_prescribed_flow_admits_no_flow_bc():
    mesh = structured_quads(2, 2)
    with pytest.raises(ValueError):
        PoromechanicsSolver(mesh, material(),
                            flow_bc=FlowBC(flux_facets=[0]))


def test_only_direct_linear_solver_for_now():
    mesh = structured_quads(2, 2)
    with pytest.raises(ValueError):
        PoromechanicsSolver(mesh, material(), linear_solver="gmres")


def test_a_fracture_set_must_map_to_a_driver():
    mesh = structured_box(2, 2, 2)
    tags = facets_on_plane(mesh, [0.5, 0, 0], [1, 0, 0])
    with pytest.raises(TypeError):
        PoromechanicsSolver(mesh, material(), fractures={"fault": tags})


def test_multiple_fracture_sets_are_declared_not_silently_dropped():
    mesh = structured_box(2, 2, 2)
    tags = facets_on_plane(mesh, [0.5, 0, 0], [1, 0, 0])
    d1 = ContactDriver(mesh, tags, SignoriniCoulomb(), mu=MU, lam=LAM)
    d2 = ContactDriver(mesh, tags, SignoriniCoulomb(), mu=MU, lam=LAM)
    with pytest.raises(NotImplementedError):
        PoromechanicsSolver(mesh, material(),
                            fractures={"a": d1, "b": d2})


def test_solved_flow_rejects_contact_fractures_for_now():
    mesh = structured_box(2, 2, 2)
    tags = facets_on_plane(mesh, [0.5, 0, 0], [1, 0, 0])
    driver = ContactDriver(mesh, tags, SignoriniCoulomb(), mu=MU, lam=LAM)
    with pytest.raises(NotImplementedError):
        PoromechanicsSolver(mesh, material(), fractures={"fault": driver},
                            flow="solved")


def test_solved_flow_rejects_nonzero_prescribed_flux():
    mesh = structured_quads(2, 2)
    with pytest.raises(NotImplementedError):
        PoromechanicsSolver(mesh, material(), flow="solved",
                            flow_bc=FlowBC(flux=lambda x: np.ones(len(x)),
                                           flux_facets=[0]))


# -- prescribed flow, unfractured --------------------------------------------------


def unfractured():
    mesh = structured_quads(3, 3)
    solver = PoromechanicsSolver(
        mesh, material(),
        bc=MechanicsBC(dirichlet=squeeze()),
    )
    return mesh, solver


def test_unfractured_step_is_a_plain_solve_with_empty_contact():
    mesh, solver = unfractured()
    p = np.zeros(mesh.num_cells(2))
    state = solver.step(p)
    assert state.converged and state.jump.shape[0] == 0
    assert state.solution is not None
    assert len(state.solution["stress"]) == solver.mechanics(p)[0].n_stress


def test_fracture_readbacks_refuse_without_a_driver():
    _, solver = unfractured()
    with pytest.raises(ValueError):
        solver._require_driver()


def test_pressure_accepts_an_array_or_a_callable_of_centroids():
    mesh, solver = unfractured()
    by_array = solver.pressure_cells(np.full(mesh.num_cells(2), 3.0))
    by_call = solver.pressure_cells(lambda x: np.full(len(x), 3.0))
    assert np.array_equal(by_array, by_call)


def test_the_system_is_cached_and_the_rhs_delta_is_exact():
    mesh, solver = unfractured()
    p1 = np.zeros(mesh.num_cells(2))
    p2 = np.linspace(0.0, 5.0, mesh.num_cells(2))
    _, A1, _ = solver.mechanics(p1)
    problem2, A2, rhs2 = solver.mechanics(p2)  # cache hit: rhs by delta
    assert A1 is A2  # one assembly, one matrix
    fresh = PoromechanicsSolver(mesh, material(),
                                bc=MechanicsBC(dirichlet=squeeze()))
    _, _, rhs_ref = fresh.mechanics(p2)  # full assembly at p2
    assert np.allclose(rhs2, rhs_ref, rtol=0.0, atol=1e-12)


# -- prescribed flow, fractured ----------------------------------------------------


def fractured():
    mesh = structured_box(2, 2, 2)
    tags = facets_on_plane(mesh, [0.5, 0, 0], [1, 0, 0])
    driver = ContactDriver(mesh, tags, SignoriniCoulomb(friction=0.6),
                           mu=MU, lam=LAM)
    solver = PoromechanicsSolver(
        mesh, material(), fractures={"fault": driver},
        bc=MechanicsBC(dirichlet=squeeze()),
    )
    return mesh, tags, driver, solver


def test_fractured_step_equals_the_hand_rolled_driver_path():
    mesh, tags, driver, solver = fractured()
    p = np.zeros(mesh.num_cells(3))
    state = solver.step(p)
    assert state.converged

    # the reference: the same driver fed the same mechanics by hand
    reference = ContactDriver(mesh, tags, SignoriniCoulomb(friction=0.6),
                              mu=MU, lam=LAM)
    ref_solver = PoromechanicsSolver(
        mesh, material(), fractures={"fault": reference},
        bc=MechanicsBC(dirichlet=squeeze()),
    )
    ref = reference.solve_step(
        lambda contact=None: ref_solver.mechanics(p, contact=contact),
        solver="newton",
    )
    assert np.allclose(state.jump, ref.jump, atol=1e-12)
    assert np.allclose(state.multiplier, ref.multiplier, rtol=1e-10)


def test_compression_closes_the_fracture():
    mesh, tags, driver, solver = fractured()
    state = solver.step(np.zeros(mesh.num_cells(3)))
    jump = solver.fault_jump(state)
    assert np.abs(jump[:, 0]).max() < 1e-10  # no interpenetration, no gap
    total = solver.fault_tractions(state)
    assert total[:, 0].max() < 0.0  # compressive normal traction


def test_warm_start_from_locked_reaches_the_same_state():
    mesh, tags, driver, solver = fractured()
    p = np.zeros(mesh.num_cells(3))
    cold = solver.step(p)
    warm = solver.step(p, warm_from_locked=True)
    assert np.allclose(cold.jump, warm.jump, atol=1e-9)


def test_prestress_shifts_what_the_law_sees():
    mesh, tags, driver, solver = fractured()
    p = np.zeros(mesh.num_cells(3))
    pre = np.zeros((len(tags), 3))
    pre[:, 0] = -10.0  # strong in-situ compression: surely stuck
    state = solver.step(p, prestress=pre)
    assert state.converged
    assert np.allclose(driver.prestress[:, 0], -10.0)


def test_fracture_centroids_are_general_ambient_points():
    mesh, tags, driver, solver = fractured()
    pts = solver.fracture_centroids("fault")
    assert pts.shape == (len(tags), 3)
    assert np.allclose(pts[:, 0], 0.5)  # the plane the fault was cut on
    assert np.array_equal(pts, solver.fracture_centroids())


# -- solved flow -------------------------------------------------------------------


def column():
    """A tiny Terzaghi column: loaded top, sealed and confined elsewhere."""
    mesh = structured_quads(1, 6, lengths=(0.25, 1.5))
    from mimetika.assembly.mixed import boundary_facets

    centroids = mesh.geometry.centroids(1)
    boundary = boundary_facets(mesh)
    top = [f for f in boundary if abs(centroids[f][1] - 1.5) < 1e-9]
    confined = [f for f in boundary if f not in set(top)]

    load = np.zeros((3, 3))
    load[1, 1] = -1.0
    applied = lambda x: np.broadcast_to(  # noqa: E731
        load, (len(np.atleast_2d(x)), 3, 3))
    zero = lambda x: np.zeros((len(np.atleast_2d(x)), 3))  # noqa: E731
    bc = MechanicsBC(dirichlet=zero, traction=applied, traction_facets=top,
                     roller_facets=confined)
    flow_bc = FlowBC(flux_facets=confined)
    return mesh, bc, flow_bc


def test_solved_flow_steps_and_drains():
    mesh, bc, flow_bc = column()
    engine = PoromechanicsSolver(mesh, material(), flow="solved",
                                 bc=bc, flow_bc=flow_bc)
    early = engine.flow_step(previous=None, dt=1e-3)
    late = engine.flow_step(previous=early, dt=10.0)
    p_early = early["pressure"]
    assert p_early.min() > 0.0  # undrained: the fluid carries the push
    assert late["pressure"].max() < 0.1 * p_early.max()  # ...then it drains


def test_constant_dt_fast_path_matches_per_step_assembly():
    mesh, bc, flow_bc = column()
    dt = 0.05
    fast = PoromechanicsSolver(mesh, material(), flow="solved",
                               bc=bc, flow_bc=flow_bc,
                               time=RKTimeStepping(dt=dt))
    slow = PoromechanicsSolver(mesh, material(), flow="solved",
                               bc=bc, flow_bc=flow_bc)
    prev_fast = prev_slow = None
    for _ in range(3):
        prev_fast = fast.flow_step(previous=prev_fast)
        prev_slow = slow.flow_step(previous=prev_slow, dt=dt)
    assert np.allclose(prev_fast["pressure"], prev_slow["pressure"],
                       rtol=1e-8)
    assert np.allclose(prev_fast["displacement"], prev_slow["displacement"],
                       rtol=1e-8)


def test_solved_flow_needs_a_step_size():
    mesh, bc, flow_bc = column()
    engine = PoromechanicsSolver(mesh, material(), flow="solved",
                                 bc=bc, flow_bc=flow_bc)
    with pytest.raises(ValueError):
        engine.flow_step(previous=None)


def test_the_two_step_entry_points_guard_their_modes():
    mesh, bc, flow_bc = column()
    solved = PoromechanicsSolver(mesh, material(), flow="solved",
                                 bc=bc, flow_bc=flow_bc)
    with pytest.raises(ValueError):
        solved.step(np.zeros(mesh.num_cells(2)))
    prescribed = PoromechanicsSolver(mesh, material(), bc=bc)
    with pytest.raises(ValueError):
        prescribed.flow_step(previous=None, dt=0.1)
