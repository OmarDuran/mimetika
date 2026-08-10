r"""Fully mixed Biot poromechanics, and the three robustness limits.

The benchmarks this solver exists for (Novikov et al., fault reactivation) run
at parameter values that break a displacement-based scheme, so the three limits
are checked here as first-class requirements rather than as afterthoughts:

===========================  ===================================================
requirement                  why the mixed form survives it
===========================  ===================================================
``nu = 1/2``                 the scheme inverts ``C^{-1}``, never ``C``; the
  (incompressible solid)     compliance ``a = nu/(1-2nu+d nu) -> 1/d`` is finite
                             and ``lambda`` never appears
``1/M = 0``                  the storage coefficient merely vanishes, turning the
  (incompressible fluid)     pressure row into a constraint
high contrast                moduli are per cell and enter only through local
                             inner products
===========================  ===================================================

Each is checked in **2D and 3D**, on **simplices and polytopes**, against a
closed-form answer -- not merely for "it ran".
"""

import numpy as np
import pytest

from mimetika.assembly.mixed import boundary_facets
# The whole module runs the **standard** formulation -- the four-field split
# with the solid pressure explicit.  The robustness limits below are therefore
# statements about the formulation users actually get; the classic five-field
# system keeps its coverage through the congruence tests in test_four_field.py.
from mimetika.assembly.four_field import FourFieldPoroMechanics as PoroMechanics
from mimetika.materials import Material, compliance_coefficient
from mimetika.mesh import (
    structured_box,
    structured_quads,
    structured_tets,
    structured_triangles,
)

MESHES = {
    "hex": (lambda: structured_box(2, 2, 2), 3),
    "tet": (lambda: structured_tets(1, 1, 1), 3),
    "quad": (lambda: structured_quads(2, 2), 2),
    "tri": (lambda: structured_triangles(2, 2), 2),
}
ALL = list(MESHES)
SIMPLEX = ["tet", "tri"]
# 0.5 exactly is included on purpose: it is the limit the benchmarks need
POISSON = [0.15, 0.45, 0.49999, 0.5]


def uniform(name, nu, biot=0.9, pressure=-25.0, shear=1.0):
    """A uniform total-stress state and the displacement that produces it.

    Returns ``(problem, stress_fn, displacement_fn, strain)`` where the stress is
    constant, so the discrete solution must reproduce it exactly for *any*
    Poisson ratio -- there is no discretisation error to hide a locking failure.
    """
    mk, d = MESHES[name]
    mesh = mk()
    material = Material(shear_modulus=shear, poisson=nu, biot=biot)
    problem = PoroMechanics(mesh, material)

    full = np.zeros((3, 3))
    full[:d, :d] = np.diag([3.0, -1.0, -2.0])[:d, :d]
    a = compliance_coefficient(d, nu)
    total = full[:d, :d] + biot * pressure * np.eye(d)
    strain = (total - a * np.trace(total) * np.eye(d)) / (2 * shear)

    grad = np.zeros((3, 3))
    grad[:d, :d] = strain

    def stress(x):
        return np.broadcast_to(full, (len(np.atleast_2d(x)), 3, 3))

    def displacement(x):
        return np.atleast_2d(x) @ grad.T

    return problem, stress, displacement, strain


def face(problem, value=1.0, axis=1):
    """Boundary facets on the plane ``x[axis] == value``."""
    centroids = problem.mesh.geometry.centroids(problem.d - 1)
    return [
        f for f in boundary_facets(problem.mesh)
        if abs(centroids[f][axis] - value) < 1e-12
    ]


# -- the trace operator ---------------------------------------------------------


@pytest.mark.parametrize("name", ALL)
def test_trace_operator_reproduces_the_trace_exactly(name):
    """``(T tau)_E / |E| = tr(tau)`` -- the only new operator poromechanics adds."""
    problem, _, _, _ = uniform(name, 0.25)
    d = problem.d
    full = np.zeros((3, 3))
    full[:d, :d] = np.array([[2.0, 0.7, -0.3], [0.7, -1.0, 0.2], [-0.3, 0.2, 4.0]])[
        :d, :d
    ]
    dofs = problem.mechanics.interpolate_stress(
        lambda x: np.broadcast_to(full, (len(np.atleast_2d(x)), 3, 3))
    )
    trace = (problem.trace_operator() @ dofs) / problem.mesh.geometry.measure(d)
    assert np.allclose(trace, np.trace(full[:d, :d]), atol=1e-12)


def test_trace_operator_annihilates_a_deviatoric_state():
    """A traceless stress must give exactly zero -- no spurious volumetric coupling."""
    problem, _, _, _ = uniform("hex", 0.25)
    dev = np.diag([1.0, 1.0, -2.0])
    dofs = problem.mechanics.interpolate_stress(
        lambda x: np.broadcast_to(dev, (len(np.atleast_2d(x)), 3, 3))
    )
    assert np.allclose(problem.trace_operator() @ dofs, 0.0, atol=1e-12)


# -- traction boundary conditions ------------------------------------------------


@pytest.mark.parametrize("name", ALL)
def test_traction_moments_match_the_interpolant(name):
    """Prescribed traction DOFs must equal what ``interpolate_stress`` produces.

    Regression guard.  The two used different normal conventions -- one applied
    the incidence sign on top of a caller-supplied ambient traction -- which
    cancelled on facets with ``s = +1`` and silently flipped the sign elsewhere.
    """
    problem, stress, _, _ = uniform(name, 0.25)
    facets = face(problem)
    dofs, values = problem.mechanics.traction_moments(facets, stress)
    assert np.allclose(values, problem.mechanics.interpolate_stress(stress)[dofs])


@pytest.mark.parametrize("name", ALL)
def test_vector_and_tensor_traction_agree(name):
    """The ``(nq, 3)`` form must match the ``(nq, 3, 3)`` form on the same facets.

    The vector form is taken against the *canonical* facet normal, so the two
    agree only if that convention is honoured -- including on facets whose
    incidence sign is negative.
    """
    problem, stress, _, _ = uniform(name, 0.25)
    facets = face(problem)
    full = stress(np.zeros((1, 3)))[0]
    geometry = problem.mesh.geometry

    for f in facets:
        normal = geometry.facet_frame(int(f))[0]
        _, tensor = problem.mechanics.traction_moments([f], stress)
        _, vector = problem.mechanics.traction_moments(
            [f], lambda x, n=normal: np.broadcast_to(full @ n, (len(np.atleast_2d(x)), 3))
        )
        assert np.allclose(tensor, vector, atol=1e-12)


# -- requirement 1: incompressible solid ------------------------------------------


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("nu", POISSON)
def test_uniform_stress_is_exact_up_to_the_incompressible_limit(name, nu):
    """No locking: the error must not grow as ``nu -> 1/2``, and must hold *at* 1/2."""
    problem, stress, displacement, strain = uniform(name, nu)
    solution = problem.solve(
        dt=None,
        dirichlet=displacement,
        pressure=-25.0,
        traction=stress,
        traction_facets=face(problem),
    )
    exact = problem.mechanics.interpolate_stress(stress)
    error = np.abs(solution["stress"] - exact).max() / np.abs(exact).max()
    assert error < 1e-11, f"{name} nu={nu}: rel stress error {error:.3e}"
    assert np.allclose(
        problem.volumetric_strain(solution), np.trace(strain), atol=1e-11
    )


@pytest.mark.parametrize("name", ALL)
def test_divergence_free_exactly_at_one_half(name):
    """``div u`` is not merely small at ``nu = 1/2`` -- it is identically zero."""
    problem, stress, displacement, _ = uniform(name, 0.5)
    solution = problem.solve(
        dt=None,
        dirichlet=displacement,
        pressure=-25.0,
        traction=stress,
        traction_facets=face(problem),
    )
    assert np.abs(problem.volumetric_strain(solution)).max() == 0.0


@pytest.mark.parametrize("name", ALL)
def test_compliance_null_space_is_hydrostatic_at_one_half(name):
    """``C^{-1}`` becomes the deviatoric projector, so ``M_E`` *must* be singular.

    A positive-definite local inner product at ``nu = 1/2`` would be wrong.  The
    null space has one hydrostatic direction per scalar mode, ``d + 1`` of them,
    on simplices and polytopes alike -- the stabilization acts on ``ker(N^T)``,
    which is a different subspace, and does not fill it in.
    """
    from mimetika.operators.elasticity import ElasticityInnerProduct
    from mimetika.operators.inner_product import assemble_local_inner_product

    mk, d = MESHES[name]
    inner = ElasticityInnerProduct(
        mk(), material=Material(shear_modulus=1.0, poisson=0.5)
    )
    N, R, Kbar, volume, _ = inner.local_matrices(0)
    local = assemble_local_inner_product(N, R, Kbar, volume)

    nullity = lambda w: int((np.abs(w) < 1e-10 * np.abs(w).max()).sum())  # noqa: E731
    assert nullity(np.linalg.eigvalsh(Kbar)) == d + 1
    assert nullity(np.linalg.eigvalsh(local)) == d + 1
    # strong consistency survives the limit: this is what the pseudo-inverse buys
    assert np.abs(local @ N - R).max() < 1e-11


@pytest.mark.parametrize("name", ALL)
def test_saddle_point_stays_nonsingular_at_one_half(name):
    """The singular ``M`` is harmless: the constraint blocks control it."""
    problem, stress, _, _ = uniform(name, 0.5)
    matrix, _, _ = problem.assemble(
        dt=None,
        dirichlet=lambda x: np.zeros((len(np.atleast_2d(x)), 3)),
        pressure=-25.0,
        traction=stress,
        traction_facets=face(problem),
    )
    singular_values = np.linalg.svd(matrix.toarray(), compute_uv=False)
    assert singular_values[-1] > 1e-8 * singular_values[0]


@pytest.mark.parametrize("name", SIMPLEX)
def test_stabilization_still_vanishes_on_simplices_at_one_half(name):
    """The simplex property is a property of ``N``, so ``nu`` cannot disturb it."""
    from mimetika.operators.elasticity import ElasticityInnerProduct

    mk, _ = MESHES[name]
    mesh = mk()
    for nu in (0.15, 0.5):
        inner = ElasticityInnerProduct(
            mesh, material=Material(shear_modulus=1.0, poisson=nu)
        )
        assert inner.stabilization_dim(0) == 0


def test_compliance_coefficient_is_finite_at_one_half():
    for d in (2, 3):
        assert compliance_coefficient(d, 0.5) == pytest.approx(1.0 / d)
        assert Material(poisson=0.5).inverse_modulus(d) == 0.0
        assert Material(poisson=0.5).pressure_coupling(d) == 0.0
        assert np.isfinite(Material(poisson=0.5).storage(d))


# -- requirement 2: incompressible fluid -------------------------------------------


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("nu", [0.15, 0.45, 0.49999])
def test_undrained_response_with_an_incompressible_fluid(name, nu):
    """``1/M = 0`` and sealed walls: ``div u = 0`` and ``p = -tr(Sigma)/(d alpha)``.

    With no storage, the fluid content cannot change, so the load is carried by
    the pore pressure alone -- Skempton ``B = 1``.  Both statements are exact,
    which is a far sharper test than checking the solve merely completes.
    """
    mk, d = MESHES[name]
    mesh = mk()
    biot = 0.9
    material = Material(
        shear_modulus=1.0,
        poisson=nu,
        biot=biot,
        inverse_biot_modulus=0.0,
        permeability=1.0,
        viscosity=1.0,
    )
    problem = PoroMechanics(mesh, material)
    load = np.zeros((3, 3))
    load[:d, :d] = -2.0 * np.eye(d)

    solution = problem.solve(
        dt=1.0,
        dirichlet=lambda x: np.zeros((len(np.atleast_2d(x)), 3)),
        traction=lambda x: np.broadcast_to(load, (len(np.atleast_2d(x)), 3, 3)),
        traction_facets=face(problem),
        no_flow=boundary_facets(mesh),
    )
    trace = (problem.trace_operator() @ solution["stress"]) / mesh.geometry.measure(d)
    assert np.abs(problem.volumetric_strain(solution)).max() < 1e-11
    assert np.allclose(solution["pressure"], -trace / (d * biot), atol=1e-11)


def test_storage_vanishes_only_when_both_terms_do():
    """``S = d alpha^2 / K + 1/M``: the two limits are independent."""
    fluid = Material(poisson=0.25, biot=0.9, inverse_biot_modulus=0.0)
    both = Material(poisson=0.5, biot=0.9, inverse_biot_modulus=0.0)
    assert fluid.storage(3) > 0.0  # compressible skeleton still stores
    assert both.storage(3) == 0.0  # both incompressible: no storage at all


# -- requirement 3: high material contrast --------------------------------------------
#
# Choosing the test problem matters here, because a *uniform* stress state is
# not in general the solution of the continuous problem under a heterogeneous
# ``C``: the strain ``eps = C^{-1} sigma`` is then discontinuous in its in-plane
# components, so no displacement produces it.  That is a statement about the
# manufactured solution, not about the method -- and it is avoidable, because
# ``C^{-1}`` splits cleanly:
#
#     ``C^{-1} sigma = dev(sigma) / 2mu  +  tr(sigma) I / (d^2 K)``
#
# so each part sees only *one* modulus.  Contrast in ``mu`` alone under a
# **hydrostatic** load, and contrast in ``K`` alone under a **deviatoric** load,
# therefore both admit a uniform stress with a linear displacement -- an exact
# patch test at arbitrary contrast, on a genuine **checkerboard**.  Contrast in
# ``mu`` and ``K`` together is covered by the layered pure-shear problem below,
# where compatibility is restored by the layering rather than by the load.


def checkerboard(mesh, values, split=0.5):
    """Per-cell array alternating between ``values`` on a checkerboard pattern."""
    centroids = mesh.geometry.centroids(mesh.dim)[:, : mesh.dim]
    parity = np.floor(centroids / split).astype(int).sum(axis=1) % 2
    return np.where(parity == 0, values[0], values[1])


def poisson_for(shear, inverse_modulus, dim):
    """The ``nu`` that pairs with ``shear`` to give a prescribed bulk compliance.

    Inverting ``1/(dK) = (1-2nu) / (2mu (1-2nu+d nu))``.  Holding ``K`` fixed
    while raising ``mu`` drives ``nu`` towards ``-1``: auxetic, but strictly
    inside the admissible range ``-1 < nu < 1/2`` where ``C`` stays definite.
    """
    t = 2.0 * np.asarray(shear, dtype=float) * inverse_modulus
    c = (1.0 - t) / (t * dim)
    return c / (1.0 + 2.0 * c)


def uniform_state(problem, tensor):
    """``(stress_fn, displacement_fn)`` for a uniform stress with per-cell moduli.

    Only valid when ``C^{-1} tensor`` is the *same* in every cell -- which is
    what the two checkerboard tests below arrange.
    """
    d = problem.d
    material = problem.material
    a = np.atleast_1d(material.compliance_coefficient(d))[0]
    shear = np.atleast_1d(material.shear_modulus)[0]
    strain = (tensor[:d, :d] - a * np.trace(tensor[:d, :d]) * np.eye(d)) / (2 * shear)
    grad = np.zeros((3, 3))
    grad[:d, :d] = strain

    def stress(x):
        return np.broadcast_to(tensor, (len(np.atleast_2d(x)), 3, 3))

    return stress, (lambda x: np.atleast_2d(x) @ grad.T), strain


@pytest.mark.parametrize("contrast", [1e2, 1e4, 1e6])
def test_checkerboard_shear_contrast_under_a_hydrostatic_load(contrast):
    """``mu`` alternating cell by cell, ``K`` matched: uniform stress is exact.

    A hydrostatic stress engages only the volumetric part of ``C^{-1}``, which
    is identical in every cell, so the shear contrast cannot make the strain
    incompatible -- any error is the solver's.
    """
    mesh = structured_box(2, 2, 2)
    d, bulk = 3, 1.0
    shear = checkerboard(mesh, (1.0, contrast))
    poisson = poisson_for(shear, 1.0 / (d * bulk), d)
    problem = PoroMechanics(
        mesh, Material(shear_modulus=shear, poisson=poisson, biot=0.9)
    )
    # the premise: every cell really does share one bulk compliance
    inverse = problem.material.inverse_modulus(d)
    assert np.allclose(inverse, inverse[0], rtol=1e-12)
    assert np.ptp(shear) > 0.9 * contrast  # ... and the contrast is really there

    stress, displacement, strain = uniform_state(problem, 2.0 * np.eye(3))
    solution = problem.solve(dt=None, dirichlet=displacement, pressure=0.0)
    exact = problem.mechanics.interpolate_stress(stress)
    error = np.abs(solution["stress"] - exact).max() / np.abs(exact).max()
    assert error < 1e-13 * contrast + 1e-11, f"contrast {contrast:g}: {error:.3e}"
    assert np.allclose(
        problem.volumetric_strain(solution), np.trace(strain), rtol=1e-9
    )


@pytest.mark.parametrize("name", ["hex", "tet", "quad", "tri"])
def test_checkerboard_bulk_contrast_under_a_deviatoric_load(name):
    """``K`` alternating cell by cell (up to the incompressible limit), ``mu`` uniform.

    A deviatoric stress engages only ``dev/2mu``, so the bulk contrast -- here
    spanning ``nu = 0.15`` to ``nu = 1/2``, i.e. finite ``K`` next to infinite --
    leaves the exact solution untouched.  This is the checkerboard analogue of
    the incompressible-limit test, with both regimes present in one mesh.
    """
    mk, d = MESHES[name]
    mesh = mk()
    problem = PoroMechanics(
        mesh,
        Material(
            shear_modulus=1.0,
            poisson=checkerboard(mesh, (0.15, 0.5)),
            biot=0.9,
        ),
    )
    deviator = np.zeros((3, 3))
    deviator[0, 1] = deviator[1, 0] = 0.5
    stress, displacement, _ = uniform_state(problem, deviator)
    solution = problem.solve(dt=None, dirichlet=displacement, pressure=0.0)
    exact = problem.mechanics.interpolate_stress(stress)
    assert np.abs(solution["stress"] - exact).max() / np.abs(exact).max() < 1e-11
    # a deviatoric state is volume preserving in compressible and stiff cells alike
    assert np.abs(problem.volumetric_strain(solution)).max() < 1e-11


def test_checkerboard_pattern_really_alternates():
    """Guard the guard: neighbouring cells must actually differ."""
    mesh = structured_box(2, 2, 2)
    values = checkerboard(mesh, (1.0, 7.0))
    assert set(np.unique(values)) == {1.0, 7.0}
    assert (values == 1.0).sum() == (values == 7.0).sum()


def layered(mesh, values, axis=1, split=0.5):
    """Per-cell array taking ``values[0]`` below ``split`` and ``values[1]`` above."""
    centroids = mesh.geometry.centroids(mesh.dim)
    return np.where(centroids[:, axis] < split, values[0], values[1])


def layered_shear(mesh, shear, amplitude=0.5, split=0.5):
    """Exact pure-shear solution of a two-layer medium stacked along ``y``.

    ``sigma = s (e_x ox e_y + e_y ox e_x)`` is uniform and equilibrated; the only
    non-zero strain is ``eps_xy = s / 2mu``, which jumps between layers.  With
    ``u_y = u_z = 0`` and ``du_x/dy = s / mu`` the displacement is continuous and
    piecewise linear, so the jump in strain is perfectly admissible.
    """
    stress_tensor = np.zeros((3, 3))
    stress_tensor[0, 1] = stress_tensor[1, 0] = amplitude
    lower, upper = float(shear[0]), float(shear[1])

    def displacement(x):
        x = np.atleast_2d(x)
        y = x[:, 1]
        below = amplitude * y / lower
        above = amplitude * split / lower + amplitude * (y - split) / upper
        out = np.zeros((len(x), 3))
        out[:, 0] = np.where(y < split, below, above)
        return out

    def stress(x):
        return np.broadcast_to(stress_tensor, (len(np.atleast_2d(x)), 3, 3))

    return stress, displacement


@pytest.mark.parametrize("contrast", [1e3, 1e6, 1e9])
def test_layered_shear_is_exact_under_a_shear_modulus_jump(contrast):
    """The stress stays exact to the accuracy the conditioning allows.

    The attainable accuracy degrades like ``contrast * eps_machine`` -- that is
    forced by solving in double precision at a condition number proportional to
    the modulus jump, and it is the *optimal* rate, not a defect.  The bound
    below encodes that rate, so an error growing like ``contrast^2`` would fail.
    """
    mesh = structured_box(2, 2, 2)
    shear = (1.0, contrast)
    problem = PoroMechanics(
        mesh, Material(shear_modulus=layered(mesh, shear), poisson=0.3, biot=0.9)
    )
    stress, displacement = layered_shear(mesh, shear)
    solution = problem.solve(dt=None, dirichlet=displacement, pressure=0.0)
    exact = problem.mechanics.interpolate_stress(stress)
    error = np.abs(solution["stress"] - exact).max() / np.abs(exact).max()
    assert error < 1e-13 * contrast + 1e-12, f"contrast {contrast:g}: {error:.3e}"


@pytest.mark.parametrize("contrast", [1.0, 1e3, 1e6, 1e9])
def test_discrete_equilibrium_is_exact_at_every_contrast(contrast):
    """``div_h sigma = f`` is metric-free, so no contrast can perturb it.

    This is the part of the answer that must stay at round-off however badly
    conditioned the inner product becomes: the divergence block contains only
    incidence signs and measures, no moduli at all.
    """
    mesh = structured_box(2, 2, 2)
    shear = (1.0, contrast)
    problem = PoroMechanics(
        mesh, Material(shear_modulus=layered(mesh, shear), poisson=0.3, biot=0.9)
    )
    _, displacement = layered_shear(mesh, shear)
    solution = problem.solve(dt=None, dirichlet=displacement, pressure=0.0)
    _, divergence, skew = problem.mechanics.assemble_operators()
    scale = np.abs(solution["stress"]).max()
    assert np.abs(divergence @ solution["stress"]).max() < 1e-11 * scale
    assert np.abs(skew @ solution["stress"]).max() < 1e-11 * scale


def test_contrast_error_grows_only_linearly():
    """Pin the *rate*: a thousand-fold contrast jump costs a thousand-fold error.

    An absolute tolerance loose enough to admit ``contrast = 1e9`` has little
    teeth on its own; this compares the two directly instead.
    """
    mesh = structured_box(2, 2, 2)
    errors = {}
    for contrast in (1e6, 1e9):
        shear = (1.0, contrast)
        problem = PoroMechanics(
            mesh, Material(shear_modulus=layered(mesh, shear), poisson=0.3, biot=0.9)
        )
        stress, displacement = layered_shear(mesh, shear)
        solution = problem.solve(dt=None, dirichlet=displacement, pressure=0.0)
        exact = problem.mechanics.interpolate_stress(stress)
        errors[contrast] = np.abs(solution["stress"] - exact).max() / np.abs(exact).max()
    assert errors[1e9] / errors[1e6] < 1e4  # linear would be 1e3


@pytest.mark.parametrize("contrast", [1e4, 1e8])
def test_contrast_together_with_the_incompressible_limit(contrast):
    """Compressible cells adjacent to exactly-incompressible ones, in one mesh.

    Pure shear is traceless, so the answer is independent of ``nu`` -- which is
    exactly what makes it a clean probe: any sensitivity to the ``nu = 1/2``
    cells is numerical, not physical.
    """
    mesh = structured_box(2, 2, 2)
    shear = (1.0, contrast)
    problem = PoroMechanics(
        mesh,
        Material(
            shear_modulus=layered(mesh, shear),
            poisson=layered(mesh, (0.15, 0.5)),
            biot=0.9,
        ),
    )
    stress, displacement = layered_shear(mesh, shear)
    solution = problem.solve(dt=None, dirichlet=displacement, pressure=0.0)
    exact = problem.mechanics.interpolate_stress(stress)
    error = np.abs(solution["stress"] - exact).max() / np.abs(exact).max()
    assert error < 1e-13 * contrast + 1e-12
    # pure shear is volume preserving everywhere, incompressible layer or not
    assert np.abs(problem.volumetric_strain(solution)).max() < 1e-9


def test_the_contrast_problem_actually_has_contrast():
    """Guard the guard: the stiff layer must barely shear compared with the soft one.

    Without this, a test that passed because the contrast never reached the
    operator would look identical to one that passed on merit.
    """
    mesh = structured_box(2, 2, 2)
    _, displacement = layered_shear(mesh, (1.0, 1e6))
    at = lambda y: displacement(np.array([[0.0, y, 0.0]]))[0, 0]  # noqa: E731
    soft = at(0.5) - at(0.0)  # displacement gained across the soft layer
    stiff = at(1.0) - at(0.5)  # ... and across the stiff one
    assert soft > 0.0
    assert stiff < 1e-5 * soft


def test_permeability_contrast_sustains_a_pressure_jump():
    """A tight layer draining through a permeable one must hold the pressure back.

    Sealed walls with a uniform source give a *uniform* pressure and no flow at
    all, so the contrast would never be exercised; the top face is drained here,
    and the step is long enough for the permeable layer to actually drain.
    """
    mesh = structured_box(2, 2, 4)
    material = Material(
        shear_modulus=1.0,
        poisson=0.25,
        biot=0.9,
        inverse_biot_modulus=0.1,
        permeability=layered(mesh, (1e-8, 1.0)),
        viscosity=1.0,
    )
    problem = PoroMechanics(mesh, material)
    centroids = mesh.geometry.centroids(2)
    sealed = [f for f in boundary_facets(mesh) if abs(centroids[f][1] - 1.0) > 1e-12]

    solution = problem.solve(
        dt=1e3,
        dirichlet=lambda x: np.zeros((len(np.atleast_2d(x)), 3)),
        source=lambda x: np.ones(len(np.atleast_2d(x))),
        no_flow=sealed,
    )
    tight = solution["pressure"][layered(mesh, (True, False))]
    open_ = solution["pressure"][layered(mesh, (False, True))]
    assert np.all(np.isfinite(solution["pressure"]))
    assert tight.min() > 1e3 * open_.max()


# -- the transient coupled system ------------------------------------------------------


def test_sealed_walls_conserve_fluid_mass_exactly():
    """``int (alpha div u + p/M) = dt int r`` when no flux crosses the boundary."""
    mesh = structured_box(2, 2, 2)
    volume = mesh.geometry.measure(3)
    material = Material(
        shear_modulus=1.0,
        poisson=0.25,
        biot=0.9,
        inverse_biot_modulus=0.1,
        permeability=1.0,
        viscosity=1.0,
    )
    problem = PoroMechanics(mesh, material)
    dt = 0.1
    solution = problem.solve(
        dt=dt,
        dirichlet=lambda x: np.zeros((len(np.atleast_2d(x)), 3)),
        source=lambda x: np.ones(len(np.atleast_2d(x))),
        no_flow=boundary_facets(mesh),
    )
    content = (
        material.biot * problem.volumetric_strain(solution)
        + material.inverse_biot_modulus * solution["pressure"]
    ) @ volume
    assert content == pytest.approx(dt * volume.sum(), abs=1e-12)


def test_no_flow_facets_are_genuinely_sealed():
    mesh = structured_box(2, 2, 2)
    sealed = boundary_facets(mesh)
    problem = PoroMechanics(
        mesh,
        Material(
            shear_modulus=1.0,
            poisson=0.25,
            biot=0.9,
            inverse_biot_modulus=0.1,
            permeability=1.0,
            viscosity=1.0,
        ),
    )
    solution = problem.solve(
        dt=0.1,
        dirichlet=lambda x: np.zeros((len(np.atleast_2d(x)), 3)),
        source=lambda x: np.ones(len(np.atleast_2d(x))),
        no_flow=sealed,
    )
    ndf = problem._ndf_q
    dofs = (ndf * np.asarray(sealed)[:, None] + np.arange(ndf)).ravel()
    assert np.all(solution["flux"][dofs] == 0.0)


def test_draining_is_the_default_and_loses_mass():
    """Guards the mixed BC convention: no ``no_flow`` means ``p = 0``, not sealed.

    Getting this backwards is the natural mistake -- in a primal formulation
    "prescribe nothing" *is* no-flow -- and it silently changes the physics.
    """
    mesh = structured_box(2, 2, 2)
    volume = mesh.geometry.measure(3)
    material = Material(
        shear_modulus=1.0,
        poisson=0.25,
        biot=0.9,
        inverse_biot_modulus=0.1,
        permeability=1.0,
        viscosity=1.0,
    )
    problem = PoroMechanics(mesh, material)
    dt = 0.1
    drained = problem.solve(
        dt=dt,
        dirichlet=lambda x: np.zeros((len(np.atleast_2d(x)), 3)),
        source=lambda x: np.ones(len(np.atleast_2d(x))),
    )
    content = (
        material.biot * problem.volumetric_strain(drained)
        + material.inverse_biot_modulus * drained["pressure"]
    ) @ volume
    assert content < 0.5 * dt * volume.sum()  # most of the injected fluid left
    assert np.abs(drained["flux"][boundary_facets(mesh)]).max() > 1e-6


def test_previous_state_advances_the_solution():
    """Two steps of ``dt`` must differ from one, and accumulate content."""
    mesh = structured_box(2, 2, 2)
    volume = mesh.geometry.measure(3)
    material = Material(
        shear_modulus=1.0,
        poisson=0.25,
        biot=0.9,
        inverse_biot_modulus=0.1,
        permeability=1.0,
        viscosity=1.0,
    )
    problem = PoroMechanics(mesh, material)
    sealed = boundary_facets(mesh)
    args = dict(
        dt=0.1,
        dirichlet=lambda x: np.zeros((len(np.atleast_2d(x)), 3)),
        source=lambda x: np.ones(len(np.atleast_2d(x))),
        no_flow=sealed,
    )
    first = problem.solve(**args)
    second = problem.solve(previous=first, **args)
    content = lambda s: (  # noqa: E731
        material.biot * problem.volumetric_strain(s)
        + material.inverse_biot_modulus * s["pressure"]
    ) @ volume
    assert content(second) == pytest.approx(2.0 * content(first), rel=1e-10)


# -- sizes and plumbing ------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL)
def test_block_sizes(name):
    problem, _, _, _ = uniform(name, 0.25)
    d, mesh = problem.d, problem.mesh
    assert problem.n_stress == d * d * mesh.num_cells(d - 1)
    assert problem.n_cells == mesh.num_cells(d)
    assert problem.n_flux == problem._ndf_q * mesh.num_cells(d - 1)
    assert problem.n_skew == d * (d - 1) // 2
    assert problem.trace_operator().shape == (problem.n_cells, problem.n_stress)


@pytest.mark.parametrize("name", ALL)
def test_quasi_steady_returns_the_prescribed_pressure(name):
    """``dt=None`` treats the pressure as data, so it comes back unchanged."""
    problem, stress, displacement, _ = uniform(name, 0.25)
    field = lambda x: 3.0 - np.atleast_2d(x)[:, 1]  # noqa: E731
    solution = problem.solve(
        dt=None,
        dirichlet=displacement,
        pressure=field,
        traction=stress,
        traction_facets=face(problem),
    )
    centroids = problem.mesh.geometry.centroids(problem.d)
    assert np.allclose(solution["pressure"], field(centroids))
    assert np.all(solution["flux"] == 0.0)


def test_derham_flow_block_reproduces_a_linear_pressure_state():
    """One backward-Euler step from the exact linear-pressure state is exact.

    With ``alpha = 0`` the flow decouples; a linear pressure with its constant
    flux is a steady state of the de Rham flow block, so the step must return
    it to machine precision -- this pins the sign and scaling of the
    boundary-pressure pairing on the moment DOFs.
    """
    mesh = structured_box(2, 2, 2)
    problem = PoroMechanics(
        mesh,
        Material(
            shear_modulus=1.0,
            poisson=0.25,
            biot=0.0,
            inverse_biot_modulus=1.0,
            permeability=1.0,
            viscosity=1.0,
        ),
    )
    assert problem._ndf_q == mesh.dim  # the de Rham flow space is the default

    from mimetika.assembly.mixed import MixedSolution

    p_lin = lambda x: 2.0 + np.atleast_2d(x)[:, 0] - 3.0 * np.atleast_2d(x)[:, 1]
    centroids = mesh.geometry.centroids(3)
    previous = MixedSolution(
        {"stress": np.zeros(problem.n_stress), "pressure": p_lin(centroids)}
    )
    solution = problem.solve(
        dt=0.5,
        dirichlet=lambda x: np.zeros((len(np.atleast_2d(x)), 3)),
        pressure_bc=p_lin,
        previous=previous,
    )
    assert np.allclose(solution["pressure"], p_lin(centroids), atol=1e-9)
    # the flux is the interpolant of the exact constant field q = -grad p
    from mimetika.geometry.local_cell import LocalCell

    q_exact = np.array([-1.0, 3.0, 0.0])
    ndf, done = problem._ndf_q, set()
    q_I = np.zeros(problem.n_flux)
    for c in range(problem.n_cells):
        lc = LocalCell.build(mesh.geometry, c, problem.flow.frame)
        for i, fid in enumerate(lc.facet_ids):
            if fid in done:
                continue
            done.add(fid)
            qn = (lc.frame.T @ q_exact) @ lc.facet_normals[i]
            q_I[ndf * fid] = lc.signs[i] * lc.facet_measures[i] * qn
    assert np.allclose(solution["flux"], q_I, atol=1e-8)
