r"""Kinematic-rotation elasticity: the rotation defined, not enforced.

The closure replaces the multiplier row ``A sigma = 0`` by the blended
kinematic row (see :mod:`mimetika.assembly.kinematic`).  The headline checks:

* the **patch test** stays exact on every cell family -- the least-squares
  skew gradient is exact for affine displacements, so replacing the rotation
  row costs no consistency;
* the pure-definition limit (``blend = 1``) is **rejected**: it carries the
  constant-skew gauge kernel (globally constant in 2D, facet-normal jump
  fields in 3D), which is a property of keeping the stress independent, not an
  implementation detail;
* three- and four-field kinematic systems are **congruent** -- the solid
  pressure split never touches the rotation row;
* the scheme **converges** without any rotation stabilization: there is no
  zero block, hence no multiplier inf-sup to fail;
* the system is quasi-symmetric, so MINRES is refused rather than silently
  misused.
"""

import numpy as np
import pytest

from mimetika.assembly.kinematic import (
    KinematicRotationElasticity,
    KinematicRotationFourField,
    skew_gradient,
)
from mimetika.materials import Material
from mimetika.mesh import (
    structured_box,
    structured_quads,
    structured_tets,
    structured_triangles,
)
from mimetika.operators.elasticity import ElasticityInnerProduct

MU, LAM = 1.3, 2.7

U_A = np.array([0.31, -0.42, 0.17])
U_B = np.array([[0.5, -0.3, 0.2], [0.15, 0.4, -0.25], [-0.1, 0.35, 0.6]])


def linear_displacement(x):
    return U_A + np.atleast_2d(x) @ U_B.T


def grad_linear(x):
    return np.broadcast_to(U_B, (len(np.atleast_2d(x)), 3, 3))


def exact_stress(d):
    eps = 0.5 * (U_B + U_B.T)[:d, :d]
    return 2 * MU * eps + LAM * np.trace(eps) * np.eye(d)


def quadratic_displacement(x):
    x = np.atleast_2d(x)
    u = np.zeros((len(x), 3))
    u[:, 0] = 0.3 * x[:, 0] ** 2 - 0.2 * x[:, 0] * x[:, 1] + 0.1 * x[:, 1] ** 2
    u[:, 1] = 0.1 * x[:, 0] ** 2 + 0.25 * x[:, 0] * x[:, 1] - 0.15 * x[:, 1] ** 2
    return u


def grad_quadratic(x):
    x = np.atleast_2d(x)
    G = np.zeros((len(x), 3, 3))
    G[:, 0, 0] = 0.6 * x[:, 0] - 0.2 * x[:, 1]
    G[:, 0, 1] = -0.2 * x[:, 0] + 0.2 * x[:, 1]
    G[:, 1, 0] = 0.2 * x[:, 0] + 0.25 * x[:, 1]
    G[:, 1, 1] = 0.25 * x[:, 0] - 0.3 * x[:, 1]
    return G


def quadratic_stress(x):
    G = grad_quadratic(x)
    eps = 0.5 * (G + np.swapaxes(G, 1, 2))
    tr = np.einsum("qii->q", eps)
    S = 2 * MU * eps
    S[:, 0, 0] += LAM * tr
    S[:, 1, 1] += LAM * tr
    return S


def quadratic_body_force(x):
    h = 1e-6
    x = np.atleast_2d(x)
    out = np.zeros((len(x), 3))
    for k in range(2):
        for j in range(2):
            xp = x.copy()
            xp[:, j] += h
            xm = x.copy()
            xm[:, j] -= h
            out[:, k] += (quadratic_stress(xp)[:, k, j] - quadratic_stress(xm)[:, k, j]) / (
                2 * h
            )
    return out


MESHES = {
    "quad": lambda: structured_quads(3, 3),
    "tri": lambda: structured_triangles(2, 2),
    "hex": lambda: structured_box(2, 2, 2),
    "tet": lambda: structured_tets(1, 1, 1),
}
CLASSES = [KinematicRotationElasticity, KinematicRotationFourField]


# -- consistency ----------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("name", list(MESHES))
def test_patch_test_is_exact(name, cls):
    """Linear displacement reproduced exactly -- stress, displacement, rotation."""
    mesh = MESHES[name]()
    d = mesh.dim
    full = np.zeros((3, 3))
    full[:d, :d] = exact_stress(d)
    stress_fn = lambda x: np.broadcast_to(full, (len(np.atleast_2d(x)), 3, 3))  # noqa: E731

    problem = cls(mesh, MU, LAM)
    sol = problem.solve(dirichlet=linear_displacement)
    scale = np.abs(full).max()
    assert np.abs(sol["stress"] - problem.interpolate_stress(stress_fn)).max() < 1e-11 * scale
    assert np.abs(
        sol["displacement"] - problem.interpolate_displacement(linear_displacement)
    ).max() < 1e-12
    assert np.abs(sol["rotation"] - problem.interpolate_rotation(grad_linear)).max() < 1e-12
    if "solid_pressure" in sol.blocks:
        assert np.allclose(
            sol["solid_pressure"], np.trace(full[:d, :d]) / d, atol=1e-11 * scale
        )


def test_skew_gradient_is_exact_for_affine_fields():
    """The least-squares fit reproduces ``|E| gens : grad u`` for affine ``u``."""
    for name in MESHES:
        mesh = MESHES[name]()
        d = mesh.dim
        problem = KinematicRotationElasticity(mesh, MU, LAM)
        G, b = skew_gradient(mesh, problem.inner.frame, linear_displacement)
        u = problem.interpolate_displacement(linear_displacement)
        vol = mesh.geometry.measure(d)
        got = (G @ u + b).reshape(-1, problem.n_skew)
        exact = problem.interpolate_rotation(grad_linear).reshape(-1, problem.n_skew)
        assert np.allclose(got, 2.0 * vol[:, None] * exact, atol=1e-12), name


# -- the gauge kernel and its exclusion ------------------------------------------


@pytest.mark.parametrize("blend", [0.0, 1.0])
def test_degenerate_blends_are_rejected(blend):
    """``theta = 1`` has the constant-skew gauge kernel; ``theta = 0`` is the
    zero-block multiplier form -- both refuse rather than mislead."""
    problem = KinematicRotationElasticity(structured_quads(2, 2), MU, LAM)
    problem.blend = blend
    with pytest.raises(ValueError, match="blend"):
        problem.assemble(dirichlet=linear_displacement)


def test_blended_system_is_nonsingular_where_pure_definition_is_not():
    """The blend removes every gauge mode -- checked on the 3D system whose
    pure-definition kernel is larger than the global rotations."""
    mesh = structured_tets(1, 1, 1)
    problem = KinematicRotationElasticity(mesh, MU, LAM)
    S, _ = problem.assemble(dirichlet=linear_displacement)
    sv = np.linalg.svd(S.toarray(), compute_uv=False)
    assert sv[-1] > 1e-10 * sv[0]


def test_minres_is_refused():
    problem = KinematicRotationElasticity(structured_quads(2, 2), MU, LAM)
    with pytest.raises(ValueError, match="quasi-symmetric"):
        problem.solve(dirichlet=linear_displacement, method="minres")


# -- congruence of the three- and four-field arrangements ------------------------


def test_three_and_four_field_kinematic_agree():
    """The solid-pressure split never touches the rotation row."""
    mesh = structured_quads(4, 4)
    three = KinematicRotationElasticity(mesh, MU, LAM)
    four = KinematicRotationFourField(mesh, MU, LAM)
    kwargs = dict(dirichlet=quadratic_displacement, body_force=quadratic_body_force)
    s3 = three.solve(**kwargs)
    s4 = four.solve(**kwargs)
    for field in ("stress", "displacement", "rotation"):
        assert np.allclose(s4[field], s3[field], rtol=1e-9, atol=1e-12), field


# -- convergence without any rotation stabilization ------------------------------


@pytest.mark.parametrize("maker", [structured_quads, structured_triangles],
                         ids=["quads", "tris"])
def test_convergence_needs_no_stabilization(maker):
    """``u`` and ``sigma`` converge at better than first order, ``s`` at least
    at first order -- with no zero block there is no inf-sup to fail."""
    errors = []
    for n in (4, 8, 16):
        mesh = maker(n, n)
        problem = KinematicRotationElasticity(mesh, MU, LAM)
        sol = problem.solve(
            dirichlet=quadratic_displacement, body_force=quadratic_body_force
        )
        vol = mesh.geometry.measure(2)
        sI = problem.interpolate_stress(quadratic_stress)
        eu = np.sqrt(
            vol
            @ (
                (
                    sol["displacement"]
                    - problem.interpolate_displacement(quadratic_displacement)
                ).reshape(-1, 2)
                ** 2
            ).sum(1)
        )
        es = np.linalg.norm(sol["stress"] - sI) / np.linalg.norm(sI)
        er = np.sqrt(vol @ (sol["rotation"] - problem.interpolate_rotation(grad_quadratic)) ** 2)
        errors.append((eu, es, er))
    rates = [
        np.log2(np.asarray(a) / np.asarray(b)) for a, b in zip(errors, errors[1:])
    ]
    last = rates[-1]
    assert last[0] > 1.5, f"u rate {last[0]:.2f}"
    assert last[1] > 1.4, f"sigma rate {last[1]:.2f}"
    assert last[2] > 0.8, f"s rate {last[2]:.2f}"


@pytest.mark.parametrize("nu", [0.3, 0.45, 0.49999])
def test_the_incompressible_range_solves(nu):
    """No degeneracy approaching ``nu = 1/2`` -- the closure never divides by
    the volumetric compliance."""
    mesh = structured_quads(3, 3)
    inner = ElasticityInnerProduct(
        mesh, material=Material(shear_modulus=MU, poisson=nu)
    )
    problem = KinematicRotationFourField(mesh, inner=inner)
    sol = problem.solve(
        dirichlet=quadratic_displacement, body_force=quadratic_body_force
    )
    assert np.all(np.isfinite(sol["stress"]))
    assert np.all(np.isfinite(sol["solid_pressure"]))


# -- fracture mechanics: the jump algebra is formulation-independent -------------


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_the_fracture_jump_is_recovered_exactly(cls):
    """A prescribed jump across a soft fracture is read back exactly.

    Two things are on trial.  The jump *readout* -- the residual of the
    unfractured constitutive row -- must be formulation-independent, since the
    kinematic closure never touches that row.  And the skew-gradient stencil
    must treat the fracture as the internal boundary it is: differencing the
    displacement across it would smear the jump into a spurious rotation of
    the fault-adjacent cells at amplitude ``~jump/h``, which is why fracture
    facets are excluded from the fit automatically (via ``self.contact``).
    Exactness on two meshes pins both the value and its h-independence.
    """
    from mimetika.contact import ContactDriver, SignoriniCoulomb
    from mimetika.mesh.fracture import facets_on_plane
    from mimetika.solver.saddle import solve_saddle

    offset = 0.02
    recovered = []
    for n in (2, 4):
        mesh = structured_box(n, n, n)
        tags = facets_on_plane(mesh, [0.5, 0, 0], [1, 0, 0])
        driver = ContactDriver(
            mesh, tags, SignoriniCoulomb(friction=0.6), mu=1.0, lam=1.0
        )

        def prescribed(x):
            x = np.atleast_2d(x)
            out = np.zeros((len(x), 3))
            out[:, 1] = offset if x[:, 0].mean() > 0.5 else 0.0
            return out

        problem = cls(
            mesh, mu=1.0, lam=1.0,
            contact=driver.contact_geometry(np.diag([1e6, 1e6, 1e6])),
        )
        matrix, rhs = problem.assemble_constrained(dirichlet=prescribed)
        sol = problem.split(
            solve_saddle(matrix, rhs, problem.block_sizes, method="direct")
        )
        vector = np.concatenate([sol[k] for k in sol.blocks])
        jump = (driver.jump_operator(problem) @ vector).reshape(
            driver.n_points, driver.dim
        )
        recovered.append(np.abs(jump[:, 1]).max())
        # the soft fracture takes the whole offset; the bulk carries only the
        # physical traction jump/compliance
        assert np.abs(sol["stress"]).max() < 1e-7
    assert recovered[0] == pytest.approx(offset, rel=1e-6)
    assert recovered[1] == pytest.approx(recovered[0], rel=1e-6)
