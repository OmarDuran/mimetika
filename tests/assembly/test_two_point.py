r"""The two-point stress scheme: lumped inner product + kinematic rotation.

:class:`~mimetika.assembly.kinematic.TwoPointFourField` assembles every
structural property the lumping programme aimed at, and this module pins each
one:

* ``M`` diagonal, the ``(s, s)`` and ``(p_s, p_s)`` blocks diagonal, and every
  facet row coupling **exactly two cells** -- the minimal fill-in of a
  cell-centred method, reached from the traction side;
* the patch test exact on tensor-product grids, *including graded ones* (the
  cell-centre line crosses each facet at its centroid there, so the two-point
  facet average is exact for affine fields);
* the layered-shear state exact across a shear-modulus contrast: continuity of
  the shear traction gives ``mu_L g_L = mu_R g_R``, which is precisely the
  cancellation the ``mu/delta``-weighted facet average is built on;
* a prescribed fracture jump recovered exactly, with the reduced
  (``d``-per-facet) compliance block and the fracture facets excluded from the
  facet-average stencil;
* convergence at first-to-second order with **no** stabilization of any kind
  and no inf-sup condition anywhere;
* the orthogonality guard inherited from the lumped space: non-orthogonal
  meshes are rejected at construction.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from mimetika.assembly.contact import FractureContact
from mimetika.assembly.kinematic import TwoPointElasticity, TwoPointFourField
from mimetika.materials import Material
from mimetika.mesh import graded_quads, structured_box, structured_quads
from mimetika.solver.saddle import solve_saddle

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
            out[:, k] += (
                quadratic_stress(xp)[:, k, j] - quadratic_stress(xm)[:, k, j]
            ) / (2 * h)
    return out


def alternating_axis(n):
    w = np.tile([1.0, 2.0], n // 2)[:n]
    return np.concatenate([[0.0], np.cumsum(w)]) / w.sum()


MESHES = {
    "quad": lambda: structured_quads(3, 3),
    "hex": lambda: structured_box(2, 2, 2),
    "graded-1:2": lambda: graded_quads(alternating_axis(4), alternating_axis(4)),
}
CLASSES = [TwoPointElasticity, TwoPointFourField]


# -- consistency ----------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("name", list(MESHES))
def test_patch_test_is_exact_on_tensor_grids(name, cls):
    mesh = MESHES[name]()
    d = mesh.dim
    full = np.zeros((3, 3))
    full[:d, :d] = exact_stress(d)
    stress_fn = lambda x: np.broadcast_to(full, (len(np.atleast_2d(x)), 3, 3))  # noqa: E731

    problem = cls(mesh, MU, LAM)
    sol = problem.solve(dirichlet=linear_displacement)
    scale = np.abs(full).max()
    assert np.abs(sol["stress"] - problem.interpolate_stress(stress_fn)).max() < 1e-11 * scale
    assert np.abs(sol["rotation"] - problem.interpolate_rotation(grad_linear)).max() < 1e-12
    if "solid_pressure" in sol.blocks:
        assert np.allclose(
            sol["solid_pressure"], np.trace(full[:d, :d]) / d, atol=1e-11 * scale
        )


@pytest.mark.parametrize("contrast", [1e3, 1e6])
def test_layered_shear_is_exact_across_the_contrast(contrast):
    """The mu-weighted facet average is built on ``mu_L g_L = mu_R g_R``."""
    mesh = structured_quads(4, 4)
    cent = mesh.geometry.centroids(2)
    shear = np.where(cent[:, 1] < 0.5, 1.0, contrast)
    problem = TwoPointFourField(
        mesh, material=Material(shear_modulus=shear, poisson=0.3)
    )

    amplitude, split = 0.5, 0.5
    tensor = np.zeros((3, 3))
    tensor[0, 1] = tensor[1, 0] = amplitude

    def displacement(x):
        x = np.atleast_2d(x)
        y = x[:, 1]
        out = np.zeros((len(x), 3))
        out[:, 0] = np.where(
            y < split,
            amplitude * y / 1.0,
            amplitude * split / 1.0 + amplitude * (y - split) / contrast,
        )
        return out

    sol = problem.solve(dirichlet=displacement)
    exact = problem.interpolate_stress(
        lambda x: np.broadcast_to(tensor, (len(np.atleast_2d(x)), 3, 3))
    )
    scale = np.abs(exact).max()
    assert np.abs(sol["stress"] - exact).max() / scale < 1e-13 * contrast + 1e-11
    assert np.abs(sol["solid_pressure"]).max() < (1e-13 * contrast + 1e-11) * scale


# -- structure -------------------------------------------------------------------


def test_the_assembled_system_has_the_two_point_structure():
    """Diagonal ``M``, diagonal ``(p_s, p_s)`` and ``(s, s)`` blocks, and every
    facet row coupling exactly two cells -- the minimal cell-centred fill-in."""
    mesh = structured_quads(3, 3)
    problem = TwoPointFourField(mesh, MU, LAM)
    S, _ = problem.assemble(dirichlet=linear_displacement)
    n1, nc = problem.n_stress, problem.n_cells

    M = S[:n1, :n1]
    assert (M - sp.diags(np.asarray(M.diagonal()).ravel())).nnz == 0
    B = S[n1 : n1 + nc, n1 : n1 + nc]
    assert (B - sp.diags(np.asarray(B.diagonal()).ravel())).nnz == 0
    Ss = S[n1 + 3 * nc :, n1 + 3 * nc :]
    assert (Ss - sp.diags(np.asarray(Ss.diagonal()).ravel())).nnz == 0

    csr = S.tocsr()
    d = 2
    for e in range(mesh.num_cells(1)):
        for k in range(d):
            cols = csr[d * e + k].indices
            cells = set()
            for start, stride in ((n1, 1), (n1 + nc, d), (n1 + 3 * nc, 1)):
                size = nc * stride
                cells |= {(c - start) // stride for c in cols if start <= c < start + size}
            assert len(cells) <= 2, f"facet {e} couples {len(cells)} cells"


def test_non_orthogonal_meshes_are_rejected():
    """The lumped guard is inherited: consistency lives on orthogonal complexes."""
    from mimetika.mesh import structured_triangles

    with pytest.raises(ValueError, match="orthogonal"):
        TwoPointFourField(structured_triangles(2, 2), MU, LAM)


# -- congruence and convergence ---------------------------------------------------


def test_three_and_four_field_two_point_agree():
    mesh = structured_quads(4, 4)
    three = TwoPointElasticity(mesh, MU, LAM)
    four = TwoPointFourField(mesh, MU, LAM)
    kwargs = dict(dirichlet=quadratic_displacement, body_force=quadratic_body_force)
    s3 = three.solve(**kwargs)
    s4 = four.solve(**kwargs)
    for field in ("stress", "displacement", "rotation"):
        assert np.allclose(s4[field], s3[field], rtol=1e-9, atol=1e-12), field


def test_convergence_with_no_stabilization_anywhere():
    errors = []
    for n in (4, 8, 16):
        mesh = structured_quads(n, n)
        problem = TwoPointFourField(mesh, MU, LAM)
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
        er = np.sqrt(
            vol @ (sol["rotation"] - problem.interpolate_rotation(grad_quadratic)) ** 2
        )
        errors.append((eu, es, er))
    last = np.log2(np.asarray(errors[-2]) / np.asarray(errors[-1]))
    assert last[0] > 1.5, f"u rate {last[0]:.2f}"
    assert last[1] > 1.4, f"sigma rate {last[1]:.2f}"
    assert last[2] > 1.4, f"s rate {last[2]:.2f}"


@pytest.mark.parametrize("nu", [0.3, 0.45, 0.49999])
def test_the_incompressible_range_solves(nu):
    mesh = structured_quads(3, 3)
    problem = TwoPointFourField(
        mesh, material=Material(shear_modulus=MU, poisson=nu)
    )
    sol = problem.solve(
        dirichlet=quadratic_displacement, body_force=quadratic_body_force
    )
    assert np.all(np.isfinite(sol["stress"]))
    assert np.all(np.isfinite(sol["solid_pressure"]))


# -- fracture mechanics -----------------------------------------------------------


def test_the_fracture_jump_is_recovered_exactly():
    """Soft fracture, prescribed tangential step: the jump is read back exactly.

    The compliance block uses the reduced (constant-basis) Gram, and the
    fracture facets leave the facet-average stencil -- the fault-adjacent
    cells close their surface integral one-sidedly.  The readout is the
    residual of the unfractured constitutive rows, whose fracture entries are
    the constant-basis coefficients of the jump directly.
    """
    from mimetika.mesh.fracture import facets_on_plane

    offset = 0.02
    recovered = []
    for n in (2, 4):
        mesh = structured_box(n, n, n)
        tags = facets_on_plane(mesh, [0.5, 0, 0], [1, 0, 0])
        contact = FractureContact(
            mesh, tags, facet_compliance=np.diag([1e6, 1e6, 1e6]), dofs_per_facet=3
        )

        def prescribed(x):
            x = np.atleast_2d(x)
            out = np.zeros((len(x), 3))
            out[:, 1] = offset if x[:, 0].mean() > 0.5 else 0.0
            return out

        problem = TwoPointFourField(mesh, mu=1.0, lam=1.0, contact=contact)
        matrix, rhs = problem.assemble_constrained(dirichlet=prescribed)
        sol = problem.split(
            solve_saddle(matrix, rhs, problem.block_sizes, method="direct")
        )
        x = np.concatenate([sol[k] for k in sol.blocks])
        residual = -(problem.constitutive_rows(contact=False) @ x)
        jumps = np.array([residual[3 * int(f) + np.arange(3)] for f in tags])
        recovered.append(np.abs(jumps[:, 1]).max())
        assert np.abs(sol["stress"]).max() < 1e-7  # only the physical traction
    assert recovered[0] == pytest.approx(offset, rel=1e-5)
    assert recovered[1] == pytest.approx(recovered[0], rel=1e-5)
