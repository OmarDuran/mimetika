r"""Nondimensionalising the saddle system before the factorisation.

In physical units the leading block is a compliance of order ``1/G`` while the
constraint block is a discrete divergence of order one.  For rock
(``G ~ 6.5e9``) that spread makes ``cond(A) ~ 2e11``, and MUMPS reports a zero
pivot -- ``KSP_DIVERGED_PC_FAILED`` -- on systems SuperLU happens to survive.

The conditioning is **intrinsic to the units**, not a diagonal artefact, so row
equilibration cannot remove it: row one already mixes both scales, so its
maximum is set by the constraint entries and dividing by it leaves the
compliance untouched.  Scaling the two *fields* against each other does work,
and is what :func:`block_scaling` returns.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from mimetika.solver.saddle import block_scaling, solve_saddle


def saddle(n=12, modulus=1.0, seed=0):
    """A symmetric indefinite ``[[M, B^T], [B, 0]]`` with ``M`` scaled by 1/modulus."""
    rng = np.random.default_rng(seed)
    root = rng.standard_normal((n, n))
    M = (root @ root.T + n * np.eye(n)) / modulus
    B = rng.standard_normal((n // 2, n))
    A = sp.csr_matrix(np.block([[M, B.T], [B, np.zeros((n // 2, n // 2))]]))
    return A, (n, n // 2)


def test_scaling_is_skipped_when_the_blocks_already_match():
    """No transform when it would buy nothing -- ``None`` means 'leave it alone'."""
    A, blocks = saddle(modulus=1.0)
    assert block_scaling(A, blocks) is None


def test_scaling_collapses_the_condition_number():
    A, blocks = saddle(modulus=6.5e9)
    scaling = block_scaling(A, blocks)
    assert scaling is not None

    D = sp.diags(scaling)
    before = np.linalg.cond(A.toarray())
    after = np.linalg.cond((D @ A @ D).toarray())
    assert before > 1e9
    assert after < 1e-6 * before


def test_row_equilibration_does_not_help():
    """Pins *why* the block form is needed rather than the obvious alternative."""
    A, _ = saddle(modulus=6.5e9)
    rows = np.asarray(abs(A).max(axis=1).todense()).ravel()
    rows[rows <= 0] = 1.0
    D = sp.diags(1.0 / np.sqrt(rows))
    before = np.linalg.cond(A.toarray())
    after = np.linalg.cond((D @ A @ D).toarray())
    assert after > 0.1 * before  # essentially unchanged


@pytest.mark.parametrize("modulus", [1.0, 1e6, 6.5e9, 1e12])
def test_the_solution_is_unchanged_by_scaling(modulus):
    """It is a similarity transform: same answer, both ways, at every scale."""
    A, blocks = saddle(modulus=modulus)
    rhs = np.arange(A.shape[0], dtype=float) + 1.0
    scaled = solve_saddle(A, rhs, blocks, backend="scipy", method="direct")
    plain = solve_saddle(
        A, rhs, blocks, backend="scipy", method="direct", scale_blocks=False
    )
    reference = np.linalg.solve(A.toarray(), rhs)
    assert np.allclose(scaled, reference, rtol=1e-8)
    assert np.allclose(plain, reference, rtol=1e-6)


def test_scaling_survives_a_degenerate_block():
    """An all-zero constraint block has no scale to match: return ``None``."""
    n = 6
    M = sp.eye(n, format="csr")
    A = sp.bmat([[M, None], [None, sp.csr_matrix((3, 3))]], format="csr")
    assert block_scaling(A, (n, 3)) is None


def test_block_sizes_out_of_range_are_ignored():
    A, _ = saddle()
    assert block_scaling(A, (0, A.shape[0])) is None
    assert block_scaling(A, (A.shape[0],)) is None


def test_the_stiff_poromechanics_system_factorises():
    """The regression this exists for: a rock-modulus system must solve cleanly.

    Ran green under scipy's SuperLU and failed under PETSc/MUMPS before the
    scaling, so it is checked on whichever backend is installed.
    """
    from mimetika.assembly.mixed import boundary_facets
    from mimetika.assembly.poromechanics import PoroMechanics
    from mimetika.materials import Material
    from mimetika.mesh import structured_quads

    shear, poisson, biot, depletion = 6500e6, 0.15, 0.9, -25e6
    mesh = structured_quads(6, 6)
    problem = PoroMechanics(
        mesh, Material(shear_modulus=shear, poisson=poisson, biot=biot)
    )
    centroids = mesh.geometry.centroids(1)
    boundary = boundary_facets(mesh)
    top = [f for f in boundary if abs(centroids[f][1] - 1.0) < 1e-12]

    solution = problem.solve(
        dt=None,
        dirichlet=lambda x: np.zeros((len(np.atleast_2d(x)), 3)),
        pressure=depletion,
        traction=lambda x: np.zeros((len(np.atleast_2d(x)), 3, 3)),
        traction_facets=top,
        roller_facets=[f for f in boundary if f not in set(top)],
    )
    exact = biot * depletion / (2 * shear * (1 - poisson) / (1 - 2 * poisson))
    assert problem.volumetric_strain(solution).mean() == pytest.approx(
        exact, rel=1e-10
    )
