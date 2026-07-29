import numpy as np
import pytest

from mimetika.mesh import structured_box
from mimetika.mesh.reference import reference_cells
from mimetika.operators.diffusion import DiffusionInnerProduct
from mimetika.operators.inner_product import (
    assemble_local_inner_product,
    consistency_residual,
)

ANISO_K = np.array([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]])
CELLS = [c for c in reference_cells() if c.dim >= 1]
IDS = [c.name for c in CELLS]
BASES = ["const", "rt0"]


def _local(rc, basis, K=ANISO_K):
    ip = DiffusionInnerProduct(rc.mesh, K=K, basis=basis)
    N, R, Kbar, vol, lc = ip.local_matrices(0)
    return ip, N, R, Kbar, vol, lc


@pytest.mark.parametrize("basis", BASES)
@pytest.mark.parametrize("rc", CELLS, ids=IDS)
def test_local_matrix_is_spd(rc, basis):
    _, N, R, Kbar, vol, _ = _local(rc, basis)
    M = assemble_local_inner_product(N, R, Kbar, vol)
    assert np.allclose(M, M.T)
    assert np.linalg.eigvalsh(M).min() > 0


@pytest.mark.parametrize("basis", BASES)
@pytest.mark.parametrize("rc", CELLS, ids=IDS)
def test_gram_identity_holds(rc, basis):
    """``N^T R = |E| Kbar`` -- every column of R is canonical here."""
    _, N, R, Kbar, vol, _ = _local(rc, basis)
    assert np.abs(N.T @ R - vol * Kbar).max() < 1e-10


@pytest.mark.parametrize("basis", BASES)
@pytest.mark.parametrize("rc", CELLS, ids=IDS)
def test_strong_consistency(rc, basis):
    _, N, R, Kbar, vol, _ = _local(rc, basis)
    M = assemble_local_inner_product(N, R, Kbar, vol)
    assert consistency_residual(M, N, R) < 1e-10


@pytest.mark.parametrize("basis", BASES)
@pytest.mark.parametrize("rc", CELLS, ids=IDS)
def test_dof_and_mode_counts(rc, basis):
    ip, N, _, _, _, lc = _local(rc, basis)
    assert N.shape == (lc.n_facets, ip.n_modes(lc.dim))


def test_rt0_stabilization_vanishes_on_every_simplex():
    for name in ("segment-unit", "triangle-unit", "triangle-irregular", "tet-reference", "tet-irregular"):
        rc = next(c for c in CELLS if c.name == name)
        ip = DiffusionInnerProduct(rc.mesh, basis="rt0")
        assert ip.stabilization_dim(0) == 0, name


def test_stabilization_dims_on_polytopes():
    expected = {
        # name: (const, rt0)
        "square-unit": (2, 1),
        "hexagon-regular": (4, 3),
        "cube-unit": (3, 2),
        "prism-triangular": (2, 1),
        "pyramid-square": (2, 1),
        "cube-dented": (6, 5),
    }
    for name, (n_const, n_rt0) in expected.items():
        rc = next(c for c in CELLS if c.name == name)
        assert DiffusionInnerProduct(rc.mesh, basis="const").stabilization_dim(0) == n_const
        assert DiffusionInnerProduct(rc.mesh, basis="rt0").stabilization_dim(0) == n_rt0


@pytest.mark.parametrize("rc", CELLS, ids=IDS)
def test_exact_on_constant_fluxes(rc):
    """``g^T M g`` reproduces ``int K^{-1} F.F`` for any constant flux."""
    ip, N, R, Kbar, vol, lc = _local(rc, "const")
    M = assemble_local_inner_product(N, R, Kbar, vol)
    Kloc = lc.project_tensor(ANISO_K)
    Kinv = np.linalg.inv(Kloc)
    rng = np.random.default_rng(0)
    for _ in range(3):
        F = rng.standard_normal(lc.dim)
        g = lc.facet_normals @ F
        assert np.isclose(g @ M @ g, vol * (F @ Kinv @ F))


def test_global_assembly_symmetric_and_spd():
    mesh = structured_box(2, 1, 1)
    A = DiffusionInnerProduct(mesh).assemble()
    assert A.shape == (mesh.num_cells(2), mesh.num_cells(2))
    assert (abs(A - A.T) > 1e-12).nnz == 0
    assert np.linalg.eigvalsh(A.toarray()).min() > 0
