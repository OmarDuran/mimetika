import numpy as np

from conftest import boundary_vertices
from mimetika.assembly import LinearSystem, apply_dirichlet, hodge_laplacian
from mimetika.operators import DiagonalHodge
from mimetika.solver import solve


def test_scipy_solver_recovers_harmonic_field(box_333):
    # On a uniform grid the lumped Hodge-Laplacian is the graph Laplacian
    # (up to h); u = x is discretely harmonic, so a Dirichlet solve with
    # boundary data u=x must recover x in the interior to machine precision.
    mesh = box_333
    h = DiagonalHodge(mesh.geometry)
    L = hodge_laplacian(mesh, h, 0)

    u_exact = mesh.geometry.points[:, 0].copy()
    b = np.zeros(mesh.num_cells(0))

    bverts = boundary_vertices(mesh)
    A, rhs = apply_dirichlet(L, b, bverts, u_exact[bverts])

    x = solve(LinearSystem(A, rhs), backend="scipy")
    assert np.allclose(x, u_exact, atol=1e-8)


def test_solver_auto_backend_runs(box_222):
    mesh = box_222
    h = DiagonalHodge(mesh.geometry)
    L = hodge_laplacian(mesh, h, 0)
    b = np.zeros(mesh.num_cells(0))
    bverts = boundary_vertices(mesh)
    vals = mesh.geometry.points[bverts, 1]
    A, rhs = apply_dirichlet(L, b, bverts, vals)

    x = solve(LinearSystem(A, rhs))  # auto -> scipy here (no petsc4py)
    assert x.shape == (mesh.num_cells(0),)
    assert np.allclose(x[bverts], vals)
