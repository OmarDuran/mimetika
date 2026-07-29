"""Shared fixtures for the mimetika test suite."""

import numpy as np
import pytest

from mimetika.mesh import structured_box


@pytest.fixture
def unit_cell():
    """A single unit hexahedron."""
    return structured_box(1, 1, 1)


@pytest.fixture
def box_222():
    """A 2x2x2 structured box over the unit cube."""
    return structured_box(2, 2, 2)


@pytest.fixture
def box_333():
    """A 3x3x3 structured box over the unit cube."""
    return structured_box(3, 3, 3)


def counts(nx, ny, nz):
    """Expected (V, E, F, C) for a structured nx*ny*nz box."""
    V = (nx + 1) * (ny + 1) * (nz + 1)
    E = (
        nx * (ny + 1) * (nz + 1)
        + (nx + 1) * ny * (nz + 1)
        + (nx + 1) * (ny + 1) * nz
    )
    F = (
        (nx + 1) * ny * nz
        + nx * (ny + 1) * nz
        + nx * ny * (nz + 1)
    )
    C = nx * ny * nz
    return V, E, F, C


def boundary_vertices(mesh, tol=1e-12):
    """Indices of vertices on the boundary of the unit cube."""
    p = mesh.geometry.points
    on = np.zeros(len(p), dtype=bool)
    for axis in range(3):
        on |= np.isclose(p[:, axis], 0.0, atol=tol)
        on |= np.isclose(p[:, axis], 1.0, atol=tol)
    return np.where(on)[0]
