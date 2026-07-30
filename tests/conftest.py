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


# -- small tagged fracture meshes ---------------------------------------------
#
# The fracture examples are driven from real .vtu files so the tests exercise
# the whole path: build -> tag -> write -> read back -> resolve tags.  Facet
# numbering is re-derived on read, so a round trip is itself the check that
# tags are stored by vertex set rather than by index.


@pytest.fixture
def tagged_vtu(tmp_path):
    """Factory building a small mesh with a tagged fracture plane, on disk."""
    from mimetika.mesh import structured_box, structured_tets
    from mimetika.mesh.fracture import facets_on_plane, write_fracture_tags
    from mimetika.postprocess import export_vtu

    def build(
        shape=(2, 2, 2),
        point=(0.0, 0.0, 0.5),
        normal=(0.0, 0.0, 1.0),
        name="fractured.vtu",
        tets=False,
        keep=None,
    ):
        make = structured_tets if tets else structured_box
        mesh = make(*shape)
        tags = facets_on_plane(mesh, point, normal)
        if keep is not None:  # partial fracture -> an immersed tip
            tags = tags[:keep]
        path = export_vtu(tmp_path / name, mesh)
        write_fracture_tags(path, mesh, tags)
        return path, mesh, tags

    return build
