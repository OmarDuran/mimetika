import numpy as np

from conftest import counts
from mimetika.mesh import structured_box


def test_dd_is_zero(box_222):
    assert box_222.complex.verify_complex()


def test_incidence_products_vanish(box_222):
    cx = box_222.complex
    for k in (1, 2):
        prod = cx.boundary_matrix(k) @ cx.boundary_matrix(k + 1)
        assert prod.nnz == 0 or np.abs(prod.data).max() < 1e-13


def test_cell_counts_match_formula():
    for nx, ny, nz in [(1, 1, 1), (2, 2, 2), (3, 2, 1)]:
        mesh = structured_box(nx, ny, nz)
        V, E, F, C = counts(nx, ny, nz)
        assert mesh.num_cells(0) == V
        assert mesh.num_cells(1) == E
        assert mesh.num_cells(2) == F
        assert mesh.num_cells(3) == C


def test_euler_characteristic_of_solid_box():
    # A solid box is contractible: V - E + F - C = 1.
    for nx, ny, nz in [(1, 1, 1), (2, 3, 2)]:
        mesh = structured_box(nx, ny, nz)
        chi = (
            mesh.num_cells(0)
            - mesh.num_cells(1)
            + mesh.num_cells(2)
            - mesh.num_cells(3)
        )
        assert chi == 1


def test_incidence_shapes(box_222):
    cx = box_222.complex
    assert cx.boundary_matrix(1).shape == (cx.num_cells(0), cx.num_cells(1))
    assert cx.boundary_matrix(2).shape == (cx.num_cells(1), cx.num_cells(2))
    assert cx.boundary_matrix(3).shape == (cx.num_cells(2), cx.num_cells(3))


def test_interior_facets_shared_by_two_cells(box_222):
    # Each column of boundary[3] has entries; interior facets appear in two
    # cells with opposite sign, boundary facets in one.
    b3 = box_222.complex.boundary_matrix(3).tocsr()
    per_facet = np.asarray(np.abs(b3).sum(axis=1)).ravel()
    assert set(np.unique(per_facet)).issubset({1.0, 2.0})
