import numpy as np

from mimetika.operators import curl, div, grad


def test_curl_grad_is_zero(box_222):
    cg = curl(box_222) @ grad(box_222)
    assert cg.nnz == 0 or np.abs(cg.data).max() < 1e-13


def test_div_curl_is_zero(box_222):
    dc = div(box_222) @ curl(box_222)
    assert dc.nnz == 0 or np.abs(dc.data).max() < 1e-13


def test_operator_shapes(box_222):
    m = box_222
    assert grad(m).shape == (m.num_cells(1), m.num_cells(0))
    assert curl(m).shape == (m.num_cells(2), m.num_cells(1))
    assert div(m).shape == (m.num_cells(3), m.num_cells(2))


def test_grad_of_linear_field_matches_edge_differences(unit_cell):
    # For a 0-form sampled at vertices, (grad u)_e = u(head) - u(tail).
    p = unit_cell.geometry.points
    u = p[:, 0] + 2 * p[:, 1] - 3 * p[:, 2]
    ev = unit_cell.complex.edge_vertices
    expected = u[ev[:, 1]] - u[ev[:, 0]]
    assert np.allclose(grad(unit_cell) @ u, expected)
