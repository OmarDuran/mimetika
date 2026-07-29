import numpy as np


def test_unit_cell_measures(unit_cell):
    g = unit_cell.geometry
    assert np.allclose(g.measure(0), 1.0)
    assert np.allclose(g.measure(1), 1.0)  # all edges length 1
    assert np.allclose(g.measure(2), 1.0)  # all faces area 1
    assert np.allclose(g.measure(3), 1.0)  # volume 1


def test_volume_sums_to_domain(box_222):
    assert np.isclose(box_222.geometry.measure(3).sum(), 1.0)
    assert np.allclose(box_222.geometry.measure(3), 1.0 / 8.0)


def test_facet_normals_unit(box_222):
    n = box_222.geometry.facet_normals()
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)


def test_centroids_inside_unit_cube(box_222):
    for k in range(4):
        c = box_222.geometry.centroids(k)
        assert c.shape == (box_222.num_cells(k), 3)
        assert (c >= -1e-12).all() and (c <= 1 + 1e-12).all()


def test_edge_length_scaling():
    from mimetika.mesh import structured_box

    mesh = structured_box(2, 2, 2, lengths=(2.0, 2.0, 2.0))
    # cell size h = 1.0 in each direction -> all edges length 1
    assert np.allclose(mesh.geometry.measure(1), 1.0)
    assert np.isclose(mesh.geometry.measure(3).sum(), 8.0)
