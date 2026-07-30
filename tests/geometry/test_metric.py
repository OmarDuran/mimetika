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


# -- quadrature caching -------------------------------------------------------


def test_quadrature_cache_returns_identical_results():
    """Caching must not change any value (facets are shared by two cells)."""
    from mimetika.mesh.reference import reference_cells

    for rc in reference_cells():
        if rc.dim == 0:
            continue
        g = rc.mesh.geometry
        first = [g.quadrature(rc.dim, 0)]
        g.clear_quadrature_cache()
        second = [g.quadrature(rc.dim, 0)]
        for (p1, w1), (p2, w2) in zip(first, second):
            assert np.array_equal(p1, p2) and np.array_equal(w1, w2), rc.name


def test_quadrature_cache_is_used():
    from mimetika.mesh import structured_box

    mesh = structured_box(2, 2, 2)
    g = mesh.geometry
    g.clear_quadrature_cache()
    a = g.quadrature(3, 0)
    b = g.quadrature(3, 0)
    assert a[0] is b[0]  # same object -> served from the cache
    g.clear_quadrature_cache()
    assert g.quadrature(3, 0)[0] is not a[0]


def test_polygon_quadrature_handles_degenerate_fan_triangles():
    """A repeated vertex gives a zero-area fan triangle; it must contribute 0."""
    from mimetika.mesh.mesh import Mesh

    # a unit square with a duplicated collinear point on one edge
    pts = np.array(
        [[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float
    )
    mesh = Mesh.from_polygons(pts, [[0, 1, 2, 3, 4]])
    p, w = mesh.geometry.quadrature(2, 0)
    assert np.isclose(w.sum(), 1.0)
    assert np.isclose((w * p[:, 0] ** 2).sum(), 1 / 3)
