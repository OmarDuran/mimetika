"""Topological and geometric checks over the whole reference cell collection."""

import numpy as np
import pytest

from mimetika.mesh.reference import reference_cells

ALL = reference_cells()
POSITIVE_DIM = [c for c in ALL if c.dim >= 1]


def ids(cells):
    return [c.name for c in cells]


# -- topology -----------------------------------------------------------------


@pytest.mark.parametrize("rc", ALL, ids=ids(ALL))
def test_complex_is_valid(rc):
    """dd = 0 on every cell of every dimension."""
    assert rc.mesh.complex.verify_complex()


@pytest.mark.parametrize("rc", ALL, ids=ids(ALL))
def test_dimension_and_single_top_cell(rc):
    assert rc.mesh.dim == rc.dim
    assert rc.mesh.num_cells(rc.dim) == 1


@pytest.mark.parametrize("rc", ALL, ids=ids(ALL))
def test_euler_characteristic_is_one(rc):
    """A single closed cell is contractible: alternating sum of counts == 1."""
    m = rc.mesh
    chi = sum((-1) ** k * m.num_cells(k) for k in range(rc.dim + 1))
    assert chi == 1


@pytest.mark.parametrize("rc", POSITIVE_DIM, ids=ids(POSITIVE_DIM))
def test_facets_belong_to_the_single_cell_once(rc):
    """Every facet of the one top cell appears exactly once, with unit sign."""
    m = rc.mesh
    facets = m.complex.facets_of(rc.dim, 0)
    assert len(facets) == m.num_cells(rc.dim - 1)
    assert {abs(s) for _, s in facets} == {1}
    assert len({f for f, _ in facets}) == len(facets)  # no repeats


@pytest.mark.parametrize("rc", POSITIVE_DIM, ids=ids(POSITIVE_DIM))
def test_boundary_of_boundary_columns_vanish(rc):
    """The signed facets of a cell form a closed boundary."""
    m = rc.mesh
    if rc.dim < 2:
        pytest.skip("no second boundary operator in 1D")
    prod = m.complex.boundary_matrix(rc.dim - 1) @ m.complex.boundary_matrix(rc.dim)
    assert prod.nnz == 0 or np.abs(prod.data).max() < 1e-13


def test_simplex_detection():
    by_name = {c.name: c for c in ALL}
    assert by_name["segment-unit"].mesh.complex.is_simplex(0)
    assert by_name["triangle-unit"].mesh.complex.is_simplex(0)
    assert by_name["tet-reference"].mesh.complex.is_simplex(0)
    assert by_name["tet-irregular"].mesh.complex.is_simplex(0)
    assert not by_name["square-unit"].mesh.complex.is_simplex(0)
    assert not by_name["cube-unit"].mesh.complex.is_simplex(0)
    assert not by_name["pyramid-square"].mesh.complex.is_simplex(0)


def test_expected_counts_for_known_cells():
    by_name = {c.name: c for c in ALL}
    # (vertices, edges, faces, cells)
    tet = by_name["tet-reference"].mesh
    assert [tet.num_cells(k) for k in range(4)] == [4, 6, 4, 1]
    cube = by_name["cube-unit"].mesh
    assert [cube.num_cells(k) for k in range(4)] == [8, 12, 6, 1]
    prism = by_name["prism-triangular"].mesh
    assert [prism.num_cells(k) for k in range(4)] == [6, 9, 5, 1]
    pyr = by_name["pyramid-square"].mesh
    assert [pyr.num_cells(k) for k in range(4)] == [5, 8, 5, 1]
    hexagon = by_name["hexagon-regular"].mesh
    assert [hexagon.num_cells(k) for k in range(3)] == [6, 6, 1]


# -- geometry -----------------------------------------------------------------


@pytest.mark.parametrize("rc", ALL, ids=ids(ALL))
def test_measure_matches_analytic_value(rc):
    assert np.isclose(rc.mesh.geometry.measure(rc.dim)[0], rc.measure, rtol=0, atol=1e-12)


@pytest.mark.parametrize("rc", POSITIVE_DIM, ids=ids(POSITIVE_DIM))
def test_quadrature_weights_sum_to_measure(rc):
    """Independent cross-check: signed decomposition vs. the measure formula."""
    _, w = rc.mesh.geometry.quadrature(rc.dim, 0)
    assert np.isclose(w.sum(), rc.measure)


@pytest.mark.parametrize("rc", POSITIVE_DIM, ids=ids(POSITIVE_DIM))
def test_facet_measures_are_positive(rc):
    assert (rc.mesh.geometry.measure(rc.dim - 1) > 0).all()


@pytest.mark.parametrize("rc", POSITIVE_DIM, ids=ids(POSITIVE_DIM))
def test_centroid_lies_in_bounding_box(rc):
    m = rc.mesh
    c = m.geometry.centroids(rc.dim)[0]
    p = m.geometry.points
    assert (c >= p.min(0) - 1e-12).all() and (c <= p.max(0) + 1e-12).all()


def test_centroids_of_symmetric_cells():
    by_name = {c.name: c for c in ALL}

    def centroid(name):
        rc = by_name[name]
        return rc.mesh.geometry.centroids(rc.dim)[0]

    assert np.allclose(centroid("segment-unit"), [0.5, 0, 0])
    assert np.allclose(centroid("square-unit"), [0.5, 0.5, 0])
    assert np.allclose(centroid("triangle-unit"), [1 / 3, 1 / 3, 0])
    assert np.allclose(centroid("hexagon-regular"), [0, 0, 0], atol=1e-14)
    # L-shape = [0,2]x[0,1] (area 2) + [0,1]x[1,2] (area 1)
    assert np.allclose(centroid("lshape-nonconvex"), [2.5 / 3, 2.5 / 3, 0])
    assert np.allclose(centroid("cube-unit"), [0.5, 0.5, 0.5])
    assert np.allclose(centroid("tet-reference"), [0.25, 0.25, 0.25])
    assert np.allclose(centroid("prism-triangular"), [1 / 3, 1 / 3, 1.0])
    # pyramid centroid sits at h/4 above the base
    assert np.allclose(centroid("pyramid-square"), [0.5, 0.5, 0.3])


def test_quadrature_integrates_quadratics_exactly():
    """Against hand-computed integrals of x^2 on canonical shapes."""
    by_name = {c.name: c for c in ALL}
    expected = {
        "segment-unit": 1 / 3,
        "triangle-unit": 1 / 12,
        "square-unit": 1 / 3,
        "tet-reference": 1 / 60,
        "cube-unit": 1 / 3,
    }
    for name, exact in expected.items():
        rc = by_name[name]
        p, w = rc.mesh.geometry.quadrature(rc.dim, 0)
        assert np.isclose((w * p[:, 0] ** 2).sum(), exact), name


def test_tilted_cells_match_their_flat_originals():
    """A rigid rotation must not change any measure."""
    by_name = {c.name: c for c in ALL}
    for flat, tilted in [("square-unit", "square-tilted"), ("lshape-nonconvex", "lshape-tilted")]:
        a, b = by_name[flat], by_name[tilted]
        assert np.isclose(a.mesh.geometry.measure(2)[0], b.mesh.geometry.measure(2)[0])
        assert np.allclose(
            np.sort(a.mesh.geometry.measure(1)), np.sort(b.mesh.geometry.measure(1))
        )


def test_facet_normals_are_unit_and_outward_on_polyhedra():
    for rc in reference_cells(dim=3):
        g = rc.mesh.geometry
        n = g.facet_normals()
        assert np.allclose(np.linalg.norm(n, axis=1), 1.0)
        # signed outward normals must integrate to zero over a closed surface
        facets = rc.mesh.complex.facets_of(3, 0)
        area = g.measure(2)
        total = sum(s * area[f] * n[f] for f, s in facets)
        assert np.allclose(total, 0.0, atol=1e-12), rc.name
