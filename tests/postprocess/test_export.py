import numpy as np

from mimetika.postprocess import export_vtk, l2_error


def test_export_vtk_writes_polyhedra(box_222, tmp_path):
    path = export_vtk(
        tmp_path / "mesh.vtk",
        box_222,
        point_data={"u": np.arange(box_222.num_cells(0), dtype=float)},
        cell_data={"rho": np.ones(box_222.num_cells(3))},
    )
    text = path.read_text()
    assert "UNSTRUCTURED_GRID" in text
    assert "\n42" in text  # VTK_POLYHEDRON cell type
    assert f"POINTS {box_222.num_cells(0)}" in text
    assert "SCALARS u double" in text
    assert "SCALARS rho double" in text


def test_l2_error_zero_when_exact(box_222):
    # Discrete field equals the exact field sampled at centroids -> zero error.
    centroids = box_222.geometry.centroids(0)
    exact = lambda x: x[:, 0]  # noqa: E731
    discrete = exact(centroids)
    assert l2_error(box_222, 0, discrete, exact) < 1e-14


def test_l2_error_positive_when_off(box_222):
    discrete = np.zeros(box_222.num_cells(0))
    err = l2_error(box_222, 0, discrete, lambda x: np.ones(len(x)))
    assert err > 0
