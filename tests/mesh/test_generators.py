import numpy as np

from conftest import counts
from mimetika.mesh import Mesh, structured_box


def test_structured_box_counts():
    mesh = structured_box(3, 2, 4)
    V, E, F, C = counts(3, 2, 4)
    assert len(mesh.geometry.points) == V
    assert mesh.num_cells(3) == C


def test_structured_box_is_valid_complex():
    assert structured_box(2, 2, 2).complex.verify_complex()


def test_origin_and_lengths():
    mesh = structured_box(1, 1, 1, lengths=(2.0, 3.0, 4.0), origin=(1.0, 1.0, 1.0))
    p = mesh.geometry.points
    assert np.allclose(p.min(0), [1.0, 1.0, 1.0])
    assert np.allclose(p.max(0), [3.0, 4.0, 5.0])


def test_from_cells_roundtrip(unit_cell):
    # Rebuild directly through Mesh.from_cells and check the complex is valid.
    pts = unit_cell.geometry.points
    cells = [
        [list(unit_cell.complex.facet_vertices[fid]) for fid, _ in entry]
        for entry in unit_cell.complex.cell_facets
    ]
    rebuilt = Mesh.from_cells(pts, cells)
    assert rebuilt.complex.verify_complex()
    assert rebuilt.num_cells(3) == 1
