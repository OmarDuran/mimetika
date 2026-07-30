"""VTU (VTK XML) export: structure, component counts, and round-tripping."""

import re

import numpy as np
import pytest

from mimetika.mesh import structured_box, structured_tets
from mimetika.mesh.readers import read_vtu
from mimetika.postprocess import export_vtu

MESHES = [("hex", structured_box(2, 2, 2)), ("tet", structured_tets(1, 1, 1))]
IDS = [n for n, _ in MESHES]
ONLY = [m for _, m in MESHES]


def _arrays(text):
    """Map ``name -> NumberOfComponents`` for every DataArray in the file."""
    out = {}
    for m in re.finditer(r"<DataArray([^>]*)>", text):
        attrs = m.group(1)
        name = re.search(r'Name="([^"]+)"', attrs)
        comps = re.search(r'NumberOfComponents="(\d+)"', attrs)
        if name:
            out[name.group(1)] = int(comps.group(1)) if comps else 1
    return out


@pytest.mark.parametrize("mesh", ONLY, ids=IDS)
def test_writes_polyhedral_cells(mesh, tmp_path):
    path = export_vtu(tmp_path / "m.vtu", mesh)
    text = path.read_text()
    assert 'type="UnstructuredGrid"' in text
    assert f'NumberOfCells="{mesh.num_cells(3)}"' in text
    assert f'NumberOfPoints="{mesh.num_cells(0)}"' in text
    assert "faces" in text and "faceoffsets" in text  # polyhedron face streams
    assert re.search(r'Name="types"[^>]*>\s*(42\s*)+', text)


@pytest.mark.parametrize("mesh", ONLY, ids=IDS)
def test_topology_arrays_are_single_component(mesh, tmp_path):
    """A flat array is n rows of one component -- not one row of n."""
    path = export_vtu(tmp_path / "m.vtu", mesh)
    comps = _arrays(path.read_text())
    for name in ("connectivity", "offsets", "types", "faces", "faceoffsets"):
        assert comps[name] == 1, name
    assert comps["Points"] == 3


def test_field_shapes_map_to_component_counts(tmp_path):
    mesh = structured_box(2, 2, 2)
    n = mesh.num_cells(3)
    path = export_vtu(
        tmp_path / "m.vtu",
        mesh,
        cell_data={
            "scalar": np.arange(n, dtype=float),
            "vector": np.zeros((n, 3)),
            "tensor": np.zeros((n, 3, 3)),
        },
    )
    comps = _arrays(path.read_text())
    assert comps["scalar"] == 1
    assert comps["vector"] == 3
    assert comps["tensor"] == 9


def test_values_round_trip_through_the_file(tmp_path):
    mesh = structured_box(2, 2, 2)
    n = mesh.num_cells(3)
    rng = np.random.default_rng(0)
    scalar = rng.standard_normal(n)
    vector = rng.standard_normal((n, 3))
    path = export_vtu(
        tmp_path / "m.vtu", mesh, cell_data={"s": scalar, "v": vector}
    )
    text = path.read_text()

    def read(name, ncomp):
        m = re.search(r'Name="%s"[^>]*>(.*?)</DataArray>' % name, text, re.S)
        return np.fromstring(m.group(1), sep=" ").reshape(-1, ncomp)

    assert np.allclose(read("s", 1).ravel(), scalar)
    assert np.allclose(read("v", 3), vector)


@pytest.mark.parametrize("mesh", ONLY, ids=IDS)
def test_written_mesh_can_be_read_back(mesh, tmp_path):
    """The writer and the reader agree: geometry and topology survive."""
    path = export_vtu(tmp_path / "m.vtu", mesh)
    back = read_vtu(path)
    assert back.complex.verify_complex()
    for k in range(4):
        assert back.num_cells(k) == mesh.num_cells(k)
    assert np.isclose(
        back.geometry.measure(3).sum(), mesh.geometry.measure(3).sum()
    )


def test_point_data_is_written(tmp_path):
    mesh = structured_box(1, 1, 1)
    path = export_vtu(
        tmp_path / "m.vtu",
        mesh,
        point_data={"u": np.arange(mesh.num_cells(0), dtype=float)},
    )
    text = path.read_text()
    assert "<PointData>" in text and 'Name="u"' in text
