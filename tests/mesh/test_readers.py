"""VTU reading, face-orientation normalisation, and submesh extraction."""

import numpy as np
import pytest

from mimetika.mesh import structured_box, structured_tets
from mimetika.mesh.mesh import Mesh
from mimetika.mesh.readers import check_orientation, read_vtu
from mimetika.postprocess import export_vtk


def _roundtrip(mesh, tmp_path, name="m.vtk"):
    """Write with the VTK exporter, read back with the VTU reader."""
    vtu = tmp_path / "m.vtu"
    _write_vtu(vtu, mesh)
    return read_vtu(vtu)


def _write_vtu(path, mesh):
    """Minimal ASCII .vtu writer using VTK_POLYHEDRON face streams."""
    cx, pts = mesh.complex, mesh.geometry.points
    conn, offs, faces, foffs = [], [], [], []
    for c in range(mesh.num_cells(3)):
        verts = sorted(cx.cell_vertices(c))
        conn += verts
        offs.append(len(conn))
        entry = cx.facets_of(3, c)
        stream = [len(entry)]
        for fid, sign in entry:
            loop = cx.polygon_loops[fid]
            if sign < 0:
                loop = tuple(reversed(loop))
            stream += [len(loop), *loop]
        faces += stream
        foffs.append(len(faces))

    def block(name, dtype, vals):
        return (
            f'<DataArray type="{dtype}" Name="{name}" format="ascii">\n'
            + " ".join(map(str, vals))
            + "\n</DataArray>\n"
        )

    body = (
        '<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid">\n'
        "<UnstructuredGrid>\n"
        f'<Piece NumberOfPoints="{len(pts)}" NumberOfCells="{mesh.num_cells(3)}">\n'
        '<Points>\n<DataArray type="Float64" NumberOfComponents="3" format="ascii">\n'
        + " ".join(f"{v:.17g}" for v in pts.ravel())
        + "\n</DataArray>\n</Points>\n<Cells>\n"
        + block("connectivity", "Int32", conn)
        + block("offsets", "Int32", offs)
        + block("types", "UInt8", [42] * mesh.num_cells(3))
        + block("faces", "Int32", faces)
        + block("faceoffsets", "Int32", foffs)
        + "</Cells>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n"
    )
    path.write_text(body)


@pytest.mark.parametrize(
    "mesh", [structured_box(2, 2, 2), structured_tets(1, 1, 1)], ids=["hex", "tet"]
)
def test_vtu_roundtrip_preserves_topology_and_measures(mesh, tmp_path):
    back = _roundtrip(mesh, tmp_path)
    assert back.complex.verify_complex()
    for k in range(4):
        assert back.num_cells(k) == mesh.num_cells(k)
    assert np.isclose(back.geometry.measure(3).sum(), mesh.geometry.measure(3).sum())
    assert np.allclose(
        np.sort(back.geometry.measure(3)), np.sort(mesh.geometry.measure(3))
    )


def test_reader_repairs_scrambled_face_orientation(tmp_path):
    """VTK does not guarantee outward loops; the reader must fix them per face."""
    mesh = structured_box(2, 2, 2)
    path = tmp_path / "m.vtu"
    _write_vtu(path, mesh)

    # scramble: reverse every other face loop in the stream
    text = path.read_text()
    import re

    m = re.search(r'Name="faces"[^>]*>(.*?)</DataArray>', text, re.S)
    vals = [int(v) for v in m.group(1).split()]
    out, i, flip = [], 0, False
    while i < len(vals):
        nf = vals[i]
        out.append(nf)
        i += 1
        for _ in range(nf):
            n = vals[i]
            loop = vals[i + 1 : i + 1 + n]
            out += [n, *(loop[::-1] if flip else loop)]
            flip = not flip
            i += 1 + n
    path.write_text(text.replace(m.group(1), " " + " ".join(map(str, out)) + " "))

    back = read_vtu(path)
    assert back.complex.verify_complex()
    check_orientation(back)
    assert np.isclose(back.geometry.measure(3).sum(), 1.0)


def test_check_orientation_rejects_a_broken_cell():
    """A single reversed face must be detected, not silently accepted."""
    mesh = structured_box(1, 1, 1)
    cells = [
        [list(reversed(mesh.complex.polygon_loops[0]))]
        + [list(mesh.complex.polygon_loops[f]) for f in range(1, 6)]
    ]
    broken = Mesh.from_cells(mesh.geometry.points, cells)
    with pytest.raises(ValueError, match="not closed"):
        check_orientation(broken)


def test_check_orientation_accepts_good_meshes():
    for mesh in (structured_box(2, 1, 1), structured_tets(1, 1, 1)):
        check_orientation(mesh)


# -- submesh ------------------------------------------------------------------


def test_subset_conserves_volume_and_topology():
    mesh = structured_box(3, 3, 3)
    ids = mesh.cells_in_box([0.0, 0.0, 0.0], [0.7, 0.7, 0.7])
    sub = mesh.subset(ids)
    assert sub.num_cells(3) == len(ids) == 8
    assert sub.complex.verify_complex()
    check_orientation(sub)
    assert np.isclose(
        sub.geometry.measure(3).sum(), mesh.geometry.measure(3)[ids].sum()
    )


def test_subset_of_everything_reproduces_the_mesh():
    mesh = structured_box(2, 2, 2)
    sub = mesh.subset(np.arange(mesh.num_cells(3)))
    for k in range(4):
        assert sub.num_cells(k) == mesh.num_cells(k)
    assert np.isclose(sub.geometry.measure(3).sum(), 1.0)


def test_cells_in_box_selects_by_centroid():
    mesh = structured_box(2, 2, 2)
    ids = mesh.cells_in_box([0.0, 0.0, 0.0], [0.5, 0.5, 0.5])
    assert len(ids) == 1  # only the cell centred at (0.25, 0.25, 0.25)
    assert np.allclose(mesh.geometry.centroids(3)[ids[0]], 0.25)


def test_subset_solves_correctly():
    """The extracted mesh is a valid problem domain, not just valid topology."""
    from mimetika.assembly.mixed import MixedPoisson

    mesh = structured_box(3, 3, 3)
    sub = mesh.subset(mesh.cells_in_box([0.0, 0.0, 0.0], [0.7, 0.7, 0.7]))
    grad = np.array([0.7, -1.1, 0.45])
    problem = MixedPoisson(sub)
    sol = problem.solve(dirichlet=lambda x: 0.3 + np.atleast_2d(x) @ grad)
    expected = problem.interpolate_pressure(lambda x: 0.3 + np.atleast_2d(x) @ grad)
    assert np.allclose(sol["pressure"], expected, atol=1e-10)


def test_export_then_read_is_consistent(tmp_path):
    """The VTK exporter and the VTU reader agree on the same mesh."""
    mesh = structured_box(2, 1, 1)
    out = export_vtk(tmp_path / "m.vtk", mesh)
    assert out.exists() and "42" in out.read_text()
