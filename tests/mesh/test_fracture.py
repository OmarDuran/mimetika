"""Fracture tagging, ``.vtu`` round trip, and extraction of the 2D subdomain.

Everything runs through real files: build a small mesh, tag a fracture plane,
write a ``.vtu``, read it back, resolve the tags.  Facet numbering is re-derived
on read, so the round trip is itself the proof that tags are stored by vertex
set rather than by index.
"""

import numpy as np
import pytest

from mimetika.mesh import structured_box, structured_tets
from mimetika.mesh.fracture import (
    facets_on_plane,
    fracture_mesh,
    interior_facets,
    read_fracture_tags,
    write_fracture_tags,
)
from mimetika.mesh.readers import check_orientation, read_vtu
from mimetika.postprocess import export_vtu


# -- tagging ------------------------------------------------------------------


def test_facets_on_plane_finds_the_expected_count():
    mesh = structured_box(2, 2, 2)
    tags = facets_on_plane(mesh, [0, 0, 0.5], [0, 0, 1])
    assert len(tags) == 4  # the 2x2 layer of horizontal faces at mid-height
    centroids = mesh.geometry.centroids(2)[tags]
    assert np.allclose(centroids[:, 2], 0.5)


def test_facets_on_plane_excludes_boundary_by_default():
    """A boundary facet has one side and cannot carry a fracture."""
    mesh = structured_box(2, 2, 2)
    assert len(facets_on_plane(mesh, [0, 0, 0.0], [0, 0, 1])) == 0
    outer = facets_on_plane(mesh, [0, 0, 0.0], [0, 0, 1], interior_only=False)
    assert len(outer) == 4
    assert set(outer).isdisjoint(set(interior_facets(mesh)))


def test_every_tagged_facet_is_interior():
    mesh = structured_box(3, 3, 3)
    tags = facets_on_plane(mesh, [0, 0, 1 / 3], [0, 0, 1])
    assert len(tags) == 9
    assert set(tags) <= set(interior_facets(mesh))


def test_tagging_works_on_a_tetrahedral_mesh():
    mesh = structured_tets(2, 2, 2)
    tags = facets_on_plane(mesh, [0, 0, 0.5], [0, 0, 1])
    assert len(tags) > 0
    # every tagged facet is a triangle lying in the plane
    for f in tags:
        loop = mesh.complex.polygon_loops[f]
        assert len(loop) == 3
        assert np.allclose(mesh.geometry.points[list(loop)][:, 2], 0.5)


# -- .vtu round trip ----------------------------------------------------------


def test_tags_survive_a_vtu_round_trip(tagged_vtu):
    path, mesh, tags = tagged_vtu()
    reread = read_vtu(path)
    resolved = read_fracture_tags(path, reread)
    assert sorted(resolved) == sorted(tags)


def test_tags_are_resolved_by_vertex_set_not_index(tagged_vtu):
    """The re-read mesh derives its own facet numbering; tags must still land."""
    path, mesh, tags = tagged_vtu()
    reread = read_vtu(path)
    resolved = read_fracture_tags(path, reread)
    a = np.sort(mesh.geometry.centroids(2)[tags], axis=0)
    b = np.sort(reread.geometry.centroids(2)[resolved], axis=0)
    assert np.allclose(a, b)


def test_file_without_tags_returns_empty(tmp_path):
    mesh = structured_box(1, 1, 1)
    path = export_vtu(tmp_path / "plain.vtu", mesh)
    assert len(read_fracture_tags(path, mesh)) == 0


def test_unknown_tagged_face_is_rejected(tmp_path):
    """A stored loop that is not a facet of the mesh must not pass silently."""
    mesh = structured_box(1, 1, 1)
    path = export_vtu(tmp_path / "m.vtu", mesh)
    text = path.read_text().replace(
        "<UnstructuredGrid>",
        '<UnstructuredGrid>\n<FieldData>\n'
        '<DataArray type="Int64" Name="fracture_faces" format="ascii">\n0 1 2\n'
        "</DataArray>\n"
        '<DataArray type="Int64" Name="fracture_face_offsets" format="ascii">\n3\n'
        "</DataArray>\n</FieldData>\n",
        1,
    )
    path.write_text(text)
    with pytest.raises(KeyError, match="not a facet"):
        read_fracture_tags(path, mesh)


def test_reader_still_finds_points_when_fielddata_precedes_them(tagged_vtu):
    """Tags are written before <Points>; the reader must not index by position."""
    path, mesh, _ = tagged_vtu()
    assert path.read_text().index("<FieldData>") < path.read_text().index("<Points>")
    reread = read_vtu(path)
    assert np.allclose(
        np.sort(reread.geometry.points, axis=0), np.sort(mesh.geometry.points, axis=0)
    )


# -- extraction ---------------------------------------------------------------


def test_fracture_mesh_matches_the_tagged_facets(tagged_vtu):
    _, mesh, tags = tagged_vtu()
    frac, facet_of = fracture_mesh(mesh, tags)

    assert frac.dim == 2
    assert frac.num_cells(2) == len(tags)
    assert frac.complex.verify_complex()
    assert np.isclose(
        frac.geometry.measure(2).sum(), mesh.geometry.measure(2)[tags].sum()
    )
    assert np.allclose(
        frac.geometry.centroids(2), mesh.geometry.centroids(2)[facet_of]
    )


def test_fracture_mesh_compacts_vertices(tagged_vtu):
    _, mesh, tags = tagged_vtu()
    frac, _ = fracture_mesh(mesh, tags)
    # a 2x2 patch of quads uses a 3x3 grid of nodes, not the parent's 27
    assert frac.num_cells(0) == 9
    assert frac.num_cells(0) < mesh.num_cells(0)


def test_fracture_mesh_is_a_connected_2d_complex(tagged_vtu):
    _, mesh, tags = tagged_vtu()
    frac, _ = fracture_mesh(mesh, tags)
    # Euler characteristic of a simply-connected 2D patch
    chi = frac.num_cells(0) - frac.num_cells(1) + frac.num_cells(2)
    assert chi == 1


def test_fracture_mesh_on_tets_gives_triangles():
    mesh = structured_tets(2, 2, 2)
    tags = facets_on_plane(mesh, [0, 0, 0.5], [0, 0, 1])
    frac, _ = fracture_mesh(mesh, tags)
    assert frac.num_cells(2) == len(tags)
    assert all(len(loop) == 3 for loop in frac.complex.polygon_loops)
    assert np.isclose(frac.geometry.measure(2).sum(), 1.0)


def test_fracture_of_a_single_interior_face():
    """The minimal case: two cells sharing one face."""
    mesh = structured_box(2, 1, 1)
    tags = facets_on_plane(mesh, [0.5, 0, 0], [1, 0, 0])
    assert len(tags) == 1
    frac, _ = fracture_mesh(mesh, tags)
    assert frac.num_cells(2) == 1 and frac.num_cells(0) == 4
    assert np.isclose(frac.geometry.measure(2)[0], 1.0)


def test_partial_fracture_keeps_a_tip(tagged_vtu):
    """Tagging only part of a plane leaves an immersed tip."""
    _, mesh, tags = tagged_vtu(keep=2)
    assert len(tags) == 2
    frac, _ = fracture_mesh(mesh, tags)
    assert frac.num_cells(2) == 2
    assert frac.complex.verify_complex()


def test_extraction_does_not_touch_the_parent_mesh(tagged_vtu):
    _, mesh, tags = tagged_vtu()
    before = (
        mesh.num_cells(0), mesh.num_cells(1), mesh.num_cells(2), mesh.num_cells(3)
    )
    volume = mesh.geometry.measure(3).sum()
    fracture_mesh(mesh, tags)
    after = (
        mesh.num_cells(0), mesh.num_cells(1), mesh.num_cells(2), mesh.num_cells(3)
    )
    assert before == after
    assert np.isclose(mesh.geometry.measure(3).sum(), volume)
    assert mesh.complex.verify_complex()
    check_orientation(mesh)
