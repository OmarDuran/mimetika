"""Facet DOF numbering, with and without fracture duplication.

The fracture makes the normal trace discontinuous, which is a property of the
discrete *space*: the mesh, the complex and ``dd = 0`` are untouched and only
the numbering changes.  Two things must hold:

* with no tags the map is the identity, so every un-fractured result is
  unchanged; and
* with tags the two sides of a fracture facet are independent, so the mass
  exchanged with the fracture, ``un+ + un-``, is free rather than identically
  zero -- which is the whole reason the duplication exists.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from mimetika.assembly.mixed import discrete_divergence
from mimetika.dof import FacetDofMap
from mimetika.mesh import structured_box, structured_tets
from mimetika.mesh.fracture import facets_on_plane
from mimetika.mesh.readers import read_vtu
from mimetika.operators.diffusion import DiffusionInnerProduct


def _mesh_and_tags(nx=2, ny=2, nz=2, tets=False):
    mesh = (structured_tets if tets else structured_box)(nx, ny, nz)
    return mesh, facets_on_plane(mesh, [0, 0, 0.5], [0, 0, 1])


# -- the conforming map is the identity ---------------------------------------


@pytest.mark.parametrize("ndf", [1, 9])
def test_conforming_map_is_the_identity(ndf):
    mesh, _ = _mesh_and_tags()
    dm = FacetDofMap(mesh, ndf)
    assert dm.is_conforming()
    assert dm.n_duplicated == 0
    assert dm.n_dofs == ndf * mesh.num_cells(2)
    for c in range(mesh.num_cells(3)):
        for f, _ in mesh.complex.facets_of(3, c):
            assert np.array_equal(dm.dofs(c, f), ndf * f + np.arange(ndf))


def test_conforming_map_reproduces_the_default_divergence():
    mesh, _ = _mesh_and_tags()
    B = discrete_divergence(mesh, FacetDofMap(mesh, 1))
    assert (abs(B - discrete_divergence(mesh)) > 1e-14).nnz == 0


def test_conforming_map_reproduces_the_default_inner_product():
    mesh, _ = _mesh_and_tags()
    ip = DiffusionInnerProduct(mesh)
    assert (abs(ip.assemble(FacetDofMap(mesh, 1)) - ip.assemble()) > 1e-14).nnz == 0


# -- duplication ---------------------------------------------------------------


def test_duplication_adds_one_block_per_tagged_facet():
    mesh, tags = _mesh_and_tags()
    dm = FacetDofMap(mesh, 1, frozenset(tags))
    assert not dm.is_conforming()
    assert dm.n_duplicated == len(tags)
    assert dm.n_dofs == mesh.num_cells(2) + len(tags)


def test_the_two_sides_of_a_tagged_facet_are_distinct():
    mesh, tags = _mesh_and_tags()
    dm = FacetDofMap(mesh, 1, frozenset(tags))
    for f in tags:
        sides = dm.sides(int(f))
        assert len(sides) == 2
        (c0, b0), (c1, b1) = sides
        assert c0 != c1 and b0 != b1


def test_untagged_facets_still_share_a_block():
    mesh, tags = _mesh_and_tags()
    dm = FacetDofMap(mesh, 1, frozenset(tags))
    for f in set(range(mesh.num_cells(2))) - set(int(t) for t in tags):
        blocks = {b for _, b in dm.sides(f)}
        assert len(blocks) == 1


def test_tagging_a_boundary_facet_is_ignored():
    """A boundary facet already has one side; there is nothing to duplicate."""
    mesh = structured_box(2, 2, 2)
    outer = facets_on_plane(mesh, [0, 0, 0], [0, 0, 1], interior_only=False)
    dm = FacetDofMap(mesh, 1, frozenset(outer))
    assert dm.n_duplicated == 0
    assert dm.n_dofs == mesh.num_cells(2)


@pytest.mark.parametrize("ndf", [1, 9])
def test_dofs_per_facet_scales_the_numbering(ndf):
    mesh, tags = _mesh_and_tags()
    dm = FacetDofMap(mesh, ndf, frozenset(tags))
    assert dm.n_dofs == ndf * (mesh.num_cells(2) + len(tags))
    for f in tags:
        (c0, _), (c1, _) = dm.sides(int(f))
        assert set(dm.dofs(c0, int(f))).isdisjoint(set(dm.dofs(c1, int(f))))


def test_cell_dofs_follows_facet_order():
    mesh, tags = _mesh_and_tags()
    dm = FacetDofMap(mesh, 1, frozenset(tags))
    for c in range(mesh.num_cells(3)):
        fids = [f for f, _ in mesh.complex.facets_of(3, c)]
        expected = np.concatenate([dm.dofs(c, f) for f in fids])
        assert np.array_equal(dm.cell_dofs(c, fids), expected)
        assert np.array_equal(dm.cell_dofs(c), expected)


def test_every_dof_is_owned_by_exactly_one_or_two_cells():
    mesh, tags = _mesh_and_tags()
    dm = FacetDofMap(mesh, 1, frozenset(tags))
    owners = {}
    for c in range(mesh.num_cells(3)):
        for d in dm.cell_dofs(c):
            owners.setdefault(int(d), []).append(c)
    assert set(range(dm.n_dofs)) == set(owners)
    assert all(len(v) in (1, 2) for v in owners.values())
    # duplicated dofs are single-owner, and there are 2 per tagged facet
    singles = [d for d, v in owners.items() if len(v) == 1]
    boundary = mesh.num_cells(2) - len(list(mesh.complex.facets_of(3, 0))) * 0
    assert len(singles) == 24 + 2 * len(tags)  # 24 boundary faces of a 2x2x2 box


# -- what the duplication buys -------------------------------------------------


def test_exchange_flux_is_identically_zero_without_duplication():
    """The point of the whole exercise: sum of the two sides cancels."""
    mesh, tags = _mesh_and_tags()
    dm = FacetDofMap(mesh, 1)
    B = discrete_divergence(mesh, dm)
    for f in tags:
        col = B[:, int(f)]
        # one shared dof, opposite signs => the fracture can receive nothing
        assert np.isclose(col.sum(), 0.0)
        assert col.nnz == 2


def test_exchange_flux_is_free_with_duplication():
    mesh, tags = _mesh_and_tags()
    dm = FacetDofMap(mesh, 1, frozenset(tags))
    B = discrete_divergence(mesh, dm)
    for f in tags:
        for _, b in dm.sides(int(f)):
            col = B[:, b]
            assert col.nnz == 1  # one cell only -> no cancellation
            assert not np.isclose(col.sum(), 0.0)


def test_duplication_preserves_the_divergence_theorem():
    """A constant field still has zero discrete divergence in every cell."""
    mesh, tags = _mesh_and_tags()
    dm = FacetDofMap(mesh, 1, frozenset(tags))
    B = discrete_divergence(mesh, dm)

    field = np.array([0.7, -1.1, 0.45])
    normals, area = mesh.geometry.facet_normals(), mesh.geometry.measure(2)
    u = np.zeros(dm.n_dofs)
    for c in range(mesh.num_cells(3)):
        for f, _ in mesh.complex.facets_of(3, c):
            u[dm.dofs(c, f)] = normals[f] @ field  # canonical orientation
    assert np.allclose(B @ u, 0.0, atol=1e-12)
    assert np.isclose(area.sum(), mesh.geometry.measure(2).sum())


def test_duplicated_dofs_do_not_couple_in_the_inner_product():
    """The two sides live in different cells, so M has no entry between them."""
    mesh, tags = _mesh_and_tags()
    dm = FacetDofMap(mesh, 1, frozenset(tags))
    M = DiffusionInnerProduct(mesh).assemble(dm)
    assert M.shape == (dm.n_dofs, dm.n_dofs)
    assert np.linalg.eigvalsh(M.toarray()).min() > 0
    for f in tags:
        (_, b0), (_, b1) = dm.sides(int(f))
        assert np.isclose(M[b0, b1], 0.0)


def test_matrix_stays_connected_around_an_immersed_tip():
    """A partially tagged plane must not cut the domain in two."""
    mesh = structured_box(2, 2, 2)
    plane = facets_on_plane(mesh, [0, 0, 0.5], [0, 0, 1])

    def n_components(tags):
        dm = FacetDofMap(mesh, 1, frozenset(tags))
        shared = {}
        for c in range(mesh.num_cells(3)):
            for d in dm.cell_dofs(c):
                shared.setdefault(int(d), []).append(c)
        pairs = [v for v in shared.values() if len(v) == 2]
        n = mesh.num_cells(3)
        g = sp.coo_matrix(
            (np.ones(len(pairs)), ([p[0] for p in pairs], [p[1] for p in pairs])),
            shape=(n, n),
        )
        return sp.csgraph.connected_components(g, directed=False)[0]

    assert n_components([]) == 1
    assert n_components(plane[:2]) == 1  # immersed tip: still connected
    assert n_components(plane) == 2  # fully cut: two halves, joined only via the fracture


def test_duplication_works_on_a_tetrahedral_mesh():
    mesh, tags = _mesh_and_tags(2, 2, 2, tets=True)
    dm = FacetDofMap(mesh, 1, frozenset(tags))
    assert dm.n_duplicated == len(tags) > 0
    M = DiffusionInnerProduct(mesh).assemble(dm)
    assert np.linalg.eigvalsh(M.toarray()).min() > 0


def test_dofmap_built_from_a_vtu_with_tags(tagged_vtu):
    """End to end: read the mesh and its tags from file, then build the space."""
    from mimetika.mesh.fracture import read_fracture_tags

    path, _, tags = tagged_vtu()
    mesh = read_vtu(path)
    resolved = read_fracture_tags(path, mesh)
    dm = FacetDofMap(mesh, 1, frozenset(resolved))
    assert dm.n_duplicated == len(tags)
    assert dm.n_dofs == mesh.num_cells(2) + len(tags)
