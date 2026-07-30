r"""Mixed-dimensional VTU output and the PVD time series.

The claim under test is that a mixed-dimensional solution survives the round
trip to disk *as two parts*.  Flattening it into one grid would either drop the
fracture unknowns -- fracture pressure, fracture flux, contact traction,
displacement jump have no bulk counterpart -- or smear them onto the cells
beside the fracture, and neither is recoverable afterwards.

The files are also checked for being well-formed XML with the cell/point counts
they declare, because a ``.vtu`` that ParaView silently refuses to open is
indistinguishable from one that was never written.
"""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from mimetika.assembly.mixed_dimensional import Fracture, MixedDimensionalDarcy
from mimetika.mesh import structured_box, structured_quads
from mimetika.mesh.fracture import facets_on_plane
from mimetika.postprocess import (
    MixedDimensionalSeries,
    darcy_fields,
    export_facets,
    export_vtu,
    facet_vectors,
)

VTK_LINE, VTK_POLYGON, VTK_POLYHEDRON = 3, 7, 42


def parse(path):
    """``(root, piece)`` of a ``.vtu``, failing loudly if it is not valid XML."""
    root = ET.parse(path).getroot()
    return root, root.find("./UnstructuredGrid/Piece")


def array_of(piece, section, name):
    for a in piece.findall(f"./{section}/DataArray"):
        if a.get("Name") == name:
            return np.fromstring(a.text, sep=" ")
    raise KeyError(name)


# -- the writer is dimension-generic -------------------------------------------------


@pytest.mark.parametrize(
    "mesh,expected",
    [(structured_quads(2, 2), VTK_POLYGON), (structured_box(2, 2, 2), VTK_POLYHEDRON)],
    ids=["2d-polygons", "3d-polyhedra"],
)
def test_bulk_export_writes_the_right_cell_type(tmp_path, mesh, expected):
    """2D is not a flattened 3D case -- it is what a fracture mesh looks like."""
    n = mesh.num_cells(mesh.dim)
    out = export_vtu(tmp_path / "b.vtu", mesh, {"p": np.arange(n, dtype=float)})
    _, piece = parse(out)
    assert int(piece.get("NumberOfCells")) == n
    assert set(array_of(piece, "Cells", "types")) == {float(expected)}
    assert np.allclose(array_of(piece, "CellData", "p"), np.arange(n))


@pytest.mark.parametrize(
    "mesh,expected",
    [(structured_quads(2, 2), VTK_LINE), (structured_box(2, 2, 2), VTK_POLYGON)],
    ids=["2d-fracture-lines", "3d-fracture-polygons"],
)
def test_fracture_export_drops_a_dimension(tmp_path, mesh, expected):
    """A fracture has no mesh of its own: it is a tagged set of ``(d-1)``-facets."""
    facets = facets_on_plane(mesh, [0.5, 0, 0], [1, 0, 0])
    assert len(facets) > 0
    out = export_facets(tmp_path / "f.vtu", mesh, facets,
                        {"t": np.ones((len(facets), 3))})
    _, piece = parse(out)
    assert int(piece.get("NumberOfCells")) == len(facets)
    assert set(array_of(piece, "Cells", "types")) == {float(expected)}


def test_fracture_export_writes_only_the_vertices_it_uses(tmp_path):
    """A fracture file must not carry the whole ambient point cloud."""
    mesh = structured_box(6, 6, 6)
    facets = facets_on_plane(mesh, [0.5, 0, 0], [1, 0, 0])
    _, piece = parse(export_facets(tmp_path / "f.vtu", mesh, facets))
    assert int(piece.get("NumberOfPoints")) < 0.2 * len(mesh.geometry.points)


def test_an_empty_fracture_is_not_an_error(tmp_path):
    mesh = structured_box(2, 2, 2)
    _, piece = parse(export_facets(tmp_path / "f.vtu", mesh, []))
    assert int(piece.get("NumberOfCells")) == 0


# -- the PVD collection ---------------------------------------------------------------


@pytest.fixture
def series(tmp_path):
    mesh = structured_box(3, 3, 3)
    facets = facets_on_plane(mesh, [1 / 3, 0, 0], [1, 0, 0])
    assert len(facets) > 0
    return MixedDimensionalSeries(tmp_path / "run", mesh, facets), mesh, facets


def test_each_step_writes_both_parts(series):
    """Bulk and fracture are separate *parts* at the same timestep, not one grid."""
    s, mesh, facets = series
    for step, t in enumerate((0.0, 0.5, 2.0)):
        s.write(t, bulk={"p": np.full(mesh.num_cells(3), float(step))},
                fracture={"q": np.full(len(facets), float(step))})
    sets = ET.parse(s.collection).getroot().findall("./Collection/DataSet")
    assert len(sets) == 6
    assert [d.get("part") for d in sets] == ["0", "1"] * 3
    assert [float(d.get("timestep")) for d in sets] == [0, 0, 0.5, 0.5, 2, 2]
    assert all((s.collection.parent / d.get("file")).exists() for d in sets)


def test_the_timesteps_are_the_ones_given(series):
    """A PVD exists to carry time; writing the index instead would be silent."""
    s, mesh, facets = series
    for t in (0.0, 1.5, 17.25):
        s.write(t, bulk={"p": np.zeros(mesh.num_cells(3))})
    sets = ET.parse(s.collection).getroot().findall("./Collection/DataSet")
    assert [float(d.get("timestep")) for d in sets] == [0.0, 1.5, 17.25]


def test_the_collection_is_readable_after_every_step(series):
    """Rewritten each step, so an interrupted run still leaves a loadable file."""
    s, mesh, facets = series
    for step in range(3):
        s.write(float(step), bulk={"p": np.zeros(mesh.num_cells(3))})
        found = ET.parse(s.collection).getroot().findall("./Collection/DataSet")
        assert len(found) == step + 1


def test_a_part_may_be_omitted(series):
    """Elasticity writes only the fracture; there is no bulk field to plot."""
    s, mesh, facets = series
    s.write(0.0, fracture={"t": np.zeros(len(facets))})
    sets = ET.parse(s.collection).getroot().findall("./Collection/DataSet")
    assert len(sets) == 1 and "fracture" in sets[0].get("file")


# -- facet-frame to ambient ------------------------------------------------------------


def test_facet_vectors_reproduce_the_frame(tmp_path):
    """A unit normal in the facet frame must come back as the ambient normal."""
    mesh = structured_box(2, 2, 2)
    facets = facets_on_plane(mesh, [0.5, 0, 0], [1, 0, 0])
    unit_normal = np.zeros((len(facets), 3))
    unit_normal[:, 0] = 1.0
    got = facet_vectors(mesh, facets, unit_normal)
    for i, f in enumerate(facets):
        assert np.allclose(got[i], mesh.geometry.facet_frame(int(f))[0])


def test_facet_vectors_preserve_length(tmp_path):
    """The frame is orthonormal, so the change of basis is an isometry."""
    mesh = structured_box(2, 2, 2)
    facets = facets_on_plane(mesh, [0.5, 0, 0], [1, 0, 0])
    rng = np.random.default_rng(0)
    values = rng.standard_normal((len(facets), 3))
    got = facet_vectors(mesh, facets, values)
    assert np.allclose(np.linalg.norm(got, axis=1), np.linalg.norm(values, axis=1))


# -- the Darcy fields --------------------------------------------------------------------


@pytest.fixture(scope="module")
def darcy():
    mesh = structured_box(4, 4, 4)
    tags = facets_on_plane(mesh, [0.5, 0, 0], [1, 0, 0])
    problem = MixedDimensionalDarcy(mesh, Fracture(facets=tags, aperture=1e-2))
    solution = problem.solve(
        dirichlet=lambda x: 1.0 - np.atleast_2d(x)[:, 0], method="direct"
    )
    return problem, solution, mesh, tags


def test_darcy_gives_pressure_and_flux_on_both_dimensions(darcy):
    problem, solution, mesh, tags = darcy
    bulk, fracture = darcy_fields(problem, solution)
    assert bulk["pressure"].shape == (mesh.num_cells(3),)
    assert bulk["velocity"].shape == (mesh.num_cells(3), 3)
    assert fracture["pressure"].shape == (len(tags),)
    assert fracture["velocity"].shape == (len(tags), 3)


def test_the_fracture_velocity_is_tangential(darcy):
    """The whole point of the lower-dimensional unknowns: flow *along* the fracture.

    It is tangential by construction -- reconstructed on the fracture's own mesh
    from its own flux DOFs -- so any normal component would mean the wrong mesh
    or the wrong unknowns were used.
    """
    problem, solution, mesh, tags = darcy
    _, fracture = darcy_fields(problem, solution)
    normal = mesh.geometry.facet_frame(int(tags[0]))[0]
    assert np.abs(fracture["velocity"] @ normal).max() < 1e-12


def test_the_bulk_velocity_uses_the_right_side_of_a_duplicated_facet(darcy):
    """With duplicated fracture DOFs, indexing flux by facet id reads the wrong one.

    Guard: a pressure drop across ``x`` must drive the bulk velocity in ``+x``
    everywhere, which fails if a cell is handed its neighbour's flux.
    """
    problem, solution, mesh, tags = darcy
    bulk, _ = darcy_fields(problem, solution)
    assert np.all(bulk["velocity"][:, 0] > 0)


def test_darcy_fields_survive_the_round_trip(darcy, tmp_path):
    problem, solution, mesh, tags = darcy
    s = MixedDimensionalSeries(tmp_path / "d", mesh, tags)
    bulk, fracture = darcy_fields(problem, solution)
    s.write(0.0, bulk=bulk, fracture=fracture)
    _, piece = parse(tmp_path / "d_fracture_0000.vtu")
    assert np.allclose(array_of(piece, "CellData", "pressure"), fracture["pressure"])
    assert int(piece.get("NumberOfCells")) == len(tags)


# -- the dim tag ------------------------------------------------------------------------


def test_every_cell_is_tagged_with_its_dimension(tmp_path):
    """``dim`` survives merging the parts, which the block structure does not.

    In ParaView the two parts arrive as separate blocks, but the moment they are
    merged that distinction is gone -- a ``Threshold`` on ``dim`` is then the only
    way to isolate the bulk or the fracture, and it works without the user having
    to know which block was which.
    """
    mesh = structured_box(3, 3, 3)
    facets = facets_on_plane(mesh, [1 / 3, 0, 0], [1, 0, 0])
    s = MixedDimensionalSeries(tmp_path / "run", mesh, facets)
    s.write(0.0, bulk={"p": np.zeros(mesh.num_cells(3))},
            fracture={"t": np.zeros(len(facets))})

    _, bulk = parse(tmp_path / "run_bulk_0000.vtu")
    _, frac = parse(tmp_path / "run_fracture_0000.vtu")
    assert set(array_of(bulk, "CellData", "dim")) == {3.0}
    assert set(array_of(frac, "CellData", "dim")) == {2.0}
    assert len(array_of(bulk, "CellData", "dim")) == mesh.num_cells(3)
    assert len(array_of(frac, "CellData", "dim")) == len(facets)


def test_the_dim_tag_follows_the_mesh_dimension(tmp_path):
    """A 2D problem tags rock as 2 and the fault as 1, not 3 and 2."""
    mesh = structured_quads(4, 4)
    facets = facets_on_plane(mesh, [0.5, 0, 0], [1, 0, 0])
    assert len(facets) > 0
    s = MixedDimensionalSeries(tmp_path / "run", mesh, facets)
    s.write(0.0, bulk={"p": np.zeros(mesh.num_cells(2))},
            fracture={"t": np.zeros(len(facets))})
    _, bulk = parse(tmp_path / "run_bulk_0000.vtu")
    _, frac = parse(tmp_path / "run_fracture_0000.vtu")
    assert set(array_of(bulk, "CellData", "dim")) == {2.0}
    assert set(array_of(frac, "CellData", "dim")) == {1.0}


def test_a_caller_supplied_dim_is_not_overwritten(tmp_path):
    """Defaulted, not forced -- a caller with its own convention keeps it."""
    mesh = structured_box(2, 2, 2)
    s = MixedDimensionalSeries(tmp_path / "run", mesh)
    s.write(0.0, bulk={"dim": np.full(mesh.num_cells(3), 7.0)})
    _, piece = parse(tmp_path / "run_bulk_0000.vtu")
    assert set(array_of(piece, "CellData", "dim")) == {7.0}
