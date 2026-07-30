"""Mixed-dimensional Darcy: a 3D matrix coupled to a 2D fracture.

The reference case is 1D flow across a single planar fracture, where the total
resistance is a **series sum** in closed form,

    R = 1/k_m  +  eps/k_n            (matrix, then both half-apertures)

so the scheme can be checked against an exact number rather than against
itself.  The lateral walls must be **no-flow**: prescribing ``p = 1 - x`` all
round is data consistent with the *unfractured* solution, and once the fracture
raises the upstream pressure the sides start draining it.
"""

import numpy as np
import pytest

from mimetika.assembly.mixed import boundary_facets
from mimetika.assembly.mixed_dimensional import Fracture, MixedDimensionalDarcy
from mimetika.mesh import structured_box, structured_tets
from mimetika.mesh.fracture import facets_on_plane, read_fracture_tags
from mimetika.mesh.readers import read_vtu

KM = 1.0
SHAPES = [(2, 1, 1), (2, 2, 2), (4, 4, 4)]


def linear_pressure(x):
    return 1.0 - np.atleast_2d(x)[:, 0]


def lateral_walls(mesh):
    """Boundary facets other than the inlet (x=0) and outlet (x=1)."""
    fc = mesh.geometry.centroids(2)
    return [
        f
        for f in boundary_facets(mesh)
        if abs(fc[f][0]) > 1e-12 and abs(fc[f][0] - 1.0) > 1e-12
    ]


def build(shape=(2, 2, 2), kn=0.01, eps=0.1, kt=1.0, tets=False):
    mesh = (structured_tets if tets else structured_box)(*shape)
    tags = facets_on_plane(mesh, [0.5, 0, 0], [1, 0, 0])
    fr = Fracture(
        tags, aperture=eps, normal_permeability=kn, tangential_permeability=kt
    )
    return mesh, tags, MixedDimensionalDarcy(mesh, fr, K=KM * np.eye(3))


def solve_1d(problem, mesh):
    return problem.solve(
        dirichlet=linear_pressure, no_flow=lateral_walls(mesh), method="direct"
    )


# -- structure -----------------------------------------------------------------


def test_fracture_parameters():
    fr = Fracture([0], aperture=0.1, normal_permeability=0.01, tangential_permeability=2.0)
    assert np.isclose(fr.kappa, 2 * 0.01 / 0.1)  # 2 k_n / eps
    assert np.isclose(fr.transmissivity, 0.1 * 2.0)  # eps k_t


def test_block_sizes_and_duplication():
    mesh, tags, pb = build((2, 2, 2))
    assert pb.n_flux3 == mesh.num_cells(2) + len(tags)  # one extra DOF per side
    assert pb.dofmap.n_duplicated == len(tags)
    assert pb.n_p2 == len(tags) == pb.frac_mesh.num_cells(2)
    assert pb.n_flux2 == pb.frac_mesh.num_cells(1)


def test_system_is_symmetric():
    mesh, _, pb = build((2, 2, 2))
    A, _, _ = pb.assemble(dirichlet=linear_pressure, no_flow=lateral_walls(mesh))
    assert (abs(A - A.T) > 1e-12).nnz == 0


def test_interface_stiffness_is_area_over_kappa():
    mesh, tags, pb = build((2, 2, 2))
    S = pb.interface_stiffness().diagonal()
    area = mesh.geometry.measure(2)
    assert np.count_nonzero(S) == 2 * len(tags)  # both sides of every facet
    for f in tags:
        for cell, _ in pb.dofmap.sides(int(f)):
            dof = pb.dofmap.dofs(cell, int(f))[0]
            assert np.isclose(S[dof], area[f] / pb.fracture.kappa)


def test_coupling_is_the_adjoint_of_the_pressure_pairing():
    """``C`` appears as ``C^T`` in the flux row and ``C`` in the mass balance."""
    mesh, tags, pb = build((2, 2, 2))
    C = pb.coupling()
    assert C.shape == (pb.n_p2, pb.n_flux3)
    area = mesh.geometry.measure(2)
    for fc, f in enumerate(pb.facet_of_cell):
        row = C[fc].toarray().ravel()
        assert np.count_nonzero(row) == 2  # two independent sides
        assert np.isclose(np.abs(row).sum(), 2 * area[f])
        assert np.isclose(row.sum(), 0.0)  # equal and opposite signs


# -- the analytic series resistance --------------------------------------------


@pytest.mark.parametrize("shape", SHAPES, ids=[str(s) for s in SHAPES])
@pytest.mark.parametrize("kn", [1e8, 1.0, 1e-2, 1e-6])
def test_series_resistance_is_exact(shape, kn):
    """R = 1/k_m + eps/k_n, to round-off, independent of the mesh."""
    eps = 0.1
    mesh, tags, pb = build(shape, kn=kn, eps=eps)
    sol = solve_1d(pb, mesh)
    q, _ = pb.side_fluxes(sol, int(tags[0]))
    assert np.isclose(1.0 / abs(q), 1.0 / KM + eps / kn, rtol=1e-9)


@pytest.mark.parametrize("eps", [1.0, 0.1, 1e-3])
def test_resistance_scales_with_aperture(eps):
    mesh, tags, pb = build((2, 2, 2), kn=0.01, eps=eps)
    sol = solve_1d(pb, mesh)
    q, _ = pb.side_fluxes(sol, int(tags[0]))
    assert np.isclose(1.0 / abs(q), 1.0 / KM + eps / 0.01, rtol=1e-9)


def test_flux_is_equal_and_opposite_across_the_fracture():
    mesh, tags, pb = build((2, 2, 2))
    sol = solve_1d(pb, mesh)
    for f in tags:
        a, b = pb.side_fluxes(sol, int(f))
        assert np.isclose(a, -b, atol=1e-12)


def test_no_exchange_when_there_is_no_tangential_gradient():
    """Pure cross-flow: what enters one side leaves the other, so Q = 0."""
    mesh, tags, pb = build((2, 2, 2))
    sol = solve_1d(pb, mesh)
    assert np.allclose(pb.exchange(sol), 0.0, atol=1e-12)
    assert np.allclose(sol["fracture_flux"], 0.0, atol=1e-12)


def test_fracture_pressure_sits_midway_by_symmetry():
    mesh, tags, pb = build((2, 2, 2))
    sol = solve_1d(pb, mesh)
    assert np.allclose(sol["fracture_pressure"], 0.5, atol=1e-10)


# -- limits ---------------------------------------------------------------------


def test_conductive_limit_recovers_the_unfractured_solution():
    """kappa -> infinity: the fracture disappears and R -> 1/k_m."""
    from mimetika.assembly.mixed import MixedPoisson

    mesh, tags, pb = build((2, 2, 2), kn=1e10, eps=1e-6)
    sol = solve_1d(pb, mesh)
    q, _ = pb.side_fluxes(sol, int(tags[0]))
    assert np.isclose(abs(q), KM, rtol=1e-6)

    plain = MixedPoisson(mesh, K=KM * np.eye(3)).solve(
        dirichlet=linear_pressure, method="direct"
    )
    assert np.allclose(sol["pressure"], plain["pressure"], atol=1e-6)


def test_blocking_limit_stops_the_flow_and_opens_a_jump():
    """kappa -> 0: no flux across, and the pressure jump takes the whole drop."""
    mesh, tags, pb = build((2, 2, 2), kn=1e-10, eps=0.1)
    sol = solve_1d(pb, mesh)
    q, _ = pb.side_fluxes(sol, int(tags[0]))
    assert abs(q) < 1e-8

    p = sol["pressure"]
    centroids = mesh.geometry.centroids(3)
    upstream = p[centroids[:, 0] < 0.5].mean()
    downstream = p[centroids[:, 0] > 0.5].mean()
    assert upstream > 0.9 and downstream < 0.1  # essentially the full drop
    assert np.isclose(upstream - downstream, 1.0, atol=1e-6)


# -- conservation ----------------------------------------------------------------


def test_mass_is_conserved_in_every_matrix_cell():
    from mimetika.assembly.mixed import discrete_divergence

    mesh, _, pb = build((2, 2, 2))
    sol = solve_1d(pb, mesh)
    B = discrete_divergence(mesh, pb.dofmap)
    assert np.allclose(B @ sol["flux"], 0.0, atol=1e-10)


def test_global_balance_inlet_equals_outlet():
    mesh, _, pb = build((2, 2, 2))
    sol = solve_1d(pb, mesh)
    fc = mesh.geometry.centroids(2)
    area = mesh.geometry.measure(2)
    inlet = [f for f in boundary_facets(mesh) if abs(fc[f][0]) < 1e-12]
    outlet = [f for f in boundary_facets(mesh) if abs(fc[f][0] - 1.0) < 1e-12]
    qin = sum(area[f] * abs(sol["flux"][f]) for f in inlet)
    qout = sum(area[f] * abs(sol["flux"][f]) for f in outlet)
    assert np.isclose(qin, qout, rtol=1e-10)


def test_exchange_matches_the_fracture_divergence():
    """The fracture mass balance ``B2 u2 = Q`` holds after the solve."""
    from mimetika.assembly.mixed import discrete_divergence

    mesh, _, pb = build((2, 2, 2), kt=1e3)  # conductive: exchange is active
    sol = pb.solve(
        dirichlet=linear_pressure, no_flow=lateral_walls(mesh), method="direct"
    )
    B2 = discrete_divergence(pb.frac_mesh)
    assert np.allclose(B2 @ sol["fracture_flux"], pb.exchange(sol), atol=1e-10)


# -- other meshes and configurations ----------------------------------------------


def test_works_on_a_tetrahedral_mesh():
    mesh, tags, pb = build((2, 2, 2), kn=0.01, eps=0.1, tets=True)
    assert len(tags) > 0
    sol = solve_1d(pb, mesh)
    q, _ = pb.side_fluxes(sol, int(tags[0]))
    assert np.isclose(1.0 / abs(q), 1.0 / KM + 0.1 / 0.01, rtol=1e-8)


def test_immersed_tip_is_constrained_to_no_flow():
    """A partially tagged plane leaves a tip, which must carry zero flux."""
    mesh = structured_box(4, 4, 4)
    plane = facets_on_plane(mesh, [0.5, 0, 0], [1, 0, 0])
    fr = Fracture(plane[:4], aperture=0.1, normal_permeability=0.01)
    pb = MixedDimensionalDarcy(mesh, fr, K=KM * np.eye(3))

    dir_facets = pb.dirichlet_facets(lateral_walls(mesh))
    tips = pb.fracture_no_flow(dir_facets)
    assert len(tips) > 0
    sol = solve_1d(pb, mesh)
    assert np.allclose(sol["fracture_flux"][tips], 0.0, atol=1e-12)


def test_driven_from_a_tagged_vtu_file(tagged_vtu):
    """End to end: mesh and fracture tags both read from a .vtu."""
    path, _, tags = tagged_vtu(
        shape=(2, 2, 2), point=(0.5, 0, 0), normal=(1, 0, 0), name="cross.vtu"
    )
    mesh = read_vtu(path)
    resolved = read_fracture_tags(path, mesh)
    fr = Fracture(resolved, aperture=0.1, normal_permeability=0.01)
    pb = MixedDimensionalDarcy(mesh, fr, K=KM * np.eye(3))
    sol = solve_1d(pb, mesh)
    q, _ = pb.side_fluxes(sol, int(resolved[0]))
    assert np.isclose(1.0 / abs(q), 1.0 / KM + 0.1 / 0.01, rtol=1e-9)


def test_ambient_mesh_is_never_modified():
    """The duplication lives in the space; the geometry must be untouched."""
    mesh, tags, pb = build((2, 2, 2))
    counts = [mesh.num_cells(k) for k in range(4)]
    volume = mesh.geometry.measure(3).sum()
    solve_1d(pb, mesh)
    assert [mesh.num_cells(k) for k in range(4)] == counts
    assert np.isclose(mesh.geometry.measure(3).sum(), volume)
    assert mesh.complex.verify_complex()
