"""THE SAME PROBLEM IN EVERY DIMENSION AND EVERY CELL FAMILY, through Python.

The Python-side mirror of tests/model/test_confined_compression.cpp: the same meshes, the
same closed form, the same tolerances. Confined uniaxial compression has no
freedom left to get wrong -- rollers on the base and sides, a uniform
compressive traction on top, zero lateral strain forced by geometry alone -- so
elasticity gives the whole answer in closed form, in ANY dimension:

    sigma_nn  = lam/(lam + 2 mu) sigma_axial    on the confined facets
    eps_axial = sigma_axial / K_oed,            K_oed = lam + 2 mu

Running it on quadrilaterals, triangles, hexahedra and tetrahedra is what
separates "the binding has a 2D branch" from "the 2D discretization is the same
method".
"""

import numpy as np
import pytest

import mimetika_cxx as mk
from _meshes import cube, square

MU, LAM, LOAD = 1.0, 1.0, 1.0

DERHAM = mk.StressRealization.derham_bdm
DERHAM_RT = mk.StressRealization.derham_rt
STABILIZED = mk.StressRealization.stabilized_bdm


class Result:
    """What one confined-compression solve is worth measuring."""

    def __init__(self):
        self.sigma_lateral = 0.0  # worst error against lam/(lam+2mu) sigma_axial
        self.displacement = 0.0  # worst error against eps_axial * x_axial
        self.n_stabilized = 0
        self.n_dofs = 0
        self.solvable = True
        self.reason = ""


def confined(mesh, d, how=DERHAM) -> Result:
    """Solve confined compression on `mesh` and measure both closed forms."""
    prob = mk.CauchyMechanicsProblem(mesh, d, MU, LAM, how)
    axis = d - 1  # the column axis is the last coordinate

    loaded, confined_facets = [], []
    for f in prob.boundary_facets():
        x = prob.centroid(d - 1, f)
        (loaded if abs(x[axis] - 1.0) < 1e-9 else confined_facets).append(f)

    applied = [0.0] * 9
    applied[axis * 3 + axis] = -LOAD
    prob.impose_traction("s_0", loaded, lambda _x: applied)
    prob.impose_free_slip("s_0", confined_facets)
    prob.freeze_constraints()

    converged, reason, x = prob.solve()

    k_oed = LAM + 2.0 * MU
    s_exact = LAM / k_oed * (-LOAD)
    e_exact = -LOAD / k_oed

    out = Result()
    out.n_stabilized = prob.n_stabilized
    out.n_dofs = prob.n_dofs
    out.solvable = converged
    out.reason = reason
    # a singular saddle point has no solution to measure; the caller asserts on
    # `solvable` and the error fields stay at zero
    if not converged:
        return out

    # the normal traction on every confined facet, read through the FORM that
    # imposed nothing there: n . (sigma n), divided by the measure it was
    # integrated against
    for f in confined_facets:
        fr = prob.facet_frame(f)
        # only the LATERAL facets carry lam/(lam+2mu) sigma_axial. The base is
        # confined too, but the axial load transmits straight through it, so its
        # normal traction is sigma_axial itself -- a different closed form, and
        # checking it against the lateral one would fail by exactly 2/3 here.
        if abs(fr["normal"][axis]) > 1e-9:
            continue
        t = sum(fr["normal"][k] * x[prob.dof("s_0", d - 1, f, 0, k)] for k in range(d))
        out.sigma_lateral = max(out.sigma_lateral, abs(t / fr["measure"] - s_exact))

    # u is stored as the cell integral, with the opposite sign of the convention
    # the closed form is written in
    for e in range(prob.count(d)):
        vol = prob.measure(d, e)
        z = prob.centroid(d, e)[axis]
        uz = -x[prob.dof("u_0", d, e, 0, axis)] / vol
        out.displacement = max(out.displacement, abs(uz - e_exact * z))
    return out


# name, mesh factory, dim, cells, stabilized cells under the STABILIZED star
CASES = [
    ("2D quadrilateral", lambda: square(3, False), 2, 9, 9),
    ("2D triangle", lambda: square(3, True), 2, 18, 0),
    ("3D hexahedron", lambda: cube(2, False), 3, 8, 8),
    ("3D tetrahedron", lambda: cube(2, True), 3, 48, 0),
]
IDS = [c[0] for c in CASES]


# THE 2 x 2 MATRIX: two dimensions, two cell families, one closed form.
@pytest.mark.parametrize("name,mesh,dim,cells,stab", CASES, ids=IDS)
def test_confined_compression_is_exact_in_every_dimension_and_family(
    name, mesh, dim, cells, stab
):
    r = confined(mesh(), dim)
    print(
        f"  {name:<18} {r.n_dofs:6d} dofs   sigma_lat {r.sigma_lateral:.2e}"
        f"   u {r.displacement:.2e}   stabilized {r.n_stabilized}"
    )
    # the de Rham realization is unisolvent by construction: nothing stabilizes
    assert r.n_stabilized == 0
    # and the closed form is reproduced, not approximated
    assert r.sigma_lateral < 1e-11
    assert r.displacement < 1e-11


def test_the_rt_layer_is_locally_sound_but_globally_unstable():
    """The RT star is the smaller space and a locally surjective one -- and is
    still not an element, because the global saddle point is singular."""
    mesh_bdm, mesh_rt = cube(2, True), cube(2, True)

    # LOCAL soundness: the stacked constraint block [Dv; As] has full row rank
    # on both stars, so neither is locally deficient
    for how, label in ((DERHAM, "derham_bdm"), (DERHAM_RT, "derham_rt")):
        prob = mk.CauchyMechanicsProblem(cube(2, True), 3, MU, LAM, how)
        b = prob.constraint_block(0)
        scale = np.max(np.abs(b))
        rank = np.linalg.matrix_rank(b, tol=1e-10 * (scale if scale > 0 else 1.0))
        print(f"  {label} local [Dv; As] {b.shape[0]}x{b.shape[1]}   "
              f"rank {rank} of {b.shape[0]} needed")
        assert rank == b.shape[0]

    bdm = confined(mesh_bdm, 3, DERHAM)
    rt = confined(mesh_rt, 3, DERHAM_RT)
    print(f"  derham      {bdm.n_dofs:6d} dofs   solved                 "
          f"sigma_lat {bdm.sigma_lateral:.2e}   u {bdm.displacement:.2e}")
    print(f"  derham_rt   {rt.n_dofs:6d} dofs   {rt.reason:<22} (no solution to measure)")

    assert bdm.solvable
    assert bdm.sigma_lateral < 1e-11
    assert bdm.displacement < 1e-11
    assert rt.n_dofs < bdm.n_dofs  # genuinely the smaller space
    assert not rt.solvable  # and genuinely not an element


@pytest.mark.parametrize("name,mesh,dim,cells,stab", CASES, ids=IDS)
def test_the_stabilized_bdm_is_exact_in_every_dimension_and_family(
    name, mesh, dim, cells, stab
):
    r = confined(mesh(), dim, STABILIZED)
    print(
        f"  {name:<18} {r.n_dofs:6d} dofs   sigma_lat {r.sigma_lateral:.2e}"
        f"   u {r.displacement:.2e}   stabilized {r.n_stabilized} of {cells} cells"
    )
    assert r.solvable
    # a simplex mesh never stabilizes; a polytopal one stabilizes every cell
    assert r.n_stabilized == stab
    assert r.sigma_lateral < 1e-11
    assert r.displacement < 1e-11
