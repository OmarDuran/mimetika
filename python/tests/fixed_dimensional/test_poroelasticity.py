"""Poroelasticity's structure through the Python interface.

The Python-side mirror of tests/model/test_poroelasticity.cpp. Nothing here
solves anything: what is pinned is the shape of the system -- which fields
exist, how many degrees of freedom each carries, and the sign convention every
coupling block was written in.
"""

import numpy as np
import pytest

import mimetika_cxx as mk
from _meshes import cube

MU, LAM = 1.0, 1.0


def hex_grid(n):
    """The fixture the C++ test meshes on: n x n x n hexahedra."""
    return cube(n, False)


def dense_jacobian(model):
    """The assembled Jacobian, densely enough to test its structure."""
    n = model.n_dofs
    rows, cols, vals = model.jacobian_coo()
    a = np.zeros((n, n))
    np.add.at(a, (rows, cols), vals)
    return a


def test_poroelasticity_adds_no_field_of_its_own():
    m = hex_grid(2)

    mech = mk.build_composition("linear_elasticity")
    flow = mk.build_composition("single_phase_flow")
    poro = mk.build_composition("poroelasticity")
    assert mech.size == 1 and flow.size == 1 and poro.size == 3

    sm, sf, sp = mech.space(m, 3), flow.space(m, 3), poro.space(m, 3)
    assert sm.n_fields == 3  # s, u, g
    assert sf.n_fields == 2  # q, p
    assert sp.n_fields == 5  # and nothing more
    assert sp.size == sm.size + sf.size
    for f in ("s_0", "u_0", "g_0", "q_0", "p_0"):
        assert sp.has(f)

    # the coupling needs both physics, and says so when one is absent
    bad = mk.Composition()
    bad.add_flow()
    bad.add_poro_coupling()
    with pytest.raises(ValueError, match="'displacement'"):
        bad.validate(3)


# THE STRESS DEGREES OF FREEDOM ARE d^2 PER FACET, and the rotation carries the
# d(d-1)/2 components weak symmetry needs. Getting either count wrong changes
# the method rather than breaking it, so it is pinned.
def test_the_mixed_elasticity_space_has_the_afw_counts():
    m = hex_grid(2)
    s = mk.build_composition("linear_elasticity").space(m, 3)

    n_facets = m.count(2)
    n_cells = m.count(3)
    assert s.field_size("s_0") == 9 * n_facets
    assert s.field_size("u_0") == 3 * n_cells
    assert s.field_size("g_0") == 3 * n_cells
    assert s.field_degree("s_0") == 2  # facet cochain
    assert s.field_degree("u_0") == 3  # cell-wise


# THE SYSTEM IS A SADDLE POINT WITH ADJOINT COUPLINGS, in the convention the
# flux term set: [M, -B^T; +B, 0]. So a constitutive block is symmetric and
# every off-diagonal pair is the NEGATIVE transpose of its partner.
def test_the_poroelastic_system_is_a_saddle_point_with_adjoint_couplings():
    m = hex_grid(2)

    # WHETHER A CELL STABILIZES IS A PROPERTY OF THE SPACE, not of the cell.
    # The default de Rham realization reconstructs each stress row on the
    # enriched scalar space, so its N is square and unisolvent and there is
    # nothing left for a stabilization to see -- on hexahedra as on simplices.
    # The AFW realization reconstructs on the full linear tensor space, whose
    # moments a hexahedron does not determine, so there every cell stabilizes.
    size, stabilized = mk.stress_operator_counts(m, 3, MU, LAM, mk.StressRealization.derham_afw)
    assert size == m.count(3)
    assert stabilized == 0
    afw_size, afw_stab = mk.stress_operator_counts(
        m, 3, MU, LAM, mk.StressRealization.stabilized_afw
    )
    assert afw_stab == afw_size

    model = mk.AssembledModel(m, 3, "poroelasticity", MU, LAM)
    model.seed_state()
    model.freeze_constraints()

    rows, _, _ = model.jacobian_coo()
    assert len(rows) > 0

    a = dense_jacobian(model)
    s0, s1 = model.field_range("s_0")
    u0, u1 = model.field_range("u_0")
    g0, g1 = model.field_range("g_0")
    p0, p1 = model.field_range("p_0")
    q0, q1 = model.field_range("q_0")

    # ADJOINTNESS everywhere: |A_ij| = |A_ji|, because every coupling was
    # written from one array of coefficients
    assert np.max(np.abs(np.abs(a) - np.abs(a.T))) < 1e-10

    # AND THE SIGNS, which adjointness-in-magnitude cannot see. The two
    # couplings of a poroelastic system are not the same kind of object:
    #
    #   (s,p)  the BIOT coupling, a constitutive symmetry. Both blocks are
    #          second derivatives of one free energy, so A_ij = +A_ji.
    #   (q,p)  the DARCY pair, adjoint differential operators, which exokal
    #          writes antisymmetrically as [M, -B^T; +B, 0]: A_ij = -A_ji.
    def pairing(a0, a1, b0, b1, want):
        blk, tr = a[a0:a1, b0:b1], a[b0:b1, a0:a1].T
        return np.max(np.abs(blk - want * tr)), np.max(np.abs(blk))

    biot_err, biot_mag = pairing(s0, s1, p0, p1, +1.0)
    assert biot_mag > 1e-12  # the block must exist before its sign means anything
    assert biot_err < 1e-10
    darcy_err, darcy_mag = pairing(q0, q1, p0, p1, -1.0)
    assert darcy_mag > 1e-12
    assert darcy_err < 1e-10

    # and the multiplier blocks are empty: no (u,u), no (g,g), no (u,g)
    def empty(a0, a1, b0, b1):
        return np.max(np.abs(a[a0:a1, b0:b1]))

    assert empty(u0, u1, u0, u1) == 0.0
    assert empty(g0, g1, g0, g1) == 0.0
    assert empty(u0, u1, g0, g1) == 0.0
    assert empty(u0, u1, p0, p1) == 0.0  # the Biot coupling goes through the STRESS

    # the Biot coupling is present, and it is the PLAIN transpose
    assert empty(s0, s1, p0, p1) > 1e-9
    assert empty(p0, p1, s0, s1) > 1e-9

    # and so are the mechanics couplings, in the same convention
    assert np.max(np.abs(a[s0:s1, u0:u1] + a[u0:u1, s0:s1].T)) < 1e-12

    # the constitutive block itself IS symmetric -- it is an inner product
    assert np.max(np.abs(a[s0:s1, s0:s1] - a[s0:s1, s0:s1].T)) < 1e-10
