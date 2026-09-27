"""The inf-sup constants of the mixed contact formulation, guarded.

A cheap version of python/examples/contact_inf_sup.py, which is where the study
and the reasoning live -- imported rather than reimplemented, so the numbers a
test asserts and the numbers a run reports come from one place.

WHAT IS WORTH A TEST HERE is not the value of beta_h, which is a property of the
formulation and of the norms, but the two things a future change to the coupling
could silently cost:

  * dim ker B^T = 0. B surjective is the structural half of inf-sup. Lose it and
    the saddle point has spurious multiplier modes -- most plausibly at a
    fracture intersection, where this construction carries no unknowns -- and
    every solve still "works" while answering a different problem.
  * beta_h and C_h not collapsing between two levels. A short ladder cannot
    measure a rate, but it catches a constant that has started to decay.

The norm pair is s_t = 1/2, s_u = 0: t_c in the h^(1/2)-weighted surrogate for
H^-1/2(Gamma), u_c in plain L^2(Gamma). That is the pair both Brezzi constants
are uniform in; in L^2 x L^2, C_h grows like h^(-1/2), which is the trace
inequality's scaling rather than an instability.
"""

import sys
from pathlib import Path

import numpy as np

import pytest

import mimetika_cxx as mk

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
study = pytest.importorskip("contact_inf_sup")

PAIR = (0.5, 0.0)


def constants(dim, n, count, c_n=0.0):
    study.args_lame = 1.0  # nu = 1/4, as the study's default
    mesh, sets = study.case(dim, n, count)
    model, K, blocks, pinned = study.assemble(mesh, dim, sets, "derham_bdm", c_n, 0.0)
    assert model.n_fracture_faces > 0
    ix, ixw = study.indices(blocks, study.X_FIELDS, pinned)
    iy, iyw = study.indices(blocks, study.Y_FIELDS, pinned)

    # the grouping, checked: x carries the mass terms, y is the multipliers, so
    # the (y, y) block has to be exactly zero or every number below is about a
    # different problem
    yy = K[np.ix_(iy, iy)]
    assert not yy.nnz or np.abs(yy.data).max() < 1e-12 * np.abs(K.data).max()

    grams = study.face_grams(K, blocks, model.n_fracture_faces)
    h = 1.0 / n
    M_X, M_Y, _, B = study.norms(mesh, dim, K, blocks, pinned, ix, iy, ixw, iyw,
                                 grams, PAIR, h)
    beta, kernel, _ = study.inf_sup(M_X, M_Y, B)
    c_h = study.kernel_coercivity(M_X, B, ixw, iyw, grams, (PAIR[0], PAIR[0]), h)
    return beta, kernel, c_h


@pytest.mark.parametrize("count", [1, 2])
def test_the_constraint_operator_stays_surjective(count):
    """dim ker B^T = 0 at both levels: no spurious multiplier modes, and none
    appearing where two fractures cross."""
    for n in (4, 8):
        _, kernel, _ = constants(2, n, count)
        assert kernel == 0, f"{kernel} spurious mode(s) with {count} fracture(s) at n = {n}"


@pytest.mark.parametrize("count", [1, 2])
def test_both_brezzi_constants_hold_up_under_one_refinement(count):
    """Halving h must not halve the constants. Two levels cannot give a rate --
    the example's four-level ladder does that -- but a collapse shows here."""
    coarse = constants(2, 4, count)
    fine = constants(2, 8, count)
    beta_c, _, ch_c = coarse
    beta_f, _, ch_f = fine
    assert beta_f > 0.5, beta_f
    assert beta_f > 0.8 * beta_c, (beta_c, beta_f)  # flat, in fact rising
    assert ch_f < 1.25 * ch_c, (ch_c, ch_f)  # flat: C_h is ~2 and ~2.5 here


def test_the_constants_do_not_depend_on_the_contact_law():
    """B = [B 0; C 0; T J^T] carries no A_c, so beta_h and C_h are properties of
    the constraint and the norms alone. One consequence matters: the BONDED case
    is the extreme one for coercivity on the kernel, since A_c >= 0 only helps,
    so a study run bonded covers every law."""
    bonded = constants(2, 4, 2, c_n=0.0)
    stiff = constants(2, 4, 2, c_n=1e2)
    assert np.allclose(bonded[0], stiff[0], rtol=1e-12)
    assert np.allclose(bonded[2], stiff[2], rtol=1e-12)
    assert bonded[1] == stiff[1] == 0


def test_the_l2_pair_is_the_wrong_one_for_the_coercivity_bound():
    """Measured in L^2(Gamma), C_h grows like h^(-1/2) -- the trace inequality's
    scaling, not an instability. This pins WHY the weighted norm is the reported
    one, so a future simplification back to L^2 fails here rather than quietly
    reporting a degrading constant."""
    study.args_lame = 1.0
    ch = {}
    for n in (4, 16):
        mesh, sets = study.case(2, n, 1)
        model, K, blocks, pinned = study.assemble(mesh, 2, sets, "derham_bdm", 0.0, 0.0)
        ix, ixw = study.indices(blocks, study.X_FIELDS, pinned)
        iy, iyw = study.indices(blocks, study.Y_FIELDS, pinned)
        grams = study.face_grams(K, blocks, model.n_fracture_faces)
        h = 1.0 / n
        M_X, _, _, B = study.norms(mesh, 2, K, blocks, pinned, ix, iy, ixw, iyw,
                                   grams, (0.0, 0.0), h)
        ch[n] = study.kernel_coercivity(M_X, B, ixw, iyw, grams, (0.0, 0.0), h)
    # two halvings of h, so h^(-1/2) is a factor of 2
    assert 1.7 < ch[16] / ch[4] < 2.3, ch
