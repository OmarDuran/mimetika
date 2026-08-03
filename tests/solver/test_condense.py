r"""Exact stress condensation: correct, and honest about when it applies."""

import numpy as np
import pytest
import scipy.sparse.linalg as spla

from mimetika.assembly.four_field import FourFieldElasticity
from mimetika.assembly.kinematic import TwoPointFourField
from mimetika.assembly.mixed import MixedElasticity
from mimetika.mesh import structured_box, structured_quads
from mimetika.operators.lumped import LumpedDeviatoricStress
from mimetika.solver.condense import eliminate_leading_diagonal, solve_condensed

MU, LAM = 1.3, 2.7
U_B = np.array([[0.5, -0.3, 0.2], [0.15, 0.4, -0.25], [-0.1, 0.35, 0.6]])
displacement = lambda x: np.atleast_2d(x) @ U_B.T  # noqa: E731


@pytest.mark.parametrize("mesh_maker", [lambda: structured_quads(4, 4),
                                        lambda: structured_box(3, 3, 3)],
                         ids=["quads", "hex"])
@pytest.mark.parametrize("formulation", ["multiplier", "kinematic"])
def test_condensed_equals_full(mesh_maker, formulation):
    """Reduce, solve, back-substitute -- identical to the full solve."""
    mesh = mesh_maker()
    if formulation == "multiplier":
        problem = FourFieldElasticity(
            mesh, inner=LumpedDeviatoricStress(mesh, mu=MU, lam=LAM)
        )
    else:
        problem = TwoPointFourField(mesh, MU, LAM)
    S, rhs = problem.assemble(dirichlet=displacement)

    full = spla.spsolve(S.tocsc(), rhs)
    condensed = solve_condensed(S, rhs, problem.n_stress)
    scale = np.abs(full).max()
    assert np.abs(condensed - full).max() < 1e-10 * scale

    # the reduced system is genuinely cell-centred
    reduced, _, _ = eliminate_leading_diagonal(S, rhs, problem.n_stress)
    d, nsk = problem.d, problem.n_skew
    assert reduced.shape[0] == (1 + d + nsk) * problem.n_cells


def test_afw_is_refused():
    """The AFW stress block couples the facets of a cell: no exact condensation."""
    problem = FourFieldElasticity(structured_quads(3, 3), MU, LAM)
    S, rhs = problem.assemble(dirichlet=displacement)
    with pytest.raises(ValueError, match="not diagonal"):
        eliminate_leading_diagonal(S, rhs, problem.n_stress)


def test_three_field_lumped_is_refused():
    """The volumetric fold-in re-couples the facets: the three-field lumped
    arrangement cannot be condensed -- the four-field split is what makes the
    diagonal survive assembly."""
    mesh = structured_quads(3, 3)
    problem = MixedElasticity(
        mesh, inner=LumpedDeviatoricStress(mesh, mu=MU, lam=LAM)
    )
    S, rhs = problem.assemble(dirichlet=displacement)
    with pytest.raises(ValueError, match="not diagonal"):
        eliminate_leading_diagonal(S, rhs, problem.n_stress)
