import numpy as np

from mimetika.dof import DofHandler, MixedDofHandler


def test_single_space_counts(box_222):
    for k in range(4):
        dh = DofHandler(box_222, k)
        assert dh.n_dofs == box_222.num_cells(k)


def test_local_to_global_identity(box_222):
    dh = DofHandler(box_222, 1)
    idx = np.array([0, 5, 9])
    assert np.array_equal(dh.local_to_global(idx), idx)


def test_mixed_offsets(box_222):
    flux = DofHandler(box_222, 2)  # facets
    pot = DofHandler(box_222, 3)  # cells
    mixed = MixedDofHandler([flux, pot])
    assert mixed.offsets == [0, flux.n_dofs]
    assert mixed.n_dofs == flux.n_dofs + pot.n_dofs
    assert mixed.global_range(1) == (flux.n_dofs, flux.n_dofs + pot.n_dofs)
