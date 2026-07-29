import numpy as np

from mimetika.operators import DiagonalHodge


def test_diagonal_hodge_matches_measures(box_222):
    h = DiagonalHodge(box_222.geometry)
    for k in range(4):
        diag = h.matrix(k).diagonal()
        assert np.allclose(diag, box_222.geometry.measure(k))


def test_diagonal_hodge_spd(box_222):
    h = DiagonalHodge(box_222.geometry)
    for k in range(4):
        assert (h.matrix(k).diagonal() > 0).all()


def test_inverse_is_reciprocal(box_222):
    h = DiagonalHodge(box_222.geometry)
    for k in range(1, 4):
        prod = h.matrix(k).diagonal() * h.inverse(k).diagonal()
        assert np.allclose(prod, 1.0)
