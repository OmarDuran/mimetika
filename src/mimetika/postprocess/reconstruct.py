r"""Reconstruct cell-centred fields from facet degrees of freedom.

Mimetic unknowns live *on facets*: a normal flux ``u_e`` per face for Darcy, and
traction moments ``\int_e sigma n_e`` per face for elasticity.  Neither is
directly plottable -- to visualise a velocity field or a stress tensor you first
have to rebuild a cell-centred object from the facet data.

Both reconstructions come from the same identity that underpins the whole
method.  For a closed cell with planar faces,

    ``sum_e |e| n_e (x_e - x_E)^T = |E| I`` ,

which is the divergence theorem applied to the linear field ``x - x_E``.  It
gives reconstructions that are **exact for constant fields**:

* velocity   ``u_E     = (1/|E|) sum_e (|e| u_e) (x_e - x_E)``
* stress     ``sigma_E = (1/|E|) sum_e t_e (x) (x_e - x_E)``   (outer product)

with ``t_e = \int_e sigma n_e`` the facet traction vector.  Because the schemes
reproduce constant fluxes and constant stresses exactly, these are the natural
partners of the discretisation rather than an ad-hoc smoothing.

All DOFs are taken in the global (canonical facet orientation) convention; the
incidence sign converts them to the cell's outward convention internally.
"""

from __future__ import annotations

import numpy as np

from mimetika.mesh.mesh import Mesh


def reconstruct_flux(mesh: Mesh, flux: np.ndarray) -> np.ndarray:
    """Cell-centred velocity vectors ``(n_cells, 3)`` from facet normal fluxes.

    Exact whenever the discrete flux is the interpolant of a constant field.
    """
    d = mesh.dim
    flux = np.asarray(flux, dtype=float)
    areas = mesh.geometry.measure(d - 1)
    fcent = mesh.geometry.centroids(d - 1)
    ccent = mesh.geometry.centroids(d)
    vol = mesh.geometry.measure(d)

    out = np.zeros((mesh.num_cells(d), 3))
    for c in range(mesh.num_cells(d)):
        acc = np.zeros(3)
        for fid, sign in mesh.complex.facets_of(d, c):
            acc += (sign * flux[fid] * areas[fid]) * (fcent[fid] - ccent[c])
        out[c] = acc / vol[c]
    return out


def reconstruct_stress(
    mesh: Mesh, stress: np.ndarray, dofs_per_facet: int | None = None
) -> np.ndarray:
    """Cell-centred stress tensors ``(n_cells, 3, 3)`` from traction moments.

    Uses the constant (``b = 0``) moment of each facet, which is exactly the
    facet traction ``\\int_e sigma n_e``.  Exact for constant stress fields.
    """
    d = mesh.dim
    ndf = d * d if dofs_per_facet is None else dofs_per_facet
    stress = np.asarray(stress, dtype=float).reshape(-1, ndf)
    fcent = mesh.geometry.centroids(d - 1)
    ccent = mesh.geometry.centroids(d)
    vol = mesh.geometry.measure(d)

    # within a facet block the DOFs are ordered (component k, basis b) -> k*d + b
    const_moment = stress[:, [k * d for k in range(d)]]  # (n_facets, d)

    out = np.zeros((mesh.num_cells(d), 3, 3))
    for c in range(mesh.num_cells(d)):
        acc = np.zeros((d, d))
        for fid, sign in mesh.complex.facets_of(d, c):
            acc += np.outer(sign * const_moment[fid], fcent[fid] - ccent[c])
        out[c, :d, :d] = acc / vol[c]
    return out


# -- derived scalar measures --------------------------------------------------


def von_mises(stress: np.ndarray) -> np.ndarray:
    """von Mises equivalent stress of ``(n, 3, 3)`` tensors."""
    s = np.asarray(stress, dtype=float)
    dev = s - np.einsum("nii->n", s)[:, None, None] / 3.0 * np.eye(3)
    return np.sqrt(1.5 * np.einsum("nij,nij->n", dev, dev))


def mean_stress(stress: np.ndarray) -> np.ndarray:
    """Mean (hydrostatic) stress ``tr(sigma)/3``."""
    return np.einsum("nii->n", np.asarray(stress, dtype=float)) / 3.0


def principal_stresses(stress: np.ndarray) -> np.ndarray:
    """Principal stresses ``(n, 3)``, ascending, of the symmetric part."""
    s = np.asarray(stress, dtype=float)
    sym = 0.5 * (s + np.swapaxes(s, 1, 2))
    return np.linalg.eigvalsh(sym)
