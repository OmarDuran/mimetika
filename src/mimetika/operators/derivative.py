"""Exterior derivative operators (the mimetic core).

The discrete exterior derivative on k-forms is the *coboundary*, i.e. the
transpose of the boundary operator on (k+1)-cells:

    ``D_k = boundary[k+1].T :  C^k -> C^{k+1}``

These operators are purely topological -- they carry no metric information --
and satisfy ``D_{k+1} @ D_k == 0`` exactly.  In 3D they are, in order:

    ``D_0 = grad``   (vertices -> edges)
    ``D_1 = curl``   (edges    -> facets)
    ``D_2 = div``    (facets   -> cells)

Because they come straight from the incidence matrices, the classic vector
identities ``curl grad = 0`` and ``div curl = 0`` hold to machine zero.
"""

from __future__ import annotations

import scipy.sparse as sp

from mimetika.mesh.mesh import Mesh


def exterior_derivative(mesh: Mesh, k: int) -> sp.csr_matrix:
    """Discrete exterior derivative ``D_k : C^k -> C^{k+1}``.

    Valid for ``0 <= k < dim``.
    """
    if not 0 <= k < mesh.dim:
        raise ValueError(f"D_k defined for 0..{mesh.dim - 1}, got k={k}")
    return mesh.complex.boundary_matrix(k + 1).transpose().tocsr()


def grad(mesh: Mesh) -> sp.csr_matrix:
    """0-forms (vertices) -> 1-forms (edges)."""
    return exterior_derivative(mesh, 0)


def curl(mesh: Mesh) -> sp.csr_matrix:
    """1-forms (edges) -> 2-forms (facets)."""
    return exterior_derivative(mesh, 1)


def div(mesh: Mesh) -> sp.csr_matrix:
    """2-forms (facets) -> 3-forms (cells)."""
    return exterior_derivative(mesh, 2)
