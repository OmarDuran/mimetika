"""Assembly layer: global operators, linear systems, backend conversion."""

from mimetika.assembly.backend import petsc_available, to_petsc_mat, to_petsc_vec
from mimetika.assembly.system import LinearSystem, apply_dirichlet, hodge_laplacian

__all__ = [
    "LinearSystem",
    "hodge_laplacian",
    "apply_dirichlet",
    "petsc_available",
    "to_petsc_mat",
    "to_petsc_vec",
]
