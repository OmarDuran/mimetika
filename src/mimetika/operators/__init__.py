"""Operators layer: exterior derivative (topology) + Hodge/mass (geometry)."""

from mimetika.operators.derivative import curl, div, exterior_derivative, grad
from mimetika.operators.diffusion import DiffusionInnerProduct
from mimetika.operators.elasticity import ElasticityInnerProduct
from mimetika.operators.hodge import DiagonalHodge, HodgeOperator
from mimetika.operators.lumped import LumpedDeviatoricStress
from mimetika.operators.inner_product import (
    assemble_local_inner_product,
    consistency_matrix,
    nullspace_basis,
    stabilization_dim,
)

__all__ = [
    "exterior_derivative",
    "grad",
    "curl",
    "div",
    "HodgeOperator",
    "DiagonalHodge",
    "assemble_local_inner_product",
    "consistency_matrix",
    "nullspace_basis",
    "stabilization_dim",
    "DiffusionInnerProduct",
    "ElasticityInnerProduct",
    "LumpedDeviatoricStress",
]
