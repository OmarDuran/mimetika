"""Mesh layer: container tying topology + geometry, plus generators/readers."""

from mimetika.mesh.generators import (
    single_hexahedron,
    single_tetrahedron,
    structured_box,
)
from mimetika.mesh.mesh import Mesh

__all__ = [
    "Mesh",
    "structured_box",
    "single_tetrahedron",
    "single_hexahedron",
]
