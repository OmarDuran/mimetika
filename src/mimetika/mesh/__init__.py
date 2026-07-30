"""Mesh layer: container tying topology + geometry, plus generators/readers."""

from mimetika.mesh.generators import (
    single_hexahedron,
    single_tetrahedron,
    structured_box,
    structured_quads,
    structured_tets,
    structured_triangles,
)
from mimetika.mesh.mesh import Mesh

__all__ = [
    "Mesh",
    "structured_box",
    "structured_tets",
    "structured_quads",
    "structured_triangles",
    "single_tetrahedron",
    "single_hexahedron",
]
