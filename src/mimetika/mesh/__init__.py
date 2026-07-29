"""Mesh layer: container tying topology + geometry, plus generators/readers."""

from mimetika.mesh.generators import structured_box
from mimetika.mesh.mesh import Mesh

__all__ = ["Mesh", "structured_box"]
