"""Postprocessing: field reconstruction, VTK/VTU export and error norms."""

from mimetika.postprocess.export import export_vtk, l2_error
from mimetika.postprocess.reconstruct import (
    mean_stress,
    principal_stresses,
    reconstruct_flux,
    reconstruct_stress,
    von_mises,
)
from mimetika.postprocess.series import (
    MixedDimensionalSeries,
    contact_fields,
    darcy_fields,
    mechanics_fields,
    facet_vectors,
)
from mimetika.postprocess.vtu import export_facets, export_vtu

__all__ = [
    "export_vtk",
    "export_vtu",
    "export_facets",
    "MixedDimensionalSeries",
    "contact_fields",
    "darcy_fields",
    "mechanics_fields",
    "facet_vectors",
    "l2_error",
    "reconstruct_flux",
    "reconstruct_stress",
    "von_mises",
    "mean_stress",
    "principal_stresses",
]
