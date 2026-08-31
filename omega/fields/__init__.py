"""Field interpolation and layered-model utilities."""

from omega.fields.interpolation import (
    FieldInterpolator,
    interpolate_to_coords,
)
from omega.fields.mesh_fields import assign_field, node_coordinates
from omega.fields.stratigraphy import IntervalObservations, LayerModel
from omega.fields.surfaces import (
    GaussianKernelSurface,
    GridSurface,
    Surface,
    clamp_monotonic,
)

__all__ = [
    "FieldInterpolator",
    "interpolate_to_coords",
    "IntervalObservations",
    "LayerModel",
    "assign_field",
    "node_coordinates",
    "Surface",
    "GaussianKernelSurface",
    "GridSurface",
    "clamp_monotonic",
]
