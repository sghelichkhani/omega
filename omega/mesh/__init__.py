"""Mesh generation utilities."""

from omega.mesh.builder import MeshBuilder
from omega.mesh.extrusion import ExtrusionConfig
from omega.mesh.surface import SurfaceMesh, generate_surface_mesh
from omega.mesh.transform import (
    apply_terrain_transform,
    apply_terrain_transform_inplace,
    compute_layer_thickness,
    validate_terrain_data,
)

__all__ = [
    "MeshBuilder",
    "ExtrusionConfig",
    "SurfaceMesh",
    "generate_surface_mesh",
    "apply_terrain_transform",
    "apply_terrain_transform_inplace",
    "compute_layer_thickness",
    "validate_terrain_data",
]
