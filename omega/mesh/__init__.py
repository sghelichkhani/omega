"""Mesh generation utilities."""

from omega.mesh.builder import build_mesh_hierarchy
from omega.mesh.extrusion import ExtrusionConfig
from omega.mesh.surface import SurfaceMesh
from omega.mesh.transform import (
    apply_terrain_transform,
    apply_terrain_transform_inplace,
    validate_terrain_data,
)

__all__ = [
    "SurfaceMesh",
    "build_mesh_hierarchy",
    "ExtrusionConfig",
    "apply_terrain_transform",
    "apply_terrain_transform_inplace",
    "validate_terrain_data",
]
