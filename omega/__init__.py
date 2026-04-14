"""OMEGA - Optimised Modelling of Groundwater in Australia.

A utility package for generating extruded meshes and loading spatial data
for the G-ADOPT groundwater module.

Example:
    >>> from omega import SurfaceMesh, build_mesh_hierarchy
    >>> from omega.geometry import Polygon
    >>> polygon = Polygon([(0, 0), (280000, 0), (280000, 130000), (0, 130000)])
    >>> sm = SurfaceMesh(polygon, resolution=3500)
    >>> sm.generate()
    >>> mesh_2d = sm.to_firedrake_mesh()
    >>> hierarchy = build_mesh_hierarchy(
    ...     mesh_2d,
    ...     elevation_coords, elevation_values,
    ...     depth_coords, depth_values,
    ...     n_layers=300,
    ... )
"""

from omega.exceptions import (
    DataLoadError,
    InterpolationError,
    MeshGenerationError,
    OmegaError,
    PolygonError,
)
from omega.geometry import Polygon
from omega.mesh import ExtrusionConfig, SurfaceMesh, build_mesh_hierarchy

__version__ = "0.1.0"

__all__ = [
    # Main API
    "SurfaceMesh",
    "build_mesh_hierarchy",
    "Polygon",
    "ExtrusionConfig",
    # Exceptions
    "OmegaError",
    "PolygonError",
    "MeshGenerationError",
    "InterpolationError",
    "DataLoadError",
]
