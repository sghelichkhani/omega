"""OMEGA - Optimised Modelling of Groundwater in Australia.

A utility package for generating extruded meshes and loading spatial data
for the G-ADOPT groundwater module.

Example:
    >>> from omega import MeshBuilder
    >>> mesh = (
    ...     MeshBuilder(polygon_coords)
    ...     .set_horizontal_resolution(3500)
    ...     .set_layer_heights(layer_height=1/300, n_layers=300)
    ...     .load_surface_elevation("elevation_data.csv")
    ...     .load_bedrock("bedrock_data.csv")
    ...     .build()
    ... )
    >>> from omega.io import save_mesh
    >>> save_mesh(mesh, "output.h5")
"""

from omega.exceptions import (
    DataLoadError,
    InterpolationError,
    MeshGenerationError,
    OmegaError,
    PolygonError,
)
from omega.geometry import Polygon
from omega.mesh import ExtrusionConfig, MeshBuilder

__version__ = "0.1.0"

__all__ = [
    # Main API
    "MeshBuilder",
    "Polygon",
    "ExtrusionConfig",
    # Exceptions
    "OmegaError",
    "PolygonError",
    "MeshGenerationError",
    "InterpolationError",
    "DataLoadError",
]
