"""Shared pytest fixtures for OMEGA tests."""

import numpy as np
import pytest


@pytest.fixture
def simple_polygon_coords():
    """Simple rectangular polygon coordinates."""
    return [(0, 0), (100, 0), (100, 100), (0, 100)]


@pytest.fixture
def murrumbidgee_polygon_coords():
    """Lower Murrumbidgee basin polygon coordinates."""
    return [
        (0, 35000),
        (140000, 0),
        (280000, 0),
        (280000, 68000),
        (201000, 130000),
        (121000, 130000),
        (0, 100000),
    ]


@pytest.fixture
def sample_elevation_data():
    """Sample elevation data for testing."""
    # Create a grid of points
    x = np.linspace(0, 100, 11)
    y = np.linspace(0, 100, 11)
    xx, yy = np.meshgrid(x, y)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    # Elevation varies gently
    values = 100 + 0.1 * coords[:, 0] + 0.05 * coords[:, 1]
    return coords, values


@pytest.fixture
def sample_depth_data():
    """Sample depth-to-bedrock data for testing."""
    # Create a grid of points
    x = np.linspace(0, 100, 11)
    y = np.linspace(0, 100, 11)
    xx, yy = np.meshgrid(x, y)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    # Constant depth
    values = np.full(len(coords), 50.0)
    return coords, values


@pytest.fixture
def simple_surface_mesh(simple_polygon_coords):
    """Generate a SurfaceMesh from the simple rectangle polygon.

    Returns a SurfaceMesh that has already been generated with resolution=20.
    """
    from omega.geometry.polygon import Polygon
    from omega.mesh.surface import SurfaceMesh

    polygon = Polygon(simple_polygon_coords)
    sm = SurfaceMesh(polygon, resolution=20.0)
    sm.generate()
    return sm
