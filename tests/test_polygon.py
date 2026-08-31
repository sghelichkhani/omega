"""Tests for omega.geometry.polygon module."""

import numpy as np
import pytest

from omega.geometry import Polygon
from omega.exceptions import PolygonError


class TestPolygon:
    """Tests for the Polygon class."""

    def test_create_from_coords(self, simple_polygon_coords):
        """Test creating a polygon from coordinate list."""
        polygon = Polygon(simple_polygon_coords)

        assert polygon.n_vertices == 4
        assert polygon.crs is None

    def test_create_with_crs(self, simple_polygon_coords):
        """Test creating a polygon with CRS."""
        polygon = Polygon(simple_polygon_coords, crs="EPSG:28354")

        assert polygon.crs == "EPSG:28354"

    def test_coords_property(self, simple_polygon_coords):
        """Test coords property returns correct coordinates."""
        polygon = Polygon(simple_polygon_coords)
        coords = polygon.coords

        assert len(coords) == 4
        assert coords[0] == (0, 0)
        assert coords[2] == (100, 100)

    def test_coords_array_property(self, simple_polygon_coords):
        """Test coords_array returns numpy array."""
        polygon = Polygon(simple_polygon_coords)
        coords = polygon.coords_array

        assert isinstance(coords, np.ndarray)
        assert coords.shape == (4, 2)

    def test_bounds_property(self, simple_polygon_coords):
        """Test bounds property."""
        polygon = Polygon(simple_polygon_coords)
        bounds = polygon.bounds

        assert bounds == (0.0, 0.0, 100.0, 100.0)

    def test_area_property(self, simple_polygon_coords):
        """Test area calculation."""
        polygon = Polygon(simple_polygon_coords)

        assert polygon.area == 10000.0  # 100 x 100

    def test_centroid_property(self, simple_polygon_coords):
        """Test centroid calculation."""
        polygon = Polygon(simple_polygon_coords)
        centroid = polygon.centroid

        assert centroid == (50.0, 50.0)

    def test_simplify(self, murrumbidgee_polygon_coords):
        """Test polygon simplification."""
        polygon = Polygon(murrumbidgee_polygon_coords)
        simplified = polygon.simplify(tolerance=10000)

        # Simplified should have same or fewer vertices
        assert simplified.n_vertices <= polygon.n_vertices

    def test_buffer(self, simple_polygon_coords):
        """Test polygon buffering."""
        polygon = Polygon(simple_polygon_coords)
        buffered = polygon.buffer(10)

        # Buffered polygon should be larger
        assert buffered.area > polygon.area

    def test_invalid_polygon_self_intersecting(self):
        """Test that self-intersecting polygon raises error."""
        # Bowtie shape (self-intersecting)
        coords = [(0, 0), (100, 100), (100, 0), (0, 100)]

        with pytest.raises(PolygonError):
            Polygon(coords)

    def test_from_shapely(self, simple_polygon_coords):
        """Test creating Polygon from shapely object."""
        from shapely.geometry import Polygon as ShapelyPolygon

        shapely_poly = ShapelyPolygon(simple_polygon_coords)
        polygon = Polygon.from_shapely(shapely_poly, crs="EPSG:4326")

        assert polygon.n_vertices == 4
        assert polygon.crs == "EPSG:4326"

    def test_shapely_property(self, simple_polygon_coords):
        """Test accessing underlying shapely object."""
        from shapely.geometry import Polygon as ShapelyPolygon

        polygon = Polygon(simple_polygon_coords)

        assert isinstance(polygon.shapely, ShapelyPolygon)

    def test_repr(self, simple_polygon_coords):
        """Test string representation."""
        polygon = Polygon(simple_polygon_coords, crs="EPSG:28354")
        repr_str = repr(polygon)

        assert "n_vertices=4" in repr_str
        assert "EPSG:28354" in repr_str
