"""Tests for omega.fields.interpolation module."""

import numpy as np
import pytest

from omega.fields.interpolation import FieldInterpolator, interpolate_to_coords
from omega.exceptions import InterpolationError


class TestFieldInterpolator:
    """Tests for the FieldInterpolator class."""

    def test_create_interpolator(self, sample_elevation_data):
        """Test creating an interpolator."""
        coords, values = sample_elevation_data
        interp = FieldInterpolator(coords, values)

        assert interp.n_points == len(values)
        assert interp.ndim == 2

    def test_nearest_interpolation(self, sample_elevation_data):
        """Test nearest neighbor interpolation."""
        coords, values = sample_elevation_data
        interp = FieldInterpolator(coords, values)

        # Interpolate at a known point
        target = np.array([[50, 50]])
        result = interp.nearest(target)

        assert len(result) == 1
        # Should be close to the value at (50, 50)
        expected = 100 + 0.1 * 50 + 0.05 * 50
        assert np.isclose(result[0], expected, rtol=0.1)

    def test_idw_interpolation(self, sample_elevation_data):
        """Test inverse distance weighting interpolation."""
        coords, values = sample_elevation_data
        interp = FieldInterpolator(coords, values)

        # Interpolate at a known point
        target = np.array([[50, 50]])
        result = interp.idw(target, k=4, power=2.0)

        assert len(result) == 1
        # Should be close to the value at (50, 50)
        expected = 100 + 0.1 * 50 + 0.05 * 50
        assert np.isclose(result[0], expected, rtol=0.1)

    def test_interpolation_at_source_points(self, sample_elevation_data):
        """Test interpolation exactly at source points."""
        coords, values = sample_elevation_data
        interp = FieldInterpolator(coords, values)

        # Interpolate at source points
        result = interp.nearest(coords)

        np.testing.assert_array_almost_equal(result, values)

    def test_callable_interface(self, sample_elevation_data):
        """Test the __call__ interface."""
        coords, values = sample_elevation_data
        interp = FieldInterpolator(coords, values)

        target = np.array([[50, 50]])

        # Test nearest via __call__
        result_nearest = interp(target, method="nearest")
        result_idw = interp(target, method="idw")

        assert len(result_nearest) == 1
        assert len(result_idw) == 1

    def test_invalid_method(self, sample_elevation_data):
        """Test that invalid method raises error."""
        coords, values = sample_elevation_data
        interp = FieldInterpolator(coords, values)

        with pytest.raises(InterpolationError):
            interp(np.array([[50, 50]]), method="invalid")

    def test_from_spatial_data(self, sample_elevation_data):
        """Test creating interpolator from SpatialData."""
        from omega.io.readers import SpatialData

        coords, values = sample_elevation_data
        data = SpatialData(coords, values, name="test")

        interp = FieldInterpolator.from_spatial_data(data)

        assert interp.n_points == len(values)

    def test_mismatched_shapes(self):
        """Test that mismatched coords/values raises error."""
        coords = np.array([[0, 0], [1, 1], [2, 2]])
        values = np.array([1, 2])  # Wrong length

        with pytest.raises(InterpolationError):
            FieldInterpolator(coords, values)

    def test_3d_target_coords(self, sample_elevation_data):
        """Test interpolation with 3D target coordinates (uses only x,y)."""
        coords, values = sample_elevation_data
        interp = FieldInterpolator(coords, values)

        # 3D target (z should be ignored)
        target = np.array([[50, 50, 999]])
        result = interp.nearest(target)

        assert len(result) == 1


class TestInterpolateToCoords:
    """Tests for the interpolate_to_coords convenience function."""

    def test_convenience_function(self, sample_elevation_data):
        """Test the convenience function."""
        coords, values = sample_elevation_data
        target = np.array([[50, 50], [25, 75]])

        result = interpolate_to_coords(coords, values, target, method="nearest")

        assert len(result) == 2
