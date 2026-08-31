"""Tests for omega.mesh.transform module."""

import numpy as np
import pytest

from omega.mesh.transform import (
    apply_terrain_transform,
    validate_terrain_data,
)


class TestApplyTerrainTransform:
    """Tests for the terrain-following coordinate transform."""

    def test_transform_at_z_zero(self):
        """Test transform at z=0 gives bedrock elevation."""
        # z=0 should give elevation - depth (bedrock elevation)
        coords = np.array([[0, 0, 0], [10, 10, 0], [20, 20, 0]])
        elevation = np.array([100, 110, 120])
        depth = np.array([50, 60, 70])

        result = apply_terrain_transform(coords, elevation, depth)

        expected = elevation - depth  # Bedrock elevation
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_at_z_one(self):
        """Test transform at z=1 gives surface elevation."""
        # z=1 should give elevation (surface)
        coords = np.array([[0, 0, 1], [10, 10, 1], [20, 20, 1]])
        elevation = np.array([100, 110, 120])
        depth = np.array([50, 60, 70])

        result = apply_terrain_transform(coords, elevation, depth)

        np.testing.assert_array_almost_equal(result, elevation)

    def test_transform_at_z_half(self):
        """Test transform at z=0.5 gives midpoint."""
        coords = np.array([[0, 0, 0.5]])
        elevation = np.array([100])
        depth = np.array([50])

        result = apply_terrain_transform(coords, elevation, depth)

        # z=0.5: elevation - depth * (1 - 0.5) = 100 - 50 * 0.5 = 75
        expected = np.array([75])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_preserves_xy(self):
        """Test that transform only affects z coordinate."""
        coords = np.array([[10, 20, 0.5], [30, 40, 0.5]])
        elevation = np.array([100, 100])
        depth = np.array([50, 50])

        # Transform only returns new z values
        result = apply_terrain_transform(coords, elevation, depth)

        assert len(result) == 2
        # XY should not be in the result (only z is returned)

    def test_transform_formula_correctness(self):
        """Test the exact formula: z_physical = depth*z + elevation - depth."""
        coords = np.array([[0, 0, 0.3]])
        elevation = np.array([100])
        depth = np.array([50])

        result = apply_terrain_transform(coords, elevation, depth)

        # z_physical = depth * z + elevation - depth
        # = 50 * 0.3 + 100 - 50 = 15 + 50 = 65
        expected = np.array([65])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_with_varying_depth(self):
        """Test transform with spatially varying depth."""
        coords = np.array([
            [0, 0, 0],    # z=0, bedrock
            [0, 0, 1],    # z=1, surface
            [10, 0, 0],   # z=0, bedrock (different location)
            [10, 0, 1],   # z=1, surface (different location)
        ])
        elevation = np.array([100, 100, 150, 150])
        depth = np.array([30, 30, 80, 80])

        result = apply_terrain_transform(coords, elevation, depth)

        expected = np.array([
            100 - 30,  # 70 (bedrock at first location)
            100,       # 100 (surface at first location)
            150 - 80,  # 70 (bedrock at second location)
            150,       # 150 (surface at second location)
        ])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_shape_validation(self):
        """Test that invalid shapes raise errors."""
        # 2D coords (missing z)
        coords_2d = np.array([[0, 0], [10, 10]])
        elevation = np.array([100, 110])
        depth = np.array([50, 60])

        with pytest.raises(ValueError):
            apply_terrain_transform(coords_2d, elevation, depth)

    def test_transform_length_validation(self):
        """Test that mismatched lengths raise errors."""
        coords = np.array([[0, 0, 0], [10, 10, 0.5]])
        elevation = np.array([100, 110, 120])  # Wrong length
        depth = np.array([50, 60])

        with pytest.raises(ValueError):
            apply_terrain_transform(coords, elevation, depth)


class TestValidateTerrainData:
    """Tests for terrain data validation."""

    def test_valid_data(self):
        """Test validation passes for valid data."""
        elevation = np.array([100, 110, 120])
        depth = np.array([50, 60, 70])  # All positive

        is_valid, message = validate_terrain_data(elevation, depth)

        assert is_valid
        assert "valid" in message.lower()

    def test_negative_depth(self):
        """Test validation fails for negative depth."""
        elevation = np.array([100, 110, 120])
        depth = np.array([50, -10, 70])  # One negative

        is_valid, message = validate_terrain_data(elevation, depth)

        assert not is_valid
        assert "negative" in message.lower()

    def test_negative_depth_with_tolerance(self):
        """Test validation with tolerance for small negative values."""
        elevation = np.array([100, 110, 120])
        depth = np.array([50, -0.5, 70])  # Small negative

        # Should fail without tolerance
        is_valid, _ = validate_terrain_data(elevation, depth, tolerance=0.0)
        assert not is_valid

        # Should pass with tolerance
        is_valid, _ = validate_terrain_data(elevation, depth, tolerance=1.0)
        assert is_valid

    def test_nan_elevation(self):
        """Test validation fails for NaN in elevation."""
        elevation = np.array([100, np.nan, 120])
        depth = np.array([50, 60, 70])

        is_valid, message = validate_terrain_data(elevation, depth)

        assert not is_valid
        assert "NaN" in message or "nan" in message.lower()

    def test_inf_depth(self):
        """Test validation fails for infinite depth."""
        elevation = np.array([100, 110, 120])
        depth = np.array([50, np.inf, 70])

        is_valid, message = validate_terrain_data(elevation, depth)

        assert not is_valid
        assert "inf" in message.lower() or "NaN" in message
