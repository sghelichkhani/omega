"""Tests for omega.mesh.extrusion module."""

import numpy as np
import pytest

from omega.mesh.extrusion import ExtrusionConfig


class TestExtrusionConfig:
    """Tests for the ExtrusionConfig class."""

    def test_uniform_layers_from_scalar(self):
        """Test creating uniform layers from scalar height."""
        config = ExtrusionConfig(layer_heights=10.0, n_layers=5)

        assert config.n_layers == 5
        assert config.total_height == 50.0
        assert config.is_uniform

    def test_uniform_layers_from_array(self):
        """Test creating uniform layers from array of equal heights."""
        config = ExtrusionConfig(layer_heights=[10, 10, 10, 10, 10])

        assert config.n_layers == 5
        assert config.total_height == 50.0
        assert config.is_uniform

    def test_variable_layers(self):
        """Test creating variable layer heights."""
        config = ExtrusionConfig(layer_heights=[5, 10, 15, 20])

        assert config.n_layers == 4
        assert config.total_height == 50.0
        assert not config.is_uniform

    def test_normalized_heights(self):
        """Test normalized heights sum to 1."""
        config = ExtrusionConfig(layer_heights=[5, 10, 15, 20])
        normalized = config.normalized_heights

        assert np.isclose(normalized.sum(), 1.0)
        assert len(normalized) == 4

    def test_normalized_layer_height_uniform(self):
        """Test normalized layer height for uniform layers."""
        config = ExtrusionConfig(layer_heights=10.0, n_layers=4)
        normalized = config.normalized_layer_height

        assert np.isclose(normalized, 0.25)

    def test_normalized_layer_height_variable(self):
        """Test normalized layer height for variable layers."""
        config = ExtrusionConfig(layer_heights=[5, 10, 15, 20])
        normalized = config.normalized_layer_height

        assert isinstance(normalized, np.ndarray)
        assert np.isclose(normalized.sum(), 1.0)

    def test_layer_boundaries_normalized(self):
        """Test layer boundary computation (normalized)."""
        config = ExtrusionConfig(layer_heights=10.0, n_layers=4)
        boundaries = config.get_layer_boundaries(normalized=True)

        assert len(boundaries) == 5  # n_layers + 1
        assert boundaries[0] == 0.0
        assert np.isclose(boundaries[-1], 1.0)

    def test_layer_boundaries_absolute(self):
        """Test layer boundary computation (absolute)."""
        config = ExtrusionConfig(layer_heights=10.0, n_layers=4)
        boundaries = config.get_layer_boundaries(normalized=False)

        assert len(boundaries) == 5
        assert boundaries[0] == 0.0
        assert boundaries[-1] == 40.0

    def test_uniform_classmethod(self):
        """Test the uniform() class method."""
        config = ExtrusionConfig.uniform(layer_height=10.0, n_layers=5)

        assert config.n_layers == 5
        assert config.total_height == 50.0
        assert config.is_uniform

    def test_graded_classmethod_uniform(self):
        """Test graded() with grading=1.0 gives uniform layers."""
        config = ExtrusionConfig.graded(total_height=100, n_layers=10, grading=1.0)

        assert config.n_layers == 10
        assert np.isclose(config.total_height, 100.0)
        assert config.is_uniform

    def test_graded_classmethod_increasing(self):
        """Test graded() with grading > 1 gives increasing layer heights."""
        config = ExtrusionConfig.graded(total_height=100, n_layers=10, grading=1.2)
        heights = config.layer_heights

        assert config.n_layers == 10
        assert np.isclose(config.total_height, 100.0)
        # Each layer should be larger than the previous
        for i in range(1, len(heights)):
            assert heights[i] > heights[i - 1]

    def test_graded_classmethod_decreasing(self):
        """Test graded() with grading < 1 gives decreasing layer heights."""
        config = ExtrusionConfig.graded(total_height=100, n_layers=10, grading=0.8)
        heights = config.layer_heights

        # Each layer should be smaller than the previous
        for i in range(1, len(heights)):
            assert heights[i] < heights[i - 1]

    def test_scalar_without_n_layers_raises(self):
        """Test that scalar height without n_layers raises error."""
        with pytest.raises(ValueError):
            ExtrusionConfig(layer_heights=10.0)

    def test_mismatched_n_layers_raises(self):
        """Test that mismatched n_layers raises error."""
        with pytest.raises(ValueError):
            ExtrusionConfig(layer_heights=[10, 10, 10], n_layers=5)

    def test_negative_height_raises(self):
        """Test that negative heights raise error."""
        with pytest.raises(ValueError):
            ExtrusionConfig(layer_heights=[-10, 10, 10])

    def test_repr(self):
        """Test string representation."""
        config = ExtrusionConfig(layer_heights=10.0, n_layers=5)
        repr_str = repr(config)

        assert "n_layers=5" in repr_str
        assert "uniform" in repr_str
