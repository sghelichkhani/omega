"""Vertical extrusion configuration for 3D mesh generation."""

from __future__ import annotations

import numpy as np


class ExtrusionConfig:
    """Configuration for vertical mesh extrusion.

    Handles layer height specification following Firedrake's ExtrudedMesh convention.
    Layer heights are normalized to sum to 1.0 for use with terrain-following
    coordinate transforms.

    Args:
        layer_heights: Either a constant thickness or array of thicknesses.
            If constant (float), n_layers must be provided.
            If array, defines each layer's thickness from bottom to top.
        n_layers: Number of layers (required if layer_heights is a scalar).

    Example:
        >>> # Uniform layers
        >>> config = ExtrusionConfig(layer_heights=10.0, n_layers=50)
        >>> config.n_layers
        50

        >>> # Variable layers (graded)
        >>> config = ExtrusionConfig(layer_heights=[5, 5, 10, 10, 20, 20])
        >>> config.n_layers
        6
    """

    def __init__(
        self,
        layer_heights: float | list[float] | np.ndarray,
        n_layers: int | None = None,
    ):
        # Handle scalar vs array input
        if isinstance(layer_heights, (int, float)):
            if n_layers is None:
                raise ValueError(
                    "n_layers must be provided when layer_heights is a scalar"
                )
            self._layer_heights = np.full(n_layers, float(layer_heights))
            self._uniform = True
        else:
            self._layer_heights = np.asarray(layer_heights, dtype=float)
            if self._layer_heights.ndim != 1:
                raise ValueError("layer_heights must be 1D array")
            if n_layers is not None and n_layers != len(self._layer_heights):
                raise ValueError(
                    f"n_layers ({n_layers}) does not match length of "
                    f"layer_heights ({len(self._layer_heights)})"
                )
            self._uniform = np.allclose(
                self._layer_heights, self._layer_heights[0]
            )

        if np.any(self._layer_heights <= 0):
            raise ValueError("All layer heights must be positive")

    @classmethod
    def uniform(cls, layer_height: float, n_layers: int) -> ExtrusionConfig:
        """Create configuration with uniform layer heights.

        Args:
            layer_height: Height of each layer.
            n_layers: Number of layers.

        Returns:
            ExtrusionConfig with uniform layers.
        """
        return cls(layer_heights=layer_height, n_layers=n_layers)

    @classmethod
    def graded(
        cls,
        total_height: float,
        n_layers: int,
        grading: float = 1.0,
    ) -> ExtrusionConfig:
        """Create configuration with geometrically graded layers.

        Layers are graded from bottom to top with a geometric progression.
        Grading > 1 means layers get thicker towards the top (surface).
        Grading < 1 means layers get thicker towards the bottom.

        Args:
            total_height: Total height of all layers combined.
            n_layers: Number of layers.
            grading: Ratio of successive layer heights. Default: 1.0 (uniform).

        Returns:
            ExtrusionConfig with graded layers.
        """
        if grading == 1.0:
            layer_height = total_height / n_layers
            return cls.uniform(layer_height, n_layers)

        # Geometric progression: h_i = h_0 * grading^i
        # Total: sum(h_0 * grading^i for i in 0..n-1) = h_0 * (grading^n - 1) / (grading - 1)
        h0 = total_height * (grading - 1) / (grading**n_layers - 1)
        heights = h0 * (grading ** np.arange(n_layers))

        return cls(layer_heights=heights)

    @property
    def n_layers(self) -> int:
        """Number of vertical layers."""
        return len(self._layer_heights)

    @property
    def layer_heights(self) -> np.ndarray:
        """Layer heights as numpy array."""
        return self._layer_heights.copy()

    @property
    def total_height(self) -> float:
        """Total height of all layers combined."""
        return float(self._layer_heights.sum())

    @property
    def is_uniform(self) -> bool:
        """Return True if all layers have equal height."""
        return self._uniform

    @property
    def normalized_heights(self) -> np.ndarray:
        """Layer heights normalized to sum to 1.0.

        This is the format expected by Firedrake's ExtrudedMesh for use with
        terrain-following coordinate transforms where z ranges from 0 to 1.
        """
        return self._layer_heights / self.total_height

    @property
    def normalized_layer_height(self) -> float | np.ndarray:
        """Normalized layer height for Firedrake ExtrudedMesh.

        For uniform layers, returns a scalar (1/n_layers).
        For variable layers, returns array of normalized heights.
        """
        if self._uniform:
            return 1.0 / self.n_layers
        return self.normalized_heights

    def get_layer_boundaries(self, normalized: bool = True) -> np.ndarray:
        """Return z-coordinates of layer boundaries.

        Args:
            normalized: If True, return values in [0, 1]. If False, return
                actual heights.

        Returns:
            Array of shape (n_layers + 1,) with boundary z-coordinates.
        """
        heights = self.normalized_heights if normalized else self._layer_heights
        boundaries = np.zeros(self.n_layers + 1)
        boundaries[1:] = np.cumsum(heights)
        return boundaries

    def __repr__(self) -> str:
        mode = "uniform" if self._uniform else "variable"
        return (
            f"ExtrusionConfig(n_layers={self.n_layers}, "
            f"total_height={self.total_height:.2f}, mode={mode})"
        )
