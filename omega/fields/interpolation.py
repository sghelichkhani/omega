"""Field interpolation using scipy.spatial.cKDTree."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.spatial import cKDTree

from omega.exceptions import InterpolationError
from omega.io.readers import SpatialData


class FieldInterpolator:
    """Interpolator for scattered spatial data using cKDTree.

    Supports nearest neighbor and inverse distance weighting (IDW) interpolation.

    Args:
        source_coords: Coordinates of source data points, shape (n, dim).
        source_values: Values at source points, shape (n,).

    Example:
        >>> interpolator = FieldInterpolator(coords, values)
        >>> target_values = interpolator.nearest(target_coords)
        >>> target_values = interpolator.idw(target_coords, k=4, power=2.0)
    """

    def __init__(
        self,
        source_coords: np.ndarray,
        source_values: np.ndarray,
    ):
        self._coords = np.asarray(source_coords)
        self._values = np.asarray(source_values)

        if self._coords.ndim != 2:
            raise InterpolationError("source_coords must be 2D array")
        if self._values.ndim != 1:
            raise InterpolationError("source_values must be 1D array")
        if len(self._coords) != len(self._values):
            raise InterpolationError(
                f"coords and values must have same length, "
                f"got {len(self._coords)} and {len(self._values)}"
            )

        self._tree = cKDTree(self._coords)

    @classmethod
    def from_spatial_data(cls, data: SpatialData) -> FieldInterpolator:
        """Create interpolator from SpatialData object.

        Args:
            data: SpatialData object containing coordinates and values.

        Returns:
            New FieldInterpolator instance.
        """
        return cls(data.coords, data.values)

    @property
    def n_points(self) -> int:
        """Number of source data points."""
        return len(self._values)

    @property
    def ndim(self) -> int:
        """Spatial dimension of source coordinates."""
        return self._coords.shape[1]

    def nearest(self, target_coords: np.ndarray) -> np.ndarray:
        """Nearest neighbor interpolation.

        Args:
            target_coords: Coordinates to interpolate to, shape (m, dim).
                Only the first `ndim` columns are used if more are provided.

        Returns:
            Interpolated values at target coordinates, shape (m,).
        """
        target = np.asarray(target_coords)
        if target.ndim == 1:
            target = target.reshape(1, -1)

        # Use only the spatial dimensions matching source data
        target_query = target[:, : self.ndim]

        _, indices = self._tree.query(target_query, k=1)
        return self._values[indices]

    def idw(
        self,
        target_coords: np.ndarray,
        k: int = 4,
        power: float = 2.0,
        eps: float = 1e-12,
    ) -> np.ndarray:
        """Inverse distance weighting interpolation.

        Args:
            target_coords: Coordinates to interpolate to, shape (m, dim).
                Only the first `ndim` columns are used if more are provided.
            k: Number of nearest neighbors to use. Default: 4.
            power: Power parameter for distance weighting. Default: 2.0.
            eps: Small value to avoid division by zero. Default: 1e-12.

        Returns:
            Interpolated values at target coordinates, shape (m,).
        """
        target = np.asarray(target_coords)
        if target.ndim == 1:
            target = target.reshape(1, -1)

        # Use only the spatial dimensions matching source data
        target_query = target[:, : self.ndim]

        # Query k nearest neighbors
        k = min(k, self.n_points)
        distances, indices = self._tree.query(target_query, k=k)

        # Handle case when k=1
        if k == 1:
            distances = distances.reshape(-1, 1)
            indices = indices.reshape(-1, 1)

        # Compute weights (inverse distance raised to power)
        weights = 1.0 / (distances**power + eps)

        # Normalize weights
        weights_sum = weights.sum(axis=1, keepdims=True)
        weights = weights / weights_sum

        # Weighted average of neighbor values
        neighbor_values = self._values[indices]
        result = (weights * neighbor_values).sum(axis=1)

        return result

    def __call__(
        self,
        target_coords: np.ndarray,
        method: Literal["nearest", "idw"] = "nearest",
        **kwargs,
    ) -> np.ndarray:
        """Interpolate to target coordinates.

        Args:
            target_coords: Coordinates to interpolate to.
            method: Interpolation method ('nearest' or 'idw').
            **kwargs: Additional arguments passed to the interpolation method.

        Returns:
            Interpolated values at target coordinates.
        """
        if method == "nearest":
            return self.nearest(target_coords, **kwargs)
        elif method == "idw":
            return self.idw(target_coords, **kwargs)
        else:
            raise InterpolationError(
                f"Unknown interpolation method: {method}. "
                f"Supported methods: 'nearest', 'idw'"
            )


def interpolate_to_coords(
    source_coords: np.ndarray,
    source_values: np.ndarray,
    target_coords: np.ndarray,
    method: Literal["nearest", "idw"] = "nearest",
    **kwargs,
) -> np.ndarray:
    """Convenience function for one-shot interpolation.

    Args:
        source_coords: Coordinates of source data points.
        source_values: Values at source points.
        target_coords: Coordinates to interpolate to.
        method: Interpolation method ('nearest' or 'idw').
        **kwargs: Additional arguments passed to the interpolation method.

    Returns:
        Interpolated values at target coordinates.
    """
    interpolator = FieldInterpolator(source_coords, source_values)
    return interpolator(target_coords, method=method, **kwargs)
