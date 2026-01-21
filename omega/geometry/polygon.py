"""Polygon handling for domain definition."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import make_valid

from omega.exceptions import PolygonError


class Polygon:
    """Domain boundary polygon with validation and manipulation utilities.

    Wraps shapely.geometry.Polygon to provide validation, simplification,
    and coordinate extraction for mesh generation.

    Args:
        coords: List of (x, y) tuples defining the polygon boundary.
            Does not need to be closed (first point != last point).
        crs: Coordinate reference system (e.g., "EPSG:28354").
            Stored for metadata purposes; no transformation is applied.

    Raises:
        PolygonError: If the polygon is invalid or cannot be repaired.
    """

    def __init__(
        self,
        coords: Sequence[tuple[float, float]],
        crs: str | None = None,
    ):
        self._crs = crs
        self._coords = list(coords)

        # Create shapely polygon
        self._polygon = ShapelyPolygon(self._coords)

        # Validate and repair if needed
        if not self._polygon.is_valid:
            self._polygon = make_valid(self._polygon)
            if not self._polygon.is_valid:
                raise PolygonError(
                    f"Invalid polygon geometry: {self._polygon.is_valid}"
                )

        # Ensure counter-clockwise orientation (exterior ring)
        if not self._polygon.exterior.is_ccw:
            self._polygon = ShapelyPolygon(self._polygon.exterior.coords[::-1])

        # Check for simple polygon (no self-intersections)
        if not self._polygon.is_simple:
            raise PolygonError("Polygon has self-intersections")

    @classmethod
    def from_shapely(
        cls,
        polygon: ShapelyPolygon,
        crs: str | None = None,
    ) -> Polygon:
        """Create Polygon from shapely Polygon object.

        Args:
            polygon: Shapely Polygon object.
            crs: Coordinate reference system.

        Returns:
            New Polygon instance.
        """
        coords = list(polygon.exterior.coords)[:-1]  # Remove closing point
        return cls(coords, crs=crs)

    @property
    def coords(self) -> list[tuple[float, float]]:
        """Return polygon coordinates as list of (x, y) tuples."""
        # Exclude the closing point (shapely auto-closes)
        return list(self._polygon.exterior.coords)[:-1]

    @property
    def coords_array(self) -> np.ndarray:
        """Return polygon coordinates as numpy array of shape (n, 2)."""
        return np.array(self.coords)

    @property
    def crs(self) -> str | None:
        """Return coordinate reference system identifier."""
        return self._crs

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return bounding box as (xmin, ymin, xmax, ymax)."""
        return self._polygon.bounds

    @property
    def area(self) -> float:
        """Return polygon area in coordinate units squared."""
        return self._polygon.area

    @property
    def centroid(self) -> tuple[float, float]:
        """Return polygon centroid as (x, y)."""
        c = self._polygon.centroid
        return (c.x, c.y)

    @property
    def n_vertices(self) -> int:
        """Return number of vertices."""
        return len(self.coords)

    @property
    def shapely(self) -> ShapelyPolygon:
        """Return underlying shapely Polygon object."""
        return self._polygon

    def simplify(self, tolerance: float) -> Polygon:
        """Return simplified polygon using Douglas-Peucker algorithm.

        Args:
            tolerance: Maximum distance from original geometry.

        Returns:
            New simplified Polygon instance.
        """
        simplified = self._polygon.simplify(tolerance, preserve_topology=True)
        return Polygon.from_shapely(simplified, crs=self._crs)

    def buffer(self, distance: float) -> Polygon:
        """Return buffered polygon.

        Args:
            distance: Buffer distance (positive = expand, negative = shrink).

        Returns:
            New buffered Polygon instance.
        """
        buffered = self._polygon.buffer(distance)
        return Polygon.from_shapely(buffered, crs=self._crs)

    def __repr__(self) -> str:
        return (
            f"Polygon(n_vertices={self.n_vertices}, "
            f"bounds={self.bounds}, crs={self._crs})"
        )
