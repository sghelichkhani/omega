"""Local metric frame for placing a domain in projected metres.

:class:`LocalFrame` is a small affine value object that maps longitude/latitude to
a local easting/northing in metres and back, using a fixed origin and a constant
number of metres per degree in each direction. It is the one georeference a demo
or model shares so that every data source (a DEM request box, scattered borehole
positions, the mesh polygon) lands in the same metric frame.

This is an *approximate equirectangular linearisation*, not a map projection: the
metres-per-degree factors are constants fitted around the domain, so it is only
accurate over a region small enough that the Earth's curvature and the cosine
narrowing of longitude can be treated as constant. It is deliberately distinct
from a named CRS (e.g. an ``"EPSG:..."`` string a :class:`Polygon` may carry):
that names a true projection, this is a local tangent-plane approximation.
"""

from __future__ import annotations

import numpy as np


class LocalFrame:
    """Affine lon/lat <-> local-metre frame about a fixed origin.

    Args:
        origin_lon: Longitude (degrees) mapped to local x = 0.
        origin_lat: Latitude (degrees) mapped to local y = 0.
        m_per_deg_lon: Metres per degree of longitude (positive).
        m_per_deg_lat: Metres per degree of latitude (positive).

    Raises:
        ValueError: If either metres-per-degree factor is not positive.
    """

    def __init__(
        self,
        origin_lon: float,
        origin_lat: float,
        m_per_deg_lon: float,
        m_per_deg_lat: float,
    ):
        if m_per_deg_lon <= 0 or m_per_deg_lat <= 0:
            raise ValueError("metres-per-degree factors must be positive")
        self.origin_lon = float(origin_lon)
        self.origin_lat = float(origin_lat)
        self.m_per_deg_lon = float(m_per_deg_lon)
        self.m_per_deg_lat = float(m_per_deg_lat)

    def to_local(self, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        """Convert lon/lat (degrees) to local easting/northing (metres).

        Accepts scalars or arrays; returns arrays shaped like the inputs (a scalar
        comes back as a 0-d array). Use :meth:`bbox_to_lonlat` for plain floats.
        """
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        x = (lon - self.origin_lon) * self.m_per_deg_lon
        y = (lat - self.origin_lat) * self.m_per_deg_lat
        return x, y

    def to_lonlat(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Convert local easting/northing (metres) to lon/lat (degrees)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        lon = self.origin_lon + x / self.m_per_deg_lon
        lat = self.origin_lat + y / self.m_per_deg_lat
        return lon, lat

    def bbox_to_lonlat(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
    ) -> tuple[float, float, float, float]:
        """Convert a local-metre bounding box to a lon/lat bounding box.

        Returns ``(lon_min, lat_min, lon_max, lat_max)``. Because the transform is
        axis-aligned and monotonic, the local corners map straight to lon/lat
        corners.
        """
        lon_min, lat_min = self.to_lonlat(x_min, y_min)
        lon_max, lat_max = self.to_lonlat(x_max, y_max)
        return float(lon_min), float(lat_min), float(lon_max), float(lat_max)

    def __repr__(self) -> str:
        return (
            f"LocalFrame(origin=({self.origin_lon}, {self.origin_lat}), "
            f"m_per_deg=({self.m_per_deg_lon}, {self.m_per_deg_lat}))"
        )
