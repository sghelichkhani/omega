"""Tests for omega.geometry.crs.LocalFrame."""

import numpy as np
import pytest

from omega.geometry.crs import LocalFrame

# The Lower Murrumbidgee demo's frame.
MURRUMBIDGEE = LocalFrame(143.01, -35.76, 91_800.0, 110_170.0)


class TestLocalFrame:
    def test_origin_maps_to_zero(self):
        x, y = MURRUMBIDGEE.to_local(143.01, -35.76)
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)

    def test_known_offset(self):
        # One degree east / one degree north -> the m/deg factors.
        x, y = MURRUMBIDGEE.to_local(144.01, -34.76)
        assert x == pytest.approx(91_800.0)
        assert y == pytest.approx(110_170.0)

    def test_round_trip_scalar(self):
        lon, lat = 145.3, -34.9
        x, y = MURRUMBIDGEE.to_local(lon, lat)
        back_lon, back_lat = MURRUMBIDGEE.to_lonlat(x, y)
        assert back_lon == pytest.approx(lon)
        assert back_lat == pytest.approx(lat)

    def test_round_trip_array(self):
        lon = np.array([143.0, 144.5, 145.9])
        lat = np.array([-35.0, -34.2, -33.8])
        x, y = MURRUMBIDGEE.to_local(lon, lat)
        back_lon, back_lat = MURRUMBIDGEE.to_lonlat(x, y)
        np.testing.assert_allclose(back_lon, lon)
        np.testing.assert_allclose(back_lat, lat)

    def test_to_local_returns_arrays(self):
        x, y = MURRUMBIDGEE.to_local([143.0, 144.0], [-35.0, -34.0])
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
        assert x.shape == (2,)

    def test_bbox_to_lonlat(self):
        # A local box maps to a lon/lat box; corners round-trip.
        bbox = MURRUMBIDGEE.bbox_to_lonlat(0.0, 0.0, 91_800.0, 110_170.0)
        lon_min, lat_min, lon_max, lat_max = bbox
        assert lon_min == pytest.approx(143.01)
        assert lat_min == pytest.approx(-35.76)
        assert lon_max == pytest.approx(144.01)
        assert lat_max == pytest.approx(-34.76)

    def test_bbox_returns_plain_floats(self):
        bbox = MURRUMBIDGEE.bbox_to_lonlat(0.0, 0.0, 1000.0, 2000.0)
        assert all(isinstance(v, float) for v in bbox)

    @pytest.mark.parametrize(
        "lon_factor, lat_factor",
        [(0.0, 110_170.0), (91_800.0, 0.0), (-1.0, 110_170.0)],
    )
    def test_non_positive_factors_raise(self, lon_factor, lat_factor):
        with pytest.raises(ValueError, match="positive"):
            LocalFrame(143.01, -35.76, lon_factor, lat_factor)

    def test_southern_hemisphere_negative_offset(self):
        # A point south-west of the origin -> negative local coords, round-tripped.
        lon, lat = 142.0, -36.5
        x, y = MURRUMBIDGEE.to_local(lon, lat)
        assert x < 0 and y < 0
        back_lon, back_lat = MURRUMBIDGEE.to_lonlat(x, y)
        assert back_lon == pytest.approx(lon)
        assert back_lat == pytest.approx(lat)

    def test_matches_demo_domain_bbox(self):
        # Regression guard for the Item 5 migration: LocalFrame must reproduce the
        # demo's domain_lonlat_bbox for the real (offset) DOMAIN polygon exactly.
        active = [
            (0, 35_000), (140_000, 0), (280_000, 0), (280_000, 68_000),
            (201_000, 130_000), (121_000, 130_000), (0, 100_000),
        ]
        offset = (45_000, 73_800)
        domain = [(x + offset[0], y + offset[1]) for x, y in active]
        xs = [p[0] for p in domain]
        ys = [p[1] for p in domain]

        # The demo's domain_lonlat_bbox computation, inline.
        demo_min = MURRUMBIDGEE.to_lonlat(min(xs), min(ys))
        demo_max = MURRUMBIDGEE.to_lonlat(max(xs), max(ys))
        demo_bbox = (
            float(demo_min[0]), float(demo_min[1]),
            float(demo_max[0]), float(demo_max[1]),
        )

        got = MURRUMBIDGEE.bbox_to_lonlat(min(xs), min(ys), max(xs), max(ys))
        assert got == pytest.approx(demo_bbox)

    def test_repr_is_informative(self):
        assert "LocalFrame" in repr(MURRUMBIDGEE)
        assert "143.01" in repr(MURRUMBIDGEE)
