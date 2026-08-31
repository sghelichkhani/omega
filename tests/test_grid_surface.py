"""Tests for omega.fields.surfaces.GridSurface.

The point of GridSurface is that it is the *same* interpolation the array-based
build_mesh_hierarchy used before the Surface refactor, wrapped in the Surface
contract. So the load-bearing test here is bit-exactness against
scipy.interpolate.griddata, not approximate agreement: a mesh rebuilt through a
GridSurface must reproduce a pre-refactor mesh exactly, or the runs it underpins
have to be repeated.
"""

import numpy as np
import pytest
from scipy.interpolate import griddata

from omega.exceptions import InterpolationError
from omega.fields.surfaces import GaussianKernelSurface, GridSurface, Surface


def _grid(n=21, span=1000.0):
    """A regular lattice, standing in for a DEM or a modelled grid."""
    x = np.linspace(0, span, n)
    xx, yy = np.meshgrid(x, x)
    return np.column_stack([xx.ravel(), yy.ravel()])


def _terrain(coords):
    """A smooth synthetic relief with a ridge, so flattening is detectable."""
    x, y = coords[:, 0], coords[:, 1]
    return 100.0 + 0.02 * x + 30.0 * np.exp(-((x - 500.0) ** 2) / (2 * 120.0**2)) + 0.005 * y


class TestGridSurfaceContract:
    def test_is_a_surface_and_returns_one_value_per_point(self):
        coords = _grid()
        surf = GridSurface(coords, _terrain(coords))
        assert isinstance(surf, Surface)
        out = surf(np.array([[100.0, 100.0], [200.0, 300.0]]))
        assert out.shape == (2,)

    def test_accepts_3d_query_points_and_uses_only_xy(self):
        # Mesh node arrays arrive as (m, 3); the surface is a function of (x, y).
        coords = _grid()
        surf = GridSurface(coords, _terrain(coords))
        q = np.array([[250.0, 250.0]])
        q3 = np.array([[250.0, 250.0, -42.0]])
        assert np.array_equal(surf(q), surf(q3))

    def test_reproduces_source_values_at_source_points(self):
        # Linear interpolation is exact at the nodes of its own triangulation.
        coords = _grid()
        values = _terrain(coords)
        surf = GridSurface(coords, values)
        assert np.allclose(surf(coords), values)

    def test_reproduces_an_affine_field_exactly(self):
        # Piecewise-linear interpolation is exact on an affine field. This is the
        # property GaussianKernelSurface cannot have (a weighted mean is bounded
        # by its inputs), and the reason a dense grid should not be smoothed.
        coords = _grid()
        values = 5.0 + 0.3 * coords[:, 0] - 0.2 * coords[:, 1]
        surf = GridSurface(coords, values)
        q = _grid(n=13, span=1000.0)
        assert np.allclose(surf(q), 5.0 + 0.3 * q[:, 0] - 0.2 * q[:, 1])

    def test_composes_with_surface_arithmetic(self):
        # The extrusion builds bedrock as top - thickness, so a GridSurface has to
        # survive the combinators like any other surface.
        coords = _grid()
        top = GridSurface(coords, _terrain(coords))
        thickness = GridSurface(coords, np.full(len(coords), 50.0))
        bed = top - thickness
        q = np.array([[300.0, 400.0]])
        assert np.allclose(bed(q), top(q) - 50.0)

    def test_clamp_min_applies(self):
        coords = _grid()
        surf = GridSurface(coords, np.full(len(coords), 2.0)).clamp_min(5.0)
        assert np.allclose(surf(np.array([[100.0, 100.0]])), 5.0)


class TestBitExactAgainstGriddata:
    """GridSurface must equal griddata(linear) + nearest backfill, exactly."""

    def test_matches_griddata_inside_the_hull(self):
        rng = np.random.default_rng(0)
        coords = _grid()
        values = _terrain(coords)
        q = rng.uniform(50.0, 950.0, (500, 2))

        expected = griddata(coords, values, q, method="linear")
        assert not np.any(np.isnan(expected))  # the sample really is inside
        assert np.array_equal(GridSurface(coords, values)(q), expected)

    def test_matches_griddata_with_the_nearest_backfill(self):
        # This is the exact sequence the pre-refactor builder ran: linear, then
        # refill the convex-hull NaNs from the nearest source point.
        rng = np.random.default_rng(1)
        coords = _grid()
        values = _terrain(coords)
        q = rng.uniform(-300.0, 1300.0, (800, 2))

        expected = griddata(coords, values, q, method="linear")
        outside = np.isnan(expected)
        assert np.any(outside)  # the sample really does escape the hull
        expected[outside] = griddata(coords, values, q[outside], method="nearest")

        assert np.array_equal(GridSurface(coords, values)(q), expected)

    def test_matches_griddata_on_scattered_sources_too(self):
        # Nothing in the implementation assumes a lattice; the legacy path was
        # given whatever arrays the caller had.
        rng = np.random.default_rng(2)
        coords = rng.uniform(0.0, 1000.0, (400, 2))
        values = _terrain(coords)
        q = rng.uniform(0.0, 1000.0, (300, 2))

        expected = griddata(coords, values, q, method="linear")
        outside = np.isnan(expected)
        expected[outside] = griddata(coords, values, q[outside], method="nearest")

        assert np.array_equal(GridSurface(coords, values)(q), expected)


class TestFillBehaviour:
    def test_nan_fill_leaves_the_outside_visible(self):
        # A caller that would rather detect a mesh escaping its terrain data than
        # silently get a flat step can ask for the NaNs.
        coords = _grid()
        surf = GridSurface(coords, _terrain(coords), fill="nan")
        out = surf(np.array([[-500.0, -500.0], [500.0, 500.0]]))
        assert np.isnan(out[0])
        assert np.isfinite(out[1])

    def test_nearest_fill_is_piecewise_constant_outside(self):
        # Documented limitation, asserted so it cannot change unnoticed: outside
        # the hull the value is the nearest source value, not an extrapolation.
        coords = _grid()
        values = _terrain(coords)
        surf = GridSurface(coords, values)
        corner = values[np.argmin(np.sum((coords - np.array([0.0, 0.0])) ** 2, axis=1))]
        far = surf(np.array([[-5000.0, -5000.0], [-9000.0, -9000.0]]))
        assert np.allclose(far, corner)

    def test_rejects_unknown_fill(self):
        coords = _grid()
        with pytest.raises(InterpolationError, match="fill must be"):
            GridSurface(coords, _terrain(coords), fill="linear")


class TestDeterminismAndPurity:
    """The MPI contract: pure, deterministic, replicated, no collectives."""

    def test_repeated_evaluation_is_identical(self):
        coords = _grid()
        surf = GridSurface(coords, _terrain(coords))
        q = _grid(n=9, span=1000.0)
        assert np.array_equal(surf(q), surf(q))

    def test_two_instances_from_the_same_data_agree_bitwise(self):
        # Stands in for two MPI ranks each constructing the surface locally.
        coords = _grid()
        values = _terrain(coords)
        q = _grid(n=9, span=1000.0)
        assert np.array_equal(GridSurface(coords, values)(q), GridSurface(coords, values)(q))

    def test_query_splitting_is_consistent(self):
        # Ranks evaluate different subsets of the nodes; a point's value must not
        # depend on which other points were in the same call.
        rng = np.random.default_rng(3)
        coords = _grid()
        surf = GridSurface(coords, _terrain(coords))
        q = rng.uniform(0.0, 1000.0, (200, 2))
        whole = surf(q)
        split = np.concatenate([surf(q[:71]), surf(q[71:])])
        assert np.array_equal(whole, split)

    def test_triangulation_is_built_lazily(self):
        # Constructing a surface no rank evaluates must not pay for the Delaunay.
        coords = _grid()
        surf = GridSurface(coords, _terrain(coords))
        assert surf._linear is None
        surf(np.array([[500.0, 500.0]]))
        assert surf._linear is not None


class TestValidation:
    def test_rejects_wrong_coord_shape(self):
        with pytest.raises(InterpolationError, match="shape"):
            GridSurface(np.zeros((10, 3)), np.zeros(10))

    def test_rejects_length_mismatch(self):
        with pytest.raises(InterpolationError, match="match coords"):
            GridSurface(_grid(), np.zeros(3))

    def test_rejects_too_few_points(self):
        with pytest.raises(InterpolationError, match="at least three"):
            GridSurface(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([1.0, 2.0]))

    def test_rejects_nan_values(self):
        coords = _grid()
        values = _terrain(coords)
        values[5] = np.nan
        with pytest.raises(InterpolationError, match="finite"):
            GridSurface(coords, values)

    def test_rejects_nan_coords(self):
        coords = _grid()
        coords[5, 0] = np.nan
        with pytest.raises(InterpolationError, match="finite"):
            GridSurface(coords, _terrain(coords))

    def test_rejects_bad_query_shape(self):
        surf = GridSurface(_grid(), _terrain(_grid()))
        with pytest.raises(InterpolationError, match="xy must have shape"):
            surf(np.array([1.0, 2.0]))


class TestWhyNotTheKernelOnDenseGrids:
    def test_kernel_regression_flattens_relief_that_grid_surface_keeps(self):
        # The justification for keeping two primitives. On a dense lattice the
        # linear surface reproduces the ridge crest, while a kernel mean at a
        # bandwidth comparable to the mesh scale clips it -- a weighted mean is
        # bounded by its inputs and cannot reach a local maximum.
        coords = _grid(n=41, span=1000.0)
        values = _terrain(coords)
        crest = np.array([[500.0, 500.0]])
        exact = _terrain(crest)[0]

        linear = GridSurface(coords, values)(crest)[0]
        smoothed = GaussianKernelSurface(coords, values, sigma=150.0, k=200)(crest)[0]

        assert abs(linear - exact) < 1e-9
        assert smoothed < exact - 1.0
