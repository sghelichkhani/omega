"""Tests for omega.fields.surfaces (the GaussianKernelSurface primitive)."""

import numpy as np
import pytest

from omega.exceptions import InterpolationError
from omega.fields.surfaces import (
    GaussianKernelSurface,
    Surface,
    clamp_monotonic,
)


def _grid(n=11, span=100.0):
    x = np.linspace(0, span, n)
    xx, yy = np.meshgrid(x, x)
    return np.column_stack([xx.ravel(), yy.ravel()])


class TestGaussianKernelSurface:
    def test_is_a_surface_and_callable(self):
        coords = _grid()
        surf = GaussianKernelSurface(coords, np.ones(len(coords)), sigma=10.0)
        assert isinstance(surf, Surface)
        out = surf(np.array([[50.0, 50.0]]))
        assert out.shape == (1,)

    def test_constant_field_reproduced_exactly(self):
        # A weighted mean of identical values is that value, for any sigma/k.
        coords = _grid()
        surf = GaussianKernelSurface(coords, np.full(len(coords), 7.5), sigma=5.0)
        out = surf(_grid(n=7, span=100.0))
        assert np.allclose(out, 7.5)

    def test_reproduces_source_values_with_small_sigma(self):
        # With sigma well below the spacing, the nearest pick dominates, so the
        # surface passes close to each pick value.
        coords = _grid(n=11, span=100.0)
        values = 100 + 0.1 * coords[:, 0] + 0.05 * coords[:, 1]
        surf = GaussianKernelSurface(coords, values, sigma=1.5, k=8)
        out = surf(coords)
        assert np.allclose(out, values, atol=1.0)

    def test_seamless_everywhere_no_nan_far_outside_data(self):
        coords = _grid()
        surf = GaussianKernelSurface(coords, coords[:, 0], sigma=10.0)
        far = np.array([[1e6, 1e6], [-1e6, -1e6], [50.0, 50.0]])
        out = surf(far)
        assert np.all(np.isfinite(out))

    def test_deterministic(self):
        coords = _grid()
        surf = GaussianKernelSurface(coords, coords[:, 1], sigma=10.0)
        q = _grid(n=5)
        assert np.array_equal(surf(q), surf(q))

    def test_density_normalisation_declusters(self):
        # A cluster of 50 co-located bores (value 0) and one lone bore (value 10),
        # queried at the midpoint. A plain weighted mean would be dragged to ~0.2
        # by the cluster's 50 votes; density-normalisation collapses the cluster to
        # ~one effective vote, so the symmetric midpoint returns ~5.
        cluster = np.zeros((50, 2))
        coords = np.vstack([cluster, [100.0, 0.0]])
        values = np.concatenate([np.zeros(50), [10.0]])
        surf = GaussianKernelSurface(coords, values, sigma=10.0, k=51)
        out = surf(np.array([[50.0, 0.0]]))
        assert abs(out[0] - 5.0) < 0.3
        # And it is nowhere near the un-declustered ~0.2.
        assert out[0] > 3.0

    def test_k_clamped_to_pick_count_no_crash(self):
        # A sparse formation with fewer picks than k must not crash (cKDTree would
        # otherwise pad with out-of-range indices / inf distances).
        coords = np.array([[0.0, 0.0], [10.0, 0.0]])
        surf = GaussianKernelSurface(coords, np.array([0.0, 10.0]), sigma=5.0, k=100)
        out = surf(np.array([[5.0, 0.0]]))
        assert np.isfinite(out[0])

    def test_k_equals_one_returns_nearest(self):
        coords = np.array([[0.0, 0.0], [100.0, 0.0]])
        surf = GaussianKernelSurface(coords, np.array([0.0, 10.0]), sigma=5.0, k=1)
        assert surf(np.array([[1.0, 0.0]]))[0] == 0.0
        assert surf(np.array([[99.0, 0.0]]))[0] == 10.0

    def test_warns_at_construction_when_k_truncates_contributors(self):
        # Dense picks, large sigma, tiny k: the k-th nearest pick sits well within
        # 3 sigma, so real contributors are truncated. The warning is raised once
        # at construction (deterministic, MPI-rank-independent), not per call.
        coords = _grid(n=21, span=100.0)
        with pytest.warns(UserWarning, match="truncates contributors"):
            GaussianKernelSurface(coords, coords[:, 0], sigma=50.0, k=4)

    def test_cluster_larger_than_k_warns(self):
        # 200 co-located bores + one lone control point, k well below the cluster
        # size. Density-normalisation cannot pull in the lone point the k-NN never
        # selects, so the surface must warn rather than silently swallow it.
        coords = np.vstack([np.zeros((200, 2)), [500.0, 0.0]])
        values = np.concatenate([np.zeros(200), [10.0]])
        with pytest.warns(UserWarning, match="truncates contributors"):
            GaussianKernelSurface(coords, values, sigma=20.0, k=20)

    def test_no_warning_when_k_covers_three_sigma(self):
        import warnings

        coords = _grid(n=11, span=100.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            surf = GaussianKernelSurface(coords, coords[:, 0], sigma=2.0, k=8)
            surf(_grid(n=5, span=100.0))

    def test_nan_values_rejected_at_construction(self):
        coords = _grid()
        values = coords[:, 0].copy()
        values[3] = np.nan
        with pytest.raises(InterpolationError):
            GaussianKernelSurface(coords, values, sigma=10.0)

    def test_large_query_crosses_chunk_boundary(self):
        # > 50k query points exercises the internal chunk loop; the first points
        # must match a standalone small query (per-point independence).
        coords = _grid(n=12, span=100.0)
        surf = GaussianKernelSurface(coords, coords[:, 0], sigma=2.0, k=6)
        big = _grid(n=250, span=100.0)  # 62,500 points
        out = surf(big)
        assert out.shape == (len(big),)
        assert np.all(np.isfinite(out))
        assert np.allclose(out[:5], surf(big[:5]))

    def test_query_splitting_is_consistent(self):
        # Each query point is evaluated independently, so splitting a query and
        # concatenating must match a single pass (guards the chunked evaluation).
        coords = _grid(n=15, span=100.0)
        values = np.sin(coords[:, 0] / 20.0)
        surf = GaussianKernelSurface(coords, values, sigma=8.0, k=12)
        q = _grid(n=40, span=100.0)
        ref = surf(q)
        half = len(q) // 2
        out = np.concatenate([surf(q[:half]), surf(q[half:])])
        assert np.allclose(out, ref)

    def test_rho_chunking_matches_unchunked(self, monkeypatch):
        # The density (rho) self-query is chunked at construction; forcing a small
        # chunk must not change results vs a single-pass build.
        import omega.fields.surfaces as surfaces_mod

        coords = _grid(n=20, span=100.0)
        values = np.sin(coords[:, 0] / 10.0)
        ref = GaussianKernelSurface(coords, values, sigma=8.0, k=12)
        monkeypatch.setattr(surfaces_mod, "_RHO_CHUNK", 37)
        chunked = GaussianKernelSurface(coords, values, sigma=8.0, k=12)
        q = _grid(n=10, span=100.0)
        np.testing.assert_allclose(ref(q), chunked(q))

    def test_callable_sigma_reserved_but_not_implemented(self):
        coords = _grid()
        with pytest.raises(NotImplementedError):
            GaussianKernelSurface(coords, coords[:, 0], sigma=lambda c: np.ones(len(c)))

    @pytest.mark.parametrize(
        "coords, values, sigma",
        [
            (np.zeros((0, 2)), np.zeros(0), 1.0),  # no picks
            (np.zeros((3, 3)), np.zeros(3), 1.0),  # wrong coord shape
            (np.zeros((3, 2)), np.zeros(4), 1.0),  # mismatched lengths
            (np.zeros((3, 2)), np.zeros(3), 0.0),  # non-positive sigma
        ],
    )
    def test_input_validation(self, coords, values, sigma):
        with pytest.raises(InterpolationError):
            GaussianKernelSurface(coords, values, sigma=sigma)


class TestCombinators:
    def test_arithmetic_top_minus_thickness(self):
        coords = _grid()
        top = GaussianKernelSurface(coords, np.full(len(coords), 100.0), sigma=10.0)
        thickness = GaussianKernelSurface(coords, np.full(len(coords), 30.0), sigma=10.0)
        bedrock = top - thickness
        assert isinstance(bedrock, Surface)
        assert np.allclose(bedrock(_grid(n=5)), 70.0)

    def test_scalar_arithmetic_both_sides(self):
        coords = _grid()
        s = GaussianKernelSurface(coords, np.full(len(coords), 10.0), sigma=10.0)
        q = _grid(n=5)
        assert np.allclose((s + 5)(q), 15.0)
        assert np.allclose((100 - s)(q), 90.0)
        assert np.allclose((2 * s)(q), 20.0)

    def test_clamp_min_with_scalar_floor(self):
        coords = _grid()
        thin = GaussianKernelSurface(coords, np.full(len(coords), 1.0), sigma=10.0)
        floored = thin.clamp_min(5.0)
        assert np.allclose(floored(_grid(n=5)), 5.0)

    def test_clamp_monotonic_non_increasing_elevations(self):
        coords = _grid()
        # Deliberately inverted: "lower" layer fitted above the "upper" one.
        upper = GaussianKernelSurface(coords, np.full(len(coords), 0.0), sigma=10.0)
        lower = GaussianKernelSurface(coords, np.full(len(coords), 50.0), sigma=10.0)
        c_upper, c_lower = clamp_monotonic([upper, lower], increasing=False)
        q = _grid(n=5)
        assert np.allclose(c_upper(q), 0.0)
        assert np.allclose(c_lower(q), 0.0)  # clamped down to the one above

    def test_clamp_monotonic_increasing_depths(self):
        coords = _grid()
        shallow = GaussianKernelSurface(coords, np.full(len(coords), 80.0), sigma=10.0)
        deep = GaussianKernelSurface(coords, np.full(len(coords), 20.0), sigma=10.0)
        c_shallow, c_deep = clamp_monotonic([shallow, deep], increasing=True)
        q = _grid(n=5)
        assert np.allclose(c_shallow(q), 80.0)
        assert np.allclose(c_deep(q), 80.0)  # clamped up to the one above

    def test_clamp_monotonic_three_layer_fold(self):
        coords = _grid()
        a = GaussianKernelSurface(coords, np.full(len(coords), 10.0), sigma=10.0)
        b = GaussianKernelSurface(coords, np.full(len(coords), 30.0), sigma=10.0)
        c = GaussianKernelSurface(coords, np.full(len(coords), 5.0), sigma=10.0)
        ca, cb, cc = clamp_monotonic([a, b, c], increasing=True)
        q = _grid(n=4)
        assert np.allclose(ca(q), 10.0)
        assert np.allclose(cb(q), 30.0)
        assert np.allclose(cc(q), 30.0)  # 5 clamped up past b's 30

    def test_clamp_monotonic_empty_raises(self):
        with pytest.raises(ValueError):
            clamp_monotonic([])
