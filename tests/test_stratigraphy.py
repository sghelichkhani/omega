"""Tests for omega.fields.stratigraphy and the mesh_fields seam."""

import importlib.util

import numpy as np
import pytest

from omega.fields.stratigraphy import IntervalObservations, LayerModel
from omega.fields.surfaces import Surface


def _two_layer_observations():
    """Two flat layers over a 4-borehole grid.

    'sand' spans elevation [10, 20]; 'clay' spans [0, 10]. Four boreholes at the
    corners of a 100 m square, each logging both layers.
    """
    xs = [0.0, 100.0, 0.0, 100.0]
    ys = [0.0, 0.0, 100.0, 100.0]
    x, y, top, bottom, label = [], [], [], [], []
    for px, py in zip(xs, ys, strict=True):
        x += [px, px]
        y += [py, py]
        top += [20.0, 10.0]
        bottom += [10.0, 0.0]
        label += ["sand", "clay"]
    return IntervalObservations(x, y, top, bottom, label)


class TestIntervalObservations:
    def test_basic_construction(self):
        obs = _two_layer_observations()
        assert obs.n_points == 8
        assert obs.labels == ["sand", "clay"]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            IntervalObservations([0, 1], [0], [1], [0], ["a"])

    def test_inverted_interval_raises(self):
        with pytest.raises(ValueError, match="inverted"):
            IntervalObservations([0], [0], [0.0], [10.0], ["a"])

    def test_non_finite_raises(self):
        with pytest.raises(ValueError, match="finite"):
            IntervalObservations([0], [0], [np.nan], [0.0], ["a"])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            IntervalObservations([], [], [], [], [])

    def test_subset(self):
        obs = _two_layer_observations()
        sand = obs.subset("sand")
        assert sand.n_points == 4
        assert set(sand.labels) == {"sand"}

    def test_subset_unknown_raises(self):
        obs = _two_layer_observations()
        with pytest.raises(ValueError, match="no observations"):
            obs.subset("granite")

    def test_labels_normalised_to_str(self):
        # Non-string labels are coerced so labels/subset/mapping stay consistent.
        obs = IntervalObservations([0, 1], [0, 1], [1.0, 1.0], [0.0, 0.0], [1, 2])
        assert obs.labels == ["1", "2"]
        assert obs.subset("1").n_points == 1


class TestLayerModel:
    def test_empty_order_raises(self):
        obs = _two_layer_observations()
        with pytest.raises(ValueError, match="at least one layer"):
            LayerModel(obs, [], sigma=20.0)

    def test_unknown_layer_raises(self):
        obs = _two_layer_observations()
        with pytest.raises(ValueError, match="no observations"):
            LayerModel(obs, ["sand", "granite"], sigma=20.0)

    def test_boundary_elevation_flat(self):
        model = LayerModel(_two_layer_observations(), ["sand", "clay"], sigma=20.0)
        query = np.array([[50.0, 50.0], [25.0, 75.0]])
        np.testing.assert_allclose(model.boundary_elevation("sand", "top", query), 20.0)
        np.testing.assert_allclose(model.boundary_elevation("sand", "bottom", query), 10.0)
        np.testing.assert_allclose(model.boundary_elevation("clay", "bottom", query), 0.0)

    def test_boundary_bad_side_raises(self):
        model = LayerModel(_two_layer_observations(), ["sand", "clay"], sigma=20.0)
        with pytest.raises(ValueError, match="side"):
            model.boundary_elevation("sand", "middle", np.array([[0.0, 0.0]]))

    def test_classify_inside_bands(self):
        model = LayerModel(_two_layer_observations(), ["sand", "clay"], sigma=20.0)
        pts = np.array(
            [
                [50.0, 50.0, 15.0],  # in sand  -> 0
                [50.0, 50.0, 5.0],   # in clay  -> 1
            ]
        )
        np.testing.assert_array_equal(model.classify(pts), [0, 1])

    def test_classify_gap_takes_nearest(self):
        model = LayerModel(_two_layer_observations(), ["sand", "clay"], sigma=20.0)
        # Above the whole stack -> nearest is sand (0); below it -> clay (1).
        pts = np.array([[50.0, 50.0, 100.0], [50.0, 50.0, -50.0]])
        np.testing.assert_array_equal(model.classify(pts), [0, 1])

    def test_value_field_mapping(self):
        model = LayerModel(_two_layer_observations(), ["sand", "clay"], sigma=20.0)
        pts = np.array([[50.0, 50.0, 15.0], [50.0, 50.0, 5.0]])
        values = model.value_field(pts, {"sand": 1e-3, "clay": 1e-6})
        np.testing.assert_allclose(values, [1e-3, 1e-6])

    def test_value_field_default(self):
        model = LayerModel(_two_layer_observations(), ["sand", "clay"], sigma=20.0)
        pts = np.array([[50.0, 50.0, 15.0]])
        values = model.value_field(pts, {"clay": 1e-6}, default=42.0)
        np.testing.assert_allclose(values, [42.0])

    def test_classify_bad_shape_raises(self):
        model = LayerModel(_two_layer_observations(), ["sand", "clay"], sigma=20.0)
        with pytest.raises(ValueError, match="m, 3"):
            model.classify(np.array([[0.0, 0.0]]))

    def test_single_borehole_layer_uses_nearest(self):
        # One borehole, one layer: every query takes that single pick.
        obs = IntervalObservations([10.0], [20.0], [5.0], [0.0], ["z"])
        model = LayerModel(obs, ["z"], sigma=20.0)
        out = model.boundary_elevation("z", "top", np.array([[0.0, 0.0], [99.0, 99.0]]))
        np.testing.assert_allclose(out, 5.0)
        assert model.classify(np.array([[0.0, 0.0, 2.0]]))[0] == 0

    def test_enforce_monotonic_resolves_crossing(self):
        # 'lower' is fitted to span 0..12, poking above 'upper' (5..10).
        x = [0.0, 100.0, 0.0, 100.0]
        y = [0.0, 0.0, 100.0, 100.0]
        obs = IntervalObservations(
            x * 2, y * 2,
            [10.0] * 4 + [12.0] * 4,
            [5.0] * 4 + [0.0] * 4,
            ["upper"] * 4 + ["lower"] * 4,
        )
        pt = np.array([[50.0, 50.0, 11.0]])  # above upper's top, inside raw lower

        monotonic = LayerModel(obs, ["upper", "lower"], sigma=20.0, enforce_monotonic=True)
        assert monotonic.classify(pt)[0] == 0  # clamped: nearest is upper

        raw = LayerModel(obs, ["upper", "lower"], sigma=20.0, enforce_monotonic=False)
        assert raw.classify(pt)[0] == 1  # unclamped: wrongly inside lower

    def test_dipping_layer_symmetric_midpoint_only(self):
        # A single layer whose top dips linearly from 30 (x=0) to 10 (x=100). A
        # Gaussian weighted mean is bounded by its inputs and does NOT reproduce
        # the linear dip -- it under-extrapolates the slope (documented limitation,
        # see surfaces.py). Only the centre is exact, by four-corner symmetry.
        x = [0.0, 100.0, 0.0, 100.0]
        y = [0.0, 0.0, 100.0, 100.0]
        top = [30.0, 10.0, 30.0, 10.0]
        bottom = [0.0, 0.0, 0.0, 0.0]
        obs = IntervalObservations(x, y, top, bottom, ["z"] * 4)
        model = LayerModel(obs, ["z"], sigma=20.0)

        mid = model.boundary_elevation("z", "top", np.array([[50.0, 50.0]]))
        np.testing.assert_allclose(mid, 20.0, atol=1e-9)  # exact only by symmetry

        # Off-centre, the true linear dip would be 25 at x=25; the Gaussian mean
        # stays near the local corner value instead. Assert it does NOT track it.
        off = model.boundary_elevation("z", "top", np.array([[25.0, 50.0]]))
        assert abs(off[0] - 25.0) > 1.0


class _Const(Surface):
    """A flat surface at a constant elevation (test double)."""

    def __init__(self, value):
        self._value = float(value)

    def __call__(self, xy):
        return np.full(len(xy), self._value)


class _PlaneX(Surface):
    """A surface that dips linearly in x: value = intercept + slope * x."""

    def __init__(self, intercept, slope):
        self._intercept = float(intercept)
        self._slope = float(slope)

    def __call__(self, xy):
        return self._intercept + self._slope * np.asarray(xy)[:, 0]


class TestLayerModelFromDepths:
    def _flat_model(self, enforce_monotonic=True, open_bottom=True):
        # Top at 100; bases at depths 10, 30, 60 -> elevation bands
        # shep [90,100], cal [70,90], renmark [40,70] (or [-inf,70] if open).
        return LayerModel.from_depths(
            _Const(100.0),
            {"shep": _Const(10.0), "cal": _Const(30.0), "renmark": _Const(60.0)},
            enforce_monotonic=enforce_monotonic,
            open_bottom=open_bottom,
        )

    def test_layer_order_follows_insertion(self):
        model = self._flat_model()
        assert model.layer_order == ["shep", "cal", "renmark"]

    def test_boundaries_are_depths_below_top(self):
        model = self._flat_model(open_bottom=False)  # closed to check the base
        q = np.array([[5.0, 5.0], [50.0, 50.0]])
        np.testing.assert_allclose(model.boundary_elevation("shep", "top", q), 100.0)
        np.testing.assert_allclose(model.boundary_elevation("shep", "bottom", q), 90.0)
        np.testing.assert_allclose(model.boundary_elevation("cal", "top", q), 90.0)
        np.testing.assert_allclose(model.boundary_elevation("cal", "bottom", q), 70.0)
        np.testing.assert_allclose(model.boundary_elevation("renmark", "bottom", q), 40.0)

    def test_classify_by_depth(self):
        model = self._flat_model()
        pts = np.array(
            [
                [0.0, 0.0, 95.0],  # 5 m down  -> shep (0)
                [0.0, 0.0, 80.0],  # 20 m down -> cal (1)
                [0.0, 0.0, 50.0],  # 50 m down -> renmark (2)
            ]
        )
        np.testing.assert_array_equal(model.classify(pts), [0, 1, 2])

    def test_open_bottom_extends_deepest_layer_downward(self):
        # With open_bottom (default) any point below the stack is the deepest
        # layer, however deep -- the renmark base depth (60) does not cap it.
        model = self._flat_model()
        pts = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, -1000.0], [0.0, 0.0, 150.0]])
        # Far below -> renmark (2); above the top -> shep (0).
        np.testing.assert_array_equal(model.classify(pts), [2, 2, 0])

    def test_closed_bottom_falls_to_nearest_below_stack(self):
        closed = LayerModel.from_depths(
            _Const(100.0),
            {"shep": _Const(10.0), "cal": _Const(30.0), "renmark": _Const(60.0)},
            open_bottom=False,
        )
        # Just below the renmark base (40) is still nearest renmark; this only
        # differs from open_bottom in that the band is closed at 40.
        assert closed.classify(np.array([[0.0, 0.0, 35.0]]))[0] == 2

    def test_classify_matches_demo_where_logic(self):
        # The contract Item 5 depends on: from_depths + classify must reproduce the
        # demo's hand-rolled depth classification (monotone-clamped bases, then
        # depth = surface - z, then nested np.where), including crossing surfaces
        # and points below the stack.
        top = _Const(100.0)
        shep = _PlaneX(20.0, 0.10)   # base depth 20 -> 30 across x
        cal = _PlaneX(50.0, -0.15)   # base depth 50 -> 35, crosses shep
        model = LayerModel.from_depths(
            top, {"shep": shep, "cal": cal, "renmark": _Const(80.0)}, open_bottom=True
        )

        rng = np.random.default_rng(0)
        xs = rng.uniform(0.0, 100.0, 5000)
        zs = rng.uniform(-50.0, 130.0, 5000)
        ys = np.zeros_like(xs)
        pts = np.column_stack([xs, ys, zs])
        got = model.classify(pts)

        xy = np.column_stack([xs, ys])
        surface = top(xy)
        base_shep = shep(xy)
        base_cal = np.maximum(cal(xy), base_shep)  # demo's monotone clamp
        depth = surface - zs
        want = np.where(depth <= base_shep, 0, np.where(depth <= base_cal, 1, 2))
        np.testing.assert_array_equal(got, want)

    def test_value_field(self):
        model = self._flat_model()
        pts = np.array([[0.0, 0.0, 95.0], [0.0, 0.0, 50.0]])
        ks = model.value_field(pts, {"shep": 2.5e-5, "cal": 1e-3, "renmark": 5e-4})
        np.testing.assert_allclose(ks, [2.5e-5, 5e-4])

    def test_clamped_depths_keep_thickness_and_classifier_consistent(self):
        # The demo's failure mode: independently-fitted base depths cross (Renmark
        # base fitted SHALLOWER than the Calivil base). If the mesh thickness used
        # the raw Renmark depth while the classifier clamped the bands, the mesh
        # floor could sit above the Calivil base and leave zero Renmark cells.
        # Pre-clamping the depths monotone (as the demo does) keeps both consistent.
        from omega.fields.surfaces import clamp_monotonic

        top = _Const(100.0)
        raw = [_Const(20.0), _Const(60.0), _Const(40.0)]  # renmark(40) < cal(60): crossing
        shep, cal, ren = clamp_monotonic(raw, increasing=True)
        depths = {"shep": shep, "cal": cal, "ren": ren}
        model = LayerModel.from_depths(top, depths, open_bottom=True)

        q = np.array([[0.0, 0.0]])
        mesh_bottom = top(q) - ren(q)            # what the demo passes as thickness
        cal_base = top(q) - cal(q)               # classifier's Renmark top boundary
        # Clamped Renmark depth (60) >= Calivil depth (60), so the mesh floor is at
        # or below the Calivil base: a Renmark band exists.
        assert mesh_bottom[0] <= cal_base[0]
        # A node just below the Calivil base classifies as Renmark (index 2).
        pt = np.array([[0.0, 0.0, cal_base[0] - 1.0]])
        assert model.classify(pt)[0] == 2

    def test_classify_collapses_repeated_columns(self):
        # classify evaluates surfaces once per unique (x, y) and broadcasts down the
        # column (the mesh repeats each column's xy per layer). Repeated columns at
        # different z must classify per-z, identically to one-off points.
        model = self._flat_model()
        # Two columns, each sampled at three depths (shep / cal / renmark bands).
        zs = [95.0, 80.0, 50.0]
        pts = np.array(
            [[10.0, 20.0, z] for z in zs] + [[300.0, 400.0, z] for z in zs]
        )
        got = model.classify(pts)
        np.testing.assert_array_equal(got, [0, 1, 2, 0, 1, 2])

    def test_top_surface_dip_propagates_to_bands(self):
        # Top dips from 100 (x=0) to 0 (x=100); a 10 m-thick first layer follows it.
        # Closed so the single layer's bottom is top - 10 rather than an open floor.
        model = LayerModel.from_depths(
            _PlaneX(100.0, -1.0), {"a": _Const(10.0)}, open_bottom=False
        )
        q = np.array([[0.0, 0.0], [100.0, 0.0]])
        np.testing.assert_allclose(model.boundary_elevation("a", "top", q), [100.0, 0.0])
        np.testing.assert_allclose(model.boundary_elevation("a", "bottom", q), [90.0, -10.0])

    def test_monotonic_clamp_keeps_bands_non_overlapping(self):
        # Crossing depths: cal base (depth 10) shallower than shep base (depth 30).
        crossing = LayerModel.from_depths(
            _Const(100.0),
            {"shep": _Const(30.0), "cal": _Const(10.0)},
            enforce_monotonic=True,
        )
        bands = crossing._bands(np.array([[0.0, 0.0]]))
        for top, bottom in bands.values():
            assert bottom[0] <= top[0]  # no inverted band
        # And the clamped stack descends: each band sits at or below the one above.
        tops = [bands[name][0][0] for name in crossing.layer_order]
        assert tops[1] <= bands["shep"][1][0]

    def test_empty_layer_depths_raises(self):
        with pytest.raises(ValueError, match="at least one layer"):
            LayerModel.from_depths(_Const(0.0), {})

    def test_non_surface_top_raises(self):
        with pytest.raises(TypeError, match="top must be a Surface"):
            LayerModel.from_depths(100.0, {"a": _Const(10.0)})

    def test_non_surface_depth_raises(self):
        with pytest.raises(TypeError, match="must be a Surface"):
            LayerModel.from_depths(_Const(100.0), {"a": 10.0})


# --- Firedrake seam: only runs where Firedrake is installed --------------------
# NB: gate the class, not the module -- a module-level importorskip would skip the
# pure-numpy tests above it too.

_HAS_FIREDRAKE = importlib.util.find_spec("firedrake") is not None


@pytest.mark.skipif(not _HAS_FIREDRAKE, reason="firedrake not installed")
class TestMeshFields:
    def _extruded_unit_mesh(self):
        from firedrake import ExtrudedMesh, UnitSquareMesh

        base = UnitSquareMesh(4, 4)
        return ExtrudedMesh(base, layers=4, layer_height=0.25)

    def test_node_coordinates_shape(self):
        from firedrake import FunctionSpace

        from omega.fields.mesh_fields import node_coordinates

        mesh = self._extruded_unit_mesh()
        V = FunctionSpace(mesh, "CG", 1)
        coords = node_coordinates(V)
        assert coords.shape[1] == 3
        assert len(coords) == V.dim()

    def test_assign_field_matches_valuator(self):
        from firedrake import FunctionSpace

        from omega.fields.mesh_fields import assign_field, node_coordinates

        mesh = self._extruded_unit_mesh()
        V = FunctionSpace(mesh, "CG", 1)

        def valuator(coords):
            return coords[:, 2] * 10.0  # value = 10 * z

        field = assign_field(V, valuator, name="test")
        expected = node_coordinates(V)[:, 2] * 10.0
        np.testing.assert_allclose(field.dat.data_ro, expected)

    def test_assign_field_with_layer_model(self):
        from firedrake import FunctionSpace

        from omega.fields.mesh_fields import assign_field

        mesh = self._extruded_unit_mesh()  # z in [0, 1]
        V = FunctionSpace(mesh, "CG", 1)

        # One layer 'upper' over [0.5, 1], 'lower' over [0, 0.5].
        x = [0.0, 1.0, 0.0, 1.0]
        y = [0.0, 0.0, 1.0, 1.0]
        obs = IntervalObservations(
            x * 2,
            y * 2,
            [1.0] * 4 + [0.5] * 4,
            [0.5] * 4 + [0.0] * 4,
            ["upper"] * 4 + ["lower"] * 4,
        )
        model = LayerModel(obs, ["upper", "lower"], sigma=0.5)
        field = assign_field(V, lambda c: model.value_field(c, {"upper": 1.0, "lower": 9.0}))
        # Every dof should be either 1.0 or 9.0.
        assert set(np.unique(field.dat.data_ro)).issubset({1.0, 9.0})

    def test_assign_field_wrong_length_raises(self):
        from firedrake import FunctionSpace

        from omega.fields.mesh_fields import assign_field

        mesh = self._extruded_unit_mesh()
        V = FunctionSpace(mesh, "CG", 1)
        with pytest.raises(ValueError, match="expected"):
            assign_field(V, lambda c: np.zeros(3))
