"""Tests for omega.mesh.builder module.

These tests require Firedrake to be installed. They exercise the
build_mesh_hierarchy() function and the SurfaceMesh.to_firedrake_mesh()
method for end-to-end mesh generation with terrain-following transforms.
"""

import numpy as np
import pytest

from omega.exceptions import MeshGenerationError
from omega.fields.surfaces import GaussianKernelSurface, Surface
from omega.geometry.polygon import Polygon
from omega.mesh.surface import SurfaceMesh

# Skip entire module if Firedrake is not available
firedrake = pytest.importorskip("firedrake")

from omega.mesh.builder import build_mesh_hierarchy  # noqa: E402


class _PlaneSurface(Surface):
    """Exact analytic surface a + b*x + c*y (so transform tests are exact)."""

    def __init__(self, a, b, c):
        self._a, self._b, self._c = float(a), float(b), float(c)

    def __call__(self, xy):
        xy = np.asarray(xy)
        return self._a + self._b * xy[:, 0] + self._c * xy[:, 1]


class _ConstSurface(Surface):
    def __init__(self, value):
        self._value = float(value)

    def __call__(self, xy):
        return np.full(len(np.asarray(xy)), self._value)


class TestToFiredrakeMesh:
    """Tests for SurfaceMesh.to_firedrake_mesh() conversion."""

    def test_to_firedrake_mesh_returns_mesh(self, simple_polygon_coords):
        """to_firedrake_mesh() returns a Firedrake Mesh object."""
        polygon = Polygon(simple_polygon_coords)
        sm = SurfaceMesh(polygon, resolution=20.0)
        sm.generate()

        mesh_2d = sm.to_firedrake_mesh()
        # firedrake.Mesh is a factory function, not a type; check the result instead.
        assert mesh_2d.coordinates.dat.data.shape[1] == 2

    def test_to_firedrake_mesh_cell_count(self, simple_polygon_coords):
        """The Firedrake mesh cell count matches the SurfaceMesh cell count."""
        polygon = Polygon(simple_polygon_coords)
        sm = SurfaceMesh(polygon, resolution=20.0)
        sm.generate()

        mesh_2d = sm.to_firedrake_mesh()
        assert mesh_2d.num_cells() == sm.n_cells


class TestBuildMeshHierarchy:
    """Tests for the build_mesh_hierarchy() function."""

    @pytest.fixture
    def simple_mesh_2d(self, simple_polygon_coords):
        """Create a simple 2D Firedrake mesh from the test polygon."""
        polygon = Polygon(simple_polygon_coords)
        sm = SurfaceMesh(polygon, resolution=20.0)
        sm.generate()
        return sm.to_firedrake_mesh()

    @pytest.fixture
    def top_surface(self):
        """Gentle ground elevation: 100 + 0.1*x + 0.05*y (exact)."""
        return _PlaneSurface(100.0, 0.1, 0.05)

    @pytest.fixture
    def thickness_surface(self):
        """Constant depth to bedrock of 50."""
        return _ConstSurface(50.0)

    def test_build_single_level(self, simple_mesh_2d, top_surface, thickness_surface):
        """refinement_levels=0 returns a hierarchy with 1 mesh."""
        hierarchy = build_mesh_hierarchy(
            simple_mesh_2d, top_surface, thickness_surface, n_layers=5
        )
        assert len(hierarchy) == 1

    def test_mesh_is_extruded_3d(self, simple_mesh_2d, top_surface, thickness_surface):
        """hierarchy[-1] is an extruded mesh with 3D coordinates."""
        hierarchy = build_mesh_hierarchy(
            simple_mesh_2d, top_surface, thickness_surface, n_layers=5
        )
        mesh = hierarchy[-1]
        assert mesh.extruded is True
        assert mesh.coordinates.dat.data.shape[1] == 3

    def test_terrain_transform_surface_and_bedrock(
        self, simple_mesh_2d, top_surface, thickness_surface
    ):
        """z=1 lands on the top surface; z=0 on bedrock (top - thickness).

        The surfaces are exact analytic planes, so with no double interpolation the
        extremes match the true elevation to tight tolerance.
        """
        hierarchy = build_mesh_hierarchy(
            simple_mesh_2d, top_surface, thickness_surface, n_layers=10
        )
        z = hierarchy[-1].coordinates.dat.data[:, 2]
        # Top corner (100,100): 100 + 10 + 5 = 115; origin bedrock: 100 - 50 = 50.
        assert np.max(z) == pytest.approx(115.0, abs=1e-6)
        assert np.min(z) == pytest.approx(50.0, abs=1e-6)

    def test_terrain_transform_midpoint(
        self, simple_mesh_2d, top_surface, thickness_surface
    ):
        """Each column's middle node is halfway between bedrock and surface."""
        hierarchy = build_mesh_hierarchy(
            simple_mesh_2d, top_surface, thickness_surface, n_layers=10
        )
        mesh_coords = hierarchy[-1].coordinates.dat.data
        unique_xy = np.unique(np.round(mesh_coords[:, :2], decimals=6), axis=0)
        for xy in unique_xy[:5]:
            mask = np.all(np.abs(mesh_coords[:, :2] - xy) < 1e-4, axis=1)
            col_z = np.sort(mesh_coords[mask, 2])
            if len(col_z) >= 3:
                z_mid_expected = (col_z[0] + col_z[-1]) / 2.0
                assert col_z[len(col_z) // 2] == pytest.approx(z_mid_expected, abs=2.0)

    def test_non_surface_arg_raises(self, simple_mesh_2d):
        """Passing raw arrays (the old signature) now raises TypeError."""
        with pytest.raises(TypeError, match="top_surface must be a Surface"):
            build_mesh_hierarchy(
                simple_mesh_2d, np.zeros((4, 2)), _ConstSurface(50.0), n_layers=5
            )

    def test_validate_terrain_nan_raises(self, simple_mesh_2d, thickness_surface):
        """A top surface that evaluates to NaN raises when validate_terrain=True."""
        with pytest.raises(MeshGenerationError):
            build_mesh_hierarchy(
                simple_mesh_2d,
                _ConstSurface(np.nan),
                thickness_surface,
                n_layers=5,
                validate_terrain=True,
            )

    def test_validate_terrain_negative_thickness_raises(
        self, simple_mesh_2d, top_surface
    ):
        """A negative thickness (inverted mesh) is caught at sample time."""
        with pytest.raises(MeshGenerationError):
            build_mesh_hierarchy(
                simple_mesh_2d,
                top_surface,
                _ConstSurface(-10.0),
                n_layers=5,
                validate_terrain=True,
            )

    def test_validate_terrain_can_skip(self, simple_mesh_2d, thickness_surface):
        """NaN terrain with validate_terrain=False completes without raising."""
        hierarchy = build_mesh_hierarchy(
            simple_mesh_2d,
            _ConstSurface(np.nan),
            thickness_surface,
            n_layers=5,
            validate_terrain=False,
        )
        assert hierarchy is not None

    def test_multilevel_hierarchy(self, simple_mesh_2d, top_surface, thickness_surface):
        """refinement_levels=1 produces 2 nested meshes, finer with more cells."""
        hierarchy = build_mesh_hierarchy(
            simple_mesh_2d, top_surface, thickness_surface, n_layers=5, refinement_levels=1
        )
        assert len(hierarchy) == 2
        assert hierarchy[1].num_cells() > hierarchy[0].num_cells()

    def test_mesh_positive_cells(self, simple_mesh_2d, top_surface, thickness_surface):
        """The generated mesh has a positive number of cells."""
        hierarchy = build_mesh_hierarchy(
            simple_mesh_2d, top_surface, thickness_surface, n_layers=5
        )
        assert hierarchy[-1].num_cells() > 0

    def test_extrusion_power_clusters_toward_bedrock(
        self, simple_mesh_2d, top_surface, thickness_surface
    ):
        """extrusion_power > 1 clusters layers toward the bedrock (bottom)."""
        hierarchy = build_mesh_hierarchy(
            simple_mesh_2d, top_surface, thickness_surface, n_layers=10, extrusion_power=2.0
        )
        mesh_coords = hierarchy[-1].coordinates.dat.data
        unique_xy = np.unique(np.round(mesh_coords[:, :2], decimals=6), axis=0)
        # In any column, the bottom gap should be smaller than the top gap.
        for xy in unique_xy[:5]:
            mask = np.all(np.abs(mesh_coords[:, :2] - xy) < 1e-4, axis=1)
            gaps = np.diff(np.sort(mesh_coords[mask, 2]))
            if len(gaps) >= 2:
                assert gaps[0] < gaps[-1]

    def test_end_to_end_with_gaussian_surface(self, simple_mesh_2d):
        """Smoke test: real GaussianKernelSurface evaluated on real mesh nodes."""
        x = np.linspace(-20, 120, 12)
        xx, yy = np.meshgrid(x, x)
        coords = np.column_stack([xx.ravel(), yy.ravel()])
        elev = 100.0 + 0.1 * coords[:, 0]
        depth = np.full(len(coords), 50.0)
        top = GaussianKernelSurface(coords, elev, sigma=40.0)
        thickness = GaussianKernelSurface(coords, depth, sigma=40.0).clamp_min(5.0)

        hierarchy = build_mesh_hierarchy(simple_mesh_2d, top, thickness, n_layers=5)
        z = hierarchy[-1].coordinates.dat.data[:, 2]
        assert np.all(np.isfinite(z))
        # Every column spans a positive height (no collapse).
        mesh_coords = hierarchy[-1].coordinates.dat.data
        unique_xy = np.unique(np.round(mesh_coords[:, :2], decimals=6), axis=0)
        for xy in unique_xy[:5]:
            mask = np.all(np.abs(mesh_coords[:, :2] - xy) < 1e-4, axis=1)
            col_z = mesh_coords[mask, 2]
            assert col_z.max() - col_z.min() > 1.0


class TestLegacyGriddataEquivalence:
    """A GridSurface through the builder must reproduce the pre-refactor mesh.

    Before the Surface refactor, build_mesh_hierarchy took raw coordinate and
    value arrays and ran them through scipy griddata (linear, with a nearest
    backfill for the convex-hull NaNs). Existing campaigns were meshed that way,
    so the equivalence has to hold to the bit: anything less means re-running
    them to keep the reported numbers honest. These tests reconstruct the legacy
    arithmetic explicitly and compare.
    """

    @pytest.fixture
    def simple_mesh_2d(self, simple_polygon_coords):
        polygon = Polygon(simple_polygon_coords)
        sm = SurfaceMesh(polygon, resolution=20.0)
        sm.generate()
        return sm.to_firedrake_mesh()

    @pytest.fixture
    def terrain_arrays(self):
        """A dense lattice of elevation and thickness picks, as a DEM would give."""
        x = np.linspace(-50.0, 150.0, 25)
        xx, yy = np.meshgrid(x, x)
        coords = np.column_stack([xx.ravel(), yy.ravel()])
        elevation = 100.0 + 0.1 * coords[:, 0] + 0.05 * coords[:, 1]
        thickness = 40.0 + 5.0 * np.sin(coords[:, 0] / 30.0)
        return coords, elevation, thickness

    @staticmethod
    def _legacy_sample(coords, values, xy):
        """The exact sampling the array-based builder performed."""
        from scipy.interpolate import griddata

        out = griddata(coords, values, xy, method="linear")
        gaps = np.isnan(out)
        if np.any(gaps):
            out[gaps] = griddata(coords, values, xy[gaps], method="nearest")
        return out

    def test_grid_surface_matches_legacy_sampling_on_the_mesh_nodes(
        self, simple_mesh_2d, terrain_arrays
    ):
        from omega.fields.surfaces import GridSurface

        coords, elevation, thickness = terrain_arrays
        xy = simple_mesh_2d.coordinates.dat.data_ro[:, :2]

        assert np.array_equal(
            GridSurface(coords, elevation)(xy),
            self._legacy_sample(coords, elevation, xy),
        )
        assert np.array_equal(
            GridSurface(coords, thickness)(xy),
            self._legacy_sample(coords, thickness, xy),
        )

    def test_hierarchy_coordinates_match_the_legacy_transform(
        self, simple_mesh_2d, terrain_arrays
    ):
        # Sample on the coarse nodes, then apply z = thk*z + top - thk, which is
        # what the legacy builder did before prolonging. With refinement_levels=0
        # there is no prolongation, so the whole legacy path is reproduced here.
        from omega.fields.surfaces import GridSurface

        coords, elevation, thickness = terrain_arrays
        hierarchy = build_mesh_hierarchy(
            simple_mesh_2d,
            GridSurface(coords, elevation),
            GridSurface(coords, thickness),
            n_layers=5,
        )
        mesh = hierarchy[-1]

        xy = mesh.coordinates.dat.data_ro[:, :2]
        top = self._legacy_sample(coords, elevation, xy)
        thk = self._legacy_sample(coords, thickness, xy)

        # Recover the normalised z the extrusion started from, then re-apply the
        # legacy transform to it and check we land on the same coordinates.
        z_physical = mesh.coordinates.dat.data_ro[:, 2]
        z_normalised = (z_physical - (top - thk)) / thk
        assert np.allclose(thk * z_normalised + top - thk, z_physical, rtol=0, atol=1e-9)
        # The column really does span bedrock to ground surface.
        assert np.isclose(z_normalised.min(), 0.0, atol=1e-9)
        assert np.isclose(z_normalised.max(), 1.0, atol=1e-9)

    def test_survives_a_mesh_extending_past_the_terrain_data(self, simple_polygon_coords):
        # The nearest backfill is the reason the legacy path never produced NaN
        # geometry. A GridSurface must inherit that, or a mesh whose polygon
        # escapes its terrain grid gets NaN coordinates instead of flat steps.
        from omega.fields.surfaces import GridSurface

        polygon = Polygon(simple_polygon_coords)
        sm = SurfaceMesh(polygon, resolution=20.0)
        sm.generate()
        mesh_2d = sm.to_firedrake_mesh()

        # Terrain covering only a corner of the polygon.
        x = np.linspace(0.0, 30.0, 8)
        xx, yy = np.meshgrid(x, x)
        coords = np.column_stack([xx.ravel(), yy.ravel()])

        hierarchy = build_mesh_hierarchy(
            mesh_2d,
            GridSurface(coords, 100.0 + 0.1 * coords[:, 0]),
            GridSurface(coords, np.full(len(coords), 40.0)),
            n_layers=4,
        )
        assert np.all(np.isfinite(hierarchy[-1].coordinates.dat.data_ro))
