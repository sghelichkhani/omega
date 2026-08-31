"""Smoke test for the demo modelling pipeline.

Exercises the same staged API the Lower Murrumbidgee demo uses -- SurfaceMesh ->
Gaussian-kernel surfaces -> monotone clamp -> LayerModel.from_depths ->
build_mesh_hierarchy -> assign_field classify -> formation/conductivity lookup --
end to end on a small synthetic domain, with no network or external data. The
real demo (lower_murrumbidgee_mesh.py) adds only the ausdem/austrata data
sources on top of this.
"""

import numpy as np
import pytest

# Skip if Firedrake is not available (the build/extrude/assign steps need it).
firedrake = pytest.importorskip("firedrake")


@pytest.mark.demo
class TestDemoPipeline:
    def _surfaces(self):
        """A DEM top surface and three crossing base-depth surfaces over a box."""
        from omega import GaussianKernelSurface

        x = np.linspace(-50, 150, 12)
        xx, yy = np.meshgrid(x, x)
        coords = np.column_stack([xx.ravel(), yy.ravel()])
        dem = GaussianKernelSurface(coords, 100.0 + 0.05 * coords[:, 0], sigma=60.0)
        # Properly stacked base depths (shep < cal < renmark) so all three
        # formations appear in the mesh. (The crossing/clamp consistency case is
        # covered by the pure-numpy test in tests/test_stratigraphy.py.)
        shep = GaussianKernelSurface(coords, np.full(len(coords), 10.0), sigma=60.0)
        cal = GaussianKernelSurface(coords, np.full(len(coords), 40.0), sigma=60.0)
        ren = GaussianKernelSurface(coords, np.full(len(coords), 90.0), sigma=60.0)
        return dem, shep, cal, ren

    def test_pipeline_builds_mesh_and_classifies(self):
        from omega import (
            LayerModel,
            Polygon,
            SurfaceMesh,
            assign_field,
            build_mesh_hierarchy,
            clamp_monotonic,
        )
        from firedrake import Function, FunctionSpace

        dem, shep, cal, ren = self._surfaces()
        order = ["Shepparton", "Calivil", "Renmark"]
        ks = {"Shepparton": 2.5e-5, "Calivil": 1e-3, "Renmark": 5e-4}

        clamped = clamp_monotonic([shep, cal, ren], increasing=True)
        depths = dict(zip(order, clamped))
        model = LayerModel.from_depths(dem, depths, open_bottom=True)
        thickness = depths["Renmark"].clamp_min(5.0)

        sm = SurfaceMesh(Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]), resolution=25.0)
        sm.generate()
        hierarchy = build_mesh_hierarchy(sm.to_firedrake_mesh(), dem, thickness, n_layers=8)
        mesh = hierarchy[-1]
        assert mesh.extruded and mesh.num_cells() > 0

        V = FunctionSpace(mesh, "CG", 1)
        layer_idx = assign_field(V, lambda c: model.classify(c).astype(float))
        idx = np.rint(layer_idx.dat.data_ro).astype(int)
        # All three formations must appear (a #1 regression that collapsed Renmark
        # would leave {0, 1}), and conductivity is a clean lookup.
        assert set(np.unique(idx)) == {0, 1, 2}
        conductivity = Function(V)
        conductivity.dat.data[:] = np.array([ks[f] for f in order])[idx]
        assert set(np.unique(conductivity.dat.data_ro)).issubset(set(ks.values()))

    def test_top_of_mesh_follows_dem(self):
        from omega import (
            LayerModel,
            Polygon,
            SurfaceMesh,
            build_mesh_hierarchy,
            clamp_monotonic,
        )

        dem, shep, cal, ren = self._surfaces()
        depths = dict(zip(["a", "b", "c"], clamp_monotonic([shep, cal, ren], increasing=True)))
        LayerModel.from_depths(dem, depths)  # constructs without error

        sm = SurfaceMesh(Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]), resolution=25.0)
        sm.generate()
        thickness = depths["c"].clamp_min(5.0)
        mesh = build_mesh_hierarchy(sm.to_firedrake_mesh(), dem, thickness, n_layers=8)[-1]
        z = mesh.coordinates.dat.data[:, 2]
        # Mesh top should track the DEM (~100-105 over the [0,100] box).
        assert 95.0 < z.max() < 115.0
