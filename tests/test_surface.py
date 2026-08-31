"""Tests for omega.mesh.surface module.

These tests exercise the SurfaceMesh class without requiring Firedrake.
Only gmsh (available at PYTHONPATH=/opt/homebrew/lib) is needed.
"""

import tempfile
from pathlib import Path

import gmsh
import numpy as np
import pytest

from omega.exceptions import MeshGenerationError
from omega.geometry.polygon import Polygon
from omega.mesh.surface import SurfaceMesh


class TestSurfaceMeshGenerate:
    """Tests for SurfaceMesh.generate() and the resulting array properties."""

    def test_generate_stores_coords(self, simple_surface_mesh):
        """After generate(), coords is an ndarray with shape (n, 2) and float64 dtype."""
        coords = simple_surface_mesh.coords
        assert isinstance(coords, np.ndarray)
        assert coords.ndim == 2
        assert coords.shape[1] == 2
        assert coords.dtype == np.float64

    def test_generate_stores_cells(self, simple_surface_mesh):
        """After generate(), cells is an ndarray with shape (m, 3) and integer dtype."""
        cells = simple_surface_mesh.cells
        assert isinstance(cells, np.ndarray)
        assert cells.ndim == 2
        assert cells.shape[1] == 3
        assert np.issubdtype(cells.dtype, np.integer)
        # All indices must be in [0, n_nodes)
        assert np.all(cells >= 0)
        assert np.all(cells < simple_surface_mesh.n_nodes)

    def test_generate_stores_boundary_edges(self, simple_surface_mesh):
        """boundary_edges is an ndarray with shape (e, 2) and valid node indices."""
        edges = simple_surface_mesh.boundary_edges
        assert isinstance(edges, np.ndarray)
        assert edges.ndim == 2
        assert edges.shape[1] == 2
        assert np.issubdtype(edges.dtype, np.integer)
        assert np.all(edges >= 0)
        assert np.all(edges < simple_surface_mesh.n_nodes)

    def test_n_nodes_positive(self, simple_surface_mesh):
        """n_nodes returns a positive integer after generation."""
        assert isinstance(simple_surface_mesh.n_nodes, int)
        assert simple_surface_mesh.n_nodes > 0

    def test_n_cells_positive(self, simple_surface_mesh):
        """n_cells returns a positive integer after generation."""
        assert isinstance(simple_surface_mesh.n_cells, int)
        assert simple_surface_mesh.n_cells > 0

    def test_n_nodes_consistent(self, simple_surface_mesh):
        """n_nodes matches the first dimension of coords."""
        assert simple_surface_mesh.n_nodes == simple_surface_mesh.coords.shape[0]

    def test_n_cells_consistent(self, simple_surface_mesh):
        """n_cells matches the first dimension of cells."""
        assert simple_surface_mesh.n_cells == simple_surface_mesh.cells.shape[0]


class TestSurfaceMeshBeforeGenerate:
    """Tests that accessing properties before generate() raises correctly."""

    def test_coords_before_generate_raises(self, simple_polygon_coords):
        """Accessing coords before generate() raises MeshGenerationError."""
        polygon = Polygon(simple_polygon_coords)
        sm = SurfaceMesh(polygon, resolution=20.0)
        with pytest.raises(MeshGenerationError):
            _ = sm.coords

    def test_cells_before_generate_raises(self, simple_polygon_coords):
        """Accessing cells before generate() raises MeshGenerationError."""
        polygon = Polygon(simple_polygon_coords)
        sm = SurfaceMesh(polygon, resolution=20.0)
        with pytest.raises(MeshGenerationError):
            _ = sm.cells

    def test_boundary_edges_before_generate_raises(self, simple_polygon_coords):
        """Accessing boundary_edges before generate() raises MeshGenerationError."""
        polygon = Polygon(simple_polygon_coords)
        sm = SurfaceMesh(polygon, resolution=20.0)
        with pytest.raises(MeshGenerationError):
            _ = sm.boundary_edges

    def test_n_nodes_before_generate_raises(self, simple_polygon_coords):
        """Accessing n_nodes before generate() raises MeshGenerationError."""
        polygon = Polygon(simple_polygon_coords)
        sm = SurfaceMesh(polygon, resolution=20.0)
        with pytest.raises(MeshGenerationError):
            _ = sm.n_nodes

    def test_n_cells_before_generate_raises(self, simple_polygon_coords):
        """Accessing n_cells before generate() raises MeshGenerationError."""
        polygon = Polygon(simple_polygon_coords)
        sm = SurfaceMesh(polygon, resolution=20.0)
        with pytest.raises(MeshGenerationError):
            _ = sm.n_cells


class TestSurfaceMeshWrite:
    """Tests for SurfaceMesh.write() file output."""

    def test_write_creates_file(self, simple_surface_mesh):
        """write() creates a .msh file that exists on disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = Path(tmpdir) / "output.msh"
            simple_surface_mesh.write(str(outpath))
            assert outpath.exists()
            assert outpath.stat().st_size > 0

    def test_write_roundtrip(self, simple_surface_mesh):
        """Write a mesh, then re-read with gmsh: node and element counts match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = Path(tmpdir) / "roundtrip.msh"
            simple_surface_mesh.write(str(outpath))

            original_n_nodes = simple_surface_mesh.n_nodes
            original_n_cells = simple_surface_mesh.n_cells

            # Re-read with gmsh to verify consistency
            gmsh.initialize()
            try:
                gmsh.option.setNumber("General.Terminal", 0)
                gmsh.open(str(outpath))
                node_tags, _, _ = gmsh.model.mesh.getNodes()
                elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=2)
                n_nodes_read = len(node_tags)
                n_cells_read = sum(len(t) for t in elem_tags)
            finally:
                gmsh.finalize()

            assert n_nodes_read == original_n_nodes
            assert n_cells_read == original_n_cells


class TestSurfaceMeshPolygons:
    """Tests for different polygon geometries."""

    def test_simple_rectangle(self, simple_polygon_coords):
        """A 4-vertex rectangle produces a valid mesh with nodes and cells."""
        polygon = Polygon(simple_polygon_coords)
        sm = SurfaceMesh(polygon, resolution=20.0)
        sm.generate()

        assert sm.n_nodes > 0
        assert sm.n_cells > 0
        assert sm.coords.shape == (sm.n_nodes, 2)
        assert sm.cells.shape == (sm.n_cells, 3)

    def test_murrumbidgee_polygon(self, murrumbidgee_polygon_coords):
        """The 7-vertex Murrumbidgee polygon produces a valid mesh."""
        polygon = Polygon(murrumbidgee_polygon_coords)
        sm = SurfaceMesh(polygon, resolution=5000.0)
        sm.generate()

        assert sm.n_nodes > 0
        assert sm.n_cells > 0
        assert sm.coords.shape == (sm.n_nodes, 2)
        assert sm.cells.shape == (sm.n_cells, 3)


class TestSurfaceMeshResolution:
    """Tests for resolution effects on mesh density."""

    def test_resolution_affects_mesh_density(self, simple_polygon_coords):
        """Lower resolution value (finer mesh) should produce more nodes/cells."""
        polygon = Polygon(simple_polygon_coords)

        sm_coarse = SurfaceMesh(polygon, resolution=50.0)
        sm_coarse.generate()

        sm_fine = SurfaceMesh(polygon, resolution=10.0)
        sm_fine.generate()

        # Finer resolution (smaller value) means more nodes and cells
        assert sm_fine.n_nodes > sm_coarse.n_nodes
        assert sm_fine.n_cells > sm_coarse.n_cells


class TestSurfaceMeshValidation:
    """Tests for mesh validity checks."""

    def test_cells_reference_valid_nodes(self, simple_surface_mesh):
        """Every index in cells must be a valid index into coords."""
        cells = simple_surface_mesh.cells
        n_nodes = simple_surface_mesh.n_nodes
        assert np.all(cells >= 0)
        assert np.all(cells < n_nodes)

    def test_boundary_edges_form_closed_loop(self, simple_surface_mesh):
        """Boundary edges should form one or more closed loops.

        In a closed loop, every node that appears in the boundary edges
        must appear exactly twice: once as a start node and once as an end
        node (considering the edges form chains).
        """
        edges = simple_surface_mesh.boundary_edges
        # Each boundary node should appear the same number of times
        # as a source and as a destination (degree balance for closed loops).
        from_nodes = edges[:, 0]
        to_nodes = edges[:, 1]
        all_nodes = np.concatenate([from_nodes, to_nodes])
        unique_nodes, counts = np.unique(all_nodes, return_counts=True)

        # In a valid closed boundary, every node appears exactly twice
        # (once per edge it connects to).
        assert np.all(counts == 2), (
            f"Not all boundary nodes appear exactly twice; "
            f"counts: {dict(zip(unique_nodes[counts != 2], counts[counts != 2], strict=False))}"
        )


class TestSurfaceMeshProperties:
    """Tests for SurfaceMesh metadata properties."""

    def test_polygon_property(self, simple_polygon_coords):
        """sm.polygon returns the original Polygon object."""
        polygon = Polygon(simple_polygon_coords)
        sm = SurfaceMesh(polygon, resolution=20.0)
        assert sm.polygon is polygon

    def test_resolution_property(self, simple_polygon_coords):
        """sm.resolution returns the resolution that was set."""
        polygon = Polygon(simple_polygon_coords)
        sm = SurfaceMesh(polygon, resolution=42.5)
        assert sm.resolution == 42.5

    def test_coords_within_polygon_bounds(self, simple_surface_mesh):
        """Generated mesh coordinates should lie within (or close to) the polygon bounds."""
        coords = simple_surface_mesh.coords
        polygon = simple_surface_mesh.polygon
        minx, miny, maxx, maxy = polygon.bounds

        # Allow a small tolerance for floating point
        tol = simple_surface_mesh.resolution * 0.1
        assert np.all(coords[:, 0] >= minx - tol)
        assert np.all(coords[:, 0] <= maxx + tol)
        assert np.all(coords[:, 1] >= miny - tol)
        assert np.all(coords[:, 1] <= maxy + tol)
