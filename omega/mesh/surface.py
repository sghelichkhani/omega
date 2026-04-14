"""2D surface mesh generation using gmsh."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import gmsh
import numpy as np

from omega.exceptions import MeshGenerationError

if TYPE_CHECKING:
    from omega.geometry.polygon import Polygon


class SurfaceMesh:
    """2D triangular mesh generator from polygon boundary.

    Uses gmsh Python API to generate unstructured triangular meshes suitable
    for extrusion into 3D. After calling generate(), mesh arrays (coords,
    cells, boundary_edges) are available as properties.

    Args:
        polygon: Domain boundary polygon.
        resolution: Target element size in polygon coordinate units (e.g., meters).

    Example:
        >>> from omega.geometry import Polygon
        >>> polygon = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        >>> sm = SurfaceMesh(polygon, resolution=10.0)
        >>> sm.generate()
        >>> print(sm.n_nodes, sm.n_cells)
    """

    def __init__(self, polygon: Polygon, resolution: float):
        self._polygon = polygon
        self._resolution = resolution
        self._boundary_tags: dict[str, int] = {}
        self._generated = False

        # Arrays populated by generate()
        self._coords: np.ndarray | None = None
        self._cells: np.ndarray | None = None
        self._boundary_edges: np.ndarray | None = None

        # Path to the gmsh temp file (kept alive for write())
        self._temp_path: Path | None = None

    @property
    def polygon(self) -> Polygon:
        """Return the domain polygon."""
        return self._polygon

    @property
    def resolution(self) -> float:
        """Return target mesh resolution."""
        return self._resolution

    @property
    def boundary_tags(self) -> dict[str, int]:
        """Return mapping of boundary names to physical group tags."""
        return self._boundary_tags.copy()

    @property
    def coords(self) -> np.ndarray:
        """Node coordinates, shape (n_nodes, 2).

        Raises:
            MeshGenerationError: If mesh has not been generated yet.
        """
        if not self._generated:
            raise MeshGenerationError(
                "Mesh has not been generated yet. Call generate() first."
            )
        return self._coords

    @property
    def cells(self) -> np.ndarray:
        """Triangle connectivity, shape (n_cells, 3), 0-indexed.

        Raises:
            MeshGenerationError: If mesh has not been generated yet.
        """
        if not self._generated:
            raise MeshGenerationError(
                "Mesh has not been generated yet. Call generate() first."
            )
        return self._cells

    @property
    def boundary_edges(self) -> np.ndarray:
        """Boundary edge connectivity, shape (n_edges, 2), 0-indexed.

        Raises:
            MeshGenerationError: If mesh has not been generated yet.
        """
        if not self._generated:
            raise MeshGenerationError(
                "Mesh has not been generated yet. Call generate() first."
            )
        return self._boundary_edges

    @property
    def n_nodes(self) -> int:
        """Number of mesh nodes.

        Raises:
            MeshGenerationError: If mesh has not been generated yet.
        """
        if not self._generated:
            raise MeshGenerationError(
                "Mesh has not been generated yet. Call generate() first."
            )
        return self._coords.shape[0]

    @property
    def n_cells(self) -> int:
        """Number of triangular cells.

        Raises:
            MeshGenerationError: If mesh has not been generated yet.
        """
        if not self._generated:
            raise MeshGenerationError(
                "Mesh has not been generated yet. Call generate() first."
            )
        return self._cells.shape[0]

    def generate(
        self,
        output_path: str | Path | None = None,
        mesh_format: str = "msh2",
    ) -> Path:
        """Generate 2D mesh and extract arrays.

        Args:
            output_path: Path for output mesh file. If None, uses a temporary
                file internally (kept alive for subsequent write() calls).
            mesh_format: Gmsh mesh format version ("msh2" or "msh4").
                Default: "msh2" for best Firedrake compatibility.

        Returns:
            Path to the generated mesh file.

        Raises:
            MeshGenerationError: If mesh generation fails.
        """
        if output_path is None:
            fd, tmp_path = tempfile.mkstemp(suffix=".msh")
            output_path = Path(tmp_path)
            self._temp_path = output_path
        else:
            output_path = Path(output_path)
            self._temp_path = output_path

        try:
            self._generate_mesh(output_path, mesh_format)
            self._generated = True
            return output_path
        except Exception as e:
            raise MeshGenerationError(f"Mesh generation failed: {e}") from e

    def _generate_mesh(self, output_path: Path, mesh_format: str) -> None:
        """Internal mesh generation using gmsh API."""
        gmsh.initialize()

        try:
            gmsh.option.setNumber("General.Terminal", 0)

            if mesh_format == "msh2":
                gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            elif mesh_format == "msh4":
                gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

            gmsh.model.add("surface_mesh")

            # Create points from polygon vertices
            poly_coords = self._polygon.coords
            points = []
            for x, y in poly_coords:
                p = gmsh.model.geo.addPoint(x, y, 0, self._resolution)
                points.append(p)

            # Create lines connecting points
            lines = []
            n = len(points)
            for i in range(n):
                p1 = points[i]
                p2 = points[(i + 1) % n]
                line = gmsh.model.geo.addLine(p1, p2)
                lines.append(line)

            # Create curve loop and surface
            curve_loop = gmsh.model.geo.addCurveLoop(lines)
            surface = gmsh.model.geo.addPlaneSurface([curve_loop])

            gmsh.model.geo.synchronize()

            # Physical groups for boundaries and domain
            sides_tag = gmsh.model.addPhysicalGroup(1, lines, 1)
            gmsh.model.setPhysicalName(1, sides_tag, "sides")
            self._boundary_tags["sides"] = sides_tag

            domain_tag = gmsh.model.addPhysicalGroup(2, [surface])
            gmsh.model.setPhysicalName(2, domain_tag, "domain")
            self._boundary_tags["domain"] = domain_tag

            # Generate 2D mesh
            gmsh.model.mesh.generate(2)

            # Extract node data. gmsh uses 1-based, non-contiguous tags.
            node_tags, node_coords_flat, _ = gmsh.model.mesh.getNodes()
            all_coords = np.array(node_coords_flat).reshape(-1, 3)
            self._coords = all_coords[:, :2].copy()

            # Build tag-to-index mapping (1-based tags -> 0-based indices)
            tag_to_idx = {int(tag): idx for idx, tag in enumerate(node_tags)}

            # Extract 2D triangular elements
            elem_types_2d, _, elem_node_tags_2d = (
                gmsh.model.mesh.getElements(dim=2)
            )
            cell_node_tags = np.array(elem_node_tags_2d[0])
            cell_node_tags_reshaped = cell_node_tags.reshape(-1, 3)
            self._cells = np.array(
                [[tag_to_idx[int(t)] for t in row]
                 for row in cell_node_tags_reshaped],
                dtype=np.int64,
            )

            # Extract 1D boundary elements
            elem_types_1d, _, elem_node_tags_1d = (
                gmsh.model.mesh.getElements(dim=1)
            )
            if len(elem_node_tags_1d) > 0:
                edge_node_tags = np.array(elem_node_tags_1d[0])
                edge_node_tags_reshaped = edge_node_tags.reshape(-1, 2)
                self._boundary_edges = np.array(
                    [[tag_to_idx[int(t)] for t in row]
                     for row in edge_node_tags_reshaped],
                    dtype=np.int64,
                )
            else:
                self._boundary_edges = np.empty((0, 2), dtype=np.int64)

            # Write mesh file
            gmsh.write(str(output_path))

        finally:
            gmsh.finalize()

    def write(self, path: str | Path, mesh_format: str = "msh2") -> Path:
        """Write the generated mesh to a file.

        Copies the internally stored mesh file to the given path. If the mesh
        format differs from the one used during generate(), the mesh is
        reconstructed from stored arrays using gmsh.

        Args:
            path: Output file path.
            mesh_format: Gmsh mesh format ("msh2" or "msh4"). Default: "msh2".

        Returns:
            Path to the written file.

        Raises:
            MeshGenerationError: If mesh has not been generated yet.
        """
        if not self._generated:
            raise MeshGenerationError(
                "Mesh has not been generated yet. Call generate() first."
            )

        path = Path(path)

        # If we have the temp file and same format, just copy
        if self._temp_path is not None and self._temp_path.exists():
            if mesh_format == "msh2":
                shutil.copy2(str(self._temp_path), str(path))
                return path

        # Otherwise reconstruct from arrays via gmsh
        self._write_from_arrays(path, mesh_format)
        return path

    def _write_from_arrays(self, path: Path, mesh_format: str) -> None:
        """Reconstruct mesh from stored arrays and write to file."""
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)

            if mesh_format == "msh2":
                gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            elif mesh_format == "msh4":
                gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

            gmsh.model.add("reconstructed_mesh")

            n_nodes = self._coords.shape[0]
            tags = list(range(1, n_nodes + 1))
            coords_3d = np.zeros((n_nodes, 3))
            coords_3d[:, :2] = self._coords

            # Create discrete entities first, then add nodes and elements
            gmsh.model.addDiscreteEntity(2, 1)
            gmsh.model.mesh.addNodes(
                2, 1, tags, coords_3d.ravel().tolist()
            )

            # Add triangular elements (convert 0-indexed to 1-indexed)
            n_cells = self._cells.shape[0]
            cell_tags = list(range(1, n_cells + 1))
            cell_nodes = (self._cells + 1).ravel().tolist()
            gmsh.model.mesh.addElements(
                2, 1, [2], [cell_tags], [cell_nodes]
            )

            # Add boundary edges
            if self._boundary_edges.shape[0] > 0:
                gmsh.model.addDiscreteEntity(1, 1)
                n_edges = self._boundary_edges.shape[0]
                edge_tags = list(
                    range(n_cells + 1, n_cells + n_edges + 1)
                )
                edge_nodes = (self._boundary_edges + 1).ravel().tolist()
                gmsh.model.mesh.addElements(
                    1, 1, [1], [edge_tags], [edge_nodes]
                )

                # Physical group for boundaries
                gmsh.model.addPhysicalGroup(1, [1], 1)
                gmsh.model.setPhysicalName(1, 1, "sides")

            # Physical group for domain
            gmsh.model.addPhysicalGroup(2, [1])
            gmsh.model.setPhysicalName(2, 1, "domain")

            gmsh.write(str(path))

        finally:
            gmsh.finalize()

    def to_firedrake_mesh(self, comm=None):
        """Convert to a Firedrake Mesh via an intermediate .msh file.

        Writes the mesh to a temporary .msh file and loads it using
        Firedrake's standard gmsh reader, which preserves physical group
        labels (boundary tags). This is MPI-safe: Firedrake's Mesh()
        handles parallel distribution internally.

        Args:
            comm: MPI communicator. Defaults to COMM_WORLD.

        Returns:
            Firedrake Mesh object with boundary labels preserved.

        Raises:
            MeshGenerationError: If mesh has not been generated yet.
        """
        if not self._generated:
            raise MeshGenerationError(
                "Mesh has not been generated yet. Call generate() first."
            )

        from firedrake import Mesh

        # Use the existing .msh file (written during generate) to preserve
        # physical groups and boundary labels. plex_from_cell_list would
        # lose all boundary tagging.
        if self._temp_path is not None and self._temp_path.exists():
            msh_path = self._temp_path
        else:
            # If the temp file was removed, reconstruct one
            fd, tmp = tempfile.mkstemp(suffix=".msh")
            msh_path = Path(tmp)
            self._write_from_arrays(msh_path, "msh2")

        kwargs = {}
        if comm is not None:
            kwargs["comm"] = comm

        return Mesh(str(msh_path), **kwargs)
