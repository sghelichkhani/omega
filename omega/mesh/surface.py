"""2D surface mesh generation using gmsh."""

from __future__ import annotations

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
    for extrusion into 3D.

    Args:
        polygon: Domain boundary polygon.
        resolution: Target element size in polygon coordinate units (e.g., meters).

    Example:
        >>> from omega.geometry import Polygon
        >>> polygon = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        >>> mesh = SurfaceMesh(polygon, resolution=10.0)
        >>> mesh_path = mesh.generate()
    """

    def __init__(self, polygon: Polygon, resolution: float):
        self._polygon = polygon
        self._resolution = resolution
        self._mesh_path: Path | None = None
        self._boundary_tags: dict[str, int] = {}

    @property
    def polygon(self) -> Polygon:
        """Return the domain polygon."""
        return self._polygon

    @property
    def resolution(self) -> float:
        """Return target mesh resolution."""
        return self._resolution

    @property
    def mesh_path(self) -> Path | None:
        """Return path to generated mesh file, or None if not yet generated."""
        return self._mesh_path

    @property
    def boundary_tags(self) -> dict[str, int]:
        """Return mapping of boundary names to physical group tags."""
        return self._boundary_tags.copy()

    def generate(
        self,
        output_path: str | Path | None = None,
        mesh_format: str = "msh2",
    ) -> Path:
        """Generate 2D mesh and write to file.

        Args:
            output_path: Path for output mesh file. If None, uses a temporary file.
            mesh_format: Gmsh mesh format version ('msh2' or 'msh4'). Default: 'msh2'.
                Use 'msh2' for best compatibility with Firedrake.

        Returns:
            Path to the generated mesh file.

        Raises:
            MeshGenerationError: If mesh generation fails.
        """
        if output_path is None:
            # Create temporary file
            fd, tmp_path = tempfile.mkstemp(suffix=".msh")
            output_path = Path(tmp_path)
        else:
            output_path = Path(output_path)

        try:
            self._generate_mesh(output_path, mesh_format)
            self._mesh_path = output_path
            return output_path

        except Exception as e:
            raise MeshGenerationError(f"Mesh generation failed: {e}") from e

    def _generate_mesh(self, output_path: Path, mesh_format: str) -> None:
        """Internal mesh generation using gmsh API."""
        gmsh.initialize()

        try:
            # Suppress terminal output
            gmsh.option.setNumber("General.Terminal", 0)

            # Set mesh format
            if mesh_format == "msh2":
                gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            elif mesh_format == "msh4":
                gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

            gmsh.model.add("surface_mesh")

            # Create points from polygon vertices
            coords = self._polygon.coords
            points = []
            for x, y in coords:
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

            # Synchronize geometry
            gmsh.model.geo.synchronize()

            # Create physical groups for boundaries and domain
            # All boundary edges grouped together as "sides"
            sides_tag = gmsh.model.addPhysicalGroup(1, lines, 1)
            gmsh.model.setPhysicalName(1, sides_tag, "sides")
            self._boundary_tags["sides"] = sides_tag

            # Domain surface
            domain_tag = gmsh.model.addPhysicalGroup(2, [surface])
            gmsh.model.setPhysicalName(2, domain_tag, "domain")
            self._boundary_tags["domain"] = domain_tag

            # Generate 2D mesh
            gmsh.model.mesh.generate(2)

            # Write mesh file
            gmsh.write(str(output_path))

        finally:
            gmsh.finalize()

    def get_mesh_info(self) -> dict:
        """Return information about the generated mesh.

        Returns:
            Dictionary with mesh statistics.

        Raises:
            MeshGenerationError: If mesh has not been generated yet.
        """
        if self._mesh_path is None:
            raise MeshGenerationError("Mesh has not been generated yet")

        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.open(str(self._mesh_path))

            # Get mesh statistics
            node_tags, _, _ = gmsh.model.mesh.getNodes()
            elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=2)

            n_nodes = len(node_tags)
            n_elements = sum(len(tags) for tags in elem_tags)

            return {
                "n_nodes": n_nodes,
                "n_elements": n_elements,
                "resolution": self._resolution,
                "mesh_path": str(self._mesh_path),
            }

        finally:
            gmsh.finalize()


def generate_surface_mesh(
    polygon: Polygon,
    resolution: float,
    output_path: str | Path | None = None,
) -> Path:
    """Convenience function to generate 2D surface mesh.

    Args:
        polygon: Domain boundary polygon.
        resolution: Target element size.
        output_path: Path for output mesh file. If None, uses a temporary file.

    Returns:
        Path to the generated mesh file.
    """
    mesh = SurfaceMesh(polygon, resolution)
    return mesh.generate(output_path)
