"""High-level MeshBuilder API for terrain-following mesh generation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

from omega.exceptions import MeshGenerationError, OmegaError
from omega.fields.interpolation import FieldInterpolator
from omega.geometry.polygon import Polygon
from omega.io.readers import SpatialData, read_csv
from omega.mesh.extrusion import ExtrusionConfig
from omega.mesh.surface import SurfaceMesh
from omega.mesh.transform import (
    apply_terrain_transform_inplace,
    validate_terrain_data,
)

if TYPE_CHECKING:
    pass


class MeshBuilder:
    """High-level API for building terrain-following extruded meshes.

    Orchestrates the full mesh generation workflow:
    1. Define domain from polygon geometry
    2. Configure horizontal resolution and vertical layers
    3. Load elevation and bedrock data
    4. Generate 2D surface mesh
    5. Extrude to 3D with Firedrake
    6. Apply terrain-following coordinate transform

    Args:
        polygon: Domain boundary as Polygon object or list of (x, y) coordinates.
        crs: Coordinate reference system (e.g., "EPSG:28354"). Stored for metadata.

    Example:
        >>> from omega import MeshBuilder
        >>> polygon = [(0, 0), (280000, 0), (280000, 130000), (0, 130000)]
        >>> mesh = (
        ...     MeshBuilder(polygon)
        ...     .set_horizontal_resolution(3500)
        ...     .set_layer_heights(layer_height=1/300, n_layers=300)
        ...     .load_surface_elevation("elevation_data.csv")
        ...     .load_bedrock("bedrock_data.csv")
        ...     .build()
        ... )
    """

    def __init__(
        self,
        polygon: Polygon | Sequence[tuple[float, float]],
        crs: str | None = None,
    ):
        if isinstance(polygon, Polygon):
            self._polygon = polygon
        else:
            self._polygon = Polygon(list(polygon), crs=crs)

        self._crs = crs or self._polygon.crs

        # Configuration (set via builder methods)
        self._horizontal_resolution: float | None = None
        self._extrusion_config: ExtrusionConfig | None = None

        # Data (loaded via builder methods)
        self._elevation_data: SpatialData | None = None
        self._bedrock_data: SpatialData | None = None

        # Generated objects (created during build)
        self._surface_mesh: SurfaceMesh | None = None
        self._mesh = None  # Firedrake mesh

    @property
    def polygon(self) -> Polygon:
        """Return the domain polygon."""
        return self._polygon

    @property
    def crs(self) -> str | None:
        """Return coordinate reference system identifier."""
        return self._crs

    @property
    def is_configured(self) -> bool:
        """Return True if all required parameters are set."""
        return (
            self._horizontal_resolution is not None
            and self._extrusion_config is not None
            and self._elevation_data is not None
            and self._bedrock_data is not None
        )

    def set_horizontal_resolution(self, dx: float) -> MeshBuilder:
        """Set target horizontal mesh resolution.

        Args:
            dx: Target element size in coordinate units (e.g., meters).

        Returns:
            Self for method chaining.
        """
        if dx <= 0:
            raise ValueError("Resolution must be positive")
        self._horizontal_resolution = dx
        return self

    def set_layer_heights(
        self,
        layer_heights: float | list[float] | np.ndarray,
        n_layers: int | None = None,
    ) -> MeshBuilder:
        """Set vertical layer configuration.

        Args:
            layer_heights: Either a constant thickness or array of thicknesses.
                For normalized coordinates [0, 1], use layer_height=1/n_layers
                for uniform layers.
            n_layers: Number of layers (required if layer_heights is a scalar).

        Returns:
            Self for method chaining.
        """
        self._extrusion_config = ExtrusionConfig(
            layer_heights=layer_heights,
            n_layers=n_layers,
        )
        return self

    def set_extrusion_config(self, config: ExtrusionConfig) -> MeshBuilder:
        """Set vertical extrusion configuration directly.

        Args:
            config: ExtrusionConfig object.

        Returns:
            Self for method chaining.
        """
        self._extrusion_config = config
        return self

    def load_surface_elevation(
        self,
        path: str | Path,
        x_col: str = "x",
        y_col: str = "y",
        value_col: str = "z",
    ) -> MeshBuilder:
        """Load surface elevation data from CSV file.

        Args:
            path: Path to CSV file.
            x_col: Name of x-coordinate column.
            y_col: Name of y-coordinate column.
            value_col: Name of elevation value column.

        Returns:
            Self for method chaining.
        """
        self._elevation_data = read_csv(
            path,
            name="elevation",
            x_col=x_col,
            y_col=y_col,
            value_col=value_col,
        )
        return self

    def load_bedrock(
        self,
        path: str | Path,
        x_col: str = "x",
        y_col: str = "y",
        value_col: str = "z",
    ) -> MeshBuilder:
        """Load bedrock elevation data from CSV file.

        Args:
            path: Path to CSV file.
            x_col: Name of x-coordinate column.
            y_col: Name of y-coordinate column.
            value_col: Name of bedrock elevation value column.

        Returns:
            Self for method chaining.
        """
        self._bedrock_data = read_csv(
            path,
            name="bedrock",
            x_col=x_col,
            y_col=y_col,
            value_col=value_col,
        )
        return self

    def set_elevation_data(self, data: SpatialData) -> MeshBuilder:
        """Set surface elevation data directly.

        Args:
            data: SpatialData object with elevation values.

        Returns:
            Self for method chaining.
        """
        self._elevation_data = data
        return self

    def set_bedrock_data(self, data: SpatialData) -> MeshBuilder:
        """Set bedrock elevation data directly.

        Args:
            data: SpatialData object with bedrock elevation values.

        Returns:
            Self for method chaining.
        """
        self._bedrock_data = data
        return self

    def _validate_configuration(self) -> None:
        """Validate that all required parameters are set."""
        if self._horizontal_resolution is None:
            raise OmegaError(
                "Horizontal resolution not set. "
                "Call set_horizontal_resolution() first."
            )
        if self._extrusion_config is None:
            raise OmegaError(
                "Layer configuration not set. Call set_layer_heights() first."
            )
        if self._elevation_data is None:
            raise OmegaError(
                "Elevation data not loaded. Call load_surface_elevation() first."
            )
        if self._bedrock_data is None:
            raise OmegaError(
                "Bedrock data not loaded. Call load_bedrock() first."
            )

    def build(
        self,
        surface_mesh_path: str | Path | None = None,
        validate_terrain: bool = True,
        interpolation_method: str = "linear",
    ):
        """Build the terrain-following extruded mesh.

        Args:
            surface_mesh_path: Optional path to save the 2D surface mesh.
                If None, a temporary file is used.
            validate_terrain: If True, validate that elevation >= bedrock.
                Set to False to skip validation for problematic datasets.
            interpolation_method: Method for interpolating elevation/bedrock data.
                Options: "linear" (scipy.interpolate.griddata), "idw" (inverse
                distance weighting), "nearest" (nearest neighbor). Default: "linear".

        Returns:
            Firedrake mesh object with terrain-following coordinates.

        Raises:
            OmegaError: If required parameters are not set.
            MeshGenerationError: If mesh generation fails.
        """
        # Import Firedrake here to allow module import without Firedrake
        from firedrake import ExtrudedMesh, Mesh, VectorFunctionSpace, assemble, interpolate

        self._validate_configuration()

        # Step 1: Generate 2D surface mesh
        self._surface_mesh = SurfaceMesh(
            self._polygon, self._horizontal_resolution
        )
        mesh_path = self._surface_mesh.generate(output_path=surface_mesh_path)

        # Step 2: Load 2D mesh in Firedrake
        mesh2d = Mesh(str(mesh_path))

        # Step 3: Create extruded 3D mesh with normalized z in [0, 1]
        n_layers = self._extrusion_config.n_layers
        layer_height = self._extrusion_config.normalized_layer_height

        mesh3d = ExtrudedMesh(
            mesh2d,
            n_layers,
            layer_height=layer_height,
            extrusion_type="uniform",
            name="mesh",
        )

        # Step 4: Get mesh coordinates for interpolation
        W = VectorFunctionSpace(mesh3d, "CG", 1)
        X = assemble(interpolate(mesh3d.coordinates, W))
        mesh_coords = X.dat.data_ro

        # Step 5: Interpolate elevation and bedrock to mesh coordinates
        xy_coords = mesh_coords[:, :2]

        if interpolation_method == "linear":
            # Use scipy.interpolate.griddata (same as original demo)
            from scipy.interpolate import griddata

            elevation_values = griddata(
                self._elevation_data.coords,
                self._elevation_data.values,
                xy_coords,
                method="linear",
            )
            bedrock_values = griddata(
                self._bedrock_data.coords,
                self._bedrock_data.values,
                xy_coords,
                method="linear",
            )
        else:
            # Use cKDTree-based interpolation
            elevation_interp = FieldInterpolator.from_spatial_data(
                self._elevation_data
            )
            bedrock_interp = FieldInterpolator.from_spatial_data(
                self._bedrock_data
            )

            if interpolation_method == "idw":
                elevation_values = elevation_interp.idw(xy_coords, k=4)
                bedrock_values = bedrock_interp.idw(xy_coords, k=4)
            elif interpolation_method == "nearest":
                elevation_values = elevation_interp.nearest(xy_coords)
                bedrock_values = bedrock_interp.nearest(xy_coords)
            else:
                raise ValueError(
                    f"Unknown interpolation method: {interpolation_method}. "
                    f"Supported: 'linear', 'idw', 'nearest'"
                )

        # Step 6: Validate terrain data (optional)
        if validate_terrain:
            is_valid, message = validate_terrain_data(
                elevation_values, bedrock_values, tolerance=1.0
            )
            if not is_valid:
                raise MeshGenerationError(f"Invalid terrain data: {message}")

        # Step 7: Apply terrain-following coordinate transform
        apply_terrain_transform_inplace(mesh3d, elevation_values, bedrock_values)

        self._mesh = mesh3d
        return mesh3d

    def get_surface_mesh(self) -> SurfaceMesh | None:
        """Return the 2D surface mesh object (available after build)."""
        return self._surface_mesh

    def get_mesh_info(self) -> dict:
        """Return information about the built mesh.

        Returns:
            Dictionary with mesh configuration and statistics.
        """
        info = {
            "polygon_vertices": self._polygon.n_vertices,
            "polygon_bounds": self._polygon.bounds,
            "crs": self._crs,
            "horizontal_resolution": self._horizontal_resolution,
        }

        if self._extrusion_config:
            info["n_layers"] = self._extrusion_config.n_layers
            info["uniform_layers"] = self._extrusion_config.is_uniform

        if self._surface_mesh:
            info.update(self._surface_mesh.get_mesh_info())

        return info
