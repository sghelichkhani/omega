"""Terrain-following coordinate transformations."""

from __future__ import annotations

import numpy as np


def apply_terrain_transform(
    mesh_coords: np.ndarray,
    elevation: np.ndarray,
    bedrock: np.ndarray,
) -> np.ndarray:
    """Transform normalized z-coordinates to terrain-following physical coordinates.

    Maps the normalized extrusion coordinate z in [0, 1] to physical elevations
    between bedrock and ground surface:

        z_physical = bedrock + z * (elevation - bedrock)

    Where:
        - z = 0 corresponds to the bedrock surface
        - z = 1 corresponds to the ground surface (DEM elevation)

    Args:
        mesh_coords: Mesh node coordinates, shape (n_nodes, 3).
            The z-values (column 2) are expected to be in [0, 1].
        elevation: Surface elevation at each node, shape (n_nodes,).
        bedrock: Bedrock elevation at each node, shape (n_nodes,).

    Returns:
        Transformed z-coordinates, shape (n_nodes,).

    Raises:
        ValueError: If input shapes are incompatible.

    Example:
        >>> coords = mesh.coordinates.dat.data_ro
        >>> z_normalized = coords[:, 2]
        >>> elevation = interpolate_elevation(coords[:, :2])
        >>> bedrock = interpolate_bedrock(coords[:, :2])
        >>> new_z = apply_terrain_transform(coords, elevation, bedrock)
        >>> mesh.coordinates.dat.data[:, 2] = new_z
    """
    mesh_coords = np.asarray(mesh_coords)
    elevation = np.asarray(elevation)
    bedrock = np.asarray(bedrock)

    if mesh_coords.ndim != 2 or mesh_coords.shape[1] < 3:
        raise ValueError("mesh_coords must have shape (n_nodes, 3)")

    n_nodes = len(mesh_coords)
    if len(elevation) != n_nodes:
        raise ValueError(
            f"elevation length ({len(elevation)}) must match "
            f"number of nodes ({n_nodes})"
        )
    if len(bedrock) != n_nodes:
        raise ValueError(
            f"bedrock length ({len(bedrock)}) must match "
            f"number of nodes ({n_nodes})"
        )

    z_normalized = mesh_coords[:, 2]

    # Terrain-following transform
    # z_physical = bedrock + z * (elevation - bedrock)
    # This is equivalent to: bedrock * (1 - z) + elevation * z
    thickness = elevation - bedrock
    z_physical = bedrock + z_normalized * thickness

    return z_physical


def apply_terrain_transform_inplace(
    mesh,
    elevation: np.ndarray,
    bedrock: np.ndarray,
) -> None:
    """Apply terrain-following transform to mesh coordinates in place.

    Modifies the mesh coordinate data directly. This is the recommended approach
    for Firedrake meshes as it preserves mesh topology and function space
    compatibility.

    Args:
        mesh: Firedrake mesh object with coordinates in normalized [0, 1] z-range.
        elevation: Surface elevation at each node.
        bedrock: Bedrock elevation at each node.

    Example:
        >>> from firedrake import ExtrudedMesh, Mesh
        >>> mesh2d = Mesh("surface.msh")
        >>> mesh3d = ExtrudedMesh(mesh2d, layers=100, layer_height=0.01)
        >>> apply_terrain_transform_inplace(mesh3d, elevation, bedrock)
    """
    coords = mesh.coordinates.dat.data_ro
    new_z = apply_terrain_transform(coords, elevation, bedrock)
    mesh.coordinates.dat.data[:, 2] = new_z


def compute_layer_thickness(
    elevation: np.ndarray,
    bedrock: np.ndarray,
) -> np.ndarray:
    """Compute the total aquifer thickness at each point.

    Args:
        elevation: Surface elevation values.
        bedrock: Bedrock elevation values.

    Returns:
        Thickness values (elevation - bedrock).
    """
    return np.asarray(elevation) - np.asarray(bedrock)


def validate_terrain_data(
    elevation: np.ndarray,
    bedrock: np.ndarray,
    tolerance: float = 0.0,
) -> tuple[bool, str]:
    """Validate terrain data for mesh generation.

    Checks that:
    - Elevation is always >= bedrock (within tolerance)
    - No NaN or infinite values

    Args:
        elevation: Surface elevation values.
        bedrock: Bedrock elevation values.
        tolerance: Allowed tolerance for elevation < bedrock (default: 0.0).

    Returns:
        Tuple of (is_valid, message).
    """
    elevation = np.asarray(elevation)
    bedrock = np.asarray(bedrock)

    # Check for NaN/inf
    if np.any(~np.isfinite(elevation)):
        return False, "elevation contains NaN or infinite values"
    if np.any(~np.isfinite(bedrock)):
        return False, "bedrock contains NaN or infinite values"

    # Check elevation >= bedrock
    thickness = elevation - bedrock
    invalid_mask = thickness < -tolerance
    if np.any(invalid_mask):
        n_invalid = np.sum(invalid_mask)
        min_thickness = np.min(thickness)
        return False, (
            f"{n_invalid} points have elevation below bedrock "
            f"(minimum thickness: {min_thickness:.2f})"
        )

    return True, "Terrain data is valid"
