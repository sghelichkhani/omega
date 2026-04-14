"""Terrain-following coordinate transformations."""

from __future__ import annotations

import numpy as np


def apply_terrain_transform(
    mesh_coords: np.ndarray,
    elevation: np.ndarray,
    depth: np.ndarray,
) -> np.ndarray:
    """Transform normalized z-coordinates to terrain-following physical coordinates.

    Maps the normalized extrusion coordinate z in [0, 1] to physical elevations
    between bedrock and ground surface using the formula:

        z_physical = elevation - depth * (1 - z)

    Which is equivalent to:
        z_physical = depth * z + elevation - depth

    Where:
        - z = 0 corresponds to the bedrock surface (elevation - depth)
        - z = 1 corresponds to the ground surface (elevation)
        - depth is the thickness of the aquifer (distance from surface to bedrock)

    Args:
        mesh_coords: Mesh node coordinates, shape (n_nodes, 3).
            The z-values (column 2) are expected to be in [0, 1].
        elevation: Surface elevation at each node, shape (n_nodes,).
        depth: Depth to bedrock (aquifer thickness) at each node, shape (n_nodes,).
            This is a positive value representing the distance from surface to bedrock.

    Returns:
        Transformed z-coordinates, shape (n_nodes,).

    Raises:
        ValueError: If input shapes are incompatible.

    Example:
        >>> coords = mesh.coordinates.dat.data_ro
        >>> z_normalized = coords[:, 2]
        >>> elevation = interpolate_elevation(coords[:, :2])
        >>> depth = interpolate_depth(coords[:, :2])
        >>> new_z = apply_terrain_transform(coords, elevation, depth)
        >>> mesh.coordinates.dat.data[:, 2] = new_z
    """
    mesh_coords = np.asarray(mesh_coords)
    elevation = np.asarray(elevation)
    depth = np.asarray(depth)

    if mesh_coords.ndim != 2 or mesh_coords.shape[1] < 3:
        raise ValueError("mesh_coords must have shape (n_nodes, 3)")

    n_nodes = len(mesh_coords)
    if len(elevation) != n_nodes:
        raise ValueError(
            f"elevation length ({len(elevation)}) must match "
            f"number of nodes ({n_nodes})"
        )
    if len(depth) != n_nodes:
        raise ValueError(
            f"depth length ({len(depth)}) must match "
            f"number of nodes ({n_nodes})"
        )

    z_normalized = mesh_coords[:, 2]

    # Terrain-following transform matching the original gw_demos convention:
    # z_physical = depth * z + elevation - depth
    # Which simplifies to: z_physical = elevation - depth * (1 - z)
    # At z=0: z_physical = elevation - depth (bedrock elevation)
    # At z=1: z_physical = elevation (surface elevation)
    z_physical = depth * z_normalized + elevation - depth

    return z_physical


def apply_terrain_transform_inplace(
    mesh,
    elevation: np.ndarray,
    depth: np.ndarray,
) -> None:
    """Apply terrain-following transform to mesh coordinates in place.

    Modifies the mesh coordinate data directly. This is the recommended approach
    for Firedrake meshes as it preserves mesh topology and function space
    compatibility.

    Args:
        mesh: Firedrake mesh object with coordinates in normalized [0, 1] z-range.
        elevation: Surface elevation at each node.
        depth: Depth to bedrock (aquifer thickness) at each node.

    Example:
        >>> from firedrake import ExtrudedMesh, Mesh
        >>> mesh2d = Mesh("surface.msh")
        >>> mesh3d = ExtrudedMesh(mesh2d, layers=100, layer_height=0.01)
        >>> apply_terrain_transform_inplace(mesh3d, elevation, depth)
    """
    coords = mesh.coordinates.dat.data_ro
    new_z = apply_terrain_transform(coords, elevation, depth)
    mesh.coordinates.dat.data[:, 2] = new_z


def validate_terrain_data(
    elevation: np.ndarray,
    depth: np.ndarray,
    tolerance: float = 0.0,
) -> tuple[bool, str]:
    """Validate terrain data for mesh generation.

    Checks that:
    - Depth values are positive (within tolerance)
    - No NaN or infinite values

    Args:
        elevation: Surface elevation values.
        depth: Depth to bedrock values (should be positive).
        tolerance: Allowed tolerance for negative depth (default: 0.0).

    Returns:
        Tuple of (is_valid, message).
    """
    elevation = np.asarray(elevation)
    depth = np.asarray(depth)

    # Check for NaN/inf
    if np.any(~np.isfinite(elevation)):
        return False, "elevation contains NaN or infinite values"
    if np.any(~np.isfinite(depth)):
        return False, "depth contains NaN or infinite values"

    # Check depth is positive
    invalid_mask = depth < -tolerance
    if np.any(invalid_mask):
        n_invalid = np.sum(invalid_mask)
        min_depth = np.min(depth)
        return False, (
            f"{n_invalid} points have negative depth "
            f"(minimum depth: {min_depth:.2f})"
        )

    return True, "Terrain data is valid"
