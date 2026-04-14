"""Terrain-following extruded mesh hierarchy builder.

Provides a single function, build_mesh_hierarchy(), that takes a 2D Firedrake
mesh and terrain data, then produces a terrain-following extruded mesh
hierarchy suitable for multigrid solvers.
"""

from __future__ import annotations

import numpy as np

from omega.exceptions import MeshGenerationError


def build_mesh_hierarchy(
    mesh_2d,
    elevation_coords: np.ndarray,
    elevation_values: np.ndarray,
    depth_coords: np.ndarray,
    depth_values: np.ndarray,
    n_layers: int,
    refinement_levels: int = 0,
    refinement_ratio: int = 1,
    extrusion_power: float = 1.0,
    interpolation_method: str = "linear",
    validate_terrain: bool = True,
):
    """Build a terrain-following extruded mesh hierarchy.

    Takes a 2D Firedrake mesh and terrain data (elevation and depth-to-bedrock),
    builds a mesh hierarchy with optional refinement, applies vertical grading,
    interpolates terrain onto the coarsest level, prolongs to finer levels,
    and applies the terrain-following coordinate transform.

    The terrain transform maps normalised z in [0, 1] to physical coordinates:
        z_physical = bed * z + elevation - bed

    where bed = elevation - depth (the bedrock surface).

    Args:
        mesh_2d: Firedrake 2D Mesh object.
        elevation_coords: Source coordinates for elevation data, shape (n, 2).
        elevation_values: Surface elevation values, shape (n,).
        depth_coords: Source coordinates for depth data, shape (n, 2).
        depth_values: Depth-to-bedrock values (positive), shape (n,).
        n_layers: Number of vertical layers in the extrusion.
        refinement_levels: Number of uniform mesh refinement levels for the
            multigrid hierarchy. Default: 0 (no refinement, single mesh).
        refinement_ratio: Ratio of vertical refinement per level. Default: 1.
        extrusion_power: Power-law exponent for vertical grading. Values < 1
            cluster cells near the top (surface), values > 1 cluster near
            the bottom (bedrock). Default: 1.0 (uniform spacing).
        interpolation_method: Interpolation method for terrain data. Options:
            "linear" (scipy griddata), "nearest". Default: "linear".
        validate_terrain: If True, validate that depth values are positive
            and data contains no NaN/Inf. Default: True.

    Returns:
        Firedrake mesh hierarchy (HierarchyBase) with terrain-following
        coordinates applied at every level.

    Raises:
        MeshGenerationError: If terrain validation fails or mesh construction
            encounters an error.
        ValueError: If input arrays have incompatible shapes.
    """
    # Deferred Firedrake imports so the module can be imported without Firedrake
    from firedrake import (
        ExtrudedMeshHierarchy,
        Function,
        FunctionSpace,
        HierarchyBase,
        MeshHierarchy,
        prolong,
    )
    from firedrake.ufl_expr import as_vector, split
    from ufl import project

    # Validate input arrays
    elevation_coords = np.asarray(elevation_coords)
    elevation_values = np.asarray(elevation_values)
    depth_coords = np.asarray(depth_coords)
    depth_values = np.asarray(depth_values)

    if elevation_coords.ndim != 2 or elevation_coords.shape[1] != 2:
        raise ValueError(
            f"elevation_coords must have shape (n, 2), "
            f"got {elevation_coords.shape}"
        )
    if elevation_values.ndim != 1:
        raise ValueError(
            f"elevation_values must be 1-D, got shape {elevation_values.shape}"
        )
    if len(elevation_coords) != len(elevation_values):
        raise ValueError(
            f"elevation_coords ({len(elevation_coords)}) and "
            f"elevation_values ({len(elevation_values)}) length mismatch"
        )
    if depth_coords.ndim != 2 or depth_coords.shape[1] != 2:
        raise ValueError(
            f"depth_coords must have shape (n, 2), got {depth_coords.shape}"
        )
    if depth_values.ndim != 1:
        raise ValueError(
            f"depth_values must be 1-D, got shape {depth_values.shape}"
        )
    if len(depth_coords) != len(depth_values):
        raise ValueError(
            f"depth_coords ({len(depth_coords)}) and "
            f"depth_values ({len(depth_values)}) length mismatch"
        )

    # Optional terrain validation
    if validate_terrain:
        from omega.mesh.transform import validate_terrain_data

        is_valid, message = validate_terrain_data(
            elevation_values, depth_values, tolerance=1.0
        )
        if not is_valid:
            raise MeshGenerationError(f"Invalid terrain data: {message}")

    # Step 1: Build 2D mesh hierarchy
    mh2d = MeshHierarchy(mesh_2d, refinement_levels)

    # Step 2: Extrude to 3D hierarchy with normalised height [0, 1]
    mh3d = ExtrudedMeshHierarchy(
        mh2d,
        height=1.0,
        base_layer=n_layers,
        refinement_ratio=refinement_ratio,
    )

    # Step 3: Apply power-law vertical grading if requested
    if extrusion_power != 1.0:
        for mesh in mh3d:
            coords_old = mesh.coordinates
            coord_space = coords_old.function_space()
            x, y, s = split(coords_old)
            new_coords_expr = as_vector([x, y, s ** extrusion_power])
            new_coords_fn = project(new_coords_expr, coord_space)
            mesh.coordinates.assign(new_coords_fn)

    # Step 4: Interpolate terrain data on the coarsest mesh
    #
    # We interpolate depth-to-bedrock and elevation separately, matching the
    # original gw_demos convention. The variable names "bed" and "elv" follow
    # the original code where "bed" is the depth-to-bedrock (positive value),
    # NOT the bedrock elevation. The terrain transform at Step 6 uses:
    #   z_physical = bed * z + elv - bed
    # which gives: z=0 → elv - bed (bedrock elevation), z=1 → elv (surface).
    from scipy.interpolate import griddata

    V0 = FunctionSpace(mh3d[0], "CG", 1)
    coords_coarse = mh3d[0].coordinates
    xy_coarse = coords_coarse.dat.data_ro[:, :2]

    bed_interp = griddata(
        depth_coords,
        depth_values,
        xy_coarse,
        method=interpolation_method,
    )
    elv_interp = griddata(
        elevation_coords,
        elevation_values,
        xy_coarse,
        method=interpolation_method,
    )

    # Handle NaN from griddata (points outside convex hull) with nearest
    nan_mask_bed = np.isnan(bed_interp)
    nan_mask_elv = np.isnan(elv_interp)
    if np.any(nan_mask_bed):
        bed_interp[nan_mask_bed] = griddata(
            depth_coords,
            depth_values,
            xy_coarse[nan_mask_bed],
            method="nearest",
        )
    if np.any(nan_mask_elv):
        elv_interp[nan_mask_elv] = griddata(
            elevation_coords,
            elevation_values,
            xy_coarse[nan_mask_elv],
            method="nearest",
        )

    bed0 = Function(V0)
    bed0.dat.data[:] = bed_interp

    elv0 = Function(V0)
    elv0.dat.data[:] = elv_interp

    # Step 5: Prolong terrain fields from coarsest to finer levels
    beds = [bed0]
    elvs = [elv0]
    for lev in range(1, len(mh3d)):
        Vi = FunctionSpace(mh3d[lev], "CG", 1)
        bed = Function(Vi)
        prolong(beds[-1], bed)
        elv = Function(Vi)
        prolong(elvs[-1], elv)
        beds.append(bed)
        elvs.append(elv)

    # Step 6: Apply terrain-following transform at each level
    for lev, m in enumerate(mh3d):
        coords = m.coordinates
        z = coords.dat.data[:, 2]
        bed = beds[lev].dat.data[:]
        elv = elvs[lev].dat.data[:]
        coords.dat.data[:, 2] = bed * z + elv - bed

    # Step 7: Re-wrap as a proper hierarchy
    mh3d_moved = HierarchyBase(
        list(mh3d),
        mh3d.coarse_to_fine_cells,
        mh3d.fine_to_coarse_cells,
        refinements_per_level=mh3d.refinements_per_level,
        nested=mh3d.nested,
    )

    return mh3d_moved
