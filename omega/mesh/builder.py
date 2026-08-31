"""Terrain-following extruded mesh hierarchy builder.

Provides a single function, build_mesh_hierarchy(), that takes a 2D Firedrake
mesh and two terrain :class:`~omega.fields.surfaces.Surface` objects (the top
surface and the depth-to-bedrock thickness), then produces a terrain-following
extruded mesh hierarchy suitable for multigrid solvers.
"""

from __future__ import annotations

import numpy as np

from omega.exceptions import MeshGenerationError
from omega.fields.surfaces import Surface


def build_mesh_hierarchy(
    mesh_2d,
    top_surface: Surface,
    thickness_surface: Surface,
    n_layers: int,
    refinement_levels: int = 0,
    refinement_ratio: int = 1,
    extrusion_power: float = 1.0,
    validate_terrain: bool = True,
):
    """Build a terrain-following extruded mesh hierarchy.

    Takes a 2D Firedrake mesh and two terrain surfaces -- the top (ground/DEM)
    elevation and the depth-to-bedrock thickness -- builds a mesh hierarchy with
    optional refinement, applies vertical grading, evaluates the surfaces on the
    coarsest level, prolongs to finer levels, and applies the terrain-following
    coordinate transform.

    The transform maps normalised z in [0, 1] to physical coordinates:
        z_physical = thickness * z + top - thickness

    so z=1 lands on the top surface and z=0 on the bedrock (``top - thickness``).

    Surfaces are evaluated directly on the coarse mesh nodes (per MPI rank), so no
    intermediate gridding or re-interpolation happens here -- whatever fit the
    caller built (e.g. a Gaussian-kernel surface) is honoured exactly. The surface
    contract (pure, deterministic, replicated on every rank) is what makes the
    per-rank evaluation produce identical geometry everywhere.

    Terrain is sampled on the coarsest level and prolonged to finer levels (so the
    levels stay geometrically nested for multigrid). Fine-level geometry is
    therefore the prolonged coarse terrain, not the surface re-evaluated at fine
    resolution: to resolve terrain at the finest level, build the hierarchy from a
    finer base mesh rather than relying on refinement.

    Args:
        mesh_2d: Firedrake 2D Mesh object.
        top_surface: Surface giving the ground/DEM elevation at ``(x, y)``.
        thickness_surface: Surface giving the depth to bedrock (a positive
            thickness) at ``(x, y)``. Apply any minimum-thickness floor before
            passing it in, e.g. ``surface.clamp_min(5.0)``.
        n_layers: Number of vertical layers in the extrusion.
        refinement_levels: Number of uniform mesh refinement levels for the
            multigrid hierarchy. Default: 0 (no refinement, single mesh).
        refinement_ratio: Ratio of vertical refinement per level. Default: 1.
        extrusion_power: Power-law exponent for vertical grading. Values < 1
            cluster cells near the top (surface), values > 1 cluster near
            the bottom (bedrock). Default: 1.0 (uniform spacing).
        validate_terrain: If True, validate (after sampling on the coarse nodes)
            that the thickness is positive and the terrain has no NaN/Inf.
            Default: True.

    Returns:
        Firedrake mesh hierarchy (HierarchyBase) with terrain-following
        coordinates applied at every level.

    Raises:
        MeshGenerationError: If terrain validation fails or mesh construction
            encounters an error.
        TypeError: If the surfaces are not :class:`Surface` instances.
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
    from firedrake import project
    from ufl import as_vector, split

    if not isinstance(top_surface, Surface):
        raise TypeError("top_surface must be a Surface")
    if not isinstance(thickness_surface, Surface):
        raise TypeError("thickness_surface must be a Surface")

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

    # Step 4: Evaluate the terrain surfaces on the coarsest mesh nodes. "thk" is
    # the depth-to-bedrock (positive); "top" is the surface elevation. The
    # transform at Step 6 uses z_physical = thk * z + top - thk, giving z=0 ->
    # top - thk (bedrock elevation) and z=1 -> top (surface).
    V0 = FunctionSpace(mh3d[0], "CG", 1)
    xy_coarse = mh3d[0].coordinates.dat.data_ro[:, :2]

    thk_values = thickness_surface(xy_coarse)
    top_values = top_surface(xy_coarse)

    if validate_terrain:
        from omega.mesh.transform import validate_terrain_data

        is_valid, message = validate_terrain_data(top_values, thk_values, tolerance=1.0)
        if not is_valid:
            raise MeshGenerationError(f"Invalid terrain data: {message}")

        # A thickness at or below zero passes the tolerance check but collapses the
        # column (every layer maps to the top), silently degrading the mesh. Warn
        # so a caller who forgot a clamp_min floor finds out before a big build.
        min_thk = float(np.min(thk_values))
        if min_thk <= 0.0:
            import warnings

            warnings.warn(
                f"thickness reaches {min_thk:.3g} <= 0 at some node; those columns "
                f"collapse to zero height. Pass thickness.clamp_min(floor) to avoid "
                f"degenerate cells.",
                stacklevel=2,
            )

    thk0 = Function(V0)
    thk0.dat.data[:] = thk_values

    top0 = Function(V0)
    top0.dat.data[:] = top_values

    # Step 5: Prolong terrain fields from coarsest to finer levels (keeps the
    # levels geometrically nested for multigrid, rather than re-sampling each).
    thks = [thk0]
    tops = [top0]
    for lev in range(1, len(mh3d)):
        Vi = FunctionSpace(mh3d[lev], "CG", 1)
        thk = Function(Vi)
        prolong(thks[-1], thk)
        top = Function(Vi)
        prolong(tops[-1], top)
        thks.append(thk)
        tops.append(top)

    # Step 6: Apply terrain-following transform at each level
    for lev, m in enumerate(mh3d):
        coords = m.coordinates
        z = coords.dat.data[:, 2]
        thk = thks[lev].dat.data[:]
        top = tops[lev].dat.data[:]
        coords.dat.data[:, 2] = thk * z + top - thk

    # Step 7: Re-wrap as a proper hierarchy
    mh3d_moved = HierarchyBase(
        list(mh3d),
        mh3d.coarse_to_fine_cells,
        mh3d.fine_to_coarse_cells,
        refinements_per_level=mh3d.refinements_per_level,
        nested=mh3d.nested,
    )

    return mh3d_moved
