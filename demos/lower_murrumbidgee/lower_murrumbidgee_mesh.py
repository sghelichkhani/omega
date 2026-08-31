"""OMEGA demo: Lower Murrumbidgee mesh + hydraulic-conductivity field.

A single G-ADOPT / Firedrake script built on OMEGA's composable abstractions,
with two external data sources:

  * ausdem   - the high-precision GA SRTM DEM, which IS the top surface. OMEGA's
               gmsh surface mesh is draped onto it and the extrusion hangs below.
  * austrata - NGIS borehole stratigraphy, which gives the DEPTH (below the
               surface) to each layer boundary. We never compare borehole
               elevations to the DEM; the DEM owns the top and every layer is
               measured downward from it, which sidesteps any datum mismatch.

Everything is one currency -- an :class:`omega.Surface` (a callable ``(x,y)->z``):

  * omega.LocalFrame          - the single lon/lat <-> local-metre georeference
  * omega.GaussianKernelSurface - the one interpolation primitive (cKDTree k-NN,
                                density-normalised). Fits the DEM AND each borehole
                                depth surface from scattered points; no gridding.
  * omega.SurfaceMesh         - 2D gmsh polygon mesh
  * omega.build_mesh_hierarchy - terrain-following extrusion from two Surfaces
                                (top = DEM, thickness = depth to bedrock)
  * omega.LayerModel.from_depths - depth-below-top layer model; classifies nodes
  * omega.assign_field        - bind the classifier onto the Firedrake mesh

Vertical model (depth d below the DEM surface), after the morrow2026 paper:

    0      <= d <= d_shep   Shepparton formation   Ks = 2.5e-05 m/s
    d_shep  < d <= d_cal    Calivil formation      Ks = 1.0e-03 m/s
    d_cal   < d            Upper Renmark group     Ks = 5.0e-04 m/s

d_shep / d_cal are the depths to the base of the upper formations, fitted as
Gaussian-kernel surfaces from the borehole picks. The Renmark is the open bottom
layer; the depth to its base sets the mesh bottom (the extrusion thickness).

Output: lower_murrumbidgee.pvd (+ .vtu) and murrumbidgee_dem.tif next to this
script. Open the .pvd in ParaView (use strong vertical exaggeration: the domain
is ~280 km wide but only ~650 m deep).

Run with the G-ADOPT / Firedrake interpreter, e.g.
    ~/Workplace/firedrake-2026-03-03/venv-firedrake/bin/python \
        demos/lower_murrumbidgee/lower_murrumbidgee_mesh.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

import ausdem
from austrata import NGISClient
from firedrake import Function, FunctionSpace, VTKFile

from omega import (
    GaussianKernelSurface,
    LayerModel,
    LocalFrame,
    Polygon,
    SurfaceMesh,
    assign_field,
    build_mesh_hierarchy,
    clamp_monotonic,
    node_coordinates,
)
from omega.io import save_mesh_and_functions

HERE = Path(__file__).parent

# --- one georeference for everything ------------------------------------------
# Local metric frame: origin 143.01 E / -35.76 S (the CSIRO/SKM full-grid SW
# corner, fitted to towns in full_modflow_domain.md), 91,800 m/deg lon,
# 110,170 m/deg lat. The active alluvium polygon sits offset inside that frame;
# this places it over the REAL Murrumbidgee (where the boreholes are), so the
# DEM box and the bores share one frame.
FRAME = LocalFrame(143.01, -35.76, 91_800.0, 110_170.0)

_ACTIVE = [
    (0, 35_000), (140_000, 0), (280_000, 0), (280_000, 68_000),
    (201_000, 130_000), (121_000, 130_000), (0, 100_000),
]
_OFFSET = (45_000, 73_800)
DOMAIN = [(x + _OFFSET[0], y + _OFFSET[1]) for x, y in _ACTIVE]

MODEL_FORMATIONS = ["Shepparton", "Calivil", "Renmark"]  # top to bottom
KS = {"Shepparton": 2.5e-05, "Calivil": 1.0e-03, "Renmark": 5.0e-04}  # m/s
FORMATION_ID = {"Shepparton": 1.0, "Calivil": 2.0, "Renmark": 3.0}

# Horizontal mesh size (m) and vertical layer count. Override from the shell to
# compare resolutions, e.g. OMEGA_DX=1000 OMEGA_NLAYERS=300 python ...
RESOLUTION = float(os.environ.get("OMEGA_DX", 1_500.0))
N_LAYERS = int(os.environ.get("OMEGA_NLAYERS", 150))
DEM_RES_DEG = 0.0025   # ~250 m DEM request
MIN_THICKNESS = 5.0    # floor on depth-to-bedrock so the mesh stays valid (m)

# Borehole picks are sparse and clustered: many bores sit within one mesh cell of
# each other and disagree by tens of metres. The Gaussian-kernel surface handles
# both -- it averages (so conflicting bores cancel rather than spiking) and is
# density-normalised (so a cluster of co-located bores cannot dominate by count).
# All we add at load is a gross-outlier clip; no binning/median is needed.
DEPTH_CLIP_PCT = 99.0  # drop depth picks above this percentile (gross outliers)

# Gaussian-kernel smoothing radius (sigma, m) for the borehole depth surfaces.
SIGMA = float(os.environ.get("OMEGA_SIGMA", 5_000.0))
# Neighbours per query for the (sparse) borehole surfaces. Clamped to the pick
# count, so a formation with few bores uses them all.
BORE_K = int(os.environ.get("OMEGA_BORE_K", 100))

# DEM smoothing length (sigma, m). There is no point carrying topographic detail
# finer than the mesh can represent, so the dense DEM is fed to the SAME Gaussian
# surface with sigma ~ the mesh size: the kernel regression IS the smoothing, and
# sampling onto mesh nodes is the same call. Defaults to the mesh size.
DEM_SMOOTH = float(os.environ.get("OMEGA_DEM_SIGMA", RESOLUTION))
# The DEM is dense, so its smoothing footprint spans many points; the neighbour
# count must cover ~3 sigma or the surface is smoothed only to the k-th-neighbour
# radius (not the full sigma) and warns about truncation. At ~250 m DEM spacing
# and sigma = DEM_SMOOTH, ~3 sigma holds roughly pi*(3*sigma/250)^2 points, so we
# size k to that (with a floor). Tune via OMEGA_DEM_K.
_DEM_SPACING = DEM_RES_DEG * FRAME.m_per_deg_lat  # ~ metres between DEM samples
# Points within ~3 sigma on a regular grid, with ~20% headroom so the discrete
# k-th neighbour sits safely past 3 sigma (and the truncation warning stays quiet).
DEM_K = int(os.environ.get(
    "OMEGA_DEM_K",
    max(400, int(1.2 * np.pi * (3.0 * DEM_SMOOTH / _DEM_SPACING) ** 2)),
))


def domain_lonlat_bbox():
    """Lon/lat bounding box of the local DOMAIN polygon, via the shared frame."""
    xs = [p[0] for p in DOMAIN]
    ys = [p[1] for p in DOMAIN]
    return FRAME.bbox_to_lonlat(min(xs), min(ys), max(xs), max(ys))


def fold_formation(unit):
    """Map an NGIS unit name onto a model formation, or None."""
    if not unit:
        return None
    if unit == "Shepparton Formation":
        return "Shepparton"
    if unit == "Calivil Formation":
        return "Calivil"
    if "Renmark" in unit:
        return "Renmark"
    return None


def fetch_dem_surface():
    """GA SRTM DEM over the domain as a Gaussian-kernel top surface.

    The dense DEM is returned as scattered (local-metre) points and wrapped in a
    GaussianKernelSurface with sigma = DEM_SMOOTH; the kernel regression both
    smooths to the mesh scale and samples onto any query point, so no separate
    grid filtering or re-interpolation happens.
    """
    bbox = domain_lonlat_bbox()
    print(f"DEM bbox (lon/lat): {tuple(round(v, 3) for v in bbox)}")
    dem = ausdem.get_dem(bbox, dataset="srtm_1s_dem", resolution=DEM_RES_DEG)
    ausdem.save_geotiff(dem, str(HERE / "murrumbidgee_dem.tif"))
    print(f"DEM grid: {dem.sizes['x']} x {dem.sizes['y']} cells; "
          f"smoothing to {DEM_SMOOTH:.0f} m via the kernel (sigma)")

    lon = dem["x"].to_numpy()
    lat = dem["y"].to_numpy()
    gx_lon, gy_lat = np.meshgrid(lon, lat)
    x, y = FRAME.to_local(gx_lon.ravel(), gy_lat.ravel())
    vals = dem.to_numpy().astype(float).ravel()
    finite = np.isfinite(vals)
    coords = np.column_stack([x[finite], y[finite]])
    return GaussianKernelSurface(coords, vals[finite], sigma=DEM_SMOOTH, k=DEM_K)


def boundary_depth_points(bores):
    """Depth (m below surface) to the base of each formation, as scattered picks.

    Returns {formation: (coords_local (n,2), depth (n,))}. The base of a
    formation at a borehole is the deepest logged interval bottom for that
    formation; gross outliers are clipped at load (the Gaussian surface handles
    the clustering and disagreement, so no binning/median is applied here).
    """
    bores.load_logs("stratigraphy")
    g = bores.stratigraphy_geodataframe()
    g = g[g["valid"] & g["bottom_depth_m"].notna()].copy()
    g["formation"] = g["unit"].map(fold_formation)
    g = g[g["formation"].notna()]

    lx, ly = FRAME.to_local(g["longitude"].to_numpy(), g["latitude"].to_numpy())
    g["lx"], g["ly"] = lx, ly
    inside = MplPath(DOMAIN).contains_points(g[["lx", "ly"]].to_numpy())
    g = g[inside]

    base = (
        g.groupby(["lx", "ly", "formation"])["bottom_depth_m"].max().reset_index()
    )
    out = {}
    for form in MODEL_FORMATIONS:
        sub = base[base["formation"] == form]
        xy = sub[["lx", "ly"]].to_numpy()
        depth = sub["bottom_depth_m"].to_numpy()
        if len(depth):
            keep = depth <= np.percentile(depth, DEPTH_CLIP_PCT)
            xy, depth = xy[keep], depth[keep]
        out[form] = (xy, depth)
        median = np.median(depth) if len(depth) else float("nan")
        print(f"  {form}: {len(depth)} picks (median {median:.0f} m)")
    return out


def main():
    dem_surface = fetch_dem_surface()

    ngis = NGISClient()
    bores = ngis.boreholes("NSW", bbox=domain_lonlat_bbox())
    print(f"NSW boreholes over the domain box: {len(bores)}")
    depths = boundary_depth_points(bores)
    raw_surfaces = {
        form: GaussianKernelSurface(coords, depth, sigma=SIGMA, k=BORE_K)
        for form, (coords, depth) in depths.items()
    }
    print(f"Gaussian smoothing radius sigma = {SIGMA:.0f} m")

    # The three base-depth surfaces are fitted independently and can cross where
    # bores disagree. Clamp them to a monotone (non-crossing) stack ONCE, so the
    # classifier and the mesh bottom use the SAME boundaries -- otherwise a raw
    # Renmark base shallower than the clamped Calivil base would put the mesh floor
    # above the Calivil base and leave zero Renmark cells.
    clamped = clamp_monotonic(
        [raw_surfaces[form] for form in MODEL_FORMATIONS], increasing=True
    )
    depth_surfaces = dict(zip(MODEL_FORMATIONS, clamped))

    # Depth-below-top layer model: the DEM owns the top, each formation is a depth
    # measured down from it. The Renmark is the open bottom layer (everything below
    # the Calivil base), so its depth surface is not used for classification -- it
    # sets the mesh bottom instead, as the extrusion thickness.
    model = LayerModel.from_depths(dem_surface, depth_surfaces, open_bottom=True)
    thickness = depth_surfaces["Renmark"].clamp_min(MIN_THICKNESS)

    # 2D gmsh surface mesh of the domain, extruded between DEM and bedrock.
    surface = SurfaceMesh(Polygon(DOMAIN), resolution=RESOLUTION)
    surface.generate()
    mesh_2d = surface.to_firedrake_mesh()
    hierarchy = build_mesh_hierarchy(
        mesh_2d,
        dem_surface,   # top surface = DEM
        thickness,     # depth to bedrock = extrusion thickness
        n_layers=N_LAYERS,
    )
    mesh = hierarchy[-1]
    print(f"Mesh: {mesh.num_cells()} cells, {N_LAYERS} layers")

    V = FunctionSpace(mesh, "CG", 1)
    # Formation index 0/1/2 at every node (classify collapses to the unique columns
    # internally, so the surfaces are evaluated once per column, not per node). The
    # conductivity is a pure lookup off that index.
    layer_idx = assign_field(V, lambda c: model.classify(c).astype(float), name="LayerIndex")
    idx = np.rint(layer_idx.dat.data_ro).astype(int)

    formation = Function(V, name="Formation")
    formation.dat.data[:] = np.array([FORMATION_ID[f] for f in MODEL_FORMATIONS])[idx]
    conductivity = Function(V, name="SaturatedConductivity")
    conductivity.dat.data[:] = np.array([KS[f] for f in MODEL_FORMATIONS])[idx]

    # Top-surface (DEM) elevation carried at every node: the z of the top of each
    # vertical column, broadcast down the column. The extrusion is monotonic, so the
    # column top is just the max node z per shared (x, y); depth below the surface at
    # any node is then SurfaceElevation - z. (Geometric, so no DEM re-query.)
    coords = node_coordinates(V)
    _, inverse = np.unique(coords[:, :2], axis=0, return_inverse=True)
    inverse = inverse.ravel()
    top_per_column = np.full(inverse.max() + 1, -np.inf)
    np.maximum.at(top_per_column, inverse, coords[:, 2])
    surface_elevation = Function(V, name="SurfaceElevation")
    surface_elevation.dat.data[:] = top_per_column[inverse]

    tag = f"{int(RESOLUTION)}m_{N_LAYERS}L"

    # Firedrake checkpoint (mesh + fields) for loading into a simulation.
    ckpt = HERE / f"lower_murrumbidgee_{tag}.h5"
    save_mesh_and_functions(
        mesh,
        {
            "Formation": formation,
            "SaturatedConductivity": conductivity,
            "SurfaceElevation": surface_elevation,
        },
        ckpt,
    )
    mesh_name = mesh.name if isinstance(mesh.name, str) else mesh.name()
    print(f"Wrote checkpoint {ckpt}  (mesh name: {mesh_name!r})")

    # VTK for viewing (skip with OMEGA_VTK=0 to save the big write at high res).
    if os.environ.get("OMEGA_VTK", "1") != "0":
        out = HERE / f"lower_murrumbidgee_{tag}.pvd"
        VTKFile(str(out)).write(formation, conductivity, surface_elevation)
        print(f"Wrote {out}")
    print("\n" + bores.citation())


if __name__ == "__main__":
    main()
