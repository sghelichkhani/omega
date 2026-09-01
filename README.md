# OMEGA

Optimised Modelling of Environmental Groundwater in Australia.

OMEGA builds terrain-following extruded meshes for groundwater simulation with
[G-ADOPT](https://gadopt.org) and Firedrake. It covers four steps: a region
boundary as a polygon, a 2D triangular mesh from gmsh, a vertical extrusion that
follows the topography and the basement, and the interpolation of scattered
field data onto the result.

## Installation

```bash
pip install -e .
```

OMEGA needs `numpy`, `scipy`, `shapely`, `gmsh` and `firedrake`. Install it into
the same environment as Firedrake, because the mesh and field code imports
Firedrake directly.

## The one currency: a Surface

A `Surface` is a callable that maps horizontal position to a value:
`(m, 2) array -> (m,) array`. Every other part of the package composes on this
one contract. Two implementations exist:

- `GaussianKernelSurface` fits scattered points with a density-normalised
  Gaussian kernel over a `scipy.spatial.cKDTree` k-nearest-neighbour query.
- `GridSurface` samples a dense regular grid, for example a DEM raster.

Combinators make derived surfaces without leaving the abstraction:
`clamp_min`, `clamp_monotonic`, and the arithmetic operators.

```python
top = GaussianKernelSurface(dem_coords, dem_elevation, sigma=1500)
thickness = GaussianKernelSurface(bed_coords, bed_depth, sigma=5000).clamp_min(5.0)
bedrock = top - thickness
```

## Worked example

```python
from firedrake import FunctionSpace
from omega import (
    Polygon, SurfaceMesh, build_mesh_hierarchy,
    GaussianKernelSurface, LayerModel, assign_field, LocalFrame,
)

# 1. Fit the two surfaces that bound the domain in the vertical direction.
top = GaussianKernelSurface(dem_coords, dem_elevation, sigma=1500)
thickness = GaussianKernelSurface(bed_coords, bed_depth, sigma=5000).clamp_min(5.0)

# 2. Mesh the polygon in 2D, then extrude it into a multigrid hierarchy.
surface_mesh = SurfaceMesh(Polygon(polygon_coords), resolution=3500)
surface_mesh.generate()
hierarchy = build_mesh_hierarchy(
    surface_mesh.to_firedrake_mesh(), top, thickness, n_layers=300
)
mesh = hierarchy[-1]

# 3. Classify each node into a geological layer and bind the result to the mesh.
model = LayerModel.from_depths(top, {"upper": d_upper, "lower": d_lower})
V = FunctionSpace(mesh, "CG", 1)
formation = assign_field(V, lambda c: model.classify(c).astype(float))
```

`build_mesh_hierarchy` evaluates the surfaces on the coarse mesh nodes and
propagates the result through the hierarchy, so the geometry stays consistent
between multigrid levels and across MPI ranks. There is no intermediate
gridding step.

## Package layout

```
omega/
├── geometry/
│   ├── polygon.py       # Polygon loading, validation, simplification
│   └── crs.py           # LocalFrame: lon/lat <-> local-metre affine frame
├── mesh/
│   ├── surface.py       # SurfaceMesh: 2D triangular mesh from gmsh
│   ├── extrusion.py     # Vertical layer configuration
│   ├── transform.py     # Terrain-following coordinate mapping
│   └── builder.py       # build_mesh_hierarchy
├── fields/
│   ├── surfaces.py      # Surface, GaussianKernelSurface, GridSurface
│   ├── interpolation.py # FieldInterpolator: nearest and IDW escape hatch
│   ├── stratigraphy.py  # IntervalObservations, LayerModel
│   └── mesh_fields.py   # assign_field
└── io/                  # CSV readers, mesh writers
```

## Terrain-following transform

The extrusion coordinate `z` in the interval `[0, 1]` maps to a physical
elevation between the basement and the ground:

```
z_physical = bedrock + z * (elevation - bedrock)
```

At `z = 0` the node sits on the basement, and at `z = 1` it sits on the ground
surface. The layers therefore conform to the topography and to the basement
geometry at the same time.

## Boundary tags

After extrusion, Firedrake exposes `"top"` for the ground surface and
`"bottom"` for the basement. The numeric physical groups from the 2D mesh
identify the lateral boundaries.

## Demos

`demos/lower_murrumbidgee` builds a terrain-following mesh of the Lower
Murrumbidgee floodplain with a hydraulic-conductivity field, and writes a
Firedrake checkpoint. `demos/bom_water` pulls observed surface-water timeseries
from the Bureau of Meteorology. Each demo directory has its own README.

## Tests

```bash
pytest tests/
```

The tests that need Firedrake skip themselves when Firedrake is not
importable, so the geometry and surface tests run in a plain Python
environment.

## Licence

See `LICENSE`.
