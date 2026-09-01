# Lower Murrumbidgee mesh demo

This demo builds a terrain-following 3D mesh of the Lower Murrumbidgee floodplain
together with a saturated–hydraulic–conductivity field, and writes it as a
Firedrake checkpoint ready to load into a groundwater simulation. It is the
worked example behind the Murrumbidgee case study in the Richards/Morrow paper.

Three packages cooperate. `ausdem` supplies the high-precision Geoscience
Australia SRTM digital elevation model, `austrata` supplies the Bureau of
Meteorology NGIS borehole stratigraphy, and `omega` does the meshing, the
terrain-following extrusion, and the field assignment. Run it with the
G-ADOPT / Firedrake interpreter:

```bash
python lower_murrumbidgee_mesh.py
```

The only file tracked here is the script (`lower_murrumbidgee_mesh.py`).
Running the script writes the rest next to it: the checkpoint
(`lower_murrumbidgee_1500m_150L.h5`), its VTK (`*.pvd` plus the matching
subdirectory) and the regenerated DEM (`murrumbidgee_dem.tif`). Those come to
several hundred megabytes, so they stay out of the repository. The separate
borehole/log
investigation scripts live in `borehole_investigation/`.

## The final output

`lower_murrumbidgee_1500m_150L.h5` is a Firedrake `CheckpointFile`: the extruded
mesh (1500 m horizontal triangles, 150 vertical layers) plus three CG1 fields,
`Formation` (1 Shepparton / 2 Calivil / 3 Renmark), `SaturatedConductivity` (m/s),
and `SurfaceElevation` (the top-of-column / DEM elevation in m, carried at every
node so depth below the surface is `SurfaceElevation - z`). Load it with:

```python
from firedrake import CheckpointFile

with CheckpointFile("lower_murrumbidgee_1500m_150L.h5", "r") as f:
    mesh = f.load_mesh("firedrake_default_extruded")
    Ks = f.load_function(mesh, "SaturatedConductivity")
    formation = f.load_function(mesh, "Formation")
    surface = f.load_function(mesh, "SurfaceElevation")
```

## The choices behind it

**One georeference, placed over the real alluvium.** Everything lives in a single
local metric frame: origin at 143.01°E / −35.76°S, scaled by 91,800 m per degree
of longitude and 110,170 m per degree of latitude. The active-alluvium polygon is
offset by (+45 km E, +73.8 km N) inside that frame so it sits over the true
Murrumbidgee, where the boreholes actually are. The DEM request box and the
borehole projection are both driven by this one transform, so topography and
geology line up. The fit and its town tie-points come from the
CSIRO/SKM (2010) groundwater modelling report for the Lower Murrumbidgee.
(An earlier convention that pinned the polygon directly
at the origin placed it ~45 × 74 km too far south-west; we do not use it.)

**The DEM owns the top; layers are depths measured down from it.** Rather than
compare borehole elevations against the DEM — two independent sources that never
agree to the metre — we take the DEM as the authoritative surface and measure
each geological boundary as a *depth below it*, straight from the borehole logs.
This sidesteps the datum mismatch entirely: a mesh node at elevation `z` has depth
`DEM(x,y) − z`, and that depth decides which formation it is in.

**Three formations, with the paper's conductivities.** Top to bottom the model is
the Shepparton formation (`Ks = 2.5e-5` m/s), the Calivil formation (`1e-3`), and
the Upper Renmark group (`5e-4`); below the Renmark is impermeable basement, which
is the mesh bottom. NGIS splits the Renmark into Upper/Middle/Lower, which we fold
back into one Renmark layer. The base of the Renmark sets the depth to bedrock and
hence the extrusion thickness.

**Robust, smooth depth surfaces from sparse clustered bores.** The borehole picks
are sparse and badly clustered — many bores sit within one mesh cell of each other
and disagree by tens of metres. Feeding those straight into a linear interpolation
produces near-vertical facets, i.e. spikes, which is what made the first meshes
look shredded. So the picks are first cleaned (drop the top-percentile outliers,
then bin to a 2 km grid and take the median per cell), and then turned into a
surface by **Gaussian-kernel regression**: every mesh point's depth is the
`exp(−d²/2σ²)`-weighted average of *all* the picks. Because it averages rather than
interpolates, conflicting bores cancel instead of spiking. There is no distance
cutoff and no nearest-neighbour fallback — far points simply get a weighted mean
dominated by the nearest data (made numerically stable with a max-log-weight
subtraction), so the surface is smooth and seamless everywhere. The smoothing
radius is `σ = 5 km` by default (`OMEGA_SIGMA`).

**The DEM is smoothed to the mesh scale.** There is no point carrying topographic
detail finer than the mesh can represent, so the dense DEM is Gaussian-filtered on
its grid to roughly the horizontal mesh size before it is sampled. The eastern
rise toward the ranges is *real* terrain and is left as-is; smoothing only removes
sub-cell roughness, it does not flatten genuine relief.

**Monotone layers.** The three depth surfaces are fitted independently, so they are
clamped at each node to a non-crossing stack (Shepparton ≤ Calivil ≤ bedrock)
before classification, so no layer can invert or poke through another.

## Known limitations

The bedrock (base of the Renmark) is the data-starved surface: the deep bores
cluster in the central-eastern alluvium, so the western third of the domain has
essentially no Renmark control and its bedrock is smooth extrapolation from distant
bores, not measurement — honest, but uncertain. Improving it would need a larger
smoothing radius, a regional trend, or an external depth-to-basement grid. This is
a capability demonstration, not a calibrated hydrogeological model.

## Regenerating at other resolutions

The resolution and smoothing are environment-configurable; outputs are tagged by
resolution so runs do not overwrite each other.

| variable | meaning | default |
|---|---|---|
| `OMEGA_DX` | horizontal mesh size (m) | 1500 |
| `OMEGA_NLAYERS` | vertical extrusion layers | 150 |
| `OMEGA_SIGMA` | borehole-surface smoothing radius (m) | 5000 |
| `OMEGA_BORE_K` | neighbours per query for borehole surfaces | 100 |
| `OMEGA_DEM_SIGMA` | DEM smoothing length (m) | = `OMEGA_DX` |
| `OMEGA_DEM_K` | neighbours per query for the DEM surface | sized to ~3σ |
| `OMEGA_VTK` | also write VTK for viewing (`0` to skip) | 1 |

The checkpoint and VTK here were produced with:

```bash
OMEGA_DX=1500 OMEGA_NLAYERS=150 python lower_murrumbidgee_mesh.py
```
