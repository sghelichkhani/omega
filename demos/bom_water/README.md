# BoM Water Data Online demo

Pulls observed surface-water timeseries from the Bureau of Meteorology Sensor
Observation Service (SOS2) and turns them into a point table ready to drive an
OMEGA groundwater simulation.

## What it does

`fetch_surface_water.py` requests quality-checked daily-mean watercourse
discharge for a lat/lon bounding box (set to the Lower Murrumbidgee here),
flattens the per-station xarray datasets into one `(lon, lat, value)` table,
and writes it to CSV. Those columns feed straight into
`omega.fields.FieldInterpolator` for mapping onto a mesh.

## Setup

`pybomwater` is cloned at `~/Workplace/pybomwater` and installed editable into
the default Python env. The first time you ever construct `BomWater()` it
downloads a capabilities + station cache to `~/pybomwater/cache`, which takes a
minute or two; after that it is fast.

```bash
~/Workplace/python3.12/bin/python3.12 fetch_surface_water.py
```

## Notes

- BoM expects bounding-box corners as `"lat lon"` strings in EPSG:4326,
  lower-left then upper-right.
- Other useful properties on the same service: `Water_Course_Level`,
  `Storage_Level`, `Storage_Volume`, `Ground_Water_Level`, `Rainfall`,
  `Evaporation`. Swap the `prop`/`proc` lines to fetch them.
- The BoM network is point gauges, so this is scattered data over the basin.
  Interior coverage is sparse, so interpolation onto a continuous surface field
  should be treated with care away from the rivers.
- `pybomwater` pins old `xarray`/`geopandas` in its metadata, but works fine
  with the current versions installed here, so those pins were overridden.
