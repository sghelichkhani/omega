"""
BoM Water Data Online demo for OMEGA

Pulls observed surface-water timeseries (watercourse discharge) from the
Bureau of Meteorology Sensor Observation Service (SOS2) for a region of
interest, using the `pybomwater` package.

The idea is to build an archive of "water at the surface" that can later act
as a driving force / boundary condition for groundwater simulations. The BoM
network is a set of point gauges, so what we get back is scattered point data
(station lon/lat + a timeseries each). That is exactly the shape OMEGA's
FieldInterpolator (cKDTree) expects, so the last step here flattens the result
into an (x, y, value) table ready for interpolation onto a mesh.

Run:
    ~/Workplace/python3.12/bin/python3.12 fetch_surface_water.py

The first time you ever instantiate BomWater() it downloads a capabilities +
station cache to ~/pybomwater/cache, which takes a minute or two. After that
it is fast.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import pybomwater.bom_water as bw


# --- Region of interest -------------------------------------------------------
# Lower Murrumbidgee, NSW (same basin as the lower_murrumbidgee mesh demo).
# BoM expects EPSG:4326 corners as "lat lon" strings, lower-left then upper-right.
LOW_LEFT_LAT, LOW_LEFT_LON = -35.5, 143.5
UP_RIGHT_LAT, UP_RIGHT_LON = -33.5, 146.5

# --- What to fetch ------------------------------------------------------------
# Watercourse Discharge (streamflow), aggregated to daily means.
# Other useful properties: Water_Course_Level, Storage_Level, Storage_Volume,
# Ground_Water_Level, Rainfall, Evaporation.
T_BEGIN = "2015-01-01T00:00:00+10"
T_END = "2015-12-31T00:00:00+10"

OUTPUT_CSV = Path(__file__).parent / "lower_murrumbidgee_discharge_2015.csv"


def main():
    bom = bw.BomWater()  # first call builds the station cache (slow once)

    prop = bom.properties.Water_Course_Discharge
    proc = bom.procedures.Pat4_C_B_1_DailyMean  # quality-checked daily mean

    lower_corner = f"{LOW_LEFT_LAT} {LOW_LEFT_LON}"
    upper_corner = f"{UP_RIGHT_LAT} {UP_RIGHT_LON}"
    bbox = [lower_corner, upper_corner]

    print(f"Requesting {prop} ({proc})")
    print(f"  region : [{lower_corner}] -> [{upper_corner}]")
    print(f"  period : {T_BEGIN} -> {T_END}")

    # get_observations: finds stations in the bbox, then batches the timeseries
    # requests. Returns a list of dicts keyed by station URL -> xarray.Dataset.
    results = bom.get_observations(
        features=None,
        property=prop,
        procedure=proc,
        t_begin=T_BEGIN,
        t_end=T_END,
        bbox=bbox,
    )

    # Flatten the chunked list-of-dicts into one {station: Dataset} mapping.
    stations = {}
    for chunk in results:
        stations.update(chunk)

    if not stations:
        print("\nNo stations returned for this region/period. Try widening the "
              "bbox or the time window.")
        return

    print(f"\nGot {len(stations)} stations with data. Summary:\n")

    # Build a point table: one row per station with its coordinates and the
    # mean discharge over the period. This is the (x, y, value) form that
    # omega.fields.FieldInterpolator consumes.
    rows = []
    for url, ds in stations.items():
        # The discharge variable is the one column that is not Quality/Interpolation.
        value_vars = [v for v in ds.data_vars if v not in ("Quality", "Interpolation")]
        if not value_vars:
            continue
        varname = value_vars[0]
        series = ds[varname].to_pandas()

        # BoM attaches coordinates as a geojson [lon, lat] list (from the
        # bbox feature-of-interest response). Older paths can hand back a
        # "lat lon" string, so handle both.
        coords = ds.attrs.get("coordinates", "")
        lat = lon = np.nan
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            lon, lat = float(coords[0]), float(coords[1])
        elif isinstance(coords, str) and coords.strip():
            parts = coords.split()
            if len(parts) >= 2:
                lat, lon = float(parts[0]), float(parts[1])

        station_id = url.rsplit("/", 1)[-1]
        rows.append({
            "station_id": station_id,
            "name": ds.attrs.get("long_name", ""),
            "lon": lon,
            "lat": lat,
            "units": ds.attrs.get("units", ""),
            "n_obs": int(series.notna().sum()),
            "mean_discharge": float(series.mean(skipna=True)),
            "max_discharge": float(series.max(skipna=True)),
        })

    table = pd.DataFrame(rows).sort_values("mean_discharge", ascending=False)
    pd.set_option("display.width", 120)
    print(table.to_string(index=False))

    table.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote point table -> {OUTPUT_CSV}")
    print("Columns lon/lat/mean_discharge feed straight into "
          "omega.fields.FieldInterpolator for mapping onto a mesh.")


if __name__ == "__main__":
    main()
