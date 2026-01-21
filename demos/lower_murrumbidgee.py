"""
Lower Murrumbidgee Mesh Generation Demo

This script demonstrates using OMEGA to generate a terrain-following
extruded mesh for the Lower Murrumbidgee River Basin, equivalent to
the original gw_demos implementation.

Usage:
    python lower_murrumbidgee.py

The script will:
1. Define the basin polygon geometry
2. Generate a 2D surface mesh with gmsh
3. Extrude to 3D with 300 vertical layers
4. Apply terrain-following coordinate transform
5. Save the mesh to HDF5 for later use
"""

from pathlib import Path

from omega import MeshBuilder
from omega.io import save_mesh


# Path to data files (from gw_demos)
DATA_DIR = Path("/Users/sghelichkhani/Workplace/gw_demos/demos/groundwater/lower_murrumbidgee")


def main():
    # Define the Lower Murrumbidgee basin polygon
    # Coordinates in local projected CRS (meters)
    polygon = [
        (0, 35000),
        (140000, 0),
        (280000, 0),
        (280000, 68000),
        (201000, 130000),
        (121000, 130000),
        (0, 100000),
    ]

    # Mesh parameters (matching original demo)
    horizontal_resolution = 3500  # meters
    n_layers = 300

    print("Building Lower Murrumbidgee mesh...")
    print(f"  Horizontal resolution: {horizontal_resolution} m")
    print(f"  Vertical layers: {n_layers}")

    # Build the mesh using OMEGA
    mesh = (
        MeshBuilder(polygon)
        .set_horizontal_resolution(horizontal_resolution)
        .set_layer_heights(layer_heights=1/n_layers, n_layers=n_layers)
        .load_surface_elevation(DATA_DIR / "elevation_data.csv")
        .load_bedrock(DATA_DIR / "bedrock_data.csv")
        .build(validate_terrain=False)
    )

    # Print mesh statistics
    coords = mesh.coordinates.dat.data
    print(f"\nMesh generated successfully:")
    print(f"  Number of cells: {mesh.num_cells()}")
    print(f"  Number of nodes: {len(coords)}")
    print(f"  X range: [{coords[:, 0].min():.0f}, {coords[:, 0].max():.0f}] m")
    print(f"  Y range: [{coords[:, 1].min():.0f}, {coords[:, 1].max():.0f}] m")
    print(f"  Z range: [{coords[:, 2].min():.1f}, {coords[:, 2].max():.1f}] m")

    # Save mesh for later use
    output_path = Path(__file__).parent / "lower_murrumbidgee_mesh.h5"
    save_mesh(mesh, output_path)
    print(f"\nMesh saved to: {output_path}")

    return mesh


if __name__ == "__main__":
    main()
