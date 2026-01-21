"""Mesh and data export utilities."""

from __future__ import annotations

from pathlib import Path


def save_mesh(mesh, path: str | Path) -> None:
    """Save mesh to HDF5 checkpoint file.

    Uses Firedrake's CheckpointFile for efficient parallel I/O.
    The mesh can be loaded later using load_mesh().

    Args:
        mesh: Firedrake mesh object.
        path: Path for output HDF5 file (typically .h5 extension).

    Example:
        >>> from omega.io import save_mesh, load_mesh
        >>> save_mesh(mesh, "output/mesh.h5")
        >>> # Later:
        >>> mesh = load_mesh("output/mesh.h5")
    """
    from firedrake import CheckpointFile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with CheckpointFile(str(path), "w") as f:
        f.save_mesh(mesh)


def load_mesh(path: str | Path, name: str = "mesh"):
    """Load mesh from HDF5 checkpoint file.

    Args:
        path: Path to HDF5 checkpoint file.
        name: Name of the mesh in the checkpoint file. Default: "mesh"
            (matches the name used by MeshBuilder).

    Returns:
        Firedrake mesh object.

    Example:
        >>> mesh = load_mesh("output/mesh.h5")
    """
    from firedrake import CheckpointFile

    with CheckpointFile(str(path), "r") as f:
        return f.load_mesh(name=name)


def save_function(function, path: str | Path, name: str | None = None) -> None:
    """Save Firedrake function to HDF5 checkpoint file.

    Args:
        function: Firedrake Function object.
        path: Path for output HDF5 file.
        name: Name for the function. If None, uses function.name().

    Example:
        >>> save_function(pressure_head, "output/fields.h5", name="pressure")
    """
    from firedrake import CheckpointFile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    func_name = name or function.name()

    with CheckpointFile(str(path), "w") as f:
        f.save_function(function, name=func_name)


def save_mesh_and_functions(
    mesh,
    functions: dict,
    path: str | Path,
) -> None:
    """Save mesh and multiple functions to a single HDF5 checkpoint file.

    Args:
        mesh: Firedrake mesh object.
        functions: Dictionary mapping names to Firedrake Function objects.
        path: Path for output HDF5 file.

    Example:
        >>> save_mesh_and_functions(
        ...     mesh,
        ...     {"elevation": elev_func, "bedrock": bed_func},
        ...     "output/model.h5"
        ... )
    """
    from firedrake import CheckpointFile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with CheckpointFile(str(path), "w") as f:
        f.save_mesh(mesh)
        for name, func in functions.items():
            f.save_function(func, name=name)


def load_function(path: str | Path, mesh, name: str):
    """Load Firedrake function from HDF5 checkpoint file.

    Args:
        path: Path to HDF5 checkpoint file.
        mesh: Firedrake mesh the function is defined on.
        name: Name of the function in the checkpoint file.

    Returns:
        Firedrake Function object.
    """
    from firedrake import CheckpointFile

    with CheckpointFile(str(path), "r") as f:
        return f.load_function(mesh, name=name)
