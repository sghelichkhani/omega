"""Data readers for spatial data files."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from omega.exceptions import DataLoadError


class SpatialData:
    """Container for spatial data with coordinates and values.

    Args:
        coords: Coordinate array of shape (n, 2) for 2D or (n, 3) for 3D.
        values: Value array of shape (n,).
        name: Optional name for the data field.
    """

    def __init__(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        name: str | None = None,
    ):
        self.coords = np.asarray(coords)
        self.values = np.asarray(values)
        self.name = name

        if self.coords.ndim != 2:
            raise ValueError("coords must be 2D array of shape (n, dim)")
        if self.values.ndim != 1:
            raise ValueError("values must be 1D array")
        if len(self.coords) != len(self.values):
            raise ValueError(
                f"coords and values must have same length, "
                f"got {len(self.coords)} and {len(self.values)}"
            )

    @property
    def n_points(self) -> int:
        """Number of data points."""
        return len(self.values)

    @property
    def ndim(self) -> int:
        """Spatial dimension of coordinates."""
        return self.coords.shape[1]

    def __repr__(self) -> str:
        return (
            f"SpatialData(n_points={self.n_points}, ndim={self.ndim}, "
            f"name={self.name!r})"
        )


class DataReader(Protocol):
    """Protocol for data readers."""

    def read(self, path: str | Path) -> SpatialData:
        """Read spatial data from file."""
        ...


class CSVReader:
    """Reader for CSV files with spatial data.

    Expected format: CSV with columns for x, y coordinates and a value column.
    Default column names are 'x', 'y', 'z' (where 'z' is the value).

    Args:
        x_col: Name of x-coordinate column. Default: 'x'.
        y_col: Name of y-coordinate column. Default: 'y'.
        value_col: Name of value column. Default: 'z'.
        delimiter: CSV delimiter. Default: ','.
    """

    def __init__(
        self,
        x_col: str = "x",
        y_col: str = "y",
        value_col: str = "z",
        delimiter: str = ",",
    ):
        self.x_col = x_col
        self.y_col = y_col
        self.value_col = value_col
        self.delimiter = delimiter

    def read(self, path: str | Path, name: str | None = None) -> SpatialData:
        """Read spatial data from CSV file.

        Args:
            path: Path to CSV file.
            name: Optional name for the data field.

        Returns:
            SpatialData object with coordinates and values.

        Raises:
            DataLoadError: If file cannot be read or is malformed.
        """
        path = Path(path)

        if not path.exists():
            raise DataLoadError(f"File not found: {path}")

        try:
            # Use numpy for efficient reading
            # First, read header to get column indices
            with open(path, "r", encoding="utf-8-sig") as f:
                header = f.readline().strip().split(self.delimiter)

            # Clean header (remove BOM if present, strip whitespace)
            header = [col.strip() for col in header]

            try:
                x_idx = header.index(self.x_col)
                y_idx = header.index(self.y_col)
                value_idx = header.index(self.value_col)
            except ValueError as e:
                raise DataLoadError(
                    f"Missing column in CSV. Expected columns: "
                    f"{self.x_col}, {self.y_col}, {self.value_col}. "
                    f"Found: {header}"
                ) from e

            # Load data, skipping header
            data = np.loadtxt(
                path,
                delimiter=self.delimiter,
                skiprows=1,
                usecols=[x_idx, y_idx, value_idx],
            )

            if data.ndim == 1:
                data = data.reshape(1, -1)

            coords = data[:, :2]
            values = data[:, 2]

            return SpatialData(coords, values, name=name or path.stem)

        except Exception as e:
            if isinstance(e, DataLoadError):
                raise
            raise DataLoadError(f"Failed to read CSV file {path}: {e}") from e


def read_csv(
    path: str | Path,
    name: str | None = None,
    x_col: str = "x",
    y_col: str = "y",
    value_col: str = "z",
) -> SpatialData:
    """Convenience function to read CSV spatial data.

    Args:
        path: Path to CSV file.
        name: Optional name for the data field.
        x_col: Name of x-coordinate column. Default: 'x'.
        y_col: Name of y-coordinate column. Default: 'y'.
        value_col: Name of value column. Default: 'z'.

    Returns:
        SpatialData object with coordinates and values.
    """
    reader = CSVReader(x_col=x_col, y_col=y_col, value_col=value_col)
    return reader.read(path, name=name)
