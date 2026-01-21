"""Custom exceptions for the OMEGA package."""


class OmegaError(Exception):
    """Base exception for omega package."""

    pass


class PolygonError(OmegaError):
    """Invalid polygon geometry."""

    pass


class MeshGenerationError(OmegaError):
    """Mesh generation failed."""

    pass


class InterpolationError(OmegaError):
    """Field interpolation failed."""

    pass


class DataLoadError(OmegaError):
    """Failed to load data from file."""

    pass
