"""Layered geological model from scattered borehole interval data.

Boreholes log the subsurface as a stack of depth intervals, each tagged with a
categorical label (a geological formation, an aquifer name, a material class).
This module turns those sparse, scattered columns into a continuous layered
model that can be queried anywhere in 3D:

* :class:`IntervalObservations` is a plain value object holding the picks: for
  each interval, its horizontal position ``(x, y)``, its top and bottom
  elevation, and its label. Coordinates and elevations share whatever frame the
  caller supplies (e.g. local metres + m AHD); projection is not done here.
* :class:`LayerModel` holds, per layer, a top and a bottom boundary
  :class:`~omega.fields.surfaces.Surface` (always in elevation), then classifies
  any 3D point into a layer and maps layers to values. It has two constructors:
  :meth:`LayerModel.__init__` fits the boundaries from elevation picks, while
  :meth:`LayerModel.from_depths` builds them from a top surface plus depth-below-
  top surfaces -- the right model when an authoritative DEM owns the top and each
  layer boundary is logged as a depth measured down from it (sidestepping the
  datum mismatch between borehole and DEM elevations).

The fitted boundary surfaces are exposed directly (see
:meth:`LayerModel.boundary_elevation`) so a caller can inspect or extract one --
for instance the bottom of the lowest aquifer as a bedrock surface -- before any
mesh is built.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from omega.fields.surfaces import DEFAULT_K, GaussianKernelSurface, Surface


class _ConstantSurface(Surface):
    """A flat surface at a fixed elevation. Used internally for an open floor."""

    def __init__(self, value: float):
        self._value = float(value)

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        return np.full(len(np.asarray(xy)), self._value)


class IntervalObservations:
    """Scattered borehole interval picks with a categorical label.

    All five arrays are parallel and share a single length ``n``. Elevations and
    coordinates are in the caller's frame; ``top_elev`` is the shallower
    (higher) end of the interval and must be greater than or equal to
    ``bottom_elev``.

    Args:
        x: Horizontal x-coordinate of each interval, shape (n,).
        y: Horizontal y-coordinate of each interval, shape (n,).
        top_elev: Elevation of the interval top (shallower end), shape (n,).
        bottom_elev: Elevation of the interval bottom (deeper end), shape (n,).
        label: Categorical layer label for each interval, shape (n,).

    Raises:
        ValueError: If the arrays differ in length, are not 1-D, contain
            non-finite elevations, or have any inverted interval
            (``top_elev < bottom_elev``).
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        top_elev: np.ndarray,
        bottom_elev: np.ndarray,
        label: np.ndarray,
    ):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.top_elev = np.asarray(top_elev, dtype=float)
        self.bottom_elev = np.asarray(bottom_elev, dtype=float)
        # Normalise labels to str so lookups (subset, mapping) match `labels`.
        self.label = np.array([str(v) for v in np.asarray(label).ravel()], dtype=object)

        arrays = (self.x, self.y, self.top_elev, self.bottom_elev, self.label)
        if any(a.ndim != 1 for a in arrays):
            raise ValueError("all observation arrays must be 1-D")
        if len({len(a) for a in arrays}) != 1:
            raise ValueError("all observation arrays must have the same length")
        if len(self.x) == 0:
            raise ValueError("need at least one interval observation")
        if not np.all(np.isfinite(self.top_elev)) or not np.all(np.isfinite(self.bottom_elev)):
            raise ValueError("top_elev and bottom_elev must be finite")
        if np.any(self.top_elev < self.bottom_elev):
            raise ValueError("found inverted interval(s) with top_elev < bottom_elev")

    @property
    def n_points(self) -> int:
        """Number of interval observations."""
        return len(self.x)

    @property
    def labels(self) -> list[str]:
        """Unique labels, in first-appearance order."""
        return list(dict.fromkeys(str(v) for v in self.label))

    def subset(self, label: str) -> IntervalObservations:
        """Return the observations carrying ``label``."""
        mask = self.label == label
        if not np.any(mask):
            raise ValueError(f"no observations with label {label!r}")
        return IntervalObservations(
            self.x[mask],
            self.y[mask],
            self.top_elev[mask],
            self.bottom_elev[mask],
            self.label[mask],
        )

    def __repr__(self) -> str:
        return f"IntervalObservations(n_points={self.n_points}, labels={self.labels})"


def _aggregate_picks(obs: IntervalObservations) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse a label's intervals to one top/bottom pick per unique (x, y).

    A borehole may log the same label over several intervals; for boundary
    fitting we take the full vertical extent at each location (shallowest top,
    deepest bottom). Returns ``(coords_xy, top, bottom)``.
    """
    xy = np.column_stack([obs.x, obs.y])
    uniq, inverse = np.unique(xy, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    top = np.full(len(uniq), -np.inf)
    bottom = np.full(len(uniq), np.inf)
    np.maximum.at(top, inverse, obs.top_elev)
    np.minimum.at(bottom, inverse, obs.bottom_elev)
    return uniq, top, bottom


class LayerModel:
    """Continuous layered model: per-layer top and bottom elevation surfaces.

    Each layer carries a top and a bottom boundary
    :class:`~omega.fields.surfaces.Surface`, always in elevation. A 3D point is
    assigned to the first layer (in layer order, top to bottom) whose
    ``[bottom, top]`` band brackets its elevation; points above, below, or in a
    gap between bands are assigned to the nearest layer by vertical distance.

    Boundaries are fitted layer-by-layer and so, away from data, could in
    principle cross or overlap. With ``enforce_monotonic=True`` (the default) the
    bands are clamped at query time to a non-increasing, non-overlapping stack --
    each layer's band sits at or below the one above -- so the "first matching band
    wins" rule is well defined. Set it False to see the raw surfaces (diagnostics).

    Use :meth:`__init__` when the picks are absolute elevations; use
    :meth:`from_depths` when an authoritative top surface owns the surface and each
    boundary is a depth measured down from it.

    Args:
        observations: The borehole interval picks (elevations).
        layer_order: Layer labels ordered from the top of the sequence to the
            bottom. Every label must appear in ``observations``.
        sigma: Gaussian bandwidth (length scale, coordinate units) for the
            boundary surfaces (see :class:`GaussianKernelSurface`).
        k: Neighbour count for the boundary surfaces. Default: ``DEFAULT_K``.
        enforce_monotonic: Clamp bands to an ordered, non-overlapping stack at
            query time. Default: True.

    Raises:
        ValueError: If ``layer_order`` is empty or names a label with no
            observations.
    """

    def __init__(
        self,
        observations: IntervalObservations,
        layer_order: list[str],
        sigma: float,
        k: int = DEFAULT_K,
        enforce_monotonic: bool = True,
    ):
        if not layer_order:
            raise ValueError("layer_order must name at least one layer")
        order = [str(name) for name in layer_order]

        # One pick per unique borehole location (shallowest top, deepest bottom),
        # fitted into a top and a bottom elevation surface per layer.
        top_surfaces: dict[str, Surface] = {}
        bottom_surfaces: dict[str, Surface] = {}
        for name in order:
            coords, top, bottom = _aggregate_picks(observations.subset(name))
            top_surfaces[name] = GaussianKernelSurface(coords, top, sigma, k)
            bottom_surfaces[name] = GaussianKernelSurface(coords, bottom, sigma, k)
        self._init_bands(order, top_surfaces, bottom_surfaces, enforce_monotonic)

    @classmethod
    def from_depths(
        cls,
        top: Surface,
        layer_depths: dict[str, Surface],
        enforce_monotonic: bool = True,
        open_bottom: bool = True,
    ) -> LayerModel:
        """Build a model from a top surface and depth-below-top boundary surfaces.

        This is the model for "the DEM (or any authoritative surface) owns the top,
        and each layer boundary is a depth measured down from it" -- which
        sidesteps the datum mismatch between borehole and DEM elevations. Layer
        *i*'s elevation band runs from ``top - d[i-1]`` down to ``top - d[i]``,
        where ``d[i]`` is the depth to the base of layer *i* (``d[-1] = 0``, i.e.
        the first layer starts at the top surface). Adjacent layers therefore share
        a boundary exactly, forming a layer cake.

        Args:
            top: The top (ground/DEM) elevation surface.
            layer_depths: Ordered mapping of layer label to the surface giving the
                depth (positive, below ``top``) to the *base* of that layer, top to
                bottom. Insertion order is the layer order.
            enforce_monotonic: Clamp bands to an ordered stack at query time.
            open_bottom: If True (default), the deepest layer is an open half-space
                extending downward without bound, so any point below the stack is
                still that layer -- matching how the deepest logged formation is
                read geologically. Its depth surface is then unused for
                classification (it typically sets the mesh bottom separately). If
                False, the deepest layer is closed at ``top - d[last]`` and points
                below the stack fall to the nearest band.

        Returns:
            A new :class:`LayerModel`.

        Raises:
            ValueError: If ``layer_depths`` is empty.
            TypeError: If ``top`` or any depth is not a :class:`Surface`.
        """
        if not layer_depths:
            raise ValueError("layer_depths must name at least one layer")
        if not isinstance(top, Surface):
            raise TypeError("top must be a Surface")
        order = [str(name) for name in layer_depths]

        top_surfaces: dict[str, Surface] = {}
        bottom_surfaces: dict[str, Surface] = {}
        upper: Surface = top  # elevation of the current layer's top boundary
        last = len(order) - 1
        for i, (name, depth) in enumerate(zip(order, layer_depths.values(), strict=True)):
            if not isinstance(depth, Surface):
                raise TypeError(f"depth for layer {name!r} must be a Surface")
            top_surfaces[name] = upper
            if i == last and open_bottom:
                bottom_surfaces[name] = _ConstantSurface(-np.inf)
            else:
                bottom_surfaces[name] = top - depth
            upper = bottom_surfaces[name]

        obj = cls.__new__(cls)
        obj._init_bands(order, top_surfaces, bottom_surfaces, enforce_monotonic)
        return obj

    def _init_bands(
        self,
        order: list[str],
        top_surfaces: dict[str, Surface],
        bottom_surfaces: dict[str, Surface],
        enforce_monotonic: bool,
    ) -> None:
        self._order = order
        self._top = top_surfaces
        self._bottom = bottom_surfaces
        self._monotonic = enforce_monotonic

    @property
    def layer_order(self) -> list[str]:
        """Layer labels, top to bottom."""
        return list(self._order)

    def boundary_elevation(
        self,
        name: str,
        side: Literal["top", "bottom"],
        coords_xy: np.ndarray,
    ) -> np.ndarray:
        """Evaluate a layer boundary surface at horizontal points.

        Args:
            name: Layer label.
            side: ``"top"`` or ``"bottom"``.
            coords_xy: Horizontal query points, shape (m, 2) (extra columns
                ignored).

        Returns:
            Boundary elevation at each query point, shape (m,).
        """
        if name not in self._top:
            raise ValueError(f"unknown layer {name!r}; known: {self._order}")
        if side not in ("top", "bottom"):
            raise ValueError("side must be 'top' or 'bottom'")
        surface = self._top[name] if side == "top" else self._bottom[name]
        return surface(np.asarray(coords_xy, dtype=float))

    def _bands(self, coords_xy: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Evaluate every layer's top and bottom surface at the query points.

        With ``enforce_monotonic`` the bands are clamped top-to-bottom so no band
        overlaps the one above it (each layer's top sits at or below the previous
        layer's bottom, and each band's bottom at or below its own top).
        """
        xy = np.asarray(coords_xy, dtype=float)
        raw = {
            name: (self._top[name](xy), self._bottom[name](xy))
            for name in self._order
        }
        if not self._monotonic:
            return raw

        bands: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        ceiling: np.ndarray | None = None
        for name in self._order:
            top, bottom = raw[name]
            if ceiling is not None:
                top = np.minimum(top, ceiling)
            bottom = np.minimum(bottom, top)
            bands[name] = (top, bottom)
            ceiling = bottom
        return bands

    def classify(self, coords_xyz: np.ndarray) -> np.ndarray:
        """Assign each 3D point to a layer index (position in ``layer_order``).

        Args:
            coords_xyz: Query points, shape (m, 3) -- columns x, y, z (elevation).

        Returns:
            Integer layer index per point, shape (m,). Every point is assigned,
            so an index is not a containment guarantee: points inside a band take
            that layer, but points above/below the stack or in a gap take the
            nearest band by vertical distance. With ``enforce_monotonic`` the
            bands are ordered and non-overlapping, so the nearest-band choice is
            unambiguous.
        """
        pts = np.asarray(coords_xyz, dtype=float)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError("coords_xyz must have shape (m, 3)")
        z = pts[:, 2]

        # Bands depend only on (x, y). An extruded mesh repeats each column's (x, y)
        # once per layer, so we evaluate the (k-NN) surfaces on the unique columns
        # and broadcast back -- turning millions of node queries into ~thousands.
        uniq, inverse = np.unique(pts[:, :2], axis=0, return_inverse=True)
        inverse = inverse.ravel()
        bands_u = self._bands(uniq)
        bands = {name: (top[inverse], bottom[inverse]) for name, (top, bottom) in bands_u.items()}

        # Pass 1: containment, first matching layer (top to bottom) wins.
        assigned = np.full(len(pts), -1, dtype=int)
        for idx, name in enumerate(self._order):
            top, bottom = bands[name]
            inside = (z <= top) & (z >= bottom) & (assigned < 0)
            assigned[inside] = idx

        # Pass 2: points still in a gap take the nearest band by vertical distance.
        gap = assigned < 0
        if np.any(gap):
            dist = np.stack(
                [
                    np.maximum.reduce([bottom - z, z - top, np.zeros_like(z)])
                    for top, bottom in (bands[name] for name in self._order)
                ]
            )
            assigned[gap] = np.argmin(dist[:, gap], axis=0)
        return assigned

    def value_field(
        self,
        coords_xyz: np.ndarray,
        mapping: dict[str, float],
        default: float = 0.0,
    ) -> np.ndarray:
        """Map each 3D point to a value via its layer.

        Args:
            coords_xyz: Query points, shape (m, 3).
            mapping: Value for each layer label. Labels absent from the mapping
                fall back to ``default``.
            default: Value for layers not present in ``mapping``.

        Returns:
            Float value per point, shape (m,).
        """
        layer_idx = self.classify(coords_xyz)
        values_by_idx = np.array(
            [float(mapping.get(name, default)) for name in self._order],
            dtype=float,
        )
        return values_by_idx[layer_idx]
