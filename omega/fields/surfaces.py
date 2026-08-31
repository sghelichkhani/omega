"""Scattered-data scalar surfaces, fitted with one primitive.

A :class:`Surface` is any callable mapping horizontal points ``(m, 2)`` to scalar
values ``(m,)``. It is the single currency the rest of OMEGA composes on: terrain
elevation, layer-boundary depths, and aquifer thickness are all surfaces, and the
mesh extrusion and stratigraphy layers consume them through this one contract.

The default fitter is :class:`GaussianKernelSurface` -- density-normalised
Gaussian kernel regression over scattered picks, evaluated through a
:class:`scipy.spatial.cKDTree` k-nearest-neighbour query. It is smooth and
seamless everywhere (no convex hull, no fill hack, no triangulation facets), it
declusters co-located picks instead of letting them dominate by count, and the
k-NN query keeps per-point cost bounded so a surface can be evaluated directly on
the nodes of a large mesh. Use it for scattered, clustered, mutually inconsistent
data: boreholes.

:class:`GridSurface` is the counterpart for sources that are already dense and
regular, such as a DEM or a modelled grid. It interpolates (piecewise-linear over
a Delaunay triangulation, nearest-neighbour outside the hull) rather than
averaging, because on a lattice finer than the mesh there is nothing to smooth
and a kernel mean would only flatten real relief. It is also the bit-exact route
to meshes built before this refactor, when the extrusion took raw coordinate and
value arrays and ran them through ``scipy.interpolate.griddata``.

Surfaces must be **pure, deterministic, and replicated on every MPI rank** (they
hold their full source picks and do no collective operations); this is what lets
``build_mesh_hierarchy`` evaluate them per-rank and still get identical geometry.

Combinators -- arithmetic, :func:`clamp_monotonic`, :meth:`Surface.clamp_min` --
let callers build derived surfaces (e.g. a bedrock elevation as ``top - thickness``,
or a non-crossing stack of layer boundaries) without leaving the abstraction.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
from scipy.spatial import cKDTree

from omega.exceptions import InterpolationError

#: Default neighbour count. Adequate for sparse borehole picks (it is clamped to
#: the pick count, so a layer with few bores uses them all). Dense sources whose
#: smoothing radius spans many points (e.g. a DEM) should pass a larger ``k``;
#: the surface warns when ``k`` truncates contributors within 3 sigma.
DEFAULT_K = 100

#: Chunk size for the per-pick density (rho) self-query at construction, so a dense
#: source does not allocate an (n_picks x k) neighbour array all at once.
_RHO_CHUNK = 50_000


class Surface(ABC):
    """A scalar field over horizontal space: ``(m, 2) -> (m,)``.

    Subclasses implement :meth:`__call__`. The base class provides arithmetic
    against other surfaces or scalars (``+``, ``-``, ``*``) and :meth:`clamp_min`,
    so derived surfaces compose without dropping back to raw arrays.
    """

    @abstractmethod
    def __call__(self, xy: np.ndarray) -> np.ndarray:
        """Evaluate the surface at horizontal points ``(m, 2)`` -> ``(m,)``."""

    def clamp_min(self, floor: float | Surface) -> Surface:
        """Return a surface clamped from below by ``floor`` (scalar or surface)."""
        return _BinOp(self, floor, np.maximum)

    def clamp_max(self, ceiling: float | Surface) -> Surface:
        """Return a surface clamped from above by ``ceiling`` (scalar or surface)."""
        return _BinOp(self, ceiling, np.minimum)

    def __add__(self, other: float | Surface) -> Surface:
        return _BinOp(self, other, np.add)

    def __sub__(self, other: float | Surface) -> Surface:
        return _BinOp(self, other, np.subtract)

    def __mul__(self, other: float | Surface) -> Surface:
        return _BinOp(self, other, np.multiply)

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other: float | Surface) -> Surface:
        return _BinOp(other, self, np.subtract)


def _as_values(obj: float | Surface, xy: np.ndarray) -> np.ndarray:
    """Evaluate ``obj`` at ``xy`` if it is a Surface, else broadcast the scalar."""
    if isinstance(obj, Surface):
        return obj(xy)
    return np.full(len(xy), float(obj))


class _BinOp(Surface):
    """A surface that applies a binary numpy op to two operands (surface/scalar)."""

    def __init__(
        self,
        left: float | Surface,
        right: float | Surface,
        op: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ):
        self._left = left
        self._right = right
        self._op = op

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        return self._op(_as_values(self._left, xy), _as_values(self._right, xy))


class GaussianKernelSurface(Surface):
    """Density-normalised Gaussian kernel regression over scattered picks.

    At a query point the value is the Gaussian-weighted mean of its ``k`` nearest
    picks, with each pick additionally down-weighted by the local pick density so
    that a cluster of co-located bores cannot dominate by sheer count. Weights are
    ``exp(-d^2 / 2 sigma^2) / rho``, where ``rho`` is the kernel density at the
    pick (computed once at construction); the per-query sum is stabilised with a
    max-log-weight subtraction so the nearest pick always has weight 1.

    Because it averages rather than interpolates, conflicting picks cancel instead
    of spiking, and there is no convex hull: far query points get a smooth
    weighted mean dominated by the nearest data. Two consequences worth knowing:

    * a weighted mean is bounded by its inputs and cannot reproduce an affine trend
      from few points, so genuine slopes are under-extrapolated where control is
      sparse (use :class:`~omega.fields.interpolation.FieldInterpolator.idw` if a
      slope-tracking surface is needed there);
    * declustering works only *within the k-NN window*. A cluster of co-located
      bores larger than ``k`` swallows the whole neighbour list, so a lone control
      point outside it contributes nothing -- not 1/N. This is surfaced by the
      construction-time truncation warning (raise ``k``, ideally past the largest
      expected cluster), not silently corrected. Symmetrically, an isolated pick
      sitting near a dense cluster has its own density inflated and is therefore
      down-weighted; "divide by local density" is symmetric and cannot tell a lone
      control point from redundant ones.

    Args:
        coords: Pick coordinates, shape (n, 2).
        values: Scalar value at each pick, shape (n,).
        sigma: Gaussian bandwidth (length scale) in coordinate units. A callable
            ``sigma(coords) -> array`` is accepted to reserve adaptive bandwidth as
            a future extension, but is not yet implemented.
        k: Number of nearest picks per query. Clamped to ``n``. Choose
            ``k > (picks within ~3 sigma)``; the surface warns once if ``k``
            truncates contributors (the k-th neighbour falls inside 3 sigma).

    Raises:
        InterpolationError: If shapes are inconsistent or there are no picks.
        NotImplementedError: If ``sigma`` is a callable (adaptive bandwidth).
    """

    def __init__(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        sigma: float | Callable[[np.ndarray], np.ndarray],
        k: int = DEFAULT_K,
    ):
        if callable(sigma):
            raise NotImplementedError("adaptive (callable) sigma is not yet implemented")

        self._coords = np.asarray(coords, dtype=float)
        self._values = np.asarray(values, dtype=float)
        self._sigma = float(sigma)

        if self._coords.ndim != 2 or self._coords.shape[1] != 2:
            raise InterpolationError("coords must have shape (n, 2)")
        if self._values.ndim != 1 or len(self._values) != len(self._coords):
            raise InterpolationError("values must be 1-D and match coords")
        if len(self._coords) == 0:
            raise InterpolationError("need at least one pick")
        if not np.all(np.isfinite(self._values)):
            raise InterpolationError("values must be finite (a single NaN poisons every query)")
        if not np.all(np.isfinite(self._coords)):
            raise InterpolationError("coords must be finite")
        if self._sigma <= 0:
            raise InterpolationError("sigma must be positive")

        n = len(self._coords)
        self._k = max(1, min(int(k), n))
        self._inv_two_s2 = 1.0 / (2.0 * self._sigma * self._sigma)
        self._tree = cKDTree(self._coords)

        # Per-pick kernel density rho_i = sum_j exp(-d_ij^2 / 2 sigma^2) over the
        # k nearest picks (self included, distance 0 -> weight 1, so rho >= 1).
        # Dividing each pick's query weight by rho declusters co-located picks.
        # Chunked so a dense source (e.g. a DEM, ~1e6 points) doesn't allocate an
        # (n x k) array at once. While here, count how many picks have their k-th
        # neighbour within 3 sigma -- the deterministic basis for the truncation
        # warning (done on the picks, so it is identical on every MPI rank, unlike a
        # query-time check over rank-local nodes).
        self._log_inv_rho = np.empty(n)
        n_truncated = 0
        for s in range(0, n, _RHO_CHUNK):
            dist, _ = self._query(self._coords[s : s + _RHO_CHUNK])
            self._log_inv_rho[s : s + _RHO_CHUNK] = -np.log(
                np.exp(-dist * dist * self._inv_two_s2).sum(axis=1)
            )
            n_truncated += int(np.count_nonzero(dist[:, -1] < 3.0 * self._sigma))
        self._warn_truncation(n_truncated / n)

    def _query(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """k-NN query returning ``(distances, indices)`` shaped ``(m, k)``.

        ``k`` is clamped to the pick count at construction, so the query never
        returns missing neighbours (no inf distances / out-of-range indices). The
        ``k == 1`` case is reshaped to 2-D for uniform downstream handling.
        """
        dist, idx = self._tree.query(xy, k=self._k)
        if self._k == 1:
            return dist[:, None], idx[:, None]
        return dist, idx

    def _warn_truncation(self, frac: float) -> None:
        """Warn if k truncates real contributors (k-th nearest pick < 3 sigma)."""
        if frac > 0.01:
            warnings.warn(
                f"GaussianKernelSurface: k={self._k} truncates contributors for "
                f"{frac:.0%} of picks (k-th nearest within 3 sigma). Increase k or "
                f"expect a biased k-subset mean (and clusters larger than k will "
                f"swallow the neighbour list).",
                stacklevel=2,
            )

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        if xy.ndim != 2 or xy.shape[1] < 2:
            raise InterpolationError("xy must have shape (m, >=2)")
        query = xy[:, :2]

        out = np.empty(len(query))
        chunk = 50_000  # cap the (chunk x k) weight arrays
        for s in range(0, len(query), chunk):
            dist, idx = self._query(query[s : s + chunk])
            # Density-corrected log weights, then a per-row stable softmax.
            logw = -dist * dist * self._inv_two_s2 + self._log_inv_rho[idx]
            logw -= logw.max(axis=1, keepdims=True)
            w = np.exp(logw)
            out[s : s + chunk] = (w * self._values[idx]).sum(axis=1) / w.sum(axis=1)
        return out

    def __repr__(self) -> str:
        return (
            f"GaussianKernelSurface(n={len(self._coords)}, "
            f"sigma={self._sigma:g}, k={self._k})"
        )


class GridSurface(Surface):
    """Piecewise-linear surface over a dense source, with a nearest-neighbour fill.

    A :class:`Surface` wrapper around Delaunay-based linear interpolation --
    exactly what ``scipy.interpolate.griddata(method="linear")`` performs -- with
    the points outside the convex hull filled from the nearest source point.

    This is the right primitive when the source is already dense and trustworthy:
    a DEM or a modelled grid sampled on a regular lattice finer than the mesh. In
    that regime the failure modes that motivate :class:`GaussianKernelSurface` do
    not arise. There are no co-located conflicting picks to decluster, and the
    triangulation facets that "spike" on disagreeing boreholes are, on a regular
    lattice, simply the bilinear surface the grid already implies. Averaging such
    a source instead of interpolating it only removes real relief: a Gaussian
    kernel is bounded by its inputs, so it flattens ridges and fills valleys at a
    scale set by sigma.

    Prefer :class:`GaussianKernelSurface` for scattered, clustered, mutually
    inconsistent picks (boreholes). Prefer this for dense gridded sources, and for
    reproducing meshes built before the Surface refactor: the interpolation here
    is the same call the array-based ``build_mesh_hierarchy`` used, so a mesh
    rebuilt through a ``GridSurface`` matches the original to the bit.

    The convex-hull fill is a genuine limitation, not a hidden fix. Outside the
    hull the value is piecewise constant, so a mesh extending past its terrain
    data gets flat steps rather than a smooth relaxation. Keep the source
    footprint larger than the mesh, or use :class:`GaussianKernelSurface`, which
    has no hull at all.

    Like every surface, this one is pure, deterministic and replicated on every
    MPI rank: it holds its own source arrays and does no collective operations.
    Qhull's triangulation is a deterministic function of the input points, so
    every rank builds the same one and evaluates identically. The triangulation is
    built lazily on first call and then cached, so constructing a surface that is
    never evaluated (a coarse level that a rank does not own) costs nothing.

    Args:
        coords: Source coordinates, shape (n, 2).
        values: Scalar value at each source point, shape (n,).
        fill: How to value query points outside the convex hull. ``"nearest"``
            (default) takes the nearest source value. ``"nan"`` leaves them NaN,
            which lets a caller detect that the mesh escaped its data rather than
            silently extrapolating.

    Raises:
        InterpolationError: If shapes are inconsistent, values are not finite,
            there are fewer than three points, or ``fill`` is not recognised.
    """

    def __init__(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        fill: str = "nearest",
    ):
        self._coords = np.asarray(coords, dtype=float)
        self._values = np.asarray(values, dtype=float)
        self._fill = str(fill)

        if self._coords.ndim != 2 or self._coords.shape[1] != 2:
            raise InterpolationError("coords must have shape (n, 2)")
        if self._values.ndim != 1 or len(self._values) != len(self._coords):
            raise InterpolationError("values must be 1-D and match coords")
        # A Delaunay triangulation needs three non-collinear points; below that
        # there is no linear patch to interpolate over.
        if len(self._coords) < 3:
            raise InterpolationError("need at least three points for a linear surface")
        if not np.all(np.isfinite(self._values)):
            raise InterpolationError("values must be finite (a single NaN poisons every query)")
        if not np.all(np.isfinite(self._coords)):
            raise InterpolationError("coords must be finite")
        if self._fill not in ("nearest", "nan"):
            raise InterpolationError(f"fill must be 'nearest' or 'nan', got {fill!r}")

        # Built on first evaluation; see the class docstring on laziness.
        self._linear = None
        self._nearest = None

    def _interpolators(self):
        """Build and cache the linear (and, if filling, nearest) interpolators.

        ``LinearNDInterpolator`` over a ``Delaunay`` of the source points is the
        same object ``griddata(method="linear")`` constructs internally, so the
        values match it exactly rather than merely closely.
        """
        if self._linear is None:
            from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

            self._linear = LinearNDInterpolator(self._coords, self._values)
            if self._fill == "nearest":
                self._nearest = NearestNDInterpolator(self._coords, self._values)
        return self._linear, self._nearest

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        if xy.ndim != 2 or xy.shape[1] < 2:
            raise InterpolationError("xy must have shape (m, >=2)")
        query = xy[:, :2]

        linear, nearest = self._interpolators()
        out = np.asarray(linear(query), dtype=float)

        # Outside the convex hull the linear interpolator returns NaN. Backfill
        # from the nearest source point unless the caller asked to see the gap.
        if nearest is not None:
            outside = np.isnan(out)
            if np.any(outside):
                out[outside] = nearest(query[outside])
        return out

    def __repr__(self) -> str:
        return f"GridSurface(n={len(self._coords)}, fill={self._fill!r})"


def clamp_monotonic(surfaces: list[Surface], increasing: bool = False) -> list[Surface]:
    """Clamp a list of surfaces to a non-crossing stack, in list order.

    Layer boundaries fitted independently can cross where data is sparse. This
    returns one derived surface per input that, at every query point, clamps the
    fitted values so the sequence is monotone in list order -- each member sitting
    at or beyond its predecessor.

    Args:
        surfaces: Surfaces in stack order.
        increasing: If False (default), each member is clamped to ``<=`` the one
            before it (elevations, top to bottom). If True, each is clamped to
            ``>=`` the one before it (depths, shallow to deep).

    Returns:
        A list of derived surfaces, parallel to the input.
    """
    if not surfaces:
        raise ValueError("clamp_monotonic needs at least one surface")
    return [_MonotoneMember(surfaces, i, increasing) for i in range(len(surfaces))]


class _MonotoneMember(Surface):
    """Member ``index`` of a monotone-clamped stack (see :func:`clamp_monotonic`)."""

    def __init__(self, surfaces: list[Surface], index: int, increasing: bool):
        self._surfaces = surfaces
        self._index = index
        self._increasing = increasing

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        clamp = np.maximum if self._increasing else np.minimum
        # Running fold: each surface is clamped against the clamped value above it.
        value = self._surfaces[0](xy)
        for surface in self._surfaces[1 : self._index + 1]:
            value = clamp(surface(xy), value)
        return value
