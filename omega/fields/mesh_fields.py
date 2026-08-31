"""Bind numpy-valued fields onto Firedrake function spaces.

This is the seam between OMEGA's pure-numpy field machinery (interpolators,
:class:`~omega.fields.stratigraphy.LayerModel`) and Firedrake. A *valuator* is
any callable that maps node coordinates -- shape ``(n_nodes, gdim)`` -- to a
1-D array of values; :func:`assign_field` evaluates it at the nodes of a
function space and returns a populated :class:`firedrake.Function`.

Firedrake is imported lazily inside the functions so the rest of the package
(and its tests) can be used without a Firedrake install.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def node_coordinates(function_space) -> np.ndarray:
    """Return the physical coordinates of a function space's nodes.

    Interpolates the mesh coordinate field into the vector-valued analogue of
    ``function_space``'s element, so the returned rows line up one-to-one with
    the degrees of freedom of a scalar :class:`firedrake.Function` on that
    space. This is the standard idiom for reading where each dof lives.

    Args:
        function_space: A scalar Firedrake ``FunctionSpace``.

    Returns:
        Node coordinates, shape (n_nodes, gdim), in mesh dof order.
    """
    from firedrake import VectorFunctionSpace, assemble, interpolate

    mesh = function_space.mesh()
    vector_space = VectorFunctionSpace(mesh, function_space.ufl_element())
    coords = assemble(interpolate(mesh.coordinates, vector_space))
    return np.asarray(coords.dat.data_ro)


def assign_field(
    function_space,
    valuator: Callable[[np.ndarray], np.ndarray],
    name: str | None = None,
):
    """Build a Firedrake ``Function`` by evaluating a valuator at the nodes.

    Args:
        function_space: Scalar Firedrake ``FunctionSpace`` to populate.
        valuator: Callable mapping node coordinates ``(n_nodes, gdim)`` to a
            1-D array of ``n_nodes`` values. Examples:
            :meth:`~omega.fields.stratigraphy.LayerModel.value_field`,
            :class:`~omega.fields.interpolation.FieldInterpolator`.
        name: Optional name for the resulting ``Function``.

    Returns:
        A :class:`firedrake.Function` on ``function_space`` carrying the values.

    Raises:
        ValueError: If the valuator returns the wrong number of values.
    """
    from firedrake import Function

    coords = node_coordinates(function_space)
    values = np.asarray(valuator(coords))
    if values.shape != (len(coords),):
        raise ValueError(
            f"valuator returned shape {values.shape}, expected ({len(coords)},)"
        )

    field = Function(function_space, name=name)
    field.dat.data[:] = values
    return field
