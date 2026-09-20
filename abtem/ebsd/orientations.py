"""Zone-axis sampling and the rotated slabs that go with them.

A reference pattern covers the whole hemisphere of outgoing directions, which is
far more than one multislice run can resolve: the calculation is only valid for
directions close to the beam direction it was set up for. So the hemisphere is
tiled into patches, each with its own zone axis, and each patch gets its own
slab of crystal cut with that zone axis along ``z``.

This module provides the two halves of that: :func:`fibonacci_hemisphere`
chooses the patch centres, and :func:`rotated_slab` cuts the slab for one of
them.

.. warning::
    :func:`rotated_slab` cuts a box out of a rotated bulk crystal, which for a
    general zone axis is **not** periodic across the box faces. The multislice
    algorithm nevertheless applies periodic boundary conditions to it, so each
    patch carries an artificial discontinuity at its edges. In practice the
    patches are small and the resulting artefacts average out across the
    hemisphere, but this is an approximation rather than an exact calculation.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

__all__ = [
    "fibonacci_hemisphere",
    "zone_axis_rotation",
    "estimate_repetitions",
    "bulk_block",
    "rotated_slab",
]


def fibonacci_hemisphere(n: int) -> np.ndarray:
    """Evenly distributed directions on the northern hemisphere.

    Uses a Fibonacci lattice, which spaces the points near-uniformly in solid
    angle without the clustering a latitude/longitude grid produces at the pole.

    Parameters
    ----------
    n : int
        Number of directions to generate.

    Returns
    -------
    directions : np.ndarray
        Unit vectors of shape ``(n, 3)``, ordered from the pole down towards
        the equator.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")

    i = np.arange(n)
    golden_ratio_inverse = (np.sqrt(5.0) - 1.0) / 2.0

    # Halving the step compared to a full sphere keeps z in (0, 1).
    z = 1.0 - (2.0 * i + 1.0) / (2.0 * n)
    r = np.sqrt(1.0 - z**2)
    theta = 2.0 * np.pi * i * golden_ratio_inverse

    return np.column_stack((r * np.cos(theta), r * np.sin(theta), z))


def zone_axis_rotation(zone_axis: np.ndarray) -> np.ndarray:
    """Rotation matrix taking a zone axis onto ``+z``.

    The returned matrix ``R`` maps the crystal frame to the slab frame, so a
    direction transforms as ``R @ d`` and a stack of row vectors as
    ``directions @ R.T``. Its transpose maps back.

    This is the same rotation :meth:`ase.Atoms.rotate` applies for
    ``atoms.rotate(zone_axis, "z")``, built explicitly so the transform is
    available without going through an :class:`~ase.Atoms` object.

    Parameters
    ----------
    zone_axis : np.ndarray
        Beam direction in the crystal frame, of shape ``(3,)``. Need not be
        normalized.

    Returns
    -------
    rotation : np.ndarray
        Rotation matrix of shape ``(3, 3)``.
    """
    zone_axis = np.asarray(zone_axis, dtype=float).ravel()

    if zone_axis.shape != (3,):
        raise ValueError(f"zone_axis must have shape (3,), got {zone_axis.shape}")

    norm = np.linalg.norm(zone_axis)
    if norm == 0.0:
        raise ValueError("zone_axis must be nonzero")

    v = zone_axis / norm
    target = np.array([0.0, 0.0, 1.0])

    cos_angle = float(np.dot(v, target))
    axis = np.cross(v, target)
    sin_angle = float(np.linalg.norm(axis))

    if sin_angle < 1e-7:
        # Parallel or antiparallel to z: any perpendicular axis will do, and
        # ASE picks them in this order.
        axis = np.cross((0.0, 0.0, 1.0), target)
        if np.linalg.norm(axis) < 1e-7:
            axis = np.cross((1.0, 0.0, 0.0), target)
        axis = axis / np.linalg.norm(axis)
    else:
        axis = axis / sin_angle

    # Rodrigues' rotation formula in matrix form.
    cross_matrix = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )

    return (
        cos_angle * np.eye(3)
        + sin_angle * cross_matrix
        + (1.0 - cos_angle) * np.outer(axis, axis)
    )


def estimate_repetitions(
    atoms: Atoms, cell: tuple[float, float, float]
) -> tuple[int, int, int]:
    """Repetitions of `atoms` needed to fill `cell` at any orientation.

    The slab is cut from a bulk block after rotation, so the block has to
    contain the slab's bounding sphere whichever way it is turned. The longest
    diagonal of a ``(a, b, c)`` box is ``sqrt(a**2 + b**2 + c**2)``, so the
    block must span at least that along every axis.

    Parameters
    ----------
    atoms : ase.Atoms
        The unit cell to repeat.
    cell : tuple of three float
        Dimensions of the slab to be cut [Å].

    Returns
    -------
    repetitions : tuple of three int
    """
    diagonal = float(np.linalg.norm(cell))
    lengths = np.asarray(atoms.cell.lengths(), dtype=float)

    if np.any(lengths <= 0.0):
        raise ValueError("atoms must have a cell with nonzero lengths")

    return tuple(int(np.ceil(diagonal / length)) for length in lengths)  # type: ignore[return-value]


def bulk_block(
    atoms: Atoms,
    cell: tuple[float, float, float],
    repetitions: tuple[int, int, int] | None = None,
) -> Atoms:
    """The repeated block that :func:`rotated_slab` cuts a slab out of.

    Building the block is the expensive part of cutting a slab, and it does not
    depend on the zone axis, so a reference pattern covering many zone axes
    should build it once here and pass it to :func:`rotated_slab` with
    ``repetitions=(1, 1, 1)``.

    Parameters
    ----------
    atoms : ase.Atoms
        The unit cell of the crystal.
    cell : tuple of three float
        Dimensions of the slab to be cut [Å].
    repetitions : tuple of three int, optional
        Repetitions of `atoms`. If not given, :func:`estimate_repetitions`
        chooses the smallest block that can contain the slab at any
        orientation.

    Returns
    -------
    block : ase.Atoms
    """
    if repetitions is None:
        repetitions = estimate_repetitions(atoms, cell)

    return atoms * repetitions


def rotated_slab(
    atoms: Atoms,
    zone_axis: np.ndarray,
    cell: tuple[float, float, float],
    repetitions: tuple[int, int, int] | None = None,
) -> tuple[Atoms, np.ndarray]:
    """Cut an orthogonal slab with the given zone axis along ``z``.

    The unit cell is repeated into a block large enough to contain the slab at
    any orientation, rotated so `zone_axis` points along ``+z``, and a box of
    the requested size is cut from the middle of it.

    See the module docstring for why the result is not periodic in ``x`` and
    ``y``.

    Parameters
    ----------
    atoms : ase.Atoms
        The unit cell of the crystal.
    zone_axis : np.ndarray
        Beam direction in the crystal frame, of shape ``(3,)``.
    cell : tuple of three float
        Dimensions of the slab to cut [Å]. The third entry is the thickness
        along the beam.
    repetitions : tuple of three int, optional
        Repetitions of `atoms` used to build the block that the slab is cut
        from. If not given, :func:`estimate_repetitions` chooses the smallest
        block that can contain the slab at any orientation. Pass
        ``(1, 1, 1)`` when `atoms` is already a block built by
        :func:`bulk_block`, which is how to avoid rebuilding it for every zone
        axis.

        The slab is cut around the *centre* of the block, so the repetitions
        also fix which point of the crystal ends up at the centre of the slab.
        Changing them therefore shifts the origin of the cut, which changes the
        result even when the block was already large enough.

    Returns
    -------
    slab : ase.Atoms
        The cut slab, with its cell set to `cell`.
    rotation : np.ndarray
        The rotation of shape ``(3, 3)`` taking the crystal frame to the slab
        frame, as returned by :func:`zone_axis_rotation`.
    """
    cell_array = np.asarray(cell, dtype=float)

    if cell_array.shape != (3,):
        raise ValueError(f"cell must have shape (3,), got {cell_array.shape}")

    zone_axis = np.asarray(zone_axis, dtype=float).ravel()
    rotation = zone_axis_rotation(zone_axis)

    if repetitions == (1, 1, 1):
        # `atoms` is already the block; copy it because the rotation and the
        # recentring below are in place, and a caller reusing one block across
        # many zone axes must not see it mutated.
        block = atoms.copy()
    else:
        block = bulk_block(atoms, cell, repetitions)

    # ASE's own rotation is used rather than `rotation` so that the cut is
    # reproducible against code that calls Atoms.rotate directly; the two agree
    # to floating-point precision (see test_zone_axis_rotation_matches_ase).
    block.rotate(zone_axis / np.linalg.norm(zone_axis), "z", rotate_cell=True)

    positions = block.positions - np.mean(block.positions, axis=0)
    half = cell_array / 2.0

    inside = np.logical_and.reduce(
        (
            positions[:, 0] > -half[0],
            positions[:, 0] < half[0],
            positions[:, 1] > -half[1],
            positions[:, 1] < half[1],
            positions[:, 2] > -half[2],
            positions[:, 2] < half[2],
        )
    )

    block.positions = positions
    slab = block[inside]
    slab.translate(half)
    slab.cell = cell_array

    return slab, rotation
