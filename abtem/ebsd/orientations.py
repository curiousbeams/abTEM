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

from typing import Optional

import numpy as np
from ase import Atoms

from abtem.inelastic.phonons import BaseFrozenPhonons, FrozenPhonons

__all__ = [
    "is_centrosymmetric",
    "fibonacci_hemisphere",
    "zone_axis_rotation",
    "estimate_repetitions",
    "central_origin",
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
    contain the slab's bounding sphere whichever way it is turned: a ball whose
    diameter is the slab's longest diagonal, ``sqrt(a**2 + b**2 + c**2)``.

    What bounds a ball inside a parallelepiped is the separation of its
    opposite faces, not the length of its edges, and for a non-orthogonal cell
    the two differ -- for the primitive cell of an fcc crystal the faces are
    ``a / sqrt(3)`` apart against edges of ``a / sqrt(2)``. Sizing on the edges
    leaves the block too thin across those faces, and a slab turned towards
    them loses atoms from its corners and ends: silicon cut 40 x 40 x 100 Å
    that way comes out with vacuum in 123 of 200 patches.

    One cell more is added along each axis, because the slab is not cut about
    the geometric centre of the block but about a point up to half a cell from
    it: the centroid of the atoms, or an origin moved into the central copy of
    the cell by :func:`central_origin`.

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
    lattice = np.asarray(atoms.cell.array, dtype=float)

    volume = abs(float(np.linalg.det(lattice)))
    if volume == 0.0:
        raise ValueError("atoms must have a cell with nonzero volume")

    # Separation of the faces spanned by the other two lattice vectors.
    separations = [
        volume / np.linalg.norm(np.cross(lattice[(i + 1) % 3], lattice[(i + 2) % 3]))
        for i in range(3)
    ]

    return tuple(int(np.ceil(diagonal / s)) + 1 for s in separations)  # type: ignore[return-value]


def central_origin(
    atoms: Atoms, repetitions: tuple[int, int, int], origin: np.ndarray
) -> np.ndarray:
    """Move a point of `atoms` to the equivalent point in the middle of its block.

    An origin is given in the frame of the unrepeated `atoms`, but the block
    ``atoms * repetitions`` grows away from that frame's corner, so the point as
    given sits at the edge of the block and a slab cut about it is mostly
    vacuum. Shifting it by whole lattice vectors to the most central copy picks
    out the same point of the crystal with the whole block around it.

    With ``repetitions == (1, 1, 1)`` -- an atomic model holding one feature --
    a point inside the cell is left where it is.

    Parameters
    ----------
    atoms : ase.Atoms
        The unrepeated cell.
    repetitions : tuple of three int
        Repetitions the block was built with.
    origin : np.ndarray
        Cartesian point of `atoms`, of shape ``(3,)``.

    Returns
    -------
    origin : np.ndarray
        The equivalent Cartesian point of the block.
    """
    lattice = np.asarray(atoms.cell.array, dtype=float)
    origin = np.asarray(origin, dtype=float).ravel()

    fractional = np.linalg.solve(lattice.T, origin)
    shift = np.round(np.asarray(repetitions, dtype=float) / 2.0 - fractional)

    # np.round sends 0.5 to 0, so a point inside a (1, 1, 1) cell never moves.
    return origin + shift @ lattice


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
    origin: np.ndarray | None = None,
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

        The slab is cut around `origin`, which defaults to the centre of the
        block, so the repetitions also fix which point of the crystal ends up
        at the centre of the slab. Changing them therefore shifts the cut,
        which changes the result even when the block was already large enough.

        Repeating a cell that contains a defect makes a periodic array of it.
        For an atomic model that already holds the feature of interest -- an MD
        cell, say -- pass ``(1, 1, 1)``, or check that
        :func:`estimate_repetitions` returns it.
    origin : np.ndarray, optional
        Point of `atoms`, in Cartesian coordinates, to place at the centre of
        the slab and to rotate about. Defaults to the centroid of the block.

        Give it to anchor the cut on a feature: a dislocation core, an
        interface. The centroid is only the feature's position when the atoms
        happen to be distributed symmetrically about it, which a void, a
        surface or an off-centre defect all break.

        When the block is built here, the origin is moved to the equivalent
        point of its central copy by :func:`central_origin`. When `atoms` is a
        prebuilt block (``repetitions=(1, 1, 1)``) it is used as given, so
        pass it through :func:`central_origin` first -- a point of the
        unrepeated cell sits at the corner of the block, and the slab cut
        about it is mostly empty.

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

    if origin is not None:
        origin = np.asarray(origin, dtype=float).ravel()
        if origin.shape != (3,):
            raise ValueError(f"origin must have shape (3,), got {origin.shape}")

    if repetitions == (1, 1, 1):
        # `atoms` is already the block; copy it because the rotation and the
        # recentring below are in place, and a caller reusing one block across
        # many zone axes must not see it mutated.
        block = atoms.copy()
    else:
        if repetitions is None:
            repetitions = estimate_repetitions(atoms, cell)
        block = bulk_block(atoms, cell, repetitions)
        if origin is not None:
            origin = central_origin(atoms, repetitions, origin)

    anchor = np.mean(block.positions, axis=0) if origin is None else origin

    # ASE's own rotation is used rather than `rotation` so that the cut is
    # reproducible against code that calls Atoms.rotate directly; the two agree
    # to floating-point precision (see test_zone_axis_rotation_matches_ase).
    # Rotating about the anchor keeps it fixed, so a feature placed there stays
    # at the centre of the slab whatever the zone axis.
    block.rotate(
        zone_axis / np.linalg.norm(zone_axis), "z", center=anchor, rotate_cell=True
    )

    positions = block.positions - anchor
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


def _crystal_and_displacements(
    atoms: Atoms | BaseFrozenPhonons,
) -> tuple[Atoms, Optional[FrozenPhonons]]:
    """The crystal the slabs are cut from, and how to displace them thermally.

    Each atom is tagged with its index in the crystal, so that displacements
    given atom by atom follow it through the block into every slab.
    """
    if isinstance(atoms, FrozenPhonons):
        frozen_phonons, atoms = atoms, atoms.atoms
    elif isinstance(atoms, BaseFrozenPhonons):
        raise TypeError(
            f"slabs are cut from one crystal, and a {type(atoms).__name__} holds "
            f"several; give Atoms, or FrozenPhonons for thermal displacements"
        )
    else:
        frozen_phonons = None

    atoms = atoms.copy()
    atoms.set_array("cell_index", np.arange(len(atoms)))
    return atoms, frozen_phonons


def _displaced(
    slab: Atoms, frozen_phonons: Optional[FrozenPhonons]
) -> Atoms | FrozenPhonons:
    """The slab, displaced as `frozen_phonons` displaces the crystal it came from.

    The same number of configurations, the same seeds, and each atom the
    displacements of its own site in the crystal.
    """
    if frozen_phonons is None:
        return slab

    sigmas = frozen_phonons.sigmas
    if not isinstance(sigmas, dict):
        # given atom by atom, for the atoms of the crystal
        sigmas = np.asarray(sigmas)[slab.arrays["cell_index"]]

    return FrozenPhonons(
        slab,
        num_configs=frozen_phonons.num_configs,
        sigmas=sigmas,
        directions=frozen_phonons.directions,
        seed=frozen_phonons.seed,
    )


def is_centrosymmetric(atoms: Atoms, symprec: float = 1e-3) -> bool:
    """Whether a crystal has a centre of inversion.

    For a centrosymmetric crystal the backscatter pattern obeys
    ``I(-k) = I(k)``, so either hemisphere follows from the other; for one
    without -- GaN, GaAs, ZnO -- the two differ, and the difference is what an
    EBSD polarity measurement reads.

    Parameters
    ----------
    atoms : ase.Atoms
        The crystal's unit cell.
    symprec : float, optional
        Tolerance on positions [Å] for the symmetry search (default 1e-3).

    Returns
    -------
    centrosymmetric : bool
    """
    import warnings

    import spglib  # type: ignore[import-untyped]

    cell = (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers)
    with warnings.catch_warnings():
        # spglib deprecates its own internal error handling; nothing the
        # caller can act on, and it would otherwise fail under -W error.
        warnings.simplefilter("ignore", DeprecationWarning)
        symmetry = spglib.get_symmetry(cell, symprec=symprec)
    if symmetry is None:
        raise ValueError("spglib could not determine the symmetry of the crystal")
    inversion = -np.eye(3, dtype=int)
    return any(
        np.array_equal(rotation, inversion) for rotation in symmetry["rotations"]
    )
