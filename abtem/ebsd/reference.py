"""Reference patterns: EBSD intensity over the whole hemisphere.

A single reciprocity calculation is only valid for directions near the beam
direction it was set up for, because the slab is cut with that direction along
``z`` and the propagator is accurate over a limited angle. Covering the whole
hemisphere therefore means tiling it into patches, running one calculation per
patch with its own slab, and stitching the results back together in the crystal
frame.

:class:`EBSDReferencePattern` does the tiling, the bookkeeping between the
crystal frame and each slab frame, and the stitching.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Optional

import numpy as np
from ase import Atoms

from abtem.array import validate_lazy
from abtem.core.diagnostics import TqdmWrapper
from abtem.core.energy import energy2wavelength
from abtem.core.utils import CopyMixin, EqualityMixin
from abtem.ebsd.detectors import BackscatterDetector
from abtem.ebsd.measurements import SphericalPattern
from abtem.ebsd.orientations import bulk_block, fibonacci_hemisphere, rotated_slab
from abtem.ebsd.projections import HemisphereProjection, validate_projection
from abtem.ebsd.reciprocity import EBSD, AntialiasLossWarning, DepthWeight
from abtem.potentials.iam import Potential
from abtem.scan import CustomScan
from abtem.waves import Probe

__all__ = [
    "EBSDReferencePattern",
    "patch_half_angle",
    "fft_friendly_gpts",
    "maximum_sampling",
    "potential_sampling",
    "recommended_sampling",
]

#: Fraction of the antialias-limited sampling used by default.
#:
#: A margin is essential rather than cosmetic: a plane wave launched right at
#: the aperture edge scatters straight past it, and sampling exactly at the
#: limit (safety = 1.0) loses about 96% of its intensity. Measured on a 40 Å
#: silicon slab collecting 132 mrad, the loss falls off as
#:
#:     safety  1.0     0.9    0.8    0.7    0.6
#:     loss    95.9%   4.3%   2.4%   1.3%   0.7%
#:
#: 0.8 keeps the recommended sampling comfortably below the 5% loss that
#: :class:`~abtem.ebsd.reciprocity.AntialiasLossWarning` reports, so a
#: default-configured run does not warn about its own defaults. 0.9 sits right
#: on that boundary and trips it for some detector geometries.
_SAMPLING_SAFETY = 0.8

#: Density of the direction grid relative to the output image, so that binning
#: leaves no empty pixels at the rim where the grid thins out.
_DIRECTION_OVERSAMPLING = 1.15


def fft_friendly_gpts(extent: tuple[float, float], sampling: float) -> tuple[int, int]:
    """Grid points covering `extent` at `sampling` or finer, sized for the FFT.

    The calculation is almost entirely Fresnel propagation, so its cost is set
    by how fast an FFT of this size is rather than by how many points there
    are. Those are not the same thing: a grid of 71 points -- prime, so the
    transform falls back to Bloom/Rader -- takes five times as long as one of
    72, and longer than one of 128. Rounding *up* to the next size with only
    small prime factors makes the sampling finer, so it can only help the
    antialias margin, and it is typically several times faster.

    Parameters
    ----------
    extent : two float
        Lateral extent of the grid [Å].
    sampling : float
        Largest acceptable sampling [Å].

    Returns
    -------
    gpts : two int
    """
    from scipy.fft import next_fast_len  # type: ignore[import-untyped]

    return tuple(  # type: ignore[return-value]
        int(next_fast_len(int(np.ceil(length / sampling)))) for length in extent[:2]
    )


def patch_half_angle(n_patches: int, coverage: float = 1.2) -> float:
    """Angular radius of each patch needed to tile the hemisphere.

    A Fibonacci lattice of `n_patches` points on the hemisphere gives each point
    a cell of solid angle :math:`2\\pi / n`. Treating the cell as a regular
    hexagon and covering it with a disc through its corners gives the angular
    radius below, which `coverage` then inflates so neighbouring patches overlap
    rather than leaving seams.

    Parameters
    ----------
    n_patches : int
        Number of patches tiling the hemisphere.
    coverage : float, optional
        Factor by which the discs are enlarged past the bare covering radius
        (default 1.2).

    Returns
    -------
    half_angle : float
        Angular radius of one patch [mrad].
    """
    if n_patches < 1:
        raise ValueError(f"n_patches must be at least 1, got {n_patches}")

    return float(
        np.sqrt(8.0 * np.pi / (3.0 * np.sqrt(3.0) * n_patches)) * 1e3 * coverage
    )


def maximum_sampling(
    energy: float, max_angle: float, safety: float = _SAMPLING_SAFETY
) -> float:
    """Coarsest sampling that still carries a given scattering angle.

    The antialias aperture passes spatial frequencies up to two thirds of the
    Nyquist frequency, so collecting a scattering angle ``a`` needs a sampling
    finer than ``1 / (2 k) * 2 / 3`` with ``k = sin(a) / wavelength``.

    This is an upper bound and not on its own a good choice: it says the grid
    can *carry* the collected angles, not that it *resolves the potential*
    producing them. See :func:`potential_sampling` for the second condition and
    :func:`recommended_sampling` for both together.

    Parameters
    ----------
    energy : float
        Electron energy [eV].
    max_angle : float
        Largest collected scattering angle [mrad].
    safety : float, optional
        Fraction of the antialias limit to use (default 0.8). Sampling right at
        the limit loses almost all the intensity of the steepest plane wave, so
        the margin matters; see ``_SAMPLING_SAFETY``.

    Returns
    -------
    sampling : float
        Largest acceptable sampling [Å].
    """
    wave_number = np.sin(max_angle * 1e-3) / energy2wavelength(energy)
    return float(1.0 / (2.0 * wave_number) * 2.0 / 3.0 * safety)


@lru_cache(maxsize=32)
def _scattering_power_cutoff(number: int, tolerance: float) -> float:
    """Spatial frequency containing all but `tolerance` of an atom's scattering.

    The projected potential of an atom is sharply peaked, and its transform
    decays slowly -- the electron scattering factor has a Rutherford tail, so
    there is no frequency beyond which it truly vanishes. What can be asked is
    where all but a given fraction of its power lies.
    """
    # The tail of the spectrum matters to the total power, so the measuring
    # grid has to reach well past the cutoff it is looking for. Converging on
    # silicon: 512 points gives 0.066 A, 1024 gives 0.063, 2048 gives 0.062 and
    # 4096 gives 0.0619. 1024 is within about 1.5% of the limit for light
    # elements and 3% for gold, and costs a quarter of a second once per
    # element. The slice thickness makes no difference at all for an isolated
    # atom, and the box only a couple of percent.
    extent, gpts, thickness = 8.0, 1024, 2.0

    atoms = Atoms(
        numbers=[number],
        positions=[(extent / 2, extent / 2, thickness / 2)],
        cell=(extent, extent, thickness),
    )
    projected = (
        Potential(atoms, gpts=gpts, slice_thickness=thickness, projection="finite")
        .build(lazy=False)
        .array[0]
    )

    power = np.abs(np.fft.fft2(np.asarray(projected))) ** 2
    frequencies = np.fft.fftfreq(gpts, d=extent / gpts)
    radial = np.hypot(*np.meshgrid(frequencies, frequencies, indexing="ij"))

    order = np.argsort(radial.ravel())
    cumulative = np.cumsum(power.ravel()[order])
    cumulative /= cumulative[-1]

    return float(radial.ravel()[order][np.searchsorted(cumulative, 1.0 - tolerance)])


def potential_sampling(atoms: Atoms, tolerance: float = 0.01) -> float:
    """Sampling that resolves the projected potential of `atoms`.

    The antialias aperture discards whatever the grid cannot carry, so
    scattering power beyond it is simply lost. This returns the sampling that
    keeps all but `tolerance` of an isolated atom's projected scattering power
    inside the aperture, taking the most demanding element present.

    Calibrated against a convergence test on silicon at 30 kV, where the error
    in the pattern relative to a far finer grid ran at one to three times the
    power left outside the aperture:

    ==========  ===============  ==============
    sampling    power captured   pattern error
    ==========  ===============  ==============
    0.20 Å      90.5%            33%
    0.14 Å      94.8%            16%
    0.10 Å      97.3%            5.5%
    0.07 Å      98.7%            0.6%
    ==========  ===============  ==============

    So the default ``tolerance=0.01`` targets roughly a percent, and the
    returned value should be read as an estimate good to a factor of about two
    rather than a guarantee. A convergence test remains the only proof.

    Parameters
    ----------
    atoms : ase.Atoms
        The specimen. Only which elements are present matters.
    tolerance : float, optional
        Fraction of the scattering power allowed to fall outside the antialias
        aperture (default 0.01).

    Returns
    -------
    sampling : float
        Recommended sampling [Å].
    """
    if not 0.0 < tolerance < 1.0:
        raise ValueError(f"tolerance must be between 0 and 1, got {tolerance}")

    if len(atoms) == 0:
        raise ValueError("cannot estimate a sampling for an empty cell")

    cutoff = max(
        _scattering_power_cutoff(int(number), tolerance)
        for number in np.unique(atoms.numbers)
    )
    # the aperture passes |k| < 1 / (3 * sampling)
    return float(1.0 / (3.0 * cutoff))


def recommended_sampling(
    energy: float,
    max_angle: float,
    atoms: Optional[Atoms] = None,
    tolerance: float = 0.01,
    safety: float = _SAMPLING_SAFETY,
) -> float:
    """Sampling that both carries the collected angles and resolves the atoms.

    Two separate conditions have to hold, and for a typical EBSD geometry the
    second is much the stricter: :func:`maximum_sampling` asks that the grid
    can carry the collected scattering angles, :func:`potential_sampling` that
    it resolves the potential doing the scattering. This returns the finer.

    Parameters
    ----------
    energy : float
        Electron energy [eV].
    max_angle : float
        Largest collected scattering angle [mrad].
    atoms : ase.Atoms, optional
        The specimen. Without it only the angular condition is applied, which
        on its own is not enough for a converged pattern.
    tolerance : float, optional
        Passed to :func:`potential_sampling` (default 0.01).
    safety : float, optional
        Passed to :func:`maximum_sampling`.

    Returns
    -------
    sampling : float
        Recommended sampling [Å].
    """
    angular = maximum_sampling(energy, max_angle, safety=safety)

    if atoms is None:
        return angular

    return min(angular, potential_sampling(atoms, tolerance=tolerance))


class EBSDReferencePattern(CopyMixin, EqualityMixin):
    """A hemisphere-wide EBSD reference pattern, tiled from zone-axis patches.

    Parameters
    ----------
    atoms : ase.Atoms
        The unit cell of the crystal.
    probe : Probe
        The incident beam.
    n_patches : int, optional
        Number of zone-axis patches tiling the hemisphere (default 400). More
        patches means smaller patches, so a smaller collection angle, a coarser
        sampling and a cheaper run each -- but more runs.
    slab_cell : tuple of three float, optional
        Dimensions of the slab cut for each patch [Å] (default
        ``(10.0, 10.0, 40.0)``). The third entry is the thickness along the
        beam.
    gpts : int, optional
        Pixels along each axis of the projected image (default 128). Only sets
        the default density of the sampled directions; the projection itself
        happens in :meth:`SphericalPattern.project`.
    projection : str or HemisphereProjection, optional
        Projection whose even grid the directions are drawn from. One of
        ``'lambert'`` (default) or ``'stereographic'``.

        Lambert is the better choice and the default for two reasons. It is
        equal-area, so it samples the hemisphere uniformly: a stereographic
        grid puts three times more directions per steradian at the equator than
        at the pole, which is the wrong way round, since the poles carry the
        zone-axis detail. And it covers the whole hemisphere including the
        near-equator azimuths that fall in the corners of its square, which a
        stereographic disk never reaches -- so a Lambert-sampled pattern can be
        interpolated to either projection without gaps, while the reverse
        leaves holes.
    sampling : float, optional
        Real-space sampling of the potential [Å]. Defaults to
        :func:`recommended_sampling`, which is the finer of what the patch
        half-angle needs and what resolving `atoms` needs -- usually the
        latter, by a factor of two or so. A value too coarse for the collected
        angles is warned about; one too coarse to resolve the atoms is not,
        since that is a convergence question rather than an outright failure.
    slice_thickness : float, optional
        Multislice slice thickness [Å] (default 1.0).
    direction_gpts : int, optional
        Number of points per axis of the grid the directions are drawn from.
        Defaults to ``ceil(1.15 * gpts)``, slightly denser than the output image
        so binning leaves no empty pixels.
    max_angle : float, optional
        Angular radius of each patch [mrad]. Defaults to
        :func:`patch_half_angle` for `n_patches`.
    overlap_tolerance : float, optional
        Directions within this angle [rad] of belonging to a second patch are
        calculated in both, and the results averaged when the pattern is
        projected. Default 0.0, ie. every direction belongs to its nearest
        zone axis alone.
    repetitions : tuple of three int, optional
        Repetitions of `atoms` used to build the block the slabs are cut from.
        Defaults to the smallest block that can contain the slab at any
        orientation. Note that this also fixes which point of the crystal sits
        at the centre of each slab.
    origin : np.ndarray, optional
        Point of `atoms` to place at the centre of every slab, passed to
        :func:`rotated_slab`. Defaults to the centroid.
    probe_positions : int, optional
        Side of the square grid of probe positions each patch is averaged over
        (default 1, a single probe at the centre of the slab).

        A reference pattern is meant to be a property of the crystal, but a
        single probe samples one arbitrary position within the unit cell, and
        the result depends on which: measured on silicon, individual positions
        vary by up to 10% in a given direction and a centred probe sits about
        5% from the average, with a shape correlation of 0.969. Averaging
        incoherently over positions is what removes that dependence.

        It is nearly free -- the reciprocity waves dominate the cost and do not
        depend on where the probe is -- so 3 costs about 1.16x and 5 about
        1.49x. What matters is the *spacing* rather than the span: 3 leaves
        about 3% and a correlation of 0.991, while 5 reaches 1% and 0.999.

        Note this is the right thing to do for a crystal and the wrong thing
        for a defect, which it would average away. For a specimen with a
        feature, use :meth:`~abtem.ebsd.reciprocity.EBSD.scan` at chosen
        positions for one orientation instead.
    probe_extent : float, optional
        Span of that grid [Å]. Defaults to the largest lattice constant of
        `atoms`, so the grid covers one unit cell, over which the average is
        complete by periodicity.
    potential_weighting, depth_weight, device :
        Passed to :class:`~abtem.ebsd.reciprocity.EBSD`.
    """

    def __init__(
        self,
        atoms: Atoms,
        probe: Probe,
        n_patches: int = 400,
        slab_cell: tuple[float, float, float] = (10.0, 10.0, 40.0),
        gpts: int = 128,
        projection: str | HemisphereProjection = "lambert",
        sampling: Optional[float] = None,
        slice_thickness: float = 1.0,
        direction_gpts: Optional[int] = None,
        max_angle: Optional[float] = None,
        overlap_tolerance: float = 0.0,
        repetitions: Optional[tuple[int, int, int]] = None,
        origin: Optional[np.ndarray] = None,
        probe_positions: int = 1,
        probe_extent: Optional[float] = None,
        potential_weighting: bool = True,
        depth_weight: Optional[DepthWeight] = None,
        device: Optional[str] = None,
    ):
        self._atoms = atoms
        self._probe = probe
        self._n_patches = int(n_patches)
        self._slab_cell = (
            float(slab_cell[0]),
            float(slab_cell[1]),
            float(slab_cell[2]),
        )
        self._gpts = int(gpts)
        self._projection = validate_projection(projection)
        self._slice_thickness = float(slice_thickness)
        self._overlap_tolerance = float(overlap_tolerance)
        self._repetitions = repetitions
        self._origin = origin
        self._probe_positions = int(probe_positions)
        self._probe_extent = probe_extent
        self._potential_weighting = potential_weighting

        if self._probe_positions < 1:
            raise ValueError(
                f"probe_positions must be at least 1, got {probe_positions}"
            )
        self._depth_weight = depth_weight
        self._device = device

        self._max_angle = (
            patch_half_angle(self._n_patches) if max_angle is None else float(max_angle)
        )

        energy = probe._valid_energy
        limit = maximum_sampling(energy, self._max_angle, safety=1.0)

        if sampling is None:
            self._sampling = recommended_sampling(energy, self._max_angle, atoms=atoms)
        else:
            self._sampling = float(sampling)
            if self._sampling >= limit:
                warnings.warn(
                    f"a sampling of {self._sampling:.3f} Å cannot resolve the "
                    f"{self._max_angle:.0f} mrad patch half-angle; the antialias "
                    f"aperture will clip the reciprocity waves. Use less than "
                    f"{limit:.3f} Å."
                )

        self._direction_gpts = (
            int(np.ceil(_DIRECTION_OVERSAMPLING * self._gpts))
            if direction_gpts is None
            else int(direction_gpts)
        )

    @property
    def atoms(self) -> Atoms:
        """The unit cell of the crystal."""
        return self._atoms

    @property
    def probe(self) -> Probe:
        """The incident beam."""
        return self._probe

    @property
    def slab_cell(self) -> tuple[float, float, float]:
        """Dimensions of the slab cut for each patch [Å]."""
        return self._slab_cell

    @property
    def n_patches(self) -> int:
        """Number of zone-axis patches tiling the hemisphere."""
        return self._n_patches

    @property
    def gpts(self) -> int:
        """Pixels per axis of the projected image."""
        return self._gpts

    @property
    def projection(self) -> HemisphereProjection:
        """Projection the sampled directions are drawn from."""
        return self._projection

    @property
    def probe_scan(self) -> Optional[CustomScan]:
        """Probe positions each patch is averaged over, or None for one probe.

        A square grid about the centre of the slab, which is where a feature
        anchored by `origin` sits.
        """
        if self._probe_positions == 1:
            return None

        extent = (
            float(np.max(self._atoms.cell.lengths()))
            if self._probe_extent is None
            else float(self._probe_extent)
        )

        n = self._probe_positions
        # One period sampled without repeating its endpoints, then centred on
        # the slab: the last point is dropped, so the grid spans
        # extent * (n - 1) / n and its midpoint is half of that, not extent / 2.
        offsets = np.linspace(0.0, extent * (n - 1) / n, n)
        offsets = offsets - offsets.mean()
        centre = np.array(self._slab_cell[:2]) / 2.0

        return CustomScan(
            [[centre[0] + dx, centre[1] + dy] for dx in offsets for dy in offsets]
        )

    @property
    def zone_axes(self) -> np.ndarray:
        """The zone axis of each patch, as unit vectors of shape ``(M, 3)``."""
        return fibonacci_hemisphere(self._n_patches)

    @property
    def directions(self) -> np.ndarray:
        """Every sampled direction in the crystal frame, of shape ``(N, 3)``."""
        return self._projection.grid(self._direction_gpts)

    @property
    def max_angle(self) -> float:
        """Angular radius of each patch [mrad]."""
        return self._max_angle

    @property
    def sampling(self) -> float:
        """Largest acceptable real-space sampling of the potential [Å]."""
        return self._sampling

    @property
    def potential_gpts(self) -> tuple[int, int]:
        """Grid the potential is built on.

        Derived from :attr:`sampling` and rounded up to a size the FFT handles
        quickly; see :func:`fft_friendly_gpts`. The actual sampling is
        therefore a little finer than :attr:`sampling`.
        """
        return fft_friendly_gpts(self._slab_cell[:2], self._sampling)

    def _assign_directions(self) -> list[np.ndarray]:
        """Group the sampled directions by the patch that will calculate them.

        Each direction goes to its nearest zone axis, plus any zone axis within
        `overlap_tolerance` of being the nearest.
        """
        directions = self.directions
        zone_axes = self.zone_axes

        cosines = directions @ zone_axes.T
        nearest = np.max(cosines, axis=1, keepdims=True)

        # Widening the acceptance by an angle, rather than by a fraction of the
        # cosine, keeps the overlap band the same width everywhere.
        threshold = np.cos(
            np.arccos(np.clip(nearest, -1.0, 1.0)) + self._overlap_tolerance
        )
        assigned = cosines >= threshold

        return [np.where(assigned[:, j])[0] for j in range(len(zone_axes))]

    def build(
        self,
        max_batch_directions: int | str = "auto",
        lazy: Optional[bool] = None,
        pbar: bool = True,
    ) -> SphericalPattern:
        """Run every patch and stitch the results into one pattern.

        Named for the abTEM builders it follows -- :meth:`abtem.Probe.build`,
        :meth:`abtem.SMatrix.build` -- which leaves ``compute`` to mean what it
        means everywhere else: realize a lazy array. So ``build(lazy=True)``
        returns a pattern whose ``compute()`` runs it.

        Parameters
        ----------
        max_batch_directions : int or str, optional
            Directions propagated at once within each patch. Passed to
            :meth:`~abtem.ebsd.reciprocity.EBSD.scan`.
        lazy : bool, optional
            If True, return a pattern backed by a dask graph with one task per
            patch, rather than running them here. Defaults to the abTEM
            configuration.
        pbar : bool, optional
            If True (default), show a progress bar over the patches. Ignored
            when lazy, where nothing runs yet.

        Returns
        -------
        pattern : SphericalPattern
            Backscattered intensity at every sampled direction, in the crystal
            frame. Use :meth:`SphericalPattern.project` to turn it into an
            image.
        """
        lazy = validate_lazy(lazy)

        directions = self.directions
        zone_axes = self.zone_axes
        assignment = self._assign_directions()

        block = bulk_block(self._atoms, self._slab_cell, self._repetitions)
        probe_scan = self.probe_scan

        # The origin is given in the frame of `atoms`; the block repeats it, so
        # a feature anchored in the unrepeated cell keeps its coordinates.
        origin = self._origin

        # Every patch shares a slab shape and a sampling, so the grid is chosen
        # once -- at a size the FFT likes, which dominates the runtime.
        gpts = self.potential_gpts

        cutoff = np.cos(self._max_angle * 1e-3)

        patterns: list[SphericalPattern] = []
        max_loss = 0.0

        progress = TqdmWrapper(
            total=len(zone_axes), enabled=pbar and not lazy, leave=False
        )
        try:
            for zone_axis, indices in zip(zone_axes, assignment):
                slab, rotation = rotated_slab(
                    block,
                    zone_axis,
                    self._slab_cell,
                    repetitions=(1, 1, 1),
                    origin=origin,
                )

                # Crystal frame -> slab frame, then drop whatever this patch
                # cannot legitimately collect.
                local = directions[indices] @ rotation.T
                keep = local[:, 2] > cutoff
                local, indices = local[keep], indices[keep]

                progress.update_if_exists(1)

                if len(local) == 0:
                    continue

                # The loss is reported once for the whole run below, so the
                # per-patch warnings would be several hundred copies of it.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", AntialiasLossWarning)
                    result = EBSD(
                        Potential(
                            slab,
                            gpts=gpts,
                            slice_thickness=self._slice_thickness,
                            projection="finite",
                            device=self._device,
                        ),
                        probe=self._probe,
                        detector=BackscatterDetector(directions=local),
                        potential_weighting=self._potential_weighting,
                        depth_weight=self._depth_weight,
                        device=self._device,
                    ).scan(
                        scan=probe_scan,
                        max_batch_directions=max_batch_directions,
                        lazy=lazy,
                    )

                if not lazy:
                    max_loss = max(max_loss, result.metadata["antialias_loss_max"])

                array = result.array
                if probe_scan is not None:
                    # Incoherent average over the probe positions: the
                    # generation events at different positions are independent.
                    array = array.mean(axis=0)

                patterns.append(SphericalPattern(array, directions=directions[indices]))
        finally:
            progress.close_if_exists()

        if not patterns:
            raise RuntimeError("no directions were assigned to any patch")

        pattern = SphericalPattern.concatenate(patterns)
        pattern.metadata.update(
            {
                "energy": self._probe.energy,
                "label": "backscattered intensity",
                "n_patches": self._n_patches,
                "max_angle": self._max_angle,
                "sampling": self._sampling,
                "gpts": gpts,
                "projection": self._projection.name,
                "probe_positions": self._probe_positions,
                **({} if lazy else {"antialias_loss_max": max_loss}),
            }
        )

        if not lazy and max_loss > 0.05:
            warnings.warn(
                f"the antialias aperture removed up to {max_loss:.1%} of the "
                f"intensity of a reciprocity plane wave; consider a sampling "
                f"finer than {self._sampling:.3f} Å",
                AntialiasLossWarning,
            )

        return pattern
