"""EBSD patterns from multislice reciprocity.

The physical process is an electron entering the specimen, scattering
inelastically somewhere inside it, and leaving through the surface in some
direction. Simulating that directly would mean a separate calculation for every
scattering site. Reciprocity turns it around: the amplitude for an electron
generated at :math:`(\\mathbf{r}, z)` to leave in direction :math:`\\mathbf{k}`
equals the amplitude at :math:`(\\mathbf{r}, z)` of a plane wave travelling in
:math:`-\\mathbf{k}` that entered at the surface. One multislice run per
collected direction therefore covers every generation site at once.

The collected intensity for direction :math:`\\mathbf{k}` is

.. math::
    I(\\mathbf{k}) = \\frac{1}{N_0} \\sum_z w_z
    \\sum_{\\mathbf{r}} |\\psi_{\\mathbf{k}}(\\mathbf{r}, z)|^2
    \\, s(\\mathbf{r}, z)

where :math:`\\psi_{\\mathbf{k}}` is the reciprocity plane wave, :math:`s` is
the density of inelastic scattering events, and :math:`w_z` is the probability
that an electron generated at depth :math:`z` escapes. The sum over
:math:`\\mathbf{r}` is incoherent: the generation events at different points are
independent, so intensities add rather than amplitudes.

The plane waves are normalized to unit modulus and :math:`N_0` is the total
intensity of the incident beam, so a specimen that scatters uniformly and
absorbs nothing gives :math:`I = 1` in every direction: the values are
backscatter yields relative to a featureless specimen. Note that :math:`N_0` is
fixed at the entrance surface rather than recomputed at each depth, so that a
beam attenuated by an absorptive potential correctly yields less backscatter
from deep in the specimen.
"""

from __future__ import annotations

import warnings
from typing import Callable, Literal, Optional, Sequence

import numpy as np
from ase import Atoms

from abtem.antialias import AntialiasAperture
from abtem.core.axes import AxisMetadata, OrdinalAxis
from abtem.core.backend import get_array_module, validate_device
from abtem.core.chunks import chunk_ranges, validate_chunks
from abtem.core.diagnostics import TqdmWrapper
from abtem.core.utils import CopyMixin, EqualityMixin, get_dtype
from abtem.ebsd.detectors import BackscatterDetector
from abtem.ebsd.measurements import SphericalPattern
from abtem.measurements import DiffractionPatterns
from abtem.multislice import FresnelPropagator, conventional_multislice_step
from abtem.potentials.iam import BasePotential, validate_potential
from abtem.prism.utils import plane_waves
from abtem.scan import BaseScan
from abtem.waves import Probe, Waves

__all__ = ["EBSD", "AntialiasLossWarning"]


class AntialiasLossWarning(UserWarning):
    """The antialias aperture clipped a reciprocity plane wave.

    Raised as its own class so that a caller running many calculations -- a
    reference pattern, say -- can quiet the individual warnings and report the
    loss over the whole run instead.
    """


DepthWeight = float | Sequence[float] | np.ndarray | Callable[[np.ndarray], np.ndarray]


def _validate_depth_weight(
    depth_weight: Optional[DepthWeight], depths: np.ndarray
) -> np.ndarray:
    """Resolve the escape-probability weight of each slice.

    The weights are normalized to sum to one, so the result is always a
    weighted average over depth and does not change scale with the number of
    slices or the thickness.
    """
    if depth_weight is None:
        weights = np.ones(len(depths))
    elif callable(depth_weight):
        weights = np.asarray(depth_weight(depths), dtype=float)
    elif np.isscalar(depth_weight):
        escape_depth = float(depth_weight)  # type: ignore[arg-type]
        if escape_depth <= 0.0:
            raise ValueError(
                f"an escape depth must be positive, got {escape_depth}; give an "
                f"array to use arbitrary weights"
            )
        weights = np.exp(-depths / escape_depth)
    else:
        weights = np.asarray(depth_weight, dtype=float)

    if weights.shape != depths.shape:
        raise ValueError(
            f"depth_weight resolved to shape {weights.shape}, expected "
            f"{depths.shape} (one weight per slice)"
        )

    if np.any(weights < 0.0):
        raise ValueError("depth weights must not be negative")

    total = weights.sum()
    if total <= 0.0:
        raise ValueError("depth weights must not be all zero")

    return weights / total


class EBSD(CopyMixin, EqualityMixin):
    """Backscatter diffraction patterns from a specimen, by reciprocity.

    Parameters
    ----------
    potential : BasePotential or Atoms
        The specimen. Given as atoms, a default potential is created. The beam
        travels along ``z``, so for a reference pattern the slab should already
        be cut with its zone axis along ``z`` (see
        :func:`abtem.ebsd.rotated_slab`).
    probe : Probe
        The incident beam. Its grid is matched to the potential.
    detector : BackscatterDetector
        The directions to collect.
    potential_weighting : bool, optional
        If True (default), the density of inelastic scattering events is taken
        to be the beam intensity times the squared projected potential of the
        slice, localizing backscattering on the atomic columns roughly as a
        screened Rutherford cross-section would.

        The weight of each slice is then rescaled so that its total matches the
        unweighted beam intensity. This keeps the depth profile of the
        generation rate unchanged, but it also **removes the overall magnitude**
        of the potential: a slice of heavy atoms generates no more backscatter
        than a slice of light ones, only a more sharply peaked distribution. For
        a single-element specimen this is immaterial; for a compound it
        suppresses compositional contrast.

        If False, the events follow the beam intensity alone.
    depth_weight : float or array or callable, optional
        Probability that an electron generated at a given depth escapes.

        The elastic multislice algorithm is unitary, so the reciprocity plane
        waves carry no attenuation with depth on their own: without a weight,
        an event 200 Å deep counts as much as one just below the surface. Give
        a float to weight the depths by ``exp(-z / escape_depth)`` [Å], an array
        of one weight per slice, or a callable mapping an array of depths to
        weights. The weights are normalized to sum to one.

        The default, None, weights every depth equally, which is what the
        unattenuated calculation does. A more complete treatment is an
        absorptive potential, which attenuates the beam and the reciprocity
        waves alike and additionally reproduces anomalous absorption; pass a
        complex potential for that.
    order : {1, 2, 'exact'}, optional
        Order of the Fresnel propagator (default ``'exact'``). The collected
        angles are large enough that the small-angle propagators are usually a
        poor approximation, so the default should rarely be changed.
    device : str, optional
        'cpu' or 'gpu'. Defaults to the user configuration.
    """

    def __init__(
        self,
        potential: BasePotential | Atoms,
        probe: Probe,
        detector: BackscatterDetector,
        potential_weighting: bool = True,
        depth_weight: Optional[DepthWeight] = None,
        order: Literal[1, 2, "exact"] = "exact",
        device: Optional[str] = None,
    ):
        self._potential = validate_potential(potential)
        self._probe = probe
        self._detector = detector
        self._potential_weighting = bool(potential_weighting)
        self._depth_weight = depth_weight
        self._order = order
        self._device = validate_device(device)

    @property
    def potential(self) -> BasePotential:
        """The specimen potential."""
        return self._potential

    @property
    def probe(self) -> Probe:
        """The incident beam."""
        return self._probe

    @property
    def detector(self) -> BackscatterDetector:
        """The collected directions."""
        return self._detector

    @property
    def device(self) -> str:
        """Device the calculation runs on."""
        return self._device

    def _direction_blocks(self, max_batch: int | str) -> list[tuple[int, int]]:
        """Split the collected directions into batches that fit in memory."""
        gpts = self._potential.gpts
        if gpts is None:
            raise RuntimeError("the potential has no grid")

        if isinstance(max_batch, int):
            max_batch = max_batch * int(np.prod(gpts))

        chunks = validate_chunks(
            shape=(len(self._detector),) + tuple(gpts),
            chunks=("auto", -1, -1),
            max_elements=max_batch,
            dtype=np.dtype("complex64"),
            device=self._device,
        )
        return list(chunk_ranges(chunks)[0])

    def scan(
        self,
        scan: Optional[BaseScan | Sequence] = None,
        max_batch_directions: int | str = "auto",
        lazy: bool = False,
        pbar: bool = False,
    ) -> DiffractionPatterns | SphericalPattern:
        """Calculate the backscatter pattern at each probe position.

        Parameters
        ----------
        scan : BaseScan or array of xy-positions, optional
            Probe positions. If not given, a single probe at the centre of the
            potential is used.
        max_batch_directions : int or str, optional
            Number of directions propagated at once. 'auto' (default) picks a
            batch from the abTEM chunk-size configuration. The source is
            re-propagated once per batch, so larger batches are faster but use
            more memory.
        lazy : bool, optional
            Not implemented; must be False.
        pbar : bool, optional
            If True, show a progress bar over the slices.

        Returns
        -------
        patterns : DiffractionPatterns or SphericalPattern
            :class:`~abtem.measurements.DiffractionPatterns` if the detector is
            a grid, otherwise a
            :class:`~abtem.ebsd.measurements.SphericalPattern` holding one
            value per collected direction. Any probe positions appear as
            leading ensemble axes.
        """
        if lazy:
            raise NotImplementedError(
                "the reciprocity EBSD calculation runs eagerly; pass lazy=False"
            )

        xp = get_array_module(self._device)

        potential = self._potential.build(lazy=False)
        num_slices = potential.num_slices

        depths = np.cumsum(np.asarray(potential.slice_thickness, dtype=float))
        weights = _validate_depth_weight(self._depth_weight, depths)

        probe = self._probe.copy()
        probe.grid.match(potential)
        energy = probe._valid_energy

        source = probe.build(scan=scan, lazy=False)
        source = source.copy_to_device(self._device)
        ensemble_axes_metadata = list(source.ensemble_axes_metadata)
        ensemble_shape = source.shape[:-2]

        # Total intensity of the incident beam, which sets the scale of the
        # result. Taken at the entrance surface and held fixed, so that a beam
        # losing intensity to an absorptive potential generates correspondingly
        # less backscatter deeper in.
        incident_norm = xp.sum(
            xp.abs(source.array) ** 2, axis=(-2, -1), dtype=xp.float64
        ).reshape(-1)

        wave_vectors = xp.asarray(
            self._detector.transverse_wave_vectors(energy),
            dtype=get_dtype(complex=False),
        )

        intensities = xp.zeros(
            ensemble_shape + (len(self._detector),), dtype=xp.float32
        )
        antialias_loss = xp.zeros(len(self._detector), dtype=xp.float32)

        blocks = self._direction_blocks(max_batch_directions)

        progress = TqdmWrapper(
            total=len(blocks) * num_slices, enabled=pbar, leave=False
        )
        try:
            for start, stop in blocks:
                self._propagate_block(
                    potential=potential,
                    source=source,
                    incident_norm=incident_norm,
                    wave_vectors=wave_vectors[start:stop],
                    weights=weights,
                    energy=energy,
                    intensities=intensities,
                    antialias_loss=antialias_loss,
                    start=start,
                    stop=stop,
                    progress=progress,
                )
        finally:
            progress.close_if_exists()

        max_loss = float(xp.max(antialias_loss))
        if max_loss > 0.05:
            warnings.warn(
                f"the antialias aperture removed {max_loss:.1%} of the intensity "
                f"of at least one reciprocity plane wave; the collected angles "
                f"are too large for this sampling",
                AntialiasLossWarning,
            )

        return self._to_measurement(
            intensities,
            ensemble_axes_metadata=ensemble_axes_metadata,
            energy=energy,
            antialias_loss=antialias_loss,
        )

    def _propagate_block(
        self,
        potential,
        source: Waves,
        incident_norm,
        wave_vectors,
        weights: np.ndarray,
        energy: float,
        intensities,
        antialias_loss,
        start: int,
        stop: int,
        progress: TqdmWrapper,
    ) -> None:
        """Co-propagate the source and one batch of reciprocity waves.

        Both wavefields are advanced through the same slice before their
        overlap is accumulated, so the depth-resolved source never has to be
        held in memory all at once. That is what makes a scan of many probe
        positions affordable; the cost is re-propagating the source once per
        batch of directions.
        """
        xp = get_array_module(self._device)

        # Unit-modulus plane waves: the normalization of the result comes from
        # dividing by the source weight of each slice, not from these.
        array = plane_waves(wave_vectors, potential.extent, potential.gpts)

        reciprocity = Waves(
            array,
            energy=energy,
            extent=potential.extent,
            ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(len(array))))],
        )
        initial_norm = xp.sum(xp.abs(reciprocity.array) ** 2, axis=(-2, -1))

        # The source is re-propagated for every block, so start from a copy.
        beam = source.copy()

        propagator = FresnelPropagator()
        antialias_aperture = AntialiasAperture()

        for index, potential_slice in enumerate(potential.generate_slices()):
            potential_slice = potential_slice.copy_to_device(self._device)

            # Build the transmission function once and hand it to both
            # wavefields; they share an energy and a grid, so recomputing it
            # per wavefield would double the cost of every slice.
            transmission_function = potential_slice.transmission_function(energy=energy)
            transmission_function = antialias_aperture.bandlimit(
                transmission_function, in_place=True
            )

            beam = conventional_multislice_step(
                beam,
                transmission_function,
                propagator=propagator,
                antialias_aperture=antialias_aperture,
                order=self._order,
            )
            reciprocity = conventional_multislice_step(
                reciprocity,
                transmission_function,
                propagator=propagator,
                antialias_aperture=antialias_aperture,
                order=self._order,
            )

            weight = float(weights[index])
            if weight > 0.0:
                self._accumulate(
                    beam=beam,
                    reciprocity=reciprocity,
                    projected_potential=potential_slice.array[0],
                    incident_norm=incident_norm,
                    weight=weight,
                    intensities=intensities,
                    start=start,
                    stop=stop,
                )

            progress.update_if_exists(1)

        final_norm = xp.sum(xp.abs(reciprocity.array) ** 2, axis=(-2, -1))
        antialias_loss[start:stop] = 1.0 - final_norm / initial_norm

    def _accumulate(
        self,
        beam: Waves,
        reciprocity: Waves,
        projected_potential,
        incident_norm,
        weight: float,
        intensities,
        start: int,
        stop: int,
    ) -> None:
        """Add one slice's contribution to the collected intensities."""
        xp = get_array_module(self._device)

        beam_intensity = xp.abs(beam.array) ** 2

        if self._potential_weighting:
            cross_section = xp.abs(projected_potential) ** 2
            source = beam_intensity * cross_section

            # Rescale so this slice contributes the same total weight as the
            # unweighted beam: the cross-section sets where backscattering
            # happens within the slice, not how much of it there is.
            total = xp.sum(source, axis=(-2, -1), keepdims=True)
            unweighted = xp.sum(beam_intensity, axis=(-2, -1), keepdims=True)

            # A slice holding no atoms has nothing to scatter off and so
            # contributes nothing, rather than being rescaled by 0/0.
            empty = total == 0.0
            source = source * xp.where(
                empty, 0.0, unweighted / xp.where(empty, 1.0, total)
            )
        else:
            source = beam_intensity

        # (directions, pixels) @ (pixels, positions) -> (directions, positions)
        reciprocity_flat = xp.abs(reciprocity.array).reshape(stop - start, -1) ** 2
        source_flat = source.reshape(-1, reciprocity_flat.shape[1])

        overlap = reciprocity_flat @ source_flat.T
        overlap = overlap / incident_norm[None]

        intensities[..., start:stop] += weight * overlap.T.reshape(
            intensities.shape[:-1] + (stop - start,)
        )

    def _to_measurement(
        self,
        intensities,
        ensemble_axes_metadata: list[AxisMetadata],
        energy: float,
        antialias_loss,
    ) -> DiffractionPatterns | SphericalPattern:
        """Wrap the collected intensities in the matching measurement type."""
        array = np.asarray(
            intensities.get() if hasattr(intensities, "get") else intensities
        )
        loss = np.asarray(
            antialias_loss.get() if hasattr(antialias_loss, "get") else antialias_loss
        )

        metadata = {
            "energy": energy,
            "label": "backscattered intensity",
            "antialias_loss_max": float(loss.max()),
            "antialias_loss_mean": float(loss.mean()),
        }

        if self._detector.is_grid:
            gpts = self._detector.gpts
            return DiffractionPatterns(
                array.reshape(array.shape[:-1] + (gpts, gpts)),
                sampling=self._detector.reciprocal_sampling(energy),
                fftshift=True,
                ensemble_axes_metadata=ensemble_axes_metadata,
                metadata=metadata,
            )

        return SphericalPattern(
            array,
            directions=self._detector.directions,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )
