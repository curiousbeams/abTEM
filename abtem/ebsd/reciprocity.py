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

import dask.array as da
import numpy as np
from ase import Atoms

from abtem.antialias import AntialiasAperture
from abtem.array import validate_lazy
from abtem.core.axes import AxisMetadata, EnergyAxis, OrdinalAxis
from abtem.core.backend import get_array_module, validate_device
from abtem.core.chunks import chunk_ranges, validate_chunks
from abtem.core.diagnostics import TqdmWrapper
from abtem.core.utils import CopyMixin, EqualityMixin, get_dtype
from abtem.distributions import BaseDistribution
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


def _validate_backscatter_energy(
    backscatter_energy, beam_energy: float
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the backscattered energies and their weights.

    The beam travels at `beam_energy`; a backscattered electron has lost some
    of that, so the reciprocity waves belong at or below it.
    """
    if backscatter_energy is None:
        return np.array([beam_energy]), np.array([1.0])

    if isinstance(backscatter_energy, BaseDistribution):
        energies = np.asarray(backscatter_energy.values, dtype=float)
        weights = np.asarray(backscatter_energy.weights, dtype=float)
    else:
        energies = np.atleast_1d(np.asarray(backscatter_energy, dtype=float))
        weights = np.ones(len(energies))

    if energies.ndim != 1 or len(energies) == 0:
        raise ValueError(
            f"backscatter_energy must be a scalar or a 1d sequence, got shape "
            f"{energies.shape}"
        )

    if np.any(energies <= 0.0):
        raise ValueError("backscattered energies must be positive")

    if np.any(energies > beam_energy):
        raise ValueError(
            f"a backscattered electron cannot carry more than the beam energy "
            f"of {beam_energy:.0f} eV; got up to {energies.max():.0f} eV"
        )

    total = weights.sum()
    if total <= 0.0:
        raise ValueError("the energy weights must not sum to zero")

    return energies, weights / total


def _worker_count() -> int:
    """Threads dask will run tasks on, as a target for the number of blocks."""
    from dask.system import CPU_COUNT

    return max(1, int(CPU_COUNT))


def _antialias_message(loss: float) -> str:
    return (
        f"the antialias aperture removed {loss:.1%} of the intensity of at "
        f"least one reciprocity plane wave; the collected angles are too large "
        f"for this sampling"
    )


def _build_potential(potential: BasePotential) -> BasePotential:
    """Build a potential as its own task, shared by the blocks that use it."""
    return potential.build(lazy=False)


def _potential_configurations(
    potential: BasePotential,
) -> list[tuple[BasePotential, float]]:
    """Split a potential into its frozen-phonon configurations and their weights.

    A potential built on :class:`~abtem.FrozenPhonons` or
    :class:`~abtem.AtomsEnsemble` carries an ensemble axis, and the propagation
    below runs on one configuration at a time -- so the axis has to be taken
    apart here rather than left for ``generate_slices``, which would silently
    walk the first configuration only.

    The configurations are averaged with equal weight, which is what a
    backscattered yield is: an incoherent sum over the thermal displacements
    the specimen passes through, not a coherent one. A configuration axis is
    therefore never carried through to the measurement, and an ensemble asked
    to keep one says so.

    Returns
    -------
    configurations : list of (BasePotential, float)
        Single-configuration potentials and the weight each carries. A
        potential with no ensemble axis gives the single pair
        ``[(potential, 1.0)]``.
    """
    ensemble_shape = potential.ensemble_shape

    if len(ensemble_shape) == 0:
        return [(potential, 1.0)]

    if any(not getattr(axis, "_ensemble_mean", True)
           for axis in potential.ensemble_axes_metadata):
        warnings.warn(
            "the potential asks to keep its configurations separate, but a "
            "backscattered yield is an incoherent sum over them; they are "
            "averaged",
            UserWarning,
        )

    configurations = [
        block.ravel()[0] for _, _, block in potential.generate_blocks(1)
    ]
    weight = 1.0 / len(configurations)

    return [(configuration, weight) for configuration in configurations]


def _propagate_block_intensities(
    ebsd: "EBSD", potential: BasePotential, arguments: dict
) -> np.ndarray:
    """One block of directions, as a dask task.

    The antialias loss cannot reach a lazy measurement's metadata, which is
    fixed when the graph is built rather than when it runs, so the check that
    would have produced it is made here instead and warns from inside the task.
    """
    intensities, loss = ebsd._propagate_block(potential=potential, **arguments)

    maximum = float(loss.max())
    if maximum > 0.05:
        warnings.warn(_antialias_message(maximum), AntialiasLossWarning)

    return np.asarray(intensities.get() if hasattr(intensities, "get") else intensities)


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
    backscatter_energy : float or sequence of float or BaseDistribution, optional
        Energy of the backscattered electrons [eV]. The beam travels at the
        probe's energy; a backscattered electron has lost some of it, so these
        must not exceed it. Giving more than one adds a leading
        :class:`.EnergyAxis` and costs proportionally more: the reciprocity
        waves travel at this energy, so each one needs its own wavelength,
        propagator and transmission function, and only the beam's propagation
        is shared. Defaults to the probe's energy, the elastic case, where the
        two wavefields share a transmission function.

        The weights of a distribution are recorded in the metadata rather than
        applied, since summing the energy axis is the consumer's business and
        the weights usually come from a Monte Carlo spectrum.
    depth_tolerance : float, optional
        Stop propagating once the depth weight still to be collected falls
        below this fraction of the total (default 1e-4). Slices past a short
        escape depth carry no weight, and propagating them is the dominant
        cost. Has no effect under the default uniform weighting, where the
        remaining weight only vanishes at the last slice.

        The bound is on the *discarded weight*; the error in any one direction
        can be a few times that, because the slices dropped are not average
        ones. Set to 0 to disable.
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
        backscatter_energy: Optional[
            float | Sequence[float] | np.ndarray | BaseDistribution
        ] = None,
        depth_tolerance: float = 1e-4,
        order: Literal[1, 2, "exact"] = "exact",
        device: Optional[str] = None,
    ):
        if isinstance(probe.energy, BaseDistribution):
            # The reciprocity waves would have to propagate at the backscattered
            # energy while the beam stays at its own, so the two wavefields could
            # no longer share a transmission function. Refuse it plainly rather
            # than letting _valid_energy raise "Energy is not defined".
            raise NotImplementedError(
                "EBSD does not support an energy ensemble; give the probe a "
                "single energy and combine the results yourself if you need a "
                "spread of backscattered energies"
            )

        self._potential = validate_potential(potential)
        self._probe = probe
        self._detector = detector
        self._potential_weighting = bool(potential_weighting)
        self._depth_weight = depth_weight
        self._backscatter_energy = backscatter_energy
        self._depth_tolerance = float(depth_tolerance)
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

    def _direction_blocks(
        self, max_batch: int | str, lazy: bool = False
    ) -> list[tuple[int, int]]:
        """Split the collected directions into batches that fit in memory.

        Blocks are the unit of work, so when they are going to become dask
        tasks there have to be enough of them to occupy the workers. The
        memory-derived size alone can leave three tasks on a twelve core
        machine; this only ever splits further, never coarser.
        """
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
        blocks = list(chunk_ranges(chunks)[0])

        if lazy and len(blocks) < _worker_count():
            size = int(np.ceil(len(self._detector) / _worker_count()))
            size = min(size, max(stop - start for start, stop in blocks))
            blocks = [
                (start, min(start + size, len(self._detector)))
                for start in range(0, len(self._detector), size)
            ]

        return blocks

    def scan(
        self,
        scan: Optional[BaseScan | Sequence] = None,
        max_batch_directions: int | str = "auto",
        lazy: Optional[bool] = None,
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
            If True, build a dask graph instead of computing, with one task per
            block of directions per backscattered energy. Defaults to the abTEM
            configuration. A lazy measurement carries no ``antialias_loss`` in
            its metadata, since that is only known once the graph runs; the
            check warns from inside the tasks instead.

            Defaults to abTEM's ``dask.lazy`` configuration, which ships as
            True, so ask for ``lazy=False`` to get an array back directly.
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
        lazy = validate_lazy(lazy)

        xp = get_array_module(self._device)

        # Only the geometry is needed to lay out the work, and every frozen
        # -phonon configuration shares it. Building each one is deferred to the
        # point of use, so that 400 of them are neither built serially while
        # the graph is assembled nor held in it at once.
        num_slices = self._potential.num_slices
        configurations = _potential_configurations(self._potential)

        depths = np.cumsum(np.asarray(self._potential.slice_thickness, dtype=float))
        weights = _validate_depth_weight(self._depth_weight, depths)

        probe = self._probe.copy()
        probe.grid.match(self._potential)
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

        backscatter_energies, energy_weights = _validate_backscatter_energy(
            self._backscatter_energy, energy
        )
        n_energies = len(backscatter_energies)

        blocks = self._direction_blocks(max_batch_directions, lazy=lazy)

        def block_arguments(backscatter_energy, start, stop):
            wave_vectors = xp.asarray(
                self._detector.transverse_wave_vectors(backscatter_energy),
                dtype=get_dtype(complex=False),
            )[start:stop]
            return dict(
                source=source,
                incident_norm=incident_norm,
                wave_vectors=wave_vectors,
                weights=weights,
                energy=energy,
                backscatter_energy=float(backscatter_energy),
                n_directions=stop - start,
                ensemble_shape=ensemble_shape,
            )

        if lazy:
            return self._lazy_measurement(
                configurations=configurations,
                block_arguments=block_arguments,
                blocks=blocks,
                backscatter_energies=backscatter_energies,
                energy_weights=energy_weights,
                ensemble_shape=ensemble_shape,
                ensemble_axes_metadata=ensemble_axes_metadata,
                energy=energy,
            )

        intensities = xp.zeros(
            (n_energies,) + ensemble_shape + (len(self._detector),), dtype=xp.float32
        )
        antialias_loss = xp.zeros((n_energies, len(self._detector)), dtype=xp.float32)

        progress = TqdmWrapper(
            total=len(configurations) * n_energies * len(blocks) * num_slices,
            enabled=pbar,
            leave=False,
        )
        try:
            for configuration, configuration_weight in configurations:
                built = configuration.build(lazy=False)
                for i, backscatter_energy in enumerate(backscatter_energies):
                    for start, stop in blocks:
                        block, loss = self._propagate_block(
                            potential=built,
                            progress=progress,
                            **block_arguments(backscatter_energy, start, stop),
                        )
                        intensities[i][..., start:stop] += configuration_weight * block
                        antialias_loss[i][start:stop] += configuration_weight * loss
        finally:
            progress.close_if_exists()

        max_loss = float(xp.max(antialias_loss))
        if max_loss > 0.05:
            warnings.warn(_antialias_message(max_loss), AntialiasLossWarning)

        if n_energies == 1:
            intensities = intensities[0]
            antialias_loss = antialias_loss[0]
        else:
            ensemble_axes_metadata = [
                EnergyAxis(values=tuple(float(e) for e in backscatter_energies))
            ] + ensemble_axes_metadata

        return self._to_measurement(
            intensities,
            ensemble_axes_metadata=ensemble_axes_metadata,
            energy=energy,
            antialias_loss=antialias_loss,
            energy_weights=energy_weights,
            backscatter_energies=backscatter_energies,
        )

    def _lazy_measurement(
        self,
        configurations,
        block_arguments,
        blocks,
        backscatter_energies,
        energy_weights,
        ensemble_shape,
        ensemble_axes_metadata,
        energy: float,
    ) -> DiffractionPatterns | SphericalPattern:
        """Assemble the same per-block computation into a dask graph.

        Each block of directions, at each backscattered energy and each frozen
        -phonon configuration, is one task. The blocks of a configuration share
        its built potential and the source, which the threaded scheduler passes
        by reference rather than copying. The wavefields a task propagates are
        its own, so only one block of them is resident at a time however many
        directions were asked for.
        """
        import dask

        n_energies = len(backscatter_energies)

        array = None
        for configuration, configuration_weight in configurations:
            # One task per configuration, so its blocks share the built
            # potential and dask frees it once they are done with it.
            built = dask.delayed(_build_potential, pure=True)(configuration)

            rows = []
            for backscatter_energy in backscatter_energies:
                columns = []
                for start, stop in blocks:
                    block = dask.delayed(_propagate_block_intensities, pure=True)(
                        self, built, block_arguments(backscatter_energy, start, stop)
                    )
                    columns.append(
                        da.from_delayed(
                            block,
                            shape=ensemble_shape + (stop - start,),
                            dtype=np.float32,
                        )
                    )
                rows.append(da.concatenate(columns, axis=-1))

            stacked = configuration_weight * da.stack(rows, axis=0)
            array = stacked if array is None else array + stacked

        if n_energies == 1:
            array = array[0]
        else:
            ensemble_axes_metadata = [
                EnergyAxis(values=tuple(float(e) for e in backscatter_energies))
            ] + ensemble_axes_metadata

        if self._detector.is_grid:
            # The two base axes of a diffraction pattern have to be one chunk.
            array = array.rechunk(array.chunks[:-1] + (-1,))

        return self._to_measurement(
            array,
            ensemble_axes_metadata=ensemble_axes_metadata,
            energy=energy,
            antialias_loss=None,
            energy_weights=energy_weights,
            backscatter_energies=backscatter_energies,
        )

    def _propagate_block(
        self,
        potential,
        source: Waves,
        incident_norm,
        wave_vectors,
        weights: np.ndarray,
        energy: float,
        backscatter_energy: float,
        n_directions: int,
        ensemble_shape: tuple[int, ...],
        progress: Optional[TqdmWrapper] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Co-propagate the source and one batch of reciprocity waves.

        Returns the block's own intensities and antialias losses rather than
        writing into a shared buffer, so the same routine serves the eager
        assembly and the lazy one, where each block is a separate task.

        Both wavefields are advanced through the same slice before their
        overlap is accumulated, so the depth-resolved source never has to be
        held in memory all at once. That is what makes a scan of many probe
        positions affordable; the cost is re-propagating the source once per
        batch of directions.

        The source travels at the beam energy and the reciprocity waves at the
        backscattered energy. When those differ the two wavefields need their
        own transmission function and propagator, since both depend on the
        wavelength; when they agree the pair is shared, halving the per-slice
        transmission work.
        """
        xp = get_array_module(self._device)

        # Unit-modulus plane waves: the normalization of the result comes from
        # dividing by the source weight of each slice, not from these.
        array = plane_waves(wave_vectors, potential.extent, potential.gpts)

        reciprocity = Waves(
            array,
            energy=backscatter_energy,
            extent=potential.extent,
            ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(len(array))))],
        )
        initial_norm = xp.sum(xp.abs(reciprocity.array) ** 2, axis=(-2, -1))

        intensities = xp.zeros(incident_norm.shape + (n_directions,), dtype=xp.float32)

        # The source is re-propagated for every block, so start from a copy.
        beam = source.copy()

        elastic = backscatter_energy == energy

        # Weight still to be collected at and below each slice. Once it is
        # negligible there is nothing left to gather and the remaining slices
        # are wasted propagation -- which is most of the specimen when the
        # escape depth is short compared to its thickness.
        remaining = np.cumsum(weights[::-1])[::-1]

        beam_propagator = FresnelPropagator()
        reciprocity_propagator = beam_propagator if elastic else FresnelPropagator()
        antialias_aperture = AntialiasAperture()

        for index, potential_slice in enumerate(potential.generate_slices()):
            if remaining[index] < self._depth_tolerance:
                break

            potential_slice = potential_slice.copy_to_device(self._device)

            beam_transmission = potential_slice.transmission_function(energy=energy)
            beam_transmission = antialias_aperture.bandlimit(
                beam_transmission, in_place=True
            )

            if elastic:
                reciprocity_transmission = beam_transmission
            else:
                reciprocity_transmission = potential_slice.transmission_function(
                    energy=backscatter_energy
                )
                reciprocity_transmission = antialias_aperture.bandlimit(
                    reciprocity_transmission, in_place=True
                )

            beam = conventional_multislice_step(
                beam,
                beam_transmission,
                propagator=beam_propagator,
                antialias_aperture=antialias_aperture,
                order=self._order,
            )
            reciprocity = conventional_multislice_step(
                reciprocity,
                reciprocity_transmission,
                propagator=reciprocity_propagator,
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
                    n_directions=n_directions,
                )

            if progress is not None:
                progress.update_if_exists(1)

        final_norm = xp.sum(xp.abs(reciprocity.array) ** 2, axis=(-2, -1))

        # incident_norm flattens the scan positions; give them back their shape
        intensities = intensities.reshape(ensemble_shape + (n_directions,))

        return intensities, 1.0 - final_norm / initial_norm

    def _accumulate(
        self,
        beam: Waves,
        reciprocity: Waves,
        projected_potential,
        incident_norm,
        weight: float,
        intensities,
        n_directions: int,
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
        #
        # The squared magnitude of the reciprocity waves is the largest array in
        # the loop, and taking it with abs()**2 costs more than the contraction
        # it feeds. Viewing the complex array as float32 gives the real and
        # imaginary parts interleaved and contiguous, so squaring every
        # component and contracting against the source with each of its values
        # repeated twice computes the same sum in one pass over contiguous
        # memory -- about seven times faster than abs()**2 on the strided
        # halves, and exactly equal.
        real_view = reciprocity.array.view(get_dtype(complex=False))
        real_view = real_view.reshape(n_directions, -1)

        # The interleave is built from a one-dimensional repeat rather than
        # `repeat(source_flat, 2, axis=-1)`. The two are exactly equal -- row
        # -major order puts each position's pixels contiguously, so duplicating
        # pairwise reproduces the interleave -- but cupy's axis-ed path is
        # pathologically slow: 482 ms against 0.0 ms for half a megabyte, which
        # was the whole of a 169x slowdown on the GPU.
        source_flat = source.reshape(-1, real_view.shape[1] // 2)
        source_interleaved = xp.repeat(source_flat.ravel(), 2).reshape(
            source_flat.shape[0], -1
        )

        overlap = (real_view * real_view) @ source_interleaved.T
        overlap = overlap / incident_norm[None]

        intensities += weight * overlap.T.reshape(intensities.shape)

    def _to_measurement(
        self,
        intensities,
        ensemble_axes_metadata: list[AxisMetadata],
        energy: float,
        antialias_loss,
        energy_weights: Optional[np.ndarray] = None,
        backscatter_energies: Optional[np.ndarray] = None,
    ) -> DiffractionPatterns | SphericalPattern:
        """Wrap the collected intensities in the matching measurement type."""
        array: np.ndarray | da.core.Array
        if isinstance(intensities, da.core.Array):
            array = intensities
        else:
            array = np.asarray(
                intensities.get() if hasattr(intensities, "get") else intensities
            )
        metadata = {"energy": energy, "label": "backscattered intensity"}

        if antialias_loss is not None:
            loss = np.asarray(
                antialias_loss.get()
                if hasattr(antialias_loss, "get")
                else antialias_loss
            )
            metadata["antialias_loss_max"] = float(loss.max())
            metadata["antialias_loss_mean"] = float(loss.mean())

        if (
            backscatter_energies is not None
            and energy_weights is not None
            and len(backscatter_energies) > 1
        ):
            # Kept rather than applied: the weights belong to whoever sums the
            # energy axis, and a Monte Carlo spectrum is the usual source of
            # them.
            metadata["backscatter_energies"] = [float(e) for e in backscatter_energies]
            metadata["energy_weights"] = [float(w) for w in energy_weights]

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
