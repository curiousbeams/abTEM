"""How fine the multislice grid has to be.

One condition only: that the grid resolves the atoms' potential. The antialias
aperture passes spatial frequencies up to a third of the inverse sampling, and
whatever the potential scatters beyond it is lost -- silently, since the
transmission function is band-limited before it acts. The collected directions
set no condition of their own: the reciprocity waves are carried as periodic
envelopes (see :mod:`abtem.ebsd.reciprocity`), so a wave's direction uses up
none of the aperture, which limits only the scattering about it.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
from ase import Atoms

from abtem.potentials.iam import Potential

__all__ = ["AntialiasLossWarning", "potential_sampling", "scattering_power_lost"]

#: Fraction of the scattering power a sampling may leave out before it is
#: warned about. Measured on silicon, the error in the pattern runs at one to
#: three times this fraction (see :func:`potential_sampling`).
_UNDERSAMPLED = 0.02


class AntialiasLossWarning(UserWarning):
    """The sampling leaves part of the atoms' scattering outside the aperture.

    Its own class so that a caller constructing many calculations -- a
    reference pattern, say -- can quiet the individual warnings and give one
    for the whole run instead.
    """


@lru_cache(maxsize=32)
def _scattering_power(number: int) -> tuple[np.ndarray, np.ndarray]:
    """An atom's projected scattering power, accumulated outwards in frequency.

    The projected potential of an atom is sharply peaked, and its transform
    decays slowly -- the electron scattering factor has a Rutherford tail, so
    there is no frequency beyond which it truly vanishes. What can be asked is
    where all but a given fraction of its power lies.

    Returns
    -------
    frequencies, fraction : np.ndarray
        Spatial frequencies [1 / Å] in increasing order, and the fraction of
        the power at or below each.
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
    return radial.ravel()[order], cumulative / cumulative[-1]


def _scattering_power_cutoff(number: int, tolerance: float) -> float:
    """Spatial frequency holding all but `tolerance` of an atom's scattering."""
    frequencies, fraction = _scattering_power(number)
    return float(frequencies[np.searchsorted(fraction, 1.0 - tolerance)])


def scattering_power_lost(atoms: Atoms, sampling: float) -> float:
    """Fraction of the atoms' scattering power a sampling leaves out.

    The antialias aperture passes spatial frequencies up to a third of the
    inverse sampling, and the scattering of the potential beyond it is simply
    lost -- silently, since the transmission function is band-limited before it
    acts. This is that fraction, for the most demanding element present.

    Parameters
    ----------
    atoms : ase.Atoms
        The specimen. Only which elements are present matters.
    sampling : float
        Real-space sampling [Å].

    Returns
    -------
    fraction : float
    """
    aperture = 1.0 / (3.0 * float(sampling))
    lost = 0.0
    for number in np.unique(atoms.numbers):
        frequencies, fraction = _scattering_power(int(number))
        index = (
            min(np.searchsorted(frequencies, aperture, side="right"), len(fraction)) - 1
        )
        lost = max(lost, 1.0 - float(fraction[max(index, 0)]))
    return lost


def _warn_if_undersampled(atoms: Atoms, sampling: float, what: str) -> None:
    """Warn when `sampling` loses more of the scattering than is worth risking."""
    lost = scattering_power_lost(atoms, sampling)
    if lost > _UNDERSAMPLED:
        warnings.warn(
            f"{what} sampling of {sampling:.3f} Å leaves {lost:.1%} of the atoms' "
            f"scattering power outside the antialias aperture; "
            f"{potential_sampling(atoms):.3f} Å or finer resolves them",
            AntialiasLossWarning,
        )


def potential_sampling(atoms: Atoms, tolerance: float = 0.01) -> float:
    """Sampling that resolves the projected potential of `atoms`.

    The one condition on the sampling. The reciprocity waves are carried as
    periodic envelopes (see :mod:`abtem.ebsd.reciprocity`), so a wave's
    direction uses up none of the antialias aperture, and the collected angles
    set no condition of their own: the aperture limits only the scattering
    about each wave, and scattering power beyond it is simply lost. This
    returns the sampling that keeps all but `tolerance` of an isolated atom's
    projected scattering power inside the aperture, taking the most demanding
    element present.

    Calibrated against a convergence test on silicon at 30 kV, with the
    earlier potential-weighted source, where the error in the pattern relative
    to a far finer grid ran at one to three times the power left outside the
    aperture:

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

    Static atoms emit from points, which a finer grid resolves ever more
    sharply against the peaks the waves channel onto them: along ``[001]`` in
    silicon the yield still moves by 2% between 0.063 and 0.032 Å. Atoms in
    thermal motion do not -- with :class:`~abtem.FrozenPhonons` at 0.076 Å it
    moves by 0.3%.

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
