"""Electron backscatter diffraction (EBSD) by multislice reciprocity."""

from abtem.ebsd.detectors import BackscatterDetector, small_angle_error
from abtem.ebsd.emsoft import write_emsoft_master_pattern
from abtem.ebsd.measurements import (
    ReferencePatternImages,
    SparseProjectionWarning,
    SphericalPattern,
)
from abtem.ebsd.orientations import (
    bulk_block,
    estimate_repetitions,
    fibonacci_hemisphere,
    rotated_slab,
    zone_axis_rotation,
)
from abtem.ebsd.projections import (
    HemisphereProjection,
    SquareLambertProjection,
    StereographicProjection,
    bin_directions,
    pixel_centers,
    validate_projection,
)
from abtem.ebsd.reciprocity import EBSD, AntialiasLossWarning
from abtem.ebsd.reference import (
    EBSDReferencePattern,
    fft_friendly_gpts,
    maximum_sampling,
    patch_half_angle,
    potential_sampling,
    recommended_sampling,
)

__all__ = [
    "EBSD",
    "AntialiasLossWarning",
    "EBSDReferencePattern",
    "patch_half_angle",
    "fft_friendly_gpts",
    "maximum_sampling",
    "potential_sampling",
    "recommended_sampling",
    "BackscatterDetector",
    "small_angle_error",
    "SphericalPattern",
    "ReferencePatternImages",
    "SparseProjectionWarning",
    "write_emsoft_master_pattern",
    "pixel_centers",
    "fibonacci_hemisphere",
    "zone_axis_rotation",
    "bulk_block",
    "estimate_repetitions",
    "rotated_slab",
    "HemisphereProjection",
    "StereographicProjection",
    "SquareLambertProjection",
    "validate_projection",
    "bin_directions",
]
