"""Electron backscatter diffraction (EBSD) by multislice reciprocity."""

from abtem.ebsd.detectors import BackscatterDetector, small_angle_error
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
from abtem.ebsd.reciprocity import EBSD
from abtem.ebsd.reference import (
    EBSDReferencePattern,
    patch_half_angle,
    recommended_sampling,
)

__all__ = [
    "EBSD",
    "EBSDReferencePattern",
    "patch_half_angle",
    "recommended_sampling",
    "BackscatterDetector",
    "small_angle_error",
    "SphericalPattern",
    "ReferencePatternImages",
    "SparseProjectionWarning",
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
