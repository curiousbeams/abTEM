"""Backscattered intensity sampled on the unit hemisphere."""

from __future__ import annotations

from typing import Optional, Sequence, cast

import numpy as np
import zarr

from abtem.core.axes import AxisMetadata, RealSpaceAxis
from abtem.core.utils import CopyMixin, EqualityMixin
from abtem.ebsd.projections import (
    HemisphereProjection,
    bin_directions,
    validate_projection,
)
from abtem.measurements import Images

__all__ = ["ReferencePatternImages", "SphericalPattern"]


class ReferencePatternImages(Images):
    """An EBSD reference pattern projected onto a square image.

    Identical to :class:`~abtem.measurements.Images` except that the two base
    axes are the dimensionless coordinates of the projection, spanning
    ``[-1, 1]``, rather than a distance in Ångström.
    """

    @property
    def base_axes_metadata(self) -> list[AxisMetadata]:
        return [
            RealSpaceAxis(
                label="x", sampling=self.sampling[0], units="", tex_label="$x$"
            ),
            RealSpaceAxis(
                label="y", sampling=self.sampling[1], units="", tex_label="$y$"
            ),
        ]


class SphericalPattern(CopyMixin, EqualityMixin):
    """Backscattered intensity sampled at directions on the unit hemisphere.

    This is the natural output of a reference-pattern calculation: the
    directions are chosen to tile the hemisphere evenly, which makes them an
    unstructured list rather than a grid. Use :meth:`project` to turn it into a
    square image.

    Unlike most abTEM measurements this is not an
    :class:`~abtem.array.ArrayObject`. Its base axis is a list of ``(N, 3)``
    direction vectors, which does not fit the scalar-valued axis metadata that
    machinery serializes; :meth:`to_zarr` stores the directions as their own
    array instead.

    Parameters
    ----------
    array : np.ndarray
        Intensities of shape ``(..., N)``, where the last axis runs over the
        directions and any leading axes are ensemble axes.
    directions : np.ndarray
        Unit vectors of shape ``(N, 3)`` in the crystal frame.
    ensemble_axes_metadata : list of AxisMetadata, optional
        Metadata for the leading axes of `array`.
    metadata : dict, optional
        Measurement metadata.
    """

    def __init__(
        self,
        array: np.ndarray,
        directions: np.ndarray,
        ensemble_axes_metadata: Optional[list[AxisMetadata]] = None,
        metadata: Optional[dict] = None,
    ):
        array = np.asarray(array)
        directions = np.asarray(directions, dtype=float)

        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError(
                f"directions must have shape (N, 3), got {directions.shape}"
            )

        if array.shape[-1] != len(directions):
            raise ValueError(
                f"the last axis of array has length {array.shape[-1]} but there "
                f"are {len(directions)} directions"
            )

        self._array = array
        self._directions = directions
        self._ensemble_axes_metadata = list(ensemble_axes_metadata or [])
        self._metadata = dict(metadata or {})

        if len(self._ensemble_axes_metadata) != array.ndim - 1:
            raise ValueError(
                f"got {len(self._ensemble_axes_metadata)} ensemble axes for an "
                f"array with {array.ndim - 1} leading axes"
            )

    @property
    def array(self) -> np.ndarray:
        """The intensities, of shape ``(..., N)``."""
        return self._array

    @property
    def directions(self) -> np.ndarray:
        """The sampled directions, of shape ``(N, 3)``."""
        return self._directions

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        """Metadata for the leading axes of :attr:`array`."""
        return self._ensemble_axes_metadata

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        """Shape of the leading axes of :attr:`array`."""
        return self._array.shape[:-1]

    @property
    def metadata(self) -> dict:
        """Measurement metadata."""
        return self._metadata

    def __len__(self) -> int:
        return len(self._directions)

    def project(
        self,
        gpts: int,
        projection: str | HemisphereProjection = "stereographic",
    ) -> ReferencePatternImages:
        """Bin the intensities into a square image.

        Parameters
        ----------
        gpts : int
            Number of pixels along each axis of the output image.
        projection : str or HemisphereProjection, optional
            One of ``'stereographic'`` (default) or ``'lambert'``.

        Returns
        -------
        images : ReferencePatternImages
            The projected pattern. Pixels no direction falls into are zero.
        """
        projection = validate_projection(projection)

        flat = self._array.reshape(-1, len(self._directions))
        images = np.stack(
            [
                bin_directions(self._directions, values, gpts, projection)
                for values in flat
            ]
        )
        images = images.reshape(self.ensemble_shape + (gpts, gpts))

        return ReferencePatternImages(
            images,
            sampling=2.0 / gpts,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata={**self._metadata, "projection": projection.name},
        )

    def show(
        self,
        gpts: int = 256,
        projection: str | HemisphereProjection = "stereographic",
        **kwargs,
    ):
        """Project the pattern and show it.

        Parameters
        ----------
        gpts : int, optional
            Number of pixels along each axis of the image shown (default 256).
        projection : str or HemisphereProjection, optional
            One of ``'stereographic'`` (default) or ``'lambert'``.
        kwargs :
            Passed to :meth:`abtem.measurements.Images.show`.
        """
        return self.project(gpts, projection).show(**kwargs)

    def to_zarr(self, url: str, overwrite: bool = False) -> None:
        """Write the pattern to a zarr store.

        Parameters
        ----------
        url : str
            Location of the zarr store.
        overwrite : bool, optional
            If True, replace an existing store (default False).
        """
        from abtem.core.axes import axis_to_dict

        root = zarr.open_group(url, mode="w" if overwrite else "w-")
        root.create_array("array", shape=self._array.shape, dtype=self._array.dtype)[
            :
        ] = self._array
        root.create_array(
            "directions", shape=self._directions.shape, dtype=self._directions.dtype
        )[:] = self._directions
        root.attrs["metadata"] = self._metadata
        root.attrs["ensemble_axes_metadata"] = [
            axis_to_dict(axis) for axis in self._ensemble_axes_metadata
        ]

    @classmethod
    def from_zarr(cls, url: str) -> "SphericalPattern":
        """Read a pattern written by :meth:`to_zarr`.

        Parameters
        ----------
        url : str
            Location of the zarr store.

        Returns
        -------
        pattern : SphericalPattern
        """
        from abtem.core.axes import axis_from_dict

        root = zarr.open_group(url, mode="r")
        # zarr types its attrs as arbitrary JSON; these two were written by
        # to_zarr, so they are a list and a mapping.
        axes = cast(list, root.attrs["ensemble_axes_metadata"])
        metadata = cast(dict, root.attrs["metadata"])

        return cls(
            np.asarray(root["array"]),
            np.asarray(root["directions"]),
            ensemble_axes_metadata=[axis_from_dict(d) for d in axes],
            metadata=dict(metadata),
        )

    @classmethod
    def concatenate(cls, patterns: Sequence["SphericalPattern"]) -> "SphericalPattern":
        """Join patterns sampled at different directions.

        Used to assemble the per-patch results of a reference-pattern
        calculation into a single pattern covering the hemisphere.

        Parameters
        ----------
        patterns : sequence of SphericalPattern
            Patterns to join. All must share the same ensemble shape.

        Returns
        -------
        pattern : SphericalPattern
        """
        patterns = list(patterns)

        if not patterns:
            raise ValueError("cannot concatenate an empty sequence of patterns")

        shapes = {pattern.ensemble_shape for pattern in patterns}
        if len(shapes) != 1:
            raise ValueError(f"patterns have mismatched ensemble shapes: {shapes}")

        return cls(
            np.concatenate([pattern.array for pattern in patterns], axis=-1),
            np.concatenate([pattern.directions for pattern in patterns], axis=0),
            ensemble_axes_metadata=patterns[0].ensemble_axes_metadata,
            metadata=patterns[0].metadata,
        )
