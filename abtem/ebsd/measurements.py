"""Backscattered intensity sampled on the unit hemisphere."""

from __future__ import annotations

import warnings
from typing import Optional, Sequence, cast

import dask.array as da
import numpy as np
import zarr

from abtem.core.axes import AxisMetadata, RealSpaceAxis
from abtem.core.utils import CopyMixin, EqualityMixin
from abtem.ebsd.projections import (
    HemisphereProjection,
    bin_directions,
    pixel_centers,
    validate_projection,
)
from abtem.measurements import Images

__all__ = ["ReferencePatternImages", "SphericalPattern", "SparseProjectionWarning"]


class SparseProjectionWarning(UserWarning):
    """The projected image has pixels no sampled direction reached."""


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
        array: np.ndarray | da.core.Array,
        directions: np.ndarray,
        ensemble_axes_metadata: Optional[list[AxisMetadata]] = None,
        metadata: Optional[dict] = None,
    ):
        # A dask array is kept as it is; np.asarray would compute it.
        if not isinstance(array, da.core.Array):
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
    def array(self) -> np.ndarray | da.core.Array:
        """The intensities, of shape ``(..., N)``."""
        return self._array

    @property
    def is_lazy(self) -> bool:
        """Whether the intensities are a dask array awaiting computation."""
        return isinstance(self._array, da.core.Array)

    def compute(self, **kwargs) -> "SphericalPattern":
        """Compute a lazy pattern, returning one holding a plain array.

        Parameters
        ----------
        kwargs :
            Passed to :meth:`dask.array.Array.compute`.

        Returns
        -------
        pattern : SphericalPattern
            Self, if the pattern was not lazy.
        """
        if not isinstance(self._array, da.core.Array):
            return self

        return self.__class__(
            self._array.compute(**kwargs),
            self._directions,
            ensemble_axes_metadata=self._ensemble_axes_metadata,
            metadata=self._metadata,
        )

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

    def interpolate(
        self,
        gpts: int,
        projection: str | HemisphereProjection = "lambert",
    ) -> ReferencePatternImages:
        """Sample the intensities at the nodes of a square grid.

        This differs from :meth:`project`, which *bins* the directions into
        pixels and therefore needs more directions than pixels to avoid holes.
        Here the value is evaluated *at* each grid node by interpolating the
        sampled directions, so any grid size is well defined.

        Node sampling is what a master pattern file wants, because a consumer
        reading it interpolates between nodes. When the pattern was calculated
        on the very grid requested -- build it with the matching `projection`
        and ``direction_gpts=gpts`` -- the interpolation is exact and this is a
        lossless repackaging.

        Parameters
        ----------
        gpts : int
            Number of grid nodes along each axis, spanning ``[-1, 1]``.
        projection : str or HemisphereProjection, optional
            One of ``'lambert'`` (default, the master-pattern convention) or
            ``'stereographic'``.

        Returns
        -------
        images : ReferencePatternImages
            Nodes outside the projection's domain -- the corners the
            stereographic disk does not cover -- are zero.
        """
        from scipy.interpolate import griddata  # type: ignore[import-untyped]

        projection = validate_projection(projection)

        if self.is_lazy:
            raise RuntimeError(
                "compute() the pattern before interpolating it; the "
                "interpolation is not built as a dask graph"
            )

        northern = self._directions[:, 2] >= 0.0
        source = projection.project(self._directions[northern])

        nodes = np.linspace(-1.0, 1.0, gpts)
        x, y = np.meshgrid(nodes, nodes, indexing="ij")
        target = np.stack([x.ravel(), y.ravel()], axis=1)
        inside = projection.domain_mask(target)

        flat = self._array.reshape(-1, len(self._directions))[:, northern]

        images = np.zeros((len(flat), gpts * gpts))
        fraction_filled = 0.0
        for i, values in enumerate(flat):
            interpolated = griddata(source, values, target[inside], method="linear")
            # Nodes beyond the convex hull of the sampled directions come back
            # as NaN; fall back to the nearest sample rather than a hole.
            missing = np.isnan(interpolated)
            if missing.any():
                fraction_filled = max(fraction_filled, float(missing.mean()))
                interpolated[missing] = griddata(
                    source, values, target[inside][missing], method="nearest"
                )
            images[i, inside] = interpolated

        if fraction_filled > 0.001:
            warnings.warn(
                f"{fraction_filled:.1%} of the nodes lie outside the sampled "
                f"directions and were filled from the nearest one, which shows "
                f"up as flat patches near the edges. Sample the pattern in the "
                f"{projection.name} projection to cover the grid exactly.",
                SparseProjectionWarning,
            )

        images = images.reshape(self.ensemble_shape + (gpts, gpts))

        return ReferencePatternImages(
            images,
            sampling=2.0 / (gpts - 1),
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata={**self._metadata, "projection": projection.name},
        )

    def project(
        self,
        gpts: int,
        projection: str | HemisphereProjection = "stereographic",
    ) -> ReferencePatternImages:
        """Bin the intensities into a square image.

        Each pixel is the mean of the directions falling inside it, so the
        directions have to outnumber the pixels or the image has holes. Use
        :meth:`interpolate` to evaluate at grid nodes instead, which is what a
        master pattern file needs.

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

        if self.is_lazy:
            raise RuntimeError(
                "compute() the pattern before projecting it; the binning is "
                "not built as a dask graph"
            )

        flat = self._array.reshape(-1, len(self._directions))
        binned = [
            bin_directions(
                self._directions, values, gpts, projection, return_counts=True
            )
            for values in flat
        ]
        images = np.stack([image for image, _ in binned])
        images = images.reshape(self.ensemble_shape + (gpts, gpts))

        self._warn_if_sparse(binned[0][1], gpts, projection)

        return ReferencePatternImages(
            images,
            sampling=2.0 / gpts,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata={**self._metadata, "projection": projection.name},
        )

    @staticmethod
    def _warn_if_sparse(counts, gpts, projection, tolerance=0.01):
        """Warn if the image has holes the directions never reached.

        Directions sampled on one projection's even grid are uneven in any
        other, so projecting a pattern through a projection it was not sampled
        for leaves a moire of empty pixels. Pixels outside the projection's
        domain -- the corners the stereographic disk does not cover -- are
        legitimately empty and do not count.
        """
        inside = projection.domain_mask(pixel_centers(gpts)).reshape(gpts, gpts)
        if not inside.any():
            return

        empty = float((counts[inside] == 0).mean())
        if empty > tolerance:
            warnings.warn(
                f"{empty:.1%} of the pixels inside the {projection.name} "
                f"projection have no sampled direction in them. Sample the "
                f"directions in the projection you mean to view, or lower gpts.",
                SparseProjectionWarning,
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

        if self.is_lazy:
            raise RuntimeError("compute() the pattern before writing it")

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
