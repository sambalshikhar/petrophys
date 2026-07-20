"""
Borehole image log containers and DLIS readers.

Electrical image tools such as Schlumberger FMI export per-depth arrays of
pad/flap button values. These are not ordinary scalar curves: the useful
object is a 2D image with depth on one axis and borehole samples on the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np

try:
    from dlisio import dlis

    HAS_DLISIO = True
except ImportError:
    HAS_DLISIO = False

try:
    import lasio

    HAS_LASIO = True
except ImportError:
    HAS_LASIO = False

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


DEFAULT_FMI_CHANNELS = (
    "FMPA",
    "FMFA",
    "FMPB",
    "FMFB",
    "FMPC",
    "FMFC",
    "FMPD",
    "FMFD",
)


def _clean_channel_array(values: np.ndarray, null_value: float) -> np.ndarray:
    arr = np.squeeze(np.asarray(values))
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Image channel must be 2D after squeeze; got {values.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32, copy=True)
        arr[arr == null_value] = np.nan
    return arr


def _finite_min_max(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.nan, np.nan
    return float(np.nanmin(finite)), float(np.nanmax(finite))


@dataclass
class BoreholeImageData:
    """
    Borehole image log data.

    Parameters
    ----------
    depth : np.ndarray
        Measured depth array.
    image : np.ndarray
        2D image data with shape ``(n_depth, n_borehole_samples)``.
    channel_names : list of str
        DLIS/LAS channels concatenated into the image, in left-to-right order.
    samples_per_channel : list of int
        Number of button/samples contributed by each channel.
    image_type : str
        Image style, for example ``"dynamic"`` or ``"static"``.
    depth_unit : str
        Depth unit.
    value_unit : str
        Image value unit, often blank for processed FMI dynamic image values.
    metadata : dict
        Source and frame metadata.
    """

    depth: np.ndarray
    image: np.ndarray
    channel_names: list[str]
    samples_per_channel: list[int]
    image_type: str = "dynamic"
    depth_unit: str = "m"
    value_unit: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(len(self.depth))

    @property
    def n_borehole_samples(self) -> int:
        return int(self.image.shape[1])

    @property
    def top(self) -> Optional[float]:
        finite = np.asarray(self.depth, dtype=float)
        finite = finite[np.isfinite(finite)]
        return float(np.nanmin(finite)) if finite.size else None

    @property
    def bottom(self) -> Optional[float]:
        finite = np.asarray(self.depth, dtype=float)
        finite = finite[np.isfinite(finite)]
        return float(np.nanmax(finite)) if finite.size else None

    @classmethod
    def from_arrays(
        cls,
        depth: Union[np.ndarray, Sequence[float]],
        image: Union[np.ndarray, Sequence[Sequence[float]]],
        channel_names: Optional[Sequence[str]] = None,
        samples_per_channel: Optional[Sequence[int]] = None,
        image_type: str = "dynamic",
        depth_unit: str = "m",
        value_unit: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> "BoreholeImageData":
        depth_arr = np.asarray(depth, dtype=float)
        image_arr = np.asarray(image)
        if image_arr.ndim != 2:
            raise ValueError("image must have shape (n_depth, n_borehole_samples)")
        if image_arr.shape[0] != depth_arr.size:
            raise ValueError("image first dimension must match depth length")

        if channel_names is None:
            channel_names = ["IMAGE"]
        if samples_per_channel is None:
            samples_per_channel = [int(image_arr.shape[1])]
        if sum(samples_per_channel) != image_arr.shape[1]:
            raise ValueError("samples_per_channel must sum to image width")

        return cls(
            depth=depth_arr,
            image=image_arr,
            channel_names=[str(name) for name in channel_names],
            samples_per_channel=[int(value) for value in samples_per_channel],
            image_type=image_type,
            depth_unit=depth_unit,
            value_unit=value_unit,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_dlis(
        cls,
        filepath: Union[str, Path],
        frame_name: Optional[str] = None,
        image_channels: Optional[Sequence[str]] = None,
        logical_file: int = 0,
        image_type: str = "dynamic",
        null_value: float = -999.25,
        normalize: bool = False,
    ) -> "BoreholeImageData":
        """
        Load a borehole image frame from DLIS.

        By default, this detects common FMI processed image channels
        (``FMPA/FMFA/.../FMFD``). If those are absent, it uses channels whose
        DLIS dimensions indicate more than one per-depth sample.
        """
        if not HAS_DLISIO:
            raise ImportError("dlisio is required: pip install dlisio")

        physical_file = dlis.load(str(filepath))
        try:
            logical_files = list(physical_file)
            if logical_file < 0 or logical_file >= len(logical_files):
                raise IndexError(f"Logical file index out of range: {logical_file}")
            logical = logical_files[logical_file]

            frame = _select_image_frame(logical.frames, frame_name, image_channels)
            frame_data = frame.curves()
            depth_name = getattr(frame, "index", None) or "TDEP"
            if depth_name not in frame_data.dtype.names:
                raise ValueError(f"Depth channel {depth_name!r} not found in frame")

            depth = np.asarray(frame_data[depth_name], dtype=float).reshape(-1)
            depth_channel = _channel_by_name(frame.channels, depth_name)
            depth_unit = getattr(depth_channel, "units", "") or "m"

            selected = _select_image_channels(frame.channels, image_channels)
            arrays = []
            names = []
            widths = []
            units = []
            for channel in selected:
                arr = _clean_channel_array(frame_data[channel.name], null_value)
                arrays.append(arr)
                names.append(channel.name)
                widths.append(int(arr.shape[1]))
                units.append(getattr(channel, "units", "") or "")

            if not arrays:
                raise ValueError("No image channels found in DLIS frame")

            image = np.concatenate(arrays, axis=1)
            if normalize:
                image = _normalize_image(image)

            origins = list(getattr(logical, "origins", []) or [])
            origin = origins[0] if origins else None
            metadata = {
                "source": str(filepath),
                "logical_file": logical_file,
                "frame": frame.name,
                "spacing": getattr(frame, "spacing", None),
                "index": depth_name,
                "well": getattr(origin, "well_name", "") if origin else "",
                "field": getattr(origin, "field_name", "") if origin else "",
                "producer": getattr(origin, "producer_name", "") if origin else "",
            }

            return cls.from_arrays(
                depth=depth,
                image=image,
                channel_names=names,
                samples_per_channel=widths,
                image_type=image_type,
                depth_unit=depth_unit,
                value_unit=_common_unit(units),
                metadata=metadata,
            )
        finally:
            physical_file.close()

    @classmethod
    def from_las(
        cls,
        filepath: Union[str, Path],
        channel_prefixes: Optional[Sequence[str]] = None,
        image_type: str = "dynamic",
        null_value: Optional[float] = None,
        normalize: bool = False,
    ) -> "BoreholeImageData":
        """
        Reconstruct a borehole image from flattened LAS button curves.

        LAS cannot natively preserve DLIS-style array channels, so FMI images
        are commonly exported as curves such as ``FMPA_0`` ... ``FMPA_23``.
        This loader groups those numbered curves back into a 2D image.
        """
        if not HAS_LASIO:
            raise ImportError("lasio is required: pip install lasio")

        las = lasio.read(str(filepath))
        curve_names = [curve.mnemonic for curve in las.curves]
        depth_name = curve_names[0] if curve_names else None
        depth = np.asarray(las.index, dtype=float)
        depth_unit = getattr(las.curves[0], "unit", "") if las.curves else ""

        if null_value is None:
            try:
                null_value = float(las.well.NULL.value)
            except (AttributeError, TypeError, ValueError):
                null_value = -999.25

        groups = _collect_flattened_curve_groups(
            [name for name in curve_names if name != depth_name],
            channel_prefixes,
        )
        if not groups:
            raise ValueError("No flattened borehole image curves found in LAS")

        units = {curve.mnemonic.upper(): curve.unit or "" for curve in las.curves}
        arrays = []
        names = []
        widths = []
        for base, names_for_base in groups:
            cols = []
            for name in names_for_base:
                arr = np.asarray(las[name], dtype=float)
                arr[arr == null_value] = np.nan
                if arr.size != depth.size:
                    raise ValueError(
                        f"Curve {name!r} length {arr.size} does not match depth "
                        f"length {depth.size}"
                    )
                cols.append(arr.reshape(-1))
            arrays.append(np.column_stack(cols))
            names.append(base)
            widths.append(len(cols))

        image = np.concatenate(arrays, axis=1)
        if normalize:
            image = _normalize_image(image)

        metadata = {
            "source": str(filepath),
            "source_format": "LAS",
            "well": las.well.WELL.value if hasattr(las.well, "WELL") else "",
            "index": depth_name,
        }

        return cls.from_arrays(
            depth=depth,
            image=image,
            channel_names=names,
            samples_per_channel=widths,
            image_type=image_type,
            depth_unit=depth_unit or "m",
            value_unit=_common_unit(
                [
                    units.get(name.upper(), "")
                    for _, names_for_base in groups
                    for name in names_for_base
                ]
            ),
            metadata=metadata,
        )

    @classmethod
    def from_well(
        cls,
        well: "Well",
        channel_prefixes: Optional[Sequence[str]] = None,
        image_type: str = "dynamic",
        normalize: bool = False,
    ) -> "BoreholeImageData":
        """Reconstruct a borehole image from flattened curves in a Well."""
        if getattr(well, "depth", None) is None:
            raise ValueError("Well depth is required to assemble a borehole image")

        depth = np.asarray(well.depth, dtype=float)
        groups = _collect_flattened_curve_groups(
            getattr(well, "curves", []), channel_prefixes
        )
        if not groups:
            raise ValueError("No flattened borehole image curves found in Well")

        arrays = []
        names = []
        widths = []
        units = []
        for base, names_for_base in groups:
            cols = []
            for name in names_for_base:
                curve = well.get(name)
                if curve is None:
                    raise ValueError(f"Curve {name!r} not found in Well")
                arr = np.asarray(curve.data, dtype=float)
                if arr.size != depth.size:
                    raise ValueError(
                        f"Curve {name!r} length {arr.size} does not match depth "
                        f"length {depth.size}"
                    )
                cols.append(arr.reshape(-1))
                units.append(curve.unit or "")
            arrays.append(np.column_stack(cols))
            names.append(base)
            widths.append(len(cols))

        image = np.concatenate(arrays, axis=1)
        if normalize:
            image = _normalize_image(image)

        return cls.from_arrays(
            depth=depth,
            image=image,
            channel_names=names,
            samples_per_channel=widths,
            image_type=image_type,
            depth_unit=getattr(well, "depth_unit", "") or "m",
            value_unit=_common_unit(units),
            metadata={
                "source": "Well",
                "source_format": "Well",
                "well": getattr(well, "name", ""),
            },
        )

    def filter_by_depth(self, top: float, bottom: float) -> "BoreholeImageData":
        lo, hi = sorted((float(top), float(bottom)))
        mask = (self.depth >= lo) & (self.depth <= hi)
        metadata = dict(self.metadata)
        navigation = metadata.get("_navigation")
        if isinstance(navigation, dict):
            metadata["_navigation"] = {
                key: np.asarray(value)[mask]
                for key, value in navigation.items()
                if np.asarray(value).shape[:1] == self.depth.shape[:1]
            }
        return BoreholeImageData(
            depth=self.depth[mask],
            image=self.image[mask],
            channel_names=list(self.channel_names),
            samples_per_channel=list(self.samples_per_channel),
            image_type=self.image_type,
            depth_unit=self.depth_unit,
            value_unit=self.value_unit,
            metadata=metadata,
        )

    def normalized_image(self) -> np.ndarray:
        """Return image values scaled to 0-1 for display."""
        return _normalize_image(self.image)

    def per_depth_statistics(self) -> dict[str, np.ndarray]:
        """Return image mean, min, max, standard deviation, and coverage."""
        arr = np.asarray(self.image, dtype=float)
        finite = np.isfinite(arr)
        any_row = finite.any(axis=1)
        stats = {
            "mean": np.full(arr.shape[0], np.nan, dtype=float),
            "min": np.full(arr.shape[0], np.nan, dtype=float),
            "max": np.full(arr.shape[0], np.nan, dtype=float),
            "std": np.full(arr.shape[0], np.nan, dtype=float),
            "coverage": np.mean(finite, axis=1),
        }
        with np.errstate(invalid="ignore"):
            if np.any(any_row):
                valid = arr[any_row]
                stats["mean"][any_row] = np.nanmean(valid, axis=1)
                stats["min"][any_row] = np.nanmin(valid, axis=1)
                stats["max"][any_row] = np.nanmax(valid, axis=1)
                stats["std"][any_row] = np.nanstd(valid, axis=1)
        return stats

    def to_well_curves(
        self, well: Optional["Well"] = None, prefix: str = "IMG"
    ) -> "Well":
        """
        Add scalar image-summary curves to a Well.

        This is intentionally lossy and meant for QC/workflow integration. The
        original image stays in ``BoreholeImageData.image``.
        """
        from petrophys.model.data_types import DataType
        from petrophys.model.well import Well

        if well is None:
            well = Well(self.metadata.get("well", ""))
            well.depth = self.depth
            well.depth_unit = self.depth_unit

        stats = self.per_depth_statistics()
        descriptions = {
            "mean": "Borehole image mean value",
            "min": "Borehole image minimum value",
            "max": "Borehole image maximum value",
            "std": "Borehole image standard deviation",
            "coverage": "Borehole image finite-sample coverage",
        }
        for name, values in stats.items():
            unit = "" if name == "coverage" else self.value_unit
            well.add_curve(
                f"{prefix}_{name.upper()}",
                values,
                unit=unit,
                description=descriptions[name],
                data_type=DataType.COMPUTED,
            )
        return well

    def plot(
        self,
        depth_range: Optional[tuple[float, float]] = None,
        ax: Any = None,
        cmap: str = "gray",
        normalize: bool = True,
    ) -> Any:
        """Plot the borehole image with depth increasing downward."""
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for plotting")

        if ax is None:
            _, ax = plt.subplots(figsize=(5, 10))

        data = self.filter_by_depth(*depth_range) if depth_range else self
        image = data.normalized_image() if normalize else data.image
        top = data.top
        bottom = data.bottom
        if top is None or bottom is None:
            raise ValueError("Cannot plot image with empty/non-finite depth")

        ax.imshow(
            image,
            aspect="auto",
            cmap=cmap,
            extent=(0, data.n_borehole_samples, bottom, top),
            interpolation="nearest",
        )
        ax.set_xlabel("Borehole image sample")
        ax.set_ylabel(f"Depth [{data.depth_unit}]")
        return ax


def _normalize_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    low, high = _finite_min_max(arr)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.full(arr.shape, np.nan, dtype=np.float32)
    return (arr - low) / (high - low)


def _channel_by_name(channels: Iterable[Any], name: str) -> Optional[Any]:
    upper = str(name).upper()
    for channel in channels:
        if str(channel.name).upper() == upper:
            return channel
    return None


def _select_image_frame(
    frames: Sequence[Any],
    frame_name: Optional[str],
    image_channels: Optional[Sequence[str]],
) -> Any:
    if frame_name:
        for frame in frames:
            if str(frame.name) == str(frame_name):
                return frame
        raise ValueError(f"Frame not found: {frame_name}")

    requested = {name.upper() for name in image_channels or DEFAULT_FMI_CHANNELS}
    for frame in frames:
        names = {str(channel.name).upper() for channel in frame.channels}
        if requested & names:
            return frame

    for frame in frames:
        if _select_image_channels(frame.channels, None):
            return frame
    raise ValueError("No image-like frame found")


def _select_image_channels(
    channels: Sequence[Any],
    image_channels: Optional[Sequence[str]],
) -> list[Any]:
    if image_channels is not None:
        requested = {name.upper() for name in image_channels}
        return [
            channel for channel in channels if str(channel.name).upper() in requested
        ]

    ordered = []
    names = {str(channel.name).upper(): channel for channel in channels}
    for name in DEFAULT_FMI_CHANNELS:
        if name in names:
            ordered.append(names[name])
    if ordered:
        return ordered

    result = []
    for channel in channels:
        dims = getattr(channel, "dimension", None) or []
        try:
            width = int(np.prod(dims))
        except TypeError:
            width = 1
        if width > 1:
            result.append(channel)
    return result


def _common_unit(units: Sequence[str]) -> str:
    unique = {unit for unit in units if unit}
    return next(iter(unique)) if len(unique) == 1 else ""


def _collect_flattened_curve_groups(
    curve_names: Sequence[str],
    channel_prefixes: Optional[Sequence[str]],
) -> list[tuple[str, list[str]]]:
    names = [str(name) for name in curve_names]

    def _matches_prefix(prefix: str) -> list[tuple[int, str]]:
        pattern = re.compile(rf"^{re.escape(prefix)}(?:[_\-.]?)(\d+)$", re.IGNORECASE)
        matches = []
        for name in names:
            match = pattern.match(name)
            if match:
                matches.append((int(match.group(1)), name))
        return sorted(matches, key=lambda item: item[0])

    groups: list[tuple[str, list[str]]] = []
    prefixes = list(channel_prefixes or DEFAULT_FMI_CHANNELS)
    for prefix in prefixes:
        matches = _matches_prefix(str(prefix))
        if matches:
            groups.append((str(prefix).upper(), [name for _, name in matches]))

    if groups:
        return groups

    buckets: dict[str, list[tuple[int, str]]] = {}
    generic_pattern = re.compile(r"^(.+?)(?:[_\-.]?)(\d+)$")
    for name in names:
        match = generic_pattern.match(name)
        if not match:
            continue
        base = match.group(1).upper()
        buckets.setdefault(base, []).append((int(match.group(2)), name))

    for base in sorted(buckets):
        matches = sorted(buckets[base], key=lambda item: item[0])
        if len(matches) >= 2:
            groups.append((base, [name for _, name in matches]))
    return groups


def read_borehole_image_dlis(
    filepath: Union[str, Path],
    **kwargs: Any,
) -> BoreholeImageData:
    """Read an FMI/borehole-image DLIS frame."""
    return BoreholeImageData.from_dlis(filepath, **kwargs)


def read_borehole_image_las(
    filepath: Union[str, Path],
    **kwargs: Any,
) -> BoreholeImageData:
    """Read a flattened FMI/borehole-image LAS export."""
    return BoreholeImageData.from_las(filepath, **kwargs)
