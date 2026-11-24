"""
FMI orientation, equalization, and display helpers.

This module builds on :class:`petrophys.interpretation.borehole_imaging.BoreholeImageData` for
Schlumberger FMI-style electrical image logs. It keeps the generic image
container separate from FMI-specific tool geometry and navigation handling.

References
----------
Ekstrom, M.P., Dahan, C.A., Chen, M.-Y., Lloyd, P.M. and Rossi, D.J. (1987).
Formation imaging with microelectrical scanning arrays. *The Log Analyst*
28(3), 294-306.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union
import warnings

import numpy as np

from petrophys.interpretation.borehole_imaging.borehole_image import BoreholeImageData, DEFAULT_FMI_CHANNELS

try:
    from dlisio import dlis

    HAS_DLISIO = True
except ImportError:
    HAS_DLISIO = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


N_ARRAYS = 8
ARRAY_SPACING_DEG = 360.0 / N_ARRAYS
ARRAY_ORDER = tuple(DEFAULT_FMI_CHANNELS)
ARRAY_OFFSET_DEG = {
    name.upper(): index * ARRAY_SPACING_DEG for index, name in enumerate(ARRAY_ORDER)
}
DEFAULT_PAD_ARC_FRACTION = 0.70
NULL_SENTINELS = (-999.25, -9999.0, -999.0)
DEFAULT_NAV_CHANNELS = ("P1AZ", "HAZI", "HAZIM", "DEVI", "DEVIM", "RB")


def read_fmi_dlis(
    filepath: Union[str, Path],
    image_channels: Optional[Sequence[str]] = None,
    nav_channels: Sequence[str] = DEFAULT_NAV_CHANNELS,
    logical_file: int = 0,
    null_value: float = -999.25,
) -> BoreholeImageData:
    """
    Read processed FMI image channels and navigation from a DLIS file.

    Navigation curves can live in a coarser frame than the high-resolution image
    frame. This reader loads the image frame, finds navigation curves in any
    frame in the same logical file, and interpolates them onto image depth.
    """
    if not HAS_DLISIO:
        raise ImportError('dlisio is required; install the DLIS extra: pip install "petrophys[dlis]"')

    physical_file = dlis.load(str(filepath))
    try:
        logical_files = list(physical_file)
        if logical_file < 0 or logical_file >= len(logical_files):
            raise IndexError(f"Logical file index out of range: {logical_file}")
        logical = logical_files[logical_file]

        frame = _select_fmi_image_frame(logical.frames, image_channels)
        frame_data = frame.curves()
        depth_name = _frame_index_name(frame, frame_data.dtype.names)
        depth = np.asarray(frame_data[depth_name], dtype=float).reshape(-1)
        depth_channel = _channel_by_name(frame.channels, depth_name)
        depth_unit = getattr(depth_channel, "units", "") or "m"

        order = list(image_channels or DEFAULT_FMI_CHANNELS)
        channels_by_name = {
            str(channel.name).upper(): channel.name for channel in frame.channels
        }
        arrays = []
        names = []
        widths = []
        units = []
        for mnemonic in order:
            key = str(mnemonic).upper()
            if key not in channels_by_name:
                continue
            channel_name = channels_by_name[key]
            arr = _clean_image_channel(frame_data[channel_name], null_value)
            if arr.shape[0] != depth.size:
                raise ValueError(
                    f"{mnemonic}: length {arr.shape[0]} does not match "
                    f"depth length {depth.size}"
                )
            channel = _channel_by_name(frame.channels, channel_name)
            arrays.append(arr)
            names.append(str(mnemonic).upper())
            widths.append(int(arr.shape[1]))
            units.append(getattr(channel, "units", "") if channel is not None else "")

        if not arrays:
            raise ValueError("No FMI image channels found in selected DLIS frame")

        image = np.concatenate(arrays, axis=1)
        navigation = _read_navigation_curves(logical, depth, nav_channels)

        origins = list(getattr(logical, "origins", []) or [])
        origin = origins[0] if origins else None
        metadata = {
            "source": str(filepath),
            "source_format": "DLIS",
            "logical_file": logical_file,
            "frame": frame.name,
            "spacing": getattr(frame, "spacing", None),
            "index": depth_name,
            "well": getattr(origin, "well_name", "") if origin else "",
            "field": getattr(origin, "field_name", "") if origin else "",
            "producer": getattr(origin, "producer_name", "") if origin else "",
            "image_channels": list(names),
            "navigation_keys": sorted(navigation),
        }
        data = BoreholeImageData.from_arrays(
            depth=depth,
            image=image,
            channel_names=names,
            samples_per_channel=widths,
            image_type="fmi_raw",
            depth_unit=depth_unit,
            value_unit=_common_unit(units),
            metadata=metadata,
        )
        return with_navigation(data, navigation)
    finally:
        if hasattr(physical_file, "close"):
            physical_file.close()


def with_navigation(
    data: BoreholeImageData,
    navigation: dict[str, Sequence[float]],
) -> BoreholeImageData:
    """Return an image object with per-depth navigation stored in metadata.

    Each navigation curve must have one value per image depth sample; resample
    coarser navigation onto the image depth before calling this function.
    """
    metadata = dict(data.metadata)
    metadata["_navigation"] = {
        str(key).upper(): np.asarray(value, dtype=np.float32).reshape(-1).copy()
        for key, value in navigation.items()
    }
    n_depth = np.asarray(data.depth).shape[0]
    mismatched = {
        key: arr.size for key, arr in metadata["_navigation"].items() if arr.size != n_depth
    }
    if mismatched:
        raise ValueError(
            f"Navigation curves {mismatched} must have one value per image depth "
            f"sample ({n_depth}); resample them onto the image depth first."
        )
    metadata["navigation_keys"] = sorted(metadata["_navigation"])
    return BoreholeImageData(
        depth=data.depth,
        image=data.image,
        channel_names=list(data.channel_names),
        samples_per_channel=list(data.samples_per_channel),
        image_type=data.image_type,
        depth_unit=data.depth_unit,
        value_unit=data.value_unit,
        metadata=metadata,
    )


def get_navigation(data: BoreholeImageData) -> dict[str, np.ndarray]:
    """Return copied navigation curves stored on an image object."""
    navigation = data.metadata.get("_navigation", {})
    if not isinstance(navigation, dict):
        return {}
    return {
        str(key).upper(): np.asarray(value, dtype=np.float32).copy()
        for key, value in navigation.items()
    }


def filter_by_depth_with_navigation(
    data: BoreholeImageData,
    top: float,
    bottom: float,
) -> BoreholeImageData:
    """Filter an image by depth while preserving aligned navigation curves."""
    return data.filter_by_depth(top, bottom)


def orient_to_azimuth(
    data: BoreholeImageData,
    n_azimuth: int = 360,
    reference: str = "north",
    pad_arc_fraction: float = DEFAULT_PAD_ARC_FRACTION,
    interpolate_within_pads: bool = True,
) -> BoreholeImageData:
    """
    Map FMI pad/flap arrays onto a regular 0-360 degree azimuth grid.

    ``reference="north"`` uses ``P1AZ``. ``reference="highside"`` uses ``RB``.
    Unimaged wedges between pads are left as NaN. Each pad/flap is placed
    across ``pad_arc_fraction`` of its 45 degree sector (default 0.70), so
    the imaged share of the wall follows that setting, not the caliper.
    When ``interpolate_within_pads`` is true, each pad/flap's button values are
    linearly resampled across that pad arc to avoid artificial white gaps caused
    by placing 24 buttons directly onto a finer azimuth grid.
    """
    if n_azimuth <= 0:
        raise ValueError("n_azimuth must be positive")
    if not (0.0 < pad_arc_fraction <= 1.0):
        raise ValueError("pad_arc_fraction must be in the interval (0, 1]")

    reference_key = _reference_navigation_key(reference)
    navigation = _align_navigation_to_depth(get_navigation(data), data.depth)
    if reference_key not in navigation:
        raise ValueError(
            f"orient_to_azimuth(reference={reference!r}) requires {reference_key}"
        )

    pad1_azimuth = np.asarray(navigation[reference_key], dtype=float)
    n_depth = data.image.shape[0]
    image = np.asarray(data.image, dtype=np.float32)
    widths = list(data.samples_per_channel)
    names = [str(name).upper() for name in data.channel_names]
    starts = np.cumsum([0] + widths)[:-1]
    bin_width = 360.0 / float(n_azimuth)
    half_arc = 0.5 * ARRAY_SPACING_DEG * pad_arc_fraction

    sums = np.zeros((n_depth, n_azimuth), dtype=np.float32)
    counts = np.zeros((n_depth, n_azimuth), dtype=np.uint16)
    used_arrays = []

    for name, start, width in zip(names, starts, widths):
        offset = ARRAY_OFFSET_DEG.get(name)
        if offset is None:
            continue
        block = image[:, start : start + width]
        local_azimuth = (
            np.array([0.0], dtype=float)
            if width == 1
            else np.linspace(-half_arc, half_arc, width)
        )
        if interpolate_within_pads and width >= 2:
            values_added = _accumulate_interpolated_pad(
                sums=sums,
                counts=counts,
                block=block,
                pad1_azimuth=pad1_azimuth,
                offset=offset,
                local_azimuth=local_azimuth,
                bin_width=bin_width,
            )
        else:
            values_added = _accumulate_sparse_buttons(
                sums=sums,
                counts=counts,
                block=block,
                pad1_azimuth=pad1_azimuth,
                offset=offset,
                local_azimuth=local_azimuth,
                bin_width=bin_width,
            )
        if values_added:
            used_arrays.append(name)

    if not used_arrays:
        raise ValueError(
            "No known FMI pad/flap channel names were found; expected "
            f"one or more of {', '.join(ARRAY_ORDER)}"
        )

    oriented = np.full((n_depth, n_azimuth), np.nan, dtype=np.float32)
    populated = counts > 0
    oriented[populated] = sums[populated] / counts[populated]

    metadata = dict(data.metadata)
    metadata.update(
        {
            "oriented": True,
            "orientation_reference": reference,
            "orientation_key": reference_key,
            "n_azimuth": n_azimuth,
            "pad_arc_fraction": float(pad_arc_fraction),
            "interpolate_within_pads": bool(interpolate_within_pads),
            "used_arrays": used_arrays,
        }
    )
    metadata["_navigation"] = navigation
    return BoreholeImageData(
        depth=data.depth.copy(),
        image=oriented,
        channel_names=["AZIMUTH"],
        samples_per_channel=[n_azimuth],
        image_type="fmi_oriented",
        depth_unit=data.depth_unit,
        value_unit=data.value_unit,
        metadata=metadata,
    )



def equalize_static(image: np.ndarray, clip: tuple[float, float] = (2.0, 98.0)) -> np.ndarray:
    """Apply one percentile clip/rescale function to the whole image."""
    arr = np.asarray(image, dtype=np.float32)
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return out
    lo, hi = np.percentile(arr[finite], clip)
    if hi <= lo:
        out[finite] = 0.5
        return out
    out[finite] = np.clip((arr[finite] - lo) / (hi - lo), 0.0, 1.0)
    return out


def equalize_dynamic(
    image: np.ndarray,
    depth: np.ndarray,
    window_m: float = 0.5,
    clip: tuple[float, float] = (2.0, 98.0),
    depth_unit: str = "m",
) -> np.ndarray:
    """Apply sliding-window percentile equalization along the depth axis.

    ``window_m`` is in metres and is converted to ``depth_unit`` (``"m"``,
    ``"ft"``, ``"in"`` or ``"0.1 in"``) before sizing the window.
    """
    arr = np.asarray(image, dtype=np.float32)
    dep = np.asarray(depth, dtype=float)
    if arr.shape[0] != dep.size:
        raise ValueError("image depth dimension must match depth length")
    if arr.shape[0] < 2:
        return equalize_static(arr, clip=clip)

    step = np.nanmedian(np.abs(np.diff(dep)))
    if not np.isfinite(step) or step <= 0:
        return equalize_static(arr, clip=clip)
    window = float(window_m) / _metres_per_depth_unit(depth_unit)
    half_window_rows = max(1, int(round((window / step) / 2.0)))
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    for row in range(arr.shape[0]):
        start = max(0, row - half_window_rows)
        stop = min(arr.shape[0], row + half_window_rows + 1)
        window = arr[start:stop]
        finite_window = window[np.isfinite(window)]
        finite_row = np.isfinite(arr[row])
        if finite_window.size == 0 or not np.any(finite_row):
            continue
        lo, hi = np.percentile(finite_window, clip)
        if hi <= lo:
            out[row, finite_row] = 0.5
        else:
            out[row, finite_row] = np.clip(
                (arr[row, finite_row] - lo) / (hi - lo),
                0.0,
                1.0,
            )
    return out


def azimuthal_coverage(oriented: BoreholeImageData) -> np.ndarray:
    """Return the fraction of azimuth bins imaged at each depth."""
    return np.isfinite(np.asarray(oriented.image, dtype=float)).mean(axis=1)


def safe_per_depth_statistics(data: BoreholeImageData) -> dict[str, np.ndarray]:
    """Return per-depth image statistics without warnings for all-NaN rows."""
    arr = np.asarray(data.image, dtype=float)
    finite = np.isfinite(arr)
    any_row = finite.any(axis=1)
    stats = {
        "mean": np.full(arr.shape[0], np.nan, dtype=float),
        "min": np.full(arr.shape[0], np.nan, dtype=float),
        "max": np.full(arr.shape[0], np.nan, dtype=float),
        "std": np.full(arr.shape[0], np.nan, dtype=float),
        "coverage": finite.mean(axis=1),
    }
    if np.any(any_row):
        valid = arr[any_row]
        stats["mean"][any_row] = np.nanmean(valid, axis=1)
        stats["min"][any_row] = np.nanmin(valid, axis=1)
        stats["max"][any_row] = np.nanmax(valid, axis=1)
        stats["std"][any_row] = np.nanstd(valid, axis=1)
    return stats


def fmi_colormap() -> Any:
    """Return a conventional conductive-dark to resistive-light FMI colormap."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required")
    colors = [
        (0.00, "#1a0d00"),
        (0.25, "#5c2e00"),
        (0.50, "#9e5c1f"),
        (0.70, "#d49a4e"),
        (0.88, "#f2d9a0"),
        (1.00, "#fff7e6"),
    ]
    return LinearSegmentedColormap.from_list("fmi_brown", colors)


def plot_fmi(
    data: BoreholeImageData,
    top: Optional[float] = None,
    bottom: Optional[float] = None,
    ax: Any = None,
    equalize: str = "dynamic",
    window_m: float = 0.5,
    interpolation: str = "bilinear",
) -> Any:
    """Plot FMI image data with brown image-log colours (dark conductive, light
    resistive) and an azimuth axis; ``top`` and ``bottom`` set the depth window.
    Draws on ``ax`` when given; returns a :class:`~petrophys.visualization.Plot`."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")

    if ax is None:
        from petrophys.visualization.style import plot_style

        with plot_style():
            _, ax = plt.subplots(figsize=(5, 11))

    from petrophys.visualization._data import depth_limits

    window = depth_limits(data.depth, top, bottom)
    display_data = data.filter_by_depth(*window) if window is not None else data
    top = display_data.top
    bottom = display_data.bottom
    if top is None or bottom is None:
        raise ValueError("Cannot plot image with empty/non-finite depth")

    mode = str(equalize or "none").strip().lower()
    if mode == "dynamic":
        image = equalize_dynamic(
            display_data.image,
            display_data.depth,
            window_m=window_m,
            depth_unit=display_data.depth_unit,
        )
        vmin, vmax = 0.0, 1.0
    elif mode == "static":
        image = equalize_static(display_data.image)
        vmin, vmax = 0.0, 1.0
    elif mode in {"none", "raw", "false"}:
        image = np.asarray(display_data.image, dtype=float)
        vmin = vmax = None
    else:
        raise ValueError("equalize must be 'dynamic', 'static', or 'none'")

    oriented = bool(display_data.metadata.get("oriented"))
    xmax = 360 if oriented else display_data.n_borehole_samples
    ax.imshow(
        image,
        aspect="auto",
        cmap=fmi_colormap(),
        extent=(0, xmax, bottom, top),
        interpolation=interpolation,
        vmin=vmin,
        vmax=vmax,
    )
    highside = (
        str(display_data.metadata.get("orientation_reference", "north")).strip().lower()
        == "highside"
    )
    if not oriented:
        ax.set_xlabel("Borehole image sample")
    elif highside:
        ax.set_xlabel("Angle from high side [deg]")
    else:
        ax.set_xlabel("Azimuth [deg]")
    ax.set_ylabel(f"Depth [{display_data.depth_unit}]")
    if oriented:
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xticklabels(
            ["HS", "90", "LS", "270", "HS"] if highside else ["N", "E", "S", "W", "N"]
        )
    from petrophys.visualization.base import Plot

    return Plot(ax.figure, ax)


def _accumulate_sparse_buttons(
    sums: np.ndarray,
    counts: np.ndarray,
    block: np.ndarray,
    pad1_azimuth: np.ndarray,
    offset: float,
    local_azimuth: np.ndarray,
    bin_width: float,
) -> bool:
    n_depth = block.shape[0]
    n_azimuth = sums.shape[1]
    button_azimuth = np.mod(
        pad1_azimuth[:, None] + offset + local_azimuth[None, :],
        360.0,
    )
    finite = np.isfinite(block) & np.isfinite(button_azimuth)
    if not np.any(finite):
        return False
    safe_azimuth = np.where(np.isfinite(button_azimuth), button_azimuth, 0.0)
    bins = np.floor(safe_azimuth / bin_width).astype(int)
    bins = np.clip(bins, 0, n_azimuth - 1)
    rows = np.repeat(np.arange(n_depth), block.shape[1]).reshape(block.shape)
    np.add.at(sums, (rows[finite], bins[finite]), block[finite])
    np.add.at(counts, (rows[finite], bins[finite]), 1)
    return True


def _accumulate_interpolated_pad(
    sums: np.ndarray,
    counts: np.ndarray,
    block: np.ndarray,
    pad1_azimuth: np.ndarray,
    offset: float,
    local_azimuth: np.ndarray,
    bin_width: float,
    chunk_rows: int = 4096,
) -> bool:
    n_depth = block.shape[0]
    n_azimuth = sums.shape[1]
    azimuth_centers = (np.arange(n_azimuth, dtype=float) + 0.5) * bin_width
    values_added = False

    for chunk_start in range(0, n_depth, chunk_rows):
        chunk_stop = min(n_depth, chunk_start + chunk_rows)
        chunk_block = block[chunk_start:chunk_stop]
        pad_az = pad1_azimuth[chunk_start:chunk_stop]
        centre = pad_az[:, None] + offset
        local_grid = ((azimuth_centers[None, :] - centre + 180.0) % 360.0) - 180.0
        finite_centre = np.isfinite(pad_az)

        for index in range(local_azimuth.size - 1):
            x0 = float(local_azimuth[index])
            x1 = float(local_azimuth[index + 1])
            if x1 <= x0:
                continue

            lower = local_grid >= x0
            upper = local_grid <= x1 if index == local_azimuth.size - 2 else local_grid < x1
            inside = lower & upper
            y0 = chunk_block[:, index]
            y1 = chunk_block[:, index + 1]
            finite_rows = finite_centre & np.isfinite(y0) & np.isfinite(y1)
            if not np.any(finite_rows):
                continue

            valid = inside & finite_rows[:, None]
            if not np.any(valid):
                continue

            rows_local, cols = np.nonzero(valid)
            fraction = (local_grid[rows_local, cols] - x0) / (x1 - x0)
            values = y0[rows_local] + fraction * (y1[rows_local] - y0[rows_local])
            rows = rows_local + chunk_start
            np.add.at(sums, (rows, cols), values.astype(np.float32))
            np.add.at(counts, (rows, cols), 1)
            values_added = True

    return values_added


def _select_fmi_image_frame(
    frames: Sequence[Any],
    image_channels: Optional[Sequence[str]],
) -> Any:
    wanted = {str(name).upper() for name in (image_channels or DEFAULT_FMI_CHANNELS)}
    candidates = []
    for frame in frames:
        names = {str(channel.name).upper() for channel in frame.channels}
        if wanted.issubset(names):
            n_samples = 0
            try:
                n_samples = len(frame.curves())
            except Exception:
                pass
            candidates.append((n_samples, frame))
    if not candidates:
        raise ValueError(f"No frame contains all expected FMI channels {sorted(wanted)}")
    if len(candidates) > 1:
        candidates.sort(key=lambda item: item[0], reverse=True)
        warnings.warn(
            "Multiple frames contain the FMI image channels; using the one with "
            "the most samples.",
            stacklevel=2,
        )
    return candidates[0][1]


def _read_navigation_curves(
    logical: Any,
    image_depth: np.ndarray,
    nav_channels: Sequence[str],
) -> dict[str, np.ndarray]:
    navigation: dict[str, np.ndarray] = {}
    canonical = {"HAZIM": "HAZI", "DEVIM": "DEVI"}
    for mnemonic in nav_channels:
        nav_depth, values = _curve_from_any_frame(logical, mnemonic)
        if nav_depth is None or values is None:
            continue
        key = canonical.get(str(mnemonic).upper(), str(mnemonic).upper())
        if key in navigation:
            continue
        if key in {"P1AZ", "HAZI", "RB"}:
            navigation[key] = _interp_angle(image_depth, nav_depth, values).astype(
                np.float32
            )
        else:
            navigation[key] = _interp_linear(image_depth, nav_depth, values).astype(
                np.float32
            )
    return navigation


def _curve_from_any_frame(
    logical: Any,
    mnemonic: str,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    target = str(mnemonic).upper()
    for frame in logical.frames:
        channels_by_name = {
            str(channel.name).upper(): channel.name for channel in frame.channels
        }
        if target not in channels_by_name:
            continue
        frame_data = frame.curves()
        depth_name = _frame_index_name(frame, frame_data.dtype.names)
        values = np.asarray(frame_data[channels_by_name[target]], dtype=float).reshape(-1)
        depth = np.asarray(frame_data[depth_name], dtype=float).reshape(-1)
        values = _clean_scalar(values)
        if depth.size == values.size:
            return depth, values
    return None, None


def _frame_index_name(frame: Any, dtype_names: Sequence[str]) -> str:
    names = list(dtype_names or [])
    names_upper = {str(name).upper(): str(name) for name in names}
    index = getattr(frame, "index", None)
    candidates = [
        getattr(index, "name", None),
        index,
        "TDEP",
        "DEPTH",
        "DEPT",
        "MD",
        names[0] if names else None,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate)
        if text in names:
            return text
        if text.upper() in names_upper:
            return names_upper[text.upper()]
    raise ValueError("Could not resolve frame index/depth channel")


def _clean_image_channel(values: np.ndarray, null_value: float) -> np.ndarray:
    arr = np.squeeze(np.asarray(values))
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Image channel must be 2D after squeeze; got {values.shape}")
    arr = arr.astype(np.float32, copy=True)
    for sentinel in (null_value, *NULL_SENTINELS):
        arr[arr == sentinel] = np.nan
    return arr


def _clean_scalar(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    for sentinel in NULL_SENTINELS:
        arr[arr == sentinel] = np.nan
    return arr


def _interp_linear(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    support_x, support_y = _unique_depth_mean(xp, fp)
    out = np.full(np.asarray(x, dtype=float).shape, np.nan, dtype=float)
    if support_x.size == 0:
        return out
    return np.interp(x, support_x, support_y, left=np.nan, right=np.nan)


def _interp_angle(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(fp)
    support_x_sin, support_sin = _unique_depth_mean(xp, np.sin(radians))
    support_x_cos, support_cos = _unique_depth_mean(xp, np.cos(radians))
    out = np.full(np.asarray(x, dtype=float).shape, np.nan, dtype=float)
    if support_x_sin.size == 0 or support_x_cos.size == 0:
        return out
    sin_i = np.interp(x, support_x_sin, support_sin, left=np.nan, right=np.nan)
    cos_i = np.interp(x, support_x_cos, support_cos, left=np.nan, right=np.nan)
    valid = np.isfinite(sin_i) & np.isfinite(cos_i)
    out[valid] = np.mod(np.rad2deg(np.arctan2(sin_i[valid], cos_i[valid])), 360.0)
    return out


def _unique_depth_mean(
    depth: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(depth, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return np.array([], dtype=float), np.array([], dtype=float)
    x = x[valid]
    y = y[valid]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique, inverse = np.unique(x, return_inverse=True)
    sums = np.zeros(unique.size, dtype=float)
    counts = np.zeros(unique.size, dtype=float)
    np.add.at(sums, inverse, y)
    np.add.at(counts, inverse, 1.0)
    return unique, sums / counts


def _align_navigation_to_depth(
    navigation: dict[str, np.ndarray],
    depth: np.ndarray,
) -> dict[str, np.ndarray]:
    n_samples = np.asarray(depth).shape[0]
    aligned: dict[str, np.ndarray] = {}
    for key, values in navigation.items():
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.size == n_samples:
            aligned[key] = arr.copy()
        elif arr.size > n_samples:
            aligned[key] = arr[:n_samples].copy()
        else:
            filled = np.full(n_samples, np.nan, dtype=np.float32)
            filled[: arr.size] = arr
            aligned[key] = filled
    return aligned


def _metres_per_depth_unit(depth_unit: str) -> float:
    unit = str(depth_unit or "m").strip().lower().replace(".", "")
    if unit in {"ft", "f", "feet", "foot"}:
        return 0.3048
    if unit in {"in", "inch", "inches"}:
        return 0.0254
    if unit in {"01 in", "01in"}:
        return 0.00254
    return 1.0


def _reference_navigation_key(reference: str) -> str:
    key = str(reference).strip().lower()
    if key == "north":
        return "P1AZ"
    if key == "highside":
        return "RB"
    raise ValueError("reference must be 'north' or 'highside'")


def _channel_by_name(channels: Sequence[Any], name: str) -> Optional[Any]:
    upper = str(name).upper()
    for channel in channels:
        if str(channel.name).upper() == upper:
            return channel
    return None


def _common_unit(units: Sequence[str]) -> str:
    unique = {str(unit) for unit in units if unit}
    return next(iter(unique)) if len(unique) == 1 else ""


__all__ = [
    "ARRAY_OFFSET_DEG",
    "ARRAY_ORDER",
    "ARRAY_SPACING_DEG",
    "DEFAULT_NAV_CHANNELS",
    "DEFAULT_PAD_ARC_FRACTION",
    "N_ARRAYS",
    "NULL_SENTINELS",
    "azimuthal_coverage",
    "equalize_dynamic",
    "equalize_static",
    "filter_by_depth_with_navigation",
    "fmi_colormap",
    "get_navigation",
    "orient_to_azimuth",
    "plot_fmi",
    "read_fmi_dlis",
    "safe_per_depth_statistics",
    "with_navigation",
]
