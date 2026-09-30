"""
DLIS file reader for well log data.

This module provides functionality to read DLIS (Digital Log Interchange Standard)
files and convert them to Well objects.
"""

from typing import TYPE_CHECKING, Optional, Union, Dict, Any, List, Iterable, Mapping, Tuple
from pathlib import Path
import warnings
import numpy as np

if TYPE_CHECKING:
    from petrophys.model.well import Well

try:
    from dlisio import dlis

    HAS_DLISIO = True
except ImportError:
    HAS_DLISIO = False


DEPTH_CHANNELS = {"TDEP", "DEPTH", "DEPT", "MD"}
FRAME_NUMBER_FIELDS = {"FRAMENO"}
COMMON_NULL_VALUES = (-999.25, -999.0, -9999.0, -99999.0)


def _unit_text(unit: str) -> str:
    return (unit or "").strip().lower().replace(" ", "")


def _depth_to_m_factor(unit: str) -> Optional[float]:
    unit_norm = _unit_text(unit)
    if unit_norm in {"m", "meter", "meters", "metre", "metres"}:
        return 1.0
    if unit_norm in {"ft", "feet", "foot"}:
        return 0.3048
    if unit_norm in {"in", "inch", "inches"}:
        return 0.0254
    if unit_norm in {"0.1in", ".1in", "1/10in", "0.1inch", "0.1inches"}:
        return 0.00254
    return None


def _convert_depth(
    depth: np.ndarray,
    source_unit: str,
    target_unit: Optional[str] = None,
) -> Tuple[np.ndarray, str]:
    """Return depth in a plotting/workflow-friendly unit."""
    arr = np.asarray(depth, dtype=float).copy()
    source = source_unit or ""
    source_norm = _unit_text(source)

    if target_unit is None:
        # DLIS tool depth is commonly stored as integer tenths of an inch.
        # Treat it as an encoding and expose measured depth in metres.
        target_unit = "m" if source_norm in {"0.1in", ".1in", "1/10in"} else source

    if not target_unit:
        return arr, source

    source_to_m = _depth_to_m_factor(source)
    target_to_m = _depth_to_m_factor(target_unit)
    if source_to_m is None or target_to_m is None:
        return arr, source

    return arr * source_to_m / target_to_m, target_unit


def _channel_field_names(frame, frame_data) -> List[Optional[str]]:
    """Map DLIS channels to structured-array field names."""
    dtype_names = list(frame_data.dtype.names or [])
    fields = [name for name in dtype_names if name.upper() not in FRAME_NUMBER_FIELDS]
    channels = list(frame.channels)

    if len(fields) >= len(channels):
        return fields[:len(channels)]

    used: set[str] = set()
    resolved: List[Optional[str]] = []
    for channel in channels:
        name = str(channel.name)
        exact = next((field for field in fields if field == name and field not in used), None)
        if exact is None:
            prefix = f"{name}."
            exact = next(
                (field for field in fields if field.startswith(prefix) and field not in used),
                None,
            )
        if exact is not None:
            used.add(exact)
        resolved.append(exact)
    return resolved


def _clean_numeric(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=float).copy()
    for null_value in COMMON_NULL_VALUES:
        arr[np.isclose(arr, null_value, equal_nan=False)] = np.nan
    return arr


def _as_scalar_curve(data: np.ndarray, n_samples: int) -> Optional[np.ndarray]:
    arr = np.asarray(data)
    squeezed = np.squeeze(arr)
    if squeezed.ndim == 1 and len(squeezed) == n_samples:
        return _clean_numeric(squeezed)
    if squeezed.ndim == 2 and squeezed.shape[0] == n_samples and squeezed.shape[1] == 1:
        return _clean_numeric(squeezed[:, 0])
    return None


def _finite_depth_range(depth: np.ndarray) -> Optional[Tuple[float, float]]:
    finite = np.asarray(depth, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.nanmin(finite)), float(np.nanmax(finite))


def _median_step(depth: np.ndarray) -> Optional[float]:
    finite = np.asarray(depth, dtype=float)
    finite = np.unique(finite[np.isfinite(finite)])
    if finite.size < 2:
        return None
    steps = np.diff(np.sort(finite))
    steps = steps[np.isfinite(steps) & (steps > 0)]
    if steps.size == 0:
        return None
    return float(np.nanmedian(steps))


def _prepare_for_resample(
    depth: np.ndarray,
    values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    depth_arr = np.asarray(depth, dtype=float)
    value_arr = np.asarray(values, dtype=float)
    valid = np.isfinite(depth_arr) & np.isfinite(value_arr)
    if np.count_nonzero(valid) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    depth_arr = depth_arr[valid]
    value_arr = value_arr[valid]
    order = np.argsort(depth_arr)
    depth_arr = depth_arr[order]
    value_arr = value_arr[order]

    unique_depth, inverse = np.unique(depth_arr, return_inverse=True)
    if unique_depth.size == depth_arr.size:
        return depth_arr, value_arr

    sums = np.zeros(unique_depth.size, dtype=float)
    counts = np.zeros(unique_depth.size, dtype=float)
    np.add.at(sums, inverse, value_arr)
    np.add.at(counts, inverse, 1.0)
    return unique_depth, sums / counts


def _resample_to_depth(
    source_depth: np.ndarray,
    values: np.ndarray,
    target_depth: np.ndarray,
) -> np.ndarray:
    depth, data = _prepare_for_resample(source_depth, values)
    if depth.size < 2:
        return np.full_like(target_depth, np.nan, dtype=float)
    return np.interp(target_depth, depth, data, left=np.nan, right=np.nan)


def _name_set(names: Optional[Iterable[str]]) -> Optional[set[str]]:
    if names is None:
        return None
    if isinstance(names, str):
        return {names.upper()}
    return {str(name).upper() for name in names}


def _channel_requested(channel_name: str, requested: Optional[set[str]]) -> bool:
    if requested is None:
        return True
    name = str(channel_name).upper()
    if name in requested:
        return True
    try:
        from petrophys.preprocessing.standardization.alias_map import get_standard_mnemonic
    except Exception:
        return False
    return get_standard_mnemonic(name).upper() in requested


def _channel_quality_score(
    channel: Any,
    preferred_mnemonic: Optional[str] = None,
) -> Tuple[int, List[str]]:
    """Rank processed scalar channels ahead of raw/QC alternatives."""
    name = str(getattr(channel, "name", "") or "").strip().upper()
    long_name = str(getattr(channel, "long_name", "") or "")
    properties = getattr(channel, "properties", None)
    if isinstance(properties, (list, tuple, set)):
        property_text = " ".join(str(value) for value in properties)
    else:
        property_text = str(properties or "")
    text = f"{name} {long_name} {property_text}".upper()

    score = 0
    reasons: List[str] = []
    if preferred_mnemonic and name == str(preferred_mnemonic).upper():
        score += 10_000
        reasons.append("explicit user preference")

    negative_markers = {
        "UNCALIBRATED": -200,
        "UN-CALIBRATED": -200,
        "RAW": -40,
        "QUALITY CONTROL": -80,
        "QC FLAG": -80,
        "QUALITY INDICATOR": -80,
    }
    positive_markers = {
        "CALIBRATED": 80,
        "BOREHOLE CORRECTED": 60,
        "BOREHOLE-CORRECTED": 60,
        "ENVIRONMENTALLY CORRECTED": 60,
        "ENVIRONMENTAL CORRECTION": 50,
        "FINAL": 30,
        "PROCESSED": 20,
    }
    has_uncalibrated = "UNCALIBRATED" in text or "UN-CALIBRATED" in text
    for marker, weight in negative_markers.items():
        if marker in text:
            score += weight
            reasons.append(marker.lower())
    for marker, weight in positive_markers.items():
        if marker in text and not (marker == "CALIBRATED" and has_uncalibrated):
            score += weight
            reasons.append(marker.lower())
    return score, reasons


def _normalize_channel_preferences(
    preferences: Optional[Mapping[str, str]],
) -> Dict[str, str]:
    if not preferences:
        return {}
    from petrophys.preprocessing.standardization.alias_map import get_standard_mnemonic

    return {
        get_standard_mnemonic(str(standard).upper()).upper(): str(raw).upper()
        for standard, raw in preferences.items()
    }


class DLISReader:
    """
    Reader for DLIS (Digital Log Interchange Standard) files.

    DLIS files can contain multiple logical files and frames.
    This reader provides access to the data within these structures.

    Examples
    --------
    >>> reader = DLISReader("well_data.dlis")
    >>> well = reader.to_well()
    """

    def __init__(self, filepath: Union[str, Path]):
        """
        Initialize DLIS reader.

        Parameters
        ----------
        filepath : str or Path
            Path to DLIS file
        """
        if not HAS_DLISIO:
            raise ImportError(
                "dlisio is required for DLIS file reading. "
                'Install the DLIS extra: pip install "petrophys[dlis]"'
            )

        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"DLIS file not found: {filepath}")

        # Open DLIS file
        self._physical_file = dlis.load(str(self.filepath))
        self._logical_files = list(self._physical_file)
        self._active_file_idx = 0

    @property
    def n_logical_files(self) -> int:
        """Return number of logical files."""
        return len(self._logical_files)

    @property
    def active_file(self):
        """Return currently active logical file."""
        return self._logical_files[self._active_file_idx]

    def set_active_file(self, index: int):
        """Set the active logical file by index."""
        if 0 <= index < len(self._logical_files):
            self._active_file_idx = index
        else:
            raise IndexError(f"Logical file index out of range: {index}")

    def get_frames(self) -> List[str]:
        """Return list of frame names in active logical file."""
        frames = []
        for frame in self.active_file.frames:
            frames.append(frame.name)
        return frames

    def get_channels(self, frame_name: Optional[str] = None) -> List[str]:
        """Return list of channel names in a frame."""
        frames = self.active_file.frames
        if not frames:
            return []

        if frame_name:
            for frame in frames:
                if frame.name == frame_name:
                    return [ch.name for ch in frame.channels]
            return []

        # Return channels from first frame
        return [ch.name for ch in frames[0].channels]

    def get_well_info(self) -> Dict[str, Any]:
        """Extract well information from DLIS origin."""
        info = {}
        origins = self.active_file.origins
        if origins:
            origin = origins[0]
            info["well_name"] = getattr(origin, "well_name", "")
            info["field_name"] = getattr(origin, "field_name", "")
            info["company"] = getattr(origin, "company", "")
            info["creation_time"] = getattr(origin, "creation_time", None)
        return info

    def to_well(
        self,
        frame_name: Optional[str] = None,
        channels: Optional[List[str]] = None,
        depth_unit: Optional[str] = None,
        include_vector_channels: bool = True,
    ) -> "Well":
        """
        Convert DLIS data to Well object.

        Parameters
        ----------
        frame_name : str, optional
            Name of frame to read (default: first frame)
        channels : list of str, optional
            Specific channels to read (default: all)
        depth_unit : str, optional
            Target depth unit. DLIS ``0.1 in`` tool depth is converted to metres
            by default.

        Returns
        -------
        Well
            Well object with loaded data
        """
        from petrophys.model.well import Well
        from petrophys.model.curve import Curve, CurveMetadata
        from petrophys.model.data_types import DataType

        # Get well info
        well_info = self.get_well_info()
        well = Well(name=well_info.get("well_name", ""))

        # Set header
        well.header.well_name = well_info.get("well_name", "")
        well.header.field = well_info.get("field_name", "")
        well.header.company = well_info.get("company", "")

        # Get frame
        frames = self.active_file.frames
        if not frames:
            warnings.warn("No frames found in DLIS file")
            return well

        frame = None
        if frame_name:
            for f in frames:
                if f.name == frame_name:
                    frame = f
                    break
            if frame is None:
                raise ValueError(f"Frame not found: {frame_name}")
        else:
            frame = frames[0]

        requested = _name_set(channels)

        # Read frame data
        frame_data = frame.curves()
        field_names = _channel_field_names(frame, frame_data)

        # Find depth channel
        depth_data = None
        target_depth_unit = depth_unit
        depth_channel = None

        for ch, field_name in zip(frame.channels, field_names):
            if ch.name.upper() in DEPTH_CHANNELS and field_name is not None:
                depth_channel = ch.name
                depth_data = frame_data[field_name]
                source_depth_unit = ch.units or "ft"
                break

        if depth_data is not None:
            # Handle multi-dimensional depth (take first column if needed)
            if len(depth_data.shape) > 1:
                depth_data = depth_data[:, 0]
            well._depth, well._depth_unit = _convert_depth(
                depth_data.flatten(),
                source_depth_unit,
                target_depth_unit,
            )

        # Add channels as curves
        for ch, field_name in zip(frame.channels, field_names):
            if not _channel_requested(ch.name, requested):
                continue
            if ch.name == depth_channel:
                continue
            if ch.name.upper() in DEPTH_CHANNELS:
                continue
            if field_name is None:
                warnings.warn(f"Skipping DLIS channel {ch.name!r}: no data field found")
                continue

            data = frame_data[field_name]

            # Handle multi-dimensional data. FMI/image channels often arrive
            # as (n_depth, 1, n_buttons); squeeze singleton dimensions, then
            # split true per-depth columns into separate scalar curves.
            if len(data.shape) > 1:
                if not include_vector_channels:
                    continue
                squeezed = np.squeeze(data)
                if squeezed.ndim == 1:
                    squeezed = squeezed.reshape(-1, 1)
                if squeezed.ndim != 2 or squeezed.shape[0] != well.n_samples:
                    warnings.warn(
                        f"Skipping multidimensional DLIS channel {ch.name!r} "
                        f"with unsupported shape {data.shape}; use "
                        "BoreholeImageData.from_dlis() for image frames."
                    )
                    continue

                for i in range(squeezed.shape[1]):
                    curve_name = f"{ch.name}_{i}" if squeezed.shape[1] > 1 else ch.name
                    metadata = CurveMetadata(
                        description=ch.long_name or "",
                        source="DLIS",
                        data_type=DataType.CURVE,
                    )
                    curve = Curve(
                        mnemonic=curve_name,
                        data=_clean_numeric(squeezed[:, i].flatten()),
                        unit=ch.units or "",
                        depth=well._depth,
                        depth_unit=well._depth_unit,
                        metadata=metadata,
                    )
                    well._curves[curve_name.upper()] = curve
            else:
                metadata = CurveMetadata(
                    description=ch.long_name or "",
                    source="DLIS",
                    data_type=DataType.CURVE,
                )
                curve = Curve(
                    mnemonic=ch.name,
                    data=_clean_numeric(data.flatten()),
                    unit=ch.units or "",
                    depth=well._depth,
                    depth_unit=well._depth_unit,
                    metadata=metadata,
                )
                well._curves[ch.name.upper()] = curve

        return well

    def to_merged_well(
        self,
        frame_names: Optional[Iterable[str]] = None,
        channels: Optional[Iterable[str]] = None,
        depth_unit: str = "m",
        step: Optional[float] = None,
        target_depth: Optional[np.ndarray] = None,
        standardize: bool = True,
        step_strategy: str = "finest",
        max_samples: int = 2_000_000,
        channel_preference: Optional[Mapping[str, str]] = None,
    ) -> "Well":
        """
        Merge scalar channels from multiple DLIS frames onto one depth index.

        This path is intended for conventional scalar log interpretation. Vector
        channels such as spectra and borehole images are skipped; image DLIS files
        should use the imaging readers.
        """
        from petrophys.model.well import Well
        from petrophys.model.curve import Curve, CurveMetadata
        from petrophys.model.data_types import DataType
        from petrophys.preprocessing.standardization.alias_map import get_standard_mnemonic

        selected_frames = _name_set(frame_names)
        requested = _name_set(channels)
        preferences = _normalize_channel_preferences(channel_preference)
        well_info = self.get_well_info()

        frame_payloads: List[Dict[str, Any]] = []
        for frame in self.active_file.frames:
            if selected_frames and frame.name.upper() not in selected_frames:
                continue

            try:
                frame_data = frame.curves()
            except Exception as exc:
                warnings.warn(f"Skipping DLIS frame {frame.name!r}: {exc}")
                continue

            field_names = _channel_field_names(frame, frame_data)
            depth = None
            converted_unit = None
            for ch, field_name in zip(frame.channels, field_names):
                if ch.name.upper() in DEPTH_CHANNELS and field_name is not None:
                    raw_depth = np.asarray(frame_data[field_name], dtype=float)
                    if raw_depth.ndim > 1:
                        raw_depth = raw_depth[:, 0]
                    depth, converted_unit = _convert_depth(
                        raw_depth.flatten(),
                        ch.units or "",
                        depth_unit,
                    )
                    break
            if depth is None or _finite_depth_range(depth) is None:
                continue
            if converted_unit != depth_unit:
                warnings.warn(
                    f"Skipping DLIS frame {frame.name!r}: depth unit "
                    f"{converted_unit!r} could not be converted to "
                    f"{depth_unit!r}."
                )
                continue

            curves: List[Dict[str, Any]] = []
            n_samples = len(depth)
            for ch, field_name in zip(frame.channels, field_names):
                name = ch.name.upper()
                if not _channel_requested(name, requested):
                    continue
                if name in DEPTH_CHANNELS or field_name is None:
                    continue

                values = _as_scalar_curve(frame_data[field_name], n_samples)
                if values is None or not np.any(np.isfinite(values)):
                    continue

                standard_name = get_standard_mnemonic(name).upper()
                quality_score, quality_reasons = _channel_quality_score(
                    ch,
                    preferences.get(standard_name),
                )

                curves.append({
                    "name": ch.name,
                    "unit": ch.units or "",
                    "description": ch.long_name or "",
                    "values": values,
                    "frame": frame.name,
                    "quality_score": quality_score,
                    "quality_reasons": quality_reasons,
                })

            if curves:
                frame_payloads.append({
                    "name": frame.name,
                    "depth": depth,
                    "curves": curves,
                    "step": _median_step(depth),
                })

        well = Well(name=well_info.get("well_name", ""))
        well.header.well_name = well_info.get("well_name", "")
        well.header.field = well_info.get("field_name", "")
        well.header.company = well_info.get("company", "")

        if not frame_payloads:
            warnings.warn("No scalar DLIS frames could be merged")
            return well

        if target_depth is None:
            ranges = [_finite_depth_range(payload["depth"]) for payload in frame_payloads]
            ranges = [item for item in ranges if item is not None]
            top = min(item[0] for item in ranges)
            bottom = max(item[1] for item in ranges)
            if step is None:
                steps = [
                    payload["step"]
                    for payload in frame_payloads
                    if payload["step"] is not None and payload["step"] > 0
                ]
                strategy = str(step_strategy).strip().lower()
                if strategy == "finest":
                    step = min(steps) if steps else (bottom - top)
                elif strategy == "coarsest":
                    step = max(steps) if steps else (bottom - top)
                elif strategy == "median":
                    step = float(np.median(steps)) if steps else (bottom - top)
                else:
                    raise ValueError(
                        "step_strategy must be 'finest', 'median', or 'coarsest'"
                    )
            if not np.isfinite(step) or float(step) <= 0:
                raise ValueError("DLIS merge step must be a positive finite value")
            estimated_samples = int(np.floor((bottom - top) / float(step))) + 1
            if estimated_samples > max_samples:
                raise ValueError(
                    f"DLIS merge would create {estimated_samples:,} samples at "
                    f"step={step:g} {depth_unit}; provide an explicit coarser step "
                    f"or raise max_samples (currently {max_samples:,})."
                )
            target_depth = np.arange(top, bottom + float(step) * 0.5, float(step))
        else:
            target_depth = np.asarray(target_depth, dtype=float)

        well._depth = np.asarray(target_depth, dtype=float)
        well._depth_unit = depth_unit

        best_rank: Dict[str, Tuple[int, int]] = {}
        source_frames: Dict[str, str] = {}
        source_quality: Dict[str, Dict[str, Any]] = {}
        for payload in frame_payloads:
            source_depth = payload["depth"]
            for curve_info in payload["curves"]:
                mnemonic = str(curve_info["name"]).upper()
                resampled = _resample_to_depth(
                    source_depth,
                    curve_info["values"],
                    well._depth,
                )
                valid_count = int(np.count_nonzero(np.isfinite(resampled)))
                if valid_count == 0:
                    continue
                rank = (int(curve_info["quality_score"]), valid_count)
                if mnemonic in best_rank and rank <= best_rank[mnemonic]:
                    continue

                metadata = CurveMetadata(
                    description=curve_info["description"],
                    source="DLIS",
                    data_type=DataType.CURVE,
                    custom={
                        "frame": curve_info["frame"],
                        "original_channel": mnemonic,
                        "standardization_priority": int(curve_info["quality_score"]),
                        "channel_quality_reasons": list(curve_info["quality_reasons"]),
                    },
                )
                well._curves[mnemonic] = Curve(
                    mnemonic=mnemonic,
                    data=resampled,
                    unit=curve_info["unit"],
                    depth=well._depth,
                    depth_unit=well._depth_unit,
                    metadata=metadata,
                )
                best_rank[mnemonic] = rank
                source_frames[mnemonic] = curve_info["frame"]
                source_quality[mnemonic] = {
                    "score": int(curve_info["quality_score"]),
                    "reasons": list(curve_info["quality_reasons"]),
                }

        lower_quality_sources = {
            name: details
            for name, details in source_quality.items()
            if details["score"] < 0
        }
        if lower_quality_sources:
            names = ", ".join(sorted(lower_quality_sources))
            warnings.warn(
                f"DLIS merge retained lower-quality/raw channels because no better "
                f"candidate was available: {names}. Review dlis_merge source_quality "
                "or pass channel_preference explicitly.",
                stacklevel=2,
            )

        well._parameters["dlis_merge"] = {
            "frames": [payload["name"] for payload in frame_payloads],
            "depth_unit": depth_unit,
            "step": float(step) if step is not None else None,
            "step_strategy": step_strategy,
            "source_frames": source_frames,
            "source_quality": source_quality,
            "channel_preference": preferences,
        }

        if standardize:
            from petrophys.preprocessing.standardization import standardize_curves

            well = standardize_curves(well, rename_curves=True, convert_units=True)
        return well

    def close(self):
        """Close the DLIS file."""
        if hasattr(self, "_physical_file"):
            self._physical_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def read_dlis(
    filepath: Union[str, Path],
    frame_name: Optional[str] = None,
    channels: Optional[List[str]] = None,
    logical_file: int = 0,
    depth_unit: Optional[str] = None,
    include_vector_channels: bool = True,
    merge_frames: Optional[bool] = None,
    standardize: bool = True,
    step: Optional[float] = None,
    step_strategy: str = "finest",
    max_samples: int = 2_000_000,
    channel_preference: Optional[Mapping[str, str]] = None,
    **kwargs,
) -> "Well":
    """
    Read DLIS file and return Well object.

    Parameters
    ----------
    filepath : str or Path
        Path to DLIS file
    frame_name : str, optional
        Name of frame to read
    channels : list of str, optional
        Specific channels to read
    logical_file : int, optional
        Index of logical file to read (default: 0)
    depth_unit : str, optional
        Target depth unit. DLIS ``0.1 in`` depth is converted to metres by default.
    merge_frames : bool, optional
        Merge scalar channels from all selected frames onto one depth index.
        By default, this is enabled when no frame_name is supplied so DLIS behaves
        like a full well-log input. Set False to read a single frame.
    **kwargs
        Additional arguments (reserved)

    Returns
    -------
    Well
        Well object with loaded data

    Examples
    --------
    >>> from petrophys.io import read_dlis
    >>> well = read_dlis("well_data.dlis")
    """
    with DLISReader(filepath) as reader:
        if logical_file > 0:
            reader.set_active_file(logical_file)
        do_merge = frame_name is None if merge_frames is None else merge_frames
        if do_merge:
            frame_names = [frame_name] if frame_name else None
            return reader.to_merged_well(
                frame_names=frame_names,
                channels=channels,
                depth_unit=depth_unit or "m",
                step=step,
                standardize=standardize,
                step_strategy=step_strategy,
                max_samples=max_samples,
                channel_preference=channel_preference,
            )
        return reader.to_well(
            frame_name=frame_name,
            channels=channels,
            depth_unit=depth_unit,
            include_vector_channels=include_vector_channels,
        )


def read_dlis_merged(
    filepath: Union[str, Path],
    frame_names: Optional[Iterable[str]] = None,
    channels: Optional[Iterable[str]] = None,
    logical_file: int = 0,
    depth_unit: str = "m",
    step: Optional[float] = None,
    standardize: bool = True,
    step_strategy: str = "finest",
    max_samples: int = 2_000_000,
    channel_preference: Optional[Mapping[str, str]] = None,
) -> "Well":
    """Read scalar DLIS log frames, merge them, and return one Well."""
    with DLISReader(filepath) as reader:
        if logical_file > 0:
            reader.set_active_file(logical_file)
        return reader.to_merged_well(
            frame_names=frame_names,
            channels=channels,
            depth_unit=depth_unit,
            step=step,
            standardize=standardize,
            step_strategy=step_strategy,
            max_samples=max_samples,
            channel_preference=channel_preference,
        )
