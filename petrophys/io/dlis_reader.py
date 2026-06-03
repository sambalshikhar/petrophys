"""
DLIS file reader for well log data.

This module provides functionality to read DLIS (Digital Log Interchange Standard)
files and convert them to Well objects.
"""

from typing import Optional, Union, Dict, Any, List
from pathlib import Path
import warnings
import numpy as np

try:
    from dlisio import dlis

    HAS_DLISIO = True
except ImportError:
    HAS_DLISIO = False


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
                "Install with: pip install dlisio"
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
        self, frame_name: Optional[str] = None, channels: Optional[List[str]] = None
    ) -> "Well":
        """
        Convert DLIS data to Well object.

        Parameters
        ----------
        frame_name : str, optional
            Name of frame to read (default: first frame)
        channels : list of str, optional
            Specific channels to read (default: all)

        Returns
        -------
        Well
            Well object with loaded data
        """
        from petrophys.base.well import Well
        from petrophys.base.curve import Curve, CurveMetadata
        from petrophys.base.data_types import DataType

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

        # Read frame data
        frame_data = frame.curves()

        # Find depth channel
        depth_data = None
        depth_unit = "ft"
        depth_channel = None

        for ch in frame.channels:
            if ch.name.upper() in ["TDEP", "DEPTH", "DEPT", "MD"]:
                depth_channel = ch.name
                depth_data = frame_data[ch.name]
                depth_unit = ch.units or "ft"
                break

        if depth_data is not None:
            # Handle multi-dimensional depth (take first column if needed)
            if len(depth_data.shape) > 1:
                depth_data = depth_data[:, 0]
            well._depth = depth_data.flatten()
            well._depth_unit = depth_unit

        # Add channels as curves
        for ch in frame.channels:
            if channels and ch.name not in channels:
                continue
            if ch.name == depth_channel:
                continue

            data = frame_data[ch.name]

            # Handle multi-dimensional data. FMI/image channels often arrive
            # as (n_depth, 1, n_buttons); squeeze singleton dimensions, then
            # split true per-depth columns into separate scalar curves.
            if len(data.shape) > 1:
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
                        data=squeezed[:, i].flatten(),
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
                    data=data.flatten(),
                    unit=ch.units or "",
                    depth=well._depth,
                    depth_unit=well._depth_unit,
                    metadata=metadata,
                )
                well._curves[ch.name.upper()] = curve

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
        return reader.to_well(frame_name=frame_name, channels=channels)
