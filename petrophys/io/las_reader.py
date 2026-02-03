"""
LAS file reader for well log data.

This module provides functionality to read LAS (Log ASCII Standard)
files and convert them to Well objects.
"""

from typing import Optional, Union, Dict, Any, List
from pathlib import Path
import numpy as np

try:
    import lasio
    HAS_LASIO = True
except ImportError:
    HAS_LASIO = False


class LASReader:
    """
    Reader for LAS (Log ASCII Standard) files.

    Examples
    --------
    >>> reader = LASReader("well_data.las")
    >>> well = reader.to_well()
    """

    def __init__(self, filepath: Union[str, Path]):
        """
        Initialize LAS reader.

        Parameters
        ----------
        filepath : str or Path
            Path to LAS file
        """
        if not HAS_LASIO:
            raise ImportError(
                "lasio is required for LAS file reading. "
                "Install with: pip install lasio"
            )

        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"LAS file not found: {filepath}")

        self._las = lasio.read(str(self.filepath))

    @property
    def las(self):
        """Return underlying lasio object."""
        return self._las

    @property
    def curves(self) -> List[str]:
        """Return list of curve mnemonics."""
        return [c.mnemonic for c in self._las.curves]

    @property
    def well_name(self) -> str:
        """Return well name from header."""
        return self._las.well.WELL.value if hasattr(self._las.well, 'WELL') else ""

    def get_header_value(self, key: str, default: Any = None) -> Any:
        """Get value from LAS header."""
        try:
            if hasattr(self._las.well, key):
                return getattr(self._las.well, key).value
            if hasattr(self._las.params, key):
                return getattr(self._las.params, key).value
        except (AttributeError, KeyError):
            pass
        return default

    def get_curve_data(self, mnemonic: str) -> Optional[np.ndarray]:
        """Get data for a specific curve."""
        if mnemonic in self.curves:
            return self._las[mnemonic]
        return None

    def get_curve_unit(self, mnemonic: str) -> str:
        """Get unit for a specific curve."""
        for curve in self._las.curves:
            if curve.mnemonic == mnemonic:
                return curve.unit or ""
        return ""

    def to_well(self) -> "Well":
        """
        Convert LAS data to Well object.

        Returns
        -------
        Well
            Well object with loaded data
        """
        from petrophys.core.well import Well
        from petrophys.core.curve import Curve, CurveMetadata
        from petrophys.core.data_types import DataType

        well = Well(name=self.well_name)

        # Set header information
        well.header.well_name = self.well_name
        well.header.uwi = self.get_header_value("UWI", "")
        well.header.api = self.get_header_value("API", "")
        well.header.field = self.get_header_value("FLD", "")
        well.header.company = self.get_header_value("COMP", "")
        well.header.location = self.get_header_value("LOC", "")
        well.header.county = self.get_header_value("CNTY", "")
        well.header.state = self.get_header_value("STAT", "")
        well.header.country = self.get_header_value("CTRY", "")

        # Get numeric header values
        well.header.kb_elevation = self._parse_float(self.get_header_value("EKBF"))
        well.header.td = self._parse_float(self.get_header_value("TD"))
        well.header.bht = self._parse_float(self.get_header_value("BHT"))
        well.header.rm = self._parse_float(self.get_header_value("RM"))
        well.header.rmf = self._parse_float(self.get_header_value("RMF"))
        well.header.rmc = self._parse_float(self.get_header_value("RMC"))
        well.header.bit_size = self._parse_float(self.get_header_value("BS"))

        # Set depth
        depth_curve = None
        for name in ["DEPT", "DEPTH", "MD", "TVD"]:
            if name in self.curves:
                depth_curve = name
                break

        if depth_curve:
            well._depth = self._las[depth_curve]
            # Get depth unit
            for curve in self._las.curves:
                if curve.mnemonic == depth_curve:
                    well._depth_unit = curve.unit or "ft"
                    break

        # Add curves
        for las_curve in self._las.curves:
            if las_curve.mnemonic == depth_curve:
                continue

            metadata = CurveMetadata(
                description=las_curve.descr or "",
                source="LAS",
                data_type=DataType.CURVE
            )

            curve = Curve(
                mnemonic=las_curve.mnemonic,
                data=las_curve.data,
                unit=las_curve.unit or "",
                depth=well._depth,
                depth_unit=well._depth_unit,
                metadata=metadata
            )

            well._curves[las_curve.mnemonic.upper()] = curve

        return well

    @staticmethod
    def _parse_float(value: Any) -> Optional[float]:
        """Parse value to float, return None if not possible."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None


def read_las(
    filepath: Union[str, Path],
    **kwargs
) -> "Well":
    """
    Read LAS file and return Well object.

    Parameters
    ----------
    filepath : str or Path
        Path to LAS file
    **kwargs
        Additional arguments (reserved for future use)

    Returns
    -------
    Well
        Well object with loaded data

    Examples
    --------
    >>> from petrophys.io import read_las
    >>> well = read_las("well_data.las")
    >>> print(well.curves)
    ['GR', 'RHOB', 'NPHI', 'RT']
    """
    reader = LASReader(filepath)
    return reader.to_well()
