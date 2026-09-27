"""
LAS file reader for well log data.

This module provides functionality to read LAS (Log ASCII Standard)
files and convert them to Well objects.
"""

from typing import Optional, Union, Dict, Any, List
from pathlib import Path
import warnings
import numpy as np

try:
    import lasio
    HAS_LASIO = True
except ImportError:
    HAS_LASIO = False


def _warn_on_depth_issues(depth: Optional[np.ndarray], source: object = "") -> None:
    """Warn when finite depth samples contain duplicates or change direction."""
    if depth is None:
        return
    arr = np.asarray(depth, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return
    diffs = np.diff(finite)
    issues = []
    if np.any(diffs == 0):
        issues.append("duplicate depth values")
    nonzero_diffs = diffs[diffs != 0]
    if np.any(nonzero_diffs > 0) and np.any(nonzero_diffs < 0):
        issues.append("depth changes direction")
    if issues:
        warnings.warn(
            f"Depth index issue in {source!s}: {', '.join(issues)}. "
            "Verify sample order before interpolation or interval operations.",
            stacklevel=3,
        )


def _normalize_depth_unit(unit: str) -> str:
    text = str(unit or "").strip()
    normalized = text.lower().replace(".", "")
    if normalized in {"f", "ft", "feet", "foot"}:
        return "ft"
    if normalized in {"m", "meter", "meters", "metre", "metres"}:
        return "m"
    return text or "ft"


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

    def get_header_unit(self, key: str, default: str = "") -> str:
        """Get unit text from LAS well or parameter header."""
        try:
            if hasattr(self._las.well, key):
                return str(getattr(self._las.well, key).unit or default)
            if hasattr(self._las.params, key):
                return str(getattr(self._las.params, key).unit or default)
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
        from petrophys.model.well import Well
        from petrophys.model.curve import Curve, CurveMetadata
        from petrophys.model.data_types import DataType

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
        well.header.bht_unit = self.get_header_unit("BHT", well.header.bht_unit) or well.header.bht_unit
        well.header.rm = self._parse_float(self.get_header_value("RM"))
        well.header.rmf = self._parse_float(self.get_header_value("RMF"))
        well.header.rmc = self._parse_float(self.get_header_value("RMC"))
        well.header.bit_size = self._parse_float(self.get_header_value("BS"))

        # Persist ~Parameter metadata on the Well so workflows/tutorials can
        # access zoned petrophysical inputs such as RW_Z1, RMF_Z1, BHT_Z1, etc.
        self._load_parameters(well)

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
                    well._depth_unit = _normalize_depth_unit(curve.unit)
                    break
            _warn_on_depth_issues(well._depth, self.filepath)

        # Add curves
        for las_curve in self._las.curves:
            if las_curve.mnemonic == depth_curve:
                continue

            metadata = CurveMetadata(
                description=las_curve.descr or "",
                source="LAS",
                data_type=DataType.CURVE
            )

            try:
                curve = Curve(
                    mnemonic=las_curve.mnemonic,
                    data=las_curve.data,
                    unit=las_curve.unit or "",
                    depth=well._depth,
                    depth_unit=well._depth_unit,
                    metadata=metadata
                )
            except ValueError:
                warnings.warn(
                    f"Skipping non-numeric LAS curve {las_curve.mnemonic!r}: "
                    "data could not be converted to float."
                )
                continue

            well._curves[las_curve.mnemonic.upper()] = curve

        return well

    def _load_parameters(self, well: "Well") -> None:
        """Copy LAS ~Parameter entries into ``well`` parameters."""
        try:
            params = list(self._las.params)
        except Exception:
            params = []

        for item in params:
            mnemonic = str(getattr(item, "mnemonic", "") or "").strip().upper()
            if not mnemonic:
                continue

            raw_value = getattr(item, "value", None)
            parsed = self._parse_float(raw_value)
            value = parsed if parsed is not None else raw_value
            well.set_parameter(mnemonic, value)

            unit = str(getattr(item, "unit", "") or "").strip()
            descr = str(getattr(item, "descr", "") or "").strip()
            if unit:
                well.set_parameter(f"{mnemonic}__UNIT", unit)
            if descr:
                well.set_parameter(f"{mnemonic}__DESCR", descr)

        # Promote single-valued zoned metadata into the header when the base
        # header is blank, so existing code can benefit without knowing about
        # the zoned LAS convention.
        if well.header.bht is None:
            bht = self._get_constant_zoned_float("BHT")
            bht_unit = self._get_constant_zoned_unit("BHT").lower()
            if bht is not None and "c" in bht_unit and "f" not in bht_unit:
                bht = (bht * 9.0 / 5.0) + 32.0
            if bht is not None:
                well.header.bht = bht
                well.header.bht_unit = "F"

        if well.header.rmf is None:
            rmf = self._get_constant_zoned_float("RMF")
            if rmf is not None:
                well.header.rmf = rmf

        if not well.header.field:
            field_name = self._get_constant_zoned_text("FLD")
            if field_name:
                well.header.field = field_name

    def _get_constant_zoned_float(self, prefix: str) -> Optional[float]:
        """Return a single unique float value from PARAM_Zn entries."""
        values = set()
        prefix = prefix.upper()

        try:
            params = list(self._las.params)
        except Exception:
            params = []

        for item in params:
            mnemonic = str(getattr(item, "mnemonic", "") or "").strip().upper()
            if not mnemonic.startswith(f"{prefix}_Z"):
                continue

            parsed = self._parse_float(getattr(item, "value", None))
            if parsed is not None:
                values.add(float(parsed))

        if len(values) == 1:
            return next(iter(values))
        return None

    def _get_constant_zoned_text(self, prefix: str) -> str:
        """Return a single unique text value from PARAM_Zn entries."""
        values = set()
        prefix = prefix.upper()

        try:
            params = list(self._las.params)
        except Exception:
            params = []

        for item in params:
            mnemonic = str(getattr(item, "mnemonic", "") or "").strip().upper()
            if not mnemonic.startswith(f"{prefix}_Z"):
                continue

            value = str(getattr(item, "value", "") or "").strip()
            if value:
                values.add(value)

        if len(values) == 1:
            return next(iter(values))
        return ""

    def _get_constant_zoned_unit(self, prefix: str) -> str:
        """Return a single unique unit string from PARAM_Zn entries."""
        values = set()
        prefix = prefix.upper()

        try:
            params = list(self._las.params)
        except Exception:
            params = []

        for item in params:
            mnemonic = str(getattr(item, "mnemonic", "") or "").strip().upper()
            if not mnemonic.startswith(f"{prefix}_Z"):
                continue

            value = str(getattr(item, "unit", "") or "").strip()
            if value:
                values.add(value)

        if len(values) == 1:
            return next(iter(values))
        return ""

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
        Reserved for future use; unsupported arguments raise TypeError.

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
    if kwargs:
        raise TypeError(f"Unsupported LAS reader options: {', '.join(sorted(kwargs))}")
    reader = LASReader(filepath)
    return reader.to_well()
