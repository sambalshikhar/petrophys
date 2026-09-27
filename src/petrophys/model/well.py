"""
Well data container class for petrophysical analysis.

This module provides the Well class which serves as the primary
container for well log data, header information, and computed results.
"""

from typing import Optional, Union, Dict, List, Any, Iterator
import copy
from pathlib import Path
import numpy as np
import pandas as pd

from petrophys.model.curve import Curve, CurveMetadata
from petrophys.model.data_types import DataType
from petrophys.model.units import normalize_depth_unit


class WellHeader:
    """
    Container for well header information.

    Attributes
    ----------
    well_name : str
        Name of the well
    uwi : str
        Unique well identifier
    api : str
        API number
    field : str
        Field name
    company : str
        Operating company
    location : str
        Well location
    county : str
        County/region
    state : str
        State/province
    country : str
        Country
    latitude : float
        Latitude coordinate
    longitude : float
        Longitude coordinate
    kb_elevation : float
        Kelly bushing elevation
    gl_elevation : float
        Ground level elevation
    td : float
        Total depth
    custom : dict
        Additional custom header values
    """

    def __init__(self):
        self.well_name: str = ""
        self.uwi: str = ""
        self.api: str = ""
        self.field: str = ""
        self.company: str = ""
        self.location: str = ""
        self.county: str = ""
        self.state: str = ""
        self.country: str = ""
        self.latitude: Optional[float] = None
        self.longitude: Optional[float] = None
        self.kb_elevation: Optional[float] = None
        self.gl_elevation: Optional[float] = None
        self.td: Optional[float] = None
        self.bht: Optional[float] = None  # Bottom hole temperature
        self.bht_unit: str = "F"
        self.rm: Optional[float] = None  # Mud resistivity
        self.rmf: Optional[float] = None  # Mud filtrate resistivity
        self.rmc: Optional[float] = None  # Mud cake resistivity
        self.rm_temp: Optional[float] = None  # Temperature at which Rm was measured
        self.bit_size: Optional[float] = None  # Bit size in inches
        self.mud_type: str = ""  # WBM, OBM, etc.
        self.custom: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get header value by key name (supports dict-style access)."""
        return self.to_dict().get(key, default)

    def __repr__(self) -> str:
        return f"WellHeader(name={self.well_name}, uwi={self.uwi})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert header to dictionary."""
        return {
            "well_name": self.well_name,
            "uwi": self.uwi,
            "api": self.api,
            "field": self.field,
            "company": self.company,
            "location": self.location,
            "county": self.county,
            "state": self.state,
            "country": self.country,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "kb_elevation": self.kb_elevation,
            "gl_elevation": self.gl_elevation,
            "td": self.td,
            "bht": self.bht,
            "bht_unit": self.bht_unit,
            "rm": self.rm,
            "rmf": self.rmf,
            "rmc": self.rmc,
            "rm_temp": self.rm_temp,
            "bit_size": self.bit_size,
            "mud_type": self.mud_type,
            **self.custom
        }


class Well:
    """
    Primary container for well log data and metadata.

    The Well class serves as the central data structure for petrophysical
    analysis, containing log curves, header information, and computed results.

    Parameters
    ----------
    name : str, optional
        Well name

    Examples
    --------
    >>> from petrophys import Well
    >>> well = Well.from_las("example.las")
    >>> print(well.curves)
    ['DEPTH', 'GR', 'RHOB', 'NPHI', 'RT']

    >>> # Access curves
    >>> gr = well["GR"]
    >>> print(gr.mean())
    65.3

    >>> # Add computed curve
    >>> well["PHID"] = (2.65 - well["RHOB"]) / (2.65 - 1.0)
    """

    def __init__(self, name: str = ""):
        self._curves: Dict[str, Curve] = {}
        self._depth: Optional[np.ndarray] = None
        self._depth_unit: str = "ft"
        self.header = WellHeader()
        self.header.well_name = name

        # Store computed results
        self._results: Dict[str, Any] = {}
        self._parameters: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        """Return well name."""
        return self.header.well_name

    @name.setter
    def name(self, value: str):
        """Set well name."""
        self.header.well_name = value

    @property
    def depth(self) -> Optional[np.ndarray]:
        """Return depth array."""
        return self._depth

    @depth.setter
    def depth(self, value: Union[np.ndarray, List[float]]):
        """Set depth array."""
        self._depth = np.asarray(value, dtype=np.float64)

    @property
    def depth_unit(self) -> str:
        """Return depth unit."""
        return self._depth_unit

    @depth_unit.setter
    def depth_unit(self, value: str):
        """Set depth unit (``"m"`` or ``"ft"``; aliases such as ``"M"`` or ``"feet"`` are normalized)."""
        self._depth_unit = normalize_depth_unit(value)

    @property
    def curves(self) -> List[str]:
        """Return list of curve mnemonics."""
        return list(self._curves.keys())

    @property
    def n_curves(self) -> int:
        """Return number of curves."""
        return len(self._curves)

    @property
    def n_samples(self) -> int:
        """Return number of depth samples."""
        if self._depth is not None:
            return len(self._depth)
        if self._curves:
            return len(next(iter(self._curves.values())))
        return 0

    @property
    def top(self) -> Optional[float]:
        """Return top depth."""
        if self._depth is not None:
            finite = self._depth[np.isfinite(self._depth)]
            if finite.size == 0:
                return None
            return float(np.min(finite))
        return None

    @property
    def bottom(self) -> Optional[float]:
        """Return bottom depth."""
        if self._depth is not None:
            finite = self._depth[np.isfinite(self._depth)]
            if finite.size == 0:
                return None
            return float(np.max(finite))
        return None

    @property
    def start_depth(self) -> Optional[float]:
        """Alias for top (for visualization module compatibility)."""
        return self.top

    @property
    def stop_depth(self) -> Optional[float]:
        """Alias for bottom (for visualization module compatibility)."""
        return self.bottom

    @property
    def step(self) -> Optional[float]:
        """Return depth step (assuming uniform sampling)."""
        if self._depth is not None:
            finite = self._depth[np.isfinite(self._depth)]
            if len(finite) > 1:
                steps = np.abs(np.diff(finite))
                steps = steps[np.isfinite(steps) & (steps > 0)]
                if steps.size:
                    return float(np.median(steps))
        return None

    def __repr__(self) -> str:
        return f"Well({self.name}, curves={self.n_curves}, samples={self.n_samples})"

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        """Return number of curves."""
        return self.n_curves

    def __contains__(self, mnemonic: str) -> bool:
        """Check if curve exists."""
        return mnemonic.upper() in self._curves

    def __iter__(self) -> Iterator[str]:
        """Iterate over curve mnemonics."""
        return iter(self._curves.keys())

    def __getitem__(self, mnemonic: str) -> Curve:
        """Get curve by mnemonic."""
        mnemonic = mnemonic.upper()
        if mnemonic not in self._curves:
            raise KeyError(f"Curve not found: {mnemonic}")
        return self._curves[mnemonic]

    def __setitem__(self, mnemonic: str, value: Union[Curve, np.ndarray, List[float]]):
        """Set or add a curve."""
        mnemonic = mnemonic.upper()
        if isinstance(value, Curve):
            n_samples = self.n_samples
            if n_samples and len(value.data) != n_samples:
                raise ValueError(
                    f"Curve {mnemonic!r} has {len(value.data)} samples but "
                    f"well has {n_samples}; lengths must match."
                )
            value.mnemonic = mnemonic
            self._curves[mnemonic] = value
        else:
            # Create new Curve from array
            self._curves[mnemonic] = Curve(
                mnemonic=mnemonic,
                data=value,
                depth=self._depth,
                depth_unit=self._depth_unit,
                metadata=CurveMetadata(data_type=DataType.COMPUTED)
            )

    def __delitem__(self, mnemonic: str):
        """Delete a curve."""
        mnemonic = mnemonic.upper()
        if mnemonic in self._curves:
            del self._curves[mnemonic]

    def get(self, mnemonic: str, default: Any = None) -> Optional[Curve]:
        """Get curve or return default if not found."""
        try:
            return self[mnemonic]
        except KeyError:
            return default

    def add_curve(
        self,
        mnemonic: str,
        data: Union[np.ndarray, List[float]],
        unit: str = "",
        description: str = "",
        data_type: DataType = DataType.CURVE
    ) -> Curve:
        """
        Add a new curve to the well.

        Parameters
        ----------
        mnemonic : str
            Curve mnemonic
        data : array-like
            Curve data values
        unit : str, optional
            Unit of measurement
        description : str, optional
            Curve description
        data_type : DataType, optional
            Data type classification

        Returns
        -------
        Curve
            The newly added curve
        """
        metadata = CurveMetadata(
            description=description,
            data_type=data_type
        )
        curve = Curve(
            mnemonic=mnemonic,
            data=data,
            unit=unit,
            depth=self._depth,
            depth_unit=self._depth_unit,
            metadata=metadata
        )
        self._curves[mnemonic.upper()] = curve
        return curve

    def remove_curve(self, mnemonic: str) -> bool:
        """Remove a curve by mnemonic."""
        mnemonic = mnemonic.upper()
        if mnemonic in self._curves:
            del self._curves[mnemonic]
            return True
        return False

    def rename_curve(self, old_name: str, new_name: str) -> bool:
        """Rename a curve."""
        old_name = old_name.upper()
        new_name = new_name.upper()
        if old_name in self._curves:
            curve = self._curves.pop(old_name)
            curve.mnemonic = new_name
            self._curves[new_name] = curve
            return True
        return False

    def has_curve(self, mnemonic: str) -> bool:
        """Check if curve exists."""
        return mnemonic.upper() in self._curves

    def has_curves(self, mnemonics: List[str]) -> bool:
        """Check if all specified curves exist."""
        return all(self.has_curve(m) for m in mnemonics)

    def to_dataframe(
        self,
        curves: Optional[List[str]] = None,
        include_depth: bool = True
    ) -> pd.DataFrame:
        """
        Convert well data to pandas DataFrame.

        Parameters
        ----------
        curves : list of str, optional
            Specific curves to include (default: all)
        include_depth : bool, optional
            Whether to include depth column

        Returns
        -------
        pd.DataFrame
            DataFrame with curve data
        """
        if curves is None:
            curves = list(self._curves.keys())

        data = {}
        if include_depth and self._depth is not None:
            data["DEPTH"] = self._depth

        for mnemonic in curves:
            if mnemonic.upper() in self._curves:
                data[mnemonic.upper()] = self._curves[mnemonic.upper()].data

        return pd.DataFrame(data)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        depth_column: str = "DEPTH",
        depth_unit: str = "ft",
        name: str = ""
    ) -> "Well":
        """
        Create Well from pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with curve data
        depth_column : str, optional
            Name of depth column
        depth_unit : str, optional
            Unit for depth
        name : str, optional
            Well name

        Returns
        -------
        Well
            New Well instance
        """
        well = cls(name=name)

        if depth_column in df.columns:
            well._depth = pd.to_numeric(
                df[depth_column],
                errors="coerce",
            ).to_numpy(dtype=float)
            well._depth_unit = normalize_depth_unit(depth_unit)

        skipped_non_numeric: Dict[str, List[str]] = {}
        for col in df.columns:
            if col == depth_column:
                continue

            series = df[col]
            numeric = pd.to_numeric(series, errors="coerce")
            has_numeric = bool(numeric.notna().any())
            has_value = bool(series.notna().any())
            if has_numeric or not has_value:
                well.add_curve(col, numeric.to_numpy(dtype=float))
                continue

            values = series.dropna().astype(str).unique()[:20].tolist()
            skipped_non_numeric[col] = values
            if (
                not well.name
                and col.upper() in {"WELL", "WELLNAME", "WELL_NAME"}
                and len(values) == 1
            ):
                well.name = values[0]

        if skipped_non_numeric:
            well.set_parameter("SKIPPED_NON_NUMERIC_COLUMNS", sorted(skipped_non_numeric))
            well.set_parameter("SKIPPED_NON_NUMERIC_VALUES", skipped_non_numeric)

        return well

    @classmethod
    def from_las(cls, filepath: Union[str, Path], **kwargs) -> "Well":
        """
        Load well from LAS file.

        Parameters
        ----------
        filepath : str or Path
            Path to LAS file
        **kwargs
            Additional arguments passed to lasio

        Returns
        -------
        Well
            New Well instance
        """
        from petrophys.io.las_reader import read_las
        return read_las(filepath, **kwargs)

    @classmethod
    def from_dlis(cls, filepath: Union[str, Path], **kwargs) -> "Well":
        """
        Load well from DLIS file.

        Parameters
        ----------
        filepath : str or Path
            Path to DLIS file
        **kwargs
            Additional arguments

        Returns
        -------
        Well
            New Well instance
        """
        from petrophys.io.dlis_reader import read_dlis
        return read_dlis(filepath, **kwargs)

    def to_las(self, filepath: Union[str, Path], **kwargs):
        """
        Export well to LAS file.

        Parameters
        ----------
        filepath : str or Path
            Output file path
        **kwargs
            Additional arguments
        """
        from petrophys.io.export import export_las
        export_las(self, filepath, **kwargs)

    def to_csv(self, filepath: Union[str, Path], **kwargs):
        """
        Export well to CSV file.

        Parameters
        ----------
        filepath : str or Path
            Output file path
        **kwargs
            Additional arguments passed to DataFrame.to_csv
        """
        df = self.to_dataframe()
        df.to_csv(filepath, index=False, **kwargs)

    def depth_slice(
        self,
        top: Optional[float] = None,
        bottom: Optional[float] = None
    ) -> "Well":
        """
        Extract well data within depth range.

        Parameters
        ----------
        top : float, optional
            Top depth of interval
        bottom : float, optional
            Bottom depth of interval

        Returns
        -------
        Well
            New Well with sliced data
        """
        if self._depth is None:
            raise ValueError("Well has no depth data")

        mask = np.ones(len(self._depth), dtype=bool)
        if top is not None:
            mask &= self._depth >= top
        if bottom is not None:
            mask &= self._depth <= bottom

        new_well = Well(name=self.name)
        new_well._depth = self._depth[mask]
        new_well._depth_unit = self._depth_unit
        new_well.header = copy.deepcopy(self.header)
        new_well._parameters = copy.deepcopy(self._parameters)
        new_well._results = copy.deepcopy(self._results)

        for mnemonic, curve in self._curves.items():
            new_well._curves[mnemonic] = curve[mask]

        return new_well

    def set_parameter(self, name: str, value: Any):
        """Set a parameter value."""
        self._parameters[name] = value

    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a parameter value."""
        return self._parameters.get(name, default)

    def set_result(self, name: str, value: Any):
        """Store a computed result."""
        self._results[name] = value

    def get_result(self, name: str, default: Any = None) -> Any:
        """Get a computed result."""
        return self._results.get(name, default)

    def info(self) -> str:
        """Return summary information about the well."""
        lines = [
            f"Well: {self.name}",
            f"UWI: {self.header.uwi}",
            f"Field: {self.header.field}",
            f"Depth range: {self.top} - {self.bottom} {self._depth_unit}",
            f"Step: {self.step} {self._depth_unit}",
            f"Number of curves: {self.n_curves}",
            f"Number of samples: {self.n_samples}",
            "",
            "Curves:",
        ]
        for mnemonic, curve in self._curves.items():
            lines.append(f"  {mnemonic}: {curve.unit}, {curve.valid_count()} valid")

        return "\n".join(lines)

    def _extract_well_df_from_well_obj(self, well_obj: Any) -> pd.DataFrame:
        """Safely extract tabular curve data from various Well object implementations."""
        if hasattr(well_obj, "to_df"): return well_obj.to_df()
        if hasattr(well_obj, "to_dataframe"): return well_obj.to_dataframe()
        if hasattr(well_obj, "df") and well_obj.df is not None: return well_obj.df.copy()
        if hasattr(well_obj, "data") and hasattr(well_obj.data, "to_df"): return well_obj.data.to_df()

        # Fallback for dict-like or attribute-based curve containers
        try:
            df = pd.DataFrame({k: v for k, v in well_obj.__dict__.items() if k not in {"name", "metadata", "header"}})
            if df.empty: df = pd.DataFrame(dict(well_obj))
            return df
        except Exception as e:
            raise ValueError(f"Cannot extract DataFrame from Well object. Implement .to_df() or check structure. Error: {e}")

    def _reconstruct_well_from_slices_well_obj(self, slices: List[Any]) -> Any:
        """Concatenate processed depth slices, sort by DEPTH, and reconstruct a Well object."""
        if not slices:
            return self

        dfs = []
        for s in slices:
            try:
                df = self._extract_well_df_from_well_obj(s)
                if "DEPTH" in df.columns:
                    dfs.append(df)
            except Exception as e:
                print(f"⚠️ Warning: Could not extract data from slice {getattr(s, 'name', 'unknown')}: {e}")
                continue

        if not dfs:
            return self

        combined = pd.concat(dfs, axis=0, ignore_index=True)
        if "DEPTH" in combined.columns:
            combined = combined.sort_values("DEPTH").reset_index(drop=True)
            combined = combined.drop_duplicates(subset=["DEPTH"], keep="first")
        else:
            raise ValueError("No 'DEPTH' column found in extracted well data.")

        cls = self.__class__
        well_name = getattr(self, "name", None)

        try:
            if hasattr(cls, "from_df"): return cls.from_df(combined, name=well_name)
            if hasattr(cls, "from_dataframe"): return cls.from_dataframe(combined, name=well_name)
            if hasattr(cls, "from_las") and hasattr(combined, "to_las"): return cls.from_las(combined.to_las())
            return cls(data=combined, name=well_name)
        except Exception as e:
            print(f"⚠️ Well reconstruction failed ({e}). Returning DataFrame instead.")
            return combined
