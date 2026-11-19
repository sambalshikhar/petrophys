"""
Well data container class for petrophysical analysis.

This module provides the Well class which serves as the primary
container for well log data, header information, and computed results.
"""

from petrophys._internal.advice import Advice
from typing import Optional, Union, Dict, List, Any, Iterator, Tuple
import copy
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

from petrophys._internal.benchmark import reproducing_benchmark
from petrophys.model.curve import Curve, CurveMetadata
from petrophys.model.data_types import DataType
from petrophys.model.units import normalize_depth_unit


#: Header parameters reported by :meth:`Well.describe` when present.
_DESCRIBED_PARAMETERS = ("BHT", "RW", "RWT", "RMF", "RMFT", "CLFL", "RM", "RMT", "MUD", "DFD", "BS", "TD")


def _raise_non_numeric_depth(series: pd.Series, depth_column: Any) -> None:
    """Raise ValueError for a depth column without a numeric value."""
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        raise ValueError(f"Depth column {depth_column!r} has no values.")
    examples = values.unique()[:3].tolist()
    hint = ""
    if values.str.fullmatch(r"[+-]?\d{1,3}(,\d{3})+(\.\d+)?").any():
        hint = " The values use thousands separators; pass thousands=','."
    elif values.str.contains(",", regex=False).any():
        hint = " The values use decimal commas; pass decimal=','."
    raise ValueError(
        f"Depth column {depth_column!r} has no numeric values (e.g. {examples}).{hint}"
    )


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
        Total depth in the well's depth unit (``Well.depth_unit``)
    td_source_unit : str
        TD unit as the source file wrote it (the LAS reader converts TD
        from it)
    bit_size : float
        Bit size in inches
    bit_size_source_unit : str
        Bit size unit as the source file wrote it (the LAS reader converts
        mm, cm and m to inches)
    bit_size_intervals : list of (float, float, float)
        Bit size of each bit run as (top, bottom, inches) in the well's depth
        unit, the hole each bit drilled (top exclusive, bottom inclusive);
        empty when the source records no numbered bit runs
    las_well_items : dict
        Other LAS ~Well items (e.g. SRVC, DATE, EKB), mnemonic to
        ``{"unit", "value", "descr"}``, which ``export_las`` writes back
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
        self.td: Optional[float] = None  # In the well's depth unit
        self.td_source_unit: str = ""
        self.bht: Optional[float] = None  # Bottom hole temperature
        self.bht_unit: str = "F"
        self.rm: Optional[float] = None  # Mud resistivity
        self.rmf: Optional[float] = None  # Mud filtrate resistivity
        self.rmc: Optional[float] = None  # Mud cake resistivity
        self.rm_temp: Optional[float] = None  # Temperature at which Rm was measured
        self.bit_size: Optional[float] = None  # Bit size in inches
        self.bit_size_source_unit: str = ""
        self.bit_size_intervals: List[Tuple[float, float, float]] = []
        self.mud_type: str = ""  # WBM, OBM, etc.
        self.las_well_items: Dict[str, Dict[str, Any]] = {}
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


class CurveNames(list):
    """The well's curve mnemonics; a name also looks up its curve.

    ``well.curves["GR"]`` returns the curve, as ``well["GR"]`` does; integer
    indexes and slices work as on a list.
    """

    def __init__(self, names: List[str], well: "Well"):
        super().__init__(names)
        self._well = well

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._well[key]
        return super().__getitem__(key)

    def __reduce__(self):
        # Pickle and copy as a plain list, without the well.
        return (list, (list(self),))


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
    >>> import petrophys as pp
    >>> well = pp.Well.from_las(pp.sample("HW-3"))
    >>> print(well.curves)
    ['GR', 'RHOB', 'NPHI', 'DT', 'PHIE_CC', 'RT', 'SW_INPUT', 'PERM_CC']

    >>> # Access curves
    >>> gr = well["GR"]
    >>> print(round(gr.mean(), 1))
    12.3

    >>> # Add computed curve (arithmetic keeps the RHOB unit, so set the result's)
    >>> well["PHID"] = (2.65 - well["RHOB"]) / (2.65 - 1.0)
    >>> well["PHID"].unit = "v/v"
    """

    def __init__(self, name: str = ""):
        self._curves: Dict[str, Curve] = {}
        self._depth: Optional[np.ndarray] = None
        self._depth_unit: str = "ft"
        self._source_depth_unit_label: Optional[str] = None
        #: Deepest log sample of the well these samples were selected from
        #: (None for a well that is not a selection), in the depth unit.
        self._logged_bottom: Optional[float] = None
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
        """Set depth array; the curves take the new depth (lengths must match)."""
        depth = np.asarray(value, dtype=np.float64)
        if reproducing_benchmark():
            self._depth = depth
            return
        for mnemonic, curve in self._curves.items():
            if len(curve.data) != len(depth):
                raise ValueError(
                    f"Depth has {len(depth)} samples but curve {mnemonic!r} has "
                    f"{len(curve.data)}; lengths must match."
                )
        self._depth = depth
        for curve in self._curves.values():
            curve.depth = depth

    @property
    def depth_unit(self) -> str:
        """Return depth unit."""
        return self._depth_unit

    @depth_unit.setter
    def depth_unit(self, value: str):
        """Set depth unit (``"m"`` or ``"ft"``; aliases such as ``"M"`` or ``"feet"`` are normalized).

        The curves take the new unit too. It only relabels the depth; set
        :attr:`depth` to converted values first when changing units.
        """
        self._depth_unit = normalize_depth_unit(value)
        if not reproducing_benchmark():
            for curve in self._curves.values():
                curve.depth_unit = self._depth_unit

    @property
    def curves(self) -> "CurveNames":
        """Return the list of curve mnemonics; ``curves["GR"]`` gives the curve."""
        return CurveNames(list(self._curves.keys()), self)

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
        """Return the depth step: the median spacing of the depth samples, in the well's depth unit.

        None for a well with fewer than two finite depths.
        """
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
        """Set or add a curve.

        A curve stored under a new mnemonic is stored as a copy, so the
        caller's curve keeps its name. A curve without depth takes the well
        depth; one with a different depth is refused.
        """
        mnemonic = mnemonic.upper()
        if isinstance(value, Curve):
            n_samples = self.n_samples
            if n_samples and len(value.data) != n_samples:
                raise ValueError(
                    f"Curve {mnemonic!r} has {len(value.data)} samples but "
                    f"well has {n_samples}; lengths must match."
                )
            if not reproducing_benchmark():
                if value.mnemonic != mnemonic:
                    value = value.copy()
                if self._depth is not None:
                    if value.depth is None:
                        value.depth = self._depth
                        value.depth_unit = self._depth_unit
                    elif not np.array_equal(value.depth, self._depth, equal_nan=True):
                        raise ValueError(
                            f"Curve {mnemonic!r} has a different depth from the well; "
                            "resample it onto well.depth first."
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

    def get(self, mnemonic: str, default: Optional[Curve] = None) -> Optional[Curve]:
        """Get curve or return default if not found."""
        try:
            curve: Curve = self[mnemonic]
        except KeyError:
            return default
        return curve

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

        A curve already stored under ``mnemonic`` is replaced.

        Parameters
        ----------
        mnemonic : str
            Curve mnemonic (stored in upper case)
        data : array-like
            Curve data values, one per depth sample; another length raises
            ValueError when the well has a depth
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
        """Rename a curve.

        Raises
        ------
        ValueError
            If another curve already has ``new_name``.
        """
        old_name = old_name.upper()
        new_name = new_name.upper()
        if (
            old_name in self._curves
            and new_name != old_name
            and new_name in self._curves
            and not reproducing_benchmark()
        ):
            raise ValueError(
                f"Cannot rename {old_name!r} to {new_name!r}: the well already has "
                f"a curve {new_name!r}; remove it first."
            )
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
            DataFrame with curve data. The depth index is the first column,
            ``DEPTH``; a curve named DEPTH is the column ``DEPTH_CURVE``
            then, with a warning.

        Raises
        ------
        KeyError
            If a requested curve is not in the well.
        """
        if curves is None:
            curves = list(self._curves.keys())
        elif not reproducing_benchmark():
            missing = [name for name in curves if name.upper() not in self._curves]
            if missing:
                raise KeyError(f"Curves not found: {', '.join(missing)}")

        data = {}
        if include_depth and self._depth is not None:
            data["DEPTH"] = self._depth

        for mnemonic in curves:
            if mnemonic.upper() in self._curves:
                column = mnemonic.upper()
                if column in data and not reproducing_benchmark():
                    column = f"{column}_CURVE"
                    while column in data or column in self._curves:
                        column += "_"
                    warnings.warn(
                        f"Curve {mnemonic.upper()!r} has the name of the depth column; "
                        f"it is the column {column!r}.",
                        UserWarning,
                        stacklevel=2,
                    )
                data[column] = self._curves[mnemonic.upper()].data

        return pd.DataFrame(data)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        depth_column: str = "DEPTH",
        depth_unit: Optional[str] = None,
        name: str = "",
        header_units: bool = True,
    ) -> "Well":
        """
        Create Well from pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with curve data
        depth_column : str, optional
            Name of depth column (matched case-insensitively). A frame whose
            index carries this name, as ``lasio``'s ``LASFile.df()`` returns,
            uses the index as depth.
        depth_unit : {"m", "ft"}
            Unit of ``depth_column``. Required when the frame has a depth
            column, unless its header gives the unit (``DEPTH (m)``); when
            both are given they must agree.
        name : str, optional
            Well name
        header_units : bool, optional
            Read a column header ``NAME (unit)`` or ``NAME [unit]`` as curve
            ``NAME`` with that unit, as :func:`petrophys.io.read_csv` does
            (default: True). False keeps each header as the curve name.

        Returns
        -------
        Well
            New Well instance. Values of mostly numeric columns that are not
            numbers become NaN and are listed in the
            ``COERCED_NON_NUMERIC_VALUES`` parameter.

        Raises
        ------
        ValueError
            If two columns become the same curve name (they differ only in
            case or unit; curve names are upper case), the depth column is
            missing or has no numeric value, or the depth header's unit
            differs from ``depth_unit``.
        """
        well = cls(name=name)

        checked = not reproducing_benchmark()
        units: Dict[str, str] = {}
        if checked and header_units:
            from petrophys.io.ascii_reader import _split_unit_header

            split = [_split_unit_header(col) for col in df.columns]
            if any(unit is not None for _, unit in split):
                named: Dict[str, Any] = {}
                for col, (curve_name, _) in zip(df.columns, split):
                    if curve_name in named:
                        raise ValueError(
                            f"Columns {named[curve_name]!r} and {col!r} both become curve "
                            f"{curve_name!r}; rename one of them."
                        )
                    named[curve_name] = col
                df = df.rename(columns={
                    col: curve_name for col, (curve_name, unit) in zip(df.columns, split) if unit is not None
                })
                units = {curve_name: unit for curve_name, unit in split if unit}
                depth_name, depth_header_unit = _split_unit_header(depth_column)
                if depth_header_unit is not None:
                    depth_column = depth_name
        if checked:
            if depth_column not in df.columns:
                matches = [col for col in df.columns if str(col).upper() == str(depth_column).upper()]
                index_name = df.index.name
                if len(matches) == 1:
                    depth_column = matches[0]
                elif (
                    not matches
                    and index_name is not None
                    and str(index_name).upper() == str(depth_column).upper()
                ):
                    df = df.reset_index()
                    depth_column = df.columns[0]
            if depth_column not in df.columns:
                hint = ""
                if len(df.columns) == 1 and any(sep in str(df.columns[0]) for sep in (";", ",", "\t", "|", " ")):
                    hint = (
                        f" The header was read as the single column {str(df.columns[0])!r}; "
                        "pass the delimiter the file uses (e.g. delimiter=';')."
                    )
                raise ValueError(
                    f"Depth column {depth_column!r} not found. Available columns: "
                    f"{[str(col) for col in df.columns]}. Set depth_column explicitly.{hint}"
                )
            seen: Dict[str, Any] = {}
            for col in df.columns:
                if col == depth_column:
                    continue
                key = str(col).upper()
                if key in seen:
                    raise ValueError(
                        f"Columns {seen[key]!r} and {col!r} both become curve {key!r}; "
                        "rename one of them."
                    )
                seen[key] = col

        if depth_column in df.columns:
            if units:
                from petrophys.io.ascii_reader import _depth_unit_from_header

                depth_unit = _depth_unit_from_header(
                    units.get(str(depth_column).upper()), depth_unit, str(depth_column), "DataFrame"
                )
            if depth_unit is None:
                raise ValueError(
                    Advice("Well.from_dataframe needs depth_unit='m' or depth_unit='ft' "
                    f"for depth column {depth_column!r}.", ('depth_unit',))
                )
            well._depth = pd.to_numeric(
                df[depth_column],
                errors="coerce",
            ).to_numpy(dtype=float)
            if checked and len(well._depth) and not np.isfinite(well._depth).any():
                _raise_non_numeric_depth(df[depth_column], depth_column)
            well._depth_unit = normalize_depth_unit(depth_unit)

        skipped_non_numeric: Dict[str, List[str]] = {}
        coerced_non_numeric: Dict[str, List[str]] = {}
        for col in df.columns:
            if col == depth_column:
                continue

            series = df[col]
            numeric = pd.to_numeric(series, errors="coerce")
            has_numeric = bool(numeric.notna().any())
            has_value = bool(series.notna().any())
            if checked:
                col = str(col)
            if has_numeric or not has_value:
                well.add_curve(col, numeric.to_numpy(dtype=float))
                lost = series.notna() & numeric.isna()
                if checked and bool(lost.any()):
                    coerced_non_numeric[col.upper()] = series[lost].astype(str).unique()[:20].tolist()
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
        if coerced_non_numeric:
            well.set_parameter("COERCED_NON_NUMERIC_VALUES", coerced_non_numeric)
        for mnemonic, unit in units.items():
            if mnemonic in well._curves:
                well._curves[mnemonic].unit = unit

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
            Options of :func:`petrophys.io.read_las` (``encoding``)

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
            Options of :func:`petrophys.io.read_dlis` (``frame_name``,
            ``channels``, ``standardize``, ...)

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
            Options of :func:`petrophys.io.export_las` (e.g. ``curves=[...]``)
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

        The header, parameters and results are copied, and the new well
        remembers the deepest log sample of this one (see :meth:`select`).

        Parameters
        ----------
        top : float, optional
            Top depth of interval, in the well's depth unit
        bottom : float, optional
            Bottom depth of interval, in the well's depth unit

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
        return self.select(mask)

    def select(self, index: Any) -> "Well":
        """
        Extract the samples selected by a slice, boolean mask or index array.

        Every curve is selected the same way (:meth:`Curve.select`); the
        header, parameters and results are copied. Use it for a set of samples
        that is not one depth range, e.g. a lithology group. The selection
        remembers the deepest log sample of the well it came from, so an
        interpretation of it places a bottom-hole temperature at the same
        depth as an interpretation of the whole well.

        Parameters
        ----------
        index : slice, array of bool or array of int
            The samples to keep, over the well's depth index.

        Returns
        -------
        Well
            New Well with the selected samples

        Examples
        --------
        >>> sand = well.select(well.get("VCL").data < 0.3)    # doctest: +SKIP
        """
        if self._depth is None:
            raise ValueError("Well has no depth data")
        if isinstance(index, np.ndarray) and index.dtype == bool and index.shape != self._depth.shape:
            raise ValueError(f"mask has {index.shape[0]} samples, the well {self._depth.shape[0]}")

        new_well = Well(name=self.name)
        new_well._depth = self._depth[index]
        new_well._depth_unit = self._depth_unit
        new_well._logged_bottom = self._logged_bottom if self._logged_bottom is not None else self.bottom
        new_well.header = copy.deepcopy(self.header)
        new_well._parameters = copy.deepcopy(self._parameters)
        new_well._results = copy.deepcopy(self._results)

        for mnemonic, curve in self._curves.items():
            new_well._curves[mnemonic] = curve.select(index)

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

    def describe(self) -> Dict[str, Any]:
        """Summarize the well as JSON-serializable data.

        Returns
        -------
        dict
            ``name``, header identifiers, ``depth_unit``, ``top``,
            ``bottom``, ``step``, ``n_samples``, ``curves`` (mnemonic, unit,
            description and share of valid samples), selected header
            parameters (BHT, RW, RMF, ...) and attached geology. Missing
            values are ``None``.

        Examples
        --------
        >>> import petrophys as pp
        >>> summary = pp.read(pp.sample("30_6-5")).describe()
        >>> summary["depth_unit"], len(summary["curves"]) > 10
        ('m', True)
        """
        from petrophys._internal.serialization import json_object

        curves = []
        for mnemonic, curve in self._curves.items():
            size = int(np.asarray(curve.data).size)
            curves.append({
                "mnemonic": mnemonic,
                "unit": curve.unit,
                "description": curve.metadata.description,
                "valid_fraction": round(curve.valid_count() / size, 3) if size else 0.0,
            })
        header = {
            key: getattr(self.header, key, None)
            for key in ("uwi", "api", "field", "company", "location", "country")
            if getattr(self.header, key, None)
        }
        geology = self.get_parameter("_geology_context") or {}
        return json_object({
            "name": self.name,
            "header": header,
            "depth_unit": self._depth_unit,
            "top": self.top,
            "bottom": self.bottom,
            "step": self.step,
            "n_samples": self.n_samples,
            "curves": curves,
            "header_parameters": {
                name: self.get_parameter(name)
                for name in _DESCRIBED_PARAMETERS
                if self.get_parameter(name) is not None
            },
            "formations": [row.get("formation_name") for row in geology.get("formations", [])],
            "lithology_intervals": len(geology.get("lithology", [])),
        })

    def info(self) -> str:
        """Return summary information about the well.

        Depths are given to at most three decimals and the step (the median
        sample spacing) to four significant figures, in the well's depth unit.
        """
        def number(value: Optional[float], spec: str) -> str:
            if value is None:
                return "None"
            return format(value, spec).rstrip("0").rstrip(".") if spec.endswith("f") else format(value, spec)

        lines = [
            f"Well: {self.name}",
            f"UWI: {self.header.uwi}",
            f"Field: {self.header.field}",
            f"Depth range: {number(self.top, '.3f')} - {number(self.bottom, '.3f')} {self._depth_unit}",
            f"Step: {number(self.step, '.4g')} {self._depth_unit}",
            f"Number of curves: {self.n_curves}",
            f"Number of samples: {self.n_samples}",
            "",
            "Curves:",
        ]
        for mnemonic, curve in self._curves.items():
            lines.append(f"  {mnemonic}: {curve.unit}, {curve.valid_count()} valid")

        return "\n".join(lines)
