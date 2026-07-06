"""
XRF (X-Ray Fluorescence) geochemistry data from core analysis.

Loads elemental concentration logs measured on core by XRF scanning,
computes standard elemental ratios, and derives mineral proxy indices
useful for lithology discrimination and multi-mineral interpretation.
"""

from typing import Optional, Union, Dict, List, Any
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np

from petrophys.utils.statistics import finite_min_max

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import lasio
    HAS_LASIO = True
except ImportError:
    HAS_LASIO = False


# Common XRF elements and their typical mineral associations
ELEMENT_MINERAL_MAP = {
    "Si": ["quartz", "feldspar", "clay"],
    "Al": ["clay", "feldspar"],
    "Ca": ["calcite", "dolomite", "anhydrite"],
    "Fe": ["pyrite", "siderite", "chlorite", "hematite"],
    "K":  ["illite", "feldspar", "glauconite"],
    "Mg": ["dolomite", "chlorite"],
    "S":  ["pyrite", "anhydrite", "gypsum"],
    "Mn": ["siderite", "rhodochrosite"],
    "Rb": ["illite", "feldspar"],
    "Sr": ["calcite", "anhydrite"],
    "Ti": ["rutile", "ilmenite"],
    "Ba": ["barite"],
}


@dataclass
class XRFData:
    """
    XRF geochemistry measurements along a core depth.

    Parameters
    ----------
    depth : np.ndarray
        Measured depth of each sample (m or ft).
    elements : dict
        Mapping of element symbol → concentration array (% or ppm).
    units : dict
        Mapping of element symbol → unit string (e.g. "%", "ppm").
    errors : dict, optional
        Mapping of element symbol → measurement error array.
    metadata : dict
        Additional header information from the LAS file.

    Examples
    --------
    >>> xrf = XRFData.from_las("core_xrf.las")
    >>> ratios = xrf.elemental_ratios()
    >>> proxies = xrf.mineral_proxies()
    """

    depth: np.ndarray
    elements: Dict[str, np.ndarray]
    units: Dict[str, str] = field(default_factory=dict)
    errors: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---- Properties ----

    @property
    def element_names(self) -> List[str]:
        """List of available element names."""
        return list(self.elements.keys())

    @property
    def n_samples(self) -> int:
        """Number of depth samples."""
        return len(self.depth)

    @property
    def top(self) -> Optional[float]:
        """Shallowest depth."""
        return finite_min_max(self.depth)[0]

    @property
    def bottom(self) -> Optional[float]:
        """Deepest depth."""
        return finite_min_max(self.depth)[1]

    # ---- Loaders ----

    @classmethod
    def from_las(
        cls,
        filepath: Union[str, Path],
        error_suffix: str = "_ERR",
        null_value: float = -999.25,
    ) -> "XRFData":
        """
        Load XRF data from a LAS file.

        Depth is read via ``lasio``'s ``las.index`` property (always the
        first column, regardless of mnemonic name).  Element curves are any
        curve whose mnemonic does not end with ``error_suffix``.

        Parameters
        ----------
        filepath : str or Path
            Path to the LAS file.
        error_suffix : str
            Suffix that identifies error/uncertainty curves (default: "_ERR").
        null_value : float
            Value to replace with NaN (default: -999.25).
        """
        if not HAS_LASIO:
            raise ImportError("lasio is required: pip install lasio")

        las = lasio.read(str(filepath))
        depth = np.array(las.index, dtype=float)
        depth_mnemonic = las.curves[0].mnemonic  # for skip-check below

        elements: Dict[str, np.ndarray] = {}
        units: Dict[str, str] = {}
        errors: Dict[str, np.ndarray] = {}

        for curve in las.curves:
            mn = curve.mnemonic
            if mn == depth_mnemonic:
                continue
            data = np.array(curve.data, dtype=float)
            data[data == null_value] = np.nan

            if mn.endswith(error_suffix):
                base = mn[: -len(error_suffix)]
                errors[base] = data
            else:
                elements[mn] = data
                units[mn] = curve.unit or ""

        meta = {
            "well": las.well.WELL.value if hasattr(las.well, "WELL") else "",
            "source": str(filepath),
        }

        return cls(depth=depth, elements=elements, units=units,
                   errors=errors, metadata=meta)

    @classmethod
    def from_dataframe(
        cls,
        df: "pd.DataFrame",
        depth_col: str = "DEPTH",
        element_cols: Optional[List[str]] = None,
        error_cols: Optional[Dict[str, str]] = None,
        unit_map: Optional[Dict[str, str]] = None,
    ) -> "XRFData":
        """
        Create XRFData from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
        depth_col : str
        element_cols : list, optional
            Columns to treat as element concentrations.
            Defaults to all numeric columns except depth_col.
        error_cols : dict, optional
            Mapping of element name → error column name.
        unit_map : dict, optional
            Mapping of element name → unit string.
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required")

        depth = df[depth_col].values.astype(float)

        if element_cols is None:
            element_cols = [c for c in df.select_dtypes(include=np.number).columns
                            if c != depth_col]

        elements = {col: df[col].values.astype(float) for col in element_cols}
        units = unit_map or {}
        errors: Dict[str, np.ndarray] = {}
        if error_cols:
            for elem, ecol in error_cols.items():
                if ecol in df.columns:
                    errors[elem] = df[ecol].values.astype(float)

        return cls(depth=depth, elements=elements, units=units, errors=errors)

    # ---- Core methods ----

    def get(self, element: str) -> Optional[np.ndarray]:
        """Return concentration array for an element, or None if not present.

        Lookup is case-insensitive so both chemical-symbol keys ("Si") and
        upper-cased LAS mnemonics ("SI") resolve to the same element.
        """
        # NOTE: `a or b` on numpy arrays raises "truth value ambiguous"; pick
        # explicitly instead.
        value = self.elements.get(element)
        if value is None:
            value = self.elements.get(element.upper())
        if value is None:
            value = self.elements.get(element.capitalize())
        if value is None:
            folded = element.casefold()
            for key, data in self.elements.items():
                if key.casefold() == folded:
                    value = data
                    break
        return value

    def _ratio(self, num: str, den: str) -> np.ndarray:
        """Compute element ratio, returning NaN where denominator is zero."""
        a = self.get(num)
        b = self.get(den)
        if a is None or b is None:
            return np.full(self.n_samples, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(b > 0, a / b, np.nan)
        return r

    def elemental_ratios(self) -> Dict[str, np.ndarray]:
        """
        Compute standard geochemical ratios used in mineral discrimination.

        Returns
        -------
        dict
            Ratio name → array. Only ratios where both elements are present
            are included. All others are arrays of NaN.

        Notes
        -----
        Ratios and their petrophysical meaning:

        =========== =========================================================
        Ratio       Interpretation
        =========== =========================================================
        Si_Al       Quartz/clay ratio — high = clean sand
        Ca_Al       Carbonate index — high = limestone/dolomite
        Fe_Al       Iron enrichment — pyrite, siderite, chlorite
        K_Al        Illite/feldspar proxy
        Mg_Ca       Dolomite vs calcite — high = dolomite
        Mg_Al       Chlorite proxy
        S_Fe        Pyrite indicator — FeS₂ has S/Fe ≈ 1.15 by weight
        Ca_Ca_Mg    Calcite fraction in carbonate (Ca/(Ca+Mg))
        Al_Al_Si    Clay fraction (Al/(Al+Si))
        =========== =========================================================
        """
        return {
            "Si_Al":    self._ratio("Si", "Al"),
            "Ca_Al":    self._ratio("Ca", "Al"),
            "Fe_Al":    self._ratio("Fe", "Al"),
            "K_Al":     self._ratio("K",  "Al"),
            "Mg_Ca":    self._ratio("Mg", "Ca"),
            "Mg_Al":    self._ratio("Mg", "Al"),
            "S_Fe":     self._ratio("S",  "Fe"),
            "Rb_K":     self._ratio("Rb", "K"),
            "Mn_Fe":    self._ratio("Mn", "Fe"),
            "Ca_Ca_Mg": self._ca_fraction(),
            "Al_Al_Si": self._clay_fraction(),
        }

    def _ca_fraction(self) -> np.ndarray:
        """Ca/(Ca+Mg) — calcite fraction within carbonates."""
        ca = self.get("Ca")
        mg = self.get("Mg")
        if ca is None or mg is None:
            return np.full(self.n_samples, np.nan)
        denom = ca + mg
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(denom > 0, ca / denom, np.nan)

    def _clay_fraction(self) -> np.ndarray:
        """Al/(Al+Si) — proxy for clay fraction in the solid matrix."""
        al = self.get("Al")
        si = self.get("Si")
        if al is None or si is None:
            return np.full(self.n_samples, np.nan)
        denom = al + si
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(denom > 0, al / denom, np.nan)

    def mineral_proxies(self) -> Dict[str, np.ndarray]:
        """
        Derive normalized mineral proxy indices (0–1) from elemental data.

        These are quick lithology discriminators, not quantitative volumes.
        Use the multi-mineral solver for quantitative mineral fractions.

        Returns
        -------
        dict
            Proxy name → array (0-1 scale where possible).
        """
        proxies: Dict[str, np.ndarray] = {}

        # Quartz index: high Si, low Al, low Ca, low Fe
        si = self.get("Si")
        al = self.get("Al")
        ca = self.get("Ca")
        fe = self.get("Fe")

        if si is not None and al is not None:
            zeros = np.zeros_like(si, dtype=float)
            total = np.nansum([
                si,
                al if al is not None else zeros,
                ca if ca is not None else zeros,
                fe if fe is not None else zeros,
            ], axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                proxies["quartz_index"] = np.where(total > 0, si / total, np.nan)

        # Carbonate index: Ca + Mg normalized by total
        mg = self.get("Mg")
        if ca is not None:
            carb = ca if mg is None else ca + mg
            proxies["carbonate_index"] = self._normalize(carb)

        # Clay index: Al normalized
        if al is not None:
            proxies["clay_index"] = self._normalize(al)

        # Pyrite index: Fe with high S/Fe ratio
        s = self.get("S")
        if fe is not None and s is not None:
            # Pyrite (FeS2) weight ratio S/Fe ≈ 1.15; flag elevated S/Fe
            with np.errstate(divide="ignore", invalid="ignore"):
                s_fe = np.where(fe > 0, s / fe, np.nan)
            proxies["pyrite_index"] = self._normalize(s_fe)

        # Dolomite indicator: Mg/Ca ratio
        if ca is not None and mg is not None:
            proxies["dolomite_index"] = self._normalize(self._ratio("Mg", "Ca"))

        # Siderite indicator: Fe with low S (to exclude pyrite)
        if fe is not None:
            siderite_signal = fe.copy()
            if s is not None:
                # Penalize high S (associated with pyrite/anhydrite)
                with np.errstate(invalid="ignore"):
                    siderite_signal = np.where(s > 0.5, fe * 0.3, fe)
            proxies["siderite_index"] = self._normalize(siderite_signal)

        return proxies

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        """Min-max normalize to [0, 1], ignoring NaN."""
        arr = np.asarray(arr, dtype=float)
        out = np.full(arr.shape, np.nan, dtype=float)
        finite = np.isfinite(arr)
        if not np.any(finite):
            return out
        mn = float(np.nanmin(arr[finite]))
        mx = float(np.nanmax(arr[finite]))
        if mx == mn:
            out[finite] = 0.0
            return out
        out[finite] = (arr[finite] - mn) / (mx - mn)
        return out

    # ---- Utilities ----

    def filter_by_depth(self, top: float, bottom: float) -> "XRFData":
        """Return a new XRFData filtered to [top, bottom]."""
        mask = (self.depth >= top) & (self.depth <= bottom)
        return XRFData(
            depth=self.depth[mask],
            elements={k: v[mask] for k, v in self.elements.items()},
            units=dict(self.units),
            errors={k: v[mask] for k, v in self.errors.items()},
            metadata=dict(self.metadata),
        )

    def to_dataframe(self) -> "pd.DataFrame":
        """Export to a pandas DataFrame with depth + all elements."""
        if not HAS_PANDAS:
            raise ImportError("pandas is required")
        data = {"DEPTH": self.depth}
        data.update(self.elements)
        for elem, err in self.errors.items():
            data[f"{elem}_ERR"] = err
        return pd.DataFrame(data)

    def __repr__(self) -> str:
        elems = ", ".join(self.element_names[:8])
        if len(self.element_names) > 8:
            elems += f" … (+{len(self.element_names) - 8} more)"
        depth_text = (
            f"{self.top:.1f}-{self.bottom:.1f}"
            if self.top is not None and self.bottom is not None
            else "None-None"
        )
        return (f"XRFData(samples={self.n_samples}, "
                f"depth={depth_text}, "
                f"elements=[{elems}])")
