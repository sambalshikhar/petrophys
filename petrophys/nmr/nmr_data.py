"""
NMR log data container and analysis.

Handles pre-processed NMR logs containing T2 distribution bins
(P1–P8), total NMR porosity (MPHI), Free Fluid Index (MFFI), and
Bulk Volume Irreducible (MBVI), as produced by Halliburton MRIL-Prime
and similar tools.

T2 bin centres default to [4, 8, 16, 32, 64, 128, 256, 512] ms with
a default T2 cutoff of 32 ms (clastic); bins with T2 < cutoff sum to
MBVI, bins with T2 >= cutoff sum to MFFI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

# Default T2 bin centres (ms) — Halliburton MRIL-Prime / Philliec459 convention
DEFAULT_T2_CENTERS = np.array([4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0])

# Standard CSV column names for the 8 bins
DEFAULT_BIN_COLS = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]

# T2 cutoff defaults (ms)
T2_CUTOFF_CLASTIC = 32.0  # Gulf of Mexico clastics (matches bin boundary)
T2_CUTOFF_CARBONATE = 92.0  # Carbonates

CMR_MPHI_ALIASES = ("TCMR", "CMR_TOT", "CMR_TOTAL", "CMRPHI", "NMRP", "PHINMR")
CMR_MFFI_ALIASES = ("CMFF", "CMR_FFI", "NMR_FFI", "FFI")
CMR_MBVI_ALIASES = ("CBVI", "CMR_BVI", "NMR_BVI", "BVI")
T2LM_ALIASES = ("T2ML", "T2LM_CMR", "T2LOGMEAN")
T2_CUTOFF_ALIASES = ("T2_CUTOFF", "T2CUT", "T2CO")
SDR_PERM_ALIASES = ("KSDR", "K_SDR", "KSDR_CMR")
TIMUR_PERM_ALIASES = ("KTIM", "K_TIM", "KTC", "K_COATES")
CMR_EXTRA_ALIASES = (
    "CMRP_3MS",
    "CMR_TEMP",
    "CDF",
    "STIT",
    "TENS",
    "HD1_PPC1",
    "HD2_PPC1",
    "BS",
    "GR_EDTC",
)


def _finite_median(values: np.ndarray, default: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.nanmedian(finite)) if finite.size else float(default)


def _looks_fractional(values: np.ndarray) -> bool:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return bool(finite.size and np.nanmax(np.abs(finite)) <= 1.5)


def _porosity_to_pu(
    values: Optional[np.ndarray],
    unit: str = "",
    mnemonic: str = "",
) -> Optional[np.ndarray]:
    """Return porosity in percent/porosity units, preserving old NMR semantics."""
    if values is None:
        return None
    arr = np.asarray(values, dtype=float).copy()
    unit_text = (unit or "").strip().lower().replace(" ", "")
    mnemonic_text = (mnemonic or "").strip().upper()
    fraction_unit = (
        "v/v" in unit_text
        or unit_text in {"vv", "frac", "fraction", "decimal"}
        or unit_text.endswith("/v")
    )
    fraction_mnemonic = mnemonic_text in {
        "TCMR",
        "CMFF",
        "CMRP_3MS",
        "CMR_FFI",
        "CMR_BVI",
        "NMR_FFI",
        "NMR_BVI",
    }
    if fraction_unit or (fraction_mnemonic and _looks_fractional(arr)):
        arr = arr * 100.0
    return arr


@dataclass
class NMRData:
    """
    NMR log data container.

    Parameters
    ----------
    depth : np.ndarray
        Measured depth (ft or m).
    mphi : np.ndarray
        Total NMR porosity — sum of all T2 bins (porosity units, pu).
    mffi : np.ndarray, optional
        Free Fluid Index (pu).  If absent, computed from bins.
    mbvi : np.ndarray, optional
        Bulk Volume Irreducible (pu).  If absent, computed from bins.
    bins : np.ndarray, optional
        Shape ``(n_depth, n_bins)`` — porosity contribution (pu) per T2 bin.
    t2lm_stored : np.ndarray, optional
        Pre-computed T2 log mean (ms) loaded directly from the LAS file.
        Used by ``t2_lm()`` when bin data are not available.
    t2_centers : np.ndarray
        T2 bin centre times (ms).  Default: [4, 8, 16, 32, 64, 128, 256, 512] ms.
    t2_cutoff : float
        T2 cutoff separating MBVI from MFFI (ms).  Default: 32 ms.
    metadata : dict
        Source file info, well name, etc.

    Examples
    --------
    >>> nmr = NMRData.from_las(petrophys.sample("daydream1_cmr"),
    ...                        mphi_mnemonic="TCMR", mffi_mnemonic="CMFF")
    >>> t2lm  = nmr.t2_lm()
    >>> k_sdr = nmr.permeability_sdr()
    >>> k_tc  = nmr.permeability_coates()
    >>> cal   = nmr.calibrate_sdr(core_depth, core_perm)
    >>> k_cal = nmr.permeability_sdr(calibration=cal)
    """

    depth: np.ndarray
    mphi: np.ndarray
    mffi: Optional[np.ndarray] = None
    mbvi: Optional[np.ndarray] = None
    bins: Optional[np.ndarray] = None
    t2lm_stored: Optional[np.ndarray] = None
    t2_cutoff_curve: Optional[np.ndarray] = None
    t2_centers: np.ndarray = field(default_factory=lambda: DEFAULT_T2_CENTERS.copy())
    t2_cutoff: float = T2_CUTOFF_CLASTIC
    stored_permeability: Dict[str, np.ndarray] = field(default_factory=dict)
    extra_curves: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        return len(self.depth)

    @property
    def n_bins(self) -> int:
        return len(self.t2_centers)

    @property
    def top(self) -> Optional[float]:
        return finite_min_max(self.depth)[0]

    @property
    def bottom(self) -> Optional[float]:
        return finite_min_max(self.depth)[1]

    @property
    def has_bins(self) -> bool:
        return self.bins is not None

    @property
    def has_t2lm(self) -> bool:
        """True if T2lm is available either from bins or a stored curve."""
        return self.has_bins or self.t2lm_stored is not None

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        filepath: Union[str, Path],
        depth_col: str = "Depth",
        mphi_col: str = "MPHI",
        mffi_col: Optional[str] = "MFFI",
        mbvi_col: Optional[str] = "MBVI",
        bin_cols: Optional[List[str]] = None,
        t2lm_col: Optional[str] = None,
        t2_centers: Optional[np.ndarray] = None,
        t2_cutoff: float = T2_CUTOFF_CLASTIC,
        null_value: float = -999.25,
        **read_csv_kwargs,
    ) -> "NMRData":
        """
        Load NMR data from a CSV file.

        Default column layout matches the standard MRIL-style CSV export::

            Depth, MPHI, P1, P2, P3, P4, P5, P6, P7, P8, MFFI, MBVI

        Parameters
        ----------
        filepath : str or Path
        depth_col : str
        mphi_col : str
        mffi_col, mbvi_col : str, optional
        bin_cols : list of str, optional
            T2 bin columns in T2-ascending order.
            Default: ["P1"..."P8"] when present.
        t2lm_col : str, optional
            Column containing a pre-computed T2 log mean (ms).
        t2_centers : np.ndarray, optional
            T2 bin centre values (ms).
        t2_cutoff : float
        null_value : float
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required: pip install pandas")
        df = pd.read_csv(filepath, **read_csv_kwargs)
        return cls.from_dataframe(
            df,
            depth_col=depth_col,
            mphi_col=mphi_col,
            mffi_col=mffi_col,
            mbvi_col=mbvi_col,
            bin_cols=bin_cols,
            t2lm_col=t2lm_col,
            t2_centers=t2_centers,
            t2_cutoff=t2_cutoff,
            null_value=null_value,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: "pd.DataFrame",
        depth_col: str = "Depth",
        mphi_col: str = "MPHI",
        mffi_col: Optional[str] = "MFFI",
        mbvi_col: Optional[str] = "MBVI",
        bin_cols: Optional[List[str]] = None,
        t2lm_col: Optional[str] = None,
        t2_centers: Optional[np.ndarray] = None,
        t2_cutoff: float = T2_CUTOFF_CLASTIC,
        null_value: float = -999.25,
    ) -> "NMRData":
        """Create NMRData from a pandas DataFrame."""
        if not HAS_PANDAS:
            raise ImportError("pandas is required")

        if bin_cols is None:
            bin_cols = [c for c in DEFAULT_BIN_COLS if c in df.columns]

        def _col(name: Optional[str]) -> Optional[np.ndarray]:
            if not name or name not in df.columns:
                return None
            arr = df[name].values.astype(float)
            arr[arr == null_value] = np.nan
            return arr

        depth = df[depth_col].values.astype(float)
        mphi = _col(mphi_col)
        if mphi is None:
            raise ValueError(f"MPHI column '{mphi_col}' not found")

        bins = None
        if bin_cols:
            bins = np.column_stack([_col(c) for c in bin_cols]).astype(float)

        t2c = t2_centers if t2_centers is not None else DEFAULT_T2_CENTERS.copy()

        return cls(
            depth=depth,
            mphi=mphi,
            mffi=_col(mffi_col),
            mbvi=_col(mbvi_col),
            bins=bins,
            t2lm_stored=_col(t2lm_col),
            t2_centers=np.asarray(t2c, dtype=float),
            t2_cutoff=t2_cutoff,
        )

    @classmethod
    def from_las(
        cls,
        filepath: Union[str, Path],
        mphi_mnemonic: str = "MPHI",
        mffi_mnemonic: Optional[str] = "MFFI",
        mbvi_mnemonic: Optional[str] = "MBVI",
        t2lm_mnemonic: Optional[str] = "T2LM",
        t2_cutoff_mnemonic: Optional[str] = "T2CUTOFF",
        ksdr_mnemonic: Optional[str] = "KSDR",
        ktim_mnemonic: Optional[str] = "KTIM",
        bin_mnemonics: Optional[List[str]] = None,
        t2_centers: Optional[np.ndarray] = None,
        t2_cutoff: float = T2_CUTOFF_CLASTIC,
        null_value: float = -999.25,
        derive_mbvi: bool = True,
    ) -> "NMRData":
        """
        Load NMR data from a LAS file.

        Depth is read via ``lasio``'s ``las.index`` property, which always
        returns the first column regardless of its mnemonic name.

        Parameters
        ----------
        filepath : str or Path
        mphi_mnemonic : str
        mffi_mnemonic, mbvi_mnemonic : str, optional
        t2lm_mnemonic : str, optional
            Mnemonic for a pre-computed T2 log mean curve (e.g. ``"T2LM"``
            from Halliburton MRIL output).  Used when bin curves are absent.
        bin_mnemonics : list of str, optional
            Ordered T2 bin curve mnemonics (e.g. ``["T2B01",…,"T2B08"]``).
        t2_centers : np.ndarray, optional
        t2_cutoff : float
        null_value : float
        """
        if not HAS_LASIO:
            raise ImportError("lasio is required: pip install lasio")

        las = lasio.read(str(filepath))
        key_lookup = {str(key).upper(): str(key) for key in las.keys()}
        unit_lookup = {
            str(curve.mnemonic).upper(): (curve.unit or "") for curve in las.curves
        }

        def _resolve(
            mnemonic: Optional[str],
            aliases: tuple[str, ...] = (),
        ) -> Optional[str]:
            candidates = []
            if mnemonic:
                candidates.append(str(mnemonic))
            candidates.extend(aliases)
            for candidate in candidates:
                key = key_lookup.get(str(candidate).upper())
                if key is not None:
                    return key
            return None

        def _get_resolved(mn: Optional[str]) -> Optional[np.ndarray]:
            if mn is None:
                return None
            arr = np.array(las[mn], dtype=float)
            arr[arr == null_value] = np.nan
            return arr

        def _unit(mn: Optional[str]) -> str:
            return unit_lookup.get(str(mn or "").upper(), "")

        mphi_name = _resolve(mphi_mnemonic, CMR_MPHI_ALIASES)
        mffi_name = _resolve(mffi_mnemonic, CMR_MFFI_ALIASES) if mffi_mnemonic else None
        mbvi_name = _resolve(mbvi_mnemonic, CMR_MBVI_ALIASES) if mbvi_mnemonic else None
        t2lm_name = _resolve(t2lm_mnemonic, T2LM_ALIASES) if t2lm_mnemonic else None
        t2_cutoff_name = (
            _resolve(t2_cutoff_mnemonic, T2_CUTOFF_ALIASES)
            if t2_cutoff_mnemonic
            else None
        )
        ksdr_name = _resolve(ksdr_mnemonic, SDR_PERM_ALIASES) if ksdr_mnemonic else None
        ktim_name = (
            _resolve(ktim_mnemonic, TIMUR_PERM_ALIASES) if ktim_mnemonic else None
        )

        depth = np.array(las.index, dtype=float)
        mphi = _porosity_to_pu(
            _get_resolved(mphi_name),
            _unit(mphi_name),
            mphi_name or "",
        )
        if mphi is None:
            raise ValueError(f"MPHI mnemonic '{mphi_mnemonic}' not found in {filepath}")

        mffi = _porosity_to_pu(
            _get_resolved(mffi_name),
            _unit(mffi_name),
            mffi_name or "",
        )
        mbvi = _porosity_to_pu(
            _get_resolved(mbvi_name),
            _unit(mbvi_name),
            mbvi_name or "",
        )
        if mbvi is None and derive_mbvi and mffi is not None:
            with np.errstate(invalid="ignore"):
                mbvi = np.where(
                    np.isfinite(mphi) & np.isfinite(mffi), mphi - mffi, np.nan
                )
                mbvi = np.where(mbvi >= 0, mbvi, np.nan)

        bins = None
        resolved_bins = None
        if bin_mnemonics is None:
            p_bins = [_resolve(mn) for mn in DEFAULT_BIN_COLS]
            t2_bins = [_resolve(f"T2B{i:02d}") for i in range(1, 9)]
            if all(name is not None for name in p_bins):
                resolved_bins = p_bins
            elif all(name is not None for name in t2_bins):
                resolved_bins = t2_bins
        else:
            resolved_bins = [_resolve(mn) for mn in bin_mnemonics]

        if resolved_bins:
            arrays = [
                _porosity_to_pu(_get_resolved(mn), _unit(mn), mn or "")
                for mn in resolved_bins
            ]
            if all(a is not None for a in arrays):
                bins = np.column_stack(arrays).astype(float)

        t2c = t2_centers if t2_centers is not None else DEFAULT_T2_CENTERS.copy()
        t2_cutoff_curve = _get_resolved(t2_cutoff_name)
        cutoff_value = (
            _finite_median(t2_cutoff_curve, t2_cutoff)
            if t2_cutoff_curve is not None
            else float(t2_cutoff)
        )

        stored_permeability: Dict[str, np.ndarray] = {}
        ksdr = _get_resolved(ksdr_name)
        ktim = _get_resolved(ktim_name)
        if ksdr is not None:
            stored_permeability["KSDR"] = ksdr
        if ktim is not None:
            stored_permeability["KTIM"] = ktim

        extra_curves: Dict[str, np.ndarray] = {}
        for name in CMR_EXTRA_ALIASES:
            resolved = _resolve(name)
            if resolved is None:
                continue
            arr = _get_resolved(resolved)
            if arr is None:
                continue
            if resolved.upper() == "CMRP_3MS":
                arr = _porosity_to_pu(arr, _unit(resolved), resolved)
            extra_curves[resolved.upper()] = arr
        if t2_cutoff_curve is not None:
            extra_curves["T2CUTOFF"] = t2_cutoff_curve
        for key, value in stored_permeability.items():
            extra_curves[key] = value

        meta = {
            "well": las.well.WELL.value if hasattr(las.well, "WELL") else "",
            "source": str(filepath),
            "profile": (
                "schlumberger_cmr"
                if (mphi_name or "").upper() == "TCMR"
                else "generic_nmr"
            ),
            "porosity_unit": "pu",
            "source_mnemonics": {
                "mphi": mphi_name,
                "mffi": mffi_name,
                "mbvi": mbvi_name,
                "t2lm": t2lm_name,
                "t2_cutoff": t2_cutoff_name,
                "ksdr": ksdr_name,
                "ktim": ktim_name,
                "bins": resolved_bins or [],
            },
            "mbvi_derived": bool(mbvi_name is None and mbvi is not None),
        }

        return cls(
            depth=depth,
            mphi=mphi,
            mffi=mffi,
            mbvi=mbvi,
            bins=bins,
            t2lm_stored=_get_resolved(t2lm_name),
            t2_cutoff_curve=t2_cutoff_curve,
            t2_centers=np.asarray(t2c, dtype=float),
            t2_cutoff=cutoff_value,
            stored_permeability=stored_permeability,
            extra_curves=extra_curves,
            metadata=meta,
        )

    @classmethod
    def from_cmr_las(
        cls,
        filepath: Union[str, Path],
        **kwargs,
    ) -> "NMRData":
        """Load a processed Schlumberger CMR LAS summary log."""
        defaults = {
            "mphi_mnemonic": "TCMR",
            "mffi_mnemonic": "CMFF",
            "mbvi_mnemonic": None,
            "t2_cutoff_mnemonic": "T2CUTOFF",
            "ksdr_mnemonic": "KSDR",
            "ktim_mnemonic": "KTIM",
            "derive_mbvi": True,
        }
        defaults.update(kwargs)
        return cls.from_las(filepath, **defaults)

    # ------------------------------------------------------------------
    # T2 distribution analysis
    # ------------------------------------------------------------------

    def t2_lm(self) -> np.ndarray:
        """
        Logarithmic mean T2 (T2lm) in ms.

        Computed from T2 bins when available (preferred); falls back to
        the ``t2lm_stored`` curve loaded from LAS.  Raises ``ValueError``
        if neither is present.

        .. math::
            T2_{lm} = \\exp\\!\\left(
                \\frac{\\sum_i \\phi_i \\ln T2_i}{\\sum_i \\phi_i}
            \\right)
        """
        if self.has_bins:
            ln_t2 = np.log(self.t2_centers)
            phi = self.bins
            total = np.nansum(phi, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(
                    total > 0,
                    np.exp(np.nansum(phi * ln_t2, axis=1) / total),
                    np.nan,
                )

        if self.t2lm_stored is not None:
            return self.t2lm_stored.copy()

        raise ValueError(
            "T2lm requires either T2 bin data (P1–P8) or a T2LM curve "
            "loaded from the LAS file (pass t2lm_mnemonic='T2LM' to from_las)."
        )

    def bvi_from_bins(self, cutoff_ms: Optional[float] = None) -> np.ndarray:
        """
        Bulk Volume Irreducible from T2 bins (bins with T2 < cutoff).

        Parameters
        ----------
        cutoff_ms : float, optional
            T2 cutoff (ms).  Defaults to ``self.t2_cutoff``.
        """
        if not self.has_bins:
            raise ValueError("T2 bin data required for bvi_from_bins()")
        cutoff = cutoff_ms
        if cutoff is None:
            cutoff = (
                self.t2_cutoff_curve
                if self.t2_cutoff_curve is not None
                else self.t2_cutoff
            )
        if np.isscalar(cutoff):
            return np.nansum(self.bins[:, self.t2_centers < float(cutoff)], axis=1)

        cutoff_arr = np.asarray(cutoff, dtype=float)
        if cutoff_arr.shape != (self.n_samples,):
            raise ValueError("Per-depth T2 cutoff must match the number of samples")
        mask = self.t2_centers[None, :] < cutoff_arr[:, None]
        return np.nansum(np.where(mask, self.bins, 0.0), axis=1)

    def ffi_from_bins(self, cutoff_ms: Optional[float] = None) -> np.ndarray:
        """
        Free Fluid Index from T2 bins (bins with T2 >= cutoff).

        Parameters
        ----------
        cutoff_ms : float, optional
            T2 cutoff (ms).  Defaults to ``self.t2_cutoff``.
        """
        if not self.has_bins:
            raise ValueError("T2 bin data required for ffi_from_bins()")
        cutoff = cutoff_ms
        if cutoff is None:
            cutoff = (
                self.t2_cutoff_curve
                if self.t2_cutoff_curve is not None
                else self.t2_cutoff
            )
        if np.isscalar(cutoff):
            return np.nansum(self.bins[:, self.t2_centers >= float(cutoff)], axis=1)

        cutoff_arr = np.asarray(cutoff, dtype=float)
        if cutoff_arr.shape != (self.n_samples,):
            raise ValueError("Per-depth T2 cutoff must match the number of samples")
        mask = self.t2_centers[None, :] >= cutoff_arr[:, None]
        return np.nansum(np.where(mask, self.bins, 0.0), axis=1)

    def t2_distribution(self, depth_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (t2_centers, bin_porosities) at one depth index."""
        if not self.has_bins:
            raise ValueError("T2 bin data required")
        return self.t2_centers.copy(), self.bins[depth_index].copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ffi(self) -> np.ndarray:
        if self.mffi is not None:
            return self.mffi
        if self.has_bins:
            return self.ffi_from_bins()
        raise ValueError("MFFI data required — load mffi_col or provide bins")

    def _bvi(self) -> np.ndarray:
        if self.mbvi is not None:
            return self.mbvi
        if self.has_bins:
            return self.bvi_from_bins()
        raise ValueError("MBVI data required — load mbvi_col or provide bins")

    # ------------------------------------------------------------------
    # Permeability
    # ------------------------------------------------------------------

    def permeability_sdr(
        self,
        c: float = 4.0,
        calibration: Optional["NMRCalibration"] = None,
        prefer_stored: bool = True,
    ) -> np.ndarray:
        """
        SDR (Schlumberger-Doll Research) NMR permeability.

        .. math::
            k_{SDR} = C \\cdot \\phi_{NMR}^{4} \\cdot T2_{lm}^{2}

        where φ is in fraction (0–1) and T2lm in ms.

        Parameters
        ----------
        c : float
            Empirical constant (default 4.0 — Kenyon 1992).
            Ignored when ``calibration`` is supplied.
        calibration : NMRCalibration, optional
            Result of ``calibrate_sdr()``.  When provided, the fitted
            constant overrides ``c``.

        Returns
        -------
        np.ndarray
            Permeability (mD).
        """
        if calibration is None and prefer_stored and "KSDR" in self.stored_permeability:
            return self.stored_permeability["KSDR"].copy()
        if calibration is not None:
            c = calibration.params["c"]
        phi = self.mphi / 100.0
        t2lm = self.t2_lm()
        with np.errstate(invalid="ignore"):
            return c * phi**4 * t2lm**2

    def permeability_coates(
        self,
        c: float = 10.0,
        a: float = 4.0,
        b: float = 2.0,
        calibration: Optional["NMRCalibration"] = None,
        prefer_stored: bool = True,
    ) -> np.ndarray:
        """
        Timur-Coates NMR permeability.

        .. math::
            k_{TC} = \\left(\\frac{\\phi_{NMR}}{C}\\right)^{a}
                     \\cdot \\left(\\frac{FFI}{BVI}\\right)^{b}

        Parameters
        ----------
        c, a, b : float
            Coates parameters.  Ignored when ``calibration`` is supplied.
        calibration : NMRCalibration, optional
            Result of ``calibrate_coates()``.  When provided, fitted
            parameters override ``c``, ``a``, ``b``.

        Returns
        -------
        np.ndarray
            Permeability (mD).  NaN where BVI = 0.
        """
        if calibration is None and prefer_stored and "KTIM" in self.stored_permeability:
            return self.stored_permeability["KTIM"].copy()
        if calibration is not None:
            c = calibration.params["c"]
            a = calibration.params.get("a", a)
            b = calibration.params.get("b", b)

        phi = self.mphi
        ffi = self._ffi()
        bvi = self._bvi()

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(bvi > 0, ffi / bvi, np.nan)
            return (phi / c) ** a * ratio**b

    # ------------------------------------------------------------------
    # Calibration (delegates to calibration.py)
    # ------------------------------------------------------------------

    def calibrate_sdr(
        self,
        core_depth: Union[np.ndarray, list],
        core_perm: Union[np.ndarray, list],
    ) -> "NMRCalibration":
        """
        Calibrate the SDR constant ``c`` against core permeability.

        Finds ``c`` that minimises the mean squared error between
        log10(k_SDR) and log10(k_core).  Exponents a=4 and b=2 are
        kept at their physically-derived values.

        Parameters
        ----------
        core_depth : array-like
            Depths of core plug measurements (same units as ``self.depth``).
        core_perm : array-like
            Core air/Klinkenberg permeability (mD).

        Returns
        -------
        NMRCalibration
        """
        from petrophys.nmr.calibration import calibrate_sdr as _cal

        return _cal(self, core_depth, core_perm)

    def calibrate_coates(
        self,
        core_depth: Union[np.ndarray, list],
        core_perm: Union[np.ndarray, list],
        fit_exponents: bool = False,
    ) -> "NMRCalibration":
        """
        Calibrate Timur-Coates parameters against core permeability.

        Parameters
        ----------
        core_depth : array-like
        core_perm : array-like
            Core permeability (mD).
        fit_exponents : bool
            If False (default), only ``c`` is fitted (a=4, b=2 fixed).
            If True, ``a``, ``b``, and ``c`` are all fitted via linear
            regression in log space.

        Returns
        -------
        NMRCalibration
        """
        from petrophys.nmr.calibration import calibrate_coates as _cal

        return _cal(self, core_depth, core_perm, fit_exponents=fit_exponents)

    # ------------------------------------------------------------------
    # Pore characterisation
    # ------------------------------------------------------------------

    def pore_size(
        self,
        surface_relaxivity: float = 10.0,
        geometry: str = "sphere",
    ) -> np.ndarray:
        """
        Estimate modal pore body radius from T2lm.

        .. math::
            r = F \\cdot \\rho_2 \\cdot T2_{lm}

        where F = 3 (sphere), 2 (cylinder), or 1 (slit), ρ₂ is surface
        relaxivity (µm/s), and T2lm is in s.

        Parameters
        ----------
        surface_relaxivity : float
            ρ₂ (µm/s).  Typical: 10–30 µm/s clastics, 1–10 µm/s carbonates.
        geometry : {"sphere", "cylinder", "slit"}

        Returns
        -------
        np.ndarray
            Pore body radius (µm).
        """
        factor = {"sphere": 3.0, "cylinder": 2.0, "slit": 1.0}.get(
            geometry.lower(), 3.0
        )
        return factor * surface_relaxivity * self.t2_lm() * 1e-3  # ms → s

    def swirr_nmr(self) -> np.ndarray:
        """
        NMR irreducible water saturation: MBVI / MPHI.

        Returns
        -------
        np.ndarray
            Swirr (0–1).  NaN where MPHI = 0.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.mphi > 0, self._bvi() / self.mphi, np.nan)

    def bin_cumulative(self) -> np.ndarray:
        """
        Cumulative T2 distribution per depth, normalised to MPHI.

        Returns
        -------
        np.ndarray
            Shape (n_depth, n_bins).  Each row sums to 1 where MPHI > 0.
        """
        if not self.has_bins:
            raise ValueError("T2 bin data required")
        total = np.nansum(self.bins, axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            normed = np.where(total > 0, self.bins / total, 0.0)
        return np.cumsum(normed, axis=1)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def filter_by_depth(self, top: float, bottom: float) -> "NMRData":
        """Return a new NMRData filtered to [top, bottom]."""
        lo, hi = sorted((float(top), float(bottom)))
        mask = (self.depth >= lo) & (self.depth <= hi)
        return NMRData(
            depth=self.depth[mask],
            mphi=self.mphi[mask],
            mffi=self.mffi[mask] if self.mffi is not None else None,
            mbvi=self.mbvi[mask] if self.mbvi is not None else None,
            bins=self.bins[mask] if self.bins is not None else None,
            t2lm_stored=(
                self.t2lm_stored[mask] if self.t2lm_stored is not None else None
            ),
            t2_cutoff_curve=(
                self.t2_cutoff_curve[mask] if self.t2_cutoff_curve is not None else None
            ),
            t2_centers=self.t2_centers.copy(),
            t2_cutoff=self.t2_cutoff,
            stored_permeability={
                key: value[mask] for key, value in self.stored_permeability.items()
            },
            extra_curves={key: value[mask] for key, value in self.extra_curves.items()},
            metadata=dict(self.metadata),
        )

    def to_dataframe(self, include_derived: bool = True) -> "pd.DataFrame":
        """
        Export to a pandas DataFrame.

        Parameters
        ----------
        include_derived : bool
            When True, appends T2LM, K_SDR, K_COATES, PORE_R, and SWIRR
            columns (requires T2lm to be computable).
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required")

        data: Dict[str, np.ndarray] = {"DEPTH": self.depth, "MPHI": self.mphi}
        if self.mffi is not None:
            data["MFFI"] = self.mffi
        if self.mbvi is not None:
            data["MBVI"] = self.mbvi
        if self.t2lm_stored is not None:
            data["T2LM_LAS"] = self.t2lm_stored
        if self.t2_cutoff_curve is not None:
            data["T2CUTOFF"] = self.t2_cutoff_curve
        if self.has_bins:
            for i, t2 in enumerate(self.t2_centers):
                data[f"P{i+1}"] = self.bins[:, i]
        for key, values in self.extra_curves.items():
            if key not in data:
                data[key] = values
        for key, values in self.stored_permeability.items():
            data[f"{key}_LAS"] = values

        if include_derived and self.has_t2lm:
            data["T2LM"] = self.t2_lm()
            data["K_SDR"] = self.permeability_sdr()
            try:
                data["K_COATES"] = self.permeability_coates()
            except ValueError:
                pass
            data["PORE_R"] = self.pore_size()

        if include_derived:
            if "KSDR" in self.stored_permeability and "K_SDR" not in data:
                data["K_SDR"] = self.permeability_sdr()
            if "KTIM" in self.stored_permeability and "K_COATES" not in data:
                data["K_COATES"] = self.permeability_coates()
            try:
                data["SWIRR"] = self.swirr_nmr()
            except ValueError:
                pass

        return pd.DataFrame(data)

    def __repr__(self) -> str:
        depth_text = (
            f"{self.top:.1f}-{self.bottom:.1f}"
            if self.top is not None and self.bottom is not None
            else "None-None"
        )
        parts = [f"samples={self.n_samples}", f"depth={depth_text}"]
        if self.has_bins:
            parts.append(f"{self.n_bins} T2 bins")
        if self.t2lm_stored is not None:
            parts.append("T2LM(stored)")
        if self.mffi is not None:
            parts.append("MFFI")
        if self.mbvi is not None:
            parts.append("MBVI")
        if self.stored_permeability:
            parts.append("stored perm")
        profile = self.metadata.get("profile")
        if profile:
            parts.append(str(profile))
        parts.append(f"cutoff={self.t2_cutoff}ms")
        return f"NMRData({', '.join(parts)})"


# Avoid circular import — referenced only in type hint strings above
from petrophys.nmr.calibration import NMRCalibration  # noqa: E402
