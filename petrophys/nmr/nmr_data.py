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
T2_CUTOFF_CLASTIC   = 32.0   # Gulf of Mexico clastics (also matches bin boundary)
T2_CUTOFF_CARBONATE = 92.0   # Carbonates


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
        Free Fluid Index (pu).  If absent, computed from bins at
        ``t2_cutoff``.
    mbvi : np.ndarray, optional
        Bulk Volume Irreducible (pu).  If absent, computed from bins.
    bins : np.ndarray, optional
        Shape ``(n_depth, n_bins)`` — porosity contribution (pu) for
        each T2 bin at each depth.
    t2_centers : np.ndarray
        T2 bin centre times (ms).  Length must equal ``bins.shape[1]``.
        Default: [4, 8, 16, 32, 64, 128, 256, 512] ms.
    t2_cutoff : float
        T2 cutoff separating MBVI from MFFI (ms).  Default: 32 ms.
    metadata : dict
        Source file info, well name, etc.

    Examples
    --------
    >>> nmr = NMRData.from_csv("nmr.csv")
    >>> t2lm  = nmr.t2_lm()
    >>> k_sdr = nmr.permeability_sdr()
    >>> k_tc  = nmr.permeability_coates()
    """

    depth:      np.ndarray
    mphi:       np.ndarray
    mffi:       Optional[np.ndarray]     = None
    mbvi:       Optional[np.ndarray]     = None
    bins:       Optional[np.ndarray]     = None
    t2_centers: np.ndarray               = field(default_factory=lambda: DEFAULT_T2_CENTERS.copy())
    t2_cutoff:  float                    = T2_CUTOFF_CLASTIC
    metadata:   Dict[str, Any]           = field(default_factory=dict)

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
    def top(self) -> float:
        return float(np.nanmin(self.depth))

    @property
    def bottom(self) -> float:
        return float(np.nanmax(self.depth))

    @property
    def has_bins(self) -> bool:
        return self.bins is not None

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        filepath: Union[str, Path],
        depth_col:  str        = "Depth",
        mphi_col:   str        = "MPHI",
        mffi_col:   Optional[str] = "MFFI",
        mbvi_col:   Optional[str] = "MBVI",
        bin_cols:   Optional[List[str]] = None,
        t2_centers: Optional[np.ndarray] = None,
        t2_cutoff:  float = T2_CUTOFF_CLASTIC,
        null_value: float = -999.25,
        **read_csv_kwargs,
    ) -> "NMRData":
        """
        Load NMR data from a CSV file.

        The default column layout matches the provided ``nmr.csv`` format::

            Depth, MPHI, P1, P2, P3, P4, P5, P6, P7, P8, MFFI, MBVI

        Parameters
        ----------
        filepath : str or Path
        depth_col : str
        mphi_col : str
            Total NMR porosity column.
        mffi_col : str, optional
        mbvi_col : str, optional
        bin_cols : list of str, optional
            Ordered list of T2 bin column names.
            Default: ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"].
        t2_centers : np.ndarray, optional
            T2 bin centre values (ms).
            Default: [4, 8, 16, 32, 64, 128, 256, 512] ms.
        t2_cutoff : float
            T2 cutoff for MBVI/MFFI partitioning (ms).
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
            t2_centers=t2_centers,
            t2_cutoff=t2_cutoff,
            null_value=null_value,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: "pd.DataFrame",
        depth_col:  str        = "Depth",
        mphi_col:   str        = "MPHI",
        mffi_col:   Optional[str] = "MFFI",
        mbvi_col:   Optional[str] = "MBVI",
        bin_cols:   Optional[List[str]] = None,
        t2_centers: Optional[np.ndarray] = None,
        t2_cutoff:  float = T2_CUTOFF_CLASTIC,
        null_value: float = -999.25,
    ) -> "NMRData":
        """Create NMRData from a pandas DataFrame."""
        if not HAS_PANDAS:
            raise ImportError("pandas is required")

        if bin_cols is None:
            # Use DEFAULT_BIN_COLS that exist in the dataframe
            bin_cols = [c for c in DEFAULT_BIN_COLS if c in df.columns]

        def _col(name: Optional[str]) -> Optional[np.ndarray]:
            if not name or name not in df.columns:
                return None
            arr = df[name].values.astype(float)
            arr[arr == null_value] = np.nan
            return arr

        depth = df[depth_col].values.astype(float)
        mphi  = _col(mphi_col)
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
            t2_centers=np.asarray(t2c, dtype=float),
            t2_cutoff=t2_cutoff,
        )

    @classmethod
    def from_las(
        cls,
        filepath: Union[str, Path],
        depth_mnemonic: str        = "DEPT",
        mphi_mnemonic:  str        = "MPHI",
        mffi_mnemonic:  Optional[str] = "MFFI",
        mbvi_mnemonic:  Optional[str] = "MBVI",
        bin_mnemonics:  Optional[List[str]] = None,
        t2_centers:     Optional[np.ndarray] = None,
        t2_cutoff:      float = T2_CUTOFF_CLASTIC,
        null_value:     float = -999.25,
    ) -> "NMRData":
        """
        Load NMR data from a LAS file.

        Parameters
        ----------
        filepath : str or Path
        depth_mnemonic : str
        mphi_mnemonic : str
        mffi_mnemonic : str, optional
        mbvi_mnemonic : str, optional
        bin_mnemonics : list of str, optional
            Ordered list of T2 bin curve mnemonics (e.g. ["T2B1",…,"T2B8"]).
        t2_centers : np.ndarray, optional
        t2_cutoff : float
        null_value : float
        """
        if not HAS_LASIO:
            raise ImportError("lasio is required: pip install lasio")

        las = lasio.read(str(filepath))

        def _get(mn: Optional[str]) -> Optional[np.ndarray]:
            if mn is None or mn not in las.keys():
                return None
            arr = np.array(las[mn], dtype=float)
            arr[arr == null_value] = np.nan
            return arr

        depth = np.array(las[depth_mnemonic], dtype=float)
        mphi  = _get(mphi_mnemonic)
        if mphi is None:
            raise ValueError(f"MPHI mnemonic '{mphi_mnemonic}' not found")

        bins = None
        if bin_mnemonics:
            arrays = [_get(mn) for mn in bin_mnemonics]
            if all(a is not None for a in arrays):
                bins = np.column_stack(arrays).astype(float)

        t2c = t2_centers if t2_centers is not None else DEFAULT_T2_CENTERS.copy()

        meta = {
            "well":   las.well.WELL.value if hasattr(las.well, "WELL") else "",
            "source": str(filepath),
        }

        return cls(
            depth=depth,
            mphi=mphi,
            mffi=_get(mffi_mnemonic),
            mbvi=_get(mbvi_mnemonic),
            bins=bins,
            t2_centers=np.asarray(t2c, dtype=float),
            t2_cutoff=t2_cutoff,
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # T2 distribution analysis
    # ------------------------------------------------------------------

    def t2_lm(self) -> np.ndarray:
        """
        Logarithmic mean T2 (T2lm) from the T2 bin distribution.

        .. math::
            T2_{lm} = \\exp\\!\\left(
                \\frac{\\sum_i \\phi_i \\ln T2_i}{\\sum_i \\phi_i}
            \\right)

        Returns
        -------
        np.ndarray
            T2lm in ms.  NaN where total bin porosity is zero.

        Raises
        ------
        ValueError
            If bin data are not loaded.
        """
        if not self.has_bins:
            raise ValueError(
                "T2 bin data (P1–P8) required for T2lm. "
                "Load a file that includes the bin columns."
            )
        ln_t2 = np.log(self.t2_centers)          # shape (n_bins,)
        phi   = self.bins                          # shape (n, n_bins)
        total = np.nansum(phi, axis=1)             # shape (n,)

        with np.errstate(divide="ignore", invalid="ignore"):
            t2lm = np.where(
                total > 0,
                np.exp(np.nansum(phi * ln_t2, axis=1) / total),
                np.nan,
            )
        return t2lm

    def bvi_from_bins(
        self, cutoff_ms: Optional[float] = None
    ) -> np.ndarray:
        """
        Bulk Volume Irreducible computed directly from T2 bins.

        Sums all bins with T2 centre **strictly less than** ``cutoff_ms``.

        Parameters
        ----------
        cutoff_ms : float, optional
            T2 cutoff in ms.  Defaults to ``self.t2_cutoff``.

        Returns
        -------
        np.ndarray
            BVI (pu).
        """
        if not self.has_bins:
            raise ValueError("Bin data required for bvi_from_bins()")
        cutoff = cutoff_ms if cutoff_ms is not None else self.t2_cutoff
        mask   = self.t2_centers < cutoff
        return np.nansum(self.bins[:, mask], axis=1)

    def ffi_from_bins(
        self, cutoff_ms: Optional[float] = None
    ) -> np.ndarray:
        """
        Free Fluid Index computed directly from T2 bins.

        Sums all bins with T2 centre **>= cutoff_ms**.

        Parameters
        ----------
        cutoff_ms : float, optional
            T2 cutoff in ms.  Defaults to ``self.t2_cutoff``.
        """
        if not self.has_bins:
            raise ValueError("Bin data required for ffi_from_bins()")
        cutoff = cutoff_ms if cutoff_ms is not None else self.t2_cutoff
        mask   = self.t2_centers >= cutoff
        return np.nansum(self.bins[:, mask], axis=1)

    def t2_distribution(self, depth_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the T2 distribution at one depth sample.

        Parameters
        ----------
        depth_index : int
            Index into the depth array.

        Returns
        -------
        (t2_centers, bin_porosities)
            T2 centres in ms and corresponding porosity (pu).
        """
        if not self.has_bins:
            raise ValueError("Bin data required")
        return self.t2_centers.copy(), self.bins[depth_index].copy()

    # ------------------------------------------------------------------
    # Permeability
    # ------------------------------------------------------------------

    def _ffi(self) -> np.ndarray:
        """MFFI from stored data or computed from bins."""
        if self.mffi is not None:
            return self.mffi
        if self.has_bins:
            return self.ffi_from_bins()
        raise ValueError("MFFI data required (load mffi_col or provide bins)")

    def _bvi(self) -> np.ndarray:
        """MBVI from stored data or computed from bins."""
        if self.mbvi is not None:
            return self.mbvi
        if self.has_bins:
            return self.bvi_from_bins()
        raise ValueError("MBVI data required (load mbvi_col or provide bins)")

    def permeability_sdr(self, c: float = 4.0) -> np.ndarray:
        """
        SDR (Schlumberger-Doll Research) NMR permeability.

        .. math::
            k_{SDR} = C \\cdot \\phi_{NMR}^{4} \\cdot T2_{lm}^{2}

        where φ is in fraction (0–1) and T2lm in ms.

        Parameters
        ----------
        c : float
            Empirical constant (default 4.0 — Kenyon 1992).
            Calibrate against core permeability for best results.

        Returns
        -------
        np.ndarray
            Permeability estimate (mD).
        """
        phi  = self.mphi / 100.0      # pu → fraction
        t2lm = self.t2_lm()           # ms
        with np.errstate(invalid="ignore"):
            return c * phi ** 4 * t2lm ** 2

    def permeability_coates(
        self,
        c: float = 10.0,
        a: float = 4.0,
        b: float = 2.0,
    ) -> np.ndarray:
        """
        Timur-Coates NMR permeability.

        .. math::
            k_{TC} = \\left(\\frac{\\phi_{NMR}}{C}\\right)^{a}
                     \\cdot \\left(\\frac{FFI}{BVI}\\right)^{b}

        where φ, FFI, BVI are all in the same porosity units (pu or
        fraction — they cancel in the FFI/BVI ratio).

        Parameters
        ----------
        c : float
            Coates constant (default 10.0).
        a : float
            Porosity exponent (default 4.0).
        b : float
            FFI/BVI exponent (default 2.0).

        Returns
        -------
        np.ndarray
            Permeability estimate (mD).  NaN where BVI = 0.
        """
        phi = self.mphi          # pu — kept as pu (standard Coates form)
        ffi = self._ffi()
        bvi = self._bvi()

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(bvi > 0, ffi / bvi, np.nan)
            return (phi / c) ** a * ratio ** b

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

        NMR surface relaxation:

        .. math::
            \\frac{1}{T2} = \\rho_2 \\cdot \\frac{S}{V}

        For a sphere S/V = 3/r, so r = 3 ρ₂ T2lm (converted from ms to s).

        Parameters
        ----------
        surface_relaxivity : float
            ρ₂ in µm/s.  Typical: 10–30 µm/s for clastics,
            1–10 µm/s for carbonates.
        geometry : {"sphere", "cylinder", "slit"}
            Pore model for the S/V geometry factor
            (sphere→3, cylinder→2, slit→1).

        Returns
        -------
        np.ndarray
            Pore body radius (µm).
        """
        geom_factor = {"sphere": 3.0, "cylinder": 2.0, "slit": 1.0}.get(
            geometry.lower(), 3.0
        )
        t2lm_s = self.t2_lm() * 1e-3          # ms → s
        return geom_factor * surface_relaxivity * t2lm_s  # µm

    def swirr_nmr(self) -> np.ndarray:
        """
        NMR-derived irreducible water saturation.

        .. math::
            S_{wirr} = \\frac{MBVI}{MPHI}

        Returns
        -------
        np.ndarray
            Swirr (0–1).  NaN where MPHI = 0.
        """
        bvi = self._bvi()
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.mphi > 0, bvi / self.mphi, np.nan)

    def bin_cumulative(self) -> np.ndarray:
        """
        Cumulative T2 distribution (fraction of MPHI) for each depth.

        Returns
        -------
        np.ndarray
            Shape (n_depth, n_bins).  Each row sums to 1 where MPHI > 0.
        """
        if not self.has_bins:
            raise ValueError("Bin data required")
        total = np.nansum(self.bins, axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            normed = np.where(total > 0, self.bins / total, 0.0)
        return np.cumsum(normed, axis=1)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def filter_by_depth(self, top: float, bottom: float) -> "NMRData":
        """Return a new NMRData filtered to [top, bottom]."""
        mask = (self.depth >= top) & (self.depth <= bottom)
        return NMRData(
            depth=self.depth[mask],
            mphi=self.mphi[mask],
            mffi=self.mffi[mask]  if self.mffi is not None else None,
            mbvi=self.mbvi[mask]  if self.mbvi is not None else None,
            bins=self.bins[mask]  if self.bins is not None else None,
            t2_centers=self.t2_centers.copy(),
            t2_cutoff=self.t2_cutoff,
            metadata=dict(self.metadata),
        )

    def to_dataframe(self, include_derived: bool = True) -> "pd.DataFrame":
        """
        Export to a pandas DataFrame.

        Parameters
        ----------
        include_derived : bool
            If True (default), append T2lm, SDR and Coates permeability,
            pore size, and Swirr columns when bins are available.
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required")

        data: Dict[str, np.ndarray] = {"DEPTH": self.depth, "MPHI": self.mphi}

        if self.mffi is not None:
            data["MFFI"] = self.mffi
        if self.mbvi is not None:
            data["MBVI"] = self.mbvi

        if self.has_bins:
            for i, t2 in enumerate(self.t2_centers):
                data[f"P{i+1}"] = self.bins[:, i]

        if include_derived and self.has_bins:
            data["T2LM"]     = self.t2_lm()
            data["K_SDR"]    = self.permeability_sdr()
            data["K_COATES"] = self.permeability_coates()
            data["PORE_R"]   = self.pore_size()
            data["SWIRR"]    = self.swirr_nmr()

        return pd.DataFrame(data)

    def __repr__(self) -> str:
        info = [f"samples={self.n_samples}", f"depth={self.top:.1f}-{self.bottom:.1f}"]
        if self.has_bins:
            info.append(f"{self.n_bins} T2 bins")
        if self.mffi is not None:
            info.append("MFFI")
        if self.mbvi is not None:
            info.append("MBVI")
        info.append(f"cutoff={self.t2_cutoff}ms")
        return f"NMRData({', '.join(info)})"
