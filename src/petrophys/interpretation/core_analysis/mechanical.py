"""
Core mechanical properties from high-resolution scanning.

Loads UCS (Unconfined Compressive Strength), P-wave velocity (Vp),
and S-wave velocity (Vs) measured on continuous core at centimetre
resolution, and derives dynamic elastic moduli.

References
----------
Mavko, G., Mukerji, T. and Dvorkin, J. (2009). *The Rock Physics Handbook*,
2nd ed. Cambridge University Press.
"""

from typing import Optional, Union, Dict, List, Any
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np

from petrophys._internal.statistics import finite_min_max

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


@dataclass
class CoreMechanical:
    """
    Core mechanical property measurements.

    Parameters
    ----------
    depth : np.ndarray
        Measured depth of each sample (m or ft).
    ucs : np.ndarray
        Uniaxial Compressive Strength (MPa) at the primary scale.
    vp : np.ndarray, optional
        P-wave velocity (m/s).
    vs : np.ndarray, optional
        S-wave velocity (m/s).
    poisson : np.ndarray, optional
        Dynamic Poisson's ratio (dimensionless, measured directly).
    extra_scales : dict, optional
        Additional UCS arrays at other averaging scales,
        e.g. {"5cm": array, "30cm": array}.
    metadata : dict
        Additional header information.

    Examples
    --------
    >>> mech = CoreMechanical.from_las(
    ...     "core_strength.las",
    ...     ucs_mnemonic="UCS_1CM",
    ...     vp_mnemonic="VP_5CM",
    ...     vs_mnemonic="VS_5CM",
    ... )
    >>> moduli = mech.dynamic_moduli(density_gcm3=2.55)
    >>> print(moduli["young_modulus"])
    """

    depth: np.ndarray
    ucs: np.ndarray
    vp: Optional[np.ndarray] = None
    vs: Optional[np.ndarray] = None
    poisson: Optional[np.ndarray] = None
    extra_scales: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---- Properties ----

    @property
    def n_samples(self) -> int:
        return len(self.depth)

    @property
    def top(self) -> Optional[float]:
        return finite_min_max(self.depth)[0]

    @property
    def bottom(self) -> Optional[float]:
        return finite_min_max(self.depth)[1]

    # ---- Loaders ----

    @classmethod
    def from_las(
        cls,
        filepath: Union[str, Path],
        ucs_mnemonic: str = "UCS",
        vp_mnemonic: Optional[str] = "VP",
        vs_mnemonic: Optional[str] = "VS",
        poisson_mnemonic: Optional[str] = None,
        extra_ucs_mnemonics: Optional[Dict[str, str]] = None,
        null_value: float = -999.25,
    ) -> "CoreMechanical":
        """
        Load mechanical properties from a LAS file.

        Depth is read via ``lasio``'s ``las.index`` property (always the
        first column, regardless of mnemonic name).

        Parameters
        ----------
        filepath : str or Path
        ucs_mnemonic : str
            Mnemonic for the primary UCS curve.
        vp_mnemonic : str, optional
            Mnemonic for P-wave velocity (m/s).
        vs_mnemonic : str, optional
            Mnemonic for S-wave velocity (m/s).
        poisson_mnemonic : str, optional
            Mnemonic for dynamic Poisson's ratio.
        extra_ucs_mnemonics : dict, optional
            Additional UCS curves keyed by a label, e.g.
            {"5cm": "UCS_5CM", "30cm": "UCS_30CM"}.
        null_value : float
            LAS null value to replace with NaN.
        """
        if not HAS_LASIO:
            raise ImportError("lasio is required: pip install lasio")

        las = lasio.read(str(filepath))

        def _get(mn):
            if mn is None or mn not in las.keys():
                return None
            arr = np.array(las[mn], dtype=float)
            arr[arr == null_value] = np.nan
            return arr

        depth = np.array(las.index, dtype=float)
        ucs = _get(ucs_mnemonic)
        if ucs is None:
            raise ValueError(f"UCS mnemonic '{ucs_mnemonic}' not found in {filepath}")

        extra: Dict[str, np.ndarray] = {}
        if extra_ucs_mnemonics:
            for label, mn in extra_ucs_mnemonics.items():
                arr = _get(mn)
                if arr is not None:
                    extra[label] = arr

        meta = {
            "well": las.well.WELL.value if hasattr(las.well, "WELL") else "",
            "source": str(filepath),
        }

        return cls(
            depth=depth,
            ucs=ucs,
            vp=_get(vp_mnemonic),
            vs=_get(vs_mnemonic),
            poisson=_get(poisson_mnemonic),
            extra_scales=extra,
            metadata=meta,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: "pd.DataFrame",
        depth_col: str = "DEPTH",
        ucs_col: str = "UCS",
        vp_col: Optional[str] = None,
        vs_col: Optional[str] = None,
        poisson_col: Optional[str] = None,
        extra_ucs_cols: Optional[Dict[str, str]] = None,
    ) -> "CoreMechanical":
        """
        Create CoreMechanical from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
        depth_col : str
        ucs_col : str
        vp_col, vs_col, poisson_col : str, optional
        extra_ucs_cols : dict, optional
            Additional UCS columns, e.g. {"5cm": "UCS_5CM"}.
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required")

        def _col(name):
            if name and name in df.columns:
                return df[name].values.astype(float)
            return None

        depth = df[depth_col].values.astype(float)
        ucs = _col(ucs_col)
        if ucs is None:
            raise ValueError(f"UCS column '{ucs_col}' not found")

        extra: Dict[str, np.ndarray] = {}
        if extra_ucs_cols:
            for label, col in extra_ucs_cols.items():
                arr = _col(col)
                if arr is not None:
                    extra[label] = arr

        return cls(
            depth=depth,
            ucs=ucs,
            vp=_col(vp_col),
            vs=_col(vs_col),
            poisson=_col(poisson_col),
            extra_scales=extra,
        )

    # ---- Computed properties ----

    def vp_vs_ratio(self) -> np.ndarray:
        """Vp/Vs ratio — sensitive to fluid content and lithology."""
        if self.vp is None or self.vs is None:
            raise ValueError("Both Vp and Vs are required")
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.vs > 0, self.vp / self.vs, np.nan)

    def dynamic_moduli(
        self,
        density_gcm3: Union[float, np.ndarray] = 2.65,
    ) -> Dict[str, np.ndarray]:
        """
        Compute dynamic elastic moduli from Vp, Vs, and bulk density.

        Parameters
        ----------
        density_gcm3 : float or np.ndarray
            Bulk density in g/cm³. Scalar or depth array.
            Use RHOB log interpolated to core depths for best results.

        Returns
        -------
        dict with keys:
            poisson_ratio   Poisson's ratio (dimensionless)
            shear_modulus   Shear modulus G (GPa)
            young_modulus   Young's modulus E (GPa)
            bulk_modulus    Bulk modulus K (GPa)

        Notes
        -----
        Equations (SI units, density in kg/m³, velocities in m/s):

        .. math::
            \\nu = \\frac{V_P^2 - 2V_S^2}{2(V_P^2 - V_S^2)}

            G = \\rho V_S^2

            E = 2G(1 + \\nu)

            K = \\rho \\left(V_P^2 - \\frac{4}{3}V_S^2\\right)
        """
        if self.vp is None or self.vs is None:
            raise ValueError("Both Vp and Vs are required for dynamic moduli")

        rho = np.asarray(density_gcm3, dtype=float) * 1000.0  # kg/m³
        vp = self.vp
        vs = self.vs

        vp2 = vp ** 2
        vs2 = vs ** 2

        with np.errstate(divide="ignore", invalid="ignore"):
            nu = (vp2 - 2 * vs2) / (2 * (vp2 - vs2))

        g = rho * vs2 / 1e9         # GPa
        k = rho * (vp2 - 4/3 * vs2) / 1e9  # GPa
        e = 2 * g * (1 + nu)        # GPa

        return {
            "poisson_ratio": nu,
            "shear_modulus": g,
            "young_modulus": e,
            "bulk_modulus":  k,
        }

    def computed_poisson(self) -> np.ndarray:
        """
        Compute Poisson's ratio from Vp and Vs (overrides measured value).

        Use when the LAS file does not contain a measured Poisson's ratio
        or when you want to verify it.
        """
        return self.dynamic_moduli()["poisson_ratio"]

    def scale_average(
        self,
        window_m: float,
        step_m: Optional[float] = None,
    ) -> "CoreMechanical":
        """
        Return a version of this object smoothed over a depth window.

        Parameters
        ----------
        window_m : float
            Averaging window in depth units, metres for EpsLog data
            (e.g. 0.30 for 30 cm). Each output averages the samples within
            the window centred on its depth, so samples across a depth gap
            are not mixed in.
        step_m : float, optional
            Output sample spacing in depth units.
            Defaults to the input depths.

        Returns
        -------
        CoreMechanical
            New instance with smoothed arrays at the same or coarser spacing.
        """
        if self.n_samples < 2:
            return self

        diffs = np.abs(np.diff(np.asarray(self.depth, dtype=float)))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if diffs.size == 0:
            return self
        dz = float(np.median(diffs))
        half = int(round(window_m / dz / 2))
        # Half-window in depth: the same samples as +/- half steps on
        # regularly sampled data
        half_depth = (half + 1e-6) * dz

        depth = np.asarray(self.depth, dtype=float)
        if step_m is not None and step_m > 0:
            finite_depth = depth[np.isfinite(depth)]
            top, bottom = float(finite_depth.min()), float(finite_depth.max())
            n_out = int(np.floor((bottom - top) / step_m + 1e-9)) + 1
            out_depth = top + step_m * np.arange(n_out)
        else:
            out_depth = depth.copy()

        def _smooth(arr):
            if arr is None:
                return None
            arr = np.asarray(arr, dtype=float)
            out = np.full(out_depth.shape, np.nan)
            for i, z in enumerate(out_depth):
                in_window = np.abs(depth - z) <= half_depth
                finite = arr[in_window][np.isfinite(arr[in_window])]
                out[i] = float(np.mean(finite)) if finite.size else np.nan
            return out

        extra = {k: _smooth(v) for k, v in self.extra_scales.items()}

        return CoreMechanical(
            depth=out_depth,
            ucs=_smooth(self.ucs),
            vp=_smooth(self.vp),
            vs=_smooth(self.vs),
            poisson=_smooth(self.poisson),
            extra_scales=extra,
            metadata=dict(self.metadata),
        )

    # ---- Utilities ----

    def filter_by_depth(self, top: float, bottom: float) -> "CoreMechanical":
        """Return a new CoreMechanical filtered to [top, bottom]."""
        mask = (self.depth >= top) & (self.depth <= bottom)
        return CoreMechanical(
            depth=self.depth[mask],
            ucs=self.ucs[mask],
            vp=self.vp[mask] if self.vp is not None else None,
            vs=self.vs[mask] if self.vs is not None else None,
            poisson=self.poisson[mask] if self.poisson is not None else None,
            extra_scales={k: v[mask] for k, v in self.extra_scales.items()},
            metadata=dict(self.metadata),
        )

    def to_dataframe(self) -> "pd.DataFrame":
        """Export to a pandas DataFrame."""
        if not HAS_PANDAS:
            raise ImportError("pandas is required")
        data: Dict[str, np.ndarray] = {"DEPTH": self.depth, "UCS": self.ucs}
        if self.vp is not None:
            data["VP"] = self.vp
        if self.vs is not None:
            data["VS"] = self.vs
        if self.poisson is not None:
            data["POISSON"] = self.poisson
        for label, arr in self.extra_scales.items():
            data[f"UCS_{label.upper()}"] = arr
        return pd.DataFrame(data)

    def __repr__(self) -> str:
        has = ["ucs"]
        if self.vp is not None:
            has.append("vp")
        if self.vs is not None:
            has.append("vs")
        if self.poisson is not None:
            has.append("poisson")
        if self.extra_scales:
            has.append(f"{len(self.extra_scales)} extra scales")
        depth_text = (
            f"{self.top:.1f}-{self.bottom:.1f}"
            if self.top is not None and self.bottom is not None
            else "None-None"
        )
        return (f"CoreMechanical(samples={self.n_samples}, "
                f"depth={depth_text}, "
                f"data=[{', '.join(has)}])")
