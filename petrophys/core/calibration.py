"""
Core-to-log calibration utilities.

Provides functions for:
- Porosity calibration (core vs log)
- Permeability transform fitting (poro-perm from core)
- Grain density correction for density porosity
- Archie parameter estimation from core Sw
- Statistical comparison (R², bias, RMSE)
"""

from typing import Optional, Tuple, Dict, Union
from dataclasses import dataclass, field
import numpy as np

from petrophys.core.core_data import CoreData


@dataclass
class CalibrationResult:
    """Result of a core-vs-log calibration."""
    property_name: str
    n_points: int
    r_squared: float
    bias: float          # mean(log - core)
    rmse: float
    slope: float         # linear fit slope
    intercept: float     # linear fit intercept
    core_values: np.ndarray
    log_values: np.ndarray


@dataclass
class PoroPermTransform:
    """
    Poro-perm transform derived from core data.

    Model: log10(K) = A * porosity + B
    i.e.   K = 10^B * 10^(A * porosity)
    """
    A: float             # slope in log10(K) vs phi space
    B: float             # intercept
    r_squared: float
    n_points: int
    method: str = "core_fit"

    def predict(self, porosity: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Predict permeability from porosity using the fitted transform."""
        return 10.0 ** (self.A * np.asarray(porosity) + self.B)

    def __repr__(self) -> str:
        return (f"PoroPermTransform(log10(K) = {self.A:.2f}*PHI + ({self.B:.2f}), "
                f"R²={self.r_squared:.3f}, n={self.n_points})")


def interpolate_log_at_core(
    well,
    core: CoreData,
    curve_name: str,
) -> np.ndarray:
    """
    Interpolate a well log curve at core plug depths.

    Parameters
    ----------
    well : Well
        Well object with log curves.
    core : CoreData
        Core data with plug depths.
    curve_name : str
        Name of the log curve to interpolate.

    Returns
    -------
    np.ndarray
        Log values interpolated at core depths.
    """
    curve = well.get(curve_name)
    if curve is None:
        raise ValueError(f"Curve '{curve_name}' not found on well")
    return np.interp(core.depth, well.depth, curve.data)


def calibrate_porosity(
    well,
    core: CoreData,
    log_curve: str = "PHIE",
) -> CalibrationResult:
    """
    Compare core porosity against a log porosity curve.

    Parameters
    ----------
    well : Well
        Well object with computed porosity.
    core : CoreData
        Core data with porosity measurements.
    log_curve : str
        Name of the log porosity curve (default ``"PHIE"``).

    Returns
    -------
    CalibrationResult
        Statistics and data for the comparison.
    """
    if core.porosity is None:
        raise ValueError("CoreData has no porosity measurements")

    log_vals = interpolate_log_at_core(well, core, log_curve)
    core_vals = core.porosity

    # Filter valid points
    valid = ~np.isnan(core_vals) & ~np.isnan(log_vals) & (core_vals > 0)
    core_v = core_vals[valid]
    log_v = log_vals[valid]

    return _compute_calibration(f"Porosity ({log_curve})", core_v, log_v)


def calibrate_permeability(
    well,
    core: CoreData,
    log_curve: str = "PERM",
    log_scale: bool = True,
) -> CalibrationResult:
    """
    Compare core permeability against log-derived permeability.

    Parameters
    ----------
    well : Well
        Well object with computed permeability.
    core : CoreData
        Core data with permeability measurements.
    log_curve : str
        Name of the log permeability curve.
    log_scale : bool
        If True, compute statistics in log10 space.

    Returns
    -------
    CalibrationResult
    """
    if core.permeability is None:
        raise ValueError("CoreData has no permeability measurements")

    log_vals = interpolate_log_at_core(well, core, log_curve)
    core_vals = core.permeability

    valid = (~np.isnan(core_vals) & ~np.isnan(log_vals)
             & (core_vals > 0) & (log_vals > 0))
    core_v = core_vals[valid]
    log_v = log_vals[valid]

    if log_scale:
        core_v = np.log10(core_v)
        log_v = np.log10(log_v)

    return _compute_calibration(
        f"Permeability ({log_curve}, {'log10' if log_scale else 'linear'})",
        core_v, log_v,
    )


def fit_poro_perm_transform(
    core: CoreData,
    min_perm: float = 0.001,
) -> PoroPermTransform:
    """
    Fit a porosity-permeability transform from core data.

    Fits ``log10(K) = A * porosity + B`` using least-squares.

    Parameters
    ----------
    core : CoreData
        Core data with porosity and permeability.
    min_perm : float
        Minimum permeability to include (mD).

    Returns
    -------
    PoroPermTransform
        Fitted transform that can predict permeability from porosity.
    """
    if core.porosity is None or core.permeability is None:
        raise ValueError("CoreData needs both porosity and permeability")

    valid = (~np.isnan(core.porosity) & ~np.isnan(core.permeability)
             & (core.porosity > 0) & (core.permeability > min_perm))

    phi = core.porosity[valid]
    log_k = np.log10(core.permeability[valid])

    coeffs = np.polyfit(phi, log_k, 1)
    A, B = float(coeffs[0]), float(coeffs[1])

    # R²
    predicted = A * phi + B
    ss_res = np.sum((log_k - predicted) ** 2)
    ss_tot = np.sum((log_k - np.mean(log_k)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return PoroPermTransform(
        A=A, B=B, r_squared=r2, n_points=int(np.sum(valid)),
    )


def apply_poro_perm_transform(
    well,
    transform: PoroPermTransform,
    porosity_curve: str = "PHIE",
    output_curve: str = "PERM_CORE",
) -> np.ndarray:
    """
    Apply a core-derived poro-perm transform to a well.

    Parameters
    ----------
    well : Well
        Well object with a porosity curve.
    transform : PoroPermTransform
        Fitted transform from ``fit_poro_perm_transform``.
    porosity_curve : str
        Name of the porosity curve to use as input.
    output_curve : str
        Name for the output permeability curve added to the well.

    Returns
    -------
    np.ndarray
        Predicted permeability array.
    """
    phie = well.get(porosity_curve)
    if phie is None:
        raise ValueError(f"Curve '{porosity_curve}' not found")

    perm = transform.predict(phie.data)
    perm = np.clip(perm, 0.001, 100000)

    well.add_curve(output_curve, perm, "mD",
                   f"Permeability (core transform: log10K={transform.A:.1f}*PHI+{transform.B:.1f})")

    return perm


def correct_porosity_with_grain_density(
    well,
    core: CoreData,
    rhob_curve: str = "RHOB",
    rho_fluid: float = 1.0,
    default_rho_ma: float = 2.65,
    output_curve: str = "PHID_CORE",
) -> np.ndarray:
    """
    Recalculate density porosity using core grain density.

    Interpolates core grain density to log depths within the cored interval
    and uses the assumed ``default_rho_ma`` outside it.

    Parameters
    ----------
    well : Well
        Well object with RHOB.
    core : CoreData
        Core data with grain density.
    rhob_curve : str
        Name of the bulk density curve.
    rho_fluid : float
        Fluid density (g/cm3).
    default_rho_ma : float
        Assumed matrix density outside the cored interval.
    output_curve : str
        Name for the output porosity curve.

    Returns
    -------
    np.ndarray
        Corrected density porosity array.
    """
    if core.grain_density is None:
        raise ValueError("CoreData has no grain density measurements")

    rhob = well.get(rhob_curve)
    if rhob is None:
        raise ValueError(f"Curve '{rhob_curve}' not found")

    depth = well.depth

    # Build per-depth grain density: interpolate within cored interval
    rho_ma = np.full_like(depth, default_rho_ma)
    core_mask = (depth >= core.top) & (depth <= core.bottom)
    if np.any(core_mask):
        rho_ma[core_mask] = np.interp(
            depth[core_mask], core.depth, core.grain_density,
        )

    # PHID = (rho_ma - RHOB) / (rho_ma - rho_fluid)
    phid = (rho_ma - rhob.data) / (rho_ma - rho_fluid)
    phid = np.clip(phid, 0, 0.5)

    well.add_curve(output_curve, phid, "v/v",
                   "Density Porosity (core grain density corrected)")

    return phid


def estimate_archie_m(
    well,
    core: CoreData,
    rw: float,
    rt_curve: str = "RDEP",
    a: float = 1.0,
    n: float = 2.0,
    min_porosity: float = 0.05,
    min_sw: float = 0.1,
) -> Dict[str, float]:
    """
    Back-calculate Archie cementation exponent (m) from core Sw.

    From Archie: Rt = a * Rw / (phi^m * Sw^n)
    Rearranging: m = log(a * Rw / (Rt * Sw^n)) / log(phi)

    Parameters
    ----------
    well : Well
        Well object with resistivity.
    core : CoreData
        Core data with porosity and Sw.
    rw : float
        Formation water resistivity at formation temperature.
    rt_curve : str
        Name of the true resistivity curve.
    a : float
        Tortuosity factor.
    n : float
        Saturation exponent.
    min_porosity : float
        Minimum porosity to include (avoids log(0)).
    min_sw : float
        Minimum Sw to include.

    Returns
    -------
    dict
        Keys: ``m_mean``, ``m_median``, ``m_std``, ``n_points``, ``m_values``.
    """
    if core.porosity is None or core.sw is None:
        raise ValueError("CoreData needs both porosity and Sw for Archie m estimation")

    rt_log = interpolate_log_at_core(well, core, rt_curve)

    valid = (~np.isnan(core.porosity) & ~np.isnan(core.sw)
             & ~np.isnan(rt_log)
             & (core.porosity > min_porosity)
             & (core.sw > min_sw)
             & (rt_log > 0))

    phi = core.porosity[valid]
    sw = core.sw[valid]
    rt = rt_log[valid]

    # m = log10(a * Rw / (Rt * Sw^n)) / log10(phi)
    numerator = np.log10(a * rw / (rt * sw ** n))
    denominator = np.log10(phi)
    m_values = numerator / denominator

    # Filter outliers
    m_clean = m_values[(m_values > 0.5) & (m_values < 5.0)]

    return {
        "m_mean": float(np.mean(m_clean)) if len(m_clean) > 0 else np.nan,
        "m_median": float(np.median(m_clean)) if len(m_clean) > 0 else np.nan,
        "m_std": float(np.std(m_clean)) if len(m_clean) > 0 else np.nan,
        "n_points": int(len(m_clean)),
        "m_values": m_clean,
    }


# ---- Internal helpers ----

def _compute_calibration(
    name: str,
    core_vals: np.ndarray,
    log_vals: np.ndarray,
) -> CalibrationResult:
    """Compute calibration statistics between two matched arrays."""
    n = len(core_vals)
    if n < 2:
        return CalibrationResult(
            property_name=name, n_points=n,
            r_squared=np.nan, bias=np.nan, rmse=np.nan,
            slope=np.nan, intercept=np.nan,
            core_values=core_vals, log_values=log_vals,
        )

    # R²
    corr = np.corrcoef(core_vals, log_vals)[0, 1]
    r2 = corr ** 2

    # Bias and RMSE
    diff = log_vals - core_vals
    bias = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # Linear fit: log = slope * core + intercept
    coeffs = np.polyfit(core_vals, log_vals, 1)

    return CalibrationResult(
        property_name=name,
        n_points=n,
        r_squared=r2,
        bias=bias,
        rmse=rmse,
        slope=float(coeffs[0]),
        intercept=float(coeffs[1]),
        core_values=core_vals,
        log_values=log_vals,
    )
