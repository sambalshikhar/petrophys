"""
Core-to-log calibration utilities.

Provides functions for:
- Porosity calibration (core vs log)
- Permeability transform fitting (poro-perm from core)
- Grain density correction for density porosity
- Archie parameter estimation from core Sw
- Statistical comparison (R², bias, RMSE)
"""

from typing import Optional, Tuple, Dict, Union, Iterable
from dataclasses import dataclass, field
import numpy as np

from petrophys.model.core import CoreData
from petrophys._internal.analysis_helpers import get_curve


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
class PorosityCorrection:
    """
    Linear porosity correction derived from core porosity.

    Model: PHI_core_equivalent = A * PHI_log + B
    """
    A: float
    B: float
    r_squared: float
    n_points: int
    rmse: float = np.nan
    input_curve: str = "PHIE"
    method: str = "core_linear"

    def predict(self, porosity: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Predict core-equivalent porosity from a log porosity curve."""
        corrected = self.A * np.asarray(porosity, dtype=float) + self.B
        return np.clip(corrected, 0.0, 0.5)

    def __repr__(self) -> str:
        return (f"PorosityCorrection(PHIcore = {self.A:.3f}*{self.input_curve} + "
                f"{self.B:.3f}, R^2={self.r_squared:.3f}, RMSE={self.rmse:.4f}, n={self.n_points})")


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
    rmse: float = np.nan
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
    if well.depth is None:
        raise ValueError("Well has no depth data")
    depth = np.asarray(well.depth, dtype=float).reshape(-1)
    values = np.asarray(curve.data, dtype=float).reshape(-1)
    if depth.size != values.size:
        raise ValueError(
            f"Curve '{curve_name}' length ({values.size}) does not match "
            f"well depth length ({depth.size})"
        )
    return _interp_curve_at_depth(depth, values, core.depth)


def core_data_from_well(
    well,
    porosity_curves: Iterable[str] = ("PHIC", "CPOR", "CPOR-AMB", "CPOR_AMB"),
    perm_curves: Iterable[str] = (
        "KH_CORE",
        "CPERM",
        "K_CORE",
        "PERM_CORE",
        "CPERM-AMB-KL",
        "CPERM_AMB_KL",
        "CPERM-AMB-AIR",
        "CPERM_AMB_AIR",
        "CPERM-AB-AIR",
    ),
    gd_curves: Iterable[str] = ("GDEN", "CGD"),
    sw_curves: Iterable[str] = ("SWC", "CSW", "CSW-AMB", "CSW_AMB"),
    so_curves: Iterable[str] = ("SOC", "CSO", "CSO-AMB", "CSO_AMB"),
    lithology_curves: Iterable[str] = ("LITHOLOGY_GEOLINK",),
    top: Optional[float] = None,
    bottom: Optional[float] = None,
) -> CoreData:
    """
    Build ``CoreData`` directly from well curves.

    This is intended for LAS files that already carry laboratory core
    measurements such as ``PHIC``, ``CPOR-AMB``, ``KH_CORE``,
    ``CPERM-AMB-KL``, ``GDEN``, ``SWC``, and ``CSW-AMB``. The returned object
    keeps only depths where at least one requested core property exists.
    """
    if well.depth is None:
        raise ValueError("Well has no depth data")

    def _first_curve(names: Iterable[str]) -> tuple[Optional[np.ndarray], Optional[str]]:
        curve = get_curve(well, list(names))
        if curve is not None:
            return np.asarray(curve.data, dtype=float), curve.mnemonic
        return None, None

    depth = np.asarray(well.depth, dtype=float)
    porosity, porosity_source = _first_curve(porosity_curves)
    permeability, permeability_source = _first_curve(perm_curves)
    grain_density, grain_density_source = _first_curve(gd_curves)
    sw, sw_source = _first_curve(sw_curves)
    so, so_source = _first_curve(so_curves)

    lithology = None
    for name in lithology_curves:
        curve = well.get(name)
        if curve is not None:
            lithology = np.asarray(curve.data)
            break

    has_property = np.zeros_like(depth, dtype=bool)
    for array in (porosity, permeability, grain_density, sw, so):
        if array is not None:
            has_property |= np.isfinite(array)

    if top is not None:
        has_property &= depth >= float(top)
    if bottom is not None:
        has_property &= depth <= float(bottom)

    if not np.any(has_property):
        raise ValueError("No core curves found on well for the requested interval")

    def _slice(array: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if array is None:
            return None
        return array[has_property]

    metadata = {
        "well_name": well.name,
        "field": well.header.field,
        "core_porosity_curve": porosity_source,
        "core_permeability_curve": permeability_source,
        "core_grain_density_curve": grain_density_source,
        "core_sw_curve": sw_source,
        "core_so_curve": so_source,
    }

    return CoreData(
        depth=depth[has_property],
        porosity=_slice(porosity),
        permeability=_slice(permeability),
        grain_density=_slice(grain_density),
        sw=_slice(sw),
        so=_slice(so),
        lithology=_slice(lithology),
        metadata=metadata,
    )


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


def fit_porosity_correction(
    well,
    core: CoreData,
    log_curve: str = "PHIE",
    min_porosity: float = 0.01,
) -> PorosityCorrection:
    """
    Fit a linear correction from log porosity to core-equivalent porosity.

    Fits ``PHI_core = A * PHI_log + B`` using core porosity as the target.
    This is the appropriate direction for applying a correction back onto the
    full log.
    """
    if core.porosity is None:
        raise ValueError("CoreData has no porosity measurements")

    log_vals = interpolate_log_at_core(well, core, log_curve)
    core_vals = core.porosity

    valid = (~np.isnan(core_vals) & ~np.isnan(log_vals)
             & (core_vals > min_porosity) & (log_vals > min_porosity))
    if np.count_nonzero(valid) < 2:
        raise ValueError("At least two valid core/log porosity pairs are required")

    phi_log = log_vals[valid]
    phi_core = core_vals[valid]
    coeffs = np.polyfit(phi_log, phi_core, 1)
    A, B = float(coeffs[0]), float(coeffs[1])

    predicted = A * phi_log + B
    ss_res = np.sum((phi_core - predicted) ** 2)
    ss_tot = np.sum((phi_core - np.mean(phi_core)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean((phi_core - predicted) ** 2)))

    return PorosityCorrection(
        A=A,
        B=B,
        r_squared=float(r2),
        n_points=int(np.count_nonzero(valid)),
        rmse=rmse,
        input_curve=log_curve,
    )


def apply_porosity_correction(
    well,
    correction: PorosityCorrection,
    input_curve: Optional[str] = None,
    output_curve: str = "PHIE_CC",
) -> np.ndarray:
    """
    Apply a porosity correction to a full well log.

    Parameters
    ----------
    well : Well
        Well object with a porosity curve.
    correction : PorosityCorrection
        Fitted correction from ``fit_porosity_correction``.
    input_curve : str, optional
        Input porosity curve. Defaults to the curve used during fitting.
    output_curve : str
        Name for the corrected output curve.
    """
    curve_name = input_curve or correction.input_curve
    porosity_curve = well.get(curve_name)
    if porosity_curve is None:
        raise ValueError(f"Curve '{curve_name}' not found")

    corrected = correction.predict(porosity_curve.data)
    well.add_curve(
        output_curve,
        corrected,
        "v/v",
        f"Porosity corrected to core ({correction.A:.3f}*{curve_name}+{correction.B:.3f})",
    )
    return corrected


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


def calibrate_saturation(
    well,
    core: CoreData,
    log_curve: str = "SW",
    core_unit: Optional[str] = None,
) -> CalibrationResult:
    """
    Compare core water saturation against a log-derived saturation curve.

    This diagnostic does not replace the log saturation, alter workflow
    routing, or treat core saturation as the preferred continuous
    interpretation. Fraction and percentage inputs are normalized to
    fractions before comparison.

    Parameters
    ----------
    well : Well
        Well object containing the log-derived saturation curve.
    core : CoreData
        Core data with water-saturation measurements.
    log_curve : str
        Name of the log water-saturation curve.
    core_unit : str, optional
        Unit for ``core.sw``. Use ``"%"`` for percentage data. When omitted,
        consistently percentage-like values are detected conservatively.

    Returns
    -------
    CalibrationResult
        Statistics and matched fraction-valued core/log saturation arrays.
    """
    if core.sw is None:
        raise ValueError("CoreData has no water-saturation measurements")

    curve = well.get(log_curve)
    if curve is None:
        raise ValueError(f"Curve '{log_curve}' not found on well")

    log_vals = interpolate_log_at_core(well, core, log_curve)
    core_vals = _saturation_to_fraction(core.sw, unit=core_unit)
    log_vals = _saturation_to_fraction(log_vals, unit=curve.unit)

    valid = (
        np.isfinite(core_vals)
        & np.isfinite(log_vals)
        & (core_vals >= 0.0)
        & (core_vals <= 1.0)
        & (log_vals >= 0.0)
        & (log_vals <= 1.0)
    )
    return _compute_calibration(
        f"Water saturation ({log_curve})",
        core_vals[valid],
        log_vals[valid],
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
    if np.count_nonzero(valid) < 2:
        raise ValueError("At least two valid core porosity/permeability pairs are required")

    phi = core.porosity[valid]
    log_k = np.log10(core.permeability[valid])

    coeffs = np.polyfit(phi, log_k, 1)
    A, B = float(coeffs[0]), float(coeffs[1])

    # R²
    predicted = A * phi + B
    ss_res = np.sum((log_k - predicted) ** 2)
    ss_tot = np.sum((log_k - np.mean(log_k)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean((log_k - predicted) ** 2)))

    return PoroPermTransform(
        A=A, B=B, r_squared=r2, n_points=int(np.sum(valid)), rmse=rmse,
    )


def fit_log_poro_perm_transform(
    well,
    core: CoreData,
    porosity_curve: str = "PHIE_CC",
    min_perm: float = 0.001,
) -> PoroPermTransform:
    """
    Fit a poro-perm transform using log porosity corrected to core.

    Fits ``log10(K_core) = A * PHI_log_corrected + B`` at core depths, so the
    resulting transform can be applied directly to the corrected log porosity.
    """
    if core.permeability is None:
        raise ValueError("CoreData has no permeability measurements")

    phi_log = interpolate_log_at_core(well, core, porosity_curve)
    perm_core = core.permeability

    valid = (~np.isnan(phi_log) & ~np.isnan(perm_core)
             & (phi_log > 0) & (perm_core > min_perm))
    if np.count_nonzero(valid) < 2:
        raise ValueError("At least two valid porosity/permeability pairs are required")

    phi = phi_log[valid]
    log_k = np.log10(perm_core[valid])

    coeffs = np.polyfit(phi, log_k, 1)
    A, B = float(coeffs[0]), float(coeffs[1])

    predicted = A * phi + B
    ss_res = np.sum((log_k - predicted) ** 2)
    ss_tot = np.sum((log_k - np.mean(log_k)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean((log_k - predicted) ** 2)))

    return PoroPermTransform(
        A=A,
        B=B,
        r_squared=float(r2),
        n_points=int(np.count_nonzero(valid)),
        rmse=rmse,
        method=f"log_fit:{porosity_curve}",
    )


def fit_log_poro_perm_transform_from_correction(
    well,
    core: CoreData,
    correction: PorosityCorrection,
    porosity_curve: str = "PHIE",
    min_perm: float = 0.001,
) -> PoroPermTransform:
    """
    Fit a poro-perm transform from a log porosity curve after core correction.

    This avoids having to add an intermediate corrected porosity curve to the
    source well just to train the transform.
    """
    if core.permeability is None:
        raise ValueError("CoreData has no permeability measurements")

    phi_log = interpolate_log_at_core(well, core, porosity_curve)
    phi_corrected = correction.predict(phi_log)
    perm_core = core.permeability

    valid = (~np.isnan(phi_corrected) & ~np.isnan(perm_core)
             & (phi_corrected > 0) & (perm_core > min_perm))
    if np.count_nonzero(valid) < 2:
        raise ValueError("At least two valid corrected-porosity/permeability pairs are required")

    phi = phi_corrected[valid]
    log_k = np.log10(perm_core[valid])

    coeffs = np.polyfit(phi, log_k, 1)
    A, B = float(coeffs[0]), float(coeffs[1])

    predicted = A * phi + B
    ss_res = np.sum((log_k - predicted) ** 2)
    ss_tot = np.sum((log_k - np.mean(log_k)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean((log_k - predicted) ** 2)))

    return PoroPermTransform(
        A=A,
        B=B,
        r_squared=float(r2),
        n_points=int(np.count_nonzero(valid)),
        rmse=rmse,
        method=f"log_fit:{porosity_curve}+{correction.method}",
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

    if well.depth is None:
        raise ValueError("Well has no depth data")
    depth = np.asarray(well.depth, dtype=float).reshape(-1)
    core_top = core.top
    core_bottom = core.bottom
    if core_top is None or core_bottom is None:
        raise ValueError("CoreData has no finite depth interval")

    # Build per-depth grain density: interpolate within cored interval
    rho_ma = np.full_like(depth, default_rho_ma)
    core_mask = (depth >= core_top) & (depth <= core_bottom)
    if np.any(core_mask):
        rho_ma[core_mask] = _interp_curve_at_depth(
            core.depth, core.grain_density, depth[core_mask],
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


def _interp_curve_at_depth(
    source_depth: np.ndarray,
    source_values: np.ndarray,
    target_depth: np.ndarray,
) -> np.ndarray:
    """Interpolate numeric values after sorting and de-duplicating depth support."""
    source_depth = np.asarray(source_depth, dtype=float).reshape(-1)
    source_values = np.asarray(source_values, dtype=float).reshape(-1)
    targets = np.asarray(target_depth, dtype=float).reshape(-1)
    if source_depth.size != source_values.size:
        raise ValueError("source_depth and source_values must have the same length")

    out = np.full(targets.shape, np.nan, dtype=float)
    valid = np.isfinite(source_depth) & np.isfinite(source_values)
    if not np.any(valid):
        return out

    x = source_depth[valid]
    y = source_values[valid]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique, inverse = np.unique(x, return_inverse=True)
    sums = np.zeros(unique.size, dtype=float)
    counts = np.zeros(unique.size, dtype=float)
    np.add.at(sums, inverse, y)
    np.add.at(counts, inverse, 1.0)
    support = sums / counts

    finite_targets = np.isfinite(targets)
    if np.any(finite_targets):
        out[finite_targets] = np.interp(
            targets[finite_targets],
            unique,
            support,
            left=np.nan,
            right=np.nan,
        )
    return out


def _saturation_to_fraction(
    values: np.ndarray,
    unit: Optional[str] = None,
) -> np.ndarray:
    """Normalize one consistently scaled saturation array to fraction units."""
    saturation = np.asarray(values, dtype=float).copy()
    finite = saturation[np.isfinite(saturation)]
    normalized_unit = (unit or "").strip().lower().replace(" ", "")
    is_percent_unit = normalized_unit in {"%", "percent", "percentage", "pu"}
    is_fraction_unit = normalized_unit in {"v/v", "frac", "fraction", "decimal"}
    appears_percent = (
        not normalized_unit
        and finite.size > 0
        and np.nanmedian(np.abs(finite)) > 1.0
    )
    if is_percent_unit or (appears_percent and not is_fraction_unit):
        saturation /= 100.0
    return saturation


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
