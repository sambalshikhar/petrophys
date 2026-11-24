"""
Flag generation for net pay determination.
"""

from typing import Dict, Iterable, Optional, Union
import numpy as np

DEFAULT_HYDROCARBON_FLUIDS = frozenset(
    {
        "HC",
        "HYDROCARBON",
        "HYDROCARBON BEARING",
        "OIL",
        "LIGHT OIL",
        "HEAVY OIL",
        "GAS",
        "WET GAS",
        "DRY GAS",
        "CONDENSATE",
        "GAS CONDENSATE",
        "OIL GAS",
    }
)
DEFAULT_NON_PAY_FLUIDS = frozenset(
    {
        "WATER",
        "WATER BEARING",
        "BRINE",
        "AQUIFER",
        "NON PAY",
        "NONPAY",
        "NO HYDROCARBON",
    }
)


def net_flags(
    vcl: np.ndarray,
    phie: np.ndarray,
    sw: np.ndarray,
    vcl_cutoff: float = 0.3,
    phie_cutoff: Union[float, np.ndarray] = 0.08,
    sw_cutoff: float = 0.7,
) -> Dict[str, np.ndarray]:
    """
    Generate net rock, reservoir, and pay flags.

    Net rock is ``VCL < vcl_cutoff``; net reservoir adds
    ``PHIE > phie_cutoff`` and net pay adds ``Sw < sw_cutoff``. A sample
    exactly at a cutoff, or with a missing (NaN) value, fails that test.

    Parameters
    ----------
    vcl : np.ndarray
        Clay volume
    phie : np.ndarray
        Effective porosity
    sw : np.ndarray
        Water saturation
    vcl_cutoff : float
        VCL cutoff (default: 0.3)
    phie_cutoff : float or np.ndarray
        PHIE cutoff. Scalar applies uniformly; array allows per-depth
        cutoffs (e.g. 0.08 for clastics, 0.045 for carbonates).
    sw_cutoff : float
        Sw cutoff (default: 0.7)

    Returns
    -------
    dict
        Dictionary with flag arrays
    """
    # Net Rock: VCL < cutoff
    net_rock = vcl < vcl_cutoff

    # Net Reservoir: Net Rock + PHIE > cutoff
    # numpy broadcasts correctly for both scalar and array phie_cutoff
    net_reservoir = net_rock & (phie > phie_cutoff)

    # Net Pay: Net Reservoir + Sw < cutoff
    net_pay = net_reservoir & (sw < sw_cutoff)

    return {
        "net_rock": net_rock,
        "net_reservoir": net_reservoir,
        "net_pay": net_pay,
    }


def volumetric_pay_flag(
    fluid_type: Union[str, np.ndarray],
    reservoir_flag: Optional[Union[float, np.ndarray]] = None,
    hydrocarbon_fluids: Iterable[str] = DEFAULT_HYDROCARBON_FLUIDS,
    non_pay_fluids: Iterable[str] = DEFAULT_NON_PAY_FLUIDS,
) -> Union[float, np.ndarray]:
    """
    Generate a conservative pay flag from a volumetric main-fluid label.

    Recognized hydrocarbon labels map to 1 and recognized non-pay labels map
    to 0. Unknown or missing labels remain NaN. When ``reservoir_flag`` is
    supplied, samples outside reservoir are forced to 0 and samples with an
    unknown reservoir state remain NaN.

    The label collections are configurable because fluid nomenclature varies
    between interpretation systems and operators.
    """
    fluids = np.asarray(fluid_type, dtype=object)
    scalar_input = fluids.ndim == 0 and (
        reservoir_flag is None or np.asarray(reservoir_flag).ndim == 0
    )

    if reservoir_flag is not None:
        fluids, reservoir = np.broadcast_arrays(
            fluids,
            np.asarray(reservoir_flag, dtype=float),
        )
    else:
        reservoir = None

    hydrocarbon = _normalize_fluid_labels(hydrocarbon_fluids)
    non_pay = _normalize_fluid_labels(non_pay_fluids)
    overlap = hydrocarbon & non_pay
    if overlap:
        labels = ", ".join(sorted(overlap))
        raise ValueError(f"Fluid labels cannot be both pay and non-pay: {labels}")

    result = np.full(fluids.shape, np.nan, dtype=float)
    for index in np.ndindex(fluids.shape):
        label = _normalize_fluid_label(fluids[index])
        if label in hydrocarbon:
            result[index] = 1.0
        elif label in non_pay:
            result[index] = 0.0

    if reservoir is not None:
        finite_reservoir = np.isfinite(reservoir)
        result[finite_reservoir & (reservoir <= 0.5)] = 0.0
        result[~finite_reservoir] = np.nan

    if scalar_input:
        return float(result.item())
    return result


def add_volumetric_pay_flag(
    well,
    fluid_type: Union[str, np.ndarray],
    reservoir_flag: Optional[Union[float, np.ndarray]] = None,
    output_curve: str = "PAY_FLAG_VOLUMETRIC",
    **kwargs,
) -> np.ndarray:
    """Generate and add ``PAY_FLAG_VOLUMETRIC`` to a well."""
    flag = np.asarray(
        volumetric_pay_flag(
            fluid_type,
            reservoir_flag=reservoir_flag,
            **kwargs,
        ),
        dtype=float,
    )
    if flag.ndim == 0:
        flag = np.full(well.n_samples, float(flag), dtype=float)
    if flag.shape != (well.n_samples,):
        raise ValueError(
            "Volumetric fluid labels must be scalar or match the well sample count"
        )
    well.add_curve(
        output_curve,
        flag,
        "",
        "Pay flag from volumetric main-fluid classification",
    )
    return flag


def _normalize_fluid_label(value: object) -> str:
    """Normalize one operator/vendor fluid label for exact matching."""
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return ""
    label = str(value).strip().upper()
    for separator in ("_", "-", "/", "&"):
        label = label.replace(separator, " ")
    return " ".join(label.split())


def _normalize_fluid_labels(values: Iterable[str]) -> set[str]:
    """Normalize a label collection while ignoring blank entries."""
    labels = {_normalize_fluid_label(value) for value in values}
    labels.discard("")
    return labels
