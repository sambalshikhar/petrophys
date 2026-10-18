"""
Moveable Hydrocarbon Index calculation module.
"""

from typing import TYPE_CHECKING, Optional, Union
import numpy as np

from petrophys._internal.analysis_helpers import get_curve_data_depthwise

if TYPE_CHECKING:
    from petrophys.model.well import Well


class MhiModule:
    """Moveable Hydrocarbon Index calculator."""

    @staticmethod
    def diagnostic_sxo_mask(
        well: "Well",
        sxo: Optional[np.ndarray] = None,
        sw: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Return rows where SXO has diagnostic invaded-zone support.

        SXO computed only from the quick-look fallback ``SW^(1/5)`` is not
        suitable for MHI, so MHI is only kept where shallow resistivity, RMF,
        PHIE/PHIT, SW, and SXO are all finite.
        """
        n_samples = well.n_samples
        if sxo is None:
            sxo, _, _ = get_curve_data_depthwise(well, ["SXO", "SWXOA", "SXO-LOG"])
        if sw is None:
            sw, _, _ = get_curve_data_depthwise(well, ["SW", "SWA", "SWT-LOG"])
        rxo, _, _ = get_curve_data_depthwise(well, ["RSHA", "RXO", "RESS"])
        rmf, _, _ = get_curve_data_depthwise(well, ["RMF"])
        phie, _, _ = get_curve_data_depthwise(well, ["PHIE", "PHIT"])

        if any(value is None for value in (sxo, sw, rxo, rmf, phie)):
            return np.zeros(n_samples, dtype=bool)

        sxo = np.asarray(sxo, dtype=float)
        sw = np.asarray(sw, dtype=float)
        rxo = np.asarray(rxo, dtype=float)
        rmf = np.asarray(rmf, dtype=float)
        phie = np.asarray(phie, dtype=float)
        return (
            np.isfinite(sxo)
            & np.isfinite(sw)
            & np.isfinite(rxo)
            & np.isfinite(rmf)
            & np.isfinite(phie)
            & (rxo > 0.01)
            & (rmf > 0.0001)
            & (phie > 0.001)
        )

    def calculate(
        self,
        well: "Well",
        sxo: Optional[np.ndarray] = None,
        sw: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Calculate MHI."""
        if sxo is None:
            sxo, _, _ = get_curve_data_depthwise(well, ["SXO", "SWXOA", "SXO-LOG"])
            if sxo is None:
                raise ValueError("SXO with finite samples required")

        if sw is None:
            sw, _, _ = get_curve_data_depthwise(well, ["SW", "SWA", "SWT-LOG"])
            if sw is None:
                raise ValueError("SW with finite samples required")

        mhi = np.asarray(calculate_mhi(sxo, sw), dtype=float)
        diagnostic = self.diagnostic_sxo_mask(well, sxo=sxo, sw=sw)
        mhi[~diagnostic] = np.nan
        if not np.any(np.isfinite(mhi)):
            raise ValueError("MHI skipped because SXO was only fifth-root fallback or lacked invaded-zone support")
        return mhi

    def add_to_well(self, well: "Well") -> "Well":
        """Add MHI curve to well."""
        mhi = self.calculate(well)
        if not np.any(np.isfinite(mhi)):
            return well
        well.add_curve("MHI", mhi, "", "Moveable Hydrocarbon Index")
        return well


def calculate_mhi(
    sxo: Union[float, np.ndarray],
    sw: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate Moveable Hydrocarbon Index.

    MHI = Sw / Sxo

    Parameters
    ----------
    sxo : float or np.ndarray
        Flushed zone water saturation
    sw : float or np.ndarray
        Uninvaded zone water saturation

    Returns
    -------
    float or np.ndarray
        Moveable hydrocarbon index; NaN where Sxo <= 0 (MHI undefined)
    """
    from petrophys._internal.benchmark import reproducing_benchmark

    if reproducing_benchmark():
        # The benchmark packages floored Sxo at 0.01.
        if isinstance(sxo, np.ndarray):
            sxo_safe = np.where(sxo <= 0, 0.01, sxo)
        else:
            sxo_safe = max(sxo, 0.01)
    elif isinstance(sxo, np.ndarray):
        with np.errstate(invalid="ignore"):
            sxo_safe = np.where(sxo > 0, sxo, np.nan)
    else:
        sxo_safe = sxo if sxo > 0 else np.nan

    mhi = sw / sxo_safe
    # Do NOT clip the upper bound: MHI = Sw/Sxo > 1 is diagnostic (Sxo < Sw means
    # no moveable hydrocarbon, or a shallow/no-invasion / bad-Rxo QC flag). Clipping
    # to 1 would collapse that entire interpretive band. Only floor at 0.
    return np.maximum(mhi, 0)
