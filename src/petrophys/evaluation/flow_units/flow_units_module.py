"""
Hydraulic Flow Units (Amaefule et al., 1993).

A hydraulic flow unit (HFU) groups rock with a similar Flow Zone Indicator
(FZI) so that, within a unit, permeability is a single-valued function of
porosity. On a log-log Reservoir Quality Index (RQI) vs normalized-porosity
(phi_z) plot each HFU plots as a unit-slope line whose intercept is the unit's
mean FZI.

    RQI   = 0.0314 * sqrt(k / phi_e)          [microns]
    phi_z = phi_e / (1 - phi_e)
    FZI   = RQI / phi_z                        [microns]
    k     = phi_e * (FZI * phi_z / 0.0314)^2   [mD]   (per-unit prediction)

This module determines the units from the FZI distribution (1-D k-means on
log10 FZI, or explicit cutoffs) and predicts permeability from each sample's
unit-mean FZI.

References
----------
Amaefule, J.O., Altunbay, M., Tiab, D., Kersey, D.G. and Keelan, D.K.
(1993). Enhanced reservoir description: using core and log data to identify
hydraulic (flow) units and predict permeability in uncored intervals/wells.
SPE 26436, SPE Annual Technical Conference and Exhibition.
"""
from dataclasses import dataclass, field
import warnings
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np

from petrophys._internal.advice import Advice
from petrophys.evaluation.rock_quality.rock_quality_module import (
    calculate_rqi,
    calculate_phi_z,
    calculate_fzi,
)
from petrophys._internal.analysis_helpers import get_curve_data_depthwise

if TYPE_CHECKING:
    from petrophys.model.well import Well
    from petrophys.model.core import CoreData

_RQI_CONST = 0.0314


@dataclass
class FlowUnitResult:
    """Result of a hydraulic-flow-unit determination."""

    hfu: np.ndarray              # integer unit label per sample (1..n_units, 0 = no data)
    n_units: int
    rqi: np.ndarray
    phi_z: np.ndarray
    fzi: np.ndarray
    perm_predicted: np.ndarray   # k from the sample's unit-mean FZI (mD)
    fzi_unit_mean: np.ndarray    # geometric-mean FZI per unit (length n_units)
    fzi_boundaries: np.ndarray   # FZI values separating adjacent units (length n_units-1)
    counts: np.ndarray = field(default=None)  # samples per unit
    depth: Optional[np.ndarray] = field(default=None, repr=False)  # sample depths, from core data

    def to_dataframe(self):
        """A row per sample: depth (when known), HFU, RQI, phi_z, FZI and the
        unit-predicted permeability."""
        import pandas as pd

        frame = pd.DataFrame({"HFU": self.hfu, "RQI": self.rqi, "PHI_Z": self.phi_z,
                              "FZI": self.fzi, "PERM_HFU": self.perm_predicted})
        if self.depth is not None:
            frame.insert(0, "DEPTH", self.depth)
        return frame

    def summary(self):
        """A row per flow unit: samples and its geometric-mean FZI."""
        import pandas as pd

        units = np.arange(1, int(self.n_units) + 1)
        counts = [int(np.count_nonzero(np.asarray(self.hfu) == unit)) for unit in units]
        return pd.DataFrame({"Samples": counts, "Mean FZI": np.asarray(self.fzi_unit_mean, dtype=float)},
                            index=pd.Index(units, name="HFU"))


def _kmeans_1d(x: np.ndarray, k: int, n_iter: int = 200) -> np.ndarray:
    """Deterministic 1-D k-means; returns cluster centers (sorted ascending)."""
    x = np.asarray(x, dtype=float)
    if k <= 1 or x.size == 0:
        return np.array([float(np.mean(x))]) if x.size else np.array([0.0])
    distinct = np.unique(x)
    if distinct.size <= k:
        # No more units than distinct values: one per value.
        return distinct
    centers = np.quantile(x, np.linspace(0.0, 1.0, k + 2)[1:-1])
    for _ in range(n_iter):
        labels = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
        new = centers.copy()
        for j in range(k):
            members = x[labels == j]
            if members.size:
                new[j] = members.mean()
        if np.allclose(new, centers):
            break
        centers = np.sort(new)
    return np.sort(centers)


def _within_ss(x: np.ndarray, centers: np.ndarray) -> float:
    labels = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
    return float(np.sum((x - centers[labels]) ** 2))


def _auto_n_units(
    log_fzi: np.ndarray,
    max_units: int,
    min_sep: float = 0.20,
    min_frac: float = 0.02,
) -> int:
    """Largest number of units whose FZI centers stay distinct and populated.

    Units are kept only while adjacent log10(FZI) centers differ by at least
    ``min_sep`` (a ~1.6x FZI factor) and every unit holds at least ``min_frac``
    of the samples. This avoids splitting a single mode into near-duplicate
    units, the failure mode of plain elbow / variance-ratio criteria on 1-D data.
    """
    n = log_fzi.size
    if n < 6:
        return 1
    upper = int(min(max_units, max(2, n // 2)))
    min_count = max(2, int(round(min_frac * n)))
    best = 1
    for k in range(2, upper + 1):
        centers = _kmeans_1d(log_fzi, k)
        labels = np.argmin(np.abs(log_fzi[:, None] - centers[None, :]), axis=1)
        counts = np.array([np.sum(labels == j) for j in range(k)])
        gaps = np.diff(np.sort(centers))
        if gaps.size and gaps.min() >= min_sep and counts.min() >= min_count:
            best = k
        else:
            break  # finer k only brings centers closer together
    return best


def assign_flow_units(
    fzi: Union[np.ndarray, Sequence[float]],
    n_units: Optional[int] = None,
    method: str = "kmeans",
    fzi_cutoffs: Optional[Sequence[float]] = None,
    max_units: int = 6,
):
    """Partition samples into hydraulic flow units from their FZI.

    Parameters
    ----------
    fzi : array
        Flow Zone Indicator per sample (microns).
    n_units : int, optional
        Number of units. If ``None`` (and no cutoffs) it is chosen automatically.
    method : {"kmeans", "cutoffs"}
        ``kmeans`` clusters log10(FZI); ``cutoffs`` uses ``fzi_cutoffs``.
    fzi_cutoffs : sequence of float, optional
        Ascending FZI boundaries (used when method="cutoffs" or when provided).
    max_units : int
        Cap on the number of units for automatic selection.

    Returns
    -------
    labels : numpy.ndarray
        Unit of each sample, 1..n_units; 0 where FZI is missing or not
        positive.
    unit_mean_fzi : numpy.ndarray
        Geometric mean FZI of each unit.
    boundaries : numpy.ndarray
        FZI boundaries between adjacent units: the sorted cutoffs, or the
        geometric midpoints between the k-means unit centres.
    """
    if method not in ("kmeans", "cutoffs"):
        raise ValueError(f"method must be 'kmeans' or 'cutoffs'; got {method!r}")
    fzi = np.asarray(fzi, dtype=float)
    labels = np.zeros(fzi.shape, dtype=int)
    valid = np.isfinite(fzi) & (fzi > 0.0)
    if not np.any(valid):
        return labels, np.asarray([]), np.asarray([])

    log_fzi = np.log10(fzi[valid])

    if fzi_cutoffs is not None or method == "cutoffs":
        if fzi_cutoffs is None:
            raise ValueError(Advice("method='cutoffs' requires fzi_cutoffs", ("fzi_cutoffs",)))
        bounds = np.sort(np.asarray(fzi_cutoffs, dtype=float))
        unit_idx = np.digitize(fzi[valid], bounds)  # 0..len(bounds)
        labels[valid] = unit_idx + 1
        n = int(bounds.size + 1)
        means = np.array([
            10.0 ** np.mean(log_fzi[unit_idx == j]) if np.any(unit_idx == j) else np.nan
            for j in range(n)
        ])
        return labels, means, bounds

    requested = n_units
    if n_units is None:
        n_units = _auto_n_units(log_fzi, max_units)
    n_units = int(max(1, n_units))

    centers = _kmeans_1d(log_fzi, n_units)
    if requested is not None and centers.size < n_units:
        warnings.warn(
            f"n_units={requested} but the {log_fzi.size} valid FZI samples hold only "
            f"{centers.size} distinct values: {centers.size} units assigned.",
            UserWarning, stacklevel=2,
        )
    n_units = centers.size
    unit_idx = np.argmin(np.abs(log_fzi[:, None] - centers[None, :]), axis=1)
    labels[valid] = unit_idx + 1
    unit_mean_fzi = 10.0 ** centers
    # Boundaries = geometric midpoints between adjacent unit centers.
    boundaries = 10.0 ** ((centers[:-1] + centers[1:]) / 2.0) if n_units > 1 else np.asarray([])
    return labels, unit_mean_fzi, boundaries


def predict_perm_from_fzi(
    fzi: Union[float, np.ndarray],
    phi_e: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Permeability (mD) from FZI and porosity via the RQI relationship."""
    fzi = np.asarray(fzi, dtype=float)
    phi_e = np.asarray(phi_e, dtype=float)
    out = np.full(np.broadcast_shapes(fzi.shape, phi_e.shape), np.nan)
    fzi_b, phi_b = np.broadcast_arrays(fzi, phi_e)
    ok = np.isfinite(fzi_b) & np.isfinite(phi_b) & (phi_b > 0.0) & (phi_b < 1.0)
    phi_z = phi_b[ok] / (1.0 - phi_b[ok])
    rqi = fzi_b[ok] * phi_z
    out[ok] = phi_b[ok] * (rqi / _RQI_CONST) ** 2
    if out.ndim == 0:
        return float(out)
    return out


def calculate_flow_units(
    perm_k: Union[np.ndarray, Sequence[float], "CoreData"],
    phi_e: Optional[Union[np.ndarray, Sequence[float]]] = None,
    n_units: Optional[int] = None,
    method: str = "kmeans",
    fzi_cutoffs: Optional[Sequence[float]] = None,
    max_units: int = 6,
) -> FlowUnitResult:
    """Determine hydraulic flow units from porosity and permeability."""
    depth = None
    if phi_e is None and hasattr(perm_k, "permeability") and hasattr(perm_k, "porosity"):
        # Core data: its plugs with both porosity and permeability.
        core = perm_k
        if core.porosity is None or core.permeability is None:
            raise ValueError("The core data needs porosity and permeability for flow units")
        plugs = np.isfinite(core.porosity) & np.isfinite(core.permeability)
        perm_k, phi_e = np.asarray(core.permeability)[plugs], np.asarray(core.porosity)[plugs]
        depth = None if core.depth is None else np.asarray(core.depth, dtype=float)[plugs]
    elif phi_e is None:
        raise ValueError("calculate_flow_units takes core data, or permeability and porosity arrays")
    perm_k = np.asarray(perm_k, dtype=float)
    phi_e = np.asarray(phi_e, dtype=float)
    rqi = np.asarray(calculate_rqi(perm_k, phi_e), dtype=float)
    phi_z = np.asarray(calculate_phi_z(phi_e), dtype=float)
    fzi = np.asarray(calculate_fzi(rqi, phi_z), dtype=float)

    labels, unit_mean_fzi, boundaries = assign_flow_units(
        fzi, n_units=n_units, method=method, fzi_cutoffs=fzi_cutoffs, max_units=max_units
    )
    n = int(unit_mean_fzi.size)

    perm_predicted = np.full(fzi.shape, np.nan)
    for unit in range(1, n + 1):
        sel = labels == unit
        if np.any(sel) and np.isfinite(unit_mean_fzi[unit - 1]):
            perm_predicted[sel] = predict_perm_from_fzi(unit_mean_fzi[unit - 1], phi_e[sel])

    counts = np.array([int(np.sum(labels == unit)) for unit in range(1, n + 1)])

    return FlowUnitResult(
        hfu=labels,
        n_units=n,
        rqi=rqi,
        phi_z=phi_z,
        fzi=fzi,
        perm_predicted=perm_predicted,
        fzi_unit_mean=unit_mean_fzi,
        fzi_boundaries=boundaries,
        counts=counts,
        depth=depth,
    )


class FlowUnitModule:
    """Hydraulic-flow-unit calculator (Amaefule FZI method)."""

    def __init__(
        self,
        n_units: Optional[int] = None,
        method: str = "kmeans",
        fzi_cutoffs: Optional[Sequence[float]] = None,
        max_units: int = 6,
    ):
        self.n_units = n_units
        self.method = method
        self.fzi_cutoffs = fzi_cutoffs
        self.max_units = max_units

    def calculate(
        self,
        well: "Well",
        perm: Optional[np.ndarray] = None,
        phi_e: Optional[np.ndarray] = None,
    ) -> FlowUnitResult:
        """Determine the well's hydraulic flow units.

        Parameters
        ----------
        well : Well
            Well whose permeability and porosity curves are read when
            ``perm`` or ``phi_e`` is not given.
        perm : array, optional
            Permeability per depth sample, mD. By default, at each depth the
            first of PERM_CC, PERM, PERM_CORE and PERM_TIMUR with a value.
        phi_e : array, optional
            Effective porosity per depth sample, v/v. By default, at each
            depth the first of PHIE_CC, PHIE and PHIT with a value.

        Returns
        -------
        FlowUnitResult
            Per sample, the flow unit (``hfu``, 0 without data), RQI, phi_z,
            FZI and the permeability predicted from its unit's mean FZI; and
            the units' mean FZI and boundaries.

        Raises
        ------
        ValueError
            If the well has no permeability or no porosity curve with finite
            samples and none is given.
        """
        if perm is None:
            perm, _, _ = get_curve_data_depthwise(
                well, ["PERM_CC", "PERM", "PERM_CORE", "PERM_TIMUR"]
            )
            if perm is None:
                raise ValueError("Permeability curve with finite samples required")
        if phi_e is None:
            phi_e, _, _ = get_curve_data_depthwise(well, ["PHIE_CC", "PHIE", "PHIT"])
            if phi_e is None:
                raise ValueError("PHIE_CC, PHIE, or PHIT with finite samples required")
        return calculate_flow_units(
            perm, phi_e,
            n_units=self.n_units, method=self.method,
            fzi_cutoffs=self.fzi_cutoffs, max_units=self.max_units,
        )

    def add_to_well(self, well: "Well", result: Optional[FlowUnitResult] = None) -> "Well":
        """Add the flow-unit curves to the well.

        Parameters
        ----------
        well : Well
            Well to add the curves to.
        result : FlowUnitResult, optional
            Result of :meth:`calculate` for this well; calculated from the
            well's curves when omitted.

        Returns
        -------
        Well
            ``well``, with ``RQI`` (microns), ``PHI_Z`` (v/v), ``FZI``
            (microns), ``HFU`` (flow unit number, 0 without data) and
            ``PERM_HFU`` (mD) added or replaced.
        """
        if result is None:
            result = self.calculate(well)
        well.add_curve("RQI", result.rqi, "microns", "Reservoir Quality Index")
        well.add_curve("PHI_Z", result.phi_z, "v/v", "Normalized Porosity")
        well.add_curve("FZI", result.fzi, "microns", "Flow Zone Indicator")
        well.add_curve("HFU", result.hfu.astype(float), "", "Hydraulic Flow Unit")
        # Class names and the flow-unit plot's colours, for plots coloured by HFU.
        n_units = max(int(result.n_units), 1)
        custom = well["HFU"].metadata.custom
        custom["category_labels"] = {0: "no unit", **{k: f"HFU {k}" for k in range(1, n_units + 1)}}
        try:
            from matplotlib import colormaps
        except ImportError:  # plotting is optional
            pass
        else:
            palette = colormaps["plasma"](np.linspace(0.0, 0.85, n_units))
            custom["category_colors"] = {0: "#999999", **{k: tuple(float(c) for c in palette[k - 1])
                                                          for k in range(1, n_units + 1)}}
        well.add_curve("PERM_HFU", result.perm_predicted, "mD", "Permeability from HFU")
        return well
