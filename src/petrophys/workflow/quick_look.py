"""
Quick look analysis for rapid well evaluation.
"""

from typing import TYPE_CHECKING, Optional, Dict, List
from dataclasses import dataclass, field
import numpy as np

from petrophys._internal.analysis_helpers import (
    cell_thickness,
    resolve_well_depth_range,
    well_depth_array,
)

if TYPE_CHECKING:
    from petrophys.model.well import Well


def _first_curve(well: "Well", preferred: List[str], standard: str):
    """The first of ``preferred``, else of the standard mnemonic's aliases, present on the well."""
    from petrophys.preprocessing.standardization.alias_map import get_aliases

    for name in dict.fromkeys([*preferred, standard, *get_aliases(standard)]):
        curve = well.get(name)
        if curve is not None:
            return curve
    return None


def _nanmean_or_nan(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else np.nan


@dataclass
class QuickLookResult:
    """Quick look analysis result."""

    well_name: str
    top_depth: float
    bottom_depth: float

    # Data quality
    has_gr: bool = False
    has_resistivity: bool = False
    has_density: bool = False
    has_neutron: bool = False
    has_sonic: bool = False

    # Quick estimates
    avg_gr: float = np.nan
    avg_rt: float = np.nan
    avg_rhob: float = np.nan
    avg_nphi: float = np.nan

    # Estimated properties: estimated_vcl and estimated_porosity are means of
    # the per-sample estimates; estimated_sw is the pore-volume-weighted mean
    # of the per-sample Archie Sw.
    estimated_vcl: float = np.nan
    estimated_porosity: float = np.nan
    estimated_sw: float = np.nan

    # Zones of interest; their top, bottom and thickness are in depth_unit
    potential_pay_zones: List[Dict] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    #: Depth unit of the well (and of the pay zones)
    depth_unit: str = ""


class QuickLook:
    """
    Quick look analysis for rapid well evaluation.

    Provides fast estimates without full interpretation.

    Parameters
    ----------
    gr_clean, gr_shale : float
        GR endpoints (API) of the linear VCL estimate.
    rho_matrix, rho_fluid : float
        Density endpoints (g/cm3) of the porosity estimate.
    rw : float, default 0.05
        Formation water resistivity (ohm.m, at formation temperature) of the
        Archie Sw estimate (a = 1, m = n = 2). It is not read from the well.
    vcl_cutoff : float, default 0.4
        Potential pay is cleaner than this linear GR clay volume, v/v: GR
        below ``gr_clean + vcl_cutoff * (gr_shale - gr_clean)`` (60 API with
        the default endpoints).
    rt_cutoff : float, default 10.0
        Potential pay reads a deep resistivity above this, ohm.m.
    min_thickness : float, default 2.0
        Thinnest potential pay zone reported, in the well's depth unit.
    """

    def __init__(
        self,
        gr_clean: float = 20.0,
        gr_shale: float = 120.0,
        rho_matrix: float = 2.65,
        rho_fluid: float = 1.0,
        rw: float = 0.05,
        vcl_cutoff: float = 0.4,
        rt_cutoff: float = 10.0,
        min_thickness: float = 2.0,
    ):
        self.gr_clean = gr_clean
        self.gr_shale = gr_shale
        self.rho_matrix = rho_matrix
        self.rho_fluid = rho_fluid
        self.rw = rw
        self.vcl_cutoff = vcl_cutoff
        self.rt_cutoff = rt_cutoff
        self.min_thickness = min_thickness

    def analyze(
        self,
        well: "Well",
        top_depth: Optional[float] = None,
        bottom_depth: Optional[float] = None
    ) -> QuickLookResult:
        """
        Perform quick look analysis.

        Parameters
        ----------
        well : Well
            Well object
        top_depth : float, optional
            Top depth
        bottom_depth : float, optional
            Bottom depth

        Returns
        -------
        QuickLookResult
            Quick look results
        """
        depth = well_depth_array(well)
        top_depth, bottom_depth = resolve_well_depth_range(well, top_depth, bottom_depth)

        result = QuickLookResult(
            well_name=well.name or "Unknown",
            top_depth=top_depth,
            bottom_depth=bottom_depth,
            depth_unit=getattr(well, "depth_unit", "") or "",
        )

        mask = (depth >= top_depth) & (depth <= bottom_depth)

        # Check data availability (standard mnemonics and their common aliases)
        gr_curve = _first_curve(well, ["GR"], "GR")
        rt_curve = _first_curve(well, ["RT", "LLD", "ILD", "RDEP"], "RDEP")
        rhob_curve = _first_curve(well, ["RHOB"], "RHOB")
        nphi_curve = _first_curve(well, ["NPHI"], "NPHI")
        result.has_gr = gr_curve is not None
        result.has_resistivity = rt_curve is not None
        result.has_density = rhob_curve is not None
        result.has_neutron = nphi_curve is not None
        result.has_sonic = _first_curve(well, ["DT"], "DTC") is not None

        # Quick averages; the estimates are made per sample, then averaged
        if gr_curve is not None:
            gr = np.asarray(gr_curve.data[mask], dtype=float)
            result.avg_gr = _nanmean_or_nan(gr)
            result.estimated_vcl = _nanmean_or_nan(self._quick_vcl(gr))

        rt = None
        if rt_curve is not None:
            rt = np.asarray(rt_curve.data[mask], dtype=float)
            result.avg_rt = _nanmean_or_nan(rt)

        phi = None
        if rhob_curve is not None:
            rhob = np.asarray(rhob_curve.data[mask], dtype=float)
            result.avg_rhob = _nanmean_or_nan(rhob)
            phi = self._quick_porosity(rhob)
            result.estimated_porosity = _nanmean_or_nan(phi)

        if nphi_curve is not None:
            nphi = nphi_curve.data[mask]
            result.avg_nphi = _nanmean_or_nan(nphi)

        # Sw: Archie per sample, averaged with pore-volume (PHI * h) weights,
        # so the samples that hold the most pore volume count most.
        if rt is not None and phi is not None:
            sw = self._quick_sw(rt, phi, rw=self.rw)
            weight = phi * cell_thickness(depth)[mask]
            valid = np.isfinite(sw) & np.isfinite(weight) & (weight > 0)
            if np.any(valid):
                result.estimated_sw = float(np.sum(sw[valid] * weight[valid]) / np.sum(weight[valid]))

        # Find potential pay zones
        result.potential_pay_zones = self._find_pay_zones(well, mask)

        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)

        return result

    def _quick_vcl(self, gr):
        """Linear GR clay volume within 0 and 1 (NaN when the GR endpoints are equal)."""
        gr = np.asarray(gr, dtype=float)
        if self.gr_shale == self.gr_clean:
            igr = np.full(gr.shape, np.nan)
        else:
            igr = np.clip((gr - self.gr_clean) / (self.gr_shale - self.gr_clean), 0, 1)
        return igr if igr.ndim else float(igr)

    def _quick_porosity(self, rhob):
        """Density porosity within 0 and 0.45."""
        phi = (self.rho_matrix - np.asarray(rhob, dtype=float)) / (self.rho_matrix - self.rho_fluid)
        phi = np.clip(phi, 0, 0.45)
        return phi if phi.ndim else float(phi)

    def _quick_sw(
        self,
        rt,
        porosity,
        rw: float = 0.05,
        a: float = 1.0,
        m: float = 2.0,
        n: float = 2.0
    ):
        """Archie Sw within 0 and 1; NaN where Rt is not positive, 1 at zero porosity."""
        rt = np.asarray(rt, dtype=float)
        porosity = np.asarray(porosity, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            sw = np.clip((a * rw / (porosity ** m * rt)) ** (1.0 / n), 0, 1)
            sw = np.where(porosity <= 0, 1.0, sw)
            sw = np.where(rt > 0, sw, np.nan)
        return sw if sw.ndim else float(sw)

    def _find_pay_zones(
        self,
        well: "Well",
        mask: np.ndarray
    ) -> List[Dict]:
        """Potential pay: GR clay volume below ``vcl_cutoff`` and Rt above ``rt_cutoff``.

        Contiguous runs at least ``min_thickness`` thick (in the well's
        depth unit) are reported.
        """
        zones = []
        depth = well_depth_array(well)
        depth_unit = getattr(well, "depth_unit", "") or ""

        gr = _first_curve(well, ["GR"], "GR")
        rt = _first_curve(well, ["RT", "LLD", "ILD", "RDEP"], "RDEP")

        if gr is None or rt is None:
            return zones

        gr_data = np.asarray(gr.data, dtype=float).copy()
        rt_data = np.asarray(rt.data, dtype=float).copy()

        with np.errstate(invalid="ignore"):
            clean = self._quick_vcl(gr_data) < self.vcl_cutoff
            high_rt = rt_data > self.rt_cutoff

        # Find contiguous zones
        pay_flag = clean & high_rt & mask
        in_zone = False
        zone_start = 0

        def _append_zone(start: int, stop: int) -> None:
            thickness = abs(depth[stop - 1] - depth[start])
            if thickness >= self.min_thickness:
                zones.append({
                    "top": float(min(depth[start], depth[stop - 1])),
                    "bottom": float(max(depth[start], depth[stop - 1])),
                    "thickness": float(thickness),
                    "depth_unit": depth_unit,
                    "avg_gr": _nanmean_or_nan(gr_data[start:stop]),
                    "avg_rt": _nanmean_or_nan(rt_data[start:stop]),
                })

        for i in range(len(pay_flag)):
            if pay_flag[i] and not in_zone:
                zone_start = i
                in_zone = True
            elif not pay_flag[i] and in_zone:
                _append_zone(zone_start, i)
                in_zone = False

        if in_zone:
            _append_zone(zone_start, len(pay_flag))

        return zones

    def _generate_recommendations(self, result: QuickLookResult) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Data quality recommendations
        if not result.has_gr:
            recommendations.append("No GR curve - VCL estimation limited")

        if not result.has_resistivity:
            recommendations.append("No resistivity - saturation cannot be calculated")

        if not result.has_density and not result.has_neutron:
            recommendations.append("No porosity logs - consider core data")

        # Interpretation recommendations
        if result.estimated_vcl > 0.5:
            recommendations.append("High shale content - consider shaly sand Sw model")

        if result.estimated_porosity < 0.08:
            recommendations.append("Low porosity - check for tight rock or bad hole")

        if result.estimated_sw < 0.5:
            recommendations.append("Potentially hydrocarbon-bearing - detailed analysis recommended")

        if len(result.potential_pay_zones) > 0:
            recommendations.append(
                f"Found {len(result.potential_pay_zones)} potential pay zones"
            )

        if not recommendations:
            recommendations.append("Standard interpretation workflow recommended")

        return recommendations


def quick_look_analysis(
    well: "Well",
    top_depth: Optional[float] = None,
    bottom_depth: Optional[float] = None,
    **kwargs
) -> QuickLookResult:
    """
    Perform quick look analysis.

    Parameters
    ----------
    well : Well
        Well object
    top_depth : float, optional
        Top depth
    bottom_depth : float, optional
        Bottom depth
    **kwargs
        Parameters of :class:`QuickLook`: ``gr_clean``, ``gr_shale``,
        ``rho_matrix``, ``rho_fluid``, ``rw``, ``vcl_cutoff``, ``rt_cutoff``
        and ``min_thickness`` (in the well's depth unit)

    Returns
    -------
    QuickLookResult
        Analysis results
    """
    ql = QuickLook(**kwargs)
    return ql.analyze(well, top_depth, bottom_depth)
