"""
Quick look analysis for rapid well evaluation.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import numpy as np


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

    # Estimated properties
    estimated_vcl: float = np.nan
    estimated_porosity: float = np.nan
    estimated_sw: float = np.nan

    # Zones of interest
    potential_pay_zones: List[Dict] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)


class QuickLook:
    """
    Quick look analysis for rapid well evaluation.

    Provides fast estimates without full interpretation.
    """

    def __init__(
        self,
        gr_clean: float = 20.0,
        gr_shale: float = 120.0,
        rho_matrix: float = 2.65,
        rho_fluid: float = 1.0
    ):
        self.gr_clean = gr_clean
        self.gr_shale = gr_shale
        self.rho_matrix = rho_matrix
        self.rho_fluid = rho_fluid

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
        depth = well.depth

        if top_depth is None:
            top_depth = well.start_depth or np.nanmin(depth)
        if bottom_depth is None:
            bottom_depth = well.stop_depth or np.nanmax(depth)

        result = QuickLookResult(
            well_name=well.header.get("WELL", "Unknown"),
            top_depth=top_depth,
            bottom_depth=bottom_depth
        )

        mask = (depth >= top_depth) & (depth <= bottom_depth)

        # Check data availability
        result.has_gr = well.get("GR") is not None
        result.has_resistivity = any(well.get(c) is not None for c in ["RT", "LLD", "ILD", "RDEP"])
        result.has_density = well.get("RHOB") is not None
        result.has_neutron = well.get("NPHI") is not None
        result.has_sonic = well.get("DT") is not None

        # Calculate quick averages
        if result.has_gr:
            gr = well.get("GR").data[mask]
            result.avg_gr = float(np.nanmean(gr))
            result.estimated_vcl = self._quick_vcl(result.avg_gr)

        if result.has_resistivity:
            for name in ["RT", "LLD", "ILD", "RDEP"]:
                rt = well.get(name)
                if rt is not None:
                    result.avg_rt = float(np.nanmean(rt.data[mask]))
                    break

        if result.has_density:
            rhob = well.get("RHOB").data[mask]
            result.avg_rhob = float(np.nanmean(rhob))
            result.estimated_porosity = self._quick_porosity(result.avg_rhob)

        if result.has_neutron:
            nphi = well.get("NPHI").data[mask]
            result.avg_nphi = float(np.nanmean(nphi))

        # Estimate Sw if we have enough data
        if result.has_resistivity and result.estimated_porosity > 0:
            result.estimated_sw = self._quick_sw(
                result.avg_rt,
                result.estimated_porosity
            )

        # Find potential pay zones
        result.potential_pay_zones = self._find_pay_zones(well, mask)

        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)

        return result

    def _quick_vcl(self, gr: float) -> float:
        """Quick VCL estimate from GR."""
        igr = (gr - self.gr_clean) / (self.gr_shale - self.gr_clean)
        igr = np.clip(igr, 0, 1)
        return float(igr)

    def _quick_porosity(self, rhob: float) -> float:
        """Quick porosity estimate from density."""
        phi = (self.rho_matrix - rhob) / (self.rho_matrix - self.rho_fluid)
        phi = np.clip(phi, 0, 0.45)
        return float(phi)

    def _quick_sw(
        self,
        rt: float,
        porosity: float,
        rw: float = 0.05,
        a: float = 1.0,
        m: float = 2.0,
        n: float = 2.0
    ) -> float:
        """Quick Sw estimate using Archie."""
        if rt <= 0 or porosity <= 0:
            return 1.0

        sw_n = a * rw / (porosity ** m * rt)
        sw = sw_n ** (1.0 / n)
        sw = np.clip(sw, 0, 1)
        return float(sw)

    def _find_pay_zones(
        self,
        well: "Well",
        mask: np.ndarray
    ) -> List[Dict]:
        """Find potential pay zones based on log response."""
        zones = []
        depth = well.depth

        # Look for low GR, high resistivity intervals
        gr = well.get("GR")
        rt = None
        for name in ["RT", "LLD", "ILD", "RDEP"]:
            rt = well.get(name)
            if rt is not None:
                break

        if gr is None or rt is None:
            return zones

        gr_data = gr.data.copy()
        rt_data = rt.data.copy()

        # Simple thresholds
        low_gr = gr_data < 60
        high_rt = rt_data > 10

        # Find contiguous zones
        pay_flag = low_gr & high_rt & mask
        in_zone = False
        zone_start = 0

        for i in range(len(pay_flag)):
            if pay_flag[i] and not in_zone:
                zone_start = i
                in_zone = True
            elif not pay_flag[i] and in_zone:
                zone_end = i
                thickness = depth[zone_end - 1] - depth[zone_start]
                if thickness >= 2.0:  # Minimum 2 ft
                    zones.append({
                        "top": float(depth[zone_start]),
                        "bottom": float(depth[zone_end - 1]),
                        "thickness": float(thickness),
                        "avg_gr": float(np.nanmean(gr_data[zone_start:zone_end])),
                        "avg_rt": float(np.nanmean(rt_data[zone_start:zone_end])),
                    })
                in_zone = False

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
        Additional parameters for QuickLook

    Returns
    -------
    QuickLookResult
        Analysis results
    """
    ql = QuickLook(**kwargs)
    return ql.analyze(well, top_depth, bottom_depth)
