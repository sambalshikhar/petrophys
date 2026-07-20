"""
Per-interval petrophysical analysis.

Given a Well object and one or more Interval objects, compute
summaries including net pay, average properties, and HCPV.
"""

from typing import Optional, List, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from petrophys.model.intervals import Interval
from petrophys.model.formation_tops import FormationTops
from petrophys.evaluation.summaries.zone_summary import ZoneSummary, calculate_zone_summary


@dataclass
class IntervalSummary:
    """
    Complete petrophysical summary for a single interval.

    Attributes
    ----------
    interval : Interval
        The analyzed interval
    zone_summary : ZoneSummary
        Net pay / NTG / pay-averaged properties (from summation module)
    avg_gr : float
        Mean GR over entire interval
    avg_rhob : float
        Mean RHOB over entire interval
    avg_nphi : float
        Mean NPHI over entire interval
    avg_rdep : float
        Mean RDEP over entire interval
    avg_phit : float
        Mean PHIT over entire interval
    dominant_lithology : str
        Most common lithology in the interval (from LITHO_CLASS curve)
    data_coverage_pct : float
        Percentage of samples with valid RHOB data
    """

    interval: Interval
    zone_summary: ZoneSummary

    avg_gr: float = np.nan
    avg_rhob: float = np.nan
    avg_nphi: float = np.nan
    avg_rdep: float = np.nan
    avg_phit: float = np.nan

    dominant_lithology: str = ""
    data_coverage_pct: float = 0.0


class IntervalAnalyzer:
    """
    Compute petrophysical summaries for depth intervals.

    Delegates net-pay math to ``calculate_zone_summary`` from the
    summation module and adds whole-interval curve averages,
    dominant lithology, and data coverage.

    Parameters
    ----------
    well : Well
        Well object with computed curves (VCL, PHIE, SW, PERM, NET_PAY, etc.)
    net_pay_curve : str
        Name of the net pay flag curve
    net_reservoir_curve : str
        Name of the net reservoir flag curve
    net_rock_curve : str
        Name of the net rock flag curve

    Examples
    --------
    >>> from petrophys.evaluation.intervals import IntervalAnalyzer, Formation
    >>> analyzer = IntervalAnalyzer(well)
    >>> summary = analyzer.analyze(Formation("Brent", 2500, 2700))
    >>> print(summary.zone_summary.net_pay)
    """

    def __init__(
        self,
        well,
        net_pay_curve: str = "NET_PAY",
        net_reservoir_curve: str = "NET_RES",
        net_rock_curve: str = "NET_ROCK",
    ):
        self.well = well
        self.net_pay_curve = net_pay_curve
        self.net_reservoir_curve = net_reservoir_curve
        self.net_rock_curve = net_rock_curve

    def analyze(self, interval: Interval) -> IntervalSummary:
        """
        Compute summary for a single interval.

        Parameters
        ----------
        interval : Interval
            Depth interval to analyze

        Returns
        -------
        IntervalSummary
        """
        well = self.well
        depth = well.depth

        # Zone summary (net pay, NTG, pay-averaged properties)
        zs = calculate_zone_summary(
            well,
            top_depth=interval.top_depth,
            bottom_depth=interval.bottom_depth,
            zone_name=interval.name,
            net_pay_curve=self.net_pay_curve,
            net_reservoir_curve=self.net_reservoir_curve,
            net_rock_curve=self.net_rock_curve,
        )

        # Whole-interval mask
        mask = (depth >= interval.top_depth) & (depth <= interval.bottom_depth)

        # Curve averages over entire interval
        avg_gr = _curve_mean(well, "GR", mask)
        avg_rhob = _curve_mean(well, "RHOB", mask)
        avg_nphi = _curve_mean(well, "NPHI", mask)
        avg_rdep = _curve_mean(well, "RDEP", mask)
        avg_phit = _curve_mean(well, "PHIT", mask)

        # Dominant lithology
        dominant_lithology = _dominant_lithology(well, mask)

        # Data coverage (% valid RHOB samples)
        coverage = _data_coverage(well, "RHOB", mask)

        return IntervalSummary(
            interval=interval,
            zone_summary=zs,
            avg_gr=avg_gr,
            avg_rhob=avg_rhob,
            avg_nphi=avg_nphi,
            avg_rdep=avg_rdep,
            avg_phit=avg_phit,
            dominant_lithology=dominant_lithology,
            data_coverage_pct=coverage,
        )

    def analyze_all(
        self,
        tops: Union[FormationTops, List[Interval]],
        clip_to_well: bool = True,
    ) -> List[IntervalSummary]:
        """
        Analyze all intervals in a collection.

        Parameters
        ----------
        tops : FormationTops or list of Interval
            Intervals to analyze
        clip_to_well : bool
            If True, clip intervals to well depth range first

        Returns
        -------
        list of IntervalSummary
        """
        if isinstance(tops, FormationTops):
            if clip_to_well:
                tops = tops.clip_to_well(self.well)
            intervals = tops.intervals
        else:
            intervals = list(tops)

        return [self.analyze(iv) for iv in intervals]

    @staticmethod
    def to_dataframe(summaries: List[IntervalSummary]):
        """
        Convert analysis results to a pandas DataFrame.

        Parameters
        ----------
        summaries : list of IntervalSummary

        Returns
        -------
        pd.DataFrame
        """
        import pandas as pd

        rows = []
        for s in summaries:
            zs = s.zone_summary
            rows.append({
                "Name": s.interval.name,
                "Kind": s.interval.kind.value,
                "Top": s.interval.top_depth,
                "Bottom": s.interval.bottom_depth,
                "Gross": zs.gross_thickness,
                "Net Rock": zs.net_rock,
                "Net Reservoir": zs.net_reservoir,
                "Net Pay": zs.net_pay,
                "NTG": zs.ntg,
                "Avg PHIE": zs.avg_phie,
                "Avg Sw": zs.avg_sw,
                "Avg VCL": zs.avg_vcl,
                "Avg Perm (mD)": zs.avg_perm,
                "Avg GR": s.avg_gr,
                "Avg RHOB": s.avg_rhob,
                "Avg NPHI": s.avg_nphi,
                "Avg RDEP": s.avg_rdep,
                "Avg PHIT": s.avg_phit,
                "HCPV": zs.hcpv,
                "BVW": zs.bvw,
                "Dominant Lithology": s.dominant_lithology,
                "Samples": zs.n_samples,
                "Pay Samples": zs.n_pay_samples,
                "Data Coverage %": s.data_coverage_pct,
            })

        return pd.DataFrame(rows)

    @staticmethod
    def to_csv(
        summaries: List[IntervalSummary],
        filepath: Union[str, Path],
    ) -> None:
        """Export results to CSV."""
        df = IntervalAnalyzer.to_dataframe(summaries)
        df.to_csv(filepath, index=False, float_format="%.4f")

    @staticmethod
    def print_report(summaries: List[IntervalSummary]) -> None:
        """Print a formatted summary report to stdout."""
        if not summaries:
            print("No intervals to report.")
            return

        print(f"\n{'=' * 95}")
        print("INTERVAL ANALYSIS REPORT")
        print(f"{'=' * 95}")

        header = (
            f"{'Name':<25s} {'Top':>8s} {'Bot':>8s} "
            f"{'Gross':>7s} {'NtPay':>7s} {'NTG':>6s} "
            f"{'PHIE':>6s} {'Sw':>6s} {'VCL':>6s} {'Perm':>8s}"
        )
        print(header)
        print("-" * 95)

        total_gross = 0.0
        total_pay = 0.0

        for s in summaries:
            zs = s.zone_summary
            total_gross += zs.gross_thickness
            total_pay += zs.net_pay

            phie_str = f"{zs.avg_phie:.3f}" if not np.isnan(zs.avg_phie) else "  -  "
            sw_str = f"{zs.avg_sw:.3f}" if not np.isnan(zs.avg_sw) else "  -  "
            vcl_str = f"{zs.avg_vcl:.3f}" if not np.isnan(zs.avg_vcl) else "  -  "
            perm_str = f"{zs.avg_perm:.2f}" if not np.isnan(zs.avg_perm) else "    -   "

            print(
                f"{s.interval.name:<25s} "
                f"{s.interval.top_depth:>8.1f} {s.interval.bottom_depth:>8.1f} "
                f"{zs.gross_thickness:>7.1f} {zs.net_pay:>7.1f} "
                f"{zs.ntg:>6.1%} "
                f"{phie_str:>6s} {sw_str:>6s} {vcl_str:>6s} {perm_str:>8s}"
            )

        print("-" * 95)
        print(f"{'TOTAL':<25s} {'':>8s} {'':>8s} "
              f"{total_gross:>7.1f} {total_pay:>7.1f}")
        print(f"{'=' * 95}")


# ====================================================================== #
#  Convenience function                                                    #
# ====================================================================== #

def analyze_interval(
    well,
    interval: Interval,
    **kwargs,
) -> IntervalSummary:
    """
    One-shot analysis for a single interval.

    Parameters
    ----------
    well : Well
        Well with computed curves
    interval : Interval
        Interval to analyze
    **kwargs
        Passed to IntervalAnalyzer (net_pay_curve, etc.)

    Returns
    -------
    IntervalSummary
    """
    return IntervalAnalyzer(well, **kwargs).analyze(interval)


# ====================================================================== #
#  Internal helpers                                                        #
# ====================================================================== #

def _curve_mean(well, curve_name: str, mask: np.ndarray) -> float:
    """Mean of a curve over a depth mask, ignoring NaN."""
    curve = well.get(curve_name)
    if curve is None:
        return np.nan
    data = curve.data[mask]
    valid = data[~np.isnan(data)]
    return float(np.mean(valid)) if len(valid) > 0 else np.nan


def _dominant_lithology(well, mask: np.ndarray) -> str:
    """Find most common lithology value in the interval."""
    for curve_name in ("LITHO_CLASS", "LITHOLOGY", "LITH"):
        curve = well.get(curve_name)
        if curve is not None:
            data = curve.data[mask]
            # Handle object/string arrays
            valid = [v for v in data if v is not None and str(v) != "nan"]
            if valid:
                from collections import Counter
                return Counter(valid).most_common(1)[0][0]
    return ""


def _data_coverage(well, curve_name: str, mask: np.ndarray) -> float:
    """Percentage of valid (non-NaN) samples for a curve in the interval."""
    curve = well.get(curve_name)
    if curve is None:
        return 0.0
    data = curve.data[mask]
    total = len(data)
    if total == 0:
        return 0.0
    try:
        valid = np.sum(~np.isnan(data.astype(float)))
    except (ValueError, TypeError):
        valid = sum(1 for v in data if v is not None)
    return 100.0 * valid / total
