"""
Formation-level analysis module.

This module provides tools for analyzing well log data by geological
formation. It reads formation tops from CSV files, runs per-formation
summaries, and exports comprehensive reports.

Supported CSV formats:
    1. Simple: formation_name, top_depth, bottom_depth
    2. NPD stratigraphic picks: Well identifier, Surface, MD columns
    3. Danish format: wellName, formationName, topDepth, bottomDepth
"""

from typing import Optional, List, Dict, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import csv

from petrophys.evaluation.summaries.zone_summary import ZoneSummary, calculate_zone_summary
from petrophys.model.intervals import Formation
from petrophys._internal.statistics import finite_min_max


@dataclass
class FormationSummary:
    """Summary results for a single formation."""
    formation: Formation
    zone_summary: ZoneSummary

    # Per-formation curve statistics (over entire formation, not just pay)
    avg_gr: float = np.nan
    avg_rhob: float = np.nan
    avg_nphi: float = np.nan
    avg_rdep: float = np.nan
    avg_phit: float = np.nan

    # Data coverage
    data_coverage_pct: float = 0.0


@dataclass
class FormationAnalysisResult:
    """Complete formation analysis results for a well."""
    well_name: str
    formations: List[Formation]
    summaries: List[FormationSummary]
    depth_unit: str = "m"

    @property
    def total_gross(self) -> float:
        return sum(s.zone_summary.gross_thickness for s in self.summaries)

    @property
    def total_net_pay(self) -> float:
        return sum(s.zone_summary.net_pay for s in self.summaries)

    def to_dataframe(self):
        """Convert results to a pandas DataFrame."""
        import pandas as pd

        rows = []
        for s in self.summaries:
            zs = s.zone_summary
            rows.append({
                "Formation": s.formation.name,
                "Top": s.formation.top_depth,
                "Bottom": s.formation.bottom_depth,
                f"Gross ({self.depth_unit})": zs.gross_thickness,
                f"Net Rock ({self.depth_unit})": zs.net_rock,
                f"Net Reservoir ({self.depth_unit})": zs.net_reservoir,
                f"Net Pay ({self.depth_unit})": zs.net_pay,
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
                "Samples": zs.n_samples,
                "Pay Samples": zs.n_pay_samples,
                "Data Coverage %": s.data_coverage_pct,
            })

        return pd.DataFrame(rows)

    def to_csv(self, filepath: Union[str, Path]) -> None:
        """Export results to CSV."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False, float_format="%.4f")

    def print_report(self) -> None:
        """Print a formatted report to stdout."""
        print(f"\n{'=' * 90}")
        print(f"FORMATION ANALYSIS REPORT - {self.well_name}")
        print(f"{'=' * 90}")

        header = (f"{'Formation':<25s} {'Top':>8s} {'Bot':>8s} "
                  f"{'Gross':>7s} {'NtPay':>7s} {'NTG':>6s} "
                  f"{'PHIE':>6s} {'Sw':>6s} {'VCL':>6s} {'Perm':>8s}")
        print(header)
        print("-" * 90)

        for s in self.summaries:
            zs = s.zone_summary
            phie_str = f"{zs.avg_phie:.3f}" if not np.isnan(zs.avg_phie) else "  -  "
            sw_str = f"{zs.avg_sw:.3f}" if not np.isnan(zs.avg_sw) else "  -  "
            vcl_str = f"{zs.avg_vcl:.3f}" if not np.isnan(zs.avg_vcl) else "  -  "
            perm_str = f"{zs.avg_perm:.2f}" if not np.isnan(zs.avg_perm) else "    -   "

            print(f"{s.formation.name:<25s} "
                  f"{s.formation.top_depth:>8.1f} {s.formation.bottom_depth:>8.1f} "
                  f"{zs.gross_thickness:>7.1f} {zs.net_pay:>7.1f} "
                  f"{zs.ntg:>6.1%} "
                  f"{phie_str:>6s} {sw_str:>6s} {vcl_str:>6s} {perm_str:>8s}")

        print("-" * 90)
        print(f"{'TOTAL':<25s} "
              f"{'':>8s} {'':>8s} "
              f"{self.total_gross:>7.1f} {self.total_net_pay:>7.1f}")
        print(f"{'=' * 90}")


def load_formations_csv(
    filepath: Union[str, Path],
    well_name: Optional[str] = None,
    format: str = "auto"
) -> List[Formation]:
    """
    Load formation definitions from a CSV file.

    Supported formats:
        - "simple": Columns: formation_name, top_depth, bottom_depth [, lithology]
        - "npd": NPD stratigraphic picks with Well identifier, Surface, MD columns
        - "danish": wellName, formationName, topDepth, bottomDepth [, majorityLithology]
        - "auto": Detect format from column headers

    Parameters
    ----------
    filepath : str or Path
        Path to CSV file
    well_name : str, optional
        Well name to filter (required for NPD format with multiple wells)
    format : str
        CSV format: "auto", "simple", "npd", or "danish"

    Returns
    -------
    list of Formation
        Formation definitions sorted by top depth
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Formation CSV not found: {filepath}")

    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = [h.strip() for h in reader.fieldnames] if reader.fieldnames else []

        if format == "auto":
            format = _detect_csv_format(headers)

        rows = list(reader)

    if format == "npd":
        return _parse_npd_format(rows, well_name)
    elif format == "danish":
        return _parse_danish_format(rows, well_name)
    else:
        return _parse_simple_format(rows)


def _detect_csv_format(headers: List[str]) -> str:
    """Auto-detect CSV format from column headers."""
    headers_lower = [h.lower().strip() for h in headers]

    if "well identifier" in headers_lower or "surface" in headers_lower:
        return "npd"
    elif "wellname" in headers_lower and "formationname" in headers_lower:
        return "danish"
    else:
        return "simple"


def _parse_simple_format(rows: List[Dict]) -> List[Formation]:
    """Parse simple CSV: formation_name, top_depth, bottom_depth [, lithology]."""
    formations = []
    for row in rows:
        # Normalize keys
        keys = {k.strip().lower(): v for k, v in row.items()}

        name = (keys.get("formation_name") or keys.get("formation") or
                keys.get("name") or keys.get("zone") or "Unknown")
        top = _safe_float(keys.get("top_depth") or keys.get("top") or
                          keys.get("topdepth"))
        bottom = _safe_float(keys.get("bottom_depth") or keys.get("bottom") or
                             keys.get("bottomdepth"))
        lithology = (keys.get("lithology") or keys.get("majoritylithology") or "")

        if top is not None and bottom is not None:
            formations.append(Formation(
                name=name.strip(),
                top_depth=top,
                bottom_depth=bottom,
                lithology=lithology.strip()
            ))

    formations.sort(key=lambda f: f.top_depth)
    return formations


def _parse_npd_format(
    rows: List[Dict],
    well_name: Optional[str] = None
) -> List[Formation]:
    """
    Parse NPD stratigraphic picks format.

    NPD provides formation tops (not intervals), so we construct intervals
    by sorting tops and using the next top as the bottom of each formation.
    """
    # Filter by well
    filtered = []
    for row in rows:
        keys = {k.strip(): v for k, v in row.items()}
        well_id = keys.get("Well identifier", "").strip()

        if well_name and well_id != well_name:
            continue

        surface = keys.get("Surface", "").strip()
        md = _safe_float(keys.get("MD"))
        age = keys.get("Geological age", "")

        if md is not None and surface:
            filtered.append({
                "name": surface.replace(" Top", ""),
                "md": md,
                "age": str(age) if age else ""
            })

    if not filtered:
        return []

    # Sort by depth
    filtered.sort(key=lambda x: x["md"])

    # Convert tops to intervals
    formations = []
    for i in range(len(filtered)):
        top = filtered[i]["md"]
        if i + 1 < len(filtered):
            bottom = filtered[i + 1]["md"]
        else:
            bottom = top + 100.0  # Default 100m thickness for last formation

        if bottom > top:
            formations.append(Formation(
                name=filtered[i]["name"],
                top_depth=top,
                bottom_depth=bottom,
                age=filtered[i]["age"]
            ))

    return formations


def _parse_danish_format(
    rows: List[Dict],
    well_name: Optional[str] = None
) -> List[Formation]:
    """Parse Danish format: wellName, formationName, topDepth, bottomDepth."""
    formations = []
    for row in rows:
        keys = {k.strip().lower(): v for k, v in row.items()}

        wn = keys.get("wellname", "").strip()
        if well_name and wn != well_name:
            continue

        name = keys.get("formationname", "Unknown").strip()
        top = _safe_float(keys.get("topdepth"))
        bottom = _safe_float(keys.get("bottomdepth"))
        lithology = keys.get("majoritylithology", "").strip()

        if top is not None and bottom is not None:
            formations.append(Formation(
                name=name,
                top_depth=top,
                bottom_depth=bottom,
                lithology=lithology
            ))

    formations.sort(key=lambda f: f.top_depth)
    return formations


def _safe_float(value: Any) -> Optional[float]:
    """Safely convert to float, return None on failure."""
    if value is None:
        return None
    try:
        v = float(str(value).strip())
        return v if np.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def _get_formation_average(well, curve_name: str, mask: np.ndarray) -> float:
    """Get mean of a curve over a depth mask."""
    curve = well.get(curve_name)
    if curve is None:
        return np.nan
    data = curve.data[mask]
    valid = data[~np.isnan(data)]
    return float(np.mean(valid)) if len(valid) > 0 else np.nan


def analyze_formations(
    well,
    formations: List[Formation],
    net_pay_curve: str = "NET_PAY",
    net_reservoir_curve: str = "NET_RES",
    net_rock_curve: str = "NET_ROCK"
) -> FormationAnalysisResult:
    """
    Run formation-level analysis on a well with computed curves.

    The well should already have petrophysical curves computed
    (VCL, PHIE, SW, PERM, NET_PAY, etc.) before calling this function.

    Parameters
    ----------
    well : Well
        Well object with computed curves (after workflow execution)
    formations : list of Formation
        Formation definitions (from load_formations_csv or manual)
    net_pay_curve : str
        Name of the net pay flag curve
    net_reservoir_curve : str
        Name of the net reservoir flag curve
    net_rock_curve : str
        Name of the net rock flag curve

    Returns
    -------
    FormationAnalysisResult
        Formation-level analysis results

    Examples
    --------
    >>> formations = load_formations_csv("formations.csv")
    >>> result = analyze_formations(well, formations)
    >>> result.print_report()
    >>> result.to_csv("formation_results.csv")
    """
    depth = getattr(well, "depth", None)
    depth_unit = getattr(well, "_depth_unit", "ft")

    if depth is None:
        return FormationAnalysisResult(
            well_name=well.header.well_name or "Unknown",
            formations=[],
            summaries=[],
            depth_unit=depth_unit,
        )
    depth = np.asarray(depth, dtype=float)

    # Filter formations to well depth range
    well_top, well_bottom = finite_min_max(depth)
    if well_top is None or well_bottom is None:
        return FormationAnalysisResult(
            well_name=well.header.well_name or "Unknown",
            formations=[],
            summaries=[],
            depth_unit=depth_unit,
        )

    valid_formations = []
    for fm in formations:
        if fm.bottom_depth > well_top and fm.top_depth < well_bottom:
            # Clip to well range
            clipped = Formation(
                name=fm.name,
                top_depth=max(fm.top_depth, well_top),
                bottom_depth=min(fm.bottom_depth, well_bottom),
                lithology=fm.lithology,
                age=fm.age
            )
            valid_formations.append(clipped)

    # Guard: if formations were supplied but NONE overlap the log, the tops are
    # almost certainly in a different depth reference/unit (e.g. metres vs feet).
    if formations and not valid_formations:
        import warnings
        fm_top = min(fm.top_depth for fm in formations)
        fm_bot = max(fm.bottom_depth for fm in formations)
        warnings.warn(
            f"None of the {len(formations)} formation tops "
            f"({fm_top:.0f}-{fm_bot:.0f}) overlap the well depth range "
            f"({well_top:.0f}-{well_bottom:.0f} {depth_unit}). Check that the tops "
            "use the same depth reference and unit (ft vs m) as the log.",
            stacklevel=2,
        )

    summaries = []
    for fm in valid_formations:
        # Zone summary (net pay, NTG, averages over pay)
        zs = calculate_zone_summary(
            well,
            top_depth=fm.top_depth,
            bottom_depth=fm.bottom_depth,
            zone_name=fm.name,
            net_pay_curve=net_pay_curve,
            net_reservoir_curve=net_reservoir_curve,
            net_rock_curve=net_rock_curve
        )

        # Formation-wide curve averages (over entire formation, not just pay)
        mask = (depth >= fm.top_depth) & (depth <= fm.bottom_depth)

        avg_gr = _get_formation_average(well, "GR", mask)
        avg_rhob = _get_formation_average(well, "RHOB", mask)
        avg_nphi = _get_formation_average(well, "NPHI", mask)
        avg_rdep = _get_formation_average(well, "RDEP", mask)
        avg_phit = _get_formation_average(well, "PHIT", mask)

        # Data coverage: % of samples with valid RHOB in this formation
        rhob_curve = well.get("RHOB")
        if rhob_curve is not None:
            rhob_in_fm = rhob_curve.data[mask]
            valid_count = np.sum(~np.isnan(rhob_in_fm))
            total_count = len(rhob_in_fm)
            coverage = 100.0 * valid_count / total_count if total_count > 0 else 0.0
        else:
            coverage = 0.0

        summaries.append(FormationSummary(
            formation=fm,
            zone_summary=zs,
            avg_gr=avg_gr,
            avg_rhob=avg_rhob,
            avg_nphi=avg_nphi,
            avg_rdep=avg_rdep,
            avg_phit=avg_phit,
            data_coverage_pct=coverage,
        ))

    return FormationAnalysisResult(
        well_name=well.header.well_name or "Unknown",
        formations=valid_formations,
        summaries=summaries,
        depth_unit=depth_unit,
    )


def run_formation_analysis(
    las_path: Union[str, Path],
    formations_csv_path: Union[str, Path],
    well_name: Optional[str] = None,
    csv_format: str = "auto",
    workflow_config: Optional[Any] = None,
    standardize: bool = True,
    **workflow_kwargs
) -> FormationAnalysisResult:
    """
    End-to-end formation analysis: load LAS, run workflow, analyze by formation.

    This is a convenience function that combines:
    1. Loading the LAS file
    2. Standardizing curves
    3. Running the dynamic workflow
    4. Loading formation definitions from CSV
    5. Computing per-formation summaries

    Parameters
    ----------
    las_path : str or Path
        Path to LAS file
    formations_csv_path : str or Path
        Path to formations CSV file
    well_name : str, optional
        Well name for filtering formations (needed for NPD format)
    csv_format : str
        CSV format: "auto", "simple", "npd", "danish"
    workflow_config : DynamicWorkflowConfig, optional
        Workflow configuration (if None, uses workflow_kwargs)
    standardize : bool
        Whether to standardize curves before workflow
    **workflow_kwargs
        Additional workflow configuration parameters

    Returns
    -------
    FormationAnalysisResult
        Formation-level analysis results

    Examples
    --------
    >>> result = run_formation_analysis(
    ...     "well.las",
    ...     "formations.csv",
    ...     rw=0.05,
    ...     lithology="clastic",
    ... )
    >>> result.print_report()
    >>> result.to_csv("output.csv")
    """
    from petrophys.io import read_las
    from petrophys.preprocessing.standardization import standardize_curves
    from petrophys.workflow import run_dynamic_workflow

    # 1. Load LAS
    well = read_las(str(las_path))

    # 2. Standardize
    if standardize:
        well = standardize_curves(well, rename_curves=True, convert_units=True)

    # 3. Run dynamic workflow
    wf_result = run_dynamic_workflow(well, config=workflow_config, **workflow_kwargs)

    # 4. Load formations
    formations = load_formations_csv(
        formations_csv_path,
        well_name=well_name,
        format=csv_format
    )

    if not formations:
        raise ValueError(
            f"No formations found in {formations_csv_path}"
            + (f" for well '{well_name}'" if well_name else "")
        )

    # 5. Analyze by formation
    result = analyze_formations(well, formations)

    return result
