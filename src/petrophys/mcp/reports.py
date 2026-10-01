"""Model-sized reports built on the library's QC, geochemistry and summaries.

Each function returns compact, JSON-serializable data that answers a
question directly (which curves are bad and how to fix them, what the
thorium/uranium ratio says, how much pay each formation holds), so a model
does not need to read raw curve values.
"""

from __future__ import annotations

import copy
import difflib
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from petrophys.mcp.serialize import compact_object, header_notes, round_float

if TYPE_CHECKING:
    from petrophys.model.well import Well

#: Curves that legitimately repeat values (bit size, codes, flags).
_CONSTANT_BY_DESIGN = ("BS", "BIT", "MUDWEIGHT", "LITH", "FLAG", "CODE", "ZONE", "QC", "TEST",
                       "FLUID", "USE_FIT")

#: Curves read by pad tools, which bad hole affects most.
_PAD_CURVES = ("RHOB", "DRHO", "NPHI", "PEF")


def _fmt(value: Optional[float]) -> str:
    return "?" if value is None else f"{round_float(float(value)):g}"


def _intervals(depth: np.ndarray, flag: np.ndarray, *, merge_gap: float = 0.0
               ) -> List[Tuple[float, float, int]]:
    """Contiguous flagged intervals as (top, bottom, samples), merging small gaps."""
    if depth.size == 0 or not np.any(flag):
        return []
    padded = np.concatenate([[False], flag.astype(bool), [False]])
    changes = np.diff(padded.astype(int))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    runs: List[List[Any]] = []
    for start, end in zip(starts, ends):
        top, bottom = float(depth[start]), float(depth[end])
        count = int(end - start + 1)
        if runs and merge_gap > 0 and top - runs[-1][1] <= merge_gap:
            runs[-1][1] = bottom
            runs[-1][2] += count
        else:
            runs.append([top, bottom, count])
    return [(top, bottom, count) for top, bottom, count in runs]


def _describe_intervals(runs: Sequence[Tuple[float, float, int]], limit: int = 4) -> str:
    ranked = sorted(runs, key=lambda item: item[1] - item[0], reverse=True)[:limit]
    text = ", ".join(f"{_fmt(top)}-{_fmt(bottom)}" for top, bottom, _ in sorted(ranked))
    if len(runs) > limit:
        text += f" (+{len(runs) - limit} smaller)"
    return text


def _is_categorical(name: str, unit: str, values: np.ndarray) -> bool:
    upper = name.upper()
    if any(token in upper for token in ("LITH", "FLAG", "CODE", "ZONE", "FACIES")):
        return True
    if unit.strip() in {"_", "-", "code", "flag"}:
        finite = values[np.isfinite(values)]
        return finite.size == 0 or bool(np.all(finite == np.round(finite)))
    return False


# -- data quality -------------------------------------------------------------------

def data_quality_report(
    well: "Well",
    *,
    curves: Optional[Sequence[str]] = None,
    top: Optional[float] = None,
    bottom: Optional[float] = None,
) -> Dict[str, Any]:
    """Per-curve data-quality problems with recommended fixes.

    Uses the library's own checks: plausible-range screening with unit
    conversion (:mod:`petrophys.preprocessing.standardization`), held
    readings (:func:`~petrophys.preprocessing.quality.find_held_readings`),
    bad-hole flags (:func:`~petrophys.preprocessing.quality.generate_bad_hole_flags`),
    coverage, values pinned at a limit and duplicate curves.

    Raises
    ------
    ValueError
        If a requested curve is missing or the window holds no samples.
    """
    from petrophys.preprocessing.quality import find_held_readings, generate_bad_hole_flags
    from petrophys.preprocessing.standardization.curve_standardizer import CurveStandardizer

    names = list(curves) if curves else list(well.curves)
    missing = [name for name in names if well.get(name) is None]
    if missing:
        raise ValueError(
            f"Curve(s) not on the well: {', '.join(missing)}. Curves: {', '.join(well.curves)}."
        )
    window = well.depth_slice(top, bottom)
    depth = np.asarray(window.depth, dtype=float)
    if depth.size == 0:
        raise ValueError(
            f"No samples between {top} and {bottom} {well.depth_unit}; the well covers "
            f"{_fmt(well.top)}-{_fmt(well.bottom)} {well.depth_unit}."
        )
    unit = well.depth_unit
    span = float(np.nanmax(depth) - np.nanmin(depth)) or 1.0
    standardizer = CurveStandardizer(rename_curves=True, convert_units=True)

    entries: Dict[str, Dict[str, Any]] = {}
    numeric: Dict[str, np.ndarray] = {}
    sparse: List[List[Any]] = []
    sparse_notes: List[str] = []

    def issue(name: str, severity: int, text: str, fix: str) -> None:
        entry = entries[name]
        entry["severity"] = max(entry["severity"], severity)
        entry["issues"].append(text)
        if fix and fix not in entry["fixes"]:
            entry["fixes"].append(fix)

    for name in names:
        curve = window.get(name)
        if curve is None:
            continue
        curve_unit = curve.unit or ""
        try:
            values = np.asarray(curve.data, dtype=float)
        except (TypeError, ValueError):
            continue
        finite = np.isfinite(values)
        valid = int(np.count_nonzero(finite))
        entries[name] = {"curve": name, "unit": curve_unit,
                         "valid_%": round(100.0 * valid / values.size, 1) if values.size else 0.0,
                         "severity": 0, "issues": [], "fixes": []}
        if valid == 0:
            issue(name, 2, "no valid samples in the window", "Do not use this curve here.")
            continue
        # Coverage: where the curve has data, and gaps inside that range.
        valid_depth = depth[finite]
        first, last = float(valid_depth.min()), float(valid_depth.max())
        inside = (depth >= first) & (depth <= last)
        if valid < 0.5 * np.count_nonzero(inside):
            # Point data such as core plugs: report where, not as a problem.
            sparse.append([name, curve_unit, valid, first, last])
            del entries[name]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                standardizer.standardize_curve(curve)
            sparse_notes.extend(str(item.message).rstrip(".") for item in caught)
            continue
        if (last - first) < 0.9 * span:
            issue(name, 1,
                  f"logged only {_fmt(first)}-{_fmt(last)} {unit} "
                  f"({entries[name]['valid_%']}% of the window)",
                  f"Restrict analyses that use {name} to {_fmt(first)}-{_fmt(last)} {unit}.")
        gaps = _intervals(depth, inside & ~finite)
        gap_samples = sum(count for _, _, count in gaps)
        if gaps and gap_samples > 0.05 * np.count_nonzero(inside):
            issue(name, 2 if gap_samples > 0.2 * np.count_nonzero(inside) else 1,
                  f"{len(gaps)} gaps inside its logged range ({gap_samples} samples), "
                  f"largest {_describe_intervals(gaps, 2)} {unit}",
                  "Gaps stay null in interpret_well; do not interpolate across long gaps.")
        if _is_categorical(name, curve_unit, values):
            entries[name]["kind"] = "categorical"
            continue
        numeric[name] = values

        # Plausible-range screening, with the library's unit conversion.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            screened = standardizer.standardize_curve(curve)
        report = screened.metadata.custom.get("range_screening") if screened.metadata else None
        messages = [str(item.message) for item in caught
                    if "outside the plausible" not in str(item.message)]
        for message in messages:
            issue(name, 1, message.rstrip("."), "")
        if report:
            fraction = float(report["fraction"])
            low, high = report["low"], report["high"]
            found = values[finite]
            observed = f"observed {_fmt(found.min())} to {_fmt(found.max())}"
            if fraction >= 0.5:
                hint = f"; {report['hint']}" if report.get("hint") else ""
                issue(name, 3,
                      f"{100 * fraction:.0f}% of values outside the plausible "
                      f"{report['standard_curve']} range {low:g}-{high:g} {screened.unit or ''} "
                      f"({observed}{hint}): wrong unit, scale or curve type",
                      f"Check the unit label and scale of {name} in the source file; until "
                      "fixed, interpret_well removes these values, so the curve is effectively "
                      "unused.")
            else:
                issue(name, 2 if fraction >= 0.01 else 1,
                      f"{report['masked']} samples ({100 * fraction:.1f}%) outside "
                      f"{low:g}-{high:g} {screened.unit or ''} ({observed}): spikes or bad hole",
                      "interpret_well sets out-of-range samples to null; check them against "
                      "caliper and DRHO if they fall in the reservoir.")
        if entries[name]["severity"] >= 3:
            continue
        # Values pinned at the curve's maximum (tool or file limit).
        found = values[finite]
        peak = float(found.max())
        pinned = int(np.count_nonzero(found == peak))
        if pinned >= 5 and pinned >= 0.001 * found.size and np.unique(found).size > 20:
            issue(name, 1, f"{pinned} samples pinned at the maximum {_fmt(peak)} {curve_unit}",
                  f"Treat those samples as '>= {_fmt(peak)}' (tool or recording limit).")

    # Held (stuck or padded) readings on measured curves still worth using.
    held_names = [name for name in numeric
                  if entries[name]["severity"] < 3
                  and not any(token in name.upper() for token in _CONSTANT_BY_DESIGN)]
    if held_names:
        held = find_held_readings(window, held_names, min_run=10)
        tolerance = 0.02 * span
        for name in held_names:
            table: Any = held.runs
            runs = table[table["curve"] == name]
            if runs.empty:
                continue
            logged = depth[np.isfinite(numeric[name])]
            at_base = runs[runs["bottom"] >= float(logged.max()) - tolerance]
            at_top = runs[runs["top"] <= float(logged.min()) + tolerance]
            at_top = at_top[~at_top.index.isin(at_base.index)]
            elsewhere = runs[~runs.index.isin(at_base.index) & ~runs.index.isin(at_top.index)]
            if not at_top.empty:
                end = float(at_top["bottom"].max())
                count = int(at_top["samples"].sum())
                issue(name, 2 if count >= 0.01 * depth.size else 1,
                      f"held (repeated) value above {_fmt(end)} {unit} ({count} samples at the "
                      "top of the log): filled before the tool was logging",
                      f"Exclude {name} above {_fmt(end)} {unit}.")
            if not at_base.empty:
                start = float(at_base["top"].min())
                count = int(at_base["samples"].sum())
                issue(name, 2 if count >= 0.01 * depth.size else 1,
                      f"held (repeated) value below {_fmt(start)} {unit} ({count} samples at the "
                      "base of the log): padding below the sensor",
                      f"Exclude {name} below {_fmt(start)} {unit}, or end the analysis window "
                      "above it.")
            if not elsewhere.empty:
                count = int(elsewhere["samples"].sum())
                longest = elsewhere.sort_values("samples", ascending=False).iloc[0]
                issue(name, 1,
                      f"{count} held (repeated) samples in {len(elsewhere)} run(s), longest "
                      f"{_fmt(longest['top'])}-{_fmt(longest['bottom'])} {unit} at "
                      f"{_fmt(longest['value'])}",
                      "Check these runs for a stuck tool (a caliper in gauge hole can repeat "
                      "legitimately); exclude them if not geological.")

    # Duplicate curves (identical arrays under different names).
    seen: Dict[bytes, str] = {}
    for name, values in numeric.items():
        key = np.where(np.isfinite(values), values, np.nan).tobytes()
        if key in seen:
            issue(name, 1, f"identical to {seen[key]} (a copy, not a separate measurement)",
                  f"Use only one of {seen[key]} and {name}.")
        else:
            seen[key] = name

    # Bad hole: caliper versus bit size, DRHO and very large hole.
    bad_hole: Dict[str, Any] = {}
    flags = generate_bad_hole_flags(window)
    has_inputs = any(window.get(name) is not None for name in ("CALI", "CAL", "DRHO"))
    if has_inputs:
        combined = np.asarray(flags["bad_hole"], dtype=bool)
        runs = _intervals(depth, combined, merge_gap=1.0 if unit == "m" else 3.0)
        bad_hole = {
            "criteria": "CALI - BS > 1.5 in, |DRHO| > 0.15 g/cm3, or BS > 17.5 in",
            "flagged_%": round(100.0 * np.count_nonzero(combined) / depth.size, 1),
            "washout_%": round(100.0 * np.count_nonzero(flags["washout"]) / depth.size, 1),
            "large_hole_%": round(100.0 * np.count_nonzero(flags["large_hole"]) / depth.size, 1),
            "main_intervals": _describe_intervals(runs, 5) + (f" {unit}" if runs else ""),
        }
        if runs:
            bad_hole["fix"] = (
                "Pad-tool curves (" + ", ".join(n for n in _PAD_CURVES if window.get(n) is not None)
                + ") read the mud there; interpret_well flags these samples (bad_hole step). "
                "Exclude the intervals from reservoir averages or rely on sonic porosity there."
            )
    else:
        bad_hole = {"note": "No caliper or DRHO curve: bad hole cannot be assessed."}

    problem = sorted((entry for entry in entries.values() if entry["issues"]),
                     key=lambda entry: (-entry["severity"], entry["curve"]))
    labels = {3: "high", 2: "medium", 1: "low"}
    for entry in problem:
        entry["severity"] = labels[entry["severity"]]
        entry.pop("kind", None)
    ok = [entry["curve"] for entry in entries.values() if not entry["issues"]]
    counts = {label: sum(1 for entry in problem if entry["severity"] == label)
              for label in ("high", "medium", "low")}
    overview = (
        f"{len(entries)} curves checked over {_fmt(depth.min())}-{_fmt(depth.max())} {unit}: "
        + ", ".join(f"{count} {label}" for label, count in counts.items() if count)
        + (" severity" if problem else "no problems found")
    )
    if bad_hole.get("flagged_%"):
        overview += f"; bad hole in {bad_hole['flagged_%']}% of the window"
    data: Dict[str, Any] = {
        "well": well.name,
        "depth_unit": unit,
        "window": {"top": float(depth.min()), "bottom": float(depth.max())},
        "overview": overview,
        "problem_curves": problem,
        "ok_curves": ok,
        "bad_hole": bad_hole,
        "header_notes": header_notes(well),
    }
    if sparse:
        data["point_data"] = {
            "columns": ["curve", "unit", "samples", "top", "bottom"],
            "rows": sparse,
            "note": "Sparse curves (e.g. core plugs): sampled points, not continuous logs.",
        }
        if sparse_notes:
            data["point_data"]["issues"] = sparse_notes
    return compact_object(data)


# -- spectral gamma -------------------------------------------------------------------

def spectral_gamma_report(
    well: "Well",
    *,
    top: Optional[float] = None,
    bottom: Optional[float] = None,
    max_intervals: int = 6,
) -> Dict[str, Any]:
    """Spectral gamma ray summary: inputs, statistics, Th/U environment and clay index.

    Built on :func:`~petrophys.interpretation.geochemistry.interpret_spectral_gamma`,
    :func:`~petrophys.interpretation.geochemistry.classify_th_u_environment` and
    :func:`~petrophys.interpretation.geochemistry.thorium_clay_index`. Held
    (padded) readings are excluded before any statistic.

    Raises
    ------
    ValueError
        If the well has no processed spectral gamma curves, or the window has
        no spectral data.
    """
    from petrophys.interpretation.geochemistry import (
        TH_U_ENVIRONMENT_THRESHOLDS,
        TH_U_ENVIRONMENTS,
        classify_th_u_environment,
        has_spectral_gamma_inputs,
        interpret_spectral_gamma,
        thorium_clay_index,
    )
    from petrophys.preprocessing.quality import held_reading_mask

    if not has_spectral_gamma_inputs(well):
        raise ValueError(
            f"Well {well.name!r} has no processed spectral gamma curves (potassium, thorium or "
            f"uranium such as POTA/THOR/URAN, or SGR/CGR). Curves: {', '.join(well.curves)}. "
            "Among the samples, 25_5-1 has THOR and URAN."
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = interpret_spectral_gamma(copy.deepcopy(well), add_curves=False)
    depth = np.asarray(well.depth, dtype=float)
    window = np.isfinite(depth)
    if top is not None:
        window &= depth >= float(top)
    if bottom is not None:
        window &= depth <= float(bottom)
    unit = well.depth_unit

    measurements = dict(result.measurements)
    held_notes = []
    usable = window.copy()
    for name, values in measurements.items():
        mask = held_reading_mask(values, min_run=10)
        if np.any(mask & window):
            runs = _intervals(depth, mask & window)
            held_notes.append(f"{result.sources.get(name, name)} held (padded) at "
                              f"{_describe_intervals(runs, 3)} {unit}; excluded")
        usable &= ~mask
    present = [name for name in ("POTASSIUM", "THORIUM", "URANIUM", "SGR", "CGR")
               if name in measurements]
    any_valid = np.zeros(depth.size, dtype=bool)
    for name in present:
        any_valid |= np.isfinite(measurements[name])
    if not np.any(any_valid & usable):
        logged = _intervals(depth, any_valid)
        raise ValueError(
            f"No spectral gamma data between {top} and {bottom} {unit}; it is logged at "
            f"{_describe_intervals(logged, 3)} {unit}."
        )
    logged_depth = depth[any_valid & usable]
    statistics: Dict[str, Any] = {}
    for name in present:
        values = measurements[name][usable]
        finite = values[np.isfinite(values)]
        if finite.size:
            p10, p50, p90 = np.percentile(finite, [10, 50, 90])
            statistics[f"{name} ({result.units.get(name, '')})"] = {
                "valid": int(finite.size), "p10": p10, "median": p50, "mean": finite.mean(),
                "p90": p90}

    data: Dict[str, Any] = {
        "well": well.name,
        "depth_unit": unit,
        "inputs": {name: f"{result.sources[name]} ({result.conversions.get(name, '')})"
                   for name in present},
        "spectral_data_interval": {"top": float(logged_depth.min()),
                                   "bottom": float(logged_depth.max())},
        "statistics": statistics,
    }
    if "THORIUM" in measurements and "URANIUM" in measurements:
        thorium = np.where(usable, measurements["THORIUM"], np.nan)
        uranium = np.where(usable, measurements["URANIUM"], np.nan)
        environment = classify_th_u_environment(thorium, uranium)
        rated = environment.rated
        count = int(np.count_nonzero(rated))
        shares = {label: 100.0 * np.count_nonzero(environment.environment == label) / count
                  if count else None for label in TH_U_ENVIRONMENTS}
        low, high = TH_U_ENVIRONMENT_THRESHOLDS
        th_u: Dict[str, Any] = {
            "rated_samples": count,
            "not_rated": f"samples with U < {environment.uranium_floor_ppm:g} ppm (noise)",
            "classes": f"Th/U < {low:g} marine reducing; {low:g}-{high:g} marine; > {high:g} "
                       "continental oxidising (Adams and Weaver 1958; Fertl 1979)",
            "share_%": shares,
        }
        if count:
            ratios = environment.th_u_ratio[rated]
            th_u.update({"median_th_u": float(np.median(ratios)),
                         "p10_th_u": float(np.percentile(ratios, 10)),
                         "p90_th_u": float(np.percentile(ratios, 90))})
            th_u["by_interval"] = _environment_by_interval(well, depth, environment,
                                                           any_valid & usable, max_intervals)
            has_tops = bool((well.get_parameter("_geology_context") or {}).get("formations"))
            th_u["by_interval_basis"] = (
                "attached formations" if has_tops else
                "intervals of one dominant class (running median of Th/U; no formation tops "
                "attached)"
            )
            reducing = _intervals(depth, environment.environment == TH_U_ENVIRONMENTS[0],
                                  merge_gap=1.0 if unit == "m" else 3.0)
            thick = [run for run in reducing
                     if run[1] - run[0] >= (1.0 if unit == "m" else 3.0)]
            if thick:
                th_u["uranium_rich_beds"] = (
                    f"Th/U < {low:g} over {_describe_intervals(thick, 5)} {unit}"
                )
        data["th_u_environment"] = th_u
        finite_th = np.isfinite(thorium)
        if np.count_nonzero(finite_th) >= 10:
            index = thorium_clay_index(thorium, mask=finite_th)
            values = index.index[finite_th]
            data["thorium_clay_index"] = {
                "clean_ppm": index.clean_ppm, "shale_ppm": index.shale_ppm,
                "median": float(np.median(values)),
                "basis": "linear Th index, baselines at P5/P95 of thorium in the window",
            }
    derived = {}
    for name in ("TH_K_RATIO", "U_K_RATIO"):
        if name in result.derived:
            values = result.derived[name][usable]
            finite = values[np.isfinite(values)]
            if finite.size:
                derived[f"{name} ({result.units.get(name, '')})"] = {
                    "median": float(np.median(finite)), "p10": float(np.percentile(finite, 10)),
                    "p90": float(np.percentile(finite, 90))}
    if derived:
        data["ratios"] = derived
    notes = held_notes + [str(item.message) for item in caught] + list(result.warnings)
    notes.append("Read the classes over intervals, not sample by sample, and confirm them "
                 "with tops, lithology or core.")
    data["notes"] = notes
    return compact_object(data)


def _environment_by_interval(well: "Well", depth: np.ndarray, environment: Any,
                             usable: np.ndarray, max_intervals: int) -> List[Dict[str, Any]]:
    """Th/U class shares per attached formation, else per equal-thickness depth bin."""
    from petrophys.interpretation.geochemistry import TH_U_ENVIRONMENTS

    zones: List[Tuple[str, float, float]] = []
    records = (well.get_parameter("_geology_context") or {}).get("formations") or []
    for row in records:
        top, bottom = row.get("top_depth"), row.get("bottom_depth")
        if top is not None and bottom is not None:
            zones.append((str(row.get("formation_name")), float(top), float(bottom)))
    rated = environment.rated & usable
    if zones:
        zones = [zone for zone in zones
                 if np.any(rated & (depth >= zone[1]) & (depth <= zone[2]))]
    if not zones:
        zones = _class_intervals(depth, environment, rated, max_intervals,
                                 min_thickness=5.0 if well.depth_unit == "m" else 15.0)
    rows = []
    for name, top, bottom in zones:
        selection = rated & (depth >= top) & (depth <= bottom)
        count = int(np.count_nonzero(selection))
        if not count:
            continue
        labels = environment.environment[selection]
        dominant = max(TH_U_ENVIRONMENTS, key=lambda label: np.count_nonzero(labels == label))
        rows.append({
            "interval": name, "top": top, "bottom": bottom, "samples": count,
            "median_th_u": float(np.median(environment.th_u_ratio[selection])),
            "dominant": dominant,
            "dominant_%": 100.0 * np.count_nonzero(labels == dominant) / count,
        })
    return rows


def _class_intervals(depth: np.ndarray, environment: Any, rated: np.ndarray,
                     max_intervals: int, min_thickness: float) -> List[Tuple[str, float, float]]:
    """Depth intervals of one dominant Th/U class, from a running median of Th/U.

    Runs thinner than ``min_thickness`` are absorbed into a neighbour, and the
    thinnest runs are merged until at most ``max_intervals`` remain.
    """
    from petrophys.interpretation.geochemistry import TH_U_ENVIRONMENT_THRESHOLDS

    index = np.flatnonzero(rated)
    if index.size == 0:
        return []
    ratio = environment.th_u_ratio[index]
    window = max(3, int(round(min_thickness / max(float(np.median(np.diff(depth[index]))), 1e-6)
                              / 2)) | 1) if index.size > 1 else 1
    half = window // 2
    smooth = np.array([np.median(ratio[max(0, i - half):i + half + 1]) for i in range(ratio.size)])
    labels = np.digitize(smooth, np.asarray(TH_U_ENVIRONMENT_THRESHOLDS), right=True)
    runs: List[List[Any]] = []
    for position, label in enumerate(labels):
        if runs and runs[-1][2] == label:
            runs[-1][1] = position
        else:
            runs.append([position, position, label])

    def thickness(run: List[Any]) -> float:
        return float(depth[index[run[1]]] - depth[index[run[0]]])

    while len(runs) > 1:
        thin = min(range(len(runs)), key=lambda k: thickness(runs[k]))
        if thickness(runs[thin]) >= min_thickness and len(runs) <= max_intervals:
            break
        neighbours = [k for k in (thin - 1, thin + 1) if 0 <= k < len(runs)]
        target = max(neighbours, key=lambda k: thickness(runs[k]))
        start, end = min(runs[thin][0], runs[target][0]), max(runs[thin][1], runs[target][1])
        runs[target] = [start, end, runs[target][2]]
        del runs[thin]
        merged: List[List[Any]] = []
        for run in runs:
            if merged and merged[-1][2] == run[2]:
                merged[-1][1] = run[1]
            else:
                merged.append(run)
        runs = merged
    zones = []
    for start, end, _ in runs:
        top, bottom = float(depth[index[start]]), float(depth[index[end]])
        zones.append((f"{_fmt(top)}-{_fmt(bottom)}", top, bottom))
    return zones


# -- zone tables ----------------------------------------------------------------------

#: Columns of the per-formation table, in order (the depth unit is filled in).
ZONE_COLUMNS = ("Formation", "Rank", "Top", "Bottom", "Gross ({u})", "Net Reservoir ({u})",
                "Net Pay ({u})", "NTG", "Avg PHIE", "Avg Sw", "Avg VCL", "Avg Perm (mD)")


def zone_table(
    well: "Well",
    *,
    formations: Any = None,
    rank: Optional[str] = None,
    names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Per-formation table of an interpreted well, built on ``formation_summary``.

    Raises
    ------
    ValueError
        If no formations are given or attached, or ``names`` match none.
    """
    from petrophys.io.geology_reader import read_formations
    from petrophys.workflow.summary import formation_table

    if formations is None:
        records = (well.get_parameter("_geology_context") or {}).get("formations") or []
        if not records:
            raise ValueError(
                "No formations are attached to this well. Load it with formations (load_well "
                "formations=<file>, or load_sample_well, which attaches bundled tops), pass "
                "formations=<tops file>, or give top and bottom for one zone."
            )
        ranks = {str(row.get("formation_name")): row.get("rank") or "" for row in records}
    else:
        table = read_formations(formations, well=well)
        ranks = {}
        if "rank" in table and "formation_name" in table:
            ranks = dict(zip(table["formation_name"].astype(str), table["rank"].fillna("")))
    levels = {value for value in ranks.values() if value}
    if rank is not None and levels and rank not in levels:
        raise ValueError(f"rank={rank!r} matches no formations; ranks here: "
                         f"{', '.join(sorted(levels))}.")
    frame = formation_table(well, formations, rank=rank)
    unit = well.depth_unit
    columns = [column.format(u=unit) for column in ZONE_COLUMNS]
    frame = frame.copy()
    frame["Rank"] = [ranks.get(str(name), "") for name in frame["Formation"]]
    frame = frame[[column for column in columns if column in frame]]
    if names:
        wanted = [str(name).strip().lower() for name in names]
        keep = [any(token in str(label).lower() for token in wanted)
                for label in frame["Formation"]]
        if not any(keep):
            raise ValueError(
                f"No formation matches {list(names)}. Formations: "
                f"{', '.join(map(str, frame['Formation']))}."
            )
        frame = frame[keep]
    rows = frame.to_dict("records")
    data: Dict[str, Any] = {
        "depth_unit": unit,
        "zones": rows,
        "averages_basis": "Avg PHIE/Sw/VCL are means over net-pay samples (null where a zone "
                          "has no pay); Avg Perm is a geometric mean",
    }
    mixed = rank is None and len(levels) > 1 and not names
    if mixed:
        data["note"] = (
            f"Rows mix levels ({', '.join(sorted(levels))}): formations lie inside groups, so do "
            "not add them up. Pass rank='formation' or rank='group' for one level and totals."
        )
    elif rows:
        data["total"] = {
            "gross": float(frame[f"Gross ({unit})"].sum()),
            "net_pay": float(frame[f"Net Pay ({unit})"].sum()),
        }
    return compact_object(data)


# -- plot tracks ------------------------------------------------------------------------

#: Words a model may use for a track, mapped to words that appear in track names.
TRACK_SYNONYMS = {
    "GR": ("GR", "GAMMA"), "GAMMA": ("GAMMA", "GR"), "GAMMA RAY": ("GAMMA", "GR"),
    "VCL": ("VCL",), "VSH": ("VCL",), "CLAY": ("VCL",), "SHALE": ("VCL",),
    "CALI": ("CALI", "HOLE"), "CALIPER": ("CALI", "HOLE"), "HOLE": ("HOLE", "CALI"),
    "SP": ("SP",), "RES": ("RESISTIVITY",), "RT": ("RESISTIVITY",), "RDEP": ("RESISTIVITY",),
    "ILD": ("RESISTIVITY",), "RESISTIVITY": ("RESISTIVITY",),
    "RHOB": ("DENSITY",), "DEN": ("DENSITY",), "DENSITY": ("DENSITY",),
    "NPHI": ("NEUTRON",), "NEU": ("NEUTRON",), "NEUTRON": ("NEUTRON",),
    "DT": ("SONIC",), "DTC": ("SONIC",), "SONIC": ("SONIC",),
    "PHI": ("POROSITY",), "PHIE": ("POROSITY",), "PHIT": ("POROSITY",),
    "POROSITY": ("POROSITY",), "SW": ("SATURATION", "SW"), "SXO": ("SATURATION",),
    "SATURATION": ("SATURATION",), "PERM": ("PERMEABILITY",), "K": ("PERMEABILITY",),
    "PERMEABILITY": ("PERMEABILITY",), "LITH": ("LITHOLOGY",), "LITHO": ("LITHOLOGY",),
    "LITHOLOGY": ("LITHOLOGY",), "MINERALS": ("MINERAL",), "MINERAL": ("MINERAL",),
    "NET": ("FLAGS",), "PAY": ("FLAGS",), "NET_PAY": ("FLAGS",), "FLAGS": ("FLAGS",),
    "DEPTH": ("DEPTH",),
}


def resolve_tracks(requested: Sequence[str], available: Sequence[str]) -> List[str]:
    """Map loose track names ("GR", "res", "porosity") onto a template's track names.

    Matching ignores case. A name matches a track exactly, as one of the
    words between slashes ("GR" -> "GR / VCL"), through a synonym ("RHOB"
    -> "DENSITY / NEUTRON"), or as a prefix ("RES" -> "RESISTIVITY").

    Raises
    ------
    ValueError
        If a name matches no track; the message lists the valid names.
    """
    tracks = list(available)
    upper = [track.upper() for track in tracks]
    words = [{part.strip() for part in track.split("/")} for track in upper]
    chosen: List[str] = []
    unknown: List[str] = []
    for name in requested:
        key = str(name).strip().upper()
        if key not in TRACK_SYNONYMS:
            key = key.replace("_", " ")
        match = None
        if key in upper:
            match = tracks[upper.index(key)]
        if match is None:
            for index, parts in enumerate(words):
                if key in parts:
                    match = tracks[index]
                    break
        if match is None:
            for synonym in TRACK_SYNONYMS.get(key, ()):
                hits = [tracks[index] for index, parts in enumerate(words)
                        if any(part == synonym or part.startswith(synonym) for part in parts)]
                if hits:
                    match = hits[0]
                    break
        if match is None:
            hits = [tracks[index] for index, track in enumerate(upper)
                    if any(part.startswith(key) for part in words[index]) or key in track]
            match = hits[0] if hits else None
        if match is None:
            unknown.append(str(name))
        elif match not in chosen:
            chosen.append(match)
    if unknown:
        close = {name: difflib.get_close_matches(name.upper(), upper, n=1, cutoff=0.4)
                 for name in unknown}
        raise ValueError(
            f"Unknown track name(s) {unknown} for this template. Valid tracks: {tracks}. "
            "Short names work too, e.g. 'GR', 'RES', 'POROSITY', 'SW'."
            + "".join(f" {name!r} -> maybe {tracks[upper.index(match[0])]!r}."
                      for name, match in close.items() if match)
        )
    return chosen


def find_zone(well: "Well", name: str) -> Tuple[str, float, float]:
    """The attached formation whose name matches ``name`` (case-insensitive, partial).

    Raises
    ------
    ValueError
        If no attached formation matches.
    """
    records = (well.get_parameter("_geology_context") or {}).get("formations") or []
    zones = [(str(row.get("formation_name")), row.get("top_depth"), row.get("bottom_depth"),
              row.get("rank") or "") for row in records]
    zones = [zone for zone in zones if zone[1] is not None and zone[2] is not None]
    if not zones:
        raise ValueError("No formations with depths are attached to this well; give top and "
                         "bottom instead.")
    key = name.strip().lower()
    exact = [zone for zone in zones if zone[0].lower() == key]
    partial = [zone for zone in zones if key in zone[0].lower()]
    matches = exact or sorted(partial, key=lambda zone: zone[3] != "formation")
    if not matches:
        raise ValueError(
            f"No attached formation matches {name!r}. Formations: "
            f"{', '.join(zone[0] for zone in zones)}."
        )
    label, top, bottom, _ = matches[0]
    return label, float(top), float(bottom)
