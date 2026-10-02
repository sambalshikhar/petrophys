"""Conservative formation-pressure picks from overbalanced MDT pretest traces.

One stationary probe setting per trace, with mud pressure recorded before and
after the test. Repeated drawdown/buildup cycles are supported. This is a
plateau picker, not pressure-transient extrapolation or a seal-state decoder.
"""
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import find_peaks


_PBAR = {"bar": 1.0, "psi": 0.06894757293168, "kPa": 0.01, "MPa": 10.0}
_DM = {"m": 1.0, "ft": 0.3048}
_TS = {"s": 1.0, "ms": 0.001, "min": 60.0}


@dataclass(frozen=True)
class MDTPickingConfig:
    """Analysis thresholds in the units in each name, not gauge specifications.

    Plateau tolerance limits both 5th-to-95th percentile pressure spread and
    fitted pressure change over the final tail. The preceding tail must agree.
    Review defaults for acquisition resolution and measurement precision.
    """

    edge_seconds: float = 15.0
    tail_seconds: float = 30.0
    guard_seconds: float = 3.0
    min_drawdown_bar: float = 0.5
    min_recovery_bar: float = 0.2
    plateau_tolerance_bar: float = 0.05
    repeat_tolerance_bar: float = 0.1
    mud_agreement_bar: float = 2.0
    max_depth_drift_m: float = 0.25
    max_gap_seconds: float = 5.0

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"{f.name} must be finite and positive")


@dataclass(frozen=True)
class MDTBuildup:
    """One detected recovery; pressures retain input units, times are seconds."""

    minimum_time: float
    window_start: float
    window_end: float
    candidate_pressure: float
    drift_pressure_per_second: float
    pressure_spread: float
    depth_span_m: float
    status: str
    flags: tuple


@dataclass(frozen=True)
class MDTPressurePick:
    """Auditable automatic pick. Only status='accepted' supplies pressure.

    A review candidate is diagnostic and must not be used as formation pressure
    automatically. Acceptance means a repeatable stable plateau under the
    configured rules, not proof of reservoir equilibrium or absence of invasion.
    """

    status: str
    pressure: object
    candidate_pressure: object
    station_depth: object
    pressure_unit: str
    depth_unit: str
    depth_reference: str
    source: str
    mud_before: object
    mud_after: object
    buildups: tuple
    selected_buildup: object
    flags: tuple
    config: MDTPickingConfig


@dataclass
class MDTPretest:
    """One probe setting, potentially including repeated pretests.

    The probe-depth curve is MD unless an explicit TVD reference is supplied.
    No well trajectory or datum conversion is inferred from the trace.
    """

    time: np.ndarray
    pressure: np.ndarray
    depth: object = None
    time_unit: str = "s"
    pressure_unit: str = "bar"
    depth_unit: str = "m"
    depth_reference: str = "MD"
    source: str = ""

    def pick_pressure(self, config=None):
        """Detect drawdown/recovery and return a stable pre-retraction pick."""
        return pick_mdt_pressure(
            self.time, self.pressure, depth=self.depth, time_unit=self.time_unit,
            pressure_unit=self.pressure_unit, depth_unit=self.depth_unit,
            depth_reference=self.depth_reference, source=self.source, config=config,
        )


def pick_mdt_pressure(time, pressure, *, depth=None, time_unit="s",
                      pressure_unit="bar", depth_unit="m", depth_reference="MD",
                      source="", config=None):
    """Pick a stabilized buildup from one raw overbalanced probe setting.

    Algorithm: establish mud baselines at both ends; find the pressure excursion
    below mud; detect drawdown minima; inspect the final two tail windows of
    each recovery before the next drawdown or probe retraction. Require small
    drift/spread, actual recovery, stationary probe depth (when present), and
    agreement between accepted repeat buildups. Use the last stable buildup.

    Returns review (pressure=None) for missing/gapped data, incomplete tests,
    inconsistent repeats, drifting recoveries or no clear pressure response.
    No report pick, fluid code or contact is an input. No asymptotic pressure
    is extrapolated from incomplete buildups. Pressure-only data cannot prove
    a good seal, identify valve states or correct supercharging: automatic
    acceptance is a QC proposal. Underbalanced, pumping and multistation traces
    are outside this picker. Supply one complete probe setting at a time.
    """
    cfg = config if config is not None else MDTPickingConfig()
    if not isinstance(cfg, MDTPickingConfig):
        raise TypeError("config must be MDTPickingConfig")
    if time_unit not in _TS or pressure_unit not in _PBAR or depth_unit not in _DM:
        raise ValueError("Use time s/ms/min, pressure bar/psi/kPa/MPa, depth m/ft")
    if depth_reference not in {"MD", "TVDSS", "TVD_MSL", "TVD_RKB", "TVD_RT"}:
        raise ValueError("depth_reference must be MD or an explicit positive-down TVD datum")
    t = np.asarray(time, dtype=float)*_TS[time_unit]
    original = np.asarray(pressure, dtype=float)
    if t.ndim != 1 or original.shape != t.shape:
        raise ValueError("time and pressure must be matching 1-D arrays")
    z = None if depth is None else np.asarray(depth, dtype=float)
    if z is not None and z.shape != t.shape:
        raise ValueError("depth must match the time series")
    factor = _PBAR[pressure_unit]
    p = original*factor

    def output(flags, cycles=(), selected=None, before=None, after=None):
        candidate = cycles[-1].candidate_pressure if cycles else None
        station_depth = None
        if cycles and z is not None:
            cycle = cycles[-1] if selected is None else cycles[selected]
            zz = z[(t >= cycle.window_start) & (t <= cycle.window_end)]
            if len(zz) and np.isfinite(zz).all():
                station_depth = float(np.median(zz))
        accepted = selected is not None and not flags
        return MDTPressurePick(
            "accepted" if accepted else "review",
            cycles[selected].candidate_pressure if accepted else None,
            candidate, station_depth, pressure_unit, depth_unit, depth_reference,
            str(source), before, after, tuple(cycles), selected, tuple(flags), cfg,
        )

    if len(t) < 10 or not np.isfinite(t).all() or not np.isfinite(p).all():
        return output(["insufficient_or_nonfinite_time_pressure"])
    if np.any(np.diff(t) <= 0):
        raise ValueError("Acquisition time must increase strictly; do not sort or merge pretests")
    dt = float(np.median(np.diff(t)))
    if np.max(np.diff(t)) > max(cfg.max_gap_seconds, 5*dt):
        return output(["time_gap_requires_review"])
    if t[-1]-t[0] < 2*cfg.edge_seconds+2*cfg.tail_seconds:
        return output(["trace_too_short_for_baselines_and_recovery"])
    first = p[t <= t[0]+cfg.edge_seconds]
    last = p[t >= t[-1]-cfg.edge_seconds]
    before, after = float(np.median(first)), float(np.median(last))
    baseline = dict(before=before/factor, after=after/factor)
    if abs(before-after) > cfg.mud_agreement_bar:
        return output(["mud_baselines_disagree_or_incomplete_trace"], **baseline)
    # A stationary mud baseline may contain acquisition noise. Robust range is
    # checked separately from the stricter formation-plateau precision.
    if any(np.ptp(np.quantile(edge, [.1, .9])) > cfg.min_drawdown_bar for edge in (first, last)):
        return output(["unstable_mud_baseline"], **baseline)
    smooth = median_filter(p, size=3, mode="nearest")
    below = smooth < min(before, after)-cfg.min_drawdown_bar
    changes = np.diff(np.r_[False, below, False].astype(int))
    runs = [(a, b) for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))
            if t[b-1]-t[a] >= 2*cfg.tail_seconds]
    if len(runs) != 1:
        return output(["no_clear_drawdown_or_seal" if not runs else "multiple_probe_settings_or_mud_returns"], **baseline)
    a, b = runs[0]
    if t[a]-t[0] < cfg.edge_seconds or t[-1]-t[b-1] < cfg.edge_seconds:
        return output(["incomplete_mud_bracketing"], **baseline)
    minima, _ = find_peaks(-smooth, prominence=cfg.min_drawdown_bar)
    chosen = [int(i) for i in minima if a <= i < b]
    if not chosen:
        return output(["no_resolved_drawdown_recovery"], **baseline)
    cycles = []
    for j, minimum in enumerate(chosen):
        # A deeper repeat drawdown can have its prominence base before the
        # first pretest. Bound the intervening plateau by adjacent minima.
        stop = (b-1 if j == len(chosen)-1 else
                minimum+int(np.argmax(smooth[minimum:chosen[j+1]])))
        if stop <= minimum:
            continue
        # Remove the beginning of a sharp late pressure rise on retraction;
        # otherwise it may contaminate the pressure window below the mud cutoff.
        late = np.flatnonzero((t >= t[stop]-cfg.edge_seconds) & (t < t[stop])
                              & (t > t[minimum]+2*cfg.tail_seconds))
        jumps = late[np.diff(smooth)[late] > cfg.min_drawdown_bar]
        if len(jumps):
            stop = int(jumps[0])
        end = t[stop]-cfg.guard_seconds
        tail = np.flatnonzero((t >= end-cfg.tail_seconds) & (t <= end) & (t > t[minimum]))
        previous = np.flatnonzero((t >= end-2*cfg.tail_seconds) & (t < end-cfg.tail_seconds)
                                 & (t > t[minimum]))
        flags = []
        if len(tail) < 5 or len(previous) < 5 or end-t[minimum] < 2*cfg.tail_seconds:
            # Do not manufacture a numerical pick from an unresolved recovery.
            continue
        level = float(np.median(p[tail]))
        drift = float(np.polyfit(t[tail]-t[tail].mean(), p[tail]-level, 1)[0])
        spread = float(np.ptp(np.quantile(p[tail], [.05, .95])))
        if abs(drift)*cfg.tail_seconds > cfg.plateau_tolerance_bar:
            flags.append("buildup_still_drifting")
        if spread > cfg.plateau_tolerance_bar:
            flags.append("unstable_pressure_plateau")
        if abs(level-np.median(p[previous])) > cfg.plateau_tolerance_bar:
            flags.append("successive_tail_windows_disagree")
        if level-smooth[minimum] < cfg.min_recovery_bar:
            flags.append("insufficient_pressure_recovery")
        depth_span = float("nan")
        if z is not None:
            # Check the whole recovery, not just the stable tail.
            zz = z[(t >= t[minimum]) & (t <= end)]
            if not np.isfinite(zz).all():
                flags.append("missing_probe_depth")
            else:
                depth_span = float(np.ptp(zz)*_DM[depth_unit])
                if depth_span > cfg.max_depth_drift_m:
                    flags.append("probe_moved_during_recovery")
        cycles.append(MDTBuildup(float(t[minimum]), float(t[tail[0]]), float(t[tail[-1]]),
                                level/factor, drift/factor, spread/factor, depth_span,
                                "accepted" if not flags else "review", tuple(flags)))
    if not cycles:
        return output(["no_sufficient_buildup_window"], **baseline)
    flags = list(cycles[-1].flags)
    # A final drawdown without a usable recovery must not silently fall back
    # to an earlier stable cycle.
    if cycles[-1].minimum_time != t[chosen[-1]]:
        flags.append("final_recovery_incomplete")
    stable = [c.candidate_pressure*factor for c in cycles if c.status == "accepted"]
    if len(stable) > 1 and np.ptp(stable) > cfg.repeat_tolerance_bar:
        flags.append("repeat_pretests_disagree")
    return output(flags, cycles, len(cycles)-1 if not flags else None, **baseline)


def read_mdt_pretest(source, *, time_column=None, pressure_column=None,
                     depth_column=None, time_unit=None, pressure_unit=None,
                     depth_unit=None, depth_reference="MD", csv_kwargs=None):
    """Read one raw MDT pretest from LAS/CSV, lasio.LASFile or DataFrame.

    The inspected native schema TIME/PQQP1/PQDVP1 is recognized. Custom columns
    are configurable; ambiguous pressure channels require explicit selection.
    Units come from LAS headers or explicit arguments/CSV names (TIME_S,
    PRES_BAR, DEPTH_M, etc.). Probe depth defaults to MD, never assumed TVD.
    """
    import lasio

    las = None
    label = "in-memory pretest"
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    else:
        if isinstance(source, lasio.LASFile):
            las = source
        else:
            path = Path(source)
            label = path.name
            if path.suffix.lower() == ".las":
                las = lasio.read(str(path))
            elif path.suffix.lower() == ".csv":
                frame = pd.read_csv(path, **(csv_kwargs or {}))
            else:
                raise ValueError("Supply a LAS/CSV path, LASFile or DataFrame")
        if las is not None:
            frame = las.df().reset_index()
    columns = {str(c).upper(): c for c in frame.columns}
    if len(columns) != len(frame.columns):
        raise ValueError("Column names must be unique (case insensitive)")

    def choose(explicit, aliases, purpose, optional=False):
        matches = [columns[c] for c in aliases if c in columns]
        if explicit is not None:
            if str(explicit).upper() not in columns:
                raise ValueError(f"Missing {purpose} column {explicit}")
            return columns[str(explicit).upper()]
        if optional and not matches:
            return None
        if len(matches) != 1:
            raise ValueError(f"Specify {purpose}_column; found {matches}")
        return matches[0]

    tc = choose(time_column, ["TIME", "TIME_S", "TIME_MS", "TIME_MIN", "ETIM"], "time")
    pc = choose(pressure_column, ["PQQP1", "PRES", "PRESSURE", "PRES_BAR", "PRES_PSI",
                                 "PRES_KPA", "PRES_MPA"], "pressure")
    dc = choose(depth_column, ["PQDVP1", "DEPTH", "DEPT", "DEPTH_M", "DEPTH_FT"], "depth", True)
    aliases = {"s": "s", "sec": "s", "ms": "ms", "min": "min", "m": "m",
               "ft": "ft", "psi": "psi", "bar": "bar", "kpa": "kPa", "mpa": "MPa"}

    def unit(explicit, col, valid):
        text = explicit
        if text is None and las is not None and col is not None:
            text = las.curves[col].unit
        if text is None and col is not None:
            text = str(col).rsplit("_", 1)[-1]
        value = aliases.get(str(text).strip().lower())
        if value not in valid:
            raise ValueError(f"Specify unit for {col}: {', '.join(valid)}")
        return value

    tu, pu = unit(time_unit, tc, _TS), unit(pressure_unit, pc, _PBAR)
    du = unit(depth_unit, dc, _DM) if dc is not None else (depth_unit or "m")
    return MDTPretest(frame[tc].to_numpy(dtype=float), frame[pc].to_numpy(dtype=float),
                      None if dc is None else frame[dc].to_numpy(dtype=float),
                      tu, pu, du, depth_reference, label)


def formation_pressures_from_mdt(picks, *, tvd=None, depth_unit=None,
                                depth_reference=None):
    """Bridge picks to unlabeled station data; review candidates stay excluded.

    Supply one TVD per input trace (including rejected traces) and its unit/datum
    unless all picks already have an explicit common TVD reference. MD is never
    used as TVD implicitly. Mixed pressure units are normalized to bar. Raw
    probe depths and review flags are retained; pressure is null for review rows.
    Reservoir/compartment and acquisition-reference selection remains the caller's
    responsibility before running infer_contacts().
    """
    from .formation_pressure import read_formation_pressure

    picks = list(picks)
    if not picks:
        raise ValueError("Supply at least one MDT pressure pick")
    if tvd is None:
        if any(p.depth_reference == "MD" for p in picks):
            raise ValueError("Provide TVD depths and a datum; probe MD is not automatically TVD")
        references = {p.depth_reference for p in picks}
        units = {p.depth_unit for p in picks}
        if len(references) != 1 or len(units) != 1:
            raise ValueError("Mixed depth metadata: supply consistent TVD depths explicitly")
        native_ref, native_unit = next(iter(references)), next(iter(units))
        if depth_reference not in {None, native_ref} or depth_unit not in {None, native_unit}:
            raise ValueError("Changing a depth datum/unit requires explicit converted TVD values")
        depth_reference, depth_unit = native_ref, native_unit
        tvd = [p.station_depth for p in picks]
    elif depth_reference is None or depth_unit is None:
        raise ValueError("Provide depth_reference and depth_unit for supplied TVD values")
    depths = np.asarray(tvd, dtype=float)
    if depths.shape != (len(picks),):
        raise ValueError("Supply one TVD per pressure pick, including review rows")
    frame = pd.DataFrame({
        "TVD": depths,
        "PRES_BAR": [np.nan if p.pressure is None else p.pressure*_PBAR[p.pressure_unit] for p in picks],
        "USE_FIT": [p.status == "accepted" for p in picks],
        "PICK_STATUS": [p.status for p in picks],
        "QC_FLAGS": [";".join(p.flags) for p in picks],
        "SOURCE": [p.source for p in picks],
        "PROBE_DEPTH": [p.station_depth for p in picks],
        "PROBE_DEPTH_UNIT": [p.depth_unit for p in picks],
    })
    return read_formation_pressure(frame, depth_column="TVD", depth_unit=depth_unit,
                                    depth_reference=depth_reference)
