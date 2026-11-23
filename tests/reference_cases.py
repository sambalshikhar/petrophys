"""Interpretations of bundled wells whose numbers the reference test holds fixed.

Each scenario runs ``pp.interpret`` on a bundled sample and reduces the run to a
signature: the steps and methods used, net thickness, statistics of every
curve and, for the benchmark profiles, the exported context. The stored
signatures live in ``tests/reference/``; ``python tests/make_reference.py``
rewrites them after an intended change.
"""

import warnings
from pathlib import Path

import _support  # noqa: F401  (import path and plotting backend)
import numpy as np

import petrophys as pp

REFERENCE_DIR = Path(__file__).resolve().parent / "reference"

S = pp.sample


def _30_6_5():
    return pp.interpret(S("30_6-5"), formations=S("formations_30_6-5"), rw=0.07, rw_meas_temp_f=230)


def _30_6_5_geolink():
    return pp.interpret(S("30_6-5"), calculation_profile="geolink")


def _hw3_costa():
    return pp.interpret(S("HW-3"), context_profile="costa")


SCENARIOS = {
    "30_6-5": _30_6_5,
    "30_6-5_geolink_profile": _30_6_5_geolink,
    "HW-3_costa_profile": _hw3_costa,
    "25_5-1": lambda: pp.interpret(S("25_5-1")),
    "16_7-1": lambda: pp.interpret(S("16_7-1")),
    "15_9-14": lambda: pp.interpret(S("15_9-14")),
    "cheal-1": lambda: pp.interpret(S("cheal-1"), formations=S("formations_cheal-1")),
    "ede_01_composite": lambda: pp.interpret(S("ede_01_composite")),
    "teapot_17-wx-21": lambda: pp.interpret(S("teapot_17-wx-21")),
}

# Scenarios whose exported context is held fixed too (the benchmark profiles).
CONTEXT_SCENARIOS = {"30_6-5_geolink_profile", "HW-3_costa_profile"}


def _stats(values):
    data = np.asarray(values, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return {"finite": 0}
    p10, p50, p90 = np.percentile(finite, [10, 50, 90])
    return {"finite": int(finite.size), "mean": float(finite.mean()), "p10": float(p10),
            "p50": float(p50), "p90": float(p90), "min": float(finite.min()), "max": float(finite.max())}


def _plain(value):
    """JSON-safe copy: numpy scalars to Python, NaN and infinities to None, other objects to text."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)


def signature(name, result):
    """Reduce an interpretation to the numbers and choices the reference holds fixed."""
    summary = result.summary()
    well = result.well
    curves = {}
    for curve in sorted(well.curves, key=str):
        values = np.asarray(well[curve].data if hasattr(well[curve], "data") else well[curve])
        if values.dtype.kind in "fiub":
            curves[str(curve)] = _stats(values)
    out = {
        "status": result.status,
        "lithology_domain": summary.get("lithology_domain"),
        "resolved": summary.get("resolved"),
        "steps": [[step["name"], step["method"], bool(step["success"])] for step in summary["executed_steps"]],
        "skipped_steps": sorted(summary.get("skipped_steps") or {}),
        "net_thickness": summary.get("net_thickness"),
        "depth": _stats(well.depth),
        "curves": curves,
    }
    if name in CONTEXT_SCENARIOS:
        out["context"] = pp.export_context(result)
    return _plain(out)


def run(name):
    """Run one scenario quietly and return its signature."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = SCENARIOS[name]()
        return signature(name, result)


def reference_path(name):
    return REFERENCE_DIR / f"{name}.json"
