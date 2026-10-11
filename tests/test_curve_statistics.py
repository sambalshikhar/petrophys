"""curve_statistics: the per-curve statistics of the reference context, public and unrounded. The exporter's payload
is now these rounded, and must be exactly what it was."""
import numpy as np
import pytest

from petrophys.evaluation.summaries import curve_statistics, percentile_key
from petrophys.evaluation.summaries.statistics import STATISTICS
from petrophys.reporting.reference_export import _curve_statistics_payload, _normalize_numeric_array, _round_stat


def _payload_before(data):
    """The exporter's payload as written before curve_statistics existed (frozen copy)."""
    arr = _normalize_numeric_array(data)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    positive = finite[finite > 0]
    rounded = np.round(finite, 4)
    unique, counts = np.unique(rounded, return_counts=True)
    mode_value = float(unique[int(np.argmax(counts))]) if unique.size else float(finite[0])
    if positive.size:
        geometric_mean = float(np.exp(np.nanmean(np.log(positive))))
        harmonic_mean = float(positive.size / np.sum(1.0 / positive))
    else:
        geometric_mean = 0.0
        harmonic_mean = 0.0
    third_power_average = float(np.cbrt(np.nanmean(np.power(finite, 3))))
    return {
        "minimum": _round_stat(np.nanmin(finite)),
        "maximum": _round_stat(np.nanmax(finite)),
        "arithmetic_mean": _round_stat(np.nanmean(finite)),
        "geometric_mean": _round_stat(geometric_mean),
        "harmonic_mean": _round_stat(harmonic_mean),
        "third_power_average": _round_stat(third_power_average),
        "median": _round_stat(np.nanmedian(finite)),
        "mode": _round_stat(mode_value),
        "standard_deviation": _round_stat(np.nanstd(finite)),
    }


def _cases():
    rng = np.random.default_rng(7)
    yield rng.normal(0.2, 0.05, 500)
    yield np.where(rng.random(400) < 0.2, np.nan, rng.lognormal(1.0, 2.0, 400))   # PERM-like with gaps
    yield rng.normal(0.0, 1.0, 300)                                               # negatives: no geometric mean
    yield np.zeros(10)
    yield np.array([3.0])
    yield np.array([1, 2, 2, 3], dtype=int)
    yield np.array(["0.1", "x", None, 0.3], dtype=object)
    yield np.full(5, np.nan)
    yield np.array([])


def test_payload_unchanged():
    for data in _cases():
        assert _curve_statistics_payload(data) == _payload_before(data)


def test_statistics_unrounded_with_count_variance_and_percentiles():
    x = np.array([0.12345, 0.2, np.nan, 0.31, -0.05])
    s = curve_statistics(x, percentiles=[10, 50, 2.5])
    f = x[np.isfinite(x)]
    assert s["count"] == 4 and list(s)[1:10] == list(STATISTICS)
    assert s["arithmetic_mean"] == pytest.approx(f.mean(), abs=0) and s["minimum"] == -0.05
    assert s["geometric_mean"] == pytest.approx(np.exp(np.log(f[f > 0]).mean()))
    assert s["variance"] == pytest.approx(f.var()) and s["standard_deviation"] == pytest.approx(f.std())
    assert s["p10"] == pytest.approx(np.percentile(f, 10)) and "p2.5" in s and percentile_key(50) == "p50"


def test_no_finite_sample_and_bad_percentile():
    assert curve_statistics([np.nan, None]) is None
    with pytest.raises(ValueError, match="outside 0-100"):
        curve_statistics([1.0, 2.0], percentiles=[150])
