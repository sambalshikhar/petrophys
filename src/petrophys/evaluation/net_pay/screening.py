"""
Screening confidence of net-pay intervals, and which net-pay cutoff removes how
much rock.

A cutoff flag such as ``NET_PAY`` is a screen, not a pay call: an interval in
rock that cannot hold producible hydrocarbons, in washed-out hole or thinner
than the tools resolve is an anomaly. :func:`grade_interval` says which, and
:func:`cutoff_attrition` says which cutoff (shale volume, porosity, saturation)
removes the rock that is not pay.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

MISSING_LABELS = ("NO_LITHOLOGY_DATA", "", "NAN", "NONE")


def lithology_in(label: str, names: Iterable[str]) -> bool:
    """Whether a lithology label is one of the listed rock types.

    A label belongs to a type when it is that type or its head noun is (the last
    word names the rock, earlier words modify it): ``CALCAREOUS SHALE`` is a
    shale and ``SHALY SANDSTONE`` a sandstone, so one list covers any naming
    scheme's modifiers.

    Examples
    --------
    >>> lithology_in("calcareous shale", ["SHALE"]), lithology_in("SHALY SANDSTONE", ["SHALE"])
    (True, False)
    """
    names = {str(n).upper() for n in names}
    label = str(label).upper().strip()
    return label in names or (label.split()[-1] in names if label else False)


def majority_label(labels: Sequence[str], missing: Iterable[str] = MISSING_LABELS) -> str:
    """The most frequent label, ties broken alphabetically; ``"UNKNOWN"`` when every label is missing."""
    skip = {str(m).upper() for m in missing}
    kept = [str(label) for label in labels if str(label).upper() not in skip]
    if not kept:
        return "UNKNOWN"
    counts = {label: kept.count(label) for label in set(kept)}
    return sorted(counts, key=lambda label: (-counts[label], label))[0]


def grade_interval(
    lithology: str,
    bad_hole_fraction: Optional[float],
    thickness: float,
    *,
    non_reservoir_lithologies: Iterable[str] = (),
    marginal_lithologies: Iterable[str] = (),
    max_bad_hole_fraction: float = 0.5,
    min_thickness: float = 1.0,
) -> Tuple[str, List[str]]:
    """Screening confidence of a flagged interval.

    Parameters
    ----------
    lithology : str
        The interval's lithology (e.g. :func:`majority_label` of its samples).
    bad_hole_fraction : float or None
        Share of the interval's samples in bad hole (None: not known).
    thickness : float
        Interval thickness, in the unit of ``min_thickness``.
    non_reservoir_lithologies, marginal_lithologies : iterable of str
        Rock types that cannot hold producible hydrocarbons, and fine-grained
        rock where a log-derived low SW is often a clay effect.
    max_bad_hole_fraction : float
        Bad-hole share above which the interval is low confidence.
    min_thickness : float
        Thickness below which the interval is below tool resolution.

    Returns
    -------
    confidence : str
        ``"higher"`` or ``"low"``.
    reasons : list of str
        Why it is low (empty when higher).

    Examples
    --------
    >>> grade_interval("SILT", 0.1, 2.0, marginal_lithologies=["SILT"])
    ('low', ['silt is marginal reservoir rock'])
    """
    reasons = []
    if lithology_in(lithology, non_reservoir_lithologies):
        reasons.append(f"{lithology.lower()} is not reservoir rock")
    elif lithology_in(lithology, marginal_lithologies):
        reasons.append(f"{lithology.lower()} is marginal reservoir rock")
    if bad_hole_fraction is not None and bad_hole_fraction > max_bad_hole_fraction:
        reasons.append(f"bad hole {100 * bad_hole_fraction:.0f}%")
    if thickness < min_thickness:
        reasons.append(f"thinner than {min_thickness} m")
    return ("low" if reasons else "higher"), reasons


def cutoff_attrition(net_rock, net_reservoir, net_pay) -> List[dict]:
    """Which net-pay cutoff removes how much rock, in order.

    The shale-volume cutoff removes what is not net rock, the porosity cutoff
    net rock that is not net reservoir, the saturation cutoff net reservoir that
    is not net pay. A pay verdict names the first cutoff that fails.

    Parameters
    ----------
    net_rock, net_reservoir, net_pay : array-like
        The flags over the same samples (above 0.5 = set).

    Returns
    -------
    list of dict
        One per cutoff applied while rock remains: ``cutoff`` (``shale_volume``,
        ``porosity``, ``saturation``), ``fails`` (the share of the rock it tested
        that fails it) and ``tested`` (samples tested). Empty without samples.

    Examples
    --------
    >>> [round(s["fails"], 2) for s in cutoff_attrition([1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0])]
    [0.25, 0.33, 0.5]
    """
    rock, res, pay = (np.asarray(f, dtype=float) > 0.5 for f in (net_rock, net_reservoir, net_pay))
    n = rock.size
    if n == 0:
        return []
    steps = [("shale_volume", ~rock, n), ("porosity", rock & ~res, int(rock.sum())),
             ("saturation", res & ~pay, int(res.sum()))]
    out = []
    for cutoff, fails, tested in steps:
        if tested == 0:
            break
        out.append({"cutoff": cutoff, "fails": float(fails.sum()) / tested, "tested": tested})
    return out


__all__ = ["cutoff_attrition", "grade_interval", "lithology_in", "majority_label"]
