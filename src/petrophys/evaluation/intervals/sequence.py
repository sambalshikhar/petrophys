"""
Vertical stacking patterns of zones: read from base to top, steps of a
shale-indicating curve (GR, VCL) in one direction form a package. Decreasing
upward is cleaning-up (coarsening-upward); increasing upward is dirtying-up
(fining-upward). A single step is an abrupt change across a contact, not a
graded package. Steps smaller than a threshold count as no change, so weak
trends are not called sequences.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

# The smallest step that counts, per curve: a GR change of a few API or a VCL change of a few percent is noise.
MIN_STEP = {"GR": 3.0, "VCL": 0.03}


def stacking_pattern(
    values: Sequence[float],
    mnemonic: Optional[str] = None,
    min_change: Optional[float] = None,
) -> Dict[str, object]:
    """Steps and packages of zone values read from base to top.

    Parameters
    ----------
    values : sequence of float
        One value per zone (e.g. zone-mean GR), base zone first.
    mnemonic : str, optional
        The curve, for its smallest counting step (:data:`MIN_STEP`).
    min_change : float, optional
        The smallest change that counts as a step. Default: 10% of the range of
        the values, at least the curve's :data:`MIN_STEP`.

    Returns
    -------
    dict
        ``threshold``; ``changes`` and ``steps`` per zone (the change from the
        zone below, and ``"cleaner"``, ``"dirtier"``, ``"no change"`` or ``""``
        for the base zone or a missing value); ``packages``, each ``start`` and
        ``end`` (zone indices, base first) and ``kind``: ``"cleaning_up"``,
        ``"dirtying_up"`` (two or more steps in one direction),
        ``"abrupt_decrease"`` or ``"abrupt_increase"`` (one step).

    Examples
    --------
    >>> p = stacking_pattern([90.0, 70.0, 50.0, 52.0, 85.0], "GR")
    >>> p["steps"], [(q["start"], q["end"], q["kind"]) for q in p["packages"]]
    (['', 'cleaner', 'cleaner', 'no change', 'dirtier'], [(0, 2, 'cleaning_up'), (3, 4, 'abrupt_increase')])
    """
    v = np.asarray(values, dtype=float)
    fin = v[np.isfinite(v)]
    floor = MIN_STEP.get(str(mnemonic or "").upper(), 0.0)
    thr = float(min_change) if min_change is not None else max(0.1 * (fin.max() - fin.min()) if fin.size else 0.0, floor)
    changes: List[float] = []
    steps: List[str] = []
    for i in range(v.size):
        ch = v[i] - v[i - 1] if i and np.isfinite(v[i]) and np.isfinite(v[i - 1]) else float("nan")
        changes.append(float(ch))
        steps.append("" if i == 0 or not np.isfinite(ch) else
                     "cleaner" if ch <= -thr else "dirtier" if ch >= thr else "no change")
    packages: List[Dict[str, object]] = []
    k = 1
    while k < v.size:
        if steps[k] in ("cleaner", "dirtier"):
            j = k
            while j + 1 < v.size and steps[j + 1] == steps[k]:
                j += 1
            if j > k:
                kind = "cleaning_up" if steps[k] == "cleaner" else "dirtying_up"
            else:
                kind = "abrupt_decrease" if steps[k] == "cleaner" else "abrupt_increase"
            packages.append({"start": k - 1, "end": j, "kind": kind})
            k = j + 1
        else:
            k += 1
    return {"threshold": thr, "changes": changes, "steps": steps, "packages": packages}


__all__ = ["MIN_STEP", "stacking_pattern"]
