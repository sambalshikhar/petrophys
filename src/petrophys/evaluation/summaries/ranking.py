"""
Which zone is best and which weakest for a property: the direction in which a
property is better (more porosity, permeability and pay is better; more water,
clay and bad hole worse) and a ranking that leaves out zones in bad hole.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

# Curves for which a higher zone value is the better reservoir, and a lower one.
HIGHER_IS_BETTER = ("PHIE", "PHIE_CC", "PHIT", "PERM", "PERM_CC", "FZI", "RQI", "NET_PAY", "NET_RES", "RDEP")
LOWER_IS_BETTER = ("SW", "VCL", "BVW", "BAD_HOLE")


def property_direction(mnemonic: str) -> Optional[str]:
    """``"higher"`` or ``"lower"`` (the better direction of a curve), None when it has none.

    Examples
    --------
    >>> property_direction("PHIE"), property_direction("sw"), property_direction("GR")
    ('higher', 'lower', None)
    """
    name = str(mnemonic).upper()
    return "higher" if name in HIGHER_IS_BETTER else "lower" if name in LOWER_IS_BETTER else None


def best_and_weakest(
    values: Sequence[float],
    direction: str,
    *,
    bad_hole: Optional[Sequence[float]] = None,
    max_bad_hole_fraction: float = 0.5,
) -> Dict[str, object]:
    """The best and the weakest of zone values, in a property's better direction.

    Zones whose bad-hole share is above ``max_bad_hole_fraction`` are left out;
    when that leaves none, every zone is ranked (and ``basis`` says so). Fewer
    than two zones are not ranked: one zone is not both best and weakest.

    Parameters
    ----------
    values : sequence of float
        One value per zone (e.g. zone means); non-finite values are not ranked.
    direction : {"higher", "lower"}
        The better direction (:func:`property_direction`).
    bad_hole : sequence of float, optional
        Bad-hole share per zone.
    max_bad_hole_fraction : float
        The largest bad-hole share a ranked zone may have.

    Returns
    -------
    dict
        ``best`` and ``weakest`` (zone indices, None when not ranked) and
        ``basis``: ``"good_hole"``, ``"all_in_bad_hole"`` (every zone with a
        value is in bad hole, so all are ranked) or ``"too_few"``.

    Examples
    --------
    >>> best_and_weakest([0.2, 0.1, 0.3], "higher", bad_hole=[0.0, 0.0, 0.9])
    {'best': 0, 'weakest': 1, 'basis': 'good_hole'}
    """
    if direction not in ("higher", "lower"):
        raise ValueError(f"direction must be 'higher' or 'lower', not {direction!r}")
    ok = [i for i, v in enumerate(values) if v is not None and np.isfinite(v)]
    good = [i for i in ok if bad_hole is None or bad_hole[i] <= max_bad_hole_fraction]
    pick = good or ok
    basis = "good_hole" if good else "all_in_bad_hole"
    if len(pick) < 2:
        return {"best": None, "weakest": None, "basis": "too_few"}
    better, worse = (max, min) if direction == "higher" else (min, max)
    return {"best": better(pick, key=lambda i: values[i]), "weakest": worse(pick, key=lambda i: values[i]),
            "basis": basis}


__all__ = ["HIGHER_IS_BETTER", "LOWER_IS_BETTER", "best_and_weakest", "property_direction"]
