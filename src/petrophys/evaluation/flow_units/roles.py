"""
Flow roles of zones from their mean permeability: the strongest flow interval,
the largest flow capacity (permeability-thickness) and possible baffles, with
the qualifiers that make a role a weaker statement.

The standard flow-unit reading of a permeability track: zones are ranked by
mean permeability; a zone is a possible baffle when it is shale-dominated,
poor, or far (decades) below the strongest zone. Bad hole, permeability held at
a model limit, a shaly zone with good modelled permeability (the model likely
overstates it) and water-bearing rock qualify a role without changing it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def _value(seq: Optional[Sequence], i: int) -> Optional[float]:
    if seq is None:
        return None
    v = seq[i]
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v


def flow_roles(
    perm: Sequence[Optional[float]],
    thickness: Sequence[float],
    *,
    sw: Optional[Sequence[Optional[float]]] = None,
    vcl: Optional[Sequence[Optional[float]]] = None,
    bad_hole: Optional[Sequence[Optional[float]]] = None,
    seal: Optional[Sequence[Optional[float]]] = None,
    at_limit: Optional[Sequence[bool]] = None,
    good_perm: float = 10.0,
    poor_perm: float = 1.0,
    baffle_decades: float = 2.0,
    seal_fraction: float = 0.5,
    sw_max: Optional[float] = None,
    vcl_max: Optional[float] = None,
    max_bad_hole_fraction: float = 0.5,
) -> Dict[str, object]:
    """Flow roles of zones.

    Parameters
    ----------
    perm : sequence of float or None
        Mean permeability per zone (mD); None where unknown.
    thickness : sequence of float
        Zone thicknesses (for permeability-thickness).
    sw, vcl, bad_hole, seal : sequences of float or None, optional
        Per-zone mean water saturation, clay volume, bad-hole share and
        seal-flag share.
    at_limit : sequence of bool, optional
        Whether the zone's permeability is held at the model's limit.
    good_perm, poor_perm : float
        Permeability (mD) above which modelled permeability counts as good, and
        at or below which a zone is poor (a possible baffle).
    baffle_decades : float
        Decades below the strongest zone at which a zone is a possible baffle.
    seal_fraction : float
        Seal-flag share at or above which a zone is shale-dominated.
    sw_max, vcl_max : float, optional
        The saturation and shale-volume cutoffs (water-bearing above ``sw_max``;
        shaly above ``vcl_max``).
    max_bad_hole_fraction : float
        Bad-hole share above which a role is low confidence.

    Returns
    -------
    dict
        ``zones``: per zone ``role`` (``"strongest"``, ``"baffle"`` or
        ``"flow"``; None without a permeability), ``kh`` and ``qualifiers``, a
        list of ``(code, value)`` with codes ``bad_hole`` (the share),
        ``at_model_limit``, ``shaly`` (the VCL) and ``water_bearing`` (the SW);
        ``strongest`` and ``capacity``: the indices of the highest mean
        permeability and the largest permeability-thickness (None without one).

    Examples
    --------
    >>> r = flow_roles([150.0, 0.5, 40.0], [10.0, 5.0, 30.0], sw=[0.3, 0.9, 0.8], sw_max=0.7)
    >>> [z["role"] for z in r["zones"]], r["strongest"], r["capacity"]
    (['strongest', 'baffle', 'flow'], 0, 0)
    """
    n = len(perm)
    if len(thickness) != n:
        raise ValueError(f"perm has {n} zones but thickness {len(thickness)}")
    k = [_value(perm, i) for i in range(n)]
    kh = [k[i] * thickness[i] if k[i] is not None else None for i in range(n)]
    ranked = sorted((i for i in range(n) if k[i] is not None), key=lambda i: -k[i])
    strongest = ranked[0] if ranked else None
    capacity = max(ranked, key=lambda i: kh[i]) if ranked else None
    zones: List[Dict[str, object]] = []
    for i in range(n):
        ki, quals = k[i], []
        seal_i, bh_i, vcl_i, sw_i = _value(seal, i), _value(bad_hole, i), _value(vcl, i), _value(sw, i)
        baffle = ki is not None and ((seal_i is not None and seal_i >= seal_fraction) or ki <= poor_perm or
                                     (strongest is not None and k[strongest] > 0 and ki > 0 and
                                      np.log10(k[strongest] / ki) >= baffle_decades))
        if bh_i is not None and bh_i > max_bad_hole_fraction:
            quals.append(("bad_hole", bh_i))
        if at_limit is not None and at_limit[i]:
            quals.append(("at_model_limit", None))
        if vcl_max is not None and vcl_i is not None and vcl_i > vcl_max and ki is not None and ki >= good_perm:
            quals.append(("shaly", vcl_i))
        if sw_max is not None and sw_i is not None and sw_i > sw_max:
            quals.append(("water_bearing", sw_i))
        role = None if ki is None else "strongest" if i == strongest else "baffle" if baffle else "flow"
        zones.append({"role": role, "kh": kh[i], "qualifiers": quals})
    return {"zones": zones, "strongest": strongest, "capacity": capacity}


__all__ = ["flow_roles"]
