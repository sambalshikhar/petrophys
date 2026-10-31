"""
The resistivity invasion profile a bed should show: which of a shallow and a
deep resistivity reading is higher, from the mud filtrate and the formation
water (Rmf/Rw) and the bed's fluid.

Archie in the flushed and the uninvaded zone gives Rxo/Rt = (Rmf/Rw) x
(Sw/Sxo)^n with Sxo >= Sw: water-bearing beds drilled with fresh mud read higher
shallow (resistive invasion), with salt mud higher deep (conductive invasion);
in hydrocarbon-bearing beds the flushed hydrocarbons lower the shallow reading
(Asquith and Krygowski, Basic Well Log Analysis, 2nd ed., AAPG 2004, ch. 1).

References
----------
Archie, G.E. (1942). The electrical resistivity log as an aid in determining
some reservoir characteristics. *Transactions of the AIME* 146, 54-62.

Asquith, G. and Krygowski, D. (2004). *Basic Well Log Analysis*, 2nd ed. AAPG
Methods in Exploration 16.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

# Resistivity mnemonics from the shallowest to the deepest radial depth of investigation.
DEPTH_OF_INVESTIGATION = ("RMIC", "RXO", "RSHA", "RMED", "RDEP")

# Rmf/Rw within this factor of 1 counts as similar: water-bearing rock then shows no diagnostic separation (the
# shallow reading also mixes flushed and invaded rock, so a smaller contrast is not readable).
SIMILAR_FACTOR = 1.25

MUD_REGIMES: Dict[str, str] = {
    "fresh": "filtrate fresher than formation water",
    "saline": "filtrate more saline than formation water",
    "similar": "filtrate and formation water of similar resistivity",
}

# Expected order of the shallowest and the deepest reading: "shallow" (shallow reads higher), "deep" (deep reads
# higher) or "either" (no definite order), with the reason.
EXPECTED_ORDER: Dict[str, Dict[str, Dict[str, str]]] = {
    "fresh": {
        "water": {"order": "shallow", "why": "resistive invasion"},
        "hydrocarbon": {"order": "either", "why": "deep reads higher where hydrocarbons remain beyond the flushed zone"},
    },
    "saline": {
        "water": {"order": "deep", "why": "conductive invasion"},
        "hydrocarbon": {"order": "deep", "why": "saline filtrate and flushed hydrocarbons both lower the shallow reading"},
    },
    "similar": {
        "water": {"order": "either", "why": "little separation"},
        "hydrocarbon": {"order": "deep", "why": "flushed hydrocarbons lower the shallow reading"},
    },
}


def mud_regime(rmf: float, rw: float, similar_factor: float = SIMILAR_FACTOR) -> str:
    """``"fresh"`` (Rmf above Rw by more than ``similar_factor``), ``"saline"`` (below by more) or ``"similar"``.

    Both resistivities at formation temperature.

    Examples
    --------
    >>> mud_regime(0.3, 0.05), mud_regime(0.045, 0.05), mud_regime(0.055, 0.05)
    ('fresh', 'similar', 'similar')
    """
    if not (rmf > 0 and rw > 0):
        raise ValueError(f"Rmf and Rw must be positive (got {rmf}, {rw})")
    ratio = rmf / rw
    return "fresh" if ratio > similar_factor else "saline" if ratio < 1 / similar_factor else "similar"


def expected_resistivity_order(mud: str, fluid: Optional[str]) -> str:
    """The expected order of the shallowest and the deepest reading: ``"shallow"``, ``"deep"`` or ``"either"``.

    ``fluid`` is ``"water"`` or ``"hydrocarbon"``; None (not known) gives the order water and hydrocarbon agree on,
    else ``"either"``.

    Examples
    --------
    >>> expected_resistivity_order("fresh", "water"), expected_resistivity_order("saline", None)
    ('shallow', 'deep')
    """
    table = EXPECTED_ORDER[mud]
    if fluid is not None:
        return table[fluid]["order"]
    orders = {table[f]["order"] for f in ("water", "hydrocarbon")}
    return orders.pop() if len(orders) == 1 else "either"


def shallowest_and_deepest(curves: Iterable[str],
                           ranking: Iterable[str] = DEPTH_OF_INVESTIGATION) -> Optional[Tuple[str, str]]:
    """The shallowest- and the deepest-reading of the given resistivity curves; None unless every curve is ranked
    and at least two differ.

    Examples
    --------
    >>> shallowest_and_deepest(["RDEP", "RSHA", "RMED"])
    ('RSHA', 'RDEP')
    """
    rank = list(ranking)
    curves = list(curves)
    if not curves or not set(curves) <= set(rank):
        return None
    shallow, deep = min(curves, key=rank.index), max(curves, key=rank.index)
    return None if shallow == deep else (shallow, deep)


__all__ = [
    "DEPTH_OF_INVESTIGATION",
    "EXPECTED_ORDER",
    "MUD_REGIMES",
    "SIMILAR_FACTOR",
    "expected_resistivity_order",
    "mud_regime",
    "shallowest_and_deepest",
]
