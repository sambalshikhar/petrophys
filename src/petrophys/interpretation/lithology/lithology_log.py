"""
Lithology log from interval definitions.

This module creates a per-depth lithology classification array from
a CSV file or list of (top, bottom, lithology_name) intervals.

The resulting array is used by downstream modules (saturation, net pay)
to apply lithology-dependent parameters at each depth sample.
"""

from typing import Optional, List, Union
from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np


@dataclass
class LithologyInterval:
    """A depth interval with assigned lithology."""
    top: float
    bottom: float
    lithology: str  # "clastic", "carbonate", "shale", etc.


def load_lithology_csv(
    filepath: Union[str, Path],
    top_col: Optional[str] = None,
    bottom_col: Optional[str] = None,
    lithology_col: Optional[str] = None,
) -> List[LithologyInterval]:
    """
    Load lithology intervals from a CSV file.

    Expected CSV format::

        top,bottom,lithology
        200.0,1500.0,clastic
        1500.0,2200.0,carbonate
        2200.0,3500.0,clastic

    Column names are auto-detected (case-insensitive). Common variants
    are recognized: top/top_depth, bottom/bottom_depth/base,
    lithology/lith/rock_type.

    Parameters
    ----------
    filepath : str or Path
        Path to CSV file
    top_col : str, optional
        Override column name for top depth
    bottom_col : str, optional
        Override column name for bottom depth
    lithology_col : str, optional
        Override column name for lithology

    Returns
    -------
    list of LithologyInterval
        Sorted by top depth
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Lithology CSV not found: {filepath}")

    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = {h.strip().lower(): h.strip() for h in reader.fieldnames or []}
        rows = list(reader)

    # Auto-detect columns
    if top_col is None:
        top_col = _find_column(headers, ["top", "top_depth", "topdepth", "from"])
    if bottom_col is None:
        bottom_col = _find_column(headers, ["bottom", "bottom_depth", "bottomdepth",
                                             "base", "to"])
    if lithology_col is None:
        lithology_col = _find_column(headers, ["lithology", "lith", "rock_type",
                                                "rocktype", "litho", "formation_type"])

    if not all([top_col, bottom_col, lithology_col]):
        raise ValueError(
            f"Could not detect required columns (top, bottom, lithology) "
            f"in headers: {list(headers.values())}. "
            f"Use top_col/bottom_col/lithology_col to specify manually."
        )

    intervals = []
    for row in rows:
        top_val = _safe_float(row.get(top_col) or row.get(headers.get(top_col.lower(), "")))
        bot_val = _safe_float(row.get(bottom_col) or row.get(headers.get(bottom_col.lower(), "")))
        lith = (row.get(lithology_col) or row.get(headers.get(lithology_col.lower(), "")) or "").strip()

        if top_val is not None and bot_val is not None and lith:
            # Normalize lithology names
            lith_lower = lith.lower()
            if lith_lower in ("sandstone", "sand", "clastic", "clastics", "sst"):
                lith_normalized = "clastic"
            elif lith_lower in ("limestone", "carbonate", "chalk", "dolomite", "carb"):
                lith_normalized = "carbonate"
            elif lith_lower in ("shale", "mudstone", "claystone", "siltstone", "marl"):
                lith_normalized = "clastic"  # Shale uses clastic cutoffs
            else:
                lith_normalized = "clastic"  # Default

            intervals.append(LithologyInterval(
                top=top_val,
                bottom=bot_val,
                lithology=lith_normalized,
            ))

    intervals.sort(key=lambda x: x.top)
    return intervals


def build_lithology_log(
    depth: np.ndarray,
    intervals: List[LithologyInterval],
    default_lithology: str = "clastic"
) -> np.ndarray:
    """
    Create a per-depth lithology array from interval definitions.

    Parameters
    ----------
    depth : np.ndarray
        Depth array (one value per sample)
    intervals : list of LithologyInterval
        Lithology interval definitions
    default_lithology : str
        Lithology for depths not covered by any interval

    Returns
    -------
    np.ndarray
        String array with lithology at each depth sample.
        Values are "clastic" or "carbonate".
    """
    lithology = np.full(len(depth), default_lithology, dtype=object)

    for interval in intervals:
        mask = (depth >= interval.top) & (depth <= interval.bottom)
        lithology[mask] = interval.lithology

    return lithology


def _find_column(headers: dict, candidates: list) -> Optional[str]:
    """Find a column name from a list of candidates."""
    for candidate in candidates:
        if candidate in headers:
            return headers[candidate]
        # Check original case headers
        for lower_key, orig_key in headers.items():
            if lower_key == candidate:
                return orig_key
    return None


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(str(value).strip())
        return v if np.isfinite(v) else None
    except (ValueError, TypeError):
        return None
