"""
Formation tops management.

Load, validate, sort, and query collections of formation intervals
from CSV files, dictionaries, or DataFrames.
"""

from typing import Optional, List, Dict, Union, Any
from pathlib import Path
import csv
import numpy as np

from petrophys.formation.intervals import (
    IntervalKind,
    Interval,
    Formation,
    Zone,
    SubZone,
    ManualInterval,
)


# Map IntervalKind to the concrete class
_KIND_CLASS = {
    IntervalKind.FORMATION: Formation,
    IntervalKind.ZONE: Zone,
    IntervalKind.SUB_ZONE: SubZone,
    IntervalKind.MANUAL: ManualInterval,
}


class FormationTops:
    """
    Manager for a collection of formation/zone intervals.

    Provides loading from CSV (simple, NPD, Danish formats),
    validation (overlap detection), querying by depth/name,
    and export to DataFrame/CSV.

    Parameters
    ----------
    intervals : list of Interval, optional
        Initial intervals to manage

    Examples
    --------
    >>> tops = FormationTops.from_csv("formations.csv")
    >>> tops.validate()
    >>> fm = tops.get_by_name("Brent")
    >>> at_depth = tops.get_at_depth(2500.0)
    """

    def __init__(self, intervals: Optional[List[Interval]] = None):
        self._intervals: List[Interval] = []
        if intervals:
            for iv in intervals:
                self._intervals.append(iv)
            self._sort()

    # ------------------------------------------------------------------ #
    #  Loading class methods                                               #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_csv(
        cls,
        filepath: Union[str, Path],
        well_name: Optional[str] = None,
        format: str = "auto",
        interval_kind: IntervalKind = IntervalKind.FORMATION,
    ) -> "FormationTops":
        """
        Load intervals from a CSV file.

        Supported formats:
            - "simple": formation_name, top_depth, bottom_depth [, lithology]
            - "npd": NPD stratigraphic picks (Well identifier, Surface, MD)
            - "danish": wellName, formationName, topDepth, bottomDepth
            - "auto": detect from column headers

        Parameters
        ----------
        filepath : str or Path
            Path to CSV file
        well_name : str, optional
            Well name filter (required for multi-well NPD/Danish files)
        format : str
            CSV format: "auto", "simple", "npd", "danish"
        interval_kind : IntervalKind
            Which Interval subclass to instantiate (default: FORMATION)

        Returns
        -------
        FormationTops
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Formation CSV not found: {filepath}")

        with open(filepath, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = [h.strip() for h in reader.fieldnames] if reader.fieldnames else []
            rows = list(reader)

        if format == "auto":
            format = _detect_csv_format(headers)

        if format == "npd":
            intervals = _parse_npd_format(rows, well_name, interval_kind)
        elif format == "danish":
            intervals = _parse_danish_format(rows, well_name, interval_kind)
        else:
            intervals = _parse_simple_format(rows, interval_kind)

        return cls(intervals)

    @classmethod
    def from_dict(
        cls,
        data: List[Dict[str, Any]],
        interval_kind: IntervalKind = IntervalKind.FORMATION,
    ) -> "FormationTops":
        """
        Create from a list of dictionaries.

        Each dict must have keys: "name", "top_depth", "bottom_depth".
        Optional keys: "lithology", "age", "parent_formation", "parent_zone".

        Parameters
        ----------
        data : list of dict
        interval_kind : IntervalKind

        Returns
        -------
        FormationTops
        """
        intervals = []
        klass = _KIND_CLASS.get(interval_kind, ManualInterval)

        for d in data:
            kwargs = {
                "name": d["name"],
                "top_depth": float(d["top_depth"]),
                "bottom_depth": float(d["bottom_depth"]),
                "lithology": d.get("lithology", ""),
            }
            # Pass subclass-specific fields
            if interval_kind == IntervalKind.FORMATION and "age" in d:
                kwargs["age"] = d["age"]
            elif interval_kind == IntervalKind.ZONE and "parent_formation" in d:
                kwargs["parent_formation"] = d["parent_formation"]
            elif interval_kind == IntervalKind.SUB_ZONE and "parent_zone" in d:
                kwargs["parent_zone"] = d["parent_zone"]

            intervals.append(klass(**kwargs))

        return cls(intervals)

    @classmethod
    def from_dataframe(
        cls,
        df,
        name_col: str = "name",
        top_col: str = "top_depth",
        bottom_col: str = "bottom_depth",
        lithology_col: Optional[str] = "lithology",
        interval_kind: IntervalKind = IntervalKind.FORMATION,
    ) -> "FormationTops":
        """
        Create from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
        name_col, top_col, bottom_col : str
            Column names for required fields
        lithology_col : str, optional
            Column name for lithology
        interval_kind : IntervalKind

        Returns
        -------
        FormationTops
        """
        records = []
        for _, row in df.iterrows():
            d = {
                "name": str(row[name_col]),
                "top_depth": float(row[top_col]),
                "bottom_depth": float(row[bottom_col]),
            }
            if lithology_col and lithology_col in df.columns:
                d["lithology"] = str(row[lithology_col]) if row[lithology_col] else ""
            records.append(d)

        return cls.from_dict(records, interval_kind=interval_kind)

    # ------------------------------------------------------------------ #
    #  Validation                                                          #
    # ------------------------------------------------------------------ #

    def validate(self) -> List[str]:
        """
        Validate intervals: check for overlaps within same kind.

        Returns
        -------
        list of str
            Validation issues (empty if all OK)

        Raises
        ------
        ValueError
            If overlapping intervals are found
        """
        issues = self.find_overlaps()
        if issues:
            raise ValueError(
                f"Found {len(issues)} overlapping interval(s):\n"
                + "\n".join(issues)
            )
        return issues

    def find_overlaps(self) -> List[str]:
        """
        Find overlapping intervals of the same kind.

        Returns
        -------
        list of str
            Descriptions of overlapping pairs (empty if none)
        """
        issues = []
        for i in range(len(self._intervals)):
            for j in range(i + 1, len(self._intervals)):
                a, b = self._intervals[i], self._intervals[j]
                if a.kind == b.kind and a.overlaps(b):
                    issues.append(
                        f"Overlap: {a.name} ({a.top_depth}-{a.bottom_depth}) "
                        f"and {b.name} ({b.top_depth}-{b.bottom_depth})"
                    )
        return issues

    def has_overlaps(self) -> bool:
        """Return True if any intervals of the same kind overlap."""
        return len(self.find_overlaps()) > 0

    # ------------------------------------------------------------------ #
    #  Querying                                                            #
    # ------------------------------------------------------------------ #

    def get_by_name(self, name: str) -> Optional[Interval]:
        """Get first interval matching name (case-insensitive)."""
        name_lower = name.lower()
        for iv in self._intervals:
            if iv.name.lower() == name_lower:
                return iv
        return None

    def get_at_depth(self, depth: float) -> List[Interval]:
        """Return all intervals containing this depth."""
        return [iv for iv in self._intervals
                if iv.top_depth <= depth <= iv.bottom_depth]

    def get_in_range(self, top: float, bottom: float) -> List[Interval]:
        """Return intervals overlapping with [top, bottom] range."""
        return [iv for iv in self._intervals
                if iv.bottom_depth > top and iv.top_depth < bottom]

    def filter_by_kind(self, kind: IntervalKind) -> List[Interval]:
        """Return intervals of a specific kind."""
        return [iv for iv in self._intervals if iv.kind == kind]

    def clip_to_well(self, well) -> "FormationTops":
        """
        Return a new FormationTops with intervals clipped to well depth range.

        Intervals completely outside the well range are excluded.
        Partially overlapping intervals are clipped to well boundaries.

        Parameters
        ----------
        well : Well
            Well object with depth array

        Returns
        -------
        FormationTops
        """
        well_top = float(np.nanmin(well.depth))
        well_bottom = float(np.nanmax(well.depth))

        clipped = []
        for iv in self._intervals:
            if iv.bottom_depth <= well_top or iv.top_depth >= well_bottom:
                continue
            klass = type(iv)
            # Build kwargs from the instance's fields
            kwargs = {
                "name": iv.name,
                "top_depth": max(iv.top_depth, well_top),
                "bottom_depth": min(iv.bottom_depth, well_bottom),
                "lithology": iv.lithology,
                "metadata": iv.metadata.copy(),
            }
            # Preserve subclass-specific fields
            if isinstance(iv, Formation):
                kwargs["age"] = iv.age
            elif isinstance(iv, Zone):
                kwargs["parent_formation"] = iv.parent_formation
            elif isinstance(iv, SubZone):
                kwargs["parent_zone"] = iv.parent_zone

            clipped.append(klass(**kwargs))

        return FormationTops(clipped)

    # ------------------------------------------------------------------ #
    #  Mutation                                                            #
    # ------------------------------------------------------------------ #

    def add(self, interval: Interval):
        """Add an interval and re-sort."""
        self._intervals.append(interval)
        self._sort()

    def remove(self, name: str) -> bool:
        """Remove first interval matching name. Returns True if removed."""
        name_lower = name.lower()
        for i, iv in enumerate(self._intervals):
            if iv.name.lower() == name_lower:
                self._intervals.pop(i)
                return True
        return False

    # ------------------------------------------------------------------ #
    #  Export                                                               #
    # ------------------------------------------------------------------ #

    def to_dataframe(self):
        """Convert intervals to a pandas DataFrame."""
        import pandas as pd

        rows = []
        for iv in self._intervals:
            row = {
                "name": iv.name,
                "top_depth": iv.top_depth,
                "bottom_depth": iv.bottom_depth,
                "thickness": iv.thickness,
                "kind": iv.kind.value,
                "lithology": iv.lithology,
            }
            if isinstance(iv, Formation):
                row["age"] = iv.age
            elif isinstance(iv, Zone):
                row["parent_formation"] = iv.parent_formation
            elif isinstance(iv, SubZone):
                row["parent_zone"] = iv.parent_zone
            rows.append(row)

        return pd.DataFrame(rows)

    def to_csv(self, filepath: Union[str, Path]) -> None:
        """Export intervals to CSV."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False, float_format="%.2f")

    # ------------------------------------------------------------------ #
    #  Collection interface                                                #
    # ------------------------------------------------------------------ #

    @property
    def intervals(self) -> List[Interval]:
        """Return list of intervals (copy)."""
        return self._intervals.copy()

    @property
    def names(self) -> List[str]:
        """Return list of interval names."""
        return [iv.name for iv in self._intervals]

    def _sort(self):
        """Sort intervals by top_depth."""
        self._intervals.sort(key=lambda iv: iv.top_depth)

    def __len__(self) -> int:
        return len(self._intervals)

    def __iter__(self):
        return iter(self._intervals)

    def __getitem__(self, key: Union[int, str]) -> Interval:
        if isinstance(key, int):
            return self._intervals[key]
        iv = self.get_by_name(key)
        if iv is None:
            raise KeyError(f"Interval not found: {key}")
        return iv

    def __repr__(self) -> str:
        return f"FormationTops({len(self._intervals)} intervals)"


# ====================================================================== #
#  CSV parsing helpers                                                     #
# ====================================================================== #

def _detect_csv_format(headers: List[str]) -> str:
    """Auto-detect CSV format from column headers."""
    headers_lower = [h.lower().strip() for h in headers]
    if ("well identifier" in headers_lower or "surface" in headers_lower
            or ("well" in headers_lower and "formation" in headers_lower and "md_top" in headers_lower)):
        return "npd"
    elif "wellname" in headers_lower and "formationname" in headers_lower:
        return "danish"
    return "simple"


def _parse_simple_format(
    rows: List[Dict],
    interval_kind: IntervalKind,
) -> List[Interval]:
    """Parse simple CSV: formation_name, top_depth, bottom_depth [, lithology]."""
    klass = _KIND_CLASS.get(interval_kind, ManualInterval)
    intervals = []

    for row in rows:
        keys = {k.strip().lower(): v for k, v in row.items()}

        name = (keys.get("formation_name") or keys.get("formation")
                or keys.get("name") or keys.get("zone") or "Unknown")
        top = _safe_float(keys.get("top_depth") or keys.get("top")
                          or keys.get("topdepth"))
        bottom = _safe_float(keys.get("bottom_depth") or keys.get("bottom")
                             or keys.get("bottomdepth"))
        lithology = (keys.get("lithology") or keys.get("majoritylithology") or "")

        if top is not None and bottom is not None:
            intervals.append(klass(
                name=name.strip(),
                top_depth=top,
                bottom_depth=bottom,
                lithology=lithology.strip(),
            ))

    intervals.sort(key=lambda iv: iv.top_depth)
    return intervals


def _parse_npd_format(
    rows: List[Dict],
    well_name: Optional[str],
    interval_kind: IntervalKind,
) -> List[Interval]:
    """
    Parse NPD stratigraphic picks format.

    NPD provides formation tops (not intervals), so we construct intervals
    by sorting tops and using the next top as the bottom of each formation.
    """
    klass = _KIND_CLASS.get(interval_kind, Formation)
    filtered = []

    target_well = _normalize_well_identifier(well_name) if well_name else None

    for row in rows:
        keys = {k.strip(): v for k, v in row.items()}
        keys_lower = {k.strip().lower(): v for k, v in row.items()}
        well_id = str(keys.get("Well identifier", keys_lower.get("well", ""))).strip()
        if target_well and _normalize_well_identifier(well_id) != target_well:
            continue

        surface = str(keys.get("Surface", keys_lower.get("formation", ""))).strip()
        md = _safe_float(keys.get("MD", keys_lower.get("md_top")))
        age = keys.get("Geological age", keys_lower.get("age", ""))

        if md is not None and surface:
            filtered.append({
                "name": surface.replace(" Top", ""),
                "md": md,
                "age": str(age) if age else "",
            })

    if not filtered:
        return []

    filtered.sort(key=lambda x: x["md"])

    intervals = []
    for i in range(len(filtered)):
        top = filtered[i]["md"]
        bottom = filtered[i + 1]["md"] if i + 1 < len(filtered) else top + 100.0

        if bottom > top:
            kwargs = {
                "name": filtered[i]["name"],
                "top_depth": top,
                "bottom_depth": bottom,
            }
            if interval_kind == IntervalKind.FORMATION:
                kwargs["age"] = filtered[i]["age"]
            intervals.append(klass(**kwargs))

    return intervals


def _parse_danish_format(
    rows: List[Dict],
    well_name: Optional[str],
    interval_kind: IntervalKind,
) -> List[Interval]:
    """Parse Danish format: wellName, formationName, topDepth, bottomDepth."""
    klass = _KIND_CLASS.get(interval_kind, Formation)
    intervals = []

    target_well = _normalize_well_identifier(well_name) if well_name else None

    for row in rows:
        keys = {k.strip().lower(): v for k, v in row.items()}
        wn = keys.get("wellname", "").strip()
        if target_well and _normalize_well_identifier(wn) != target_well:
            continue

        name = keys.get("formationname", "Unknown").strip()
        top = _safe_float(keys.get("topdepth"))
        bottom = _safe_float(keys.get("bottomdepth"))
        lithology = keys.get("majoritylithology", "").strip()

        if top is not None and bottom is not None:
            intervals.append(klass(
                name=name,
                top_depth=top,
                bottom_depth=bottom,
                lithology=lithology,
            ))

    intervals.sort(key=lambda iv: iv.top_depth)
    return intervals


def _safe_float(value: Any) -> Optional[float]:
    """Safely convert to float, return None on failure."""
    if value is None:
        return None
    try:
        v = float(str(value).strip())
        return v if np.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def _normalize_well_identifier(value: Any) -> str:
    """Normalize well identifiers for lightweight cross-file matching."""
    text = str(value or "").strip().lower().replace("_", "/")
    if " " in text:
        text = text.split(" ")[0]
    return text
