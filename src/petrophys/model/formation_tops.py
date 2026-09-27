"""
Formation tops management.

Load, validate, sort, and query collections of formation intervals
from CSV files, dictionaries, or DataFrames.
"""

from typing import Optional, List, Dict, Union, Any
from pathlib import Path
import numpy as np

from petrophys.model.intervals import (
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
        self.import_report: Dict[str, Any] = {}
        if intervals:
            for iv in intervals:
                self._intervals.append(iv)
            self._sort()

    # ------------------------------------------------------------------ #
    #  Loading class methods                                               #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_csv(
        cls, filepath: Union[str, Path], well_name: Optional[str] = None,
        format: str = "auto", interval_kind: IntervalKind = IntervalKind.FORMATION,
        *, terminal_depth: Optional[float] = None, rank: Optional[str] = None,
        **read_options,
    ) -> "FormationTops":
        """Read bounded intervals through the shared geology importer.

        Simple, NPD and Danish headers share one validated schema. Top-only
        files require an explicit terminal_depth to derive intervals; no bottom
        is invented. Select rank before deriving intervals from hierarchical
        picks. read_options accepts the read_formations table/depth controls.
        """
        from petrophys.io.geology_reader import read_formations

        if format not in {"auto", "simple", "npd", "danish"}:
            raise ValueError(f"Unknown formation format: {format!r}")
        table = read_formations(filepath, well_name=well_name, **read_options)
        return cls._from_table(table, interval_kind, terminal_depth, rank)

    @classmethod
    def _from_table(cls, table, interval_kind, terminal_depth=None, rank=None):
        if rank is not None:
            if "rank" not in table:
                raise ValueError("Supply a rank column before selecting a stratigraphic level.")
            table = table.loc[table["rank"].astype(str).str.lower() == str(rank).lower()].copy()
            if table.empty:
                raise ValueError(f"No formations found at rank {rank!r}.")
        if "bottom_depth" not in table:
            if terminal_depth is None:
                raise ValueError(
                    "Top-only picks require terminal_depth to build bounded intervals. "
                    "For annotation without bottoms, use read_formations() directly."
                )
            terminal_depth = float(terminal_depth)
            if not np.isfinite(terminal_depth) or terminal_depth <= table["top_depth"].max():
                raise ValueError("terminal_depth must be finite and below the deepest top.")
            if table["top_depth"].duplicated().any() or (
                "rank" in table and table["rank"].nunique() > 1
            ):
                message = "Hierarchical/coincident picks require a single rank or explicit bottoms."
                if "rank" in table:
                    ranks = ", ".join(repr(str(value)) for value in table["rank"].dropna().unique())
                    message += f" Select one level with rank=..., e.g. one of: {ranks}."
                raise ValueError(message)
            table = table.copy()
            table["bottom_depth"] = table["top_depth"].shift(-1).fillna(terminal_depth)
        klass = _KIND_CLASS.get(interval_kind, ManualInterval)
        intervals = []
        for _, row in table.iterrows():
            metadata = {
                key: row[key] for key in ("rank", "parent", "well")
                if key in table and str(row[key]).strip()
            }
            metadata.update({key: table.attrs[key] for key in ("depth_unit", "depth_reference", "datum")
                             if table.attrs.get(key) is not None})
            kwargs = dict(name=row["formation_name"], top_depth=float(row["top_depth"]),
                          bottom_depth=float(row["bottom_depth"]),
                          lithology=row.get("lithology", ""), metadata=metadata)
            if interval_kind == IntervalKind.FORMATION:
                kwargs["age"] = row.get("age", "")
            intervals.append(klass(**kwargs))
        result = cls(intervals)
        result.import_report = dict(table.attrs.get("import_report", {}))
        return result

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
                "metadata": dict(d.get("metadata", {})),
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
        cls, df, name_col: str = "name", top_col: str = "top_depth",
        bottom_col: str = "bottom_depth", lithology_col: Optional[str] = "lithology",
        interval_kind: IntervalKind = IntervalKind.FORMATION,
        *, terminal_depth: Optional[float] = None, rank: Optional[str] = None,
        **read_options,
    ) -> "FormationTops":
        """Read a table using the same validation as CSV and plotting."""
        from petrophys.io.geology_reader import read_formations

        columns = {canonical: column for canonical, column in (
            ("formation_name", name_col), ("top_depth", top_col),
            ("bottom_depth", bottom_col), ("lithology", lithology_col),
        ) if column is not None and column in df.columns}
        columns.update(read_options.pop("columns", {}) or {})
        table = read_formations(df, columns=columns, **read_options)
        return cls._from_table(table, interval_kind, terminal_depth, rank)

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
        Find overlapping peers, allowing explicitly linked parent containment.

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
                    contained_parent = any(
                        str(child.metadata.get("parent", "")).strip().lower() == parent.name.lower()
                        and parent.top_depth <= child.top_depth
                        and parent.bottom_depth >= child.bottom_depth
                        and parent.metadata.get("rank") != child.metadata.get("rank")
                        for parent, child in ((a, b), (b, a))
                    )
                    if contained_parent:
                        continue
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
        well_top = getattr(well, "top", None)
        well_bottom = getattr(well, "bottom", None)
        if well_top is None or well_bottom is None:
            return FormationTops([])

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
            row.update({key: value for key, value in iv.metadata.items()
                        if key in {"rank", "parent", "well", "depth_unit", "depth_reference", "datum"}})
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
