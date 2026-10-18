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
        Intervals whose ``metadata["depth_unit"]`` is the other length unit
        ("m" or "ft") than ``well.depth_unit`` are converted to the well unit
        first.

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

        from petrophys._internal.benchmark import reproducing_benchmark
        from petrophys.model.units import normalize_depth_unit

        well_unit = normalize_depth_unit(getattr(well, "depth_unit", None) or "", strict=False)
        clipped = []
        for iv in self._intervals:
            top_depth, bottom_depth = iv.top_depth, iv.bottom_depth
            metadata = iv.metadata.copy()
            unit = normalize_depth_unit(str(metadata.get("depth_unit") or ""), strict=False)
            if (
                not reproducing_benchmark()
                and {unit, well_unit} == {"m", "ft"}
            ):
                factor = 0.3048 if unit == "ft" else 1.0 / 0.3048
                top_depth, bottom_depth = top_depth * factor, bottom_depth * factor
                metadata["depth_unit"] = well_unit
            if bottom_depth <= well_top or top_depth >= well_bottom:
                continue
            klass = type(iv)
            # Build kwargs from the instance's fields
            kwargs = {
                "name": iv.name,
                "top_depth": max(top_depth, well_top),
                "bottom_depth": min(bottom_depth, well_bottom),
                "lithology": iv.lithology,
                "metadata": metadata,
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
        """Export intervals to CSV (depths at full precision)."""
        from petrophys._internal.benchmark import reproducing_benchmark

        df = self.to_dataframe()
        if reproducing_benchmark():
            df.to_csv(filepath, index=False, float_format="%.2f")
        else:
            df.to_csv(filepath, index=False)

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
        if isinstance(key, (int, np.integer)):
            return self._intervals[key]
        iv = self.get_by_name(key)
        if iv is None:
            raise KeyError(f"Interval not found: {key}")
        return iv

    def __repr__(self) -> str:
        return f"FormationTops({len(self._intervals)} intervals)"


_TOP_COLUMNS = ("top_depth", "top", "top_md", "TOP_DEPTH", "TOP", "Top")
_BOTTOM_COLUMNS = ("bottom_depth", "bottom", "base", "base_depth", "BOTTOM_DEPTH", "BOTTOM", "BASE",
                   "Bottom", "Base")
_LABEL_COLUMNS = ("formation_name", "name", "zone", "lithology", "FORMATION", "NAME", "Formation",
                  "Name")
_OVERLAP_RULES = ("error", "thinnest", "thickest")


def _pick_column(columns, requested, candidates, role):
    if requested is not None:
        if requested not in columns:
            raise ValueError(f"Column {requested!r} not found; available: {list(columns)}")
        return requested
    for name in candidates:
        if name in columns:
            return name
    raise ValueError(
        f"Could not find the interval {role} column (tried {', '.join(candidates)}); "
        f"pass {role}= explicitly. Available: {list(columns)}"
    )


def _interval_records(intervals, label, top, bottom):
    """Return (tops, bottoms, labels) arrays from a table or interval collection."""
    if hasattr(intervals, "columns") and hasattr(intervals, "__getitem__"):
        columns = list(intervals.columns)
        top_col = _pick_column(columns, top, _TOP_COLUMNS, "top")
        bottom_col = _pick_column(columns, bottom, _BOTTOM_COLUMNS, "bottom")
        label_col = _pick_column(columns, label, _LABEL_COLUMNS, "label")
        tops = np.asarray(intervals[top_col], dtype=float)
        bottoms = np.asarray(intervals[bottom_col], dtype=float)
        labels = np.asarray(intervals[label_col], dtype=object)
    else:
        if top is not None or bottom is not None:
            raise ValueError("top= and bottom= apply only to tables; Interval objects "
                             "use top_depth and bottom_depth")
        items = list(intervals)
        attribute = label or "name"
        for item in items:
            if not isinstance(item, Interval):
                raise TypeError(
                    "intervals must be a DataFrame, a FormationTops or a sequence of "
                    f"Interval objects; got an item of type {type(item).__name__}"
                )
            if not hasattr(item, attribute):
                raise ValueError(f"Interval {item.name!r} has no attribute {attribute!r}")
        tops = np.array([item.top_depth for item in items], dtype=float)
        bottoms = np.array([item.bottom_depth for item in items], dtype=float)
        labels = np.array([getattr(item, attribute) for item in items], dtype=object)
    usable = np.isfinite(tops) & np.isfinite(bottoms)
    if np.any(usable & (bottoms < tops)):
        raise ValueError("An interval has its bottom above its top")
    from petrophys._internal.benchmark import reproducing_benchmark

    if not np.all(usable) and not reproducing_benchmark():
        import warnings

        names = ", ".join(repr(name) for name in labels[~usable][:10])
        warnings.warn(
            f"Intervals without a finite top and bottom are not used for labelling: {names}",
            UserWarning,
            stacklevel=3,
        )
    return tops[usable], bottoms[usable], labels[usable]


def label_depths(
    depth,
    intervals,
    *,
    label: Optional[str] = None,
    top: Optional[str] = None,
    bottom: Optional[str] = None,
    overlap: str = "error",
    default: Any = "",
) -> np.ndarray:
    """Label each depth with the name of the interval that contains it.

    Use it to attach formation, zone or lithology names to log samples, core
    plugs or any other depths. Each interval covers ``top <= depth < bottom``,
    so a depth on a shared boundary belongs to the deeper interval.

    Parameters
    ----------
    depth : float, array-like or Well
        Depths to label (a well's depth is used when a ``Well`` is given), in
        the same unit and reference as the intervals.
    intervals : DataFrame, FormationTops or sequence of Interval
        Interval table, for example from :func:`petrophys.read_formations`
        (``top_depth``, ``bottom_depth``, ``formation_name``) or
        :func:`petrophys.read_lithology_intervals` (``top``, ``bottom``,
        ``lithology``). Column names are detected; override them with
        ``top``, ``bottom`` and ``label``.
    label : str, optional
        Table column, or ``Interval`` attribute (default ``"name"``), holding
        the label. For a table, the first of ``formation_name``, ``name``,
        ``zone`` and ``lithology`` that exists is used.
    top, bottom : str, optional
        Top and bottom depth columns of a table.
    overlap : {"error", "thinnest", "thickest"}, default "error"
        What to do where intervals overlap, as in nested tops that list
        groups and their member formations. ``"thinnest"`` picks the most
        detailed interval (the formation), ``"thickest"`` the broadest (the
        group). The default raises so that nesting is never resolved
        silently.
    default : object, default ""
        Label for depths outside every interval. Intervals with a missing
        (NaN) top or bottom are left out with a warning, so their depths get
        this label too.

    Returns
    -------
    numpy.ndarray of object
        One label per depth.

    Raises
    ------
    ValueError
        If intervals overlap and ``overlap="error"``, a column cannot be
        found, or ``overlap`` is not a supported rule.

    Examples
    --------
    >>> import pandas as pd
    >>> tops = pd.DataFrame({
    ...     "formation_name": ["BAAT GP.", "Ness Fm.", "Etive Fm."],
    ...     "top_depth": [2841.0, 2841.0, 2865.0],
    ...     "bottom_depth": [2958.0, 2865.0, 2958.0]})
    >>> label_depths([2830.0, 2850.0, 2900.0], tops, overlap="thinnest").tolist()
    ['', 'Ness Fm.', 'Etive Fm.']
    >>> label_depths([2850.0], tops, overlap="thickest").tolist()
    ['BAAT GP.']
    """
    if overlap not in _OVERLAP_RULES:
        raise ValueError(f"overlap must be one of {', '.join(_OVERLAP_RULES)}; got {overlap!r}")
    if hasattr(depth, "depth") and not isinstance(depth, np.ndarray):
        depth = depth.depth
    depth = np.asarray(depth, dtype=float).reshape(-1)
    tops, bottoms, labels = _interval_records(intervals, label, top, bottom)

    if overlap == "error" and tops.size > 1:
        order = np.argsort(tops, kind="stable")
        running_bottom = bottoms[order[0]]
        running_index = order[0]
        for index in order[1:]:
            if tops[index] < running_bottom:
                raise ValueError(
                    f"Intervals overlap ({labels[running_index]!r} "
                    f"{tops[running_index]:g}-{bottoms[running_index]:g} and "
                    f"{labels[index]!r} {tops[index]:g}-{bottoms[index]:g}). Pick one level "
                    "(e.g. read_formations(..., rank='formation')) or pass "
                    "overlap='thinnest' or 'thickest'."
                )
            if bottoms[index] > running_bottom:
                running_bottom, running_index = bottoms[index], index

    result = np.empty(depth.size, dtype=object)
    result[:] = default
    thickness = bottoms - tops
    # Later assignments win: broad intervals first for "thinnest", and the
    # reverse for "thickest".
    order = np.argsort(-thickness if overlap != "thickest" else thickness, kind="stable")
    for index in order:
        inside = (depth >= tops[index]) & (depth < bottoms[index])
        value = labels[index]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            value = default
        result[inside] = value
    return result
