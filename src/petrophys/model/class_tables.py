"""
Class tables: the classes each petrophysical property is graded into.

:data:`CLASS_TABLES` holds one canonical table per property, with the unit of
its values and the source of its edges. The classification enums of
:mod:`petrophys.model.classifications`, the ``classify_*`` helpers built on
them and the diagnostics analyzers (with their default ``"current"`` class
tables) all classify through it, so a property gets the same class wherever it
is graded. A sample without a valid value gets the table's ``missing`` label,
never a class.

A one-curve table is a :class:`Bands`: a base label for every valid sample,
then each ``(op, threshold, label)`` step in order, later steps winning. The
data therefore also say on which side of an edge a value equal to it falls,
for the analyzer that classifies with them and for anyone reading its classes.
Reservoir quality, graded from effective porosity, permeability and clay volume
with limits per lithology family, is a :class:`ReservoirQualityTable`.

Examples
--------
>>> from petrophys.model.class_tables import CLASS_TABLES
>>> vcl = CLASS_TABLES["VCL"]
>>> list(vcl.classify([0.1, 0.25, 0.5, 0.6, float("nan")]))
['CLEAN_FORMATION', 'MODERATE_CLAY', 'MODERATE_CLAY', 'SHALY_FORMATION', 'NO_VCL_DATA']
>>> vcl.unit, vcl.bands.edges
('v/v', (0.25, 0.5))
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from petrophys.model.constants import DefaultParameters

_OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}


@dataclass(frozen=True)
class Bands:
    """A one-curve classification: ``base`` for every valid sample, then each
    ``(op, threshold, label)`` step in order (``op`` one of ``>``, ``>=``,
    ``<``, ``<=``).

    Examples
    --------
    >>> b = Bands("POOR", ((">", 1.0, "FAIR"), (">", 10.0, "GOOD")))
    >>> list(b.classify([0.5, 1.0, 5.0, 50.0])), b.edges, b.labels
    (['POOR', 'POOR', 'FAIR', 'GOOD'], (1.0, 10.0), ('POOR', 'FAIR', 'GOOD'))
    """

    base: str
    steps: Tuple[Tuple[str, float, str], ...]

    def __post_init__(self):
        bad = [op for op, _, _ in self.steps if op not in _OPS]
        if bad:
            raise ValueError(f"unknown comparison(s) {bad}; use one of {sorted(_OPS)}")

    def classify(self, values, valid: Optional[np.ndarray] = None, missing: str = "NO_DATA") -> np.ndarray:
        """The label of every sample (``missing`` where the value is not valid; default validity: finite)."""
        x = np.asarray(values, dtype=float)
        ok = np.isfinite(x) if valid is None else np.asarray(valid, dtype=bool)
        out = np.full(x.shape, missing, dtype=object)
        out[ok] = self.base
        with np.errstate(invalid="ignore"):
            for op, threshold, label in self.steps:
                out[ok & _OPS[op](x, threshold)] = label
        return out

    def label_of(self, value: float) -> Optional[str]:
        """The label of one value (None when it is not finite)."""
        if value is None or not np.isfinite(value):
            return None
        return str(self.classify(np.array([float(value)]))[0])

    @property
    def edges(self) -> Tuple[float, ...]:
        """The thresholds, ascending."""
        return tuple(sorted({float(t) for _, t, _ in self.steps}))

    @property
    def labels(self) -> Tuple[str, ...]:
        """The labels from the lowest to the highest values."""
        e = self.edges
        if not e:
            return (self.base,)
        gaps = [e[0] - 1.0] + [(a + b) / 2 for a, b in zip(e, e[1:])] + [e[-1] + 1.0]
        order = []
        for v in gaps:
            label = self.label_of(v)
            if label not in order:
                order.append(label)
        return tuple(order)

    def without_edges(self, *edges: float) -> "Bands":
        """A coarsening of these bands: the steps at ``edges`` dropped.

        The classes either side of a dropped edge merge, under the label the
        remaining steps give them.

        Examples
        --------
        >>> b = Bands("POOR", ((">", 1.0, "FAIR"), ("<=", 0.1, "VERY_POOR")))
        >>> b.without_edges(0.1).labels
        ('POOR', 'FAIR')
        """
        drop = {float(edge) for edge in edges}
        unknown = drop - set(self.edges)
        if unknown:
            raise ValueError(f"no step at {sorted(unknown)}; the edges are {self.edges}")
        return Bands(self.base, tuple(step for step in self.steps if float(step[1]) not in drop))

    def ranges(self) -> Tuple[Tuple[str, float, bool, float, bool], ...]:
        """The range of each label, from the lowest to the highest values.

        Each item is ``(label, lower, lower_closed, upper, upper_closed)``; an
        open end is ``-inf`` or ``inf``.

        Raises
        ------
        ValueError
            If a label covers more than one range.
        """
        e = self.edges
        if not e:
            return ((self.base, -np.inf, False, np.inf, False),)
        # Probe below the first edge, at each edge, between edges and above the last.
        points = [(e[0] - 1.0, None)]
        for i, edge in enumerate(e):
            points.append((edge, edge))
            points.append(((edge + e[i + 1]) / 2 if i + 1 < len(e) else edge + 1.0, None))
        runs: list = []
        for i, (value, _) in enumerate(points):
            label = self.label_of(value)
            if runs and runs[-1][0] == label:
                runs[-1][2] = i
            else:
                runs.append([label, i, i])
        seen = set()
        result = []
        for label, first, last in runs:
            if label in seen:
                raise ValueError(f"{label} covers more than one range")
            seen.add(label)
            if first == 0:
                lower, lower_closed = -np.inf, False
            elif points[first][1] is not None:
                lower, lower_closed = points[first][1], True
            else:
                lower, lower_closed = points[first - 1][1], False
            if last == len(points) - 1:
                upper, upper_closed = np.inf, False
            elif points[last][1] is not None:
                upper, upper_closed = points[last][1], True
            else:
                upper, upper_closed = points[last + 1][1], False
            result.append((label, lower, lower_closed, upper, upper_closed))
        return tuple(result)

    def describe(self) -> str:
        """The range of each label as text, e.g. ``"LOW < 0.5; HIGH >= 0.5"``."""
        parts = []
        for label, lower, lower_closed, upper, upper_closed in self.ranges():
            low = f"{'>=' if lower_closed else '>'} {lower:g}"
            high = f"{'<=' if upper_closed else '<'} {upper:g}"
            if np.isfinite(lower) and np.isfinite(upper):
                parts.append(f"{label} {low} and {high}")
            elif np.isfinite(lower):
                parts.append(f"{label} {low}")
            elif np.isfinite(upper):
                parts.append(f"{label} {high}")
            else:
                parts.append(f"{label} for every value")
        return "; ".join(parts)


@dataclass(frozen=True)
class TwoCurveBands:
    """A two-curve classification: ``base`` for every sample where both curves
    are valid, then each ``(label, x_above, y_above)`` step in order where
    ``x > x_above`` and ``y > y_above``; later steps win.

    Examples
    --------
    >>> b = TwoCurveBands("LOW", (("MID", 0.1, 1.0), ("HIGH", 0.2, 10.0)))
    >>> list(b.classify([0.05, 0.15, 0.25, 0.25], [50.0, 50.0, 5.0, 50.0]))
    ['LOW', 'MID', 'MID', 'HIGH']
    """

    base: str
    steps: Tuple[Tuple[str, float, float], ...]

    def classify(self, x, y, valid: Optional[np.ndarray] = None, missing: str = "NO_DATA") -> np.ndarray:
        """The label of every sample (``missing`` where either value is not valid; default validity: both finite)."""
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        ok = (np.isfinite(xa) & np.isfinite(ya)) if valid is None else np.asarray(valid, dtype=bool)
        out = np.full(xa.shape, missing, dtype=object)
        out[ok] = self.base
        with np.errstate(invalid="ignore"):
            for label, x_above, y_above in self.steps:
                out[ok & (xa > x_above) & (ya > y_above)] = label
        return out

    @property
    def labels(self) -> Tuple[str, ...]:
        """The labels from the lowest to the highest grade."""
        return (self.base, *(label for label, _, _ in self.steps))


@dataclass(frozen=True)
class ClassTable:
    """The canonical classes of one property.

    Attributes
    ----------
    name : str
        The property's curve name, e.g. ``"VCL"``.
    quantity : str
        What the values are, e.g. ``"clay volume"``.
    unit : str
        Unit of the values (``""`` when dimensionless).
    bands : Bands
        The classes.
    missing : str
        Label of a sample without a valid value.
    source : str
        Where the edges come from.
    """

    name: str
    quantity: str
    unit: str
    bands: Bands
    missing: str
    source: str

    def classify(self, values, valid: Optional[np.ndarray] = None) -> np.ndarray:
        """The label of every sample, ``missing`` where the value is not valid (default validity: finite)."""
        return self.bands.classify(values, valid, missing=self.missing)

    def label_of(self, value: float) -> Optional[str]:
        """The label of one value (None when it is not finite)."""
        return self.bands.label_of(value)

    @property
    def labels(self) -> Tuple[str, ...]:
        """The labels from the lowest to the highest values."""
        return self.bands.labels

    def describe(self, bands: Optional[Bands] = None) -> str:
        """The classes as threshold text; ``bands``, a coarsening of the table, to describe that instead."""
        unit = f", {self.unit}" if self.unit else ""
        return (
            f"{(bands or self.bands).describe()} ({self.name}{unit}); "
            f"{self.missing} where {self.name} is missing."
        )


@dataclass(frozen=True)
class TwoCurveClassTable:
    """The canonical classes of a property graded from two curves.

    Attributes
    ----------
    name : str
        The property's name.
    quantities : tuple of str
        What the two curves are, in the order :meth:`classify` takes them.
    units : tuple of str
        Units of the two curves.
    bands : TwoCurveBands
        The classes.
    missing : str
        Label of a sample where either curve has no valid value.
    source : str
        Where the edges come from.
    """

    name: str
    quantities: Tuple[str, str]
    units: Tuple[str, str]
    bands: TwoCurveBands
    missing: str
    source: str

    def classify(self, x, y, valid: Optional[np.ndarray] = None) -> np.ndarray:
        """The label of every sample, ``missing`` where either value is not valid (default validity: both finite)."""
        return self.bands.classify(x, y, valid, missing=self.missing)

    @property
    def labels(self) -> Tuple[str, ...]:
        """The labels from the lowest to the highest grade."""
        return self.bands.labels

    def describe(self) -> str:
        """The classes as threshold text."""
        (x_name, y_name), (x_unit, y_unit) = self.quantities, self.units
        parts = [
            f"{label}: {x_name} > {x_above:g} {x_unit} and {y_name} > {y_above:g} {y_unit}"
            for label, x_above, y_above in reversed(self.bands.steps)
        ]
        lowest = self.bands.steps[0][0] if self.bands.steps else "any class"
        parts.append(f"{self.bands.base}: below {lowest}")
        return "; ".join(parts) + f"; {self.missing} where either curve is missing."


def _family_limit(limit: Union[float, Mapping[str, float]], family: str) -> float:
    """A limit's value for ``family``: its entry in a per-family mapping, or the one value."""
    if isinstance(limit, Mapping):
        if family not in limit:
            raise ValueError(f"no limit for lithology family {family!r}; the limits cover {sorted(limit)}")
        return float(limit[family])
    return float(limit)


@dataclass(frozen=True)
class ReservoirGrade:
    """The limits of one reservoir-quality grade.

    A sample gets the grade where its effective porosity is above
    ``porosity_above``, its permeability above ``permeability_above`` and its
    clay volume below ``clay_below``. Each limit is one value, or a mapping of
    lithology family (``"clastic"``, ``"carbonate"``) to value.

    Examples
    --------
    >>> grade = ReservoirGrade("FAIR", {"clastic": 0.08, "carbonate": 0.045}, 0.1, 0.4)
    >>> grade.limits("carbonate")
    (0.045, 0.1, 0.4)
    """

    label: str
    porosity_above: Union[float, Mapping[str, float]]
    permeability_above: Union[float, Mapping[str, float]]
    clay_below: Union[float, Mapping[str, float]]

    def limits(self, family: str) -> Tuple[float, float, float]:
        """The porosity, permeability and clay-volume limits for ``family``."""
        return (
            _family_limit(self.porosity_above, family),
            _family_limit(self.permeability_above, family),
            _family_limit(self.clay_below, family),
        )


@dataclass(frozen=True)
class ReservoirQualityTable:
    """The canonical reservoir-quality grades: effective porosity, permeability
    and clay volume, with limits per lithology family.

    A sample where all three values are valid gets ``base``, then each grade
    of ``grades`` (lowest first) whose limits it passes for its lithology
    family; later grades win. Each grade's limits imply those of the grade
    before it, so a sample gets the highest grade it passes.

    Attributes
    ----------
    name : str
        The property's name, ``"RESERVOIR_QUALITY"``.
    quantities : tuple of str
        What the three curves are, in the order :meth:`classify` takes them.
    units : tuple of str
        Units of the three curves.
    base : str
        Label of a sample that passes no grade.
    grades : tuple of ReservoirGrade
        The grades, lowest first.
    missing : str
        Label of a sample where any of the three curves has no valid value.
    families : tuple of str
        The lithology families the limits are given for.
    source : str
        Where the limits come from.

    Examples
    --------
    >>> table = CLASS_TABLES["RESERVOIR_QUALITY"]
    >>> list(table.classify([0.06, 0.06, 0.2], [5.0, 5.0, 50.0], [0.1, 0.1, 0.1],
    ...                     ["clastic", "carbonate", "carbonate"]))
    ['NON_RESERVOIR', 'FAIR_RESERVOIR', 'EXCELLENT_RESERVOIR']
    """

    name: str
    quantities: Tuple[str, str, str]
    units: Tuple[str, str, str]
    base: str
    grades: Tuple[ReservoirGrade, ...]
    missing: str
    families: Tuple[str, ...]
    source: str

    def __post_init__(self):
        for family in self.families:
            for lower, upper in zip(self.grades, self.grades[1:]):
                (phi_low, k_low, clay_low), (phi_up, k_up, clay_up) = lower.limits(family), upper.limits(family)
                if phi_up < phi_low or k_up < k_low or clay_up > clay_low:
                    raise ValueError(f"{upper.label} must imply {lower.label} in {family} rock")

    def classify(self, porosity, permeability, clay, family="clastic", valid: Optional[np.ndarray] = None) -> np.ndarray:
        """The grade of every sample, ``missing`` where a value is not valid (default validity: all three finite).

        ``family`` is the lithology family of every sample, or of each sample.

        Raises
        ------
        ValueError
            If a family is not one of :attr:`families`.
        """
        phi, k, clay_volume = np.broadcast_arrays(
            np.asarray(porosity, dtype=float),
            np.asarray(permeability, dtype=float),
            np.asarray(clay, dtype=float),
        )
        families = np.broadcast_to(np.asarray(family, dtype=object), phi.shape)
        unknown = sorted(set(np.unique(families.astype(str)).tolist()) - set(self.families))
        if unknown:
            raise ValueError(f"unknown lithology family {unknown}; use one of {self.families}")
        if valid is None:
            ok = np.isfinite(phi) & np.isfinite(k) & np.isfinite(clay_volume)
        else:
            ok = np.asarray(valid, dtype=bool)
        out = np.full(phi.shape, self.missing, dtype=object)
        out[ok] = self.base
        with np.errstate(invalid="ignore"):
            for name in self.families:
                in_family = ok & (families == name)
                for grade in self.grades:
                    phi_above, k_above, clay_below = grade.limits(name)
                    out[in_family & (phi > phi_above) & (k > k_above) & (clay_volume < clay_below)] = grade.label
        return out

    @property
    def labels(self) -> Tuple[str, ...]:
        """The labels from the lowest to the highest grade."""
        return (self.base, *(grade.label for grade in self.grades))

    def describe(self, families: Optional[Sequence[str]] = None) -> str:
        """The grades as threshold text, for ``families`` (default: every family)."""
        wanted = set(self.families if families is None else families)
        chosen = [name for name in self.families if name in wanted] or list(self.families)
        (phi_name, k_name, clay_name), (phi_unit, k_unit, clay_unit) = self.quantities, self.units
        blocks = []
        for name in chosen:
            parts = []
            for grade in reversed(self.grades):
                phi_above, k_above, clay_below = grade.limits(name)
                parts.append(
                    f"{grade.label}: {phi_name} > {phi_above:g} {phi_unit}, {k_name} > {k_above:g} {k_unit} "
                    f"and {clay_name} < {clay_below:g} {clay_unit}"
                )
            blocks.append(f"{name.capitalize()} rock: " + "; ".join(parts) + ".")
        lowest = self.grades[0].label if self.grades else "any grade"
        return " ".join(blocks) + (
            f" {self.base}: below {lowest}; {self.missing} where {phi_name}, {k_name} or {clay_name} is missing."
        )


def bands_of(analyzer, name: str) -> Bands:
    """The :class:`Bands` an analyzer publishes under ``name`` (e.g. ``"DISTRIBUTION_BANDS"``).

    An analyzer class gives its benchmark table (the class attribute); an
    analyzer instance gives the table it classifies with.
    """
    bands = getattr(analyzer, name, None)
    if not isinstance(bands, Bands):
        owner = getattr(analyzer, "__name__", None) or type(analyzer).__name__
        raise ValueError(f"{owner} has no Bands named {name!r}")
    return bands


_FZI_GHE_EDGES = (0.0938, 0.1875, 0.375, 0.75, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0)

_TABLES = (
    ClassTable(
        "VCL", "clay volume", "v/v",
        Bands("CLEAN_FORMATION", ((">=", 0.25, "MODERATE_CLAY"), (">", 0.50, "SHALY_FORMATION"))),
        "NO_VCL_DATA",
        "Library convention, as in VclClassification: no industry-standard clay-volume classes "
        "exist. Net-pay, Sw-model routing, lithology-log, crossover and seal cutoffs are separate "
        "cutoffs for those jobs.",
    ),
    ClassTable(
        "PHI", "porosity", "v/v",
        Bands("POOR_POROSITY", ((">", 0.06, "FAIR_POROSITY"), (">", 0.12, "GOOD_POROSITY"),
                                (">", 0.20, "EXCELLENT_POROSITY"))),
        "NO_POROSITY_DATA",
        "Library convention, as in PorosityClassification. Published scales differ; "
        "Koesoemadinata (1980), for example, grades 5-10 % poor, 10-15 % fair, 15-20 % good, "
        "20-25 % very good and above 25 % excellent.",
    ),
    ClassTable(
        "PERM", "permeability", "mD",
        Bands("POOR_PERMEABILITY", ((">", 1.0, "FAIR_PERMEABILITY"), (">", 10.0, "GOOD_PERMEABILITY"),
                                    (">", 100.0, "VERY_GOOD_PERMEABILITY"),
                                    (">", 1000.0, "EXCELLENT_PERMEABILITY"),
                                    ("<=", 0.1, "VERY_POOR_PERMEABILITY"))),
        "NO_PERMEABILITY_DATA",
        "PermeabilityClassification: decade edges at 1, 10, 100 and 1000 mD, and 0.1 mD, the "
        "US FERC tight-gas limit (in situ permeability of 0.1 mD or less; AAPG Wiki, \"Tight gas "
        "reservoirs: evaluation\").",
    ),
    ClassTable(
        "SW", "water saturation", "v/v",
        Bands("EXCELLENT_HC_SATURATION", ((">=", 0.20, "GOOD_HC_SATURATION"), (">=", 0.40, "FAIR_HC_SATURATION"),
                                          (">=", 0.60, "WATER_BEARING"))),
        "NO_SATURATION_DATA",
        "Library convention, as in HcSaturationClassification; the 0.60 edge equals the "
        "clastic net-pay Sw cutoff (STANDARD_CUTOFFS).",
    ),
    ClassTable(
        "BVW", "bulk volume water", "v/v",
        Bands("VERY_LOW_BVW", ((">=", 0.03, "LOW_BVW"), (">=", 0.06, "MODERATE_BVW"), (">=", 0.10, "HIGH_BVW"))),
        "NO_BVW_DATA",
        "BVW = PHIE x SW. It is near constant in rock at irreducible water saturation "
        "(Buckles, 1965), at a level that rises as grain and pore size fall, so the classes "
        "describe BVW rather than producibility. Library convention for the edges.",
    ),
    ClassTable(
        "RQI", "reservoir quality index", "um",
        Bands("LOW_RQI", ((">=", 0.5, "MODERATE_RQI"), (">", 1.0, "HIGH_RQI"))),
        "NO_RQI_DATA",
        "RQI = 0.0314 sqrt(k / PHIE), in um (Amaefule et al., 1993). Library convention for "
        "the edges, as in RQIClassification.",
    ),
    ClassTable(
        "FZI", "flow zone indicator", "um",
        Bands("BELOW_GHE_1", tuple((">=", edge, f"GHE_{number}")
                                   for number, edge in enumerate(_FZI_GHE_EDGES, start=1))),
        "NO_FZI_DATA",
        "Global hydraulic elements (Corbett and Potter, 2004, SCA2004-30, Table 1): GHE 1 to 10 "
        "start at FZI 0.0938, 0.1875, 0.375, 0.75, 1.5, 3, 6, 12, 24 and 48 um, with FZI = RQI / "
        "PHI_Z (Amaefule et al., 1993). Hydraulic units are field-specific, so no other "
        "field-independent FZI table exists; GHE numbers rise with FZI, as the library's "
        "hydraulic flow units do.",
    ),
    ClassTable(
        "MHI", "moveable hydrocarbon index", "",
        Bands("HIGH_MOVEABLE_HC", ((">=", 0.5, "MODERATE_MOVEABLE_HC"), (">=", 0.7, "LOW_MOVEABLE_HC"),
                                   (">=", 0.9, "MINIMAL_MOVEABLE_HC"))),
        "NO_MHI_DATA",
        "MHI = Sw / Sxo. At 1 or above no hydrocarbon moved during invasion; below 0.7 in "
        "sandstones (0.6 in carbonates) moveable hydrocarbons are indicated (Asquith, 1982; "
        "Nordeng, 2007, NDGS Geologic Investigations 48). Edges as in MhiClassification.",
    ),
    ClassTable(
        "RXO_RT", "flushed-to-deep resistivity ratio Rxo/Rt", "",
        Bands("CONDUCTIVE_INVASION", ((">=", 0.8, "MINIMAL_INVASION"), (">", 1.2, "RESISTIVE_INVASION"))),
        "NO_INVASION_DATA",
        "Library convention, as in InvasionClassification: a ratio within 0.8-1.2 shows no "
        "invasion contrast.",
    ),
)

#: Net reservoir rock of each lithology family, as net pay counts it
#: (STANDARD_CUTOFFS): effective porosity (v/v) above the net-reservoir cutoff
#: and clay volume (v/v) below the net-rock cutoff.
_NET_RESERVOIR_POROSITY = MappingProxyType({
    "clastic": DefaultParameters.PHIE_CUTOFF_CLASTIC,
    "carbonate": DefaultParameters.PHIE_CUTOFF_CARBONATE,
})
_NET_ROCK_CLAY = MappingProxyType({
    "clastic": DefaultParameters.VCL_CUTOFF_CLASTIC,
    "carbonate": DefaultParameters.VCL_CUTOFF_CARBONATE,
})
#: Clay volume (v/v) below which rock is CLEAN_FORMATION in the VCL table.
_CLEAN_CLAY = next(
    upper
    for label, _, _, upper, _ in next(table for table in _TABLES if table.name == "VCL").bands.ranges()
    if label == "CLEAN_FORMATION"
)

_RESERVOIR_QUALITY = ReservoirQualityTable(
    "RESERVOIR_QUALITY",
    ("effective porosity", "permeability", "clay volume"),
    ("v/v", "mD", "v/v"),
    "NON_RESERVOIR",
    (
        ReservoirGrade("FAIR_RESERVOIR", _NET_RESERVOIR_POROSITY, 0.1, _NET_ROCK_CLAY),
        ReservoirGrade("GOOD_RESERVOIR", 0.10, 1.0, _NET_ROCK_CLAY),
        ReservoirGrade("EXCELLENT_RESERVOIR", 0.15, 10.0, _CLEAN_CLAY),
    ),
    "NO_QUALITY_DATA",
    ("clastic", "carbonate"),
    "Every grade is net reservoir rock of the sample's lithology family, as net pay counts it "
    "(STANDARD_CUTOFFS): effective porosity above the family's net-reservoir cutoff and clay "
    "volume below its net-rock cutoff, PHIE > 0.08 and VCL < 0.40 in clastic rock, PHIE > 0.045 "
    "and VCL < 0.30 in carbonate rock. Both clay cutoffs lie inside MODERATE_CLAY (0.25-0.50) of "
    "the VCL table, so no SHALY_FORMATION rock is graded. FAIR_RESERVOIR adds permeability above "
    "0.1 mD, the tight limit of the PERM table. GOOD_RESERVOIR needs 10 % effective porosity and "
    "1 mD and keeps the net-rock clay limit, below the 0.50 edge of MODERATE_CLAY: moderately "
    "clayey net rock can be good. EXCELLENT_RESERVOIR needs 15 % and 10 mD in CLEAN_FORMATION "
    "rock (VCL < 0.25): only clean rock is excellent. The 10 % and 15 % edges start the fair and "
    "good porosity classes of Koesoemadinata (1980); 1 and 10 mD are edges of the PERM table. A "
    "sample without effective porosity, permeability or clay volume is NO_QUALITY_DATA: it gets "
    "no grade, NON_RESERVOIR included.",
)

#: The canonical class table of each property, keyed by its curve name: VCL,
#: PHI (porosity), PERM, SW, BVW, RQI, FZI, MHI, RXO_RT (Rxo/Rt) and
#: RESERVOIR_QUALITY (effective porosity, permeability and clay volume, with
#: limits per lithology family).
CLASS_TABLES: Mapping[str, Union[ClassTable, ReservoirQualityTable]] = MappingProxyType(
    {table.name: table for table in (*_TABLES, _RESERVOIR_QUALITY)}
)

__all__ = [
    "Bands",
    "TwoCurveBands",
    "ClassTable",
    "TwoCurveClassTable",
    "ReservoirGrade",
    "ReservoirQualityTable",
    "CLASS_TABLES",
    "bands_of",
]
