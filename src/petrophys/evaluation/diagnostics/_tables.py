"""
The class-table sets of the diagnostics analyzers.

``"current"`` (default) classifies with the canonical tables of
:data:`~petrophys.model.class_tables.CLASS_TABLES`, so a property gets the
same class in every analyzer. ``"geolink"`` keeps the tables of the GeoLink
eval package: frozen copies that each analyzer holds as class attributes and
inline rules, together with their threshold texts, which never read the
registry.

Under ``"current"`` the analyzers that grade reservoir quality (porosity,
permeability, hydrocarbon and lithology) read the same three curves and apply
the limits of each sample's lithology family (:func:`sample_families`), so a
sample gets one reservoir-quality class in every category.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from petrophys._internal.analysis_helpers import format_curve_source, get_curve_data
from petrophys.model.class_tables import CLASS_TABLES

if TYPE_CHECKING:
    from petrophys.model.well import Well

#: The class-table sets an analyzer takes.
CLASS_TABLE_SETS = ("current", "geolink")


def class_table_set(value: str) -> str:
    """Validate an analyzer's ``class_tables`` argument."""
    text = str(value).strip().lower()
    if text == "benchmark":
        raise ValueError('class_tables "benchmark" is now "geolink": the GeoLink eval package\'s tables')
    if text not in CLASS_TABLE_SETS:
        raise ValueError(f"class_tables must be one of {CLASS_TABLE_SETS}, not {value!r}")
    return text


def rxo_rt_ratio(well: "Well") -> Tuple[Optional[np.ndarray], str]:
    """The Rxo/Rt ratio the current class tables classify, and its source.

    The workflow's INVASION_RATIO when the well has one, so the classes agree
    with the workflow's invasion statistics; otherwise RXO (then RSHA, RESS)
    over RDEP (then RT), from
    :func:`~petrophys.interpretation.electrical.invasion.calculate_invasion_ratio`.
    ``(None, "unavailable")`` without either.
    """
    from petrophys.interpretation.electrical.invasion.invasion_module import calculate_invasion_ratio

    ratio, source = get_curve_data(well, ["INVASION_RATIO"])
    if ratio is not None:
        return ratio, format_curve_source(source)
    rxo, rxo_source = get_curve_data(well, ["RXO", "RSHA", "RESS"])
    rdep, rdep_source = get_curve_data(well, ["RDEP", "RT"])
    if rxo is None or rdep is None:
        return None, "unavailable"
    ratio = np.asarray(calculate_invasion_ratio(rxo, rdep), dtype=float)
    return ratio, f"{format_curve_source(rxo_source)} / {format_curve_source(rdep_source)}"


#: Clay-volume curves the reservoir-quality table reads, in order of
#: preference: net pay's order, so a grade's clay limit tests the clay volume
#: that net pay tests.
QUALITY_CLAY_CURVES = ("VCL", "VCLAY", "VSH")

#: Short rock names read as the rock they stand for, as the dynamic workflow reads them.
_ROCK_ABBREVIATIONS = {"lime": "limestone", "ls": "limestone", "dolo": "dolomite"}


def lithology_family(lithology) -> Optional[str]:
    """The lithology family, ``"clastic"`` or ``"carbonate"``, of a lithology or family name.

    A name is read by the rock it ends with, so SANDY LIMESTONE is carbonate
    and CALCAREOUS SHALE clastic; digits and punctuation are ignored
    (DOLOMITE2). The rock's family is that of
    :func:`~petrophys.evaluation.net_pay.cutoffs.cutoff_family`: limestone,
    chalk, dolomite and marl are carbonate, every other rock clastic. None
    when the value names nothing (None, NaN or an empty name).

    Examples
    --------
    >>> [lithology_family(name) for name in ("CHALK", "SANDY LIMESTONE", "CALCAREOUS SHALE", "DOLOMITE2", "")]
    ['carbonate', 'carbonate', 'clastic', 'carbonate', None]
    """
    from petrophys.evaluation.net_pay.cutoffs import cutoff_family

    if lithology is None or (isinstance(lithology, float) and not np.isfinite(lithology)):
        return None
    words = re.findall(r"[a-z]+", str(lithology).lower())
    if not words or words[-1] in {"nan", "none"}:
        return None
    return cutoff_family(_ROCK_ABBREVIATIONS.get(words[-1], words[-1]))


def sample_families(families, n_samples: int) -> np.ndarray:
    """Each sample's lithology family for the reservoir-quality table.

    ``families`` is None, one lithology or family name for every sample, or
    one per sample; :func:`lithology_family` reads each. A sample whose
    entry names nothing, and every sample when ``families`` is None, is
    clastic, the library's default lithology domain.

    Raises
    ------
    ValueError
        If a per-sample sequence does not have ``n_samples`` entries.
    """
    if families is None or isinstance(families, str):
        return np.full(n_samples, lithology_family(families) or "clastic", dtype=object)
    values = np.asarray(families, dtype=object)
    if values.shape != (n_samples,):
        raise ValueError(f"lithology_family has {values.size} entries for {n_samples} samples")
    resolved: dict = {}
    out = np.empty(n_samples, dtype=object)
    for index, value in enumerate(values):
        key = value if isinstance(value, str) else repr(value)
        if key not in resolved:
            resolved[key] = lithology_family(value) or "clastic"
        out[index] = resolved[key]
    return out


def reservoir_quality(well: "Well", porosity, permeability, families) -> Tuple[np.ndarray, Optional[str]]:
    """Each sample's grade in the canonical reservoir-quality table, and the clay-volume curve read.

    ``porosity`` and ``permeability`` are the analyzer's effective porosity
    and permeability (None when the well has none); the clay volume is read
    from :data:`QUALITY_CLAY_CURVES`, and ``families`` (one per sample, see
    :func:`sample_families`) selects each sample's limits. Every sample is
    the table's missing label when a curve is missing.
    """
    table = CLASS_TABLES["RESERVOIR_QUALITY"]
    clay, clay_source = get_curve_data(well, list(QUALITY_CLAY_CURVES))
    if porosity is None or permeability is None or clay is None:
        return np.full(well.n_samples, table.missing, dtype=object), clay_source
    return table.classify(porosity, permeability, clay, families), clay_source


def reservoir_quality_text(families) -> str:
    """The reservoir-quality table's grades as text, for the lithology families among ``families``."""
    table = CLASS_TABLES["RESERVOIR_QUALITY"]
    present = set(np.asarray(families, dtype=object).ravel().tolist())
    return table.describe([name for name in table.families if name in present])
