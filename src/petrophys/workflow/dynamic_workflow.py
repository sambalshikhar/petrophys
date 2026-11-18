"""
Dynamic petrophysics workflow.

This module provides a DynamicWorkflow class that automatically detects
available input curves and runs the appropriate steps in the correct
order. It supports both a full run (:meth:`DynamicWorkflow.run`) and
selected steps (:meth:`DynamicWorkflow.run_selected_steps`).

Pipeline order (:meth:`DynamicWorkflow.get_pipeline`):
    1. QC: curve standardization, temperature profile, bad-hole flag
    2. Clay volume and clay resistivity
    3. Mud-filtrate resistivity and the opt-in hydrocarbon correction
    4. Porosity
    5. Lithology (mineral solve)
    6. Core porosity correction
    7. Formation water resistivity and invasion
    8. Water saturation
    9. Permeability
    10. Rock quality (RQI, FZI), bulk volume water, moveable hydrocarbon index
    11. Net pay

The spectral gamma, elemental spectroscopy and porosity refinement steps
are not run by :meth:`DynamicWorkflow.run`;
:meth:`DynamicWorkflow.run_selected_steps` runs them when asked.
"""

from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from pathlib import Path
import re
from petrophys._internal.benchmark import reproducing_benchmark
from petrophys.model.well import Well
from petrophys.model.class_tables import Bands
from petrophys.model.data_types import DataType
from petrophys.model.units import UnitConverter, temperature_to_fahrenheit, temperature_unit
from petrophys._internal.analysis_helpers import (
    curve_source_mnemonic,
    get_curve,
    get_curve_data_depthwise,
    interval_strings,
)
from petrophys.preprocessing.quality.bad_hole.flag_generator import (
    _known_bit_size,
    _plausible_bit_size,
)
import pandas as pd
from petrophys.io import ascii_reader, las_reader

DEFAULT_RESISTIVITY_REFERENCE_TEMP_F = 75.0
APPARENT_POROSITY_METHODS = {
    "density_only": {
        "source": "density-derived",
        "basis": "PHID",
        "reason": "only density porosity (PHID) was available",
    },
    "neutron_only": {
        "source": "neutron-derived",
        "basis": "PHIN",
        "reason": "only neutron porosity (PHIN) was available",
    },
    "sonic_only": {
        "source": "sonic-derived",
        "basis": "PHIS",
        "reason": "only sonic porosity (PHIS) was available",
    },
}


def _apparent_porosity_info(phit_method: Optional[str]) -> Optional[Dict[str, str]]:
    """Return provenance for single-log porosity routes."""
    return APPARENT_POROSITY_METHODS.get(str(phit_method or "").strip().lower())


class _SpRwRejected(Exception):
    """The SP estimate of Rw sat at the method's limits; the next Rw route is tried."""


class WorkflowStep(Enum):
    """Individual workflow steps."""
    NORMALIZATION = "normalization"
    BAD_HOLE = "bad_hole"
    SPECTRAL_GAMMA = "spectral_gamma"
    ELEMENTAL_SPECTROSCOPY = "elemental_spectroscopy"
    TEMPERATURE = "temperature"
    VCLAY = "vclay"
    RCL = "rcl"
    LITHOLOGY = "lithology"
    RMF = "rmf"
    RW = "rw"
    INVASION = "invasion"
    HC_CORRECTION = "hc_correction"
    POROSITY = "porosity"
    POROSITY_REFINEMENT = "porosity_refinement"
    CORE_CORRECTION = "core_correction"
    SATURATION = "saturation"
    PERMEABILITY = "permeability"
    ROCK_QUALITY = "rock_quality"
    BVW = "bvw"
    MHI = "mhi"
    NET_PAY = "net_pay"


def _parameter_float(well: "Well", name: str) -> Optional[float]:
    """Fetch a numeric well parameter if present."""
    value = well.get_parameter(name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parameter_text(well: "Well", name: str) -> str:
    """Fetch a text well parameter if present."""
    value = well.get_parameter(name)
    return str(value or "").strip()


def _constant_zoned_parameter(well: "Well", prefix: str) -> Optional[float]:
    """
    Return a unique zoned LAS parameter value, e.g. RW_Z1/RW_Z2 -> RW.

    If multiple distinct zoned values exist, return None so callers do not
    silently choose the wrong interval constant.
    """
    values = set()
    pattern = re.compile(rf"^{re.escape(prefix.upper())}_Z\d+$")

    for key, value in getattr(well, "_parameters", {}).items():
        if not isinstance(key, str) or "__" in key:
            continue
        if not pattern.match(key.upper()):
            continue
        try:
            values.add(float(value))
        except (TypeError, ValueError):
            continue

    if len(values) == 1:
        return next(iter(values))
    return None


def _parameter_temperature_f(well: "Well", name: str) -> Optional[float]:
    """Fetch a numeric temperature parameter and normalize it to Fahrenheit."""
    value = _parameter_float(well, name)
    unit = _parameter_text(well, f"{name}__UNIT")
    if value is not None:
        return temperature_to_fahrenheit(value, unit)

    value = _constant_zoned_parameter(well, name)
    unit = _parameter_text(well, f"{name}_Z1__UNIT")
    if value is not None:
        return temperature_to_fahrenheit(value, unit)

    return None


def _metadata_bht_f(well: "Well") -> Optional[float]:
    """Return BHT metadata in Fahrenheit."""
    if well.header.bht is not None:
        unit = _parameter_text(well, "BHT__UNIT") or getattr(well.header, "bht_unit", "F")
        return temperature_to_fahrenheit(float(well.header.bht), unit)

    value = _parameter_float(well, "BHT")
    unit = _parameter_text(well, "BHT__UNIT").lower()
    if value is not None:
        return temperature_to_fahrenheit(value, unit)

    value = _constant_zoned_parameter(well, "BHT")
    unit = _parameter_text(well, "BHT_Z1__UNIT").lower()
    if value is not None:
        return temperature_to_fahrenheit(value, unit)

    return None


GROUND_TRUTH_CODE_TO_GROUP = {
    1: "sandstone",
    2: "sandstone",
    3: "sandstone",
    4: "sandstone",
    5: "silt",
    6: "silt",
    7: "shale",
    8: "shale",
    9: "limestone",
    10: "sandstone",
    11: "limestone",
    12: "limestone",
    13: "limestone",
    14: "calcareous_shale",
    15: "dolomite",
    16: "limestone",
    17: "clastic_coarse",
    18: "coal",
    19: "igneous",
    20: "igneous",
    21: "halite",
    22: "anhydrite",
    23: "igneous",
    24: "igneous",
    25: "shale",
    26: "evaporite",
    27: "halite",
    28: "igneous",
    29: "chert",
    30: "sandstone",
    31: "clastic_coarse",
    32: "halite",
    33: "halite",
    34: "halite",
    35: "sandstone",
    36: "clastic_coarse",
}

COMPUTED_LITHO_CODE_TO_GROUP = {
    1: "sandstone",
    2: "limestone",
    3: "dolomite",
    4: "shale",
    5: "anhydrite",
    6: "halite",
    7: "coal",
}

LITHOLOGY_GROUP_TO_MATRIX = {
    "sandstone": "sandstone",
    "silt": "sandstone",
    "clastic_coarse": "sandstone",
    "igneous": "quartz",
    "chert": "quartz",
    "limestone": "limestone",
    "dolomite": "dolomite",
    "shale": "shale",
    "calcareous_shale": "shale",
    "anhydrite": "anhydrite",
    "evaporite": "gypsum",
    "halite": "halite",
    "coal": "coal",
}

# The "current" calculations regroup a few lithology codes; the "geolink" profile
# keeps GROUND_TRUTH_CODE_TO_GROUP. Code 10 is "Porous Limestone" in the Kingdom
# lithology codebook ("Calcareous Sandstone" is only its display pattern) and
# logs as a carbonate; code 26, "Gap", records no lithology. The potash salts
# get their own matrix endpoints: sylvite and carnallite read lighter than
# halite on the density log, and the hydrated salts read high on the neutron
# log. Lithology intervals may also name gypsum.
_CURRENT_GROUND_TRUTH_CODE_TO_GROUP = {
    **GROUND_TRUTH_CODE_TO_GROUP,
    10: "limestone",
    26: None,
    27: "sylvinite",
    32: "polyhalite",
    33: "carnallite",
    34: "kainite",
}

_CURRENT_LITHOLOGY_GROUP_TO_MATRIX = {
    **LITHOLOGY_GROUP_TO_MATRIX,
    # Total porosity counts the clay-bound water, so shale takes the grain
    # endpoints of its solids, as phi_cl does: the wet-shale bulk density
    # (2.58 g/cm3) as a matrix would count that water as rock.
    "shale": "sandstone",
    "calcareous_shale": "limestone",
    "gypsum": "gypsum",
    "sylvinite": "sylvite",
    "polyhalite": "polyhalite",
    "carnallite": "carnallite",
    "kainite": "kainite",
}

#: Lithology groups counted as clastic when a codebook resolves the lithology domain.
_CLASTIC_GROUPS = frozenset({"sandstone", "silt", "shale", "clastic_coarse", "calcareous_shale"})

#: Lithology groups that are evaporites (halite and other salts, anhydrite,
#: gypsum). A "current" run never counts them as net reservoir or net pay.
EVAPORITE_LITHOLOGY_GROUPS = frozenset({
    "evaporite", "halite", "salt", "anhydrite", "gypsum",
    "sylvinite", "carnallite", "kainite", "polyhalite",
})

#: Other lithology groups that a "current" run never counts as net reservoir
#: or net pay. Coal logs a bulk density of 1.2-1.8 g/cm3 and a high
#: resistivity (Asquith and Krygowski, 2004), so it passes the porosity and
#: saturation cutoffs although it holds no movable hydrocarbon. The igneous,
#: metamorphic and chert groups stay in: fractured basement and porous tuff
#: or spiculite can be reservoirs.
NON_RESERVOIR_LITHOLOGY_GROUPS = frozenset({"coal"})


def _halite_log_signature(well: "Well") -> np.ndarray:
    """Samples whose logs read like halite (rock salt).

    Halite logs a bulk density of 2.03 g/cm3, a neutron porosity of about
    zero, a photoelectric factor of 4.65 b/e and a slowness of 67 us/ft, has
    almost no radioactivity (Schlumberger, *Log Interpretation Charts*,
    1989) and almost no conductivity. A sample reads like halite with RHOB
    1.95-2.15 g/cm3, GR below 15 API, and a deep resistivity above
    500 ohm.m or, where none is logged, a caliper more than 0.5 in under the
    bit size (salt creeps into the hole). Where they are logged, NPHI must
    not exceed 0.05 and PEF must lie between 4.0 and 5.5 b/e, which rules
    out porous clean sandstone or limestone of the same bulk density.
    """
    def data(*names: str) -> np.ndarray:
        values, _, _ = get_curve_data_depthwise(well, list(names))
        if values is None:
            return np.full(well.n_samples, np.nan)
        return np.asarray(values, dtype=float)

    rhob, gr, rt = data("RHOB"), data("GR", "SGR", "HSGR"), data("RDEP", "RT")
    nphi, pef, cali = data("NPHI"), data("PEF"), data("CALI", "CAL")
    bit_size = _known_bit_size(well)
    with np.errstate(invalid="ignore"):
        halite = (rhob >= 1.95) & (rhob <= 2.15) & (gr < 15.0)
        halite &= (rt > 500.0) | (~np.isfinite(rt) & (cali < bit_size - 0.5))
        halite &= ~(nphi > 0.05) & ~((pef < 4.0) | (pef > 5.5))
    return halite


def _coal_log_signature(well: "Well") -> np.ndarray:
    """Samples whose logs read like coal.

    Coal logs a bulk density of 1.2-1.8 g/cm3 and a high apparent neutron
    porosity (Schlumberger, *Log Interpretation Charts*, 1989). A sample
    reads like coal with RHOB below 1.8 g/cm3 and, where NPHI is logged, NPHI
    above 0.35; no porous rock has so low a bulk density. Samples within half
    the density log's vertical resolution (0.23 m) of such a reading are
    included: the tool averages them with the seam.
    """
    def data(*names: str) -> np.ndarray:
        values, _, _ = get_curve_data_depthwise(well, list(names))
        if values is None:
            return np.full(well.n_samples, np.nan)
        return np.asarray(values, dtype=float)

    rhob, nphi = data("RHOB"), data("NPHI")
    with np.errstate(invalid="ignore"):
        coal = (rhob < 1.8) & ~(nphi <= 0.35)
    depth = np.asarray(well.depth, dtype=float) if well.depth is not None else None
    if depth is None or depth.shape != coal.shape:
        return coal
    # Within half the density log's vertical resolution of a seam the reading
    # mixes coal with the bed beside it (shoulder effect).
    half = 0.5 * DENSITY_VERTICAL_RESOLUTION_M / (0.3048 if str(well.depth_unit).lower() in {"ft", "feet"} else 1.0)
    return _within(coal, depth, half)


def _group_mask(lithology: Union[str, np.ndarray], groups: frozenset, n_samples: int) -> np.ndarray:
    """Samples whose per-depth lithology group is one of ``groups``."""
    if not isinstance(lithology, np.ndarray) or lithology.shape != (n_samples,):
        return np.zeros(n_samples, dtype=bool)
    return np.array([_coerce_lithology_group(group) in groups for group in lithology], dtype=bool)


#: Vertical resolution of a standard density log, about 18 in (0.46 m): a
#: layer thinner than that between two salt readings is not resolved.
DENSITY_VERTICAL_RESOLUTION_M = 0.46


def _fill_thin_gaps(flag: np.ndarray, depth: np.ndarray, width: float) -> np.ndarray:
    """``flag`` with runs of unflagged samples thinner than ``width`` (depth
    units) filled where flagged samples bound them on both sides."""
    filled = np.array(flag, dtype=bool, copy=True)
    index = np.flatnonzero(filled)
    if index.size < 2:
        return filled
    steps = np.abs(np.diff(depth[np.isfinite(depth)]))
    step = float(np.median(steps[steps > 0])) if np.any(steps > 0) else 0.0
    for start, stop in zip(index[:-1], index[1:]):
        # The gap is the unflagged samples' own thickness, one step each.
        if stop - start > 1 and (stop - start - 1) * step <= width + 1e-9:
            filled[start + 1:stop] = True
    return filled


def _within(flag: np.ndarray, depth: np.ndarray, distance: float) -> np.ndarray:
    """Samples within ``distance`` (depth units) of a flagged sample."""
    near = np.array(flag, dtype=bool, copy=True)
    flagged = depth[np.asarray(flag, dtype=bool) & np.isfinite(depth)]
    if flagged.size == 0:
        return near
    flagged = np.sort(flagged)
    position = np.searchsorted(flagged, depth)
    below = np.abs(depth - flagged[np.clip(position, 0, flagged.size - 1)])
    above = np.abs(depth - flagged[np.clip(position - 1, 0, flagged.size - 1)])
    with np.errstate(invalid="ignore"):
        near |= np.minimum(below, above) <= distance
    return near


def _log_halite(well: "Well", lithology: Union[str, np.ndarray]) -> np.ndarray:
    """Halite read from the logs (:func:`_halite_log_signature`) where no
    lithology group already calls the sample an evaporite.

    A layer thinner than the density log resolves (about 0.46 m) between two
    salt readings is taken as salt: such 1-3 sample "porous" readings inside
    salt are interbeds or log transitions, not reservoir.
    """
    evaporite = _group_mask(lithology, EVAPORITE_LITHOLOGY_GROUPS, well.n_samples)
    halite = _halite_log_signature(well)
    depth = np.asarray(well.depth, dtype=float) if well.depth is not None else None
    if depth is not None and depth.shape == halite.shape and not reproducing_benchmark():
        width = DENSITY_VERTICAL_RESOLUTION_M / (0.3048 if str(well.depth_unit).lower() in {"ft", "feet"} else 1.0)
        halite = _fill_thin_gaps(halite, depth, width)
    return halite & ~evaporite


def _non_reservoir_samples(well: "Well", lithology: Union[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Samples kept out of net reservoir and net pay under ``"current"``, by kind.

    * ``evaporite``: per-depth lithology groups in
      :data:`EVAPORITE_LITHOLOGY_GROUPS`. Salt, anhydrite and gypsum have no
      effective porosity or permeability; they seal reservoirs rather than
      form them (Warren, 2006). Porosity computed with a sandstone or
      limestone matrix still passes the cutoffs there: halite reads
      2.03 g/cm3 on the density log (Schlumberger chartbook), an apparent
      density porosity of 0.38 on a 2.65 g/cm3 matrix with 1.0 g/cm3 water,
      and its high resistivity gives a low Sw. The lithology domain alone
      never excludes samples. On a lithology code log the evaporites are the
      halite and anhydrite groups and the potash salts; code 26, "Gap",
      records no lithology and has no group.
    * ``coal``: per-depth lithology groups in
      :data:`NON_RESERVOIR_LITHOLOGY_GROUPS`.
    * ``halite_log_signature``: samples that no lithology group calls an
      evaporite but whose logs read like halite
      (:func:`_halite_log_signature`), such as salt below the end of a
      lithology log.
    * ``coal_log_signature``: samples no lithology group calls coal whose
      logs read like coal (:func:`_coal_log_signature`), such as a coal seam
      inside an interval a lithology log calls sandstone.
    * ``bad_hole``: samples with ``BAD_HOLE`` = 1 (washout or large hole).
      There the density pad loses contact with the borehole wall, so bulk
      density and neutron porosity, and the PHIE and Sw built on them, are
      unreliable (Asquith and Krygowski, 2004). The ``"current"`` bad-hole
      step assumes no bit size: where none is known, BAD_HOLE comes from
      ``|DRHO|`` > 0.15 g/cm3 alone.
    """
    n_samples = well.n_samples
    excluded: Dict[str, np.ndarray] = {}
    coal_group = _group_mask(lithology, NON_RESERVOIR_LITHOLOGY_GROUPS, n_samples)
    kinds = [
        ("evaporite", _group_mask(lithology, EVAPORITE_LITHOLOGY_GROUPS, n_samples)),
        ("coal", coal_group),
        ("halite_log_signature", _log_halite(well, lithology)),
    ]
    if not reproducing_benchmark():
        kinds.append(("coal_log_signature", _coal_log_signature(well) & ~coal_group))
    for kind, mask in kinds:
        if mask.any():
            excluded[kind] = mask
    bad_hole = well.get("BAD_HOLE")
    if bad_hole is not None:
        data = np.asarray(bad_hole.data, dtype=float)
        flagged = np.isfinite(data) & (data > 0.5)
        if flagged.any():
            excluded["bad_hole"] = flagged
    return excluded


def _cutoff_families(lithology: Union[str, np.ndarray], fallback: str) -> Union[str, np.ndarray]:
    """Lithology family that sets each sample's net-pay cutoffs.

    Limestone, chalk, dolomite and marl groups take the carbonate standards
    and every other group the clastic ones
    (:func:`~petrophys.evaluation.net_pay.cutoffs.cutoff_family`); samples
    without a lithology group take the family ``fallback``.
    """
    from petrophys.evaluation.net_pay.cutoffs import cutoff_family

    if not isinstance(lithology, np.ndarray):
        return cutoff_family(lithology) if lithology else fallback
    families: Dict[Any, str] = {}
    for group in lithology:
        if group not in families:
            name = _coerce_lithology_group(group)
            families[group] = cutoff_family(name) if name else fallback
    return np.array([families[group] for group in lithology], dtype=object)


def _net_flags_by_family(
    well: "Well",
    families: Union[str, np.ndarray],
    vcl: Optional[np.ndarray],
    phie: np.ndarray,
    sw: np.ndarray,
    explicit: Dict[str, Optional[float]],
    excluded_masks: Dict[str, np.ndarray],
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Net flags from each sample's family cutoffs, less the non-reservoir samples.

    A cutoff in ``explicit`` applies to every sample; one left as None takes
    the standard (:data:`~petrophys.evaluation.STANDARD_CUTOFFS`) of the
    sample's family in ``families``. The samples in ``excluded_masks``
    (:func:`_non_reservoir_samples`) are then removed from net reservoir and
    net pay; net rock keeps them.

    Returns
    -------
    tuple
        The :class:`~petrophys.evaluation.net_pay.NetPayModule`, its
        ``NetPayResult`` and a dict with the resolved ``cutoffs``, their
        ``cutoff_source``, ``cutoffs_by_family`` (with sample counts) and,
        per excluded kind, the samples flagged and the net-reservoir and
        net-pay samples removed (``excluded``).
    """
    from petrophys.evaluation.net_pay.cutoffs import standard_cutoffs
    from petrophys.evaluation.net_pay.net_pay_module import NetPayModule

    from collections.abc import Mapping

    family_array = np.broadcast_to(np.asarray(families, dtype=str), (well.n_samples,))
    # A cutoff given per family applies sample by sample; a family it leaves
    # out keeps its standard.
    applied = {
        key: (
            np.array([_lithology_value(value, family, family, standard_cutoffs(family)[key])
                      for family in family_array], dtype=float)
            if isinstance(value, Mapping) else value
        )
        for key, value in explicit.items()
    }
    module = NetPayModule(**applied, lithology=families)
    result = module.calculate(well, vcl=vcl, phie=phie, sw=sw)

    names, counts = np.unique(family_array, return_counts=True)
    by_family: Dict[str, Dict[str, Any]] = {}
    for index in np.argsort(-counts, kind="stable"):  # most samples first
        family = str(names[index])
        standard = standard_cutoffs(family)
        item: Dict[str, Any] = {
            key: _lithology_value(value, family, family, standard[key])
            for key, value in explicit.items()
        }
        item["samples"] = int(counts[index])
        by_family[str(names[index])] = item
    cutoffs = {
        key: _scalar_or_lithology_dependent([item[key] for item in by_family.values()])
        for key in explicit
    }
    source = {
        key: "config" if value is not None else "lithology_family_standard"
        for key, value in explicit.items()
    }

    excluded: Dict[str, Dict[str, int]] = {}
    usable = np.ones(well.n_samples, dtype=bool)
    for kind, mask in excluded_masks.items():
        removed = mask & usable
        excluded[kind] = {
            "samples": int(np.count_nonzero(mask)),
            "net_reservoir_samples_removed": int(np.count_nonzero(result.net_reservoir & removed)),
            "net_pay_samples_removed": int(np.count_nonzero(result.net_pay & removed)),
        }
        usable &= ~mask
    if excluded:
        result = module.calculate(well, vcl=vcl, phie=phie, sw=sw, quality_mask=usable)
    details = {
        "cutoffs": cutoffs,
        "cutoff_source": source,
        "cutoffs_by_family": by_family,
        "excluded": excluded,
    }
    return module, result, details


def _interval_text(well: "Well", mask: np.ndarray, limit: int = 3) -> str:
    """The first ``limit`` depth intervals where ``mask`` holds, as text."""
    labels = np.where(np.asarray(mask, dtype=bool), "in", "")
    intervals = [
        text.strip()
        for text in interval_strings(well.depth, labels, well.depth_unit or "").get("in", [])
    ]
    text = ", ".join(intervals[:limit])
    if len(intervals) > limit:
        text += f" and {len(intervals) - limit} more intervals"
    return text


def _representative_finite_value(values: Any) -> Optional[float]:
    """Return the modal finite value, falling back to median."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    rounded = np.round(finite, 4)
    unique, counts = np.unique(rounded, return_counts=True)
    if unique.size:
        return float(unique[np.argmax(counts)])
    return float(np.nanmedian(finite))


def _coerce_lithology_group(value: Any) -> Optional[str]:
    """Normalize lithology strings to workflow group names."""
    text = str(value or "").strip().lower()
    if not text or text in {"nan", "none", "-999.25"}:
        return None

    aliases = {
        "sand": "sandstone",
        "ss": "sandstone",
        "lime": "limestone",
        "ls": "limestone",
        "dolo": "dolomite",
        "salt": "halite",
    }
    return aliases.get(text, text)


def _group_array_from_numeric_codes(data: np.ndarray, mapping: Mapping[int, Optional[str]]) -> np.ndarray:
    groups = np.full(data.shape, None, dtype=object)
    finite = np.isfinite(data)
    if not np.any(finite):
        return groups

    rounded = np.rint(data[finite]).astype(int)
    grouped = [mapping.get(code) for code in rounded]
    groups[finite] = grouped
    return groups


def _finite_unique_numeric(values: Any) -> list[float]:
    """Return sorted finite unique numeric values rounded for stable summaries."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return []
    return sorted(float(value) for value in np.unique(np.round(finite, 6)))


def _family_cutoff_text(by_family: Dict[str, Dict[str, Any]], key: str) -> str:
    """A net-pay cutoff for the trace: ``0.4``, or ``0.4 clastic, 0.3 carbonate``."""
    values = {family: item[key] for family, item in by_family.items()}
    if len(set(values.values())) == 1:
        return f"{next(iter(values.values())):g}"
    return ", ".join(f"{value:g} {family}" for family, value in values.items())


def _net_pay_detail_decisions(
    source: Dict[str, str],
    by_family: Dict[str, Dict[str, Any]],
    excluded: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Trace lines: where the net-pay cutoffs came from, and the samples left out."""
    lines = []
    standard = [key for key, origin in source.items() if origin != "config"]
    if standard:
        lines.append(
            f"{', '.join(standard)} not set: lithology-family standards (STANDARD_CUTOFFS) for "
            + ", ".join(f"{item['samples']} {family} samples" for family, item in by_family.items())
        )
    reasons = {
        "evaporite": "Evaporite lithology (salt, anhydrite, gypsum) is not reservoir rock",
        "coal": "Coal is not reservoir rock (low density and high resistivity pass the cutoffs)",
        "halite_log_signature": (
            "Halite on the logs where no lithology group calls the rock an evaporite (RHOB "
            "1.95-2.15 g/cm3, GR < 15 API, RT > 500 ohm.m or, without RT, CALI < BS - 0.5 in; "
            "NPHI <= 0.05 and PEF 4.0-5.5 where logged)"
        ),
        "coal_log_signature": (
            "Coal on the logs where no lithology group calls the rock coal (RHOB < 1.8 g/cm3, "
            "NPHI > 0.35 where logged)"
        ),
        "bad_hole": "Bad hole (BAD_HOLE = 1): density, neutron, PHIE and SW are unreliable",
    }
    for kind, item in excluded.items():
        unit = item["depth_unit"]
        lines.append(
            f"{reasons.get(kind, kind)}: {item['samples']} samples left out of NET_RES and "
            f"NET_PAY, removing {item['net_reservoir_removed']:.1f} {unit} of net reservoir and "
            f"{item['net_pay_removed']:.1f} {unit} of net pay"
        )
    return lines


def _scalar_or_lithology_dependent(values: Any) -> Any:
    """Return a scalar when all values match, otherwise a compact label."""
    unique = _finite_unique_numeric(values)
    if len(unique) == 1:
        return unique[0]
    return "lithology_dependent"


def _is_shale_group(name: Any) -> bool:
    text = str(name or "").strip().lower()
    return "shale" in text


def _ordered_unique_values(values: List[Any]) -> List[Any]:
    seen = set()
    ordered: List[Any] = []
    for value in values:
        key = repr(value)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


def _trace_array_summary(value: np.ndarray) -> Dict[str, Any]:
    """Summarize array values without writing full curve payloads to logs."""
    data = np.asarray(value)
    payload: Dict[str, Any] = {
        "type": "ndarray",
        "shape": list(data.shape),
        "size": int(data.size),
    }
    if data.ndim == 0:
        payload["value"] = _json_safe_trace_value(data.item())
        return payload

    sample_count = min(5, int(data.size))
    payload["sample"] = [
        _json_safe_trace_value(item) for item in list(data.flat[:sample_count])
    ]

    if np.issubdtype(data.dtype, np.number):
        numeric = data.astype(float, copy=False)
        finite = numeric[np.isfinite(numeric)]
        payload.update({
            "finite_count": int(finite.size),
            "nan_count": int(np.isnan(numeric).sum()),
            "inf_count": int(np.isinf(numeric).sum()),
        })
        if finite.size:
            payload.update({
                "min": float(np.nanmin(finite)),
                "max": float(np.nanmax(finite)),
                "mean": float(np.nanmean(finite)),
                "p05": float(np.nanpercentile(finite, 5)),
                "p50": float(np.nanpercentile(finite, 50)),
                "p95": float(np.nanpercentile(finite, 95)),
            })

    return payload


def _json_safe_trace_value(value: Any) -> Any:
    """Convert trace metadata to JSON-safe, compact values."""
    if isinstance(value, np.ndarray):
        return _trace_array_summary(value)
    if isinstance(value, np.generic):
        return _json_safe_trace_value(value.item())
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_safe_trace_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_trace_value(item) for item in value]
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def _curve_summaries_for_trace(
    well: "Well",
    curve_names: Optional[List[str]],
) -> Dict[str, Any]:
    """Build compact per-curve summaries for workflow trace exports."""
    if not curve_names:
        return {}

    summaries: Dict[str, Any] = {}
    for name in curve_names:
        curve = well.get(name) if hasattr(well, "get") else None
        if curve is None:
            continue
        summaries[str(name).upper()] = {
            "unit": getattr(curve, "unit", ""),
            "description": getattr(getattr(curve, "metadata", None), "description", ""),
            "data": _trace_array_summary(np.asarray(curve.data)),
        }
    return summaries


@dataclass
class DynamicWorkflowConfig:
    """
    Settings for :func:`petrophys.interpret` and the dynamic workflow.

    Every setting is optional. The workflow inspects the available curves
    and runs each step it can; settings supply values that logs cannot
    (formation water resistivity, temperatures, mud properties) or override
    automatic choices. Values are validated when the config is created: an
    invalid choice raises ``ValueError`` listing the accepted values.

    Temperatures are in degrees Fahrenheit, resistivities in ohm.m, densities
    in g/cm3 and slownesses in us/ft, whatever the depth unit of the well.
    Fields left as ``None`` are read from the LAS header where available or
    derived from the data, and the trace records where each value came from.

    Examples
    --------
    >>> from petrophys import DynamicWorkflowConfig
    >>> config = DynamicWorkflowConfig(rw=0.05, bht_f=220, sw_model="indonesia")
    >>> config.matrix_type
    'sandstone'
    """

    # -- Input handling ------------------------------------------------------
    #: Standardize curve aliases and units before interpretation: vendor
    #: mnemonics get canonical aliases, units such as kg/m3 or percent
    #: porosity are converted and implausible values are screened, with a
    #: warning. None (default) means True under the ``"current"`` calculation
    #: profile and False under the "geolink" profile.
    standardize_input: Optional[bool] = None
    #: Rename vendor mnemonics to canonical names instead of adding aliases.
    rename_input_curves: bool = False
    #: Use a 0.05 ohm.m default Rw when no Rw source is found (the warnings
    #: say so). When False, saturation is not computed without an Rw source.
    allow_default_rw: bool = True
    # -- Temperature ---------------------------------------------------------
    #: Surface (mean annual) temperature, degF, at depth 0 of the linear
    #: temperature profile (offshore, the sea surface rather than the seabed).
    #: Set explicitly, it (like ``bht_f``) puts the profile ahead of an input
    #: TEMP log, which is kept as TEMP_LOG.
    surface_temp_f: float = 68.0
    #: Bottom-hole temperature, degF. Defaults to the LAS header BHT. It is
    #: taken as the static formation temperature at TD, with no correction for
    #: mud circulation (such as a Horner plot). Set, the temperature profile
    #: follows it rather than an input TEMP log (kept as TEMP_LOG).
    bht_f: Optional[float] = None
    #: Total depth for the BHT, in feet. Defaults to the LAS header TD, else
    #: the deepest sample. Under ``"current"`` a total depth above the deepest
    #: log sample (a depth in metres, for instance) is not used, with a
    #: warning: the BHT is placed at the header TD, else at the deepest sample.
    td_ft: Optional[float] = None
    #: Geothermal gradient, degF/100 ft (1 degC/100 m is 0.549 degF/100 ft),
    #: from ``surface_temp_f`` at depth 0. Defaults to the LAS header TGRAD,
    #: used only when there is no BHT: a BHT anchors the profile at TD. Set
    #: explicitly, it is used instead of the header BHT; set it or ``bht_f``,
    #: not both. On a well with a TEMP log (half or more of its samples
    #: finite) the log stays the temperature source and the gradient only
    #: fills the log's gaps, whereas ``bht_f`` and ``surface_temp_f`` set
    #: explicitly replace the log.
    gradient_f_per_100ft: Optional[float] = None

    # -- Mud properties ------------------------------------------------------
    #: ``"WBM"`` (water-based) or ``"OBM"`` (oil-based) mud. With oil-based mud
    #: no SXO or MHI is computed, and Rw is not estimated from the SP or Rxo/Rt:
    #: they need a conductive water-based filtrate (the "geolink" profile
    #: computes them).
    mud_type: str = "WBM"
    #: The mud contains clay or KCl, so the SP log is not used: Rw is not
    #: estimated from the SP, and no VCL_SP is computed.
    mud_has_clay_or_kcl: bool = False
    #: Mud-filtrate resistivity Rmf at ``rmf_meas_temp_f``, ohm.m. Rmf comes
    #: from the first of: this setting; ``rmf_salinity_ppm``,
    #: ``rmf_chlorides_mg_l`` or ``rm`` set by you; the LAS header RMF; then
    #: the header's filtrate salinity (FLSAL), chlorides (CLFL) or RM. The
    #: "geolink" profile takes the header RMF before ``rmf_salinity_ppm``,
    #: ``rmf_chlorides_mg_l`` and ``rm``.
    rmf_at_meas_temp: Optional[float] = None
    #: Temperature at which ``rmf_at_meas_temp`` was measured, degF. Unset
    #: (None), the LAS header RMFT is used, or else MFST (mud filtrate sample
    #: temperature), which the "geolink" profile does not read, and otherwise
    #: 75 degF. An explicit value, 75 included, is used as it stands.
    rmf_meas_temp_f: Optional[float] = None
    #: Mud-filtrate salinity, ppm NaCl equivalent (alternative to Rmf).
    rmf_salinity_ppm: Optional[float] = None
    #: Mud-filtrate chlorides (Cl-), mg/l, as on the mud report and the LAS
    #: header CLFL. Without Rmf or ``rmf_salinity_ppm``, Rmf comes from its NaCl
    #: equivalent: Cl x 58.44/35.45, divided by the brine density to give ppm
    #: by weight. The "geolink" profile reads CLFL as ``rmf_salinity_ppm``.
    rmf_chlorides_mg_l: Optional[float] = None
    #: Mud resistivity Rm at ``rm_meas_temp_f``, ohm.m (Rmf is derived from it).
    rm: Optional[float] = None
    #: Temperature at which ``rm`` was measured, degF. Unset (None), the LAS
    #: header RMT is used, or else MST (mud sample temperature), which the
    #: "geolink" profile does not read, and otherwise 75 degF. An explicit
    #: value, 75 included, is used as it stands.
    rm_meas_temp_f: Optional[float] = None
    #: ``"fresh"`` or ``"salt"`` mud, which sets the Rm-to-Rmf factor.
    mud_water: str = "fresh"

    # -- Formation water -----------------------------------------------------
    #: Formation water resistivity Rw, ohm.m, at ``rw_meas_temp_f``.
    rw: Optional[float] = None
    #: Temperature of ``rw``, degF. When omitted, ``rw`` is taken at the median
    #: formation temperature, as recorded in the trace; either way the RW
    #: curve follows the temperature profile from that point. Set without
    #: ``rw`` under ``"current"``, it is the measurement temperature of the LAS
    #: header RW, in place of the header RWT; with neither ``rw`` nor a header
    #: RW it is not used, with a warning.
    rw_meas_temp_f: Optional[float] = None
    #: Formation water salinity, ppm NaCl equivalent: another way to give the
    #: formation water. Without ``rw``, Rw follows this salinity across the
    #: temperature profile, ahead of a header RW. With ``rw`` set, ``rw`` gives
    #: the RW curve and this salinity is not used: where saturation chooses its
    #: equation by salinity (Simandoux at 20000 ppm or more), it takes the
    #: salinity of the Rw the run uses. Above 350,000 ppm a warning notes that
    #: a NaCl brine holds less; above 400,000 ppm the value is refused.
    salinity_ppm: Optional[float] = None

    # -- Matrix properties ---------------------------------------------------
    #: ``"auto"`` (resolved from lithology evidence) or a mineral name such as
    #: ``"sandstone"``, ``"limestone"`` or ``"dolomite"``; see
    #: :meth:`petrophys.model.constants.MatrixProperties.names`. Default:
    #: ``"sandstone"`` in the clastic domain, otherwise ``"auto"`` within
    #: ``lithology_domain``. It gives the matrix endpoints of the porosity
    #: logs where no lithology group is known, and the clean line of VCL_ND.
    #: On a well with a lithology log or lithology intervals the group of each
    #: depth sets the matrix there, so ``matrix_type`` changes the porosity of
    #: few samples; ``rho_ma``, ``nphi_ma`` and ``dt_ma`` hold at every depth.
    matrix_type: Optional[str] = None
    #: Matrix density, g/cm3. Unset: the value for ``matrix_type``, or per
    #: depth that of each lithology-log group. Set, it holds at every depth.
    rho_ma: Optional[float] = None
    #: Matrix neutron porosity, v/v. Unset: the value for ``matrix_type``, or
    #: per depth that of each lithology-log group. Set, it holds at every depth.
    nphi_ma: Optional[float] = None
    #: Pore-fluid density, g/cm3, used by the porosity, hydrocarbon-correction
    #: and mineral steps alike. Default (None): 1.0, or the COSTA
    #: calibration's 1.1 brine when ``carbonate_calibration="costa"`` is set;
    #: a value set here is used everywhere, the COSTA solve included.
    rho_fl: Optional[float] = None
    #: Clay density, g/cm3, used by the carbonate mineral solve and the
    #: neutron-density hydrocarbon correction. Default (None): 2.45, or the
    #: COSTA calibration's 2.52 illite when ``carbonate_calibration="costa"``
    #: is set. The clastic porosity equations do not read it, and the clastic
    #: mineral solve keeps its clay endpoint (density, neutron, PEF and sonic
    #: of one clay point), which a density alone would not move consistently.
    rho_cl: Optional[float] = None
    #: Matrix sonic slowness, us/ft. Unset: the value for ``matrix_type``, or
    #: per depth that of each lithology-log group. Set, it holds at every depth.
    dt_ma: Optional[float] = None
    #: Pore-fluid sonic slowness, us/ft.
    dt_fl: float = 189.0

    # -- Lithology -----------------------------------------------------------
    #: Broad interpretation domain: ``"clastic"`` (the default when unset),
    #: ``"carbonate"``, ``"evaporite"`` or ``"auto"``. This selects the
    #: mineral-solving family and cutoff defaults; it is distinct from
    #: per-depth lithology. ``"evaporite"`` runs the clastic mineral solve
    #: (anhydrite among its minerals) with an automatic matrix that no family
    #: limits, and clastic cutoffs; evaporite samples are left out of net pay.
    #: A matrix left unset follows the domain, also when the domain is changed
    #: on a copy of the config. A rock name (``"dolomite"``, ``"chalk"``,
    #: ``"anhydrite"`` ...) sets its family as the domain and, when
    #: ``matrix_type`` is unset, the rock as the matrix where it has matrix
    #: properties.
    lithology_domain: Optional[str] = None
    #: Alias for ``lithology_domain``, the unambiguous name. It fills
    #: ``lithology_domain`` only when that is unset; when both are set,
    #: ``lithology_domain`` is used (:func:`petrophys.interpret` refuses
    #: both together).
    lithology: Optional[str] = None
    #: Lithology intervals CSV (``top,bottom,lithology``); the file must exist.
    lithology_csv: Optional[str] = None
    #: Lithology intervals as a table (DataFrame or list of rows) or the path
    #: of an existing CSV/Excel file.
    lithology_intervals: Optional[Any] = None
    #: Lithology source. ``"auto"`` (default) takes the lithology intervals
    #: when there are any, else the lithology code curve, else ``"computed"``.
    #: ``"external"`` requires intervals and ``"vendor"`` a lithology code
    #: curve. ``"computed"`` and ``"config"`` both ignore the lithology log
    #: and the intervals: every depth takes ``lithology_domain`` and
    #: ``matrix_type``. With ``"computed"`` an ``"auto"`` domain or matrix is
    #: resolved from the well's core and logs; ``"config"`` refuses ``"auto"``
    #: for either.
    lithology_source: str = "auto"
    #: Mnemonic of the per-sample lithology code curve, e.g. ``"FACIES"``. The
    #: same curve is the vendor lithology that ``lithology_source`` selects,
    #: resolves an ``"auto"`` matrix and gives the per-depth lithology groups,
    #: and its codes are decoded with the built-in lithology code table
    #: (Kingdom nomenclature). None (default) reads ``LITHOLOGY_GEOLINK``. Under
    #: ``"current"``, with ``lithology_source`` ``"auto"`` or ``"vendor"``, a
    #: well without it is read from the first lithology-code mnemonic
    #: (``LITHOLOGY``, ``LITHOFACIES``, ``FACIES``, ``LITH_CODE``,
    #: ``ROCK_TYPE``, ...) whose curve holds lithology codes: 20 or more finite
    #: whole numbers from 0 to 99, at most 50 distinct, half or more of them
    #: codes of that table. A warning names a curve other than ``LITHOLOGY_GEOLINK``,
    #: and a curve set here that has no finite samples. Under ``"current"`` a
    #: curve set here is held to the same checks (with ``lithology_codes``,
    #: against its codes and without the 0-99 range), and one that fails them,
    #: such as a GR log, raises ``ValueError``. Unless this is set, the
    #: "geolink" profile takes its per-depth lithology groups from
    #: ``LITHOLOGY_GEOLINK`` alone (see ``calculation_profile``).
    lithology_curve: Optional[str] = None
    #: The codebook of the lithology code curve, when its codes are not those of
    #: the built-in lithology code table: a mapping of code to rock name, or a table
    #: (path or DataFrame) that :func:`petrophys.io.read_lithology_codes` reads.
    #: Each rock name is read as a lithology group by
    #: :func:`petrophys.interpretation.lithology.labels.lithology_group_for_label`
    #: ("Sst", "shaly sandstone", "LST", "Dolostone", ...); codes whose name
    #: is no recognized rock are left without a group, with a warning. Not
    #: used by the benchmark profiles.
    lithology_codes: Optional[Any] = None
    #: Mineral solver for the clastic mineral step. ``1`` (default) is
    #: :class:`~petrophys.interpretation.lithology.MultiMineralSolverV1`:
    #: VCL and PHIE are fixed inputs, each equation row is
    #: weighted by its nominal tool uncertainty (unweighted under the "geolink"
    #: profile), and samples that fail the 0.05 closure gate are left
    #: unsolved. ``2`` is
    #: :class:`~petrophys.interpretation.lithology.MultiMineralSolver`, which
    #: solves porosity jointly with the minerals. The carbonate domain uses its
    #: own calcite-dolomite solve (``carbonate_calibration``) either way.
    mineral_solver_version: int = 1
    #: Calibration of the carbonate-domain mineral solve, which takes calcite
    #: and dolomite from bulk density and closure. ``"generic"``: clay volume
    #: from VCL with density ``rho_cl``, effective porosity, pore fluid
    #: ``rho_fl``, calcite 2.71 and dolomite 2.87 g/cm3. ``"costa"``: the COSTA
    #: field calibration, illite from GR (``-0.0424035 + 0.00562239 * GR``),
    #: total porosity, a 1.1 g/cm3 brine (unless ``rho_fl`` is set) and a
    #: 2.847 g/cm3 dolomite. Default:
    #: ``"generic"``; ``"costa"`` under the "geolink" profile. Both read the
    #: porosity step's PHIE or PHIT, computed with the ``matrix_type`` density,
    #: so the calcite-dolomite split largely follows ``matrix_type`` where that
    #: porosity is density-derived.
    carbonate_calibration: Optional[str] = None
    #: Endpoints of the clastic mineral solve that replace the built-in table,
    #: by mineral (``quartz``, ``calcite``, ``dolomite``, ``anhydrite``,
    #: ``clay``) or pore fluid (``water``), each a mapping of any of
    #: ``rhob`` (g/cm3), ``nphi`` (v/v), ``pef`` (b/e), ``u`` (b/cm3) and
    #: ``dt`` (us/ft): e.g. ``{"quartz": {"rhob": 2.65, "nphi": -0.04},
    #: "water": {"nphi": 1.0}}``. The solve's water takes ``rho_fl`` and
    #: ``dt_fl`` unless set here; porosity always uses ``rho_fl`` and
    #: ``dt_fl``. Not available under the "geolink" profile.
    mineral_endpoints: Optional[Dict[str, Dict[str, float]]] = None
    #: Raise on the first failed step instead of recording it and continuing.
    raise_on_error: bool = False

    # -- Archie parameters (None: lithology-dependent defaults) ---------------
    #: Tortuosity factor a. One value for every sample, or one per lithology,
    #: e.g. ``{"sandstone": 0.81, "carbonate": 1.0}``: a lithology of the
    #: Archie table, a family (``clastic``, ``carbonate``), or ``"*"`` for
    #: every other lithology. A lithology not given takes the LAS header's
    #: value (``A``, or ``A_Z1`` inside its zone), else its default.
    a: Optional[Any] = None
    #: Cementation exponent m; one value, or one per lithology as for ``a``.
    m: Optional[Any] = None
    #: Saturation exponent n; one value, or one per lithology as for ``a``.
    n: Optional[Any] = None

    # -- Waxman-Smits parameters ---------------------------------------------
    #: Cation exchange capacity, meq/100 g (0 to 150; pure smectite is 80-150).
    #: Waxman-Smits runs only with ``cec`` and ``b_coeff``, from lab data or the
    #: LAS header, where a CEC in MEQ/G is converted; neither is estimated. A
    #: value under 1 gets a warning, as meq/g values look like that. The
    #: "geolink" profile passes ``cec`` to the solver as meq/g.
    cec: Optional[float] = None
    #: Equivalent counter-ion conductance B, (S/m)/(meq/cm3): at 25 degC
    #: 4.6 * (1 - 0.6 * exp(-0.77 / Rw)) (Waxman and Smits, 1968), and higher at
    #: reservoir temperature.
    b_coeff: Optional[float] = None
    #: Waxman-Smits cementation exponent m*.
    m_star: float = 2.0

    # -- Dual-water parameters -----------------------------------------------
    #: Bound-water resistivity, ohm.m, from lab data or the LAS header. Dual
    #: Water runs only with it: ``sw_model="auto"`` gives shaly carbonate
    #: samples without it the clastic routing (Waxman-Smits with ``cec`` and
    #: ``b_coeff``, Simandoux at a ``salinity_ppm`` of 20000 or more, otherwise
    #: Indonesia), and ``sw_model="dual_water"`` stops with an error. Rwb is not
    #: estimated (the "geolink" profile takes 0.3 Rw).
    rwb: Optional[float] = None
    #: Clay (shale) porosity, v/v. A value set is used as it stands, 0.15
    #: included. Unset (None), it is derived from the
    #: well's shale samples where they support it, and 0.15 is the last
    #: resort. Under ``"current"`` a LAS
    #: header shale point, PNSH (the neutron log's reading in shale) with RBSH
    #: (shale density), comes before the shale samples; PNSH alone is not a
    #: clay porosity and is not used. The derived value is one number for the
    #: whole well: it does not follow the compaction of shale with depth, and
    #: samples without a neutron reading get the neutron-density value.
    phi_cl: Optional[float] = None

    # -- Hydrocarbon correction ----------------------------------------------
    #: Apply the neutron-density hydrocarbon correction (opt-in). It writes the
    #: corrected curves NPHI_HC and RHOB_HC, and the porosity step (so SW and
    #: net pay) reads them instead of NPHI and RHOB. It runs before the
    #: saturation step, so it needs an input flushed-zone saturation (SXO);
    #: without one the step is skipped with a warning. The "geolink" profile
    #: writes them for inspection only.
    apply_hc_correction: bool = False
    #: Hydrocarbon type, ``"oil"`` or ``"gas"``: the fluid of the
    #: neutron-density hydrocarbon correction and, under ``"current"``, the PHIT
    #: rule in neutron-density crossover (PHID > PHIN + 0.05): the
    #: root-mean-square of PHID and PHIN (Gaymard and Poupon, 1968) for gas,
    #: density porosity for oil.
    hc_type: str = "oil"
    #: Hydrocarbon density in the flushed zone, g/cm3, for the hydrocarbon
    #: correction and the core matrix-density check of auto lithology. Unset:
    #: 0.8 for oil and 0.2 for gas in the correction (0.25 for gas in the
    #: matrix check).
    rho_hc: Optional[float] = None
    #: Hydrogen index of the hydrocarbon for the neutron hydrocarbon
    #: correction. Unset: 1.0 for oil, 0.3 for gas.
    hc_hydrogen_index: Optional[float] = None

    # -- Net-pay cutoffs (None: lithology-family standards) --------------------
    #: Maximum clay volume for net rock, v/v. Unset, each sample takes the
    #: standard of its lithology family (``STANDARD_CUTOFFS``: 0.40 clastic,
    #: 0.30 carbonate); the "geolink" profile uses 0.3. Each cutoff takes one
    #: value, or one per family, e.g. ``{"clastic": 0.35, "carbonate": 0.2}``
    #: (a lithology name stands for its family; ``"*"`` for the others); a
    #: family left out takes the LAS header's value, else its standard. Unless
    #: ``sw_routing_vcl_cutoff`` is set, saturation switches to a shaly-sand
    #: model above this clay volume, or above 0.3 when it is unset. The same
    #: clay volume bounds the reservoir flag that picks the SWIRR_APP samples
    #: (built when ``swirr`` is unset and the well has no SWIRR curve or NMR
    #: Swirr), so ``vcl_cutoff`` also changes SW, PERM, RQI and FZI, not only
    #: net pay.
    vcl_cutoff: Optional[Any] = None
    #: Maximum water saturation for net pay, v/v. Unset: 0.60 clastic, 0.70
    #: carbonate; 0.7 under the "geolink" profile.
    sw_cutoff: Optional[Any] = None
    #: Minimum effective porosity for net reservoir, v/v. Unset: 0.08 clastic,
    #: 0.045 carbonate. It is also the least porosity of the reservoir flag
    #: that picks the SWIRR_APP samples (unset: 0.08 in every lithology under
    #: ``"current"``), built when ``swirr`` is unset and the well has no SWIRR
    #: curve or NMR Swirr, so it changes PERM, RQI and FZI, not only net pay.
    phie_cutoff: Optional[Any] = None

    # -- Gamma-ray clay volume ----------------------------------------------
    #: Clean-rock gamma-ray endpoint, API, used as given. None picks P5 of the
    #: log, lowered to 20 API when higher (a guard against a nearly flat GR,
    #: reported in the trace as ``gr_min_used``, with a warning when the pick
    #: is above 30 API).
    gr_min: Optional[float] = None
    #: Shale gamma-ray endpoint, API, used as given. None picks P95 of the
    #: log, raised to 100 API when lower.
    gr_max: Optional[float] = None
    #: Transform: ``"linear"``, ``"larionov_tertiary"``, ``"larionov_older"``,
    #: ``"stieber"`` (stored as ``"steiber"``) or ``"clavier"``.
    gr_method: str = "linear"

    # -- SP clay volume (None: picked from the log) --------------------------
    #: Shale SP baseline, mV: the line of least SP deflection. None picks P95
    #: of the SP log. With a normal SP, where sands deflect negative, it is the
    #: higher of the two baselines.
    sp_min: Optional[float] = None
    #: Clean-sand SP line (static SP), mV: the line of greatest deflection.
    #: None picks P5 of the SP log. The two lines are also those of the
    #: SP-derived Rw. VCL_SP joins the minimum-of-indicators VCL
    #: only when both ``sp_min`` and ``sp_max`` are set. Both are constant
    #: lines applied over the whole log, so set them only for an SP without
    #: baseline drift (e.g. a drift-corrected SP): lines picked in one interval
    #: of a drifting SP misread every other interval, and the minimum takes
    #: that reading. Without them VCL_SP is reported and only fills depths with
    #: no GR or neutron-density value. A warning flags user baselines whose
    #: order contradicts the SP polarity; the automatic P5/P95 picks assume a
    #: normal SP and are not checked. The "geolink" profile always puts VCL_SP
    #: in the minimum.
    sp_max: Optional[float] = None

    # -- Neutron-density clay volume -----------------------------------------
    #: Clay point of VCL_ND, ``(RHOB g/cm3, NPHI v/v)``. None (default) picks
    #: it under ``"current"`` in the well's good-hole shales: the median of 20
    #: or more shale-lithology samples, else of the VCL_GR > 0.9 samples, else
    #: (2.62, 0.38). The pick is one median over the whole log, so where
    #: shallow, undercompacted shale dominates it the point is lighter than
    #: the shales of a deeper reservoir; set the point from a crossplot of the
    #: interval you evaluate. A point less than 0.10 v/v of neutron above the
    #: clean line leaves VCL_ND out of VCL, with a warning. VCL_ND is measured
    #: from the clean line of the run's matrix at every depth, carbonate beds
    #: of a clastic well included. Zoned runs give every zone the whole-well
    #: point. The "geolink" profile uses (2.62, 0.38) and ignores this setting.
    nd_clay: Optional[Tuple[float, float]] = None

    # -- Bad hole --------------------------------------------------------------
    #: Thresholds of the BAD_HOLE flag, any of ``cali`` (washout where
    #: CALI - BS exceeds it, in; default 1.5), ``drho`` (washout where absolute DRHO
    #: exceeds it, g/cm3; default 0.15) and ``large_hole`` (bit sizes above
    #: it, in; default 17.5), e.g. ``{"cali": 2.0}``. Not available under the
    #: "geolink" profile.
    bad_hole: Optional[Dict[str, float]] = None

    # -- Permeability --------------------------------------------------------
    #: Permeability method. ``"auto"`` (default) writes PERM from the empirical
    #: equation ``perm_fallback_method`` at every depth and, where the well's
    #: core (or peer-well core) gives a core model, the core-calibrated
    #: PERM_CC beside it. Naming an equation writes PERM from it and no
    #: PERM_CC: ``"timur"``, ``"coates"`` (Coates-Denoo) and ``"wyllie_rose"``
    #: from porosity and Swirr (``perm_coefficients`` gives the equations),
    #: ``"lith_vcl"`` from total porosity, clay volume and lithology, without
    #: Swirr.
    perm_method: str = "auto"
    #: Empirical equation of PERM under ``perm_method="auto"``: ``"coates"``
    #: (default), ``"timur"``, ``"wyllie_rose"`` or ``"lith_vcl"``. It sets
    #: PERM at every depth, also where a core model is fitted: the
    #: core-calibrated permeability is PERM_CC.
    perm_fallback_method: str = "coates"
    #: Coefficients of the empirical permeability equation, any of ``a`` (the
    #: coefficient), ``b`` (the porosity exponent) and ``c`` (the Swirr
    #: exponent): Timur k = a (100 PHIE)^b / (100 Swirr)^c (defaults 0.136,
    #: 4.4, 2), Coates-Denoo k = a PHIE^b ((1 - Swirr) / Swirr)^c (10000, 4,
    #: 2), Wyllie-Rose k = a PHIE^b / Swirr^c (62500, 6, 2). They apply to
    #: the equation the run uses (``perm_method``, or ``perm_fallback_method``
    #: where no core model is fitted); ``lith_vcl`` takes none. Not available
    #: under the "geolink" profile.
    perm_coefficients: Optional[Dict[str, float]] = None
    #: Irreducible water saturation, v/v, for the permeability equations. One
    #: value for every sample, or one per lithology as for ``a``, e.g.
    #: ``{"sandstone": 0.12, "limestone": 0.20}`` (SCAL or NMR values by rock
    #: type); a lithology not given takes the automatic estimate. Defaults to
    #: a SWIRR curve or NMR data, else the Buckles bulk-volume-water estimate.
    #: Under ``"current"`` it must be above 0.
    swirr: Optional[Any] = None
    #: Clay porosity behind the PHIE that picks the SWIRR_APP samples, i.e. the
    #: reservoir flag built before permeability when the well has no RESFLAG or
    #: NET_RES curve. ``"workflow"`` uses the reported PHIE. ``"all_shale_phit"``
    #: recomputes PHIE for that flag only, as ``clip(PHIT - VCL * phi_cl, 0, PHIT)``
    #: with ``phi_cl`` the mean PHIT of all shale-lithology samples, the rule
    #: used to select SWIRR_APP samples for the GeoLink benchmark context.
    #: This flag calculation does not alter the reported PHIE or SW.
    swirr_flag_phi_cl: str = "workflow"
    # -- Calculation profile -------------------------------------------------
    #: Interpretation configuration. ``"current"`` is the default;
    #: ``"geolink"`` supplies the GeoLink evaluation configuration, including
    #: dataset-specific calculation and classification rules. Explicit settings
    #: take precedence over profile defaults. Resolved parameters and selected
    #: methods are recorded in the workflow trace.
    #:
    #: For GeoLink, COSTA or Danish benchmark context, use ``context_profile``
    #: in ``petrophys.export_context`` to select the collection's preparation,
    #: calibration, interpretation and grouping rules. See the benchmark context
    #: guide for required inputs and commands.
    #:
    #: Library calls read this setting from the supplied configuration. The MCP
    #: server can also select it through ``--calculation-profile`` or
    #: ``PETROPHYS_CALCULATION_PROFILE``.
    calculation_profile: str = "current"
    #: Reuse a RESFLAG or NET_RES curve already on the well as the reservoir
    #: flag that picks the SWIRR_APP samples before permeability. False
    #: (default) recomputes the flag from this run's cutoffs and warns that the
    #: input flag was set aside, so an interpreted LAS fed back in cannot carry
    #: another run's cutoffs into Swirr and permeability. True keeps a supplied
    #: flag, e.g. a vendor reservoir flag you mean to honour.
    reuse_input_resflag: bool = False
    #: Offset wells whose core data calibrate permeability: a list of
    #: interpreted ``Well`` objects (``petrophys.interpret(peer).well``, which
    #: has the PHIE or PHIT its core is fitted to). A peer is used only when
    #: ``use_peer_core=True``, it has the same ``header.field`` as this well,
    #: both wells have formation tops with a formation name in common, and it
    #: is interpreted. Under ``"current"`` a warning names the condition a
    #: peer fails.
    peer_wells: Optional[List[Any]] = None
    #: Formation intervals (``FormationTops``) for per-formation core
    #: calibration. Defaults to formations attached by ``petrophys.read``.
    formation_tops: Optional[Any] = None
    #: Calibrate permeability with core data from ``peer_wells``, under the
    #: conditions given there; without ``peer_wells`` it has no effect.
    use_peer_core: bool = False

    # -- Water saturation ----------------------------------------------------
    #: ``"auto"`` (routed per sample by clay content) or one of
    #: ``"archie"``, ``"simandoux"`` (Sw in the clay term, Bardon-Pied),
    #: ``"modified_simandoux"`` (sand term over 1 - Vcl), ``"indonesia"``,
    #: ``"dual_water"``, ``"waxman_smits"``, ``"north_sea"``,
    #: ``"nigeria_schlumberger"``.
    sw_model: str = "auto"
    #: Clay volume, v/v, from which ``sw_model="auto"`` uses a shaly-sand
    #: equation rather than Archie. None (default) uses ``vcl_cutoff``, or 0.3
    #: when that is unset. Unset net-pay cutoffs resolve per lithology family
    #: under ``"current"`` (clastic VCL < 0.40), so net rock with 0.3-0.4 clay
    #: gets the shaly-sand equations; under the "geolink" profile (VCL < 0.3)
    #: every net-rock sample is evaluated with Archie. A lower value, e.g. 0.15,
    #: also applies the shaly-sand equations to moderately shaly reservoir rock.
    #: Under ``"current"`` the SWIRR_APP samples are picked below the same clay
    #: volume. One value, or one per lithology family as for ``vcl_cutoff``,
    #: e.g. ``{"clastic": 0.2, "carbonate": 0.3}``.
    sw_routing_vcl_cutoff: Optional[Any] = None
    #: Shale resistivity, ohm.m. Defaults to the computed clay resistivity.
    rtsh: Optional[float] = None
    #: Shale exponent of the North Sea model.
    vsh_exp: float = 1.6
    #: Shale exponent of the Nigeria-Schlumberger model.
    shale_exp: float = 1.6

    # -- Zones ---------------------------------------------------------------
    #: Settings that differ by depth: a list of zones, each a mapping with
    #: ``top``, ``bottom`` and the settings for that interval (any setting
    #: but the whole-run ones, such as ``calculation_profile`` or
    #: ``skip_steps``), e.g. ``[{"top": 176, "bottom": 1908, "rw": 0.5,
    #: "rw_meas_temp_f": 60}]``; or a table or CSV/Excel file with those
    #: columns (an empty cell sets nothing). Depths are in the well's depth
    #: unit unless a ``depth_unit`` is given; zones may not overlap. Each zone
    #: is interpreted with its settings over the rest and merged into the
    #: well. Inside a zone its values win over everything; outside, the
    #: whole-well settings apply. LAS header zones (``RW_Z1`` with
    #: ``TOP_Z1``/``BOT_Z1``) rank below the settings you pass. A zone may
    #: also give only a ``name`` (or ``zone``) and its depths: it then
    #: changes nothing and names the interval for
    #: :meth:`DynamicWorkflowResult.zone_summary` and context exports. A
    #: ``well`` column lets one table hold the zones of several wells; a well
    #: takes the rows with its name and the rows without one. Not available
    #: under the "geolink" profile.
    zones: Optional[Any] = None

    # -- Step selection ------------------------------------------------------
    #: Workflow steps to skip, e.g. ``["permeability"]``; see
    #: :class:`WorkflowStep` for the names. A skipped step writes none of its
    #: curves, and the steps that read them are not skipped with it: one that
    #: finds its inputs missing is skipped with the reason, and one that runs
    #: without them fails. Skipping ``"rw"`` leaves saturation without an Rw:
    #: the saturation step fails and the run's status is ``"failed"``, so skip
    #: the steps that depend on a skipped step as well.
    skip_steps: List[str] = field(default_factory=list)
    #: The settings the caller passed, which rank above LAS header values even
    #: when they equal the default (kept by ``dataclasses.replace``). Set by
    #: the library; not a setting.
    _explicit_settings: Optional[frozenset] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.lithology is not None:
            # The ``lithology`` alias fills an unset domain and is not mirrored
            # back, so dataclasses.replace(config, lithology_domain=...) works.
            alias = str(self.lithology).strip()
            if self.lithology_domain is None:
                self.lithology_domain = alias
            # Otherwise lithology_domain wins: dataclasses.replace() passes the
            # old mirrored alias along with a new domain.
        if self.lithology_domain is None:
            self.lithology_domain = _DEFAULT_LITHOLOGY_DOMAIN
        _validate_workflow_config(self)
        self.lithology = self.lithology_domain


def _explicit(config: Any) -> frozenset:
    """The settings the caller passed (``_explicit_settings``), empty when unknown."""
    return getattr(config, "_explicit_settings", None) or frozenset()


def _record_explicit_settings(cls):
    """Record on each config the settings its caller passed (``_explicit_settings``).

    A header value fills only a setting left at its default; one passed
    explicitly, even with the default's value, is kept. ``dataclasses.replace``
    passes the record on (it is a field), and ``copy.deepcopy`` keeps it.
    """
    import inspect

    init = cls.__init__
    names = [item.name for item in dataclasses.fields(cls) if item.init and not item.name.startswith("_")]

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        if kwargs.get("_explicit_settings") is None:
            passed = set(names[:len(args)]) | (set(kwargs) - {"_explicit_settings"})
            if len(passed) < len(names):
                self._explicit_settings = frozenset(passed)

    __init__.__doc__ = init.__doc__
    __init__.__signature__ = inspect.signature(init)
    cls.__init__ = __init__
    return cls


DynamicWorkflowConfig = _record_explicit_settings(DynamicWorkflowConfig)


#: The settings that stand in for each header zone value the workflow reads
#: step by step (``RW_Z1`` ...): one of them set makes the zone value unused.
_HEADER_ZONE_SETTINGS = {
    "RW": ("rw", "salinity_ppm"),
    "RMF": ("rmf_at_meas_temp", "rmf_salinity_ppm", "rmf_chlorides_mg_l"),
    "RM": ("rm",),
    "PNSH": ("phi_cl",),
    "RBSH": ("phi_cl",),
}
#: Settings that also take one value per lithology, ``{lithology: value}``.
_ARCHIE_FIELDS = ("a", "m", "n")
_CUTOFF_FIELDS = ("vcl_cutoff", "phie_cutoff", "sw_cutoff")
_FAMILIES = ("clastic", "carbonate")


def _per_lithology_values(config: "DynamicWorkflowConfig", errors: List[str]) -> None:
    """Check and normalize the ``{lithology: value}`` forms of Archie and cutoff settings.

    Archie keys are lithologies of the Archie table (``sandstone``,
    ``limestone``, ``dolomite``, ``shale``, aliases such as ``sand``) or a
    family, ``clastic`` or ``carbonate``, for every lithology in it. Cutoff
    keys are families, or lithologies standing for their family. ``"*"``
    stands for every lithology not listed. Each value is checked as the single
    value would be.
    """
    from collections.abc import Mapping

    from petrophys.evaluation.net_pay.cutoffs import cutoff_family
    from petrophys.model.constants import ArchieParameterDefaults

    known = set(ArchieParameterDefaults.names())
    for name in (*_ARCHIE_FIELDS, "swirr", *_CUTOFF_FIELDS, "sw_routing_vcl_cutoff"):
        value = getattr(config, name)
        if not isinstance(value, Mapping):
            continue
        if config.calculation_profile in _BENCHMARK_PROFILES:
            if name != "sw_routing_vcl_cutoff":  # which the "geolink" profile takes as given
                errors.append(f'{name} per lithology is not available under the "geolink" calculation profile')
            continue
        normalized: Dict[str, float] = {}
        sources: Dict[str, str] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key).strip().lower().replace(" ", "_")
            if key == "*":
                target = "*"  # every lithology not listed
            elif name in _ARCHIE_FIELDS or name == "swirr":
                if key in _FAMILIES:
                    target = key
                elif key in known:
                    target = ArchieParameterDefaults.get(key).lithology
                else:
                    errors.append(
                        f"{name}: unknown lithology {raw_key!r}; use clastic, carbonate or one of "
                        f"{', '.join(sorted(n for n in known if '_' not in n))}"
                    )
                    continue
            else:
                target = key if key in _FAMILIES else cutoff_family(key)
                if key not in _FAMILIES and key not in known:
                    errors.append(f"{name}: unknown lithology {raw_key!r}; use clastic or carbonate")
                    continue
            try:
                single = DynamicWorkflowConfig(**{name: raw_value})
            except (TypeError, ValueError) as exc:
                detail = str(exc).replace("Invalid DynamicWorkflowConfig:\n  - ", "")
                errors.append(f"{name}[{raw_key!r}]: {detail}")
                continue
            number = float(getattr(single, name))
            if target in normalized and normalized[target] != number:
                errors.append(
                    f"{name}: {sources[target]!r} and {raw_key!r} both give the {target} value "
                    f"({normalized[target]:g} and {number:g})"
                    + (": cutoffs are per family, clastic or carbonate" if name in _CUTOFF_FIELDS else "")
                )
                continue
            normalized[target], sources[target] = number, str(raw_key)
        if not normalized and not errors:
            errors.append(f"{name}: give at least one lithology and value")
        setattr(config, name, normalized)


def _lithology_value(value: Any, lithology: str, family: str, default: float) -> float:
    """A setting for one sample: the single value, else the sample's lithology's, family's or the default."""
    from collections.abc import Mapping

    if value is None:
        return float(default)
    if isinstance(value, Mapping):
        for key in (lithology, family, "*"):
            if key in value:
                return float(value[key])
        return float(default)
    return float(value)


#: Settings that hold for the whole run and cannot differ by zone.
_WHOLE_RUN_SETTINGS = frozenset({
    "zones", "calculation_profile", "raise_on_error", "standardize_input", "rename_input_curves",
    "reuse_input_resflag", "formation_tops", "peer_wells", "use_peer_core", "lithology",
    "lithology_intervals", "lithology_csv", "lithology_source", "lithology_codes", "lithology_curve",
    "skip_steps",
})
_ZONE_TOP_NAMES = ("top", "top_depth", "ztop")
_ZONE_BOTTOM_NAMES = ("bottom", "bottom_depth", "base", "bot")


def _zone_rows(zones: Any) -> List[Dict[str, Any]]:
    """The rows of a ``zones`` value: a list of mappings, a table, or a CSV/Excel path."""
    from collections.abc import Mapping

    import pandas as pd

    if isinstance(zones, (str, Path)):
        from petrophys.io.geology_reader import _read_table

        zones = _read_table(Path(zones))
    if isinstance(zones, pd.DataFrame):
        labels = [str(column).strip().lower() for column in zones.columns]
        repeated = sorted({label for label in labels if labels.count(label) > 1})
        if repeated:
            raise ValueError(f"the table has more than one {', '.join(repeated)} column")
        if len(zones.columns) == 1 and ";" in str(zones.columns[0]):
            raise ValueError("the file has one column with ';' in its header: save it comma-separated")
        return [dict(row) for _, row in zones.iterrows()]
    if isinstance(zones, Mapping):
        raise TypeError("zones takes a list of zones (one mapping each), a table or a CSV/Excel path")
    return [dict(row) for row in zones]


def _zone_value(name: str, value: Any) -> Any:
    """A zone table cell as its setting expects it (tables read numbers as text)."""
    if isinstance(value, str):
        text = value.strip()
        if name in _BOOL_FIELDS:
            lowered = text.lower()
            if lowered in {"true", "yes", "1"}:
                return True
            if lowered in {"false", "no", "0"}:
                return False
            return text
        try:
            number = float(text)
        except ValueError:
            return text
        return int(number) if name == "mineral_solver_version" and number.is_integer() else number
    if isinstance(value, np.generic):
        value = value.item()
    if name == "mineral_solver_version" and isinstance(value, float) and value.is_integer():
        return int(value)  # a table column with a blank cell reads 2 as 2.0
    return value


def _normalized_zones(config: "DynamicWorkflowConfig", errors: List[str], raw: Any = None) -> Optional[tuple]:
    """``config.zones`` as ``({"top", "bottom", "depth_unit", "name", "settings"}, ...)``.

    Depths stay in their own unit (``depth_unit``, else the well's) until a
    run converts them; settings are checked as a config would check them.
    """
    import difflib
    from collections.abc import Mapping

    import pandas as pd

    if raw is None:
        raw = config.zones
    if raw is None:
        return None
    if config.calculation_profile in _BENCHMARK_PROFILES:
        errors.append('zones is not available under the "geolink" calculation profile')
        return None
    try:
        rows = _zone_rows(raw)
    except (TypeError, ValueError, OSError) as exc:
        errors.append(f"zones: {exc}")
        return None
    fields = {item.name for item in dataclasses.fields(DynamicWorkflowConfig) if not item.name.startswith("_")}
    zones = []
    for number, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            errors.append(f"zones[{number}]: each zone is a mapping such as {{'top': ..., 'bottom': ..., 'rw': ...}}")
            continue
        if isinstance(row.get("settings"), Mapping):
            # A zone as this function returns it (a copied config).
            row = {**{key: value for key, value in row.items() if key != "settings"}, **row["settings"]}
        names = [str(key).strip().lower() for key in row]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            errors.append(f"zones: zone {number}: {', '.join(repeated)} given more than once")
            continue
        cells = {
            str(key).strip().lower(): value for key, value in row.items()
            if not (value is None or (not isinstance(value, str) and np.ndim(value) == 0 and pd.isna(value))
                    or (isinstance(value, str) and not value.strip()))
        }
        label = str(cells.pop("name", cells.pop("zone", "")) or "").strip() or str(number)
        well_name = cells.pop("well", None)
        depth_unit = cells.pop("depth_unit", None)
        if depth_unit is not None:
            from petrophys.model.units import normalize_depth_unit

            if normalize_depth_unit(str(depth_unit), strict=False) not in {"m", "ft"}:
                errors.append(f"zones: zone {label}: depth_unit {depth_unit!r} is not m or ft")
                continue
        top_key = next((key for key in _ZONE_TOP_NAMES if key in cells), None)
        bottom_key = next((key for key in _ZONE_BOTTOM_NAMES if key in cells), None)
        if top_key is None or bottom_key is None:
            errors.append(f"zones: zone {label} needs a top and a bottom depth")
            continue
        try:
            top, bottom = float(cells.pop(top_key)), float(cells.pop(bottom_key))
        except (TypeError, ValueError):
            errors.append(f"zones: zone {label} has a top or bottom that is not a number")
            continue
        if not (np.isfinite(top) and np.isfinite(bottom)) or top >= bottom:
            errors.append(f"zones: zone {label} needs a top above its bottom; got {top:g}-{bottom:g}")
            continue
        settings = {}
        reported = len(errors)
        for name, value in cells.items():
            if name in _WHOLE_RUN_SETTINGS:
                errors.append(f"zones: zone {label}: {name} holds for the whole run and cannot be set per zone")
            elif name in fields:
                settings[name] = _zone_value(name, value)
            else:
                from petrophys.workflow.zoned_batch import PARAMETER_ALIASES

                hint = PARAMETER_ALIASES.get(name.upper())
                if hint is None:
                    close = difflib.get_close_matches(name, sorted(fields), n=1)
                    hint = close[0] if close else None
                errors.append(
                    f"zones: zone {label}: unknown setting {name!r}"
                    + (f" (did you mean {hint!r}?)" if hint else "")
                )
        if len(errors) > reported:
            continue
        if settings:
            try:
                DynamicWorkflowConfig(**settings)
            except (TypeError, ValueError) as exc:
                detail = str(exc).replace("Invalid DynamicWorkflowConfig:\n  - ", "").replace("\n  - ", "; ")
                errors.append(f"zones: zone {label}: {detail}")
                continue
        zone = {
            "name": label, "top": top, "bottom": bottom,
            "depth_unit": None if depth_unit is None else str(depth_unit).strip(),
            "settings": settings,
        }
        if well_name is not None:
            zone["well"] = str(well_name).strip()
        zones.append(zone)
    # Zones of different wells may share depths.
    for name in sorted({_well_name_key(zone.get("well")) for zone in zones}):
        same_well = [zone for zone in zones if _well_name_key(zone.get("well")) == name]
        if len({zone["depth_unit"] for zone in same_well}) > 1:
            continue
        ordered = sorted(same_well, key=lambda zone: zone["top"])
        for upper, lower in zip(ordered, ordered[1:]):
            if lower["top"] < upper["bottom"]:
                errors.append(
                    f"zones: zone {upper['name']} ({upper['top']:g}-{upper['bottom']:g}) and "
                    f"zone {lower['name']} ({lower['top']:g}-{lower['bottom']:g}) overlap"
                )
    return tuple(zones)


def _well_name_key(value: Any) -> str:
    """A well name as zone tables match it: without case, spacing or the ``_``/``/`` difference."""
    import re

    text = str(value or "").strip().upper().replace("\\", "/").replace("_", "/")
    return re.sub(r"\s+", " ", text)


def _zones_for_well(zones: Sequence[Mapping[str, Any]], *names: Any) -> List[Dict[str, Any]]:
    """The zones that apply to a well named any of ``names``.

    Zones without a ``well`` apply to every well; the others to the well
    whose name they give.
    """
    keys = {_well_name_key(name) for name in names if str(name or "").strip()}
    return [dict(zone) for zone in zones if not zone.get("well") or _well_name_key(zone["well"]) in keys]


def _zone_intervals_for_well(well: "Well", zones: Any) -> List[Dict[str, Any]]:
    """The ``name``, ``top`` and ``bottom`` (well depth unit) of the zones of a table for ``well``."""
    from petrophys.workflow.zoned_batch import _convert_depth_between_units

    errors: List[str] = []
    normalized = _normalized_zones(DynamicWorkflowConfig(), errors, zones) or ()
    if errors:
        raise ValueError("; ".join(errors))
    names = (well.name, getattr(getattr(well, "header", None), "well_name", ""))
    intervals = []
    for zone in _zones_for_well(normalized, *names):
        unit = zone["depth_unit"] or well.depth_unit
        intervals.append({
            "name": zone["name"],
            "top": _convert_depth_between_units(zone["top"], unit, well.depth_unit),
            "bottom": _convert_depth_between_units(zone["bottom"], unit, well.depth_unit),
        })
    return sorted(intervals, key=lambda zone: zone["top"])



def _default_matrix_type(lithology_domain: Any) -> str:
    """``matrix_type`` left unset: sandstone in the clastic domain, else "auto"."""
    from petrophys.evaluation.net_pay.cutoffs import cutoff_family

    domain = str(lithology_domain).strip().lower()
    clastic = domain not in {"auto", "evaporite"} and cutoff_family(domain) != "carbonate"
    return "sandstone" if clastic else "auto"


def _copy_config(
    config: DynamicWorkflowConfig, *, explicit: Optional[Sequence[str]] = None, **changes: Any
) -> DynamicWorkflowConfig:
    """``dataclasses.replace(config, **changes)`` that keeps the record of the
    caller's settings, adding ``explicit`` (default: every changed setting).

    A new ``lithology_domain`` re-resolves a ``matrix_type`` that was left
    unset, as a new config would ("geolink": kept).
    """
    changes = dict(changes)
    if "lithology_domain" in changes and "lithology" not in changes:
        # The copy mirrors its old domain in the ``lithology`` alias otherwise.
        changes["lithology"] = None
    if (
        "lithology_domain" in changes
        and "matrix_type" not in changes
        and getattr(config, "_matrix_defaulted", False)
        and config.calculation_profile not in _BENCHMARK_PROFILES
        and not reproducing_benchmark()
    ):
        changes["matrix_type"] = None
    copy = dataclasses.replace(config, **changes)
    if "matrix_type" not in changes and getattr(config, "_matrix_defaulted", False):
        copy._matrix_defaulted = True
    added = set(changes) - {"lithology", "matrix_type"} if explicit is None else set(explicit)
    copy._explicit_settings = _explicit(config) | added
    return copy


def _config_with_header_settings(
    well: "Well", config: Optional[DynamicWorkflowConfig], options: Optional[Dict[str, Any]] = None
) -> DynamicWorkflowConfig:
    """The run config: ``options`` over ``config``, LAS header settings filling
    the settings left at their defaults.

    Header values with a native LAS route (RW/RWT, BHT, RMF, RM) stay on the
    well, where each step reads them with its own unit and reference handling.
    """
    import copy as _copy

    from petrophys.workflow.zoned_batch import _NATIVE_HEADER_FIELDS, _extract_global_parameter_overrides

    options = dict(options or {})
    profile = str(
        options.get("calculation_profile", getattr(config, "calculation_profile", "current"))
    ).strip().lower()
    metadata_options, _ = _extract_global_parameter_overrides(well, calculation_profile=profile)
    # A header TD is kept, converted to feet from its own unit (the "geolink"
    # profile read it from the well header instead).
    historical = profile in _BENCHMARK_PROFILES or reproducing_benchmark()
    native = {name: metadata_options.pop(name) for name in _NATIVE_HEADER_FIELDS if name in metadata_options}
    if historical:
        metadata_options.pop("td_ft", None)
    # Your values of a mapping setting, then the header's for the keys you leave out.
    yours = {
        name: options[name] if name in options
        else (getattr(config, name, None) if config is not None and name in _explicit(config) else None)
        for name in _MAPPING_SETTINGS
    }
    mapping_notes: List[str] = []
    for name in _MAPPING_SETTINGS:
        header_value = metadata_options.pop(name, None)
        if not header_value:
            continue
        options[name] = _merged_mapping(header_value, yours[name])
        if yours[name] and _merged_mapping(header_value, yours[name]) != _merged_mapping(yours[name], header_value):
            mapping_notes.append(
                f"The LAS header gives {name}={header_value!r}; your {name}={yours[name]!r} is used where they differ"
            )
    if config is None:
        config = DynamicWorkflowConfig(**{**metadata_options, **options})
        # Header values are not the caller's settings.
        config._explicit_settings = _explicit(config) - (set(metadata_options) - set(options))
        if not historical:
            _note_header_settings(well, config, {**metadata_options, **native}, yours, mapping_notes)
        return config
    settings = {
        name: value
        for name, value in metadata_options.items()
        # A header value fills a setting the config leaves at its default
        # ("geolink": a setting left unset).
        if (getattr(config, name, None) is None if historical else _left_at_default(config, name))
        # Under the "geolink" profile an unset cutoff already holds its
        # profile default (VCL 0.3, SW 0.7), which a header does not replace.
        and not (profile in _BENCHMARK_PROFILES and name in _PROFILE_DEFAULTS)
    }
    settings.update(options)
    if historical:
        if "lithology_domain" in settings and "lithology" not in settings:
            settings["lithology"] = None
        explicit = _explicit(config) | set(options)
        result = dataclasses.replace(config, **settings) if settings else _copy.deepcopy(config)
        result._explicit_settings = frozenset(explicit)
        return result
    result = _copy_config(config, explicit=list(options), **settings)
    _note_header_settings(well, result, {**metadata_options, **native}, yours, mapping_notes)
    return result


def _note_header_settings(
    well: "Well",
    config: DynamicWorkflowConfig,
    header: Dict[str, Any],
    yours: Optional[Dict[str, Any]] = None,
    mapping_notes: Optional[List[str]] = None,
) -> None:
    """Record on ``config`` the header values a setting overrides, and the header's zones.

    ``_header_notes`` name the whole-well and zone header values the caller's
    settings override; ``_header_zones`` are the LAS header zones (``M_Z1``
    with ``TOP_Z1``/``BOT_Z1``, ...) with the settings the run applies inside
    them. The values the steps read from the header themselves (``RW_Z1``,
    ``RMF_Z1``, ...) are left to those steps.
    """
    from petrophys.workflow.zoned_batch import _NATIVE_HEADER_FIELDS, extract_zoned_parameters

    explicit = _explicit(config)
    notes = list(mapping_notes or [])
    for name, value in header.items():
        given = getattr(config, name, None)
        if name in explicit and isinstance(given, dict) and "*" not in given:
            # Your values for the lithologies you list; the header's for the rest.
            setattr(config, name, {**given, "*": float(value)})
            notes.append(
                f"The LAS header gives {name}={value!r}: used for the lithologies {name} does not list"
            )
        elif name in explicit and value != given:
            notes.append(
                f"The LAS header gives {name}={value!r}; not used, {name}={given!r} is set"
            )
    zones = []
    for zone in extract_zoned_parameters(well, calculation_profile=config.calculation_profile):
        settings = {}
        for name, value in zone.config_overrides.items():
            if name in _NATIVE_HEADER_FIELDS or name in _WHOLE_RUN_SETTINGS:
                continue
            if name in _MAPPING_SETTINGS:
                # Inside the zone: the zone's values over the whole-well ones, yours over both.
                settings[name] = _merged_mapping(
                    _merged_mapping(getattr(config, name, None), value), (yours or {}).get(name)
                )
                continue
            given = getattr(config, name, None)
            if name in explicit and isinstance(given, dict):
                # Inside the zone, the zone's value fills the lithologies yours leave out.
                listed = {key: item for key, item in given.items() if key != "*"}
                settings[name] = {**listed, "*": float(value)}
                continue
            if name in explicit:
                notes.append(
                    f"The LAS header zone Z{zone.index} ({zone.top:g}-{zone.bottom:g} {well.depth_unit}) gives "
                    f"{name}={value!r}; not used, {name}={getattr(config, name)!r} is set"
                )
                continue
            settings[name] = value
        if settings:
            zones.append({
                "name": f"Z{zone.index} (LAS header)", "top": float(zone.top), "bottom": float(zone.bottom),
                "depth_unit": None, "settings": settings,
            })
    config._header_notes = notes
    config._header_zones = tuple(zones)

def _attached_formation_tops(records: List[Dict[str, Any]], terminal_depth: float) -> Any:
    """Build validated, non-overlapping FormationTops from attached records.

    Raises
    ------
    ValueError
        If the records cannot form bounded, non-overlapping intervals.
    """
    import pandas as pd
    from petrophys.model.formation_tops import FormationTops

    table = pd.DataFrame(records)
    if "bottom_depth" in table and table["bottom_depth"].isna().all():
        table = table.drop(columns="bottom_depth")
    tops = FormationTops.from_dataframe(
        table, name_col="formation_name", terminal_depth=terminal_depth
    )
    tops.validate()
    return tops


# --- DynamicWorkflowConfig validation ---------------------------------------

# Matrices that belong to each explicitly chosen lithology domain (first = default).
_DOMAIN_MATRICES = {"clastic": ("sandstone",), "carbonate": ("limestone", "dolomite")}

_MUD_TYPES = ("WBM", "OBM")
_MUD_WATERS = ("fresh", "salt")
_LITHOLOGY_FAMILIES = ("auto", "clastic", "carbonate", "evaporite")
#: ``lithology_domain`` when neither it nor its ``lithology`` alias is set.
_DEFAULT_LITHOLOGY_DOMAIN = "clastic"
#: Logged vendor density and sonic porosities, in order of preference, that
#: PHID and PHIS take where their values are in :data:`_POROSITY_RANGE`.
_VENDOR_DENSITY_POROSITY = ("DPOR", "DPHI", "DPHZ", "PORD")
_VENDOR_SONIC_POROSITY = ("SPOR", "SPHI", "PORS")
#: Porosity range (v/v) of the computed PHID and PHIS, which a vendor value must lie in.
_POROSITY_RANGE = (0.0, 0.5)
_PERCENT_UNITS = {"%", "pu", "p.u.", "percent", "pct"}
_FRACTION_UNITS = {"v/v", "frac", "fraction", "dec", "decimal", "m3/m3", "ft3/ft3"}
_LITHOLOGY_SOURCES = ("auto", "external", "vendor", "computed", "config")
_HC_TYPES = ("oil", "gas")
_SWIRR_FLAG_PHI_CL = ("workflow", "all_shale_phit")
_MINERAL_SOLVER_VERSIONS = (1, 2)
_CARBONATE_CALIBRATIONS = ("generic", "costa")
#: Profiles that keep the eval package's calculations (see
#: DynamicWorkflowConfig.calculation_profile).
_BENCHMARK_PROFILES = ("geolink",)
_CALCULATION_PROFILES = ("current", *_BENCHMARK_PROFILES)
#: A profile name that is refused with a pointer to the "geolink" profile.
_RENAMED_PROFILE = "benchmark"
_RENAMED_PROFILE_MESSAGE = (
    'calculation_profile "benchmark" is not a profile; use "geolink" for the calculations '
    'of the GeoLink eval package. To reproduce a published context package, use context_profile '
    '("geolink", "costa" or "danish") in export_context.'
)
#: The eval package's wells that its re-run interpreted, as compared by
#: :func:`_normalized_well_name`. The "geolink" profile gives them the re-run's
#: calculations.
_RERUN_WELLS = frozenset({"15_9-17", "34_10-23"})
#: True inside :func:`_rerun_calculations`: the "geolink" profile then gives
#: every well the calculations of the eval package's re-run.
_RERUN_CALCULATIONS_REQUESTED: ContextVar[bool] = ContextVar(
    "petrophys_rerun_calculations_requested", default=False
)
#: The published context packages besides GeoLink's eval package whose
#: calculations the "geolink" profile keeps. All were interpreted with the
#: calculations of the GeoLink re-run; the COSTA package also with the two
#: saturation rules of :meth:`DynamicWorkflow._costa_saturation`. The GeoLink
#: training package ("geolink_train") also corrected Rw estimates for
#: temperature with Arps (:meth:`DynamicWorkflow._train_rw_curve`), calibrated
#: core porosity only together with core permeability
#: (:meth:`DynamicWorkflow._train_lacks_core_permeability`,
#: :meth:`DynamicWorkflow._train_core_porosity`), read GR alone for VCL,
#: grouped a well without lithology codes by the LITHO of its lithology step
#: (:meth:`DynamicWorkflow._train_lithology_codes`), and flagged bad hole
#: from |DRHO| alone where the well records no bit size.
_DATASET_CALCULATIONS = ("danish", "costa", "geolink_train")
#: The dataset inside :func:`_dataset_calculations`, else None.
_DATASET_CALCULATIONS_REQUESTED: ContextVar[Optional[str]] = ContextVar(
    "petrophys_dataset_calculations_requested", default=None
)
#: Settings whose default depends on ``calculation_profile``, as (default under
#: ``"current"``, default under the "geolink" profile). A setting left as None
#: takes the default of its profile; a value set explicitly always wins. Unset
#: net-pay cutoffs stay None under ``"current"``: net pay resolves them per
#: lithology family.
_PROFILE_DEFAULTS: Dict[str, Tuple[Any, Any]] = {
    "standardize_input": (True, False),
    "vcl_cutoff": (None, 0.3),
    "sw_cutoff": (None, 0.7),
}
#: Rxo/Rt classes behind the invasion step's conductive, minimal and resistive
#: percentages under the "geolink" profile: the eval package's table, a frozen
#: copy kept apart from the canonical CLASS_TABLES["RXO_RT"].
_BENCHMARK_INVASION_BANDS = Bands(
    "MINIMAL_INVASION", (("<", 0.8, "CONDUCTIVE_INVASION"), (">", 1.2, "RESISTIVE_INVASION"))
)
#: The curve the invasion ratio reads first under the "geolink" profile: the
#: shallow RSHA, then RXO and RESS. "current" reads the flushed-zone RXO first.
_BENCHMARK_RXO_CURVE = "RSHA"
# Matrix slowness (us/ft) that the "geolink" profile uses in place of the
# shared mineral table's chartbook values: coal 250 (table 120) everywhere, and
# gypsum 83 (table 52.5) for the gypsum lithology group.
_BENCHMARK_SLOWNESS = {"coal": 250.0}
_BENCHMARK_GROUP_SLOWNESS = {**_BENCHMARK_SLOWNESS, "gypsum": 83.0}
#: Limits the "current" profile adds to the standardizer's implausible-value
#: screening (CURVE_PLAUSIBLE_RANGES). No rock matrix is faster than
#: 38.5 us/ft (dolomite, 23,000-26,000 ft/s; calcite 43.5-47.6, anhydrite 50,
#: quartz 51-55.5 us/ft: Schlumberger, 1989), so a lower compressional
#: slowness is a unit or scale error. A formation's static SP lies within
#: about 200 mV of the shale baseline; readings beyond +/-500 mV (the
#: library's value-range rule for SP) come from the tool or the recording.
_CURRENT_RANGE_LIMITS = {"DTC": (38.5, 300.0), "SP": (-500.0, 500.0)}
_BOOL_FIELDS = (
    "standardize_input", "rename_input_curves", "allow_default_rw",
    "mud_has_clay_or_kcl", "raise_on_error", "apply_hc_correction", "use_peer_core",
    "reuse_input_resflag",
)
_POSITIVE_FIELDS = (
    "rw", "rmf_at_meas_temp", "rm", "rmf_salinity_ppm", "rmf_chlorides_mg_l", "salinity_ppm", "td_ft",
    "rho_ma", "rho_fl", "rho_cl", "dt_ma", "dt_fl", "a", "m", "n", "m_star",
    "rwb", "rtsh", "vsh_exp", "shale_exp", "b_coeff",
)
_FRACTION_FIELDS = (
    "vcl_cutoff", "sw_cutoff", "phie_cutoff", "swirr", "phi_cl", "sw_routing_vcl_cutoff",
)
_FINITE_FIELDS = (
    "surface_temp_f", "bht_f", "rmf_meas_temp_f", "rm_meas_temp_f",
    "rw_meas_temp_f", "nphi_ma", "gr_min", "gr_max", "sp_min", "sp_max", "cec",
    "gradient_f_per_100ft", "rho_hc", "hc_hydrogen_index",
)


#: Least r^2 of a same-well core porosity correction that the "current"
#: calculations apply: the log porosity must explain at least half of the core
#: porosity variance. A poorer least-squares line is mostly scatter, flattened
#: towards the mean core porosity, and gives porosity to tight rock. With the
#: minimum of 8 core-log pairs (perm_module.CORE_POROSITY_MIN_POINTS) it is also
#: the 5% significance level of the correlation.
_CORE_POROSITY_MIN_R_SQUARED = 0.5

#: Core porosity mnemonics that let the "current" core-correction step run:
#: those core_data_from_well reads core porosity from. The "geolink" profile
#: ran the step on PHIC and CPOR only.
_CORE_POROSITY_CURVES = ("PHIC", "CPOR", "CPOR-AMB", "CPOR_AMB")

#: Why the current profile writes no SXO or MHI for oil-based mud.
_OBM_FLUSHED_ZONE_NOTE = (
    "mud_type='OBM' (SXO and MHI describe flushing by a conductive water-based "
    "filtrate; the 200 ohm.m RMF written for oil-based mud is a placeholder)"
)


def _cec_range_error(cec: Any) -> Optional[str]:
    """Message for a CEC (meq/100 g) outside the plausible whole-rock range."""
    from petrophys.interpretation.saturation.waxman_smits import MAX_CEC_MEQ_PER_100G

    if isinstance(cec, (bool, np.bool_)) or not isinstance(cec, (int, float, np.integer, np.floating)):
        return None
    if not np.isfinite(cec) or 0.0 <= cec <= MAX_CEC_MEQ_PER_100G:
        return None
    return (
        f"cec must be between 0 and {MAX_CEC_MEQ_PER_100G:g} meq/100 g "
        f"(pure smectite is 80-150 meq/100 g); got {cec!r}"
    )


#: Absolute zero in degrees Fahrenheit.
_ABSOLUTE_ZERO_F = -459.67
#: Plausible ranges of the "current" matrix and fluid settings: a whole-well
#: matrix density lies within 1.5-3.5 g/cm3 (sylvite 1.99, quartz 2.65,
#: anhydrite 2.98), a pore fluid is lighter than 2 g/cm3 (saturated brine
#: about 1.2) and a matrix neutron reading lies within -0.1 and 0.5 v/v.
_RHO_MA_RANGE = (1.5, 3.5)
_RHO_FL_MAX = 2.0
_NPHI_MA_RANGE = (-0.1, 0.5)
#: Most dissolved salt a formation water holds, in ppm (halite saturation is
#: about 260,000 ppm NaCl; CaCl2 brines reach about 400,000 ppm).
_MAX_SALINITY_PPM = 400_000.0
#: A salinity above this is used with a warning: NaCl brine holds about
#: 260,000 ppm at surface temperature and up to about 350,000 ppm in the
#: hottest reservoirs, and the resistivity-salinity conversions are NaCl charts.
_USUAL_MAX_SALINITY_PPM = 350_000.0
#: Limits of settings that are refused beyond them: a formation water above
#: 100 ohm.m holds under about 50 ppm of salt at 75 degF (rain water), a
#: bottom-hole temperature above 700 degF is beyond wireline logging (the
#: limit the water-sample temperatures take) and no mean annual surface
#: temperature is above 150 degF.
_MAX_RW_OHMM = 100.0
_MAX_FORMATION_TEMP_F = 700.0
_MAX_SURFACE_TEMP_F = 150.0
#: Matrix slowness lies between dolomite (38.5-43.5 us/ft) and coal (about
#: 120 us/ft); pore fluids between a saturated brine under pressure (about
#: 160 us/ft; water 189, oil about 240) and gas (several hundred us/ft). A
#: sandstone matrix in us/m reads 182.
_DT_MA_RANGE = (35.0, 130.0)
_DT_FL_RANGE = (150.0, 1000.0)
#: NaCl brine saturates near 260,000 ppm; apparent Rw implying more is not water.
_SATURATED_BRINE_PPM = 260_000.0
#: Rwa samples are compared at this temperature before the P5 is taken.
_RWA_REFERENCE_TEMP_F = 75.0
#: The Rw in use is questioned when the median Rwa of the clean porous rock
#: is below this fraction of it: Archie Sw above 1.4 (n = 2) in more than
#: half of that rock.
_RWA_CHECK_RATIO = 0.5
#: Fewest clean porous samples for that comparison.
_RWA_CHECK_MIN_SAMPLES = 30


#: Usual Archie ranges; a value outside is used, with a warning.
_ARCHIE_USUAL = {"a": (0.3, 3.0), "m": (1.0, 4.0), "n": (1.0, 5.0)}


def _unusual_setting_notes(config: "DynamicWorkflowConfig") -> List[str]:
    """Warnings for set Archie values and salinities outside their usual range ("current" runs)."""
    from collections.abc import Mapping

    if config.calculation_profile != "current":
        return []
    salinity_notes = []
    for name in ("salinity_ppm", "rmf_salinity_ppm"):
        value = getattr(config, name, None)
        if isinstance(value, (int, float, np.integer, np.floating)) and value > _USUAL_MAX_SALINITY_PPM:
            salinity_notes.append(
                f"{name}={float(value):,.0f} ppm is more than a NaCl brine holds (about "
                f"{_SATURATED_BRINE_PPM:,.0f} ppm at surface temperature, {_USUAL_MAX_SALINITY_PPM:,.0f} ppm in "
                "the hottest reservoirs); used as given. Check the unit (mg/l reads higher than ppm in "
                "dense brines)."
            )
    sources = [("", {name: getattr(config, name, None) for name in _ARCHIE_USUAL})]
    sources += [
        (f"zone {zone['name']}: ", zone.get("settings") or {}) for zone in (config.zones or ())
    ]
    notes = []
    for prefix, settings in sources:
        for name, (low, high) in _ARCHIE_USUAL.items():
            value = settings.get(name)
            items = value.items() if isinstance(value, Mapping) else [(None, value)]
            for key, item in items:
                if isinstance(item, bool) or not isinstance(item, (int, float, np.integer, np.floating)):
                    continue
                if not low <= float(item) <= high:
                    label = name if key is None else f"{name}[{key!r}]"
                    notes.append(
                        f"{prefix}{label}={float(item):g} is outside the usual {low:g}-{high:g}; "
                        "used as given. Check it is not a typo."
                    )
    return notes + salinity_notes


def _physical_range_errors(config: "DynamicWorkflowConfig") -> List[str]:
    """Messages for "current" settings outside their physical range.

    Temperatures below absolute zero, a bottom-hole temperature below the
    surface temperature, densities and matrix neutron values no rock has,
    a matrix lighter than its pore fluid and a salinity outside 0-400,000 ppm
    are refused; most are a unit slip (kg/m3 for g/cm3, degC for degF). So are
    a clay density outside the matrix range, an irreducible water saturation
    of 0, an Rw above 100 ohm.m, a bottom-hole temperature above 700 degF, a
    surface temperature above 150 degF and sonic slownesses outside the
    matrix (35-130 us/ft) and pore-fluid (150-1000 us/ft) ranges.
    """
    def number(name: str) -> Optional[float]:
        value = getattr(config, name, None)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            return None
        return float(value) if np.isfinite(value) else None

    errors = []
    for name in ("surface_temp_f", "bht_f", "rw_meas_temp_f", "rmf_meas_temp_f", "rm_meas_temp_f"):
        value = number(name)
        if value is not None and value < _ABSOLUTE_ZERO_F:
            errors.append(
                f"{name} must be a temperature in degF, at or above absolute zero ({_ABSOLUTE_ZERO_F} degF); "
                f"got {value!r}"
            )
    for name in ("rw_meas_temp_f", "rmf_meas_temp_f", "rm_meas_temp_f"):
        value = number(name)
        if value is not None and _ABSOLUTE_ZERO_F <= value and not 0.0 <= value <= 700.0:
            errors.append(
                f"{name} must be the temperature of a water sample, 0-700 degF; got {value!r}"
                + (" (a degC value? 25 degC is 77 degF)" if value < 0 else "")
            )
    gradient = number("gradient_f_per_100ft")
    if gradient is not None and not 0.0 < gradient <= 10.0:
        errors.append(
            "gradient_f_per_100ft must be a geothermal gradient above 0 and at most 10 degF/100 ft "
            f"(3 degC/100 m is 1.65 degF/100 ft); got {gradient!r}"
        )
    if gradient is not None and number("bht_f") is not None and {"bht_f", "gradient_f_per_100ft"} <= _explicit(config):
        errors.append("set bht_f or gradient_f_per_100ft, not both")
    bht, surface = number("bht_f"), number("surface_temp_f")
    if bht is not None and surface is not None and _ABSOLUTE_ZERO_F <= bht < surface:
        errors.append(
            f"bht_f ({bht:g} degF) must not be below surface_temp_f ({surface:g} degF): temperature "
            "rises with depth (set surface_temp_f for a cold surface or seabed)"
        )
    rho_ma, rho_fl, nphi_ma = number("rho_ma"), number("rho_fl"), number("nphi_ma")
    if rho_ma is not None and not _RHO_MA_RANGE[0] <= rho_ma <= _RHO_MA_RANGE[1]:
        errors.append(
            f"rho_ma must be a matrix density of {_RHO_MA_RANGE[0]:g}-{_RHO_MA_RANGE[1]:g} g/cm3 "
            f"(2650 kg/m3 is 2.65 g/cm3); got {rho_ma!r}"
        )
    if rho_fl is not None and not 0.0 < rho_fl <= _RHO_FL_MAX:
        errors.append(f"rho_fl must be a fluid density above 0 and at most {_RHO_FL_MAX:g} g/cm3; got {rho_fl!r}")
    if rho_ma is not None and rho_fl is not None and rho_ma <= rho_fl:
        errors.append(f"rho_ma ({rho_ma:g} g/cm3) must be greater than rho_fl ({rho_fl:g} g/cm3)")
    if nphi_ma is not None and not _NPHI_MA_RANGE[0] <= nphi_ma <= _NPHI_MA_RANGE[1]:
        errors.append(
            f"nphi_ma must be a matrix neutron porosity of {_NPHI_MA_RANGE[0]:g} to {_NPHI_MA_RANGE[1]:g} v/v; "
            f"got {nphi_ma!r}"
        )
    rho_hc = number("rho_hc")
    if rho_hc is not None and not 0.0 < rho_hc <= 1.2:
        errors.append(f"rho_hc must be a hydrocarbon density above 0 and at most 1.2 g/cm3; got {rho_hc!r}")
    hc_hi = number("hc_hydrogen_index")
    if hc_hi is not None and not 0.0 <= hc_hi <= 1.2:
        errors.append(f"hc_hydrogen_index must be between 0 and 1.2; got {hc_hi!r}")
    for name in ("salinity_ppm", "rmf_salinity_ppm"):
        salinity = number(name)
        if salinity is not None and not 0.0 <= salinity <= _MAX_SALINITY_PPM:
            errors.append(f"{name} must be between 0 and {_MAX_SALINITY_PPM:,.0f} ppm; got {salinity!r}")
    rho_cl = number("rho_cl")
    if rho_cl is not None and not _RHO_MA_RANGE[0] <= rho_cl <= _RHO_MA_RANGE[1]:
        errors.append(
            f"rho_cl must be a clay density of {_RHO_MA_RANGE[0]:g}-{_RHO_MA_RANGE[1]:g} g/cm3 "
            f"(2450 kg/m3 is 2.45 g/cm3); got {rho_cl!r}"
        )
    if number("swirr") == 0.0:
        errors.append(
            "swirr must be above 0: the permeability equations divide by the irreducible water "
            "saturation; got 0"
        )
    rw = number("rw")
    if rw is not None and rw > _MAX_RW_OHMM:
        errors.append(
            f"rw must be a formation water resistivity of at most {_MAX_RW_OHMM:g} ohm.m (sea water is "
            f"about 0.2 ohm.m and drinking water 10-50 ohm.m at 75 degF); got {rw!r}"
        )
    if bht is not None and bht > _MAX_FORMATION_TEMP_F:
        errors.append(
            f"bht_f must be a formation temperature of at most {_MAX_FORMATION_TEMP_F:g} degF; got {bht!r}"
        )
    if surface is not None and surface > _MAX_SURFACE_TEMP_F:
        errors.append(
            f"surface_temp_f must be a surface (mean annual) temperature of at most "
            f"{_MAX_SURFACE_TEMP_F:g} degF; got {surface!r}"
        )
    dt_ma, dt_fl = number("dt_ma"), number("dt_fl")
    if dt_ma is not None and not _DT_MA_RANGE[0] <= dt_ma <= _DT_MA_RANGE[1]:
        errors.append(
            f"dt_ma must be a matrix slowness of {_DT_MA_RANGE[0]:g}-{_DT_MA_RANGE[1]:g} us/ft (dolomite "
            f"43.5, limestone 47.6, sandstone 55.5; 182 us/m is 55.5 us/ft); got {dt_ma!r}"
        )
    if dt_fl is not None and not _DT_FL_RANGE[0] <= dt_fl <= _DT_FL_RANGE[1]:
        errors.append(
            f"dt_fl must be a pore-fluid slowness of {_DT_FL_RANGE[0]:g}-{_DT_FL_RANGE[1]:g} us/ft, above "
            f"the matrix slowness (water 189, oil about 240); got {dt_fl!r}"
        )
    return errors


def _choice_error(name: str, value: Any, choices: Sequence[str]) -> str:
    import difflib

    message = f"{name}={value!r} is not valid; use one of: {', '.join(choices)}"
    close = difflib.get_close_matches(
        str(value).strip().lower(), [c.lower() for c in choices], n=1, cutoff=0.75
    )
    if close:
        match = next(c for c in choices if c.lower() == close[0])
        message += f" (did you mean {match!r}?)"
    return message


def _normalized_choice(
    config: "DynamicWorkflowConfig",
    name: str,
    choices: Sequence[str],
    errors: List[str],
    *,
    aliases: Sequence[str] = (),
) -> None:
    value = getattr(config, name)
    if not isinstance(value, str):
        errors.append(f"{name} must be a string; got {type(value).__name__}")
        return
    key = value.strip().lower()
    canonical = {choice.lower(): choice for choice in choices}
    if key in canonical:
        setattr(config, name, canonical[key])
    elif key in {alias.lower() for alias in aliases}:
        setattr(config, name, key)
    elif name == "calculation_profile" and key == _RENAMED_PROFILE:
        errors.append(_RENAMED_PROFILE_MESSAGE)
    else:
        errors.append(_choice_error(name, value, choices))


_CARBONATE_WORDS = ("lime", "chalk", "dolo", "calcite", "marl", "ooze", "carbonate", "calcareous_debris")
#: Rock names that are also matrix types (:class:`~petrophys.model.constants.MatrixProperties`).
_MATRIX_ROCKS = ("sandstone", "limestone", "dolomite", "calcite", "anhydrite", "gypsum", "halite")
_EVAPORITE_WORDS = ("anhydrite", "gypsum", "halite", "salt", "sylvi", "carnallite", "kainite", "potash", "evaporite")


def _rock_family(name: str) -> str:
    """The lithology domain of a rock name: carbonate, evaporite or clastic."""
    if any(word in name for word in _EVAPORITE_WORDS):
        return "evaporite"
    if any(word in name for word in _CARBONATE_WORDS):
        return "carbonate"
    return "clastic"


def _validate_lithology(config: "DynamicWorkflowConfig", errors: List[str]) -> None:
    """Accept a cutoff family or any lithology with tabulated Archie coefficients."""
    from petrophys.model.constants import ArchieParameterDefaults

    value = config.lithology_domain
    if not isinstance(value, str):
        errors.append(f"lithology_domain must be a string; got {type(value).__name__}")
        return
    key = value.strip().lower()
    if key in _LITHOLOGY_FAMILIES:
        config.lithology_domain = key
        return
    name = ArchieParameterDefaults.normalize(value)
    if name in ArchieParameterDefaults.names():
        if config.calculation_profile not in _BENCHMARK_PROFILES:
            # A rock name: its family is the domain, and the rock the matrix
            # when matrix_type is unset, so the mineral solve, matrix, Archie
            # coefficients and cutoffs agree.
            config.lithology_domain = _rock_family(name)
            if getattr(config, "_matrix_defaulted", False):
                config.matrix_type = (
                    name if name in _MATRIX_ROCKS else _default_matrix_type(config.lithology_domain)
                )
        return
    import difflib

    message = (
        f"lithology_domain={value!r} is not valid; use one of: "
        f"{', '.join(_LITHOLOGY_FAMILIES)}, or a lithology with tabulated Archie "
        "coefficients (petrophys.model.constants.ArchieParameterDefaults.names())"
    )
    candidates = list(_LITHOLOGY_FAMILIES) + [name.lower() for name in ArchieParameterDefaults.names()]
    close = difflib.get_close_matches(key, candidates, n=1, cutoff=0.75)
    if close:
        message += f" (did you mean {close[0]!r}?)"
    errors.append(message)


def _validate_lithology_curve(config: "DynamicWorkflowConfig", errors: List[str]) -> None:
    """Accept the mnemonic of an input lithology code curve, in capitals."""
    from petrophys.interpretation.lithology.matrix_inference import _COMPUTED_LITHOLOGY_CURVES

    value = config.lithology_curve
    if not isinstance(value, str) or not value.strip():
        errors.append(f"lithology_curve must be a curve mnemonic; got {value!r}")
        return
    name = value.strip().upper()
    if name in _COMPUTED_LITHOLOGY_CURVES or name.startswith("GT_"):
        errors.append(
            f"lithology_curve={value!r} is a curve the workflow computes; "
            "name the lithology code curve of the input logs"
        )
        return
    config.lithology_curve = name


def _normalized_well_name(name: Any) -> str:
    """A well name as the "geolink" routing compares it.

    Case and surrounding spaces are ignored and ``/`` reads as ``_``; the rest
    must match, so ``"15/9-17 A"`` is not ``"15_9-17"``.
    """
    return str(name).strip().lower().replace("/", "_")


def _is_rerun_well(well: "Well", file_stem: Optional[str] = None) -> bool:
    """True when ``well`` is one of the two wells the eval package's re-run interpreted.

    The well's UWI is checked, then its WELL header, then ``file_stem``, the
    LAS file name the zoned batch runner read it from; any of them naming
    15/9-17 or 34/10-23 (:func:`_normalized_well_name`) identifies the well.
    """
    header = getattr(well, "header", None)
    if isinstance(header, dict):
        uwi = header.get("UWI", header.get("uwi"))
        name = header.get("WELL", header.get("well_name"))
    else:
        uwi, name = getattr(header, "uwi", None), getattr(well, "name", None)
    return any(
        _normalized_well_name(item) in _RERUN_WELLS
        for item in (uwi, name, file_stem)
        if item is not None
    )


@contextmanager
def _rerun_calculations(enabled: bool = True) -> Iterator[None]:
    """Give every "geolink" run inside the block the calculations of the eval package's re-run.

    The context exporters enter it around their workflow runs, as the re-run
    wrote the package's context; the zoned batch runner enters it for a re-run
    well it recognized, which its file name alone may identify. With
    ``enabled=False`` it does nothing, and each well keeps its own calculations.
    """
    if not enabled:
        yield
        return
    token = _RERUN_CALCULATIONS_REQUESTED.set(True)
    try:
        yield
    finally:
        _RERUN_CALCULATIONS_REQUESTED.reset(token)


@contextmanager
def _dataset_calculations(dataset: str) -> Iterator[None]:
    """Give every "geolink" run inside the block the calculations of ``dataset``'s package.

    ``dataset`` is ``"danish"``, ``"costa"`` or ``"geolink_train"``
    (:data:`_DATASET_CALCULATIONS`). These packages were interpreted with the
    calculations of the GeoLink re-run, so every run inside the block also
    gets those.
    """
    if dataset not in _DATASET_CALCULATIONS:
        raise ValueError(f"dataset must be one of {_DATASET_CALCULATIONS}, got {dataset!r}")
    token = _DATASET_CALCULATIONS_REQUESTED.set(dataset)
    try:
        with _rerun_calculations():
            yield
    finally:
        _DATASET_CALCULATIONS_REQUESTED.reset(token)


def _uses_rerun_calculations(well: "Well", file_stem: Optional[str] = None) -> bool:
    """True when a "geolink" run of ``well`` gets the calculations of the eval package's re-run.

    That is inside :func:`_rerun_calculations`, or for one of the package's
    re-run wells (:func:`_is_rerun_well`); every other "geolink" run gets the
    calculations of the package's main run.
    """
    return _RERUN_CALCULATIONS_REQUESTED.get() or _is_rerun_well(well, file_stem)


def _profile_setting(config: DynamicWorkflowConfig, field_name: str) -> Any:
    """A setting's value, with its calculation-profile default when left as None."""
    value = getattr(config, field_name)
    if value is None and field_name in _PROFILE_DEFAULTS:
        current, benchmark = _PROFILE_DEFAULTS[field_name]
        value = benchmark if config.calculation_profile in _BENCHMARK_PROFILES else current
    return value


def _left_at_default(config: DynamicWorkflowConfig, field_name: str) -> bool:
    """True when a setting is unset or still has its default value.

    A setting the caller passed explicitly (``_explicit_settings``) is never
    at its default, even with the default's value. Under the "geolink" profile a setting of :data:`_PROFILE_DEFAULTS` that
    holds its "geolink" value (e.g. ``vcl_cutoff=0.3``) also counts as left
    at the default, as the eval package's calculations treat it.
    """
    import dataclasses

    value = getattr(config, field_name, None)
    if value is None:
        return True
    if (
        field_name in _explicit(config)
        and config.calculation_profile not in _BENCHMARK_PROFILES
        and not reproducing_benchmark()
    ):
        return False
    if field_name in _PROFILE_DEFAULTS and config.calculation_profile in _BENCHMARK_PROFILES:
        return value == _PROFILE_DEFAULTS[field_name][1]
    item = DynamicWorkflowConfig.__dataclass_fields__.get(field_name)
    if item is None:
        return False
    if field_name == "matrix_type":
        # The default (None) resolves to the domain's matrix when the config is made.
        return value in ("sandstone", "auto")
    if field_name == "lithology_domain":
        # The default (None) resolves to the clastic domain when the config is made.
        return value == _DEFAULT_LITHOLOGY_DOMAIN
    if item.default is not dataclasses.MISSING:
        return value == item.default
    if item.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return value == item.default_factory()  # type: ignore[misc]
    return False


#: Settings that take a mapping: header values and zone values fill the keys
#: they give, under the keys you give.
_MAPPING_SETTINGS = ("bad_hole", "perm_coefficients", "mineral_endpoints")


def _merged_mapping(base: Any, over: Any) -> Dict[str, Any]:
    """``over``'s keys over ``base``'s, a level deeper for mapping values (``mineral_endpoints``)."""
    from collections.abc import Mapping

    merged: Dict[str, Any] = {key: (dict(value) if isinstance(value, Mapping) else value)
                              for key, value in dict(base or {}).items()}
    for key, value in dict(over or {}).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = dict(value) if isinstance(value, Mapping) else value
    return merged


_BAD_HOLE_KEYS = ("cali", "drho", "large_hole")
_PERM_COEFFICIENT_KEYS = ("a", "b", "c")
_ENDPOINT_NAMES = ("quartz", "calcite", "dolomite", "anhydrite", "clay", "water")
_ENDPOINT_KEYS = ("rhob", "nphi", "pef", "u", "dt")


def _number_mapping(value: Any, name: str, keys: Sequence[str], errors: List[str], *,
                    positive: Sequence[str] = ()) -> Optional[Dict[str, float]]:
    """A mapping of some of ``keys`` to finite numbers (positive for ``positive``), else None with an error."""
    from collections.abc import Mapping

    if not isinstance(value, Mapping):
        errors.append(f"{name} takes a mapping of {', '.join(keys)}; got {type(value).__name__}")
        return None
    numbers: Dict[str, float] = {}
    for key, item in value.items():
        label = str(key).strip().lower()
        if label not in keys:
            errors.append(f"{name}: unknown key {key!r}; use {', '.join(keys)}")
            continue
        try:
            number = float(item)
        except (TypeError, ValueError):
            errors.append(f"{name}: {label} must be a number; got {item!r}")
            continue
        if not np.isfinite(number) or (label in positive and number <= 0):
            errors.append(f"{name}: {label} must be a {'positive' if label in positive else 'finite'} number; got {item!r}")
            continue
        numbers[label] = number
    return numbers


def _validate_parameter_mappings(config: "DynamicWorkflowConfig", errors: List[str]) -> None:
    """Check ``bad_hole``, ``perm_coefficients`` and ``mineral_endpoints``."""
    from collections.abc import Mapping

    names = ("bad_hole", "perm_coefficients", "mineral_endpoints")
    given = [name for name in names if getattr(config, name) is not None]
    if given and config.calculation_profile in _BENCHMARK_PROFILES:
        errors.extend(f'{name} is not available under the "geolink" calculation profile' for name in given)
        return
    if config.bad_hole is not None:
        config.bad_hole = _number_mapping(config.bad_hole, "bad_hole", _BAD_HOLE_KEYS, errors,
                                          positive=_BAD_HOLE_KEYS)
    if config.perm_coefficients is not None:
        config.perm_coefficients = _number_mapping(config.perm_coefficients, "perm_coefficients",
                                                   _PERM_COEFFICIENT_KEYS, errors, positive=("a",))
    if config.mineral_endpoints is not None:
        if not isinstance(config.mineral_endpoints, Mapping):
            errors.append("mineral_endpoints takes a mapping of mineral to endpoints, "
                          'e.g. {"quartz": {"rhob": 2.65}}')
            config.mineral_endpoints = None
            return
        endpoints: Dict[str, Dict[str, float]] = {}
        for mineral, values in config.mineral_endpoints.items():
            label = str(mineral).strip().lower()
            if label not in _ENDPOINT_NAMES:
                errors.append(f"mineral_endpoints: unknown mineral {mineral!r}; use {', '.join(_ENDPOINT_NAMES)}")
                continue
            numbers = _number_mapping(values, f"mineral_endpoints[{label!r}]", _ENDPOINT_KEYS, errors,
                                      positive=("rhob", "pef", "u", "dt"))
            if numbers:
                endpoints[label] = numbers
        config.mineral_endpoints = endpoints


def _validate_input_tables(config: "DynamicWorkflowConfig", errors: List[str]) -> None:
    """Check that the lithology tables are tables or files that exist, and ``peer_wells`` a list of wells."""
    import os

    accepted = {
        "lithology_csv": "the path of a CSV or Excel file with top, bottom and lithology columns",
        "lithology_intervals": "a DataFrame, a list of rows, or the path of a CSV or Excel file, "
                               "with top, bottom and lithology columns",
        "lithology_codes": "a mapping of code to rock name, a DataFrame, or the path of a CSV or Excel codebook",
    }
    for name, description in accepted.items():
        value = getattr(config, name)
        if value is None:
            continue
        if isinstance(value, (str, os.PathLike)):
            if not Path(value).is_file():
                errors.append(f"{name}: file not found: {str(value)!r}; give {description}")
        elif isinstance(value, (bool, np.bool_)) or not isinstance(value, (pd.DataFrame, Mapping, list, tuple)):
            errors.append(f"{name} takes {description}; got {type(value).__name__} {value!r}")
    peers = config.peer_wells
    if peers is None:
        return
    expected = (
        "peer_wells takes a list of interpreted wells, e.g. [petrophys.interpret(peer).well]"
    )
    if isinstance(peers, Well):
        errors.append(f"{expected}; got one well (put it in a list)")
    elif not isinstance(peers, (list, tuple)):
        errors.append(f"{expected}; got {type(peers).__name__}")
    else:
        for index, peer in enumerate(peers):
            if peer is not None and not isinstance(peer, Well):
                hint = " (pass its .well)" if isinstance(getattr(peer, "well", None), Well) else ""
                errors.append(f"{expected}; item {index} is {type(peer).__name__}{hint}")


def _validate_workflow_config(config: "DynamicWorkflowConfig") -> None:
    """Validate and normalize a :class:`DynamicWorkflowConfig` in place.

    Raises
    ------
    ValueError
        Listing every invalid field, with the accepted values.
    """
    from petrophys.interpretation.clay_volume.gr_method import GR_VCL_METHODS, gr_method_name
    from petrophys.interpretation.permeability.perm_module import (
        PERMEABILITY_METHOD_ALIASES,
        PERMEABILITY_METHODS,
    )
    from petrophys.interpretation.saturation.sw_module import SwModel
    from petrophys.model.constants import MatrixProperties

    errors: List[str] = []
    _normalized_choice(config, "mud_type", _MUD_TYPES, errors)
    _normalized_choice(config, "mud_water", _MUD_WATERS, errors)
    if config.matrix_type is None:
        config.matrix_type = _default_matrix_type(config.lithology_domain)
        config._matrix_defaulted = True
    _normalized_choice(config, "matrix_type", ("auto", *MatrixProperties.names()), errors)
    _validate_lithology(config, errors)
    _normalized_choice(config, "lithology_source", _LITHOLOGY_SOURCES, errors)
    if config.lithology_curve is not None:
        _validate_lithology_curve(config, errors)
    _normalized_choice(config, "hc_type", _HC_TYPES, errors)
    _normalized_choice(config, "swirr_flag_phi_cl", _SWIRR_FLAG_PHI_CL, errors)
    _normalized_choice(config, "calculation_profile", _CALCULATION_PROFILES, errors)
    version = config.mineral_solver_version
    if isinstance(version, (bool, np.bool_)) or not isinstance(version, (int, np.integer))             or int(version) not in _MINERAL_SOLVER_VERSIONS:
        errors.append(f"mineral_solver_version must be 1 or 2; got {version!r}")
    else:
        config.mineral_solver_version = int(version)
    if config.carbonate_calibration is not None:
        _normalized_choice(config, "carbonate_calibration", _CARBONATE_CALIBRATIONS, errors)
    config.gr_method = gr_method_name(config.gr_method)
    _normalized_choice(config, "gr_method", GR_VCL_METHODS, errors)
    _normalized_choice(
        config, "perm_method", PERMEABILITY_METHODS, errors,
        aliases=tuple(PERMEABILITY_METHOD_ALIASES),
    )
    _normalized_choice(
        config, "perm_fallback_method",
        tuple(method for method in PERMEABILITY_METHODS if method != "auto"), errors,
        aliases=tuple(PERMEABILITY_METHOD_ALIASES),
    )
    _normalized_choice(config, "sw_model", ("auto", *(model.value for model in SwModel)), errors)

    step_names = tuple(step.value for step in WorkflowStep)
    if isinstance(config.skip_steps, str) or not isinstance(config.skip_steps, (list, tuple, set)):
        errors.append(
            f"skip_steps must be a list of step names; got {config.skip_steps!r}"
        )
    else:
        steps: List[str] = []
        for step in config.skip_steps:
            name = step.value if isinstance(step, WorkflowStep) else str(step).strip().lower()
            if name in step_names:
                steps.append(name)
            else:
                errors.append(_choice_error("skip_steps entry", step, step_names))
        config.skip_steps = steps

    for name in _BOOL_FIELDS:
        value = getattr(config, name)
        if value is None and name in _PROFILE_DEFAULTS:
            continue  # the calculation profile's default applies
        if not isinstance(value, (bool, np.bool_)):
            errors.append(f"{name} must be True or False; got {value!r}")

    def _number(name: str) -> Optional[float]:
        value = getattr(config, name)
        if value is None or isinstance(value, dict):
            return None  # a per-lithology value is checked by _per_lithology_values
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            errors.append(f"{name} must be a number; got {value!r}")
            return None
        if not np.isfinite(value):
            errors.append(f"{name} must be finite; got {value!r}")
            return None
        return float(value)

    _per_lithology_values(config, errors)
    for name in _FINITE_FIELDS:
        _number(name)
    for name in _POSITIVE_FIELDS:
        value = _number(name)
        if value is not None and value <= 0:
            errors.append(f"{name} must be greater than 0; got {value!r}")
    for name in _FRACTION_FIELDS:
        value = _number(name)
        if value is not None and 1.0 < value <= 100.0:
            # A fraction given in percent (swirr=15 for 0.15).
            value = value / 100.0
            setattr(config, name, value)
        if value is not None and not 0.0 <= value <= 1.0:
            errors.append(f"{name} must be a fraction between 0 and 1, or a percentage; got {value!r}")
    if config.gr_min is not None and config.gr_max is not None and config.gr_min >= config.gr_max:
        errors.append(f"gr_min ({config.gr_min}) must be less than gr_max ({config.gr_max})")
    if (
        config.calculation_profile == "current"
        and config.sp_min is not None
        and config.sp_min == config.sp_max
    ):
        errors.append(
            f"sp_min (shale baseline) and sp_max (clean-sand line) must differ; both are {config.sp_min}"
        )
    if config.nd_clay is not None:
        try:
            density, neutron = (float(value) for value in config.nd_clay)
        except (TypeError, ValueError):
            density = neutron = float("nan")
        if 1.0 <= density <= 3.5 and -0.15 <= neutron <= 1.0:
            config.nd_clay = (density, neutron)
        else:
            errors.append(
                "nd_clay must be (RHOB, NPHI) with RHOB 1.0-3.5 g/cm3 and NPHI -0.15-1.0 v/v; "
                f"got {config.nd_clay!r}"
            )
    if config.lithology_csv is not None and config.lithology_intervals is not None:
        errors.append("Supply lithology_csv or lithology_intervals, not both.")
    _validate_input_tables(config, errors)
    config.zones = _normalized_zones(config, errors)
    _validate_parameter_mappings(config, errors)
    if config.calculation_profile != "current" and config.gradient_f_per_100ft is not None:
        errors.append('gradient_f_per_100ft is not available under the "geolink" calculation profile')
    if config.calculation_profile == "current":
        cec_error = _cec_range_error(config.cec)
        if cec_error:
            errors.append(cec_error)
        errors.extend(_physical_range_errors(config))

    if errors:
        raise ValueError(
            "Invalid DynamicWorkflowConfig:\n  - " + "\n  - ".join(errors)
        )


@dataclass
class StepResult:
    """
    Result of a single workflow step with full trace metadata.

    Attributes
    ----------
    step : str
        Step name (e.g., "vclay", "porosity").
    success : bool
        Whether the step completed without error.
    message : str
        Why the step failed; empty when it succeeded (``decisions`` and
        ``parameters`` describe what it did).
    method : str
        Algorithm/equation used (e.g., "larionov_tertiary", "simandoux").
    input_curves : list
        Log curves consumed by this step.
    output_curves : list
        Curves added/modified on the well.
    parameters : dict
        All numeric parameters, thresholds, and constants used.
    decisions : list
        Human-readable trace of decisions/logic branches taken.
    data : dict, optional
        Extra step output for custom steps; the built-in steps leave it None.
    """
    step: str
    success: bool
    message: str = ""
    method: str = ""
    input_curves: List[str] = field(default_factory=list)
    output_curves: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    decisions: List[str] = field(default_factory=list)
    data: Optional[Dict[str, Any]] = None
    #: Under a benchmark profile, curves the step also wrote beyond the
    #: package's own ``output_curves`` list (which those packages' context
    #: records keep); under ``"current"`` they are in ``output_curves``.
    also_written: List[str] = field(default_factory=list)
    #: Curves the step removed from the well.
    curves_removed: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe representation of this workflow step trace."""
        payload = {
            "step": self.step,
            "success": bool(self.success),
            "method": self.method,
            "input_curves": list(self.input_curves),
            "output_curves": list(self.output_curves),
            "parameters": _json_safe_trace_value(self.parameters),
            "decisions": list(self.decisions),
            "message": self.message,
            "data": _json_safe_trace_value(self.data or {}),
        }
        if self.also_written:
            payload["also_written"] = list(self.also_written)
        if self.curves_removed:
            payload["curves_removed"] = list(self.curves_removed)
        return payload


def _curve_snapshot(well: "Well") -> Dict[str, np.ndarray]:
    """A copy of every curve's values, to tell which curves a step wrote."""
    return {name: np.array(curve.data, copy=True) for name, curve in well._curves.items()}


def _same_curve_values(old: np.ndarray, new: Any) -> bool:
    new = np.asarray(new)
    if old.shape != new.shape:
        return False
    try:
        return bool(np.array_equal(old, new, equal_nan=True))
    except TypeError:
        return bool(np.all(old == new))


def _account_step_curves(
    trace: "StepResult",
    before: Dict[str, np.ndarray],
    well: "Well",
    *,
    benchmark: bool,
    exact_outputs: bool = False,
) -> None:
    """Record on ``trace`` every curve its step created, changed or removed.

    Under ``"current"`` the curves the step wrote but did not list join
    ``output_curves``; with ``exact_outputs`` (the normalization step) the
    list becomes exactly the curves it created or changed. Under a benchmark
    profile ``output_curves`` keeps the package's own list and the others go
    to ``also_written``.
    """
    written = [
        name for name, curve in well._curves.items()
        if name not in before or not _same_curve_values(before[name], curve.data)
    ]
    trace.curves_removed = [name for name in before if name not in well._curves]
    if exact_outputs and not benchmark:
        trace.output_curves = written
        return
    listed = {str(curve).upper() for curve in trace.output_curves or []}
    unlisted = [name for name in written if name.upper() not in listed]
    if benchmark:
        trace.also_written = unlisted
    else:
        trace.output_curves = [*(trace.output_curves or []), *unlisted]


#: Warnings that count over the whole well, left out of a zone run's messages.
_WHOLE_WELL_TALLIES = ("Bad hole removed",)

#: Clay-volume aliases the workflow writes next to their standard names.
_CLAY_ALIASES = {"VCLAY": "VCL", "VCLGR": "VCL_GR", "VCLSP": "VCL_SP", "VCLND": "VCL_ND"}


def _curves_without_aliases(well: Well) -> List[str]:
    """Curve names of ``well`` less the clay-volume aliases that copy a standard curve."""
    names = []
    for name in well.curves:
        standard = well.get(_CLAY_ALIASES.get(name, "")) if name in _CLAY_ALIASES else None
        alias = well.get(name)
        if (
            standard is not None and alias is not None
            and np.array_equal(np.asarray(standard.data), np.asarray(alias.data), equal_nan=True)
        ):
            continue
        names.append(name)
    return names


@dataclass
class DynamicWorkflowResult:
    """Results from dynamic workflow execution."""
    #: The interpreted well: the input curves and the calculated ones.
    well: "Well"
    #: The settings the run used.
    config: DynamicWorkflowConfig
    #: The record of each step that ran, in run order; a step that failed is
    #: included with ``success`` False and the reason in ``message``.
    steps_executed: List[StepResult] = field(default_factory=list)
    #: Names of the steps that did not run.
    steps_skipped: List[str] = field(default_factory=list)
    #: Why each skipped step did not run, by step name.
    specific_steps_skipped: dict[str, str] = field(default_factory=dict)
    #: What made the run fail: an input the workflow cannot use, or a failed
    #: step as ``"<step>: <reason>"``. Empty for a run that succeeded.
    errors: List[str] = field(default_factory=list)
    #: Assumed values and data-quality findings to check.
    warnings: List[str] = field(default_factory=list)
    #: The lithology domain the run interpreted with, e.g. ``"clastic"`` or
    #: ``"carbonate"``.
    lithology_domain: str = "auto"

    # Computed values

    #: Formation water resistivity used (ohm.m): one value, or a value per
    #: depth sample at formation temperature. None when no Rw was computed.
    rw_computed: Optional[Union[float, np.ndarray]] = None
    #: Mud filtrate resistivity used (ohm.m), as ``rw_computed``; None when
    #: no Rmf was computed.
    rmf_computed: Optional[Union[float, np.ndarray]] = None
    #: Clay resistivity used (ohm.m); None when none was computed.
    rcl_computed: Optional[float] = None
    #: Temperature gradient of the temperature step (degF per 100 ft); None
    #: when the step did not run.
    temperature_gradient: Optional[float] = None
    # The eval package's calculations a "geolink" run gave the well, "main_run"
    # or "rerun" (None under "current"); the trace reports them.
    _benchmark_calculations: Optional[str] = field(
        default=None, init=False, repr=False, compare=False
    )
    #: The context profile whose context interpretation this is
    #: (``petrophys.interpret(..., context_profile=...)``), else None.
    context_profile: Optional[str] = None
    #: Trace entries of the context profile's preparatory step.
    preparatory_steps: List[Dict[str, Any]] = field(default_factory=list)
    #: Trace entries of the context's value-range QC and dependent-curve
    #: recomputation, applied to ``well`` after the workflow.
    post_workflow_steps: List[Dict[str, Any]] = field(default_factory=list)
    #: One run per distinct set of zone settings (``config.zones`` and LAS
    #: header zones), each merged into ``well`` over its depths; empty when
    #: no zone sets anything (a zone that only names an interval adds no run).
    zone_results: List["DynamicWorkflowResult"] = field(default_factory=list)
    #: For a run in ``zone_results``: the names of the zones it computed.
    zone_names: List[str] = field(default_factory=list)
    #: The zones of ``config.zones`` that apply to this well, in depth order:
    #: ``name``, ``top`` and ``bottom`` in the well's depth unit.
    zone_intervals: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def has_usable_interpretation(self) -> bool:
        """Return whether a successful interpretive step produced a curve."""
        non_interpretive_steps = {
            WorkflowStep.NORMALIZATION.value,
            WorkflowStep.BAD_HOLE.value,
            WorkflowStep.TEMPERATURE.value,
            WorkflowStep.RMF.value,
            WorkflowStep.RW.value,
            "input_context",
        }
        return any(
            step.success and bool(step.output_curves)
            for step in self.steps_executed
            if step.step not in non_interpretive_steps
        )

    @property
    def status(self) -> str:
        """Return the execution status.

        ``"failed"`` when the run has errors or a failed step,
        ``"interpreted"`` when it succeeded and an interpretive step wrote a
        curve, and ``"no_interpretation"`` when it succeeded without one.
        """
        if not self.success:
            return "failed"
        if self.has_usable_interpretation:
            return "interpreted"
        return "no_interpretation"

    @property
    def success(self) -> bool:
        """Return True when ``errors`` is empty and no executed step failed."""
        # Success: no hard errors AND no executed step reported failure. Step
        # failures are recorded on the steps, not in `errors`.
        step_failed = any(
            getattr(step, "success", True) is False
            for step in self.steps_executed
        )
        return len(self.errors) == 0 and not step_failed

    def raise_for_errors(self) -> "DynamicWorkflowResult":
        """Raise RuntimeError when execution failed; return this result for chaining."""
        if not self.success:
            details = self.errors or [
                step.message or step.step
                for step in self.steps_executed if not step.success
            ]
            raise RuntimeError("Workflow failed: " + "; ".join(details))
        return self

    def plot(self, **options):
        """Plot the interpretation: ``petrophys.plot(result, **options)``, with its options."""
        from petrophys.api import plot

        return plot(self, **options)

    def print_trace(self) -> None:
        """Print a comprehensive human-readable trace of all steps."""
        print(f"\n{'=' * 90}")
        print(f"WORKFLOW TRACE: {self.well.name or 'Unknown Well'}")
        print(f"{'=' * 90}")
        print(f"Overall: {'SUCCESS' if self.success else 'FAILED'} | "
              f"Executed: {len(self.steps_executed)} | Skipped: {len(self.steps_skipped)}")
        print(f"Lithology domain: {self.lithology_domain}")
        if self.rw_computed is not None:
            def _summary(value):
                arr = np.asarray(value, dtype=float)
                finite = arr[np.isfinite(arr)]
                if finite.size == 0:
                    return "NaN"
                if finite.size == 1:
                    return f"{finite[0]:.4f}"
                return f"{np.nanmin(finite):.4f}-{np.nanmax(finite):.4f}"
            print(f"Rw = {_summary(self.rw_computed)} ohm.m | Rmf = {_summary(self.rmf_computed) if self.rmf_computed is not None else 'None'} | "
                  f"Rcl = {_summary(self.rcl_computed) if self.rcl_computed is not None else 'None'}")
        print()

        for sr in self.steps_executed:
            status = "OK" if sr.success else "FAIL"
            print(f"[{status:4s}] {sr.step}")
            if sr.method:
                print(f"       Method: {sr.method}")
            if sr.input_curves:
                print(f"       Inputs: {', '.join(sr.input_curves)}")
            if sr.output_curves:
                print(f"       Outputs: {', '.join(sr.output_curves)}")
            if sr.parameters:
                for k, v in sr.parameters.items():
                    if isinstance(v, float):
                        print(f"       {k}: {v:.6g}")
                    else:
                        print(f"       {k}: {v}")
            for d in sr.decisions:
                print(f"       -> {d}")
            if sr.message and not sr.success:
                print(f"       ERROR: {sr.message}")
            print()

        if self.steps_skipped:
            print("Skipped:")
            for name in self.steps_skipped:
                reason = self.specific_steps_skipped.get(name)
                print(f"  - {name}: {reason}" if reason else f"  - {name}")
        if self.warnings:
            print(f"\nWarnings:")
            for w in self.warnings:
                print(f"  - {w}")
        print(f"{'=' * 90}")

    def to_dataframe(self):
        """Return the step trace as a DataFrame, one row per executed step.

        The columns are ``Step``, ``Status`` (``OK`` or ``FAIL``), ``Method``,
        ``Input Curves``, ``Output Curves``, ``Parameters``, ``Decisions``
        and ``Message``. For the interpreted log curves use
        ``result.well.to_dataframe()``.
        """
        import pandas as pd
        rows = []
        for sr in self.steps_executed:
            rows.append({
                "Step": sr.step,
                "Status": "OK" if sr.success else "FAIL",
                "Method": sr.method,
                "Input Curves": ", ".join(sr.input_curves),
                "Output Curves": ", ".join(sr.output_curves),
                "Parameters": str(sr.parameters) if sr.parameters else "",
                "Decisions": " | ".join(sr.decisions),
                "Message": sr.message,
            })
        return pd.DataFrame(rows)

    def to_trace_dict(
        self,
        include_curves: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Export the full workflow route as a JSON-safe dictionary.

        The trace is recorded at the workflow module boundary, so each step
        shows why that module ran, which method was selected, which curves were
        consumed, which curves were written, and the parameters used.
        """
        try:
            well_name = self.well.name
        except Exception:
            header = getattr(self.well, "header", None)
            well_name = getattr(header, "well_name", "") if header is not None else ""
            if not well_name and isinstance(header, dict):
                well_name = header.get("WELL") or header.get("well_name") or ""

        well_summary = {
            "name": well_name or "Unknown Well",
            "depth_unit": getattr(self.well, "depth_unit", ""),
            "top": _json_safe_trace_value(getattr(self.well, "top", None)),
            "bottom": _json_safe_trace_value(getattr(self.well, "bottom", None)),
            "step": _json_safe_trace_value(getattr(self.well, "step", None)),
            "n_samples": int(getattr(self.well, "n_samples", 0) or 0),
            "curve_count": int(getattr(self.well, "n_curves", 0) or 0),
            "available_curves": sorted(str(curve) for curve in getattr(self.well, "curves", [])),
        }

        from petrophys.version import __version__

        return {
            "schema": "dynamic_workflow_trace",
            "schema_version": 1,
            "petrophys_version": __version__,
            "well": well_summary,
            "success": bool(self.success),
            "status": self.status,
            "has_usable_interpretation": self.has_usable_interpretation,
            "lithology_domain": self.lithology_domain,
            **({"context_profile": self.context_profile} if self.context_profile else {}),
            "executed_steps": [step.step for step in self.steps_executed],
            "skipped_steps": list(self.steps_skipped),
            "specific_steps_skipped": _json_safe_trace_value(self.specific_steps_skipped),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "computed": {
                "rw": _json_safe_trace_value(self.rw_computed),
                "rmf": _json_safe_trace_value(self.rmf_computed),
                "rcl": _json_safe_trace_value(self.rcl_computed),
                "temperature_gradient": _json_safe_trace_value(self.temperature_gradient),
            },
            "config": _json_safe_trace_value(self._trace_settings()),
            "steps": [step.to_dict() for step in self.steps_executed],
            "curve_summaries": _curve_summaries_for_trace(self.well, include_curves),
        }

    def group_records(self, **options: Any) -> List[Dict[str, Any]]:
        """Each formation and lithology group's record, with its own routing notes, as its context gives it.

        See :func:`petrophys.reporting.context_profiles.interpretation_group_records`
        for the options (``tops_file`` and ``lithology_mapping_file`` for
        context-profile results, ``minimum_group_thickness``, ``documents``).
        """
        from petrophys.reporting.context_profiles import interpretation_group_records

        return interpretation_group_records(self, **options)

    def _trace_settings(self) -> Dict[str, Any]:
        """The run's settings as the trace reports them.

        Settings left unset show the default of the run's calculation profile.
        A benchmark run adds ``benchmark_calculations`` after
        ``calculation_profile``: the eval package's calculations the well got.
        """
        settings: Dict[str, Any] = {}
        for name in vars(self.config):
            if name.startswith("_"):
                continue  # the library's own records, not settings
            settings[name] = _profile_setting(self.config, name)
            if name == "calculation_profile" and self._benchmark_calculations is not None:
                settings["benchmark_calculations"] = self._benchmark_calculations
        return settings

    def to_json(
        self,
        path: Optional[Union[str, Path]] = None,
        include_curves: Optional[List[str]] = None,
        indent: int = 2,
    ) -> Union[str, Path]:
        """
        Export the workflow route as JSON.

        If ``path`` is provided, the JSON file is written and the output path is
        returned. Otherwise the JSON string is returned.
        """
        text = json.dumps(
            self.to_trace_dict(include_curves=include_curves),
            indent=indent,
            sort_keys=False,
            allow_nan=False,
        )
        if path is None:
            return text

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        return output_path

    def formation_summary(self, formations: Any = None, *, rank: Optional[str] = None) -> Any:
        """Gross, net and average properties for each formation.

        Parameters
        ----------
        formations : path, DataFrame, mapping or sequence, optional
            Formation tops or intervals (anything
            :func:`petrophys.read_formations` accepts). Defaults to the
            formations attached with ``petrophys.read(path, formations=...)``.
        rank : str, optional
            Level to summarize when the tops mix levels, e.g. ``"group"``;
            tops that mix levels default to ``"formation"``. NPD-style names
            (``"Brent Gp."``, ``"Etive Fm."``) carry their rank; other tables
            need a ``rank`` column to mix levels. Tops at or below the logged
            bottom are left out.

        Returns
        -------
        pandas.DataFrame
            One row per formation: top, bottom, gross, net rock, net
            reservoir and net pay thickness (in the well's depth unit),
            net reservoir / gross ("N/G (reservoir)") and net pay / gross
            ("Pay/Gross"), and PHIE, Sw, VCL and
            permeability averaged over the net-pay samples only: porosity
            and clay volume weighted by thickness, Sw by pore volume
            (porosity x thickness), permeability a geometric mean (NaN
            where a formation has no pay).
            ``DataFrame.attrs["curves"]`` names the curve each pay average is
            taken from (PHIE_CC and PERM_CC where the interpretation produced
            or used them, PHIE and PERM at the samples those do not
            cover). ``Avg GR``, ``Avg RHOB``, ``Avg NPHI``,
            ``Avg RDEP`` and ``Avg PHIT`` are means over the whole formation.
            ``HCPV`` is the hydrocarbon pore thickness, the sum of
            PHIE x (1 - Sw) x thickness over net pay (depth unit), and
            ``BVW`` the thickness-weighted mean of PHIE x Sw over net pay
            (v/v). ``Samples`` and ``Pay Samples`` count the depth samples,
            and ``Data Coverage %`` is the share of samples with RHOB.

        Raises
        ------
        ValueError
            If no formations are given or attached to the well.

        Examples
        --------
        >>> import petrophys as pp
        >>> well = pp.read(pp.sample("30_6-5"), formations=pp.sample("formations_30_6-5"))
        >>> result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230)
        >>> table = result.formation_summary()  # doctest: +SKIP
        """
        from petrophys.workflow.summary import summarize_formations

        return summarize_formations(self, formations, rank=rank)

    def zone_summary(self, zones: Any = None) -> Any:
        """Gross, net and average properties for each zone.

        Parameters
        ----------
        zones : path, DataFrame or sequence of mappings, optional
            Zones as :attr:`DynamicWorkflowConfig.zones` takes them: one row
            per zone with its ``name`` (or ``zone``), ``top`` and ``bottom``,
            in a CSV or Excel file, a table or a list. A ``well`` column
            selects the rows of this well. Defaults to the zones passed to
            ``petrophys.interpret(well, zones=...)``.

        Returns
        -------
        pandas.DataFrame
            One row per zone, with the columns of :meth:`formation_summary`;
            the first column is ``Zone``.

        Raises
        ------
        ValueError
            If no zones are given and the interpretation had none.

        Examples
        --------
        >>> import petrophys as pp
        >>> zones = [{"zone": "Upper", "top": 2841, "bottom": 2900},
        ...          {"zone": "Lower", "top": 2900, "bottom": 2958}]
        >>> result = pp.interpret(pp.read(pp.sample("30_6-5")), rw=0.07, rw_meas_temp_f=230, zones=zones)
        >>> list(result.zone_summary()["Zone"])  # doctest: +SKIP
        ['Upper', 'Lower']
        """
        from petrophys.workflow.summary import summarize_zones

        return summarize_zones(self, zones)

    def summary(
        self, top: Optional[float] = None, bottom: Optional[float] = None
    ) -> Dict[str, Any]:
        """Summarize the interpretation as compact, JSON-serializable data.

        Designed for reports and LLM agents: everything needed to judge the
        result without walking the full trace.

        Parameters
        ----------
        top, bottom : float, optional
            Depth window (well depth unit) for ``key_results`` and
            ``net_thickness``; defaults to the whole well.

        Returns
        -------
        dict
            ``petrophys_version``, ``well`` (its name), ``status``,
            ``lithology_domain``, ``resolved`` matrix and lithology,
            ``settings_used`` (non-default settings), ``summary_interval``
            (``top``, ``bottom`` and ``depth_unit``), ``key_results``
            (statistics of VCL, PHIT, PHIE, SW and PERM, and of PHIE_CC and
            PERM_CC where the pay averages use them), ``net_thickness``
            (``net_rock``, ``net_res`` and ``net_pay`` in ``depth_unit``),
            ``executed_steps``, ``skipped_steps`` with reasons, ``warnings``
            and ``errors``.
            Use :meth:`to_trace_dict` for every step's full record.

        Examples
        --------
        >>> import petrophys as pp
        >>> result = pp.interpret(pp.read(pp.sample("30_6-5")), rw=0.07, rw_meas_temp_f=230)
        >>> result.summary(top=2841, bottom=2958)["status"]
        'interpreted'
        """
        from petrophys.workflow.summary import summarize_result

        return summarize_result(self, top, bottom)

    def step(self, name: str) -> "StepResult":
        """The record of one executed step: its ``method``, ``decisions``,
        ``parameters``, ``input_curves`` and ``output_curves``.

        A step that did not run raises KeyError, with the reason it was
        skipped and the names of the steps that ran.

        >>> result.step("rw").decisions  # doctest: +SKIP
        """
        for step in self.steps_executed:
            if step.step == name:
                return step
        skipped = self.specific_steps_skipped.get(name)
        ran = ", ".join(step.step for step in self.steps_executed)
        raise KeyError(
            f"Step {name!r} did not run" + (f": {skipped}" if skipped else "") + f". Steps run: {ran}"
        )

    def statistics(
        self,
        curves: Optional[Sequence[str]] = None,
        top: Optional[float] = None,
        bottom: Optional[float] = None,
    ) -> Any:
        """Statistics of curves of the interpreted well over a depth window.

        Parameters
        ----------
        curves : str or sequence of str, optional
            Curves to describe (the well's names); default VCL, PHIT, PHIE,
            SW and PERM where present.
        top, bottom : float, optional
            Depth window in the well's depth unit; default the whole well.

        Returns
        -------
        pandas.DataFrame
            A row per curve: ``unit``, ``samples`` (finite), ``mean``,
            ``p10``, ``median`` and ``p90``.

        Raises
        ------
        KeyError
            If a named curve is not in the well.
        """
        from petrophys.workflow.summary import statistics_table

        return statistics_table(self.well, curves, top, bottom, default=("VCL", "PHIT", "PHIE", "SW", "PERM"))

    def provenance_parameters(self) -> List[Dict[str, Any]]:
        """Summarize the settings this interpretation used, as LAS parameters.

        Returns
        -------
        list of dict
            ``PP_*`` entries (``mnemonic``, ``value``, ``unit``, ``descr``):
            package version, status, resolved matrix and lithology domain,
            Rw input and its temperature-corrected range, Sw model and Archie
            exponents.
        """
        from petrophys.version import __version__

        steps = {step.step: step for step in self.steps_executed if step.success}
        items: List[Dict[str, Any]] = [
            {"mnemonic": "PP_VERSION", "value": __version__, "descr": "petrophys version"},
            {"mnemonic": "PP_STATUS", "value": self.status, "descr": "Interpretation status"},
            {"mnemonic": "PP_DOMAIN", "value": self.lithology_domain,
             "descr": "Lithology interpretation domain"},
            {"mnemonic": "PP_MATRIX", "value": self.config.matrix_type, "descr": "Matrix used"},
            {"mnemonic": "PP_LITH", "value": self.config.lithology_domain, "descr": "Lithology family used"},
        ]
        rw_step = steps.get(WorkflowStep.RW.value)
        if rw_step is not None:
            parameters = rw_step.parameters
            # Temperature-corrected runs record the input as rw_at_measured_temp.
            rw_input = parameters.get("rw", parameters.get("rw_at_measured_temp"))
            if rw_input is not None:
                reference = parameters.get("reference_temp_f", parameters.get("measured_temp_f"))
                descr = f"Rw input ({rw_step.method})"
                if reference is not None:
                    descr = f"Rw input at {reference:g} degF ({rw_step.method})"
                items.append({"mnemonic": "PP_RW", "value": rw_input, "unit": "ohm.m", "descr": descr})
            for key, mnemonic, descr in (
                ("rw_min", "PP_RWTMIN", "Minimum Rw at formation temperature"),
                ("rw_max", "PP_RWTMAX", "Maximum Rw at formation temperature"),
            ):
                if parameters.get(key) is not None:
                    items.append({"mnemonic": mnemonic, "value": float(f"{parameters[key]:.4g}"),
                                  "unit": "ohm.m", "descr": descr})
        sw_step = steps.get(WorkflowStep.SATURATION.value)
        if sw_step is not None:
            parameters = sw_step.parameters
            items.append({"mnemonic": "PP_SWMODEL", "value": sw_step.method, "descr": "Water saturation model(s)"})
            for key in ("a", "m", "n"):
                value = parameters.get(key)
                if value is None:
                    continue
                if isinstance(value, (int, float)):
                    items.append({"mnemonic": f"PP_{key.upper()}", "value": value,
                                  "descr": f"Archie {key}"})
                else:
                    items.append({"mnemonic": f"PP_{key.upper()}", "value": "per lithology",
                                  "descr": f"Archie {key} from the lithology table"})
        return items

    def to_las(self, path: Union[str, Path], **options: Any) -> Path:
        """Write the interpreted well to LAS with the settings used.

        The ``~Params`` section keeps the source file's header values and adds
        :meth:`provenance_parameters` (``PP_*``); ``~Other`` lists executed and
        skipped steps and, under the ``"current"`` calculation profile, the
        run's warnings. Use :meth:`to_json` for the complete trace.

        The clay-volume aliases the workflow keeps for other readers
        (``VCLAY``, ``VCLGR``, ``VCLSP``, ``VCLND``, copies of ``VCL``,
        ``VCL_GR``, ``VCL_SP``, ``VCL_ND``) are left out unless ``curves``
        names them.

        Parameters
        ----------
        path : str or Path
            Output LAS file.
        **options
            Passed to :func:`petrophys.io.export_las` (e.g. ``curves=[...]``).

        Returns
        -------
        Path
            The written file.
        """
        from petrophys.io.export import export_las
        from petrophys.version import __version__

        lines = [f"Interpreted with petrophys {__version__}; status {self.status}."]
        lines.append("Executed steps: " + ", ".join(step.step for step in self.steps_executed))
        for name in self.steps_skipped:
            reason = self.specific_steps_skipped.get(name)
            lines.append(f"Skipped {name}" + (f": {reason}" if reason else ""))
        if self.config.calculation_profile not in _BENCHMARK_PROFILES:
            lines.extend(f"Warning: {warning}" for warning in self.warnings)
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if options.get("curves") is None:
            options["curves"] = _curves_without_aliases(self.well)
        export_las(
            self.well,
            output,
            extra_parameters=self.provenance_parameters(),
            other="\n".join(lines),
            **options,
        )
        return output

    def to_markdown(
        self,
        path: Optional[Union[str, Path]] = None,
        include_curves: Optional[List[str]] = None,
    ) -> Union[str, Path]:
        """
        Export the workflow route as a human-readable Markdown report.

        This mirrors ``to_json`` but is easier to scan when checking per-module
        routing decisions for a single well.
        """
        trace = self.to_trace_dict(include_curves=include_curves)
        well = trace["well"]
        lines = [
            "# Dynamic Workflow Trace",
            "",
            f"- Well: `{well['name']}`",
            f"- Depth: `{well['top']} - {well['bottom']} {well['depth_unit']}`",
            f"- Samples: `{well['n_samples']}`",
            f"- Overall success: `{trace['success']}`",
            f"- Executed steps: `{', '.join(trace['executed_steps']) or 'none'}`",
            f"- Skipped steps: `{', '.join(trace['skipped_steps']) or 'none'}`",
        ]

        if trace["warnings"]:
            lines.append(f"- Warnings: `{' ; '.join(trace['warnings'])}`")
        if trace["errors"]:
            lines.append(f"- Errors: `{' ; '.join(trace['errors'])}`")

        computed = trace["computed"]
        lines.extend([
            "",
            "## Computed Values",
            f"- Rw: `{json.dumps(computed['rw'])}`",
            f"- Rmf: `{json.dumps(computed['rmf'])}`",
            f"- Rcl: `{format(computed['rcl'], '.4g') if isinstance(computed['rcl'], (int, float)) else json.dumps(computed['rcl'])}`",
            f"- Temperature gradient: `{json.dumps(computed['temperature_gradient'])}`",
        ])

        for step in trace["steps"]:
            lines.extend([
                "",
                f"## {step['step']}",
                f"- Success: `{step['success']}`",
                f"- Method: `{step['method'] or 'none'}`",
                f"- Inputs: `{', '.join(step['input_curves']) or 'none'}`",
                f"- Outputs: `{', '.join(step['output_curves']) or 'none'}`",
                f"- Parameters: `{json.dumps(step['parameters'], sort_keys=True)}`",
            ])
            if step["decisions"]:
                lines.append("- Decisions:")
                lines.extend(f"  - {decision}" for decision in step["decisions"])
            if step["message"]:
                lines.append(f"- Message: `{step['message']}`")
            if step["data"]:
                lines.append(f"- Data: `{json.dumps(step['data'], sort_keys=True)}`")

        if trace["curve_summaries"]:
            lines.extend(["", "## Curve Summaries"])
            for name, summary in trace["curve_summaries"].items():
                lines.append(f"- {name}: `{json.dumps(summary, sort_keys=True)}`")

        text = "\n".join(lines) + "\n"
        if path is None:
            return text

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        return output_path


class DynamicWorkflow:
    """
    Dynamic petrophysics workflow that auto-detects available inputs.

    This workflow inspects the well's available curves and the provided
    configuration to determine which processing steps can be executed.
    It runs modules in the correct pipeline order and passes results
    between dependent steps.

    Examples
    --------
    >>> config = DynamicWorkflowConfig(
    ...     bht_f=220, td_ft=8500, rw=0.05,
    ...     lithology_domain='clastic', salinity_ppm=25000
    ... )
    >>> workflow = DynamicWorkflow(config)
    >>> result = workflow.run(well)
    >>> print(result.steps_executed)
    """

    def __init__(self, config: Optional[DynamicWorkflowConfig] = None):
        self.config = config or DynamicWorkflowConfig()
        self._result: Optional[DynamicWorkflowResult] = None
        self._lithology_log: Optional[np.ndarray] = None
        self._lithology_source: str = "config"
        self._computed_lithology_written = False
        self._last_phit_nd_average_mask: Optional[np.ndarray] = None
        self._last_phi_cl_info: Optional[Dict[str, Any]] = None
        self._auto_matrix = False
        self._auto_lith = False
        self._costa_fluid = False
        self._lithology_pending = False
        self._lithology_resolution = None
        self._selected_lithology_source = "auto"
        self._external_lithology = None
        # The run's lithology code curve and the well it was found on
        # (_lithology_code_curve).
        self._code_curve: Optional[str] = None
        self._code_curve_well: Optional["Well"] = None
        # Whether the run gives the well the eval package's re-run calculations,
        # and the published dataset whose calculations it keeps, if any;
        # resolved per run by _start_result.
        self._rerun_selected = False
        self._dataset: Optional[str] = None
        # Input PHIE_CC/PERM_CC that every step of the run reads as they
        # stand; decided once per run (_input_core_curves_kept).
        self._input_core_basis: Tuple[str, ...] = ()

    def _start_result(self, well: "Well") -> DynamicWorkflowResult:
        """Open the result of a run on ``well``, resolving its benchmark calculations.

        Under the "geolink" profile the well gets the calculations of the eval
        package's re-run or of its main run (:func:`_uses_rerun_calculations`),
        once for the whole run; the result records which for the trace.
        """
        self._rerun_selected = _uses_rerun_calculations(well)
        self._dataset = (
            _DATASET_CALCULATIONS_REQUESTED.get() if self._reproducing_benchmark else None
        )
        result = DynamicWorkflowResult(well=well, config=self.config)
        if self._reproducing_benchmark:
            result._benchmark_calculations = self._dataset or (
                "rerun" if self._rerun_selected else "main_run"
            )
        return result

    def _prepare_input_context(self, well: "Well") -> None:
        from petrophys.io import read_lithology_intervals
        from petrophys.interpretation.lithology.matrix_inference import (
            CODE_CURVE_RECORD,
            resolve_from_lithology_log,
        )

        policy = str(self.config.lithology_source).strip().lower()
        if policy not in {"auto", "external", "vendor", "computed", "config"}:
            raise ValueError("lithology_source must be auto, external, vendor, computed, or config")
        if self.config.lithology_csv is not None and self.config.lithology_intervals is not None:
            raise ValueError("Supply lithology_csv or lithology_intervals, not both.")
        context = dict(well.get_parameter("_geology_context") or {})
        if CODE_CURVE_RECORD in context:
            # Each run records the code curve it reads; an earlier run's
            # record must not steer this one.
            del context[CODE_CURVE_RECORD]
            well.set_parameter("_geology_context", dict(context))
        self._code_curve_well = None
        self._check_lithology_curve(well)
        code_curve = self._lithology_code_curve(well)
        source = self.config.lithology_intervals
        if source is None:
            source = self.config.lithology_csv
        if source is None:
            source = context.get("lithology")
        if self._eval_lithology_detection:
            vendor_available = self._has_curve(well, "LITHOLOGY_GEOLINK") or resolve_from_lithology_log(well, warn=False) is not None
        else:
            vendor_available = code_curve is not None
        selected = policy
        if policy == "auto":
            selected = "external" if source is not None else ("vendor" if vendor_available else "computed")
        if selected == "external" and source is None:
            raise ValueError("lithology_source='external' requires lithology_intervals or lithology_csv.")
        if selected == "vendor" and not vendor_available:
            if self._eval_lithology_detection:
                raise ValueError("lithology_source='vendor' requires a recognized vendor lithology curve.")
            raise ValueError(
                "lithology_source='vendor' requires a recognized vendor lithology curve: "
                "LITHOLOGY_GEOLINK, a lithology-code mnemonic such as LITHOLOGY or FACIES "
                "that holds lithology codes, or the curve named by lithology_curve."
            )
        if selected == "config" and (
            self.config.matrix_type == "auto" or self.config.lithology_domain == "auto"
        ):
            raise ValueError("lithology_source='config' requires explicit matrix_type and lithology.")
        self._selected_lithology_source = selected
        self._external_lithology = None
        report = {"requested_source": policy, "selected_source": selected,
                  "external_available": source is not None, "vendor_available": vendor_available}
        if not self._eval_lithology_detection:
            report["vendor_curve"] = code_curve
            context[CODE_CURVE_RECORD] = code_curve
            self._warn_code_curve(code_curve)
        output_curves = []
        if selected == "external":
            table = read_lithology_intervals(source, well=well)
            self._external_lithology = table
            labels = list(dict.fromkeys(table["lithology"]))
            codes = np.full(well.depth.shape, np.nan)
            groups = np.full(well.depth.shape, None, dtype=object)
            from petrophys.interpretation.lithology.labels import lithology_group_for_label

            unknown = set()
            category_groups = {}
            for row in table.itertuples():
                # The rock the name describes ("Sst", "shaly sandstone", "LST", ...).
                group = lithology_group_for_label(row.lithology, known=self._group_to_matrix)
                if group is None:
                    unknown.add(row.lithology)
                mask = (well.depth >= row.top) & (well.depth < row.bottom)
                if row.bottom == table["bottom"].max():
                    mask |= well.depth == row.bottom
                code = labels.index(row.lithology) + 1
                codes[mask] = code
                groups[mask] = group
                category_groups[code] = group
            well.add_curve("LITHO_CLASS", codes, "", "External lithology categories")
            well["LITHO_CLASS"].metadata.custom.update({
                "category_labels": {index + 1: label for index, label in enumerate(labels)},
                "category_groups": category_groups,
                "source": "external", "code_scheme": "local_categories",
            })
            self._lithology_log = groups
            self._lithology_source = "external_lithology"
            context["lithology"] = table.to_dict("records")
            report.update({"import_report": table.attrs.get("import_report", {}),
                           "unmapped_labels": sorted(unknown),
                           "covered_samples": int(np.isfinite(codes).sum()),
                           "uncovered_samples": int(np.isnan(codes).sum())})
            output_curves = ["LITHO_CLASS"]
            if any(str(label).strip().lower() in {"marl", "marlstone"} for label in labels):
                self._result.warnings.append(
                    "Marl uses the limestone matrix and carbonate cutoff family. "
                    "This is a routing convention, not a calibrated marl model; "
                    "check clay correction and supply project-specific cutoffs."
                )
            if unknown or np.isnan(codes).any():
                self._result.warnings.append(
                    f"External lithology: unmapped labels {sorted(unknown)}, "
                    f"{np.isnan(codes).sum()} uncovered samples. Labels/gaps are preserved; "
                    "unclassified samples use configured calculation fallbacks, not inferred rock labels."
                )
        decisions = [f"Selected {selected} lithology; alternate sources were retained"]
        decisions.extend(self._select_formation_tops(well, context, report))
        context["selected_lithology_source"] = selected
        context["lithology_selection"] = report
        well.set_parameter("_geology_context", context)
        self._result.steps_executed.append(StepResult(
            step="input_context", success=True, method="validated_geology_selection",
            output_curves=output_curves, parameters=report,
            decisions=decisions,
        ))

    @property
    def _eval_lithology_detection(self) -> bool:
        """True when vendor lithology is found as the eval package's calculations found it.

        That is the "geolink" profile without ``lithology_curve``: the vendor
        lithology and its matrix come from any curve named as a lithology code
        log, the per-depth lithology groups from ``LITHOLOGY_GEOLINK`` alone.
        Otherwise one curve, :meth:`_lithology_code_curve`, serves all three.
        """
        return self._reproducing_benchmark and self.config.lithology_curve is None

    def _lithology_code_curve(self, well: "Well") -> Optional[str]:
        """Mnemonic of the run's lithology code curve, or None; found once per run.

        It is ``lithology_curve``, else ``LITHOLOGY_GEOLINK``, else (under
        ``"current"``, with ``lithology_source`` ``"auto"`` or ``"vendor"``) a
        curve of another lithology-code mnemonic that holds lithology codes
        (:func:`~petrophys.interpretation.lithology.matrix_inference.lithology_code_curve`).
        """
        if self._code_curve_well is not well:
            from petrophys.interpretation.lithology.matrix_inference import lithology_code_curve

            user_groups = self._user_code_groups()
            found = lithology_code_curve(
                well,
                explicit=self.config.lithology_curve,
                allow_aliases=(
                    not self._reproducing_benchmark
                    and self.config.lithology_source in {"auto", "vendor"}
                ),
                known_codes=(
                    None if user_groups is None
                    else {code for code in user_groups if isinstance(code, int)}
                ),
            )
            self._code_curve = found[0] if found else None
            self._code_curve_well = well
        return self._code_curve

    def _unused_setting_notes(self, well: "Well") -> List[str]:
        """Warnings for settings that are given but that this run cannot use (``"current"``).

        Waxman-Smits needs ``cec`` and ``b_coeff`` together; ``rw_meas_temp_f``
        is the temperature of ``rw`` or of the LAS header RW;
        ``rename_input_curves`` and a curve-name mapping act in the
        standardization step; peer wells calibrate core only with
        ``use_peer_core`` (:meth:`_core_peer_wells` reports each peer).
        """
        if self._historical:
            return []
        from petrophys.preprocessing.standardization.user_aliases import curve_aliases_of

        config = self.config
        zones = [zone.get("settings") or {} for zone in (config.zones or ())]

        def given(name: str) -> bool:
            return getattr(config, name) is not None or any(zone.get(name) is not None for zone in zones)

        notes = []
        for name, other in (("cec", "b_coeff"), ("b_coeff", "cec")):
            if given(name) and not given(other):
                notes.append(
                    f"{name} is set without {other}: Waxman-Smits needs both (neither is estimated), so "
                    f"{name} is not used and saturation is routed as without it."
                )
        header_rw = any(
            isinstance(key, str) and re.fullmatch(r"RW(_Z\d+)?", key.upper())
            for key in (getattr(well, "_parameters", None) or {})
        )
        if config.rw_meas_temp_f is not None and not given("rw") and (not header_rw or given("salinity_ppm")):
            notes.append(
                f"rw_meas_temp_f = {float(config.rw_meas_temp_f):g} degF is not used: it is the temperature "
                "of rw or of the LAS header RW, and this run takes Rw from neither."
            )
        if not self._setting("standardize_input") or WorkflowStep.NORMALIZATION.value in config.skip_steps:
            if config.rename_input_curves:
                notes.append(
                    "rename_input_curves=True has no effect: curves are renamed by the standardization "
                    "step, which is off (standardize_input=False or 'normalization' in skip_steps)."
                )
            if curve_aliases_of(well):
                notes.append(
                    "The curve-name mapping (curve_aliases) is not applied: it is read by the "
                    "standardization step, which is off (standardize_input=False or 'normalization' in "
                    "skip_steps)."
                )
        if config.use_peer_core and not config.peer_wells:
            notes.append("use_peer_core=True has no effect: no peer_wells are given.")
        if config.peer_wells and not config.use_peer_core:
            notes.append(
                "peer_wells are not used: their core calibrates this well only with use_peer_core=True."
            )
        return notes

    def _check_lithology_curve(self, well: "Well") -> None:
        """Refuse a ``lithology_curve`` whose curve does not hold lithology codes (``"current"``).

        The curve named is held to the checks of the automatic detection
        (:func:`~petrophys.interpretation.lithology.matrix_inference.lithology_code_curve`),
        so a log such as GR is not decoded as codes. A curve the well lacks, or
        one without finite samples, is left to the warning of :meth:`_warn_code_curve`.
        """
        name = self.config.lithology_curve
        if name is None or self._historical:
            return
        from petrophys.interpretation.lithology.matrix_inference import (
            LITHOLOGY_CODE_CLASS,
            _finite_curve,
            _validate_lithology_codes,
        )

        curve = _finite_curve(well, name)
        if curve is None:
            return
        user_groups = self._user_code_groups()
        known = None if user_groups is None else {code for code in user_groups if isinstance(code, int)}
        if _validate_lithology_codes(well, name, known) is not None:
            return
        data = np.asarray(curve.data, dtype=float)
        data = data[np.isfinite(data)]
        whole = bool(np.allclose(data, np.round(data), atol=1e-6))
        table = "lithology_codes" if known is not None else "the built-in lithology code table"
        codes = LITHOLOGY_CODE_CLASS if known is None else known
        decoded = float(np.mean([float(value).is_integer() and int(value) in codes for value in data]))
        raise ValueError(
            f"lithology_curve={name!r} does not look like lithology codes: the curve has {data.size} finite "
            f"samples and {np.unique(data).size} distinct values from {data.min():g} to {data.max():g}"
            f"{'' if whole else ', not whole numbers'}, {decoded:.0%} of them codes of {table}. "
            "A lithology code curve holds 20 or more finite "
            "samples, all whole numbers"
            + ("" if known is not None else " from 0 to 99")
            + f", at most 50 distinct, half or more of them codes of {table}. Name the well's lithology "
            "code curve, give its codebook as lithology_codes, or leave lithology_curve unset."
        )

    def _user_code_groups(self) -> Optional[Dict[Any, Optional[str]]]:
        """Lithology group of each code of ``lithology_codes``, or None when it is not set.

        Not used by the benchmark profiles. Read once per run.
        """
        if self.config.lithology_codes is None or self._reproducing_benchmark:
            return None
        if getattr(self, "_user_codes_source", None) is not self.config.lithology_codes:
            from collections.abc import Mapping

            from petrophys.interpretation.lithology.labels import lithology_group_for_label
            from petrophys.io.lithology_reader import read_lithology_codes

            source = self.config.lithology_codes
            if isinstance(source, Mapping):
                labels = {}
                for code, label in source.items():
                    try:
                        number = float(code)
                        key = int(number) if number.is_integer() else code
                    except (TypeError, ValueError):
                        key = code
                    labels[key] = str(label)
            else:
                labels = read_lithology_codes(source)
            self._user_code_labels = labels
            self._user_code_group_map = {
                code: lithology_group_for_label(label, known=self._group_to_matrix)
                for code, label in labels.items()
            }
            self._user_codes_source = source
        return self._user_code_group_map

    def _warn_code_curve(self, code_curve: Optional[str]) -> None:
        """Say which curve the lithology codes are read from, when it is not LITHOLOGY_GEOLINK."""
        requested = self.config.lithology_curve
        if requested is not None and code_curve != requested:
            instead = f"{code_curve} is read instead" if code_curve else "no lithology code curve is read"
            self._result.warnings.append(
                f"lithology_curve={requested!r}: the well has no such curve with finite samples; {instead}."
            )
        user_groups = self._user_code_groups()
        if code_curve is None and user_groups is not None:
            self._result.warnings.append(
                "lithology_codes is not used: no lithology code curve of the well holds its codes; "
                "name the curve with lithology_curve=."
            )
        if code_curve is not None and user_groups is not None:
            unmapped = sorted(
                f"{code} ({self._user_code_labels[code]})" for code, group in user_groups.items() if group is None
            )
            note = f"Lithology codes are read from {code_curve} and decoded with lithology_codes."
            if unmapped:
                note += " Codes whose name is no recognized rock get no lithology group: " + ", ".join(unmapped) + "."
            self._result.warnings.append(note)
        elif code_curve is not None and code_curve != "LITHOLOGY_GEOLINK":
            note = (
                f"Lithology codes are read from {code_curve} and decoded with the built-in "
                "lithology code table."
            )
            if code_curve != requested:
                note += (
                    " If the curve holds other codes, pass their codebook as lithology_codes, "
                    "or lithology_source='computed' or lithology intervals."
                )
            self._result.warnings.append(note)

    def _vendor_lithology_resolution(self, well: "Well"):
        """Lithology and matrix from the run's lithology code curve, or None without one."""
        from petrophys.interpretation.lithology.matrix_inference import resolve_from_lithology_log

        if self._eval_lithology_detection:
            return resolve_from_lithology_log(well, warn=False)
        code_curve = self._lithology_code_curve(well)
        if code_curve is None:
            return None
        user_groups = self._user_code_groups()
        if user_groups is not None:
            from petrophys.interpretation.lithology.matrix_inference import _finalize

            data = np.asarray(well.get(code_curve).data, dtype=float)
            groups = [
                user_groups.get(int(code)) for code in np.round(data[np.isfinite(data)]).astype(int)
            ]
            clastic = sum(group in _CLASTIC_GROUPS for group in groups)
            carbonate = sum(group == "limestone" for group in groups)
            dolomite = sum(group == "dolomite" for group in groups)
            return _finalize(
                clastic, carbonate, dolomite, "vendor_log_codebook", clastic + carbonate + dolomite, False
            )
        return resolve_from_lithology_log(well, warn=False, code_curve=code_curve)

    @property
    def _benchmark(self) -> bool:
        """True when the run uses the calculations of the eval package's main run.

        That is the "geolink" profile for a well other than the package's two
        re-run wells, outside a context export (:func:`_uses_rerun_calculations`).
        It gates what the main run did beyond the re-run's calculations.
        """
        return self._reproducing_benchmark and not self._rerun_selected

    @property
    def _reproducing_benchmark(self) -> bool:
        """True under the "geolink" profile: keep the eval package's calculations.

        It gates what the package's main run and re-run share; :attr:`_benchmark`
        gates what only the main run did.
        """
        return self.config.calculation_profile in _BENCHMARK_PROFILES

    @property
    def _carbonate_calibration(self) -> str:
        """Carbonate mineral calibration: the setting, else the profile's default."""
        if self.config.carbonate_calibration is not None:
            return self.config.carbonate_calibration
        return "costa" if self._reproducing_benchmark else "generic"

    @property
    def _crossover_gas_method(self) -> str:
        """PHIT rule in neutron-density crossover (PHID > PHIN + 0.05).

        The root-mean-square of PHID and PHIN (Gaymard and Poupon, 1968) is a
        gas-zone approximation, so ``"current"`` applies it only when
        ``hc_type`` is ``"gas"``. Oil hardly moves the neutron log, so in an
        oil well the crossover comes from the matrix or shale endpoints, and
        PHIT is density porosity there, as under the "geolink" profile.
        """
        if not self._reproducing_benchmark and str(self.config.hc_type).lower() == "gas":
            return "rms"
        return "density"

    @property
    def _group_to_matrix(self) -> Dict[str, str]:
        """Lithology group -> MatrixProperties key for this calculation profile."""
        if self._reproducing_benchmark:
            return LITHOLOGY_GROUP_TO_MATRIX
        return _CURRENT_LITHOLOGY_GROUP_TO_MATRIX

    @property
    def _skips_flushed_zone(self) -> bool:
        """True when SXO and MHI are not computed: oil-based mud, current profile.

        Oil-based filtrate does not conduct, so the RMF written for it is a
        placeholder; the invaded-zone Archie SXO, the SW^(1/5) quick-look and
        MHI = SW/SXO all describe flushing by a water-based filtrate.
        """
        return not self._reproducing_benchmark and str(self.config.mud_type).upper() == "OBM"

    def _setting(self, name: str) -> Any:
        """Config value, or the calculation profile's default when it is unset."""
        return _profile_setting(self.config, name)

    @property
    def _historical(self) -> bool:
        """True under the "geolink" profile or while a benchmark package is reproduced."""
        return self._reproducing_benchmark or reproducing_benchmark()

    @property
    def _phi_cl_setting(self) -> float:
        """The configured clay porosity, or 0.15 when it is unset."""
        from petrophys.model.constants import DefaultParameters

        if self.config.phi_cl is None:
            return float(DefaultParameters.CLAY_POROSITY)
        return float(self.config.phi_cl)

    def _measurement_temp_f(self, name: str) -> Optional[float]:
        """An explicit ``rmf_meas_temp_f`` or ``rm_meas_temp_f``, else None.

        The "geolink" profile read a value equal to the 75 degF default as
        left at the default; the current calculations use any explicit value.
        """
        value = getattr(self.config, name)
        if value is None:
            return None
        if self._historical and float(value) == DEFAULT_RESISTIVITY_REFERENCE_TEMP_F:
            return None
        return float(value)

    def _zone_only_parameters(self, well: "Well", prefix: str) -> List[Dict[str, Any]]:
        """``<prefix>_Z<n>`` header values whose zone is a part of the well.

        A zone with a top and a bottom (``TOP_Z1``/``BOT_Z1``) is an interval
        of the well, so its values hold inside it only; a zone that spans
        every depth sample is a whole-well constant and is left out here.
        Each entry carries the zone index, its ``top`` and ``bottom`` in the
        well's depth unit, the ``value`` and its unit label. Empty under the
        "geolink" profile, which read a single zoned value as a whole-well
        constant.
        """
        return [
            zone for zone in self._bounded_zone_parameters(well, prefix)
            if not self._zone_covers_well(well, zone)
        ]

    def _zone_covers_well(self, well: "Well", zone: Dict[str, Any]) -> bool:
        depth = np.asarray(well.depth, dtype=float)
        return bool(np.all(self._zone_mask(well, zone)[np.isfinite(depth)]))

    def _bounded_zone_parameters(self, well: "Well", prefix: str) -> List[Dict[str, Any]]:
        """``<prefix>_Z<n>`` header values of zones with depth bounds (``"current"``)."""
        if self._historical or not getattr(well, "_parameters", None):
            return []
        from petrophys.workflow.zoned_batch import extract_zoned_parameters

        prefix = prefix.upper()
        zones = []
        for zone in extract_zoned_parameters(well, calculation_profile=self.config.calculation_profile):
            raw = zone.raw_parameters.get(prefix)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            zones.append({
                "index": zone.index, "top": float(zone.top), "bottom": float(zone.bottom),
                "value": value, "unit": _parameter_text(well, f"{prefix}_Z{zone.index}__UNIT"),
            })
        # A setting you passed (whole-well, or a zone's in config.zones) ranks
        # above the header's zone values for it.
        explicit = _explicit(self.config)
        given = [name for name in _HEADER_ZONE_SETTINGS.get(prefix, ()) if name in explicit]
        if zones and given:
            for zone in zones:
                note = (
                    f"{prefix}_Z{zone['index']} = {zone['value']:g} in the LAS header "
                    f"({zone['top']:g}-{zone['bottom']:g} {well.depth_unit}) is not used: {given[0]} is set"
                )
                if self._result is not None and note not in self._result.warnings:
                    self._result.warnings.append(note)
            return []
        return zones

    def _zone_mask(self, well: "Well", zone: Dict[str, Any]) -> np.ndarray:
        """Samples inside a zone, both bounds included (as the zoned runner merges)."""
        depth = np.asarray(well.depth, dtype=float)
        return (depth >= zone["top"]) & (depth <= zone["bottom"])

    def _zoned_constant(self, well: "Well", prefix: str) -> Optional[float]:
        """A unique ``<prefix>_Z<n>`` value that holds for the whole well, else None.

        Zone-only values (:meth:`_zone_only_parameters`) count only when their
        zones cover every depth sample; otherwise they are left to the step,
        which applies them inside their zones or warns that they were not
        used, and the whole-well chain runs outside them.
        """
        bounded = self._bounded_zone_parameters(well, prefix)
        if not bounded:
            return _constant_zoned_parameter(well, prefix)
        values = {zone["value"] for zone in bounded if self._zone_covers_well(well, zone)}
        if len(values) == 1:
            return next(iter(values))
        return None

    def _warn_zone_only_unused(self, well: "Well", prefix: str, used_for: str) -> None:
        """Warn once per parameter that zone-only values were not applied."""
        for zone in self._zone_only_parameters(well, prefix):
            message = (
                f"{prefix}_Z{zone['index']} = {zone['value']:g} covers {zone['top']:g}-{zone['bottom']:g} "
                f"{well.depth_unit} only and was not used for {used_for}; run "
                "run_dynamic_workflow_with_zones to apply it inside its zone."
            )
            self._warn_assumed(message)

    def _warn_assumed(self, message: str) -> None:
        """Report an assumed, ignored or missing value in ``result.warnings``
        (``"current"`` runs only)."""
        if self._reproducing_benchmark or self._result is None:
            return
        if message not in self._result.warnings:
            self._result.warnings.append(message)

    def _phie_names(self, include_phit: bool = True) -> List[str]:
        """Effective-porosity curve preference.

        The "geolink" profile reads PHIE alone: neither the core-corrected PHIE_CC nor
        a PHIT fallback, so samples without PHIE stay unevaluated.
        """
        if self._benchmark:
            return ["PHIE"]
        names = ["PHIE_CC", "PHIE"]
        return [*names, "PHIT"] if include_phit else names

    def _perm_names(self, *extra: str) -> List[str]:
        """Permeability curve preference; the "geolink" profile ignores PERM_CC."""
        names = ["PERM"] if self._benchmark else ["PERM_CC", "PERM"]
        return [*names, *extra]

    def _core_calibration_options(self, well: "Well") -> Dict[str, Any]:
        """PermeabilityModule options for the same-well core porosity correction.

        Under "current" core porosity is fitted to the log PHIT: helium porosity
        of cleaned, oven-dried plugs includes the pore space of the clay-bound
        water (API RP 40, 1998; Bush and Jenkins, 1970), so it is total
        porosity. A fit that rests on few pairs, has a slope that is not
        positive or an r^2 below ``_CORE_POROSITY_MIN_R_SQUARED`` is not used.
        The core model is applied over the whole well; its permeability is
        written only where clay volume is at or below the net-rock clay
        cutoff, since a porosity-permeability transform describes reservoir
        rock, not shale.
        The "geolink" profile fits core porosity to PHIE, uses every fit and
        applies each model over its whole interval.
        """
        if self._reproducing_benchmark:
            return {"bound_core_rock": False}
        return {
            "core_porosity_curve": "PHIT",
            "min_porosity_r_squared": _CORE_POROSITY_MIN_R_SQUARED,
            "core_vcl_max": self._net_rock_vcl_cutoff(well),
        }

    def _net_rock_vcl_cutoff(self, well: "Well") -> np.ndarray:
        """Each sample's net-rock clay cutoff (v/v): the setting, else its lithology family's standard."""
        from collections.abc import Mapping

        from petrophys.evaluation.net_pay.cutoffs import standard_cutoffs

        value = self.config.vcl_cutoff
        if value is not None and not isinstance(value, Mapping):
            return np.full(well.n_samples, float(value))
        families = np.broadcast_to(
            np.asarray(self._cutoff_families(self._build_lithology_log(well)), dtype=str), (well.n_samples,))
        cutoff = np.empty(well.n_samples, dtype=float)
        for family in np.unique(families):
            name = str(family)
            cutoff[families == family] = _lithology_value(
                value, name, name, standard_cutoffs(name)["vcl_cutoff"])
        return cutoff

    def _effective_core_porosity(self, well: "Well", phit_cc: np.ndarray) -> np.ndarray:
        """PHIE_CC from core-corrected total porosity (``"current"``).

        ``PHIE_CC = PHIT_CC - (PHIT - PHIE)``, within 0 and ``PHIT_CC``: the
        clay-bound porosity that the porosity step took off PHIT is taken off
        the core-corrected PHIT_CC too. Without both PHIT and PHIE there is no
        clay-bound porosity to take off, and the input is returned as it is.
        """
        phit_cc = np.asarray(phit_cc, dtype=float)
        phit, _, _ = get_curve_data_depthwise(well, ["PHIT"])
        phie, _, _ = get_curve_data_depthwise(well, ["PHIE"])
        if phit is None or phie is None:
            return phit_cc
        with np.errstate(invalid="ignore"):
            bound = np.clip(phit - phie, 0.0, None)
            return np.clip(phit_cc - bound, 0.0, phit_cc)

    def _keep_input_core_curves(self, well: "Well", names: List[str]) -> str:
        """Keep input PHIE_CC/PERM_CC as PHIE_CC_INPUT/PERM_CC_INPUT (``"current"``).

        The run fits its own core model, so the input curves are set aside
        (renamed rather than removed) and PHIE_CC/PERM_CC come only from that
        model, where its fit is accepted. Returns the trace line, and records
        the warning.
        """
        kept = []
        for name in names:
            curve = well.get(name)
            if curve is None:
                continue
            well.remove_curve(f"{name}_INPUT")
            well.rename_curve(name, f"{name}_INPUT")
            kept.append(f"{name}_INPUT")
        message = (
            f"Input curves {', '.join(names)} were kept as {', '.join(kept)} and not used: this run "
            "fits its own core model, and PHIE_CC/PERM_CC are written only where that model's fit "
            "is accepted."
        )
        self._warn_assumed(message)
        return message

    def _core_porosity_names(self) -> Tuple[str, ...]:
        """Core porosity mnemonics that let the core-correction step run."""
        return ("PHIC", "CPOR") if self._historical else _CORE_POROSITY_CURVES

    def _peer_core_available(self) -> bool:
        """True when peer-well core is requested and peer wells are given (``"current"``)."""
        return (
            not self._historical
            and bool(self.config.use_peer_core)
            and bool(self.config.peer_wells)
        )

    def _input_core_curves_kept(self, well: "Well") -> Tuple[str, ...]:
        """Input PHIE_CC/PERM_CC that the whole run reads as they stand (``"current"``).

        Decided once, before any step. When the run fits no core model before
        saturation (no core porosity and no peer-well core, or
        ``core_correction`` in ``skip_steps``), input core-corrected curves are
        the run's core-corrected curves: SW, BVW, net pay and RQI read the
        same PHIE_CC, and no step renames or replaces them. Empty under the
        "geolink" profile and when the run fits its own model, whose
        core-correction step sets the inputs aside as PHIE_CC_INPUT/PERM_CC_INPUT.
        """
        if self._historical:
            return ()
        present = tuple(name for name in ("PHIE_CC", "PERM_CC") if well.get(name) is not None)
        if not present:
            return ()
        skipped = WorkflowStep.CORE_CORRECTION.value in self.config.skip_steps
        has_core = self._has_curve(well, *self._core_porosity_names()) or self._peer_core_available()
        if has_core and not skipped:
            return ()
        reason = (
            "core_correction is in skip_steps"
            if skipped else "the well has no core porosity and no peer-well core to fit its own core model"
        )
        self._warn_assumed(
            f"Input curves {', '.join(present)} are used as they stand by every step (SW, BVW, net pay, "
            f"RQI): {reason}."
        )
        return present

    def _core_peer_wells(self, well: "Well", decisions: List[str]) -> Optional[List["Well"]]:
        """Peer wells for core calibration.

        Under ``"current"`` a peer needs a log porosity (PHIE or PHIT) to fit
        its core to; a peer without one (not yet interpreted) is left out with
        a trace line and a warning. A peer whose formations were attached by
        :func:`petrophys.read` and not stored as its ``formation_tops``
        parameter is passed as a copy that carries them. The "geolink" profile
        passes the peer wells as they are.
        """
        import copy

        peers = self.config.peer_wells
        if self._historical or not peers or not self.config.use_peer_core:
            return peers
        prepared = []
        for peer in peers:
            if peer is None or peer is well:
                prepared.append(peer)
                continue
            name = getattr(peer, "name", None) or "unnamed"
            if get_curve(peer, ["PHIE", "PHIT"]) is None:
                message = (
                    f"Peer well {name} was not used for core calibration: it has no PHIE or PHIT "
                    "to fit its core to. Interpret it first (petrophys.interpret(peer).well)."
                )
                decisions.append(message)
                self._warn_assumed(message)
                continue
            if peer.get_parameter("formation_tops") is None:
                records = (peer.get_parameter("_geology_context") or {}).get("formations")
                if records:
                    try:
                        tops = _attached_formation_tops(records, float(np.nanmax(peer.depth)))
                    except ValueError as exc:
                        reason = " ".join(line.strip() for line in str(exc).splitlines()[:1])
                        decisions.append(
                            f"Peer well {name}: attached formations not used for core calibration: {reason}"
                        )
                    else:
                        peer = copy.copy(peer)
                        peer._parameters = {**peer._parameters, "formation_tops": tops}
                        decisions.append(
                            f"Peer well {name}: its {len(tops.intervals)} attached formation intervals "
                            "are used for formation-matched core calibration"
                        )
            prepared.append(peer)
            # Peer core applies within one field and one formation: say which
            # of these conditions leaves this peer unused.
            field = str(getattr(well.header, "field", None) or "").strip()
            peer_field = str(getattr(peer.header, "field", None) or "").strip()
            target_tops = self.config.formation_tops
            if target_tops is None:
                target_tops = well.get_parameter("formation_tops")
            peer_tops = peer.get_parameter("formation_tops")
            reason = None
            if not field:
                reason = "this well's header has no field name (FLD) to match the peer's field"
            elif peer_field.lower() != field.lower():
                reason = (
                    f"its field ({peer_field or 'none in its header'}) is not this well's field ({field}); "
                    "peer core is used within one field (header FLD)"
                )
            elif target_tops is None:
                reason = (
                    "this well has no formation tops (formations= in petrophys.read or "
                    "petrophys.interpret) to match the peer's formations"
                )
            elif peer_tops is None:
                reason = "it has no formation tops (formations= in petrophys.read) to match this well's formations"
            elif not any(peer_tops.get_by_name(interval.name) is not None for interval in target_tops.intervals):
                reason = "its formation tops share no formation name with this well's"
            if reason is not None:
                message = f"Peer well {name} was not used for core calibration: {reason}."
                decisions.append(message)
                self._warn_assumed(message)
        return prepared

    def _core_porosity_decisions(self, metadata: Dict[str, Any], phie_cc_written: bool) -> List[str]:
        """Trace lines for the "current" core porosity correction and its gate."""
        decisions = []
        fits = metadata.get("core_porosity_fits") or []
        for fit in fits:
            sign = "-" if fit["B"] < 0 else "+"
            decisions.append(
                f"Core porosity correction ({fit.get('source_formation') or 'all depths'}): "
                f"{fit['input_curve']}_CC = {fit['A']:.3f} * {fit['input_curve']} {sign} "
                f"{abs(fit['B']):.3f} (n={fit['n_points']}, r^2={fit['r_squared']:.2f})"
            )
        if phie_cc_written and fits and all(fit["input_curve"] == "PHIT" for fit in fits):
            decisions.append(
                "PHIE_CC = PHIT_CC - (PHIT - PHIE), within 0 and PHIT_CC: plug helium "
                "porosity is total porosity, so the clay-bound part is taken off again"
            )
        reasons = {
            "too_few_porosity_points": lambda item: (
                f"{item['n_porosity_points']} core-log pairs, fewer than {item['min_porosity_points']}"
            ),
            "non_positive_porosity_slope": lambda item: (
                f"slope {item['porosity_correction']['A']:.3f} is not positive"
            ),
            "low_porosity_r_squared": lambda item: (
                f"r^2 {item['r_squared']:.2f} is below {item['min_r_squared']:g}"
            ),
        }
        for item in metadata.get("core_rejected_models") or []:
            explain = reasons.get(item.get("reason"))
            if explain is None:
                continue
            decisions.append(
                f"Core porosity correction ({item.get('source_formation') or 'all depths'}) "
                f"not used: {explain(item)}; no PHIE_CC or PERM_CC from that fit"
            )
        return decisions

    def _select_formation_tops(
        self, well: "Well", context: Dict[str, Any], report: Dict[str, Any]
    ) -> List[str]:
        """Use formations attached by :func:`petrophys.read` when unambiguous.

        Formation tops drive formation-specific core permeability calibration.
        An explicit ``DynamicWorkflowConfig.formation_tops`` always wins;
        otherwise attached formations are used when they do not overlap,
        apart from groups containing their own ranked formations. Models are
        applied group first, so a formation's model takes precedence over its
        group's wherever it can be fitted.
        """
        if self.config.formation_tops is not None:
            if isinstance(self.config.formation_tops, pd.DataFrame):
                # A table from petrophys.read_formations.
                self.config.formation_tops = _attached_formation_tops(
                    self.config.formation_tops.to_dict("records"), float(np.nanmax(well.depth))
                )
            elif not hasattr(self.config.formation_tops, "intervals"):
                raise ValueError(
                    "formation_tops must be FormationTops or a table from "
                    f"petrophys.read_formations, not {type(self.config.formation_tops).__name__}."
                )
            report["formations"] = {"source": "config", "used": True}
            return []
        records = context.get("formations")
        if not records:
            return []
        try:
            tops = _attached_formation_tops(records, float(np.nanmax(well.depth)))
        except ValueError as exc:
            lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
            reason = " ".join(lines[:2])
            if not reason.endswith("."):
                reason += "."
            report["formations"] = {"source": "attached", "used": False, "reason": reason}
            return [
                "Attached formations are used for plotting and reporting only: "
                f"{reason} Pass one stratigraphic level as "
                "DynamicWorkflowConfig(formation_tops=...) to calibrate core "
                "permeability per formation."
            ]
        self.config.formation_tops = tops
        names = [interval.name for interval in tops.intervals]
        report["formations"] = {"source": "attached", "used": True, "intervals": names}
        return [
            f"Using {len(names)} attached formation intervals for formation-specific "
            "core permeability calibration"
        ]

    def _resolve_lithology_domain(self, well: "Well") -> str:
        """Record the metadata-selected lithology domain for this run."""
        self._lithology_resolution = (
            self._vendor_lithology_resolution(well)
            if self._selected_lithology_source in {"auto", "vendor"} else None
        )
        if self._result is not None:
            self._result.lithology_domain = self.config.lithology_domain
        return self.config.lithology_domain

    def _set_auto_lithology(self, res) -> list:
        """Write an auto-resolved lithology/matrix to the (copied) run config."""
        applied = []
        if self._auto_matrix:
            matrix = res.matrix if self._auto_lith else self._matrix_within_domain(res)
            self.config.matrix_type = matrix
            applied.append(f"matrix_type={matrix}")
        if self._auto_lith:
            self.config.lithology_domain = res.lithology
            self.config.lithology = res.lithology  # the lithology alias mirrors the domain
            if self._result is not None:
                self._result.lithology_domain = res.lithology
            applied.append(f"lithology={res.lithology}")
        return applied

    def _matrix_within_domain(self, res) -> str:
        """Keep an auto-resolved matrix inside an explicitly chosen lithology domain.

        ``lithology_domain="carbonate"`` with ``matrix_type="auto"`` resolves to
        limestone or dolomite, never sandstone; ``"clastic"`` (the default)
        resolves to sandstone. When the matrix evidence points outside the
        domain, the domain wins and the conflict is reported in ``warnings``.
        With ``lithology_domain="auto"`` the evidence decides both.
        """
        domain = str(self.config.lithology_domain).strip().lower()
        allowed = _DOMAIN_MATRICES.get(domain)
        if not allowed or res.matrix in allowed:
            return res.matrix
        votes = getattr(res, "votes", None) or {}
        if domain == "carbonate":
            matrix = "dolomite" if votes.get("dolomite", 0) > votes.get("limestone", 0) else "limestone"
        else:
            matrix = allowed[0]
        is_default = domain == _DEFAULT_LITHOLOGY_DOMAIN
        label = f"lithology_domain is '{domain}'" + (" (the default)" if is_default else "")
        if res.method == "default":
            message = (f"{label} with matrix_type='auto' and no matrix evidence: "
                       f"using the {domain} matrix '{matrix}'.")
        else:
            hint = ""
            if domain == "clastic" and res.matrix in _DOMAIN_MATRICES["carbonate"]:
                hint = " If the rock is a carbonate, set lithology_domain='carbonate'."
            message = (f"{label}, but {res.method} points to a {res.matrix} matrix "
                       f"({res.confidence:.0%} of {res.n} samples); using the {domain} matrix "
                       f"'{matrix}'.{hint} Set matrix_type to choose the matrix yourself, or "
                       f"lithology_domain='auto' to let the evidence decide.")
        if self._result is not None and message not in self._result.warnings:
            self._result.warnings.append(message)
        return matrix

    def _apply_lithology_resolution(self, res, iterations: int = 0) -> None:
        """Apply a resolution and record it (single-shot: vendor log / default)."""
        applied = self._set_auto_lithology(res)
        self._lithology_pending = False
        if self._result is not None and applied:
            iter_note = f", {iterations} iteration(s)" if iterations else ""
            self._result.warnings.append(
                f"Auto lithology resolved via {res.method} "
                f"({res.confidence:.0%} of {res.n} samples{iter_note}): " + ", ".join(applied)
            )

    def _resolved_cutoff(self, key: str):
        """The config cutoff, else 0.3 (VCL) or 0.7 (SW).

        Without ``sw_routing_vcl_cutoff`` the VCL value is the clay volume
        above which saturation switches from Archie to a shaly-sand model
        (:meth:`_routing_vcl_cutoff`). Net pay resolves unset cutoffs
        separately (:meth:`_current_net_flags`).
        """
        value = getattr(self.config, key)
        if value is not None:
            return value
        canonical = {"vcl_cutoff": 0.3, "sw_cutoff": 0.7}
        if key in canonical:
            return canonical[key]
        from petrophys.evaluation.net_pay.cutoffs import standard_cutoffs
        return standard_cutoffs(self.config.lithology_domain)[key]

    def _routing_vcl_cutoff(self) -> float:
        """Clay volume above which ``sw_model="auto"`` uses a shaly-sand equation.

        ``sw_routing_vcl_cutoff`` when set, else :meth:`_resolved_cutoff`.
        Saturation routing and the SWIRR_APP sample pick both use it.
        """
        if self.config.sw_routing_vcl_cutoff is not None:
            return self.config.sw_routing_vcl_cutoff
        return self._resolved_cutoff("vcl_cutoff")

    def _cutoff_families(self, lithology: Union[str, np.ndarray]) -> Union[str, np.ndarray]:
        """Lithology family of each sample's net-pay cutoffs (:func:`_cutoff_families`);
        samples without a lithology group take the family of the run's lithology."""
        from petrophys.evaluation.net_pay.cutoffs import cutoff_family

        return _cutoff_families(lithology, cutoff_family(self.config.lithology_domain))

    def _current_net_flags(
        self,
        well: "Well",
        lithology: Union[str, np.ndarray],
        vcl: Optional[np.ndarray],
        phie: np.ndarray,
        sw: np.ndarray,
        vcl_cutoff: Optional[float] = None,
        phie_cutoff: Optional[float] = None,
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """Net flags of a ``"current"`` run and the details behind them.

        A cutoff set in the config applies to every sample. An unset one takes
        the standard (:data:`~petrophys.evaluation.STANDARD_CUTOFFS`) of the
        sample's lithology family (:meth:`_cutoff_families`). Evaporite, coal,
        log-identified halite and bad-hole samples are then removed from net
        reservoir and net pay (:func:`_non_reservoir_samples`); net rock keeps
        them. ``vcl_cutoff`` and ``phie_cutoff``, when given, replace the clay
        and porosity cutoffs. See :func:`_net_flags_by_family` for the result.
        """
        explicit = {
            "vcl_cutoff": self.config.vcl_cutoff if vcl_cutoff is None else vcl_cutoff,
            "phie_cutoff": self.config.phie_cutoff if phie_cutoff is None else phie_cutoff,
            "sw_cutoff": self.config.sw_cutoff,
        }
        return _net_flags_by_family(
            well,
            self._cutoff_families(lithology),
            vcl, phie, sw, explicit,
            _non_reservoir_samples(well, lithology),
        )

    def _swirr_by_lithology(self, well: "Well") -> np.ndarray:
        """Each sample's Swirr from a per-lithology ``swirr`` setting; NaN where none is given."""
        from petrophys.evaluation.net_pay.cutoffs import cutoff_family
        from petrophys.model.constants import ArchieParameterDefaults

        n = well.n_samples
        labels = self._build_lithology_log(well)
        if not isinstance(labels, np.ndarray) or labels.shape != (n,):
            labels = np.full(n, labels or self.config.lithology_domain, dtype=object)
        fallback = self.config.lithology_domain or "generic_clean"
        values = np.full(n, np.nan)
        for index, raw in enumerate(labels):
            name = _coerce_lithology_group(raw) or fallback
            canonical = str(ArchieParameterDefaults.get(name).lithology)
            values[index] = _lithology_value(self.config.swirr, canonical, cutoff_family(name), float("nan"))
        return values

    def _archie_coefficients_for_lithology(
        self,
        lithology: Union[str, np.ndarray],
        n_samples: int,
    ) -> tuple[Any, Any, Any, Dict[str, Dict[str, Any]]]:
        """Build lithology-dependent Archie coefficients with config overrides."""
        from petrophys.model.constants import ArchieParameterDefaults

        if self._benchmark:
            # The eval package used one coefficient set for the whole well,
            # resolved from the lithology domain rather than the lithology log.
            defaults = ArchieParameterDefaults.get(self.config.matrix_type or self.config.lithology_domain)
            a = float(self.config.a) if self.config.a is not None else float(defaults.a)
            m = float(self.config.m) if self.config.m is not None else float(defaults.m)
            n = float(self.config.n) if self.config.n is not None else float(defaults.n)
            summary = {str(defaults.lithology).upper(): {"a": a, "m": m, "n": n, "samples": int(n_samples)}}
            return a, m, n, summary

        if isinstance(lithology, np.ndarray):
            lithology_values = np.asarray(lithology, dtype=object)
            if lithology_values.shape != (n_samples,):
                # Pad/truncate to n_samples; never cyclically tile (np.resize), which
                # would mis-assign per-depth Archie/matrix endpoints.
                fixed = np.full(n_samples, self.config.lithology_domain, dtype=object)
                k = min(lithology_values.size, n_samples)
                fixed[:k] = lithology_values.ravel()[:k]
                lithology_values = fixed
        else:
            lithology_values = np.full(n_samples, lithology or self.config.lithology_domain, dtype=object)

        fallback = self.config.lithology_domain or "generic_clean"
        a_values = np.full(n_samples, np.nan, dtype=float)
        m_values = np.full(n_samples, np.nan, dtype=float)
        n_values = np.full(n_samples, np.nan, dtype=float)
        summary: Dict[str, Dict[str, Any]] = {}

        from petrophys.evaluation.net_pay.cutoffs import cutoff_family

        for idx, raw_lithology in enumerate(lithology_values):
            lithology_name = _coerce_lithology_group(raw_lithology) or fallback
            coeffs = ArchieParameterDefaults.get(lithology_name)
            # A value per lithology goes by the sample's lithology, then its family.
            canonical, family = str(coeffs.lithology), cutoff_family(lithology_name)
            a_values[idx] = _lithology_value(self.config.a, canonical, family, coeffs.a)
            m_values[idx] = _lithology_value(self.config.m, canonical, family, coeffs.m)
            n_values[idx] = _lithology_value(self.config.n, canonical, family, coeffs.n)

            key = str(coeffs.lithology or lithology_name).upper()
            item = summary.setdefault(
                key,
                {
                    "a": a_values[idx],
                    "m": m_values[idx],
                    "n": n_values[idx],
                    "closest_lithology": coeffs.closest_lithology,
                    "samples": 0,
                },
            )
            item["samples"] += 1

        a_out = float(a_values[0]) if len(_finite_unique_numeric(a_values)) == 1 else a_values
        m_out = float(m_values[0]) if len(_finite_unique_numeric(m_values)) == 1 else m_values
        n_out = float(n_values[0]) if len(_finite_unique_numeric(n_values)) == 1 else n_values
        return a_out, m_out, n_out, summary

    def get_pipeline(self):
        """Return the workflow's steps in pipeline order.

        Returns
        -------
        list of tuple
            One ``(step, run, can_run)`` per step: the :class:`WorkflowStep`,
            the method that runs it on a well and returns its
            :class:`StepResult`, and the method that tells whether the well
            and settings allow it.
        """
        return [
            (WorkflowStep.NORMALIZATION, self._step_normalization, self._can_normalization),
            (WorkflowStep.TEMPERATURE, self._step_temperature, self._can_temperature),
            (WorkflowStep.BAD_HOLE, self._step_bad_hole, self._can_bad_hole),
            (WorkflowStep.SPECTRAL_GAMMA, self._step_spectral_gamma, self._can_spectral_gamma),
            (WorkflowStep.ELEMENTAL_SPECTROSCOPY, self._step_elemental_spectroscopy, self._can_elemental_spectroscopy),
            (WorkflowStep.VCLAY, self._step_vclay, self._can_vclay),
            (WorkflowStep.RCL, self._step_rcl, self._can_rcl),
            (WorkflowStep.RMF, self._step_rmf, self._can_rmf),
            (WorkflowStep.HC_CORRECTION, self._step_hc_correction, self._can_hc_correction),
            (WorkflowStep.POROSITY, self._step_porosity, self._can_porosity),
            (WorkflowStep.LITHOLOGY, self._step_lithology, self._can_lithology),
            (WorkflowStep.POROSITY_REFINEMENT, self._step_porosity_refinement, self._can_porosity_refinement),
            (WorkflowStep.CORE_CORRECTION, self._step_core_correction, self._can_core_correction),
            (WorkflowStep.RW, self._step_rw, self._can_rw),
            (WorkflowStep.INVASION, self._step_invasion, self._can_invasion),
            (WorkflowStep.SATURATION, self._step_saturation, self._can_saturation),
            (WorkflowStep.PERMEABILITY, self._step_permeability, self._can_permeability),
            (WorkflowStep.ROCK_QUALITY, self._step_rock_quality, self._can_rock_quality),
            (WorkflowStep.BVW, self._step_bvw, self._can_bvw),
            (WorkflowStep.MHI, self._step_mhi, self._can_mhi),
            (WorkflowStep.NET_PAY, self._step_net_pay, self._can_net_pay),
        ]

    def run_selected_steps(self, well: "Well", steps: list[WorkflowStep]) -> DynamicWorkflowResult:
        """
        Execute only the specified workflow steps, in pipeline order.

        Like :meth:`run`, it works on a copy of :attr:`config`.
        """
        requested_config = self.config
        self.config = _copy_config(requested_config)
        try:
            return self._run_selected_steps(well, steps)
        finally:
            self.config = requested_config

    def _grain_density_setting(self) -> float:
        """Grain density for Qv from CEC: ``rho_ma``, else the matrix's.

        The "geolink" profile takes 2.65 g/cm3 whatever the matrix.
        """
        from petrophys.model.constants import MatrixProperties

        if self.config.rho_ma is not None:
            return float(self.config.rho_ma)
        if not self._reproducing_benchmark:
            try:
                return float(MatrixProperties.get_density(self.config.matrix_type))
            except KeyError:
                pass
        return 2.65

    def _resolve_fluid_density(self) -> None:
        """Set an unset ``rho_fl`` and ``rho_cl`` on the run config.

        They are 1.0 and 2.45 g/cm3, or with ``carbonate_calibration="costa"``
        the calibration's brine and illite, so porosity, the hydrocarbon
        correction and the mineral solve read the same fluid and clay.
        """
        from petrophys.interpretation.lithology.costa_carbonate import (
            COSTA_FLUID_DENSITY,
            ILLITE_DENSITY,
        )

        costa = (
            self.config.carbonate_calibration == "costa"
            and str(self.config.lithology_domain).strip().lower() == "carbonate"
            and self.config.calculation_profile not in _BENCHMARK_PROFILES
        )
        self._costa_fluid = costa and self.config.rho_fl is None
        if self.config.rho_fl is None:
            self.config.rho_fl = COSTA_FLUID_DENSITY if self._costa_fluid else 1.0
        if self.config.rho_cl is None:
            self.config.rho_cl = ILLITE_DENSITY if costa else 2.45

    def _solver_endpoints(self) -> Optional[Dict[str, Dict[str, float]]]:
        """The run's pore fluid (``rho_fl``, ``dt_fl``) and ``mineral_endpoints`` for the clastic mineral solve.

        The benchmark profiles keep the solver's own water endpoint.
        """
        if self._reproducing_benchmark:
            return None
        endpoints: Dict[str, Dict[str, float]] = {
            "water": {"density": float(self.config.rho_fl), "dt": float(self.config.dt_fl)}
        }
        for mineral, values in (self.config.mineral_endpoints or {}).items():
            # The solver's tables name the density endpoint "density".
            endpoints.setdefault(mineral, {}).update(
                {("density" if key == "rhob" else key): value for key, value in values.items()}
            )
        return endpoints

    def _run_selected_steps(self, well: "Well", steps: list[WorkflowStep]) -> DynamicWorkflowResult:
        self._resolve_fluid_density()
        self._result = self._start_result(well)
        self._lithology_log = None
        self._lithology_source = "config"
        self._computed_lithology_written = False
        self._last_phit_nd_average_mask = None
        self._last_phi_cl_info = None
        self._lithology_resolution = None
        self._input_core_basis = ()
        try:
            before = _curve_snapshot(well)
            self._prepare_input_context(well)
            self._account_input_context(before, well)
            self._resolve_lithology_domain(well)
        except (ValueError, FileNotFoundError) as exc:
            if self.config.raise_on_error:
                raise
            self._result.errors.append(f"input_context: {exc}")
            return self._result
        self._input_core_basis = self._input_core_curves_kept(well)

        allowed = set(steps)

        for step_enum, step_fn, _ in self.get_pipeline():
            if step_enum not in allowed:
                continue

            step_name = step_enum.value
            try:
                before = _curve_snapshot(well)
                step_trace = step_fn(well)
                if step_trace is None:
                    step_trace = StepResult(step=step_name, success=True)
                step_trace.step = step_name
                _account_step_curves(
                    step_trace, before, well, benchmark=self._reproducing_benchmark,
                    exact_outputs=step_enum is WorkflowStep.NORMALIZATION,
                )
                # Record a StepResult (same type as run()) for consistent consumers.
                self._result.steps_executed.append(step_trace)
                if not step_trace.success:
                    self._result.errors.append(f"{step_name}: {step_trace.message}")
            except Exception as e:
                if self.config.raise_on_error or isinstance(e, Warning):
                    raise
                self._result.errors.append(f"{step_name}: {e}")
                self._result.specific_steps_skipped[step_name] = str(e)
                self._result.steps_executed.append(
                    StepResult(step=step_name, success=False, message=str(e))
                )

        return self._result

    def run(self, well: "Well") -> DynamicWorkflowResult:
        """
        Run the dynamic workflow on a well.

        Automatically detects available curves and executes
        applicable processing steps in pipeline order.

        Parameters
        ----------
        well : Well
            Well object with raw log data

        Returns
        -------
        DynamicWorkflowResult
            ``result.config`` is the configuration used for this run, with any
            ``"auto"`` matrix/lithology replaced by the resolved values.

        Notes
        -----
        The settings are used as given: LAS header parameters are applied by
        :func:`run_dynamic_workflow` and :func:`petrophys.interpret`, which
        build the config from them, not here.

        Each run works on its own copy of :attr:`config`, so a workflow can be
        reused across wells: values resolved for one well (e.g. an ``"auto"``
        matrix) never carry over to the next, and :attr:`config` itself is
        left unchanged.
        """
        notes = list(getattr(self.config, "_header_notes", ()) or ())
        notes += _unusual_setting_notes(self.config)
        notes += self._unused_setting_notes(well)
        self._check_lithology_curve(well)
        if self.config.zones or getattr(self.config, "_header_zones", None):
            result = self._run_zones(well)
        else:
            requested_config = self.config
            self.config = _copy_config(requested_config)
            try:
                result = self._run(well)
            finally:
                self.config = requested_config
        result.warnings.extend(note for note in notes if note not in result.warnings)
        return result

    def _run_zones(self, well: "Well") -> DynamicWorkflowResult:
        """Run the well, then each distinct set of zone settings, merged over the zones.

        Zones are ``config.zones`` (the caller's) and the LAS header zones
        that :func:`_config_with_header_settings` found (``_header_zones``);
        where they overlap, the caller's values win. Each set of zone settings
        is a full run of the well with those settings over the rest, so
        data-derived picks (GR baselines, irreducible water saturation, the
        core model) are taken over the whole well, not over the zone's samples
        alone; a pick that depends on a zone setting (Swirr from SW, for
        instance) follows that setting. Its curves replace the whole-well ones
        inside the zone.
        """
        import copy as _copy

        from petrophys.model.units import normalize_depth_unit
        from petrophys.workflow.zoned_batch import _convert_depth_between_units

        requested = self.config
        depth = np.asarray(well.depth, dtype=float)
        finite = depth[np.isfinite(depth)]
        step = float(np.nanmedian(np.diff(finite))) if finite.size > 1 else 0.0
        notes: List[str] = []

        def placed(zone: Dict[str, Any], user: bool) -> Optional[Dict[str, Any]]:
            unit = zone["depth_unit"] or well.depth_unit
            if normalize_depth_unit(unit, strict=False) not in {"m", "ft"}:
                raise ValueError(f"zones: zone {zone['name']}: depth unit {unit!r} is not m or ft")
            top = _convert_depth_between_units(zone["top"], unit, well.depth_unit)
            bottom = _convert_depth_between_units(zone["bottom"], unit, well.depth_unit)
            label = f"Zone {zone['name']} ({top:g}-{bottom:g} {well.depth_unit})"
            if not finite.size or not ((depth >= top) & (depth <= bottom)).any():
                notes.append(f"{label} lies outside the logged depths"
                             + ("; its settings are not used" if zone["settings"] else ""))
                return None
            # Beyond the log by more than 1 % of the zone or five samples.
            slack = max(5 * step, 0.01 * (bottom - top))
            if user and (top < finite.min() - slack or bottom > finite.max() + slack):
                notes.append(
                    f"{label} extends beyond the logged depths ({finite.min():g}-{finite.max():g}); "
                    "it applies where the well is logged"
                )
            return {**zone, "top": top, "bottom": bottom, "label": label, "user": user}

        names = (well.name, getattr(getattr(well, "header", None), "well_name", ""))
        mine_only = _zones_for_well(requested.zones or (), *names)
        if requested.zones and not mine_only:
            notes.append(f"No zone of the zones table is for well {well.name}")
        user_zones = [z for z in (placed(zone, True) for zone in mine_only) if z]
        header_zones = [z for z in (placed(zone, False) for zone in getattr(requested, "_header_zones", ()) or ()) if z]
        ordered = sorted(user_zones, key=lambda zone: zone["top"])
        for upper, lower in zip(ordered, ordered[1:]):
            if lower["top"] < upper["bottom"]:
                raise ValueError(f"zones: {upper['label']} and {lower['label']} overlap")

        # Pieces between every zone boundary, each with its settings: the
        # header zones' (the later zone where they overlap), then yours.
        edges = sorted({zone[key] for zone in user_zones + header_zones for key in ("top", "bottom")})
        groups: Dict[tuple, Dict[str, Any]] = {}
        for top, bottom in zip(edges, edges[1:]):
            middle = 0.5 * (top + bottom)
            covering = [zone for zone in header_zones + user_zones if zone["top"] <= middle <= zone["bottom"]]
            if not covering:
                continue
            settings: Dict[str, Any] = {}
            mine: List[str] = []
            for zone in covering:
                for name, value in zone["settings"].items():
                    # A zone's mapping values fill the keys it gives.
                    settings[name] = (
                        _merged_mapping(settings.get(name, getattr(requested, name, None)), value)
                        if name in _MAPPING_SETTINGS else value
                    )
                if zone["user"]:
                    mine.extend(zone["settings"])
            if not settings:
                continue  # only named zones here: the whole-well run holds
            for zone in covering:
                if not zone["user"]:
                    overridden = sorted(set(zone["settings"]) & set(mine))
                    for name in overridden:
                        note = f"{zone['label']}: its {name} is not used where your zone sets {name}"
                        if note not in notes:
                            notes.append(note)
            key = tuple(sorted((name, repr(value)) for name, value in settings.items()))
            group = groups.setdefault(key, {"settings": settings, "mine": set(), "pieces": [], "labels": []})
            group["mine"].update(mine)
            group["pieces"].append((top, bottom))
            for zone in covering:
                if zone["label"] not in group["labels"]:
                    group["labels"].append(zone["label"])

        # Each zone's settings over the whole-well ones make a valid config
        # before anything runs; a header zone that does not is left out.
        runs = []
        for group in groups.values():
            try:
                config = _copy_config(requested, explicit=sorted(group["mine"]), zones=None, **group["settings"])
            except ValueError as exc:
                reason = str(exc).replace("Invalid DynamicWorkflowConfig:\n  - ", "").replace("\n  - ", "; ")
                if group["mine"]:
                    raise ValueError(f"zones: {', '.join(group['labels'])}: {reason}") from None
                notes.append(f"{', '.join(group['labels'])}: settings not used ({reason})")
                continue
            runs.append((group, config))

        source = _copy.deepcopy(well)
        before = set(well.curves)
        result = DynamicWorkflow(_copy_config(requested, explicit=[], zones=None)).run(well)
        result.config.zones = requested.zones
        result.zone_intervals = [
            {"name": zone["name"], "top": zone["top"], "bottom": zone["bottom"]}
            for zone in sorted(user_zones, key=lambda zone: zone["top"])
        ]
        result.warnings.extend(notes)
        for group, config in runs:
            zone_well = _copy.deepcopy(source)
            zone_result = DynamicWorkflow(config).run(zone_well)
            label = ", ".join(group["labels"])
            inside = np.zeros(depth.shape, dtype=bool)
            for top, bottom in group["pieces"]:
                inside |= (depth >= top) & (depth <= bottom)
            computed = set(zone_well.curves) - before
            for name in sorted(computed | (set(well.curves) - before)):
                curve = zone_well.get(name) if name in computed else None
                if curve is None:
                    # Computed by the whole-well run only: nothing for the zone.
                    target = well.get(name)
                    data = np.array(target.data, copy=True)
                    if np.issubdtype(data.dtype, np.floating):
                        data[inside] = np.nan
                        well.add_curve(name, data, target.unit, target.metadata.description)
                    continue
                values = np.asarray(curve.data)
                target = well.get(name)
                if target is None:
                    data = (np.full(depth.shape, np.nan) if np.issubdtype(values.dtype, np.number)
                            else np.full(depth.shape, None, dtype=object))
                else:
                    data = np.array(target.data, copy=True)
                    if np.issubdtype(data.dtype, np.integer) and np.issubdtype(values.dtype, np.floating):
                        data = data.astype(float)
                data[inside] = values[inside]
                well.add_curve(name, data, curve.unit, curve.metadata.description)
            # A zone run interprets the whole well; its whole-well tallies (bad-hole
            # pay removed) do not describe the zone, so they are not repeated.
            result.warnings.extend(
                f"{label}: {message}" for message in zone_result.warnings
                if message not in result.warnings and not message.startswith(_WHOLE_WELL_TALLIES)
            )
            result.errors.extend(f"{label}: {message}" for message in zone_result.errors)
            zone_result.zone_names = list(group["labels"])
            result.zone_results.append(zone_result)
        return result

    def _run(self, well: "Well") -> DynamicWorkflowResult:
        self._auto_matrix = str(self.config.matrix_type).strip().lower() == "auto"
        self._auto_lith = str(self.config.lithology_domain).strip().lower() == "auto"
        self._resolve_fluid_density()

        self._result = self._start_result(well)
        self._lithology_log = None
        self._lithology_source = "config"
        self._computed_lithology_written = False
        self._last_phit_nd_average_mask = None
        self._last_phi_cl_info = None
        self._lithology_resolution = None
        self._input_core_basis = ()

        input_error = self._input_validation_error(well)
        if input_error is not None:
            self._result.errors.append(input_error)
            return self._result

        # Normalize before any auto-lithology routing examines the inputs. The
        # pipeline still exposes this step for discovery and selected-step runs.
        normalization_done = False
        if (
            self._setting("standardize_input")
            and WorkflowStep.NORMALIZATION.value not in self.config.skip_steps
        ):
            try:
                before = _curve_snapshot(well)
                trace = self._step_normalization(well)
                _account_step_curves(
                    trace, before, well, benchmark=self._reproducing_benchmark, exact_outputs=True
                )
                self._result.steps_executed.append(trace)
                normalization_done = True
            except Exception as exc:
                if self.config.raise_on_error or isinstance(exc, Warning):
                    raise
                message = f"normalization: {exc}"
                self._result.errors.append(message)
                self._result.steps_executed.append(
                    StepResult(step="normalization", success=False, message=str(exc))
                )
                return self._result

        try:
            before = _curve_snapshot(well)
            self._prepare_input_context(well)
            self._account_input_context(before, well)
            self._resolve_lithology_domain(well)
        except (ValueError, FileNotFoundError) as exc:
            if self.config.raise_on_error:
                raise
            self._result.errors.append(str(exc))
            return self._result
        self._input_core_basis = self._input_core_curves_kept(well)

        # Tiers 0-1: resolve auto lithology from raw-input sources up front (a vendor
        # lithology log, else core-porosity apparent matrix density), so it drives the
        # whole pipeline (N-D VCL clean line, porosity matrix, Sw, cutoffs). Both are
        # independent of VCL/porosity, so there is no circularity here.
        self._lithology_pending = self._auto_matrix or self._auto_lith
        if self._lithology_pending:
            from petrophys.interpretation.lithology.matrix_inference import (
                resolve_from_core_matrix,
            )
            rho_hc = self.config.rho_hc
            if rho_hc is None:
                rho_hc = {"oil": 0.8, "gas": 0.25, "condensate": 0.5}.get(
                    str(self.config.hc_type).lower(), 0.8
                )
            res = None
            if self._selected_lithology_source == "external":
                from petrophys.interpretation.lithology.matrix_inference import _finalize
                groups = self._lithology_log
                clastic = int(sum(value in {"sandstone", "shale", "silt", "clastic_coarse", "calcareous_shale"} for value in groups))
                carbonate = int(sum(value == "limestone" for value in groups))
                dolomite = int(sum(value == "dolomite" for value in groups))
                res = _finalize(clastic, carbonate, dolomite, "external_lithology", clastic + carbonate + dolomite, False)
            elif self._selected_lithology_source == "vendor":
                res = self._vendor_lithology_resolution(well)
            elif self._selected_lithology_source != "config":
                fluid = {} if self._historical else {"rho_fluid": self.config.rho_fl}
                res = resolve_from_core_matrix(well, rho_hc=rho_hc, warn=False, **fluid)
            if res is not None:
                self._apply_lithology_resolution(res)
            elif self._selected_lithology_source == "config":
                raise ValueError("lithology_source='config' requires explicit lithology and matrix_type, not 'auto'.")

        # Detect available curves
        available = self._detect_available_curves(well)

        # Execute pipeline in order
        pipeline = self.get_pipeline()

        for step_enum, step_fn, can_fn in pipeline:
            step_name = step_enum.value

            if step_enum is WorkflowStep.NORMALIZATION and normalization_done:
                continue

            if (
                step_enum is WorkflowStep.CORE_CORRECTION
                and self._lithology_pending
                and not self._historical
            ):
                # Settle an "auto" lithology before Rw and saturation read the
                # domain's Archie coefficients, with the resolution the end of
                # the run would apply, so result.config is what they used.
                self._resolve_pending_lithology(well)

            if step_name in self.config.skip_steps:
                self._result.steps_skipped.append(step_name)
                self._result.specific_steps_skipped[step_name] = "Disabled in skip_steps"
                continue

            if can_fn(well):
                try:
                    before = _curve_snapshot(well)
                    step_trace = step_fn(well)
                    if step_trace is None:
                        step_trace = StepResult(step=step_name, success=True)
                    step_trace.step = step_name
                    _account_step_curves(
                        step_trace, before, well, benchmark=self._reproducing_benchmark,
                        exact_outputs=step_enum is WorkflowStep.NORMALIZATION,
                    )
                    self._result.steps_executed.append(step_trace)
                    if not step_trace.success:
                        self._result.errors.append(f"{step_name}: {step_trace.message}")
                        if self.config.raise_on_error:
                            raise RuntimeError(step_trace.message)
                except Exception as e:
                    if self.config.raise_on_error or isinstance(e, Warning):
                        raise
                    msg = f"{step_name}: {str(e)}"
                    self._result.warnings.append(msg)
                    self._result.errors.append(msg)
                    self._result.steps_executed.append(
                        StepResult(step=step_name, success=False, message=str(e))
                    )
            else:
                self._result.steps_skipped.append(step_name)
                self._result.specific_steps_skipped[step_name] = self._skip_reason(step_enum, well)
                if (
                    step_enum is WorkflowStep.HC_CORRECTION
                    and self.config.apply_hc_correction
                    and not self._reproducing_benchmark
                ):
                    # Asked for explicitly, so its absence is a warning.
                    self._result.warnings.append(
                        f"hc_correction not applied: {self._skip_reason(step_enum, well)}"
                    )

        # Fallback: if auto lithology is still unresolved (no vendor log, solver
        # skipped), apply the default (sandstone/clastic) with a warning.
        if self._lithology_pending:
            self._resolve_pending_lithology(well)

        return self._result

    def _account_input_context(self, before: Dict[str, np.ndarray], well: "Well") -> None:
        """Record on the input_context step the curves it created or changed."""
        step = next((s for s in reversed(self._result.steps_executed) if s.step == "input_context"), None)
        if step is not None:
            _account_step_curves(step, before, well, benchmark=self._reproducing_benchmark)

    def _resolve_pending_lithology(self, well: "Well") -> None:
        """Resolve an "auto" lithology from the solver minerals, else the default.

        Under ``"current"`` it runs before the core-correction step (after the
        porosity and lithology steps): Rw, saturation and net pay then use the
        resolved domain and matrix, and a warning names any earlier step that
        used a provisional matrix with another clean line or endpoint.
        """
        from petrophys.interpretation.lithology.matrix_inference import (
            _default, resolve_from_minerals,
        )
        # The matrix an unresolved "auto" stood for so far (_resolve_matrix_arrays).
        domain = str(self.config.lithology_domain).strip().lower()
        provisional = _DOMAIN_MATRICES.get(domain, ("sandstone",))[0]
        self._apply_lithology_resolution(
            resolve_from_minerals(well, warn=False) or _default(warn=False)
        )
        if not self._reproducing_benchmark:
            self._warn_provisional_clean_line()
        if self._historical or self.config.rho_ma is not None:
            return
        porosity = next(
            (item for item in self._result.steps_executed if item.step == "porosity" and item.success),
            None,
        )
        constant_auto = (
            porosity is not None
            and porosity.parameters.get("matrix_mode") == "constant"
            and str(porosity.parameters.get("matrix_source")) == "config_or_default[auto]"
        )
        if constant_auto and provisional != self.config.matrix_type:
            self._result.warnings.append(
                f"The matrix was resolved ({self.config.matrix_type}) after the porosity step, which "
                f"used the provisional {provisional} matrix. Set matrix_type='{self.config.matrix_type}' "
                f"to compute porosity with the {self.config.matrix_type} matrix."
            )

    def _warn_provisional_clean_line(self) -> None:
        """Warn when VCL_ND used another clean line than the matrix resolved late in the run."""
        from petrophys.interpretation.clay_volume.nd_method import nd_clean_line

        step = next((item for item in self._result.steps_executed if item.step == "vclay"), None)
        if step is None or not step.success or not step.parameters.get("use_nd"):
            return
        used, resolved = step.parameters.get("nd_matrix"), self._vcl_nd_matrix()
        if nd_clean_line(used) != nd_clean_line(resolved):
            self._result.warnings.append(
                f"The matrix was resolved ({self.config.matrix_type}) only after the clay-volume step, "
                f"which measured VCL_ND from the provisional {used} clean line. Set "
                f"matrix_type='{self.config.matrix_type}' to measure it from the {resolved} line."
            )

    @staticmethod
    def _input_validation_error(well: "Well") -> Optional[str]:
        """Return a user-facing error when a well cannot support a workflow."""
        if well is None:
            return "A Well instance is required"
        depth = getattr(well, "depth", None)
        if depth is None:
            return "Well has no depth samples"
        try:
            depth_values = np.asarray(depth, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return "Well depth must be numeric"
        if depth_values.size == 0 or not np.any(np.isfinite(depth_values)):
            return "Well has no finite depth samples"

        for mnemonic in getattr(well, "curves", []):
            if str(mnemonic).strip().upper() in {
                "DEPTH", "DEPT", "MD", "TDEP", "TVD", "TVDSS",
            }:
                continue
            curve = well.get(mnemonic)
            if curve is None:
                continue
            try:
                values = np.asarray(curve.data, dtype=float)
            except (TypeError, ValueError):
                continue
            if values.size and np.any(np.isfinite(values)):
                return None
        return "Well has no finite log-curve samples"

    def _detect_available_curves(self, well: "Well") -> set:
        """Detect which curves are available in the well."""
        available = set()
        for name in ["GR", "SP", "NPHI", "RHOB", "DTC", "DTS", "RDEP", "RT",
                      "RESS", "RESM", "RSHA", "RXO", "PEF", "CALI", "CAL", "BS", "DRHO",
                      "VCLAY", "PHIE", "PHIT", "SWT-LOG", "SXO-LOG", "PERM", "TEMP"]:
            if get_curve(well, [name]) is not None:
                available.add(name)
        return available

    def _missing(self, well: "Well", *requirements: tuple) -> List[str]:
        """Return the labels of unmet ``(label, curve names...)`` requirements."""
        return [label for label, *names in requirements if not self._has_curve(well, *names)]

    def _skip_reason(self, step: "WorkflowStep", well: "Well") -> str:
        """Explain, in user terms, why a pipeline step cannot run on ``well``."""
        porosity = ("porosity (PHIE/PHIT)", "PHIE_CC", "PHIE", "PHIT")
        saturation = ("water saturation (SW)", "SW", "SWA", "SWT-LOG")
        deep_res = ("deep resistivity (RDEP/RT)", "RDEP", "RT")
        missing: List[str] = []
        if step is WorkflowStep.NORMALIZATION:
            return "standardize_input=False"
        if step is WorkflowStep.TEMPERATURE:
            return "The well has no depth index"
        if step is WorkflowStep.BAD_HOLE:
            if self._reproducing_benchmark:
                return "Needs a caliper (CALI) or density-correction (DRHO) curve"
            header_bit_size = getattr(well.header, "bit_size", None)
            if (
                header_bit_size is not None
                and _plausible_bit_size(header_bit_size) is None
                and self._has_curve(well, "CALI", "CAL")
            ):
                message = (
                    f"Header bit size {header_bit_size} is not a hole size in inches (3-36 in): "
                    "a null value or another unit, so it is ignored and the caliper washout test "
                    "does not run; bad hole is not assessed."
                )
                self._warn_assumed(message)
                return message
            return (
                "Needs a density-correction (DRHO) curve, or a caliper (CALI) and a bit size "
                "(BS curve or a 3-36 in header bit size)"
            )
        if step is WorkflowStep.VCLAY:
            return "Needs a gamma-ray curve, an SP curve, or neutron and density curves"
        if step is WorkflowStep.RCL:
            missing = self._missing(well, deep_res, ("clay volume (VCL)", "VCL", "VCLAY"))
        elif step is WorkflowStep.LITHOLOGY:
            if self.config.lithology_domain == "carbonate":
                clay = (
                    ("gamma ray (GR)", "GR", "CGR", "SGR")
                    if self._carbonate_calibration == "costa"
                    else ("clay volume (VCL)", "VCL", "VCLAY", "VSH")
                )
                missing = self._missing(
                    well,
                    ("bulk density (RHOB)", "RHOB"),
                    clay,
                    ("porosity (PHIT/PHIE)", "PHIT", "PHIE"),
                )
            elif not self._reproducing_benchmark:
                missing = self._missing(
                    well,
                    ("bulk density (RHOB)", "RHOB"),
                    ("neutron porosity or photoelectric factor (NPHI/PEF)", "NPHI", "PEF"),
                    ("clay volume (VCL)", "VCL"),
                    ("effective porosity (PHIE)", "PHIE"),
                )
            else:
                missing = self._missing(
                    well,
                    ("bulk density (RHOB)", "RHOB"),
                    ("neutron porosity (NPHI)", "NPHI"),
                    ("photoelectric factor or sonic (PEF/DTC)", "PEF", "DTC"),
                    ("clay volume (VCL)", "VCL"),
                    ("effective porosity (PHIE)", "PHIE"),
                )
        elif step is WorkflowStep.SPECTRAL_GAMMA:
            return (
                "Not run by the interpretation workflow; use "
                "petrophys.interpretation.geochemistry.interpret_spectral_gamma directly"
            )
        elif step is WorkflowStep.ELEMENTAL_SPECTROSCOPY:
            return (
                "Not run by the interpretation workflow; use "
                "petrophys.interpretation.geochemistry.interpret_elemental_spectroscopy directly"
            )
        elif step is WorkflowStep.RMF:
            if not self._reproducing_benchmark:
                return (
                    "No mud-filtrate resistivity source: set mud_type='OBM', rmf_at_meas_temp, "
                    "rmf_salinity_ppm, rmf_chlorides_mg_l or rm, or provide RMF, CLFL or RM "
                    "in the LAS header"
                )
            return (
                "No mud-filtrate resistivity source: set mud_type='OBM', rmf_at_meas_temp, "
                "rmf_salinity_ppm or rm, or provide RMF/RM in the LAS header"
            )
        elif step is WorkflowStep.INVASION:
            missing = self._missing(
                well,
                ("shallow or medium resistivity (RXO/RSHA/RMED)", "RSHA", "RESS", "RXO", "RMED", "RESM"),
                deep_res,
            )
        elif step is WorkflowStep.HC_CORRECTION:
            if not self.config.apply_hc_correction:
                return "apply_hc_correction=False (hydrocarbon correction is opt-in)"
            missing = self._missing(
                well, ("neutron porosity (NPHI)", "NPHI"), ("bulk density (RHOB)", "RHOB"),
                *(() if self._reproducing_benchmark else (
                    ("a flushed-zone water saturation curve (SXO), as the step runs before saturation",
                     "SXO", "SWXOA", "SXO-LOG"),
                )),
            )
        elif step is WorkflowStep.POROSITY:
            return "Needs a density, neutron or sonic curve, or an input porosity curve"
        elif step is WorkflowStep.POROSITY_REFINEMENT:
            if WorkflowStep.POROSITY.value in self.config.skip_steps:
                return "The porosity step is disabled in skip_steps"
            return "Not run by the interpretation workflow; porosity is computed once in the porosity step"
        elif step is WorkflowStep.CORE_CORRECTION:
            if self._historical:
                core = ("core porosity (PHIC/CPOR)", "PHIC", "CPOR")
            else:
                core = (
                    "core porosity (PHIC/CPOR/CPOR-AMB) or peer-well core (use_peer_core with peer_wells)",
                    *self._core_porosity_names(),
                )
            missing = self._missing(well, porosity, core)
            if self._peer_core_available() and core[0] in missing:
                missing.remove(core[0])
        elif step is WorkflowStep.SATURATION:
            missing = self._missing(well, ("effective porosity (PHIE)", "PHIE_CC", "PHIE"), deep_res)
            if self._result is not None and self._result.rw_computed is None:
                missing.append("formation water resistivity (Rw)")
        elif step is WorkflowStep.PERMEABILITY:
            missing = self._missing(well, porosity)
        elif step is WorkflowStep.ROCK_QUALITY:
            missing = self._missing(well, ("permeability (PERM)", "PERM_CC", "PERM"), porosity)
        elif step in (WorkflowStep.BVW, WorkflowStep.NET_PAY):
            missing = self._missing(well, porosity, saturation)
        elif step is WorkflowStep.MHI:
            if self._skips_flushed_zone:
                return _OBM_FLUSHED_ZONE_NOTE
            missing = self._missing(well, saturation)
            if not self._has_sxo_curve(well):
                missing.append("flushed-zone saturation (SXO)")
        if missing:
            return "Missing " + ", ".join(missing)
        return "Required usable curves or parameters are unavailable"

    def _can_normalization(self, well: "Well") -> bool:
        return bool(self._setting("standardize_input"))

    def _step_normalization(self, well: "Well") -> StepResult:
        """Standardize aliases and physical units before interpretation."""
        from petrophys.preprocessing.standardization import (
            add_standard_curve_aliases,
            standardize_curves,
        )

        before = {
            name: {
                "unit": str(getattr(well.get(name), "unit", "") or ""),
                "valid_count": int(well.get(name).valid_count()),
            }
            for name in well.curves
            if well.get(name) is not None
        }
        from petrophys.preprocessing.standardization.curve_standardizer import (
            CURVE_PLAUSIBLE_RANGES,
            RANGE_SCREENING_WARNING_FRACTION,
            _ceiling_screening_message,
            _range_screening_message,
        )
        from petrophys.preprocessing.standardization.unit_converter import benchmark_unit_table

        # The "geolink" profile keeps the benchmark packages' unit handling.
        with benchmark_unit_table() if self._reproducing_benchmark else nullcontext():
            standardize_curves(
                well,
                rename_curves=self.config.rename_input_curves,
                convert_units=True,
                plausible_ranges=(
                    None if self._reproducing_benchmark
                    else {**CURVE_PLAUSIBLE_RANGES, **_CURRENT_RANGE_LIMITS}
                ),
            )
        rename_map = dict(well.get_parameter("_rename_map") or {})
        working_aliases = {}
        if not self.config.rename_input_curves:
            working_aliases = add_standard_curve_aliases(
                well, prefer_deepest_array=not self._reproducing_benchmark
            )
        unit_changes = {}
        for original, details in before.items():
            target = rename_map.get(original, original)
            curve = well.get(target) or well.get(original)
            if curve is None:
                continue
            after_unit = str(getattr(curve, "unit", "") or "")
            if after_unit != details["unit"]:
                unit_changes[original] = {
                    "from": details["unit"],
                    "to": after_unit,
                }

        range_screening = {
            name: dict(curve.metadata.custom["range_screening"])
            for name in well.curves
            for curve in [well.get(name)]
            if "range_screening" in curve.metadata.custom
        }
        for name, item in range_screening.items():
            if item["fraction"] < RANGE_SCREENING_WARNING_FRACTION:
                continue
            if not self._reproducing_benchmark:
                # The standardizer's own warning text, so a caller that also
                # captures its DataQualityWarning sees one message.
                self._result.warnings.append(_range_screening_message(name, item))
            else:
                message = (
                    f"{name}: {item['masked']} of {item['finite']} samples were outside the "
                    f"plausible {item['standard_curve']} range {item['low']:g}-{item['high']:g} "
                    "and were removed"
                )
                if item["hint"]:
                    message += f"; {item['hint']}, check the unit label {item['source_unit']!r}"
                self._result.warnings.append(message + ".")
        ceiling_screening = {
            name: dict(curve.metadata.custom["ceiling_screening"])
            for name in well.curves
            for curve in [well.get(name)]
            if "ceiling_screening" in curve.metadata.custom
            # A working alias copies its source curve's record.
            and "working_alias_source" not in curve.metadata.custom
        }
        for name, item in ceiling_screening.items():
            self._result.warnings.append(_ceiling_screening_message(name, item))

        return StepResult(
            step="normalization",
            success=True,
            method="curve_alias_and_unit_standardization",
            input_curves=sorted(before),
            output_curves=sorted(well.curves),
            parameters={
                "rename_curves": self.config.rename_input_curves,
                "convert_units": True,
                "rename_map": rename_map,
                "working_aliases": working_aliases,
                "unit_changes": unit_changes,
                "core_saturation_normalization": {
                    name: curve.metadata.custom["core_saturation_normalization"]
                    for name in well.curves
                    for curve in [well.get(name)]
                    if "core_saturation_normalization" in curve.metadata.custom
                },
                "range_screening": range_screening,
                **({"ceiling_screening": ceiling_screening} if ceiling_screening else {}),
            },
            decisions=[
                "Standardized curve units, nulls, and plausible ranges before routing",
                (
                    "Added canonical working aliases while retaining vendor mnemonics"
                    if working_aliases
                    else "No canonical working aliases were required"
                ),
            ],
        )

    def _has_curve(self, well: "Well", *names: str) -> bool:
        """Check if any named curve has finite usable numeric samples."""
        return get_curve(well, names) is not None

    def _has_nmr_swirr_route(self, well: "Well") -> bool:
        """Return True when NMR Swirr can be read or derived."""
        if self._has_curve(well, "SWIRR_NMR", "NMR_SWIRR", "SWIRR_CMR", "CMR_SWIRR"):
            return True
        has_total = self._has_curve(
            well,
            "NMR_PHIT",
            "MPHI",
            "TCMR",
            "CMRPHI",
            "NMRP",
            "PHINMR",
        )
        has_bvi_or_ffi = self._has_curve(
            well,
            "NMR_BVI",
            "MBVI",
            "CBVI",
            "CMR_BVI",
            "BVI",
            "NMR_FFI",
            "MFFI",
            "CMFF",
            "CMR_FFI",
            "FFI",
        )
        return bool(has_total and has_bvi_or_ffi)

    # ---- CAN-RUN checks ----

    def _can_temperature(self, well: "Well") -> bool:
        return (
            well.depth is not None
            or self._has_curve(well, "DEPT", "DEPTH")
        )

    def _can_bad_hole(self, well: "Well") -> bool:
        if self._reproducing_benchmark:
            return self._has_curve(well, "CALI", "CAL") or self._has_curve(well, "DRHO")
        # Without a bit size a caliper cannot be tested for washout.
        return self._has_curve(well, "DRHO") or (
            self._has_curve(well, "CALI", "CAL") and bool(np.isfinite(_known_bit_size(well)).any())
        )

    def _can_vclay(self, well: "Well") -> bool:
        # Accept the GR family: a CGR/SGR-only well still supports GR-based VCL.
        # The step also supports SP-only and neutron-density-only wells, so a
        # GR-less well must not be silently left without any clay volume
        # (downstream Sw would otherwise treat every sample as clean sand).
        return (
            self._has_curve(
                well, "GR", "CGR", "CGR_CALC", "SGR", "SGR_CALC",
                "HSGR", "HCGR", "GRKT", "GRTO", "GRSL"
            )
            or self._has_curve(well, "SP")
            or (self._has_curve(well, "NPHI", "PHIN") and self._has_curve(well, "RHOB"))
        )

    def _can_rcl(self, well: "Well") -> bool:
        return self._has_curve(well, "RDEP", "RT") and self._has_curve(well, "VCL", "VCLAY")

    def _can_lithology(self, well: "Well") -> bool:
        if self.config.lithology_domain == "carbonate":
            clay = ("GR", "CGR", "SGR") if self._carbonate_calibration == "costa" else ("VCL", "VCLAY", "VSH")
            return (
                self._has_curve(well, "RHOB")
                and self._has_curve(well, *clay)
                and self._has_curve(well, "PHIT", "PHIE")
            )
        if not self._reproducing_benchmark:
            # Version 1 solves where NPHI is logged (with or without PEF); the
            # version 2 solve fills samples with PEF but no NPHI.
            return (
                self._has_curve(well, "RHOB")
                and self._has_curve(well, "NPHI", "PEF")
                and self._has_curve(well, "VCL")
                and self._has_curve(well, "PHIE")
            )
        return (
            self._has_curve(well, "RHOB")
            and self._has_curve(well, "NPHI")
            and self._has_curve(well, "PEF", "DTC")
            and self._has_curve(well, "VCL")
            and self._has_curve(well, "PHIE")
        )

    def _can_spectral_gamma(self, well: "Well") -> bool:
        return False

    def _can_elemental_spectroscopy(self, well: "Well") -> bool:
        return False

    def _can_rmf(self, well: "Well") -> bool:
        return (
            str(self.config.mud_type).upper() == "OBM"
            or self.config.rmf_at_meas_temp is not None
            or self.config.rmf_salinity_ppm is not None
            or self.config.rmf_chlorides_mg_l is not None
            or self.config.rm is not None
            or well.header.rmf is not None
            or well.header.rm is not None
            or _parameter_float(well, "RMF") is not None
            or self._zoned_constant(well, "RMF") is not None
            or _parameter_float(well, "RM") is not None
            or self._zoned_constant(well, "RM") is not None
            or bool(self._zone_only_parameters(well, "RMF"))
        )

    def _can_rw(self, well: "Well") -> bool:
        # Always run the Rw step so the workflow can fall back through
        # metadata, salinity, log-based heuristics, and finally the built-in
        # default when no explicit source is available.
        return True

    def _can_invasion(self, well: "Well") -> bool:
        shallow = ("RSHA", "RESS", "RXO")
        # Without the medium/deep proxy the step needs a true shallow curve.
        candidates = shallow if self._benchmark else (*shallow, "RMED", "RESM")
        return (self._has_curve(well, *candidates) and
                self._has_curve(well, "RDEP", "RT"))

    def _can_hc_correction(self, well: "Well") -> bool:
        return (
            self.config.apply_hc_correction
            and self._has_curve(well, "NPHI")
            and self._has_curve(well, "RHOB")
            # The correction scales with the flushed zone's hydrocarbon
            # (1 - SXO); the step runs before saturation, so it needs an input
            # SXO. Without one the "geolink" profile wrote uncorrected copies.
            and (self._reproducing_benchmark or self._has_curve(well, "SXO", "SWXOA", "SXO-LOG"))
        )

    def _can_porosity(self, well: "Well") -> bool:
        return self._has_curve(well, "RHOB", "NPHI", "DTC", "PHIN", "PHIT", "PHID", "PHIS")

    def _can_porosity_refinement(self, well: "Well") -> bool:
        return False

    def _can_core_correction(self, well: "Well") -> bool:
        if self._train_lacks_core_permeability(well):
            return False
        return (
            self._has_curve(well, "PHIE", "PHIT")
            and (self._has_curve(well, *self._core_porosity_names()) or self._peer_core_available())
        )

    def _train_lacks_core_permeability(self, well: "Well") -> bool:
        """True when the GeoLink training package would not run the core-correction step.

        That package ran the step only on a well with core permeability
        (KH_CORE, CPERM, K_CORE or PERM_CORE) as well as core porosity, so a
        well with core porosity alone has no PHIE_CC.
        """
        return self._dataset == "geolink_train" and not self._has_curve(
            well, "KH_CORE", "CPERM", "K_CORE", "PERM_CORE"
        )

    @staticmethod
    def _train_core_porosity(
        phi_corecal: Optional[np.ndarray], perm_core: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Core-calibrated porosity as the GeoLink training package kept it.

        That package's core calibration gave porosity only together with
        permeability: PHIE_CC holds the calibrated porosity where the
        calibrated permeability is finite, and there is none without it.
        """
        if phi_corecal is None or perm_core is None:
            return None
        kept = np.where(np.isfinite(np.asarray(perm_core, dtype=float)), phi_corecal, np.nan)
        return kept if np.any(np.isfinite(kept)) else None

    def _can_saturation(self, well: "Well") -> bool:
        has_porosity = self._has_curve(well, "PHIE_CC", "PHIE")
        can_compute_sw = self._has_curve(well, "RDEP", "RT") and self._result.rw_computed is not None
        can_compute_sxo = (
            self._has_curve(well, "RSHA", "RXO", "RESS")
            and self._result.rmf_computed is not None
            and not self._skips_flushed_zone
        )
        return has_porosity and (can_compute_sw or can_compute_sxo)

    def _can_permeability(self, well: "Well") -> bool:
        return self._has_curve(well, "PHIE") or self._has_curve(well, "PHIT")

    def _can_rock_quality(self, well: "Well") -> bool:
        return (
            self._has_curve(well, "PERM_CC", "PERM")
            and self._has_curve(well, "PHIE_CC", "PHIE", "PHIT")
        )

    def _can_bvw(self, well: "Well") -> bool:
        return self._has_curve(well, "PHIE_CC", "PHIE", "PHIT") and self._has_curve(well, "SW", "SWA", "SWT-LOG")

    def _can_mhi(self, well: "Well") -> bool:
        if self._skips_flushed_zone:
            return False
        return self._has_curve(well, "SW", "SWA", "SWT-LOG") and self._has_sxo_curve(well)

    def _can_net_pay(self, well: "Well") -> bool:
        return (
            self._has_curve(well, "PHIE_CC", "PHIE", "PHIT")
            and self._has_curve(well, "SW", "SWA", "SWT-LOG")
        )

    # ---- STEP implementations ----

    def _step_temperature(self, well: "Well") -> StepResult:
        from petrophys.interpretation.temperature import (
            calculate_temperature_profile,
            gradient_to_celsius_per_km,
        )
        from petrophys.interpretation.temperature.temp_profile import (
            store_temperature_parameters,
        )

        trace = StepResult(step="temperature", success=True, method="temperature_profile")
        surface_temp_f = (
            self.config.surface_temp_f
            if self.config.surface_temp_f is not None
            else 68.0
        )
        bht_f = self.config.bht_f
        td_ft = self.config.td_ft
        td_unit = "ft" if td_ft is not None else None

        if bht_f is None:
            bht_f = _metadata_bht_f(well)

        if td_ft is None and well.header.td is not None:
            td_ft = float(well.header.td)
            td_unit = well.depth_unit
        if td_ft is None and well.bottom is not None:
            td_ft = float(well.bottom)
            # A depth slice of a well keeps the BHT at the deepest log sample
            # of the well it was cut from.
            logged_bottom = getattr(well, "_logged_bottom", None)
            if logged_bottom is not None and not self._historical:
                td_ft = max(td_ft, float(logged_bottom))
            td_unit = well.depth_unit

        gradient = self.config.gradient_f_per_100ft
        if gradient is not None:
            if "gradient_f_per_100ft" in _explicit(self.config):
                if bht_f is not None:
                    self._warn_assumed(
                        f"The LAS header BHT ({bht_f:.1f} degF) is not used: "
                        f"gradient_f_per_100ft = {gradient:g} is set"
                    )
                bht_f = None
            elif bht_f is not None:
                trace.decisions.append(
                    f"Header TGRAD ({gradient:g} degF/100 ft) not used: the BHT anchors the profile at TD"
                )
                gradient = None

        if bht_f is not None and not self._historical and well.bottom is not None:
            # A total depth above the deepest log sample is not where the BHT
            # was measured (often metres given as feet), and the gradient to
            # it would be far too steep: the step falls back as without it.
            from petrophys.model.units import normalize_depth_unit

            def in_well_unit(value: float, unit: Optional[str]) -> float:
                given = normalize_depth_unit(str(unit or ""), strict=False)
                logged = normalize_depth_unit(str(well.depth_unit or ""), strict=False)
                if given == "ft" and logged == "m":
                    return UnitConverter.feet_to_meters(value)
                if given == "m" and logged == "ft":
                    return UnitConverter.meters_to_feet(value)
                return value

            bottom, unit = float(well.bottom), well.depth_unit or ""
            user_td = "td_ft" in _explicit(self.config) and self.config.td_ft is not None
            candidates = [(td_ft, td_unit, "td_ft" if user_td else "the LAS header TD")]
            if user_td and well.header.td is not None:
                candidates.append((float(well.header.td), well.depth_unit, "the LAS header TD"))
            rejected = []
            # (of the well a depth slice was cut from, as above)
            deepest = max(bottom, float(getattr(well, "_logged_bottom", None) or bottom))
            td_label = (
                f"the deepest log sample{'' if deepest == bottom else ' of the whole well'} ({deepest:g} {unit})"
            )
            td_ft, td_unit = deepest, well.depth_unit
            for value, value_unit, label in candidates:
                # (within the rounding of a unit conversion)
                if in_well_unit(float(value), value_unit) >= bottom * (1.0 - 1e-6):
                    td_ft, td_unit, td_label = value, value_unit, f"{label} ({value:g} {value_unit or unit})"
                    break
                rejected.append(f"{label} = {value:g} {value_unit or unit}")
            if rejected:
                message = (
                    f"{' and '.join(rejected)} is above the deepest log sample ({bottom:g} {unit}), so it is "
                    "not the total depth of the bottom-hole temperature and is not used"
                    + (" (td_ft is in feet)" if user_td else "")
                    + f". The BHT ({bht_f:g} degF) is placed at {td_label}."
                )
                self._warn_assumed(message)
                trace.decisions.append(message)

        trace.parameters = {
            "surface_temp_f": surface_temp_f,
            "bht_f": bht_f,
            # The TD may be in metres; "td" names it without a unit suffix.
            ("td_ft" if self._reproducing_benchmark else "td"): td_ft,
            "td_unit": td_unit,
        }
        if gradient is not None:
            trace.parameters["gradient_f_per_100ft"] = gradient

        try:
            result = calculate_temperature_profile(
                well,
                bht=bht_f,
                surface_temp=surface_temp_f,
                **({"gradient": gradient} if gradient is not None else {}),
                bht_unit="F",
                surface_temp_unit="F",
                td=td_ft,
                td_unit=td_unit,
                depth_unit=well.depth_unit,
            )
        except Exception as exc:
            trace.success = False
            trace.message = str(exc)
            self._result.warnings.append(f"Temperature profile failed: {exc}")
            return trace

        measured = well.get("TEMP")
        temperature = np.asarray(result.temperature, dtype=float)
        source = result.source
        if measured is not None and not self._historical and well.get("TEMP_LOG") is None:
            # Keep the input temperature log as TEMP_LOG. With at least half of
            # its samples finite it is the temperature source, and the modelled
            # profile fills its gaps; a sparser log is set aside for the profile.
            well.add_curve(
                "TEMP_LOG", np.array(measured.data, dtype=float, copy=True), measured.unit,
                measured.metadata.description or "Input temperature log",
            )
            measured_f = self._temperature_curve_f(measured)
            finite = np.isfinite(measured_f) if measured_f.size == temperature.size else np.zeros(0, bool)
            coverage = float(np.mean(finite)) if finite.size else 0.0
            # bht_f is unset (None) by default; surface_temp_f holds 68 F
            # unless the caller passed it.
            user_anchor = [
                name for name, given in (
                    ("bht_f", self.config.bht_f is not None),
                    ("surface_temp_f", "surface_temp_f" in _explicit(self.config)),
                ) if given
            ]
            if user_anchor:
                # Temperatures set explicitly win over the log (often a mud
                # temperature that reads below the formation).
                message = (
                    f"{' and '.join(user_anchor)} set: the temperature profile follows them; the input "
                    "TEMP curve is kept as TEMP_LOG and not used."
                )
            elif coverage >= 0.5:
                gaps = int(np.count_nonzero(~finite))
                temperature = np.where(finite, measured_f, temperature)
                source = "measured_temperature_log"
                message = (
                    f"The input TEMP curve ({coverage:.0%} finite) is the temperature source, "
                    f"kept as TEMP_LOG; the modelled {result.source} profile fills its {gaps} gaps."
                )
                trace.parameters["temp_log_coverage"] = coverage
                trace.parameters["temp_log_gap_samples"] = gaps
            else:
                message = (
                    f"The input TEMP curve ({coverage:.0%} finite, under the 50% a temperature "
                    "source needs) was replaced by the modelled temperature profile and kept as "
                    "TEMP_LOG; the Rw and Rmf corrections use the modelled profile."
                )
            self._warn_assumed(message)
            trace.decisions.append(message)
        well.add_curve("TEMP", temperature, "degF", "Formation Temperature")
        # The eval package stored these values without unit or description.
        store_temperature_parameters(well, result, with_metadata=not self._reproducing_benchmark)

        self._result.temperature_gradient = result.gradient
        trace.method = source
        trace.output_curves = ["TEMP"]
        trace.parameters["gradient_f_per_100ft"] = result.gradient
        trace.parameters["gradient_c_per_km"] = gradient_to_celsius_per_km(result.gradient)
        if result.reference_temperature is not None:
            trace.parameters["reference_temperature_f"] = result.reference_temperature
        if result.reference_depth is not None:
            trace.parameters["reference_depth_ft"] = result.reference_depth

        if source == "measured_temperature_log":
            if result.source == "fallback_geothermal_gradient":
                trace.decisions.append(
                    "No BHT temperature anchor found: the 30 C/km gradient fills only the TEMP log's gaps"
                )
        elif result.source == "fallback_geothermal_gradient":
            trace.decisions.append("No BHT temperature anchor found -> using fallback 30 C/km geothermal gradient")
            self._warn_assumed(
                "No bottom-hole temperature (bht_f or header BHT): temperatures follow an assumed "
                "30 C/km gradient, which sets the Rw and Rmf temperature corrections."
            )
        elif result.source == "specified_gradient":
            given = "set" if "gradient_f_per_100ft" in _explicit(self.config) else "LAS header TGRAD"
            trace.decisions.append(
                f"Temperature profile from the geothermal gradient ({given}) and the surface temperature "
                f"{surface_temp_f:g} degF at depth 0; no BHT is used"
            )
        else:
            trace.decisions.append("Temperature profile anchored to BHT metadata/input")
        trace.decisions.append(
            f"Gradient = {result.gradient:.4f} F/100ft ({trace.parameters['gradient_c_per_km']:.2f} C/km)"
        )
        trace.decisions.append(
            f"Temp range: {np.nanmin(temperature):.1f} - {np.nanmax(temperature):.1f} F"
        )
        return trace

    def _step_bad_hole(self, well: "Well") -> StepResult:
        from petrophys.preprocessing.quality.bad_hole.flag_generator import generate_bad_hole_flags

        trace = StepResult(step="bad_hole", success=True, method="caliper_drho_flags")
        inputs = []

        cali = get_curve(well, ["CALI", "CAL"])
        bs_curve = get_curve(well, ["BS"])
        drho = get_curve(well, ["DRHO"])
        rhob = get_curve(well, ["RHOB"])
        pef = get_curve(well, ["PEF"])

        if cali is not None: inputs.append(curve_source_mnemonic(cali) or cali.mnemonic)
        if bs_curve is not None: inputs.append(curve_source_mnemonic(bs_curve) or "BS")
        if drho is not None: inputs.append(curve_source_mnemonic(drho) or "DRHO")
        if rhob is not None: inputs.append(curve_source_mnemonic(rhob) or "RHOB")
        if pef is not None: inputs.append(curve_source_mnemonic(pef) or "PEF")
        trace.input_curves = inputs
        if not self._reproducing_benchmark:
            return self._flag_bad_hole_where_bit_size_known(well, trace)

        # The eval package's rule: the BS curve as logged, else the header bit
        # size whatever its value, else an assumed 8.5 in bit; |DRHO| only
        # together with the caliper test.
        if bs_curve is not None:
            bs = np.asarray(bs_curve.data, dtype=float)
            bit_size = bs
            representative_bit_size = _representative_finite_value(bs)
            bit_size_source = "BS_curve_depth_by_depth"
            finite_bs = bs[np.isfinite(bs)]
            bit_size_unique_values = sorted({float(value) for value in np.round(finite_bs, 4)}) if finite_bs.size else []
        else:
            header_bit_size = getattr(well.header, "bit_size", None)
            if header_bit_size is None:
                representative_bit_size = 8.5
                bit_size_source = "fallback_8.5"
            else:
                representative_bit_size = float(header_bit_size)
                bit_size_source = "well_header"
            bit_size = representative_bit_size
            bit_size_unique_values = [float(representative_bit_size)]

        if self._dataset == "geolink_train" and bit_size_source == "fallback_8.5":
            # The GeoLink training package assumes no bit size where the well
            # records none: |DRHO| alone flags washout, and the caliper and
            # large-hole tests are skipped.
            representative_bit_size = bit_size = bit_size_source = None
            bit_size_unique_values = []
            trace.decisions.append("No BS curve or header bit size found; CALI-BS and large-hole criteria skipped")
            flags = generate_bad_hole_flags(well, bit_size=None, drho_alone=True, **self._bad_hole_thresholds(trace))
        else:
            flags = generate_bad_hole_flags(well, bit_size=bit_size, **self._bad_hole_thresholds(trace))
        well.add_curve("BAD_HOLE", flags["bad_hole"].astype(float), "", "Bad Hole Flag")
        if self._benchmark:
            # The eval package carried the flag twice, under both mnemonics.
            well.add_curve("BADHOLEFLAG", flags["bad_hole"].astype(float), "", "Bad Hole Flag")

        n_bad = int(np.sum(flags["bad_hole"]))
        trace.output_curves = ["BAD_HOLE"]
        trace.parameters["bit_size_used"] = representative_bit_size
        trace.parameters["bit_size_source"] = bit_size_source
        trace.parameters["bit_size_unique_values"] = bit_size_unique_values
        trace.parameters["bit_size_depth_dependent"] = bs_curve is not None and len(bit_size_unique_values) > 1
        trace.decisions.append(f"{n_bad} bad-hole samples flagged out of {len(flags['bad_hole'])}")
        return trace

    def _bad_hole_value(self, key: str) -> float:
        """A bad-hole threshold: the ``bad_hole`` value set, else the default."""
        from petrophys.model.constants import DefaultParameters

        defaults = {"cali": DefaultParameters.CALI_BS_THRESHOLD, "drho": DefaultParameters.DRHO_THRESHOLD,
                    "large_hole": DefaultParameters.LARGE_HOLE_BS}
        return float((self.config.bad_hole or {}).get(key, defaults[key]))

    def _bad_hole_thresholds(self, trace: StepResult) -> Dict[str, float]:
        """The ``bad_hole`` thresholds set, as :func:`generate_bad_hole_flags` arguments, recorded in ``trace``."""
        names = {"cali": "cali_threshold", "drho": "drho_threshold", "large_hole": "large_hole_threshold"}
        given = dict(self.config.bad_hole or {})
        if given:
            trace.parameters["bad_hole_thresholds"] = given
            trace.decisions.append(
                "Bad-hole thresholds set: " + ", ".join(f"{key}={value:g}" for key, value in given.items())
            )
        return {names[key]: float(value) for key, value in given.items()}

    def _flag_bad_hole_where_bit_size_known(self, well: "Well", trace: StepResult) -> StepResult:
        """BAD_HOLE under ``"current"``, which assumes no bit size.

        The caliper washout test (CALI - BS > 1.5 in) and the large-hole test
        (BS > 17.5 in) run at the samples whose bit size is known: the BS
        curve, else a header bit size of 3-36 in.
        ``|DRHO|`` > 0.15 g/cm3 flags washout at every sample with a DRHO
        reading, so where no bit size is known BAD_HOLE flags DRHO alone. A
        sample with neither test is not assessed and stays 0; the trace says
        how many, and a warning says where a caliper reading has no bit size.
        The thresholds are those of ``bad_hole`` where it sets them.
        """
        from petrophys.preprocessing.quality.bad_hole.flag_generator import generate_bad_hole_flags

        n_samples = well.n_samples
        logged_bs, _, _ = get_curve_data_depthwise(well, ["BS"])
        from_curve = np.zeros(n_samples, dtype=bool) if logged_bs is None else np.isfinite(logged_bs)
        header_bit_size = getattr(well.header, "bit_size", None)
        if (
            header_bit_size is not None
            and _plausible_bit_size(header_bit_size) is None
            and not from_curve.all()
        ):
            # A null (-999.25) or millimetre header value would flag the
            # whole hole as washout.
            message = (
                f"Header bit size {header_bit_size} is not a hole size in inches (3-36 in): "
                "a null value or another unit, so it is ignored."
            )
            trace.decisions.append(message)
            self._warn_assumed(message)

        bit_size = _known_bit_size(well)
        known = np.isfinite(bit_size)
        cali, _, _ = get_curve_data_depthwise(well, ["CALI", "CAL"])
        drho, _, _ = get_curve_data_depthwise(well, ["DRHO"])
        has_cali = np.zeros(n_samples, dtype=bool) if cali is None else np.isfinite(cali)
        has_drho = np.zeros(n_samples, dtype=bool) if drho is None else np.isfinite(drho)
        caliper_test = has_cali & known
        flags = generate_bad_hole_flags(well, bit_size=bit_size, drho_alone=True, **self._bad_hole_thresholds(trace))
        bad_hole = np.asarray(flags["bad_hole"], dtype=bool)
        assessed = caliper_test | has_drho | np.asarray(flags["large_hole"], dtype=bool)

        unique_values = sorted({float(value) for value in np.round(bit_size[known], 4)})
        header_source = (
            "well_header_bit_runs" if getattr(well.header, "bit_size_intervals", None) else "well_header"
        )
        if not from_curve.any():
            source = header_source if known.any() else "none"
        elif (known & ~from_curve).any():
            source = "BS_curve_and_well_header"
        else:
            source = "BS_curve_depth_by_depth"
        trace.parameters["bit_size_used"] = _representative_finite_value(bit_size)
        trace.parameters["bit_size_source"] = source
        trace.parameters["bit_size_unique_values"] = unique_values
        trace.parameters["bit_size_depth_dependent"] = len(unique_values) > 1
        trace.parameters["no_bit_size_samples"] = int(np.count_nonzero(~known))
        trace.parameters["caliper_test_samples"] = int(np.count_nonzero(caliper_test))
        trace.parameters["not_assessed_samples"] = int(np.count_nonzero(~assessed))

        if not assessed.any():
            trace.decisions.append(
                "Bad hole not assessed: no DRHO reading and no caliper reading with a bit size "
                "(BS curve or header), so BAD_HOLE is not written"
            )
            return trace
        drho_test = f"|DRHO| > {self._bad_hole_value('drho'):g} g/cm3"
        untested = has_cali & ~known
        if untested.any():
            where = _interval_text(well, untested)
            message = (
                f"No bit size (BS curve or header) at {int(np.count_nonzero(untested))} samples "
                "with a caliper reading" + (f" ({where})" if where else "") + ": the caliper "
                f"washout test (CALI - BS > {self._bad_hole_value('cali'):g} in) and the "
                f"large-hole test (BS > {self._bad_hole_value('large_hole'):g} in) do not run there, "
                f"and BAD_HOLE there flags only {drho_test}."
            )
            trace.decisions.append(message)
            self._warn_assumed(message)
        elif not has_cali.any():
            trace.decisions.append(f"No caliper reading: washout is {drho_test} alone")
        if not assessed.all():
            trace.decisions.append(
                f"Bad hole not assessed at {int(np.count_nonzero(~assessed))} samples with neither "
                "a DRHO reading nor a caliper reading with a bit size; BAD_HOLE is 0 there"
            )
        well.add_curve("BAD_HOLE", bad_hole.astype(float), "", "Bad Hole Flag")
        trace.output_curves = ["BAD_HOLE"]
        trace.decisions.append(f"{int(np.count_nonzero(bad_hole))} bad-hole samples flagged out of {n_samples}")
        return trace

    def _step_spectral_gamma(self, well: "Well") -> StepResult:
        """Standardize processed K-Th-U logs and derive spectral diagnostics."""
        from petrophys.interpretation.geochemistry import interpret_spectral_gamma

        result = interpret_spectral_gamma(well)
        if result.warnings:
            self._result.warnings.extend(
                f"spectral_gamma: {warning}" for warning in result.warnings
            )

        measurements = sorted(result.measurements)
        decisions = [
            "Used processed concentration/gamma curves; raw detector spectra were not inverted",
        ]
        if measurements:
            decisions.insert(0, f"Resolved processed products: {', '.join(measurements)}")
        if "CGR_CALC" in result.derived:
            decisions.append("Reconstructed uranium-free gamma ray from potassium and thorium")
        if result.warnings:
            decisions.append(
                f"Completed with {len(result.warnings)} unit/provenance warning(s)"
            )

        return StepResult(
            step="spectral_gamma",
            success=True,
            method="processed_k_th_u_spectral_gamma",
            input_curves=result.input_curves,
            output_curves=result.output_curves,
            parameters={
                "sources": result.sources,
                "conversions": result.conversions,
                "measurements": measurements,
                "derived": sorted(result.derived),
            },
            decisions=decisions,
            message="; ".join(result.warnings),
        )

    def _step_elemental_spectroscopy(self, well: "Well") -> StepResult:
        """Resolve processed elemental dry weights and relative yields."""
        from petrophys.interpretation.geochemistry import interpret_elemental_spectroscopy

        result = interpret_elemental_spectroscopy(well)
        if result.warnings:
            self._result.warnings.extend(
                f"elemental_spectroscopy: {warning}" for warning in result.warnings
            )

        dry_elements = sorted(result.dry_weight_fractions)
        yield_products = sorted(result.relative_yields)
        decisions = [
            "Kept relative yields separate from dry-weight fractions",
            "Did not invert raw spectra or apply vendor sensitivities implicitly",
        ]
        if dry_elements:
            decisions.insert(0, f"Resolved dry-weight elements: {', '.join(dry_elements)}")
        if yield_products:
            decisions.append(f"Resolved relative-yield products: {', '.join(yield_products)}")
        if result.warnings:
            decisions.append(
                f"Completed with {len(result.warnings)} unit/provenance warning(s)"
            )

        return StepResult(
            step="elemental_spectroscopy",
            success=True,
            method="processed_elemental_spectroscopy",
            input_curves=result.input_curves,
            output_curves=result.output_curves,
            parameters={
                "sources": result.sources,
                "conversions": result.conversions,
                "dry_weight_elements": dry_elements,
                "relative_yields": yield_products,
                "derived": sorted(result.derived),
            },
            decisions=decisions,
            message="; ".join(result.warnings),
        )

    def _get_temperature_profile(self, well: "Well") -> np.ndarray:
        """Return a formation-temperature array in Fahrenheit."""
        temp_curve = well.get("TEMP")
        if temp_curve is not None:
            temp = self._temperature_curve_f(temp_curve)
            if temp.size == well.n_samples:
                return temp

        if well.depth is not None:
            from petrophys.interpretation.temperature import calculate_temperature_profile

            try:
                surface_temp_f = (
                    self.config.surface_temp_f
                    if self.config.surface_temp_f is not None
                    else 68.0
                )
                result = calculate_temperature_profile(
                    well,
                    bht=self.config.bht_f if self.config.bht_f is not None else _metadata_bht_f(well),
                    surface_temp=surface_temp_f,
                    bht_unit="F",
                    surface_temp_unit="F",
                    td=self.config.td_ft,
                    td_unit="ft" if self.config.td_ft is not None else None,
                    depth_unit=well.depth_unit,
                )
                return np.asarray(result.temperature, dtype=float)
            except Exception:
                pass

        avg_temp = self._get_avg_formation_temp(well)
        return np.full(well.n_samples, avg_temp, dtype=float)

    def _temperature_curve_f(self, curve: Any) -> np.ndarray:
        """Temperature curve values in degF.

        A curve labelled in degC or K is converted; the "geolink" profile reads
        the values as degF whatever the label.
        """
        values = np.asarray(curve.data, dtype=float)
        if not self._reproducing_benchmark and temperature_unit(curve.unit) in ("C", "K"):
            values = UnitConverter.temperature(values, curve.unit, "F")
        return values

    def _reference_temperature(self, temp_profile: np.ndarray) -> float:
        """Choose a reference temperature for scalar resistivity inputs."""
        finite = temp_profile[np.isfinite(temp_profile)]
        if finite.size == 0:
            return 150.0
        return float(np.nanmedian(finite))

    def _vcl_nd_matrix(self) -> str:
        """Matrix whose neutron-density clean line VCL_ND is measured from."""
        from petrophys.interpretation.clay_volume.nd_method import nd_clean_line

        matrix = str(self.config.matrix_type or "auto").strip().lower()
        if matrix == "auto":
            # Not resolved yet (the run settles it after the mineral solve):
            # the domain's own matrix, else sandstone, the matrix the porosity
            # step and the run's default resolution use for an unresolved
            # "auto". The end of the run warns when the resolved matrix has
            # another clean line.
            domain = str(self.config.lithology_domain).strip().lower()
            matrix = _DOMAIN_MATRICES.get(domain, ("sandstone",))[0]
        if nd_clean_line(matrix) == nd_clean_line("limestone"):
            # Calcite, and matrices without a clean line of their own.
            return "limestone"
        return matrix

    def _nd_clay_point(
        self, well: "Well", vcl_gr: Optional[np.ndarray], matrix: str
    ) -> tuple:
        """The neutron-density clay point of VCL_ND, and whether VCL_ND joins VCL.

        ``nd_clay`` when set; otherwise the median RHOB and NPHI of the well's
        good-hole shale samples: the shale-lithology samples when a lithology
        log flags at least 20 with both logs, else the samples with VCL_GR
        above 0.9, the shale control behind the clay porosity and Rcl.
        Bad-hole samples are left out, since washouts read low density and
        high neutron. With fewer than 20 samples on both bases the default
        point (2.62 g/cm3, 0.38) is kept. A point, set or picked, less than
        0.10 v/v of neutron above the clean line cannot tell shale from clean
        rock: VCL_ND is then computed on the default point, reported, and left
        out of VCL (``nd_in_vcl`` False).

        The pick is one median over the whole log, so where most VCL_GR > 0.9
        samples are shallow, undercompacted shale, the point is lighter than
        the shales of a deeper reservoir section; ``nd_clay_depth_range``
        gives the depths of the picked samples.

        Returns
        -------
        tuple
            ``((rhob, nphi), info)``: the point VCL_ND is computed on and the
            trace parameters.
        """
        from petrophys.interpretation.clay_volume.nd_method import ND_CLAY_POINT, nd_clay_separation

        info: Dict[str, Any] = {
            "nd_clay_source": "default",
            "nd_clay_basis": None,
            "nd_clay_samples": 0,
            "nd_clay_depth_range": None,
            "nd_clay_picked": None,
            "nd_clay_fallback_reason": None,
            "nd_in_vcl": True,
        }
        if self.config.nd_clay is not None:
            point = (float(self.config.nd_clay[0]), float(self.config.nd_clay[1]))
            info.update(nd_clay_source="config", nd_clay_basis="nd_clay")
        else:
            candidates = []
            lithology = self._build_lithology_log(well)
            if isinstance(lithology, np.ndarray):
                flags = np.asarray([_is_shale_group(item) for item in lithology], dtype=bool)
                candidates.append((flags, "lithology_shale", "shale-lithology"))
            if vcl_gr is not None:
                values = np.asarray(vcl_gr, dtype=float)
                candidates.append((np.isfinite(values) & (values > 0.9), "vcl_gr_gt_0.9", "VCL_GR > 0.9"))
            if not candidates:
                info["nd_clay_fallback_reason"] = "no shale lithology or gamma-ray clay volume to pick shale samples from"
                return ND_CLAY_POINT, info
            usable = np.ones(well.n_samples, dtype=bool)
            bad_hole = well.get("BAD_HOLE")
            if bad_hole is not None:
                usable &= ~(np.asarray(bad_hole.data, dtype=float) > 0.5)
            nphi, _, _ = get_curve_data_depthwise(well, ["NPHI", "PHIN"])
            rhob, _, _ = get_curve_data_depthwise(well, ["RHOB"])
            usable &= np.isfinite(nphi) & np.isfinite(rhob)
            reasons = []
            for shale, basis, label in candidates:
                picked = shale & usable
                count = int(np.count_nonzero(picked))
                info.update(nd_clay_basis=basis, nd_clay_samples=count)
                if count >= 20:
                    break
                reasons.append(f"{count} good-hole {label} samples have neutron and density; 20 are needed")
            info["nd_clay_fallback_reason"] = "; ".join(reasons) or None
            if count < 20:
                return ND_CLAY_POINT, info
            point = (float(np.median(rhob[picked])), float(np.median(nphi[picked])))
            depth = np.asarray(well.depth, dtype=float)[picked]
            info.update(
                nd_clay_source="shale_samples",
                nd_clay_depth_range=[float(depth.min()), float(depth.max())],
            )
        info["nd_clay_picked"] = list(point)
        separation = nd_clay_separation(point[0], point[1], matrix)
        if separation < 0.10:
            info.update(
                nd_clay_source="default",
                nd_in_vcl=False,
                nd_clay_fallback_reason=(
                    f"the clay point ({point[0]:.3f} g/cm3, {point[1]:.3f} v/v) lies "
                    f"{separation:.3f} v/v above the {matrix} clean line, less than the 0.10 needed"
                ),
            )
            return ND_CLAY_POINT, info
        return point, info

    def _check_sp_baselines(
        self, well: "Well", gr_curve: Any, sp_min: float, sp_max: float, decisions: List[str]
    ) -> str:
        """Check the order of the SP baselines against the SP polarity.

        A normal SP (Rmf > Rw) reads higher in shales, so its shale baseline
        ``sp_min`` lies above the clean-sand line ``sp_max``; a reversed SP
        the other way round. An SP whose polarity the logs do not show clearly
        is taken as normal, the usual case. A contradicting order is warned
        about and the values are still used as given.

        Returns
        -------
        str
            ``"normal"``, ``"reversed"`` or ``"unclear"``.
        """
        from petrophys.interpretation.clay_volume.sp_method import sp_polarity

        sp_curve = get_curve(well, ["SP"])
        polarity = 0
        if sp_curve is not None and gr_curve is not None and well.depth is not None:
            unit = str(getattr(well, "depth_unit", "") or "").strip().lower()
            window = 100.0 if unit in {"ft", "feet", "foot", "f"} else 30.0
            polarity = sp_polarity(sp_curve.data, gr_curve.data, well.depth, window=window)
        label = {1: "normal", -1: "reversed", 0: "unclear"}[polarity]
        reversed_sp = polarity < 0
        consistent = sp_min < sp_max if reversed_sp else sp_min > sp_max
        decisions.append(
            f"SP polarity: {label}" + (" (taken as normal)" if polarity == 0 else "")
            + f"; the baselines' order {'matches' if consistent else 'contradicts'} it"
        )
        if not consistent:
            kind = "reversed" if reversed_sp else "normal"
            evidence = "as the SP and GR show" if polarity else "assumed: the logs do not show the polarity clearly"
            relation = "equals" if sp_max == sp_min else ("lies above" if sp_max > sp_min else "lies below")
            self._result.warnings.append(
                "sp_min is the shale baseline and sp_max the clean-sand line, which lies "
                f"{'above' if reversed_sp else 'below'} the shale baseline for a {kind} SP "
                f"({evidence}). Here sp_max={sp_max:g} mV {relation} sp_min={sp_min:g} mV, so "
                "VCL_SP is inverted, high in clean sand and low in shale: check whether the two "
                "values are swapped."
            )
        return label

    def _step_vclay(self, well: "Well") -> StepResult:
        from petrophys.interpretation.clay_volume.vcl_module import VclModule
        from petrophys.interpretation.clay_volume.nd_method import ND_CLAY_POINT, nd_clean_line

        trace = StepResult(step="vclay", success=True)
        decisions = []
        gr_min = self.config.gr_min
        gr_max = self.config.gr_max
        gr_min_source = "user" if gr_min is not None else "auto"
        gr_max_source = "user" if gr_max is not None else "auto"

        # The GeoLink training package read GR alone, without the other
        # gamma-ray curves where GR has no value.
        train_gr = ("GR",) if self._dataset == "geolink_train" else None
        gr_candidates = list(train_gr or ("GR", "CGR", "SGR", "HSGR", "HCGR", "CGR_CALC"))
        gr_curve = get_curve(well, gr_candidates)
        gr_curve_name = gr_curve.mnemonic if gr_curve is not None else None

        if gr_min is None or gr_max is None:
            if gr_curve is not None:
                valid_gr = gr_curve.data[~np.isnan(gr_curve.data)]
                if len(valid_gr) > 0:
                    if gr_min is None:
                        gr_min = float(np.percentile(valid_gr, 5))
                    if gr_max is None:
                        gr_max = float(np.percentile(valid_gr, 95))

        # Use `is None` (not `or`) so a legitimately computed gr_min of 0.0 is kept.
        if gr_min is None:
            gr_min = 20.0
        if gr_max is None:
            gr_max = 100.0

        use_sp = (self._has_curve(well, "SP") and
                  not self.config.mud_has_clay_or_kcl)
        use_nd = self._has_curve(well, "NPHI") and self._has_curve(well, "RHOB")
        sp_min = self.config.sp_min
        sp_max = self.config.sp_max
        sp_min_source = "user" if sp_min is not None else "auto"
        sp_max_source = "user" if sp_max is not None else "auto"

        if use_sp:
            sp_curve = get_curve(well, ["SP"])
            if sp_curve is not None and (sp_min is None or sp_max is None):
                valid_sp = sp_curve.data[~np.isnan(sp_curve.data)]
                if len(valid_sp) > 0:
                    if sp_max is None:
                        sp_max = float(np.percentile(valid_sp, 5))
                    if sp_min is None:
                        sp_min = float(np.percentile(valid_sp, 95))

        if self._reproducing_benchmark:
            gr_min_label = f"{gr_min_source}, P5 of GR"
            gr_max_label = f"{gr_max_source}, P95 of GR"
        else:
            # Only auto-picked values come from the GR percentiles.
            gr_min_label = "auto, P5 of GR" if gr_min_source == "auto" else "user"
            gr_max_label = "auto, P95 of GR" if gr_max_source == "auto" else "user"
        decisions.append(f"GR_min = {gr_min:.1f} ({gr_min_label})")
        decisions.append(f"GR_max = {gr_max:.1f} ({gr_max_label})")
        if gr_curve is not None:
            decisions.append(f"Gamma-ray clay input: {gr_curve_name}")
        decisions.append(f"GR method: {self.config.gr_method}")
        decisions.append(f"SP method: {'enabled' if use_sp else 'disabled'}"
                        + (f" (mud has clay/KCl)" if self.config.mud_has_clay_or_kcl else ""))
        decisions.append(f"N-D method: {'enabled' if use_nd else 'disabled (missing NPHI or RHOB)'}")
        sp_baselines = use_sp and sp_max is not None and sp_min is not None
        if sp_baselines and self._reproducing_benchmark:
            decisions.append(f"SP_clean = {sp_max:.1f} ({sp_max_source}, P5 of SP)")
            decisions.append(f"SP_shale = {sp_min:.1f} ({sp_min_source}, P95 of SP)")
        elif sp_baselines:
            decisions.append(
                f"SP_clean = {sp_max:.1f} ({'user' if sp_max_source == 'user' else 'auto, P5 of SP'})"
            )
            decisions.append(
                f"SP_shale = {sp_min:.1f} ({'user' if sp_min_source == 'user' else 'auto, P95 of SP'})"
            )

        # Select the lowest valid clay estimate from the available methods at
        # each depth, consistent with the workflow's VCL convention.
        vcl_combine = "min"
        sp_in_vcl = use_sp
        polarity = None
        if self._reproducing_benchmark:
            # The canonical context calculation uses the fixed limestone N-D clean
            # line while the downstream porosity matrix remains configurable.
            vcl_matrix = "limestone"
        else:
            vcl_matrix = self._vcl_nd_matrix()
            if sp_baselines and "user" in (sp_min_source, sp_max_source):
                polarity = self._check_sp_baselines(well, gr_curve, sp_min, sp_max, decisions)
            # Baselines picked over the whole log mostly measure the SP baseline
            # drift (Schlumberger, 1989), so only user-set baselines put SP in
            # the minimum; otherwise VCL_SP is reported and fills only depths
            # that have no GR or N-D value.
            sp_in_vcl = use_sp and sp_min_source == sp_max_source == "user"
        # The 20/100 bound guards the P5/P95 picks against a nearly flat GR;
        # a baseline set by the user, the header or a zone is used as given.
        bound_gr = [name for name, source in (("gr_min", gr_min_source), ("gr_max", gr_max_source))
                    if source == "auto"]
        if self._reproducing_benchmark:
            from petrophys.model.constants import DefaultParameters

            # The "geolink" profile bounded every baseline, and let the log
            # extend one equal to 20 or 100 API, as it treated those as unset.
            bound_gr = ["gr_min", "gr_max"]
            gr_min = None if gr_min == DefaultParameters.GR_MIN else gr_min
            gr_max = None if gr_max == DefaultParameters.GR_MAX else gr_max
        vcl_module = VclModule(
            bound_gr=bound_gr,
            gr_min=gr_min, gr_max=gr_max,
            gr_method=self.config.gr_method,
            use_sp=use_sp, use_nd=use_nd,
            sp_min=sp_min, sp_max=sp_max,
            matrix=vcl_matrix,
            combine=vcl_combine,
            rho_fl=self.config.rho_fl,
            sp_fallback_only=not sp_in_vcl,
            nd_triangle_only=not self._reproducing_benchmark,
            gr_curves=train_gr,
            check_units=False,  # the normalization step converts units
        )
        curve_mapping = {"gr": gr_curve_name} if gr_curve_name else None
        result = vcl_module.calculate(well, curves=curve_mapping)
        nd_clay = ND_CLAY_POINT
        nd_clay_info: Dict[str, Any] = {}
        nd_in_vcl = use_nd
        if not self._reproducing_benchmark and result.vcl_nd is not None:
            nd_clay, nd_clay_info = self._nd_clay_point(well, result.vcl_gr, vcl_matrix)
            reason = nd_clay_info["nd_clay_fallback_reason"]
            if nd_clay_info["nd_clay_source"] != "default":
                vcl_module.nd_clay = nd_clay
                result = vcl_module.calculate(well, curves=curve_mapping)
            elif not nd_clay_info["nd_in_vcl"] and (result.vcl_gr is not None or result.vcl_sp is not None):
                # The well's clay point lies on the clean line, so neutron-density
                # cannot separate clay here: VCL_ND is reported only.
                reported_nd = result.vcl_nd
                vcl_module.use_nd = False
                result = vcl_module.calculate(well, curves=curve_mapping)
                result.vcl_nd = reported_nd
                nd_in_vcl = False
                self._result.warnings.append(
                    f"VCL_ND is reported but left out of VCL: {reason}, so neutron-density cannot "
                    "tell shale from clean rock in this well. Check the NPHI and RHOB logs (units, "
                    "environmental corrections)."
                )
            elif not nd_clay_info["nd_in_vcl"]:
                nd_clay_info["nd_in_vcl"] = True
                self._result.warnings.append(
                    "VCL comes from VCL_ND alone, on the default clay point (2.62 g/cm3, 0.38), "
                    f"because {reason}. Check the NPHI and RHOB logs (units, environmental "
                    "corrections)."
                )
        if result.gr_min_used is not None:
            decisions.append(
                f"Applied GR endpoints: {result.gr_min_used:g} to {result.gr_max_used:g} API"
            )
        if (
            not self._reproducing_benchmark
            and result.gr_min_used is not None
            and gr_min_source != gr_max_source
            and result.gr_min_used >= result.gr_max_used
        ):
            # One baseline set, the other picked from the log, in the wrong order.
            given, picked = ("gr_min", "gr_max") if gr_min_source == "user" else ("gr_max", "gr_min")
            used = {"gr_min": result.gr_min_used, "gr_max": result.gr_max_used}
            how = (
                "P5 of the GR log, lowered to 20 API when higher" if picked == "gr_min"
                else "P95 of the GR log, raised to 100 API when lower"
            )
            self._result.warnings.append(
                f"vclay: {given} = {used[given]:g} API (set) is {'at or above' if given == 'gr_min' else 'at or below'} "
                f"the {picked} picked automatically, {used[picked]:g} API ({how}), so VCL_GR is 0 or 1 at "
                f"almost every depth and is not a clay volume. Set {picked} as well, or check {given} "
                "(gr_min is the clean-rock GR, gr_max the shale GR)."
            )
        if result.gr_bounded and not self._reproducing_benchmark:
            moved = []
            for name, (picked, used) in result.gr_bounded.items():
                label = "clean" if name == "gr_min" else "shale"
                moved.append(f"the {label} GR pick {picked:.1f} API was {'lowered' if used < picked else 'raised'} to {used:g}")
            note = "; ".join(moved)
            decisions.append(f"GR bound (nearly flat GR guard): {note}")
            # Clean rock reads about 15-30 API (Crain); a clean pick above 30
            # suggests radioactive clean rock, which the bound reads as clay.
            if result.gr_bounded.get("gr_min", (0.0,))[0] > 30.0:
                picked = result.gr_bounded["gr_min"][0]
                self._result.warnings.append(
                    f"vclay: the clean GR pick {picked:.1f} API (P5) was lowered to 20, so GR reads "
                    "rock at that level as clay. If clean rock here is radioactive (feldspathic or "
                    "volcanic sands), set gr_min from a known clean sand (per zone with GR_MIN_Z1); "
                    "a value you set is used as given."
                )

        inputs = []
        if gr_curve is not None:
            inputs.append(curve_source_mnemonic(gr_curve) or gr_curve_name)
        outputs = []
        if use_sp: inputs.append("SP")
        if use_nd: inputs.extend(["NPHI", "RHOB"])
        well.add_curve("VCL", result.vcl, unit="v/v", description="Clay Volume")
        well.add_curve("VCLAY", result.vcl, unit="v/v", description="Clay Volume")
        outputs.extend(["VCL", "VCLAY"])

        if result.vcl_gr is not None:
            well.add_curve("VCL_GR", result.vcl_gr, unit="v/v", description="VCL from GR")
            well.add_curve("VCLGR", result.vcl_gr, unit="v/v", description="VCL from GR")
            outputs.extend(["VCL_GR", "VCLGR"])
        if result.vcl_sp is not None:
            well.add_curve("VCL_SP", result.vcl_sp, unit="v/v", description="VCL from SP")
            well.add_curve("VCLSP", result.vcl_sp, unit="v/v", description="VCL from SP")
            outputs.extend(["VCL_SP", "VCLSP"])
            if not sp_in_vcl:
                # Reported only: keeps it out of the VCL an export recombines.
                for name in ("VCL_SP", "VCLSP"):
                    well.get(name).metadata.custom["in_vcl"] = False
        if result.vcl_nd is not None:
            well.add_curve("VCL_ND", result.vcl_nd, unit="v/v", description="VCL from N-D")
            well.add_curve("VCLND", result.vcl_nd, unit="v/v", description="VCL from N-D")
            outputs.extend(["VCL_ND", "VCLND"])
            if not nd_in_vcl:
                # Reported only: keeps it out of the VCL an export recombines.
                for name in ("VCL_ND", "VCLND"):
                    well.get(name).metadata.custom["in_vcl"] = False

        nd_clean1_density, nd_clean1_neutron, nd_clean2_density, nd_clean2_neutron = (
            nd_clean_line(vcl_matrix)
        )
        vcl_range = f"{np.nanmin(result.vcl):.3f} - {np.nanmax(result.vcl):.3f}"
        if self._reproducing_benchmark:
            decisions.append(f"Final VCL = min(available methods), range: {vcl_range}")
            decisions.append("Canonical limestone N-D endpoints used")
        else:
            combined = [
                name for name, values in (("GR", result.vcl_gr), ("SP", result.vcl_sp), ("ND", result.vcl_nd))
                if values is not None and (name != "SP" or sp_in_vcl) and (name != "ND" or nd_in_vcl)
            ] or ["SP"]
            rule = f"min({', '.join(combined)})" if len(combined) > 1 else f"VCL_{combined[0]}"
            decisions.append(f"Final VCL = {rule} at each depth, range: {vcl_range}")
            if result.vcl_sp is not None and not sp_in_vcl:
                decisions.append(
                    "VCL_SP is reported but left out of the minimum: SP baselines picked over "
                    "the whole log mostly measure SP baseline drift, so it only fills depths with "
                    "no GR or N-D value. Set both sp_min and sp_max to include it; they apply to "
                    "the whole log, so only for an SP without baseline drift"
                )
            if result.vcl_nd is not None:
                decisions.append(
                    f"N-D clean line ({vcl_matrix}): RHOB {nd_clean1_density:.2f} g/cm3 at NPHI "
                    f"{nd_clean1_neutron:.2f} to RHOB {nd_clean2_density:.2f} g/cm3 at NPHI "
                    f"{nd_clean2_neutron:.2f}"
                )
                source = nd_clay_info.get("nd_clay_source")
                if source == "shale_samples":
                    picked = (
                        "shale-lithology" if nd_clay_info["nd_clay_basis"] == "lithology_shale"
                        else "VCL_GR > 0.9"
                    )
                    top, bottom = nd_clay_info["nd_clay_depth_range"]
                    decisions.append(
                        f"N-D clay point: RHOB {nd_clay[0]:.3f} g/cm3, NPHI {nd_clay[1]:.3f}, the "
                        f"median of {nd_clay_info['nd_clay_samples']} good-hole {picked} samples "
                        f"at {top:g}-{bottom:g} {well.depth_unit or 'm'}"
                        + (f" ({nd_clay_info['nd_clay_fallback_reason']})" if nd_clay_info["nd_clay_fallback_reason"] else "")
                    )
                elif source == "config":
                    decisions.append(f"N-D clay point: RHOB {nd_clay[0]:.3f} g/cm3, NPHI {nd_clay[1]:.3f} (nd_clay)")
                elif nd_in_vcl:
                    decisions.append(
                        f"N-D clay point: default RHOB {nd_clay[0]:.2f} g/cm3, NPHI {nd_clay[1]:.2f} "
                        f"({nd_clay_info.get('nd_clay_fallback_reason')})"
                    )
                else:
                    decisions.append(
                        f"N-D clay point: default RHOB {nd_clay[0]:.2f} g/cm3, NPHI {nd_clay[1]:.2f} for the "
                        f"reported VCL_ND, which stays out of VCL ({nd_clay_info['nd_clay_fallback_reason']})"
                    )
                nphi, _, _ = get_curve_data_depthwise(well, ["NPHI", "PHIN"])
                rhob, _, _ = get_curve_data_depthwise(well, ["RHOB"])
                limit = max(nd_clean2_neutron, nd_clay[1])
                with np.errstate(invalid="ignore"):
                    outside = int(np.count_nonzero(np.isfinite(rhob) & (np.asarray(nphi, dtype=float) > limit)))
                nd_clay_info["nd_outside_triangle_samples"] = outside
                if outside:
                    decisions.append(
                        f"VCL_ND undefined at {outside} samples with NPHI above {limit:.3f}, beyond the "
                        "clean line's end point and the clay point (outside the neutron-density "
                        "triangle); VCL there comes from the other indicators"
                    )

        trace.method = self.config.gr_method
        trace.input_curves = inputs
        trace.output_curves = outputs
        trace.parameters = {
            "gr_min": gr_min, "gr_max": gr_max,
            "gr_min_used": result.gr_min_used, "gr_max_used": result.gr_max_used,
            **({} if self._reproducing_benchmark else {
                "gr_bounded": {name: list(pair) for name, pair in result.gr_bounded.items()},
            }),
            "gr_curve": gr_curve_name,
            "gr_method": self.config.gr_method,
            "use_sp": use_sp, "use_nd": use_nd,
            "sp_min": sp_min, "sp_max": sp_max,
            "sp_min_source": sp_min_source if use_sp else None,
            "sp_max_source": sp_max_source if use_sp else None,
            "nd_matrix": vcl_matrix,
            "nd_clean1_density": nd_clean1_density,
            "nd_clean1_neutron": nd_clean1_neutron,
            "nd_clean2_density": nd_clean2_density,
            "nd_clean2_neutron": nd_clean2_neutron,
            "nd_clay_density": nd_clay[0],
            "nd_clay_neutron": nd_clay[1],
            "active_vcl_source": "workflow_calculation",
            "computed_vcl_curve": "VCL",
            "lithology_domain": self.config.lithology_domain,
        }
        if not self._reproducing_benchmark:
            trace.parameters.update({
                "sp_in_vcl": sp_in_vcl if use_sp else None,
                "sp_polarity": polarity,
                **nd_clay_info,
            })
        trace.decisions = decisions
        return trace

    def _step_rcl(self, well: "Well") -> StepResult:
        from petrophys.interpretation.clay_volume.rcl import RclModule

        trace = StepResult(step="rcl", success=True, method="median_Rt_in_shale_zones")
        rdep_curve = get_curve(well, ["RDEP", "RT"])
        vcl_curve = get_curve(well, ["VCL", "VCLAY"])
        trace.input_curves = [
            curve.mnemonic for curve in (rdep_curve, vcl_curve) if curve is not None
        ]
        trace.parameters = {"vcl_threshold": 0.9, "default_rcl": 5.0, "statistic": "median"}

        lithology = self._build_lithology_log(well)
        shale_mask = None
        shale_groups: List[str] = []
        if isinstance(lithology, np.ndarray):
            shale_mask = np.asarray([_is_shale_group(item) for item in lithology], dtype=bool)
            shale_groups = _ordered_unique_values(
                str(item).upper() for item in lithology.tolist() if _is_shale_group(item)
            )

        rcl_module = RclModule(statistic="median")
        self._result.rcl_computed = rcl_module.calculate(
            well,
            vcl=vcl_curve.data if vcl_curve is not None else None,
            shale_mask=shale_mask,
        )
        shale_sample_count = 0
        source = "default"
        if rdep_curve is not None:
            rdep = np.asarray(rdep_curve.data, dtype=float)
            if shale_mask is not None:
                valid = shale_mask & np.isfinite(rdep)
                if np.any(valid):
                    shale_sample_count = int(np.count_nonzero(valid))
                    source = "lithology_shale_median_rt"
            if source == "default" and vcl_curve is not None:
                vcl = np.asarray(vcl_curve.data, dtype=float)
                valid = np.isfinite(rdep) & np.isfinite(vcl) & (vcl > 0.9)
                if np.any(valid):
                    shale_sample_count = int(np.count_nonzero(valid))
                    source = "vcl_gt_0.9_median_rt"

        trace.method = source
        trace.parameters["rcl_computed"] = self._result.rcl_computed
        trace.parameters["source"] = source
        trace.parameters["shale_sample_count"] = shale_sample_count
        trace.parameters["shale_lithology_groups"] = shale_groups
        if source == "lithology_shale_median_rt":
            trace.decisions.append(
                f"Rcl = {self._result.rcl_computed:.2f} ohm.m (median RT in shale lithology samples, n={shale_sample_count})"
            )
        elif source == "vcl_gt_0.9_median_rt":
            trace.decisions.append(
                f"Rcl = {self._result.rcl_computed:.2f} ohm.m (median RT where VCL > 0.9, n={shale_sample_count})"
            )
        else:
            trace.decisions.append(
                f"Rcl = {self._result.rcl_computed:.2f} ohm.m (default because no shale lithology or VCL > 0.9 samples were available)"
            )
        if self.config.rtsh is not None and not self._reproducing_benchmark:
            trace.parameters["overridden_by"] = {"rtsh": self.config.rtsh}
            trace.decisions.append(
                f"Saturation uses rtsh = {self.config.rtsh:g} ohm.m as set, not this Rcl"
            )
        return trace

    @staticmethod
    def _train_lithology_codes(well: "Well", result: Any) -> np.ndarray:
        """LITHO codes as the GeoLink training package inferred them.

        Every sample with RHOB and a neutron reading is classified by the log
        cutoff tree of :meth:`LithologyModule._classify_sample` from RHOB,
        NPHI, GR, the screened DTC, VCL and the solved mineral volumes of
        ``result``; any other sample is unknown (code 0).
        """
        from petrophys.interpretation.lithology.lithology_module import LithologyModule

        n_samples = well.n_samples

        def log(names: List[str]) -> np.ndarray:
            data, _, _ = get_curve_data_depthwise(well, names)
            return np.asarray(data, dtype=float) if data is not None else np.full(n_samples, np.nan)

        rhob, nphi, gr = log(["RHOB"]), log(["NPHI", "PHIN"]), log(["GR"])
        dtc = LithologyModule._screen_dtc(log(["DTC", "DT"]))
        vcl, _, _ = get_curve_data_depthwise(well, ["VCL", "VCLAY", "VSH"])
        vcl = np.asarray(vcl, dtype=float) if vcl is not None else np.zeros(n_samples)
        volumes = [
            np.asarray(values, dtype=float)
            for values in (result.v_quartz, result.v_calcite, result.v_dolomite, result.v_anhydrite)
        ]
        classifier = LithologyModule(lithology="auto")
        codes = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            name = "unknown"
            if np.isfinite(rhob[i]) and np.isfinite(nphi[i]):
                name = classifier._classify_sample(
                    rhob=float(rhob[i]),
                    nphi=float(nphi[i]),
                    gr=float(gr[i]),
                    dtc=float(dtc[i]),
                    vcl=float(vcl[i]),
                    v_quartz=float(volumes[0][i]),
                    v_calcite=float(volumes[1][i]),
                    v_dolomite=float(volumes[2][i]),
                    v_anhydrite=float(volumes[3][i]),
                )
            codes[i] = LithologyModule.LITHOLOGY_CODES.get(name, LithologyModule.LITHOLOGY_CODES["mixed"])
        return codes

    def _fill_without_nphi(self, well: "Well", result, trace: StepResult):
        """Solve with version 2 the samples version 1 could not, for want of NPHI, where PEF is logged.

        Version 2 solves porosity jointly, so it needs no NPHI: against vendor
        lithology logs it matched them where version 1 could not run (25/5-1:
        90 % of the samples without NPHI). Samples without PEF either stay
        unsolved: RHOB and sonic alone do not separate the minerals.
        """
        from petrophys.interpretation.lithology.lithology_module import LithologyModule

        def log(names):
            data, _, _ = get_curve_data_depthwise(well, names)
            return np.asarray(data, dtype=float) if data is not None else np.full(well.n_samples, np.nan)

        pef = LithologyModule._screen_pef(log(["PEF"]))
        fill = (
            ~np.asarray(result.solved, dtype=bool)
            & ~np.isfinite(log(["NPHI", "PHIN"]))
            & np.isfinite(log(["RHOB"]))
            & np.isfinite(pef)
        )
        if not fill.any():
            return result
        module = LithologyModule(
            lithology="auto", solver_version=2, screen_barite_pef=True,
            solver_options={"endpoint_overrides": self._solver_endpoints()},
        )
        other = module.calculate(well, samples=fill)
        fill &= np.asarray(other.solved, dtype=bool)
        if not fill.any():
            return result
        for name in ("v_quartz", "v_calcite", "v_dolomite", "v_anhydrite", "lithology_code", "lithology_name",
                     "qc_flag", "solved"):
            values = np.array(getattr(result, name), copy=True)
            values[fill] = np.asarray(getattr(other, name))[fill]
            setattr(result, name, values)
        trace.parameters["version2_samples_without_nphi"] = int(fill.sum())
        trace.decisions.append(
            f"{int(fill.sum())} samples without NPHI solved by the version 2 solver from RHOB and PEF "
            "(porosity solved jointly)"
        )
        return result

    def _step_lithology(self, well: "Well") -> StepResult:
        from petrophys.interpretation.lithology.lithology_module import LithologyModule

        trace = StepResult(step="lithology", success=True, method="multi_mineral_solver")
        trace.input_curves = ["RHOB", "VCL"]
        nphi_curve = get_curve(well, ["NPHI", "PHIN"])
        if nphi_curve is not None:
            trace.input_curves.append(nphi_curve.mnemonic)
        if self._has_curve(well, "PHIE"):
            trace.input_curves.append("PHIE")
        elif self._has_curve(well, "PHIT"):
            trace.input_curves.append("PHIT")
        if self._has_curve(well, "PEF"):
            trace.input_curves.append("PEF")
        if self._has_curve(well, "DTC"):
            trace.input_curves.append("DTC")

        carbonate = self.config.lithology_domain == "carbonate"
        # The COSTA calibration (GR-derived illite, written as VILL) and the
        # generic one are the same closed-form calcite-dolomite solve.
        costa_carbonate = carbonate and self._carbonate_calibration == "costa"
        weighted_rows = None
        if carbonate:
            from petrophys.interpretation.lithology.carbonate_minerals import (
                solve_carbonate_minerals,
            )

            if self.config.mineral_endpoints:
                self._warn_assumed(
                    "mineral_endpoints is not used in the carbonate domain: its calcite-dolomite solve "
                    "takes rho_fl, rho_cl and carbonate_calibration; the endpoints apply to the clastic "
                    "mineral solve."
                )
            if costa_carbonate:
                # The "geolink" profile keeps the calibration's own brine and illite.
                result = solve_carbonate_minerals(
                    well, "costa",
                    rho_fluid=None if self._reproducing_benchmark else self.config.rho_fl,
                    rho_clay=None if self._reproducing_benchmark else self.config.rho_cl,
                )
            else:
                result = solve_carbonate_minerals(
                    well, "generic", rho_fluid=self.config.rho_fl, rho_clay=self.config.rho_cl
                )
            selection = dict(result.mineral_selection)
            trace.method = selection.pop("method")
            selection.pop("lithology_domain")
            clay_curves = ["GR"] if costa_carbonate else str(selection["clay_source"]).split("+")
            trace.input_curves = ["RHOB", *clay_curves, *str(selection["porosity_curve"]).split("+")]
            trace.parameters.update(selection)
        else:
            solver_version = self.config.mineral_solver_version
            # The "geolink" profile solves the version 1 rows unweighted.
            unweighted = solver_version == 1 and self._reproducing_benchmark
            # Benchmark profiles use unweighted solver rows,
            # sonic standing in for PEF, PEF always used.
            litho_module = LithologyModule(
                lithology="auto",
                solver_version=solver_version,
                solver_options=(
                    {"row_uncertainty": None, "sonic_without_pef": True} if unweighted
                    else {"sonic_without_pef": True} if solver_version == 1 and self._reproducing_benchmark
                    else {"endpoint_overrides": self._solver_endpoints()}
                ),
                screen_barite_pef=not self._reproducing_benchmark,
            )
            # Without NPHI anywhere, outside the benchmarks, version 2 solves the
            # well from RHOB and PEF (porosity solved jointly).
            whole_well_v2 = solver_version == 1 and nphi_curve is None and not self._reproducing_benchmark
            if whole_well_v2:
                solver_version = 2
                litho_module = LithologyModule(
                    lithology="auto", solver_version=2, screen_barite_pef=True,
                    solver_options={"endpoint_overrides": self._solver_endpoints()},
                )
                trace.decisions.append(
                    "No NPHI: minerals solved by the version 2 solver from RHOB and PEF (porosity solved jointly)"
                )
            if solver_version == 1:
                if nphi_curve is None:
                    raise ValueError(
                        "The canonical clastic mineral inversion requires NPHI/PHIN"
                    )
                from petrophys.workflow._canonical_calculations import CALCULATION_REVISION

                trace.method = "canonical_multi_mineral_solver"
                trace.parameters["equation_revision"] = CALCULATION_REVISION
                weighted_rows = litho_module.solver.row_uncertainty
                if weighted_rows is not None:
                    trace.parameters["row_uncertainty"] = dict(weighted_rows)
            else:
                trace.method = "multi_mineral_solver"
            trace.parameters["mineral_solver_version"] = solver_version
            result = litho_module.calculate(well)
            for note in litho_module.notes:
                trace.decisions.append(note)
                self._result.warnings.append(f"lithology: {note}")
            if solver_version == 1 and not self._reproducing_benchmark:
                result = self._fill_without_nphi(well, result, trace)
        code_curve_available = self._lithology_code_curve(well) is not None
        input_lithology_available = self._selected_lithology_source == "external" or (
            self._selected_lithology_source == "vendor" and code_curve_available
        )
        if self._selected_lithology_source == "auto":
            input_lithology_available = code_curve_available or self.config.lithology_csv is not None
        write_lithology_code = not input_lithology_available
        well.add_curve("VQUAR", result.v_quartz, "v/v", "Quartz Volume")
        well.add_curve("VCALC", result.v_calcite, "v/v", "Calcite Volume")
        well.add_curve("VDOLO", result.v_dolomite, "v/v", "Dolomite Volume")
        well.add_curve("VANHY", result.v_anhydrite, "v/v", "Anhydrite Volume")
        if costa_carbonate:
            well.add_curve("VILL", result.v_clay, "v/v", "Illite Volume")
        if write_lithology_code:
            codes = (
                self._train_lithology_codes(well, result)
                if self._dataset == "geolink_train"
                else result.lithology_code
            )
            well.add_curve(
                "LITHO",
                codes.astype(float),
                "",
                "Lithology Code",
            )
        if write_lithology_code:
            self._lithology_log = None
            self._lithology_source = "config"
            self._computed_lithology_written = True

        trace.output_curves = ["VQUAR", "VCALC", "VDOLO", "VANHY"]
        if costa_carbonate:
            trace.output_curves.append("VILL")
        if write_lithology_code:
            trace.output_curves.append("LITHO")
        trace.parameters["lithology_curve_written"] = write_lithology_code
        trace.parameters["lithology_source_priority"] = (
            "multi_mineral_fallback" if write_lithology_code else "input_lithology"
        )
        trace.parameters["multi_mineral_solved_samples"] = int(np.count_nonzero(result.solved))
        trace.parameters["reduced_quartz_calcite_samples"] = int(
            np.count_nonzero(result.reduced_qc_fallback)
            if result.reduced_qc_fallback is not None else 0
        )
        trace.parameters["multi_mineral_qc_flagged_samples"] = int(np.count_nonzero(result.qc_flag))
        trace.parameters["multi_mineral_total_samples"] = int(well.n_samples)
        trace.parameters["lithology_domain"] = self.config.lithology_domain
        if write_lithology_code:
            trace.decisions.append(
                "No lithology curve or lithology intervals were available -> inferred LITHO from mineral volumes"
            )
            if self._dataset == "geolink_train":
                trace.decisions.append(
                    "LITHO from the log cutoff tree (RHOB, NPHI, GR, DTC, VCL and the mineral volumes) "
                    "at samples with RHOB and NPHI, else unknown (0); the later steps read their "
                    "lithology groups from it"
                )
        else:
            trace.decisions.append(
                "Input lithology was available -> mineral volumes computed, LITHO not inferred or overwritten"
            )
        if not carbonate:
            # The clastic solvers read PEF and DTC.
            trace.decisions.append(
                "PEF screened for plausibility; DTC used when PEF is absent or invalid"
                if self._reproducing_benchmark or carbonate
                else "PEF screened for plausibility; without PEF the minerals are solved from RHOB and NPHI "
                "(sonic is used beside PEF, not in its place)"
            )
        if result.reduced_qc_fallback is not None and np.any(result.reduced_qc_fallback):
            trace.decisions.append(
                "Underdetermined samples used fixed PHIE with a reduced quartz-calcite inversion"
            )
        if not carbonate:
            trace.decisions.append(
                "Canonical clastic inversion uses the preserved equation set and closure gate"
            )
        if weighted_rows is not None:
            units = {"unity": "v/v", "rhob": "g/cm3", "nphi": "v/v", "u": "b/cm3", "dt": "us/ft"}
            trace.decisions.append(
                "Equation rows weighted by nominal one-sigma tool uncertainty: "
                + ", ".join(f"{row} {sigma:g} {units[row]}" for row, sigma in weighted_rows.items())
            )
        if carbonate:
            clay = (
                f"illite {trace.parameters['clay_source']}" if costa_carbonate
                else f"clay = {trace.parameters['clay_source']} at {trace.parameters['clay_density']:g} g/cm3"
            )
            trace.decisions.append(
                f"Carbonate domain ({trace.parameters['carbonate_calibration']} calibration): "
                f"calcite and dolomite from RHOB and closure with {clay}, "
                f"{trace.parameters['porosity_curve']} porosity and "
                f"{trace.parameters['fluid_density']:g} g/cm3 pore fluid; samples whose RHOB "
                f"misfit exceeds {trace.parameters['density_tolerance']:g} g/cm3 are left unsolved"
            )
            # A porosity computed with an assumed matrix density carries that
            # matrix into the split, so say which density it was.
            porosity = str(trace.parameters["porosity_curve"])
            porosity_step = next(
                (step for step in reversed(self._result.steps_executed)
                 if step.step == "porosity" and step.success),
                None,
            )
            if porosity_step is not None and set(porosity.split("+")) <= set(porosity_step.output_curves):
                low = porosity_step.parameters.get("rho_ma_min")
                high = porosity_step.parameters.get("rho_ma_max")
                if low is not None and high is not None:
                    rho_ma = f"{low:g}" if low == high else f"{low:g}-{high:g}"
                    trace.decisions.append(
                        f"{porosity} was computed in the porosity step with rho_ma {rho_ma} g/cm3, "
                        "so the calcite-dolomite split is not independent of that matrix choice"
                    )
        trace.decisions.append(
            f"Multi-mineral solved {int(np.count_nonzero(result.solved))}/{well.n_samples} samples; "
            f"QC flagged {int(np.count_nonzero(result.qc_flag))}"
        )
        return trace

    def _step_rmf(self, well: "Well") -> StepResult:
        from petrophys.interpretation.electrical.rmf.rmf_module import RmfModule
        from petrophys.interpretation.electrical.rw.salinity_rw import (
            NACL_PER_CHLORIDE,
            brine_density,
            chlorides_to_nacl_ppm,
            rw_from_salinity,
        )
        from petrophys.interpretation.temperature import correct_resistivity_temperature

        trace = StepResult(step="rmf", success=True)
        decisions = []

        temp_profile = self._get_temperature_profile(well)
        reference_temp = self._reference_temperature(temp_profile)

        zone_only_rmf = self._zone_only_parameters(well, "RMF")

        def _finalize_rmf(
            method: str,
            values: Union[float, np.ndarray],
            parameters: Dict[str, Any],
            output_curves: Optional[List[str]] = None,
        ) -> StepResult:
            rmf_curve = np.asarray(values, dtype=float)
            if rmf_curve.ndim == 0:
                rmf_curve = np.full(well.n_samples, float(rmf_curve), dtype=float)
            if zone_only_rmf:
                # A zone-bounded RMF_Z<n> holds inside its zone; the whole-well
                # chain gives Rmf outside it.
                rmf_curve = rmf_curve.copy()
                applied = []
                for zone in zone_only_rmf:
                    mask = self._zone_mask(well, zone)
                    zone_temp_f = _parameter_temperature_f(well, f"RMFT_Z{zone['index']}")
                    if zone_temp_f is None:
                        zone_temp_f = _parameter_temperature_f(well, f"MFST_Z{zone['index']}")
                    if zone_temp_f is None:
                        zone_temp_f = DEFAULT_RESISTIVITY_REFERENCE_TEMP_F
                    rmf_curve[mask] = RmfModule().calculate(
                        rmf_surface=zone["value"], temp_measured=zone_temp_f,
                        temp_formation=temp_profile[mask],
                    ).rmf
                    applied.append({
                        "parameter": f"RMF_Z{zone['index']}", "rmf": zone["value"],
                        "measured_temp_f": zone_temp_f, "top": zone["top"], "bottom": zone["bottom"],
                        "samples": int(np.count_nonzero(mask)),
                    })
                    message = (
                        f"RMF_Z{zone['index']} = {zone['value']:g} ohm.m applies to "
                        f"{zone['top']:g}-{zone['bottom']:g} {well.depth_unit} only; outside that "
                        f"zone Rmf comes from the whole-well chain ({method})."
                    )
                    self._warn_assumed(message)
                    decisions.append(message)
                parameters = {**parameters, "zone_only_rmf": applied}
            self._result.rmf_computed = rmf_curve
            well.add_curve("RMF", rmf_curve, "ohm.m", "Mud Filtrate Resistivity")
            trace.method = method
            trace.output_curves = output_curves or ["RMF"]
            trace.parameters = {
                **parameters,
                "formation_temp_min_f": float(np.nanmin(temp_profile)),
                "formation_temp_max_f": float(np.nanmax(temp_profile)),
                "rmf_min": float(np.nanmin(rmf_curve)),
                "rmf_max": float(np.nanmax(rmf_curve)),
            }
            trace.decisions = decisions
            return trace

        if self.config.mud_type == "OBM":
            decisions.append("OBM detected -> Rmf set to non-conductive default and written as a constant curve")
            return _finalize_rmf(
                "OBM_default",
                np.full(well.n_samples, 200.0, dtype=float),
                {"rmf": 200.0, "mud_type": "OBM"},
            )

        config_rmf_temp_f = self._measurement_temp_f("rmf_meas_temp_f")
        rmf_meas_temp_f = (
            config_rmf_temp_f if config_rmf_temp_f is not None else DEFAULT_RESISTIVITY_REFERENCE_TEMP_F
        )
        if config_rmf_temp_f is None and well.header.rm_temp is not None:
            rmf_meas_temp_f = float(well.header.rm_temp)
        metadata_rmf_temp_f = _parameter_temperature_f(well, "RMFT")
        if metadata_rmf_temp_f is None and not self._reproducing_benchmark:
            # MFST: the mud filtrate sample temperature of standard log headers.
            metadata_rmf_temp_f = _parameter_temperature_f(well, "MFST")
        if metadata_rmf_temp_f is not None and config_rmf_temp_f is None:
            rmf_meas_temp_f = metadata_rmf_temp_f

        las_rmf = well.header.rmf
        if las_rmf is None:
            las_rmf = _parameter_float(well, "RMF")
        if las_rmf is None:
            las_rmf = self._zoned_constant(well, "RMF")

        config_rm_temp_f = self._measurement_temp_f("rm_meas_temp_f")
        rm_meas_temp_f = (
            config_rm_temp_f if config_rm_temp_f is not None else DEFAULT_RESISTIVITY_REFERENCE_TEMP_F
        )
        if config_rm_temp_f is None and well.header.rm_temp is not None:
            rm_meas_temp_f = float(well.header.rm_temp)
        metadata_rm_temp_f = _parameter_temperature_f(well, "RMT")
        if metadata_rm_temp_f is None and not self._reproducing_benchmark:
            # MST: the mud sample temperature of standard log headers.
            metadata_rm_temp_f = _parameter_temperature_f(well, "MST")
        if metadata_rm_temp_f is not None and config_rm_temp_f is None:
            rm_meas_temp_f = metadata_rm_temp_f

        las_rm = well.header.rm
        if las_rm is None:
            las_rm = _parameter_float(well, "RM")
        if las_rm is None:
            las_rm = self._zoned_constant(well, "RM")

        if self.config.rmf_at_meas_temp is not None:
            rmf_module = RmfModule()
            result = rmf_module.calculate(
                rmf_surface=self.config.rmf_at_meas_temp,
                temp_measured=rmf_meas_temp_f,
                temp_formation=temp_profile
            )
            decisions.append(f"Rmf corrected from {self.config.rmf_at_meas_temp:.3f} ohm.m "
                           f"@ {rmf_meas_temp_f:.0f}F across the temperature profile")
            return _finalize_rmf(
                "arps_temperature_profile_correction",
                result.rmf,
                {
                    "rmf_at_surface": self.config.rmf_at_meas_temp,
                    "measured_temp_f": rmf_meas_temp_f,
                },
            )

        def _from_header_rmf() -> StepResult:
            ignored = [
                name for name in ("rmf_salinity_ppm", "rm")
                if getattr(self.config, name) is not None
            ]
            if ignored and not self._historical:
                self._warn_assumed(
                    f"Rmf comes from the LAS header RMF; {' and '.join(ignored)} "
                    f"{'was' if len(ignored) == 1 else 'were'} not used. Set rmf_at_meas_temp "
                    "to override the header."
                )
            rmf_module = RmfModule()
            if (
                metadata_rmf_temp_f is not None
                or well.header.rm_temp is not None
                or config_rmf_temp_f is not None
            ):
                result = rmf_module.calculate(
                    rmf_surface=float(las_rmf),
                    temp_measured=rmf_meas_temp_f,
                    temp_formation=temp_profile,
                )
                decisions.append(
                    f"Rmf corrected from LAS metadata: {float(las_rmf):.4f} ohm.m "
                    f"@ {rmf_meas_temp_f:.0f}F across the temperature profile"
                )
                return _finalize_rmf(
                    "las_metadata_profile_temperature_correction",
                    result.rmf,
                    {
                        "rmf_at_measured_temp": float(las_rmf),
                        "measured_temp_f": rmf_meas_temp_f,
                    },
                )
            assumed_temp_f = DEFAULT_RESISTIVITY_REFERENCE_TEMP_F
            result = rmf_module.calculate(
                rmf_surface=float(las_rmf),
                temp_measured=assumed_temp_f,
                temp_formation=temp_profile,
            )
            decisions.append(
                f"Rmf from LAS metadata used assumed reference temperature {assumed_temp_f:.1f}F and was corrected along the temperature profile"
            )
            return _finalize_rmf(
                "las_metadata_assumed_reference_temperature",
                result.rmf,
                {
                    "rmf_at_reference_temp": float(las_rmf),
                    "reference_temp_f": assumed_temp_f,
                },
            )

        def _from_salinity() -> StepResult:
            rmf_at_meas_temp = float(rw_from_salinity(self.config.rmf_salinity_ppm, rmf_meas_temp_f))
            rmf_module = RmfModule()
            result = rmf_module.calculate(
                rmf_surface=rmf_at_meas_temp,
                temp_measured=rmf_meas_temp_f,
                temp_formation=temp_profile,
            )
            # The "geolink" profile fills rmf_salinity_ppm from the CLFL header.
            source = "CLFL salinity" if self._reproducing_benchmark else "mud-filtrate salinity"
            decisions.append(
                f"Rmf from {source}: {self.config.rmf_salinity_ppm:.0f} ppm -> "
                f"{rmf_at_meas_temp:.4f} ohm.m @ {rmf_meas_temp_f:.0f}F and then along the temperature profile"
            )
            return _finalize_rmf(
                "mud_filtrate_salinity_conversion",
                result.rmf,
                {
                    "mud_filtrate_salinity_ppm": self.config.rmf_salinity_ppm,
                    "rmf_at_measured_temp": rmf_at_meas_temp,
                    "measured_temp_f": rmf_meas_temp_f,
                },
            )

        def _from_chlorides() -> StepResult:
            # Mud reports titrate filtrate chlorides in mg/l; Bateman and Konen
            # (1977) take NaCl ppm by weight. Count the chloride as NaCl and
            # divide by the brine density (McCain, 1991).
            chlorides = float(self.config.rmf_chlorides_mg_l)
            if self.config.rm is not None and not self._historical:
                self._warn_assumed(
                    "Rmf comes from the mud-filtrate chlorides (rmf_chlorides_mg_l or header "
                    "CLFL); rm was not used. Set rmf_at_meas_temp to override them."
                )
            salinity_ppm = chlorides_to_nacl_ppm(chlorides)
            density = brine_density(salinity_ppm)
            rmf_75f = float(rw_from_salinity(salinity_ppm, DEFAULT_RESISTIVITY_REFERENCE_TEMP_F))
            result = RmfModule().calculate(
                rmf_surface=rmf_75f,
                temp_measured=DEFAULT_RESISTIVITY_REFERENCE_TEMP_F,
                temp_formation=temp_profile,
            )
            decisions.append(
                f"Rmf from mud-filtrate chlorides {chlorides:.0f} mg/l: NaCl equivalent "
                f"{chlorides * NACL_PER_CHLORIDE:.0f} mg/l (Cl x 58.44/35.45) = {salinity_ppm:.0f} ppm "
                f"at a brine density of {density:.4f} g/cm3 -> {rmf_75f:.4f} ohm.m @ "
                f"{DEFAULT_RESISTIVITY_REFERENCE_TEMP_F:.0f}F and then along the temperature profile"
            )
            return _finalize_rmf(
                "mud_filtrate_chlorides_conversion",
                result.rmf,
                {
                    "mud_filtrate_chlorides_mg_l": chlorides,
                    "mud_filtrate_salinity_ppm": salinity_ppm,
                    "brine_density_g_cm3": density,
                    "rmf_at_measured_temp": rmf_75f,
                    "measured_temp_f": DEFAULT_RESISTIVITY_REFERENCE_TEMP_F,
                },
            )

        def _from_rm(rm_input: float, rm_source: str) -> StepResult:
            k = 0.75 if self.config.mud_water == "fresh" else 0.85
            rm_curve = correct_resistivity_temperature(
                float(rm_input),
                rm_meas_temp_f,
                temp_profile,
                "F",
            )
            rmf_curve = np.asarray(rm_curve, dtype=float) * k
            well.add_curve("RM", rm_curve, "ohm.m", "Mud Resistivity")
            decisions.append(
                f"Rm from {rm_source} corrected from {float(rm_input):.4f} ohm.m "
                f"@ {rm_meas_temp_f:.0f}F across the temperature profile; Rmf = Rm(T) * {k}"
            )
            return _finalize_rmf(
                f"rm_to_rmf_conversion_factor_{k}",
                rmf_curve,
                {
                    "rm_at_measured_temp": float(rm_input),
                    "rm_measured_temp_f": rm_meas_temp_f,
                    "rm_min": float(np.nanmin(rm_curve)),
                    "rm_max": float(np.nanmax(rm_curve)),
                    "conversion_factor": k,
                    "mud_water": self.config.mud_water,
                    "source_priority": rm_source,
                },
                ["RM", "RMF"],
            )

        if self._historical:
            # The "geolink" profile: header RMF, then the salinity, chlorides
            # and Rm settings (config or header).
            if las_rmf is not None:
                return _from_header_rmf()
            if self.config.rmf_salinity_ppm is not None:
                return _from_salinity()
            if self.config.rmf_chlorides_mg_l is not None:
                return _from_chlorides()
            rm_input = self.config.rm if self.config.rm is not None else las_rm
            if rm_input is not None:
                return _from_rm(rm_input, "config.rm" if self.config.rm is not None else "LAS metadata")
            trace.decisions = decisions
            return trace

        # Explicit settings win over the header, as rw does over the header RW.
        # A setting that pp.interpret or the zoned runner copied from the
        # header (CLFL as rmf_chlorides_mg_l, RM as rm) ranks with the header.
        from petrophys.workflow.zoned_batch import _extract_global_parameter_overrides

        header_derived = _extract_global_parameter_overrides(
            well, calculation_profile=self.config.calculation_profile
        )[0]

        def _explicit(name: str) -> bool:
            value = getattr(self.config, name)
            if value is None:
                return False
            header_value = header_derived.get(name)
            if not self._reproducing_benchmark and name in (
                    getattr(self.config, "_explicit_settings", None) or frozenset()):
                return True  # the caller's own value, even when it equals the header's
            return header_value is None or not np.isclose(float(value), float(header_value))

        if _explicit("rmf_salinity_ppm"):
            return _from_salinity()
        if _explicit("rmf_chlorides_mg_l"):
            return _from_chlorides()
        if _explicit("rm"):
            return _from_rm(self.config.rm, "config.rm")
        if las_rmf is not None:
            for name, label in (("rmf_salinity_ppm", "salinity"), ("rmf_chlorides_mg_l", "CLFL")):
                if getattr(self.config, name) is not None:
                    decisions.append(
                        f"Header {label} ({name}={getattr(self.config, name):g}) not used: "
                        "the header RMF is the direct measurement"
                    )
            return _from_header_rmf()
        if self.config.rmf_salinity_ppm is not None:
            return _from_salinity()
        if self.config.rmf_chlorides_mg_l is not None:
            return _from_chlorides()
        rm_input = self.config.rm if self.config.rm is not None else las_rm
        if rm_input is not None:
            return _from_rm(rm_input, "config.rm" if self.config.rm is not None else "LAS metadata")
        if zone_only_rmf:
            decisions.append("No whole-well Rmf source: RMF is known inside the header zones only")
            return _finalize_rmf(
                "zone_only_header_rmf",
                np.full(well.n_samples, np.nan, dtype=float),
                {"source_priority": "zoned LAS RMF"},
            )

        trace.decisions = decisions
        return trace

    def _costa_saturation(
        self, well: "Well", computed_sw: np.ndarray, model: str
    ) -> Tuple[List[str], Optional[str], Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
        """Write SW and SO as the COSTA package's interpretation did.

        SW takes the source saturation (SW_INPUT, SW_RAW or SW_SOURCE) where it
        is finite and within 0-1, and the computed SW elsewhere; SO = 1 - SW
        wherever SW is finite.

        Returns
        -------
        tuple
            The output curves, the source curve's name and metadata, the
            samples it gave, the samples SO covers, and the SW written.
        """
        source_sw, source_sw_source, source_sw_meta = get_curve_data_depthwise(
            well, ["SW_INPUT", "SW_RAW", "SW_SOURCE"]
        )
        if source_sw is None:
            source_sw_mask = np.zeros(well.n_samples, dtype=bool)
        else:
            source_sw_mask = np.isfinite(source_sw) & (source_sw >= 0.0) & (source_sw <= 1.0)
        final_sw = computed_sw.copy()
        if source_sw is not None:
            final_sw[source_sw_mask] = source_sw[source_sw_mask]
        well.add_curve("SW", final_sw, "v/v", f"Water Saturation ({model}; source input priority)")
        so = np.full(final_sw.shape, np.nan, dtype=float)
        so_valid = np.isfinite(final_sw)
        so[so_valid] = 1.0 - final_sw[so_valid]
        well.add_curve("SO", so, "v/v", "Oil Saturation")
        return ["SW", "SO"], source_sw_source, dict(source_sw_meta or {}), source_sw_mask, so_valid, final_sw

    def _dataset_given_rw(
        self,
        well: "Well",
        temp_profile: np.ndarray,
        las_rw_temp_f: Optional[float],
        decisions: List[str],
        finalize: Callable[..., StepResult],
    ) -> Optional[StepResult]:
        """RW from a given Rw as the Danish and COSTA packages computed it, else None.

        Those packages corrected a configured or header Rw to the formation
        temperature with Arps directly, and took an Rw without its measurement
        temperature at 75 degF (:data:`DEFAULT_RESISTIVITY_REFERENCE_TEMP_F`).
        With a measurement temperature the result equals the salinity route of
        the GeoLink package. None when no Rw is given, so the shared Rw rules
        apply.
        """
        from petrophys.interpretation.electrical.rw.temp_correction import correct_rw_temperature

        if self.config.rw is not None:
            if self.config.rw_meas_temp_f is not None:
                decisions.append(
                    f"Rw directly corrected from {self.config.rw:.4f} ohm.m @ {self.config.rw_meas_temp_f:.0f}F "
                    f"with Arps along the temperature profile"
                )
                return finalize(
                    "user_provided_temperature_profile_corrected",
                    correct_rw_temperature(self.config.rw, self.config.rw_meas_temp_f, temp_profile, "F"),
                    {
                        "rw_at_measured_temp": self.config.rw,
                        "measured_temp_f": self.config.rw_meas_temp_f,
                        "source_priority": "config.rw",
                    },
                )
            assumed_temp_f = DEFAULT_RESISTIVITY_REFERENCE_TEMP_F
            decisions.append(
                f"Rw input {self.config.rw:.4f} ohm.m used assumed reference temperature {assumed_temp_f:.1f}F "
                "and was directly corrected with Arps along the temperature profile"
            )
            return finalize(
                "user_provided_assumed_reference_temperature",
                correct_rw_temperature(self.config.rw, assumed_temp_f, temp_profile, "F"),
                {
                    "rw": float(self.config.rw),
                    "reference_temp_f": assumed_temp_f,
                    "source_priority": "config.rw",
                },
            )

        las_rw = _parameter_float(well, "RW")
        if las_rw is None:
            las_rw = _constant_zoned_parameter(well, "RW")
        if las_rw is None:
            return None
        if las_rw_temp_f is not None:
            decisions.append(
                f"Rw directly corrected from LAS metadata: {float(las_rw):.4f} ohm.m @ {las_rw_temp_f:.0f}F "
                f"with Arps along the temperature profile"
            )
            return finalize(
                "las_metadata_temperature_profile_corrected",
                correct_rw_temperature(float(las_rw), las_rw_temp_f, temp_profile, "F"),
                {
                    "rw_at_measured_temp": float(las_rw),
                    "measured_temp_f": las_rw_temp_f,
                    "source_priority": "las.RW",
                },
            )
        assumed_temp_f = DEFAULT_RESISTIVITY_REFERENCE_TEMP_F
        decisions.append(
            f"Rw from LAS metadata used assumed reference temperature {assumed_temp_f:.1f}F "
            "and was directly corrected with Arps along the temperature profile"
        )
        return finalize(
            "las_metadata_assumed_reference_temperature",
            correct_rw_temperature(float(las_rw), assumed_temp_f, temp_profile, "F"),
            {
                "rw": float(las_rw),
                "reference_temp_f": assumed_temp_f,
                "source_priority": "las.RW",
            },
        )

    def _covering_zone_rw(
        self,
        well: "Well",
        header_rw: float,
        header_temp_f: Optional[float],
        decisions: List[str],
    ) -> Tuple[float, Optional[float]]:
        """The Rw (and its temperature) of a bounded ``RW_Z<n>`` zone that covers the whole well.

        :func:`~petrophys.workflow.zoned_batch.run_dynamic_workflow_with_zones`
        re-runs such a zone with its own Rw over every sample, so the single
        run takes it too, in place of the header RW, and warns where the two
        differ. The header values are returned when no zone covers the well.
        """
        covering = [
            zone for zone in self._bounded_zone_parameters(well, "RW")
            if self._zone_covers_well(well, zone)
        ]
        if len({zone["value"] for zone in covering}) != 1:
            return header_rw, header_temp_f
        zone = covering[0]
        zone_temp_f = _parameter_temperature_f(well, f"RWT_Z{zone['index']}")
        if zone_temp_f is None:
            zone_temp_f = header_temp_f
        if not np.isclose(zone["value"], header_rw):
            message = (
                f"RW_Z{zone['index']} = {zone['value']:g} ohm.m covers the whole well "
                f"({zone['top']:g}-{zone['bottom']:g} {well.depth_unit}) and differs from the header "
                f"RW {header_rw:g} ohm.m; the zone value is used, as run_dynamic_workflow_with_zones "
                "applies it."
            )
            decisions.append(message)
            self._warn_assumed(message)
        return zone["value"], zone_temp_f

    def _check_rw_against_rwa(self, well: "Well", rw_curve: np.ndarray, trace: StepResult) -> None:
        """Warn when clean porous rock implies a water resistivity well below the Rw in use.

        The apparent water resistivity ``Rwa = Rt * phi^m / a`` equals Rw in
        clean water-bearing rock and exceeds it where hydrocarbon is present,
        so the lowest Rwa of a well marks its Rw (the Rwa method; Asquith and
        Krygowski, 2004, Basic Well Log Analysis). Rwa is taken over the
        clean porous samples (VCL < 0.3, effective porosity > 0.05), each
        against the Rw of its own depth. Where its median is below
        ``_RWA_CHECK_RATIO`` of the Rw in use, Archie gives Sw above 1.4 in
        more than half of that rock, which scatter in the logs does not
        explain: the Rw is too high for that rock (water saltier than the Rw
        implies), or ``a`` and ``m`` do not fit it. The median and 5th
        percentile of Rwa / Rw are recorded in the trace
        (``rwa_median_over_rw``, ``rwa_p5_over_rw``).
        """
        rt_curve = get_curve(well, ["RDEP", "RT"])
        vcl_curve = get_curve(well, ["VCL", "VCLAY"])
        phie, _, _ = get_curve_data_depthwise(well, self._phie_names(include_phit=False))
        if rt_curve is None or vcl_curve is None or phie is None:
            return
        rdep = np.asarray(rt_curve.data, dtype=float)
        vcl = np.asarray(vcl_curve.data, dtype=float)
        phie = np.asarray(phie, dtype=float)
        archie_a, archie_m, _, _ = self._archie_coefficients_for_lithology(
            self._build_lithology_log(well), well.n_samples)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = rdep * np.power(phie, archie_m) / archie_a / np.asarray(rw_curve, dtype=float)
        clean = np.isfinite(ratio) & (ratio > 0) & (vcl < 0.3) & (phie > 0.05)
        if int(np.count_nonzero(clean)) < _RWA_CHECK_MIN_SAMPLES:
            return
        median = float(np.median(ratio[clean]))
        trace.parameters["rwa_median_over_rw"] = median
        trace.parameters["rwa_p5_over_rw"] = float(np.percentile(ratio[clean], 5))
        trace.parameters["rwa_check_samples"] = int(np.count_nonzero(clean))
        if median >= _RWA_CHECK_RATIO:
            return
        above_one = float(np.mean(ratio[clean] < 1.0))
        message = (
            f"The Rw in use looks too high for this well: over its clean porous rock (VCL < 0.3, porosity "
            f"> 0.05, {int(np.count_nonzero(clean))} samples) the median apparent water resistivity "
            f"Rwa = Rt x phi^m / a is {median:.2f} of it, and Archie gives Sw above 1 at "
            f"{100.0 * above_one:.0f}% of those samples, where SW is held at 1. Rwa is not below Rw in "
            "water-bearing rock, so the formation water is saltier than this Rw implies, or a and m do "
            "not fit the rock. Check rw (or salinity_ppm) against a water sample or the water leg."
        )
        trace.decisions.append(message)
        self._warn_assumed(message)

    @staticmethod
    def _train_rw_curve(
        estimates: np.ndarray,
        sample_temps: np.ndarray,
        temp_profile: np.ndarray,
        statistic: Callable[[np.ndarray], float],
    ) -> Tuple[float, float, np.ndarray]:
        """Rw along ``temp_profile`` from Rw estimates, as the GeoLink training package computed it.

        Each estimate is corrected with Arps from its sample's temperature
        (``sample_temps``) to the median of those temperatures, Tref;
        ``statistic`` of the corrected estimates is Rw at Tref, which Arps
        carries along the profile: RW(T) = RW(Tref) * (Tref + 6.77) / (T + 6.77)
        in degF. Returns (Tref, Rw at Tref, the Rw curve).
        """
        from petrophys.interpretation.electrical.rw.temp_correction import correct_rw_temperature

        reference_temp = float(np.nanmedian(sample_temps))
        at_reference = float(statistic(correct_rw_temperature(estimates, sample_temps, reference_temp, "F")))
        return reference_temp, at_reference, correct_rw_temperature(at_reference, reference_temp, temp_profile, "F")

    def _step_rw(self, well: "Well") -> StepResult:
        from petrophys.interpretation.electrical.rw.rw_module import RwModule
        from petrophys.interpretation.electrical.rw.salinity_rw import rw_from_salinity, salinity_from_rw
        from petrophys.interpretation.electrical.rw.sp_rw import estimate_sp_lines, rw_from_sp, rw_from_rxo_rt

        trace = StepResult(step="rw", success=True)
        decisions = []
        temp_profile = self._get_temperature_profile(well)
        reference_temp = self._reference_temperature(temp_profile)
        las_rw_temp_f = _parameter_temperature_f(well, "RWT")

        zone_only_rw = self._zone_only_parameters(well, "RW")

        def _finalize_rw(
            method: str,
            value: Union[float, np.ndarray],
            parameters: Dict[str, Any],
            inputs: Optional[List[str]] = None,
        ) -> StepResult:
            rw_curve = np.asarray(value, dtype=float)
            if rw_curve.ndim == 0:
                rw_curve = np.full(well.n_samples, float(rw_curve), dtype=float)
            if zone_only_rw:
                # A zone-bounded RW_Z<n> holds inside its zone; the whole-well
                # chain above gives Rw outside it.
                rw_curve = rw_curve.copy()
                applied = []
                for zone in zone_only_rw:
                    mask = self._zone_mask(well, zone)
                    zone_temp_f = _parameter_temperature_f(well, f"RWT_Z{zone['index']}")
                    if zone_temp_f is None:
                        zone_temp_f = las_rw_temp_f if las_rw_temp_f is not None else reference_temp
                    salinity = float(salinity_from_rw(zone["value"], zone_temp_f))
                    rw_curve[mask] = rw_from_salinity(salinity, temp_profile[mask])
                    applied.append({
                        "parameter": f"RW_Z{zone['index']}", "rw": zone["value"],
                        "measured_temp_f": zone_temp_f, "top": zone["top"], "bottom": zone["bottom"],
                        "samples": int(np.count_nonzero(mask)),
                    })
                    message = (
                        f"RW_Z{zone['index']} = {zone['value']:g} ohm.m applies to "
                        f"{zone['top']:g}-{zone['bottom']:g} {well.depth_unit} only; outside that "
                        f"zone Rw comes from the whole-well chain ({method})."
                    )
                    self._warn_assumed(message)
                    decisions.append(message)
                parameters = {**parameters, "zone_only_rw": applied}
            self._result.rw_computed = rw_curve
            well.add_curve("RW", rw_curve, "ohm.m", "Formation Water Resistivity")
            trace.method = method
            trace.parameters = {
                **parameters,
                "rw_min": float(np.nanmin(rw_curve)),
                "rw_max": float(np.nanmax(rw_curve)),
                "formation_temp_min_f": float(np.nanmin(temp_profile)),
                "formation_temp_max_f": float(np.nanmax(temp_profile)),
            }
            trace.input_curves = inputs or []
            trace.output_curves = ["RW"]
            trace.decisions = decisions
            if not self._historical and method != "rwa_clean_wet_p5":
                self._check_rw_against_rwa(well, rw_curve, trace)
            return trace

        if self._dataset is not None:
            dataset_trace = self._dataset_given_rw(well, temp_profile, las_rw_temp_f, decisions, _finalize_rw)
            if dataset_trace is not None:
                return dataset_trace

        if self.config.rw is not None:
            if self.config.rw_meas_temp_f is not None:
                salinity_est = float(salinity_from_rw(self.config.rw, self.config.rw_meas_temp_f))
                rw_curve = rw_from_salinity(salinity_est, temp_profile)
                decisions.append(
                    f"Rw corrected from {self.config.rw:.4f} ohm.m @ {self.config.rw_meas_temp_f:.0f}F "
                    f"along the temperature profile"
                )
                return _finalize_rw(
                    "user_provided_temperature_profile_corrected",
                    rw_curve,
                    {
                        "rw_at_measured_temp": self.config.rw,
                        "measured_temp_f": self.config.rw_meas_temp_f,
                        "salinity_ppm_estimated": salinity_est,
                        "source_priority": "config.rw",
                    },
                )
            salinity_est = float(salinity_from_rw(self.config.rw, reference_temp))
            rw_curve = rw_from_salinity(salinity_est, temp_profile)
            decisions.append(
                f"Rw input {self.config.rw:.4f} ohm.m used assumed reference temperature {reference_temp:.1f}F and was corrected along the temperature profile"
            )
            self._warn_assumed(
                f"Rw {self.config.rw:g} ohm.m was given without rw_meas_temp_f: it is taken at the "
                f"median formation temperature, {reference_temp:.1f} degF."
            )
            return _finalize_rw(
                "user_provided_assumed_reference_temperature",
                rw_curve,
                {
                    "rw": float(self.config.rw),
                    "reference_temp_f": reference_temp,
                    "salinity_ppm_estimated": salinity_est,
                    "source_priority": "config.rw",
                },
            )

        las_rw = _parameter_float(well, "RW")
        if las_rw is not None and not self._historical:
            las_rw, las_rw_temp_f = self._covering_zone_rw(well, las_rw, las_rw_temp_f, decisions)
        if las_rw is None:
            las_rw = self._zoned_constant(well, "RW")
        if las_rw is not None and self.config.salinity_ppm is not None and not self._historical:
            # An explicit salinity wins over the header, as rw does.
            decisions.append(
                f"salinity_ppm is set: the header Rw {float(las_rw):g} ohm.m is not used"
            )
            las_rw = None
        if las_rw is not None and self.config.rw_meas_temp_f is not None and not self._historical:
            # rw_meas_temp_f without rw is the measurement temperature of the header Rw.
            if las_rw_temp_f is None:
                decisions.append(
                    f"Header Rw has no RWT: rw_meas_temp_f = {self.config.rw_meas_temp_f:g} degF "
                    "is its measurement temperature"
                )
            elif float(las_rw_temp_f) != float(self.config.rw_meas_temp_f):
                decisions.append(
                    f"rw_meas_temp_f = {self.config.rw_meas_temp_f:g} degF replaces the header RWT "
                    f"{float(las_rw_temp_f):g} degF as the header Rw's measurement temperature"
                )
            las_rw_temp_f = float(self.config.rw_meas_temp_f)
        if las_rw is not None and las_rw_temp_f is None and not self._historical:
            # Operators quote Rw at formation temperature: FMT, when the header gives it.
            fmt_f = _parameter_temperature_f(well, "FMT")
            if fmt_f is not None:
                las_rw_temp_f = fmt_f
                decisions.append(
                    f"Header Rw has no RWT: taken at the header formation temperature FMT, {fmt_f:.1f} degF"
                )
        if las_rw is not None:
            if las_rw_temp_f is not None:
                salinity_est = float(salinity_from_rw(float(las_rw), las_rw_temp_f))
                rw_curve = rw_from_salinity(salinity_est, temp_profile)
                decisions.append(
                    f"Rw corrected from LAS metadata: {float(las_rw):.4f} ohm.m @ {las_rw_temp_f:.0f}F "
                    f"along the temperature profile"
                )
                return _finalize_rw(
                    "las_metadata_temperature_profile_corrected",
                    rw_curve,
                    {
                        "rw_at_measured_temp": float(las_rw),
                        "measured_temp_f": las_rw_temp_f,
                        "salinity_ppm_estimated": salinity_est,
                        "source_priority": "las.RW",
                    },
                )
            salinity_est = float(salinity_from_rw(float(las_rw), reference_temp))
            rw_curve = rw_from_salinity(salinity_est, temp_profile)
            decisions.append(
                f"Rw from LAS metadata used assumed reference temperature {reference_temp:.1f}F and was corrected along the temperature profile"
            )
            self._warn_assumed(
                f"Header RW {float(las_rw):g} ohm.m has no RWT: it is taken at the median formation "
                f"temperature, {reference_temp:.1f} degF. Pass rw and rw_meas_temp_f to use another value."
            )
            return _finalize_rw(
                "las_metadata_assumed_reference_temperature",
                rw_curve,
                {
                    "rw": float(las_rw),
                    "reference_temp_f": reference_temp,
                    "salinity_ppm_estimated": salinity_est,
                    "source_priority": "las.RW",
                },
            )

        if self.config.salinity_ppm is not None:
            rw_curve = rw_from_salinity(self.config.salinity_ppm, temp_profile)
            decisions.append(
                f"Rw from salinity {self.config.salinity_ppm:.0f} ppm across the temperature profile"
            )
            return _finalize_rw(
                "salinity_conversion",
                rw_curve,
                {
                    "salinity_ppm": self.config.salinity_ppm,
                    "source_priority": "salinity_ppm",
                },
            )

        if self._skips_flushed_zone:
            # An SP needs a conductive mud column, and the Rxo/Rt ratio a
            # water-based filtrate of known resistivity (Schlumberger, 1989,
            # Log Interpretation Principles/Applications).
            decisions.append(
                "Rw SP and Rxo/Rt methods skipped: mud_type='OBM' (an SP needs a conductive "
                "mud and Rxo/Rt a water-based filtrate; the 200 ohm.m RMF written for "
                "oil-based mud is a placeholder)"
            )
        elif self._result.rmf_computed is not None:
            rmf_curve = np.asarray(self._result.rmf_computed, dtype=float)
            if rmf_curve.ndim == 0:
                rmf_curve = np.full(well.n_samples, float(rmf_curve), dtype=float)
            decisions.append(
                "Rw direct source unavailable -> using available RMF curve for heuristic estimation"
            )

            if self._has_curve(well, "SP"):
                if self.config.mud_has_clay_or_kcl:
                    decisions.append("SP method skipped because mud_has_clay_or_kcl=True")
                else:
                    try:
                        sp_module = RwModule(method="sp")
                        sp_curve = get_curve(well, ["SP"])
                        sp_data = np.asarray(sp_curve.data, dtype=float)
                        clean_mask = sp_module._build_clean_water_mask(well)
                        shale_mask = sp_module._build_shale_mask(well)
                        clean_sp = sp_data[clean_mask] if clean_mask is not None else sp_data
                        clean_sp = clean_sp[np.isfinite(clean_sp)]
                        if clean_sp.size > 0:
                            sp_clean, _ = estimate_sp_lines(clean_sp)
                        else:
                            sp_clean, _ = estimate_sp_lines(sp_data)

                        if shale_mask is not None:
                            shale_sp = sp_data[shale_mask]
                            shale_sp = shale_sp[np.isfinite(shale_sp)]
                            if shale_sp.size > 0:
                                sp_shale = float(np.nanmedian(shale_sp))
                            else:
                                _, sp_shale = estimate_sp_lines(sp_data)
                        else:
                            _, sp_shale = estimate_sp_lines(sp_data)
                        if not self._reproducing_benchmark:
                            # sp_max and sp_min are the clean-sand and shale lines.
                            if self.config.sp_max is not None:
                                sp_clean = float(self.config.sp_max)
                                decisions.append(f"SPclean = sp_max = {sp_clean:g} mV (user)")
                            if self.config.sp_min is not None:
                                sp_shale = float(self.config.sp_min)
                                decisions.append(f"SPshale = sp_min = {sp_shale:g} mV (user)")

                        rw_curve = rw_from_sp(sp_clean, sp_shale, rmf_curve, temp_profile)
                        decisions.append(
                            f"Rw via SP clean-water method with SPclean={sp_clean:.2f} mV and SPshale={sp_shale:.2f} mV across the temperature profile"
                        )
                        finite_rw = np.asarray(rw_curve, dtype=float)
                        finite_rw = finite_rw[np.isfinite(finite_rw)]
                        # rw_from_ssp returns the unclipped value: half or more
                        # of the samples outside 0.01-10 ohm.m is not a water.
                        outside = (finite_rw < 0.01) | (finite_rw > 10.0)
                        at_bound = finite_rw.size == 0 or np.mean(outside) >= 0.5
                        if at_bound and not self._reproducing_benchmark:
                            message = (
                                f"Rw from the SP (SSP={sp_clean - sp_shale:.1f} mV) lies outside the "
                                "0.01-10 ohm.m range of the SP method in "
                                f"{np.mean(outside) if finite_rw.size else 1.0:.0%} of the samples "
                                "and was rejected; set rw or salinity_ppm."
                            )
                            self._warn_assumed(message)
                            decisions.append(message)
                            raise _SpRwRejected()
                        self._warn_assumed(
                            "Rw was estimated from the SP; set rw or salinity_ppm for a measured value."
                        )
                        return _finalize_rw(
                            "sp_clean_water",
                            rw_curve,
                            {
                                "rmf_min": float(np.nanmin(rmf_curve)),
                                "rmf_max": float(np.nanmax(rmf_curve)),
                                "sp_clean_mv": float(sp_clean),
                                "sp_shale_mv": float(sp_shale),
                                "ssp_mv": float(sp_clean - sp_shale),
                                "source_priority": "rmf -> SP",
                            },
                            ["SP"],
                        )
                    except _SpRwRejected:
                        pass
                    except Exception as exc:
                        message = f"Rw SP heuristic failed: {exc}"
                        self._result.warnings.append(message)
                        decisions.append(message)
            else:
                decisions.append("SP method skipped because SP curve is unavailable")

            try:
                from petrophys.interpretation.electrical.rw.sp_rw import rw_from_rxo_rt
                rt = get_curve(well, ["RDEP", "RT"])
                if self._historical:
                    # The "geolink" profile read the shallow RSHA first.
                    rxo = get_curve(well, ["RSHA", "RXO", "RESS"])
                    rxo_data = None if rxo is None else np.asarray(rxo.data, dtype=float)
                    rxo_source = "RSHA"
                else:
                    # The flushed-zone RXO first, as the invasion step reads
                    # it, with RSHA/RESS where RXO has no reading.
                    rxo_data, rxo_source, _ = get_curve_data_depthwise(well, ["RXO", "RSHA", "RESS"])
                if rxo_data is None or rt is None:
                    raise ValueError("Rxo/Rt curves unavailable")
                rxo_data = np.asarray(rxo_data, dtype=float)
                rt_data = np.asarray(rt.data, dtype=float)
                clean_mask = RwModule()._build_clean_water_mask(well)
                if clean_mask is not None:
                    valid = clean_mask & np.isfinite(rxo_data) & np.isfinite(rt_data) & np.isfinite(rmf_curve)
                else:
                    valid = np.isfinite(rxo_data) & np.isfinite(rt_data) & np.isfinite(rmf_curve)
                if not np.any(valid):
                    raise ValueError("No valid clean-water Rxo/Rt samples")
                rw_clean = rw_from_rxo_rt(rxo_data[valid], rt_data[valid], rmf_curve[valid])
                if self._dataset == "geolink_train":
                    rw_reference_temp, rw_at_reference, rw_curve = self._train_rw_curve(
                        rw_clean, temp_profile[valid], temp_profile, np.nanmedian
                    )
                    decisions.append(
                        f"Rw via Rxo/Rt clean-water method using {int(np.count_nonzero(valid))} valid samples "
                        "and direct Arps temperature correction"
                    )
                    return _finalize_rw(
                        "rxo_rt_clean_water",
                        rw_curve,
                        {
                            "rmf_min": float(np.nanmin(rmf_curve)),
                            "rmf_max": float(np.nanmax(rmf_curve)),
                            "rw_at_reference_temp": rw_at_reference,
                            "rw_reference_temp_f": rw_reference_temp,
                            "n_points": int(np.count_nonzero(valid)),
                            "source_priority": "rmf -> rxo_rt",
                        },
                        ["RSHA", "RDEP"],
                    )
                salinity_est = float(np.nanmedian(salinity_from_rw(rw_clean, temp_profile[valid])))
                rw_curve = rw_from_salinity(salinity_est, temp_profile)
                if not self._historical:
                    self._warn_assumed(
                        "Rw was estimated from Rxo/Rt; set rw or salinity_ppm for a measured value."
                    )
                decisions.append(
                    f"Rw via Rxo/Rt clean-water method using {int(np.count_nonzero(valid))} valid samples and depth-wise temperature correction"
                )
                return _finalize_rw(
                    "rxo_rt_clean_water",
                    rw_curve,
                    {
                        "rmf_min": float(np.nanmin(rmf_curve)),
                        "rmf_max": float(np.nanmax(rmf_curve)),
                        "salinity_ppm_estimated": salinity_est,
                        "n_points": int(np.count_nonzero(valid)),
                        "source_priority": "rmf -> rxo_rt",
                    },
                    ["RSHA", "RDEP"] if self._historical else [rxo_source, rt.mnemonic],
                )
            except Exception as exc:
                message = f"Rw log heuristic failed: {exc}"
                self._result.warnings.append(message)
                decisions.append(message)
        else:
            decisions.append("Rw heuristic methods skipped because Rmf is unavailable")

        rt_curve = get_curve(well, ["RDEP", "RT"])
        phie_curve = get_curve(well, self._phie_names(include_phit=False))
        vcl_curve = get_curve(well, ["VCL", "VCLAY"])
        if rt_curve is not None and phie_curve is not None and vcl_curve is not None:
            rdep = np.asarray(rt_curve.data, dtype=float)
            phie = np.asarray(phie_curve.data, dtype=float)
            vcl = np.asarray(vcl_curve.data, dtype=float)
            lithology = self._build_lithology_log(well)
            archie_a, archie_m, _, archie_summary = self._archie_coefficients_for_lithology(
                lithology,
                well.n_samples,
            )
            clean_wet_mask = (
                np.isfinite(rdep)
                & np.isfinite(phie)
                & np.isfinite(vcl)
                & (vcl < 0.3)
                & (phie > 0.05)
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                rwa = rdep * np.power(phie, archie_m) / archie_a
            valid = clean_wet_mask & np.isfinite(rwa) & (rwa > 0)
            if np.any(valid):
                if self._dataset == "geolink_train":
                    rwa_reference_temp, rw_value, rw_curve = self._train_rw_curve(
                        rwa[valid], temp_profile[valid], temp_profile, lambda values: np.percentile(values, 5)
                    )
                    temperature_text = "then direct Arps temperature correction"
                    estimate = {}
                elif self._historical:
                    rw_value = float(np.percentile(rwa[valid], 5))
                    rwa_reference_temp = float(np.nanmedian(temp_profile[valid]))
                    salinity_est = float(salinity_from_rw(rw_value, rwa_reference_temp))
                    rw_curve = rw_from_salinity(salinity_est, temp_profile)
                    temperature_text = "then depth-wise temperature correction"
                    estimate = {"salinity_ppm_estimated": salinity_est}
                else:
                    # Minimum Rwa (P5) after bringing each sample to one temperature:
                    # Rw falls with temperature, so raw deep Rwa would pass for salty water.
                    # Minimum-Rwa method: Asquith and Krygowski (2004), Basic Well Log
                    # Analysis, 2nd ed.; temperature dependence: Arps (1953), JPT 5(10).
                    with np.errstate(invalid="ignore"):
                        salinity = np.asarray(
                            salinity_from_rw(rwa[valid], temp_profile[valid]), dtype=float
                        )
                    plausible = np.isfinite(salinity) & (salinity <= _SATURATED_BRINE_PPM)
                    rejected = int(salinity.size - np.count_nonzero(plausible))
                    if np.any(plausible):
                        rwa_at_reference = np.asarray(
                            rw_from_salinity(salinity[plausible], _RWA_REFERENCE_TEMP_F), dtype=float
                        )
                        salinity_est = float(
                            salinity_from_rw(np.percentile(rwa_at_reference, 5), _RWA_REFERENCE_TEMP_F)
                        )
                    else:
                        salinity_est = _SATURATED_BRINE_PPM
                    if rejected:
                        self._warn_assumed(
                            f"{rejected} of {salinity.size} clean samples read an apparent Rw "
                            f"saltier than saturated brine ({_SATURATED_BRINE_PPM:,.0f} ppm) and were "
                            "left out of the Rw estimate; check the deep resistivity and porosity there."
                        )
                    rwa_reference_temp = _RWA_REFERENCE_TEMP_F
                    rw_value = float(rw_from_salinity(salinity_est, rwa_reference_temp))
                    rw_curve = rw_from_salinity(salinity_est, temp_profile)
                    temperature_text = (
                        f"each sample taken to {_RWA_REFERENCE_TEMP_F:g} degF, "
                        "then depth-wise temperature correction"
                    )
                    estimate = {
                        "rwa_temperature_corrected": True,
                        "salinity_ppm_estimated": salinity_est,
                        "rwa_implausible_samples": rejected,
                    }
                decisions.append(
                    "Rw from Rwa clean-wet fallback: "
                    + "P5 of "
                    + f"{rt_curve.mnemonic} * {phie_curve.mnemonic}^m / a "
                    f"where VCL < 0.3 and {phie_curve.mnemonic} > 0.05 (n={int(np.count_nonzero(valid))}), "
                    "using lithology-dependent Archie coefficients, "
                    f"{temperature_text}"
                )
                self._warn_assumed(
                    f"Rw {rw_value:.4g} ohm.m at {rwa_reference_temp:.1f} degF "
                    f"({estimate.get('salinity_ppm_estimated', float('nan')):,.0f} ppm) is estimated "
                    "from the apparent Rw of clean samples, which assumes some of them are "
                    "water-bearing, and holds one water for the whole well. Pass rw and "
                    "rw_meas_temp_f if Rw is known, or zones= where the water changes."
                )
                return _finalize_rw(
                    "rwa_clean_wet_p5",
                    rw_curve,
                    {
                        "rwa_percentile": 5.0,
                        "rwa_clean_wet_samples": int(np.count_nonzero(valid)),
                        "rwa_min": float(np.nanmin(rwa[valid])),
                        "rwa_max": float(np.nanmax(rwa[valid])),
                        "rw_from_rwa": rw_value,
                        "archie_a": _scalar_or_lithology_dependent(archie_a),
                        "archie_m": _scalar_or_lithology_dependent(archie_m),
                        "archie_coefficients": archie_summary,
                        "archie_lithology_source": self._lithology_source,
                        "rwa_reference_temp_f": rwa_reference_temp,
                        **estimate,
                        "vcl_cutoff": 0.3,
                        "phie_cutoff": 0.05,
                        "source_priority": "Rwa clean-wet fallback",
                    },
                    [rt_curve.mnemonic, phie_curve.mnemonic, vcl_curve.mnemonic],
                )
            decisions.append("Rwa clean-wet fallback skipped because no valid VCL < 0.3 and effective porosity > 0.05 samples were available")
        else:
            decisions.append("Rwa clean-wet fallback skipped because RDEP/RT, effective porosity, or VCL is unavailable")

        if not self.config.allow_default_rw:
            message = (
                "Rw could not be determined from configuration, metadata, salinity, "
                "SP, Rxo/Rt, or clean-wet Rwa; saturation was not computed"
            )
            self._result.warnings.append(message)
            decisions.append(message)
            trace.method = "unavailable"
            trace.message = message
            trace.decisions = decisions
            return trace

        self._result.warnings.append("Rw could not be determined. Using default 0.05 ohm.m")
        decisions.append(
            "All Rw priority sources failed (config/LAS RW, salinity, SP, Rxo/Rt, Rwa clean-wet fallback) -> default 0.05"
        )
        if self._dataset == "geolink_train":
            # The training package took the default at 75 degF and corrected
            # it with Arps.
            from petrophys.interpretation.electrical.rw.temp_correction import correct_rw_temperature

            reference_temp = DEFAULT_RESISTIVITY_REFERENCE_TEMP_F
            rw_curve = correct_rw_temperature(0.05, reference_temp, temp_profile, "F")
        else:
            rw_curve = rw_from_salinity(float(salinity_from_rw(0.05, reference_temp)), temp_profile)
        return _finalize_rw(
            "default_fallback_profile",
            rw_curve,
            {
                "rw_default": 0.05,
                "reference_temp_f": reference_temp,
                "source_priority": "default",
            },
        )

    def _step_invasion(self, well: "Well") -> StepResult:
        from petrophys.interpretation.electrical.invasion.invasion_module import InvasionModule

        trace = StepResult(step="invasion", success=True, method="rxo_rt_ratio")
        invasion_module = InvasionModule(
            rxo_curve=_BENCHMARK_RXO_CURVE if self._reproducing_benchmark else "RXO",
            medium_deep_proxy=not self._benchmark,
            depthwise_curve_merge=not self._benchmark,
            class_bands=_BENCHMARK_INVASION_BANDS if self._reproducing_benchmark else None,
        )
        result = invasion_module.analyze(well)
        well.add_curve("INVASION_RATIO", result.ratio, "", "Invasion Ratio Rxo/Rt")
        trace.input_curves = [
            name for name in (
                result.statistics.get("rt_curve"),
                result.statistics.get("rxo_curve"),
            )
            if name
        ]
        trace.output_curves = ["INVASION_RATIO"]
        trace.parameters = dict(result.statistics)
        if not self._reproducing_benchmark:
            # The curves the ratio read, so the trace says which resistivity stands for Rxo.
            trace.parameters["rxo_curve_order"] = [*invasion_module.shallow_curves, *invasion_module.medium_curves]
            used = [
                f"{name} at {count}"
                for name, count in (result.statistics.get("rxo_source_counts") or {}).items()
            ]
            used_text = ", ".join(used[:-1]) + (" and " if len(used) > 1 else "") + (used[-1] if used else "no")
            proxy = invasion_module.medium_curves
            trace.decisions.append(
                f"INVASION_RATIO reads {used_text} samples over {result.statistics.get('rt_curve')}; curve "
                f"preference {', '.join(invasion_module.shallow_curves)}"
                + (f", then the {'/'.join(proxy)} medium/deep proxy" if proxy else "")
            )
        rejected = int(result.statistics.get("nan_samples") or 0)
        total = int(result.statistics.get("total_samples") or len(result.ratio))
        if result.statistics.get("fallback_proxy_used"):
            trace.decisions.append(
                "INVASION_RATIO fallback medium/deep proxy = RMED/RDEP because RXO/RSHA/RESS were unavailable or non-finite at those samples"
            )
        if rejected:
            trace.decisions.append(
                "INVASION_RATIO samples written as NaN where shallow/deep resistivity inputs were invalid, "
                f"RT/RDEP was below {result.statistics.get('rt_min_used', 0.2)}, "
                f"or RXO/RT exceeded {result.statistics.get('max_ratio_used', 100.0)} "
                f"({rejected} of {total} samples rejected)"
            )
        else:
            trace.decisions.append(f"INVASION_RATIO valid for all {total} samples")
        return trace

    def _step_hc_correction(self, well: "Well") -> StepResult:
        from petrophys.interpretation.porosity.hydrocarbon_correction.hc_correction_module import HcCorrectionModule
        from petrophys.model.constants import DefaultParameters, MatrixProperties

        trace = StepResult(step="hc_correction", success=True, method=f"{self.config.hc_type}_correction")
        decisions = []

        rho_ma = self.config.rho_ma
        rho_ma_source = "user"
        if rho_ma is None:
            try:
                rho_ma = (
                    MatrixProperties.get_density(self.config.matrix_type)
                    if self._reproducing_benchmark
                    else MatrixProperties.get_log_density(self.config.matrix_type)
                )
                rho_ma_source = f"MatrixProperties[{self.config.matrix_type}]"
            except KeyError:
                rho_ma = 2.65
                rho_ma_source = "default"
        decisions.append(f"rho_ma = {rho_ma} g/cm3 (source: {rho_ma_source})")

        hc_module = HcCorrectionModule(
            hc_type=self.config.hc_type, rho_ma=rho_ma,
            rho_fl=self.config.rho_fl, phi_cl=self._phi_cl_setting,
            rho_cl=self.config.rho_cl, rho_hc=self.config.rho_hc,
            hydrogen_index=self.config.hc_hydrogen_index,
            floor_porosity=not self._historical,
        )
        result = hc_module.calculate(well)
        hc_module.add_to_well(well, result)

        trace.input_curves = ["NPHI", "RHOB"]
        trace.output_curves = ["NPHI_HC", "RHOB_HC"]
        trace.parameters = {"hc_type": self.config.hc_type, "rho_ma": rho_ma,
                           "rho_fl": self.config.rho_fl, "phi_cl": self._phi_cl_setting,
                           "rho_cl": self.config.rho_cl}
        if not self._historical:
            gas = str(self.config.hc_type).lower() == "gas"
            trace.parameters["rho_hc"] = (
                self.config.rho_hc if self.config.rho_hc is not None else (0.2 if gas else 0.8)
            )
            trace.parameters["hc_hydrogen_index"] = (
                self.config.hc_hydrogen_index if self.config.hc_hydrogen_index is not None
                else (DefaultParameters.GAS_HYDROGEN_INDEX if gas else DefaultParameters.OIL_HYDROGEN_INDEX)
            )
        else:
            for name in ("rho_hc", "hc_hydrogen_index"):
                if getattr(self.config, name) is not None:
                    trace.parameters[name] = getattr(self.config, name)
        trace.decisions = decisions
        return trace

    def _resolve_matrix_arrays(self, well: "Well") -> Dict[str, Any]:
        from petrophys.model.constants import MatrixProperties

        benchmark = self._reproducing_benchmark

        def density_endpoint(props) -> float:
            # Density porosity takes the apparent log density where one is
            # tabulated (evaporites); the "geolink" profile keeps grain density.
            if benchmark or props.log_density is None:
                return float(props.density)
            return float(props.log_density)

        def slowness_endpoint(props, matrix_key, group: bool) -> float:
            # The "geolink" profile keeps the eval package's coal slowness and
            # gypsum-group slowness.
            if benchmark:
                kept = _BENCHMARK_GROUP_SLOWNESS if group else _BENCHMARK_SLOWNESS
                return float(kept.get(matrix_key, props.slowness))
            return float(props.slowness)

        fallback_key = self.config.matrix_type
        try:
            fallback_props = MatrixProperties.get(self.config.matrix_type)
        except KeyError:
            # An unresolved "auto" takes the domain's own matrix (limestone
            # for carbonate), else sandstone, as _vcl_nd_matrix does.
            fallback_key = "sandstone"
            if not self._reproducing_benchmark:
                domain = str(self.config.lithology_domain).strip().lower()
                fallback_key = _DOMAIN_MATRICES.get(domain, ("sandstone",))[0]
            fallback_props = MatrixProperties.get(fallback_key)

        fallback_rho_ma = float(self.config.rho_ma if self.config.rho_ma is not None else density_endpoint(fallback_props))
        fallback_dt_ma = float(
            self.config.dt_ma if self.config.dt_ma is not None
            else slowness_endpoint(fallback_props, fallback_key, group=False)
        )
        fallback_nphi_ma = float(self.config.nphi_ma if self.config.nphi_ma is not None else fallback_props.nphi)

        rho_ma = np.full(well.n_samples, fallback_rho_ma, dtype=float)
        dt_ma = np.full(well.n_samples, fallback_dt_ma, dtype=float)
        nphi_ma = np.full(well.n_samples, fallback_nphi_ma, dtype=float)

        matrix_endpoints: List[Dict[str, Any]] = []
        matrix_groups_used: List[str] = []
        matrix_mode = "constant"
        matrix_source = f"config_or_default[{self.config.matrix_type}]"
        shale_mask = np.zeros(well.n_samples, dtype=bool)

        lithology = self._build_lithology_log(well)
        if isinstance(lithology, np.ndarray):
            matrix_mode = "lithology_dependent"
            matrix_source = self._lithology_source
            valid_groups = _ordered_unique_values(
                group for group in lithology.tolist()
                if _coerce_lithology_group(group) is not None
            )
            for raw_group in valid_groups:
                group = _coerce_lithology_group(raw_group)
                matrix_key = self._group_to_matrix.get(group or "", self.config.matrix_type)
                try:
                    props = MatrixProperties.get(matrix_key)
                except KeyError:
                    props = fallback_props
                    matrix_key = self.config.matrix_type

                mask = np.asarray([item == raw_group for item in lithology], dtype=bool)
                if not np.any(mask):
                    continue
                if _is_shale_group(group):
                    shale_mask |= mask

                rho_ma[mask] = density_endpoint(props)
                dt_ma[mask] = slowness_endpoint(props, matrix_key, group=True)
                nphi_ma[mask] = float(props.nphi)
                matrix_groups_used.append(str(raw_group).upper())
                matrix_endpoints.append(
                    {
                        "lithology_group": str(raw_group).upper(),
                        "matrix_key": matrix_key,
                        "rho_ma": float(rho_ma[mask][0]),
                        "dt_ma": float(dt_ma[mask][0]),
                        "nphi_ma": float(props.nphi),
                    }
                )

        halite_samples = 0
        if not benchmark and isinstance(lithology, np.ndarray):
            # Coal endpoints (1.35 g/cm3) only where the density reads coal: a
            # sample labelled coal at RHOB 2.0 g/cm3 or more is coaly shale or
            # an interbed, read on the fallback matrix.
            rhob_values, _, _ = get_curve_data_depthwise(well, ["RHOB"])
            if rhob_values is not None:
                coal_label = np.asarray(
                    [_coerce_lithology_group(item) == "coal" for item in lithology], dtype=bool
                )
                with np.errstate(invalid="ignore"):
                    not_coal = coal_label & (np.asarray(rhob_values, dtype=float) >= 2.0)
                if np.any(not_coal):
                    rho_ma[not_coal] = fallback_rho_ma
                    dt_ma[not_coal] = fallback_dt_ma
                    nphi_ma[not_coal] = fallback_nphi_ma
                    matrix_endpoints.append({
                        "lithology_group": "COAL (RHOB >= 2.0 g/cm3)",
                        "matrix_key": self.config.matrix_type,
                        "rho_ma": float(fallback_rho_ma),
                        "dt_ma": float(fallback_dt_ma),
                        "nphi_ma": float(fallback_nphi_ma),
                        "samples": int(np.count_nonzero(not_coal)),
                    })
        if not benchmark:
            # Halite on the logs where no lithology group calls the sample an
            # evaporite takes the halite endpoints, so its density porosity
            # reads about zero rather than 0.38 on a sandstone matrix.
            halite = _log_halite(well, lithology)
            halite_samples = int(np.count_nonzero(halite))
            if halite_samples:
                props = MatrixProperties.get("halite")
                rho_ma[halite] = density_endpoint(props)
                dt_ma[halite] = slowness_endpoint(props, "halite", group=True)
                nphi_ma[halite] = float(props.nphi)
                shale_mask &= ~halite
                matrix_mode = "lithology_dependent"
                matrix_endpoints.append(
                    {
                        "lithology_group": "HALITE (LOG SIGNATURE)",
                        "matrix_key": "halite",
                        "rho_ma": density_endpoint(props),
                        "dt_ma": slowness_endpoint(props, "halite", group=True),
                        "nphi_ma": float(props.nphi),
                        "samples": halite_samples,
                    }
                )

        user_endpoints = []
        if not benchmark and matrix_mode == "lithology_dependent":
            # A matrix endpoint set explicitly holds for the whole well, over
            # the lithology groups' and the halite signature's values.
            for name, array in (("rho_ma", rho_ma), ("dt_ma", dt_ma), ("nphi_ma", nphi_ma)):
                value = getattr(self.config, name)
                if value is not None:
                    array[:] = float(value)
                    user_endpoints.append(f"{name} = {float(value):g}")
                    for endpoint in matrix_endpoints:
                        endpoint[name] = float(value)

        return {
            "user_endpoints": user_endpoints,
            "rho_ma": rho_ma,
            "dt_ma": dt_ma,
            "nphi_ma": nphi_ma,
            "matrix_mode": matrix_mode,
            "matrix_source": matrix_source,
            "matrix_groups_used": matrix_groups_used,
            "matrix_endpoints": matrix_endpoints,
            "fallback_rho_ma": fallback_rho_ma,
            "fallback_dt_ma": fallback_dt_ma,
            "fallback_nphi_ma": fallback_nphi_ma,
            # Samples of shale lithology groups and the configured matrix's
            # endpoints, the basis of the clay porosity under "current".
            "shale_mask": shale_mask,
            "reservoir_matrix": fallback_key,
            "reservoir_rho_ma": (
                fallback_rho_ma if not benchmark and self.config.rho_ma is not None
                else density_endpoint(fallback_props)
            ),
            "reservoir_nphi_ma": (
                fallback_nphi_ma if not benchmark and self.config.nphi_ma is not None
                else float(fallback_props.nphi)
            ),
            "lithology_log": isinstance(lithology, np.ndarray),
            "halite_log_samples": halite_samples,
        }

    def _resolve_phi_cl(
        self,
        well: "Well",
        vcl_curve,
        porosity_basis: Optional[np.ndarray] = None,
        porosity_basis_name: Optional[str] = None,
        porosity_basis_valid_mask: Optional[np.ndarray] = None,
        basis_matrix: Optional[str] = None,
        matrix_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Clay porosity for PHIE = PHIT - VCL * phi_cl, and where it came from.

        ``basis_matrix`` names the matrix whose endpoints the shale samples in
        ``porosity_basis`` were computed on, when that differs from their own
        lithology's endpoints. ``matrix_info`` is the porosity step's
        :meth:`_resolve_matrix_arrays` result, if at hand.

        Under ``"current"`` a configured ``phi_cl`` comes first, then a LAS
        header shale point (:meth:`_header_shale_point_phi_cl`), then the
        well's shale samples, then the configured default. The "geolink"
        profile takes the LAS PNSH header itself as ``phi_cl``, ahead of
        everything else.

        The result is one value for the whole well. It does not follow the
        compaction of shale porosity with depth, so where shale porosity
        changes down the well PHIE is too low in some intervals and too high
        in others, by VCL times the difference. Samples without a neutron
        reading (PHIT = PHID) get the neutron-density value; a shale's density
        porosity is usually the lower of the two, so they are over-corrected by
        VCL times the difference. A well without a neutron log uses the
        configured default.
        """
        if self._reproducing_benchmark:
            explicit_phi_cl = _parameter_float(well, "PNSH")
            explicit_source = "LAS PNSH parameter"
            if explicit_phi_cl is None:
                explicit_phi_cl = _constant_zoned_parameter(well, "PNSH")
                explicit_source = "zoned LAS PNSH fallback"

            if explicit_phi_cl is not None:
                return {
                    "phi_cl": float(explicit_phi_cl),
                    "phi_cl_source": explicit_source,
                    "phi_cl_basis_count": None,
                    "phi_cl_basis": "explicit_parameter",
                }

        if not _left_at_default(self.config, "phi_cl"):
            # Configured deliberately: a value derived from shale samples or,
            # under "current", from the LAS header must not quietly replace it.
            return {
                "phi_cl": self._phi_cl_setting,
                "phi_cl_source": "config",
                "phi_cl_basis_count": None,
                "phi_cl_basis": "config",
            }

        header_note: Optional[str] = None
        if not self._reproducing_benchmark:
            header = self._header_shale_point_phi_cl(well, matrix_info)
            if header is not None and "phi_cl" in header:
                return header
            if header is not None:
                header_note = header["phi_cl_header_note"]

        if self._benchmark:
            # The eval package takes clay porosity from the mean PHIT of every
            # shale-lithology sample; the "current" profile averages the
            # neutron-density averaged shale samples only.
            lithology = self._build_lithology_log(well)
            phit_curve = well.get("PHIT")
            basis = None if phit_curve is None else np.asarray(phit_curve.data, dtype=float)
            if isinstance(lithology, np.ndarray) and basis is not None:
                shale = np.asarray([_is_shale_group(item) for item in lithology], dtype=bool)
                mask = shale & np.isfinite(basis)
                if np.any(mask):
                    return {
                        "phi_cl": float(np.mean(basis[mask])),
                        "phi_cl_source": "all_shale_phit",
                        "phi_cl_basis": "all_shale_phit",
                        "phi_cl_basis_count": int(np.count_nonzero(mask)),
                    }

        def shale_control_mask() -> tuple[np.ndarray, str]:
            shale_mask = np.zeros(well.n_samples, dtype=bool)
            lithology = self._build_lithology_log(well)
            if isinstance(lithology, np.ndarray):
                shale_mask = np.asarray([_is_shale_group(item) for item in lithology], dtype=bool)
            basis = "lithology_shale"
            if not np.any(shale_mask) and vcl_curve is not None:
                vcl = np.asarray(vcl_curve.data, dtype=float)
                shale_mask = np.isfinite(vcl) & (vcl > 0.9)
                basis = "vcl_gt_0.9"
            return shale_mask, basis

        fallback_reason = ""

        def from_porosity(
            values: Any,
            label: str,
            valid_source_mask: Optional[np.ndarray] = None,
        ) -> Optional[Dict[str, Any]]:
            nonlocal fallback_reason
            if values is None:
                return None
            porosity = np.asarray(values, dtype=float)
            shale_mask, basis = shale_control_mask()
            valid_mask = shale_mask & np.isfinite(porosity)
            if valid_source_mask is None:
                fallback_reason = "PHIT clay-porosity control requires neutron-density averaged PHIT samples"
                return None
            source_mask = np.asarray(valid_source_mask, dtype=bool)
            if source_mask.shape != valid_mask.shape:
                fallback_reason = "PHIT source mask was unavailable for clay-porosity control"
                return None
            valid_mask &= source_mask
            if not np.any(valid_mask):
                fallback_reason = "no shale-control PHIT samples were calculated from neutron-density averaging"
                return None
            phi_cl = float(np.clip(np.nanmean(porosity[valid_mask]), 0.0, 0.6))
            label_clean = str(label or "porosity").lower()
            source = f"{basis}_average_{label_clean}_nd"
            if basis_matrix is not None:
                source += "_on_reservoir_matrix"
            info = {
                "phi_cl": phi_cl,
                "phi_cl_source": source,
                "phi_cl_basis_count": int(np.count_nonzero(valid_mask)),
                "phi_cl_basis": source,
                "phi_cl_fallback_reason": None,
            }
            if basis_matrix is not None:
                info["phi_cl_matrix"] = basis_matrix
            return info

        preferred = from_porosity(
            porosity_basis,
            porosity_basis_name or "porosity",
            porosity_basis_valid_mask,
        )
        if preferred is None:
            preferred = {
                "phi_cl": self._phi_cl_setting,
                "phi_cl_source": "config_or_default",
                "phi_cl_basis_count": None,
                "phi_cl_basis": "config_or_default",
                "phi_cl_fallback_reason": fallback_reason or "no usable neutron-density averaged PHIT shale-control samples",
            }
        if header_note is not None:
            preferred["phi_cl_header_note"] = header_note
        return preferred

    def _header_shale_point_phi_cl(
        self, well: "Well", matrix_info: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Clay porosity from the LAS header's neutron-density shale point (``"current"``).

        PNSH ("Neutron Shale") and RBSH ("Shale Density") are the neutron and
        density log readings in shale, beside the matrix (RHMA, PNMA) and fluid
        (RBFL) points of the same parameter set. On the endpoints of the
        reservoir PHIT they give the shale's PHIT by the porosity step's
        neutron-density rule, and that is the clay porosity of
        PHIE = PHIT - VCL * phi_cl (Asquith and Krygowski, 2004). Returns None
        without PNSH; without a usable RBSH, a ``phi_cl_header_note`` saying
        why PNSH was set aside instead of a ``phi_cl``.
        """
        from petrophys.interpretation.porosity.density_porosity import density_porosity
        from petrophys.interpretation.porosity.neutron_porosity import neutron_porosity
        from petrophys.interpretation.porosity.total_porosity import total_porosity

        pnsh, source = _parameter_float(well, "PNSH"), "LAS"
        if pnsh is None:
            pnsh, source = self._zoned_constant(well, "PNSH"), "zoned LAS"
        if pnsh is None:
            self._warn_zone_only_unused(well, "PNSH", "the clay porosity")
            return None
        rbsh = _parameter_float(well, "RBSH")
        if rbsh is None:
            rbsh = self._zoned_constant(well, "RBSH")
        if rbsh is None or not (0.0 <= pnsh < 1.0 and 1.0 < rbsh < 3.0):
            reason = (
                "and there is no RBSH (shale density) to complete the shale point"
                if rbsh is None
                else f"and with RBSH = {rbsh:g} the pair is not a v/v and g/cm3 shale point"
            )
            return {
                "phi_cl_header_note": (
                    f"{source} PNSH = {pnsh:g} set aside: it is the neutron log's reading "
                    f"in shale, not clay porosity, {reason}"
                )
            }
        if matrix_info is None:
            matrix_info = self._resolve_matrix_arrays(well)
        # The endpoints of the reservoir PHIT: the configured matrix's values
        # where a lithology log sets endpoints per group, else the constant
        # (configured or LAS header) endpoints.
        if matrix_info["matrix_mode"] == "lithology_dependent":
            rho_ma, nphi_ma = matrix_info["reservoir_rho_ma"], matrix_info["reservoir_nphi_ma"]
        else:
            rho_ma, nphi_ma = matrix_info["fallback_rho_ma"], matrix_info["fallback_nphi_ma"]
        phid = density_porosity(rbsh, rho_ma, self.config.rho_fl)
        phin = float(neutron_porosity(pnsh, nphi_matrix=nphi_ma))
        phi_cl = float(total_porosity(phin, phid, gas_method=self._crossover_gas_method))
        return {
            "phi_cl": phi_cl,
            "phi_cl_source": f"{source} PNSH and RBSH shale point",
            "phi_cl_basis": "header_shale_point_nd",
            "phi_cl_basis_count": None,
            "phi_cl_fallback_reason": None,
            "phi_cl_shale_point": {
                "PNSH": pnsh, "RBSH": rbsh, "PHIN": phin, "PHID": phid,
                "rho_ma": float(rho_ma), "nphi_ma": float(nphi_ma), "rho_fl": float(self.config.rho_fl),
            },
        }

    def _prefer_vendor_porosity(
        self,
        well: "Well",
        name: str,
        computed: Optional[np.ndarray],
        vendor_names: Sequence[str],
        description: str,
        inputs: List[str],
        outputs: List[str],
        decisions: List[str],
    ) -> Optional[np.ndarray]:
        """``name`` from a logged vendor porosity where it is in range, else ``computed``.

        The first of ``vendor_names`` on the well is read in v/v (a percent
        label, or values with a median above 1, are divided by 100). Its
        samples within :data:`_POROSITY_RANGE` replace the computed porosity;
        the computed value stays elsewhere. Returns the porosity used.
        """
        vendor_name = next((n for n in vendor_names if well.get(n) is not None), None)
        if vendor_name is None:
            return computed
        curve = well.get(vendor_name)
        values = np.asarray(curve.data, dtype=float).copy()
        values[values <= -900.0] = np.nan
        unit = str(curve.unit or "").strip().lower()
        finite = values[np.isfinite(values)]
        percent = unit in _PERCENT_UNITS or (
            unit not in _FRACTION_UNITS and finite.size and float(np.median(finite)) > 1.0
        )
        if percent:
            values = values / 100.0
        low, high = _POROSITY_RANGE
        valid = np.isfinite(values) & (values >= low) & (values <= high)
        if not valid.any():
            decisions.append(
                f"{name}: the logged {vendor_name} has no sample within {low:g}-{high:g} v/v; "
                f"{name} is computed"
            )
            return computed
        base = (
            np.full(well.n_samples, np.nan)
            if computed is None
            else np.asarray(computed, dtype=float)
        )
        merged = np.where(valid, values, base)
        well.add_curve(name, merged, "v/v", f"{description} ({vendor_name} where in range)")
        if name not in outputs:
            outputs.append(name)
        inputs.append(vendor_name)
        n_vendor = int(np.count_nonzero(valid))
        n_computed = int(np.count_nonzero(~valid & np.isfinite(base)))
        decisions.append(
            f"{name} = {vendor_name}{' / 100' if percent else ''} (logged {description.lower()}) at "
            f"{n_vendor} samples within {low:g}-{high:g} v/v; computed at {n_computed} other samples"
        )
        return merged

    def _step_porosity_refinement(self, well: "Well") -> StepResult:
        self._lithology_log = None
        self._lithology_source = "config"
        return self._step_porosity(well, refinement=True)

    def _step_porosity(self, well: "Well", refinement: bool = False) -> StepResult:
        from petrophys.interpretation.porosity.density_porosity import density_porosity
        from petrophys.interpretation.porosity.effective_porosity import effective_porosity
        from petrophys.interpretation.porosity.neutron_porosity import neutron_porosity
        from petrophys.interpretation.porosity.sonic_porosity import secondary_porosity, wyllie_time_average
        from petrophys.interpretation.porosity.total_porosity import total_porosity

        trace = StepResult(
            step="porosity_refinement" if refinement else "porosity",
            success=True,
            method="component_based_lithology_refined" if refinement else "component_based",
        )
        decisions = []
        inputs = []
        outputs = []

        nphi_curve = get_curve(well, ["NPHI"])
        rhob_curve = get_curve(well, ["RHOB"])
        hc_corrected = (
            self.config.apply_hc_correction
            and not self._historical
            and get_curve(well, ["NPHI_HC"]) is not None
            and get_curve(well, ["RHOB_HC"]) is not None
        )
        if hc_corrected:
            # apply_hc_correction=True: the hydrocarbon-corrected neutron and
            # density readings are the porosity inputs.
            nphi_curve = get_curve(well, ["NPHI_HC"])
            rhob_curve = get_curve(well, ["RHOB_HC"])
            decisions.append(
                "Porosity computed from the hydrocarbon-corrected NPHI_HC and RHOB_HC "
                "(apply_hc_correction=True)"
            )
        dtc_curve = get_curve(well, ["DTC"])
        existing_phin_curve = get_curve(well, ["PHIN"])
        existing_phid_curve = get_curve(well, ["PHID"])
        existing_phis_curve = get_curve(well, ["PHIS"])
        existing_phit_curve = get_curve(well, ["PHIT"])
        vcl_curve = get_curve(well, ["VCL", "VCLAY"])

        matrix_info = self._resolve_matrix_arrays(well)
        rho_ma = matrix_info["rho_ma"]
        dt_ma = matrix_info["dt_ma"]
        nphi_ma = matrix_info["nphi_ma"]
        if refinement:
            decisions.append(
                "Refinement pass after inferred LITHO: porosity curves recomputed using computed lithology matrix endpoints"
            )
        elif matrix_info["matrix_mode"] == "lithology_dependent" and matrix_info["lithology_log"]:
            decisions.append(
                "Initial porosity pass used an available lithology log for matrix endpoints"
            )
        else:
            decisions.append(
                "Initial porosity pass used configured matrix endpoints; it will be refined if the workflow infers LITHO"
                if self._historical
                else "Porosity used the configured matrix endpoints; it is computed once and "
                "is not recomputed if the workflow later infers LITHO"
            )
        if matrix_info["user_endpoints"]:
            decisions.append(
                f"{', '.join(matrix_info['user_endpoints'])} (user) used at every depth, "
                "in place of the lithology groups' matrix values"
            )
        if matrix_info["halite_log_samples"]:
            halite = next(
                item for item in matrix_info["matrix_endpoints"] if item["lithology_group"] == "HALITE (LOG SIGNATURE)"
            )
            decisions.append(
                f"{matrix_info['halite_log_samples']} samples read as halite on the logs (RHOB 1.95-2.15 g/cm3, "
                "GR < 15 API, RT > 500 ohm.m or an under-gauge hole) where no lithology group calls them an "
                f"evaporite: halite endpoints rho_ma {halite['rho_ma']:g} g/cm3, nphi_ma {halite['nphi_ma']:g}, "
                f"dt_ma {halite['dt_ma']:g} us/ft"
            )

        phin = None
        phid = None
        phis = None
        phit = None
        phie_basis = None
        phie_basis_name = None
        phit_method = None
        phie = None
        phit_source_counts: Dict[str, int] = {}
        phit_nd_average_mask: Optional[np.ndarray] = None

        if nphi_curve is not None:
            inputs.append(nphi_curve.mnemonic)
            phin = neutron_porosity(nphi_curve.data, nphi_matrix=nphi_ma)
            well.add_curve(
                "PHIN",
                phin,
                "v/v",
                "Neutron Porosity refined with computed lithology" if refinement else "Neutron Porosity",
            )
            outputs.append("PHIN")
            if matrix_info["matrix_mode"] == "lithology_dependent":
                decisions.append(
                    "PHIN = clip(NPHI - nphi_ma(depth), 0, 0.6) using lithology-specific matrix neutron endpoints"
                )
            else:
                decisions.append(
                    f"PHIN = clip(NPHI - nphi_ma, 0, 0.6) using nphi_ma={matrix_info['fallback_nphi_ma']:.3f}"
                )
        elif existing_phin_curve is not None:
            phin = existing_phin_curve.data
            inputs.append("PHIN")
            decisions.append("PHIN reused from existing well curve")

        if rhob_curve is not None:
            inputs.append(rhob_curve.mnemonic)
            phid = density_porosity(rhob_curve.data, rho_ma, self.config.rho_fl)
            well.add_curve(
                "PHID",
                phid,
                "v/v",
                "Density Porosity refined with computed lithology" if refinement else "Density Porosity",
            )
            outputs.append("PHID")
            if matrix_info["matrix_mode"] == "lithology_dependent":
                decisions.append("PHID = (rho_ma(depth) - RHOB) / (rho_ma(depth) - rho_fl)")
            else:
                decisions.append("PHID = (rho_ma - RHOB) / (rho_ma - rho_fl)")
            if self._costa_fluid:
                decisions.append(
                    f"rho_fl not set: the COSTA calibration's {self.config.rho_fl:g} g/cm3 brine"
                )
        elif existing_phid_curve is not None:
            phid = existing_phid_curve.data
            inputs.append("PHID")
            decisions.append("PHID reused from existing well curve")

        if dtc_curve is not None:
            inputs.append("DTC")
            phis = wyllie_time_average(dtc_curve.data, dt_ma, self.config.dt_fl)
            well.add_curve(
                "PHIS",
                phis,
                "v/v",
                "Sonic Porosity refined with computed lithology" if refinement else "Sonic Porosity",
            )
            outputs.append("PHIS")
            if matrix_info["matrix_mode"] == "lithology_dependent":
                decisions.append("PHIS from Wyllie time-average equation using dt_ma(depth)")
            else:
                decisions.append("PHIS from Wyllie time-average equation")
        elif existing_phis_curve is not None:
            phis = existing_phis_curve.data
            inputs.append("PHIS")
            decisions.append("PHIS reused from existing well curve")

        if not self._reproducing_benchmark:
            # A logged vendor density or sonic porosity takes precedence where
            # its value is a porosity the workflow would accept.
            phid = self._prefer_vendor_porosity(
                well, "PHID", phid, _VENDOR_DENSITY_POROSITY, "Density Porosity", inputs, outputs, decisions
            )
            phis = self._prefer_vendor_porosity(
                well, "PHIS", phis, _VENDOR_SONIC_POROSITY, "Sonic Porosity", inputs, outputs, decisions
            )

        if phid is not None and phin is not None:
            gas_method = self._crossover_gas_method
            phit = total_porosity(phin, phid, gas_method=gas_method)
            phie_basis = phit
            phie_basis_name = "PHIT"
            phit_method = "adaptive_nd"
            phin_arr = np.asarray(phin, dtype=float)
            phid_arr = np.asarray(phid, dtype=float)
            both = np.isfinite(phin_arr) & np.isfinite(phid_arr)
            density_missing_phin = ~np.isfinite(phin_arr) & np.isfinite(phid_arr)
            neutron_missing_phid = np.isfinite(phin_arr) & ~np.isfinite(phid_arr)
            density_gas = both & (phid_arr > phin_arr + 0.05)
            weighted_nd = both & ~density_gas
            phit_nd_average_mask = weighted_nd
            phit_arr = np.asarray(phit, dtype=float)
            phit_source_counts = {
                "density_from_missing_phin": int(np.count_nonzero(density_missing_phin)),
                f"{gas_method}_from_gas_crossover": int(np.count_nonzero(density_gas)),
                "weighted_nd": int(np.count_nonzero(weighted_nd)),
                "neutron_from_missing_phid": int(np.count_nonzero(neutron_missing_phid)),
                "nan": int(np.count_nonzero(~np.isfinite(phit_arr))),
            }
            decisions.append(
                "PHIT = adaptive(PHIN, PHID) with per-sample fallback: "
                + ", ".join(f"{name}={count}" for name, count in phit_source_counts.items())
            )
        elif phid is not None:
            phit = phid
            phie_basis = phit
            phie_basis_name = "PHIT"
            phit_method = "density_only"
            phit_source_counts = {
                "density_only": int(np.count_nonzero(np.isfinite(np.asarray(phit, dtype=float)))),
                "nan": int(np.count_nonzero(~np.isfinite(np.asarray(phit, dtype=float)))),
            }
            decisions.append("Density-only porosity: PHIT = PHID because PHIN was unavailable")
        elif phin is not None:
            phit = phin
            phie_basis = phit
            phie_basis_name = "PHIT"
            phit_method = "neutron_only"
            phit_source_counts = {
                "neutron_only": int(np.count_nonzero(np.isfinite(np.asarray(phit, dtype=float)))),
                "nan": int(np.count_nonzero(~np.isfinite(np.asarray(phit, dtype=float)))),
            }
            decisions.append("Neutron-only porosity: PHIT = PHIN because PHID was unavailable")
            self._warn_assumed(
                "Porosity is from the neutron log alone (no density porosity): it reads high in "
                "shaly rock and low in gas, so PHIE, Sw and net pay are less reliable. Add RHOB if available."
            )
        elif phis is not None:
            phit = phis
            phie_basis = phit
            phie_basis_name = "PHIT"
            phit_method = "sonic_only"
            phit_source_counts = {
                "sonic_only": int(np.count_nonzero(np.isfinite(np.asarray(phit, dtype=float)))),
                "nan": int(np.count_nonzero(~np.isfinite(np.asarray(phit, dtype=float)))),
            }
            decisions.append("Sonic-only porosity: PHIT = PHIS because PHIN and PHID were unavailable")
        else:
            if existing_phit_curve is not None:
                phit = existing_phit_curve.data
                phie_basis = phit
                phie_basis_name = "PHIT"
                phit_method = "existing_phit"
                phit_source_counts = {
                    "existing_phit": int(np.count_nonzero(np.isfinite(np.asarray(phit, dtype=float)))),
                    "nan": int(np.count_nonzero(~np.isfinite(np.asarray(phit, dtype=float)))),
                }
                inputs.append("PHIT")
                decisions.append("PHIT reused from existing well curve")

        if phit is not None:
            apparent_info = _apparent_porosity_info(phit_method)
            phit_output = "PHIT"
            if apparent_info is not None:
                phit_description = (
                    f"Apparent total porosity, {apparent_info['source']} "
                    f"from {apparent_info['basis']}"
                )
                if refinement:
                    phit_description += " with computed lithology matrix endpoints"
            else:
                phit_description = (
                    "Total Porosity refined with computed lithology"
                    if refinement
                    else "Total Porosity"
                )
            phit_curve = well.add_curve(
                phit_output,
                phit,
                "v/v",
                phit_description,
                data_type=DataType.COMPUTED,
            )
            phit_curve.metadata.custom.update(
                {
                    "porosity_role": "total",
                    "porosity_method": phit_method,
                    "porosity_apparent": apparent_info is not None,
                    "porosity_basis": apparent_info["basis"] if apparent_info else phie_basis_name,
                    "porosity_source": apparent_info["source"] if apparent_info else "workflow_calculation",
                }
            )
            if phit_output not in outputs:
                outputs.append(phit_output)
            if apparent_info is not None:
                decisions.append(
                    f"{phit_output} written as apparent total porosity "
                    f"({apparent_info['source']}) because {apparent_info['reason']}"
                )

        phi_cl_basis = phie_basis
        phi_cl_basis_mask = phit_nd_average_mask if phie_basis_name == "PHIT" else None
        phi_cl_matrix = None
        shale_mask = matrix_info["shale_mask"]
        if (
            not self._reproducing_benchmark
            and phit_method == "adaptive_nd"
            and rhob_curve is not None
            and nphi_curve is not None
            and np.any(shale_mask)
        ):
            # Under linear mixing, PHIT = PHIE + VCL * (the shale's PHIT on the
            # same matrix and fluid endpoints), the shale correction of
            # Asquith and Krygowski (2004). phi_cl is therefore the shale's
            # PHIT on the reservoir rock's endpoints, not on the wet-shale point
            # of the shale groups' own PHIT, averaged over the shale samples
            # recomputed on the configured matrix.
            basis_phid = density_porosity(
                rhob_curve.data,
                np.where(shale_mask, matrix_info["reservoir_rho_ma"], rho_ma),
                self.config.rho_fl,
            )
            basis_phin = neutron_porosity(
                nphi_curve.data,
                nphi_matrix=np.where(shale_mask, matrix_info["reservoir_nphi_ma"], nphi_ma),
            )
            phi_cl_basis = total_porosity(basis_phin, basis_phid)
            phi_cl_basis_mask = (
                np.isfinite(basis_phid)
                & np.isfinite(basis_phin)
                & ~(basis_phid > basis_phin + 0.05)
            )
            phi_cl_matrix = matrix_info["reservoir_matrix"]

        phi_cl_info = self._resolve_phi_cl(
            well,
            vcl_curve,
            porosity_basis=phi_cl_basis,
            porosity_basis_name=phie_basis_name,
            porosity_basis_valid_mask=phi_cl_basis_mask,
            basis_matrix=phi_cl_matrix,
            matrix_info=matrix_info,
        )
        phi_cl = float(phi_cl_info["phi_cl"])

        if phie_basis is not None:
            if vcl_curve is not None:
                inputs.append(vcl_curve.mnemonic)
                vcl = vcl_curve.data
            else:
                vcl = np.zeros(well.n_samples, dtype=float)

            phie = effective_porosity(phie_basis, vcl, phi_cl)
            phie_output = "PHIE"
            apparent_info = _apparent_porosity_info(phit_method)
            if apparent_info is not None:
                correction_text = (
                    f"{vcl_curve.mnemonic} correction"
                    if vcl_curve is not None
                    else "no clay-curve correction"
                )
                phie_description = (
                    f"Apparent effective porosity from {apparent_info['source']} "
                    f"PHIT and {correction_text}"
                )
                if refinement:
                    phie_description += " with computed lithology matrix endpoints"
            else:
                phie_description = (
                    "Effective Porosity refined with computed lithology"
                    if refinement
                    else "Effective Porosity"
                )
            phie_curve = well.add_curve(
                phie_output,
                phie,
                "v/v",
                phie_description,
                data_type=DataType.COMPUTED,
            )
            phie_curve.metadata.custom.update(
                {
                    "porosity_role": "effective",
                    "porosity_method": phit_method,
                    "porosity_apparent": apparent_info is not None,
                    "porosity_basis": phie_basis_name,
                    "porosity_source": apparent_info["source"] if apparent_info else "workflow_calculation",
                }
            )
            outputs.append(phie_output)
            if apparent_info is not None:
                decisions.append(
                    f"{phie_output} written as apparent effective porosity from "
                    f"{apparent_info['source']} {phie_basis_name}"
                )
            if vcl_curve is not None:
                decisions.append(
                    f"{phie_output} = {phie_basis_name} - {vcl_curve.mnemonic} * phi_cl ({phi_cl:.3f})"
                )
            else:
                decisions.append(f"{phie_output} = {phie_basis_name} (no clay curve available)")
            if phi_cl_info.get("phi_cl_matrix"):
                decisions.append(
                    "phi_cl averaged over the shale samples recomputed on the "
                    f"{phi_cl_info['phi_cl_matrix']} matrix endpoints of the PHIT it corrects"
                )
            if phi_cl_info.get("phi_cl_source") == "config_or_default" and phi_cl_info.get("phi_cl_fallback_reason"):
                decisions.append(
                    "phi_cl fallback used because "
                    + str(phi_cl_info["phi_cl_fallback_reason"])
                )
            point = phi_cl_info.get("phi_cl_shale_point")
            if point:
                decisions.append(
                    f"phi_cl is the PHIT of the LAS shale point PNSH {point['PNSH']:g}, "
                    f"RBSH {point['RBSH']:g} g/cm3 on the reservoir endpoints (rho_ma "
                    f"{point['rho_ma']:g}, nphi_ma {point['nphi_ma']:g}, rho_fl {point['rho_fl']:g}): "
                    f"PHID {point['PHID']:.3f}, PHIN {point['PHIN']:.3f}"
                )
            if phi_cl_info.get("phi_cl_header_note"):
                decisions.append(str(phi_cl_info["phi_cl_header_note"]))

        if phis is not None and phit is not None:
            phi_sec = secondary_porosity(phis, phit)
            well.add_curve("PHI_SEC", phi_sec, "v/v", "Secondary Porosity Index")
            outputs.append("PHI_SEC")
            decisions.append("PHI_SEC = PHIT - PHIS")

        trace.input_curves = inputs
        trace.output_curves = outputs
        trace.parameters = {
            "matrix_type": self.config.matrix_type,
            "matrix_mode": matrix_info["matrix_mode"],
            "matrix_source": matrix_info["matrix_source"],
            "porosity_pass": "refined_after_computed_lithology" if refinement else "initial",
            "uses_computed_lithology": matrix_info["matrix_source"] == "computed_lithology_curve",
            "matrix_groups_used": matrix_info["matrix_groups_used"],
            "matrix_endpoints": matrix_info["matrix_endpoints"],
            "nphi_ma": (
                matrix_info["fallback_nphi_ma"]
                if matrix_info["matrix_mode"] == "constant" or self.config.nphi_ma is not None and matrix_info["user_endpoints"]
                else None
            ),
            "nphi_ma_min": float(np.nanmin(nphi_ma)),
            "nphi_ma_max": float(np.nanmax(nphi_ma)),
            "rho_ma": (
                matrix_info["fallback_rho_ma"]
                if matrix_info["matrix_mode"] == "constant" or self.config.rho_ma is not None and matrix_info["user_endpoints"]
                else None
            ),
            "rho_ma_min": float(np.nanmin(rho_ma)),
            "rho_ma_max": float(np.nanmax(rho_ma)),
            "rho_fl": self.config.rho_fl,
            "dt_ma": (
                matrix_info["fallback_dt_ma"]
                if matrix_info["matrix_mode"] == "constant" or self.config.dt_ma is not None and matrix_info["user_endpoints"]
                else None
            ),
            "dt_ma_min": float(np.nanmin(dt_ma)),
            "dt_ma_max": float(np.nanmax(dt_ma)),
            "dt_fl": self.config.dt_fl,
            "phi_cl": phi_cl,
            "phi_cl_source": phi_cl_info["phi_cl_source"],
            "phi_cl_basis": phi_cl_info["phi_cl_basis"],
            "phi_cl_basis_count": phi_cl_info["phi_cl_basis_count"],
            "phi_cl_fallback_reason": phi_cl_info.get("phi_cl_fallback_reason"),
            "phie_basis": phie_basis_name,
            "phit_method": phit_method,
            "phit_source_counts": phit_source_counts,
            "phit_nd_average_samples": int(np.count_nonzero(phit_nd_average_mask)) if phit_nd_average_mask is not None else 0,
            "phit_apparent": _apparent_porosity_info(phit_method) is not None,
            "phie_apparent": _apparent_porosity_info(phit_method) is not None and phie is not None,
            "apparent_porosity_source": (
                _apparent_porosity_info(phit_method)["source"]
                if _apparent_porosity_info(phit_method) is not None
                else None
            ),
        }
        if "phi_cl_matrix" in phi_cl_info:
            trace.parameters["phi_cl_matrix"] = phi_cl_info["phi_cl_matrix"]
        for key in ("phi_cl_shale_point", "phi_cl_header_note"):
            if key in phi_cl_info:
                trace.parameters[key] = phi_cl_info[key]
        trace.decisions = decisions
        self._last_phit_nd_average_mask = phit_nd_average_mask
        self._last_phi_cl_info = phi_cl_info
        return trace

    def _step_core_correction(self, well: "Well") -> StepResult:
        """Fit same-well core correction before saturation uses porosity."""
        from petrophys.interpretation.permeability.perm_module import PermeabilityModule

        trace = StepResult(step="core_correction", success=True, method="same_well_core_correction")
        decisions: list[str] = []
        outputs: list[str] = []

        stale_core_curves = [name for name in ("PHIE_CC", "PERM_CC") if well.get(name) is not None]
        if stale_core_curves and self._historical:
            for curve_name in stale_core_curves:
                well.remove_curve(curve_name)
            decisions.append(
                "Existing core-corrected curves cleared before pre-saturation core correction"
            )
        elif stale_core_curves:
            decisions.append(self._keep_input_core_curves(well, stale_core_curves))

        perm_module = PermeabilityModule(
            method=self.config.perm_method,
            fallback_method=self.config.perm_fallback_method,
            use_peer_core=self.config.use_peer_core,
            porosity_curves=["PHIE"] if self._benchmark else None,
            coefficients=self.config.perm_coefficients,
            **self._core_calibration_options(well),
        )
        result = perm_module.calculate_core_correction(
            well,
            peer_wells=self._core_peer_wells(well, decisions),
            formation_tops=self.config.formation_tops,
        )

        phi_corecal = result.metadata.get("phi_corecal")
        perm_core = result.metadata.get("perm_core")
        if self._dataset == "geolink_train":
            phi_corecal = self._train_core_porosity(phi_corecal, perm_core)
        if phi_corecal is not None and not self._reproducing_benchmark:
            phi_corecal = self._effective_core_porosity(well, phi_corecal)
        if phi_corecal is not None:
            well.add_curve(
                "PHIE_CC",
                phi_corecal,
                "v/v",
                "Core-corrected effective porosity",
            )
            outputs.append("PHIE_CC")
            decisions.append("Core-corrected porosity stored as PHIE_CC before saturation")
        if perm_core is not None:
            well.add_curve(
                "PERM_CC",
                perm_core,
                "mD",
                "Permeability from core-calibrated poro-perm transform",
            )
            outputs.append("PERM_CC")
            decisions.append("Core-corrected permeability stored as PERM_CC")

        trace.input_curves = [result.metadata.get("porosity_curve", "PHIE")]
        trace.output_curves = outputs
        trace.parameters = {
            **trace.parameters,
            "strategy": result.metadata.get("strategy"),
            "same_well_core_models": result.metadata.get("same_well_core_models"),
            "peer_core_models": result.metadata.get("peer_core_models"),
            "peer_wells_considered": result.metadata.get("peer_wells_considered"),
            "peer_core_enabled": result.metadata.get("peer_core_enabled"),
            "core_min_points": result.metadata.get("core_min_points"),
            "core_models": result.metadata.get("core_models"),
            "core_rejected_models": result.metadata.get("core_rejected_models"),
            "core_sections_count": result.metadata.get("core_sections_count"),
            "perm_max_clip_md": result.metadata.get("perm_max_clip_md"),
        }
        if not outputs:
            decisions.append("No PHIE_CC/PERM_CC written because no same-well core model was fitted")
        for model in result.metadata.get("core_models", []):
            decisions.append(
                "Core model fitted "
                f"(source={model.get('source_type')}, n={model.get('n_core_points')}, "
                f"core_phi={model.get('core_porosity_curve') or 'unknown'}, "
                f"core_perm={model.get('core_permeability_curve') or 'unknown'})"
            )
        if not self._reproducing_benchmark:
            # Core porosity is fitted to PHIT, and PHIT - PHIE is taken off again.
            trace.input_curves = [name for name in ("PHIT", "PHIE") if get_curve(well, [name]) is not None]
            cutoff = self._net_rock_vcl_cutoff(well)
            cutoff_text = (
                f"{float(cutoff[0]):.2f}" if np.ptp(cutoff) == 0
                else f"{float(np.min(cutoff)):.2f}-{float(np.max(cutoff)):.2f} by lithology"
            )
            for model in result.metadata.get("core_models", []):
                if model.get("vcl_max") is not None:
                    decisions.append(
                        f"PERM_CC from the {model.get('source_formation') or 'whole-well'} transform only "
                        f"where VCL is at or below the net-rock clay cutoff ({cutoff_text}) and the clay "
                        f"volume its core plugs span (P95 {model['vcl_max']:.2f})"
                    )
            trace.parameters["core_porosity_fits"] = result.metadata.get("core_porosity_fits")
            trace.parameters["core_porosity_min_r_squared"] = _CORE_POROSITY_MIN_R_SQUARED
            decisions.extend(self._core_porosity_decisions(result.metadata, "PHIE_CC" in outputs))
        trace.decisions = decisions
        return trace

    def _paired_total_porosity(
        self, well: "Well", phie_selected: Optional[np.ndarray]
    ) -> tuple[Optional[np.ndarray], str]:
        """Total porosity on the basis of the effective porosity SW is paired with.

        ``PHIE_sel + (PHIT - PHIE)``: the clay-bound porosity of the porosity
        step added to the selected effective porosity, e.g. a core-corrected
        PHIE_CC, so that Waxman-Smits and Dual Water read only clay-bound
        water, not the core correction, in PHIT - PHIE_sel. It is PHIT where
        PHIE itself is selected or the bound porosity is unknown. Returns the
        array (None without PHIT or PHIE: the saturation module reads PHIT) and
        a label for the trace.
        """
        phit, _, _ = get_curve_data_depthwise(well, ["PHIT"])
        phie, _, _ = get_curve_data_depthwise(well, ["PHIE"])
        if phie_selected is None or phit is None or phie is None:
            return None, "PHIT"
        with np.errstate(invalid="ignore"):
            paired = phie_selected + np.clip(phit - phie, 0.0, None)
        shifted = (phie_selected != phie) & np.isfinite(paired)
        if not np.any(shifted):
            return phit, "PHIT"
        return np.where(shifted, paired, phit), "PHIE_CC + (PHIT - PHIE) where PHIE_CC is used, else PHIT"

    def _shaly_sand_cec(self) -> Optional[float]:
        """Check the inputs of an explicit shaly-sand model; return CEC in meq/g.

        Dual Water needs a supplied Rwb and Waxman-Smits a supplied CEC and B;
        none of them is estimated. An explicit request without them stops the
        step rather than writing an SW labelled with a model that did not run.
        """
        model = self.config.sw_model
        if model == "dual_water" and self.config.rwb is None:
            raise ValueError(
                "sw_model='dual_water' needs the bound-water resistivity rwb (ohm.m), from "
                "lab data or an RWB LAS header parameter; it is not estimated. "
                "sw_model='auto' uses Dual Water only where rwb is given."
            )
        if model == "waxman_smits" and (self.config.cec is None or self.config.b_coeff is None):
            raise ValueError(
                "sw_model='waxman_smits' needs cec (meq/100 g) and b_coeff, from lab data or "
                "CEC and B_COEFF LAS header parameters; they are not estimated."
            )
        if self.config.cec is None:
            return None
        cec_error = _cec_range_error(self.config.cec)
        if cec_error:
            raise ValueError(cec_error)
        cec = float(self.config.cec)
        if 0.0 < cec < 1.0:
            # Under 1 meq/100 g is a clean sand, while the same number in meq/g
            # is a typical shaly sand: the likelier reading is a unit slip.
            message = (
                f"cec={cec:g} meq/100 g is a clean-sand value; a meq/g value must be "
                "multiplied by 100"
            )
            if message not in self._result.warnings:
                self._result.warnings.append(message)
        # Lab CEC is reported in meq/100 g; waxman_smits() takes meq/g.
        return cec / 100.0

    def _routing_salinity(self) -> Optional[float]:
        """Formation-water salinity for the saturation model choice: that of
        the Rw the rw step used (``salinity_ppm`` where Rw came from it, else
        the NaCl equivalent of a given or header Rw at its temperature), so a
        saline water routes to Simandoux whichever way it was given. A
        ``salinity_ppm`` set next to an Rw from another source is not the
        water the saturation is computed with; the warning says so."""
        if self._reproducing_benchmark or self._historical:
            return self.config.salinity_ppm
        for step in self._result.steps_executed:
            if step.step == "rw":
                for key in ("salinity_ppm", "salinity_ppm_estimated"):
                    value = step.parameters.get(key)
                    if isinstance(value, (int, float)) and np.isfinite(value) and value > 0:
                        given = self.config.salinity_ppm
                        if given is not None and key == "salinity_ppm_estimated" and not (
                            0.5 <= float(value) / float(given) <= 2.0
                        ):
                            source = "the rw setting" if self.config.rw is not None else "another source"
                            self._warn_assumed(
                                f"salinity_ppm = {float(given):,.0f} is not used: the Rw of this run comes "
                                f"from {source}, a water of about {float(value):,.0f} ppm NaCl, and the "
                                "saturation model is chosen for that water. Set either rw or salinity_ppm."
                            )
                        return float(value)
        return self.config.salinity_ppm

    def _step_saturation(self, well: "Well") -> StepResult:
        from petrophys.interpretation.saturation.sw_module import SaturationModule
        from petrophys.workflow._canonical_calculations import (
            CanonicalSaturationModule,
        )

        trace = StepResult(step="saturation", success=True)
        cec = self.config.cec if self._reproducing_benchmark else self._shaly_sand_cec()
        lithology = self._build_lithology_log(well)
        a_value, m_value, n_value, archie_summary = self._archie_coefficients_for_lithology(
            lithology,
            well.n_samples,
        )

        vcl_curve = get_curve(well, ["VCL", "VCLAY"])
        if not self._reproducing_benchmark and self._last_phi_cl_info is not None:
            # Dual Water takes the clay porosity of the PHIE it is paired with.
            phi_cl_info = self._last_phi_cl_info
        else:
            phit_curve_for_phi_cl = get_curve(well, ["PHIT"])
            phi_cl_info = self._resolve_phi_cl(
                well,
                vcl_curve,
                porosity_basis=phit_curve_for_phi_cl.data if phit_curve_for_phi_cl is not None else None,
                porosity_basis_name="PHIT" if phit_curve_for_phi_cl is not None else None,
                porosity_basis_valid_mask=(
                    self._last_phit_nd_average_mask
                    if phit_curve_for_phi_cl is not None
                    and getattr(self, "_last_phit_nd_average_mask", None) is not None
                    else None
                ),
            )
        phi_cl = float(phi_cl_info["phi_cl"])

        routing_cutoff = self._routing_vcl_cutoff()
        module_options = dict(
            a=a_value, m=m_value, n=n_value,
            model=self.config.sw_model, vcl_cutoff=routing_cutoff,
            sxo_quicklook_fallback="whole_curve" if self._benchmark else "per_sample",
            depthwise_curve_merge=not self._benchmark,
        )
        if self._reproducing_benchmark:
            sw_module = CanonicalSaturationModule(**module_options)
        else:
            # The published equations. Waxman-Smits Sw goes on the effective
            # basis, as BVW and net pay pair SW with PHIE.
            sw_module = SaturationModule(
                **module_options, rwb_fallback=None, effective_waxman_smits=True
            )
        flushed_zone = not self._skips_flushed_zone

        sat_phie, _, _ = get_curve_data_depthwise(well, self._phie_names(include_phit=False))
        sat_phit, sat_phit_basis = None, "PHIT"
        if not self._reproducing_benchmark:
            sat_phit, sat_phit_basis = self._paired_total_porosity(well, sat_phie)
        routing_salinity = self._routing_salinity()
        result = sw_module.calculate(
            well, phie=sat_phie, phit=sat_phit,
            rw=self._result.rw_computed, rcl=self._result.rcl_computed,
            salinity=routing_salinity,
            rmf=self._result.rmf_computed if flushed_zone else None,
            lithology=lithology, cec=cec, b_coeff=self.config.b_coeff,
            rho_ma=self._grain_density_setting(), m_star=self.config.m_star,
            rwb=self.config.rwb, phi_cl=phi_cl,
            rtsh=self.config.rtsh, vsh_exp=self.config.vsh_exp,
            shale_exp=self.config.shale_exp,
        )

        computed_sw = np.asarray(result.sw, dtype=float)
        source_sw_source = None
        source_sw_meta = {}
        source_sw_mask = np.zeros(well.n_samples, dtype=bool)
        so_valid = None
        written_sw = computed_sw
        if self._dataset == "costa":
            outputs, source_sw_source, source_sw_meta, source_sw_mask, so_valid, written_sw = (
                self._costa_saturation(well, computed_sw, result.model)
            )
        else:
            well.add_curve("SW", computed_sw, "v/v", f"Water Saturation ({result.model})")
            outputs = ["SW"]
        sxo_method, sxo_reason = result.sxo_method, result.sxo_reason
        sxo_method_counts = result.sxo_method_counts or {}
        if not flushed_zone:
            sxo_method, sxo_reason, sxo_method_counts = None, _OBM_FLUSHED_ZONE_NOTE, {}
        elif result.sxo is not None:
            well.add_curve("SXO", result.sxo, "v/v", "Flushed Zone Water Saturation")
            outputs.append("SXO")

        trace.method = result.model
        _, rt_source, _ = get_curve_data_depthwise(well, ["RDEP", "RT"])
        _, phie_source, _ = get_curve_data_depthwise(well, self._phie_names())
        _, vcl_source, _ = get_curve_data_depthwise(well, ["VCL", "VCLAY"])
        trace.input_curves = [
            source for source in (source_sw_source, rt_source, phie_source, vcl_source) if source
        ]
        trace.output_curves = outputs
        trace.parameters = {
            "sw_model": result.model,
            "sw_model_requested": self.config.sw_model,
            "sw_model_sample_counts": result.model_counts or {},
            "a": _scalar_or_lithology_dependent(a_value),
            "m": _scalar_or_lithology_dependent(m_value),
            "n": _scalar_or_lithology_dependent(n_value),
            "archie_coefficients": archie_summary,
            "archie_lithology_source": self._lithology_source,
            "archie_a_source": "config_or_las" if self.config.a is not None else "ArchieParameterDefaults",
            "archie_m_source": "config_or_las" if self.config.m is not None else "ArchieParameterDefaults",
            "archie_n_source": "config_or_las" if self.config.n is not None else "ArchieParameterDefaults",
            "rw_min": (
                float(np.nanmin(np.asarray(self._result.rw_computed, dtype=float)))
                if self._result.rw_computed is not None
                else None
            ),
            "rw_max": (
                float(np.nanmax(np.asarray(self._result.rw_computed, dtype=float)))
                if self._result.rw_computed is not None
                else None
            ),
            "rcl": self._result.rcl_computed,
            "salinity_ppm": routing_salinity,
            "cec": self.config.cec,
            "cec_present": self.config.cec is not None,
            "b_coeff": self.config.b_coeff,
            "b_coeff_present": self.config.b_coeff is not None,
            "m_star": self.config.m_star,
            "rwb": self.config.rwb,
            "phi_cl": phi_cl,
            "phi_cl_source": phi_cl_info["phi_cl_source"],
            "phi_cl_basis": phi_cl_info["phi_cl_basis"],
            "phi_cl_basis_count": phi_cl_info["phi_cl_basis_count"],
            "phi_cl_fallback_reason": phi_cl_info.get("phi_cl_fallback_reason"),
            "rtsh": self.config.rtsh if self.config.rtsh is not None else self._result.rcl_computed,
            "vsh_exp": self.config.vsh_exp,
            "shale_exp": self.config.shale_exp,
            "vcl_cutoff_for_routing": routing_cutoff,
            "lithology_type": self.config.lithology_domain if isinstance(lithology, str) else "per-depth array",
            "sxo_method": sxo_method,
            "sxo_reason": sxo_reason,
            "sxo_method_counts": sxo_method_counts,
            "sxo_input_sources": result.sxo_input_sources or {},
            "source_sw_curve": source_sw_source,
            "source_sw_samples_used": int(np.count_nonzero(source_sw_mask)),
            "archie_sw_samples_used": int(np.count_nonzero(~source_sw_mask & np.isfinite(computed_sw))),
            "source_sw_input_sources": source_sw_meta.get("source_counts", {}) if source_sw_meta else {},
            "lithology_domain": self.config.lithology_domain,
        }
        if self._result.rmf_computed is not None:
            trace.parameters["rmf_min"] = float(np.nanmin(np.asarray(self._result.rmf_computed, dtype=float)))
            trace.parameters["rmf_max"] = float(np.nanmax(np.asarray(self._result.rmf_computed, dtype=float)))
        # Shaly carbonate without Rwb takes the clastic routing: model -> samples.
        fallbacks = {
            route.split("->", 1)[1]: count
            for route, count in (result.fallback_counts or {}).items()
            if route.startswith("dual_water->")
        }
        if not self._reproducing_benchmark:
            trace.parameters.update({
                "shaly_sand_equations": "published",
                "sw_model_fallback_counts": result.fallback_counts or {},
                "cec_meq_per_g": cec,
                "waxman_smits_sw_basis": "effective",
                "total_porosity_basis": sat_phit_basis,
                "waxman_smits_clipped_to_zero": result.waxman_smits_clipped_to_zero,
            })
        trace.decisions.append(f"Model selected: {result.model}")
        if self._reproducing_benchmark:
            # The equations this well's samples were routed to, with their sample counts.
            names = {
                "archie": "Archie", "indonesia": "Indonesia", "dual_water": "Dual Water",
                "simandoux": "Simandoux", "waxman_smits": "Waxman-Smits",
            }
            used = [
                f"{names.get(model, model)} ({count} samples)"
                for model, count in (result.model_counts or {}).items() if count
            ]
            trace.decisions.append(
                "Equations used: " + ", ".join(used) if used else "No saturation equation was applied"
            )
        else:
            used = result.model_counts or {}
            if "indonesia" in used:
                trace.decisions.append(
                    "Indonesia equation of Poupon and Leveaux (1971): 1/sqrt(Rt) = "
                    "[VCL^(1-VCL/2)/sqrt(Rcl) + PHIE^(m/2)/sqrt(a*Rw)] * Sw^(n/2)"
                )
            if "dual_water" in used:
                trace.decisions.append(
                    "Dual Water in the wet-zone form of Clavier, Coates and Dumanoir (1984), "
                    f"Rwb = {self.config.rwb:g} ohm.m"
                )
            if "waxman_smits" in used:
                trace.decisions.append(
                    f"Waxman-Smits (1968) with CEC = {self.config.cec:g} meq/100 g ({cec:g} meq/g) "
                    f"and B = {self.config.b_coeff:g} (S/m)/(meq/cm3); its total-porosity Swt is "
                    "reported on the effective basis, SW = 1 - (PHIT/PHIE) * (1 - Swt)"
                    + ("" if sat_phit_basis == "PHIT" else f", with PHIT = {sat_phit_basis}")
                )
                clipped = result.waxman_smits_clipped_to_zero
                if clipped:
                    trace.decisions.append(
                        f"Waxman-Smits: {clipped} samples with Swt below the bound-water fraction "
                        "(PHIT - PHIE)/PHIT were set to SW = 0: the resistivity leaves less water "
                        "than the clay-bound water of the porosity model there (a well-wide CEC "
                        "gives tight rock a large Qv)"
                    )
        if self.config.sw_model == "auto":
            trace.decisions.append("Auto-routing: depth-by-depth model selection based on VCLAY and lithology")
            if (not self._reproducing_benchmark and not self._historical
                    and routing_salinity is not None
                    and (result.model_counts or {}).get("simandoux", 0)):
                trace.decisions.append(
                    f"Simandoux selected for shaly samples at routing salinity "
                    f"{routing_salinity:,.0f} ppm (>= 20,000 ppm)"
                )
            if result.model_counts:
                trace.decisions.append(
                    "Depth routing counts: "
                    + ", ".join(f"{model}={count}" for model, count in result.model_counts.items())
                )
            if fallbacks:
                names = {"indonesia": "Indonesia", "simandoux": "Simandoux", "waxman_smits": "Waxman-Smits"}
                routed = "; ".join(
                    f"{count} shaly carbonate samples used the {names.get(model, model)} equation"
                    for model, count in fallbacks.items()
                )
                trace.decisions.append(
                    f"Dual Water needs a supplied Rwb, which is not estimated: {routed} "
                    "(set rwb to use Dual Water)"
                )
        trace.decisions.append(
            "Archie coefficients from lithology"
            + (f" ({self._lithology_source})" if self._lithology_source else "")
        )
        if np.any(source_sw_mask):
            trace.decisions.append(
                f"SW uses source input {source_sw_source} for {int(np.count_nonzero(source_sw_mask))} finite samples; "
                "Archie fills remaining samples where inputs are finite"
            )
        elif self._reproducing_benchmark or set(result.model_counts or {}) <= {"archie"}:
            trace.decisions.append("No finite source SW input was available; SW is Archie-computed where inputs are finite")
        if self._reproducing_benchmark or "dual_water" in (result.model_counts or {}):
            trace.decisions.append(
                f"Dual Water phi_cl = {phi_cl:.3f} from {phi_cl_info['phi_cl_source']}"
            )
            if phi_cl_info.get("phi_cl_source") == "config_or_default" and phi_cl_info.get("phi_cl_fallback_reason"):
                trace.decisions.append(
                    "Dual Water phi_cl fallback used because "
                    + str(phi_cl_info["phi_cl_fallback_reason"])
                )
        rcl_text = f"Rcl = {self._result.rcl_computed}"
        if not self._reproducing_benchmark and isinstance(self._result.rcl_computed, (int, float)):
            rcl_text = f"Rcl = {self._result.rcl_computed:.3f} ohm.m"
        if self.config.rtsh is not None and not self._reproducing_benchmark:
            rcl_text = f"Rsh = {self.config.rtsh:g} ohm.m (rtsh as set)"
        if trace.parameters["rw_min"] is not None and trace.parameters["rw_max"] is not None:
            trace.decisions.append(
                f"Rw range = {trace.parameters['rw_min']:.4f} to {trace.parameters['rw_max']:.4f}, {rcl_text}"
            )
        else:
            trace.decisions.append(
                f"Rw unavailable for SW; {rcl_text}"
            )
        if so_valid is not None and np.any(so_valid):
            trace.decisions.append("SO = 1 - SW where SW is finite")
        invaded_curves = "RSHA/RXO/RESS" if self._historical else "RXO/RSHA/RESS"
        if sxo_method == "archie_rxo_rmf":
            trace.decisions.append(f"SXO from Archie invaded-zone equation using {invaded_curves} and RMF")
        elif sxo_method == "archie_rxo_rmf+fifth_root_quicklook":
            counts = sxo_method_counts
            count_text = (
                " ("
                + ", ".join(f"{method}={count}" for method, count in counts.items())
                + ")"
                if counts else ""
            )
            trace.decisions.append(
                f"SXO from invaded-zone Archie where {invaded_curves} and RMF were finite; "
                "SXO = SW^(1/5) elsewhere"
                + count_text
            )
        elif sxo_method == "fifth_root_quicklook":
            trace.decisions.append(
                f"SXO fallback: SXO = SW^(1/5) because {sxo_reason}"
            )
        elif not flushed_zone:
            trace.decisions.append(f"SXO not computed: {sxo_reason}")
        if np.any(np.isfinite(written_sw)):
            trace.decisions.append(
                f"Sw range: {np.nanmin(written_sw):.3f} - {np.nanmax(written_sw):.3f}"
            )
        else:
            trace.decisions.append("Sw range: all values are NaN")
            if not self._historical:
                trace.success = False
                trace.message = "Sw is NaN at every depth: " + self._sw_nan_cause(well)
        return trace

    def _sw_nan_cause(self, well: "Well") -> str:
        """The likely reason a saturation run gave no finite Sw."""
        def empty(names: List[str]) -> bool:
            curve = get_curve(well, names)
            return curve is None or not np.any(np.isfinite(np.asarray(curve.data, dtype=float)))

        if empty(["RW"]):
            return (
                "Rw is NaN everywhere, so the Rw inputs give no valid water resistivity "
                "(check rw, rw_meas_temp_f, salinity_ppm and the temperature settings)."
            )
        if empty(["RDEP", "RT"]):
            return "the deep resistivity has no valid values."
        if empty(self._phie_names()):
            return "effective porosity has no valid values."
        return "check the Rw, resistivity and porosity inputs and the Archie settings."

    def _step_permeability(self, well: "Well") -> StepResult:
        from petrophys.interpretation.permeability.perm_module import PermeabilityModule

        trace = StepResult(step="permeability", success=True, method=self.config.perm_method)
        decisions = []
        resflag_info = None
        perm_method_norm = str(self.config.perm_method or "").strip().lower().replace("-", "_")
        perm_method_uses_swirr = perm_method_norm not in {"lith_vcl", "lithology_vcl", "dunite", "dunite_basic"}
        if (
            perm_method_uses_swirr
            and self.config.swirr is None
            and not self._has_curve(well, "SWIRR", "SWI", "SWIR")
            and not self._has_nmr_swirr_route(well)
        ):
            resflag_info = self._ensure_resflag_for_swirr(well)
        outputs = []

        if resflag_info is not None:
            if resflag_info.get("created"):
                outputs.append("RESFLAG")
                if resflag_info.get("source_curve") == "NET_RES":
                    decisions.append(
                        "RESFLAG copied from existing NET_RES before permeability "
                        f"({resflag_info['reservoir_samples']} reservoir samples)"
                    )
                else:
                    decisions.append(
                        "RESFLAG created before permeability from NetPayModule net_reservoir "
                        f"using {resflag_info['vcl_curve']}, {resflag_info['phie_curve']}, "
                        f"and {resflag_info['sw_curve']} ({resflag_info['reservoir_samples']} samples)"
                    )
            else:
                decisions.append(
                    f"{resflag_info['source_curve']} already present before permeability "
                    "and available for Swirr routing"
                )

        # A porosity calibration made earlier in this run is independent of
        # the selected permeability method (including empirical methods).
        core_phi_generated = self._result is not None and any(
            step.step == "core_correction" and step.success and "PHIE_CC" in step.output_curves
            for step in self._result.steps_executed
        )
        # Under "current" PHIE_CC is the porosity saturation read (from the
        # core-correction step, or an input kept for the whole run), so the
        # step neither clears nor replaces it; an input PERM_CC kept for the
        # whole run is left as well.
        kept_core_curves = set(self._input_core_basis)
        if not self._historical:
            kept_core_curves.add("PHIE_CC")
        stale_core_curves = [
            name for name in ("PHIE_CC", "PERM_CC")
            if well.get(name) is not None and not (name == "PHIE_CC" and core_phi_generated)
            and name not in kept_core_curves
        ]
        if stale_core_curves:
            written = {
                name for step in (self._result.steps_executed if self._result is not None else [])
                for name in step.output_curves if step.step != "normalization"
            }
            removed_inputs = [name for name in stale_core_curves if name not in written]
            if removed_inputs and not self._historical:
                decisions.append(self._keep_input_core_curves(well, removed_inputs))
            for curve_name in stale_core_curves:
                well.remove_curve(curve_name)
            decisions.append(
                "Existing core-corrected curves cleared before permeability; "
                "PHIE_CC/PERM_CC are rewritten only when the current same-well core model is fitted"
            )
        if core_phi_generated:
            decisions.append("Porosity calibration from the core-correction step retained independently of permeability")

        code_curve = self._lithology_code_curve(well)
        perm_module = PermeabilityModule(
            method=self.config.perm_method,
            fallback_method=self.config.perm_fallback_method,
            use_peer_core=self.config.use_peer_core,
            porosity_curves=["PHIE"] if self._benchmark else None,
            lithology_curve=code_curve or "LITHOLOGY_GEOLINK",
            coefficients=self.config.perm_coefficients,
            **self._core_calibration_options(well),
        )
        lithology = None
        if code_curve is not None and self._user_code_groups() is not None:
            # The user's codebook names the rocks, so lith_vcl reads names, not table codes.
            codes = np.asarray(well.get(code_curve).data, dtype=float)
            lithology = np.full(codes.shape, "", dtype=object)
            finite = np.isfinite(codes)
            lithology[finite] = [
                self._user_code_labels.get(int(code), "") for code in np.round(codes[finite]).astype(int)
            ]
        elif code_curve is None and not self._has_curve(well, "LITHO_CLASS", "LITHO"):
            lithology = self._build_lithology_log(well)

        peer_wells = self._core_peer_wells(well, decisions)
        result = perm_module.calculate(
            well,
            swi=None if isinstance(self.config.swirr, Mapping) else self.config.swirr,
            lithology=lithology,
            peer_wells=peer_wells,
            formation_tops=self.config.formation_tops,
        )
        if isinstance(self.config.swirr, Mapping):
            # Swirr by lithology; a lithology not given keeps the automatic estimate.
            given = self._swirr_by_lithology(well)
            automatic = np.broadcast_to(np.asarray(result.swi_used, dtype=float), given.shape) \
                if result.swi_used is not None else np.full(given.shape, np.nan)
            result = perm_module.calculate(
                well,
                swi=np.where(np.isfinite(given), given, automatic),
                lithology=lithology,
                peer_wells=peer_wells,
                formation_tops=self.config.formation_tops,
            )
            decisions.append(
                "Swirr by lithology: " + ", ".join(f"{key} {value:g}" for key, value in self.config.swirr.items())
                + f" ({int(np.isfinite(given).sum())} of {given.size} samples); other samples keep the "
                "automatic estimate"
            )
        empirical_perm = result.metadata.get("empirical_perm")
        empirical_method = result.metadata.get("empirical_method") or self.config.perm_fallback_method
        perm_output = empirical_perm if empirical_perm is not None else result.perm
        perm_desc = empirical_method if empirical_perm is not None else result.method
        well.add_curve("PERM", perm_output, "mD", f"Permeability ({perm_desc})")

        outputs.append("PERM")
        phi_corecal = result.metadata.get("phi_corecal")
        perm_core = result.metadata.get("perm_core")
        if phi_corecal is not None and not self._historical:
            # PHIE_CC comes from the core-correction step alone, before
            # saturation, so SW, BVW, net pay and RQI share one porosity.
            if not core_phi_generated:
                decisions.append(
                    "Core-calibrated porosity of the permeability fit not written as PHIE_CC: "
                    "saturation ran without it, and BVW, net pay and RQI keep the porosity SW used"
                )
            phi_corecal = None
        if self._dataset == "geolink_train":
            phi_corecal = self._train_core_porosity(phi_corecal, perm_core)
        if "PERM_CC" in self._input_core_basis:
            perm_core = None
        if phi_corecal is not None and not self._reproducing_benchmark:
            phi_corecal = self._effective_core_porosity(well, phi_corecal)
        if phi_corecal is not None:
            well.add_curve(
                "PHIE_CC",
                phi_corecal,
                "v/v",
                "Permeability input porosity from core calibration",
            )
            outputs.append("PHIE_CC")
        if perm_core is not None:
            well.add_curve(
                "PERM_CC",
                perm_core,
                "mD",
                "Permeability from core-calibrated poro-perm transform",
            )
            outputs.append("PERM_CC")
        swirr_metadata = dict(result.swi_metadata)
        swirr_app = swirr_metadata.pop("swirr_app", None)
        if swirr_app is not None and empirical_method != "lith_vcl":
            well.add_curve(
                "SWIRR_APP",
                swirr_app,
                "v/v",
                "Apparent irreducible water saturation from hydrocarbon-mask P5 method"
                if self._historical or result.swi_source != "buckles_bvw_irr"
                else "Apparent irreducible water saturation, Buckles: BVWirr / PHIE",
            )
            outputs.append("SWIRR_APP")
        method_metadata = result.metadata.get("method_metadata") or {}
        method_input_curves = result.metadata.get("input_curves") or []
        input_curves = (
            list(method_input_curves)
            if method_input_curves
            else [result.metadata.get("porosity_curve", "PHIE")]
        )
        if empirical_method != "lith_vcl" and result.swi_source in {"hc_mask_p5", "default_0.20", "buckles_bvw_irr"}:
            if self._historical or get_curve(well, ["SW"]) is not None:
                input_curves.append("SW")
            elif result.swi_source == "default_0.20":
                # The assumed Swirr of 0.20 is a constant, read from no curve.
                input_curves = [name for name in input_curves if name != "SWIRR/SW"]
            rt_curve = swirr_metadata.get("rt_curve")
            if rt_curve:
                input_curves.append(rt_curve)
            if swirr_metadata.get("resflag_source"):
                input_curves.append(swirr_metadata["resflag_source"])
        trace.input_curves = list(dict.fromkeys(input_curves))
        trace.output_curves = outputs
        trace.parameters = {
            "method": self.config.perm_method,
            "fallback_method": self.config.perm_fallback_method,
            "swirr_source": result.swi_source,
            "swirr_app_method": swirr_metadata.get("swirr_app_method"),
            "bvw_irr": result.bvw_irr,
            "swirr_metadata": swirr_metadata,
            "strategy": result.metadata.get("strategy"),
            "same_well_core_models": result.metadata.get("same_well_core_models"),
            "peer_core_models": result.metadata.get("peer_core_models"),
            "peer_wells_considered": result.metadata.get("peer_wells_considered"),
            "peer_core_enabled": result.metadata.get("peer_core_enabled"),
            "core_min_points": result.metadata.get("core_min_points"),
            "core_models": result.metadata.get("core_models"),
            "core_rejected_models": result.metadata.get("core_rejected_models"),
            "empirical_method": empirical_method,
            "empirical_error": result.metadata.get("empirical_error"),
            "empirical_attempted_methods": result.metadata.get("empirical_attempted_methods"),
            "empirical_method_metadata": method_metadata,
            "perm_max_clip_md": result.metadata.get("perm_max_clip_md"),
        }
        if self.config.perm_coefficients:
            trace.parameters["perm_coefficients"] = dict(self.config.perm_coefficients)
            if str(empirical_method).strip().lower() == "lith_vcl":
                decisions.append("perm_coefficients not used: lith_vcl takes no coefficients")
        if resflag_info is not None:
            trace.parameters["pre_permeability_resflag"] = resflag_info
        if result.metadata.get("core_sections_count"):
            decisions.append(
                f"Core-calibrated sections: {result.metadata['core_sections_count']} samples"
            )
        for model in result.metadata.get("core_models", []):
            source = model.get("source_well") or "unknown"
            formation = model.get("source_formation") or "all_depths"
            source_type = model.get("source_type") or "core"
            decisions.append(
                f"{source_type}: {source} / {formation} "
                f"(n={model.get('n_core_points')}, "
                f"core_phi={model.get('core_porosity_curve') or 'unknown'}, "
                f"core_perm={model.get('core_permeability_curve') or 'unknown'})"
            )
        if not self._reproducing_benchmark:
            trace.parameters["core_porosity_fits"] = result.metadata.get("core_porosity_fits")
            decisions.extend(self._core_porosity_decisions(result.metadata, "PHIE_CC" in outputs))
        if np.isscalar(result.swi_used):
            trace.parameters["swirr"] = float(result.swi_used)
            if result.swi_source == "hc_mask_p5":
                n_samples = swirr_metadata.get("hc_mask_samples")
                rt_curve = swirr_metadata.get("rt_curve", "RDEP")
                decisions.append(
                    f"SWIRR_APP = P5(SW where resflag and {rt_curve} > 5) = {float(result.swi_used):.3f} (n={n_samples})"
                )
            elif result.swi_source == "default_0.20":
                n_samples = swirr_metadata.get("hc_mask_samples")
                reason = swirr_metadata.get("reason")
                if n_samples is not None:
                    decisions.append(f"SWIRR_APP = 0.200 fallback because hydrocarbon mask had {n_samples} samples")
                elif reason:
                    decisions.append(f"SWIRR_APP = 0.200 fallback because {reason}")
                else:
                    decisions.append("SWIRR_APP = 0.200 fallback")
                self._warn_assumed(
                    "Irreducible water saturation assumed 0.20 (none could be picked from the "
                    "logs): PERM, RQI and FZI rest on that value."
                )
            else:
                decisions.append(f"Swirr = {float(result.swi_used):.3f} ({result.swi_source})")
        elif isinstance(result.swi_used, np.ndarray) and result.swi_source == "buckles_bvw_irr":
            rt_curve = swirr_metadata.get("rt_curve", "RDEP")
            decisions.append(
                f"SWIRR_APP = BVWirr / PHIE (Buckles), BVWirr = median(PHIE x SW where resflag and "
                f"{rt_curve} > 5) = {float(result.bvw_irr):.4f} (n={swirr_metadata.get('hc_mask_samples')})"
            )
        elif isinstance(result.swi_used, np.ndarray):
            decisions.append(f"Swirr resolved as curve ({result.swi_source})")
        elif result.swi_source == "not_required" and empirical_method == "lith_vcl":
            decisions.append(
                "Swirr not used because lith_vcl permeability uses PHIT, VCL, and lithology"
            )
        else:
            decisions.append("Swirr not used because permeability came from core calibration")
        if empirical_method == "lith_vcl":
            decisions.append(
                "lith_vcl empirical permeability: "
                "10 ** min(4.5, slope * PHIT * (1 - VCL * 1.2) + intercept)"
            )
            groups = method_metadata.get("lithology_parameter_groups") or []
            if groups:
                decisions.append("lith_vcl parameter groups used: " + ", ".join(groups))
        elif method_metadata.get("coefficients_set") and empirical_method in {"coates", "timur", "wyllie_rose"}:
            # The equation with the coefficients set.
            name = {"coates": "Coates-Denoo", "timur": "Timur", "wyllie_rose": "Wyllie-Rose"}[empirical_method]
            decisions.append(f"{name} empirical permeability: k = {method_metadata['equation']}")
        elif empirical_method == "coates":
            decisions.append(
                "Coates-Denoo empirical permeability: "
                "k = 10000 * PHIE^4 * ((1 - SWIRR) / SWIRR)^2"
            )
        elif empirical_method == "timur":
            decisions.append(
                "Timur empirical permeability: "
                "k = 0.136 * (100 * PHIE)^4.4 / (100 * SWIRR)^2"
            )
        elif empirical_method == "wyllie_rose":
            decisions.append(
                "Wyllie-Rose empirical permeability (Morris-Biggs oil constant): "
                "k = 62500 * PHIE^6 / SWIRR^2"
            )
        if result.metadata.get("empirical_error") and empirical_method == "lith_vcl":
            decisions.append(f"Earlier empirical method failed before lith_vcl: {result.metadata['empirical_error']}")
        if result.metadata.get("perm_max_clip_md") is not None:
            decisions.append(f"Permeability clipped to <= {result.metadata['perm_max_clip_md']:.0f} mD")
        decisions.append(f"Empirical permeability curve stored as PERM ({perm_desc})")
        if perm_core is not None:
            decisions.append("Core-corrected permeability stored as PERM_CC")
        decisions.append(f"Permeability strategy: {result.method}")
        decisions.append(f"Perm range: {np.nanmin(perm_output):.3f} - {np.nanmax(perm_output):.1f} mD")
        trace.decisions = decisions
        return trace

    def _swirr_flag_phie(
        self, well: "Well", vcl: Optional[np.ndarray], phie_curve
    ) -> tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """PHIE for the reservoir flag that picks the SWIRR_APP samples.

        Returns the reported PHIE unless ``config.swirr_flag_phi_cl`` is
        ``"all_shale_phit"``; then PHIE is recomputed with ``phi_cl`` equal to
        the mean PHIT of all shale-lithology samples (VCL > 0.9 when the
        lithology log has no shale). An explicit PNSH parameter, or missing
        PHIT/VCL/shale samples, keeps the reported PHIE; the returned details
        say which rule applied.
        """
        reported = np.asarray(phie_curve.data, dtype=float)
        if self.config.swirr_flag_phi_cl == "workflow":
            return reported, None
        details: Dict[str, Any] = {"swirr_flag_phi_cl": self.config.swirr_flag_phi_cl}
        if _parameter_float(well, "PNSH") is not None or _constant_zoned_parameter(well, "PNSH") is not None:
            details["swirr_flag_phi_cl_applied"] = "LAS PNSH parameter; reported PHIE used"
            return reported, details
        phit_curve = get_curve(well, ["PHIT"])
        if phit_curve is None or vcl is None:
            details["swirr_flag_phi_cl_applied"] = "PHIT or VCL unavailable; reported PHIE used"
            return reported, details
        phit = np.asarray(phit_curve.data, dtype=float)
        lithology = self._build_lithology_log(well)
        shale = (
            np.asarray([_is_shale_group(item) for item in lithology], dtype=bool)
            if isinstance(lithology, np.ndarray)
            else np.zeros(well.n_samples, dtype=bool)
        )
        if not np.any(shale):
            vcl_values = np.asarray(vcl, dtype=float)
            shale = np.isfinite(vcl_values) & (vcl_values > 0.9)
        control = shale & np.isfinite(phit)
        if not np.any(control):
            details["swirr_flag_phi_cl_applied"] = "no shale PHIT samples; reported PHIE used"
            return reported, details
        phi_cl = float(np.clip(np.nanmean(phit[control]), 0.0, 0.6))
        with np.errstate(invalid="ignore"):
            phie = np.clip(phit - np.asarray(vcl, dtype=float) * phi_cl, 0.0, phit)
        details.update(
            swirr_flag_phi_cl_applied="mean PHIT of all shale-lithology samples",
            swirr_flag_phi_cl_value=phi_cl,
            swirr_flag_phi_cl_samples=int(np.count_nonzero(control)),
        )
        return phie, details

    def _ensure_resflag_for_swirr(self, well: "Well") -> Optional[Dict[str, Any]]:
        """Create RESFLAG before permeability so Swirr uses workflow net-reservoir logic."""
        supplied = [
            name for name in ("RESFLAG", "NET_RES")
            if get_curve(well, [name]) is not None
        ]
        if supplied and not self.config.reuse_input_resflag:
            # An interpreted LAS carries the previous run's flag, whose cutoffs
            # may differ from this run's. Leave the curves alone but compute the
            # Swirr flag from this run's cutoffs, and say so.
            message = (
                f"{', '.join(supplied)} was already on the well; the reservoir flag behind "
                "SWIRR_APP was recomputed from this run's cutoffs. Set "
                "reuse_input_resflag=True to pick Swirr from the supplied flag."
            )
            if self._result is not None and message not in self._result.warnings:
                self._result.warnings.append(message)
        elif supplied:
            resflag_curve = get_curve(well, ["RESFLAG"])
            if resflag_curve is not None:
                data = np.asarray(resflag_curve.data, dtype=float)
                return {
                    "source_curve": "RESFLAG",
                    "created": False,
                    "reservoir_samples": int(np.nansum(data > 0.5)),
                }

            net_res_curve = get_curve(well, ["NET_RES"])
            if net_res_curve is not None:
                data = np.asarray(net_res_curve.data, dtype=float)
                well.add_curve("RESFLAG", data, "", "Reservoir Flag")
                return {
                    "source_curve": "NET_RES",
                    "created": True,
                    "vcl_curve": "existing_NET_RES",
                    "phie_curve": "existing_NET_RES",
                    "sw_curve": "existing_NET_RES",
                    "reservoir_samples": int(np.nansum(data > 0.5)),
                }

        if not self._has_curve(well, "PHIE") or not self._has_curve(well, "SW", "SWA", "SWT-LOG"):
            return None

        from petrophys.evaluation.net_pay.net_pay_module import NetPayModule

        phie_curve = self._first_curve(well, "PHIE")
        sw_curve = self._first_curve(well, "SW", "SWA", "SWT-LOG")
        vcl_curve = get_curve(well, ["VCL", "VCLAY"])
        vcl = np.asarray(vcl_curve.data, dtype=float) if vcl_curve is not None else None

        lithology = self._build_lithology_log(well)
        flag_phie, flag_phie_details = self._swirr_flag_phie(well, vcl, phie_curve)
        sw = np.asarray(sw_curve.data, dtype=float)
        if self._reproducing_benchmark:
            cutoffs = {
                "vcl_cutoff": self._setting("vcl_cutoff"),
                "phie_cutoff": self.config.phie_cutoff,
                "sw_cutoff": self._setting("sw_cutoff"),
            }
            from petrophys._internal.benchmark import historical_benchmark

            # The packages' NetPayModule took the carbonate cutoffs for the
            # label "carbonate" alone.
            with historical_benchmark():
                result = NetPayModule(**cutoffs, lithology=lithology).calculate(
                    well, vcl=vcl, phie=flag_phie, sw=sw
                )
        else:
            from petrophys.model.constants import DefaultParameters

            # Swirr is picked in clean, porous rock: below the saturation-routing
            # clay cutoff, where log Sw depends least on the shaly-sand model,
            # and above the clastic porosity standard in every lithology, so
            # tight water-bearing rock with a high resistivity does not pass as
            # hydrocarbon-bearing. The non-reservoir samples (evaporite, coal,
            # halite on the logs, bad hole) are left out as in net reservoir.
            _, result, details = self._current_net_flags(
                well, lithology, vcl, flag_phie, sw,
                vcl_cutoff=self._routing_vcl_cutoff(),
                phie_cutoff=(
                    self.config.phie_cutoff
                    if self.config.phie_cutoff is not None
                    else DefaultParameters.PHIE_CUTOFF_CLASTIC
                ),
            )
            cutoffs = {
                **details["cutoffs"],
                "excluded_samples": {
                    kind: item["samples"] for kind, item in details["excluded"].items()
                },
            }
        resflag = result.net_reservoir.astype(float)
        well.add_curve("RESFLAG", resflag, "", "Reservoir Flag")

        phie_label = phie_curve.mnemonic
        if flag_phie_details and "swirr_flag_phi_cl_value" in flag_phie_details:
            phie_label = (
                f"{phie_curve.mnemonic} recomputed with phi_cl="
                f"{flag_phie_details['swirr_flag_phi_cl_value']:.3f} (mean PHIT of "
                f"{flag_phie_details['swirr_flag_phi_cl_samples']} shale samples)"
            )
        return {
            **(flag_phie_details or {}),
            "source_curve": "NetPayModule.net_reservoir",
            "created": True,
            "vcl_curve": vcl_curve.mnemonic if vcl_curve is not None else "implicit_0",
            "phie_curve": phie_label,
            "sw_curve": sw_curve.mnemonic,
            **cutoffs,
            "reservoir_samples": int(np.nansum(resflag > 0.5)),
        }

    def _step_rock_quality(self, well: "Well") -> StepResult:
        from petrophys.evaluation.rock_quality.rock_quality_module import RockQualityModule

        trace = StepResult(step="rock_quality", success=True, method="RQI_FZI_phi_z")
        rq_module = RockQualityModule()
        perm, perm_source, _ = get_curve_data_depthwise(well, self._perm_names("PERM_CORE", "PERM_TIMUR"))
        phie, phie_source, _ = get_curve_data_depthwise(well, self._phie_names())
        result = rq_module.calculate(well, perm=perm, phi_e=phie)
        rq_module.add_to_well(well, result)
        trace.input_curves = [source for source in (perm_source, phie_source) if source]
        trace.output_curves = ["RQI", "FZI", "PHI_Z"]
        trace.decisions.append(
            f"RQI = 0.0314*sqrt({perm_source}/{phie_source}); "
            f"PHI_Z = {phie_source}/(1-{phie_source}); FZI = RQI/PHI_Z"
        )
        return trace

    def _step_bvw(self, well: "Well") -> StepResult:
        from petrophys.evaluation.bvw.bvw_module import calculate_bvw

        phie, phie_source, _ = get_curve_data_depthwise(well, self._phie_names())
        sw_curve = self._first_curve(well, "SW", "SWA", "SWT-LOG")
        sw = sw_curve.data
        bvw = calculate_bvw(phie, sw)
        well.add_curve("BVW", bvw, "v/v", "Bulk Volume Water")

        trace = StepResult(step="bvw", success=True, method="PHIE_x_SW")
        trace.input_curves = [phie_source, sw_curve.mnemonic]
        trace.output_curves = ["BVW"]
        trace.decisions.append(f"BVW = {phie_source} * {sw_curve.mnemonic}")
        return trace

    def _step_mhi(self, well: "Well") -> StepResult:
        from petrophys.evaluation.mhi.mhi_module import MhiModule

        sw_curve = self._first_curve(well, "SW", "SWA", "SWT-LOG")
        sxo_curve = self._first_sxo_curve(well)
        mhi_module = MhiModule()
        trace = StepResult(step="mhi", success=True, method="moveable_HC_index")
        trace.input_curves = [sw_curve.mnemonic, sxo_curve.mnemonic]
        if self._benchmark:
            # The eval package wrote MHI wherever SW and SXO were finite.
            diagnostic_mask = np.isfinite(np.asarray(sxo_curve.data, dtype=float)) & np.isfinite(
                np.asarray(sw_curve.data, dtype=float)
            )
        else:
            diagnostic_mask = mhi_module.diagnostic_sxo_mask(
                well,
                sxo=sxo_curve.data,
                sw=sw_curve.data,
            )
        diagnostic_samples = int(np.count_nonzero(diagnostic_mask))
        trace.parameters = {
            "diagnostic_sxo_samples": diagnostic_samples,
            "total_samples": int(well.n_samples),
            "rule": "MHI only where SXO has invaded-zone support from RSHA/RXO/RESS plus RMF",
        }
        if diagnostic_samples == 0:
            well.remove_curve("MHI")
            trace.output_curves = []
            trace.decisions.append(
                "MHI not computed because SXO was only SW^(1/5) fallback or lacked invaded-zone support"
            )
            return trace

        if self._benchmark:
            from petrophys._internal.benchmark import historical_benchmark
            from petrophys.evaluation.mhi.mhi_module import calculate_mhi

            with historical_benchmark():
                mhi = np.asarray(calculate_mhi(sxo_curve.data, sw_curve.data), dtype=float)
        else:
            mhi = mhi_module.calculate(well, sxo=sxo_curve.data, sw=sw_curve.data)
        well.add_curve("MHI", mhi, "", "Moveable Hydrocarbon Index")

        trace.output_curves = ["MHI"]
        trace.decisions.append(
            f"MHI = {sw_curve.mnemonic} / {sxo_curve.mnemonic} for {diagnostic_samples}/{well.n_samples} "
            "samples with diagnostic SXO; fallback-SXO rows were set to NaN"
        )
        return trace

    def _step_net_pay(self, well: "Well") -> StepResult:
        from petrophys.evaluation.diagnostics import HydrocarbonAnalysis
        from petrophys.evaluation.net_pay.net_pay_module import NetPayModule

        trace = StepResult(step="net_pay", success=True, method="cutoff_flags")
        lithology = self._build_lithology_log(well)

        vcl, vcl_source, _ = get_curve_data_depthwise(well, ["VCL", "VCLAY", "VSH"])
        phie, phie_source, _ = get_curve_data_depthwise(well, self._phie_names())
        sw, sw_source, _ = get_curve_data_depthwise(well, ["SW", "SWA", "SWT-LOG"])
        details: Optional[Dict[str, Any]] = None
        if self._reproducing_benchmark:
            from petrophys._internal.benchmark import historical_benchmark

            # The packages' NetPayModule took the carbonate cutoffs for the
            # label "carbonate" alone.
            with historical_benchmark():
                np_module = NetPayModule(
                    vcl_cutoff=self._setting("vcl_cutoff"),
                    phie_cutoff=self.config.phie_cutoff,
                    sw_cutoff=self._setting("sw_cutoff"),
                    lithology=lithology,
                )
                result = np_module.calculate(well, vcl=vcl, phie=phie, sw=sw)
        else:
            np_module, result, details = self._current_net_flags(well, lithology, vcl, phie, sw)
        np_module.add_to_well(well, result)

        well.add_curve("RESFLAG", result.net_reservoir.astype(float), "", "Reservoir Flag")

        if vcl is None:
            vcl_data = np.zeros(well.n_samples, dtype=float)
            vcl_source = "VCL default 0"
            if not self._historical:
                message = (
                    "No clay volume (no gamma-ray, SP or neutron-density curves): net pay assumes "
                    "VCL = 0, so the clay cutoff removes no sample and NET_ROCK, NET_RES and NET_PAY "
                    "rest on porosity and saturation alone."
                )
                trace.decisions.append(message)
                self._warn_assumed(message)
        else:
            vcl_data = np.asarray(vcl, dtype=float)
        seal_flag = np.full(vcl_data.shape, np.nan, dtype=float)
        valid_vcl = np.isfinite(vcl_data)
        seal_flag[valid_vcl] = (vcl_data[valid_vcl] >= 0.8).astype(float)
        well.add_curve("SEALFLAG", seal_flag, "", "Seal Flag")

        hc_result = HydrocarbonAnalysis(
            porosity_curves=self._phie_names(),
            class_tables="geolink" if self._reproducing_benchmark else "current",
            # Each sample's reservoir-quality limits come from its lithology
            # family, as its net-pay cutoffs do.
            lithology_family=None if self._reproducing_benchmark else self._cutoff_families(lithology),
        ).analyze(well, add_curves=True)

        trace.input_curves = [source for source in (vcl_source, phie_source, sw_source) if source]
        trace.output_curves = ["NET_ROCK", "NET_RES", "NET_PAY", "RESFLAG", "SEALFLAG"]
        if well.get("CROSSOVER_FLAG") is not None:
            trace.output_curves.append("CROSSOVER_FLAG")
        if details is None:
            trace.parameters = {
                "vcl_cutoff": self._setting("vcl_cutoff"),
                "phie_cutoff": self.config.phie_cutoff,
                "sw_cutoff": self._setting("sw_cutoff"),
                "lithology": self.config.lithology_domain if isinstance(lithology, str) else "per-depth",
                "seal_vcl_cutoff": 0.8,
            }
        else:
            trace.parameters = {
                **details["cutoffs"],
                "lithology": self.config.lithology_domain if isinstance(lithology, str) else "per-depth",
                "seal_vcl_cutoff": 0.8,
                "cutoff_source": details["cutoff_source"],
                "cutoffs_by_family": details["cutoffs_by_family"],
            }
        n_rock = int(np.nansum(well.get("NET_ROCK").data > 0.5)) if well.get("NET_ROCK") else 0
        n_res = int(np.nansum(well.get("NET_RES").data > 0.5)) if well.get("NET_RES") else 0
        n_pay = int(np.nansum(well.get("NET_PAY").data > 0.5)) if well.get("NET_PAY") else 0
        n_seal = int(np.nansum(seal_flag > 0.5))
        # Thicknesses are in the well's own depth unit (ft for the COSTA wells).
        step_len = abs(float(well.step)) if well.step else 0.15
        unit = well.depth_unit or "m"
        if details is None:
            vcl_cutoff_text = f"{self._setting('vcl_cutoff')}"
            if self.config.phie_cutoff is not None:
                phie_cutoff_text = f"{self.config.phie_cutoff}"
            else:
                from petrophys.model.constants import DefaultParameters

                phie_cutoff_text = (
                    f"{DefaultParameters.PHIE_CUTOFF_CLASTIC} "
                    f"({DefaultParameters.PHIE_CUTOFF_CARBONATE} where carbonate)"
                )
            sw_cutoff_text = f"{self._setting('sw_cutoff')}"
        else:
            vcl_cutoff_text, phie_cutoff_text, sw_cutoff_text = (
                _family_cutoff_text(details["cutoffs_by_family"], key)
                for key in ("vcl_cutoff", "phie_cutoff", "sw_cutoff")
            )
        trace.decisions.append(f"NET_ROCK: VCL < {vcl_cutoff_text} -> {n_rock} samples ({n_rock*step_len:.1f} {unit})")
        trace.decisions.append(f"NET_RES:  + {phie_source} > {phie_cutoff_text} -> {n_res} samples ({n_res*step_len:.1f} {unit})")
        trace.decisions.append(f"NET_PAY:  + {sw_source} < {sw_cutoff_text} -> {n_pay} samples ({n_pay*step_len:.1f} {unit})")
        if details is not None:
            excluded = {
                kind: {
                    **item,
                    "net_reservoir_removed": round(item["net_reservoir_samples_removed"] * step_len, 2),
                    "net_pay_removed": round(item["net_pay_samples_removed"] * step_len, 2),
                    "depth_unit": unit,
                }
                for kind, item in details["excluded"].items()
            }
            trace.parameters["excluded_from_net_reservoir_and_pay"] = excluded
            trace.decisions.extend(
                _net_pay_detail_decisions(details["cutoff_source"], details["cutoffs_by_family"], excluded)
            )
            bad_hole = excluded.get("bad_hole")
            if bad_hole and bad_hole["net_pay_removed"] > 0 and not self._reproducing_benchmark:
                self._result.warnings.append(
                    f"Bad hole removed {bad_hole['net_pay_removed']:g} {unit} of net pay "
                    f"(BAD_HOLE at {bad_hole['samples']} of {well.n_samples} samples). Check the "
                    "flag against the caliper and DRHO; bad_hole={'drho': ..., 'cali': ...} sets "
                    "its thresholds and skip_steps=['bad_hole'] keeps the pay."
                )
        trace.decisions.append(f"SEALFLAG: VCL >= 0.8 -> {n_seal} samples ({n_seal*step_len:.1f} {unit})")
        if well.get("CROSSOVER_FLAG") is not None:
            n_crossover = int(np.nansum(np.asarray(well.get("CROSSOVER_FLAG").data, dtype=float) > 0.5))
            trace.decisions.append(
                f"CROSSOVER_FLAG from hydrocarbon analysis -> {n_crossover} samples ({n_crossover*step_len:.1f} {unit})"
            )
        trace.decisions.extend(
            f"Hydrocarbon analysis: {section_name}"
            for section_name in hc_result.sections.keys()
        )
        return trace

    # ---- Helpers ----

    def _first_curve(self, well: "Well", *names: str):
        curve = get_curve(well, names)
        if curve is not None:
            return curve
        raise ValueError(f"None of the requested curves are available: {', '.join(names)}")

    def _has_sxo_curve(self, well: "Well") -> bool:
        if self._has_curve(well, "SXO", "SWXOA", "SXO-LOG"):
            return True
        return any(
            name.upper().startswith("SXO-")
            and get_curve(well, [name]) is not None
            for name in well.curves
        )

    def _first_sxo_curve(self, well: "Well"):
        for name in ("SXO", "SWXOA", "SXO-LOG"):
            curve = get_curve(well, [name])
            if curve is not None:
                return curve
        for name in well.curves:
            if name.upper().startswith("SXO-"):
                curve = get_curve(well, [name])
                if curve is not None:
                    return curve
        raise ValueError("No SXO curve is available")

    def _build_lithology_log(self, well: "Well") -> Union[str, np.ndarray]:
        """Build per-depth lithology from well curves, CSV, or global config."""
        if self._lithology_log is not None:
            return self._lithology_log

        if self._selected_lithology_source == "config":
            self._lithology_source = "config"
            return self.config.lithology_domain
        code_curve = self._lithology_code_curve(well)
        gt_curve = get_curve(well, [code_curve]) if code_curve is not None else None
        if self._selected_lithology_source not in {"auto", "vendor"}:
            gt_curve = None
        if gt_curve is not None:
            try:
                gt_data = np.asarray(gt_curve.data, dtype=float)
                user_groups = self._user_code_groups()
                if user_groups is not None:
                    mapping = {code: group for code, group in user_groups.items() if isinstance(code, int)}
                    gt_curve.metadata.custom["category_groups"] = dict(mapping)
                    gt_curve.metadata.custom["category_labels"] = {
                        code: label for code, label in self._user_code_labels.items() if isinstance(code, int)
                    }
                elif self._reproducing_benchmark:
                    mapping = GROUND_TRUTH_CODE_TO_GROUP
                else:
                    mapping = _CURRENT_GROUND_TRUTH_CODE_TO_GROUP
                    # Recorded so reports rebuild the groups this run used.
                    gt_curve.metadata.custom["category_groups"] = dict(mapping)
                self._lithology_log = _group_array_from_numeric_codes(gt_data, mapping)
                # The published packages name the source of LITHOLOGY_GEOLINK so.
                self._lithology_source = (
                    "ground_truth_lithology_geolink"
                    if code_curve == "LITHOLOGY_GEOLINK" and self._reproducing_benchmark
                    else f"vendor_lithology_log[{code_curve}]"
                )
                return self._lithology_log
            except (TypeError, ValueError):
                pass

        computed_curve = get_curve(well, ["LITHO"]) if self._dataset == "geolink_train" else None
        if computed_curve is not None:
            # The GeoLink training package grouped a well without lithology
            # codes by the LITHO its lithology step inferred, once written.
            self._lithology_log = _group_array_from_numeric_codes(
                np.asarray(computed_curve.data, dtype=float), COMPUTED_LITHO_CODE_TO_GROUP
            )
            self._lithology_source = "computed_lithology_curve"
            return self._lithology_log

        self._lithology_source = "config"
        return self.config.lithology_domain

    def _get_avg_formation_temp(self, well: "Well") -> float:
        """Get average formation temperature."""
        temp = get_curve(well, ["TEMP"])
        if temp is not None:
            return float(np.nanmean(self._temperature_curve_f(temp)))

        surface_temp_f = (
            self.config.surface_temp_f
            if self.config.surface_temp_f is not None
            else 68.0
        )
        bht_f = self.config.bht_f if self.config.bht_f is not None else _metadata_bht_f(well)
        if bht_f is not None:
            return (surface_temp_f + float(bht_f)) / 2.0

        if well.depth is not None:
            from petrophys.interpretation.temperature import calculate_temperature_profile

            try:
                result = calculate_temperature_profile(
                    well,
                    surface_temp=surface_temp_f,
                    surface_temp_unit="F",
                    td=self.config.td_ft,
                    td_unit="ft" if self.config.td_ft is not None else None,
                    depth_unit=well.depth_unit,
                )
                return float(np.nanmean(result.temperature))
            except Exception:
                pass

        return 150.0  # Last-resort estimate when depth is unavailable


def run_dynamic_workflow(
    well: "Well",
    config: Optional[DynamicWorkflowConfig] = None,
    _header_settings: bool = True,
    **kwargs
) -> DynamicWorkflowResult:
    """
    Run the dynamic workflow on a well, adding calculated curves to it.

    This is the lower-level form of :func:`petrophys.interpret`. It works on
    ``well`` itself: calculated curves are added to the object you pass in.
    Use ``petrophys.interpret(well)`` to leave the input unchanged.

    Parameters
    ----------
    well : Well
        Well object with raw log data. Modified in place.
    config : DynamicWorkflowConfig, optional
        Workflow configuration. When omitted, one is built from ``kwargs``.
    **kwargs
        DynamicWorkflowConfig settings; with ``config`` they override its
        fields on a copy.

    Returns
    -------
    DynamicWorkflowResult
        ``result.well`` is ``well``.

    Examples
    --------
    >>> result = run_dynamic_workflow(well, rw=0.05, lithology_domain='clastic')
    >>> print(f"Steps executed: {[s.step for s in result.steps_executed]}")
    >>> print(f"Steps skipped: {result.steps_skipped}")
    """
    profile = str(
        kwargs.get("calculation_profile", getattr(config, "calculation_profile", "current"))
    ).strip().lower()
    if _header_settings and profile not in _BENCHMARK_PROFILES and not reproducing_benchmark():
        # As petrophys.interpret: LAS header settings fill the settings left
        # at their defaults.
        config = _config_with_header_settings(well, config, kwargs)
    elif config is None:
        config = DynamicWorkflowConfig(**kwargs)
    elif kwargs:
        # A copy that keeps the record of the caller's settings.
        config = _copy_config(config, **kwargs)

    workflow = DynamicWorkflow(config)
    return workflow.run(well)
