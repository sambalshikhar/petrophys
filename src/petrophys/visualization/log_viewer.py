"""
Template-driven LAS/well-log plotting.

The viewer uses PetroPy-style XML templates while staying native to the local
``petrophys.model.well.Well`` data model. Templates define tracks, curves,
ranges, fills, cumulative volume tracks, lithology bands, and flag displays.
Missing curves are skipped by default and reported in ``warnings``; strict mode
turns those into errors.
"""

from __future__ import annotations

from petrophys.visualization.style import _styled_plot, save_figure

from dataclasses import dataclass, field, fields, replace
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import math
import os
import warnings
import csv
import re
from defusedxml import ElementTree as ET

import numpy as np

from petrophys._internal.statistics import finite_min_max
from petrophys.io.lithology_reader import read_lithology_intervals
from petrophys.preprocessing.standardization.alias_map import get_standard_mnemonic
from petrophys.preprocessing.standardization.unit_converter import (
    CORE_SATURATION_CURVES,
    standardize_core_saturation,
)
from petrophys.visualization.curve_labels import porosity_display_label

if TYPE_CHECKING:
    from petrophys.model.well import Well

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch, Rectangle

    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover - exercised by import environment
    HAS_MATPLOTLIB = False


BUILTIN_LOG_VIEWER_TEMPLATE_FILES: Dict[str, str] = {
    "raw": "raw.xml",
    "triple_combo": "raw.xml",
    "comprehensive": "geolink_full.xml",
    "full": "geolink_full.xml",
    "petrophysics": "petrophysics.xml",
    "interpretation": "petrophysics.xml",
    "minerals": "minerals.xml",
    "geolink": "geolink_full.xml",
    "geolink_full": "geolink_full.xml",
    "standard": "geolink_full.xml",
    "universal": "geolink_full.xml",
}


ADAPTIVE_LOG_VIEWER_TEMPLATE_NAMES = {"auto", "adaptive"}


CURVE_ALIAS_CANDIDATES: Dict[str, List[str]] = {
    "DEPT": ["DEPT", "DEPTH", "MD", "TDEP"],
    "DEPTH": ["DEPT", "DEPTH", "MD", "TDEP"],
    "GR": ["GR", "SGR", "GRC", "GR_CAL", "CGR", "HGR", "RGR", "GAM", "GRD", "GR_EDTC"],
    "SGR": ["SGR", "GR", "CGR"],
    "CALI": ["CALI", "CAL", "CAL_N", "HCAL", "DCAL", "LCAL", "XCAL"],
    # Method/measurement-specific names must not fall back to a different
    # quantity or a generic curve: the track header would claim data that was
    # never computed (DCAL is not a caliper; SPHI is sonic porosity, not SP).
    "DCAL": ["DCAL"],
    "BS": ["BS", "BIT", "BITSIZE", "BIT_SIZE"],
    "DRHO": ["DRHO", "DCOR", "DCOR_N", "ZCOR"],
    "SP": ["SP", "SSP", "SPBR"],
    "PEF": ["PEF", "PE", "PEFL", "PEFZ"],
    "ROP": ["ROP", "ROPA"],
    "RDEP": ["RDEP", "RT", "LLD", "ILD", "AT90", "RD", "RESDEEP"],
    "RT": ["RT", "RDEP", "LLD", "ILD", "AT90", "RD"],
    "RMED": ["RMED", "ILM", "LLS", "AT60", "RESM", "RM"],
    "RSHA": ["RSHA", "RSHAL", "RXO", "RESS", "MSFL", "SFL", "AT10"],
    # MSFL is a genuine flushed-zone measurement; shallow curves are not and
    # would render twice in a track that also plots RSHA.
    "RXO": ["RXO", "MSFL"],
    "RW": ["RW", "RWA", "RW_APP", "RWAPP"],
    "RWA": ["RWA", "RW_APP", "RWAPP", "RW"],
    "RMF": ["RMF", "RMFA", "RMF_APP", "RMFAPP"],
    "RMFA": ["RMFA", "RMF_APP", "RMFAPP", "RMF"],
    "RHOB": ["RHOB", "DEN", "ZDEN", "RHOZ"],
    "NPHI": ["NPHI", "TNPH", "NEUT", "NPHI_LS"],
    # HC-corrected neutron labels must not silently render the uncorrected
    # NPHI when no correction was computed.
    "NPHIC": ["NPHIC", "NPHI_HC", "NPHI_SHC"],
    "NPHI_HC": ["NPHI_HC", "NPHI_SHC", "NPHIC"],
    "DTC": ["DTC", "DT", "AC", "DTCO"],
    "DTS": ["DTS", "DTSM", "DTSH"],
    "TENS": ["TENS", "TENSION"],
    "VCL": ["VCL", "VCLAY", "VSH", "VSHGR", "VSHALE_GR", "VSHALE_GR_LINEAR", "VSHALE_GR_CLAVIER", "VSHALE_GR_STEIBER", "VSHALE_GR_LARIONOV_YOUNG", "VWCL", "VCLGR", "VCL_GR", "VCLND", "VCL_ND", "VSHALE", "VCLAV"],
    "VCLAY": [
        "VCLAY",
        "VCL",
        "VSH",
        "VSHGR",
        "VSHALE_GR",
        "VSHALE_GR_LINEAR",
    ],
    "RCL": ["RCL", "VCL", "VSH", "VSHALE_GR"],
    "PHIN": ["PHIN", "NPHI_HC", "NPHI_SHC", "NPHIC", "NPHI"],
    "PHID": ["PHID", "DPHI", "PHID_SHC", "DPOR", "DPHZ", "PORD"],
    "PHIS": ["PHIS", "SPHI", "PHIS_SHC", "SPOR", "PORS"],
    "PHIT": ["PHIT", "PHI_T", "PHIT_AVG", "PHIT_GAS_CORR", "PIGT", "PHIT_ND", "PHI_TOTAL", "TPOR"],
    "PHIE": ["PHIE", "PHIE_CC", "PHI_E", "PIGN", "PHIE_ND", "PHIEFF", "PHI_EFF", "EPOR"],
    "PHIE_CC": ["PHIE_CC"],
    "PHIT_AVG": ["PHIT_AVG", "PHIT"],
    "PHIT_GAS_CORR": ["PHIT_GAS_CORR"],
    "PHI_SEC": ["PHI_SEC", "PHIS", "PHID"],
    "PHIC": ["PHIC", "CPOR", "POR_CORE", "CORE_PHI", "CORE_POR", "CPOR_AMB"],
    "GDEN": ["GDEN", "GRAIN_DEN", "GRAIN_DENSITY"],
    "SW": ["SW", "SWA", "SWT", "SW_ARCHIE", "SW_INDONESIAN", "SW_SIMANDOUX", "SWE", "SW_E", "SWEFF", "SWU", "SUWI", "SW_AR"],
    "SW_ARCHIE": ["SW_ARCHIE"],
    "SW_INDONESIAN": ["SW_INDONESIAN", "SW_INDONESIA"],
    "SW_SIMANDOUX": ["SW_SIMANDOUX", "SW_SIM"],
    "SXO": ["SXO", "SWXO", "SWXOA", "SXOE", "SXOU", "SXOT", "SXO_E"],
    "SWXO": ["SWXO", "SXO", "SWXOA"],
    "SO": ["SO", "SH", "OIL_SAT", "1-SW"],
    "BVW": ["BVW", "BVWE", "BVWT", "BVW_E"],
    "MHI": ["MHI"],
    "SWIRR_APP": ["SWIRR_APP", "SWIRR", "SWI", "SWIR"],
    "PERM": ["PERM", "PERM_CC", "PERM_TIMUR", "PERM_COATES", "PERM_MORRIS_BIGGS_MODIFIED", "PERM_TIXIER", "PERM_WYK_EXP", "K", "KLOG", "KLOGH", "KINT", "K_TIMUR", "KPERM", "PERM_LOG", "PERMEABILITY"],
    "PERM_CC": ["PERM_CC"],
    "PERM_TIMUR": ["PERM_TIMUR"],
    "PERM_COATES": ["PERM_COATES"],
    "PERM_MORRIS_BIGGS_MODIFIED": ["PERM_MORRIS_BIGGS_MODIFIED"],
    "PERM_TIXIER": ["PERM_TIXIER"],
    "PERM_WYK_EXP": ["PERM_WYK_EXP"],
    "KH_CORE": ["KH_CORE", "K_CORE", "CPERM", "PERM_CORE", "CORE_PERM", "KAIR", "KHOR"],
    "KV_CORE": ["KV_CORE"],
    "RQI": ["RQI"],
    "FZI": ["FZI"],
    "PHI_Z": ["PHI_Z", "PHIZ"],
    "R35": ["R35"],
    "VQUAR": ["VQUAR", "VSAND", "VQTZ", "VQUARTZ", "VQUA"],
    "VSAND": ["VSAND", "VQUAR", "VQTZ", "VQUARTZ"],
    "VCALC": ["VCALC", "VLIM", "VCAL", "VCALCITE", "VCLC"],
    "VLIM": ["VLIM", "VCALC", "VCAL"],
    "VDOLO": ["VDOLO", "VDOL", "VDOLOMITE"],
    "VDOL": ["VDOL", "VDOLO", "VDOLOMITE"],
    "VANHY": ["VANHY", "VANH", "VANHYDRITE"],
    "VANH": ["VANH", "VANHY", "VANHYDRITE"],
    "MATRIX_VOL": ["MATRIX_VOL"],
    "LITHOLOGY_GEOLINK": ["LITHOLOGY_GEOLINK", "LITHOLOGY", "LITHO", "ROCK_TYPE"],
    "LITHO": ["LITHO", "LITHOLOGY_GEOLINK", "LITHOLOGY", "ROCK_TYPE"],
    "ROCK_TYPE": ["ROCK_TYPE", "LITHOLOGY_GEOLINK", "LITHO"],
    "NET_ROCK": ["NET_ROCK", "ROCKFLAG", "ROCK_FLAG", "NETROCK"],
    "NET_RES": ["NET_RES", "NET_RESERVOIR", "RESFLAG", "RES_FLAG", "NETRES", "NET_RES_FLAG", "RES"],
    "NET_PAY": ["NET_PAY", "PAYFLAG", "PAY_FLAG", "NETPAY", "PAY", "PAY_F"],
    "RESFLAG": ["RESFLAG", "NET_RES"],
    "SEALFLAG": ["SEALFLAG"],
    "CROSSOVER_FLAG": ["CROSSOVER_FLAG"],
    "BAD_HOLE": ["BAD_HOLE", "BAD_HOLE_FLAG", "BADHOLEFLAG", "WASHOUT_FLAG"],
    "BADHOLEFLAG": ["BADHOLEFLAG", "BAD_HOLE", "BAD_HOLE_FLAG"],
    "INVASION_RATIO": ["INVASION_RATIO", "IR"],
    "TEMP": ["TEMP", "TMPC", "TEMP_F", "TEMPERATURE", "MTEM", "TEMP_LOG"],
    "SAL": ["SAL", "SALINITY"],
    "SALAPP": ["SALAPP", "SAL_APP"],
    "SALMUDAPP": ["SALMUDAPP", "SAL_MUD_APP"],
    "SWC": ["SWC", "SW_CORE", "CORE_SW"],
    "SOC": ["SOC", "SO_CORE", "CORE_SO"],
    "VP": ["VP"],
    "VS": ["VS"],
    "VP_VS_RATIO": ["VP_VS_RATIO", "VPVS"],
    "POISSON": ["POISSON", "PR"],
    "YOUNG_MOD": ["YOUNG_MOD", "YME", "YM"],
    "BULK_MOD": ["BULK_MOD", "KMOD"],
    "SHEAR_MOD": ["SHEAR_MOD", "GMOD"],
    "UCS": ["UCS"],
}


NO_LITHOLOGY_LABEL = "No lithology available"


MINERAL_VOLUME_TRACK_NAMES = {
    "MINERAL VOLUMES",
    "MATRIX MINERAL VOLUMES",
    "LITHOLOGY VOLUMES",
}


MINERAL_VOLUME_CURVE_NAMES = {
    "VCL",
    "VCLAY",
    "VSH",
    "VQUAR",
    "VQTZ",
    "VSAND",
    "VCALC",
    "VCAL",
    "VLIM",
    "VDOLO",
    "VDOL",
    "VANHY",
    "VANH",
    "VANHYDRITE",
}


# Maps mineral-volume curves to the lithology whose colour + pattern they
# should inherit, so the cumulative mineral-composition track is styled identically
# to the lithology track (quartz -> sandstone dots, calcite -> limestone brick, etc.).
MINERAL_VOLUME_LITHOLOGY_LABELS: Dict[str, str] = {
    "VCLAY": "Shale", "VCL": "Shale", "VSH": "Shale", "VSHALE": "Shale", "VCLA": "Shale",
    "VQUAR": "Sandstone", "VQTZ": "Sandstone", "VSAND": "Sandstone",
    "VQUARTZ": "Sandstone", "VSND": "Sandstone",
    "VCALC": "Limestone", "VCAL": "Limestone", "VLIM": "Limestone",
    "VCALCITE": "Limestone", "VLIME": "Limestone",
    "VDOLO": "Dolomite", "VDOL": "Dolomite", "VDOLOMITE": "Dolomite",
    "VANHY": "Anhydrite", "VANH": "Anhydrite", "VANHYDRITE": "Anhydrite",
    "VHALI": "Halite", "VHALITE": "Halite", "VSALT": "Halite",
    "VSILT": "Silt",
    "VCOAL": "Coal",
    "VPYR": "Pyrite", "VPYRITE": "Pyrite",
    "VGYP": "Gypsum", "VGYPSUM": "Gypsum",
    "VFELD": "Arkose", "VFELDSPAR": "Arkose", "VKFELD": "Arkose",
}


COMPUTED_LITHOLOGY_CODE_LABELS: Dict[int, str] = {
    0: "Unknown",
    1: "Sandstone",
    2: "Limestone",
    3: "Dolomite",
    4: "Shale",
    5: "Anhydrite",
    6: "Salt",
    7: "Coal",
    8: "Mixed",
}


#: Rock names of the built-in lithology code table (Kingdom nomenclature), used
#: for any lithology code curve unless a codebook is given.
LITHOLOGY_CODE_LABELS: Dict[int, str] = {
    1: "Sandstone",
    2: "Silty Sand",
    3: "Cross Bedded Sst",
    4: "Sandy Silt",
    5: "Silt",
    6: "Shaly Silt",
    7: "Silty Shale",
    8: "Shale",
    9: "Chalk",
    10: "Porous Limestone",
    11: "Limestone",
    12: "Argillaceous Limestone",
    13: "Marlstone",
    14: "Calcareous Shale",
    15: "Dolomite",
    16: "Calcareous Cement",
    17: "Conglomerate",
    18: "Coal",
    19: "Cinerite",
    20: "Volcanic Tuff",
    21: "Halite",
    22: "Anhydrite",
    23: "Basement",
    24: "Plutonic Rock",
    25: "Biogenic Ooze",
    26: "Gap",
    27: "Sylvinite",
    28: "Volcanic Rock",
    29: "Spiculite",
    30: "Metamorphic Rock",
    31: "Calcareous Debris Flow",
    32: "Polyhalite",
    33: "Carnallite",
    34: "Kainite",
    35: "Aeolian Sandstone",
    36: "Arkose",
}


INDUSTRY_LITHOLOGY_LABEL_ALIASES: Dict[str, str] = {
    "aeolian sandstone": "SANDSTONE",
    "anhydrite": "ANHYDRITE",
    "argillaceous": "ARGILLACEOUS",
    "argillaceous limestone": "LIMESTONE",
    "arkose": "SANDSTONE",
    "basement": "IGNEOUS/INTRUSIVE METAMORPHIC",
    "biogenic ooze": "SHALE",
    "breccia": "BRECCIA",
    "calcareous": "CALCAREOUS",
    "calcareous cement": "CALCAREOUS",
    "calcareous debris flow": "BRECCIA",
    "calcareous sand": "SANDSTONE",
    "calcareous sandstone": "SANDSTONE",
    "calcareous shale": "CALCAREOUS CLAY/SHALE",
    "carnallite": "HALITE",
    "chalk": "CHALK",
    "chert": "CHERT",
    "cinerite": "TUFF",
    "clay": "CLAYSTONE/MUDSTONE",
    "clay shale": "CLAYSTONE/MUDSTONE",
    "claystone": "CLAYSTONE/MUDSTONE",
    "coal": "COAL (LIGNITE PEAT DESCRIPTORS)",
    "conglomerate": "CONGLOMERATE",
    "cross bedded sand": "SANDSTONE",
    "cross bedded sandstone": "SANDSTONE",
    "cross bedded sst": "SANDSTONE",
    "crossbedded sand": "SANDSTONE",
    "dolomite": "DOLOMITE",
    "dolomite2": "DOLOMITE",
    "dolomitic": "DOLOMITIC",
    "extrusive": "VOLCANICS",
    "foraminifera": "FORAMINIFERA",
    "gap": "UNKNOWN",
    "glauconite": "GLAUCONITE",
    "gravel": "CONGLOMERATE",
    "gypsum": "GYPSUM",
    "halite": "HALITE",
    "igneous": "IGNEOUS/INTRUSIVE METAMORPHIC",
    "igneous intrusive metamorphic": "IGNEOUS/INTRUSIVE METAMORPHIC",
    "intrusive": "IGNEOUS/INTRUSIVE METAMORPHIC",
    "intrusive acid igneous": "INTRUSIVE ACID IGNEOUS",
    "intrusive basic igneous": "INTRUSIVE BASIC IGNEOUS",
    "kainite": "HALITE",
    "kaīnite": "HALITE",
    "limestone": "LIMESTONE",
    "marl": "MARL",
    "marlstone": "MARL",
    "metamorphic": "METAMORPHIC ROCK",
    "metamorphic rock": "METAMORPHIC ROCK",
    "mica": "MICA",
    "mixed": "MIXED",
    "mudstone": "CLAYSTONE/MUDSTONE",
    "plutonic rock": "IGNEOUS/INTRUSIVE METAMORPHIC",
    "polyhalite": "POLYHALITE",
    "porous limestone": "LIMESTONE",
    "pyrite": "PYRITE",
    "sand": "SANDSTONE",
    "sandy": "SANDY",
    "sandy limestone": "LIMESTONE",
    "sandy shale": "SHALE",
    "sandy silt": "SILTSTONE/SILT",
    "sandstone": "SANDSTONE",
    "salt": "HALITE",
    "shale": "SHALE",
    "shaly sand": "SANDSTONE",
    "shaly sandstone": "SANDSTONE",
    "shaly silt": "SILTSTONE/SILT",
    "silt": "SILTSTONE/SILT",
    "siltstone": "SILTSTONE/SILT",
    "silty": "SILTY",
    "silty sand": "SANDSTONE",
    "silty shale": "SHALE",
    "spiculite": "CHERT",
    "sylvinite": "HALITE",
    "tuff": "TUFF",
    "unknown": "UNKNOWN",
    "volcanic": "VOLCANICS",
    "volcanic rock": "VOLCANICS",
    "volcanic tuff": "TUFF",
    "volcanics": "VOLCANICS",
}


def _sample_cell_fill(depth: np.ndarray, active: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Depths and mask for ``fill_betweenx`` that shade each active sample's cell.

    ``fill_betweenx(where=mask)`` fills only between consecutive active
    samples, so a single flagged sample has no height and an N-sample run is
    drawn N-1 steps thick. Each sample is drawn over depth +/- half a step
    instead (capped at half the median step across gaps).
    """
    depth = np.asarray(depth, dtype=float).reshape(-1)
    active = np.asarray(active, dtype=bool).reshape(-1)
    steps = np.diff(depth)
    finite_steps = np.abs(steps[np.isfinite(steps) & (steps != 0)])
    half_step = 0.5 * float(np.median(finite_steps)) if finite_steps.size else 0.5
    half = np.clip(np.nan_to_num(steps / 2.0, nan=half_step), -half_step, half_step)
    if half.size:
        upper = np.concatenate([half, half[-1:]])
        lower = np.concatenate([half[:1], half])
    else:
        upper = lower = np.full(depth.shape, half_step)
    edges = np.column_stack([depth - lower, depth + upper]).reshape(-1)
    return edges, np.repeat(active, 2)


def _lithology_label_key(label: Any) -> str:
    text = str(label or "").strip().lower()
    text = re.sub(r"[\s_/\-()]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _lithology_style_text(label: Any) -> str:
    key = _lithology_label_key(label)
    alias = INDUSTRY_LITHOLOGY_LABEL_ALIASES.get(key)
    if alias is None:
        return key
    return f"{key} {_lithology_label_key(alias)}"


def _lookup_lithology_style(label: Any, styles: Mapping[str, Any], default: Any = "") -> Any:
    exact_key = _lithology_label_key(label)
    sorted_styles = sorted(styles.items(), key=lambda item: len(item[0]), reverse=True)
    for key, value in sorted_styles:
        if _lithology_label_key(key) == exact_key:
            return value

    lowered = _lithology_style_text(label)
    for key, value in sorted_styles:
        # Match at a word start only, so 'basalt' does not match 'salt'.
        if re.search(r"(?<![a-z0-9])" + re.escape(_lithology_label_key(key)), lowered):
            return value
    return default


#: Alias of :data:`LITHOLOGY_CODE_LABELS`.
GEOLINK_LITHOLOGY_CODE_LABELS = LITHOLOGY_CODE_LABELS

#: Environment variable naming a lithology code table (CSV or Excel) that
#: overrides the built-in lithology code labels for every plot in the process.
LITHOLOGY_MAPPING_ENV_VAR = "PETROPHYS_LITHOLOGY_MAPPING"

#: Additional lithology code tables to consult, as absolute paths. Empty by
#: default: the built-in lithology labels are used unless a table is supplied
#: explicitly (``lithology_mapping``/``lithology_mapping_df`` arguments or
#: :data:`LITHOLOGY_MAPPING_ENV_VAR`). Files are never discovered implicitly.
GEOLINK_LITHOLOGY_MAPPING_FILENAMES: Tuple[str, ...] = ()


def _add_lithology_mapping_label(out: Dict[Any, str], code: Any, label: Any) -> None:
    if code is None or label is None:
        return
    label_text = str(label).strip()
    code_text = str(code).strip()
    if not label_text or label_text.lower() == "nan":
        return
    if not code_text or code_text.lower() == "nan":
        return

    out[code] = label_text
    out[code_text] = label_text
    try:
        numeric = float(code)
    except (TypeError, ValueError):
        return
    if np.isfinite(numeric):
        out[numeric] = label_text
        if numeric.is_integer():
            out[int(numeric)] = label_text


def _find_lithology_mapping_column(
    records: Sequence[Mapping[str, Any]],
    candidates: Sequence[str],
) -> Optional[str]:
    normalized_candidates = [candidate.lower() for candidate in candidates]
    for record in records:
        columns = {str(column).strip().lower(): str(column) for column in record.keys()}
        for candidate in normalized_candidates:
            if candidate in columns:
                return columns[candidate]
    return None


def _lithology_mapping_from_records(records: Sequence[Mapping[str, Any]]) -> Dict[Any, str]:
    if not records:
        return {}
    from petrophys.io.lithology_reader import read_lithology_codes
    out: Dict[Any, str] = {}
    for code, label in read_lithology_codes(records).items():
        _add_lithology_mapping_label(out, code, label)
    return out


def _candidate_geolink_lithology_mapping_paths() -> Tuple[Path, ...]:
    """Return explicitly configured lithology code tables, in priority order."""
    configured: List[str] = []
    env_value = os.environ.get(LITHOLOGY_MAPPING_ENV_VAR, "").strip()
    if env_value:
        configured.append(env_value)
    configured.extend(GEOLINK_LITHOLOGY_MAPPING_FILENAMES)

    paths: List[Path] = []
    for entry in configured:
        path = Path(entry).expanduser().resolve()
        if path not in paths:
            paths.append(path)
    return tuple(paths)


def _read_lithology_mapping_records(path: Path) -> List[Dict[str, Any]]:
    from petrophys.io.geology_reader import _read_table
    return _read_table(path).to_dict("records")


@lru_cache(maxsize=8)
def _load_lithology_mapping_tables(
    paths: Tuple[Path, ...],
    stamps: Tuple[Any, ...] = (),
) -> Dict[Any, str]:
    # ``stamps`` (file modification time and size) is part of the cache key
    # so an edited table is read again.
    for path in paths:
        if not path.exists():
            warnings.warn(
                f"Lithology code table {str(path)!r} does not exist; "
                "using the built-in lithology labels.",
                UserWarning,
                stacklevel=4,
            )
            continue
        try:
            mapping = _lithology_mapping_from_records(_read_lithology_mapping_records(path))
        except (ImportError, OSError, KeyError, ValueError) as exc:
            warnings.warn(
                f"Could not read lithology code table {str(path)!r} ({exc}); "
                "using the built-in lithology labels.",
                UserWarning,
                stacklevel=4,
            )
            continue
        if mapping:
            return mapping
        warnings.warn(
            f"Lithology code table {str(path)!r} has no recognizable code and "
            "label columns (e.g. 'code' and 'lithology', or 'Abbreviation' and 'Attribute'); "
            "using the built-in lithology labels.",
            UserWarning,
            stacklevel=4,
        )
    return {}


def _default_geolink_lithology_mapping() -> Dict[Any, str]:
    """Return the lithology code table configured for this process, if any."""
    paths = _candidate_geolink_lithology_mapping_paths()
    stamps = []
    for path in paths:
        try:
            stat = path.stat()
            stamps.append((stat.st_mtime_ns, stat.st_size))
        except OSError:
            stamps.append(None)
    return _load_lithology_mapping_tables(paths, tuple(stamps))


# Industry-standard lithology fill colours, keyed by normalized label.
# Follows the reference industry legend by FAMILY (yellow sands, teal carbonates,
# pink evaporites, red igneous/volcanic, orange conglomerate, brown shale) but
# each member within a family is given a clearly stepped shade so adjacent codes
# (e.g. sandstone vs cross-bedded vs arkose, or halite vs sylvinite vs carnallite)
# are visually distinguishable, not just told apart by their pattern.
LITHOLOGY_LABEL_COLORS: Dict[str, str] = {
    "unknown": "#d9d9d9",
    "gap": "#ffffff",
    # --- shale / clay (brown family) ---
    "claystone/mudstone": "#8b6208",
    "mudstone": "#8b6208",
    "claystone": "#8b6208",
    "clay": "#8b6208",
    "silty shale": "#6e7a36",                # olive (silt+clay)
    "shale": "#8b6208",                       # brown (anchor)
    "biogenic ooze": "#b58a22",               # light gold-brown
    # --- sands (yellow family) ---
    "sandstone": "#ffd000",                   # vivid yellow (anchor)
    "silty sand": "#e4d27a",                  # muted khaki-yellow
    "cross bedded sst": "#ffab00",            # amber
    "crossbedded sand": "#ffab00",
    "aeolian sandstone": "#fceeb0",           # pale cream
    "arkose": "#d49a3a",                       # goldenrod-tan (feldspathic)
    "calcareous sandstone": "#f0e08a",
    "shaly sandstone": "#e6d488",
    "shaly sand": "#e6d488",
    "sand": "#ffd000",
    # --- silts ---
    "sandy silt": "#d9b36a",                  # tan
    "silt": "#f2a06a",                         # salmon-orange
    "shaly silt": "#ae9226",                   # dark olive-gold
    "siltstone/silt": "#e6bf3f",
    "siltstone": "#e6bf3f",
    # --- conglomerate / breccia (orange family) ---
    "conglomerate": "#f57f17",                 # orange (anchor)
    "gravel": "#f57f17",
    "breccia": "#b5400c",                       # burnt orange
    "calcareous debris flow": "#f7a833",        # amber
    # --- coal ---
    "coal (lignite peat descriptors)": "#000000",
    "coal": "#000000",
    # --- carbonates (teal family) ---
    "chalk": "#cff3e4",                         # pale mint
    "porous limestone": "#74d4be",              # deeper teal
    "limestone": "#b0f0e0",                     # light teal (anchor)
    "argillaceous limestone": "#95c9d6",        # blue-gray teal (clay)
    "sandy limestone": "#a6e8d6",
    "dolomite": "#7fe4d2",                      # cyan-teal
    "dolomite2": "#7fe4d2",
    # --- marl / calcareous shale (light blue) ---
    "marl": "#bfe3ec",
    "marlstone": "#bfe3ec",
    "calcareous clay/shale": "#a4d2dc",
    "calcareous shale": "#a4d2dc",
    # --- evaporites (pink / magenta family) ---
    "halite": "#f3b2de",                        # pink (anchor)
    "salt": "#f3b2de",
    "sylvinite": "#e87bc4",                     # rose-magenta
    "carnallite": "#d646ac",                    # magenta
    "kainite": "#fbdcef",                       # pale lavender-pink
    "polyhalite": "#dca6ce",                    # mauve
    "anhydrite": "#efa6e8",                     # lilac
    "gypsum": "#eec2ea",                        # pale lilac
    "calcareous cement": "#f7f7f7",
    # --- igneous / volcanic / metamorphic (red / magenta family) ---
    "basement": "#f2705a",                      # salmon-red
    "plutonic rock": "#dc1c28",                 # deep red
    "intrusive acid igneous": "#dc1c28",
    "intrusive basic igneous": "#b82a34",
    "igneous/intrusive metamorphic": "#dc1c28",
    "intrusive": "#dc1c28",
    "volcanic rock": "#ff7a2e",                 # orange-red
    "volcanics": "#ff7a2e",
    "volcanic": "#ff7a2e",
    "extrusive": "#ff6347",
    "metamorphic rock": "#c4366a",              # crimson-purple
    "cinerite": "#c44fea",                      # purple
    "volcanic tuff": "#e83abe",                 # magenta
    "tuff": "#e83abe",
    # --- chert / spiculite ---
    "chert": "#eaeaea",                         # very light gray
    "spiculite": "#cccccc",                     # gray
    # --- accessory / modifier labels ---
    "sandy": "#f7f7f7",
    "silty": "#eeeeee",
    "argillaceous": "#fafafa",
    "limestone stringer": "#f0f0f0",
    "dolomite stringer": "#f5f5f5",
    "calcareous": "#fdfdfd",
    "dolomitic": "#f2f2f2",
    "macrofossils": "#e0e0e0",
    "foraminifera": "#e8e8e8",
    "glauconite": "#4daf4a",
    "mica": "#bdbdbd",
    "pyrite": "#9e9e9e",
    "carbonaceous": "#303030",
    "mixed": "#cab2d6",
}


FLAG_COLORS: Dict[str, str] = {
    "NET_ROCK": "#fdae61",
    "NET_RES": "#fee08b",
    "NET_PAY": "#1a9850",
    "RESFLAG": "#66bd63",
    "SEALFLAG": "#74add1",
}


RESISTIVITY_CURVE_NAMES = {
    "RDEP",
    "RT",
    "LLD",
    "ILD",
    "AT90",
    "RD",
    "RESDEEP",
    "RMED",
    "ILM",
    "LLS",
    "AT60",
    "RESM",
    "RM",
    "RSHA",
    "RSHAL",
    "RXO",
    "RESS",
    "MSFL",
    "SFL",
    "AT10",
}


FRACTION_SATURATION_CURVE_NAMES = {
    "SW",
    "SWT",
    "SWE",
    "SWA",
    "SWC",
    "SXO",
    "SO",
    "SOC",
    "SG",
    "SH",
    "SWI",
    "SWIR",
    "SWIRR",
    "SWIRR_APP",
    "NMR_SWIRR",
    "SWIRR_NMR",
    "CMR_SWIRR",
    "SWIRR_CMR",
}


DEFAULT_CURVE_DISPLAY_UNITS: Dict[str, str] = {
    "GR": "API",
    "SGR": "API",
    "CGR": "API",
    "HGR": "API",
    "CALI": "in",
    "CAL": "in",
    "BS": "in",
    "BITSIZE": "in",
    "DTC": "us/ft",
    "DT": "us/ft",
    "DTS": "us/ft",
    "RHOB": "g/cc",
    "RHOZ": "g/cc",
    "DRHO": "g/cc",
    "PEF": "b/e",
    "PE": "b/e",
    "NPHI": "v/v",
    "PHIN": "v/v",
    "PHIT": "v/v",
    "PHIE": "v/v",
    "PHIE_CC": "v/v",
    "PHIT_GAS_CORR": "v/v",
    "PHID": "v/v",
    "PHIS": "v/v",
    "PHIC": "v/v",
    "CPOR": "v/v",
    "VCL": "v/v",
    "VCLAY": "v/v",
    "VSH": "v/v",
    "BVW": "v/v",
    "PERM": "mD",
    "PERM_CC": "mD",
    "K": "mD",
    "KH": "mD",
    "RQI": "microns",
    "FZI": "microns",
    "PHIZ": "v/v",
    "PHI_Z": "v/v",
    "VQUAR": "v/v",
    "VCALC": "v/v",
    "VDOLO": "v/v",
    "VANHY": "v/v",
    "VHALI": "v/v",
}


UNIT_DISPLAY_ALIASES: Dict[str, str] = {
    "api": "API",
    "gapi": "API",
    "in": "in",
    "inch": "in",
    "inches": "in",
    "ohmm": "ohm.m",
    "ohm.m": "ohm.m",
    "ohm-m": "ohm.m",
    "ohm m": "ohm.m",
    "ohmmeter": "ohm.m",
    "ohm.meter": "ohm.m",
    "ohm-meter": "ohm.m",
    "ohmmetre": "ohm.m",
    "ohm.metre": "ohm.m",
    "ohm-metre": "ohm.m",
    "us/ft": "us/ft",
    "us/f": "us/ft",
    "usec/ft": "us/ft",
    "usec/f": "us/ft",
    "microsecond/ft": "us/ft",
    "microseconds/ft": "us/ft",
    "g/cc": "g/cc",
    "g/cm3": "g/cc",
    "g/c3": "g/cc",
    "gm/cc": "g/cc",
    "b/e": "b/e",
    "barn/e": "b/e",
    "barns/e": "b/e",
    "v/v": "v/v",
    "vol/vol": "v/v",
    "frac": "frac",
    "fraction": "frac",
    "decimal": "frac",
    "dec": "frac",
    "unitless": "frac",
    "%": "%",
    "percent": "%",
    "md": "mD",
    "millidarcy": "mD",
    "millidarcies": "mD",
    "um": "microns",
    "micron": "microns",
    "microns": "microns",
}


LITHOLOGY_HATCH_PATTERNS: Dict[str, str] = {
    "claystone/mudstone": "--",
    "mudstone": "--",
    "claystone": "--",
    "clay": "--",
    "shale": "----",
    "conglomerate": "oo",
    "gravel": "oo",
    "calcareous sandstone": "..",
    "shaly sandstone": "..",
    "shaly sand": "..",
    "sandstone": "..",
    "cross bedded sst": "..",
    "crossbedded sand": "..",
    "arkose": "..",
    "sand": "..",
    "siltstone/silt": "-.",
    "siltstone": "-.",
    "silt": "-.",
    "breccia": "xx",
    "chalk": "++",
    "sandy limestone": "+",
    "porous limestone": "+",
    "argillaceous limestone": "+",
    "limestone": "+",
    "dolomite": "//",
    "dolomite2": "//",
    "calcareous clay/shale": "+-",
    "calcareous shale": "+-",
    "marl": "--",
    "marlstone": "--",
    "halite": "++",
    "salt": "++",
    "sylvinite": "++",
    "carnallite": "++",
    "kainite": "++",
    "anhydrite": "..",
    "gypsum": "\\\\",
    "polyhalite": "xx",
    "intrusive acid igneous": "+.",
    "intrusive basic igneous": "x.",
    "igneous/intrusive metamorphic": "//",
    "basement": "//",
    "plutonic rock": "//",
    "intrusive": "//",
    "extrusive": "**",
    "cinerite": "\\\\",
    "tuff": "\\\\",
    "volcanics": "**",
    "volcanic": "**",
    "metamorphic rock": "////",
    "sandy": "..",
    "silty": "-.",
    "argillaceous": "-",
    "limestone stringer": "+-",
    "dolomite stringer": "//",
    "calcareous": "+",
    "dolomitic": "/",
    "macrofossils": "o",
    "foraminifera": "oo",
    "chert": "*",
    "glauconite": "*",
    "pyrite": "o",
    "mica": ".",
}


LITHOLOGY_CUSTOM_PATTERNS: Dict[str, str] = {
    "claystone/mudstone": "dash",
    "mudstone": "dash",
    "claystone": "dash",
    "clay": "dash",
    "shale": "horizontal_dense",
    "conglomerate": "pebbles",
    "gravel": "pebbles",
    "sandstone": "dots",
    "cross bedded sst": "dots",
    "crossbedded sand": "dots",
    "arkose": "dots",
    "sand": "dots",
    "siltstone/silt": "dash_dot",
    "siltstone": "dash_dot",
    "silt": "dash_dot",
    "breccia": "triangles",
    "coal": "solid",
    "chalk": "chalk_grid",
    "argillaceous limestone": "block_grid",
    "limestone": "block_grid",
    "dolomite": "diagonal",
    "calcareous clay/shale": "calcareous_clay",
    "calcareous shale": "calcareous_clay",
    "marl": "waves",
    "marlstone": "waves",
    "halite": "grid",
    "salt": "grid",
    "sylvinite": "grid",
    "carnallite": "grid",
    "kainite": "grid",
    "anhydrite": "carets",
    "gypsum": "chevrons_left",
    "polyhalite": "cross_diagonal",
    "intrusive acid igneous": "plus_symbols",
    "intrusive basic igneous": "x_symbols",
    "igneous/intrusive metamorphic": "igneous_dash",
    "basement": "igneous_dash",
    "plutonic rock": "igneous_dash",
    "intrusive": "igneous_dash",
    "extrusive": "volcanic_v",
    "cinerite": "tuff_slashes",
    "volcanic tuff": "tuff_slashes",
    "tuff": "tuff_slashes",
    "volcanics": "volcanic_v",
    "volcanic": "volcanic_v",
    "metamorphic rock": "diagonal_dense",
    "sandy": "dots",
    "silty": "dash_dot",
    "argillaceous": "argillaceous_dash",
    "limestone stringer": "limestone_stringer",
    "dolomite stringer": "dolomite_stringer",
    "calcareous": "calcareous_t",
    "dolomitic": "dolomitic_slash",
    "macrofossils": "macrofossils",
    "foraminifera": "foraminifera",
    "calcareous debris flow": "triangles",
    "chert": "triangle_single",
    "spiculite": "triangle_single",
    "glauconite": "star_symbols",
    "pyrite": "square_symbols",
    "mica": "mica_symbols",
    "carbonaceous": "carbonaceous_bar",
}


LITHOLOGY_TILE_PATTERNS = {
    "chalk_grid": "chalk",
    "block_grid": "limestone",
    "diagonal": "dolomite",
    "calcareous_clay": "calcareous_clay",
    "waves": "marl",
}


COMMON_LITHOLOGY_LEGEND_LABELS: Dict[str, str] = {
    "unknown": "Unknown",
    "gap": "Gap",
    "claystone/mudstone": "Shale/Claystone",
    "mudstone": "Shale/Claystone",
    "claystone": "Shale/Claystone",
    "clay": "Shale/Claystone",
    "silty shale": "Shale/Claystone",
    "shale": "Shale/Claystone",
    "biogenic ooze": "Shale/Claystone",
    "sandstone": "Sandstone",
    "silty sand": "Sandstone",
    "cross bedded sst": "Sandstone",
    "crossbedded sand": "Sandstone",
    "aeolian sandstone": "Sandstone",
    "arkose": "Sandstone",
    "sand": "Sandstone",
    "siltstone/silt": "Siltstone/Silt",
    "sandy silt": "Siltstone/Silt",
    "shaly silt": "Siltstone/Silt",
    "siltstone": "Siltstone/Silt",
    "silt": "Siltstone/Silt",
    "conglomerate": "Conglomerate/Breccia",
    "breccia": "Conglomerate/Breccia",
    "gravel": "Conglomerate/Breccia",
    "chalk": "Limestone/Chalk",
    "porous limestone": "Limestone/Chalk",
    "argillaceous limestone": "Limestone/Chalk",
    "limestone": "Limestone/Chalk",
    "dolomite": "Dolomite",
    "dolomite2": "Dolomite",
    "calcareous clay/shale": "Marl/Calcareous Shale",
    "calcareous shale": "Marl/Calcareous Shale",
    "marl": "Marl/Calcareous Shale",
    "marlstone": "Marl/Calcareous Shale",
    "halite": "Evaporite",
    "salt": "Evaporite",
    "sylvinite": "Evaporite",
    "carnallite": "Evaporite",
    "kainite": "Evaporite",
    "anhydrite": "Evaporite",
    "gypsum": "Evaporite",
    "polyhalite": "Evaporite",
    "coal": "Coal",
    "coal (lignite peat descriptors)": "Coal",
    "chert": "Chert/Siliceous",
    "spiculite": "Chert/Siliceous",
    "intrusive acid igneous": "Igneous/Volcanic",
    "intrusive basic igneous": "Igneous/Volcanic",
    "igneous/intrusive metamorphic": "Igneous/Volcanic",
    "basement": "Igneous/Volcanic",
    "plutonic rock": "Igneous/Volcanic",
    "intrusive": "Igneous/Volcanic",
    "extrusive": "Igneous/Volcanic",
    "cinerite": "Igneous/Volcanic",
    "volcanic tuff": "Igneous/Volcanic",
    "tuff": "Igneous/Volcanic",
    "volcanics": "Igneous/Volcanic",
    "volcanic": "Igneous/Volcanic",
    "metamorphic rock": "Metamorphic",
    "calcareous cement": "Calcareous Cement",
    "calcareous debris flow": "Conglomerate/Breccia",
}


BUILTIN_LITHOLOGY_LABEL_KEYS = {
    _lithology_label_key(label)
    for label in (*LITHOLOGY_CODE_LABELS.values(), *COMPUTED_LITHOLOGY_CODE_LABELS.values())
}


AUTO_TEMPLATE_SKIP_CURVES = {
    "KV_CORE",
}


DISCRETE_CURVE_NAMES = {
    "PHIC",
    "CPOR",
    "POR_CORE",
    "CORE_PHI",
    "KH_CORE",
    "K_CORE",
    "SWC",
    "SW_CORE",
    "SOC",
    "SO_CORE",
    "GDEN",
    "GRAIN_DEN",
    "GRAIN_DENSITY",
}


BAD_HOLE_CURVE_NAMES = ("BAD_HOLE", "BADHOLEFLAG", "BAD_HOLE_FLAG", "WASHOUT_FLAG")
CROSSOVER_CURVE_NAMES = ("CROSSOVER_FLAG",)


def _read_builtin_template_text(filename: str) -> str:
    try:
        return (
            resources.files("petrophys.visualization")
            .joinpath("templates", filename)
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError):
        return (Path(__file__).parent / "templates" / filename).read_text(encoding="utf-8")


DEFAULT_LOG_VIEWER_TEMPLATES: Dict[str, str] = {
    name: _read_builtin_template_text(filename)
    for name, filename in BUILTIN_LOG_VIEWER_TEMPLATE_FILES.items()
}


@dataclass
class LogCurveSpec:
    curve_name: str
    display_name: Optional[str] = None
    left: Optional[float] = None
    right: Optional[float] = None
    color: str = "black"
    line_width: float = 0.8
    line_style: str = "-"
    alpha: float = 1.0
    fill: Optional[str] = None
    fill_color: Optional[str] = None
    fill_color_map: Optional[str] = None
    fill_alpha: float = 0.28
    left_color_value: Optional[float] = None
    right_color_value: Optional[float] = None
    left_cutoff_fill: Optional[float] = None
    right_cutoff_fill: Optional[float] = None
    left_crossover: Optional[str] = None
    right_crossover: Optional[str] = None
    marker: Optional[str] = None
    marker_size: float = 0.0
    candidates: List[str] = field(default_factory=list)
    #: Style keys set through ``curve_styles``: those values are drawn as given.
    styled: frozenset = frozenset()


@dataclass
class LogTrackSpec:
    display_name: str = ""
    width: float = 1.0
    scale: str = "linear"
    left: float = 0.0
    right: float = 1.0
    cumulative: bool = False
    depth_track: bool = False
    track_type: str = "curves"
    major_lines: int = 5
    tick_spacing: Optional[float] = None
    number_spacing: Optional[float] = None
    line_spacing: Optional[float] = None
    curves: List[LogCurveSpec] = field(default_factory=list)


@dataclass
class LogViewerTemplate:
    name: str = "custom"
    top: Optional[float] = None
    height: Optional[float] = None
    tracks: List[LogTrackSpec] = field(default_factory=list)


@dataclass
class FormationTopSpec:
    name: str
    depth: float


#: Porosity curves of other packages and core tables that are often in percent.
_PERCENT_PRONE_POROSITY = {
    "DPOR", "DPHI", "DPHZ", "PORD", "SPOR", "SPHI", "PORS", "CORE_POR", "CPOR", "CPOR_AMB",
}

#: Generic tracks (three curves each) for a raw well's curves that no named track takes.
_RAW_GENERIC_TRACKS = 4

#: Flag curves; their vendor names are taken only for curves of 0 and 1.
_FLAG_FAMILIES = {"NET_PAY", "NET_RES", "NET_ROCK"}

#: Interpretation intermediates and flags drawn otherwise (shading, overlays);
#: generic tracks leave them out, ``curves=`` can still select them.
_INTERMEDIATE_CURVES = {
    "VCL_GR", "VCLGR", "VCL_SP", "VCLSP", "VCL_ND", "VCLND", "PHI_SEC", "PHI_Z",
    "INVASION_RATIO", "RESFLAG", "PRIME_OIL_TARGET", "PRIME_GAS_TARGET", "LITHO", "LITHO_CLASS",
    *BAD_HOLE_CURVE_NAMES, *CROSSOVER_CURVE_NAMES,
}

#: Depth references added as curves (pp.read(..., survey=...)); never drawn as data.
_DEPTH_CURVES = {"TVD", "TVDSS", "TVDRKB", "TVD_RKB", "TVDMSL", "TVD_MSL", "MD"}

#: Analysis outputs that are bounds or diagnostics of another curve.
_ANALYSIS_DIAGNOSTICS = {"RT_INV_LOW", "RT_INV_HIGH", "INV_MISFIT"}

#: Tracks of computed profiles (the temperature gradient, Rw and Rmf at
#: temperature), drawn only when ``tracks=`` or ``curves=`` asks for them. A
#: logged temperature curve keeps its track.
OPTIONAL_TRACKS = ("TEMPERATURE", "RW / RMF")


def _is_binary_curve(curve) -> bool:
    """True when every finite sample of ``curve`` is 0 or 1."""
    try:
        values = np.asarray(curve.data, dtype=float)
    except (TypeError, ValueError):
        return False
    finite = values[np.isfinite(values)]
    return finite.size > 0 and bool(np.isin(finite, (0.0, 1.0)).all())


def _resolve_curve_name_for_well(
    well: "Well",
    curve_name: str,
    candidates: Optional[Sequence[str]] = None,
) -> Optional[str]:
    from petrophys.preprocessing.standardization.user_aliases import curve_aliases_of

    # A curve the user's mapping names as this curve comes first.
    user = [
        name for name in getattr(well, "curves", [])
        if curve_aliases_of(well).get(str(name).upper()) == str(curve_name).upper()
    ]
    if user:
        return well.get(user[0]).mnemonic
    search_names = [str(curve_name).upper()]
    search_names.extend(str(candidate).upper() for candidate in (candidates or []))
    search_names.extend(CURVE_ALIAS_CANDIDATES.get(str(curve_name).upper(), []))
    seen = set()
    flag = str(curve_name).upper() in _FLAG_FAMILIES
    for candidate in search_names:
        key = str(candidate).upper()
        if key in seen:
            continue
        seen.add(key)
        curve = well.get(key)
        if curve is not None:
            # Another package's flag name (PAY, RES, ...) counts only for a 0/1 curve:
            # a resistivity called RES is not a reservoir flag.
            if flag and key != str(curve_name).upper() and not _is_binary_curve(curve):
                continue
            if str(curve_name).upper() == "CALI" and key == "DCAL":
                from petrophys.preprocessing.standardization.curve_standardizer import _differential_caliper

                if _differential_caliper(key, curve):
                    continue
            return curve.mnemonic
    # Fall back to the library alias map so vendor mnemonics such as GR_EDTC
    # or RHOZ fill their tracks while keeping their own names as labels.
    target = str(curve_name).upper()
    for mnemonic in well.curves:
        if get_standard_mnemonic(mnemonic) == target:
            if target == "CALI":
                from petrophys.preprocessing.standardization.curve_standardizer import _differential_caliper

                if _differential_caliper(mnemonic, well.get(mnemonic)):
                    continue
            if target in _FLAG_FAMILIES and not _is_binary_curve(well.get(mnemonic)):
                continue
            return well.get(mnemonic).mnemonic
    return None


def _adaptive_spec(
    curve_name: str,
    display_name: Optional[str] = None,
    left: Optional[float] = None,
    right: Optional[float] = None,
    color: str = "black",
    line_style: str = "-",
    fill: Optional[str] = None,
    fill_color: Optional[str] = None,
    fill_alpha: float = 0.28,
    marker: Optional[str] = None,
) -> LogCurveSpec:
    return LogCurveSpec(
        curve_name=curve_name,
        display_name=display_name or curve_name,
        left=left,
        right=right,
        color=color,
        line_style=line_style,
        fill=fill,
        fill_color=fill_color,
        fill_alpha=fill_alpha,
        marker=marker,
        marker_size=3.0 if marker else 0.0,
    )


def _adaptive_curve_range(well: "Well", curve_name: str) -> Tuple[float, float]:
    curve = well.get(curve_name)
    if curve is None:
        return 0.0, 1.0
    data = np.asarray(curve.data, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    left, right = np.nanpercentile(finite, [2, 98])
    if not np.isfinite(left) or not np.isfinite(right):
        return 0.0, 1.0
    if math.isclose(float(left), float(right)):
        padding = abs(float(left)) * 0.05 or 1.0
        return float(left - padding), float(right + padding)
    # Round outward to a tenth of the span's order of magnitude (68.3-170.05 -> 60-180).
    step = 10.0 ** (math.floor(math.log10(float(right - left))) - 1)
    return float(math.floor(left / step) * step), float(math.ceil(right / step) * step)


def _sp_range(well: "Well", curve_name: str) -> Tuple[float, float]:
    """SP scale: right edge at the shale baseline, at least 100 mV wide.

    SP readings carry an arbitrary offset (30/6-5 reads 50-110 mV), so a fixed
    -160 to 40 mV scale can miss the whole curve; only deflections matter.
    """
    data = np.asarray(well.get(curve_name).data, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return -160.0, 40.0
    low, high = np.nanpercentile(finite, [2, 98])
    right = 10.0 * math.ceil(high / 10.0)
    span = max(100.0, 10.0 * math.ceil((right - low) / 10.0))
    return right - span, right


def _curve_has_finite_values(well: "Well", curve_name: str) -> bool:
    curve = well.get(curve_name)
    if curve is None:
        return False
    data = np.asarray(curve.data, dtype=float)
    return bool(np.any(np.isfinite(data)))


def _duplicate_of(well: "Well", curve_name: str, others: Iterable[str]) -> bool:
    """True when ``curve_name`` holds the same samples as one of ``others`` (a working-alias copy)."""
    curve = well.get(curve_name)
    try:
        data = np.asarray(curve.data, dtype=float)
    except (AttributeError, TypeError, ValueError):
        return False
    if not np.isfinite(data).any():
        return False
    for name in others:
        other = well.get(name)
        if other is None or other is curve or str(name).upper() == str(curve_name).upper():
            continue
        try:
            values = np.asarray(other.data, dtype=float)
        except (TypeError, ValueError):
            continue
        if values.shape == data.shape and np.array_equal(values, data, equal_nan=True):
            return True
    return False


def _optional_track(well: "Well", track: LogTrackSpec) -> bool:
    """True for an optional track of computed profiles (see :data:`OPTIONAL_TRACKS`)."""
    if track.display_name not in OPTIONAL_TRACKS:
        return False
    if track.display_name == "TEMPERATURE":
        # A logged temperature (MTEM, ...) is shown; the workflow's gradient profile TEMP is optional.
        resolved = _resolve_curve_name_for_well(well, "TEMP")
        return well.get_parameter("temperature_profile_source") is not None and str(resolved).upper() == "TEMP"
    return True


def _lithology_code_curve(well: "Well") -> Optional[str]:
    """The well's lithology code curve, read with the built-in lithology codes; None without one.

    ``LITHOLOGY_GEOLINK``, or another lithology-code curve as
    :func:`petrophys.interpret` reads it.
    """
    from petrophys.interpretation.lithology.matrix_inference import find_lithology_code_curve

    return find_lithology_code_curve(well)


@_styled_plot
def create_adaptive_log_viewer_template(
    well: "Well",
    max_generic_tracks: int = 0,
    generic_curves_per_track: int = 3,
) -> LogViewerTemplate:
    """
    Build a plotting template from the curves actually present on a well.

    The adaptive template starts with common petrophysical curve families and
    then adds generic tracks for unfamiliar mnemonics so arbitrary LAS files
    still produce useful plots.
    """
    code_curve = _lithology_code_curve(well)
    template = LogViewerTemplate(
        name="adaptive",
        tracks=[
            LogTrackSpec(
                display_name="DEPTH",
                width=0.55,
                depth_track=True,
                track_type="depth",
                tick_spacing=50,
                number_spacing=250,
                line_spacing=100,
            )
        ],
    )
    used_resolved: set = set()

    def has(curve_name: str) -> bool:
        resolved = _resolve_curve_name_for_well(well, curve_name)
        return resolved is not None and _curve_has_finite_values(well, resolved)

    # Shading between curve pairs only when both are logged (no missing-curve warning).
    # A well with a bad-hole flag shows it as the band across all tracks; the
    # caliper-bit fill is drawn only without one, so the plot has one signal.
    flagged = any(well.get(name) is not None for name in BAD_HOLE_CURVE_NAMES)
    washout = "BS" if has("CALI") and has("BS") and not flagged else None
    density_neutron = has("RHOB") and has("NPHI")

    def add_track(
        display_name: str,
        specs: Sequence[LogCurveSpec],
        width: float = 1.2,
        scale: str = "linear",
        track_type: str = "curves",
        cumulative: bool = False,
        left: float = 0.0,
        right: float = 1.0,
        major_lines: int = 5,
    ) -> None:
        track_specs: List[LogCurveSpec] = []
        track_resolved = set()
        for spec in specs:
            resolved = _resolve_curve_name_for_well(well, spec.curve_name, spec.candidates)
            if resolved is None or resolved in track_resolved:
                continue
            # A curve an earlier track draws is not drawn again under another
            # family's alias (NPHI as PHIN in the porosity track).
            if resolved in used_resolved and resolved.upper() != spec.curve_name.upper():
                continue
            if not _curve_has_finite_values(well, resolved):
                continue
            track_resolved.add(resolved)
            used_resolved.add(resolved)
            if spec.left is None or spec.right is None:
                spec_left, spec_right = (
                    _sp_range(well, resolved) if spec.curve_name.upper() == "SP"
                    else _adaptive_curve_range(well, resolved)
                )
                spec = LogCurveSpec(**{**spec.__dict__, "left": spec_left, "right": spec_right})
            track_specs.append(spec)
        if track_specs:
            template.tracks.append(
                LogTrackSpec(
                    display_name=display_name,
                    width=width,
                    scale=scale,
                    track_type=track_type,
                    cumulative=cumulative,
                    left=left,
                    right=right,
                    major_lines=major_lines,
                    curves=track_specs,
                )
            )

    template.tracks.append(
        LogTrackSpec(
            display_name="LITHOLOGY",
            width=0.9,
            track_type="lithology",
            curves=[
                _adaptive_spec(
                    code_curve or "LITHOLOGY_GEOLINK",
                    "Lithology",
                    color="#000000",
                    fill_alpha=0.85,
                )
            ],
        )
    )
    add_track(
        "GAMMA / HOLE",
        [
            _adaptive_spec("GR", "GR", 0, 150, "#1b7837", fill="left", fill_color="#d9f0d3"),
            # VCL is a line here; clay volume is shaded in the mineral volumes track.
            _adaptive_spec("VCL", "VCL", 0, 1, "#8c510a"),
            # Washout: caliper larger than bit size.
            replace(_adaptive_spec("CALI", "CALI", 6, 16, "#d73027", fill_color="#f4a582", fill_alpha=0.45),
                    left_crossover=washout),
            _adaptive_spec("BS", "BS", 6, 16, "#000000", line_style="--"),
            # A second caliper arm pair (HD2, C2) beside the first.
            replace(_adaptive_spec("CAL2", "CAL2", 6, 16, "#f46d43", line_style=":"),
                    candidates=["HD2_PPC1", "HD2", "C2", "CALI2", "CAL2", "HCAL2", "HRCAL2"]),
            # Uranium-free gamma ray, beside the total.
            replace(_adaptive_spec("CGR", "CGR", 0, 150, "#4daf4a", line_style="--"),
                    candidates=["CGR", "HCGR", "GRKT", "CGR_EDTC"]),
            # SP is relative: its scale follows the well's shale baseline (see _sp_range).
            _adaptive_spec("SP", "SP", color="#4d4d4d", line_style="-."),
        ],
        width=1.35,
    )
    add_track(
        "RESISTIVITY",
        [
            _adaptive_spec("RDEP", "RDEP", 0.2, 2000, "#a50f15"),
            _adaptive_spec("RMED", "RMED", 0.2, 2000, "#08519c", line_style="--"),
            _adaptive_spec("RSHA", "RSHA", 0.2, 2000, "#238b45", line_style=":"),
            _adaptive_spec("RXO", "RXO", 0.2, 2000, "#41b6c4", line_style="-."),
            # Rt from the step-profile invasion inversion.
            _adaptive_spec("RT_INV", "Rt inv", 0.2, 2000, "#000000", line_style="--"),
        ],
        width=1.35,
        scale="log",
        left=0.2,
        right=2000,
    )
    add_track(
        "DENSITY / NEUTRON",
        [
            # Limestone-compatible scales (NPHI in limestone units, 0 v/v at 2.70 g/cc):
            # density left of neutron is crossover
            # (light hydrocarbon, yellow); neutron left of density is shale separation (grey).
            replace(_adaptive_spec("RHOB", "RHOB", 1.95, 2.95, "#d73027", fill_color="#bdbdbd", fill_alpha=0.35),
                    left_crossover="NPHI" if density_neutron else None),
            replace(_adaptive_spec("NPHI", "NPHI", 0.45, -0.15, "#0571b0", line_style="--",
                                   fill_color="#ffd92f", fill_alpha=0.55),
                    left_crossover="RHOB" if density_neutron else None),
            _adaptive_spec("NPHI_HC", "NPHI HC", 0.45, -0.15, "#2166ac", line_style="--"),
            _adaptive_spec("DRHO", "DRHO", -0.25, 0.25, "#4d4d4d", line_style=":"),
            _adaptive_spec("PEF", "PEF", 0, 10, "#b35806"),
            _adaptive_spec("GDEN", "Grain Den Core", 1.95, 2.95, "#000000", marker="x"),
        ],
        width=1.35,
    )
    add_track(
        "SONIC",
        [
            _adaptive_spec("DTC", "DTC", 140, 40, "#762a83"),
            _adaptive_spec("DTS", "DTS", 240, 40, "#e7298a", line_style="--"),
        ],
        width=1.1,
    )
    add_track(
        "POROSITY",
        [
            _adaptive_spec("PHIT", "PHIT", 0.45, 0, "#253494"),
            _adaptive_spec("PHIT_GAS_CORR", "PHIT Gas", 0.45, 0, "#313695", line_style="--"),
            _adaptive_spec("PHIE", "PHIE", 0.45, 0, "#d73027", fill="right", fill_color="#fee08b"),
            _adaptive_spec("PHIE_CC", "PHIE_CC", 0.45, 0, "#7b3294", line_style="--"),
            _adaptive_spec("PHID", "PHID", 0.45, 0, "#ef8a62"),
            _adaptive_spec("PHIS", "PHIS", 0.45, 0, "#1a9850"),
            _adaptive_spec("PHIN", "PHIN", 0.45, 0, "#0571b0"),
            _adaptive_spec("PHIC", "PHIC", 0.45, 0, "#000000", marker="x"),
            # NMR total porosity, free fluid and bound water (CMR, MRIL).
            replace(_adaptive_spec("NMR_PHIT", "NMR total", 0.45, 0, "#4d4d4d", line_style="-."),
                    candidates=["TCMR", "MPHI", "MPHS", "CMRP_3MS", "NMR_PHIT", "CMRPHI", "NMRP", "PHINMR"]),
            replace(_adaptive_spec("NMR_FFI", "NMR free fluid", 0.45, 0, "#1f78b4", line_style=":"),
                    candidates=["CMFF", "MFFI", "NMR_FFI", "CMR_FFI", "FFI"]),
            replace(_adaptive_spec("NMR_BVI", "NMR bound", 0.45, 0, "#b15928", line_style=":"),
                    candidates=["MBVI", "CBVI", "NMR_BVI", "CMR_BVI", "BVI"]),
            # Bulk volume water on the porosity scale: PHIE to BVW is hydrocarbon (yellow), BVW to 0 water.
            _adaptive_spec("BVW", "BVW", 0.45, 0, "#08519c", fill="right", fill_color="#6baed6", fill_alpha=0.6),
        ],
        width=1.35,
    )
    add_track(
        "SATURATION",
        [
            _adaptive_spec("SW", "Sw", 1, 0, "#0571b0", fill="right", fill_color="#9ecae1"),
            _adaptive_spec("SXO", "Sxo", 1, 0, "#41b6c4", line_style="--"),
            _adaptive_spec("MHI", "MHI", 1, 0, "#d73027"),
            _adaptive_spec("SWIRR_APP", "Swirr", 1, 0, "#762a83"),
            _adaptive_spec("SWC", "Sw Core", 1, 0, "#08519c", marker="x"),
            _adaptive_spec("SW_INPUT", "Sw input", 1, 0, "#4d4d4d", line_style="--"),
            _adaptive_spec("SOC", "So Core", 0, 1, "#b35806", marker="o"),
        ],
        width=1.3,
    )
    add_track(
        "PERMEABILITY",
        [
            _adaptive_spec("PERM", "PERM", 0.01, 10000, "#000000"),
            _adaptive_spec("PERM_CC", "PERM_CC", 0.01, 10000, "#7b3294", line_style="--"),
            _adaptive_spec("PERM_TIMUR", "Timur", 0.01, 10000, "#d73027"),
            _adaptive_spec("PERM_COATES", "Coates", 0.01, 10000, "#1a9850"),
            _adaptive_spec("KH_CORE", "Kh Core", 0.01, 10000, "#08519c", marker="x"),
            replace(_adaptive_spec("KSDR", "NMR SDR", 0.01, 10000, "#e7298a", line_style="-."),
                    candidates=["KSDR", "NMR_KSDR", "KSDR_CMR"]),
            replace(_adaptive_spec("KTIM", "NMR Timur", 0.01, 10000, "#66a61e", line_style=":"),
                    candidates=["KTIM", "NMR_KTIM", "K_COATES", "KTIM_CMR"]),
        ],
        width=1.25,
        scale="log",
        left=0.01,
        right=10000,
    )
    add_track(
        "ROCK QUALITY",
        [
            _adaptive_spec("RQI", "RQI", 0.001, 10, "#7b3294"),
            _adaptive_spec("FZI", "FZI", 0.01, 1000, "#e66101", line_style="--"),
        ],
        width=1.1,
        scale="log",
        left=0.001,
        right=10,
    )
    # Mineral volumes the interpretation computed are drawn whatever the
    # lithology track shows (a vendor lithology log or the computed one).
    add_track(
        "MINERAL VOLUMES",
        [
            _adaptive_spec("VCLAY", "VCLAY", color="#7a6a53", fill_color="#7a6a53", fill_alpha=0.85),
            _adaptive_spec("VQUAR", "VQUAR", color="#f1d27a", fill_color="#f1d27a", fill_alpha=0.85),
            _adaptive_spec("VCALC", "VCALC", color="#b8b8b8", fill_color="#b8b8b8", fill_alpha=0.85),
            _adaptive_spec("VDOLO", "VDOLO", color="#7e8fa6", fill_color="#7e8fa6", fill_alpha=0.85),
            _adaptive_spec("VANHY", "VANHY", color="#92c5de", fill_color="#92c5de", fill_alpha=0.85),
        ],
        width=1.35,
        cumulative=True,
        left=0,
        right=1,
    )
    add_track(
        "SPECTRAL GR",
        [
            replace(_adaptive_spec("POTA", "K", 0, 5, "#e41a1c"), candidates=["POTA", "HFK", "POTASSIUM", "KC", "K40"]),
            replace(_adaptive_spec("THOR", "Th", 0, 25, "#377eb8"), candidates=["THOR", "HTHO", "THORIUM", "THC", "TH232"]),
            replace(_adaptive_spec("URAN", "U", 0, 10, "#4daf4a", line_style="--"),
                    candidates=["URAN", "HURA", "URANIUM", "UC", "U238"]),
        ],
        width=1.15,
    )
    pressures = [name for name in ("OVERBURDEN", "PHYD", "PP_EATON") if _curve_has_finite_values(well, name)]
    if pressures:
        # One scale for all pressures, so overburden, hydrostatic and pore pressure compare.
        high = float(np.ceil(np.nanmax([np.nanmax(np.asarray(well.get(n).data, dtype=float)) for n in pressures]) / 100.0) * 100.0)
        add_track(
            "PRESSURE",
            [
                _adaptive_spec("OVERBURDEN", "Overburden", 0, high, "#000000"),
                _adaptive_spec("PHYD", "Hydrostatic", 0, high, "#2166ac", line_style="--"),
                _adaptive_spec("PP_EATON", "Pore pressure", 0, high, "#d73027"),
            ],
            width=1.2,
        )
    add_track(
        "TEMPERATURE",
        [_adaptive_spec("TEMP", "TEMP", color="#d6604d")],
        width=0.9,
    )
    add_track(
        "RW / RMF",
        [
            _adaptive_spec("RW", "Rw", 0.01, 10, "#2166ac"),
            _adaptive_spec("RMF", "Rmf", 0.01, 10, "#b2182b", line_style="--"),
        ],
        width=0.9,
        scale="log",
        left=0.01,
        right=10,
    )
    add_track(
        "FLAGS",
        [
            _adaptive_spec("NET_ROCK", "Clean Rock", color="#fdae61"),
            _adaptive_spec("NET_RES", "Res", color="#fee08b"),
            _adaptive_spec("NET_PAY", "Pay", color="#1a9850"),
            _adaptive_spec("SEALFLAG", "Seal", color="#74add1"),
        ],
        width=0.95,
        track_type="flags",
    )

    generic_names = []
    for curve_name in getattr(well, "curves", []):
        upper = curve_name.upper()
        if (
            upper in {"DEPT", "DEPTH", "MD"}
            or upper in used_resolved
            or upper in AUTO_TEMPLATE_SKIP_CURVES
            or upper in _INTERMEDIATE_CURVES
            or upper in _DEPTH_CURVES
            or upper in _ANALYSIS_DIAGNOSTICS
            or upper == str(code_curve or "").upper()
            or _duplicate_of(well, curve_name, used_resolved | set(generic_names))
        ):
            continue
        generic_names.append(upper)

    if max_generic_tracks > 0:
        generic_limit = max_generic_tracks * max(1, generic_curves_per_track)
        generic_names = generic_names[:generic_limit]
        colors = ["#000000", "#1b7837", "#d73027", "#0571b0", "#762a83", "#8c510a"]
        for idx in range(0, len(generic_names), max(1, generic_curves_per_track)):
            names = generic_names[idx: idx + max(1, generic_curves_per_track)]
            specs = [
                _adaptive_spec(name, name, color=colors[(idx + offset) % len(colors)])
                for offset, name in enumerate(names)
            ]
            add_track(f"OTHER {idx // max(1, generic_curves_per_track) + 1}", specs, width=1.15)

    return template


def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_float(value: Optional[str], default: Optional[float] = None) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _float_or_default(value: Optional[str], default: float) -> float:
    """Parse a template number; an explicit 0 is kept, NaN/inf use ``default``."""
    parsed = _as_float(value)
    if parsed is None or not math.isfinite(parsed):
        return default
    return parsed


def _as_int(value: Optional[str], default: int) -> int:
    parsed = _as_float(value)
    if parsed is None or not math.isfinite(parsed):
        return default
    return int(parsed)


def _first_attr(
    element: ET.Element,
    names: Sequence[str],
    default: Optional[str] = None,
) -> Optional[str]:
    for name in names:
        if name in element.attrib:
            return element.attrib[name]
    return default


def _split_candidates(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [part.strip().upper() for part in value.replace("|", ",").split(",") if part.strip()]


def _parse_curve(element: ET.Element) -> LogCurveSpec:
    name = _first_attr(element, ("curve_name", "mnemonic", "name"))
    if not name:
        raise ValueError("Template curve is missing curve_name/mnemonic/name")

    explicit_candidates = _split_candidates(
        _first_attr(element, ("candidates", "aliases", "alias"))
    )
    return LogCurveSpec(
        curve_name=str(name).upper(),
        display_name=_first_attr(element, ("display_name", "label")),
        left=_as_float(_first_attr(element, ("left", "min"))),
        right=_as_float(_first_attr(element, ("right", "max"))),
        color=_first_attr(element, ("color", "line_color"), "black") or "black",
        line_width=_float_or_default(_first_attr(element, ("line_width", "width", "linewidth")), 0.8),
        line_style=_first_attr(element, ("line_style", "linestyle"), "-") or "-",
        alpha=_float_or_default(element.attrib.get("alpha"), 1.0),
        fill=_first_attr(element, ("fill", "fill_direction")),
        fill_color=_first_attr(element, ("fill_color", "fillcolor")),
        fill_color_map=_first_attr(element, ("fill_color_map", "fill_cmap", "cmap")),
        fill_alpha=_float_or_default(_first_attr(element, ("fill_alpha", "fillalpha")), 0.28),
        left_color_value=_as_float(element.attrib.get("left_color_value")),
        right_color_value=_as_float(element.attrib.get("right_color_value")),
        left_cutoff_fill=_as_float(element.attrib.get("left_cutoff_fill")),
        right_cutoff_fill=_as_float(element.attrib.get("right_cutoff_fill")),
        left_crossover=_first_attr(element, ("left_crossover", "left_crossover_curve")),
        right_crossover=_first_attr(element, ("right_crossover", "right_crossover_curve")),
        marker=_first_attr(element, ("marker", "symbol")),
        marker_size=_float_or_default(_first_attr(element, ("marker_size", "markersize")), 0.0),
        candidates=explicit_candidates,
    )


def _infer_track_type(element: ET.Element, display_name: str, curves: List[LogCurveSpec]) -> str:
    explicit = _first_attr(element, ("track_type", "type"))
    if explicit:
        return explicit.lower()
    title = display_name.upper()
    if _as_bool(element.attrib.get("depth_track")) or (title == "DEPTH" and not curves):
        return "depth"
    if _as_bool(element.attrib.get("cumulative")):
        return "cumulative"
    if "LITHO" in title or any("LITHO" in curve.curve_name for curve in curves):
        return "lithology"
    if "FLAG" in title or title in {"FLAGS", "NET"}:
        return "flags"
    return "curves"


def _parse_track(element: ET.Element) -> LogTrackSpec:
    curves = [_parse_curve(child) for child in element if child.tag.lower() == "curve"]
    display_name = _first_attr(element, ("display_name", "name", "label"), "") or ""
    track_type = _infer_track_type(element, display_name, curves)

    return LogTrackSpec(
        display_name=display_name,
        width=_as_float(element.attrib.get("width"), 1.0) or 1.0,
        scale=(_first_attr(element, ("scale", "xscale"), "linear") or "linear").lower(),
        left=_float_or_default(_first_attr(element, ("left", "min")), 0.0),
        right=_float_or_default(_first_attr(element, ("right", "max")), 1.0),
        cumulative=track_type == "cumulative" or _as_bool(element.attrib.get("cumulative")),
        depth_track=track_type == "depth",
        track_type=track_type,
        major_lines=max(0, _as_int(element.attrib.get("major_lines"), 5)),
        tick_spacing=_as_float(element.attrib.get("tick_spacing")),
        number_spacing=_as_float(element.attrib.get("number_spacing")),
        line_spacing=_as_float(element.attrib.get("line_spacing")),
        curves=curves,
    )


def load_log_viewer_template(
    template_xml_path: Optional[str | Path] = None,
    template_defaults: str = "raw",
) -> LogViewerTemplate:
    """
    Load a log-viewer template from XML or from a built-in default.

    XML can use either a ``<template>`` root or PetroPy-style ``<graph>`` root.
    Built-in names include ``raw``, ``petrophysics``, ``minerals``,
    ``comprehensive``, and ``geolink_full``.
    """
    if template_xml_path is not None:
        root = ET.parse(template_xml_path).getroot()
        default_name = Path(template_xml_path).stem
    else:
        key = str(template_defaults or "raw").lower()
        if key not in DEFAULT_LOG_VIEWER_TEMPLATES:
            valid = ", ".join(sorted(DEFAULT_LOG_VIEWER_TEMPLATES))
            raise ValueError(f"Unknown log viewer template '{template_defaults}'. Valid templates: {valid}")
        root = ET.fromstring(DEFAULT_LOG_VIEWER_TEMPLATES[key])
        default_name = key

    if root.tag.lower() not in {"template", "graph"}:
        raise ValueError("Log viewer XML root must be <template> or <graph>")

    template = LogViewerTemplate(
        name=root.attrib.get("name", default_name),
        top=_as_float(root.attrib.get("top")),
        height=_as_float(root.attrib.get("height")),
    )
    for child in root:
        if child.tag.lower() == "track":
            template.tracks.append(_parse_track(child))

    if not template.tracks:
        raise ValueError("Log viewer template must contain at least one track")
    return template


def _display_well(well: "Well", interpreted: bool = False) -> "Well":
    """Return a copy of ``well`` with curves converted to standard units.

    Plots must not depend on whether the well has been interpreted yet:
    units such as kg/m3 density or percent porosity are converted exactly as
    the interpretation workflow does, so every curve is drawn at the right
    scale. Curve names are kept; vendor mnemonics (``GR_EDTC``, ``RHOZ``) are
    matched to tracks through the alias map when the tracks are resolved.
    """
    from copy import deepcopy

    from petrophys.preprocessing.standardization import standardize_curves

    from petrophys.model.units import PERCENT_SATURATION_UNITS
    from petrophys.preprocessing.standardization.curve_standardizer import (
        CURVE_PLAUSIBLE_RANGES,
    )

    display = deepcopy(well)
    # Depth curves stay in the well's own unit: tops, intervals and
    # depth_range are given in that unit and are not converted.
    depth_curves = {
        name: display._curves[name].copy()
        for name in CURVE_ALIAS_CANDIDATES["DEPT"]
        if name in display._curves
    }
    # Percent-labelled fraction curves (SW, SXO, BVW, PHID ...) are shown in
    # v/v; standardization converts only some of them and would otherwise
    # remove the rest as implausible or keep a '%' label on rescaled data.
    for mnemonic, curve in display._curves.items():
        limits = CURVE_PLAUSIBLE_RANGES.get(str(get_standard_mnemonic(mnemonic)).upper())
        unit = str(curve.unit or "").strip().lower()
        if limits is not None and limits[1] <= 1.0 and unit in PERCENT_SATURATION_UNITS:
            curve._data = np.asarray(curve.data, dtype=float) / 100.0
            curve.unit = "v/v"
    # An interpretation's curves are drawn with the values it used: units are
    # still converted for the tracks, but no value is screened out again.
    # Vendor and core porosities (DPOR, SPOR, CORE_POR, ...) are often in
    # percent under a blank label; they are drawn in v/v as the interpretation
    # reads them.
    for mnemonic, curve in display._curves.items():
        if mnemonic.upper() not in _PERCENT_PRONE_POROSITY:
            continue
        values = np.asarray(curve.data, dtype=float)
        finite = values[np.isfinite(values) & (values > -900.0)]
        unit = str(curve.unit or "").strip().lower()
        percent = unit in PERCENT_SATURATION_UNITS or (
            unit not in {"v/v", "frac", "fraction", "dec", "decimal"} and finite.size and float(np.median(finite)) > 1.0
        )
        if percent:
            curve._data = values / 100.0
            curve.unit = "v/v"
    standardize_curves(
        display, rename_curves=False, convert_units=True,
        plausible_ranges={} if interpreted else None,
    )
    display._curves.update(depth_curves)
    # NMR porosities standardize to porosity units (pu); the porosity track is in v/v.
    for mnemonic, curve in display._curves.items():
        if str(get_standard_mnemonic(mnemonic)).upper() in {"NMR_PHIT", "NMR_FFI", "NMR_BVI", "NMR_3MS"}                 and str(curve.unit or "").strip().lower() in PERCENT_SATURATION_UNITS:
            curve._data = np.asarray(curve.data, dtype=float) / 100.0
            curve.unit = "v/v"
    return display


class LogViewer:
    """
    PetroPy-style template-driven well log plotter.

    Parameters
    ----------
    well : Well
        Local ``Well`` object containing a depth array and curves.
    template_xml_path : str or Path, optional
        XML template file. When omitted, ``template_defaults`` is used.
    template_defaults : str
        Template mode/name. ``auto`` builds tracks from available curves.
        Built-in XML names include ``raw``, ``petrophysics``, ``minerals``,
        ``comprehensive``, and ``geolink_full``.
    strict : bool
        If True, missing curves listed by the template raise ``ValueError``.
        If False, they are skipped and recorded in ``warnings``.
    interpreted : bool
        True for an interpretation's well: units are converted for display but
        values are not screened again, so the plot shows what the
        interpretation used.
    lithology_mapping : mapping, optional
        Mapping from lithology code to label for lithology tracks.
    lithology_mapping_df : DataFrame, optional
        PetroPy-script-compatible mapping table. Columns named
        ``Abbreviation`` and ``Attribute`` are recognized. If both label
        columns exist, ``Attribute`` takes precedence over ``Lithology Attribute``.
        When neither mapping is given, codes use the built-in lithology labels,
        or the CSV/Excel table named by the ``PETROPHYS_LITHOLOGY_MAPPING``
        environment variable when it is set.
    formation_tops : mapping, sequence, DataFrame, or path, optional
        Formation tops to annotate. Accepts mappings of name to depth,
        ``(name, depth)`` pairs, CSV/Excel paths, or tabular objects.
    formation_name : str, optional
        Formation or interval name to show in the top title beside the well name.
    lithology_intervals : path or DataFrame, optional
        Single-well CSV/XLSX/table with ``top,bottom,lithology`` columns, in the
        same depth units and datum as the well. Overrides embedded lithology
        for display only, preserving interval boundaries and unfilled gaps.
    """

    def __init__(
        self,
        well: "Well",
        template_xml_path: Optional[str | Path] = None,
        template_defaults: str = "auto",
        strict: bool = False,
        figsize: Optional[Tuple[float, float]] = None,
        lithology_mapping: Optional[Mapping[Any, str]] = None,
        lithology_mapping_df: Optional[Any] = None,
        formation_tops: Optional[Any] = None,
        formation_tops_df: Optional[Any] = None,
        formation_name: Optional[str] = None,
        lithology_intervals: Optional[Any] = None,
        tracks: Optional[Sequence[str]] = None,
        interpreted: bool = False,
        curves: Optional[Sequence[str]] = None,
        curve_styles: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization")
        #: Plotted copy of the input well in standard units, with the curve
        #: names kept (GR_EDTC stays GR_EDTC and fills the GR track through the
        #: alias tables); the caller's well is unchanged.
        #: With ``interpreted`` (an interpretation's well) values are not
        #: screened again, so the plot shows what the interpretation used.
        self.well = _display_well(well, interpreted=interpreted)
        # The lithology code curve, labelled with the built-in lithology codes.
        self._code_curve = _lithology_code_curve(self.well)
        self.formation_import_report = {}
        self.suppressed_formation_labels = []
        self.omitted_tracks = []
        context = self.well.get_parameter("_geology_context") or {}
        if lithology_intervals is None:
            if context.get("selected_lithology_source") in {None, "external"}:
                lithology_intervals = context.get("lithology")
        if formation_tops is None and formation_tops_df is None:
            formation_tops = context.get("formations")
        self.lithology_source = (
            "external" if lithology_intervals is not None
            else context.get("selected_lithology_source", "embedded_or_computed")
        )
        self.lithology_intervals = (
            read_lithology_intervals(lithology_intervals, well=self.well)
            if lithology_intervals is not None else None
        )
        self._adaptive_template = False
        template_key = str(template_defaults or "auto").lower()
        if template_xml_path is None and template_key in ADAPTIVE_LOG_VIEWER_TEMPLATE_NAMES:
            # A raw well shows every logged curve: those no named track takes
            # go to generic tracks. An interpretation's plot keeps its tracks.
            self.template = create_adaptive_log_viewer_template(
                self.well, max_generic_tracks=0 if interpreted else _RAW_GENERIC_TRACKS
            )
            self._adaptive_template = True
        else:
            self.template = load_log_viewer_template(template_xml_path, template_defaults)
        if self.lithology_intervals is not None and not any(
            track.track_type == "lithology" for track in self.template.tracks
        ):
            self.template.tracks.insert(
                0, LogTrackSpec(display_name="LITHOLOGY", width=0.9, track_type="lithology")
            )
        if tracks is not None and curves is not None:
            raise ValueError(
                "Pass tracks= or curves=, not both: tracks= keeps whole tracks, curves= keeps single curves "
                "(each in its track). To add a track's curves, list them in curves=."
            )
        if tracks is not None:
            requested = [str(name).strip().upper() for name in tracks]
            available = {track.display_name.upper(): track for track in self.template.tracks}
            unknown = set(requested) - set(available)
            if unknown or not requested or len(requested) != len(set(requested)):
                raise ValueError(
                    f"Select unique track names from {list(available)}; requested {requested}."
                )
            self.template.tracks = [available[name] for name in requested]
        if curves is not None:
            self._select_curves(curves)
        #: Optional tracks left out of this plot; name one in ``tracks=`` to draw it.
        self.optional_tracks: List[str] = []
        self._optional_track_specs: List[LogTrackSpec] = []
        if self._adaptive_template and tracks is None and curves is None:
            self._optional_track_specs = [track for track in self.template.tracks if _optional_track(self.well, track)]
            self.optional_tracks = [track.display_name for track in self._optional_track_specs]
            self.template.tracks = [track for track in self.template.tracks if track not in self._optional_track_specs]
        self.strict = strict
        self.figsize = figsize
        self._style_warnings = self._apply_curve_styles(curve_styles) if curve_styles else []
        self.lithology_mapping = self._coerce_lithology_mapping(
            lithology_mapping,
            lithology_mapping_df,
        )
        self.formation_tops = self._coerce_formation_tops(
            formation_tops_df if formation_tops_df is not None else formation_tops
        )
        self.formation_name = self._clean_title_label(formation_name)
        self.fig: Optional[Figure] = None
        self.axes: List[Axes] = []
        self.header_axes: List[Axes] = []
        import_messages = list(self.formation_import_report.get("diagnostics", []))
        if self.lithology_intervals is not None:
            import_messages.extend(self.lithology_intervals.attrs.get("import_report", {}).get("diagnostics", []))
        # Formation and lithology imports can report the same depth-reference note.
        self.warnings: List[str] = list(dict.fromkeys(import_messages)) + self._style_warnings
        self.curve_lookup: Dict[str, str] = {}
        for name in self.well.curves:
            report = self.well.get(name).metadata.custom.get("range_screening") or {}
            if report and report.get("masked") == report.get("finite") and report.get("finite"):
                self.warnings.append(
                    f"{name}: all {report['finite']} values lie outside the plausible "
                    f"{report['standard_curve']} range {report['low']:g}-{report['high']:g}"
                    f"{' ' + str(report['unit']) if report.get('unit') else ''}; not drawn."
                    + (f" {report['hint'].capitalize()}." if report.get("hint") else "")
                )
        if not interpreted and self._adaptive_template and curves is None and tracks is None:
            undrawn = self._curves_without_track()
            if undrawn:
                self.warnings.append(
                    f"Curves not drawn ({len(undrawn)}; the generic tracks hold "
                    f"{_RAW_GENERIC_TRACKS * 3}; select tracks or a template to show them): "
                    + ", ".join(undrawn)
                )
        self._header_axis_by_data_axis: Dict[int, Axes] = {}
        self._header_row_counts: Dict[int, int] = {}
        self._bad_hole_legend = None

    def _select_curves(self, curves: Sequence[str]) -> None:
        """Keep only ``curves`` in the template, each in its track; others go to generic tracks.

        A name is a curve of the well (``GRD``) or a standard name (``GR``),
        found as the plot finds it: through the user's curve-name mapping and
        the alias tables.
        """
        requested = list(dict.fromkeys(str(name).strip().upper() for name in curves))
        if not requested:
            raise ValueError("curves= is empty: name the curves to draw, e.g. curves=['GR', 'RDEP'].")
        available = {name.upper(): name for name in self.well.curves}
        targets: Dict[str, str] = {}
        unknown = []
        for name in requested:
            if name in available:
                targets[name] = name
                continue
            resolved = _resolve_curve_name_for_well(self.well, name)
            if resolved is None:
                unknown.append(name)
            else:
                targets[name] = resolved.upper()
        if unknown:
            raise ValueError(
                f"No curve {unknown} on the well, by its own name or as a standard name (GR, RDEP, NPHI, ...); "
                f"it has {list(self.well.curves)}. A name the alias tables do not know can be mapped with "
                "curve_aliases={'MYNAME': 'GR'}."
            )
        target_set = set(targets.values())
        satisfied: set = set()
        placed: set = set()
        kept: List[LogTrackSpec] = []
        code = str(self._code_curve or "").upper()
        for track in self.template.tracks:
            if track.depth_track:
                kept.append(track)
                continue
            if track.track_type == "lithology":
                if self.lithology_intervals is not None or code in target_set:
                    kept.append(track)
                    satisfied.update(n for n, t in targets.items() if t == code)
                continue
            specs = []
            for spec in track.curves:
                resolved = _resolve_curve_name_for_well(self.well, spec.curve_name, spec.candidates)
                if resolved is None:
                    continue
                key = resolved.upper()
                by_family = spec.curve_name.upper() in targets
                if (key in target_set or by_family) and key not in placed:
                    placed.add(key)
                    specs.append(spec)
                    satisfied.update(n for n, t in targets.items() if t == key)
                    if by_family:
                        satisfied.add(spec.curve_name.upper())
                        placed.add(targets[spec.curve_name.upper()])
            if specs:
                # A crossover fill needs its partner curve in the same track.
                names = {spec.curve_name.upper() for spec in specs}
                track.curves = [
                    replace(spec, left_crossover=spec.left_crossover if str(spec.left_crossover or "").upper() in names else None,
                            right_crossover=spec.right_crossover if str(spec.right_crossover or "").upper() in names else None)
                    for spec in specs
                ]
                kept.append(track)
        leftover = list(dict.fromkeys(
            available[targets[name]] for name in requested
            if name not in satisfied and targets[name] not in placed and targets[name] != code
        ))
        colors = ["#000000", "#1b7837", "#d73027", "#0571b0", "#762a83", "#8c510a"]
        for index in range(0, len(leftover), 3):
            specs = []
            for offset, name in enumerate(leftover[index:index + 3]):
                left, right = _adaptive_curve_range(self.well, name)
                specs.append(LogCurveSpec(**{**_adaptive_spec(name, name, color=colors[(index + offset) % len(colors)]).__dict__,
                                             "left": left, "right": right}))
            kept.append(LogTrackSpec(display_name="OTHER", width=1.15, curves=specs))
        generic = 0
        for track in kept:
            if track.display_name.startswith("OTHER"):
                generic += 1
                track.display_name = f"OTHER {generic}"
        self.template.tracks = kept

    def _apply_curve_styles(self, styles: Mapping[str, Mapping[str, Any]]) -> List[str]:
        """Apply per-curve style changes; returns a warning for names no track draws."""
        from matplotlib.colors import is_color_like

        allowed = {item.name for item in fields(LogCurveSpec)} - {"curve_name", "candidates", "styled"}
        wanted: Dict[str, Dict[str, Any]] = {}
        for name, style in styles.items():
            unknown = set(style) - allowed
            if unknown:
                hint = " Scale (linear/log) belongs to the track; use a template for it." if "scale" in unknown else ""
                raise ValueError(f"curve_styles[{name!r}]: unknown keys {sorted(unknown)}; use {sorted(allowed)}.{hint}")
            if style.get("fill") not in (None, "left", "right"):
                raise ValueError(f"curve_styles[{name!r}]: fill must be 'left', 'right' or None, not {style['fill']!r}")
            for key in ("color", "fill_color"):
                if style.get(key) is not None and not is_color_like(style[key]):
                    raise ValueError(f"curve_styles[{name!r}]: {key} {style[key]!r} is not a colour (a name, '#rrggbb' or RGB)")
            wanted[str(name).strip().upper()] = dict(style)
        matched = set()
        for track in self.template.tracks:
            for index, spec in enumerate(track.curves):
                resolved = _resolve_curve_name_for_well(self.well, spec.curve_name, spec.candidates)
                for key in (spec.curve_name.upper(), str(resolved or "").upper()):
                    if key in wanted:
                        track.curves[index] = replace(
                            spec, **wanted[key], styled=spec.styled | frozenset(wanted[key])
                        )
                        matched.add(key)
                        break
        missing = sorted(set(wanted) - matched)
        return [f"curve_styles: no track of this plot draws {', '.join(missing)}"] if missing else []

    def _curves_without_track(self) -> List[str]:
        """The well's curves with data that no track of the template draws."""
        drawn = set()
        for track in [*self.template.tracks, *self._optional_track_specs]:
            for spec in track.curves:
                resolved = _resolve_curve_name_for_well(self.well, spec.curve_name, spec.candidates)
                if resolved is not None:
                    drawn.add(resolved.upper())
        if self._code_curve:
            drawn.add(str(self._code_curve).upper())
        skipped = {"DEPT", "DEPTH", "MD", "TDEP", *AUTO_TEMPLATE_SKIP_CURVES, *_INTERMEDIATE_CURVES,
                   *_DEPTH_CURVES, *_ANALYSIS_DIAGNOSTICS}
        return [
            name for name in self.well.curves
            if name.upper() not in drawn and name.upper() not in skipped
            and _curve_has_finite_values(self.well, name)
            and not _duplicate_of(self.well, name, drawn)
        ]

    @_styled_plot
    def create(
        self,
        depth_range: Optional[Tuple[float, float]] = None,
        top: Optional[float] = None,
        height: Optional[float] = None,
    ) -> Tuple[Figure, List[Axes]]:
        """Create and return ``(figure, axes)``."""
        depth = self._depth()
        plot_top, plot_bottom = self._resolve_depth_window(depth, depth_range, top, height)
        if not plot_bottom > plot_top:
            raise ValueError(f"The depth window {plot_top:g}-{plot_bottom:g} has no height")
        logged_top, logged_bottom = finite_min_max(depth)
        if logged_top is not None and (plot_bottom < logged_top or plot_top > logged_bottom):
            message = (f"The depth window {plot_top:g}-{plot_bottom:g} {self.well.depth_unit} is outside the "
                       f"logged interval {logged_top:g}-{logged_bottom:g} {self.well.depth_unit}: nothing to draw")
            if message not in self.warnings:
                self.warnings.append(message)
            warnings.warn(message, UserWarning, stacklevel=3)
        tracks = self.template.tracks
        if self._adaptive_template:
            tracks = self._filter_visible_tracks(tracks, depth, plot_top, plot_bottom)
        tracks = self._tracks_for_plot(tracks)

        width_ratios = [self._track_width_ratio(track) for track in tracks]
        max_label_rows = max((self._track_header_row_count(track) for track in tracks), default=1)
        figsize = self.figsize or (max(sum(width_ratios) * 1.75, 11.0), 14.5)
        # Header labels need a fixed height in inches (about 0.3 in per curve);
        # on short figures a share of the height alone would overlap them.
        header_inches = 0.55 + max_label_rows * 0.36
        needed_ratio = header_inches / max(float(figsize[1]) - header_inches, 1.0)
        header_ratio = max(min(0.30, max(0.14, 0.075 + max_label_rows * 0.032)), min(needed_ratio, 0.8))

        # Close a figure from a previous create() on this viewer so repeated
        # builds don't leak figures.
        if self.fig is not None:
            plt.close(self.fig)
        self.fig = plt.figure(figsize=figsize, constrained_layout=False)
        grid = self.fig.add_gridspec(
            2,
            len(tracks),
            height_ratios=[header_ratio, 1.0],
            width_ratios=width_ratios,
            hspace=0.015,
            wspace=0.02,
        )
        self.axes = []
        self.header_axes = []
        self._header_axis_by_data_axis = {}
        self._header_row_counts = {}

        shared_axis: Optional[Axes] = None
        for index, track in enumerate(tracks):
            header_ax = self.fig.add_subplot(grid[0, index])
            data_ax = self.fig.add_subplot(grid[1, index], sharey=shared_axis)
            if shared_axis is None:
                shared_axis = data_ax
            self.header_axes.append(header_ax)
            self.axes.append(data_ax)
            self._header_axis_by_data_axis[id(data_ax)] = header_ax
            self._header_row_counts[id(data_ax)] = self._track_header_row_count(track)
            self._format_header_axis(header_ax, track)

        for ax, track in zip(self.axes, tracks):
            self._format_track_axis(ax, track, plot_top, plot_bottom)
            if track.depth_track:
                self._plot_depth_track(ax, track, plot_top, plot_bottom)
            elif track.cumulative:
                self._plot_cumulative_track(ax, track, depth)
            elif track.track_type == "lithology":
                self._plot_lithology_track(ax, track, depth, plot_top, plot_bottom)
            elif track.track_type == "flags":
                self._plot_flag_track(ax, track, depth)
            else:
                self._plot_curve_track(ax, track, depth)

        self._draw_bad_hole_shading(depth)
        self._draw_formation_tops(plot_top, plot_bottom)

        title = self._plot_title()
        self.fig.suptitle(title, fontsize=13, fontweight="bold")
        self.fig.subplots_adjust(
            left=0.035,
            right=0.995,
            bottom=0.045,
            top=0.94,
            wspace=0.02,
            hspace=0.015,
        )
        return self.fig, self.axes

    @_styled_plot
    def show(self) -> None:
        """Display the figure."""
        if self.fig is None:
            self.create()
        plt.show()

    def close(self) -> None:
        """Close the underlying matplotlib figure and release it (frees memory)."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

    @_styled_plot
    def save(self, filename: str | Path, dpi: int = 150) -> None:
        """Save the plot to disk."""
        if self.fig is None:
            self.create()
        assert self.fig is not None
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(self.fig, output_path, dpi=dpi)

    def _well_title(self) -> str:
        header = getattr(self.well, "header", None)
        for value in (
            getattr(self.well, "name", None),
            getattr(header, "well_name", None),
            getattr(header, "uwi", None),
            getattr(header, "api", None),
        ):
            if value:
                return str(value)
        return "Unknown Well"

    def _plot_title(self) -> str:
        well_title = self._well_title()
        if self.formation_name:
            return f"{well_title} | {self.formation_name}"
        return well_title

    def _clean_title_label(self, value: Optional[Any]) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return ""
        text = re.sub(r"\s+top\.?$", "", text, flags=re.IGNORECASE).strip()
        text = text.replace("_", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _depth(self) -> np.ndarray:
        if getattr(self.well, "depth", None) is not None:
            return np.asarray(self.well.depth, dtype=float).reshape(-1)
        for name in CURVE_ALIAS_CANDIDATES["DEPT"]:
            curve = self.well.get(name)
            if curve is not None:
                return np.asarray(curve.data, dtype=float).reshape(-1)
        raise ValueError("Well has no depth array or DEPT/DEPTH/MD curve")

    def _resolve_depth_window(
        self,
        depth: np.ndarray,
        depth_range: Optional[Tuple[float, float]],
        top: Optional[float],
        height: Optional[float],
    ) -> Tuple[float, float]:
        if depth_range is not None:
            if top is not None or height is not None:
                raise ValueError("Pass either depth_range or top/height, not both")
            # (top, bottom) in either order; depth always increases downwards.
            first, second = float(depth_range[0]), float(depth_range[1])
            return min(first, second), max(first, second)
        top_value = top if top is not None else self.template.top
        height_value = height if height is not None else self.template.height
        inferred_top, inferred_bottom = finite_min_max(depth)
        if top_value is None:
            top_value = inferred_top
        if height_value is not None:
            if top_value is None:
                raise ValueError("Cannot resolve plot window without finite depth values")
            return float(top_value), float(top_value + height_value)
        if inferred_bottom is None or top_value is None:
            raise ValueError("Cannot resolve plot window without finite depth values")
        return float(top_value), float(inferred_bottom)

    def _filter_visible_tracks(
        self,
        tracks: Sequence[LogTrackSpec],
        depth: np.ndarray,
        top: float,
        bottom: float,
    ) -> List[LogTrackSpec]:
        visible = np.isfinite(depth) & (depth >= min(top, bottom)) & (depth <= max(top, bottom))
        filtered = []
        self.omitted_tracks = []
        for track in tracks:
            if (
                track.depth_track
                or track.track_type == "lithology"
                or self._track_has_visible_data(track, visible)
            ):
                filtered.append(track)
            else:
                self.omitted_tracks.append({"track": track.display_name, "reason": "no finite data in depth window"})
        if len(filtered) == 0:
            return list(tracks[:1])
        return filtered

    def _tracks_for_plot(self, tracks: Sequence[LogTrackSpec]) -> List[LogTrackSpec]:
        plot_tracks = [track for track in tracks if not track.depth_track]
        return plot_tracks or list(tracks)

    def _track_header_row_count(self, track: LogTrackSpec) -> int:
        if self._is_mineral_volume_track(track):
            return max(5, min(8, len(track.curves) + 2))
        return max(1, len(track.curves))

    def _is_mineral_volume_track(self, track: LogTrackSpec) -> bool:
        title = str(track.display_name).strip().upper()
        if title in MINERAL_VOLUME_TRACK_NAMES:
            return True
        if "MINERAL" in title and "VOLUME" in title:
            return True
        if not track.cumulative or len(track.curves) < 2:
            return False
        curve_names = {curve.curve_name.upper() for curve in track.curves}
        return len(curve_names & MINERAL_VOLUME_CURVE_NAMES) >= 2

    def _track_width_ratio(self, track: LogTrackSpec) -> float:
        width = max(track.width, 0.1)
        if track.track_type == "lithology":
            return min(max(width, 0.45), 0.62)
        return width

    def _track_has_visible_data(self, track: LogTrackSpec, visible: np.ndarray) -> bool:
        for spec in track.curves:
            curve = self._get_curve(spec)
            if curve is None:
                continue
            data = np.asarray(curve.data, dtype=float)
            if np.any(np.isfinite(data) & visible):
                return True
        return False

    def _format_header_axis(self, ax: Axes, track: LogTrackSpec) -> None:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.axis("off")
        ax.text(
            0.5,
            0.96,
            track.display_name,
            fontsize=10.2,
            fontweight="bold",
            ha="center",
            va="top",
            transform=ax.transAxes,
        )

    def _format_track_axis(
        self,
        ax: Axes,
        track: LogTrackSpec,
        top: float,
        bottom: float,
    ) -> None:
        ax.set_ylim(bottom, top)
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([])
        ax.grid(axis="y", color="0.88", linewidth=0.5)
        if ax is not self.axes[0]:
            ax.tick_params(axis="y", labelleft=False)
        else:
            unit = getattr(self.well, "depth_unit", "")
            ax.set_ylabel(f"Depth [{unit}]" if unit else "Depth")

        self._draw_track_grid(ax, track)

    def _draw_track_grid(self, ax: Axes, track: LogTrackSpec) -> None:
        if track.track_type in {"lithology", "flags"}:
            return
        if str(track.scale).lower() == "log" and track.left > 0 and track.right > 0:
            lo, hi = sorted((track.left, track.right))
            first_decade = math.floor(math.log10(lo))
            last_decade = math.ceil(math.log10(hi))
            for exponent in range(first_decade, last_decade + 1):
                for multiplier in (1, 2, 5):
                    value = multiplier * (10**exponent)
                    if lo <= value <= hi:
                        x = self._normalize_scalar(value, track.left, track.right, track.scale)
                        color = "0.78" if multiplier == 1 else "0.90"
                        width = 0.55 if multiplier == 1 else 0.35
                        ax.axvline(x, color=color, linewidth=width, zorder=0)
            return
        if track.major_lines:
            for x in np.linspace(0.0, 1.0, track.major_lines + 1):
                ax.axvline(x, color="0.90", linewidth=0.45, zorder=0)

    def _plot_depth_track(
        self,
        ax: Axes,
        track: LogTrackSpec,
        top: float,
        bottom: float,
    ) -> None:
        tick_spacing = track.tick_spacing or self._nice_depth_spacing(top, bottom, target=16)
        number_spacing = track.number_spacing or tick_spacing * 5
        line_spacing = track.line_spacing or number_spacing

        for tick in self._depth_ticks(top, bottom, tick_spacing):
            ax.axhline(tick, color="0.82", linewidth=0.45, zorder=0)

        for tick in self._depth_ticks(top, bottom, line_spacing):
            ax.axhline(tick, color="0.70", linewidth=0.6, zorder=0)

        for tick in self._depth_ticks(top, bottom, number_spacing):
            ax.text(0.50, tick, f"{tick:g}", ha="center", va="center", fontsize=7.5)

    def _plot_curve_track(self, ax: Axes, track: LogTrackSpec, depth: np.ndarray) -> None:
        label_rows = 0
        plotted_curves: List[Dict[str, Any]] = []
        for spec in track.curves:
            curve = self._get_curve(spec)
            if curve is None:
                self._warn_missing(spec.curve_name)
                continue
            if get_standard_mnemonic(curve.mnemonic) in CORE_SATURATION_CURVES:
                curve = standardize_core_saturation(curve, warn=False)
                evidence = curve.metadata.custom["core_saturation_normalization"]
                for message in evidence["warnings"]:
                    if message not in self.warnings:
                        self.warnings.append(message)
            data = np.asarray(curve.data, dtype=float)
            left = spec.left if spec.left is not None else track.left
            right = spec.right if spec.right is not None else track.right
            x = self._normalize(data, left, right, track.scale)
            finite = np.isfinite(x) & np.isfinite(depth)
            if not np.any(finite):
                self._warn_missing(f"{spec.curve_name} (no finite values after scaling)")
                continue

            label = self._curve_label(self._display_label(spec, curve), curve)
            fill_color = spec.fill_color or spec.color
            if spec.fill:
                self._plot_curve_fill(
                    ax, depth, x, finite, spec, fill_color, left, right, track.scale
                )

            self._plot_cutoff_fills(ax, depth, data, x, finite, left, right, track, spec)
            self._plot_crossover_fills(ax, depth, x, finite, left, right, track, spec)
            marker = spec.marker or self._default_marker_for_curve(curve.mnemonic)
            discrete = self._is_discrete_curve(curve.mnemonic) or bool(marker)
            if discrete:
                ax.scatter(
                    x[finite],
                    depth[finite],
                    color=spec.color,
                    marker=marker or "x",
                    s=(spec.marker_size or 4.0) ** 2,
                    alpha=spec.alpha,
                    label=label,
                    zorder=4,
                    linewidths=0.7,
                )
                line_style = "None"
            else:
                ax.plot(
                    x,
                    depth,
                    color=spec.color,
                    linewidth=spec.line_width,
                    linestyle=spec.line_style,
                    alpha=spec.alpha,
                    label=label,
                    zorder=3,
                )
                line_style = spec.line_style
            self._add_curve_header(
                ax,
                label_rows,
                label,
                left,
                right,
                spec.color,
                track.scale,
                line_style=line_style,
                marker=marker,
            )
            plotted_curves.append(
                {
                    "spec": spec,
                    "curve": curve,
                    "x": x,
                    "finite": finite,
                    "left": left,
                    "right": right,
                }
            )
            label_rows += 1
        self._plot_crossover_flag_overlay(ax, track, depth, plotted_curves)

    def _plot_curve_fill(
        self,
        ax: Axes,
        depth: np.ndarray,
        x: np.ndarray,
        finite: np.ndarray,
        spec: LogCurveSpec,
        fill_color: str,
        left: Optional[float] = None,
        right: Optional[float] = None,
        scale: str = "linear",
    ) -> None:
        direction = str(spec.fill).lower()
        baseline = 0.0 if direction == "left" else 1.0
        if spec.fill_color_map and HAS_MATPLOTLIB:
            values = x[np.isfinite(x)]
            if values.size:
                cmap = plt.get_cmap(spec.fill_color_map)
                quantiles = np.linspace(np.nanmin(values), np.nanmax(values), 16)
                color_range = (spec.left_color_value, spec.right_color_value)
                use_color_range = (
                    left is not None and right is not None
                    and any(limit is not None for limit in color_range)
                )
                for value in quantiles:
                    position = value
                    if use_color_range:
                        # Colour by the data value on the template's colour scale.
                        data_value = self._denormalize_scalar(value, left, right, scale)
                        position = self._normalize_scalar(
                            data_value,
                            left if color_range[0] is None else color_range[0],
                            right if color_range[1] is None else color_range[1],
                            scale,
                        )
                    color = cmap(np.clip(position, 0.0, 1.0))
                    where = finite & (x >= value if direction == "left" else x <= value)
                    ax.fill_betweenx(depth, baseline, x, where=where, color=color, alpha=spec.fill_alpha)
                return
        ax.fill_betweenx(
            depth,
            baseline,
            x,
            where=finite,
            color=fill_color,
            alpha=spec.fill_alpha,
            linewidth=0.0,
            zorder=1,
        )

    def _plot_cumulative_track(self, ax: Axes, track: LogTrackSpec, depth: np.ndarray) -> None:
        if self._is_mineral_volume_track(track):
            self._plot_mineral_volume_track(ax, track, depth)
            return

        cumulative = np.zeros_like(depth, dtype=float)
        label_rows = 0
        for spec in track.curves:
            curve = self._get_curve(spec)
            if curve is None:
                self._warn_missing(spec.curve_name)
                continue
            data = np.asarray(curve.data, dtype=float)
            finite = np.isfinite(data) & np.isfinite(depth)
            if not np.any(finite):
                self._warn_missing(f"{spec.curve_name} (no finite values)")
                continue

            # A missing component leaves a gap rather than being stacked as 0.
            cleaned = np.where(finite, np.clip(data, 0.0, None), np.nan)
            next_cumulative = cumulative + cleaned
            x0 = self._normalize(cumulative, track.left, track.right, track.scale)
            x1 = self._normalize(next_cumulative, track.left, track.right, track.scale)
            fill_color = spec.fill_color or spec.color
            ax.fill_betweenx(
                depth,
                x0,
                x1,
                where=np.isfinite(x0) & np.isfinite(x1),
                color=fill_color,
                alpha=spec.fill_alpha if "fill_alpha" in spec.styled else min(spec.fill_alpha, 0.85),
                linewidth=0.0,
                zorder=1,
            )
            ax.plot(x1, depth, color=spec.color, linewidth=0.45, zorder=2)
            label = self._curve_label(spec.display_name or curve.mnemonic, curve)
            self._add_curve_header(
                ax,
                label_rows,
                label,
                track.left,
                track.right,
                spec.color,
                track.scale,
                line_style=spec.line_style,
                marker=spec.marker,
            )
            label_rows += 1
            cumulative = next_cumulative

    def _plot_mineral_volume_track(self, ax: Axes, track: LogTrackSpec, depth: np.ndarray) -> None:
        cumulative = np.zeros_like(depth, dtype=float)
        plotted_items: List[Dict[str, Any]] = []

        for spec in track.curves:
            curve = self._get_curve(spec)
            if curve is None:
                self._warn_missing(spec.curve_name)
                continue
            data = np.asarray(curve.data, dtype=float)
            finite = np.isfinite(data) & np.isfinite(depth)
            if not np.any(finite):
                self._warn_missing(f"{spec.curve_name} (no finite values)")
                continue

            # A missing component leaves a gap rather than being stacked as 0.
            cleaned = np.where(finite, np.clip(data, 0.0, None), np.nan)
            next_cumulative = cumulative + cleaned
            x0 = self._normalize(cumulative, track.left, track.right, track.scale)
            x1 = self._normalize(next_cumulative, track.left, track.right, track.scale)
            valid = finite & np.isfinite(x0) & np.isfinite(x1)
            litho_label = self._mineral_lithology_label(spec, curve)
            fill_color = self._mineral_fill_color(litho_label, spec.fill_color or spec.color)
            hatch = self._lithology_hatch(litho_label) if litho_label else ""
            ax.fill_betweenx(
                depth,
                x0,
                x1,
                where=valid,
                color=fill_color,
                hatch=hatch,
                edgecolor="0.30" if hatch else "none",
                alpha=spec.fill_alpha if "fill_alpha" in spec.styled else min(spec.fill_alpha, 0.90),
                linewidth=0.0,
                zorder=1,
            )
            boundary = np.where(valid, x1, np.nan)
            ax.plot(boundary, depth, color="0.32", linewidth=0.32, alpha=0.75, zorder=2)
            plotted_items.append(
                {
                    "spec": spec,
                    "curve": curve,
                    "label": spec.display_name or curve.mnemonic,
                    "color": fill_color,
                    "hatch": hatch,
                }
            )
            cumulative = next_cumulative

        # Close the volumetric stack with the porosity wedge (white). Our clay term
        # (VCLAY) is a wet-shale volume that already carries its clay-bound water, so
        # the complement that makes minerals + clay + porosity = 1.0 is the EFFECTIVE
        # porosity (PHIE), not total (PHIT, which would double-count clay-bound water).
        if plotted_items:
            self._plot_mineral_porosity_wedge(ax, track, depth, cumulative, plotted_items)

        left_x = self._normalize_scalar(track.left, track.left, track.right, track.scale)
        right_x = self._normalize_scalar(track.right, track.left, track.right, track.scale)
        for x_value in (left_x, right_x):
            if np.isfinite(x_value):
                ax.axvline(x_value, color="0.18", linewidth=0.65, zorder=3)

        if plotted_items:
            self._add_mineral_volume_header(ax, track, plotted_items)

    def _plot_mineral_porosity_wedge(
        self,
        ax: Axes,
        track: LogTrackSpec,
        depth: np.ndarray,
        cumulative: np.ndarray,
        plotted_items: List[Dict[str, Any]],
    ) -> None:
        """Draw the effective-porosity (PHIE) wedge that closes the mineral stack to 1.0."""
        spec = LogCurveSpec("PHIE", candidates=["PHIE", "PHIE_CC", "PHIT", "PHI"])
        curve = self._get_curve(spec)
        if curve is None:
            return
        data = np.asarray(curve.data, dtype=float)
        finite = np.isfinite(data) & np.isfinite(depth)
        if not np.any(finite):
            return

        phi = np.where(finite, np.clip(data, 0.0, None), np.nan)
        next_cumulative = cumulative + phi
        x0 = self._normalize(cumulative, track.left, track.right, track.scale)
        x1 = self._normalize(next_cumulative, track.left, track.right, track.scale)
        valid = finite & np.isfinite(x0) & np.isfinite(x1)
        ax.fill_betweenx(
            depth,
            x0,
            x1,
            where=valid,
            color="white",
            edgecolor="0.30",
            linewidth=0.0,
            zorder=1,
        )
        boundary = np.where(valid, x1, np.nan)
        ax.plot(boundary, depth, color="0.32", linewidth=0.32, alpha=0.75, zorder=2)
        plotted_items.append(
            {
                "spec": spec,
                "curve": curve,
                "label": str(getattr(curve, "mnemonic", "") or "PHIE"),
                "color": "white",
                "hatch": "",
            }
        )

    def _mineral_lithology_label(self, spec: LogCurveSpec, curve: Any) -> str:
        """Resolve the lithology label whose style a mineral curve inherits."""
        for name in (spec.curve_name, getattr(curve, "mnemonic", None), spec.display_name):
            key = str(name or "").strip().upper()
            if key in MINERAL_VOLUME_LITHOLOGY_LABELS:
                return MINERAL_VOLUME_LITHOLOGY_LABELS[key]
        return ""

    def _mineral_fill_color(self, litho_label: str, default: Any) -> Any:
        if litho_label:
            color = _lookup_lithology_style(litho_label, LITHOLOGY_LABEL_COLORS, None)
            if color is not None:
                return color
        return default

    def _add_mineral_volume_header(
        self,
        ax: Axes,
        track: LogTrackSpec,
        plotted_items: Sequence[Dict[str, Any]],
    ) -> None:
        header_ax = self._header_axis_by_data_axis.get(id(ax))
        if header_ax is None:
            header_ax = ax
            transform = ax.transAxes
        else:
            transform = header_ax.transAxes

        unit = self._mineral_volume_unit(plotted_items)
        scale_y = 0.58
        header_ax.plot(
            [0.08, 0.92],
            [scale_y, scale_y],
            color="0.25",
            linewidth=0.65,
            transform=transform,
            clip_on=True,
        )
        for x_pos, label in ((0.08, f"{track.left:g}"), (0.50, "0.5"), (0.92, f"{track.right:g}")):
            header_ax.plot(
                [x_pos, x_pos],
                [scale_y - 0.018, scale_y + 0.018],
                color="0.25",
                linewidth=0.65,
                transform=transform,
                clip_on=True,
            )
            header_ax.text(
                x_pos,
                scale_y + 0.050,
                label,
                color="black",
                fontsize=7.4,
                ha="center",
                va="center",
                transform=transform,
                clip_on=True,
            )
        if unit:
            header_ax.text(
                0.50,
                scale_y - 0.070,
                f"[{unit}]",
                color="0.25",
                fontsize=7.4,
                fontweight="bold",
                ha="center",
                va="center",
                transform=transform,
                clip_on=True,
            )

        legend_top = 0.40
        legend_bottom = 0.08
        row_count = max(len(plotted_items), 1)
        row_step = 0.0 if row_count == 1 else (legend_top - legend_bottom) / (row_count - 1)
        for index, item in enumerate(plotted_items):
            y_pos = legend_top - index * row_step if row_count > 1 else legend_top
            color = str(item["color"])
            hatch = str(item.get("hatch", "") or "")
            label = self._compact_header_label(str(item["label"]).upper(), limit=18)
            header_ax.add_patch(
                Rectangle(
                    (0.17, y_pos - 0.017),
                    0.065,
                    0.034,
                    facecolor=color,
                    edgecolor="0.25",
                    linewidth=0.35,
                    hatch=hatch,
                    transform=transform,
                    clip_on=True,
                )
            )
            header_ax.text(
                0.27,
                y_pos,
                label,
                color="black",
                fontsize=8.8,
                fontweight="bold",
                ha="left",
                va="center",
                transform=transform,
                clip_on=True,
            )

    def _mineral_volume_unit(self, plotted_items: Sequence[Dict[str, Any]]) -> str:
        for item in plotted_items:
            unit = self._curve_unit(item["curve"])
            if unit:
                return unit
        return "v/v"

    def _plot_lithology_track(
        self,
        ax: Axes,
        track: LogTrackSpec,
        depth: np.ndarray,
        top: float,
        bottom: float,
    ) -> None:
        specs = track.curves or [
            LogCurveSpec(getattr(self, "_code_curve", None) or "LITHOLOGY_GEOLINK", display_name="Lithology")
        ]
        spec = specs[0]
        beds = []
        if self.lithology_intervals is not None:
            low, high = sorted((top, bottom))
            labels = list(dict.fromkeys(self.lithology_intervals["lithology"]))
            for index, label in enumerate(labels):
                rows = self.lithology_intervals[self.lithology_intervals["lithology"] == label]
                segments = [
                    (max(row.top, low), min(row.bottom, high))
                    for row in rows.itertuples()
                    if row.bottom > low and row.top < high
                ]
                if segments:
                    beds.append((label, index, segments))
        else:
            curve = self._get_curve(spec)
            if self.lithology_source == "computed":
                curve = self.well.get("LITHO")
            elif self.lithology_source == "config":
                curve = None
            if curve is not None:
                data = np.asarray(curve.data, dtype=float)
                visible = np.isfinite(data) & np.isfinite(depth)
                visible &= (depth >= min(top, bottom)) & (depth <= max(top, bottom))
                for code in np.unique(data[visible]):
                    mask = visible & np.isclose(data, code)
                    segments = [
                        self._lithology_segment_bounds(depth, indices, top, bottom)
                        for indices in self._lithology_segments(mask)
                    ]
                    beds.append((self._lithology_label(code, curve.mnemonic), code, segments))
        if not beds:
            self._plot_no_lithology(ax)
            return

        legend_handles_by_label: Dict[str, Patch] = {}
        cmap = plt.get_cmap(spec.fill_color_map or "tab20")
        for index, (label, code, segments) in enumerate(beds):
            color = self._lithology_color(code, label, cmap, index, len(beds))
            pattern = self._lithology_pattern(label)
            hatch = "" if pattern else self._lithology_hatch(label)
            legend_hatch = self._lithology_hatch(label)
            fill_alpha = spec.fill_alpha if spec.fill_alpha > 0.5 or "fill_alpha" in spec.styled else 0.85
            # Fill each contiguous bed edge-to-edge (segment bounds reach the sample
            # midpoints) so adjacent beds abut. fill_betweenx(where=mask) instead
            # leaves a one-sample white sliver at every lithology transition, which
            # stacks into visible gaps in finely interbedded sections. Genuine data
            # gaps (missing samples) remain unfilled because the segment split keeps
            # them apart.
            for y0, y1 in segments:
                if not (np.isfinite(y0) and np.isfinite(y1)) or y1 <= y0:
                    continue
                ax.fill_betweenx(
                    [y0, y1],
                    0.0,
                    1.0,
                    color=color,
                    hatch=hatch,
                    edgecolor="0.28" if hatch else "none",
                    alpha=fill_alpha,
                    linewidth=0.2 if hatch else 0.0,
                    zorder=1,
                )
                if pattern:
                    self._draw_lithology_pattern(ax, label, y0, y1)
            legend_label = self._lithology_legend_label(label)
            if legend_label not in legend_handles_by_label:
                legend_handles_by_label[legend_label] = Patch(
                    facecolor=color,
                    edgecolor="0.28",
                    hatch=legend_hatch,
                    label=legend_label,
                )

        ax.set_xlim(0.0, 1.0)
        legend_handles = list(legend_handles_by_label.values())
        if legend_handles:
            legend_handles = legend_handles[:18]
            max_cols = min(4, max(1, len(legend_handles)))
            ax.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.025),
                fontsize=8.0,
                ncol=max_cols,
                frameon=False,
            )

    def _plot_no_lithology(self, ax: Axes) -> None:
        ax.set_xlim(0.0, 1.0)
        ax.set_facecolor("#f7f7f7")
        ax.text(
            0.5,
            0.5,
            NO_LITHOLOGY_LABEL,
            color="#595959",
            fontsize=9.2,
            fontweight="bold",
            ha="center",
            va="center",
            rotation=90,
            transform=ax.transAxes,
        )
        self._header_row_counts[id(ax)] = 1

    def _lithology_segments(self, mask: np.ndarray) -> List[np.ndarray]:
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            return []
        split_at = np.flatnonzero(np.diff(indices) > 1) + 1
        return [segment for segment in np.split(indices, split_at) if segment.size > 0]

    def _lithology_segment_bounds(
        self,
        depth: np.ndarray,
        indices: np.ndarray,
        top: float,
        bottom: float,
    ) -> Tuple[float, float]:
        segment_depth = np.asarray(depth[indices], dtype=float)
        segment_depth = segment_depth[np.isfinite(segment_depth)]
        if segment_depth.size == 0:
            return float(top), float(top)

        finite_depth = np.asarray(depth[np.isfinite(depth)], dtype=float)
        if finite_depth.size > 1:
            step = float(np.nanmedian(np.abs(np.diff(finite_depth))))
            if not np.isfinite(step) or step <= 0:
                step = 1.0
        else:
            step = 1.0

        lo = float(np.nanmin(segment_depth) - step * 0.5)
        hi = float(np.nanmax(segment_depth) + step * 0.5)
        visible_lo, visible_hi = sorted((float(top), float(bottom)))
        return max(lo, visible_lo), min(hi, visible_hi)

    def _lithology_pattern(self, label: str) -> str:
        return _lookup_lithology_style(label, LITHOLOGY_CUSTOM_PATTERNS, "")

    def _lithology_legend_label(self, label: str) -> str:
        if _lithology_label_key(label) not in BUILTIN_LITHOLOGY_LABEL_KEYS:
            return str(label)
        grouped = _lookup_lithology_style(label, COMMON_LITHOLOGY_LEGEND_LABELS, "")
        return grouped or str(label)

    def _pattern_grid(
        self,
        y0: float,
        y1: float,
        rows: int,
        cols: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_low, y_high = sorted((float(y0), float(y1)))
        if rows <= 0 or cols <= 0 or y_high <= y_low:
            return np.asarray([]), np.asarray([])
        xs = np.linspace(0.14, 0.86, cols)
        ys = np.linspace(y_low, y_high, rows + 2)[1:-1]
        xx, yy = np.meshgrid(xs, ys)
        if rows > 1 and cols > 1:
            spacing = xs[1] - xs[0]
            xx[1::2] = np.clip(xx[1::2] + spacing * 0.38, 0.14, 0.90)
        return xx.ravel(), yy.ravel()

    def _lithology_tile_rgba(self, tile_name: str) -> np.ndarray:
        height, width = 46, 92
        image = np.zeros((height, width, 4), dtype=float)
        line = np.asarray([0.10, 0.10, 0.10, 0.82], dtype=float)

        def bounds(value: int, upper: int) -> int:
            return int(max(0, min(upper, value)))

        def hline(y: int, x0: int = 0, x1: Optional[int] = None, thickness: int = 1) -> None:
            x1 = width if x1 is None else x1
            y0 = bounds(y - thickness // 2, height)
            y1 = bounds(y0 + thickness, height)
            image[y0:y1, bounds(x0, width):bounds(x1, width), :] = line

        def vline(x: int, y0: int = 0, y1: Optional[int] = None, thickness: int = 1) -> None:
            y1 = height if y1 is None else y1
            x0 = bounds(x - thickness // 2, width)
            x1 = bounds(x0 + thickness, width)
            image[bounds(y0, height):bounds(y1, height), x0:x1, :] = line

        def diagonal(x0: int, y0: int, x1: int, y1: int, thickness: int = 1) -> None:
            steps = max(abs(x1 - x0), abs(y1 - y0), 1) + 1
            xs = np.linspace(x0, x1, steps).round().astype(int)
            ys = np.linspace(y0, y1, steps).round().astype(int)
            radius = max(0, thickness // 2)
            paint_points(xs, ys, radius)

        def paint_points(xs: np.ndarray, ys: np.ndarray, radius: int) -> None:
            for x, y in zip(xs, ys):
                image[
                    bounds(y - radius, height):bounds(y + radius + 1, height),
                    bounds(x - radius, width):bounds(x + radius + 1, width),
                    :,
                ] = line

        def wave(y_center: int, amplitude: int = 3, cycles: float = 2.0, thickness: int = 1) -> None:
            xs = np.arange(4, width - 4)
            phase = np.linspace(0.0, cycles * 2.0 * np.pi, xs.size)
            ys = np.rint(y_center + amplitude * np.sin(phase)).astype(int)
            paint_points(xs, ys, max(0, thickness // 2))

        def calcareous_marker(cx: int, cy: int) -> None:
            hline(cy, cx - 6, cx + 7, thickness=2)
            vline(cx, cy - 8, cy + 1, thickness=2)

        mid = height // 2
        if tile_name == "chalk":
            hline(mid, thickness=2)
            for x in (8, 14, width - 16, width - 9):
                vline(x, 0, mid, thickness=2)
            vline(width // 2, mid, height, thickness=2)
        elif tile_name == "limestone":
            hline(mid, thickness=2)
            for x in (14, width - 18):
                vline(x, 0, mid, thickness=2)
            vline(width // 2, mid, height, thickness=2)
        elif tile_name == "dolomite":
            hline(mid, thickness=2)
            for x in (24, 70):
                diagonal(x, 0, x + 14, mid, thickness=3)
                diagonal(x - 14, mid, x, height, thickness=3)
        elif tile_name == "calcareous_clay":
            for cx, cy in ((18, 13), (47, 24), (74, 14), (25, 37), (68, 36)):
                calcareous_marker(cx, cy)
        elif tile_name == "marl":
            for y_center in (12, 24, 36):
                wave(y_center, amplitude=3, cycles=2.3, thickness=2)
        return image

    def _draw_lithology_tile(
        self,
        ax: Axes,
        tile_name: str,
        y0: float,
        y1: float,
        zorder: float,
    ) -> None:
        tile = self._lithology_tile_rgba(tile_name)
        if tile.size == 0:
            return

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        y_low, y_high = sorted((float(y0), float(y1)))
        axis_span = abs(float(np.diff(ylim)[0])) or (y_high - y_low)
        interval_fraction = min(1.0, max(0.0, (y_high - y_low) / axis_span))
        bbox = ax.get_window_extent()
        width_px = int(max(tile.shape[1], bbox.width or tile.shape[1] * 3))
        height_px = int(max(tile.shape[0], (bbox.height or tile.shape[0] * 3) * interval_fraction))
        repeat_x = max(1, math.ceil(width_px / tile.shape[1]))
        repeat_y = max(1, math.ceil(height_px / tile.shape[0]))
        tiled = np.tile(tile, (repeat_y, repeat_x, 1))

        ax.imshow(
            tiled,
            extent=(0.0, 1.0, y_low, y_high),
            origin="upper",
            aspect="auto",
            interpolation="nearest",
            zorder=zorder,
            clip_on=True,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    def _draw_lithology_pattern(
        self,
        ax: Axes,
        label: str,
        y0: float,
        y1: float,
        density_fraction: Optional[float] = None,
    ) -> None:
        pattern = self._lithology_pattern(label)
        if not pattern or pattern == "solid":
            return

        y_low, y_high = sorted((float(y0), float(y1)))
        height = y_high - y_low
        if not np.isfinite(height) or height <= 0:
            return

        axis_span = abs(float(np.diff(ax.get_ylim())[0])) or height
        # Pattern density is keyed to the bed's on-screen fraction, so spacing stays
        # constant per pixel and thin beds (< ~0.6%) drop the pattern entirely. A
        # legend swatch fills its whole tile (fraction ~1) yet must still render at a
        # normal density, so callers can pass density_fraction to set the density
        # independently of the fill extent (0.05 reproduces the reference look).
        bed_fraction = (
            float(density_fraction) if density_fraction is not None
            else height / max(axis_span, 1e-9)
        )
        if bed_fraction < 0.006:
            return
        # rows grows with bed height so the vertical pattern spacing is constant per
        # pixel for every bed (the ceiling is only a safety cap); cols is fixed since
        # the track width does not change.
        rows = int(np.clip(math.ceil(bed_fraction / 0.045), 2, 40))
        cols = 4
        color = "0.15"
        if pattern == "shale_lines":
            color = "0.45"
        elif pattern == "spiculite_trees":
            color = "0.35"
        zorder = 2.8
        tile_name = LITHOLOGY_TILE_PATTERNS.get(pattern)
        if tile_name:
            self._draw_lithology_tile(ax, tile_name, y_low, y_high, zorder)
            return

        def regular_points(
            rows_count: int,
            cols_count: int,
            x0: float = 0.14,
            x1: float = 0.86,
        ) -> Tuple[np.ndarray, np.ndarray]:
            if rows_count <= 0 or cols_count <= 0:
                return np.asarray([]), np.asarray([])
            xs = np.linspace(x0, x1, cols_count)
            ys = np.linspace(y_low, y_high, rows_count + 2)[1:-1]
            xx, yy = np.meshgrid(xs, ys)
            return xx.ravel(), yy.ravel()

        def regular_scatter(
            marker: Any,
            size: float,
            rows_count: int,
            cols_count: int,
            x0: float = 0.14,
            x1: float = 0.86,
            **kwargs: Any,
        ) -> None:
            xs, ys = regular_points(rows_count, cols_count, x0=x0, x1=x1)
            if xs.size == 0:
                return
            scatter_kwargs = {
                "marker": marker,
                "s": size,
                "linewidths": kwargs.pop("linewidths", 0.6),
                "zorder": zorder,
                "clip_on": True,
            }
            if "facecolors" not in kwargs and "edgecolors" not in kwargs:
                scatter_kwargs["color"] = kwargs.pop("color", color)
            scatter_kwargs.update(kwargs)
            ax.scatter(xs, ys, **scatter_kwargs)

        def scatter(marker: Any, size: float = 12.0, rows_override: Optional[int] = None,
                    cols_override: Optional[int] = None, **kwargs: Any) -> None:
            xs, ys = self._pattern_grid(
                y_low,
                y_high,
                rows_override if rows_override is not None else rows,
                cols_override if cols_override is not None else cols,
            )
            if xs.size == 0:
                return
            scatter_kwargs = {
                "marker": marker,
                "s": size,
                "linewidths": kwargs.pop("linewidths", 0.6),
                "zorder": zorder,
                "clip_on": True,
            }
            if "facecolors" not in kwargs and "edgecolors" not in kwargs:
                scatter_kwargs["color"] = kwargs.pop("color", color)
            scatter_kwargs.update(kwargs)
            ax.scatter(xs, ys, **scatter_kwargs)

        def horizontal_lines(count: int, linewidth: float = 0.45, x0: float = 0.06, x1: float = 0.94) -> None:
            ys = np.linspace(y_low, y_high, count + 2)[1:-1]
            ax.hlines(ys, x0, x1, color=color, linewidth=linewidth, zorder=zorder, clip_on=True)

        def vertical_lines(count: int, linewidth: float = 0.45, ypad: float = 0.0) -> None:
            xs = np.linspace(0.10, 0.90, count)
            ax.vlines(xs, y_low + ypad, y_high - ypad, color=color, linewidth=linewidth, zorder=zorder, clip_on=True)

        def short_dashes(with_dots: bool = False, dot_count: int = 1) -> None:
            ys = np.linspace(y_low, y_high, rows + 2)[1:-1]
            for row, y in enumerate(ys):
                offset = 0.06 if row % 2 else 0.0
                for x_start in (0.12 + offset, 0.42 + offset, 0.72 + offset):
                    if x_start + 0.14 <= 0.94:
                        ax.hlines(y, x_start, x_start + 0.14, color=color, linewidth=0.6, zorder=zorder, clip_on=True)
                if with_dots:
                    xs = np.linspace(0.30, 0.70, dot_count)
                    ax.scatter(xs, np.full(xs.shape, y), marker=".", s=8, color=color, zorder=zorder, clip_on=True)

        def diagonal_lines(direction: int = 1, count: Optional[int] = None, linewidth: float = 0.6) -> None:
            n_lines = count or max(3, rows + 3)
            starts = np.linspace(-0.35, 0.95, n_lines)
            for start in starts:
                x_start = max(0.04, start)
                x_end = min(0.96, start + 0.38)
                if x_end <= 0.04 or x_start >= 0.96 or x_end <= x_start:
                    continue
                frac0 = (x_start - start) / 0.38
                frac1 = (x_end - start) / 0.38
                if direction > 0:
                    yy0 = y_high - frac0 * height
                    yy1 = y_high - frac1 * height
                else:
                    yy0 = y_low + frac0 * height
                    yy1 = y_low + frac1 * height
                ax.plot([x_start, x_end], [yy0, yy1], color=color, linewidth=linewidth, zorder=zorder, clip_on=True)

        def vrows(base: float, ref: float = 0.05) -> int:
            """Vertical repeat count for constant on-screen spacing: a pattern's
            base count is preserved at a reference bed (~5% of track height) and
            scales with thickness so every bed reads at the same density."""
            return int(max(1, round(base * bed_fraction / ref)))

        if pattern == "horizontal_dense":
            # Count grows with bed height so the line SPACING is constant per pixel
            # for every bed (the high ceiling is only a safety cap, not a target).
            horizontal_lines(int(np.clip(round(bed_fraction / 0.012), 1, 120)), linewidth=0.45)
        elif pattern == "shale_lines":
            horizontal_lines(int(np.clip(round(bed_fraction / 0.012), 1, 120)), linewidth=0.35)
        elif pattern == "dash":
            short_dashes()
        elif pattern == "dash_dot":
            short_dashes(with_dots=True, dot_count=1)
        elif pattern == "dots":
            scatter(".", size=10, cols_override=5,
                    rows_override=int(np.clip(round(bed_fraction / 0.012), 1, 60)))
        elif pattern == "vertical_dot_columns":
            regular_scatter(".", size=9, rows_count=7, cols_count=7)
        elif pattern == "silt_strokes":
            dash_ys = np.linspace(y_low, y_high, 7 + 2)[1:-1]
            for group_x in (0.14, 0.42, 0.70):
                ax.hlines(dash_ys, group_x, group_x + 0.16, color=color, linewidth=0.55, zorder=zorder, clip_on=True)
                ax.vlines(
                    group_x + 0.13,
                    y_low + height * 0.18,
                    y_high - height * 0.18,
                    color=color,
                    linewidth=0.55,
                    zorder=zorder,
                    clip_on=True,
                )
        elif pattern == "pebbles":
            scatter("o", size=13, rows_override=vrows(3), cols_override=4, facecolors="none", edgecolors=color)
            scatter(".", size=6, rows_override=vrows(3), cols_override=5)
        elif pattern == "vertical_open_circles":
            regular_scatter("o", size=15, rows_count=5, cols_count=6, facecolors="none", edgecolors=color)
        elif pattern == "triangles":
            scatter("^", size=18, rows_override=vrows(3), cols_override=4, color=color)
            scatter(".", size=6, rows_override=vrows(3), cols_override=5)
        elif pattern == "chalk_grid":
            vertical_lines(5, linewidth=0.55)
            horizontal_lines(2, linewidth=0.45)
        elif pattern == "carbonate_grid_dense":
            vertical_lines(4, linewidth=0.45)
            horizontal_lines(8, linewidth=0.35)
        elif pattern == "block_grid":
            vertical_lines(4, linewidth=0.55)
            horizontal_lines(2, linewidth=0.55)
        elif pattern == "argillaceous_limestone":
            stripe = Rectangle(
                (0.38, y_low + height * 0.12),
                0.24,
                height * 0.76,
                facecolor="#2f2f2f",
                edgecolor="none",
                alpha=0.18,
                zorder=zorder - 0.1,
                clip_on=True,
            )
            ax.add_patch(stripe)
            ax.hlines(
                np.linspace(y_low + height * 0.18, y_high - height * 0.18, 10),
                0.38,
                0.62,
                color="0.20",
                linewidth=0.35,
                zorder=zorder,
                clip_on=True,
            )
        elif pattern == "diagonal":
            diagonal_lines(direction=1)
        elif pattern == "diagonal_dense":
            diagonal_lines(direction=1, count=max(7, rows + 5), linewidth=0.65)
        elif pattern == "cross_diagonal":
            diagonal_lines(direction=1, linewidth=0.55)
            diagonal_lines(direction=-1, linewidth=0.55)
        elif pattern == "grid":
            vertical_lines(4, linewidth=0.55)
            horizontal_lines(vrows(2), linewidth=0.55)
        elif pattern == "evaporite_grid_dense":
            vertical_lines(4, linewidth=0.45)
            horizontal_lines(8, linewidth=0.35)
        elif pattern == "calcareous_clay":
            regular_scatter("+", size=18, rows_count=4, cols_count=5)
        elif pattern == "waves":
            centers = np.linspace(y_low, y_high, max(2, rows) + 2)[1:-1]
            xs = np.linspace(0.12, 0.88, 60)
            amplitude = height * 0.035
            for y_center in centers:
                ys = y_center + amplitude * np.sin(np.linspace(0, 4 * np.pi, xs.size))
                ax.plot(xs, ys, color=color, linewidth=0.55, zorder=zorder, clip_on=True)
        elif pattern == "waves_dense":
            centers = np.linspace(y_low, y_high, 6 + 2)[1:-1]
            xs = np.linspace(0.10, 0.90, 80)
            amplitude = height * 0.025
            for y_center in centers:
                ys = y_center + amplitude * np.sin(np.linspace(0, 4 * np.pi, xs.size))
                ax.plot(xs, ys, color=color, linewidth=0.50, zorder=zorder, clip_on=True)
        elif pattern == "carets":
            scatter("^", size=20, rows_override=vrows(2), cols_override=3,
                    facecolors="none", edgecolors=color, linewidths=0.9)
        elif pattern == "vertical_carets":
            regular_scatter("^", size=22, rows_count=4, cols_count=5, facecolors="none", edgecolors=color, linewidths=0.9)
        elif pattern == "chevrons_left":
            scatter("<", size=20, rows_override=vrows(2), cols_override=3,
                    facecolors="none", edgecolors=color, linewidths=0.9)
        elif pattern == "plus_symbols":
            scatter("+", size=24, rows_override=vrows(2), cols_override=3)
        elif pattern == "x_symbols":
            scatter("x", size=22, rows_override=vrows(2), cols_override=3)
        elif pattern == "igneous_dash":
            short_dashes()
            scatter("|", size=16, rows_override=max(2, rows), cols_override=3)
        elif pattern == "basement_symbols":
            for group_x in (0.16, 0.44, 0.72):
                dash_ys = np.linspace(y_low + height * 0.16, y_high - height * 0.16, 5)
                ax.hlines(dash_ys, group_x, group_x + 0.14, color=color, linewidth=0.65, zorder=zorder, clip_on=True)
                ax.vlines(
                    group_x + 0.11,
                    y_low + height * 0.18,
                    y_high - height * 0.18,
                    color=color,
                    linewidth=0.55,
                    zorder=zorder,
                    clip_on=True,
                )
        elif pattern == "tuff_slashes":
            scatter(r"$/$", size=22, rows_override=vrows(2), cols_override=3, linewidths=0.8)
        elif pattern == "lightning_columns":
            for x in (0.18, 0.36, 0.54, 0.72, 0.86):
                ys = np.linspace(y_low + height * 0.10, y_high - height * 0.10, 8)
                offsets = np.asarray([0.00, -0.012, 0.010, -0.014, 0.012, -0.010, 0.014, 0.00])
                ax.plot(x + offsets, ys, color=color, linewidth=0.9, zorder=zorder, clip_on=True)
        elif pattern == "volcanic_chevrons":
            regular_scatter("v", size=22, rows_count=4, cols_count=6, facecolors="none", edgecolors=color, linewidths=0.9)
        elif pattern == "volcanic_v":
            scatter("v", size=20, rows_override=vrows(2), cols_override=3,
                    facecolors="none", edgecolors=color, linewidths=0.9)
        elif pattern == "argillaceous_dash":
            horizontal_lines(max(1, rows), linewidth=0.7, x0=0.35, x1=0.65)
        elif pattern == "limestone_stringer":
            horizontal_lines(max(1, rows), linewidth=0.7, x0=0.25, x1=0.75)
            vertical_lines(2, linewidth=0.6, ypad=height * 0.30)
        elif pattern == "dolomite_stringer":
            diagonal_lines(direction=1, count=max(2, rows), linewidth=0.7)
            horizontal_lines(max(1, rows), linewidth=0.55, x0=0.25, x1=0.75)
        elif pattern == "calcareous_t":
            horizontal_lines(max(1, rows), linewidth=0.7, x0=0.35, x1=0.65)
            xs, ys = self._pattern_grid(y_low, y_high, max(1, rows), 1)
            ax.vlines(xs, ys, ys + height * 0.05, color=color, linewidth=0.7, zorder=zorder, clip_on=True)
        elif pattern == "dolomitic_slash":
            diagonal_lines(direction=1, count=max(2, rows), linewidth=0.7)
        elif pattern == "macrofossils":
            scatter(r"$\sim$", size=30, rows_override=max(1, rows), cols_override=2)
        elif pattern == "foraminifera":
            scatter("o", size=16, rows_override=max(1, rows), cols_override=3, facecolors="none", edgecolors=color, color=color)
        elif pattern == "triangle_single":
            scatter("^", size=28, rows_override=max(1, rows), cols_override=2)
        elif pattern == "star_symbols":
            scatter("*", size=28, rows_override=max(1, rows), cols_override=2)
        elif pattern == "square_symbols":
            scatter("s", size=18, rows_override=max(1, rows), cols_override=2, facecolors="none", edgecolors=color, color=color)
        elif pattern == "mica_symbols":
            scatter(r"$m$", size=26, rows_override=max(1, rows), cols_override=2)
        elif pattern == "carbonaceous_bar":
            ys = np.linspace(y_low, y_high, max(1, rows) + 2)[1:-1]
            ax.hlines(ys, 0.35, 0.65, color="black", linewidth=3.0, zorder=zorder, clip_on=True)
        elif pattern == "calcareous_cement":
            dot_ys = np.linspace(y_low + height * 0.15, y_high - height * 0.15, 5)
            ax.scatter(
                np.full(dot_ys.shape, 0.18),
                dot_ys,
                marker=".",
                s=8,
                color=color,
                zorder=zorder,
                clip_on=True,
            )
            ax.hlines(
                np.linspace(y_low + height * 0.22, y_high - height * 0.22, 8),
                0.36,
                0.66,
                color=color,
                linewidth=0.45,
                zorder=zorder,
                clip_on=True,
            )
        elif pattern == "spiculite_trees":
            for x in (0.24, 0.50, 0.76):
                ax.vlines(x, y_low + height * 0.18, y_high - height * 0.16, color=color, linewidth=0.75, zorder=zorder, clip_on=True)
            regular_scatter("^", size=24, rows_count=2, cols_count=3, facecolors=color, edgecolors=color, linewidths=0.7)

    def _plot_flag_track(self, ax: Axes, track: LogTrackSpec, depth: np.ndarray) -> None:
        available_specs: List[Tuple[LogCurveSpec, Any]] = []
        for spec in track.curves:
            curve = self._get_curve(spec)
            if curve is None:
                self._warn_missing(spec.curve_name)
                continue
            available_specs.append((spec, curve))

        if not available_specs:
            self._plot_no_flags(ax)
            return

        n_curves = len(available_specs)
        row_height = 1.0 / n_curves
        for idx, (spec, curve) in enumerate(available_specs):
            data = np.asarray(curve.data, dtype=float)
            finite = np.isfinite(data) & np.isfinite(depth)
            active = finite & (data > 0.5)
            y0 = idx * row_height
            y1 = (idx + 1) * row_height
            color = spec.fill_color or FLAG_COLORS.get(spec.curve_name.upper(), spec.color)
            cell_depth, cell_active = _sample_cell_fill(depth, active)
            ax.fill_betweenx(
                cell_depth, y0, y1, where=cell_active, color=color, alpha=0.75, linewidth=0.0
            )
            ax.axvline(y0, color="0.92", linewidth=0.4, zorder=0)
            label = spec.display_name or curve.mnemonic
            self._add_curve_header(ax, idx, label, 0.0, 1.0, spec.color, "linear")
        ax.set_xlim(0.0, 1.0)

    def _plot_no_flags(self, ax: Axes) -> None:
        ax.set_xlim(0.0, 1.0)
        ax.set_facecolor("#fbf7f7")
        ax.text(
            0.5,
            0.5,
            "No flags available",
            color="#595959",
            fontsize=9.2,
            fontweight="bold",
            ha="center",
            va="center",
            rotation=90,
            transform=ax.transAxes,
        )
        self._header_row_counts[id(ax)] = 1

    def _plot_cutoff_fills(
        self,
        ax: Axes,
        depth: np.ndarray,
        data: np.ndarray,
        x: np.ndarray,
        finite: np.ndarray,
        left: float,
        right: float,
        track: LogTrackSpec,
        spec: LogCurveSpec,
    ) -> None:
        fill_color = spec.fill_color or spec.color
        if spec.left_cutoff_fill is not None:
            cutoff_x = self._normalize_scalar(spec.left_cutoff_fill, left, right, track.scale)
            ax.fill_betweenx(
                depth,
                cutoff_x,
                x,
                where=finite & (data < spec.left_cutoff_fill),
                color=fill_color,
                alpha=spec.fill_alpha,
                linewidth=0.0,
                zorder=1,
            )
        if spec.right_cutoff_fill is not None:
            cutoff_x = self._normalize_scalar(spec.right_cutoff_fill, left, right, track.scale)
            ax.fill_betweenx(
                depth,
                cutoff_x,
                x,
                where=finite & (data > spec.right_cutoff_fill),
                color=fill_color,
                alpha=spec.fill_alpha,
                linewidth=0.0,
                zorder=1,
            )

    def _plot_crossover_fills(
        self,
        ax: Axes,
        depth: np.ndarray,
        x: np.ndarray,
        finite: np.ndarray,
        left: float,
        right: float,
        track: LogTrackSpec,
        spec: LogCurveSpec,
    ) -> None:
        fill_color = spec.fill_color or spec.color
        for attr, direction in ((spec.left_crossover, "left"), (spec.right_crossover, "right")):
            if not attr:
                continue
            other_curve = self._get_curve(LogCurveSpec(str(attr)))
            if other_curve is None:
                self._warn_missing(str(attr))
                continue
            other = np.asarray(other_curve.data, dtype=float)
            # Normalize the other curve on the scale it is drawn with in this
            # track; fall back to this curve's scale when it is not plotted here.
            other_left, other_right = left, right
            for other_spec in track.curves:
                if other_spec is spec:
                    continue
                candidate = self._get_curve(other_spec)
                if candidate is not None and candidate.mnemonic == other_curve.mnemonic:
                    other_left = other_spec.left if other_spec.left is not None else track.left
                    other_right = other_spec.right if other_spec.right is not None else track.right
                    break
            other_x = self._normalize(other, other_left, other_right, track.scale)
            valid = finite & np.isfinite(other_x)
            where = valid & (other_x < x) if direction == "left" else valid & (other_x > x)
            ax.fill_betweenx(
                depth,
                other_x,
                x,
                where=where,
                color=fill_color,
                alpha=spec.fill_alpha,
                linewidth=0.0,
                zorder=1,
            )

    def _add_curve_header(
        self,
        ax: Axes,
        row: int,
        label: str,
        left: float,
        right: float,
        color: str,
        scale: str,
        line_style: str = "-",
        marker: Optional[str] = None,
    ) -> None:
        header_ax = self._header_axis_by_data_axis.get(id(ax))
        if header_ax is None:
            header_ax = ax
            transform = ax.transAxes
            row_step = 0.044
            row_center = 1.018 - row * row_step
        else:
            transform = header_ax.transAxes
            row_count = max(self._header_row_counts.get(id(ax), 1), 1)
            if row_count == 1:
                row_step = 0.22
                row_center = 0.46
            else:
                row_step = min(0.135, 0.64 / (row_count - 1))
                row_top = min(0.72, 0.14 + row_step * (row_count - 1))
                row_center = row_top - row * row_step

        label_text, unit_text = self._split_header_label(label)
        if str(scale).lower() == "log":
            unit_text = f"{unit_text}, log" if unit_text else "log"
        label_y = row_center + row_step * 0.16
        scale_y = row_center - row_step * 0.22
        sample_style = "None" if str(line_style).lower() == "none" else line_style
        header_ax.plot(
            [0.25, 0.39],
            [label_y, label_y],
            color=color,
            linestyle=sample_style,
            linewidth=1.75,
            marker=marker,
            markersize=5.4 if marker else 0.0,
            transform=transform,
            clip_on=True,
        )
        header_ax.text(
            0.43,
            label_y,
            self._compact_header_label(label_text),
            color=color,
            fontsize=8.9,
            fontweight="bold",
            ha="left",
            va="center",
            transform=transform,
            clip_on=True,
        )
        header_ax.text(
            0.04,
            scale_y,
            f"{left:g}",
            color="black",
            fontsize=6.8,
            ha="left",
            va="center",
            transform=transform,
            clip_on=True,
        )
        if unit_text:
            header_ax.text(
                0.50,
                scale_y,
                unit_text,
                color=color,
                fontsize=6.6,
                fontweight="bold",
                ha="center",
                va="center",
                transform=transform,
                clip_on=True,
            )
        header_ax.text(
            0.96,
            scale_y,
            f"{right:g}",
            color="black",
            fontsize=6.8,
            ha="right",
            va="center",
            transform=transform,
            clip_on=True,
        )

    def _curve_label(self, label: str, curve: Any) -> str:
        unit = self._curve_unit(curve)
        if unit and f"[{unit}]" not in label:
            return f"{label} [{unit}]"
        return label

    def _display_label(self, spec: LogCurveSpec, curve: Any) -> str:
        mnemonic = str(getattr(curve, "mnemonic", spec.curve_name)).upper()
        if mnemonic in RESISTIVITY_CURVE_NAMES or spec.curve_name.upper() in RESISTIVITY_CURVE_NAMES:
            return mnemonic
        if mnemonic != spec.curve_name.upper():
            # A curve drawn under an alias keeps its own name (DPOR, not PHID):
            # the header must not claim a curve the well does not hold.
            return porosity_display_label(mnemonic, curve)
        label = spec.display_name or getattr(curve, "mnemonic", spec.curve_name)
        return porosity_display_label(label, curve)

    def _curve_unit(self, curve: Any) -> str:
        unit = getattr(curve, "unit", None) or getattr(curve, "units", None) or ""
        unit = str(unit).strip()
        mnemonic = str(getattr(curve, "mnemonic", "") or "").upper()
        normalized = self._normalize_display_unit(unit)
        if mnemonic in FRACTION_SATURATION_CURVE_NAMES and normalized in {"", "v/v", "frac"}:
            return "frac"
        if mnemonic in RESISTIVITY_CURVE_NAMES and normalized in {"", "ohm.m"}:
            return "ohm.m"
        if normalized:
            return normalized
        return DEFAULT_CURVE_DISPLAY_UNITS.get(mnemonic, "")

    def _normalize_display_unit(self, unit: str) -> str:
        text = str(unit or "").strip()
        if text.lower() in {"", "-", "none", "nan"}:
            return ""
        key = re.sub(r"\s+", " ", text.lower().replace("_", " ")).strip()
        key = key.replace("ohm-m", "ohm.m")
        key = key.replace("ohm m", "ohm.m")
        return UNIT_DISPLAY_ALIASES.get(key, text)

    def _split_header_label(self, label: str) -> Tuple[str, str]:
        match = re.match(r"^(?P<name>.*?)\s*(?P<unit>\[[^\]]+\])\s*$", str(label))
        if not match:
            return str(label), ""
        return match.group("name").strip(), match.group("unit").strip()

    def _compact_header_label(self, label: str, limit: int = 16) -> str:
        label = str(label).strip()
        if len(label) <= limit:
            return label
        return f"{label[: max(limit - 1, 1)]}."

    def _is_discrete_curve(self, curve_name: str) -> bool:
        upper = str(curve_name).upper()
        return upper in DISCRETE_CURVE_NAMES

    def _default_marker_for_curve(self, curve_name: str) -> Optional[str]:
        upper = str(curve_name).upper()
        if upper in DISCRETE_CURVE_NAMES:
            return "x"
        return None

    def _plot_crossover_flag_overlay(
        self,
        ax: Axes,
        track: LogTrackSpec,
        depth: np.ndarray,
        plotted_curves: Sequence[Dict[str, Any]],
    ) -> None:
        if not plotted_curves:
            return
        flag_curve = self._first_available_curve(CROSSOVER_CURVE_NAMES)
        if flag_curve is None:
            return
        flag_data = np.asarray(flag_curve.data, dtype=float)
        active = np.isfinite(flag_data) & (flag_data > 0.5) & np.isfinite(depth)
        if not np.any(active):
            return

        pair = self._find_plotted_pair(
            plotted_curves,
            first_candidates=("NPHI", "NPHI_HC", "NPHIC", "PHIN"),
            second_candidates=("RHOB", "DEN", "ZDEN", "RHOZ", "PHID"),
        )
        if pair is None:
            pair = self._find_plotted_pair(
                plotted_curves,
                first_candidates=("PHIN", "NPHI"),
                second_candidates=("PHID", "DPHI"),
            )
        if pair is None:
            return

        first, second = pair
        valid = active & first["finite"] & second["finite"]
        if not np.any(valid):
            return
        ax.fill_betweenx(
            depth,
            first["x"],
            second["x"],
            where=valid,
            color="#fdae61",
            alpha=0.32,
            linewidth=0.0,
            zorder=2.2,
        )

    def _find_plotted_pair(
        self,
        plotted_curves: Sequence[Dict[str, Any]],
        first_candidates: Sequence[str],
        second_candidates: Sequence[str],
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        first = self._find_plotted_curve(plotted_curves, first_candidates)
        second = self._find_plotted_curve(plotted_curves, second_candidates)
        if first is None or second is None or first is second:
            return None
        return first, second

    def _find_plotted_curve(
        self,
        plotted_curves: Sequence[Dict[str, Any]],
        candidates: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        candidate_set = {candidate.upper() for candidate in candidates}
        for item in plotted_curves:
            names = {
                str(getattr(item["curve"], "mnemonic", "")).upper(),
                str(item["spec"].curve_name).upper(),
            }
            if names & candidate_set:
                return item
        return None

    def _draw_bad_hole_shading(self, depth: np.ndarray) -> None:
        curve = self._first_available_curve(BAD_HOLE_CURVE_NAMES)
        if curve is None:
            return
        data = np.asarray(curve.data, dtype=float)
        active = np.isfinite(data) & (data > 0.5) & np.isfinite(depth)
        if not np.any(active):
            return
        cell_depth, cell_active = _sample_cell_fill(depth, active)
        for ax in self.axes:
            ax.fill_betweenx(
                cell_depth,
                0.0,
                1.0,
                where=cell_active,
                color="#d73027",
                alpha=0.075,
                linewidth=0.0,
                zorder=0.6,
            )
        self._add_bad_hole_legend()

    def _add_bad_hole_legend(self) -> None:
        if self.fig is None:
            return
        if self._bad_hole_legend is not None:
            self._bad_hole_legend.remove()
        self._bad_hole_legend = self.fig.legend(
            handles=[
                Patch(
                    facecolor="#d73027",
                    edgecolor="none",
                    alpha=0.18,
                    label="BADHOLE",
                )
            ],
            loc="lower right",
            bbox_to_anchor=(0.995, 0.006),
            fontsize=8.4,
            frameon=False,
        )

    def _draw_formation_tops(self, top: float, bottom: float) -> None:
        self.suppressed_formation_labels = []
        if not self.formation_tops or not self.axes:
            return
        lo, hi = sorted((top, bottom))
        grouped: Dict[float, List[str]] = {}
        for formation in self.formation_tops:
            if lo <= formation.depth <= hi:
                grouped.setdefault(formation.depth, []).append(formation.name)
        span = max(abs(hi - lo), 1.0)
        next_label_depth = -np.inf
        label_ax = self.axes[0]
        for depth, names in grouped.items():
            for ax in self.axes:
                ax.axhline(depth, linewidth=0.7, color="black", alpha=0.85, zorder=7)
            if depth < next_label_depth:
                self.suppressed_formation_labels.extend(
                    {"name": name, "depth": depth, "reason": "label spacing"} for name in names
                )
                continue
            label_ax.annotate(
                "\n".join(names), xy=(0.02, depth), xycoords=("axes fraction", "data"),
                xytext=(2, -2), textcoords="offset points", fontsize=8.0,
                fontweight="bold", color="black", ha="left", va="top",
                bbox={"boxstyle": "square,pad=0.12", "facecolor": "white",
                      "edgecolor": "none", "alpha": 0.72}, zorder=8,
            )
            next_label_depth = depth + span * 0.035 * len(names)

    @property
    def diagnostics(self) -> Dict[str, Any]:
        """Structured import/plot diagnostics for notebooks and automation."""
        return {
            "warnings": list(self.warnings),
            "formations": self.formation_import_report,
            "lithology_source": self.lithology_source,
            "lithology": (
                self.lithology_intervals.attrs.get("import_report", {})
                if self.lithology_intervals is not None else {}
            ),
            "suppressed_formation_labels": list(self.suppressed_formation_labels),
            "omitted_tracks": list(self.omitted_tracks),
            "optional_tracks": list(self.optional_tracks),
            "curve_lookup": dict(self.curve_lookup),
        }

    def _first_available_curve(self, curve_names: Sequence[str]):
        for curve_name in curve_names:
            resolved = _resolve_curve_name_for_well(self.well, curve_name)
            if resolved is None:
                continue
            curve = self.well.get(resolved)
            if curve is not None:
                self.curve_lookup[str(curve_name).upper()] = curve.mnemonic
                return curve
        return None

    def _get_curve(self, spec: LogCurveSpec):
        resolved = _resolve_curve_name_for_well(self.well, spec.curve_name, spec.candidates)
        if resolved is None:
            return None
        curve = self.well.get(resolved)
        if curve is not None:
            self.curve_lookup[spec.curve_name.upper()] = curve.mnemonic
        return curve

    def _warn_missing(self, curve_name: str) -> None:
        message = f"Curve '{curve_name}' not available for log viewer template"
        if self.strict:
            raise ValueError(message)
        if message not in self.warnings:
            self.warnings.append(message)

    def _normalize(self, values: np.ndarray, left: float, right: float, scale: str) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if math.isclose(float(left), float(right)):
            return np.full(values.shape, np.nan, dtype=float)
        if str(scale).lower() == "log":
            out = np.full(values.shape, np.nan, dtype=float)
            if left <= 0 or right <= 0:
                return out
            finite = np.isfinite(values) & (values > 0)
            left_log = math.log10(left)
            right_log = math.log10(right)
            out[finite] = (np.log10(values[finite]) - left_log) / (right_log - left_log)
            return out
        return (values - left) / (right - left)

    def _normalize_scalar(self, value: float, left: float, right: float, scale: str) -> float:
        arr = self._normalize(np.asarray([value], dtype=float), left, right, scale)
        return float(arr[0]) if np.isfinite(arr[0]) else np.nan

    def _denormalize_scalar(self, x: float, left: float, right: float, scale: str) -> float:
        """Inverse of :meth:`_normalize_scalar`: data value drawn at display ``x``."""
        if str(scale).lower() == "log":
            if left <= 0 or right <= 0:
                return np.nan
            left_log, right_log = math.log10(left), math.log10(right)
            return float(10.0 ** (left_log + x * (right_log - left_log)))
        return float(left + x * (right - left))

    def _depth_ticks(self, top: float, bottom: float, spacing: float) -> Iterable[float]:
        if spacing <= 0:
            return []
        first = math.ceil(min(top, bottom) / spacing) * spacing
        last = math.floor(max(top, bottom) / spacing) * spacing
        if last < first:
            return []
        return np.arange(first, last + spacing * 0.5, spacing)

    def _nice_depth_spacing(self, top: float, bottom: float, target: int = 12) -> float:
        span = abs(bottom - top)
        if not np.isfinite(span) or span <= 0:
            return 10.0
        raw = span / max(target, 1)
        exponent = math.floor(math.log10(raw))
        fraction = raw / (10**exponent)
        if fraction <= 1:
            nice = 1
        elif fraction <= 2:
            nice = 2
        elif fraction <= 5:
            nice = 5
        else:
            nice = 10
        return nice * (10**exponent)

    def _coerce_lithology_mapping(
        self,
        mapping: Optional[Mapping[Any, str]],
        mapping_df: Optional[Any],
    ) -> Dict[Any, str]:
        out: Dict[Any, str] = {}

        if mapping_df is not None:
            from petrophys.io.lithology_reader import read_lithology_codes
            for code, label in read_lithology_codes(mapping_df).items():
                _add_lithology_mapping_label(out, code, label)
        for code, label in (mapping or {}).items():
            _add_lithology_mapping_label(out, code, label)
        return out

    def _lithology_label(self, code: float, curve_name: Optional[str] = None) -> str:
        numeric = int(code) if float(code).is_integer() else code
        curve = self.well.get(curve_name) if curve_name else None
        if curve is not None:
            labels = curve.metadata.custom.get("category_labels", {})
            if numeric in labels or str(numeric) in labels:
                return labels.get(numeric, labels.get(str(numeric)))
        for key in (numeric, code, float(code)):
            if key in self.lithology_mapping:
                return self.lithology_mapping[key]

        upper_curve = str(curve_name or "").upper()
        use_geolink_defaults = (
            "GEOLINK" in upper_curve
            or upper_curve in {"LITHOLOGY", "ROCK_TYPE"}
            or upper_curve == getattr(self, "_code_curve", None)
        )
        if use_geolink_defaults:
            geolink_mapping = _default_geolink_lithology_mapping()
            for key in (numeric, code, float(code)):
                if key in geolink_mapping:
                    return geolink_mapping[key]

        default_maps: Tuple[Mapping[int, str], ...]
        if use_geolink_defaults:
            default_maps = (LITHOLOGY_CODE_LABELS, COMPUTED_LITHOLOGY_CODE_LABELS)
        else:
            default_maps = (COMPUTED_LITHOLOGY_CODE_LABELS, LITHOLOGY_CODE_LABELS)

        for default_map in default_maps:
            if numeric in default_map:
                return default_map[numeric]
        return "Unknown lithology"

    def _lithology_color(self, code: float, label: str, cmap: Any, index: int, total: int) -> Any:
        color = _lookup_lithology_style(label, LITHOLOGY_LABEL_COLORS, None)
        if color is not None:
            return color
        if total <= 1:
            return cmap(0.1)
        return cmap(index / max(total - 1, 1))

    def _lithology_hatch(self, label: str) -> str:
        return _lookup_lithology_style(label, LITHOLOGY_HATCH_PATTERNS, "")

    def _coerce_formation_tops(self, source: Optional[Any]) -> List[FormationTopSpec]:
        if source is None:
            return []
        from petrophys.io.geology_reader import read_formations

        table = read_formations(source, well=self.well)
        self.formation_import_report = dict(table.attrs.get("import_report", {}))
        # Only exact repeated picks are redundant; coincident named horizons survive.
        table = table.drop_duplicates(["formation_name", "top_depth"])
        return [
            FormationTopSpec(row.formation_name, float(row.top_depth))
            for row in table.itertuples()
        ]


@_styled_plot
def create_log_viewer(
    well: "Well",
    template_xml_path: Optional[str | Path] = None,
    template_defaults: str = "auto",
    depth_range: Optional[Tuple[float, float]] = None,
    top: Optional[float] = None,
    height: Optional[float] = None,
    strict: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    lithology_mapping: Optional[Mapping[Any, str]] = None,
    lithology_mapping_df: Optional[Any] = None,
    formation_tops: Optional[Any] = None,
    formation_tops_df: Optional[Any] = None,
    formation_name: Optional[str] = None,
    lithology_intervals: Optional[Any] = None,
    tracks: Optional[Sequence[str]] = None,
    interpreted: bool = False,
    curves: Optional[Sequence[str]] = None,
    curve_styles: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> LogViewer:
    """Create a template-driven well log viewer and immediately build it.

    ``interpreted`` marks an interpretation's well, whose values are drawn as
    the interpretation used them (see :class:`LogViewer`).
    """
    viewer = LogViewer(
        well,
        template_xml_path=template_xml_path,
        template_defaults=template_defaults,
        strict=strict,
        figsize=figsize,
        lithology_mapping=lithology_mapping,
        lithology_mapping_df=lithology_mapping_df,
        formation_tops=formation_tops,
        formation_tops_df=formation_tops_df,
        formation_name=formation_name,
        lithology_intervals=lithology_intervals,
        tracks=tracks,
        interpreted=interpreted,
        curves=curves,
        curve_styles=curve_styles,
    )
    viewer.create(depth_range=depth_range, top=top, height=height)
    return viewer
