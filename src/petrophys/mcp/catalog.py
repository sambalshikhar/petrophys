"""What the MCP server can offer: sample wells, settings and workflow step kinds.

Compact, model-oriented listings built from the library's own registries
(:mod:`petrophys.datasets`, :func:`petrophys.workflow.config_schema` and the
built-in steps of :mod:`petrophys.workflow.steps`).
"""

from __future__ import annotations

import difflib
import inspect
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Short description of each sample collection (folder below petrophys/datasets).
COLLECTIONS: Dict[str, str] = {
    "geolink": "North Sea clastic wells, depth in m (GeoLink, CC BY-SA 4.0)",
    "costa": (
        "COSTA carbonate field, depth in ft; each file covers only the KHARAIB-2 "
        "reservoir interval; header RW/RWT and Archie A given (M/N in 13 of the 17 "
        "wells); interpret with "
        "lithology_domain='carbonate' (carbonate_calibration='costa' applies the "
        "field's GR-illite relation, and a 1.1 g/cm3 brine to porosity and mineral "
        "volumes unless rho_fl is set)"
    ),
    "taranaki": (
        "Taranaki Basin, New Zealand: onshore Cheal-1 (vertical) and Tuhua-7 "
        "(deviated, Tikorangi limestone), depth in m, bundled formation intervals"
    ),
    "teapot": (
        "Teapot Dome, Wyoming: two 1970s Schlumberger logs digitized to LAS, depth in ft, "
        "with mud, temperature and bit-size values in the header"
    ),
    "kgs": "Kansas legacy LAS (Morgan No. 1-17), depth in ft",
    "daydream": "Daydream 1 (Australia) processed CMR/NMR summaries, depth in m",
    "formation_pressure": (
        "MDT data, not log suites: pressure stations (35/11-13, "
        "35/11-10; depth-indexed) and raw pretest records (IODP C0009A DLIS "
        "and LAS copies; time-indexed, read with read_mdt_pretest)"
    ),
    "epslog": "EDE-01 core-scanner data (XRF, grain size, strength, probe permeability) and matching composite well logs",
}

#: Notes on individual sample wells.
WELL_NOTES: Dict[str, str] = {
    "30_6-5": (
        "flagship: full log suite, core data, bundled formation tops and lithology; "
        "Brent Group (Ness and Etive formations) 2841-2958 m"
    ),
    "25_5-1": "spectral gamma (THOR, URAN) and core data",
    "15_9-14": "vendor lithology; mineral-solver example",
    "16_7-1": "Zechstein halite section beyond the resistivity tool range; no neutron or PEF",
    "1046123714": "bundled formation tops",
}


def sample_wells() -> List[str]:
    """Names of bundled samples that are well files (LAS or DLIS)."""
    from petrophys.datasets import list_samples, sample

    return [
        name for name in list_samples()
        if Path(sample(name)).suffix.lower() in {".las", ".dlis"}
    ]


def normalized_name(name: str) -> str:
    """``name`` in lower case without separators: '30/6-5' and '30_6-5' give '3065'."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


_key = normalized_name


def resolve_sample(name: str) -> str:
    """Return the bundled sample well matching ``name``.

    Matching ignores case, separators and the file extension, so
    ``"30/6-5"``, ``"30_6-5.las"`` and ``"hw3"`` all resolve.

    Raises
    ------
    ValueError
        If nothing (or more than one sample) matches; the message lists
        the closest names.
    """
    wells = sample_wells()
    text = str(name).strip()
    for suffix in (".las", ".dlis"):
        if text.lower().endswith(suffix):
            text = text[: -len(suffix)]
    exact = [well for well in wells if well.lower() == text.lower()]
    if exact:
        return exact[0]
    key = _key(text)
    matches = [well for well in wells if _key(well) == key]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Sample name {name!r} is ambiguous: {', '.join(matches)}.")
    keys = {_key(well): well for well in wells}
    close = [keys[item] for item in difflib.get_close_matches(key, list(keys), n=4, cutoff=0.6)]
    hint = f" Closest: {', '.join(close)}." if close else ""
    raise ValueError(
        f"No bundled sample well named {name!r}.{hint} Sample wells: {', '.join(wells)}."
    )


def sample_geology(name: str) -> Dict[str, Path]:
    """Bundled formation and lithology files for sample well ``name``."""
    from petrophys.datasets import list_samples, sample

    available = set(list_samples())
    collection = Path(sample(name)).parent.name
    found: Dict[str, Path] = {}
    for option, candidates in (
        ("formations", (f"formations_{name}", f"{name}_formation_tops",
                        f"formations_{collection}")),
        ("lithology", (f"lithology_{name}",)),
    ):
        for candidate in candidates:
            if candidate in available:
                found[option] = Path(sample(candidate))
                break
    return found


def sample_catalog() -> Dict[str, Any]:
    """Sample wells grouped by collection, with notes and the input templates."""
    from petrophys.datasets import list_samples, sample

    wells = sample_wells()
    groups: Dict[str, Dict[str, Any]] = {}
    pressure: List[str] = []
    for name in wells:
        collection = Path(sample(name)).parent.name
        if collection == "formation_pressure":
            # Pressure stations and time-indexed pretests, not log suites.
            pressure.append(name)
            continue
        group = groups.setdefault(
            collection, {"about": COLLECTIONS.get(collection, ""), "wells": []}
        )
        group["wells"].append(name)
    for group in groups.values():
        group["wells"] = sorted(group["wells"], key=_natural)
    tables = [name for name in list_samples() if name not in wells]
    return {
        "collections": groups,
        "pressure_data": {
            "about": COLLECTIONS["formation_pressure"]
            + "; not for interpret_well (read pretests with read_mdt_pretest)",
            "files": sorted(pressure, key=_natural),
        },
        "well_notes": {name: note for name, note in WELL_NOTES.items() if name in wells},
        "other_files": tables,
        "hint": (
            "load_sample_well(name) loads a well and attaches its bundled formation tops "
            "and lithology. Names ignore case and separators: '30/6-5' and 'hw3' work."
        ),
    }


def _natural(text: str) -> Tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text))


# -- interpretation settings ----------------------------------------------------

#: Settings shown by default, in this order: the ones agents need most often.
COMMON_SETTINGS: Tuple[str, ...] = (
    "rw", "rw_meas_temp_f", "salinity_ppm", "bht_f", "td_ft", "surface_temp_f",
    "lithology_domain", "matrix_type", "rho_ma", "sw_model", "a", "m", "n",
    "vcl_cutoff", "phie_cutoff", "sw_cutoff", "gr_min", "gr_max", "gr_method",
    "apply_hc_correction", "hc_type", "perm_method", "rmf_at_meas_temp",
    "rmf_meas_temp_f", "skip_steps", "allow_default_rw",
)

#: Choices worth listing for settings whose accepted values include aliases.
_SHORT_CHOICES = {
    "matrix_type": ["auto", "sandstone", "limestone", "dolomite", "anhydrite", "halite",
                    "coal", "shale", "(aliases accepted)"],
}

#: Defaults resolved at run time, stated where the schema default is None.
_RUNTIME_DEFAULTS = {
    "standardize_input": "True (False under the 'geolink' calculation profile)",
    "vcl_cutoff": "0.40 clastic, 0.30 carbonate samples (0.3 under the 'geolink' profile)",
    "sw_cutoff": "0.60 clastic, 0.70 carbonate samples (0.7 under the 'geolink' profile)",
    "phie_cutoff": "0.08 clastic, 0.045 carbonate samples",
    "bht_f": "header BHT (converted to degF)",
    "td_ft": "header TD, else deepest sample",
    "lithology_domain": "LAS header value, else clastic",
    "matrix_type": "LAS header value, else sandstone in the clastic domain, auto in the others",
    "gr_min": "P5 of the GR log, lowered to 20 API when higher",
    "gr_max": "P95 of the GR log, raised to 100 API when lower",
    "rw": "header RW (at RWT, else at the header FMT, else the median formation temperature)",
    "gradient_f_per_100ft": "header TGRAD, used only without a BHT",
    "a": "by lithology", "m": "by lithology", "n": "by lithology",
    "rho_ma": "from matrix_type",
    "carbonate_calibration": "generic ('costa' under the 'geolink' profile)",
    "nd_clay": "picked in the well's shales ((2.62, 0.38) under the 'geolink' profile)",
}


def _entry(item: Dict[str, Any], *, full: bool) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"name": item["name"]}
    if full:
        entry["type"] = item["type"].replace("Optional[", "").rstrip("]") if item["type"] else ""
    default = item["default"]
    runtime = _RUNTIME_DEFAULTS.get(item["name"])
    entry["default"] = runtime if default is None and runtime else default
    description = item["description"]
    if not full:
        description = re.split(r"(?<=\S)\. (?=[A-Z])", description, maxsplit=1)[0].rstrip(".")
    entry["description"] = description
    choices = item.get("choices")
    if choices:
        short = _SHORT_CHOICES.get(item["name"], list(choices))
        entry["choices"] = list(choices) if full else short
    if full:
        entry["group"] = item["group"]
    return entry


def settings_listing(
    names: Optional[Sequence[str]] = None, group: Optional[str] = None
) -> Dict[str, Any]:
    """Compact listing of interpretation settings, or full detail for some.

    Raises
    ------
    ValueError
        For an unknown setting name or group (with the closest names).
    """
    from petrophys.workflow.schema import config_schema

    schema = config_schema()
    by_name = {item["name"]: item for item in schema}
    groups: Dict[str, List[str]] = {}
    for item in schema:
        groups.setdefault(_short_group(item["group"]), []).append(item["name"])

    if names:
        unknown = [name for name in names if name not in by_name]
        if unknown:
            hints = []
            for name in unknown:
                close = difflib.get_close_matches(name, list(by_name), n=2, cutoff=0.5)
                suggestion = f" (did you mean {', '.join(close)}?)" if close else ""
                hints.append(f"{name!r}{suggestion}")
            raise ValueError(
                f"Unknown setting(s): {', '.join(hints)}. Call interpretation_settings() "
                "without arguments for the list."
            )
        return {"settings": [_entry(by_name[name], full=True) for name in names]}
    if group:
        wanted = group.strip().lower()
        matches = [label for label in groups if label.lower().startswith(wanted)] or [
            label for label in groups if wanted in label.lower()
        ]
        if not matches:
            raise ValueError(
                f"Unknown settings group {group!r}. Groups: {', '.join(groups)}."
            )
        selected = [name for label in matches for name in groups[label]]
        return {"group": matches, "settings": [_entry(by_name[name], full=True)
                                               for name in selected]}
    return {
        "settings": [_entry(by_name[name], full=False) for name in COMMON_SETTINGS
                     if name in by_name],
        "all_groups": groups,
        "hint": (
            "These are the commonly used settings. For any other setting, or full "
            "accepted values, call interpretation_settings(names=[...]) or (group='...')."
        ),
    }


def _short_group(label: str) -> str:
    return re.sub(r"\s*\(.*\)$", "", label or "Other")


# -- custom workflow steps ------------------------------------------------------

def _param_docs(func: Any) -> Dict[str, str]:
    """First sentence of each parameter's description in a numpydoc docstring."""
    doc = inspect.getdoc(func) or ""
    lines = doc.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "Parameters")
    except StopIteration:
        return {}
    described: Dict[str, str] = {}
    current: List[str] = []
    for line in lines[start + 2:]:
        if line.strip() in {"Returns", "Raises", "Examples", "Notes", "See Also"}:
            break
        if line and not line.startswith(" "):
            current = [name.strip() for name in line.split(":")[0].split(",") if name.strip()]
            for name in current:
                described[name] = ""
        elif current and line.strip():
            for name in current:
                described[name] = (described[name] + " " + line.strip()).strip()
    cleaned = {}
    for name, text in described.items():
        text = re.sub(r"``([^`]+)``", r"\1", re.sub(r":\w+:`~?([^`]+)`", r"\1", text))
        cleaned[name] = re.split(r"(?<=[a-z0-9)\]])\.\s+(?=[A-Z])", text, maxsplit=1)[0].rstrip(".")
    return cleaned


def _step_choices() -> Dict[str, Sequence[str]]:
    from petrophys.interpretation.permeability.perm_module import PERMEABILITY_METHODS
    from petrophys.workflow import steps

    return {
        "clay_volume.method": steps._CLAY_METHODS,
        "clay_volume.nd_matrix": steps._ND_MATRICES,
        "porosity.method": steps._POROSITY_METHODS,
        "saturation.model": tuple(
            f"{model} (options: {', '.join(options)})" if options else model
            for model, options in steps._SW_MODEL_OPTIONS.items()
        ),
        "permeability.method": tuple(PERMEABILITY_METHODS),
    }


def step_kinds(kind: Optional[str] = None) -> Dict[str, Any]:
    """Built-in workflow step kinds with their parameters, defaults and outputs.

    Raises
    ------
    ValueError
        For an unknown kind (with the closest names).
    """
    from petrophys.workflow.steps import _REGISTRY

    if kind is not None and kind not in _REGISTRY:
        close = difflib.get_close_matches(kind, list(_REGISTRY), n=2, cutoff=0.5)
        raise ValueError(
            f"Unknown step kind {kind!r}"
            + (f" (did you mean {', '.join(close)}?)" if close else "")
            + f". Kinds: {', '.join(_REGISTRY)}."
        )
    kinds = {}
    for name, factory in _REGISTRY.items():
        if kind is not None and name != kind:
            continue
        signature = inspect.signature(factory)
        docs = _param_docs(factory) if kind is not None else {}
        params: Dict[str, Any] = {}
        for parameter in signature.parameters.values():
            if parameter.name == "name":
                continue
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                params["**settings"] = "any interpret_well setting"
                continue
            default = None if parameter.default is inspect.Parameter.empty else parameter.default
            if kind is None:
                params[parameter.name] = default
            else:
                params[parameter.name] = {"default": default,
                                          "description": docs.get(parameter.name, "")}
                choices = _step_choices().get(f"{name}.{parameter.name}")
                if choices:
                    params[parameter.name]["choices"] = list(choices)
        summary = (inspect.getdoc(factory) or "").splitlines()[0]
        summary = re.sub(r"``([^`]+)``", r"\1", re.sub(r":\w+:`~?([^`]+)`", r"\1", summary))
        kinds[name] = {"does": summary, "params": params}
    if kind is not None:
        info = kinds[kind]
        step = _REGISTRY[kind]
        try:
            # The water steps need a source; the header one shows their curves.
            example = {"formation_water": lambda: step(rw="header"),
                       "mud_filtrate": lambda: step(rmf="header")}.get(kind, step)()
        except (TypeError, ValueError):
            example = None
        if example is not None:
            info["requires"] = list(example.requires)
            info["produces"] = dict(example.produces)
    return {"kinds": kinds}
