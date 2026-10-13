"""Computed reproductions of the published COSTA and Danish context packages.

The GeoLink eval package is reproduced by ``calculation_profile="benchmark"``
(see :attr:`petrophys.workflow.DynamicWorkflowConfig.calculation_profile`).
The COSTA and Danish packages were made by their own pipelines: the COSTA
package by interpretation steps of its own around the workflow and a rewrite of
the context notes, the Danish package by a preparation of its raw LAS and one
context group per formation of its tops workbook. A context profile runs that
pipeline, from the delivery's LAS files to the context JSON, with the
calculations each package used; nothing is copied from a published package.

The reproduction of each package was checked against it: the interpreted LAS
files are identical byte for byte, and every number and note of the context
is the same except the ELECTROFACIES statistics, which come from k-means
clustering that was unseeded when the packages were made.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from petrophys.reporting._costa_context import COSTA_WELLS
from petrophys.reporting._danish_context import DANISH_WELLS


@dataclass(frozen=True)
class ContextProfile:
    """A published context package and the inputs its reproduction needs."""

    name: str
    aliases: tuple[str, ...]
    description: str
    well_names: tuple[str, ...]
    required_inputs: tuple[str, ...]


_PROFILES = {
    "costa_v1": ContextProfile(
        name="costa_v1",
        aliases=("costa", "costa-v1"),
        description=(
            "The COSTA six-category context package for the 17 COSTA wells, computed from the "
            "COSTA LAS files (by default the copies bundled in petrophys.datasets) with the "
            "bundled COSTA interpretation parameters and reference carbonate volumes."
        ),
        well_names=COSTA_WELLS,
        required_inputs=(),
    ),
    "danish_v1": ContextProfile(
        name="danish_v1",
        aliases=("danish", "danish-v1"),
        description=(
            "The Danish six-category context package for five Danish wells, computed from their "
            "raw LAS files, the Danish formation tops workbook and the GeoLink lithology codebook."
        ),
        well_names=DANISH_WELLS,
        required_inputs=("tops_file", "lithology_mapping_file"),
    ),
}


def _profile_key(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def get_context_profile(name: str) -> ContextProfile:
    """Return a context profile by name, accepting its short alias."""
    key = _profile_key(name)
    for canonical, profile in _PROFILES.items():
        accepted = {canonical, *(_profile_key(alias) for alias in profile.aliases)}
        if key in accepted:
            return profile
    choices = ", ".join(sorted(_PROFILES))
    raise ValueError(f"Unknown context profile {name!r}; choose from {choices}")


def list_context_profiles() -> tuple[str, ...]:
    """Return the canonical context profile names."""
    return tuple(sorted(_PROFILES))


def describe_context_profile(name: str) -> dict:
    """Return JSON-safe profile metadata."""
    profile = get_context_profile(name)
    return {
        "name": profile.name,
        "aliases": list(profile.aliases),
        "description": profile.description,
        "well_names": list(profile.well_names),
        "well_count": len(profile.well_names),
        "required_inputs": list(profile.required_inputs),
        "output_contract": "computed; matches the published package except ELECTROFACIES statistics",
    }


def _selector(value: str) -> str:
    return Path(str(value)).stem.strip().casefold().replace("_", "-")


def identify_context_profile(
    input_path: str | Path,
    *,
    well_names: Optional[Sequence[str]] = None,
    pattern: str = "*.las",
    recursive: bool = False,
) -> Optional[ContextProfile]:
    """The COSTA or Danish profile whose wells the input files (or ``well_names``) are, else None.

    ``input_path`` may be one LAS file or a directory; in a directory every
    selected file must belong to the same profile. None means the input is
    not one of those deliveries (for example, a GeoLink well).
    """
    source = Path(input_path)
    if well_names:
        stems = {_selector(name) for name in well_names}
    elif source.is_file():
        stems = {_selector(source.name)}
    elif source.is_dir():
        iterator = source.rglob(pattern) if recursive else source.glob(pattern)
        stems = {_selector(path.name) for path in iterator if path.is_file()}
    else:
        raise FileNotFoundError(f"Input LAS path not found: {source}")
    if not stems:
        return None
    matches = [
        profile for profile in _PROFILES.values()
        if stems <= {_selector(name) for name in profile.well_names}
    ]
    return matches[0] if len(matches) == 1 else None


def export_context_profile(
    input_dir: Optional[str | Path],
    output_dir: str | Path,
    *,
    context_profile: str,
    work_dir: str | Path,
    well_names: Optional[Sequence[str]] = None,
    pattern: str = "*.las",
    recursive: bool = False,
    tops_file: Optional[str | Path] = None,
    lithology_mapping_file: Optional[str | Path] = None,
):
    """Compute a published COSTA or Danish context package from its LAS files.

    Parameters
    ----------
    input_dir : str or Path or None
        The delivery's LAS files: the Danish raw LAS, or the COSTA LAS (None
        for the COSTA copies bundled with petrophys).
    output_dir : str or Path
        Folder for the category context JSON.
    context_profile : str
        ``"costa_v1"`` or ``"danish_v1"`` (or an alias, ``"costa"``, ``"danish"``).
    work_dir : str or Path
        Folder for the intermediate LAS: ``processed_las`` (interpreted LAS
        and workflow traces) and, for the Danish package, ``prepared_las``.
    well_names : sequence of str, optional
        Only these wells.
    tops_file, lithology_mapping_file : str or Path, optional
        Required by the Danish profile: its formation tops workbook and the
        GeoLink/Kingdom lithology codebook.

    Returns
    -------
    list of ExampleFormatExportResult
        One per context file.
    """
    profile = get_context_profile(context_profile)
    if recursive:
        raise ValueError("Context profiles read one folder of LAS files; recursive is not supported")
    if well_names:
        known = {_selector(name) for name in profile.well_names}
        unknown = sorted(name for name in well_names if _selector(name) not in known)
        if unknown:
            raise ValueError(f"Wells not in context profile {profile.name}: {', '.join(unknown)}")
    if profile.name == "costa_v1":
        from petrophys.reporting._costa_context import reproduce_costa_context

        return reproduce_costa_context(
            output_dir, work_dir=work_dir, input_dir=input_dir, well_names=well_names, pattern=pattern
        )

    from petrophys.reporting._danish_context import reproduce_danish_context

    missing = [
        name for name, value in (("tops_file", tops_file), ("lithology_mapping_file", lithology_mapping_file))
        if value is None
    ]
    if missing:
        raise ValueError(f"Context profile {profile.name} requires {', '.join(missing)}")
    if input_dir is None:
        raise ValueError("Context profile danish_v1 requires the folder of Danish raw LAS files")
    return reproduce_danish_context(
        input_dir,
        output_dir,
        tops_file=tops_file,
        lithology_mapping_file=lithology_mapping_file,
        work_dir=work_dir,
        well_names=well_names,
        pattern=pattern,
    )


__all__ = [
    "ContextProfile",
    "describe_context_profile",
    "export_context_profile",
    "get_context_profile",
    "identify_context_profile",
    "list_context_profiles",
]
