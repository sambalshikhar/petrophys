"""Dataset-specific benchmark context for GeoLink, the GeoLink training package, COSTA and Danish.

Each context profile computes its collection's JSON from raw LAS using the
configured preparation, interpretation, calibration and grouping rules.
GeoLink, the GeoLink training package and Danish require their tops workbook
and lithology codebook; COSTA uses bundled inputs and calibration data. See
docs/benchmark_context.md for required inputs and examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from petrophys.reporting._costa_context import COSTA_WELLS
from petrophys.reporting._danish_context import DANISH_WELLS
from petrophys.reporting._geolink_context import GEOLINK_WELLS
from petrophys.reporting._geolink_train_context import GEOLINK_TRAIN_WELLS


@dataclass(frozen=True)
class ContextProfile:
    """A published context package and the inputs its reproduction needs."""

    name: str
    aliases: tuple[str, ...]
    description: str
    well_names: tuple[str, ...]
    required_inputs: tuple[str, ...]


_PROFILES = {
    "geolink_v1": ContextProfile(
        name="geolink_v1",
        aliases=("geolink", "geolink-v1"),
        description=(
            "The GeoLink eval six-category context package for 32 Norwegian wells, computed from "
            "their raw LAS files, the NPD formation tops workbook and the GeoLink lithology codebook."
        ),
        well_names=GEOLINK_WELLS,
        required_inputs=("tops_file", "lithology_mapping_file"),
    ),
    "geolink_train_v1": ContextProfile(
        name="geolink_train_v1",
        aliases=("geolink_train", "geolink-train", "geolink-train-v1"),
        description=(
            "The GeoLink training six-category context package for 191 Norwegian wells, computed "
            "from their raw LAS files with header interpretation parameters "
            "(LAS_with_petrophys_params), the NPD formation tops workbook and the GeoLink "
            "lithology codebook."
        ),
        well_names=GEOLINK_TRAIN_WELLS,
        required_inputs=("tops_file", "lithology_mapping_file"),
    ),
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
        "output_contract": "computed from the delivery's raw LAS with the package's own pipeline",
    }


def _selector(value: str) -> str:
    text = str(value).strip()
    if text.lower().endswith(".las"):
        text = Path(text).stem
    return text.strip().casefold().replace("/", "-").replace("_", "-")


def identify_context_profile(
    input_path: str | Path,
    *,
    well_names: Optional[Sequence[str]] = None,
    pattern: str = "*.las",
    recursive: bool = False,
) -> Optional[ContextProfile]:
    """The context profile whose wells the input files (or ``well_names``) are, else None.

    ``input_path`` may be one LAS file or a directory; in a directory every
    selected file must belong to the same profile. None means the input is
    not one of the published deliveries.
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
    """Compute a published GeoLink, GeoLink training, COSTA or Danish context package from its LAS files.

    Parameters
    ----------
    input_dir : str or Path or None
        The delivery's raw LAS files: the GeoLink ``eval`` folder, the GeoLink
        training package's ``LAS_with_petrophys_params`` folder, the Danish
        raw LAS, or the COSTA LAS (None for the COSTA copies bundled with
        petrophys).
    output_dir : str or Path
        Folder for the category context JSON.
    context_profile : str
        ``"geolink_v1"``, ``"geolink_train_v1"``, ``"costa_v1"`` or
        ``"danish_v1"`` (aliases ``"geolink"``, ``"geolink_train"``,
        ``"costa"``, ``"danish"``).
    work_dir : str or Path
        Folder for ``processed_las``: the interpreted LAS the context is
        computed from, with workflow traces.
    well_names : sequence of str, optional
        Only these wells.
    pattern : str, default ``"*.las"``
        The delivery files to read; the packages were made from LAS files.
    tops_file, lithology_mapping_file : str or Path, optional
        Required by the GeoLink, GeoLink training and Danish profiles: the
        formation tops workbook (the NPD picks for GeoLink and the training
        package, the Danish tops workbook) and the GeoLink/Kingdom lithology
        codebook.

    Returns
    -------
    list of ExampleFormatExportResult
        One per context file.
    """
    profile = get_context_profile(context_profile)
    if recursive:
        raise ValueError("Context profiles read one folder of LAS files; recursive is not supported")
    if not str(pattern).lower().endswith(".las"):
        raise ValueError(
            f"Context profile {profile.name} reproduces a LAS delivery; pattern must select .las files, "
            f"got {pattern!r}"
        )
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

    missing = [
        name for name, value in (("tops_file", tops_file), ("lithology_mapping_file", lithology_mapping_file))
        if value is None
    ]
    if missing:
        raise ValueError(f"Context profile {profile.name} requires {', '.join(missing)}")
    if input_dir is None:
        raise ValueError(f"Context profile {profile.name} requires the folder of its raw LAS files")
    if profile.name == "geolink_train_v1":
        from petrophys.reporting._geolink_train_context import reproduce_geolink_train_context

        return reproduce_geolink_train_context(
            input_dir,
            output_dir,
            tops_file=tops_file,
            lithology_mapping_file=lithology_mapping_file,
            work_dir=work_dir,
            well_names=well_names,
            pattern=pattern,
        )
    if profile.name == "geolink_v1":
        from petrophys.reporting._geolink_context import reproduce_geolink_context

        return reproduce_geolink_context(
            input_dir,
            output_dir,
            tops_file=tops_file,
            lithology_mapping_file=lithology_mapping_file,
            work_dir=work_dir,
            well_names=well_names,
            pattern=pattern,
        )

    from petrophys.reporting._danish_context import reproduce_danish_context

    return reproduce_danish_context(
        input_dir,
        output_dir,
        tops_file=tops_file,
        lithology_mapping_file=lithology_mapping_file,
        work_dir=work_dir,
        well_names=well_names,
        pattern=pattern,
    )


def interpret_with_context_profile(
    file_path: str | Path,
    context_profile: str,
    *,
    tops_file: Optional[str | Path] = None,
):
    """Interpret one well of a context package as the package's context interprets it.

    The well's raw LAS goes through its profile's preparation and
    calculations, as :func:`export_context_profile` runs them, so the
    statistics of the returned curves over a context group's depths are that
    group's context statistics. Nothing is written.

    Parameters
    ----------
    file_path : str or Path
        The well's raw LAS file from the package's delivery (for COSTA, e.g.
        the bundled copy ``petrophys.sample("HW-10")``).
    context_profile : str
        ``"geolink"``, ``"geolink_train"``, ``"costa"`` or ``"danish"`` (or
        their ``_v1`` names). The well must belong to the profile's package.
    tops_file : str or Path, optional
        Required by the Danish profile: its tops workbook, which supplies the
        lithology of wells without a lithology log. The other profiles' tops
        only group the context.

    Returns
    -------
    DynamicWorkflowResult
        The interpreted well (``result.well``) after the context's value-range
        QC and dependent-curve recomputation, with the trace of the context's
        interpretation; ``result.context_profile`` names the profile,
        ``result.preparatory_steps`` records the profile's preparatory step
        and ``result.post_workflow_steps`` the QC and recomputation.
    """
    import tempfile

    from petrophys._internal.benchmark import historical_benchmark

    profile = get_context_profile(context_profile)
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"No LAS file {path}")
    if path.suffix.lower() != ".las":
        raise ValueError(f"Context profile {profile.name} interprets the package's LAS files, got {path.name}")
    if _selector(path.stem) not in {_selector(name) for name in profile.well_names}:
        raise ValueError(
            f"{path.stem} is not a well of context profile {profile.name}; its calculations are specific "
            "to the package's wells."
        )
    if profile.name == "danish_v1" and tops_file is None:
        raise ValueError("Context profile danish_v1 requires tops_file (the Danish tops workbook)")

    from petrophys.reporting.reference_export import context_checked_well

    captured: dict = {}

    def keep(_file_path, workflow_result, preparatory_steps):
        # The context's value-range QC and dependent-curve recomputation, as
        # every context group gets them before its statistics, under the
        # profile's own rules (applied while its calculations are active).
        checked, post_workflow_steps = context_checked_well(workflow_result)
        captured.update(run=workflow_result, well=checked, steps=preparatory_steps, post=post_workflow_steps)

    options = dict(
        well_names=[path.stem],
        pattern=path.name,
        processed_las_dir=None,
        processed_las_only=True,
        on_interpreted=keep,
    )
    with historical_benchmark(), tempfile.TemporaryDirectory(prefix="petrophys-profile-") as unused_output:
        if profile.name == "costa_v1":
            from petrophys.reporting._costa_context import (
                _export_costa,
                load_costa_parameters,
                load_costa_reference_volumes,
            )

            _export_costa(
                path.parent,
                unused_output,
                parameters=load_costa_parameters(),
                reference=load_costa_reference_volumes(),
                **options,
            )
        elif profile.name == "geolink_train_v1":
            from petrophys.reporting._geolink_train_context import _export_geolink_train

            _export_geolink_train(path.parent, unused_output, **options)
        elif profile.name == "geolink_v1":
            from petrophys.reporting._geolink_context import _export_geolink

            _export_geolink(path.parent, unused_output, **options)
        else:
            from petrophys.reporting._danish_context import _export_danish, read_danish_tops

            _export_danish(path.parent, unused_output, tops=read_danish_tops(tops_file), **options)

    run = captured["run"]
    result = run.base_result
    result.well = captured["well"]
    for warning in [*run.warnings, *(w for zone in run.zone_results for w in zone.warnings)]:
        if warning not in result.warnings:
            result.warnings.append(warning)
    result.context_profile = profile.name
    result.preparatory_steps = list(captured["steps"] or [])
    result.post_workflow_steps = captured["post"]
    return result


__all__ = [
    "ContextProfile",
    "describe_context_profile",
    "export_context_profile",
    "interpret_with_context_profile",
    "get_context_profile",
    "identify_context_profile",
    "list_context_profiles",
]
