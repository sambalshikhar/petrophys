"""Compact, size-bounded JSON summaries of wells and results for LLMs and other tools."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, Optional

from petrophys._internal.serialization import json_object
from petrophys.model.well import Well
from petrophys.version import __version__


CONTEXT_SCHEMA_VERSION = "1.0"
DEFAULT_MAX_BYTES = 1_000_000
DEFAULT_MAX_TOTAL_BYTES = 50_000_000


class ContextSizeError(ValueError):
    """A context exceeds its explicitly configured UTF-8 byte budget."""


def _validate_limit(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _encode(payload: Any, max_bytes: int) -> bytes:
    _validate_limit(max_bytes, "max_bytes")
    data = json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if len(data) > max_bytes:
        raise ContextSizeError(
            f"Context requires {len(data)} UTF-8 bytes, exceeding "
            f"max_bytes={max_bytes}. Select a smaller interval/category or "
            "explicitly increase the limit; nothing was truncated."
        )
    return data


def build_context(
    data: Any,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    category: Optional[str] = None,
) -> dict:
    """Build a deterministic JSON-safe summary of a well or workflow result.

    This serializer does not invent inputs or rerun an interpretation. Pass an
    interpretation result when calculated curves and workflow provenance are
    required. The command-line generator always interprets its well inputs.
    """
    _validate_limit(max_bytes, "max_bytes")
    if isinstance(data, (str, Path)):
        from petrophys.api import read

        data = read(data)

    result = data if hasattr(data, "well") else None
    well = result.well if result is not None else data
    if not isinstance(well, Well):
        raise TypeError("build_context expects a Well, interpretation result, or well-file path")
    if result is not None and hasattr(result, "raise_for_errors"):
        result.raise_for_errors()
    if result is not None and hasattr(result, "base_result"):
        for run in [result.base_result, *result.zone_results]:
            run.raise_for_errors()

    from petrophys.reporting import reference_export as reference

    if category is not None and category not in reference.CATEGORY_CONTEXT_SPECS:
        raise ValueError(
            f"Unknown category {category!r}; choose from "
            f"{sorted(reference.CATEGORY_CONTEXT_SPECS)}"
        )

    interpretation = None
    if result is not None:
        interpretation = (
            result.summary()
            if hasattr(result, "summary")
            else {
                "base": result.base_result.summary(),
                "zones": [run.summary() for run in result.zone_results],
                "warnings": list(result.warnings),
            }
        )
    payload = {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "petrophys_version": __version__,
        "context_kind": "well_summary",
        "well": well.describe(),
        "interpretation": interpretation,
        "geology": well.get_parameter("_geology_context") or {},
        "include_raw_traces_in_context": False,
    }
    if category is not None:
        payload["category"] = category
        payload["analysis"] = reference._category_analysis_result(
            category,
            deepcopy(well),
            result,
        ).to_dict()
    payload = json_object(payload)
    _encode(payload, max_bytes)
    return payload


def _write_bytes(path: Path, data: bytes, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}; pass overwrite=True "
            "(--overwrite on the command line) to replace it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        with path.open("xb") as handle:
            handle.write(data)
        return
    temporary = None
    try:
        with NamedTemporaryFile(
            dir=path.parent,
            prefix=".petrophys-",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def export_context(
    data: Any,
    output_path: Any,
    *,
    layout: str = "well",
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    overwrite: bool = False,
    **options: Any,
):
    """Write a well summary or a fully computed context package.

    Package layouts read well files, apply optional formation and lithology
    inputs, run the workflow, and write context JSON. They never read an older
    context package or reuse archived numerical values. With a
    ``workflow_config`` under the benchmark calculation profile, every well
    gets the calculations of the eval package's re-run, which wrote its context.
    The default pattern is "*.las"; use pattern="*.dlis" for DLIS and
    reader_options={"step": 0.1524} to request a DLIS sampling step in metres.
    """
    _validate_limit(max_bytes, "max_bytes")
    _validate_limit(max_total_bytes, "max_total_bytes")
    output = Path(output_path)
    if layout == "well":
        payload = build_context(data, max_bytes=max_bytes, **options)
        _write_bytes(output, _encode(payload, max_bytes), overwrite)
        return output

    layouts = {"formation_lithology", "category_formation_lithology"}
    if layout not in layouts:
        raise ValueError(
            "Unknown context layout; choose well, formation_lithology or "
            "category_formation_lithology"
        )
    if overwrite or output.exists():
        raise FileExistsError(
            "Package exports require a new output directory; existing "
            "packages are never overwritten"
        )

    source = Path(data)
    if not source.is_dir():
        raise FileNotFoundError(f"Input well directory not found: {source}")

    from petrophys.reporting import reference_export as reference

    exporter = (
        reference.export_reference_json_work_package_by_formation_lithology
        if layout == "formation_lithology"
        else reference.export_reference_json_work_package_by_category_formation_lithology
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".petrophys-context-", dir=output.parent) as directory:
        staged = Path(directory) / "package"
        results = exporter(input_dir=source, output_dir=staged, **options)
        if not results:
            raise ValueError(
                "No context files generated; check well selection, formations, "
                "and minimum thickness"
            )
        failures = [item for item in results if not item.success]
        if failures:
            errors = [error for item in failures for error in item.errors]
            raise ValueError("Context export failed: " + "; ".join(errors))

        reference.write_reference_curve_name_mapping(staged)
        total = 0
        for path in sorted(staged.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("schema") == "petrophys_lithology_group_trace":
                    payload["context_json_file"] = str(
                        output
                        / Path(payload["context_json_file"]).relative_to(staged)
                    )
                    path.write_bytes(_encode(json_object(payload), max_bytes))
                elif path.stat().st_size > max_bytes:
                    raise ContextSizeError(
                        f"{path.name} exceeds max_bytes={max_bytes}; "
                        "no package was published"
                    )
            total += path.stat().st_size
            if total > max_total_bytes:
                raise ContextSizeError(
                    f"Package exceeds max_total_bytes={max_total_bytes}; "
                    "no package was published"
                )

        for item in results:
            for field in ("output_file", "json_file"):
                value = getattr(item, field)
                if value:
                    setattr(item, field, str(output / Path(value).relative_to(staged)))
        staged.rename(output)
    return results
