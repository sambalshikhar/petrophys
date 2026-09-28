"""Versioned, bounded context exports and explicit legacy package compatibility."""

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


def _validate_options(schema_version, max_bytes):
    if schema_version not in {CONTEXT_SCHEMA_VERSION, "legacy"}:
        raise ValueError("schema_version must be '1.0' or 'legacy'")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")


def _encode(payload, max_bytes, *, legacy=False):
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    if legacy:
        text = text.replace("\n", os.linesep)
    data = text.encode("utf-8")
    if len(data) > max_bytes:
        raise ContextSizeError(
            f"Context requires {len(data)} UTF-8 bytes, exceeding max_bytes={max_bytes}. "
            "Select a smaller interval/category or explicitly increase the limit; nothing was truncated."
        )
    return data


def build_context(
    data: Any,
    *,
    schema_version: str = CONTEXT_SCHEMA_VERSION,
    max_bytes: int = DEFAULT_MAX_BYTES,
    category: Optional[str] = None,
    workflow_result: Any = None,
    source_file: Any = None,
    original_curves: Any = None,
    **legacy_options: Any,
) -> dict:
    """Build JSON-safe context without modifying or interpreting the input.

    Parameters
    ----------
    data : Well, DynamicWorkflowResult, or path
        Raw well, interpreted result, or a file accepted by ``petrophys.read``.
        Pass ``petrophys.interpret(well, ...)`` when calculated results are needed.
    schema_version : {"1.0", "legacy"}
        Versioned summaries by default; legacy uses the existing reference
        payload builder on a copy. No raw depth-by-depth arrays are included.
    max_bytes : int
        Maximum serialized UTF-8 bytes. Exceeding it raises ContextSizeError.
    category : str, optional
        One of the existing reference-analysis category names.
    workflow_result, source_file, original_curves
        Legacy provenance inputs. An interpreted legacy result requires its
        original curve names explicitly; they must not be guessed from outputs.
    **legacy_options
        Formation/segment options of the existing reference payload builder,
        accepted only for the legacy schema.
    """
    _validate_options(schema_version, max_bytes)
    if isinstance(data, (str, Path)):
        from petrophys.api import read
        source_file = source_file or data
        data = read(data)
    result = workflow_result if workflow_result is not None else (data if hasattr(data, "well") else None)
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
        raise ValueError(f"Unknown category {category!r}; choose from {sorted(reference.CATEGORY_CONTEXT_SPECS)}")
    if schema_version == "legacy":
        if category is not None:
            raise ValueError("Legacy category packages use export_context(layout='category_formation_lithology')")
        if original_curves is None and result is not None:
            raise ValueError("Legacy interpreted context requires original_curves from before interpretation")
        copied_well, copied_result = deepcopy((well, result))
        if copied_result is None:
            from petrophys.workflow import DynamicWorkflowConfig, DynamicWorkflowResult
            copied_result = DynamicWorkflowResult(copied_well, DynamicWorkflowConfig())
        if not hasattr(copied_result, "base_result"):
            from petrophys.workflow.zoned_batch import ZonedWorkflowRunResult
            copied_result = ZonedWorkflowRunResult(copied_well, copied_result)
        payload = reference._build_reference_json_payload(
            copied_well, copied_result, Path(source_file or f"{well.name}.las"),
            set(well.curves if original_curves is None else original_curves), **legacy_options,
        )
        payload = reference._json_safe(payload)
    else:
        if legacy_options:
            raise TypeError(f"Legacy-only context options: {sorted(legacy_options)}")
        interpretation = None
        if result is not None:
            interpretation = (result.summary() if hasattr(result, "summary") else {
                "base": result.base_result.summary(),
                "zones": [run.summary() for run in result.zone_results],
                "warnings": list(result.warnings),
            })
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
            payload["analysis"] = reference._category_analysis_result(category, deepcopy(well)).to_dict()
        payload = json_object(payload)
    _encode(payload, max_bytes, legacy=schema_version == "legacy")
    return payload


def _write_bytes(path, data, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}; pass overwrite=True to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        with path.open("xb") as handle:
            handle.write(data)
        return
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=".petrophys-", delete=False) as handle:
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
    schema_version: str = CONTEXT_SCHEMA_VERSION,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    overwrite: bool = False,
    **options: Any,
):
    """Write context JSON, or an existing formation/lithology context package.

    ``layout='well'`` returns the written Path and accepts build_context options.
    ``'formation_lithology'`` and ``'category_formation_lithology'`` take a LAS
    directory and legacy exporter options (including a required ``tops_file``),
    returning the existing list of export results. Package output must be a new
    directory; failed or oversized packages are never published. ``max_bytes``
    limits each JSON file, and ``max_total_bytes`` limits the entire package.
    ``schema_version='legacy'`` preserves legacy payloads and serialization;
    versioned packages wrap each payload with explicit schema metadata.
    """
    _validate_options(schema_version, max_bytes)
    output = Path(output_path)
    if layout == "well":
        payload = build_context(data, schema_version=schema_version, max_bytes=max_bytes, **options)
        _write_bytes(output, _encode(payload, max_bytes, legacy=schema_version == "legacy"), overwrite)
        return output
    if layout not in {"formation_lithology", "category_formation_lithology"}:
        raise ValueError("Unknown context layout; choose well, formation_lithology or category_formation_lithology")
    if overwrite or output.exists():
        raise FileExistsError("Package exports require a new output directory; existing packages are never overwritten")
    _validate_options(schema_version, max_total_bytes)
    source = Path(data)
    if not source.is_dir():
        raise FileNotFoundError(f"Input LAS directory not found: {source}")
    if options.get("tops_file") is None:
        raise ValueError("Package export requires tops_file; provide a CSV/TSV/Excel table using the input templates")
    from petrophys.reporting import reference_export as reference
    exporter = (reference.export_reference_json_work_package_by_formation_lithology
                if layout == "formation_lithology"
                else reference.export_reference_json_work_package_by_category_formation_lithology)
    if schema_version != "legacy" and options.get("workflow_config") is None:
        from petrophys.workflow import DynamicWorkflowConfig
        options["workflow_config"] = DynamicWorkflowConfig()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".petrophys-context-", dir=output.parent) as directory:
        staged = Path(directory) / "package"
        results = exporter(input_dir=source, output_dir=staged, **options)
        if not results:
            raise ValueError("No context files generated; check well selection, formations, and minimum thickness")
        failures = [item for item in results if not item.success]
        if failures:
            raise ValueError("Context export failed: " + "; ".join(error for item in failures for error in item.errors))
        reference.write_reference_curve_name_mapping(staged)
        total = 0
        for path in sorted(staged.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                relocated = payload.get("schema") == "petrophys_lithology_group_trace"
                if relocated:
                    payload["context_json_file"] = str(output / Path(payload["context_json_file"]).relative_to(staged))
                if schema_version != "legacy":
                    wrapped = {"schema_version": CONTEXT_SCHEMA_VERSION, "petrophys_version": __version__,
                               "context_kind": layout, "payload": payload}
                    path.write_bytes(_encode(json_object(wrapped), max_bytes))
                elif relocated:
                    path.write_bytes(_encode(payload, max_bytes, legacy=True))
                if path.stat().st_size > max_bytes:
                    raise ContextSizeError(f"{path.name} exceeds max_bytes={max_bytes}; no package was published")
            total += path.stat().st_size
            if total > max_total_bytes:
                raise ContextSizeError(f"Package exceeds max_total_bytes={max_total_bytes}; no package was published")
        for item in results:
            for field in ("output_file", "json_file"):
                value = getattr(item, field)
                if value:
                    setattr(item, field, str(output / Path(value).relative_to(staged)))
        staged.rename(output)
    return results
