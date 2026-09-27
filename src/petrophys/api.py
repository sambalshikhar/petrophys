"""Short entry points for reading, plotting, and interpreting a single well."""

from copy import deepcopy
from pathlib import Path


def read(path, *, formations=None, lithology=None, formation_options=None,
         lithology_options=None, depth_reference=None, datum=None, **reader_options):
    """Read LAS, DLIS, CSV, or TSV and optionally attach validated geology.

    Tabular logs require an explicit depth_unit. Geology uses the well's
    depth units when not declared. Reader-specific options are forwarded.
    This function does not calculate interpretation curves.
    """
    from petrophys.io import read_csv, read_dlis, read_formations, read_las
    from petrophys.io import read_lithology_intervals

    extension = Path(path).suffix.lower()
    if extension == ".las":
        well = read_las(path, **reader_options)
    elif extension == ".dlis":
        well = read_dlis(path, **reader_options)
    elif extension in {".csv", ".tsv"}:
        if "depth_unit" not in reader_options:
            raise ValueError("CSV/TSV logs require depth_unit='m' or depth_unit='ft'.")
        if extension == ".tsv":
            reader_options.setdefault("sep", "\t")
        well = read_csv(path, **reader_options)
    else:
        raise ValueError(f"Unsupported well format {extension!r}; use LAS, DLIS, CSV, or TSV.")

    for key, value in (("depth_reference", depth_reference), ("datum", datum)):
        if value is not None:
            well.set_parameter(key, value)
    context = {}
    for name, source, reader, options in (
        ("formations", formations, read_formations, formation_options),
        ("lithology", lithology, read_lithology_intervals, lithology_options),
    ):
        if source is not None:
            table = reader(source, well=well, **(options or {}))
            # Persist normalized metadata with records, so later reads cannot
            # accidentally reinterpret converted depths in their source units.
            for key in ("depth_unit", "depth_reference", "datum"):
                if table.attrs.get(key) is not None:
                    table[key] = table.attrs[key]
            context[name] = table.to_dict("records")
            context[name + "_import_report"] = table.attrs.get("import_report", {})
    if context:
        well.set_parameter("_geology_context", context)
    return well


def plot(well, **options):
    """Build a log viewer for a Well or interpretation result."""
    from petrophys.visualization import create_log_viewer
    return create_log_viewer(getattr(well, "well", well), **options)


def interpret(well, config=None, *, inplace=False, **options):
    """Interpret a Well or path and return a diagnostic-rich workflow result.

    By default the source well is copied. Geological context is optional.
    Without a config, matrix and lithology are resolved from available data.
    Scientific assumptions remain those of DynamicWorkflowConfig; review
    result.config and result.to_trace_dict() before using calculated results.
    """
    from petrophys.workflow.dynamic_workflow import (
        DynamicWorkflowConfig, run_dynamic_workflow,
    )

    if isinstance(well, (str, Path)):
        well = read(well)
    elif not inplace:
        well = deepcopy(well)
    if config is None:
        defaults = {"matrix_type": "auto", "lithology": "auto"}
        defaults.update(options)
        config = DynamicWorkflowConfig(**defaults)
        options = {}
    else:
        config = deepcopy(config)
    return run_dynamic_workflow(well, config=config, **options)
