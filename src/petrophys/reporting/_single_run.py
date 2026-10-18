"""Preparatory step of the context profiles' single interpretation.

The GeoLink, COSTA and Danish context packages were interpreted from inputs a
first interpretation of the raw LAS prepared: the raw curves standardized by it
and stored at the precision of the delivered LAS (five decimals), without the
curves its standardization sets aside as duplicates, and some curves it
computed (the reservoir flag that selects the SWIRR_APP samples, for COSTA
also its precomputed curves). The profiles compute those inputs in memory, as
a preparatory step of one interpretation of each raw LAS, and record it in the
trace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

def _historical_standardization(function):
    """Run ``function`` with the benchmark packages' duplicate-curve handling."""
    import functools

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        from petrophys._internal.benchmark import historical_benchmark

        with historical_benchmark():
            return function(*args, **kwargs)

    return wrapper


#: Core-corrected curves the workflow clears from its input and writes again
#: when the input carries them from an earlier interpretation (not as raw logs).
REWRITTEN_CORE_CURVES = ("PHIE_CC", "PERM_CC")


def prepare_from_first_run(
    base,
    first_run,
    *,
    export_options: Mapping[str, Any],
    reader_config,
    used_curves: Sequence[tuple[str, str, str]],
    calculation_steps: Optional[Sequence[Mapping[str, Any]]] = None,
) -> dict:
    """Prepare ``base`` (the well to interpret, modified in place) from ``first_run``.

    Parameters
    ----------
    base : Well
        The well as the context reads it.
    first_run : Well
        The first interpretation of the same raw LAS.
    export_options : mapping
        ``export_las`` options the package stored the first interpretation with.
    reader_config : DynamicWorkflowConfig
        The configuration the context reads its input with.
    used_curves : sequence of (name, method, description)
        The first interpretation's curves the context's interpretation uses.
    calculation_steps : sequence of mapping, optional
        Trace entries of the first interpretation's steps, recorded with the
        first used curve.

    Returns
    -------
    dict
        ``preparatory_steps`` (trace entries) and ``after_workflow`` (orders
        the interpreted well's curves as the package's LAS: the input order,
        then the curves the workflow added or rewrote, in the order it wrote
        them).
    """
    import tempfile

    import numpy as np

    from petrophys.io import export_las
    from petrophys.reporting.reference_export import _read_context_input_well

    with tempfile.TemporaryDirectory(prefix="petrophys-context-") as directory:
        stored = Path(directory) / "input.las"
        export_las(first_run, stored, **dict(export_options))
        inputs, _ = _read_context_input_well(stored, workflow_config=reader_config)
    if not np.array_equal(np.asarray(inputs.depth, dtype=float), np.asarray(base.depth, dtype=float)):
        raise ValueError("standardized input depths differ from the well's depths")

    raw_curves = set(base.curves)
    rounded: dict[str, int] = {}
    set_aside: list[str] = []
    for name in list(base.curves):
        curve = inputs.get(name)
        if curve is None:
            base.remove_curve(name)
            set_aside.append(name)
            continue
        old = np.asarray(base.get(name).data, dtype=float)
        new = np.asarray(curve.data, dtype=float)
        changed = int(np.sum(~((old == new) | (np.isnan(old) & np.isnan(new)))))
        if changed:
            base.get(name)._data = new.copy()
            rounded[name] = changed
    entries: list[dict[str, Any]] = [
        {
            "step": "input_standardization",
            "method": "curve standardization and storage at the delivered LAS precision (5 decimals)",
            "description": "Input curves standardized as for the preparatory interpretation and stored at "
            "five decimals; curves standardization sets aside as duplicates are not used.",
            "samples_rounded": rounded,
            "curves_set_aside": set_aside,
        }
    ]
    for name, method, description in used_curves:
        curve = inputs.get(name)
        if curve is None:
            continue
        data = np.asarray(curve.data, dtype=float)
        if base.get(name) is not None:
            base.get(name)._data = data.copy()
        else:
            base.add_curve(name, data.copy(), curve.unit, getattr(curve.metadata, "description", "") or "")
        entry: dict[str, Any] = {"step": name.lower(), "output_curves": [name], "method": method, "description": description}
        if np.isin(data[np.isfinite(data)], (0.0, 1.0)).all():
            entry["samples_flagged"] = int(np.nansum(data > 0.5))
        else:
            entry["finite_samples"] = int(np.isfinite(data).sum())
        entries.append(entry)
    if calculation_steps and len(entries) > 1:
        entries[1]["calculation_steps"] = list(calculation_steps)
    order = list(inputs.curves)

    def package_curve_order(interpreted) -> None:
        rewritten = {name for name in REWRITTEN_CORE_CURVES if name in order and name not in raw_curves}
        head = [name for name in order if name in interpreted.curves and name not in rewritten]
        tail = [name for name in interpreted.curves if name not in head]
        interpreted._curves = {name: interpreted._curves[name] for name in head + tail}

    return {"preparatory_steps": entries, "after_workflow": package_curve_order}


#: The reservoir flag and BADHOLEFLAG of the first interpretation, as used by the context.
RESERVOIR_FLAGS = (
    (
        "RESFLAG",
        "net reservoir from VCL, PHIE and SW cutoffs (NetPayModule)",
        "Reservoir flag that selects the SWIRR_APP samples, from VCL, PHIE and SW computed "
        "with the parameter set listed in calculation_steps.",
    ),
    (
        "BADHOLEFLAG",
        "bad-hole flag from caliper and bit size",
        "Bad-hole flag, the same flag as BAD_HOLE under its second mnemonic.",
    ),
)
