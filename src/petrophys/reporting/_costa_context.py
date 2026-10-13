"""The COSTA eval context package, computed from the COSTA LAS files.

The package interpreted the 17 bundled COSTA wells (``petrophys/datasets/costa``,
the same files as the package's ``costa_las_raw`` folder) in three steps,
each reproduced here. Its two tables are bundled in ``petrophys/_internal/data``.

1. Curves from the well's own inputs before the workflow: PERM from the core
   permeability PERM_CC; VCL by the well's gamma-ray method
   (``costa_interpretation_parameters.csv``); calcite and dolomite volumes
   (VCALC, VDOLO), taken from ``costa_reference_carbonate_volumes.csv`` where
   that table has the depth and otherwise solved from VCL, the core porosity
   PHIE_CC and RHOB; and from them a two-group LITHOLOGY_GEOLINK (limestone 11,
   dolomite 15) and LITHO.
2. The workflow with the COSTA settings (:func:`costa_workflow_config`) and the
   COSTA calculations (:func:`petrophys.workflow.dynamic_workflow._dataset_calculations`):
   SW takes the source SW_INPUT where it has one, SO = 1 - SW. CROSSOVER_FLAG
   is added afterwards, and the flushed-zone curves, which COSTA has no
   shallow resistivity for, are dropped.
3. Context for one formation spanning each well, with the package's notes:
   the per-curve method notes of :func:`costa_method_notes` replace the
   workflow's, and context curves the package left out are removed.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

#: Wells of the COSTA package, by file stem.
COSTA_WELLS = (
    "HW-10", "HW-24", "HW-25", "HW-26", "HW-27", "HW-28", "HW-29", "HW-3", "HW-30",
    "HW-31", "HW-32", "HW-4", "HW-5", "HW-6", "HW-7", "HW-8", "HW-9",
)
FORMATION_NAME = "KHARAIB-2 FM. TOP"
_FLUID_DENSITY = 1.1
_SURFACE_TEMP_F = 68.0
_RW_MEASURED_TEMP_F = 68.0
_DEFAULT_RW_68F = 0.1
#: Lithology groups thinner than 2 m (in feet) are not exported.
_MIN_GROUP_THICKNESS_FT = 2.0 * 3.280839895
_SKIPPED_STEPS = ("vclay", "rcl", "rmf", "invasion", "core_correction", "permeability", "lithology", "hc_correction", "mhi")
#: Curves the package removed from the interpreted LAS: COSTA has no shallow resistivity.
_UNSUPPORTED_SATURATION_CURVES = ("SXO", "SWXO", "SWXOA", "MHI", "SW_INPUT", "SO_INPUT")
#: Curves the package removed from the context lists and statistics.
_DROP_CONTEXT_CURVES = {
    "SWIRR_APP", "SALMUDAPP", "SALAPP", "RWA", "RHOMAA", "VILL", "LITH_NPHI_MODEL",
    "LITH_RHOB_MODEL", "LITHOLOGY_GEOLINK", "SXO", "SWXO", "SW_INPUT", "SO_INPUT", "MHI",
}
_DROP_CONTEXT_NOTE_TOKENS = (
    "converted to salinity", "Rw75", "salinity", "apparent", "SALMUDAPP", "SALAPP", "SAL:", "RWA:",
    "SWIRR_APP", "LITHOLOGY_GEOLINK", "LITHOLOGY_GEOLINK:", "LITH_NPHI_MODEL:", "LITH_RHOB_MODEL:",
    "RHOMAA:", "VILL:", "RMF:", "mud",
)
_DROP_CONTEXT_NOTE_PREFIXES = (
    "Archie coefficients:", "BAD_HOLE:", "SXO:", "TEMP:", "Flags:", "CROSSOVER_FLAG:", "Lithology fallback:",
)
_REPLACED_CONTEXT_NOTE_PREFIXES = (
    "VCL:", "PHIN:", "PHID:", "PHIS:", "PHIT:", "PHIN/PHID/PHIS/PHIT:", "Porosity:", "PHIE:", "PERM:",
    "RW:", "SW:", "Lithology:", "Mineral volumes:", "LITHO:",
)


@dataclass(frozen=True)
class CostaWellParameters:
    """One row of ``costa_interpretation_parameters.csv``."""

    well: str
    bht_f: Optional[float]
    rw_68f: Optional[float]
    saturation_method: str
    vcl_method: str
    phie_method: str

    @property
    def archie(self) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Archie a, m and n written in the saturation method (a = 1 for Archie when unstated)."""
        text = self.saturation_method or ""
        found = {
            name: re.search(rf"\b{name}\s*=\s*([0-9.]+)", text, flags=re.IGNORECASE)
            for name in ("a", "m", "n")
        }
        a = float(found["a"].group(1)) if found["a"] else 1.0 if "archie" in text.lower() else None
        m = float(found["m"].group(1)) if found["m"] else None
        n = float(found["n"].group(1)) if found["n"] else None
        return a, m, n

    @property
    def rw_68f_used(self) -> float:
        """Rw at 68 degF: the table's, else the package's 0.1 ohm.m default."""
        return self.rw_68f if self.rw_68f is not None else _DEFAULT_RW_68F


def _optional_float(text: str) -> Optional[float]:
    text = (text or "").strip()
    return float(text) if text else None


def load_costa_parameters(path: Optional[str | Path] = None) -> dict[str, CostaWellParameters]:
    """The COSTA interpretation parameters by upper-case well name (default: the bundled table)."""
    if path is None:
        text = resources.files("petrophys").joinpath("_internal/data/costa_interpretation_parameters.csv").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    rows = csv.DictReader(text.splitlines())
    return {
        row["well"].strip().upper(): CostaWellParameters(
            well=row["well"].strip(),
            bht_f=_optional_float(row["bht_f"]),
            rw_68f=_optional_float(row["rw_68f"]),
            saturation_method=row["saturation_method"],
            vcl_method=row["vcl_method"],
            phie_method=row["phie_method"],
        )
        for row in rows
    }


def load_costa_reference_volumes(path: Optional[str | Path] = None) -> dict[str, dict[float, tuple[float, float]]]:
    """Calcite (VLIM) and dolomite (VDOL) volumes by upper-case well and depth (ft, rounded to 1e-6)."""
    if path is None:
        text = resources.files("petrophys").joinpath("_internal/data/costa_reference_carbonate_volumes.csv").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    volumes: dict[str, dict[float, tuple[float, float]]] = {}
    for row in csv.DictReader(text.splitlines()):
        vlim = _optional_float(row["vlim"])
        vdol = _optional_float(row["vdol"])
        volumes.setdefault(row["well"].strip().upper(), {})[round(float(row["depth_ft"]), 6)] = (
            np.nan if vlim is None else vlim,
            np.nan if vdol is None else vdol,
        )
    return volumes


def costa_workflow_config():
    """The workflow settings the COSTA package interpreted every well with."""
    from petrophys.workflow.dynamic_workflow import DynamicWorkflowConfig

    return DynamicWorkflowConfig(
        calculation_profile="geolink",
        surface_temp_f=_SURFACE_TEMP_F,
        lithology="carbonate",
        matrix_type="limestone",
        rho_fl=_FLUID_DENSITY,
        phi_cl=0.20,
        sw_model="archie",
        skip_steps=list(_SKIPPED_STEPS),
    )


def _vcl_from_gamma_ray(gr: np.ndarray, method: str) -> tuple[np.ndarray, str]:
    igr = np.clip((np.asarray(gr, dtype=float) - 20.0) / 80.0, 0.0, 1.0)
    lower = (method or "").lower()
    if "larionov" in lower:
        return 0.083 * ((2.0 ** (3.7 * igr)) - 1.0), "larionov_tertiary"
    if "steiber" in lower:
        return np.divide(igr, 3.0 - 2.0 * igr, out=np.zeros_like(igr), where=(3.0 - 2.0 * igr) != 0.0), "steiber"
    return igr, "linear_gr"


def _solve_calcite_dolomite(phie: float, rhob: float, vcl: float) -> tuple[float, float, float]:
    """Quartz (0), calcite and dolomite filling 1 - VCL - PHIE that best match RHOB."""
    from scipy import optimize

    if not all(np.isfinite(value) for value in (phie, rhob, vcl)):
        return (np.nan, np.nan, np.nan)
    available = 1.0 - phie - vcl
    if available < -1e-9:
        return (np.nan, np.nan, np.nan)
    available = max(0.0, float(available))
    if available <= 1e-12:
        return (0.0, 0.0, 0.0)

    def objective(volumes: np.ndarray) -> float:
        density = (2.52 * vcl + 2.71 * volumes[0] + 2.847 * volumes[1] + _FLUID_DENSITY * phie) - rhob
        return float(density * density)

    result = optimize.minimize(
        objective,
        np.array([available / 2.0, available / 2.0], dtype=float),
        bounds=((0.0, available), (0.0, available)),
        constraints=({"type": "eq", "fun": lambda volumes: (volumes[0] + volumes[1]) - available},),
        options={"disp": False, "maxiter": 100},
    )
    if not np.all(np.isfinite(result.x)):
        return (np.nan, np.nan, np.nan)
    return (0.0, float(np.clip(result.x[0], 0.0, available)), float(np.clip(result.x[1], 0.0, available)))


def _carbonate_volumes(well, reference: dict[float, tuple[float, float]]) -> dict[str, np.ndarray]:
    n = well.n_samples
    rhob_curve = well.get("RHOB")
    phie_curve = well.get("PHIE") or well.get("PHIE_CC")
    vcl_curve = well.get("VCL")
    if rhob_curve is None or phie_curve is None or vcl_curve is None:
        return {name: np.full(n, np.nan) for name in ("VQUAR", "VCALC", "VDOLO", "LITHOLOGY_GEOLINK", "LITHO")}
    rhob = np.asarray(rhob_curve.data, dtype=float)
    phie = np.asarray(phie_curve.data, dtype=float)
    vcl = np.asarray(vcl_curve.data, dtype=float)
    depth = np.asarray(well.depth, dtype=float)
    vquar = np.zeros(n)
    vcalc = np.full(n, np.nan)
    vdolo = np.full(n, np.nan)
    if reference:
        for index, value in enumerate(depth):
            if np.isfinite(value):
                vcalc[index], vdolo[index] = reference.get(round(float(value), 6), (np.nan, np.nan))
    for index in range(n):
        if np.isfinite(vcalc[index]) or np.isfinite(vdolo[index]):
            continue
        if not all(np.isfinite(value) for value in (phie[index], rhob[index], vcl[index])):
            continue
        vquar[index], vcalc[index], vdolo[index] = _solve_calcite_dolomite(phie[index], rhob[index], vcl[index])
    known = np.isfinite(vcalc) | np.isfinite(vdolo)
    limestone = np.isfinite(depth) & ~known
    vcalc[limestone] = 1.0
    vdolo[limestone] = 0.0
    known = np.isfinite(vcalc) | np.isfinite(vdolo)
    vquar[~known] = np.nan
    geolink = np.full(n, np.nan)
    litho = np.full(n, np.nan)
    for index in range(n):
        values = {
            "limestone": max(vcalc[index], 0.0) if np.isfinite(vcalc[index]) else np.nan,
            "dolomite": max(vdolo[index], 0.0) if np.isfinite(vdolo[index]) else np.nan,
        }
        finite = {name: value for name, value in values.items() if np.isfinite(value)}
        if not finite:
            continue
        if max(finite, key=finite.get) == "dolomite":
            geolink[index], litho[index] = 15.0, 3.0
        else:
            geolink[index], litho[index] = 11.0, 2.0
    return {"VQUAR": vquar, "VCALC": vcalc, "VDOLO": vdolo, "LITHOLOGY_GEOLINK": geolink, "LITHO": litho}


def _add_precomputed_curves(well, parameters: CostaWellParameters, reference) -> None:
    core_perm = well.get("PERM_CC") or well.get("CORE_PERM")
    if core_perm is not None:
        well.add_curve("PERM", core_perm.data, "mD", "Permeability from core")
    gr = well.get("GR")
    if gr is not None:
        vcl, method = _vcl_from_gamma_ray(gr.data, parameters.vcl_method)
        well.add_curve("VCL", vcl, "v/v", f"Clay volume: {method}")
    volumes = _carbonate_volumes(well, reference)
    descriptions = {
        "VQUAR": ("v/v", "Quartz volume fixed at zero for two-group carbonate lithology"),
        "VCALC": ("v/v", "Limestone volume from carbonate volume equations"),
        "VDOLO": ("v/v", "Dolomite volume from carbonate volume equations"),
        "LITHOLOGY_GEOLINK": ("", "Two-group lithology code from dominant limestone/dolomite volume"),
        "LITHO": ("", "Two-group lithology class from dominant limestone/dolomite volume"),
    }
    for mnemonic, (unit, description) in descriptions.items():
        well.add_curve(mnemonic, volumes[mnemonic], unit, description)


def _add_crossover_flag(well) -> None:
    """CROSSOVER_FLAG from HydrocarbonAnalysis, with a clean bad-hole screen (no CALI, BS or DRHO)."""
    from petrophys.evaluation.diagnostics.hydrocarbon_analysis import HydrocarbonAnalysis

    before = set(well.curves)
    temporary = well.get("BAD_HOLE") is None and well.get("BAD_HOLE_FLAG") is None
    if temporary:
        well.add_curve("BAD_HOLE", np.zeros(well.n_samples), "", "Temporary clean bad-hole screen for library CROSSOVER_FLAG calculation")
    HydrocarbonAnalysis().analyze(well, add_curves=True)
    if temporary:
        well.remove_curve("BAD_HOLE")
    for mnemonic in sorted(set(well.curves) - before):
        if mnemonic.upper() != "CROSSOVER_FLAG":
            well.remove_curve(mnemonic)
    flag = well.get("CROSSOVER_FLAG")
    if flag is not None and "CROSSOVER_FLAG" not in before and not np.any(np.isfinite(np.asarray(flag.data, dtype=float))):
        well.remove_curve("CROSSOVER_FLAG")


def _curve_summary(well, mnemonic: str) -> dict[str, Any]:
    curve = well.get(mnemonic)
    if curve is None:
        return {"present": False}
    summary: dict[str, Any] = {
        "present": True,
        "curve_name": curve.mnemonic,
        "unit": curve.unit,
        "description": curve.metadata.description if curve.metadata else "",
    }
    values = np.asarray(curve.data, dtype=float)
    finite = values[np.isfinite(values)]
    summary.update(
        {
            "numeric": True,
            "sample_count": int(values.size),
            "finite_count": int(finite.size),
            "nan_count": int(values.size - finite.size),
            "finite_percent": round(float(100.0 * finite.size / values.size), 4) if values.size else 0.0,
        }
    )
    if finite.size:
        summary.update({"min": float(finite.min()), "p50": float(np.percentile(finite, 50)), "max": float(finite.max())})
    return summary


def _has_finite(well, mnemonic: str) -> bool:
    curve = well.get(mnemonic)
    return curve is not None and bool(np.any(np.isfinite(np.asarray(curve.data, dtype=float))))


def _precomputed_trace_steps(well, parameters: CostaWellParameters) -> list[dict[str, Any]]:
    """The package's trace of the steps run before and after the workflow."""
    summaries = lambda names: {name: _curve_summary(well, name) for name in names}  # noqa: E731
    return [
        {
            "step": "costa_core_curve_aliases",
            "success": well.get("PHIE_CC") is not None or well.get("PERM_CC") is not None,
            "method": "COSTA source core curve mapping",
            "input_curves": ["PHIE or CORE_POR", "CORE_PERM"],
            "output_curves": ["PHIE_CC", "PERM_CC"],
            "parameters": {"parameter_table": "costa_interpretation_parameters.csv"},
            "decisions": [
                "Source PHIE is mapped to PHIE_CC so library PHIE remains log-derived.",
                "PHIE_CC values are from cores.",
                "PERM_CC values are from cores.",
            ],
            "output_curve_summaries": summaries(["PHIE_CC", "PERM_CC"]),
        },
        {
            "step": "costa_vcl",
            "success": _has_finite(well, "VCL"),
            "method": parameters.vcl_method,
            "input_curves": ["GR"],
            "output_curves": ["VCL"],
            "parameters": {"parameter_table": "costa_interpretation_parameters.csv"},
            "decisions": ["VCL uses the per-well method of the COSTA parameter table."],
            "input_curve_summaries": summaries(["GR"]),
            "output_curve_summaries": summaries(["VCL"]),
        },
        {
            "step": "costa_lithology_volumes",
            "success": _has_finite(well, "VCALC") or _has_finite(well, "VDOLO"),
            "method": "carbonate mineral volume equations",
            "input_curves": ["VCL", "PHIE_CC", "RHOB", "reference VLIM/VDOL where available"],
            "output_curves": ["VQUAR", "VCALC", "VDOLO", "LITHO", "LITHOLOGY_GEOLINK"],
            "parameters": {"vquar": 0.0, "rho_clay": 2.52, "rho_calcite": 2.71, "rho_dolomite": 2.847, "rho_fluid": _FLUID_DENSITY},
            "decisions": [
                "VQUAR=0 for the exported two-group COSTA carbonate lithology.",
                "Missing samples use VCALC+VDOLO=1-VCL-PHIE_CC and RHOB=2.52*VCL+2.71*VCALC+2.847*VDOLO+1.1*PHIE_CC.",
                "LITHO is the dominant lithology from VCALC versus VDOLO; exported groups are limestone and dolomite only.",
            ],
            "input_curve_summaries": summaries(["VCL", "PHIE_CC", "RHOB"]),
            "output_curve_summaries": summaries(["VQUAR", "VCALC", "VDOLO", "LITHO", "LITHOLOGY_GEOLINK"]),
        },
        {
            "step": "costa_core_permeability",
            "success": _has_finite(well, "PERM"),
            "method": "core corrected permeability",
            "input_curves": ["PERM_CC"],
            "output_curves": ["PERM"],
            "parameters": {},
            "decisions": ["PERM compatibility alias is copied from PERM_CC; module consumers prioritize PERM_CC."],
            "input_curve_summaries": summaries(["PERM_CC"]),
            "output_curve_summaries": summaries(["PERM"]),
        },
        {
            "step": "costa_crossover_flag",
            "success": _has_finite(well, "CROSSOVER_FLAG"),
            "method": "petrophys HydrocarbonAnalysis",
            "input_curves": ["PHID", "PHIN", "VCL", "PHIE_CC"],
            "output_curves": ["CROSSOVER_FLAG"],
            "parameters": {
                "nd_separation_min": 0.05,
                "vcl_max": 0.35,
                "phie_min": 0.05,
                "bad_hole_assumption": "clean screen because CALI, BS, and DRHO are unavailable",
            },
            "decisions": [
                "CROSSOVER_FLAG=1 where PHID-PHIN>0.05, VCL<0.35, PHIE_CC>0.05, and BAD_HOLE<0.5.",
                "BAD_HOLE is not exported; a temporary clean BAD_HOLE=0 screen is used only inside HydrocarbonAnalysis.",
            ],
            "input_curve_summaries": summaries(["PHID", "PHIN", "VCL", "PHIE_CC", "BAD_HOLE"]),
            "output_curve_summaries": summaries(["CROSSOVER_FLAG"]),
        },
    ]


def interpret_costa_las(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    well_names: Optional[Sequence[str]] = None,
    parameters: Optional[dict[str, CostaWellParameters]] = None,
    reference_volumes: Optional[dict[str, dict[float, tuple[float, float]]]] = None,
    pattern: str = "*.las",
):
    """Interpret the COSTA LAS files as the COSTA package did (steps 1 and 2 of the module docstring).

    Writes the interpreted LAS and a ``workflow_json`` trace per well under
    ``output_dir``, and returns one BatchLasResult per well.
    """
    from petrophys.io import export_las, read_las
    from petrophys.preprocessing.standardization import standardize_curves
    from petrophys.reporting.reference_export import _filter_input_files_by_well_names
    from petrophys.workflow.dynamic_workflow import _dataset_calculations
    from petrophys.workflow.zoned_batch import (
        BatchLasResult,
        _finalize_export_curves,
        _write_json_log,
        run_dynamic_workflow_with_zones,
    )

    parameters = parameters if parameters is not None else load_costa_parameters()
    reference_volumes = reference_volumes if reference_volumes is not None else load_costa_reference_volumes()
    source = Path(input_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = _filter_input_files_by_well_names(sorted(source.glob(pattern)), well_names)
    results = []
    for file_path in files:
        relative = file_path.relative_to(source)
        export_path = output / relative
        try:
            well = read_las(file_path)
            depth_label = getattr(well, "_source_depth_unit_label", None)
            # As the package, which kept the source's equivalent unit labels.
            well = standardize_curves(well, rename_curves=True, convert_units=True, relabel_equivalent_units=False)
            key = file_path.stem.upper()
            if key not in parameters:
                raise ValueError(f"No COSTA interpretation parameters for well {file_path.stem}")
            well_parameters = parameters[key]
            input_curves = set(well.curves)
            _add_precomputed_curves(well, well_parameters, reference_volumes.get(key, {}))
            well.set_parameter("_COSTA_VCL_METHOD", well_parameters.vcl_method)
            well.set_parameter("_COSTA_PHIE_METHOD", well_parameters.phie_method)
            well.set_parameter("_COSTA_SATURATION_METHOD", well_parameters.saturation_method)
            finalize_inputs = set(well.curves)
            with _dataset_calculations("costa"):
                workflow_result = run_dynamic_workflow_with_zones(well, config=costa_workflow_config())
            _add_crossover_flag(workflow_result.well)
            for mnemonic in _UNSUPPORTED_SATURATION_CURVES:
                if workflow_result.well.get(mnemonic) is not None:
                    workflow_result.well.remove_curve(mnemonic)
            _finalize_export_curves(workflow_result.well, finalize_inputs)
            export_las(workflow_result.well, export_path, depth_unit_label=depth_label)

            warnings = list(workflow_result.warnings)
            warnings.extend(workflow_result.base_result.warnings)
            for zone_result in workflow_result.zone_results:
                warnings.extend(zone_result.warnings)
            if well_parameters.rw_68f is None:
                warnings.append(
                    f"No numeric RW in the COSTA parameter table; default RW={_DEFAULT_RW_68F:g} ohm.m at 68 F was used."
                )
            computed = set(workflow_result.well.curves) - input_curves
            if _has_finite(workflow_result.well, "CROSSOVER_FLAG"):
                computed.add("CROSSOVER_FLAG")
            item = BatchLasResult(
                source_file=str(file_path),
                output_file=str(export_path),
                success=workflow_result.base_result.success
                and all(zone.success for zone in workflow_result.zone_results),
                warnings=warnings,
                input_curves=sorted(input_curves),
                output_curves=sorted(workflow_result.well.curves),
                computed_curves=sorted(computed),
                zones_applied=len(workflow_result.zones),
                result=workflow_result,
            )
            item.json_log_file = _write_json_log(item, relative, output)
            trace_path = Path(item.json_log_file)
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            payload["costa_precomputed_steps"] = _precomputed_trace_steps(workflow_result.well, well_parameters)
            trace_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            results.append(item)
        except Exception as exc:
            failure = BatchLasResult(source_file=str(file_path), output_file=str(export_path), success=False, errors=[str(exc)])
            failure.json_log_file = _write_json_log(failure, relative, output)
            results.append(failure)
    return results


def write_costa_formation_tops(processed_dir: str | Path, tops_file: str | Path) -> Path:
    """One formation, KHARAIB-2 FM. TOP, from the first sample of each interpreted LAS."""
    import pandas as pd

    from petrophys.io import read_las

    rows = []
    for file_path in sorted(Path(processed_dir).glob("*.las")):
        well = read_las(file_path)
        if well.top is not None:
            rows.append({"well": file_path.stem, "formation": FORMATION_NAME, "md_top": float(well.top)})
    output = Path(tops_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(output, index=False)
    return output


# -- Context notes -------------------------------------------------------------


def _stat_range_text(stats: dict[str, Any], curve: str, unit: str = "") -> Optional[str]:
    item = stats.get(f"{curve}_overall")
    if not isinstance(item, dict) or item.get("minimum") is None or item.get("maximum") is None:
        return None
    suffix = f" {unit}" if unit else ""
    return f"{float(item['minimum']):.4g} to {float(item['maximum']):.4g}{suffix}"


def _note_lithology(lithology: Any) -> str:
    text = str(lithology or "").strip().replace("_", " ").lower()
    if "dol" in text:
        return "Dolomite"
    if "lime" in text or "calc" in text:
        return "Limestone"
    return "Carbonate"


def costa_method_notes(parameters: CostaWellParameters, stats: dict[str, Any], lithology: Any = None) -> list[str]:
    """The per-curve method notes the COSTA package wrote for a context file."""
    temp_range = _stat_range_text(stats, "TEMP", "F")
    temp_suffix = f" TEMP={temp_range}." if temp_range else "."
    bht_text = f"{parameters.bht_f:g} F" if parameters.bht_f is not None else "unavailable"
    lithology_label = _note_lithology(lithology)
    a, m, n = parameters.archie
    if a is None or m is None or n is None:
        a, m, n = 1.0, 2.0, 2.0
    notes = [
        f"TEMP: TEMP=68+(BHT-68)*DEPT/TD using BHT={bht_text} and TD from well heads{temp_suffix}",
        "BAD_HOLE: not computed because CALI, BS, and DRHO inputs are unavailable.",
        f"VCL: {parameters.vcl_method}.",
        "PHIN: clip(NPHI-nphi_ma(depth),0,0.6); nphi_ma follows the library matrix endpoint for LITHO.",
        f"PHID: (rho_ma(depth)-RHOB)/(rho_ma(depth)-rho_fl), clipped to 0-0.5; rho_fl={_FLUID_DENSITY:g} g/cc.",
    ]
    if isinstance(stats.get("PHIS_overall"), dict):
        notes.append("PHIS: (DTC/DT-dt_ma(depth))/(dt_fl-dt_ma(depth)), clipped to 0-0.5; dt_fl=189 us/ft.")
    else:
        notes.append("PHIS: not computed because DTC/DT has no finite values in this segment.")
    notes.extend(
        [
            "PHIT: PHID where PHID>PHIN+0.05 or PHIN is unavailable; PHIN where PHID is unavailable; otherwise 0.6*PHID+0.4*PHIN; if only PHIS is available, PHIT=PHIS; clipped to 0-0.5.",
            "PHIE: min(max(PHIT-VCL*0.20,0),max(min(PHIT,1-VCL),0)).",
        ]
    )
    if isinstance(stats.get("PHI_SEC_overall"), dict):
        notes.append("PHI_SEC: max(PHIT-PHIS,0).")
    notes.append("PHIE_CC: values from cores.")
    notes.extend(
        [
            f"SW: {lithology_label} Archie Sw=(a*RW(T)/(PHIE_CC^m*RT))^(1/n); a={a:g}, m={m:g}, n={n:g}; computed depth-by-depth where RW(T), PHIE_CC, and RT are finite.",
            "SO: 1-SW.",
            "BVW: PHIE_CC*SW.",
            (
                "CROSSOVER_FLAG: petrophys HydrocarbonAnalysis; 1 where PHID-PHIN>0.05, "
                "VCL<0.35, PHIE_CC>0.05, with CALI/BS/DRHO screen treated as clean because "
                "those inputs are unavailable."
            ),
            "VQUAR: fixed to 0 for the exported two-mineral carbonate model.",
            "VCALC: calcite/limestone volume from VCALC+VDOLO=1-VCL-PHIE_CC and RHOB=2.52*VCL+2.71*VCALC+2.847*VDOLO+1.1*PHIE_CC.",
            "VDOLO: dolomite volume from VCALC+VDOLO=1-VCL-PHIE_CC and RHOB=2.52*VCL+2.71*VCALC+2.847*VDOLO+1.1*PHIE_CC.",
            "LITHO: dominant lithology from VCALC vs VDOLO only; groups are limestone and dolomite.",
            "PERM_CC: values from cores.",
            "RQI: 0.0314*sqrt(PERM_CC/PHIE_CC).",
            "PHI_NORM: PHIE_CC/(1-PHIE_CC).",
            "FZI: RQI/PHI_NORM.",
            "NET_ROCK: 1 where VCL<0.3; else 0.",
            "NET_RES: 1 where NET_ROCK=1 and selected effective porosity passes the library reservoir cutoff; else 0.",
            "NET_PAY: 1 where NET_RES=1 and SW<0.7; else 0.",
            "RESFLAG: same reservoir flag basis as NET_RES.",
            "SEALFLAG: 1 where VCL>=0.8; else 0.",
            "DS_SEPARATION: PHID-PHIS.",
            "PORO_QUALITY_INDEX: sqrt(max(PHIT*log10(PERM),1e-6)); clipped 0.001-5.",
        ]
    )
    rw_range = _stat_range_text(stats, "RW", "ohm.m")
    rw_suffix = f" RW(T)={rw_range}." if rw_range else ""
    rw_source = "from costa_params.xlsx" if parameters.rw_68f is not None else "default because no numeric RW was supplied"
    notes.append(
        f"RW: petrophys RW step applies Arps correction from RW={parameters.rw_68f_used:.6g} "
        f"ohm.m at {_RW_MEASURED_TEMP_F:.0f} F ({rw_source}); "
        f"RW(T)=RW68*(68+6.77)/(TEMP+6.77).{rw_suffix}"
    )
    return notes


_METHODOLOGY_REWRITES = (
    (
        "methodology",
        lambda value: "PHIE_CC" in value or "PERM_CC" in value or "empirical permeability" in value
        or "porosity=PHIT and permeability=PERM" in value,
        "Uses library log-derived PHIE for porosity analysis and PHIE_CC/PERM_CC values from cores where "
        "core-corrected inputs are prioritized; empirical permeability transforms are not used.",
    ),
    (
        "methodology",
        lambda value: value == "Maps raw lithology codes to grouped lithology classes.",
        "Maps processed COSTA carbonate lithology codes to two exported groups: limestone and dolomite.",
    ),
    (
        "calculation_thresholds",
        lambda value: "GT_{GROUP} for mapped codes" in value,
        "Exported lithology groups are limestone and dolomite only; limestone uses processed code 11 and "
        "dolomite uses processed code 15.",
    ),
    (
        "methodology",
        lambda value: "CONSISTENCY_SCORE" in value and ("sandstone" in value or "shale" in value),
        "CONSISTENCY_SCORE uses available GR/PEF carbonate checks for the COSTA limestone and dolomite groups.",
    ),
    (
        "calculation_thresholds",
        lambda value: "TIGHT_FORMATION from tight sandstone/carbonate" in value,
        "Overall interpretation is based on the exported limestone/dolomite group and reservoir-quality "
        "labels; no non-carbonate lithology groups are exported.",
    ),
)


def _rewrite_methodology(obj: Any) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            for target, matches, replacement in _METHODOLOGY_REWRITES:
                if str(key) == target and isinstance(value, str) and matches(value):
                    obj[key] = replacement
                    break
            else:
                _rewrite_methodology(value)
    elif isinstance(obj, list):
        for value in obj:
            _rewrite_methodology(value)


def patch_costa_context_notes(context_dir: str | Path, parameters: Optional[dict[str, CostaWellParameters]] = None) -> int:
    """Rewrite a COSTA context package's notes and curve lists as the package did; returns the files patched."""
    parameters = parameters if parameters is not None else load_costa_parameters()
    count = 0
    for json_path in sorted(Path(context_dir).rglob("*.json")):
        if json_path.name == "curve_name_mapping.json":
            continue
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        analysis = payload.get("analysis_results")
        if not isinstance(analysis, dict):
            continue
        well_parameters = parameters.get(str(analysis.get("well_name") or "").upper())
        if well_parameters is None:
            continue
        statistics = analysis.get("statistics") if isinstance(analysis.get("statistics"), dict) else {}
        notes = costa_method_notes(well_parameters, statistics, analysis.get("lithology") or analysis.get("lithology group"))
        analysis.pop("derived_log_computation_notes", None)
        completeness = analysis.get("data_completeness")
        if isinstance(completeness, dict):
            completeness["derived_log_computation_notes"] = notes
        _rewrite_methodology(payload)
        if isinstance(analysis.get("input_logs_present"), list):
            analysis["input_logs_present"] = [
                curve for curve in analysis["input_logs_present"] if str(curve).upper() not in _DROP_CONTEXT_CURVES
            ]
        derived = analysis.get("derived_logs_computed")
        if isinstance(derived, list):
            kept = [curve for curve in derived if str(curve).upper() not in _DROP_CONTEXT_CURVES]
            inputs = {str(curve).upper() for curve in analysis.get("input_logs_present", []) if isinstance(curve, str)}
            if "SO" not in inputs and isinstance(statistics.get("SO_overall"), dict) and "SO" not in {
                str(curve).upper() for curve in kept
            }:
                kept.append("SO")
            analysis["derived_logs_computed"] = sorted(kept, key=str)
        for key in list(statistics):
            if str(key).split("_overall", 1)[0].upper() in _DROP_CONTEXT_CURVES:
                statistics.pop(key, None)
        analysis.pop("costa_method_sources", None)
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        count += 1
    return count


def reproduce_costa_context(
    output_dir: str | Path,
    *,
    work_dir: str | Path,
    input_dir: Optional[str | Path] = None,
    well_names: Optional[Sequence[str]] = None,
    parameters_file: Optional[str | Path] = None,
    reference_volumes_file: Optional[str | Path] = None,
    pattern: str = "*.las",
):
    """Compute the COSTA context package.

    Parameters
    ----------
    output_dir : str or Path
        Folder for the category context JSON.
    work_dir : str or Path
        Folder for the interpreted LAS with their workflow traces
        (``processed_las``) and the formation tops the context used.
    input_dir : str or Path, optional
        Folder of the COSTA LAS files; default the bundled
        ``petrophys/datasets/costa`` copies.
    well_names : sequence of str, optional
        Only these wells (file stems).
    parameters_file, reference_volumes_file : str or Path, optional
        CSV tables replacing the bundled ``costa_interpretation_parameters.csv``
        and ``costa_reference_carbonate_volumes.csv``.

    Returns
    -------
    list of ExampleFormatExportResult
        One per context file.
    """
    from petrophys.reporting.reference_export import (
        export_reference_json_work_package_by_category_formation_lithology,
        write_reference_curve_name_mapping,
    )
    from petrophys.workflow.dynamic_workflow import _dataset_calculations

    parameters = load_costa_parameters(parameters_file)
    reference = load_costa_reference_volumes(reference_volumes_file)
    if input_dir is None:
        source = Path(str(resources.files("petrophys").joinpath("datasets/costa")))
    else:
        source = Path(input_dir)
    work = Path(work_dir)
    processed_dir = work / "processed_las"
    batch = interpret_costa_las(
        source, processed_dir, well_names=well_names, parameters=parameters, reference_volumes=reference, pattern=pattern
    )
    failed = [item for item in batch if not item.success]
    if failed:
        raise RuntimeError(
            "COSTA interpretation failed: "
            + "; ".join(f"{Path(item.source_file).stem}: {' | '.join(item.errors)}" for item in failed)
        )
    tops_file = write_costa_formation_tops(processed_dir, work / "costa_formation_tops_whole_well.xlsx")

    with _dataset_calculations("costa"):
        results = export_reference_json_work_package_by_category_formation_lithology(
            input_dir=processed_dir,
            output_dir=output_dir,
            tops_file=tops_file,
            minimum_group_thickness=_MIN_GROUP_THICKNESS_FT,
            well_names=[Path(item.source_file).stem for item in batch],
            raw_input_dir=source,
            workflow_config=costa_workflow_config(),
        )
    write_reference_curve_name_mapping(output_dir)
    patch_costa_context_notes(output_dir, parameters)
    return results
