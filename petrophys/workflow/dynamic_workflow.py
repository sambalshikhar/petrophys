"""
Dynamic petrophysics workflow.

This module provides a DynamicWorkflow class that automatically detects
available input curves and runs the appropriate modules in the correct
order. It supports both granular (single-module) and full pipeline
execution modes.

Pipeline execution order (from design doc):
    Module 1: QC & Pre-Processing
    Module 2: Volume of Clay (VCLAY) & Clay Resistivity
    Module 3: Lithology Determination
    Module 4: Fluid Resistivity (Rmf, Rw)
    Module 5: HC Correction
    Module 6: Porosity
    Module 7: Water Saturation
    Module 8: Permeability
    Module 9: Rock Quality (RQI, FZI)
    Module 10: Net Pay
"""

from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import re
from petrophys.base.well import Well
import pandas as pd
from petrophys.io import ascii_reader, las_reader

class WorkflowStep(Enum):
    """Individual workflow steps."""
    NORMALIZATION = "normalization"
    BAD_HOLE = "bad_hole"
    TEMPERATURE = "temperature"
    VCLAY = "vclay"
    RCL = "rcl"
    LITHOLOGY = "lithology"
    RMF = "rmf"
    RW = "rw"
    INVASION = "invasion"
    HC_CORRECTION = "hc_correction"
    POROSITY = "porosity"
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


def _temperature_to_f(value: float, unit: str) -> float:
    """Convert a temperature value to Fahrenheit when units indicate Celsius."""
    text = (unit or "").strip().lower()
    if "c" in text and "f" not in text:
        return (float(value) * 9.0 / 5.0) + 32.0
    return float(value)


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
        return _temperature_to_f(value, unit)

    value = _constant_zoned_parameter(well, name)
    unit = _parameter_text(well, f"{name}_Z1__UNIT")
    if value is not None:
        return _temperature_to_f(value, unit)

    return None


def _metadata_bht_f(well: "Well") -> Optional[float]:
    """Return BHT metadata in Fahrenheit."""
    if well.header.bht is not None:
        return float(well.header.bht)

    value = _parameter_float(well, "BHT")
    unit = _parameter_text(well, "BHT__UNIT").lower()
    if value is not None:
        if "c" in unit and "f" not in unit:
            return (float(value) * 9.0 / 5.0) + 32.0
        return float(value)

    value = _constant_zoned_parameter(well, "BHT")
    unit = _parameter_text(well, "BHT_Z1__UNIT").lower()
    if value is not None:
        if "c" in unit and "f" not in unit:
            return (float(value) * 9.0 / 5.0) + 32.0
        return float(value)

    return None


@dataclass
class DynamicWorkflowConfig:
    """
    Configuration for the dynamic workflow.

    All parameters are optional. The workflow auto-detects what can
    be computed based on available curves and provided parameters.
    """
    # Temperature
    surface_temp_f: float = 68.0
    bht_f: Optional[float] = None
    td_ft: Optional[float] = None

    # Mud properties
    mud_type: str = "WBM"  # WBM or OBM
    mud_has_clay_or_kcl: bool = False
    rmf_at_meas_temp: Optional[float] = None
    rmf_meas_temp_f: float = 75.0
    rmf_salinity_ppm: Optional[float] = None
    rm: Optional[float] = None
    mud_water: str = "fresh"  # fresh or salt

    # Formation water
    rw: Optional[float] = None
    rw_meas_temp_f: Optional[float] = None
    salinity_ppm: Optional[float] = None

    # Matrix properties
    matrix_type: str = "sandstone"
    rho_ma: Optional[float] = None
    nphi_ma: Optional[float] = None
    rho_fl: float = 1.0
    rho_cl: float = 2.45
    dt_ma: Optional[float] = None
    dt_fl: float = 189.0

    # Lithology
    lithology: str = "clastic"  # clastic or carbonate
    lithology_csv: Optional[str] = None  # Path to lithology CSV (top, bottom, lithology)

    # Archie parameters
    a: float = 1.0
    m: float = 2.0
    n: float = 2.0

    # Waxman-Smits parameters
    cec: Optional[float] = None
    b_coeff: Optional[float] = None
    m_star: float = 2.0

    # Dual Water parameters
    rwb: Optional[float] = None
    phi_cl: float = 0.15

    # HC correction
    hc_type: str = "oil"  # oil or gas

    # Cutoffs
    vcl_cutoff: float = 0.3
    sw_cutoff: float = 0.7
    phie_cutoff: Optional[float] = None

    # GR parameters (auto-detected if None)
    gr_min: Optional[float] = None
    gr_max: Optional[float] = None
    gr_method: str = "linear"

    # SP parameters (auto-detected if None)
    sp_min: Optional[float] = None
    sp_max: Optional[float] = None

    # Permeability
    perm_method: str = "auto"
    perm_fallback_method: str = "timur"
    swirr: Optional[float] = None
    peer_wells: Optional[List[Any]] = None
    formation_tops: Optional[Any] = None

    # Saturation model (auto = use routing logic)
    sw_model: str = "auto"
    rtsh: Optional[float] = None
    vsh_exp: float = 1.6
    shale_exp: float = 1.6

    # Control which steps to skip
    skip_steps: List[str] = field(default_factory=list)


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
        Error message if failed, or summary if successful.
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


@dataclass
class DynamicWorkflowResult:
    """Results from dynamic workflow execution."""
    well: "Well"
    config: DynamicWorkflowConfig
    steps_executed: List[StepResult] = field(default_factory=list)
    steps_skipped: List[str] = field(default_factory=list)
    specific_steps_skipped: dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Computed values
    rw_computed: Optional[float] = None
    rmf_computed: Optional[float] = None
    rcl_computed: Optional[float] = None
    temperature_gradient: Optional[float] = None

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    def print_trace(self) -> None:
        """Print a comprehensive human-readable trace of all steps."""
        print(f"\n{'=' * 90}")
        print(f"WORKFLOW TRACE — {self.well.name or 'Unknown Well'}")
        print(f"{'=' * 90}")
        print(f"Overall: {'SUCCESS' if self.success else 'FAILED'} | "
              f"Executed: {len(self.steps_executed)} | Skipped: {len(self.steps_skipped)}")
        if self.rw_computed is not None:
            print(f"Rw = {self.rw_computed:.4f} ohm.m | Rmf = {self.rmf_computed} | "
                  f"Rcl = {self.rcl_computed}")
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
            print(f"Skipped: {', '.join(self.steps_skipped)}")
        if self.warnings:
            print(f"\nWarnings:")
            for w in self.warnings:
                print(f"  - {w}")
        print(f"{'=' * 90}")

    def to_dataframe(self):
        """Export step trace to a pandas DataFrame."""
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
    ...     lithology='clastic', salinity_ppm=25000
    ... )
    >>> workflow = DynamicWorkflow(config)
    >>> result = workflow.run(well)
    >>> print(result.steps_executed)
    """

    def __init__(self, config: Optional[DynamicWorkflowConfig] = None):
        self.config = config or DynamicWorkflowConfig()
        self._result: Optional[DynamicWorkflowResult] = None
        self._lithology_log: Optional[np.ndarray] = None

    def get_pipeline(self):
        return [
            (WorkflowStep.TEMPERATURE, self._step_temperature, self._can_temperature),
            (WorkflowStep.BAD_HOLE, self._step_bad_hole, self._can_bad_hole),
            (WorkflowStep.VCLAY, self._step_vclay, self._can_vclay),
            (WorkflowStep.RCL, self._step_rcl, self._can_rcl),
            (WorkflowStep.LITHOLOGY, self._step_lithology, self._can_lithology),
            (WorkflowStep.RMF, self._step_rmf, self._can_rmf),
            (WorkflowStep.RW, self._step_rw, self._can_rw),
            (WorkflowStep.INVASION, self._step_invasion, self._can_invasion),
            (WorkflowStep.HC_CORRECTION, self._step_hc_correction, self._can_hc_correction),
            (WorkflowStep.POROSITY, self._step_porosity, self._can_porosity),
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
        """
        self._result = DynamicWorkflowResult(well=well, config=self.config)

        allowed = set(steps)

        for step_enum, step_fn, _ in self.get_pipeline():
            print(f"step_enum: {step_enum}")
            if step_enum not in allowed:
                continue

            try:
                step_fn(well)
                self._result.steps_executed.append(step_enum.name)
            except Exception as e:
                self._result.specific_steps_skipped[step_enum.name] = str(e)


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
        """
        self._result = DynamicWorkflowResult(well=well, config=self.config)

        # Detect available curves
        available = self._detect_available_curves(well)

        # Execute pipeline in order
        pipeline = [
            (WorkflowStep.TEMPERATURE, self._step_temperature, self._can_temperature),
            (WorkflowStep.BAD_HOLE, self._step_bad_hole, self._can_bad_hole),
            (WorkflowStep.VCLAY, self._step_vclay, self._can_vclay),
            (WorkflowStep.RCL, self._step_rcl, self._can_rcl),
            (WorkflowStep.RMF, self._step_rmf, self._can_rmf),
            (WorkflowStep.RW, self._step_rw, self._can_rw),
            (WorkflowStep.INVASION, self._step_invasion, self._can_invasion),
            (WorkflowStep.HC_CORRECTION, self._step_hc_correction, self._can_hc_correction),
            (WorkflowStep.POROSITY, self._step_porosity, self._can_porosity),
            (WorkflowStep.LITHOLOGY, self._step_lithology, self._can_lithology),
            (WorkflowStep.SATURATION, self._step_saturation, self._can_saturation),
            (WorkflowStep.PERMEABILITY, self._step_permeability, self._can_permeability),
            (WorkflowStep.ROCK_QUALITY, self._step_rock_quality, self._can_rock_quality),
            (WorkflowStep.BVW, self._step_bvw, self._can_bvw),
            (WorkflowStep.MHI, self._step_mhi, self._can_mhi),
            (WorkflowStep.NET_PAY, self._step_net_pay, self._can_net_pay),
        ]

        for step_enum, step_fn, can_fn in pipeline:
            step_name = step_enum.value

            if step_name in self.config.skip_steps:
                self._result.steps_skipped.append(step_name)
                continue

            if can_fn(well):
                try:
                    step_trace = step_fn(well)
                    if step_trace is None:
                        step_trace = StepResult(step=step_name, success=True)
                    step_trace.step = step_name
                    step_trace.success = True
                    self._result.steps_executed.append(step_trace)
                except Exception as e:
                    msg = f"{step_name}: {str(e)}"
                    self._result.warnings.append(msg)
                    self._result.steps_executed.append(
                        StepResult(step=step_name, success=False, message=str(e))
                    )
            else:
                self._result.steps_skipped.append(step_name)

        return self._result

    def _detect_available_curves(self, well: "Well") -> set:
        """Detect which curves are available in the well."""
        available = set()
        for name in ["GR", "SP", "NPHI", "RHOB", "DTC", "DTS", "RDEP", "RT",
                      "RESS", "RESM", "RXO", "PEF", "CAL", "BS", "DRHO",
                      "VCLAY", "PHIE", "PHIT", "SWT-LOG", "SXO-LOG", "PERM", "TEMP"]:
            if well.get(name) is not None:
                available.add(name)
        return available

    def _has_curve(self, well: "Well", *names: str) -> bool:
        """Check if any of the named curves exist."""
        for n in names:
            if hasattr(well, "get") and well.get(n) is not None:
                return True
            if hasattr(well, "__getitem__"):
                try:
                    if well[n] is not None:
                        return True
                except (KeyError, IndexError, TypeError):
                    pass
            if hasattr(well, "has_curve") and well.has_curve(n):
                return True
            if hasattr(well, "curves") and n in well.curves:
                return True
        return False

    # ---- CAN-RUN checks ----

    def _can_temperature(self, well: "Well") -> bool:
        return (
            (self.config.bht_f is not None and self.config.td_ft is not None)
            or _metadata_bht_f(well) is not None
        )

    def _can_bad_hole(self, well: "Well") -> bool:
        return self._has_curve(well, "CAL") or self._has_curve(well, "DRHO")

    def _can_vclay(self, well: "Well") -> bool:
        return self._has_curve(well, "GR")

    def _can_rcl(self, well: "Well") -> bool:
        return self._has_curve(well, "RDEP", "RT") and self._has_curve(well, "VCLAY")

    def _can_lithology(self, well: "Well") -> bool:
        return (self._has_curve(well, "RHOB") and
                self._has_curve(well, "NPHI") and
                self._has_curve(well, "PEF", "DTC") and
                self._has_curve(well, "PHIE") and
                self._has_curve(well, "VCL"))

    def _can_rmf(self, well: "Well") -> bool:
        return (
            self.config.rmf_at_meas_temp is not None
            or self.config.rmf_salinity_ppm is not None
            or self.config.rm is not None
            or well.header.rmf is not None
            or _parameter_float(well, "RMF") is not None
            or _constant_zoned_parameter(well, "RMF") is not None
        )

    def _can_rw(self, well: "Well") -> bool:
        # Always run the Rw step so the workflow can fall back through
        # metadata, salinity, log-based heuristics, and finally the built-in
        # default when no explicit source is available.
        return True

    def _can_invasion(self, well: "Well") -> bool:
        return (self._has_curve(well, "RESS", "RXO") and
                self._has_curve(well, "RDEP", "RT"))

    def _can_hc_correction(self, well: "Well") -> bool:
        return self._has_curve(well, "NPHI") and self._has_curve(well, "RHOB")

    def _can_porosity(self, well: "Well") -> bool:
        return self._has_curve(well, "RHOB", "NPHI", "DTC", "PHIN", "PHIT", "PHID", "PHIS")

    def _can_saturation(self, well: "Well") -> bool:
        return (self._has_curve(well, "RDEP", "RT") and
                (self._has_curve(well, "PHIE") or self._has_curve(well, "RHOB")) and
                self._result.rw_computed is not None)

    def _can_permeability(self, well: "Well") -> bool:
        return self._has_curve(well, "PHIE") or self._has_curve(well, "PHIT")

    def _can_rock_quality(self, well: "Well") -> bool:
        return self._has_curve(well, "PERM") and self._has_curve(well, "PHIE")

    def _can_bvw(self, well: "Well") -> bool:
        return self._has_curve(well, "PHIE") and self._has_curve(well, "SW", "SWA", "SWT-LOG")

    def _can_mhi(self, well: "Well") -> bool:
        return self._has_curve(well, "SW", "SWA", "SWT-LOG") and self._has_sxo_curve(well)

    def _can_net_pay(self, well: "Well") -> bool:
        return self._has_curve(well, "PHIE") and self._has_curve(well, "SW", "SWA", "SWT-LOG")

    # ---- STEP implementations ----

    def _step_temperature(self, well: "Well") -> StepResult:
        from petrophys.environmental.temperature.temp_profile import TemperatureProfile

        trace = StepResult(step="temperature", success=True, method="linear_gradient_from_BHT")
        bht_f = self.config.bht_f
        td_ft = self.config.td_ft

        if bht_f is None:
            bht_f = _metadata_bht_f(well)
            if bht_f is not None:
                trace.method = "constant_BHT_metadata"

        if td_ft is None and well.header.td is not None:
            td_ft = float(well.header.td)

        trace.parameters = {
            "surface_temp_f": self.config.surface_temp_f,
            "bht_f": bht_f,
            "td_ft": td_ft,
        }

        if bht_f is None or td_ft is None:
            temp_f = self._get_avg_formation_temp(well)
            depth_curve = well.get("DEPT") or well.get("DEPTH")
            depth_data = depth_curve.data if depth_curve is not None else well.depth
            if depth_data is not None:
                temp_array = np.full_like(np.asarray(depth_data, dtype=float), temp_f, dtype=float)
                well.add_curve("TEMP", temp_array, "degF", "Formation Temperature")
                trace.output_curves = ["TEMP"]
                trace.decisions.append(f"Constant temperature from metadata/default = {temp_f:.1f} F")
            return trace

        profile = TemperatureProfile.from_bht(
            bht=bht_f,
            td=td_ft,
            surface_temp=self.config.surface_temp_f,
            bht_unit="F"
        )
        depth_curve = well.get("DEPT") or well.get("DEPTH")
        depth_data = depth_curve.data if depth_curve is not None else well.depth
        if depth_data is not None:
            temp_array = profile.get_temperature(depth_data)
            well.add_curve("TEMP", temp_array, "degF", "Formation Temperature")
            self._result.temperature_gradient = profile.gradient
            trace.output_curves = ["TEMP"]
            trace.parameters["gradient_f_per_ft"] = profile.gradient
            trace.decisions.append(f"Gradient = {profile.gradient:.4f} F/ft")
            trace.decisions.append(f"Temp range: {np.nanmin(temp_array):.1f} - {np.nanmax(temp_array):.1f} F")
        return trace

    def _step_bad_hole(self, well: "Well") -> StepResult:
        from petrophys.qaqc.bad_hole.flag_generator import generate_bad_hole_flags

        trace = StepResult(step="bad_hole", success=True, method="caliper_drho_flags")
        inputs = []

        cali = well.get("CAL")
        bs_curve = well.get("BS")
        drho = well.get("DRHO")
        rhob = well.get("RHOB")
        pef = well.get("PEF")

        bs = bs_curve.data if bs_curve is not None else 8.5
        if cali is not None: inputs.append("CAL")
        if bs_curve is not None: inputs.append("BS")
        if drho is not None: inputs.append("DRHO")
        if rhob is not None: inputs.append("RHOB")
        if pef is not None: inputs.append("PEF")
        trace.input_curves = inputs
        bit_size = float(np.nanmean(bs)) if hasattr(bs, "__len__") and np.any(np.isfinite(bs)) else 8.5 if hasattr(bs, "__len__") else bs

        flags = generate_bad_hole_flags(
            well,
            bit_size=bit_size
        )
        well.add_curve("BAD_HOLE", flags["bad_hole"].astype(float), "", "Bad Hole Flag")
        well.add_curve("BADHOLEFLAG", flags["bad_hole"].astype(float), "", "Bad Hole Flag")

        n_bad = int(np.sum(flags["bad_hole"]))
        trace.output_curves = ["BAD_HOLE", "BADHOLEFLAG"]
        trace.parameters["bit_size_used"] = bit_size
        trace.decisions.append(f"{n_bad} bad-hole samples flagged out of {len(flags['bad_hole'])}")
        return trace

    def _step_vclay(self, well: "Well") -> StepResult:
        from petrophys.clay_volume.vcl_module import VclModule

        trace = StepResult(step="vclay", success=True)
        decisions = []

        gr_min = self.config.gr_min
        gr_max = self.config.gr_max
        gr_min_source = "user" if gr_min is not None else "auto"
        gr_max_source = "user" if gr_max is not None else "auto"

        if gr_min is None or gr_max is None:
            gr_curve = well.get("GR")
            if gr_curve is not None:
                valid_gr = gr_curve.data[~np.isnan(gr_curve.data)]
                if len(valid_gr) > 0:
                    if gr_min is None:
                        gr_min = float(np.percentile(valid_gr, 5))
                    if gr_max is None:
                        gr_max = float(np.percentile(valid_gr, 95))

        gr_min = gr_min or 20.0
        gr_max = gr_max or 100.0

        use_sp = (self._has_curve(well, "SP") and
                  not self.config.mud_has_clay_or_kcl)
        use_nd = self._has_curve(well, "NPHI") and self._has_curve(well, "RHOB")
        sp_min = self.config.sp_min
        sp_max = self.config.sp_max
        sp_min_source = "user" if sp_min is not None else "auto"
        sp_max_source = "user" if sp_max is not None else "auto"

        if use_sp:
            sp_curve = well.get("SP")
            if sp_curve is not None and (sp_min is None or sp_max is None):
                valid_sp = sp_curve.data[~np.isnan(sp_curve.data)]
                if len(valid_sp) > 0:
                    if sp_max is None:
                        sp_max = float(np.percentile(valid_sp, 5))
                    if sp_min is None:
                        sp_min = float(np.percentile(valid_sp, 95))

        decisions.append(f"GR_min = {gr_min:.1f} ({gr_min_source}, P5 of GR)")
        decisions.append(f"GR_max = {gr_max:.1f} ({gr_max_source}, P95 of GR)")
        decisions.append(f"GR method: {self.config.gr_method}")
        decisions.append(f"SP method: {'enabled' if use_sp else 'disabled'}"
                        + (f" (mud has clay/KCl)" if self.config.mud_has_clay_or_kcl else ""))
        decisions.append(f"N-D method: {'enabled' if use_nd else 'disabled (missing NPHI or RHOB)'}")
        if use_sp and sp_max is not None and sp_min is not None:
            decisions.append(f"SP_clean = {sp_max:.1f} ({sp_max_source}, P5 of SP)")
            decisions.append(f"SP_shale = {sp_min:.1f} ({sp_min_source}, P95 of SP)")

        vcl_module = VclModule(
            gr_min=gr_min, gr_max=gr_max,
            gr_method=self.config.gr_method,
            use_sp=use_sp, use_nd=use_nd,
            sp_min=sp_min, sp_max=sp_max
        )
        result = vcl_module.calculate(well)
        vcl_module.add_to_well(well, result)

        inputs = ["GR"]
        outputs = ["VCL"]
        if use_sp: inputs.append("SP")
        if use_nd: inputs.extend(["NPHI", "RHOB"])
        if result.vcl_gr is not None: outputs.append("VCLGR")
        if result.vcl_sp is not None: outputs.append("VCLSP")
        if result.vcl_nd is not None: outputs.append("VCLND")

        decisions.append(f"Final VCL = min(available methods), range: "
                        f"{np.nanmin(result.vcl):.3f} - {np.nanmax(result.vcl):.3f}")

        trace.method = self.config.gr_method
        trace.input_curves = inputs
        trace.output_curves = outputs
        trace.parameters = {
            "gr_min": gr_min, "gr_max": gr_max,
            "gr_method": self.config.gr_method,
            "use_sp": use_sp, "use_nd": use_nd,
            "sp_min": sp_min, "sp_max": sp_max,
            "sp_min_source": sp_min_source if use_sp else None,
            "sp_max_source": sp_max_source if use_sp else None,
            "nd_clean1_density": 2.71,
            "nd_clean1_neutron": 0.00,
            "nd_clean2_density": 2.03,
            "nd_clean2_neutron": 0.40,
            "nd_clay_density": 2.62,
            "nd_clay_neutron": 0.38,
        }
        trace.decisions = decisions
        return trace

    def _step_rcl(self, well: "Well") -> StepResult:
        from petrophys.clay_volume.rcl import RclModule

        trace = StepResult(step="rcl", success=True, method="mean_Rt_in_shale_zones")
        trace.input_curves = ["RT", "VCLAY"]
        trace.parameters = {"vcl_threshold": 0.9, "default_rcl": 5.0}

        rcl_module = RclModule()
        self._result.rcl_computed = rcl_module.calculate(well)

        trace.parameters["rcl_computed"] = self._result.rcl_computed
        trace.decisions.append(f"Rcl = {self._result.rcl_computed:.2f} ohm.m "
                              f"(mean RT where VCLAY > 0.9)")
        return trace

    def _step_lithology(self, well: "Well") -> StepResult:
        from petrophys.lithology.lithology_module import LithologyModule

        trace = StepResult(step="lithology", success=True, method="multi_mineral_solver")
        trace.input_curves = ["RHOB", "NPHI", "VCL", "PHIE"]
        if well.get("PEF") is not None:
            trace.input_curves.append("PEF")
        if well.get("DTC") is not None:
            trace.input_curves.append("DTC")

        litho_module = LithologyModule()
        result = litho_module.calculate(well)
        litho_module.add_to_well(well, result)

        trace.output_curves = ["VQUAR", "VCALC", "VDOLO", "VANHY", "LITHO"]
        trace.decisions.append("Raw-log multi-mineral solve with VCL/PHIE clay-porosity terms")
        trace.decisions.append("PEF screened for plausibility; DTC used when PEF is absent or invalid")
        return trace

    def _step_rmf(self, well: "Well") -> StepResult:
        from petrophys.resistivity.rmf.rmf_module import RmfModule
        from petrophys.resistivity.rw.salinity_rw import rw_from_salinity

        trace = StepResult(step="rmf", success=True)
        decisions = []

        def _add_rmf_curve(value: float) -> None:
            if well.depth is None:
                return
            rmf_data = np.full(well.n_samples, float(value), dtype=float)
            well.add_curve("RMF", rmf_data, "ohm.m", "Mud Filtrate Resistivity")
            trace.output_curves = ["RMF"]

        if self.config.mud_type == "OBM":
            self._result.rmf_computed = 200.0
            _add_rmf_curve(self._result.rmf_computed)
            trace.method = "OBM_default"
            decisions.append("OBM detected -> Rmf set to 200.0 ohm.m (non-conductive)")
            trace.parameters = {"rmf": 200.0, "mud_type": "OBM"}
            trace.decisions = decisions
            return trace

        temp_f = self._get_avg_formation_temp(well)
        rmf_meas_temp_f = self.config.rmf_meas_temp_f
        if rmf_meas_temp_f == 75.0 and well.header.rm_temp is not None:
            rmf_meas_temp_f = float(well.header.rm_temp)
        metadata_rmf_temp_f = _parameter_temperature_f(well, "RMFT")
        if metadata_rmf_temp_f is not None and self.config.rmf_meas_temp_f == 75.0:
            rmf_meas_temp_f = metadata_rmf_temp_f

        las_rmf = well.header.rmf
        if las_rmf is None:
            las_rmf = _parameter_float(well, "RMF")
        if las_rmf is None:
            las_rmf = _constant_zoned_parameter(well, "RMF")

        if self.config.rmf_at_meas_temp is not None:
            trace.method = "arps_temperature_correction"
            rmf_module = RmfModule()
            result = rmf_module.calculate(
                rmf_surface=self.config.rmf_at_meas_temp,
                temp_measured=self.config.rmf_meas_temp_f,
                temp_formation=temp_f
            )
            self._result.rmf_computed = result.rmf
            _add_rmf_curve(self._result.rmf_computed)
            trace.parameters = {
                "rmf_at_surface": self.config.rmf_at_meas_temp,
                "surface_temp_f": self.config.rmf_meas_temp_f,
                "formation_temp_f": temp_f,
                "rmf_at_formation": result.rmf,
            }
            decisions.append(f"Rmf corrected from {self.config.rmf_at_meas_temp:.3f} ohm.m "
                           f"@ {self.config.rmf_meas_temp_f:.0f}F "
                           f"-> {result.rmf:.4f} ohm.m @ {temp_f:.0f}F (Arps)")
            trace.decisions = decisions
            return trace

        if las_rmf is not None:
            rmf_module = RmfModule()
            if metadata_rmf_temp_f is not None or well.header.rm_temp is not None:
                result = rmf_module.calculate(
                    rmf_surface=float(las_rmf),
                    temp_measured=rmf_meas_temp_f,
                    temp_formation=temp_f,
                )
                self._result.rmf_computed = float(result.rmf)
                _add_rmf_curve(self._result.rmf_computed)
                trace.method = "las_metadata_arps_temperature_correction"
                trace.parameters = {
                    "rmf_at_measured_temp": float(las_rmf),
                    "measured_temp_f": rmf_meas_temp_f,
                    "formation_temp_f": temp_f,
                    "rmf_at_formation": float(result.rmf),
                }
                decisions.append(
                    f"Rmf corrected from LAS metadata: {float(las_rmf):.4f} ohm.m "
                    f"@ {rmf_meas_temp_f:.0f}F -> {float(result.rmf):.4f} ohm.m @ {temp_f:.0f}F"
                )
            else:
                self._result.rmf_computed = float(las_rmf)
                _add_rmf_curve(self._result.rmf_computed)
                trace.method = "las_metadata"
                trace.parameters = {"rmf": float(las_rmf)}
                decisions.append(f"Rmf = {float(las_rmf):.4f} ohm.m (LAS metadata)")
            trace.decisions = decisions
            return trace

        if self.config.rmf_salinity_ppm is not None:
            rmf_at_meas_temp = float(rw_from_salinity(self.config.rmf_salinity_ppm, rmf_meas_temp_f))
            rmf_module = RmfModule()
            result = rmf_module.calculate(
                rmf_surface=rmf_at_meas_temp,
                temp_measured=rmf_meas_temp_f,
                temp_formation=temp_f,
            )
            self._result.rmf_computed = float(result.rmf)
            _add_rmf_curve(self._result.rmf_computed)
            trace.method = "mud_filtrate_salinity_conversion"
            trace.parameters = {
                "mud_filtrate_salinity_ppm": self.config.rmf_salinity_ppm,
                "rmf_at_measured_temp": rmf_at_meas_temp,
                "measured_temp_f": rmf_meas_temp_f,
                "formation_temp_f": temp_f,
                "rmf_at_formation": float(result.rmf),
            }
            decisions.append(
                f"Rmf from CLFL salinity: {self.config.rmf_salinity_ppm:.0f} ppm -> "
                f"{rmf_at_meas_temp:.4f} ohm.m @ {rmf_meas_temp_f:.0f}F -> "
                f"{float(result.rmf):.4f} ohm.m @ {temp_f:.0f}F"
            )
        elif self.config.rm is not None:
            k = 0.75 if self.config.mud_water == "fresh" else 0.85
            rmf_module = RmfModule(rmf_conversion_factor=k)
            result = rmf_module.calculate(
                rm_surface=self.config.rm,
                temp_measured=rmf_meas_temp_f,
                temp_formation=temp_f,
            )
            self._result.rmf_computed = float(result.rmf)
            _add_rmf_curve(self._result.rmf_computed)
            trace.method = f"rm_to_rmf_conversion_factor_{k}"
            trace.parameters = {
                "rm": self.config.rm,
                "conversion_factor": k,
                "mud_water": self.config.mud_water,
                "measured_temp_f": rmf_meas_temp_f,
                "formation_temp_f": temp_f,
                "rmf_at_formation": float(result.rmf),
            }
            decisions.append(
                f"Rmf = Rm * {k} corrected to formation temperature: "
                f"{self.config.rm:.4f} -> {float(result.rmf):.4f} ohm.m"
            )

        trace.decisions = decisions
        return trace

    def _step_rw(self, well: "Well") -> StepResult:
        from petrophys.resistivity.rw.rw_module import RwModule
        from petrophys.resistivity.rw.salinity_rw import rw_from_salinity

        trace = StepResult(step="rw", success=True)
        decisions = []
        temp_f = self._get_avg_formation_temp(well)
        las_rw_temp_f = _parameter_temperature_f(well, "RWT")

        def _finalize_rw(
            method: str,
            value: float,
            parameters: Dict[str, Any],
            inputs: Optional[List[str]] = None,
        ) -> StepResult:
            self._result.rw_computed = float(value)
            if well.depth is not None:
                rw_data = np.full(well.n_samples, self._result.rw_computed, dtype=float)
                well.add_curve("RW", rw_data, "ohm.m", "Formation Water Resistivity")
            trace.method = method
            trace.parameters = parameters
            trace.input_curves = inputs or []
            trace.output_curves = ["RW"]
            trace.decisions = decisions
            return trace

        if self.config.rw is not None:
            if self.config.rw_meas_temp_f is not None:
                from petrophys.resistivity.rw.temp_correction import correct_rw_temperature

                rw_value = float(
                    correct_rw_temperature(self.config.rw, self.config.rw_meas_temp_f, temp_f)
                )
                decisions.append(
                    f"Rw corrected from {self.config.rw:.4f} ohm.m @ {self.config.rw_meas_temp_f:.0f}F "
                    f"-> {rw_value:.4f} ohm.m @ {temp_f:.0f}F"
                )
                return _finalize_rw(
                    "user_provided_temperature_corrected",
                    rw_value,
                    {
                        "rw_at_measured_temp": self.config.rw,
                        "measured_temp_f": self.config.rw_meas_temp_f,
                        "formation_temp_f": temp_f,
                        "rw_at_formation": rw_value,
                        "source_priority": "config.rw",
                    },
                )
            decisions.append(f"Rw = {self.config.rw:.4f} ohm.m (user-provided)")
            return _finalize_rw(
                "user_provided",
                float(self.config.rw),
                {
                    "rw": float(self.config.rw),
                    "source_priority": "config.rw",
                },
            )

        las_rw = _parameter_float(well, "RW")
        if las_rw is None:
            las_rw = _constant_zoned_parameter(well, "RW")
        if las_rw is not None:
            if las_rw_temp_f is not None:
                from petrophys.resistivity.rw.temp_correction import correct_rw_temperature

                rw_value = float(
                    correct_rw_temperature(float(las_rw), las_rw_temp_f, temp_f)
                )
                decisions.append(
                    f"Rw corrected from LAS metadata: {float(las_rw):.4f} ohm.m @ {las_rw_temp_f:.0f}F "
                    f"-> {rw_value:.4f} ohm.m @ {temp_f:.0f}F"
                )
                return _finalize_rw(
                    "las_metadata_temperature_corrected",
                    rw_value,
                    {
                        "rw_at_measured_temp": float(las_rw),
                        "measured_temp_f": las_rw_temp_f,
                        "formation_temp_f": temp_f,
                        "rw_at_formation": rw_value,
                        "source_priority": "las.RW",
                    },
                )
            decisions.append(f"Rw = {float(las_rw):.4f} ohm.m (LAS metadata)")
            return _finalize_rw(
                "las_metadata",
                float(las_rw),
                {
                    "rw": float(las_rw),
                    "source_priority": "las.RW",
                },
            )

        if self.config.salinity_ppm is not None:
            rw_value = float(rw_from_salinity(self.config.salinity_ppm, temp_f))
            decisions.append(f"Rw from salinity: {self.config.salinity_ppm:.0f} ppm "
                           f"@ {temp_f:.0f}F -> Rw = {rw_value:.4f} ohm.m")
            return _finalize_rw(
                "salinity_conversion",
                rw_value,
                {
                    "salinity_ppm": self.config.salinity_ppm,
                    "temperature_f": temp_f,
                    "rw_computed": rw_value,
                    "source_priority": "salinity_ppm",
                },
            )

        if self._result.rmf_computed is not None:
            decisions.append(
                f"Rw direct source unavailable -> using available Rmf = {self._result.rmf_computed:.4f} ohm.m for heuristic estimation"
            )

            if self._has_curve(well, "SP"):
                if self.config.mud_has_clay_or_kcl:
                    decisions.append("SP method skipped because mud_has_clay_or_kcl=True")
                else:
                    try:
                        rw_result = RwModule(method="sp").calculate(
                            well,
                            rmf=self._result.rmf_computed,
                            temperature=temp_f,
                        )
                        rw_value = float(np.nanmedian(np.asarray(rw_result.rw, dtype=float)))
                        if not np.isfinite(rw_value):
                            raise ValueError("SP-derived Rw is not finite")
                        decisions.append(
                            f"Rw via SP method ({rw_result.method}): {rw_value:.4f} ohm.m"
                        )
                        return _finalize_rw(
                            rw_result.method,
                            rw_value,
                            {
                                "rmf": self._result.rmf_computed,
                                "temperature_f": temp_f,
                                "rw_computed": rw_value,
                                "source_priority": "rmf -> SP",
                                **rw_result.metadata,
                            },
                            ["SP"],
                        )
                    except Exception as exc:
                        message = f"Rw SP heuristic failed: {exc}"
                        self._result.warnings.append(message)
                        decisions.append(message)
            else:
                decisions.append("SP method skipped because SP curve is unavailable")

            try:
                rw_result = RwModule(method="rxo_rt").calculate(
                    well,
                    rmf=self._result.rmf_computed,
                    temperature=temp_f,
                )
                rw_value = float(np.nanmedian(np.asarray(rw_result.rw, dtype=float)))
                if not np.isfinite(rw_value):
                    raise ValueError("Rxo/Rt-derived Rw is not finite")
                decisions.append(
                    f"Rw via log method ({rw_result.method}): {rw_value:.4f} ohm.m"
                )
                return _finalize_rw(
                    rw_result.method,
                    rw_value,
                    {
                        "rmf": self._result.rmf_computed,
                        "temperature_f": temp_f,
                        "rw_computed": rw_value,
                        "source_priority": "rmf -> rxo_rt",
                        **rw_result.metadata,
                    },
                    ["RSHA", "RDEP"],
                )
            except Exception as exc:
                message = f"Rw log heuristic failed: {exc}"
                self._result.warnings.append(message)
                decisions.append(message)
        else:
            decisions.append("Rw heuristic methods skipped because Rmf is unavailable")

        self._result.warnings.append("Rw could not be determined. Using default 0.05 ohm.m")
        decisions.append(
            "All Rw priority sources failed (config/LAS RW, salinity, SP, Rxo/Rt) -> default 0.05"
        )
        return _finalize_rw(
            "default_fallback",
            0.05,
            {
                "rw_default": 0.05,
                "formation_temp_f": temp_f,
                "source_priority": "default",
            },
        )

    def _step_invasion(self, well: "Well") -> StepResult:
        from petrophys.resistivity.invasion.invasion_module import InvasionModule

        trace = StepResult(step="invasion", success=True, method="rxo_rt_ratio")
        invasion_module = InvasionModule()
        result = invasion_module.analyze(well)
        well.add_curve("INVASION_RATIO", result.ratio, "", "Invasion Ratio Rxo/Rt")
        trace.input_curves = ["RT", "RESS"]
        trace.output_curves = ["INVASION_RATIO"]
        return trace

    def _step_hc_correction(self, well: "Well") -> StepResult:
        from petrophys.petrophysics.hc_correction.hc_correction_module import HcCorrectionModule
        from petrophys.base.constants import MatrixProperties

        trace = StepResult(step="hc_correction", success=True, method=f"{self.config.hc_type}_correction")
        decisions = []

        rho_ma = self.config.rho_ma
        rho_ma_source = "user"
        if rho_ma is None:
            try:
                rho_ma = MatrixProperties.get(self.config.matrix_type).density
                rho_ma_source = f"MatrixProperties[{self.config.matrix_type}]"
            except KeyError:
                rho_ma = 2.65
                rho_ma_source = "default"
        decisions.append(f"rho_ma = {rho_ma} g/cm3 (source: {rho_ma_source})")

        hc_module = HcCorrectionModule(
            hc_type=self.config.hc_type, rho_ma=rho_ma,
            rho_fl=self.config.rho_fl, phi_cl=self.config.phi_cl,
            rho_cl=self.config.rho_cl
        )
        result = hc_module.calculate(well)
        hc_module.add_to_well(well, result)

        trace.input_curves = ["NPHI", "RHOB"]
        trace.output_curves = ["NPHI_HC", "RHOB_HC"]
        trace.parameters = {"hc_type": self.config.hc_type, "rho_ma": rho_ma,
                           "rho_fl": self.config.rho_fl, "phi_cl": self.config.phi_cl,
                           "rho_cl": self.config.rho_cl}
        trace.decisions = decisions
        return trace

    def _step_porosity(self, well: "Well") -> StepResult:
        from petrophys.base.constants import MatrixProperties
        from petrophys.petrophysics.porosity.density_porosity import density_porosity
        from petrophys.petrophysics.porosity.effective_porosity import effective_porosity
        from petrophys.petrophysics.porosity.neutron_porosity import neutron_porosity
        from petrophys.petrophysics.porosity.sonic_porosity import secondary_porosity, sonic_porosity
        from petrophys.petrophysics.porosity.total_porosity import total_porosity

        trace = StepResult(step="porosity", success=True, method="component_based")
        decisions = []
        inputs = []
        outputs = []

        try:
            props = MatrixProperties.get(self.config.matrix_type)
            rho_ma = self.config.rho_ma or props.density
            dt_ma = self.config.dt_ma or props.slowness
            nphi_ma = self.config.nphi_ma if self.config.nphi_ma is not None else props.nphi
        except KeyError:
            rho_ma = self.config.rho_ma or 2.65
            dt_ma = self.config.dt_ma or 55.5
            nphi_ma = self.config.nphi_ma if self.config.nphi_ma is not None else -0.02

        nphi_curve = well.get("NPHI")
        rhob_curve = well.get("RHOB")
        dtc_curve = well.get("DTC")
        existing_phin_curve = well.get("PHIN")
        existing_phid_curve = well.get("PHID")
        existing_phis_curve = well.get("PHIS")
        existing_phit_curve = well.get("PHIT")
        vcl_curve = well.get("VCL") or well.get("VCLAY")

        phin = None
        phid = None
        phis = None
        phit = None
        phie_basis = None
        phie_basis_name = None
        phie = None

        if nphi_curve is not None:
            inputs.append("NPHI")
            phin = neutron_porosity(nphi_curve.data, nphi_matrix=nphi_ma)
            well.add_curve("PHIN", phin, "v/v", "Neutron Porosity")
            outputs.append("PHIN")
            decisions.append(f"PHIN from NPHI using matrix neutron response {nphi_ma:.3f}")
        elif existing_phin_curve is not None:
            phin = existing_phin_curve.data
            inputs.append("PHIN")
            decisions.append("PHIN reused from existing well curve")

        if rhob_curve is not None:
            inputs.append("RHOB")
            phid = density_porosity(rhob_curve.data, rho_ma, self.config.rho_fl)
            well.add_curve("PHID", phid, "v/v", "Density Porosity")
            outputs.append("PHID")
            decisions.append("PHID = (rho_ma - RHOB) / (rho_ma - rho_fl)")
        elif existing_phid_curve is not None:
            phid = existing_phid_curve.data
            inputs.append("PHID")
            decisions.append("PHID reused from existing well curve")

        if dtc_curve is not None:
            inputs.append("DTC")
            phis = sonic_porosity(dtc_curve.data, matrix=self.config.matrix_type, dtc_fluid=self.config.dt_fl)
            well.add_curve("PHIS", phis, "v/v", "Sonic Porosity")
            outputs.append("PHIS")
            decisions.append("PHIS from Wyllie time-average equation")
        elif existing_phis_curve is not None:
            phis = existing_phis_curve.data
            inputs.append("PHIS")
            decisions.append("PHIS reused from existing well curve")

        if phid is not None and phin is not None:
            phit = total_porosity(phin, phid)
            phie_basis = phit
            phie_basis_name = "PHIT"
            decisions.append("PHIT = adaptive(PHIN, PHID)")
        elif phid is not None:
            phie_basis = phid
            phie_basis_name = "PHID"
            decisions.append("Density-only porosity: PHID computed, PHIT not created")
        elif phin is not None:
            phie_basis = phin
            phie_basis_name = "PHIN"
            decisions.append("Neutron-only porosity: PHIN computed, PHIT not created")
        elif phis is not None:
            phie_basis = phis
            phie_basis_name = "PHIS"
            decisions.append("Sonic-only porosity: PHIS computed, PHIT not created")
        else:
            if existing_phit_curve is not None:
                phit = existing_phit_curve.data
                phie_basis = phit
                phie_basis_name = "PHIT"
                inputs.append("PHIT")
                decisions.append("PHIT reused from existing well curve")

        if phit is not None:
            well.add_curve("PHIT", phit, "v/v", "Total Porosity")
            if "PHIT" not in outputs:
                outputs.append("PHIT")

        if phie_basis is not None:
            if vcl_curve is not None:
                inputs.append(vcl_curve.mnemonic)
                vcl = vcl_curve.data
            else:
                vcl = np.zeros(well.n_samples, dtype=float)

            phie = effective_porosity(phie_basis, vcl, self.config.phi_cl)
            well.add_curve("PHIE", phie, "v/v", "Effective Porosity")
            outputs.append("PHIE")
            if vcl_curve is not None:
                decisions.append(
                    f"PHIE = {phie_basis_name} - {vcl_curve.mnemonic} * phi_clay ({self.config.phi_cl})"
                )
            else:
                decisions.append(f"PHIE = {phie_basis_name} (no clay curve available)")

        if phis is not None and phit is not None:
            phi_sec = secondary_porosity(phis, phit)
            well.add_curve("PHI_SEC", phi_sec, "v/v", "Secondary Porosity Index")
            outputs.append("PHI_SEC")
            decisions.append("PHI_SEC = PHIT - PHIS")

        trace.input_curves = inputs
        trace.output_curves = outputs
        trace.parameters = {
            "matrix_type": self.config.matrix_type,
            "nphi_ma": nphi_ma,
            "rho_ma": rho_ma,
            "rho_fl": self.config.rho_fl,
            "dt_ma": dt_ma,
            "dt_fl": self.config.dt_fl,
            "phi_clay": self.config.phi_cl,
        }
        trace.decisions = decisions
        return trace

    def _step_saturation(self, well: "Well") -> StepResult:
        from petrophys.petrophysics.saturation.sw_module import SaturationModule

        trace = StepResult(step="saturation", success=True)

        sw_module = SaturationModule(
            a=self.config.a, m=self.config.m, n=self.config.n,
            model=self.config.sw_model, vcl_cutoff=self.config.vcl_cutoff
        )
        lithology = self._build_lithology_log(well)

        result = sw_module.calculate(
            well, rw=self._result.rw_computed, rcl=self._result.rcl_computed,
            salinity=self.config.salinity_ppm, rmf=self._result.rmf_computed,
            lithology=lithology, cec=self.config.cec, b_coeff=self.config.b_coeff,
            rho_ma=self.config.rho_ma or 2.65, m_star=self.config.m_star,
            rwb=self.config.rwb, phi_cl=self.config.phi_cl,
            rtsh=self.config.rtsh, vsh_exp=self.config.vsh_exp,
            shale_exp=self.config.shale_exp,
        )

        well.add_curve("SW", result.sw, "v/v", f"Water Saturation ({result.model})")
        well.add_curve("SWA", result.sw, "v/v", f"Water Saturation ({result.model})")
        outputs = ["SW", "SWA"]
        if result.sxo is not None:
            well.add_curve("SXO", result.sxo, "v/v", "Flushed Zone Water Saturation")
            well.add_curve("SWXOA", result.sxo, "v/v", "Flushed Zone Water Saturation")
            well.add_curve(f"SXO-{result.model}", result.sxo, "v/v", "Flushed Zone Water Saturation")
            outputs.extend(["SXO", "SWXOA", f"SXO-{result.model}"])

        trace.method = result.model
        trace.input_curves = ["RT", "PHIE", "VCLAY"]
        trace.output_curves = outputs
        trace.parameters = {
            "sw_model": result.model,
            "sw_model_requested": self.config.sw_model,
            "a": self.config.a, "m": self.config.m, "n": self.config.n,
            "rw": self._result.rw_computed,
            "rcl": self._result.rcl_computed,
            "rtsh": self.config.rtsh if self.config.rtsh is not None else self._result.rcl_computed,
            "vsh_exp": self.config.vsh_exp,
            "shale_exp": self.config.shale_exp,
            "vcl_cutoff_for_routing": self.config.vcl_cutoff,
            "lithology_type": self.config.lithology if isinstance(lithology, str) else "per-depth array",
        }
        trace.decisions.append(f"Model selected: {result.model}")
        if self.config.sw_model == "auto":
            trace.decisions.append("Auto-routing: depth-by-depth model selection based on VCLAY and lithology")
        trace.decisions.append(f"Archie params: a={self.config.a}, m={self.config.m}, n={self.config.n}")
        trace.decisions.append(f"Rw = {self._result.rw_computed:.4f}, Rcl = {self._result.rcl_computed}")
        if np.any(np.isfinite(result.sw)):
            trace.decisions.append(f"Sw range: {np.nanmin(result.sw):.3f} - {np.nanmax(result.sw):.3f}")
        else:
            trace.decisions.append("Sw range: all values are NaN")
        return trace

    def _step_permeability(self, well: "Well") -> StepResult:
        from petrophys.petrophysics.permeability.perm_module import PermeabilityModule

        trace = StepResult(step="permeability", success=True, method=self.config.perm_method)
        decisions = []

        perm_module = PermeabilityModule(
            method=self.config.perm_method,
            fallback_method=self.config.perm_fallback_method,
        )

        result = perm_module.calculate(
            well,
            swi=self.config.swirr,
            peer_wells=self.config.peer_wells,
            formation_tops=self.config.formation_tops,
        )
        empirical_perm = result.metadata.get("empirical_perm")
        empirical_method = result.metadata.get("empirical_method") or self.config.perm_fallback_method
        perm_output = empirical_perm if empirical_perm is not None else result.perm
        perm_desc = empirical_method if empirical_perm is not None else result.method
        well.add_curve("PERM", perm_output, "mD", f"Permeability ({perm_desc})")

        outputs = ["PERM"]
        phi_corecal = result.metadata.get("phi_corecal")
        perm_core = result.metadata.get("perm_core")
        if phi_corecal is not None:
            well.add_curve(
                "PHIE_CORECAL",
                phi_corecal,
                "v/v",
                "Permeability input porosity from core calibration",
            )
            outputs.append("PHIE_CORECAL")
        if perm_core is not None:
            well.add_curve(
                "PERM_CC",
                perm_core,
                "mD",
                "Permeability from core-calibrated poro-perm transform",
            )
            outputs.append("PERM_CC")
        if isinstance(result.swi_used, np.ndarray) and result.swi_source == "apparent_buckles":
            well.add_curve("SWIRR_APP", result.swi_used, "v/v", "Apparent Irreducible Water Saturation")
            outputs.append("SWIRR_APP")
        if self._has_curve(well, "PHIS") and self._has_curve(well, "PHID"):
            from petrophys.petrophysics.permeability.impermeable_flag import flag_impermeable
            phis = well.get("PHIS").data
            phid = well.get("PHID").data
            nphi = well.get("NPHI")
            phi_n = nphi.data if nphi is not None else phid
            imperm = flag_impermeable(phis, phid, phi_n)
            well.add_curve("IMPERM_FLAG", imperm.astype(float), "", "Impermeable Flag")
            outputs.append("IMPERM_FLAG")
            decisions.append(f"Impermeable flag: {int(np.sum(imperm))} samples flagged")

        trace.input_curves = [result.metadata.get("porosity_curve", "PHIE")]
        trace.output_curves = outputs
        trace.parameters = {
            "method": self.config.perm_method,
            "fallback_method": self.config.perm_fallback_method,
            "swirr_source": result.swi_source,
            "bvw_irr": result.bvw_irr,
            "strategy": result.metadata.get("strategy"),
            "same_well_core_models": result.metadata.get("same_well_core_models"),
            "peer_core_models": result.metadata.get("peer_core_models"),
            "peer_wells_considered": result.metadata.get("peer_wells_considered"),
            "empirical_method": empirical_method,
            "empirical_error": result.metadata.get("empirical_error"),
        }
        if result.metadata.get("core_sections_count"):
            decisions.append(
                f"Core-calibrated sections: {result.metadata['core_sections_count']} samples"
            )
        for model in result.metadata.get("core_models", []):
            source = model.get("source_well") or "unknown"
            formation = model.get("source_formation") or "all_depths"
            source_type = model.get("source_type") or "core"
            decisions.append(
                f"{source_type}: {source} / {formation} (n={model.get('n_core_points')}, "
                f"perm R2={model.get('perm_r2'):.3f})"
            )
        if np.isscalar(result.swi_used):
            trace.parameters["swirr"] = float(result.swi_used)
            decisions.append(f"Swirr = {float(result.swi_used):.3f} ({result.swi_source})")
        elif isinstance(result.swi_used, np.ndarray):
            decisions.append(f"Swirr resolved as curve ({result.swi_source})")
            if result.bvw_irr is not None:
                decisions.append(f"Apparent BVWirr = {result.bvw_irr:.4f} v/v")
        else:
            decisions.append("Swirr not used because permeability came from core calibration")
        decisions.append(f"Empirical permeability curve stored as PERM ({perm_desc})")
        if perm_core is not None:
            decisions.append("Core-corrected permeability stored as PERM_CC")
        decisions.append(f"Permeability strategy: {result.method}")
        decisions.append(f"Perm range: {np.nanmin(perm_output):.3f} - {np.nanmax(perm_output):.1f} mD")
        trace.decisions = decisions
        return trace

    def _step_rock_quality(self, well: "Well") -> StepResult:
        from petrophys.reservoir.rock_quality.rock_quality_module import RockQualityModule

        trace = StepResult(step="rock_quality", success=True, method="RQI_FZI_phi_z")
        rq_module = RockQualityModule()
        result = rq_module.calculate(well)
        rq_module.add_to_well(well, result)
        trace.input_curves = ["PERM", "PHIE"]
        trace.output_curves = ["RQI", "FZI", "PHI_Z"]
        trace.decisions.append("RQI = sqrt(K/PHIE), PHI_Z = PHIE/(1-PHIE), FZI = RQI/PHI_Z")
        return trace

    def _step_bvw(self, well: "Well") -> StepResult:
        from petrophys.reservoir.bvw.bvw_module import calculate_bvw

        phie = well.get("PHIE").data
        sw_curve = self._first_curve(well, "SW", "SWA", "SWT-LOG")
        sw = sw_curve.data
        bvw = calculate_bvw(phie, sw)
        well.add_curve("BVW", bvw, "v/v", "Bulk Volume Water")

        trace = StepResult(step="bvw", success=True, method="PHIE_x_SW")
        trace.input_curves = ["PHIE", sw_curve.mnemonic]
        trace.output_curves = ["BVW"]
        trace.decisions.append(f"BVW = PHIE * {sw_curve.mnemonic}")
        return trace

    def _step_mhi(self, well: "Well") -> StepResult:
        from petrophys.reservoir.mhi.mhi_module import MhiModule

        sw_curve = self._first_curve(well, "SW", "SWA", "SWT-LOG")
        sxo_curve = self._first_sxo_curve(well)
        mhi_module = MhiModule()
        mhi = mhi_module.calculate(well, sxo=sxo_curve.data, sw=sw_curve.data)
        well.add_curve("MHI", mhi, "", "Moveable Hydrocarbon Index")

        trace = StepResult(step="mhi", success=True, method="moveable_HC_index")
        trace.input_curves = [sw_curve.mnemonic, sxo_curve.mnemonic]
        trace.output_curves = ["MHI"]
        trace.decisions.append(f"MHI = {sw_curve.mnemonic} / {sxo_curve.mnemonic}")
        return trace

    def _step_net_pay(self, well: "Well") -> StepResult:
        from petrophys.reservoir.hydrocarbon_analysis import HydrocarbonAnalysis
        from petrophys.reservoir.net_pay.net_pay_module import NetPayModule

        trace = StepResult(step="net_pay", success=True, method="cutoff_flags")
        lithology = self._build_lithology_log(well)

        np_module = NetPayModule(
            vcl_cutoff=self.config.vcl_cutoff,
            phie_cutoff=self.config.phie_cutoff,
            sw_cutoff=self.config.sw_cutoff,
            lithology=lithology
        )
        result = np_module.calculate(well)
        np_module.add_to_well(well, result)

        well.add_curve("RESFLAG", result.net_reservoir.astype(float), "", "Reservoir Flag")

        vcl_curve = well.get("VCL") or well.get("VCLAY")
        if vcl_curve is not None:
            vcl_data = np.asarray(vcl_curve.data, dtype=float)
        else:
            vcl_data = np.zeros(well.n_samples, dtype=float)
        seal_flag = np.full(vcl_data.shape, np.nan, dtype=float)
        valid_vcl = np.isfinite(vcl_data)
        seal_flag[valid_vcl] = (vcl_data[valid_vcl] >= 0.8).astype(float)
        well.add_curve("SEALFLAG", seal_flag, "", "Seal Flag")

        hc_result = HydrocarbonAnalysis().analyze(well, add_curves=True)

        sw_curve = self._first_curve(well, "SW", "SWA", "SWT-LOG")
        trace.input_curves = ["VCL", "PHIE", sw_curve.mnemonic]
        trace.output_curves = ["NET_ROCK", "NET_RES", "NET_PAY", "RESFLAG", "SEALFLAG"]
        if well.get("CROSSOVER_FLAG") is not None:
            trace.output_curves.append("CROSSOVER_FLAG")
        trace.parameters = {
            "vcl_cutoff": self.config.vcl_cutoff,
            "phie_cutoff": self.config.phie_cutoff,
            "sw_cutoff": self.config.sw_cutoff,
            "lithology": self.config.lithology if isinstance(lithology, str) else "per-depth",
            "seal_vcl_cutoff": 0.8,
        }
        n_rock = int(np.nansum(well.get("NET_ROCK").data > 0.5)) if well.get("NET_ROCK") else 0
        n_res = int(np.nansum(well.get("NET_RES").data > 0.5)) if well.get("NET_RES") else 0
        n_pay = int(np.nansum(well.get("NET_PAY").data > 0.5)) if well.get("NET_PAY") else 0
        n_seal = int(np.nansum(seal_flag > 0.5))
        step_m = well.step or 0.15
        trace.decisions.append(f"NET_ROCK: VCL < {self.config.vcl_cutoff} -> {n_rock} samples ({n_rock*step_m:.1f} m)")
        trace.decisions.append(f"NET_RES:  + PHIE > {self.config.phie_cutoff} -> {n_res} samples ({n_res*step_m:.1f} m)")
        trace.decisions.append(f"NET_PAY:  + {sw_curve.mnemonic} < {self.config.sw_cutoff} -> {n_pay} samples ({n_pay*step_m:.1f} m)")
        trace.decisions.append(f"SEALFLAG: VCL >= 0.8 -> {n_seal} samples ({n_seal*step_m:.1f} m)")
        if well.get("CROSSOVER_FLAG") is not None:
            n_crossover = int(np.nansum(np.asarray(well.get("CROSSOVER_FLAG").data, dtype=float) > 0.5))
            trace.decisions.append(
                f"CROSSOVER_FLAG from hydrocarbon analysis -> {n_crossover} samples ({n_crossover*step_m:.1f} m)"
            )
        trace.decisions.extend(
            f"Hydrocarbon analysis: {section_name}"
            for section_name in hc_result.sections.keys()
        )
        return trace

    # ---- Helpers ----

    def _first_curve(self, well: "Well", *names: str):
        for name in names:
            curve = well.get(name)
            if curve is not None:
                return curve
        raise ValueError(f"None of the requested curves are available: {', '.join(names)}")

    def _has_sxo_curve(self, well: "Well") -> bool:
        if self._has_curve(well, "SXO", "SWXOA", "SXO-LOG"):
            return True
        return any(name.upper().startswith("SXO-") for name in well.curves)

    def _first_sxo_curve(self, well: "Well"):
        for name in ("SXO", "SWXOA", "SXO-LOG"):
            curve = well.get(name)
            if curve is not None:
                return curve
        for name in well.curves:
            if name.upper().startswith("SXO-"):
                curve = well.get(name)
                if curve is not None:
                    return curve
        raise ValueError("No SXO curve is available")

    def _build_lithology_log(self, well: "Well") -> Union[str, np.ndarray]:
        """Build per-depth lithology from CSV or return global string."""
        if self._lithology_log is not None:
            return self._lithology_log

        if self.config.lithology_csv is not None:
            from petrophys.lithology.lithology_log import (
                load_lithology_csv, build_lithology_log
            )
            intervals = load_lithology_csv(self.config.lithology_csv)
            depth = well.depth
            self._lithology_log = build_lithology_log(
                depth, intervals, default_lithology=self.config.lithology
            )
            # Add LITHO_CLASS curve to well
            well.add_curve(
                "LITHO_CLASS", self._lithology_log, "", "Lithology Classification"
            )
            return self._lithology_log

        return self.config.lithology

    def _get_avg_formation_temp(self, well: "Well") -> float:
        """Get average formation temperature."""
        temp = well.get("TEMP")
        if temp is not None:
            return float(np.nanmean(temp.data))

        bht_meta = _metadata_bht_f(well)
        if bht_meta is not None:
            return float(bht_meta)

        if self.config.bht_f is not None:
            return (self.config.surface_temp_f + self.config.bht_f) / 2.0

        return 150.0  # Default estimate


def run_dynamic_workflow(
    well: "Well",
    config: Optional[DynamicWorkflowConfig] = None,
    **kwargs
) -> DynamicWorkflowResult:
    """
    Run the dynamic workflow on a well.

    Parameters
    ----------
    well : Well
        Well object with raw log data
    config : DynamicWorkflowConfig, optional
        Workflow configuration
    **kwargs
        Override config parameters

    Returns
    -------
    DynamicWorkflowResult

    Examples
    --------
    >>> result = run_dynamic_workflow(well, rw=0.05, lithology='clastic')
    >>> print(f"Steps executed: {[s.step for s in result.steps_executed]}")
    >>> print(f"Steps skipped: {result.steps_skipped}")
    """
    if config is None:
        config = DynamicWorkflowConfig(**kwargs)
    else:
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    workflow = DynamicWorkflow(config)
    return workflow.run(well)

def _coerce_to_list(value, field_name: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return value
    raise ValueError(f"{field_name} must be a string or list of strings")

def build_workflow_config(cfg: Dict[str, Any], td_ft: float) -> DynamicWorkflowConfig:
    return DynamicWorkflowConfig(
        bht_f=cfg["bht_f"],
        td_ft=td_ft,

        surface_temp_f=cfg.get("surface_temp_f"),
        mud_type=cfg.get("mud_type"),
        rmf_at_meas_temp=cfg.get("rmf_at_meas_temp"),
        rmf_meas_temp_f=cfg.get("rmf_meas_temp_f"),
        rmf_salinity_ppm=cfg.get("rmf_salinity_ppm"),

        rw=cfg["rw"],
        rw_meas_temp_f=cfg.get("rw_meas_temp_f"),
        matrix_type=cfg["matrix_type"],
        lithology=cfg["lithology"],

        a=cfg.get("a"),
        m=cfg.get("m"),
        n=cfg.get("n"),

        vcl_cutoff=cfg.get("vcl_cutoff"),
        phie_cutoff=cfg.get("phie_cutoff"),
        sw_cutoff=cfg.get("sw_cutoff"),

        gr_method=cfg.get("gr_method"),
        perm_method=cfg.get("perm_method"),
    )

def run_dynamic_workflow_by_level(
    well_names: Union[str, List[str]],
    formation_names: Optional[Union[str, List[str]]] = None,
    formation_configs: Optional[List[Any]] = None,  # Adjust type to your FormationConfig class
) -> Dict[str, Any]:
    try:
        all_well_results = []

        for well in well_names:
            # Load parent well
            parent_well = (
                ascii_reader.read_csv(f, well_name=well, formation_name=formation_names)
                if f.suffix.lower() == ".csv"
                else las_reader.read_las(f, well_name=well)
            )

            df_well = parent_well.to_dataframe()
            if not df_well.empty and "WELLNAME" in df_well.columns:
                df = df_well[df_well["WELLNAME"].astype(str).str.upper() == str(well).upper()].copy()
            else:
                df = df_well.copy()

            if formation_names and "FORMATION" in df.columns:
                df = df[df["FORMATION"].astype(str).str.upper().isin([str(x).upper() for x in formation_names])]

            if df.empty:
                all_well_results.append({"ok": False, "well_name": well, "error": "No data after filtering"})
                continue

            results = []
            has_formation_col = "FORMATION" in df.columns
            formations = sorted(df["FORMATION"].dropna().astype(str).unique().tolist()) if has_formation_col else []

            processed_slices = []  # Collect workflow-modified slices here

            for fm in formations:
                if fm not in formation_configs_by_name:
                    continue

                df_fm = df[df["FORMATION"].astype(str) == fm].copy()
                if df_fm.empty:
                    continue

                fm_top_depth = float(df_fm["DEPTH"].min())
                fm_bottom_depth = float(df_fm["DEPTH"].max())

                interval_well = parent_well.depth_slice(top=fm_top_depth, bottom=fm_bottom_depth)
                cfg = build_workflow_config(formation_configs_by_name[fm], td_ft=interval_well.bottom)
                
                # Run workflow (modifies interval_well in-place with computed curves)
                result = DynamicWorkflow(cfg).run(interval_well)
                processed_slices.append(interval_well)  #  Save modified slice

                print(
                    f"\n=== TRACE | well={well} | level=formation | "
                    f"formation={fm} | depth=[{fm_top_depth}, {fm_bottom_depth}] ==="
                )

                results.append({
                    "run_level": "formation",
                    "formation_name": fm,
                    "depth_range": [fm_top_depth, fm_bottom_depth],
                    "result": result,
                })

            # 🔁 Reconstruct single cohesive well object from all processed intervals
            reconstructed_well = parent_well._reconstruct_well_from_slices_well_obj(processed_slices)

            all_well_results.append({
                "ok": True,
                "well_name": well,
                "n_intervals": len(results),
                "results": results,
                "well_obj": reconstructed_well,  # Now contains all workflow-computed curves
            })

        return {
            "ok": True,
            "n_wells": len(all_well_results),
            "wells": all_well_results,
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}
