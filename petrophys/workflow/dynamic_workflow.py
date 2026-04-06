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
    rm: Optional[float] = None
    mud_water: str = "fresh"  # fresh or salt

    # Formation water
    rw: Optional[float] = None
    salinity_ppm: Optional[float] = None

    # Matrix properties
    matrix_type: str = "sandstone"
    rho_ma: Optional[float] = None
    rho_fl: float = 1.0
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
    perm_method: str = "timur"
    swirr: Optional[float] = None

    # Saturation model (auto = use routing logic)
    sw_model: str = "auto"
    rtsh: Optional[float] = None
    vsh_exp: float = 1.6

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
        pipeline = self.get_pipeline()
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
            self.config.bht_f is not None
            and self.config.td_ft is not None
            and self.config.surface_temp_f is not None
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
                self._has_curve(well, "PEF") and
                self._has_curve(well,"PHIE"))

    def _can_rmf(self, well: "Well") -> bool:
        return self.config.rmf_at_meas_temp is not None or self.config.rm is not None

    def _can_rw(self, well: "Well") -> bool:
        return (self.config.rw is not None or
                self.config.salinity_ppm is not None or
                (self._has_curve(well, "SP") and not self.config.mud_has_clay_or_kcl))

    def _can_invasion(self, well: "Well") -> bool:
        return (self._has_curve(well, "RESS", "RXO") and
                self._has_curve(well, "RDEP", "RT"))

    def _can_hc_correction(self, well: "Well") -> bool:
        return self._has_curve(well, "NPHI") and self._has_curve(well, "RHOB")

    def _can_porosity(self, well: "Well") -> bool:
        return self._has_curve(well, "RHOB")

    def _can_saturation(self, well: "Well") -> bool:
        return (self._has_curve(well, "RDEP", "RT") and
                (self._has_curve(well, "PHIE") or self._has_curve(well, "RHOB")) and
                self._result.rw_computed is not None)

    def _can_permeability(self, well: "Well") -> bool:
        return self._has_curve(well, "PHIE") or self._has_curve(well, "PHIT")

    def _can_rock_quality(self, well: "Well") -> bool:
        return self._has_curve(well, "PERM") and self._has_curve(well, "PHIE")

    def _can_bvw(self, well: "Well") -> bool:
        return self._has_curve(well, "PHIE") and self._has_curve(well, "SWT-LOG")

    def _can_mhi(self, well: "Well") -> bool:
        return self._has_curve(well, "SXO-LOG") and self._has_curve(well, "SWT-LOG")

    def _can_net_pay(self, well: "Well") -> bool:
        return (self._has_curve(well, "PHIE") and self._has_curve(well, "SWT-LOG"))

    # ---- STEP implementations ----

    def _step_temperature(self, well: "Well") -> StepResult:
        from petrophys.environmental.temperature.temp_profile import TemperatureProfile

        trace = StepResult(step="temperature", success=True, method="linear_gradient_from_BHT")
        trace.parameters = {
            "surface_temp_f": self.config.surface_temp_f,
            "bht_f": self.config.bht_f,
            "td_ft": self.config.td_ft,
        }

        profile = TemperatureProfile.from_bht(
            bht=self.config.bht_f,
            td=self.config.td_ft,
            surface_temp=self.config.surface_temp_f,
            bht_unit="F"
        )
        depth = well.depth
        if depth is not None:
            depth_data = np.array(depth.data, dtype=np.float64)
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

        flags = generate_bad_hole_flags(
            well,
            bit_size=float(np.nanmean(bs)) if hasattr(bs, '__len__') else bs
        )
        well.add_curve("BAD_HOLE", flags["bad_hole"].astype(float), "", "Bad Hole Flag")

        n_bad = int(np.sum(flags["bad_hole"]))
        trace.output_curves = ["BAD_HOLE"]
        trace.parameters["bit_size_used"] = float(np.nanmean(bs)) if hasattr(bs, '__len__') else bs
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

        decisions.append(f"GR_min = {gr_min:.1f} ({gr_min_source}, P5 of GR)")
        decisions.append(f"GR_max = {gr_max:.1f} ({gr_max_source}, P95 of GR)")
        decisions.append(f"GR method: {self.config.gr_method}")
        decisions.append(f"SP method: {'enabled' if use_sp else 'disabled'}"
                        + (f" (mud has clay/KCl)" if self.config.mud_has_clay_or_kcl else ""))
        decisions.append(f"N-D method: {'enabled' if use_nd else 'disabled (missing NPHI or RHOB)'}")

        vcl_module = VclModule(
            gr_min=gr_min, gr_max=gr_max,
            gr_method=self.config.gr_method,
            use_sp=use_sp, use_nd=use_nd,
            sp_min=self.config.sp_min, sp_max=self.config.sp_max
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
            "sp_min": self.config.sp_min, "sp_max": self.config.sp_max,
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
        trace.input_curves = ["RHOB", "NPHI", "PEF","PHIE"]

        litho_module = LithologyModule()
        result = litho_module.calculate(well)
        litho_module.add_to_well(well, result)

        trace.output_curves = ["QUARTZ", "CALCITE", "DOLOMITE", "CLAY_VOL"]
        trace.decisions.append("3-mineral + fluid solver (RHOB, NPHI, PEF)")
        return trace

    def _step_rmf(self, well: "Well") -> StepResult:
        from petrophys.resistivity.rmf.rmf_module import RmfModule

        trace = StepResult(step="rmf", success=True)
        decisions = []

        if self.config.mud_type == "OBM":
            self._result.rmf_computed = 200.0
            trace.method = "OBM_default"
            decisions.append("OBM detected -> Rmf set to 200.0 ohm.m (non-conductive)")
            trace.parameters = {"rmf": 200.0, "mud_type": "OBM"}
            trace.decisions = decisions
            return trace

        temp_f = self._get_avg_formation_temp(well)

        if self.config.rmf_at_meas_temp is not None:
            trace.method = "arps_temperature_correction"
            rmf_module = RmfModule()
            result = rmf_module.calculate(
                rmf_surface=self.config.rmf_at_meas_temp,
                temp_measured=self.config.rmf_meas_temp_f,
                temp_formation=temp_f
            )
            self._result.rmf_computed = result.rmf
            trace.parameters = {
                "rmf_at_surface": self.config.rmf_at_meas_temp,
                "surface_temp_f": self.config.rmf_meas_temp_f,
                "formation_temp_f": temp_f,
                "rmf_at_formation": result.rmf,
            }
            decisions.append(f"Rmf corrected from {self.config.rmf_at_meas_temp:.3f} ohm.m "
                           f"@ {self.config.rmf_meas_temp_f:.0f}F "
                           f"-> {result.rmf:.4f} ohm.m @ {temp_f:.0f}F (Arps)")
        elif self.config.rm is not None:
            k = 0.75 if self.config.mud_water == "fresh" else 0.85
            self._result.rmf_computed = self.config.rm * k
            trace.method = f"Rm_conversion_factor_{k}"
            trace.parameters = {"rm": self.config.rm, "conversion_factor": k,
                               "mud_water": self.config.mud_water,
                               "rmf_computed": self._result.rmf_computed}
            decisions.append(f"Rmf = Rm * {k} = {self.config.rm} * {k} = "
                           f"{self._result.rmf_computed:.4f} ({self.config.mud_water} mud)")

        trace.decisions = decisions
        return trace

    def _step_rw(self, well: "Well") -> StepResult:
        trace = StepResult(step="rw", success=True)
        decisions = []

        if self.config.rw is not None:
            self._result.rw_computed = self.config.rw
            trace.method = "user_provided"
            trace.parameters = {"rw": self.config.rw}
            decisions.append(f"Rw = {self.config.rw:.4f} ohm.m (user-provided)")
            trace.decisions = decisions
            return trace

        temp_f = self._get_avg_formation_temp(well)

        if self.config.salinity_ppm is not None:
            from petrophys.resistivity.rw.salinity_rw import rw_from_salinity
            self._result.rw_computed = rw_from_salinity(self.config.salinity_ppm, temp_f)
            trace.method = "salinity_conversion"
            trace.parameters = {"salinity_ppm": self.config.salinity_ppm,
                               "temperature_f": temp_f,
                               "rw_computed": self._result.rw_computed}
            decisions.append(f"Rw from salinity: {self.config.salinity_ppm:.0f} ppm "
                           f"@ {temp_f:.0f}F -> Rw = {self._result.rw_computed:.4f} ohm.m")
            trace.decisions = decisions
            return trace

        if (self._has_curve(well, "SP") and
                not self.config.mud_has_clay_or_kcl and
                self._result.rmf_computed is not None):
            from petrophys.resistivity.rw.sp_rw import estimate_sp_lines, rw_from_ssp
            sp = well.get("SP")
            valid_sp = sp.data[~np.isnan(sp.data)]
            if len(valid_sp) > 0:
                sp_clean, sp_shale = estimate_sp_lines(sp.data)
                ssp = sp_clean - sp_shale
                rw = rw_from_ssp(ssp, self._result.rmf_computed, temp_f)
                self._result.rw_computed = rw
                trace.method = "SP_method"
                trace.input_curves = ["SP"]
                trace.parameters = {
                    "sp_clean_mv": sp_clean,
                    "sp_shale_mv": sp_shale,
                    "ssp_mv": ssp,
                    "rmf": self._result.rmf_computed,
                    "temperature_f": temp_f,
                    "rw_computed": rw
                }
                decisions.append(
                    f"Rw from SP: clean line = {sp_clean:.1f} mV, shale line = {sp_shale:.1f} mV, "
                    f"SSP = {ssp:.1f} mV, Rmf = {self._result.rmf_computed:.4f}, "
                    f"T = {temp_f:.0f}F -> Rw = {rw:.4f} ohm.m"
                )
                trace.decisions = decisions
                return trace

        self._result.warnings.append("Rw could not be determined. Using default 0.05 ohm.m")
        self._result.rw_computed = 0.05
        trace.method = "default_fallback"
        trace.parameters = {"rw_default": 0.05}
        decisions.append("No Rw source available (no user Rw, no salinity, SP unavailable) -> default 0.05")
        trace.decisions = decisions
        return trace

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
            rho_fl=self.config.rho_fl, phi_cl=self.config.phi_cl
        )
        result = hc_module.calculate(well)
        hc_module.add_to_well(well, result)

        trace.input_curves = ["NPHI", "RHOB"]
        trace.output_curves = ["NPHI_HC", "RHOB_HC"]
        trace.parameters = {"hc_type": self.config.hc_type, "rho_ma": rho_ma,
                           "rho_fl": self.config.rho_fl, "phi_cl": self.config.phi_cl}
        trace.decisions = decisions
        return trace

    def _step_porosity(self, well: "Well") -> StepResult:
        from petrophys.petrophysics.porosity.porosity_module import PorosityModule

        trace = StepResult(step="porosity", success=True, method="density_neutron_adaptive")

        por_module = PorosityModule(
            matrix=self.config.matrix_type,
            rhob_fluid=self.config.rho_fl,
            phi_clay=self.config.phi_cl
        )
        result = por_module.calculate(well)
        por_module.add_to_well(well, result)

        inputs = []
        outputs = []
        for c in ["RHOB", "NPHI", "DTC"]:
            if well.get(c) is not None: inputs.append(c)
        if well.get("VCLAY") is not None: inputs.append("VCLAY")
        for c in ["PHID", "PHIT", "PHIE", "PHIS"]:
            if well.get(c) is not None: outputs.append(c)

        trace.input_curves = inputs
        trace.output_curves = outputs
        trace.parameters = {
            "matrix_type": self.config.matrix_type,
            "rho_ma": self.config.rho_ma or {"sandstone": 2.65, "limestone": 2.71, "dolomite": 2.87}.get(self.config.matrix_type, 2.65),
            "rho_fl": self.config.rho_fl,
            "phi_clay": self.config.phi_cl,
        }
        trace.decisions.append(f"Matrix: {self.config.matrix_type}")
        trace.decisions.append(f"PHID = (rho_ma - RHOB) / (rho_ma - rho_fl)")
        if "NPHI" in inputs:
            trace.decisions.append("PHIT = adaptive(NPHI, PHID)")
        trace.decisions.append(f"PHIE = PHIT - VCLAY * phi_clay ({self.config.phi_cl})")
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
        )

        well.add_curve(f"SW-{result.model}", result.sw, "v/v", f"Water Saturation ({result.model})")
        outputs = [f"SW-{result.model}"]
        if result.sxo is not None:
            well.add_curve(f"SXO-{result.model}", result.sxo, "v/v", "Flushed Zone Water Saturation")
            outputs.append(f"SXO-{result.model}")

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
            "vcl_cutoff_for_routing": self.config.vcl_cutoff,
            "lithology_type": self.config.lithology if isinstance(lithology, str) else "per-depth array",
        }
        trace.decisions.append(f"Model selected: {result.model}")
        if self.config.sw_model == "auto":
            trace.decisions.append("Auto-routing: model chosen based on lithology and VCLAY")
        trace.decisions.append(f"Archie params: a={self.config.a}, m={self.config.m}, n={self.config.n}")
        trace.decisions.append(f"Rw = {self._result.rw_computed:.4f}, Rcl = {self._result.rcl_computed}")
        trace.decisions.append(f"Sw range: {np.nanmin(result.sw):.3f} - {np.nanmax(result.sw):.3f}")
        return trace

    def _step_permeability(self, well: "Well") -> StepResult:
        from petrophys.petrophysics.permeability.perm_module import PermeabilityModule

        trace = StepResult(step="permeability", success=True, method=self.config.perm_method)
        decisions = []

        perm_module = PermeabilityModule(method=self.config.perm_method)

        swi = self.config.swirr
        swi_source = "user"
        if swi is None:
            swi_source = "estimated"
            sw_curve = well.get("SWT-LOG")
            net_pay = well.get("NET_PAY")
            if sw_curve is not None and net_pay is not None:
                pay_mask = net_pay.data > 0.5
                if np.any(pay_mask):
                    swi = float(np.nanmean(sw_curve.data[pay_mask]))
            if swi is None:
                swi_source = "not_available"
                decisions.append("Swirr not available (no pay zones yet); using equation defaults")

        if swi is not None:
            decisions.append(f"Swirr = {swi:.3f} ({swi_source})")

        result = perm_module.calculate(well, swi=swi)
        well.add_curve("PERM", result.perm, "mD", f"Permeability ({result.method})")

        outputs = ["PERM"]
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

        trace.input_curves = ["PHIE"]
        trace.output_curves = outputs
        trace.parameters = {"method": self.config.perm_method, "swirr": swi, "swirr_source": swi_source}
        decisions.append(f"Equation: {self.config.perm_method}")
        decisions.append(f"Perm range: {np.nanmin(result.perm):.3f} - {np.nanmax(result.perm):.1f} mD")
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
        sw = well.get("SWT-LOG").data
        bvw = calculate_bvw(phie, sw)
        well.add_curve("BVW", bvw, "v/v", "Bulk Volume Water")

        trace = StepResult(step="bvw", success=True, method="PHIE_x_SW")
        trace.input_curves = ["PHIE", "SWT-LOG"]
        trace.output_curves = ["BVW"]
        trace.decisions.append("BVW = PHIE * SWT-LOG")
        return trace

    def _step_mhi(self, well: "Well") -> StepResult:
        from petrophys.reservoir.mhi.mhi_module import MhiModule

        mhi_module = MhiModule()
        mhi_module.add_to_well(well)

        trace = StepResult(step="mhi", success=True, method="moveable_HC_index")
        trace.input_curves = ["SWT-LOG", "SXO-LOG"]
        trace.output_curves = ["MHI"]
        trace.decisions.append("MHI = SWT-LOG / SXO-LOG")
        return trace

    def _step_net_pay(self, well: "Well") -> StepResult:
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

        trace.input_curves = ["VCLAY", "PHIE", "SWT-LOG"]
        trace.output_curves = ["NET_ROCK", "NET_RES", "NET_PAY"]
        trace.parameters = {
            "vcl_cutoff": self.config.vcl_cutoff,
            "phie_cutoff": self.config.phie_cutoff,
            "sw_cutoff": self.config.sw_cutoff,
            "lithology": self.config.lithology if isinstance(lithology, str) else "per-depth",
        }
        n_rock = int(np.nansum(well.get("NET_ROCK").data > 0.5)) if well.get("NET_ROCK") else 0
        n_res = int(np.nansum(well.get("NET_RES").data > 0.5)) if well.get("NET_RES") else 0
        n_pay = int(np.nansum(well.get("NET_PAY").data > 0.5)) if well.get("NET_PAY") else 0
        step_m = well.step or 0.15
        trace.decisions.append(f"NET_ROCK: VCLAY < {self.config.vcl_cutoff} -> {n_rock} samples ({n_rock*step_m:.1f} m)")
        trace.decisions.append(f"NET_RES:  + PHIE > {self.config.phie_cutoff} -> {n_res} samples ({n_res*step_m:.1f} m)")
        trace.decisions.append(f"NET_PAY:  + SWT-LOG < {self.config.sw_cutoff} -> {n_pay} samples ({n_pay*step_m:.1f} m)")
        return trace

    # ---- Helpers ----

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

        rw=cfg["rw"],
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