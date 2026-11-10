"""
Deprecated fixed-configuration workflow; use pp.interpret or DynamicWorkflow instead.
"""

from typing import TYPE_CHECKING, Optional, List, Dict, Any
from dataclasses import dataclass, field
import numpy as np
import re
import warnings

from petrophys._internal.analysis_helpers import get_curve, get_curve_data_depthwise

if TYPE_CHECKING:
    from petrophys.model.well import Well


def _parameter_float(well: "Well", name: str) -> Optional[float]:
    """Fetch a numeric well parameter if present."""
    value = well.get_parameter(name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _constant_zoned_parameter(well: "Well", prefix: str) -> Optional[float]:
    """Return a single unique zoned LAS parameter value if available."""
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


@dataclass
class WorkflowConfig:
    """Configuration for standard workflow."""

    # Temperature
    surface_temp_f: float = 70.0
    bht_f: float = 200.0
    td_ft: float = 10000.0

    # Mud properties
    rmf_at_meas_temp: float = 0.5
    rmf_meas_temp_f: float = 75.0

    # Formation water
    rw: Optional[float] = None  # None: LAS header RW, salinity_ppm, the SP, else 0.05 ohm.m
    salinity_ppm: Optional[float] = None

    # Matrix properties
    matrix_type: str = "sandstone"
    lithology: str = "clastic"
    # Matrix density (g/cm3) and sonic slowness (us/ft): None takes the
    # values of matrix_type; an explicit value wins.
    matrix_density: Optional[float] = None
    fluid_density: float = 1.0
    matrix_dt: Optional[float] = None
    # Pore-fluid slowness (us/ft): None is 189 (fresh water).
    fluid_dt: Optional[float] = None
    phi_cl: float = 0.15
    rho_cl: float = 2.45

    # Archie parameters
    a: float = 1.0
    m: float = 2.0
    n: float = 2.0

    # Saturation model
    saturation_model: str = "archie"  # archie, simandoux, indonesia, north_sea, nigeria_schlumberger
    # Bound-water resistivity (ohm.m) for Dual Water, which runs only with it;
    # it is not estimated.
    rwb: Optional[float] = None
    rtsh: Optional[float] = None
    vsh_exp: float = 1.6
    shale_exp: float = 1.6
    swirr: Optional[float] = None
    perm_method: str = "auto"
    perm_fallback_method: str = "coates"
    peer_wells: Optional[List[Any]] = None
    formation_tops: Optional[Any] = None
    use_peer_core: bool = False
    mud_has_clay_or_kcl: bool = False

    # Cutoffs (None: the standard set of ``lithology``'s family, STANDARD_CUTOFFS)
    vcl_cutoff: Optional[float] = None
    phie_cutoff: Optional[float] = None
    sw_cutoff: Optional[float] = None

    # VCL parameters
    gr_min: Optional[float] = None  # None: 20 API, or the log minimum where lower
    gr_max: Optional[float] = None  # None: 100 API, or the log maximum where higher
    gr_method: str = "linear"

    # Processing options
    calculate_vcl: bool = True
    calculate_porosity: bool = True
    calculate_sw: bool = True
    calculate_perm: bool = True
    calculate_net_pay: bool = True

    def __post_init__(self) -> None:
        from petrophys.evaluation.net_pay.cutoffs import standard_cutoffs

        # Unset cutoffs follow the lithology family, as petrophys.interpret's do.
        for name, value in standard_cutoffs(self.lithology).items():
            if getattr(self, name) is None:
                setattr(self, name, value)
        models = ("archie", "simandoux", "indonesia", "north_sea", "nigeria_schlumberger", "dual_water")
        if str(self.saturation_model).strip().lower() not in models:
            raise ValueError(
                f"saturation_model must be one of {list(models)}; got {self.saturation_model!r}. "
                "Waxman-Smits needs CEC: use petrophys.interpret(..., sw_model='waxman_smits', cec=...)."
            )


@dataclass
class StepTrace:
    """Trace of a single standard workflow step."""
    step: str
    success: bool = True
    method: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_curves: List[str] = field(default_factory=list)
    output_curves: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    message: str = ""


@dataclass
class WorkflowResult:
    """Results from standard workflow."""

    well: "Well"
    config: WorkflowConfig
    success: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    steps: List[StepTrace] = field(default_factory=list)

    # Computed values
    rw_computed: Optional[float] = None
    temperature_gradient: Optional[float] = None

    def print_trace(self) -> None:
        """Print a human-readable trace of all steps."""
        print(f"\n{'=' * 80}")
        print(f"STANDARD WORKFLOW TRACE - {self.well.name or 'Unknown'}")
        print(f"{'=' * 80}")
        print(f"Overall: {'SUCCESS' if self.success else 'FAILED'} | "
              f"Steps: {len(self.steps)} | Rw = {self.rw_computed}")
        print()
        for st in self.steps:
            status = "OK" if st.success else "FAIL"
            print(f"[{status:4s}] {st.step}")
            if st.method:
                print(f"       Method: {st.method}")
            if st.input_curves:
                print(f"       Inputs: {', '.join(st.input_curves)}")
            if st.output_curves:
                print(f"       Outputs: {', '.join(st.output_curves)}")
            for k, v in st.parameters.items():
                print(f"       {k}: {v}")
            for d in st.decisions:
                print(f"       -> {d}")
            if st.message:
                print(f"       Note: {st.message}")
            print()
        if self.warnings:
            print("Warnings:")
            for w in self.warnings:
                print(f"  - {w}")
        print(f"{'=' * 80}")


class StandardWorkflow:
    """
    Fixed-configuration petrophysics interpretation workflow.

    .. deprecated:: 0.1.0
       Use :class:`petrophys.workflow.DynamicWorkflow`, which checks available
       inputs before routing each interpretation step.

    Workflow steps:
    1. Temperature profile
    2. VCL calculation
    3. Rw determination
    4. Porosity calculation
    5. Water saturation
    6. Permeability
    7. Net pay determination
    """

    def __init__(self, config: Optional[WorkflowConfig] = None):
        warnings.warn(
            "StandardWorkflow is deprecated; use DynamicWorkflow instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.config = config or WorkflowConfig()
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self._core_porosity_generated = False
        # Temperature (degF) the Rw value is given at; None: at formation temperature.
        self._rw_temp_f: Optional[float] = None
        self._input_curves: set = set()
        self._well: Optional["Well"] = None

    def run(self, well: "Well") -> WorkflowResult:
        """
        Run the standard workflow on a well.

        Parameters
        ----------
        well : Well
            Well object with raw log data

        Returns
        -------
        WorkflowResult
            Workflow results with processed well
        """
        self.errors = []
        self.warnings = []
        self._core_porosity_generated = False
        self._rw_temp_f = None
        self._input_curves = set(well.curves)
        self._well = well

        result = WorkflowResult(
            well=well,
            config=self.config
        )

        try:
            # Step 1: Temperature profile
            result.steps.append(self._calculate_temperature(well))

            # Step 2: VCL
            if self.config.calculate_vcl:
                result.steps.append(self._calculate_vcl(well))

            # Step 3: Rw
            rw, rw_trace = self._determine_rw(well)
            result.rw_computed = rw
            result.steps.append(rw_trace)

            # Step 4: Porosity
            if self.config.calculate_porosity:
                result.steps.append(self._calculate_porosity(well))

            # Step 4b: Core correction before saturation can use PHIE_CC
            if self._can_core_correction(well):
                result.steps.append(self._calculate_core_correction(well))

            # Step 5: Water saturation
            if self.config.calculate_sw:
                result.steps.append(self._calculate_saturation(well, rw))

            # Step 6: Permeability
            if self.config.calculate_perm:
                result.steps.append(self._calculate_permeability(well))

            # Step 7: Net pay
            if self.config.calculate_net_pay:
                result.steps.append(self._calculate_net_pay(well))

            for step in result.steps:
                if not step.success:
                    self.errors.append(f"{step.step}: {step.message}")
            result.success = len(self.errors) == 0
            result.errors = self.errors
            result.warnings = self.warnings

        except Exception as e:
            self.errors.append(f"Workflow error: {str(e)}")
            result.success = False
            result.errors = self.errors

        return result

    def _calculate_temperature(self, well: "Well") -> StepTrace:
        """Calculate temperature profile."""
        from petrophys.interpretation.temperature import TemperatureProfile

        trace = StepTrace(step="temperature", method="linear_gradient_from_BHT")
        trace.parameters = {
            "surface_temp_f": self.config.surface_temp_f,
            "bht_f": self.config.bht_f,
            "td_ft": self.config.td_ft,
        }
        try:
            temp_module = TemperatureProfile.from_bht(
                bht=self.config.bht_f,
                td=self.config.td_ft,
                surface_temp=self.config.surface_temp_f,
                bht_unit="F"
            )
            temp_module.add_to_well(well)
            trace.output_curves = ["TEMP"]
            trace.decisions.append(f"Temp gradient computed from BHT={self.config.bht_f}F at TD={self.config.td_ft}ft")
        except Exception as e:
            self.warnings.append(f"Temperature calculation warning: {e}")
            trace.success = False
            trace.message = str(e)
        return trace

    def _calculate_vcl(self, well: "Well") -> StepTrace:
        """Calculate clay volume."""
        from petrophys.interpretation.clay_volume import VclModule

        trace = StepTrace(step="vclay", method="default_VclModule")
        trace.input_curves = ["GR"]
        try:
            vcl_module = VclModule(
                gr_min=self.config.gr_min,
                gr_max=self.config.gr_max,
                gr_method=self.config.gr_method,
            )
            result = vcl_module.calculate(well)
            vcl_module.add_to_well(well, result)
            trace.output_curves = ["VCL"]
            trace.parameters = {
                "gr_min": self.config.gr_min,
                "gr_max": self.config.gr_max,
                "gr_method": self.config.gr_method,
            }
            trace.decisions.append("GR method with configured GR baselines")
        except Exception as e:
            self.warnings.append(f"VCL calculation warning: {e}")
            trace.success = False
            trace.message = str(e)
        return trace

    def _determine_rw(self, well: "Well") -> tuple:
        """Determine Rw. Returns (rw_value, StepTrace)."""
        trace = StepTrace(step="rw")

        if self.config.rw is not None:
            trace.method = "user_provided"
            trace.parameters = {"rw": self.config.rw}
            trace.decisions.append(f"Rw = {self.config.rw} ohm.m (user-provided)")
            return self.config.rw, trace

        las_rw = _parameter_float(well, "RW")
        if las_rw is None:
            las_rw = _constant_zoned_parameter(well, "RW")
        if las_rw is not None:
            from petrophys.workflow.dynamic_workflow import _parameter_temperature_f

            trace.method = "las_metadata"
            trace.parameters = {"rw": float(las_rw)}
            # RWT is the temperature of the header RW; saturation corrects it to TEMP.
            self._rw_temp_f = _parameter_temperature_f(well, "RWT")
            if self._rw_temp_f is not None:
                trace.parameters["rw_meas_temp_f"] = self._rw_temp_f
                trace.decisions.append(
                    f"Rw = {float(las_rw):.4f} ohm.m at {self._rw_temp_f:.0f}F (LAS metadata RW, RWT)"
                )
            else:
                trace.decisions.append(f"Rw = {float(las_rw):.4f} ohm.m (LAS metadata)")
            return float(las_rw), trace

        if self.config.salinity_ppm is not None:
            from petrophys.interpretation.electrical.rw.salinity_rw import rw_from_salinity
            temp = well.get("TEMP")
            if temp is not None:
                avg_temp = float(np.nanmean(temp.data))
            else:
                avg_temp = (self.config.surface_temp_f + self.config.bht_f) / 2

            rw = rw_from_salinity(self.config.salinity_ppm, avg_temp)
            self._rw_temp_f = avg_temp
            trace.method = "salinity_conversion"
            trace.parameters = {"salinity_ppm": self.config.salinity_ppm,
                               "temperature_f": avg_temp, "rw": rw}
            trace.decisions.append(f"Rw from {self.config.salinity_ppm:.0f} ppm "
                                  f"@ {avg_temp:.0f}F = {rw:.4f} ohm.m")
            return rw, trace

        sp = well.get("SP")
        if sp is not None and not self.config.mud_has_clay_or_kcl:
            from petrophys.interpretation.electrical.rw import RwModule
            from petrophys.interpretation.temperature.temp_corrections import arps_equation
            from petrophys.interpretation.temperature.temp_profile import temperature_curve_f
            from petrophys.workflow.dynamic_workflow import _parameter_temperature_f
            try:
                rmf_for_sp = self.config.rmf_at_meas_temp
                rmf_temp_f = self.config.rmf_meas_temp_f
                las_rmf = _parameter_float(well, "RMF")
                if las_rmf is None:
                    las_rmf = _constant_zoned_parameter(well, "RMF")
                if las_rmf is not None:
                    rmf_for_sp = float(las_rmf)
                    las_rmf_temp = _parameter_temperature_f(well, "RMFT")
                    if las_rmf_temp is None:
                        las_rmf_temp = _parameter_temperature_f(well, "MFST")
                    if las_rmf_temp is not None:
                        rmf_temp_f = las_rmf_temp

                # The SP equation takes Rmf at formation temperature.
                formation_temp = temperature_curve_f(well, ["TEMP"])
                rmf_at_formation = rmf_for_sp
                if formation_temp is not None:
                    rmf_at_formation = arps_equation(rmf_for_sp, rmf_temp_f, formation_temp)
                else:
                    self.warnings.append(
                        f"Rmf {rmf_for_sp} ohm.m at {rmf_temp_f:.0f}F used in the SP equation "
                        "without temperature correction: no TEMP curve"
                    )

                rw_module = RwModule(method="sp")
                result = rw_module.calculate(
                    well,
                    rmf=rmf_at_formation,
                    temperature=formation_temp,
                )
                rw_temp = result.metadata.get("rw_temperature_f")
                if formation_temp is not None and rw_temp is not None:
                    self._rw_temp_f = float(rw_temp)
                trace.method = result.method
                trace.input_curves = ["SP"]
                trace.parameters = {
                    "rmf": rmf_for_sp,
                    "rmf_meas_temp_f": rmf_temp_f,
                    "rw": result.rw,
                    **result.metadata,
                }
                trace.decisions.append(f"Rw from SP = {result.rw:.4f} ohm.m")
                return result.rw, trace
            except Exception as e:
                self.warnings.append(f"Rw from SP failed: {e}")

        self.warnings.append("Using default Rw = 0.05 ohm.m")
        trace.method = "default_fallback"
        trace.parameters = {"rw": 0.05}
        trace.decisions.append("No Rw source -> default 0.05 ohm.m")
        return 0.05, trace

    def _calculate_porosity(self, well: "Well") -> StepTrace:
        """Calculate porosity."""
        from petrophys.interpretation.porosity import PorosityModule

        from petrophys.model.constants import DefaultParameters

        trace = StepTrace(step="porosity", method="density_neutron")
        try:
            por_module = PorosityModule(
                matrix=self.config.matrix_type,
                rhob_fluid=self.config.fluid_density,
                phi_clay=self.config.phi_cl
            )
            # Endpoints: the matrix_type values unless the config sets them.
            matrix_density = (
                self.config.matrix_density if self.config.matrix_density is not None
                else por_module.rhob_matrix
            )
            matrix_dt = self.config.matrix_dt if self.config.matrix_dt is not None else por_module.dtc_matrix
            fluid_dt = (
                self.config.fluid_dt if self.config.fluid_dt is not None
                else DefaultParameters.FLUID_SLOWNESS
            )
            trace.parameters = {
                "matrix_type": self.config.matrix_type,
                "matrix_density": matrix_density,
                "fluid_density": self.config.fluid_density,
                "matrix_dt": matrix_dt,
                "fluid_dt": fluid_dt,
                "phi_cl": self.config.phi_cl,
                "endpoint_source": {
                    "matrix_density": "config" if self.config.matrix_density is not None else "matrix_type",
                    "matrix_dt": "config" if self.config.matrix_dt is not None else "matrix_type",
                    "fluid_dt": "config" if self.config.fluid_dt is not None else "default",
                },
            }
            por_module.rhob_matrix = matrix_density
            por_module.dtc_matrix = matrix_dt
            result = por_module.calculate(well)
            if result.phis is not None and self.config.fluid_dt is not None:
                # PorosityModule reads the default fluid slowness; a fluid_dt
                # set in the config replaces it.
                from petrophys.interpretation.porosity.sonic_porosity import (
                    secondary_porosity,
                    wyllie_time_average,
                )

                dtc = get_curve_data_depthwise(well, ["DTC", "DT"])[0]
                result.phis = wyllie_time_average(dtc, matrix_dt, fluid_dt)
                result.phi_sec = secondary_porosity(result.phis, result.phit)
            if result.phis is not None:
                trace.decisions.append(f"PHIS = (DT - {matrix_dt}) / ({fluid_dt} - {matrix_dt})")
            por_module.add_to_well(well, result)

            inputs = [c for c in ["RHOB", "NPHI", "DTC", "VCL"] if well.get(c)]
            outputs = [c for c in ["PHID", "PHIT", "PHIE", "PHIS"] if well.get(c)]
            trace.input_curves = inputs
            trace.output_curves = outputs
            trace.decisions.append(f"PHID = (rho_ma({matrix_density}) - RHOB) / "
                                  f"(rho_ma - rho_fl({self.config.fluid_density}))")
        except Exception as e:
            self.warnings.append(f"Porosity calculation warning: {e}")
            trace.success = False
            trace.message = str(e)
        return trace

    def _can_core_correction(self, well: "Well") -> bool:
        return (
            get_curve(well, ["PHIE", "PHIT"]) is not None
            and get_curve(well, ["PHIC", "CPOR"]) is not None
        )

    def _calculate_core_correction(self, well: "Well") -> StepTrace:
        """Calculate PHIE_CC before saturation when same-well core is available."""
        from petrophys.interpretation.permeability import PermeabilityModule

        trace = StepTrace(step="core_correction", method="same_well_core_correction")
        self._core_porosity_generated = False
        try:
            stale_core_curves = [name for name in ("PHIE_CC", "PERM_CC") if well.get(name) is not None]
            if stale_core_curves:
                self._warn_input_curves_removed(stale_core_curves, "core correction")
            for curve_name in stale_core_curves:
                well.remove_curve(curve_name)
            if stale_core_curves:
                trace.decisions.append("Existing core-corrected curves cleared before pre-saturation core correction")

            perm_module = PermeabilityModule(
                method=self.config.perm_method,
                fallback_method=self.config.perm_fallback_method,
                use_peer_core=self.config.use_peer_core,
            )
            result = perm_module.calculate_core_correction(
                well,
                peer_wells=self.config.peer_wells,
                formation_tops=self.config.formation_tops,
            )
            trace.input_curves = [result.metadata.get("porosity_curve", "PHIE")]
            trace.parameters["same_well_core_models"] = result.metadata.get("same_well_core_models")
            trace.parameters["peer_core_models"] = result.metadata.get("peer_core_models")
            trace.parameters["core_models"] = result.metadata.get("core_models")
            trace.parameters["core_rejected_models"] = result.metadata.get("core_rejected_models")
            trace.parameters["core_min_points"] = result.metadata.get("core_min_points")
            trace.parameters["core_sections_count"] = result.metadata.get("core_sections_count")

            phi_corecal = result.metadata.get("phi_corecal")
            perm_core = result.metadata.get("perm_core")
            if phi_corecal is not None:
                well.add_curve("PHIE_CC", phi_corecal, "v/v", "Core-corrected effective porosity")
                self._core_porosity_generated = True
                trace.output_curves.append("PHIE_CC")
                trace.decisions.append("Core-corrected porosity stored as PHIE_CC before saturation")
            if perm_core is not None:
                well.add_curve("PERM_CC", perm_core, "mD", "Permeability from core-calibrated poro-perm transform")
                trace.output_curves.append("PERM_CC")
                trace.decisions.append("Core-corrected permeability stored as PERM_CC")
            if not trace.output_curves:
                trace.decisions.append("No PHIE_CC/PERM_CC written because no same-well core model was fitted")
        except Exception as e:
            self.warnings.append(f"Core correction warning: {e}")
            trace.success = False
            trace.message = str(e)
        return trace

    def _calculate_saturation(self, well: "Well", rw: float) -> StepTrace:
        """Calculate water saturation."""
        from petrophys.interpretation.saturation import SaturationModule

        trace = StepTrace(step="saturation", method=self.config.saturation_model)
        trace.parameters = {
            "model": self.config.saturation_model,
            "rw": rw, "a": self.config.a,
            "m": self.config.m, "n": self.config.n,
            "rwb": self.config.rwb,
            "rtsh": self.config.rtsh,
            "vsh_exp": self.config.vsh_exp,
            "shale_exp": self.config.shale_exp,
        }
        try:
            from petrophys.interpretation.temperature.temp_corrections import arps_equation
            from petrophys.interpretation.temperature.temp_profile import temperature_curve_f

            rw_used: Any = rw
            formation_temp = temperature_curve_f(well, ["TEMP"]) if self._rw_temp_f is not None else None
            if formation_temp is not None:
                # Rw given at another temperature: Arps-corrected to TEMP at each depth.
                rw_used = arps_equation(rw, self._rw_temp_f, formation_temp)
                trace.parameters["rw_reference_temp_f"] = self._rw_temp_f
                trace.decisions.append(
                    f"Rw {rw:.4f} ohm.m at {self._rw_temp_f:.0f}F corrected to TEMP (Arps)"
                )
            elif self._rw_temp_f is not None:
                self.warnings.append(
                    f"Rw {rw:.4f} ohm.m at {self._rw_temp_f:.0f}F used without temperature "
                    "correction: no TEMP curve"
                )
            sw_module = SaturationModule(
                a=self.config.a, m=self.config.m,
                n=self.config.n, model=self.config.saturation_model,
                vcl_cutoff=self.config.vcl_cutoff
            )
            result = sw_module.calculate(
                well,
                rw=rw_used,
                lithology=self.config.lithology,
                rwb=self.config.rwb,
                phi_cl=self.config.phi_cl,
                rtsh=self.config.rtsh,
                vsh_exp=self.config.vsh_exp,
                shale_exp=self.config.shale_exp,
            )
            well.add_curve("SW", result.sw, "frac", f"Water Saturation ({result.model})")
            outputs = ["SW"]
            if result.sxo is not None:
                well.add_curve("SXO", result.sxo, "frac", "Flushed Zone Water Saturation")
                outputs.append("SXO")
            trace.input_curves = ["RDEP", "PHIE_CC" if well.get("PHIE_CC") is not None else "PHIE"]
            trace.output_curves = outputs
            trace.decisions.append(f"Sw ({self.config.saturation_model}): "
                                  f"a={self.config.a}, m={self.config.m}, n={self.config.n}, "
                                  f"Rw={rw:.4f}")
            for route, count in (result.fallback_counts or {}).items():
                message = (
                    f"{count} shaly carbonate samples used {route.split('->', 1)[1]} instead of "
                    "Dual Water, which needs rwb (it is not estimated)"
                )
                trace.decisions.append(message)
                self.warnings.append(message)
        except Exception as e:
            self.warnings.append(f"Saturation calculation warning: {e}")
            trace.success = False
            trace.message = str(e)
        return trace

    def _calculate_permeability(self, well: "Well") -> StepTrace:
        """Calculate permeability."""
        from petrophys.interpretation.lithology.matrix_inference import find_lithology_code_curve
        from petrophys.interpretation.permeability import PermeabilityModule

        trace = StepTrace(step="permeability", method=self.config.perm_method)
        try:
            stale_core_curves = [
                name for name in ("PHIE_CC", "PERM_CC")
                if well.get(name) is not None and not (name == "PHIE_CC" and self._core_porosity_generated)
            ]
            if stale_core_curves:
                self._warn_input_curves_removed(stale_core_curves, "permeability")
            for curve_name in stale_core_curves:
                well.remove_curve(curve_name)
            if stale_core_curves:
                trace.decisions.append(
                    "Existing core-corrected curves cleared before permeability; "
                    "PHIE_CC/PERM_CC are rewritten only when the current same-well core model is fitted"
                )

            perm_module = PermeabilityModule(
                method=self.config.perm_method,
                fallback_method=self.config.perm_fallback_method,
                use_peer_core=self.config.use_peer_core,
            )
            lithology = None
            if (
                find_lithology_code_curve(well) is None
                and get_curve(well, ["LITHO"]) is None
                and well.get("LITHO_CLASS") is None
            ):
                lithology = self.config.lithology

            result = perm_module.calculate(
                well,
                swi=self.config.swirr,
                lithology=lithology,
                peer_wells=self.config.peer_wells,
                formation_tops=self.config.formation_tops,
            )
            well.add_curve("PERM", result.perm, "mD", f"Permeability ({result.method})")
            empirical_method = result.metadata.get("empirical_method") or self.config.perm_fallback_method
            if result.metadata.get("input_curves"):
                trace.input_curves = list(dict.fromkeys(result.metadata["input_curves"]))
            else:
                trace.input_curves = [result.metadata.get("porosity_curve", "PHIE")]
            trace.output_curves = ["PERM"]
            phi_corecal = result.metadata.get("phi_corecal")
            perm_core = result.metadata.get("perm_core")
            if phi_corecal is not None:
                well.add_curve(
                    "PHIE_CC",
                    phi_corecal,
                    "v/v",
                    "Permeability input porosity from core calibration",
                )
                trace.output_curves.append("PHIE_CC")
            if perm_core is not None:
                well.add_curve(
                    "PERM_CC",
                    perm_core,
                    "mD",
                    "Permeability from core-calibrated poro-perm transform",
                )
                trace.output_curves.append("PERM_CC")
            swirr_metadata = dict(result.swi_metadata)
            swirr_app = swirr_metadata.pop("swirr_app", None)
            if swirr_app is not None and empirical_method != "lith_vcl":
                well.add_curve(
                    "SWIRR_APP",
                    swirr_app,
                    "v/v",
                    "Apparent irreducible water saturation from hydrocarbon-mask P5 method",
                )
                trace.output_curves.append("SWIRR_APP")
            if np.isscalar(result.swi_used):
                trace.parameters["swirr"] = float(result.swi_used)
                if result.swi_source == "hc_mask_p5":
                    n_samples = swirr_metadata.get("hc_mask_samples")
                    rt_curve = swirr_metadata.get("rt_curve", "RDEP")
                    trace.decisions.append(
                        f"SWIRR_APP = P5(SW where resflag and {rt_curve} > 5) = {float(result.swi_used):.3f} (n={n_samples})"
                    )
                elif result.swi_source == "default_0.20":
                    n_samples = swirr_metadata.get("hc_mask_samples")
                    if n_samples is not None:
                        trace.decisions.append(f"SWIRR_APP = 0.200 fallback because hydrocarbon mask had {n_samples} samples")
                    else:
                        trace.decisions.append("SWIRR_APP = 0.200 fallback")
                else:
                    trace.decisions.append(f"Swirr = {float(result.swi_used):.3f} ({result.swi_source})")
            elif isinstance(result.swi_used, np.ndarray):
                trace.decisions.append(f"Swirr resolved as curve ({result.swi_source})")
            elif result.swi_source == "not_required" and empirical_method == "lith_vcl":
                trace.decisions.append(
                    "Swirr not used because lith_vcl permeability uses PHIT, VCL, and lithology"
                )
            else:
                trace.decisions.append("Swirr not used because permeability came from core calibration")
            trace.parameters["swirr_source"] = result.swi_source
            trace.parameters["swirr_app_method"] = swirr_metadata.get("swirr_app_method")
            trace.parameters["swirr_metadata"] = swirr_metadata
            trace.parameters["fallback_method"] = self.config.perm_fallback_method
            trace.parameters["empirical_method"] = empirical_method
            trace.parameters["empirical_attempted_methods"] = result.metadata.get("empirical_attempted_methods")
            trace.parameters["empirical_method_metadata"] = result.metadata.get("method_metadata", {})
            trace.parameters["perm_max_clip_md"] = result.metadata.get("perm_max_clip_md")
            trace.parameters["strategy"] = result.metadata.get("strategy")
            trace.parameters["same_well_core_models"] = result.metadata.get("same_well_core_models")
            trace.parameters["peer_core_models"] = result.metadata.get("peer_core_models")
            trace.parameters["peer_core_enabled"] = result.metadata.get("peer_core_enabled")
            trace.parameters["core_min_points"] = result.metadata.get("core_min_points")
            trace.parameters["core_models"] = result.metadata.get("core_models")
            trace.parameters["core_rejected_models"] = result.metadata.get("core_rejected_models")
            if empirical_method == "lith_vcl":
                trace.decisions.append(
                    "lith_vcl empirical permeability: "
                    "10 ** min(4.5, slope * PHIT * (1 - VCL * 1.2) + intercept)"
                )
                groups = result.metadata.get("method_metadata", {}).get("lithology_parameter_groups") or []
                if groups:
                    trace.decisions.append("lith_vcl parameter groups used: " + ", ".join(groups))
            elif empirical_method == "coates":
                trace.decisions.append(
                    "Coates-Denoo empirical permeability: "
                    "k = 10000 * PHIE^4 * ((1 - SWIRR) / SWIRR)^2"
                )
            elif empirical_method == "timur":
                trace.decisions.append(
                    "Timur empirical permeability: "
                    "k = 0.136 * (100 * PHIE)^4.4 / (100 * SWIRR)^2"
                )
            if result.metadata.get("core_sections_count"):
                trace.decisions.append(
                    f"Core-calibrated sections: {result.metadata['core_sections_count']} samples"
                )
            for model in result.metadata.get("core_models", []):
                source = model.get("source_well") or "unknown"
                formation = model.get("source_formation") or "all_depths"
                source_type = model.get("source_type") or "core"
                trace.decisions.append(
                    f"{source_type}: {source} / {formation} "
                    f"(n={model.get('n_core_points')}, "
                    f"core_phi={model.get('core_porosity_curve') or 'unknown'}, "
                    f"core_perm={model.get('core_permeability_curve') or 'unknown'})"
                )
            if result.metadata.get("perm_max_clip_md") is not None:
                trace.decisions.append(f"Permeability clipped to <= {result.metadata['perm_max_clip_md']:.0f} mD")
            trace.decisions.append(f"Permeability strategy: {result.method}")
        except Exception as e:
            self.warnings.append(f"Permeability calculation warning: {e}")
            trace.success = False
            trace.message = str(e)
        return trace

    def _warn_input_curves_removed(self, curves: List[str], step: str) -> None:
        """Keep input curves (not ones this run wrote) as ``<NAME>_INPUT`` and warn.

        The caller removes ``curves`` afterwards; an input curve is renamed
        here first, so it survives as PHIE_CC_INPUT or PERM_CC_INPUT.
        """
        removed = [name for name in curves if name in self._input_curves]
        self._input_curves.difference_update(removed)
        if removed:
            for name in removed:
                if well_curve := self._well.get(name):
                    self._well.remove_curve(f"{name}_INPUT")
                    self._well.rename_curve(well_curve.mnemonic, f"{name}_INPUT")
            self.warnings.append(
                f"Input curve(s) {', '.join(removed)} kept as "
                f"{', '.join(f'{name}_INPUT' for name in removed)} before {step}; "
                "earlier steps (e.g. saturation) may have used them"
            )

    def _calculate_net_pay(self, well: "Well") -> StepTrace:
        """Calculate net pay flags."""
        from petrophys.evaluation.net_pay import NetPayModule

        trace = StepTrace(step="net_pay", method="cutoff_flags")
        trace.parameters = {
            "vcl_cutoff": self.config.vcl_cutoff,
            "phie_cutoff": self.config.phie_cutoff,
            "sw_cutoff": self.config.sw_cutoff,
        }
        try:
            np_module = NetPayModule(
                vcl_cutoff=self.config.vcl_cutoff,
                phie_cutoff=self.config.phie_cutoff,
                sw_cutoff=self.config.sw_cutoff
            )
            result = np_module.calculate(well)
            np_module.add_to_well(well, result)
            # NetPayModule reads PHIE_CC, else PHIE, else PHIT.
            porosity = get_curve(well, ["PHIE_CC", "PHIE", "PHIT"])
            porosity_name = porosity.mnemonic if porosity is not None else "PHIE"
            trace.input_curves = ["VCL", porosity_name, "SW"]
            trace.output_curves = ["NET_ROCK", "NET_RES", "NET_PAY"]
            trace.decisions.append(f"NET_ROCK: VCL < {self.config.vcl_cutoff}")
            trace.decisions.append(f"NET_RES: + {porosity_name} > {self.config.phie_cutoff}")
            trace.decisions.append(f"NET_PAY: + Sw < {self.config.sw_cutoff}")
        except Exception as e:
            self.warnings.append(f"Net pay calculation warning: {e}")
            trace.success = False
            trace.message = str(e)
        return trace


def run_standard_workflow(
    well: "Well",
    config: Optional[WorkflowConfig] = None,
    **kwargs
) -> WorkflowResult:
    """
    Run the deprecated fixed-configuration workflow on a well.

    Use :func:`petrophys.workflow.run_dynamic_workflow` for new code.

    Parameters
    ----------
    well : Well
        Well object
    config : WorkflowConfig, optional
        Workflow configuration
    **kwargs
        Override config parameters

    Returns
    -------
    WorkflowResult
        Workflow results
    """
    if config is None:
        config = WorkflowConfig(**kwargs)
    elif kwargs:
        # Override a copy of config; unknown settings raise as without config.
        import dataclasses

        config = dataclasses.replace(config, **kwargs)

    workflow = StandardWorkflow(config)
    return workflow.run(well)
