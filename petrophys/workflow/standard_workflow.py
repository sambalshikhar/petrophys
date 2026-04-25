"""
Standard petrophysics workflow.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import numpy as np
import re


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
    rw: Optional[float] = None  # If None, calculate from SP
    salinity_ppm: Optional[float] = None

    # Matrix properties
    matrix_type: str = "sandstone"
    lithology: str = "clastic"
    matrix_density: float = 2.65
    fluid_density: float = 1.0
    matrix_dt: float = 55.5
    fluid_dt: float = 189.0
    phi_cl: float = 0.15
    rho_cl: float = 2.45

    # Archie parameters
    a: float = 1.0
    m: float = 2.0
    n: float = 2.0

    # Saturation model
    saturation_model: str = "archie"  # archie, simandoux, indonesia, north_sea, nigeria_schlumberger
    rtsh: Optional[float] = None
    vsh_exp: float = 1.6
    shale_exp: float = 1.6
    swirr: Optional[float] = None
    perm_method: str = "auto"
    perm_fallback_method: str = "timur"
    peer_wells: Optional[List[Any]] = None
    formation_tops: Optional[Any] = None
    mud_has_clay_or_kcl: bool = False

    # Cutoffs
    vcl_cutoff: float = 0.3
    phie_cutoff: float = 0.08
    sw_cutoff: float = 0.7

    # VCL parameters
    gr_min: float = 20.0
    gr_max: float = 100.0
    gr_method: str = "linear"

    # Processing options
    calculate_vcl: bool = True
    calculate_porosity: bool = True
    calculate_sw: bool = True
    calculate_perm: bool = True
    calculate_net_pay: bool = True


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
    Standard petrophysics interpretation workflow.

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
        self.config = config or WorkflowConfig()
        self.errors: List[str] = []
        self.warnings: List[str] = []

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

            # Step 5: Water saturation
            if self.config.calculate_sw:
                result.steps.append(self._calculate_saturation(well, rw))

            # Step 6: Permeability
            if self.config.calculate_perm:
                result.steps.append(self._calculate_permeability(well))

            # Step 7: Net pay
            if self.config.calculate_net_pay:
                result.steps.append(self._calculate_net_pay(well))

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
        from petrophys.environmental.temperature import TemperatureProfile

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
        from petrophys.clay_volume import VclModule

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
            trace.method = "las_metadata"
            trace.parameters = {"rw": float(las_rw)}
            trace.decisions.append(f"Rw = {float(las_rw):.4f} ohm.m (LAS metadata)")
            return float(las_rw), trace

        if self.config.salinity_ppm is not None:
            from petrophys.resistivity.rw.salinity_rw import rw_from_salinity
            temp = well.get("TEMP")
            if temp is not None:
                avg_temp = float(np.nanmean(temp.data))
            else:
                avg_temp = (self.config.surface_temp_f + self.config.bht_f) / 2

            rw = rw_from_salinity(self.config.salinity_ppm, avg_temp)
            trace.method = "salinity_conversion"
            trace.parameters = {"salinity_ppm": self.config.salinity_ppm,
                               "temperature_f": avg_temp, "rw": rw}
            trace.decisions.append(f"Rw from {self.config.salinity_ppm:.0f} ppm "
                                  f"@ {avg_temp:.0f}F = {rw:.4f} ohm.m")
            return rw, trace

        sp = well.get("SP")
        if sp is not None and not self.config.mud_has_clay_or_kcl:
            from petrophys.resistivity.rw import RwModule
            try:
                rmf_for_sp = self.config.rmf_at_meas_temp
                las_rmf = _parameter_float(well, "RMF")
                if las_rmf is None:
                    las_rmf = _constant_zoned_parameter(well, "RMF")
                if las_rmf is not None:
                    rmf_for_sp = float(las_rmf)

                rw_module = RwModule(method="sp")
                result = rw_module.calculate(
                    well,
                    rmf=rmf_for_sp,
                )
                trace.method = result.method
                trace.input_curves = ["SP"]
                trace.parameters = {
                    "rmf": rmf_for_sp,
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
        from petrophys.petrophysics.porosity import PorosityModule

        trace = StepTrace(step="porosity", method="density_neutron")
        trace.parameters = {
            "matrix_type": self.config.matrix_type,
            "matrix_density": self.config.matrix_density,
            "fluid_density": self.config.fluid_density,
            "matrix_dt": self.config.matrix_dt,
            "fluid_dt": self.config.fluid_dt,
            "phi_cl": self.config.phi_cl,
        }
        try:
            por_module = PorosityModule(
                matrix=self.config.matrix_type,
                rhob_fluid=self.config.fluid_density,
                phi_clay=self.config.phi_cl
            )
            por_module.rhob_matrix = self.config.matrix_density
            por_module.dtc_matrix = self.config.matrix_dt
            result = por_module.calculate(well)
            por_module.add_to_well(well, result)

            inputs = [c for c in ["RHOB", "NPHI", "DTC", "VCL"] if well.get(c)]
            outputs = [c for c in ["PHID", "PHIT", "PHIE", "PHIS"] if well.get(c)]
            trace.input_curves = inputs
            trace.output_curves = outputs
            trace.decisions.append(f"PHID = (rho_ma({self.config.matrix_density}) - RHOB) / "
                                  f"(rho_ma - rho_fl({self.config.fluid_density}))")
        except Exception as e:
            self.warnings.append(f"Porosity calculation warning: {e}")
            trace.success = False
            trace.message = str(e)
        return trace

    def _calculate_saturation(self, well: "Well", rw: float) -> StepTrace:
        """Calculate water saturation."""
        from petrophys.petrophysics.saturation import SaturationModule

        trace = StepTrace(step="saturation", method=self.config.saturation_model)
        trace.parameters = {
            "model": self.config.saturation_model,
            "rw": rw, "a": self.config.a,
            "m": self.config.m, "n": self.config.n,
            "rtsh": self.config.rtsh,
            "vsh_exp": self.config.vsh_exp,
            "shale_exp": self.config.shale_exp,
        }
        try:
            sw_module = SaturationModule(
                a=self.config.a, m=self.config.m,
                n=self.config.n, model=self.config.saturation_model,
                vcl_cutoff=self.config.vcl_cutoff
            )
            result = sw_module.calculate(
                well,
                rw=rw,
                lithology=self.config.lithology,
                phi_cl=self.config.phi_cl,
                rtsh=self.config.rtsh,
                vsh_exp=self.config.vsh_exp,
                shale_exp=self.config.shale_exp,
            )
            well.add_curve("SW", result.sw, "v/v", f"Water Saturation ({result.model})")
            outputs = ["SW"]
            if result.sxo is not None:
                well.add_curve("SXO", result.sxo, "v/v", "Flushed Zone Water Saturation")
                outputs.append("SXO")
            trace.input_curves = ["RDEP", "PHIE"]
            trace.output_curves = outputs
            trace.decisions.append(f"Sw ({self.config.saturation_model}): "
                                  f"a={self.config.a}, m={self.config.m}, n={self.config.n}, "
                                  f"Rw={rw:.4f}")
        except Exception as e:
            self.warnings.append(f"Saturation calculation warning: {e}")
            trace.success = False
            trace.message = str(e)
        return trace

    def _calculate_permeability(self, well: "Well") -> StepTrace:
        """Calculate permeability."""
        from petrophys.petrophysics.permeability import PermeabilityModule

        trace = StepTrace(step="permeability", method=self.config.perm_method)
        try:
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
            well.add_curve("PERM", result.perm, "mD", f"Permeability ({result.method})")
            empirical_method = result.metadata.get("empirical_method") or self.config.perm_fallback_method
            if empirical_method == "lith_vcl" and result.metadata.get("input_curves"):
                trace.input_curves = list(dict.fromkeys(result.metadata["input_curves"]))
            else:
                trace.input_curves = [result.metadata.get("porosity_curve", "PHIE")]
            trace.output_curves = ["PERM"]
            phi_corecal = result.metadata.get("phi_corecal")
            perm_core = result.metadata.get("perm_core")
            if phi_corecal is not None:
                well.add_curve(
                    "PHIE_CORECAL",
                    phi_corecal,
                    "v/v",
                    "Permeability input porosity from core calibration",
                )
                trace.output_curves.append("PHIE_CORECAL")
            if perm_core is not None:
                well.add_curve(
                    "PERM_CC",
                    perm_core,
                    "mD",
                    "Permeability from core-calibrated poro-perm transform",
                )
                trace.output_curves.append("PERM_CC")
            if np.isscalar(result.swi_used):
                trace.parameters["swirr"] = float(result.swi_used)
                if result.swi_source == "hc_mask_p5":
                    n_samples = result.swi_metadata.get("hc_mask_samples")
                    rt_curve = result.swi_metadata.get("rt_curve", "RDEP")
                    trace.decisions.append(
                        f"Swirr = P5(SW where resflag and {rt_curve} > 5) = {float(result.swi_used):.3f} (n={n_samples})"
                    )
                elif result.swi_source == "default_0.20":
                    n_samples = result.swi_metadata.get("hc_mask_samples")
                    if n_samples is not None:
                        trace.decisions.append(f"Swirr = 0.200 fallback because hydrocarbon mask had {n_samples} samples")
                    else:
                        trace.decisions.append("Swirr = 0.200 fallback")
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
            trace.parameters["swirr_metadata"] = result.swi_metadata
            trace.parameters["fallback_method"] = self.config.perm_fallback_method
            trace.parameters["empirical_method"] = empirical_method
            trace.parameters["empirical_attempted_methods"] = result.metadata.get("empirical_attempted_methods")
            trace.parameters["empirical_method_metadata"] = result.metadata.get("method_metadata", {})
            trace.parameters["strategy"] = result.metadata.get("strategy")
            trace.parameters["same_well_core_models"] = result.metadata.get("same_well_core_models")
            trace.parameters["peer_core_models"] = result.metadata.get("peer_core_models")
            if empirical_method == "lith_vcl":
                trace.decisions.append(
                    "lith_vcl empirical permeability: "
                    "10 ** min(4.5, slope * PHIT * (1 - VCL * 1.2) + intercept)"
                )
                groups = result.metadata.get("method_metadata", {}).get("lithology_parameter_groups") or []
                if groups:
                    trace.decisions.append("lith_vcl parameter groups used: " + ", ".join(groups))
            if result.metadata.get("core_sections_count"):
                trace.decisions.append(
                    f"Core-calibrated sections: {result.metadata['core_sections_count']} samples"
                )
            for model in result.metadata.get("core_models", []):
                source = model.get("source_well") or "unknown"
                formation = model.get("source_formation") or "all_depths"
                source_type = model.get("source_type") or "core"
                trace.decisions.append(
                    f"{source_type}: {source} / {formation} (n={model.get('n_core_points')}, "
                    f"perm R2={model.get('perm_r2'):.3f})"
                )
            trace.decisions.append(f"Permeability strategy: {result.method}")
        except Exception as e:
            self.warnings.append(f"Permeability calculation warning: {e}")
            trace.success = False
            trace.message = str(e)
        return trace

    def _calculate_net_pay(self, well: "Well") -> StepTrace:
        """Calculate net pay flags."""
        from petrophys.reservoir.net_pay import NetPayModule

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
            trace.input_curves = ["VCL", "PHIE", "SW"]
            trace.output_curves = ["NET_ROCK", "NET_RES", "NET_PAY"]
            trace.decisions.append(f"NET_ROCK: VCL < {self.config.vcl_cutoff}")
            trace.decisions.append(f"NET_RES: + PHIE > {self.config.phie_cutoff}")
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
    Run standard workflow on a well.

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
    else:
        # Update config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    workflow = StandardWorkflow(config)
    return workflow.run(well)
