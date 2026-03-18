"""
Standard petrophysics workflow.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import numpy as np


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
    matrix_density: float = 2.65
    fluid_density: float = 1.0
    matrix_dt: float = 55.5
    fluid_dt: float = 189.0

    # Archie parameters
    a: float = 1.0
    m: float = 2.0
    n: float = 2.0

    # Saturation model
    saturation_model: str = "archie"  # archie, simandoux, indonesia

    # Cutoffs
    vcl_cutoff: float = 0.3
    phie_cutoff: float = 0.08
    sw_cutoff: float = 0.7

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
        print(f"STANDARD WORKFLOW TRACE — {self.well.name or 'Unknown'}")
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
            temp_module = TemperatureProfile(
                surface_temp=self.config.surface_temp_f,
                bht=self.config.bht_f,
                td=self.config.td_ft,
                temp_unit="F"
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
            vcl_module = VclModule()
            vcl_module.calculate(well)
            vcl_module.add_to_well(well)
            trace.output_curves = ["VCL"]
            trace.decisions.append("GR method with default GR_min/GR_max auto-detection")
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

        if self.config.salinity_ppm is not None:
            from petrophys.resistivity.rw import calculate_rw_from_salinity
            temp = well.get("TEMP")
            if temp is not None:
                avg_temp = float(np.nanmean(temp.data))
            else:
                avg_temp = (self.config.surface_temp_f + self.config.bht_f) / 2

            rw = calculate_rw_from_salinity(self.config.salinity_ppm, avg_temp)
            trace.method = "salinity_conversion"
            trace.parameters = {"salinity_ppm": self.config.salinity_ppm,
                               "temperature_f": avg_temp, "rw": rw}
            trace.decisions.append(f"Rw from {self.config.salinity_ppm:.0f} ppm "
                                  f"@ {avg_temp:.0f}F = {rw:.4f} ohm.m")
            return rw, trace

        sp = well.get("SP")
        if sp is not None:
            from petrophys.resistivity.rw import RwModule
            try:
                rw_module = RwModule()
                result = rw_module.calculate_from_sp(well, rmf=self.config.rmf_at_meas_temp)
                trace.method = "SP_method"
                trace.input_curves = ["SP"]
                trace.parameters = {"rmf": self.config.rmf_at_meas_temp, "rw": result.rw}
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
            "matrix_density": self.config.matrix_density,
            "fluid_density": self.config.fluid_density,
            "matrix_dt": self.config.matrix_dt,
            "fluid_dt": self.config.fluid_dt,
        }
        try:
            por_module = PorosityModule(
                rho_ma=self.config.matrix_density,
                rho_fl=self.config.fluid_density,
                dt_ma=self.config.matrix_dt,
                dt_fl=self.config.fluid_dt
            )
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
        from petrophys.petrophysics.saturation import SwModule

        trace = StepTrace(step="saturation", method=self.config.saturation_model)
        trace.parameters = {
            "model": self.config.saturation_model,
            "rw": rw, "a": self.config.a,
            "m": self.config.m, "n": self.config.n,
        }
        try:
            sw_module = SwModule(
                rw=rw, a=self.config.a, m=self.config.m,
                n=self.config.n, model=self.config.saturation_model
            )
            result = sw_module.calculate(well)
            sw_module.add_to_well(well, result)
            trace.input_curves = ["RDEP", "PHIE"]
            trace.output_curves = ["SW"]
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
        from petrophys.petrophysics.permeability import PermModule

        trace = StepTrace(step="permeability", method="default")
        try:
            perm_module = PermModule()
            result = perm_module.calculate(well)
            perm_module.add_to_well(well, result)
            trace.input_curves = ["PHIE"]
            trace.output_curves = ["PERM"]
            trace.decisions.append("Default permeability equation")
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
