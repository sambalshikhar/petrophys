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
class WorkflowResult:
    """Results from standard workflow."""

    well: "Well"
    config: WorkflowConfig
    success: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Computed values
    rw_computed: Optional[float] = None
    temperature_gradient: Optional[float] = None


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
            self._calculate_temperature(well)

            # Step 2: VCL
            if self.config.calculate_vcl:
                self._calculate_vcl(well)

            # Step 3: Rw
            rw = self._determine_rw(well)
            result.rw_computed = rw

            # Step 4: Porosity
            if self.config.calculate_porosity:
                self._calculate_porosity(well)

            # Step 5: Water saturation
            if self.config.calculate_sw:
                self._calculate_saturation(well, rw)

            # Step 6: Permeability
            if self.config.calculate_perm:
                self._calculate_permeability(well)

            # Step 7: Net pay
            if self.config.calculate_net_pay:
                self._calculate_net_pay(well)

            result.success = len(self.errors) == 0
            result.errors = self.errors
            result.warnings = self.warnings

        except Exception as e:
            self.errors.append(f"Workflow error: {str(e)}")
            result.success = False
            result.errors = self.errors

        return result

    def _calculate_temperature(self, well: "Well") -> None:
        """Calculate temperature profile."""
        from petrophys.environmental.temperature import TemperatureProfile

        try:
            temp_module = TemperatureProfile(
                surface_temp=self.config.surface_temp_f,
                bht=self.config.bht_f,
                td=self.config.td_ft,
                temp_unit="F"
            )
            temp_module.add_to_well(well)
        except Exception as e:
            self.warnings.append(f"Temperature calculation warning: {e}")

    def _calculate_vcl(self, well: "Well") -> None:
        """Calculate clay volume."""
        from petrophys.clay_volume import VclModule

        try:
            vcl_module = VclModule()
            vcl_module.calculate(well)
            vcl_module.add_to_well(well)
        except Exception as e:
            self.warnings.append(f"VCL calculation warning: {e}")

    def _determine_rw(self, well: "Well") -> float:
        """Determine Rw."""
        if self.config.rw is not None:
            return self.config.rw

        if self.config.salinity_ppm is not None:
            from petrophys.resistivity.rw import calculate_rw_from_salinity
            # Use formation temperature
            temp = well.get("TEMP")
            if temp is not None:
                avg_temp = float(np.nanmean(temp.data))
            else:
                avg_temp = (self.config.surface_temp_f + self.config.bht_f) / 2

            return calculate_rw_from_salinity(self.config.salinity_ppm, avg_temp)

        # Try to calculate from SP
        sp = well.get("SP")
        if sp is not None:
            from petrophys.resistivity.rw import RwModule
            try:
                rw_module = RwModule()
                result = rw_module.calculate_from_sp(
                    well,
                    rmf=self.config.rmf_at_meas_temp
                )
                return result.rw
            except Exception as e:
                self.warnings.append(f"Rw from SP failed: {e}")

        # Default Rw
        self.warnings.append("Using default Rw = 0.05 ohm.m")
        return 0.05

    def _calculate_porosity(self, well: "Well") -> None:
        """Calculate porosity."""
        from petrophys.petrophysics.porosity import PorosityModule

        try:
            por_module = PorosityModule(
                rho_ma=self.config.matrix_density,
                rho_fl=self.config.fluid_density,
                dt_ma=self.config.matrix_dt,
                dt_fl=self.config.fluid_dt
            )

            # Calculate available porosities
            result = por_module.calculate(well)
            por_module.add_to_well(well, result)

        except Exception as e:
            self.warnings.append(f"Porosity calculation warning: {e}")

    def _calculate_saturation(self, well: "Well", rw: float) -> None:
        """Calculate water saturation."""
        from petrophys.petrophysics.saturation import SwModule

        try:
            sw_module = SwModule(
                rw=rw,
                a=self.config.a,
                m=self.config.m,
                n=self.config.n,
                model=self.config.saturation_model
            )

            result = sw_module.calculate(well)
            sw_module.add_to_well(well, result)

        except Exception as e:
            self.warnings.append(f"Saturation calculation warning: {e}")

    def _calculate_permeability(self, well: "Well") -> None:
        """Calculate permeability."""
        from petrophys.petrophysics.permeability import PermModule

        try:
            perm_module = PermModule()
            result = perm_module.calculate(well)
            perm_module.add_to_well(well, result)

        except Exception as e:
            self.warnings.append(f"Permeability calculation warning: {e}")

    def _calculate_net_pay(self, well: "Well") -> None:
        """Calculate net pay flags."""
        from petrophys.reservoir.net_pay import NetPayModule

        try:
            np_module = NetPayModule(
                vcl_cutoff=self.config.vcl_cutoff,
                phie_cutoff=self.config.phie_cutoff,
                sw_cutoff=self.config.sw_cutoff
            )

            result = np_module.calculate(well)
            np_module.add_to_well(well, result)

        except Exception as e:
            self.warnings.append(f"Net pay calculation warning: {e}")


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
