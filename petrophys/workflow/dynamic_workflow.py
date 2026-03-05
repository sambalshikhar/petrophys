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

    # Control which steps to skip
    skip_steps: List[str] = field(default_factory=list)


@dataclass
class StepResult:
    """Result of a single workflow step."""
    step: str
    success: bool
    message: str = ""
    data: Optional[Dict[str, Any]] = None


@dataclass
class DynamicWorkflowResult:
    """Results from dynamic workflow execution."""
    well: "Well"
    config: DynamicWorkflowConfig
    steps_executed: List[StepResult] = field(default_factory=list)
    steps_skipped: List[str] = field(default_factory=list)
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

        for step_enum, step_fn, can_fn in pipeline:
            step_name = step_enum.value

            if step_name in self.config.skip_steps:
                self._result.steps_skipped.append(step_name)
                continue

            if can_fn(well):
                try:
                    step_fn(well)
                    self._result.steps_executed.append(
                        StepResult(step=step_name, success=True)
                    )
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
                      "RSHA", "RMED", "RXO", "PEF", "CALI", "BS", "DRHO",
                      "VCL", "PHIE", "PHIT", "SW", "SXO", "PERM", "TEMP"]:
            if well.get(name) is not None:
                available.add(name)
        return available

    def _has_curve(self, well: "Well", *names: str) -> bool:
        """Check if any of the named curves exist."""
        return any(well.get(n) is not None for n in names)

    # ---- CAN-RUN checks ----

    def _can_temperature(self, well: "Well") -> bool:
        return self.config.bht_f is not None and self.config.td_ft is not None

    def _can_bad_hole(self, well: "Well") -> bool:
        return self._has_curve(well, "CALI") or self._has_curve(well, "DRHO")

    def _can_vclay(self, well: "Well") -> bool:
        return self._has_curve(well, "GR")

    def _can_rcl(self, well: "Well") -> bool:
        return self._has_curve(well, "RDEP", "RT") and self._has_curve(well, "VCL")

    def _can_lithology(self, well: "Well") -> bool:
        return (self._has_curve(well, "RHOB") and
                self._has_curve(well, "NPHI") and
                self._has_curve(well, "PEF"))

    def _can_rmf(self, well: "Well") -> bool:
        return self.config.rmf_at_meas_temp is not None or self.config.rm is not None

    def _can_rw(self, well: "Well") -> bool:
        return (self.config.rw is not None or
                self.config.salinity_ppm is not None or
                (self._has_curve(well, "SP") and not self.config.mud_has_clay_or_kcl))

    def _can_invasion(self, well: "Well") -> bool:
        return (self._has_curve(well, "RSHA", "RXO") and
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
        return self._has_curve(well, "PHIE") and self._has_curve(well, "SW")

    def _can_mhi(self, well: "Well") -> bool:
        return self._has_curve(well, "SXO") and self._has_curve(well, "SW")

    def _can_net_pay(self, well: "Well") -> bool:
        return (self._has_curve(well, "PHIE") and self._has_curve(well, "SW"))

    # ---- STEP implementations ----

    def _step_temperature(self, well: "Well") -> None:
        from petrophys.environmental.temperature.temp_profile import TemperatureProfile

        profile = TemperatureProfile.from_bht(
            bht=self.config.bht_f,
            td=self.config.td_ft,
            surface_temp=self.config.surface_temp_f,
            temp_unit="F"
        )
        depth = well.get("DEPT") or well.get("DEPTH")
        if depth is not None:
            temp_array = profile.get_temperature(depth.data)
            well.add_curve("TEMP", temp_array, "degF", "Formation Temperature")
            self._result.temperature_gradient = profile.gradient

    def _step_bad_hole(self, well: "Well") -> None:
        from petrophys.qaqc.bad_hole.flag_generator import generate_bad_hole_flags

        cali = well.get("CALI")
        bs_curve = well.get("BS")
        drho = well.get("DRHO")
        rhob = well.get("RHOB")
        pef = well.get("PEF")

        bs = bs_curve.data if bs_curve is not None else 8.5

        flags = generate_bad_hole_flags(
            cali=cali.data if cali is not None else None,
            bs=bs,
            drho=drho.data if drho is not None else None,
            rhob=rhob.data if rhob is not None else None,
            pef=pef.data if pef is not None else None
        )

        well.add_curve("BAD_HOLE", flags["bad_hole"].astype(float), "", "Bad Hole Flag")

    def _step_vclay(self, well: "Well") -> None:
        from petrophys.clay_volume.vcl_module import VclModule

        gr_min = self.config.gr_min
        gr_max = self.config.gr_max

        # Auto-detect GR range if not provided
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

        vcl_module = VclModule(
            gr_min=gr_min,
            gr_max=gr_max,
            gr_method=self.config.gr_method,
            use_sp=use_sp,
            use_nd=self._has_curve(well, "NPHI") and self._has_curve(well, "RHOB"),
            sp_min=self.config.sp_min,
            sp_max=self.config.sp_max
        )
        result = vcl_module.calculate(well)
        vcl_module.add_to_well(well, result)

    def _step_rcl(self, well: "Well") -> None:
        from petrophys.clay_volume.rcl import RclModule

        rcl_module = RclModule()
        self._result.rcl_computed = rcl_module.calculate(well)

    def _step_lithology(self, well: "Well") -> None:
        from petrophys.lithology.lithology_module import LithologyModule

        litho_module = LithologyModule()
        result = litho_module.calculate(well)
        litho_module.add_to_well(well, result)

    def _step_rmf(self, well: "Well") -> None:
        from petrophys.resistivity.rmf.rmf_module import RmfModule

        if self.config.mud_type == "OBM":
            self._result.rmf_computed = 200.0
            return

        # Get formation temperature
        temp_f = self._get_avg_formation_temp(well)

        if self.config.rmf_at_meas_temp is not None:
            rmf_module = RmfModule()
            result = rmf_module.calculate(
                rmf_surface=self.config.rmf_at_meas_temp,
                temp_surface=self.config.rmf_meas_temp_f,
                temp_formation=temp_f
            )
            self._result.rmf_computed = result.rmf
        elif self.config.rm is not None:
            k = 0.75 if self.config.mud_water == "fresh" else 0.85
            self._result.rmf_computed = self.config.rm * k

    def _step_rw(self, well: "Well") -> None:
        if self.config.rw is not None:
            self._result.rw_computed = self.config.rw
            return

        temp_f = self._get_avg_formation_temp(well)

        if self.config.salinity_ppm is not None:
            from petrophys.resistivity.rw.salinity_rw import rw_from_salinity
            self._result.rw_computed = rw_from_salinity(
                self.config.salinity_ppm, temp_f
            )
            return

        # Try SP method
        if (self._has_curve(well, "SP") and
                not self.config.mud_has_clay_or_kcl and
                self._result.rmf_computed is not None):
            from petrophys.resistivity.rw.sp_rw import rw_from_sp

            sp = well.get("SP")
            valid_sp = sp.data[~np.isnan(sp.data)]
            if len(valid_sp) > 0:
                ssp = float(np.min(valid_sp))
                rw = rw_from_sp(ssp, self._result.rmf_computed, temp_f)
                self._result.rw_computed = rw
                return

        self._result.warnings.append("Rw could not be determined. Using default 0.05 ohm.m")
        self._result.rw_computed = 0.05

    def _step_invasion(self, well: "Well") -> None:
        from petrophys.resistivity.invasion.invasion_module import InvasionModule

        invasion_module = InvasionModule()
        result = invasion_module.analyze(well)
        well.add_curve("INVASION_RATIO", result.ratio, "", "Invasion Ratio Rxo/Rt")

    def _step_hc_correction(self, well: "Well") -> None:
        from petrophys.petrophysics.hc_correction.hc_correction_module import HcCorrectionModule
        from petrophys.core.constants import MatrixProperties

        rho_ma = self.config.rho_ma
        if rho_ma is None:
            try:
                rho_ma = MatrixProperties.get(self.config.matrix_type).density
            except KeyError:
                rho_ma = 2.65

        hc_module = HcCorrectionModule(
            hc_type=self.config.hc_type,
            rho_ma=rho_ma,
            rho_fl=self.config.rho_fl,
            phi_cl=self.config.phi_cl
        )
        result = hc_module.calculate(well)
        hc_module.add_to_well(well, result)

    def _step_porosity(self, well: "Well") -> None:
        from petrophys.petrophysics.porosity.porosity_module import PorosityModule

        por_module = PorosityModule(
            matrix=self.config.matrix_type,
            rhob_fluid=self.config.rho_fl,
            phi_clay=self.config.phi_cl
        )
        result = por_module.calculate(well)
        por_module.add_to_well(well, result)

    def _step_saturation(self, well: "Well") -> None:
        from petrophys.petrophysics.saturation.sw_module import SaturationModule

        sw_module = SaturationModule(
            a=self.config.a,
            m=self.config.m,
            n=self.config.n,
            model=self.config.sw_model,
            vcl_cutoff=self.config.vcl_cutoff
        )

        lithology = self._build_lithology_log(well)

        result = sw_module.calculate(
            well,
            rw=self._result.rw_computed,
            rcl=self._result.rcl_computed,
            salinity=self.config.salinity_ppm,
            rmf=self._result.rmf_computed,
            lithology=lithology,
            cec=self.config.cec,
            b_coeff=self.config.b_coeff,
            rho_ma=self.config.rho_ma or 2.65,
            m_star=self.config.m_star,
            rwb=self.config.rwb,
            phi_cl=self.config.phi_cl
        )

        well.add_curve("SW", result.sw, "v/v", f"Water Saturation ({result.model})")
        if result.sxo is not None:
            well.add_curve("SXO", result.sxo, "v/v", "Flushed Zone Water Saturation")

    def _step_permeability(self, well: "Well") -> None:
        from petrophys.petrophysics.permeability.perm_module import PermeabilityModule

        perm_module = PermeabilityModule(method=self.config.perm_method)

        swi = self.config.swirr
        if swi is None:
            # Estimate Swirr from net pay zones if available
            sw_curve = well.get("SW")
            net_pay = well.get("NET_PAY")
            if sw_curve is not None and net_pay is not None:
                pay_mask = net_pay.data > 0.5
                if np.any(pay_mask):
                    swi = float(np.nanmean(sw_curve.data[pay_mask]))

        result = perm_module.calculate(well, swi=swi)
        well.add_curve("PERM", result.perm, "mD", f"Permeability ({result.method})")

        # Flag impermeable zones if sonic and density porosity available
        if self._has_curve(well, "PHIS") and self._has_curve(well, "PHID"):
            from petrophys.petrophysics.permeability.impermeable_flag import flag_impermeable

            phis = well.get("PHIS").data
            phid = well.get("PHID").data
            nphi = well.get("NPHI")
            phi_n = nphi.data if nphi is not None else phid

            imperm = flag_impermeable(phis, phid, phi_n)
            well.add_curve("IMPERM_FLAG", imperm.astype(float), "", "Impermeable Flag")

    def _step_rock_quality(self, well: "Well") -> None:
        from petrophys.reservoir.rock_quality.rock_quality_module import RockQualityModule

        rq_module = RockQualityModule()
        result = rq_module.calculate(well)
        rq_module.add_to_well(well, result)

    def _step_bvw(self, well: "Well") -> None:
        from petrophys.reservoir.bvw.bvw_module import calculate_bvw

        phie = well.get("PHIE").data
        sw = well.get("SW").data
        bvw = calculate_bvw(phie, sw)
        well.add_curve("BVW", bvw, "v/v", "Bulk Volume Water")

    def _step_mhi(self, well: "Well") -> None:
        from petrophys.reservoir.mhi.mhi_module import MhiModule

        mhi_module = MhiModule()
        mhi_module.add_to_well(well)

    def _step_net_pay(self, well: "Well") -> None:
        from petrophys.reservoir.net_pay.net_pay_module import NetPayModule

        lithology = self._build_lithology_log(well)

        np_module = NetPayModule(
            vcl_cutoff=self.config.vcl_cutoff,
            phie_cutoff=self.config.phie_cutoff,
            sw_cutoff=self.config.sw_cutoff,
            lithology=lithology
        )
        result = np_module.calculate(well)
        np_module.add_to_well(well, result)

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
