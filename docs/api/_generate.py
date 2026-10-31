"""Build the grouped API reference from the area map below.

``docs/conf.py`` calls :func:`write_reference` before Sphinx reads any source.
It imports every module named in :data:`AREAS`, reads its ``__all__`` (or, for
the few modules without one, the public functions and classes defined in it)
and writes one page per area, plus a page per large subpackage, into
``docs/api/reference/`` (gitignored). Autosummary then writes one page per
public function and class next to them, from the templates in
``docs/_templates/autosummary/``. Constants are listed on their section's page.

Each object is documented once, at its canonical path: of the area-map modules
that export it, the one whose name is the longest prefix of the module that
defines it. Every other public path is an alias: the object's page lists it
("Also available as ..."), and :meth:`ReferencePlan.resolve_reference`
resolves cross-references to it, so ``petrophys.read`` and
``petrophys.workflow.Workflow`` still link to the canonical page.

Names a module exports but no curated group lists fall into automatic
Functions, Classes and Constants groups, so the reference never drifts from
the code.
"""

from __future__ import annotations

import importlib
import inspect
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence, Union

# -- Area map --------------------------------------------------------------------
# Names are space-separated. A Section is one module, or the part of one given
# by ``names``; its ``groups`` are curated (title, names) tables. Every section
# has a page of its own, so the sidebar always reads area, section page,
# objects: a Section placed directly on an area becomes a one-section page
# (``slug`` defaults to the module), and a Page holds several sections. The
# area page shows each page's ``key`` entry points (by default the first few
# objects) and links to it. A Topic is a heading on an area page. A ``flat``
# area is itself the page of its one section, and an area without a ``card``
# is left off the landing grid.


@dataclass
class Section:
    module: str
    title: Optional[str] = None
    description: Optional[str] = None
    names: Optional[str] = None
    groups: Sequence[tuple[str, str]] = ()
    note: Optional[str] = None
    slug: Optional[str] = None
    key: str = ""


@dataclass
class Page:
    slug: str
    title: str
    sections: Sequence[Section]
    key: str = ""
    description: Optional[str] = None


@dataclass
class Topic:
    title: str
    items: Sequence[Union[Section, Page]]
    description: Optional[str] = None


@dataclass
class Area:
    slug: str
    title: str
    card: Optional[str]
    intro: str
    key: str
    items: Sequence[Union[Section, Page, Topic]]
    legacy: bool = False
    flat: bool = False


INTERP = "petrophys.interpretation"
VIZ = "petrophys.visualization"

AREAS: list[Area] = [
    Area(
        slug="front-door",
        title="Front door",
        card=None,
        flat=True,
        intro="""Three calls cover most work: :py:obj:`~petrophys.api.read` a well,
:py:obj:`~petrophys.api.interpret` it, and :py:obj:`~petrophys.api.plot` the
result. All three are in the top-level namespace as ``pp.read``,
``pp.interpret`` and ``pp.plot``.

.. code-block:: python

   import petrophys as pp

   well = pp.read(pp.sample("30_6-5"))
   result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230)
   pp.plot(result, depth_range=(2800, 3000))

The rest of the reference documents what these calls are built from.""",
        key="petrophys.api.read petrophys.api.interpret petrophys.api.plot",
        items=[Section("petrophys.api", names="read interpret plot")],
    ),
    Area(
        slug="data-io",
        title="Data model and I/O",
        card="Wells, curves, intervals and core data, readers and writers for "
        "LAS, DLIS, CSV and Excel, and the bundled sample wells.",
        intro="The objects every other module works on, the functions that read "
        "them from files and write them back, and the sample data that ships "
        "with the package.",
        key="petrophys.model.Well petrophys.model.Curve petrophys.io.read_las "
        "petrophys.io.read_formations",
        items=[
            Page(
                "petrophys.model",
                "Data model",
                key="petrophys.model.Well petrophys.model.Curve "
                "petrophys.model.FormationTops petrophys.model.CoreData",
                description="Wells, curves, intervals, core data, reference "
                "properties and class tables, and the depth utilities they use.",
                sections=[
                    Section(
                        "petrophys.model",
                        title="Data structures",
                        groups=[
                            ("Wells and curves", "Well Curve DataType"),
                            (
                                "Intervals and formation tops",
                                "FormationTops Formation Zone SubZone ManualInterval "
                                "Interval IntervalKind label_depths",
                            ),
                            ("Core data", "CoreData"),
                            ("Units", "UnitConverter UnitRegistry"),
                            (
                                "Reference rock and fluid properties",
                                "MatrixProperties FluidProperties ArchieCoefficients "
                                "ArchieParameterDefaults",
                            ),
                            # CLASS_TABLES, the registry of these tables, is a
                            # constant: the section lists it under Constants.
                            (
                                "Class tables",
                                "ClassTable ReservoirQualityTable ReservoirGrade Bands",
                            ),
                            (
                                "Classifications",
                                "VclClassification PorosityClassification "
                                "HcSaturationClassification PermeabilityClassification "
                                "RQIClassification MhiClassification InvasionClassification",
                            ),
                            (
                                "Two-curve class tables, which no canonical table uses",
                                "TwoCurveClassTable TwoCurveBands",
                            ),
                        ],
                    ),
                    Section("petrophys.model.log_data", title="Depth utilities"),
                ],
            ),
            Page(
                "petrophys.io",
                "Reading and writing files",
                key="petrophys.io.read_las petrophys.io.read_formations "
                "petrophys.io.read_lithology_intervals petrophys.io.export_las",
                sections=[
                    Section(
                        "petrophys.io",
                        groups=[
                            (
                                "Well logs",
                                "read_las read_dlis read_dlis_merged read_csv "
                                "read_csv_wells read_ascii",
                            ),
                            (
                                "Formations and lithology",
                                "read_formations read_lithology_intervals "
                                "read_lithology_codes",
                            ),
                            ("Core analysis tables", "read_core"),
                            (
                                "Deviation surveys and depth datums",
                                "read_survey attach_survey depth_reference_elevation "
                                "minimum_curvature Survey",
                            ),
                            ("Export", "export_las export_csv export_excel"),
                            ("Reader classes", "LASReader DLISReader"),
                        ],
                    )
                ],
            ),
            Section(
                "petrophys.datasets",
                title="Datasets",
                groups=[
                    ("Bundled samples", "sample list_samples"),
                    ("Large fixtures", "integration_fixture list_integration_fixtures data_dir"),
                ],
            ),
            Section("petrophys.exceptions", title="Exceptions"),
        ],
    ),
    Area(
        slug="preprocessing",
        title="Preprocessing",
        card="Standardize mnemonics and units, screen value ranges and bad "
        "hole, normalize curves across wells and impute missing samples.",
        intro="Prepare raw logs before interpretation. By default ``pp.interpret`` "
        "standardizes curves and screens value ranges itself; the functions "
        "here let you run and inspect each step.",
        key="petrophys.preprocessing.standardization.standardize_curves "
        "petrophys.preprocessing.quality.generate_bad_hole_flags "
        "petrophys.preprocessing.normalization.normalize_log "
        "petrophys.preprocessing.imputation.impute_missing_curve",
        items=[
            Section(
                "petrophys.preprocessing.standardization",
                title="Standardization",
                groups=[
                    (
                        "Curves and mnemonics",
                        "standardize_curves add_standard_curve_aliases "
                        "get_standard_mnemonic CurveStandardizer",
                    ),
                    (
                        "Your curve names",
                        "read_curve_aliases attach_curve_aliases curve_aliases_of "
                        "standard_curve_names user_standard_name",
                    ),
                    (
                        "Units",
                        "standardize_units convert_curve_units standardize_core_saturation",
                    ),
                ],
            ),
            Section(
                "petrophys.preprocessing.quality",
                title="Quality control",
                groups=[
                    (
                        "Bad hole and washout",
                        "generate_bad_hole_flags generate_washout_flag generate_large_hole_flag "
                        "generate_low_density_flag generate_pef_exclusion_flag "
                        "mask_bad_hole_curves apply_quality_flags detect_washout "
                        "WashoutDetection WellboreQC",
                    ),
                    ("Held readings", "find_held_readings held_reading_mask HeldReadings"),
                    (
                        "Value ranges",
                        "apply_value_range_qc clean_curve_values value_range_qc_note",
                    ),
                ],
            ),
            Section("petrophys.preprocessing.normalization", title="Normalization"),
            Section(
                "petrophys.preprocessing.imputation",
                title="Curve imputation",
                note="Needs the ``ml`` extra (Python 3.11 or later).",
            ),
        ],
    ),
    Area(
        slug="interpretation",
        title="Interpretation",
        card="Temperature, clay volume, porosity, lithology, electrical "
        "properties, water saturation, permeability, rock physics and "
        "specialist data.",
        intro="Deterministic calculations that turn measurements into "
        "petrophysical properties. Most functions take and return NumPy "
        "arrays; the ``calculate_*`` functions and ``*Module`` classes work on "
        "a whole :class:`~petrophys.model.Well`.",
        key=f"{INTERP}.clay_volume.calculate_vcl {INTERP}.porosity.calculate_porosity "
        f"{INTERP}.saturation.calculate_sw {INTERP}.permeability.calculate_permeability",
        items=[
            Section(
                f"{INTERP}.temperature",
                title="Temperature",
                groups=[
                    ("Temperature profile", "calculate_temperature_profile TemperatureProfile"),
                    (
                        "Geothermal gradient",
                        "estimate_gradient_from_bht calculate_temperature_gradient "
                        "gradient_c_per_km_to_fahrenheit_per_100ft gradient_to_celsius_per_km",
                    ),
                    (
                        "Resistivity at temperature",
                        "correct_resistivity_temperature arps_equation",
                    ),
                ],
            ),
            Page(
                f"{INTERP}.clay_volume",
                "Clay volume",
                key=f"{INTERP}.clay_volume.calculate_vcl {INTERP}.clay_volume.vcl_from_gr "
                f"{INTERP}.clay_volume.vcl_from_nd {INTERP}.clay_volume.calculate_rcl",
                sections=[
                    Section(
                        f"{INTERP}.clay_volume",
                        groups=[
                            ("Clay volume of a well", "calculate_vcl VclModule classify_vcl"),
                            (
                                "Gamma-ray methods",
                                "vcl_from_gr vcl_linear vcl_larionov_tertiary "
                                "vcl_larionov_older vcl_steiber vcl_clavier",
                            ),
                            ("SP and neutron-density methods", "vcl_from_sp vcl_from_nd"),
                            ("Clay resistivity", "calculate_rcl RclModule"),
                            (
                                "Shale distribution (Thomas-Stieber)",
                                "thomas_stieber ThomasStieberResult laminated_sand_resistivity",
                            ),
                        ],
                    )
                ],
            ),
            Page(
                f"{INTERP}.porosity",
                "Porosity",
                description="Density, neutron, sonic, total and effective "
                "porosity, with hydrocarbon corrections.",
                key=f"{INTERP}.porosity.calculate_porosity {INTERP}.porosity.density_porosity "
                f"{INTERP}.porosity.effective_porosity "
                f"{INTERP}.porosity.hydrocarbon_correction.apply_hc_correction",
                sections=[
                    Section(
                        f"{INTERP}.porosity",
                        title="Porosity",
                        groups=[
                            (
                                "Porosity of a well",
                                "calculate_porosity PorosityModule classify_porosity",
                            ),
                            (
                                "Density, neutron and sonic porosity",
                                "density_porosity calculate_phid neutron_porosity "
                                "standardize_nphi sonic_porosity wyllie_time_average "
                                "raymer_hunt_gardner",
                            ),
                            (
                                "Total and effective porosity",
                                "total_porosity nd_weighted_porosity effective_porosity "
                                "calculate_phie",
                            ),
                        ],
                    ),
                    Section(
                        f"{INTERP}.porosity.hydrocarbon_correction",
                        title="Hydrocarbon correction",
                    ),
                ],
            ),
            Page(
                f"{INTERP}.lithology",
                "Lithology and minerals",
                key=f"{INTERP}.lithology.calculate_lithology "
                f"{INTERP}.lithology.MultiMineralSolver "
                f"{INTERP}.lithology.nd_crossplot_data {INTERP}.lithology.build_lithology_log",
                sections=[
                    Section(
                        f"{INTERP}.lithology",
                        title="Lithology",
                        groups=[
                            ("Lithology of a well", "calculate_lithology LithologyModule"),
                            (
                                "Mineral solvers",
                                "MultiMineralSolver MultiMineralSolverV1 solve_two_mineral solve_three_mineral "
                                "get_mineral_properties",
                            ),
                            (
                                "Crossplot data",
                                "nd_crossplot_data mn_crossplot_data pef_crossplot_data",
                            ),
                            (
                                "Lithology logs from intervals",
                                "build_lithology_log load_lithology_csv LithologyInterval",
                            ),
                            (
                                "Apparent matrix parameters",
                                "apparent_matrix_density apparent_matrix_slowness "
                                "apparent_matrix_volumetric_pe apparent_total_porosity "
                                "electron_density volumetric_photoelectric",
                            ),
                        ],
                    ),
                    Section(
                        f"{INTERP}.lithology.labels",
                        title="Lithology labels",
                        description="Rock names in interval files and their lithology groups.",
                    ),
                    Section(
                        f"{INTERP}.lithology.carbonate_minerals",
                        title="Carbonate minerals",
                        description="Calcite and dolomite volumes from bulk density "
                        "and closure: the carbonate-domain mineral step of "
                        "``pp.interpret``.",
                    ),
                ],
            ),
            Page(
                f"{INTERP}.electrical",
                "Electrical properties",
                description="Formation water and mud filtrate resistivity, "
                "invasion, and Archie cementation parameters.",
                key=f"{INTERP}.electrical.rw.calculate_rw {INTERP}.electrical.rw.rw_from_sp "
                f"{INTERP}.electrical.rmf.calculate_rmf "
                f"{INTERP}.electrical.cementation.pickett_analysis",
                sections=[
                    Section(
                        f"{INTERP}.electrical.rw",
                        title="Formation water resistivity (Rw)",
                        groups=[
                            ("Rw of a well", "calculate_rw RwModule"),
                            (
                                "Rw from the SP",
                                "rw_from_sp sp_rw_in_window SpRwEstimate rw_from_ssp "
                                "calculate_ssp estimate_sp_lines",
                            ),
                            (
                                "Salinity and temperature",
                                "rw_from_salinity salinity_from_rw correct_rw_temperature",
                            ),
                            ("Apparent water resistivity", "apparent_water_resistivity sw_from_rwa"),
                        ],
                    ),
                    Section(f"{INTERP}.electrical.rmf", title="Mud filtrate resistivity (Rmf)"),
                    Section(f"{INTERP}.electrical.invasion", title="Invasion"),
                    Section(
                        f"{INTERP}.electrical.cementation",
                        title="Cementation and Archie parameters",
                    ),
                ],
            ),
            Page(
                f"{INTERP}.saturation",
                "Water saturation",
                description="Saturation models, saturation height and "
                "formation pressures, and sonic saturation.",
                key=f"{INTERP}.saturation.calculate_sw {INTERP}.saturation.archie "
                f"{INTERP}.saturation.height.fit_j_function "
                f"{INTERP}.saturation.height.fit_pressure_fwl",
                sections=[
                    Section(
                        f"{INTERP}.saturation",
                        title="Saturation models",
                        groups=[
                            (
                                "Saturation of a well",
                                "calculate_sw SaturationModule SaturationResult SwModel "
                                "classify_sw",
                            ),
                            (
                                "Archie and shaly-sand equations",
                                "archie simandoux modified_simandoux indonesia north_sea_shaly_sand "
                                "nigeria_schlumberger dual_water waxman_smits",
                            ),
                            ("Flushed zone", "calculate_sxo"),
                        ],
                    ),
                    Section(
                        f"{INTERP}.saturation.height",
                        title="Saturation height",
                        groups=[
                            (
                                "Capillary pressure and J-function",
                                "pc_from_height height_from_pc convert_pc_lab_to_reservoir "
                                "leverett_j pc_from_j",
                            ),
                            (
                                "Saturation-height models",
                                "fit_j_function fit_brooks_corey saturation_height_from_logs "
                                "JFunctionModel BrooksCoreyModel",
                            ),
                            (
                                "Empirical functions from log Sw and a known FWL",
                                "fit_saturation_height fit_saturation_height_from_well "
                                "height_above_fwl FittedSaturationHeight",
                            ),
                            (
                                "Formation pressure and free-water level",
                                "read_formation_pressure fit_pressure_fwl "
                                "FormationPressureData PressureGradient PressureFWLResult",
                            ),
                            (
                                "Unlabeled pressure contacts",
                                "infer_pressure_contacts PressureBranch PressureContact "
                                "PressureContactResult",
                            ),
                            (
                                "Raw MDT pretest pressure picking",
                                "read_mdt_pretest pick_mdt_pressure formation_pressures_from_mdt "
                                "MDTPickingConfig MDTPretest MDTBuildup MDTPressurePick",
                            ),
                        ],
                    ),
                    Section(
                        f"{INTERP}.saturation.sonic",
                        title="Sonic saturation",
                        groups=[
                            (
                                "Sonic saturation of a well",
                                "sonic_water_saturation SonicSaturationModule "
                                "SonicSaturationResult",
                            ),
                            (
                                "Velocities, moduli and Gassmann",
                                "slowness_to_velocity vs_from_vp_greenberg_castagna "
                                "velocity_to_moduli dry_rock_modulus gassmann_ksat "
                                "gassmann_kfl wood_fluid_modulus sw_from_fluid_modulus "
                                "fluid_substitution_dtc_swing",
                            ),
                        ],
                    ),
                ],
            ),
            Section(
                f"{INTERP}.permeability",
                title="Permeability",
                groups=[
                    (
                        "Permeability of a well",
                        "calculate_permeability PermeabilityModule CorePermModel "
                        "classify_permeability",
                    ),
                    ("Equations", "timur coates_denoo wyllie_rose lith_vcl_permeability"),
                    (
                        "Irreducible water saturation",
                        "resolve_swirr estimate_hc_swirr estimate_swirr_app "
                        "estimate_apparent_swirr",
                    ),
                    ("Impermeable zones", "flag_impermeable"),
                ],
            ),
            Section(f"{INTERP}.rock_physics", title="Rock physics"),
            Section(
                f"{INTERP}.pore_pressure",
                title="Pore pressure",
                description="Overburden and hydrostatic pressure, normal compaction "
                "trends and Eaton's method.",
            ),
            Topic(
                "Specialist data",
                description="NMR, borehole images, geochemistry and core.",
                items=[
                    Section(
                        f"{INTERP}.nmr",
                        title="NMR",
                        groups=[
                            ("NMR data", "NMRData NMRCalibration"),
                            (
                                "Pore volumes and permeability",
                                "t2lm_from_bins bvi_from_bins ffi_from_bins "
                                "t2_fraction_below_cutoff sdr_permeability "
                                "coates_permeability",
                            ),
                        ],
                    ),
                    Page(
                        f"{INTERP}.borehole_imaging",
                        "Borehole imaging",
                        key=f"{INTERP}.borehole_imaging.read_fmi_dlis "
                        f"{INTERP}.borehole_imaging.orient_to_azimuth "
                        f"{INTERP}.borehole_imaging.equalize_dynamic "
                        f"{INTERP}.borehole_imaging.plot_fmi",
                        sections=[
                            Section(
                                f"{INTERP}.borehole_imaging",
                                groups=[
                                    (
                                        "Reading images",
                                        "read_fmi_dlis read_borehole_image_dlis "
                                        "read_borehole_image_las BoreholeImageData",
                                    ),
                                    (
                                        "Navigation and orientation",
                                        "orient_to_azimuth azimuthal_coverage "
                                        "with_navigation get_navigation "
                                        "filter_by_depth_with_navigation",
                                    ),
                                    (
                                        "Equalization and display",
                                        "equalize_static equalize_dynamic "
                                        "safe_per_depth_statistics fmi_colormap plot_fmi",
                                    ),
                                ],
                            )
                        ],
                    ),
                    Page(
                        f"{INTERP}.geochemistry",
                        "Geochemistry",
                        key=f"{INTERP}.geochemistry.interpret_spectral_gamma "
                        f"{INTERP}.geochemistry.interpret_elemental_spectroscopy "
                        f"{INTERP}.geochemistry.thorium_clay_index "
                        f"{INTERP}.geochemistry.XRFData",
                        sections=[
                            Section(
                                f"{INTERP}.geochemistry",
                                groups=[
                                    (
                                        "Spectral gamma ray",
                                        "interpret_spectral_gamma has_spectral_gamma_inputs "
                                        "SpectralGammaResult uranium_stripped_gamma_ray",
                                    ),
                                    (
                                        "Thorium and uranium",
                                        "thorium_clay_index ThoriumClayIndex "
                                        "classify_th_u_environment ThUEnvironment",
                                    ),
                                    (
                                        "Clay typing from Th/K",
                                        "classify_clay_th_k clay_typing_from_well "
                                        "ClayTypingResult",
                                    ),
                                    (
                                        "Elemental spectroscopy",
                                        "interpret_elemental_spectroscopy "
                                        "has_elemental_spectroscopy_inputs "
                                        "ElementalSpectroscopyResult oxide_closure "
                                        "OxideClosureResult",
                                    ),
                                    ("XRF", "XRFData"),
                                ],
                            )
                        ],
                    ),
                    Page(
                        f"{INTERP}.core_analysis",
                        "Core analysis",
                        key=f"{INTERP}.core_analysis.core_data_from_well "
                        f"{INTERP}.core_analysis.calibrate_porosity "
                        f"{INTERP}.core_analysis.fit_poro_perm_transform "
                        f"{INTERP}.core_analysis.find_core_depth_shift",
                        sections=[
                            Section(
                                f"{INTERP}.core_analysis",
                                groups=[
                                    (
                                        "Core data from a well",
                                        "core_data_from_well interpolate_log_at_core",
                                    ),
                                    (
                                        "Calibration against logs",
                                        "calibrate_porosity calibrate_permeability "
                                        "calibrate_saturation CalibrationResult",
                                    ),
                                    (
                                        "Porosity correction",
                                        "fit_porosity_correction apply_porosity_correction "
                                        "correct_porosity_with_grain_density "
                                        "PorosityCorrection",
                                    ),
                                    (
                                        "Poro-perm transforms",
                                        "fit_poro_perm_transform fit_log_poro_perm_transform "
                                        "fit_log_poro_perm_transform_from_correction "
                                        "apply_poro_perm_transform PoroPermTransform",
                                    ),
                                    ("Archie exponent from core", "estimate_archie_m"),
                                    (
                                        "Depth matching",
                                        "find_core_depth_shift shift_core_depths "
                                        "CoreDepthShift",
                                    ),
                                    (
                                        "Grain size, mechanics and core images",
                                        "GrainSizeData wentworth_class um_to_phi "
                                        "CoreMechanical CoreImageData",
                                    ),
                                    ("Digital core facies", "DigitalCoreFacies FaciesResult"),
                                ],
                            )
                        ],
                    ),
                ],
            ),
        ],
    ),
    Area(
        slug="evaluation",
        title="Evaluation",
        card="Bulk volume water, moveable hydrocarbons, net pay, rock quality "
        "and flow units, zone summaries, volumetrics, sensitivity and "
        "diagnostics.",
        intro="Reservoir decisions and summaries built on interpreted curves.",
        key="petrophys.evaluation.net_pay.calculate_net_pay "
        "petrophys.evaluation.net_pay.summarize_pay "
        "petrophys.evaluation.flow_units.calculate_flow_units "
        "petrophys.evaluation.sensitivity.cutoff_sensitivity",
        items=[
            Page(
                "evaluation-bvw-mhi",
                "BVW and MHI",
                description="Bulk volume water and the moveable hydrocarbon index.",
                sections=[
                    Section("petrophys.evaluation.bvw", title="Bulk volume water"),
                    Section("petrophys.evaluation.mhi", title="Moveable hydrocarbon index"),
                ],
            ),
            Page(
                "petrophys.evaluation.net_pay",
                "Net pay",
                key="petrophys.evaluation.net_pay.calculate_net_pay "
                "petrophys.evaluation.net_pay.summarize_pay "
                "petrophys.evaluation.net_pay.generate_net_flags "
                "petrophys.evaluation.net_pay.standard_cutoffs",
                sections=[
                    Section(
                        "petrophys.evaluation.net_pay",
                        groups=[
                            ("Net pay of a well", "calculate_net_pay NetPayModule NetPayResult"),
                            ("Pay summaries", "summarize_pay PaySummary"),
                            (
                                "Screening",
                                "grade_interval cutoff_attrition lithology_in majority_label",
                            ),
                            ("Cutoffs", "standard_cutoffs default_cutoffs Cutoffs"),
                            (
                                "Flags",
                                "generate_net_flags generate_volumetric_pay_flag "
                                "add_volumetric_pay_flag",
                            ),
                        ],
                    )
                ],
            ),
            Page(
                "evaluation-rock-quality",
                "Rock quality and flow units",
                description="Rock quality index, flow zone indicator and "
                "hydraulic flow units.",
                sections=[
                    Section("petrophys.evaluation.rock_quality", title="Rock quality"),
                    Section("petrophys.evaluation.flow_units", title="Flow units"),
                ],
            ),
            Page(
                "petrophys.evaluation.summaries",
                "Intervals and summaries",
                description="Interval analysis, zone and formation summaries, "
                "averages and net-to-gross.",
                key="petrophys.evaluation.summaries.analyze_formations "
                "petrophys.evaluation.summaries.calculate_zone_summary "
                "petrophys.evaluation.intervals.analyze_interval "
                "petrophys.evaluation.intervals.flag_intervals",
                sections=[
                    Section("petrophys.evaluation.intervals", title="Intervals"),
                    Section(
                        "petrophys.evaluation.summaries",
                        title="Zone and formation summaries",
                        groups=[
                            ("Zones", "calculate_zone_summary summarize_zones ZoneSummary"),
                            ("Curve statistics", "curve_statistics zone_statistics percentile_key"),
                            ("Zone ranking", "best_and_weakest property_direction"),
                            (
                                "Formations",
                                "analyze_formations run_formation_analysis "
                                "load_formations_csv FormationSummary FormationAnalysisResult",
                            ),
                            (
                                "Net to gross",
                                "calculate_net_gross calculate_ntg_by_zone NetGrossResult",
                            ),
                            (
                                "Averages",
                                "calculate_zone_averages calculate_pay_averages "
                                "calculate_weighted_averages calculate_pore_volume_weighted_sw",
                            ),
                        ],
                    ),
                ],
            ),
            Section("petrophys.evaluation.volumetrics", title="Volumetrics"),
            Section("petrophys.evaluation.sensitivity", title="Sensitivity"),
            Section("petrophys.evaluation.uncertainty", title="Monte Carlo uncertainty"),
            Section("petrophys.evaluation.heterogeneity", title="Permeability heterogeneity"),
            Section("petrophys.evaluation.zonation", title="Change-point zonation"),
            Section("petrophys.evaluation.diagnostics", title="Diagnostics"),
        ],
    ),
    Area(
        slug="workflows",
        title="Workflows",
        card="The dynamic workflow behind pp.interpret, custom workflows built "
        "from steps, batch and zoned runs, and the settings schema.",
        intro="Run complete interpretations. ``pp.interpret`` runs the dynamic "
        "workflow; :class:`~petrophys.workflow.custom.Workflow` runs the steps "
        "you choose.",
        key="petrophys.workflow.DynamicWorkflowConfig "
        "petrophys.workflow.DynamicWorkflowResult petrophys.workflow.custom.Workflow "
        "petrophys.workflow.custom.workflow_step",
        items=[
            Page(
                "petrophys.workflow",
                "The pp.interpret engine",
                description="The input-aware workflow that ``pp.interpret`` runs: "
                "settings, steps, results and traces, and the quick-look and "
                "comprehensive analyses built on it.",
                key="petrophys.workflow.DynamicWorkflowConfig "
                "petrophys.workflow.DynamicWorkflowResult "
                "petrophys.workflow.run_dynamic_workflow petrophys.workflow.DynamicWorkflow",
                sections=[
                    Section(
                        "petrophys.workflow",
                        title="Dynamic workflow",
                        description="The input-aware workflow behind ``pp.interpret``: "
                        "settings, steps, results and traces.",
                        names="DynamicWorkflowConfig DynamicWorkflowResult run_dynamic_workflow "
                        "DynamicWorkflow StepResult WorkflowStep",
                        groups=[
                            ("Settings and results", "DynamicWorkflowConfig DynamicWorkflowResult StepResult"),
                            ("Running the workflow", "run_dynamic_workflow DynamicWorkflow WorkflowStep"),
                        ],
                    ),
                    Section(
                        "petrophys.workflow",
                        title="Quick look and comprehensive analysis",
                        description="Fast screening of a well, and every diagnostic "
                        "analysis in one report.",
                        names="quick_look_analysis QuickLook run_comprehensive_analysis",
                    ),
                ],
            ),
            Page(
                "petrophys.workflow.custom",
                "Custom workflows",
                description="Build, check, run and share your own sequence of "
                "steps, from built-in steps or your own functions.",
                key="petrophys.workflow.custom.Workflow petrophys.workflow.custom.workflow_step "
                "petrophys.workflow.custom.WorkflowRunResult petrophys.workflow.steps.saturation",
                sections=[
                    Section(
                        "petrophys.workflow.custom",
                        title="Workflows and steps",
                        groups=[
                            ("Building workflows", "Workflow workflow_step Step"),
                            ("Results", "WorkflowRunResult WorkflowBatchResult"),
                            ("Errors", "WorkflowValidationError WorkflowStepError"),
                        ],
                    ),
                    Section(
                        "petrophys.workflow.steps",
                        title="Built-in steps",
                        groups=[
                            ("Preparation", "standardize temperature"),
                            (
                                "Interpretation",
                                "clay_volume porosity formation_water mud_filtrate "
                                "saturation permeability spectral_gamma",
                            ),
                            ("Evaluation", "rock_quality bvw mhi net_pay formation_summary"),
                            ("The whole pp.interpret workflow", "interpret"),
                        ],
                    ),
                ],
            ),
            Section(
                "petrophys.workflow",
                title="Batch and zoned runs",
                slug="workflows-batch",
                description="Interpret many wells, folders of LAS files, and "
                "wells whose LAS headers hold depth-varying settings.",
                groups=[
                    (
                        "Several wells",
                        "process_multiple_wells process_las_directory BatchProcessor",
                    ),
                    (
                        "Zoned settings from LAS headers",
                        "run_dynamic_workflow_with_zones extract_zoned_parameters "
                        "ZonedParameterInterval ZonedWorkflowRunResult",
                    ),
                    (
                        "Folders of LAS files and reports",
                        "process_las_directory_with_workflow BatchLasResult "
                        "write_batch_report write_curve_audit_report "
                        "write_workflow_json_traces_directory",
                    ),
                ],
                names="process_multiple_wells process_las_directory BatchProcessor "
                "run_dynamic_workflow_with_zones extract_zoned_parameters "
                "ZonedParameterInterval ZonedWorkflowRunResult "
                "process_las_directory_with_workflow BatchLasResult write_batch_report "
                "write_curve_audit_report write_workflow_json_traces_directory",
            ),
            Section("petrophys.workflow.schema", title="Schema"),
        ],
    ),
    Area(
        slug="visualization",
        title="Visualization",
        card="Log viewers and templates, crossplots, specialist panels, "
        "reservoir displays, analytics charts, core plots and batch export.",
        intro="Plots for wells, interpretations and specialist data. Every "
        "function returns the Matplotlib figure (and axes), so you can style "
        "it further or save it.",
        key="petrophys.visualization.create_log_viewer "
        "petrophys.visualization.create_triple_combo "
        "petrophys.visualization.create_pickett_plot "
        "petrophys.visualization.create_nd_crossplot",
        items=[
            Section(
                VIZ,
                title="Log viewer and templates",
                slug="visualization-log-viewer",
                description="Template-driven multi-track log displays, the "
                "triple combo, and simple log plots.",
                names="create_log_viewer create_adaptive_log_viewer_template "
                "load_log_viewer_template LogViewer LogViewerTemplate create_triple_combo "
                "create_log_plot LogPlot DEFAULT_LOG_VIEWER_TEMPLATES "
                "LITHOLOGY_CODE_LABELS",
            ),
            Section(
                VIZ,
                title="Crossplots",
                slug="visualization-crossplots",
                description="Lithology crossplots (neutron-density, M-N, MID), "
                "Pickett, Buckles and Hingle plots, and general crossplots.",
                names="create_crossplot CrossPlot create_nd_crossplot create_mn_plot "
                "create_mid_plot create_pickett_plot PickettPlot create_buckles_plot "
                "create_hingle_plot",
            ),
            Section(
                VIZ,
                title="Specialist panels",
                slug="visualization-specialist",
                description="NMR, borehole-image and XRF panels, saturation-height "
                "type curves and formation-pressure plots.",
                names="create_nmr_panel has_nmr_curves create_fmi_panel create_xrf_panel "
                "create_saturation_height_plot create_formation_pressure_plot "
                "create_pressure_contact_plot create_mdt_pretest_plot",
            ),
            Section(
                VIZ,
                title="Reservoir displays",
                slug="visualization-reservoir",
                description="Flow-unit crossplots and zone flags on logs.",
                names="create_flow_unit_plot plot_zone_flags add_zone_shading",
            ),
            Section(
                VIZ,
                title="Analytics charts",
                slug="visualization-analytics",
                description="Charts for multi-well comparisons and sensitivity "
                "studies.",
                names="create_multiwell_overlay create_bar_comparison create_tornado_plot "
                "create_histogram_comparison create_line_comparison create_heatmap",
            ),
            Section(
                VIZ,
                title="Core plots",
                slug="visualization-core",
                description="Core-versus-log calibration, poro-perm crossplots and "
                "core points on logs.",
                names="plot_calibration plot_poro_perm plot_core_on_logs",
            ),
            Section(
                VIZ,
                title="Batch export",
                slug="visualization-batch",
                description="Interpretation plots for every LAS file in a folder, "
                "with a CSV report.",
                names="export_interpretation_plots write_visualization_report "
                "BatchVisualizationResult",
            ),
        ],
    ),
    Area(
        slug="reporting",
        title="Reporting and LLM agents",
        card="Compact context for LLMs, delivery packages and audit exports, "
        "and the MCP server for LLM clients.",
        intro="Hand interpreted wells to people and to LLM tools.",
        key="petrophys.reporting.context.build_context "
        "petrophys.reporting.context.export_context "
        "petrophys.mcp.server.create_server petrophys.mcp.Session",
        items=[
            Section("petrophys.reporting.context", title="Context for LLMs"),
            Section(
                "petrophys.reporting",
                title="Exports and delivery packages",
                groups=[
                    (
                        "Context profiles",
                        "list_context_profiles describe_context_profile get_context_profile "
                        "identify_context_profile ContextProfile export_context_profile "
                        "interpret_with_context_profile interpretation_group_records "
                        "context_group_record context_group_statistics",
                    ),
                    (
                        "Example-format exports",
                        "export_example_format_las export_example_format_directory "
                        "write_example_format_report ExampleFormatExportResult",
                    ),
                    (
                        "Work packages",
                        "export_example_work_package_directory "
                        "export_reference_json_work_package_directory "
                        "export_reference_json_work_package_by_formation "
                        "export_reference_json_work_package_by_formation_lithology "
                        "export_reference_json_work_package_by_category_formation_lithology "
                        "write_reference_curve_name_mapping",
                    ),
                ],
            ),
            Page(
                "petrophys.mcp",
                "MCP server",
                description='The MCP server for LLM clients. Install with ``pip '
                'install "petrophys[mcp]"`` and run ``petrophys-mcp``; see :doc:`/mcp`.',
                sections=[
                    Section("petrophys.mcp.server", title="Server"),
                    Section("petrophys.mcp", title="Sessions"),
                ],
            ),
        ],
    ),
    Area(
        slug="legacy",
        title="Legacy",
        card="The fixed-configuration StandardWorkflow and old import paths, "
        "kept for existing code and scheduled for removal.",
        intro="Deprecated: kept for existing code and scheduled for removal. "
        "Use ``pp.interpret`` or :class:`~petrophys.workflow.DynamicWorkflow` "
        "instead.",
        key="petrophys.workflow.standard_workflow.StandardWorkflow "
        "petrophys.workflow.standard_workflow.run_standard_workflow",
        legacy=True,
        items=[Section("petrophys.workflow.standard_workflow", title="StandardWorkflow")],
    ),
]

def _wrap_sections(items: Sequence[Any]) -> list[Any]:
    """Give every section placed directly on an area (or topic) its own page."""
    wrapped = []
    for item in items:
        if isinstance(item, Section):
            item = Page(item.slug or item.module, item.title or item.module, [item],
                        key=item.key)
        elif isinstance(item, Topic):
            item.items = _wrap_sections(item.items)
        wrapped.append(item)
    return wrapped


for _area in AREAS:
    if not _area.flat:
        _area.items = _wrap_sections(_area.items)

#: Thin wrappers of another public function; the wrapper's path becomes an
#: alias of the wrapped function's page.
SAME_AS = {
    "petrophys.mcp.create_server": "petrophys.mcp.server.create_server",
    "petrophys.mcp.main": "petrophys.mcp.server.main",
}

#: One-line descriptions of constants that have no ``#:`` doc comment.
CONSTANT_DOCS = {
    "petrophys.datasets.DATA_DIR": "Folder of the bundled sample data.",
    f"{INTERP}.saturation.height.SATURATION_HEIGHT_MODELS": "Empirical "
    "saturation-height forms :func:`fit_saturation_height` fits.",
    f"{INTERP}.electrical.invasion.RESPONSE_TYPES": "Mixing laws of a resistivity "
    "reading: series (laterolog) or parallel (induction).",
    f"{INTERP}.pore_pressure.EATON_EXPONENTS": "Default Eaton exponents for sonic "
    "and resistivity.",
    "petrophys.evaluation.uncertainty.DISTRIBUTIONS": "Input distributions "
    ":func:`monte_carlo` draws from.",
    "petrophys.datasets.DAYDREAM_DATA_DIR": "Folder of the optional repository "
    "integration fixtures.",
    "petrophys.preprocessing.standardization.ALIAS_MAP": "Vendor mnemonics "
    "recognized for each standard curve name.",
    "petrophys.preprocessing.quality.VALUE_RANGE_RULES": "Plausible range of each "
    "standard curve, and what happens to values outside it.",
    "petrophys.preprocessing.quality.VALUE_RANGE_QC_PARAMETER": "Well parameter "
    "that stores the value-range QC report.",
    f"{INTERP}.lithology.MINERAL_PROPERTIES": "Density, sonic slowness, neutron "
    "porosity and PEF of common minerals.",
    f"{INTERP}.nmr.DEFAULT_T2_CENTERS": "T2 bin centres (ms) used when a "
    "distribution comes without its axis.",
    f"{INTERP}.nmr.T2_CUTOFF_CLASTIC": "Default T2 cutoff for sandstones (ms; Coates, "
    "Xiao and Prammer, 1999).",
    f"{INTERP}.nmr.T2_CUTOFF_CARBONATE": "Default T2 cutoff for carbonates (ms; Coates, "
    "Xiao and Prammer, 1999).",
    f"{INTERP}.geochemistry.DEFAULT_API_COEFFICIENTS": "Gamma-ray API per percent "
    "potassium and per ppm thorium and uranium.",
    f"{INTERP}.geochemistry.ELEMENT_NAMES": "Element names keyed by chemical symbol.",
    f"{INTERP}.saturation.height.FLUID_SYSTEMS": "Interfacial tension (dyn/cm) and "
    "contact angle (degrees) of laboratory and reservoir fluid systems.",
    f"{INTERP}.saturation.height.WATER_GRADIENT_PSI_PER_FT": "Fresh-water pressure "
    "gradient (psi/ft).",
    f"{INTERP}.saturation.height.J_CONST": "Unit constant of the Leverett "
    "J-function for psi, mD and dyn/cm.",
    f"{INTERP}.saturation.sonic.MINERAL_ELASTIC": "Bulk and shear moduli (GPa) and "
    "density (g/cc) of common minerals.",
    f"{INTERP}.saturation.sonic.FLUID_ELASTIC": "Bulk modulus (GPa) and density "
    "(g/cc) of brine, oil and gas.",
    "petrophys.evaluation.net_pay.DEFAULT_HYDROCARBON_FLUIDS": "Fluid labels the "
    "volumetric pay flag counts as hydrocarbon.",
    "petrophys.evaluation.net_pay.DEFAULT_NON_PAY_FLUIDS": "Fluid labels the "
    "volumetric pay flag never counts as pay.",
    "petrophys.visualization.DEFAULT_LOG_VIEWER_TEMPLATES": "Built-in log-viewer "
    "templates (XML), keyed by name.",
}

#: Modules with a page address ``/api/generated/<module>.html``;
#: ``docs/_extra/_redirects`` sends each to its page in the grouped reference.
OLD_MODULE_PAGES = """
petrophys.api petrophys.exceptions petrophys.model petrophys.model.log_data
petrophys.io petrophys.datasets petrophys.preprocessing.standardization
petrophys.preprocessing.normalization petrophys.preprocessing.quality
petrophys.preprocessing.imputation petrophys.interpretation.temperature
petrophys.interpretation.clay_volume petrophys.interpretation.lithology
petrophys.interpretation.porosity petrophys.interpretation.porosity.hydrocarbon_correction
petrophys.interpretation.electrical.rw petrophys.interpretation.electrical.rmf
petrophys.interpretation.electrical.invasion petrophys.interpretation.electrical.cementation
petrophys.interpretation.saturation petrophys.interpretation.permeability
petrophys.interpretation.rock_physics petrophys.interpretation.nmr
petrophys.interpretation.borehole_imaging petrophys.interpretation.geochemistry
petrophys.interpretation.core_analysis petrophys.interpretation.saturation.height
petrophys.interpretation.saturation.sonic petrophys.evaluation.bvw petrophys.evaluation.mhi
petrophys.evaluation.net_pay petrophys.evaluation.rock_quality petrophys.evaluation.flow_units
petrophys.evaluation.intervals petrophys.evaluation.summaries petrophys.evaluation.volumetrics
petrophys.evaluation.sensitivity petrophys.evaluation.diagnostics petrophys.workflow
petrophys.workflow.custom petrophys.workflow.steps petrophys.workflow.schema
petrophys.reporting petrophys.reporting.context petrophys.visualization petrophys.mcp
petrophys.mcp.server petrophys.workflow.standard_workflow
""".split()

#: Generated pages, relative to the docs folder.
OUTPUT = Path("api") / "reference"

# -- Plan -----------------------------------------------------------------------------

FUNCTION, CLASS, CONSTANT = "function", "class", "constant"
_KIND_WORDS = {FUNCTION: ("function", "functions"), CLASS: ("class", "classes"),
               CONSTANT: ("constant", "constants")}


@dataclass
class Documented:
    """One object, documented once at ``canonical``."""

    canonical: str
    kind: str
    obj: Any
    exports: list[str] = field(default_factory=list)  # every public path to it
    docname: Optional[str] = None  # the page whose table (or entry) lists it

    @property
    def module(self) -> str:
        return self.canonical.rsplit(".", 1)[0]

    @property
    def name(self) -> str:
        return self.canonical.rsplit(".", 1)[1]

    @property
    def also(self) -> list[str]:
        """Public paths other than the canonical one, shortest first."""
        others = set(self.exports) - {self.canonical}
        return sorted(others, key=lambda path: (path.count("."), path))


@dataclass
class ReferencePlan:
    """Everything :func:`write_reference` writes, computed without Sphinx."""

    files: dict[str, str]  # path relative to the docs folder -> content
    documented: dict[str, Documented]  # canonical path -> object
    exports: dict[str, str]  # every public path in the area map -> canonical path
    stubs: list[str]  # canonical paths given their own page, in toctree order
    module_pages: dict[str, str]  # module -> docname of the page with its target
    redirect_pages: dict[str, str]  # module -> page its old URL redirects to
    aliases: dict[str, str]  # alias path -> canonical path
    page_aliases: dict[str, str]  # undocumented package -> docname, for :mod:
    class_excludes: dict[str, list[str]]  # class -> attributes its docstring lists
    class_inherited: dict[str, list[str]]  # class -> fields inherited from a base
    counts: dict[str, dict[str, int]]  # area slug -> kind -> count

    @property
    def autosummary_context(self) -> dict[str, Any]:
        """Data for the autosummary templates (plain data, so it pickles)."""
        return {
            "ph_also": {key: doc.also for key, doc in self.documented.items() if doc.also},
            "ph_exclude": self.class_excludes,
            "ph_inherited": self.class_inherited,
            # Classes whose bases are worth a "Bases:" line (not just object).
            "ph_bases": sorted(
                key
                for key, doc in self.documented.items()
                if doc.kind == CLASS and doc.obj.__bases__ != (object,)
            ),
        }

    def resolve_reference(self, app, env, node, contnode):
        """``missing-reference`` handler that sends aliases to the canonical object."""
        if node.get("refdomain") != "py":
            return None
        target = node.get("reftarget", "")
        if target.startswith("pp."):
            target = "petrophys." + target[3:]
        fromdoc = node.get("refdoc", env.docname)
        canonical = self.canonical_target(target, node.get("reftype"))
        if canonical is not None:
            domain = env.get_domain("py")
            for role in dict.fromkeys((node.get("reftype"), "obj")):
                newnode = domain.resolve_xref(
                    env, fromdoc, app.builder, role, canonical, node, contnode
                )
                if newnode is not None:
                    return newnode
        if node.get("reftype") not in ("mod", "obj"):
            return None
        from sphinx.util.nodes import make_refnode

        docname = self.page_aliases.get(target)
        if docname is not None:
            return make_refnode(app.builder, fromdoc, docname, "", contnode, target)
        # A private submodule links to the section of its public package.
        parts = target.split(".")
        for cut in range(len(parts) - 1, 1, -1):
            package = ".".join(parts[:cut])
            if package in self.module_pages:
                docname = self.module_pages[package]
                return make_refnode(
                    app.builder, fromdoc, docname, f"module-{package}", contnode, target
                )
        return None

    def add_sidebar_titles(self, app, pagename, templatename, context, doctree) -> None:
        """``html-page-context`` handler: a tooltip with the full path on each
        function and class page in furo's sidebar, whose names are truncated
        to one line by the stylesheet."""
        tree = context.get("furo_navigation_tree")
        if not tree:
            return
        stubs = self._stub_set

        def titled(match):
            name = match.group(2)
            if name not in stubs:
                return match.group(0)
            return f'{match.group(1)} title="{name}">'

        tree = _SIDEBAR_LINK.sub(titled, tree)
        # The current page's own entry links to "#".
        name = pagename.rsplit("/", 1)[-1]
        if name in stubs:
            tree = tree.replace(
                '<a class="current reference internal" href="#">',
                f'<a class="current reference internal" href="#" title="{name}">',
                1,
            )
        context["furo_navigation_tree"] = tree

    @property
    def _stub_set(self) -> set:
        if not hasattr(self, "_stubs_cached"):
            self._stubs_cached = set(self.stubs)
        return self._stubs_cached

    def canonical_target(self, target: str, role: Optional[str] = None) -> Optional[str]:
        """The canonical path for an alias, or for a member of an aliased class.

        A bare class name such as ``XRFData`` (or a member, ``Workflow.run``)
        resolves when exactly one documented class has that name; a bare
        function name does too, but only from ``:func:`` and ``:meth:``.
        """
        if target in self.aliases:
            return self.aliases[target]
        parts = target.split(".")
        for cut in range(len(parts) - 1, 1, -1):
            prefix = ".".join(parts[:cut])
            if prefix in self.aliases:
                return ".".join([self.aliases[prefix], *parts[cut:]])
        if target.startswith("petrophys.") or role in ("attr", "data", "mod"):
            return None
        canonical = self.short_names.get(parts[0])
        if canonical is None:
            return None
        kind = self.documented[canonical].kind
        if kind == CLASS or (kind == FUNCTION and len(parts) == 1 and role in ("func", "meth")):
            return ".".join([canonical, *parts[1:]])
        return None

    @property
    def short_names(self) -> dict[str, str]:
        """Short name -> canonical path, for names only one object has."""
        if not hasattr(self, "_short_names"):
            seen: dict[str, list[str]] = {}
            for key, doc in self.documented.items():
                seen.setdefault(doc.name, []).append(key)
            self._short_names = {name: keys[0] for name, keys in seen.items() if len(keys) == 1}
        return self._short_names


_SIDEBAR_LINK = re.compile(
    r'(<a class="reference internal" href="(?:[^"]*/)?(petrophys\.[^"/#]+)\.html")>'
)


def _import(module: str):
    return importlib.import_module(module)


def _split(names: Optional[str]) -> list[str]:
    return names.split() if names else []


def _kind(obj: Any) -> Optional[str]:
    if inspect.ismodule(obj):
        return None
    if inspect.isclass(obj):
        return CLASS
    if callable(obj):
        return FUNCTION
    return CONSTANT


def public_names(module: str) -> list[str]:
    """``__all__``, or the public functions and classes defined in the module."""
    mod = _import(module)
    names = getattr(mod, "__all__", None)
    if names is not None:
        return list(names)
    return [
        name
        for name, obj in vars(mod).items()
        if not name.startswith("_")
        and (inspect.isfunction(obj) or inspect.isclass(obj))
        and obj.__module__ == module
    ]


def iter_sections(items: Iterable[Any], page: Optional[Page] = None
                  ) -> Iterator[tuple[Section, Optional[Page]]]:
    """Every section under ``items``, with the page it is on (None: the area page)."""
    for item in items:
        if isinstance(item, Section):
            yield item, page
        elif isinstance(item, Page):
            yield from iter_sections(item.sections, item)
        elif isinstance(item, Topic):
            yield from iter_sections(item.items, page)


def all_sections() -> list[tuple[Area, Section, Optional[Page]]]:
    return [(area, s, p) for area in AREAS for s, p in iter_sections(area.items)]


def _identity(name: str, obj: Any, kind: str) -> Any:
    # Constants such as 0.433 may share an object by accident; require the name too.
    return (name, id(obj)) if kind == CONSTANT else id(obj)


def _parent_packages(modules: Iterable[str]) -> list[str]:
    modules = set(modules)
    parents = {".".join(m.split(".")[:cut]) for m in modules for cut in range(1, m.count(".") + 1)}
    return sorted(parents - modules)


def _napoleon_attributes(cls: type) -> list[str]:
    """Attributes the class docstring documents itself, in a NumPy Attributes section.

    The class template excludes them from autodoc's member list, so each is
    documented once. A line naming several attributes (``a, b : float``)
    gives napoleon no target, so autodoc keeps documenting those.
    """
    doc = inspect.getdoc(cls) or ""
    if not re.search(r"^Attributes\n-{3,}$", doc, re.M):
        return []
    from sphinx.ext.napoleon import Config
    from sphinx.ext.napoleon.docstring import NumpyDocstring

    text = str(NumpyDocstring(doc, Config(napoleon_use_ivar=False), what="class", obj=cls))
    return re.findall(r"^\s*\.\. attribute:: ([A-Za-z_]\w*)\s*$", text, re.M)


def _inherited_fields(cls: type) -> list[str]:
    """Annotated attributes (dataclass fields) the class inherits from a base.

    Autosummary cannot tell these from the class's own fields, because a field
    without a default is not an attribute of the class.
    """
    own = set(cls.__dict__.get("__annotations__", {})) | set(cls.__dict__)
    inherited = set()
    for base in cls.__mro__[1:]:
        inherited |= set(base.__dict__.get("__annotations__", {}))
    return sorted(name for name in inherited - own if not name.startswith("_"))


def _constant_aliases(documented: Iterable[Documented]) -> dict[str, str]:
    """The paths of each constant in the submodules of its canonical module.

    A constant has no ``__module__`` to name where it is defined, but
    docstrings link to it there (``petrophys.model.class_tables.CLASS_TABLES``
    for ``petrophys.model.CLASS_TABLES``). Values such as numbers and strings
    are left out: two modules can hold the same small value by accident.
    """
    aliases: dict[str, str] = {}
    for doc in documented:
        if doc.kind != CONSTANT or isinstance(doc.obj, (bool, int, float, complex, str, bytes)):
            continue
        prefix = doc.module + "."
        for module in sorted(name for name in list(sys.modules) if name.startswith(prefix)):
            if getattr(sys.modules.get(module), doc.name, None) is doc.obj:
                aliases.setdefault(f"{module}.{doc.name}", doc.canonical)
    return aliases


def _section_names(section: Section, leftovers: dict[str, list[str]]) -> list[str]:
    """The names a section lists: its ``names``, or every public name of its module."""
    exported = public_names(section.module)
    if section.names is None:
        return exported
    names = _split(section.names)
    missing = [name for name in names if name not in exported]
    if missing:
        raise ValueError(f"API area map: {section.module} does not export {missing}")
    return names + leftovers.pop(id(section), [])


def build_plan() -> ReferencePlan:
    """Import the area-map modules and compute every page, alias and count."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    sections = all_sections()
    modules = list(dict.fromkeys(section.module for _, section, _ in sections))

    # A module split across sections: names none of them lists go to the last one.
    leftovers: dict[int, list[str]] = {}
    for module in modules:
        split = [s for _, s, _ in sections if s.module == module and s.names is not None]
        if split:
            listed = {name for s in split for name in _split(s.names)}
            rest = [n for n in public_names(module) if n not in listed]
            if any(s.names is None for _, s, _ in sections if s.module == module):
                raise ValueError(f"API area map: {module} mixes full and partial sections")
            leftovers[id(split[-1])] = rest

    # Every public path, grouped by the object it names.
    wrapped = {}
    for alias, target in SAME_AS.items():
        module, name = target.rsplit(".", 1)
        wrapped[alias] = getattr(_import(module), name)
    groups: dict[Any, list[tuple[str, str, Any, str]]] = {}
    for module in modules:
        mod = _import(module)
        for name in public_names(module):
            obj = getattr(mod, name)
            kind = _kind(obj)
            if kind is None:
                continue
            path = f"{module}.{name}"
            key = _identity(name, wrapped.get(path, obj), kind)
            groups.setdefault(key, []).append((path, module, obj, kind))

    documented: dict[str, Documented] = {}
    exports: dict[str, str] = {}
    for paths in groups.values():
        candidates = [p for p in paths if p[0] not in SAME_AS]
        defining = getattr(candidates[0][2], "__module__", None) or ""
        best = None
        for path, module, obj, kind in candidates:
            if defining == module or defining.startswith(module + "."):
                if best is None or len(module) > len(best[1]):
                    best = (path, module, obj, kind)
        path, module, obj, kind = best or candidates[0]
        documented[path] = Documented(path, kind, obj, [p[0] for p in paths])
        for other in paths:
            exports[other[0]] = path

    # Public paths outside the area map: the top level and parent packages.
    aliases = {path: canonical for path, canonical in exports.items() if path != canonical}
    for package in _parent_packages(modules):
        mod = _import(package)
        for name in getattr(mod, "__all__", []):
            obj = getattr(mod, name, None)
            kind = _kind(obj)
            key = _identity(name, obj, kind) if kind else None
            if key in groups:
                canonical = exports[groups[key][0][0]]
                documented[canonical].exports.append(f"{package}.{name}")
                aliases[f"{package}.{name}"] = canonical
    compat = getattr(_import("petrophys.io"), "_COMPAT_EXPORTS", {})
    for name, (module, attr) in compat.items():
        if f"{module}.{attr}" in exports:
            aliases[f"petrophys.io.{name}"] = exports[f"{module}.{attr}"]
    for doc in documented.values():
        if doc.kind != CONSTANT:
            real = f"{doc.obj.__module__}.{doc.obj.__qualname__}"
            if real not in documented:
                aliases.setdefault(real, doc.canonical)
    for path, canonical in _constant_aliases(documented.values()).items():
        if path not in documented:
            aliases.setdefault(path, canonical)

    writer = _Writer(documented, exports, leftovers)
    files = writer.write_all()
    unlisted = sorted(key for key, doc in documented.items() if doc.docname is None)
    if unlisted:
        raise ValueError(f"API area map: no section lists {unlisted}")

    page_aliases = {"petrophys": "api/index"}
    for package in _parent_packages(modules):
        docname = writer.package_pages.get(package)
        if docname:
            page_aliases[package] = docname
    class_excludes, class_inherited = {}, {}
    for key, doc in documented.items():
        if doc.kind == CLASS:
            if names := _napoleon_attributes(doc.obj):
                class_excludes[key] = names
            if names := _inherited_fields(doc.obj):
                class_inherited[key] = names
    return ReferencePlan(
        files=files,
        documented=documented,
        exports=exports,
        stubs=writer.stubs,
        module_pages=writer.module_pages,
        redirect_pages=writer.redirect_pages,
        aliases=aliases,
        page_aliases=page_aliases,
        class_excludes=class_excludes,
        class_inherited=class_inherited,
        counts=writer.counts,
    )


# -- Writing rst ----------------------------------------------------------------------

_UNDERLINE = "=-~^"


def _heading(text: str, level: int) -> list[str]:
    return [text, _UNDERLINE[level - 1] * len(text), ""]


def _first_line(module: str) -> str:
    doc = (inspect.getdoc(_import(module)) or "").strip()
    return doc.splitlines()[0].strip() if doc else ""


def count_text(counts: dict[str, int]) -> str:
    """``"31 functions · 12 classes"``."""
    parts = []
    for kind, (one, many) in _KIND_WORDS.items():
        if counts.get(kind):
            parts.append(f"{counts[kind]} {one if counts[kind] == 1 else many}")
    return " · ".join(parts)


def _constant_value(obj: Any) -> Optional[str]:
    if isinstance(obj, bool) or not isinstance(obj, (int, float, str, tuple)):
        return None
    text = repr(obj)
    return text if len(text) <= 60 else None


def _constant_doc(doc: Documented) -> str:
    """The ``#:`` comment of a constant, from its module or the submodule defining it."""
    from sphinx.pycode import ModuleAnalyzer

    candidates = [doc.module] + sorted(
        m for m in list(sys.modules) if m.startswith(doc.module + ".")
    )
    for candidate in candidates:
        if getattr(sys.modules.get(candidate), doc.name, None) is not doc.obj:
            continue
        try:
            comments = ModuleAnalyzer.for_module(candidate).find_attr_docs()
        except Exception:  # noqa: BLE001 - modules without source
            continue
        text = " ".join(line.strip() for line in comments.get(("", doc.name), [])).strip()
        if text:
            return text
    return CONSTANT_DOCS.get(doc.canonical, "")


class _Writer:
    def __init__(self, documented, exports, leftovers):
        self.documented: dict[str, Documented] = documented
        self.exports: dict[str, str] = exports
        self.leftovers = leftovers
        self.stubs: list[str] = []
        self.module_pages: dict[str, str] = {}
        self.package_pages: dict[str, str] = {}
        self.counts: dict[str, dict[str, int]] = {}
        self.declared: set[str] = set()  # modules with a module target
        self.redirect_pages: dict[str, str] = {}  # module -> page for old URLs
        self.pages: list[Page] = []

    def own(self, section: Section) -> list[Documented]:
        """Objects this section documents (canonical here), in export order."""
        names = _section_names(section, self.leftovers.copy())
        paths = [f"{section.module}.{name}" for name in names]
        return [self.documented[p] for p in paths if self.exports.get(p) == p]

    def reexports(self, section: Section) -> list[Documented]:
        """Functions and classes the module exports but documents elsewhere."""
        result = []
        for name in public_names(section.module):
            path = f"{section.module}.{name}"
            canonical = self.exports.get(path)
            if canonical and canonical != path and self.documented[canonical].kind != CONSTANT:
                result.append(self.documented[canonical])
        legacy = {s.module for area, s, _ in all_sections() if area.legacy}
        return sorted(result, key=lambda doc: doc.module in legacy)

    def groups(self, section: Section, own: list[Documented]):
        by_name = {doc.name: doc for doc in own}
        groups, used = [], set()
        for title, names in section.groups:
            docs = []
            for name in _split(names):
                doc = by_name.get(name)
                if doc is None or doc.kind == CONSTANT:
                    raise ValueError(
                        f"API area map: group {title!r} lists {section.module}.{name}, "
                        "which is not a function or class documented in that section"
                    )
                docs.append(doc)
                used.add(name)
            groups.append((title, docs))
        rest = [doc for doc in own if doc.name not in used]
        for kind, title in ((FUNCTION, "Functions"), (CLASS, "Classes")):
            docs = [doc for doc in rest if doc.kind == kind]
            if docs:
                groups.append((f"More {title.lower()}" if section.groups else title, docs))
        return groups, [doc for doc in rest if doc.kind == CONSTANT]

    @staticmethod
    def counts_of(docs: Iterable[Documented]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for doc in docs:
            counts[doc.kind] = counts.get(doc.kind, 0) + 1
        return counts

    # -- Blocks -----------------------------------------------------------------

    def module_block(self, section: Section, description: Optional[str]) -> list[str]:
        # Plain ``module``/``currentmodule`` (the default domain is Python):
        # autosummary's stub scanner only follows these spellings.
        directive = "currentmodule" if section.module in self.declared else "module"
        self.declared.add(section.module)
        lines = [f".. {directive}:: {section.module}", ""]
        lines += [".. container:: ph-api-module", "", f"   ``{section.module}``", ""]
        text = description or section.description or _first_line(section.module)
        if text:
            lines += [f"   {text}", ""]
        if section.note:
            lines += [".. rst-class:: ph-api-note", "", section.note, ""]
        return lines

    def table(self, docs: Sequence[Documented], docname: Optional[str]) -> list[str]:
        """An autosummary table; with ``docname`` it also gives each object its page.

        Section tables follow the section's ``module``, so they name objects
        relative to it; other tables reset the current module and use full paths.
        """
        lines = [] if docname else [".. currentmodule:: None", ""]
        lines.append(".. autosummary::")
        if docname:
            lines.append("   :toctree: .")
        lines += ["   :nosignatures:", ""]
        lines += [f"   {doc.name}" if docname else f"   ~{doc.canonical}" for doc in docs]
        if docname:
            for doc in docs:
                doc.docname = docname
                self.stubs.append(doc.canonical)
        return lines + [""]

    def constants(self, docs: Sequence[Documented], docname: str) -> list[str]:
        lines = []
        for doc in docs:
            kind = type(doc.obj)
            kind_name = kind.__qualname__
            if kind.__module__ != "builtins":
                kind_name = f"{kind.__module__}.{kind_name}"
            lines += [f".. py:data:: {doc.name}", f"   :module: {doc.module}",
                      f"   :type: {kind_name}"]
            value = _constant_value(doc.obj)
            if value is not None:
                lines.append(f"   :value: {value}")
            lines.append("")
            text = _constant_doc(doc)
            if text:
                lines += [f"   {text}", ""]
            doc.docname = docname
        return lines

    def section(self, section: Section, level: int, docname: str, merged: bool,
                description: Optional[str] = None) -> list[str]:
        """Heading, module line and tables of one section.

        A ``merged`` section is the only one on its page: the page title is its
        heading and its groups become headings, so they show in the page contents.
        """
        self.module_pages.setdefault(section.module, docname)
        lines = [] if merged else _heading(section.title or section.module, level)
        lines += self.module_block(section, description)
        groups, constants = self.groups(section, self.own(section))
        if constants:
            groups.append(("Constants", constants))
        for title, docs in groups:
            if merged:
                lines += _heading(title, level)
            else:
                lines += [f".. rubric:: {title}", "   :class: ph-api-group", ""]
            if title == "Constants":
                lines += self.constants(docs, docname)
            else:
                lines += self.table(docs, docname)
        reexports = self.reexports(section)
        if reexports and (section.names is None or section is self.first_split(section)):
            links = ", ".join(f":py:obj:`~{doc.canonical}`" for doc in reexports)
            lines += [".. rst-class:: ph-api-also", "",
                      f"Also importable from ``{section.module}``: {links}.", ""]
        return lines

    def first_split(self, section: Section) -> Section:
        return next(s for _, s, _ in all_sections() if s.module == section.module)

    def page_summary(self, page: Page, level: int, area_docs: list) -> list[str]:
        """How a subpackage page appears on its area page: entry points and a link."""
        lines = _heading(page.title, level)
        description = page.description
        if len(page.sections) == 1:
            description = description or page.sections[0].description or _first_line(
                page.sections[0].module)
        modules = list(dict.fromkeys(s.module for s in page.sections))
        if all(m == page.slug or m.startswith(page.slug + ".") for m in modules):
            modules = [page.slug]
        modules_text = " · ".join(f"``{m}``" for m in modules)
        lines += [".. container:: ph-api-module", "", f"   {modules_text}", ""]
        if description:
            lines += [f"   {description}", ""]
        docs = [doc for section in page.sections for doc in self.own(section)]
        area_docs += docs
        if page.key:
            keys = [self.documented[self.exports.get(key, key)] for key in _split(page.key)]
        else:
            # The page's first objects in display order; all of them if few.
            ordered = [
                doc
                for section in page.sections
                for _, group in self.groups(section, self.own(section))[0]
                for doc in group
            ]
            keys = ordered if len(ordered) <= 5 else ordered[:4]
        lines += self.table(keys, None)
        lines += [".. rst-class:: ph-api-more", "",
                  f":doc:`{page.title} reference <{page.slug}>` · "
                  f"{count_text(self.counts_of(docs))}", "",
                  ".. toctree::", "   :hidden:", "", f"   {page.slug}", ""]
        self.pages.append(page)
        return lines

    def items(self, items, level: int, docname: str, area_docs: list) -> list[str]:
        lines = []
        for item in items:
            if isinstance(item, Section):
                area_docs += self.own(item)
                lines += self.section(item, level, docname, merged=False)
            elif isinstance(item, Page):
                lines += self.page_summary(item, level, area_docs)
            else:
                lines += _heading(item.title, level)
                if item.description:
                    lines += [".. rst-class:: ph-api-lead", "", item.description, ""]
                lines += self.items(item.items, level + 1, docname, area_docs)
        return lines

    # -- Pages ------------------------------------------------------------------

    def area(self, area: Area) -> str:
        docname = (OUTPUT / area.slug).as_posix()
        lead = "ph-api-deprecated" if area.legacy else "ph-api-lead"
        lines = [f".. _api-{area.slug}:", ""] + _heading(area.title, 1)
        lines += [f".. rst-class:: {lead}", "", area.intro, ""]
        area_docs: list[Documented] = []
        if area.flat:
            (section,) = area.items
            area_docs += self.own(section)
            lines += self.section(section, 2, docname, merged=True)
        else:
            lines += self.items(area.items, 2, docname, area_docs)
        if area.legacy:
            lines += self.compat_paths()
        self.counts[area.slug] = self.counts_of(area_docs)
        pages_by_module: dict[str, set[str]] = {}
        for section, page in iter_sections(area.items):
            target = (OUTPUT / page.slug).as_posix() if page else docname
            pages_by_module.setdefault(section.module, set()).add(target)
        for module, targets in pages_by_module.items():
            self.redirect_pages[module] = targets.pop() if len(targets) == 1 else docname
        for section, page in iter_sections(area.items):
            if page is not None:
                self.package_pages[page.slug] = (OUTPUT / page.slug).as_posix()
        for section, page in iter_sections(area.items):
            parts = section.module.split(".")
            for cut in range(2, len(parts)):
                self.package_pages.setdefault(".".join(parts[:cut]), docname)
        return "\n".join(lines).rstrip() + "\n"

    def page(self, page: Page) -> str:
        docname = (OUTPUT / page.slug).as_posix()
        lines = [f".. _api-{page.slug}:", ""] + _heading(page.title, 1)
        if len(page.sections) == 1:
            lines += self.section(page.sections[0], 2, docname, merged=True,
                                  description=page.description)
        else:
            if page.description:
                lines += [".. rst-class:: ph-api-lead", "", page.description, ""]
            for section in page.sections:
                lines += self.section(section, 2, docname, merged=False)
        return "\n".join(lines).rstrip() + "\n"

    def compat_paths(self) -> list[str]:
        compat = getattr(_import("petrophys.io"), "_COMPAT_EXPORTS", {})
        if not compat:
            return []
        lines = _heading("Deprecated import paths", 2)
        lines += ["These names import from ``petrophys.io`` with a "
                  "``DeprecationWarning``. Import them from the module listed.", "",
                  ".. list-table::", "   :header-rows: 1", "   :class: ph-api-compat", "",
                  "   * - Deprecated", "     - Use instead"]
        for name, (module, attr) in sorted(compat.items()):
            target = self.exports.get(f"{module}.{attr}", f"{module}.{attr}")
            lines += [f"   * - ``petrophys.io.{name}``", f"     - :py:obj:`{target}`"]
        return lines + [""]

    def cards(self) -> str:
        lines = [".. grid:: 1 1 2 2", "   :gutter: 3", "   :class-container: ph-api-cards", ""]
        for area in AREAS:
            if area.card is None:
                continue
            classes = "ph-api-card ph-api-card-legacy" if area.legacy else "ph-api-card"
            keys = " ".join(f":py:obj:`~{key}`" for key in _split(area.key))
            lines += [
                f"   .. grid-item-card:: :doc:`{area.title} <reference/{area.slug}>`",
                f"      :class-card: {classes}",
                "",
                f"      {area.card}",
                "",
                "      .. rst-class:: ph-api-card-keys",
                "",
                f"      {keys}",
                "",
                "      +++",
                f"      {count_text(self.counts[area.slug])}",
                "",
            ]
        lines += [".. toctree::", "   :hidden:", ""]
        lines += [f"   reference/{area.slug}" for area in AREAS]
        return "\n".join(lines) + "\n"

    def write_all(self) -> dict[str, str]:
        texts = {area.slug: self.area(area) for area in AREAS}
        # Subpackage pages come after every area page, in sidebar order.
        texts.update({page.slug: self.page(page) for page in self.pages})
        files = {(OUTPUT / f"{slug}.rst").as_posix(): text for slug, text in texts.items()}
        files[(OUTPUT / "_cards.inc").as_posix()] = self.cards()
        return files


def clean_docstring(app, what, name, obj, options, lines) -> None:
    """``autodoc-process-docstring`` handler: drop docstrings a value inherits.

    An undocumented attribute or constant otherwise shows the docstring of its
    type, such as ``str(object='') -> str`` for a string default or the class
    docstring for every member of an enum.
    """
    if what not in ("attribute", "data") or not lines or inspect.isroutine(obj):
        return
    if isinstance(obj, property):
        return
    type_doc = inspect.getdoc(type(obj))
    if type_doc and "\n".join(lines).strip() == type_doc.strip():
        lines.clear()


def write_reference(docs_dir: Union[str, Path]) -> ReferencePlan:
    """Write the generated pages under ``docs_dir`` and return the plan.

    Files are rewritten only when their content changes, and pages left over
    from an earlier layout are removed, so incremental builds stay incremental.
    """
    docs_dir = Path(docs_dir)
    plan = build_plan()
    out = docs_dir / OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    keep = {Path(path).name for path in plan.files} | {f"{s}.rst" for s in plan.stubs}
    for path in out.iterdir():
        if path.is_file() and path.name not in keep:
            path.unlink()
    for relative, content in plan.files.items():
        target = docs_dir / relative
        if not target.exists() or target.read_text(encoding="utf-8") != content:
            target.write_text(content, encoding="utf-8")
    return plan


def redirect_rules(plan: ReferencePlan) -> list[str]:
    """Netlify rules from each old one-page-per-module URL to its new page."""
    rules = []
    for module in OLD_MODULE_PAGES:
        target = plan.redirect_pages[module]
        rules.append(f"/api/generated/{module}.html  /{target}.html  301")
        rules.append(f"/api/generated/{module}  /{target}  301")
    return rules
