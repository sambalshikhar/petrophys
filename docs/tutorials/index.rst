Tutorials
=========

The tutorial curriculum runs entirely on **real, publicly released well data**:
North Sea wells 30/6-5 and 16/7-1 (with the operator's parameter block in the
LAS headers), the 17-well COSTA carbonate benchmark, and the Daydream 1 CMR
log. Every notebook ships fully executed, so the figures and numbers below are
real outputs - and each notebook re-runs locally against the sample data
bundled with the package.

Part A - Foundations
--------------------

The single-well interpretation chain, one physical quantity at a time.

.. toctree::
   :maxdepth: 1

   quickstart
   part_a/00_orientation_and_setup
   part_a/01_loading_and_inspecting_data
   part_a/02_qaqc_and_standardization
   part_a/03_clay_volume
   part_a/04_porosity
   part_a/05_lithology_and_mineral_solver
   part_a/06_formation_water_and_environment
   part_a/07_water_saturation
   part_a/08_permeability
   part_a/09_reservoir_quality_and_net_pay

Part B - Advanced
-----------------

Workflow orchestration, formation analysis, core calibration, special logs
(NMR, borehole image), machine-learning imputation, and multi-well studies.

.. toctree::
   :maxdepth: 1

   part_b/10_dynamic_workflow_glass_box
   part_b/11_formation_and_interval_analysis
   part_b/12_core_calibration_and_digital_core
   part_b/13_ml_curve_imputation
   part_b/14_nmr_petrophysics
   part_b/15_capillary_pressure_saturation_height
   part_b/16_borehole_image_fmi
   part_b/17_visualization_masterclass
   part_b/18_multiwell_field_study
   part_b/19_llm_context_generation

Case Studies
------------

End-to-end briefs answered the way an asset team would ask them, each
validated against public ground truth (operator completion reports, NPD fact
pages).

.. toctree::
   :maxdepth: 1

   case_studies/cs1_full_formation_evaluation_30_6-5
   case_studies/cs2_carbonate_rock_typing_costa
   case_studies/cs3_field_screening_costa_17wells
   case_studies/cs4_imperfect_well_recovery_16_7-1
   case_studies/cs5_saturation_model_sensitivity_30_6-5
