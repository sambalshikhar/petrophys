# Examples

The notebooks in this directory form a progressive tutorial curriculum. They
use the installable sample wells in `src/petrophys/datasets/`, available through
`petrophys.sample()` and `petrophys.list_samples()`.

Start with [`quickstart.ipynb`](quickstart.ipynb), then work through the numbered
notebooks in order.

## Foundations

| # | Notebook | Topic |
|---|----------|-------|
| 00 | [`00_orientation_and_setup.ipynb`](part_a_foundations/00_orientation_and_setup.ipynb) | Installation, sample data, and curriculum orientation |
| 01 | [`01_loading_and_inspecting_data.ipynb`](part_a_foundations/01_loading_and_inspecting_data.ipynb) | Wells, curves, headers, and units |
| 02 | [`02_qaqc_and_standardization.ipynb`](part_a_foundations/02_qaqc_and_standardization.ipynb) | Mnemonics, units, bad-hole checks, and value-range QC |
| 03 | [`03_clay_volume.ipynb`](part_a_foundations/03_clay_volume.ipynb) | Clay volume and clay resistivity |
| 04 | [`04_porosity.ipynb`](part_a_foundations/04_porosity.ipynb) | Density, neutron, sonic, total, and effective porosity |
| 05 | [`05_lithology_and_mineral_solver.ipynb`](part_a_foundations/05_lithology_and_mineral_solver.ipynb) | Lithology crossplots and mineral solving |
| 06 | [`06_formation_water_and_environment.ipynb`](part_a_foundations/06_formation_water_and_environment.ipynb) | Formation water, mud filtrate, temperature, and invasion |
| 07 | [`07_water_saturation.ipynb`](part_a_foundations/07_water_saturation.ipynb) | Archie and shaly-sand saturation models |
| 08 | [`08_permeability.ipynb`](part_a_foundations/08_permeability.ipynb) | Empirical and core-calibrated permeability |
| 09 | [`09_reservoir_quality_and_net_pay.ipynb`](part_a_foundations/09_reservoir_quality_and_net_pay.ipynb) | Flow units, reservoir quality, and net pay |

## Advanced Workflows

| # | Notebook | Topic |
|---|----------|-------|
| 10 | [`10_dynamic_workflow_glass_box.ipynb`](part_b_advanced/10_dynamic_workflow_glass_box.ipynb) | Input-aware workflow routing and execution traces |
| 11 | [`11_formation_and_interval_analysis.ipynb`](part_b_advanced/11_formation_and_interval_analysis.ipynb) | Formation tops, zones, and interval summaries |
| 12 | [`12_core_calibration_and_digital_core.ipynb`](part_b_advanced/12_core_calibration_and_digital_core.ipynb) | Core calibration, digital facies, XRF, and geomechanics |
| 13 | [`13_ml_curve_imputation.ipynb`](part_b_advanced/13_ml_curve_imputation.ipynb) | Pretrained curve imputation |
| 14 | [`14_nmr_petrophysics.ipynb`](part_b_advanced/14_nmr_petrophysics.ipynb) | NMR porosity, T2 distributions, and permeability |
| 15 | [`15_capillary_pressure_saturation_height.ipynb`](part_b_advanced/15_capillary_pressure_saturation_height.ipynb) | Capillary pressure and saturation-height models |
| 16 | [`16_borehole_image_fmi.ipynb`](part_b_advanced/16_borehole_image_fmi.ipynb) | Borehole-image loading and visualization |
| 17 | [`17_visualization_masterclass.ipynb`](part_b_advanced/17_visualization_masterclass.ipynb) | Log panels, crossplots, and batch visualization |
| 18 | [`18_multiwell_field_study.ipynb`](part_b_advanced/18_multiwell_field_study.ipynb) | Multiwell analysis and field screening |
| 19 | [`19_llm_context_generation.ipynb`](part_b_advanced/19_llm_context_generation.ipynb) | Portable LAS-to-JSON context generation |

## Case Studies

| Notebook | Scope |
|----------|-------|
| [`cs1_full_formation_evaluation_30_6-5.ipynb`](case_studies/cs1_full_formation_evaluation_30_6-5.ipynb) | Full clastic formation evaluation |
| [`cs2_carbonate_rock_typing_costa.ipynb`](case_studies/cs2_carbonate_rock_typing_costa.ipynb) | Carbonate characterization and rock typing |
| [`cs3_field_screening_costa_17wells.ipynb`](case_studies/cs3_field_screening_costa_17wells.ipynb) | Field-scale screening and well ranking |
| [`cs4_imperfect_well_recovery_16_7-1.ipynb`](case_studies/cs4_imperfect_well_recovery_16_7-1.ipynb) | Interpretation of an incomplete log suite |
| [`cs5_saturation_model_sensitivity_30_6-5.ipynb`](case_studies/cs5_saturation_model_sensitivity_30_6-5.ipynb) | Saturation-model selection and sensitivity |

The ML examples require the optional `scikit-learn`, `xgboost`, and `joblib`
dependencies.
