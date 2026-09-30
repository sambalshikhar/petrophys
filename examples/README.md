# Examples

The notebooks in this directory form a progressive tutorial curriculum. They
use the installable sample wells in `src/petrophys/datasets/`, available through
`petrophys.sample()` and `petrophys.list_samples()`.

Start with [`quickstart.ipynb`](quickstart.ipynb), then work through the numbered
notebooks in order.

For your own LAS and optional formations/lithology, continue directly to
[`own_data.ipynb`](own_data.ipynb). It covers templates, CSV/TSV/Excel,
column and sheet mapping, well selection, depth metadata, source selection,
core saturation units, and actionable import errors.

The quickstart now plots before interpreting. Lessons 00, 01, 02, 05, 07,
10, 11, 17 and CS1 connect the short API to the detailed scientific workflows.
Geology files remain optional; calculated results require appropriate inputs
and assumptions. Advanced equation-level APIs remain part of the curriculum.

## Run and Verify

From the repository root, install the package with the extras the notebooks
use, in the environment your notebook editor runs:

```bash
python -m pip install -e ".[tutorials,ml,mcp,dlis]"
```

`tutorials` provides the kernel and execution tools (not a notebook editor).
The other extras are only needed by some notebooks:

| Extra | Needed by | Notes |
|-------|-----------|-------|
| `ml` | 12, 13, CS4 | Pretrained imputation models; Python 3.11 or later |
| `dlis` | 16 | DLIS reader; lesson 16 also needs the Git LFS image data |
| `mcp` | 20 | MCP server; Python 3.10 or later |

Every other notebook runs with `python -m pip install -e ".[tutorials]"`.
Open a notebook in VS Code/Jupyter, select that environment, and restart/run
all cells.

From the repository root, verify the thirteen primary user-journey notebooks:

```bash
python scripts/execute_tutorials.py --output-dir tutorial-results --report tutorial-results/report.json
```

Each notebook runs in a fresh kernel with a temporary working directory.
Use `--inplace` instead of `--output-dir` to refresh saved source outputs
only after successful execution. Failed runs return a nonzero exit status.
Other specialist notebooks retain their own optional data/dependency requirements.

Tutorial 19 uses `build_context`/`export_context`, schema versions and byte
budgets, and reads the ranked formation table directly. Tutorial 14 separates
vendor permeability from T2 calculations; tutorial 16 needs real Git LFS data.

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
| 20 | [`20_mcp_server.ipynb`](part_b_advanced/20_mcp_server.ipynb) | Petrophys for LLM agents through the MCP server |

## Case Studies

| Notebook | Scope |
|----------|-------|
| [`cs1_full_formation_evaluation_30_6-5.ipynb`](case_studies/cs1_full_formation_evaluation_30_6-5.ipynb) | Full clastic formation evaluation, per-formation summary and completion ranking |
| [`cs2_carbonate_rock_typing_costa.ipynb`](case_studies/cs2_carbonate_rock_typing_costa.ipynb) | Carbonate core calibration and rock typing over the Kharaib-2 interval |
| [`cs3_field_screening_costa_17wells.ipynb`](case_studies/cs3_field_screening_costa_17wells.ipynb) | Kharaib-2 screening and ranking across 17 COSTA wells |
| [`cs4_imperfect_well_recovery_16_7-1.ipynb`](case_studies/cs4_imperfect_well_recovery_16_7-1.ipynb) | Data readiness, QC flags and validated imputation on an incomplete log suite (`ml` extra) |
| [`cs5_saturation_model_sensitivity_30_6-5.ipynb`](case_studies/cs5_saturation_model_sensitivity_30_6-5.ipynb) | Saturation-model selection and parameter sensitivity |

Lessons 12, 13 and CS4 need the `ml` extra (scikit-learn, xgboost and joblib;
Python 3.11 or later); see [Run and Verify](#run-and-verify).
