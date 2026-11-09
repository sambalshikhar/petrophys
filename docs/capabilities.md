# What Petrophys can do

You have interpreted a first well and read your own files. The guide continues from there, from interpretation and review to LLM agents; this page maps each part of the library to the page that covers it.

## At a glance

| Area | What you get | Start with | Covered in |
| --- | --- | --- | --- |
| Prepare the data | LAS, DLIS and CSV reading; standard names and units; range QC; bad-hole flags; normalization; ML gap filling | `pp.read`, `pp.impute` | {doc}`own_data`, {doc}`data_model`, {ref}`review-inputs`, {ref}`analysis-gap-filling` |
| Interpret the rock | Temperature, clay volume, porosity, lithology and minerals, Rw, seven Sw models, permeability, rock physics | `pp.interpret` | {doc}`workflows` |
| Review the answer | Warnings, the step-by-step trace, the cutoffs used, net pay per formation | `result.print_trace()`, `result.formation_summary` | {ref}`review-interpretation` |
| Your own recipe | Built-in and custom steps, checked before they run, saved as JSON and run on many wells | `Workflow`, `@workflow_step` | {doc}`custom_workflows` |
| Plots and delivery | Log viewer templates, crossplots, multi-well charts; LAS, JSON, CSV and Excel exports | `pp.plot` | {doc}`visualization_reporting` |
| Core and specialist data | Core calibration, NMR, spectral gamma, saturation-height, borehole images, formation pressures | `core_data_from_well`, `NMRData` | {doc}`specialist_data` |
| Further analyses | Rock-quality and matrix indicators, shale distribution, heterogeneity, zonation, Monte Carlo, invasion, pore pressure | `zone_well`, `monte_carlo` | {doc}`analysis_tools` |
| Many wells | Batch and zoned runs over a folder of wells | `process_las_directory_with_workflow`, `BatchProcessor` | {ref}`workflows-batch`, {doc}`custom_workflows` |
| LLMs | Bounded JSON context files of interpreted wells; an MCP server that lets a model run Petrophys | `pp.export_context`, `petrophys-mcp` | {doc}`context`, {doc}`mcp` |

The {doc}`complete tour <tutorials/complete_tour>` runs every part of the library on real wells in one notebook.

## What Petrophys does not do

- No automatic depth matching between runs, or tool-depth correction.
- DLIS: processed channels, not every raw acquisition product.
- Borehole images: preparation and display, not dip or fracture picking.
- NMR: delivered distributions or curves, not raw echo trains.
- Empirical models still need local calibration and expert review.

## Next

{doc}`workflows`: interpret a well with `pp.interpret`, read its trace and review the answer.
