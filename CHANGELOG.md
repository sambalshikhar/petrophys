# Changelog

All notable changes to Petrophys are documented here.

## 3.0.0 (unreleased)

### Breaking changes

- Context generation now uses one canonical calculation path. Dataset names,
  vendor curves, and earlier context packages cannot select another workflow
  or provide archived numerical values.

- `DynamicWorkflowConfig` validates every setting when it is created and
  raises `ValueError` listing all invalid fields with the accepted values
  (previously, typos such as `hc_type="gass"` silently used defaults).
- `DynamicWorkflowConfig` uses `lithology_domain` to distinguish the
  clastic and carbonate mineral routes from per-depth lithology labels.
  `matrix_type` and the domain default to `"auto"`; insufficient evidence
  resolves to sandstone/clastic with a recorded warning.
- Tabular readers (`io.read_csv`, `read_csv_wells`, `read_ascii` without a
  units row, `Well.from_dataframe`) require `depth_unit` instead of assuming
  feet. `Well.depth_unit` accepts only metre or foot labels.
- The log viewer no longer searches parent directories for
  `meta_files/Litho_Nomenclature_Kingdom.xlsx`; pass `lithology_mapping` or
  set `PETROPHYS_LITHOLOGY_MAPPING`.

### Added

- Raw MDT pretest pressure picking through `read_mdt_pretest` and
  `pick_mdt_pressure`, configurable buildup QC, review flags, and a library
  plot of selected windows. `formation_pressures_from_mdt` passes accepted
  picks to station interpretation with explicit TVD metadata. Tutorial 15
  verifies the picker against 35 native C0020A LAS traces and their report.
- Unlabeled station-pressure contact inference through `infer_contacts`,
  exposing competing fluid hypotheses and unresolved contacts. The existing
  labeled FWL fit remains available.
- Public `build_context` / `export_context` and `petrophys-context` CLI with
  schema version 1.0, UTF-8 byte budgets, staged package publication and
  overwrite protection. The CLI always computes its context from input data.
- Standardized well-parameter and core-measurement readers and templates.
  Package generation accepts these alongside optional formation, lithology,
  and lithology-code tables.
- A COSTA carbonate mineral route using the published GR-illite relation and
  bounded calcite-dolomite density/closure solve.
- Context-package tops accept CSV/TSV/Excel, direct NPD/SODIR headers, explicit
  bottoms, depth units and rank selection without private default workbooks.
- `read_lithology_codes` validates codebooks and prefers Kingdom `Attribute`
  rock labels over a conflicting `Lithology Attribute` grouping, with a warning.
- A context API guide, data-quality notes and refreshed tutorial 19. CI now
  executes context generation and the real-file FMI tutorial.

- An MCP server (`petrophys-mcp`, `pip install "petrophys[mcp]"`) that lets
  LLM clients load wells, interpret them, inspect traces, summarize zones,
  plot and export results, with file access limited to chosen folders; a
  guide and tutorial 20 cover setup and a full session.
- `petrophys.workflow.config_schema()` and `workflow_steps()` describe every
  setting and step as data, and the documentation site serves `llms.txt`.
- Short public API: `petrophys.read`, `petrophys.plot` and the
  copy-by-default `petrophys.interpret`, with full docstrings;
  `DynamicWorkflowConfig`, `DynamicWorkflowResult` and `run_dynamic_workflow`
  are exported at the top level.
- `petrophys.DataQualityWarning` for implausible inputs: curves mostly outside
  their plausible range (with a unit hint such as "values look 1000x too
  large"), geology that does not overlap the logged interval, and
  unrecognized lithology labels.
- `DynamicWorkflowResult.to_las()` writes the interpreted well with the
  settings used (`PP_*` parameters) and step reasons; traces record the
  package version.
- Every skipped step records a specific reason, and `print_trace()` shows it.
- `formations_costa.csv` labels the logged interval of each Costa well as
  its single formation, KHARAIB-2, so formation summaries and context
  packages name the reservoir instead of treating the file as a whole well.
- `DynamicWorkflowResult.formation_summary()` tabulates gross, net and
  average properties per formation.
- NPD/GeoLink-style tops without a rank column get their stratigraphic rank
  from the name suffix (GP./Fm./Mb.), so mixed group/formation picks can be
  analyzed one level at a time with `run_formation_analysis(..., rank=...)`.
- Formations attached with `petrophys.read(..., formations=...)` calibrate
  core permeability per formation when they form a single level.
- Shared formation import for the model, viewer, and formation analysis, with
  explicit column, worksheet, well, depth-unit, reference and datum handling.
- External lithology precedence, numeric category metadata, and gap/unknown
  reports; core saturation fraction normalization with provenance.
- Input templates, an own-data guide and notebook, and fresh-kernel execution
  of the user-journey tutorials in CI.
- `config_schema()` entries carry the settings `group`. The documentation adds
  an interpretation-settings page generated from it, a page for the
  top-level `petrophys` namespace, a "Terms, curves and steps" reference and
  a review checklist for interpretations.
- `read_formations(..., rank=...)` keeps one stratigraphic level.
- MCP server, tested live with an OpenAI model on 18 realistic requests
  (all answered correctly; input tokens down from 599k to 268k and tool
  errors from 5 to 1 on the original 14):
  - new tools `check_data_quality`, `spectral_gamma`, `run_workflow` /
    `workflow_step_kinds` (custom workflows) and `interpret_wells` (many wells
    in one table);
  - rich tool descriptions and compact outputs; header values with units and
    notes (e.g. an Rw without its temperature); `interpret_well` returns its
    assumptions and review notes (whole-well totals include non-reservoir
    rock); `summarize_zones(rank=...)`; forgiving well and track names;
    `curve_values` is capped; paths are reported relative to the allowed
    folders.
- MCP deployment: `petrophys-mcp serve --http` (streamable HTTP or `--sse`)
  with bearer-token auth, required beyond localhost, and `/healthz`;
  `petrophys-mcp config <client>` prints the setup for Claude Code, Claude
  Desktop, Cursor, VS Code, Codex and OpenAI; `petrophys-mcp check` verifies
  folders and a live session; a Dockerfile and docker-compose.yml;
  `examples/mcp_openai_agent.py`; and `scripts/mcp_live_eval.py` for
  re-running the live evaluation.
- Custom workflows: `pp.Workflow` and `pp.workflow_step` build reusable
  interpretation recipes from built-in steps (`petrophys.workflow.steps`,
  including `pp.interpret` as one step) and your own functions. Inputs are
  checked before running; runs keep a full trace and export to LAS/JSON;
  `run_many` runs a field; workflows save and load as JSON. New guide page and
  tutorial 21.
- Tools promoted from the tutorials into the library:
  `find_core_depth_shift` / `shift_core_depths` (core-to-log depth matching,
  with per-core-run scans); `find_held_readings` / `held_reading_mask` (held or
  padded tool readings); `summarize_pay` / `PaySummary` (net pay, NTG, pay
  averages and hydrocarbon pore thickness for a window); the new
  `petrophys.evaluation.sensitivity` module with `cutoff_sensitivity`,
  `parameter_sensitivity` (tornado data) and `compare_saturation_models`;
  `sp_rw_in_window` (SP-derived Rw in a chosen window at a reference
  temperature); `classify_th_u_environment`, `thorium_clay_index` and
  `uranium_stripped_gamma_ray` for spectral gamma; `label_depths` and
  `flag_intervals`. Tutorials 06, 11, 12, CS3, CS5 and the complete tour call
  them instead of inline helpers.
- The API reference is reorganized: a landing page with one card per area
  (front door, data model and I/O, preprocessing, interpretation, evaluation,
  workflows, visualization, reporting and LLM agents, legacy), a page per
  section with grouped summary tables, and one page per function and class,
  each documented once with its other import paths noted. It is generated
  from each package's `__all__`, so it follows the code; old reference URLs
  redirect to the new pages.
- The user guide is rewritten as executed pages: every example on the
  quickstart, own-data, input-template, data-model, glossary, architecture,
  workflow, review, context, MCP, visualization, specialist-data and
  capabilities pages runs at each documentation build, so the outputs and
  figures shown are always current. "What Petrophys can do" is a gallery of
  every capability with the call that produces it.
- A complete-tour notebook covering the whole library on four bundled GeoLink
  wells, and a real spectral-gamma geochemistry part plus a second-well core
  calibration in tutorial 12.
- `scripts/check_notebook_outputs.py`, run in CI, rejects tutorial notebooks
  with unexecuted cells, errors or local paths in their saved outputs.

### Fixed

- Named carbonate and marl cutoffs now agree with matrix routing. Marl remains
  an explicitly warned routing convention, not a calibrated mixed-rock model.
- The bundled 30/6-5 formation CSV declares metre depths and group/formation
  hierarchy. Direct DataFrame and file codebooks use the same validation.
- Netlify uses the strict documentation build and its required dependencies.
- Log-plot warnings no longer repeat a depth-reference note reported by both
  the formation and the lithology import.
- `workflow_steps()` descriptions say which steps `pp.interpret` does not run.
- Warnings escalated to errors (for example
  `warnings.simplefilter("error", pp.DataQualityWarning)`) now stop
  `pp.interpret` instead of being recorded as a failed step.
- An invalid `lithology_domain` names that setting and suggests the closest
  domain; `run_selected_steps` records skip reasons under the same step names
  as a full run.
- Interpreted LAS files record `PP_RW` and its temperature when
  `rw_meas_temp_f` is given (it was dropped), and a lithology-dependent Archie
  `a` is written as "per lithology".
- `CoreData.filter_by_depth` slices only per-plug arrays in its metadata;
  strings and dictionaries of the same length made it fail.
- `TemperatureProfile.from_bht` reads "degC"-style BHT labels as Celsius
  (only a bare "C" was converted).
- Buckles and Hingle plot labels stay inside the frame without overprinting;
  Buckles colours permeability on a log scale; Pickett plots no longer
  overlap minor tick labels.
- `create_heatmap` centres each cell on its coordinate (cells were shifted by
  half a cell), draws unevenly and log-spaced bins such as NMR T2 correctly,
  and also accepts cell edges.
- Analytics charts no longer flip the depth axis back when drawn twice on the
  same axes.
- `core_data_from_well` returns core water and oil saturation as fractions,
  converting percentages labelled as fractions with a warning, as its
  documentation states. The conversion rule now lives in
  `petrophys.model.units.core_saturation_to_fraction`.
- The core plotting helpers (`plot_calibration`, `plot_poro_perm`,
  `plot_core_on_logs`) and `GEOLINK_LITHOLOGY_CODE_LABELS` are exported from
  `petrophys.visualization` and appear in the API reference.
- The DLIS readers name the `petrophys[dlis]` extra when dlisio is missing,
  and `result.summary()` rounds net thicknesses to 0.01.

- A workflow reused across wells no longer carries one well's resolved
  matrix into the next.
- Celsius temperatures labelled like "deg C (from log)" are converted in every
  code path; units are parsed by one shared implementation.
- Reference reports label RMF methods as RMF (they were labelled as RW).
- Vendor mnemonics and units (e.g. `GR_EDTC`, `RHOZ` in kg/m3) plot correctly
  before interpretation.
- `run_formation_analysis` refuses to summarize a failed interpretation.
- LAS files are decoded as UTF-8 first, and unreadable files raise a clear
  `ValueError`.
- The ML extras install on Python 3.9 and 3.10 (the ML stack itself needs
  3.11+), and incompatible scikit-learn versions raise
  `ModelCompatibilityError`.

### Changed

- Formation model and summary loaders no longer add a final bottom 100 units
  below the last top; supply `terminal_depth` for top-only analysis, or use
  `pp.read_formations` for point annotations.
- Invalid geology rows raise instead of being skipped silently, summary
  totals count the covered depth union rather than the sum of nested rows,
  and unknown reader or workflow options raise instead of being ignored.
- `load_lithology_csv` is kept for compatibility; new code should use the
  shared validated readers.
- Release steps and code-placement rules moved from the documentation site to
  `CONTRIBUTING.md`; the Examples page is merged into Tutorials.
- Public modules are organized under `model`, `io`, `preprocessing`,
  `interpretation`, `evaluation`, `workflow`, `reporting`, and `visualization`.
  Consult the API reference when migrating imports from 2.x.
- The bundled imputation pack predicts DTC, NPHI, and RHOB.
  Estimators must have held-out-well R2 of at least 0.5. Other measured curves
  remain usable as companions. Imputation fills existing null samples and
  records provenance in LAS exports.
- Mineral inversion reports non-resolvable and zero-volume failures as missing
  values with QC flags. A reduced calcite-dolomite model is available when
  carbonate evidence supports it. Workflow traces record mineral selection.
- Context packages are computed by the same workflow as direct
  interpretations; no archived context payload or benchmark manifest is read.
- Package checks install the wheel and execute a sample workflow. Documentation
  builds treat warnings as errors, and release tags must match package versions.
- Third-party data provenance and model validation records accompany the package.
