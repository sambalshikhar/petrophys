# Changelog

All notable changes to Petrophys are documented here.

## 3.0.0 (unreleased)

### Breaking changes

- `DynamicWorkflowConfig` validates every setting when it is created and
  raises `ValueError` listing all invalid fields with the accepted values
  (previously, typos such as `hc_type="gass"` silently used defaults).
- `DynamicWorkflowConfig` defaults `matrix_type` and `lithology` to `"auto"`,
  the same as `petrophys.interpret`; wells without lithology evidence still
  resolve to sandstone/clastic. The legacy reference-JSON exporters keep
  their previous sandstone/clastic default.
- Tabular readers (`io.read_csv`, `read_csv_wells`, `read_ascii` without a
  units row, `Well.from_dataframe`) require `depth_unit` instead of assuming
  feet. `Well.depth_unit` accepts only metre or foot labels.
- The log viewer no longer searches parent directories for
  `meta_files/Litho_Nomenclature_Kingdom.xlsx`; pass `lithology_mapping` or
  set `PETROPHYS_LITHOLOGY_MAPPING`.

### Added

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

### Fixed

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
- The vendor benchmark profile retains its versioned calculation path.
- Package checks install the wheel and execute a sample workflow. Documentation
  builds treat warnings as errors, and release tags must match package versions.
- Third-party data provenance and model validation records accompany the package.
