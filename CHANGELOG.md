# Changelog

## 3.0.0 (unreleased)

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
