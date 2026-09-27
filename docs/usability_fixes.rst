Usability Fixes
===============

This implementation addresses the audited read/plot/interpret journey. It does
not replace every specialized interface in the package.

Implemented
-----------

* Short public read, plot, and copy-by-default interpretation APIs.
* Shared formation imports for the model, viewer, and formation analysis.
* Explicit column, worksheet, well, depth-unit, reference and datum handling.
* External lithology precedence, numeric category metadata, and gap/unknown reports.
* Explicit terminal depths, preserved coincident tops, rank/parent metadata,
  documented parent containment, and union-based formation totals.
* Core saturation fraction normalization with provenance and explicit overrides.
* Plot diagnostics, track selection, workflow errors, and skipped-step reasons.
* Updated first-plot instructions, own-data guide, runnable example, templates,
  and Windows CI coverage for public API user journeys.
* Refreshed quickstart, own-data, foundations 00/01/02/05/07, advanced
  10/11/17 and CS1 notebooks, with portable kernels, regenerated outputs,
  and repeatable fresh-kernel execution.

Remaining Audit Work
--------------------

The legacy family-only lithology importer and specialized reference-report
parsers retain their historical contracts. Vendor code detection outside the
external-interval route still has legacy heuristics; arbitrary custom integer
codes require a verified codebook, not an assumption that they are GeoLink codes.
Migrating those APIs requires compatibility decisions and additional reference
fixtures.

A graphical file-mapping/preflight screen, full batch/CLI consolidation,
interactive plot redesign and richer WITSML/mixture ingestion are not implemented
here. The primary user-journey notebooks are updated; other specialist lessons
retain their detailed scientific module interfaces and optional dependencies.
See :doc:`tutorials/own_data`, :doc:`own_data` and :doc:`input_templates`.
