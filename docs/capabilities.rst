Capability Map
==============

This page maps the current library by processing lifecycle. The canonical
public namespaces are ``petrophys.model``, ``petrophys.io``,
``petrophys.preprocessing``, ``petrophys.interpretation``,
``petrophys.evaluation``, ``petrophys.workflow``, ``petrophys.reporting``, and
``petrophys.visualization``.

Data model and interchange
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Capability
     - Scope
     - Main namespace
   * - Well and curve model
     - Depth-indexed scalar curves, metadata, parameters, classifications,
       core data, formation tops, zones, and subzones
     - ``petrophys.model``
   * - LAS
     - Header, parameter, depth, and scalar-curve reading; LAS writing with
       computed-curve metadata
     - ``petrophys.io``
   * - DLIS
     - Optional scalar-frame reading and multi-frame interpolation; specialist
       vector channels are handled separately
     - ``petrophys.io``
   * - Tables
     - Wide CSV, long multiwell CSV, delimited ASCII, CSV/Excel export
     - ``petrophys.io``
   * - Packaged data
     - Small North Sea, COSTA carbonate, and processed NMR samples plus
       repository-only Daydream integration fixtures
     - ``petrophys.datasets``

Preprocessing
-------------

* **Curve standardization:** recognized vendor aliases, canonical working
  names, supported unit conversions, null cleaning, and plausible-range
  screening.
* **Wellbore quality:** washout, large-hole, bad-hole, and curve-quality masks
  driven by available caliper, bit-size, density-correction, density, and PEF
  evidence. Missing bit size is not silently replaced by a fixed value.
* **Normalization:** linear, percentile, z-score, reference-well,
  histogram-matching, and histogram-equalization utilities.
* **Curve imputation:** compatible pretrained model packs can fill only missing
  target samples while retaining observed values and recording model
  provenance. KNN, random forest, and XGBoost training utilities are available
  for building project-specific packs.

The bundled predictors support **DTC, NPHI, and RHOB**. Each retained
model/target pair has held-out-well R-squared of at least 0.5. Their best recorded
scores are 0.642, 0.699, and 0.692, respectively; these are validation
scores, not per-sample confidence estimates. Inspect the packaged
``pretrained/validation.json`` for all model metrics and exclusions. Observed
BS, CALI, GR, PEF, resistivity, SP, and DRHO can still serve as companion inputs, but the
bundled pack does not predict those targets. Out-of-distribution intervals and
missing companion measurements can remain unfilled.

Formation interpretation
------------------------

.. list-table::
   :header-rows: 1
   :widths: 23 48 29

   * - Domain
     - Methods and products
     - Package
   * - Temperature
     - Geothermal gradients, depth profiles, and resistivity-temperature
       correction
     - ``interpretation.temperature``
   * - Clay volume
     - Gamma ray, spontaneous potential, neutron-density, multi-method VCL,
       classifications, and clay resistivity
     - ``interpretation.clay_volume``
   * - Lithology
     - Density-neutron, M-N, PEF crossplot data; two- and three-mineral solving;
       interval lithology; matrix inference and volume-derived classification
     - ``interpretation.lithology``
   * - Porosity
     - Density, neutron, sonic, total, effective, secondary, and optional
       hydrocarbon-response corrections
     - ``interpretation.porosity``
   * - Electrical properties
     - Rw from supplied values, salinity, or SP; temperature correction; Rmf;
       invasion ratios; Pickett analysis and Archie parameters
     - ``interpretation.electrical``
   * - Water saturation
     - Archie, Simandoux, Indonesia, North Sea, Nigeria-Schlumberger,
       Waxman-Smits, Dual Water, Sxo, classifications, and automatic routing
     - ``interpretation.saturation``
   * - Permeability
     - Timur, Coates-Denoo, Wyllie-Rose, lithology/VCL routes, core-calibrated
       inputs, classifications, and impermeability flags
     - ``interpretation.permeability``
   * - Rock physics
     - Elastic-property calculations and sonic fluid-substitution support
     - ``interpretation.rock_physics``

Specialist data
---------------

* **NMR:** processed T2-bin distributions, logarithmic mean T2, BVI, FFI,
  clastic/carbonate cutoffs, SDR and Timur-Coates permeability, calibration,
  and NMR panels.
* **Borehole images:** DLIS or LAS pad/flap-array loading where the delivery
  exposes compatible channels; navigation, azimuthal orientation, static and
  dynamic equalization, coverage statistics, gap handling, and image panels.
* **Geochemistry:** XRF tables, spectral gamma interpretation, elemental
  spectroscopy, oxide closure, and depth-aligned multi-element panels.
* **Core:** porosity and permeability calibration, grain-density correction,
  poro-perm transforms, Archie ``m`` estimation, mechanical measurements,
  grain-size classes, digital facies, and core-image color measurements.
* **Saturation height:** capillary pressure-height conversion, laboratory to
  reservoir conversion, Leverett J, J-function and Brooks-Corey fitting, and
  log-column saturation prediction.
* **Sonic saturation:** a resistivity-independent, Gassmann-based saturation
  estimate with a detectability gate for suitable sonic-density inputs.

Reservoir evaluation
--------------------

* **Rock quality and flow units:** RQI, normalized porosity, FZI, hydraulic
  flow-unit assignment, and FZI-based permeability prediction.
* **Reservoir indicators:** BVW, moveable-hydrocarbon index, invasion and
  saturation classifications, and diagnostics.
* **Net intervals:** configurable and lithology-aware net rock, net reservoir,
  net pay, seal, and bad-hole exclusion logic.
* **Summaries:** thickness-aware averages, net-to-gross, formation and zone
  summaries, and hydrocarbon pore-volume calculations.
* **Diagnostics:** numerical, lithology, porosity, saturation, permeability,
  and hydrocarbon analysis helpers.

Orchestration and delivery
--------------------------

* **Dynamic workflow:** input-aware execution with per-step methods,
  parameters, decisions, warnings, outputs, and skipped prerequisites.
* **Zoned workflow:** depth-varying parameter overrides and formation-aware
  interpretation.
* **Batch processing:** directory processing, interpreted LAS output, trace
  JSON, curve audits, and batch reports.
* **Visualization:** adaptive log viewers, named templates, lithology patterns,
  triple combos, crossplots, Pickett plots, NMR/FMI/XRF panels, saturation
  height, flow units, and multiwell analytics.
* **Reporting:** standard LAS/CSV/Excel exports and structured LAS/JSON work
  packages with calculation provenance and category/formation/lithology
  groupings.

Scope boundaries
----------------

Petrophys is a formation-evaluation library, not a replacement for acquisition
processing or a commercial log-editing platform. In particular:

* It does not perform automatic run-to-run depth matching or tool-depth
  correction.
* DLIS support is input-aware, but it does not decode every proprietary raw
  acquisition product from every service company.
* Borehole-image APIs support compatible processed arrays and image processing;
  they do not claim complete structural or fracture interpretation.
* NMR APIs start from delivered distributions or interpreted NMR curves rather
  than reconstructing raw echo trains.
* Empirical and routed calculations still require petrophysical assumptions,
  local calibration, and professional review.
