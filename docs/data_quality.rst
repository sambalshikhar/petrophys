Data Quality and Interpretation Review
======================================

A completed run is not a calibrated reservoir interpretation. Inspect
``well.describe()``, ``result.summary()``, ``result.warnings`` and
``result.to_trace_dict()``.

Net-pay screening
-----------------

There is one context-generation workflow; vendor curves do not switch to a
different calculation profile. Unless explicitly supplied in the standardized
parameter table, the canonical net-pay route uses VCL < 0.30 and SW < 0.70,
with the established lithology-family porosity cutoff.

These flags intentionally retain the established context calculation. Missing
VCL is treated as zero clay, and ``BAD_HOLE`` or a shale label does not
automatically erase candidate pay. Therefore ``NET_ROCK``, ``NET_RES``, and
``NET_PAY`` are screening outputs, not confirmed reservoir or hydrocarbons.
Inspect them together with bad-hole QC and supplied geology.

Review Rw **and its reference temperature**, clay-volume endpoints and
method, porosity calibration, saturation model, cutoffs and hole quality.
Do not tune Rw merely to obtain expected pay. Heather/Lista and Hordaland
results require this scientific review; cutoff flags are not confirmed pay.
The official
`Heather description <https://factpages.sodir.no/nb-no/strat/pageview/litho/formations/60>`_
is a useful geological cross-check, not a replacement for interval data.

Marl convention
---------------

Marl is a clay-carbonate mixture, not pure limestone; see the
`SLB definition <https://glossary.slb.com/en/terms/m/marl>`_.
The existing limestone-matrix routing and carbonate-cutoff family are now
consistent for marl/marlstone. External marl intervals generate a warning:
this is a routing convention requiring calibration, not a validated mineral
mixture. Use project-specific endpoints and numeric cutoffs when appropriate.

Known sample limitations
------------------------

* **30/6-5 SWC/SOC:** headers say fraction but populated values are percentages.
  Standardization converts percent-scale values to fractions and records
  the correction; the source LAS is preserved. Ambiguous low percentages
  require an explicit unit override: magnitude alone cannot distinguish
  0.5 percent from a 0.5 fraction.
* **15/9-14 DTE:** magnitudes are incompatible with the sonic-slowness unit.
  DTE is not a DTS alias; range QC reports implausible values. Its true
  meaning remains unverified. Do not infer shear velocity or rescale it
  without an acquisition mnemonic dictionary.
* **Daydream CMR:** this scalar sample has neither T2 bins nor T2LM.
  ``nmr.has_t2lm`` is false; ``nmr.t2_lm()`` cannot be computed.
  Stored vendor KSDR/KCMR may still be read. Returning stored permeability
  is not recomputing SDR or calibrating a T2 distribution.
* **30/6-5 formations:** the bundled CSV declares metres, rank and parent.
  Group/formation nesting is intentional. Select one rank for
  non-overlapping formation summaries rather than flattening all levels.

Geology and codebooks
---------------------

External formations and lithology remain optional. Logs alone can drive
available workflow steps, including mineral solving when its measurements
exist. A formation name does not create mineral fractions or missing logs.

Private NPD and Kingdom workbooks are not package dependencies.
Use :doc:`input_templates` or supply a table with explicit column, sheet,
depth-unit and hierarchy options. ``pp.read_lithology_codes`` prefers
Kingdom's ``Attribute`` rock names over ``Lithology Attribute`` group labels,
warns on disagreement, and rejects duplicate codes and empty labels.
No neighboring personal directory is searched automatically.

Native DLIS fixtures
--------------------

Daydream scalar and FMI files are repository Git LFS fixtures, not wheel
contents. In a checkout:

.. code-block:: bash

   git lfs install
   git lfs pull
   python -m pip install -e ".[dlis,tutorials]"

Tutorial 16 needs the real image DLIS, not its small LFS pointer.
Inventory/checksums are in ``src/petrophys/datasets/daydream/README.md``.
Missing measurements are never fabricated to make a tutorial run.
