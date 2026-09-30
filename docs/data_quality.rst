Reviewing an Interpretation
===========================

A completed run shows that the calculations ran. It does not make the result a
calibrated reservoir interpretation. Before you use the numbers, work through
the checks below.

.. code-block:: python

   import petrophys as pp

   well = pp.read(pp.sample("30_6-5"), formations=pp.sample("formations_30_6-5"))
   print(well.describe())                     # curves, units, coverage

   result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230)
   print(result.status)                       # "interpreted", "no_interpretation" or "failed"
   print(result.summary(top=2841, bottom=2958))
   for warning in result.warnings:
       print("-", warning)
   print(result.specific_steps_skipped)       # step -> reason it did not run
   print(result.formation_summary(rank="formation"))  # net pay per formation

A review checklist
------------------

1. **Inputs.** Are all the curves you expect listed by ``well.describe()``,
   with the right units? Look for ``DataQualityWarning`` messages from
   ``pp.read`` about unit conversions and implausible values.
2. **Rw and its temperature.** Rw is quoted at a temperature. Pass
   ``rw_meas_temp_f`` whenever you pass ``rw``. Without it, Petrophys assumes
   75 degF and warns. Do not tune Rw just to get the pay you expect.
3. **Clay volume.** Check the gamma-ray endpoints (``gr_min``/``gr_max``,
   default 20 and 100 API) and the VCL method in ``result.summary()``.
4. **Porosity and saturation.** Compare the porosity with core where you
   have it, and check which saturation model the ``saturation`` step chose
   (``result.print_trace()``).
5. **Hole quality.** Look at ``BAD_HOLE`` alongside the pay flags (see below).
6. **Cutoffs.** Replace the default cutoffs with project values when you have
   them.

Net-pay flags
-------------

``NET_ROCK``, ``NET_RES`` and ``NET_PAY`` are cutoff flags. They are not
confirmed reservoir or hydrocarbons. Unless you set them, the cutoffs are
VCL < 0.30, SW < 0.70, and the standard porosity cutoff for the lithology
domain. ``result.summary()`` lists the values used.

The flags are deliberately simple:

* where VCL is missing, it is treated as zero clay;
* ``BAD_HOLE`` does not remove pay;
* a shale label from an external lithology file does not remove pay.

So inspect the flags together with bad-hole QC and your geology, and use
``result.formation_summary()`` to see where the pay sits. On 30/6-5, for
example, most of the flagged pay lies far above the Brent reservoir: in the
shallow, claystone-dominated Nordland and Hordaland groups, where the
calculated clay volume is low and the porosity high, and in the
shale-dominated Heather Formation. Summarize the reservoir interval
(``result.summary(top=2841, bottom=2958)``) and treat pay elsewhere as a
question to review, not a result. The
`Sodir description of the Heather Formation <https://factpages.sodir.no/nb-no/strat/pageview/litho/formations/60>`_
is a useful geological cross-check.

Marl
----

Marl is a clay-carbonate mixture, not pure limestone; see the
`SLB definition <https://glossary.slb.com/en/terms/m/marl>`_. Petrophys
interprets marl intervals with the limestone matrix and the carbonate cutoff
family, and warns when an external lithology file contains marl. Supply
project-specific endpoints and cutoffs for marl-rich intervals.

Known limitations of the sample data
------------------------------------

* **30/6-5 SWC/SOC:** the header says fraction, but the values are
  percentages. Petrophys converts them to fractions on reading and records the
  correction; the bundled LAS is unchanged. Magnitude alone cannot tell
  0.5 percent from a 0.5 fraction, so give an explicit unit override for
  ambiguous low values.
* **15/9-14 DTE:** the values do not fit the sonic-slowness unit in the
  header, and their meaning is unknown. DTE is not treated as shear slowness
  (DTS). Range QC reports the implausible values; do not rescale them.
* **Daydream CMR:** this sample holds scalar NMR curves only. It has neither
  T2 bins nor T2LM, so ``nmr.t2_lm()`` cannot be computed. The stored vendor
  permeability curves (KSDR, KCMR) can be read, but they are not recomputed.
  ``CMRP_3MS`` is labelled V/V but holds porosity units (median about 2.7).
* **30/6-5 formations:** the bundled table has groups and formations that
  nest. Pass ``rank="formation"`` (or ``"group"``) to
  ``result.formation_summary()`` for a table without overlaps. When the
  nested table is attached, core permeability is calibrated per group and
  per formation, and a formation's model takes precedence over its group's;
  ``pp.read(..., formation_options={"rank": "formation"})`` attaches one
  level only.
* **30/6-5 header Rw:** the LAS header gives ``RW 0.7`` ohm.m with no
  temperature, ten times the 0.07 ohm.m used in the tutorials; it is most
  likely a slipped decimal. Always pass ``rw`` and ``rw_meas_temp_f``.
* **COSTA wells (HW-*):** each LAS covers only the logged Kharaib-2 reservoir
  interval of a well, not the whole well. The bundled
  ``formations_costa.csv`` labels that interval ``KHARAIB-2``, so these wells
  cannot be used for whole-well or multi-formation studies.

Geology is optional
-------------------

Formation and lithology files are optional. Logs alone drive the workflow,
including the mineral solve when its curves exist. A formation name never
creates mineral fractions or missing logs.

Petrophys needs no proprietary workbooks. Use the tables in
:doc:`input_templates`, or pass your own table with explicit column, sheet,
depth-unit and rank options. ``pp.read_lithology_codes`` reads Kingdom
lithology codebooks: it prefers the ``Attribute`` rock names over the
``Lithology Attribute`` group labels, warns when they disagree, and rejects
duplicate codes and empty labels. Petrophys only reads the files you pass.

Native DLIS files
-----------------

The Daydream scalar and borehole-image DLIS files are stored in the
repository with Git LFS; they are not part of the installed package. To use
them from a checkout:

.. code-block:: bash

   git lfs install
   git lfs pull
   python -m pip install -e ".[dlis,tutorials]"

Tutorial 16 needs the real image DLIS, not the small LFS pointer file.
File inventory and checksums are in ``src/petrophys/datasets/daydream/README.md``.
Missing measurements are never fabricated to make a tutorial run.
