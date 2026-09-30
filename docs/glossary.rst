Terms, Curves and Steps
=======================

A reference for the names you meet in results, traces and settings.

Terms
-----

Lithology domain
   The mineral model an interpretation uses: ``clastic``, ``carbonate`` or
   ``auto``. Set with ``lithology_domain``. It is one setting for the whole
   run, not a rock label per depth.

Lithology intervals
   Optional per-depth rock labels (sandstone, shale, limestone, ...) from a
   table such as ``lithology_intervals.csv``. They take precedence over
   labels embedded in the LAS or produced by the mineral solve.

Formations
   Optional named depth intervals (groups, formations, members). They are
   used for per-formation summaries, plots and core calibration.

Rw
   Formation-water resistivity, in ohm.m, quoted at a reference temperature
   (``rw_meas_temp_f``). The workflow corrects it to formation temperature.

Cutoffs
   Limits on clay volume, porosity and water saturation that turn curves into
   net-rock, net-reservoir and net-pay flags.

Trace
   The record of a run: for each step, whether it ran, its method, input and
   output curves, parameters, decisions and warnings. See
   ``result.print_trace()`` and ``result.to_trace_dict()``.

Context
   A compact JSON summary of an interpreted well for LLMs and other tools;
   see :doc:`context`.

Where each input goes
---------------------

The same input has a different argument name in each interface:

.. list-table::
   :header-rows: 1
   :widths: 16 22 22 20 20

   * - Input
     - Python
     - ``export_context`` package
     - Command line
     - MCP ``load_well``
   * - Formations
     - ``pp.read(path, formations=...)`` or
       ``pp.read_formations(...)``
     - ``tops_file=``
     - ``--formations``
     - ``formations``
   * - Lithology intervals
     - ``pp.read(path, lithology=...)`` or
       ``pp.read_lithology_intervals(...)``
     - ``lithology_file=``
     - ``--lithology``
     - ``lithology``
   * - Well parameters
     - ``pp.interpret(well, **pp.read_well_parameters(...))``
     - ``parameters_file=``
     - ``--parameters``
     - settings in ``interpret_well``
   * - Core measurements
     - ``pp.attach_core_measurements(well, ...)``
     - ``core_file=``
     - ``--core``
     - not available
   * - Lithology codebook
     - ``pp.read_lithology_codes(...)``
     - ``lithology_mapping_file=``
     - ``--lithology-codes``
     - not available

.. warning::

   ``pp.read(path, lithology=...)`` attaches lithology *intervals*. In
   ``pp.interpret`` and ``DynamicWorkflowConfig``, ``lithology=`` is the
   deprecated name of ``lithology_domain``. Use ``lithology_domain`` there.

The file layouts are in :doc:`input_templates`.

Workflow steps
--------------

``pp.interpret`` runs these steps in order. A step runs only when its inputs
are available; ``result.specific_steps_skipped`` gives the reason for every
step that did not run.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Step
     - What it does
   * - ``normalization``
     - Canonical curve names and units; implausible-value screening
   * - ``input_context``
     - Records which formation and lithology sources were used
   * - ``temperature``
     - Formation temperature profile (``TEMP``) from BHT or a gradient
   * - ``bad_hole``
     - Washout and density-correction quality flag (``BAD_HOLE``)
   * - ``vclay``
     - Clay volume from gamma ray, SP and neutron-density (``VCL``)
   * - ``rcl``
     - Clay resistivity from shale intervals
   * - ``rmf``
     - Mud-filtrate resistivity at formation temperature (``RMF``)
   * - ``hc_correction``
     - Neutron-density hydrocarbon correction; opt-in with
       ``apply_hc_correction=True``
   * - ``porosity``
     - Density, neutron, sonic, total and effective porosity
   * - ``lithology``
     - Mineral volumes: clastic multi-mineral or carbonate solve
   * - ``core_correction``
     - Porosity and permeability calibrated to core, when core is attached
   * - ``rw``
     - Formation-water resistivity at formation temperature (``RW``)
   * - ``invasion``
     - Invasion diagnosis from shallow and deep resistivity
   * - ``saturation``
     - Water and flushed-zone saturation (``SW``, ``SXO``)
   * - ``permeability``
     - Core-calibrated or empirical permeability (``PERM``)
   * - ``rock_quality``
     - Reservoir quality index and flow zone indicator (``RQI``, ``FZI``)
   * - ``bvw``
     - Bulk volume water (``BVW``)
   * - ``mhi``
     - Moveable hydrocarbon index (``MHI``)
   * - ``net_pay``
     - Net-rock, net-reservoir and net-pay flags
   * - ``permeability_finalize``, ``rock_quality_finalize``
     - Permeability and rock quality recomputed from the final reservoir flag

Three step names appear in ``result.steps_skipped`` on every run because
``pp.interpret`` does not run them: ``spectral_gamma`` and
``elemental_spectroscopy`` (call the functions in
:mod:`petrophys.interpretation.geochemistry` directly; see
:doc:`specialist_data`) and ``porosity_refinement`` (porosity is computed
once, in the ``porosity`` step).

Curves the workflow adds
------------------------

Which curves appear depends on the inputs. Fractions are in v/v.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Curve
     - Meaning
   * - ``TEMP``
     - Formation temperature, degF
   * - ``BAD_HOLE``
     - 1 where the borehole compromises the logs
   * - ``VCL``; ``VCL_GR``, ``VCL_SP``, ``VCL_ND``
     - Clay volume; the gamma-ray, SP and neutron-density estimates
   * - ``RMF``, ``RW``
     - Mud-filtrate and formation-water resistivity at formation
       temperature, ohm.m
   * - ``PHID``, ``PHIN``, ``PHIS``
     - Density, neutron and sonic porosity
   * - ``PHIT``, ``PHIE``
     - Total and effective porosity
   * - ``PHI_SEC``
     - Secondary-porosity index
   * - ``VQUAR``, ``VCALC``, ``VDOLO``, ``VANHY``
     - Quartz, calcite, dolomite and anhydrite volumes
   * - ``SW``, ``SXO``
     - Water saturation in the uninvaded and flushed zones
   * - ``INVASION_RATIO``
     - Rxo/Rt
   * - ``PERM``; ``PHIE_CC``, ``PERM_CC``
     - Permeability, mD; core-calibrated porosity and permeability
   * - ``SWIRR_APP``
     - Apparent irreducible water saturation
   * - ``RQI``, ``PHI_Z``, ``FZI``
     - Reservoir quality index, normalized porosity, flow zone indicator
   * - ``BVW``, ``MHI``
     - Bulk volume water; moveable hydrocarbon index
   * - ``NET_ROCK``, ``NET_RES``, ``NET_PAY``
     - Cutoff flags (1 or 0); see :doc:`data_quality`
   * - ``RESFLAG``, ``SEALFLAG``
     - Reservoir and seal flags
   * - ``CROSSOVER_FLAG``, ``PRIME_OIL_TARGET``
     - Neutron-density crossover; integrated oil-target flag

Some curves are also written under a second name for compatibility (for
example ``VCLAY`` for ``VCL`` and ``VCLGR`` for ``VCL_GR``).
