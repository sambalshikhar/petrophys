Interpretation settings
=======================

Every keyword accepted by ``pp.interpret`` is a field of
:class:`~petrophys.workflow.DynamicWorkflowConfig`. Pass settings as keywords
or as one config object:

.. code-block:: python

   import petrophys as pp

   well = pp.read(pp.sample("30_6-5"))
   result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230, sw_model="auto")

   config = pp.DynamicWorkflowConfig(rw=0.07, rw_meas_temp_f=230)
   result = pp.interpret(well, config)

A default of ``None`` means the value is taken from the data, from the
standard for the lithology family, or from the calculation profile; the
description says which, and the Default column adds the value a profile sets.
``calculation_profile`` selects an interpretation configuration: ``"current"``
is the default and ``"geolink"`` supplies the GeoLink evaluation configuration
(see :ref:`calculation-profiles`). For dataset context generation, see
:doc:`../benchmark_context`. Invalid values raise
``ValueError`` before any calculation, naming the accepted values.
``result.summary()["settings_used"]`` shows the values a run actually used,
and ``result.config`` the whole configuration.

Which value a run uses, first to last: your ``zones`` values inside their
zone, a setting you pass (even one equal to its default), a LAS header zone
value (``RW_Z1`` with ``TOP_Z1``/``BOT_Z1``) inside its zone, a LAS header
whole-well value (see :doc:`../input_templates`), the default. A header value
one of yours overrides is named in the warnings. The same order holds for
``pp.interpret``, ``run_dynamic_workflow``, the batch and zoned runners,
``pp.export_context`` and the MCP server.

This page is generated from the code. The same information is available as
data from :func:`petrophys.workflow.schema.config_schema`, which is what the
:doc:`MCP server <../mcp>` gives to LLM clients.

.. include:: settings_table.inc
