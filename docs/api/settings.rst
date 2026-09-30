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

A default of ``None`` means the value is taken from the data or from the
standard for the lithology domain; the description says which. Invalid values
raise ``ValueError`` before any calculation, naming the accepted values.
``result.summary()["settings_used"]`` shows the values a run actually used.

This page is generated from the code. The same information is available as
data from :func:`petrophys.workflow.schema.config_schema`, which is what the
:doc:`MCP server <../mcp>` gives to LLM clients.

.. include:: settings_table.inc
