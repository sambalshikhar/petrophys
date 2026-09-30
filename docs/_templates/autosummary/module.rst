{% set reexport_modules = [
   "petrophys.evaluation.intervals",
   "petrophys.evaluation.summaries",
   "petrophys.interpretation.core_analysis",
] %}
{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}{% if fullname in reexport_modules %}
   :no-index:
{% endif %}{% if fullname == "petrophys.workflow" %}
   :exclude-members: StandardWorkflow, WorkflowConfig, WorkflowResult, StepTrace, run_standard_workflow
{% endif %}
