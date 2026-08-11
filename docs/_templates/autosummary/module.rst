{% set reexport_modules = [
   "petrophys.evaluation.intervals",
   "petrophys.evaluation.summaries",
   "petrophys.interpretation.core_analysis",
] %}
{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}{% if fullname in reexport_modules %}
   :no-index:
{% endif %}
