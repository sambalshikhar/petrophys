{#- Own public members only (no inherited methods or dataclass fields).
    Attributes the class docstring lists in a NumPy "Attributes" section are
    documented there, so autodoc skips them. The ph_* data comes from
    docs/api/_generate.py through autosummary_context. -#}
{%- set exclude = ph_exclude.get(fullname, []) if ph_exclude is defined else [] %}
{%- set inherited_fields = ph_inherited.get(fullname, []) if ph_inherited is defined else [] %}
{%- set own_attributes = attributes | reject("in", inherited_members) | reject("in", exclude)
      | reject("in", inherited_fields) | list %}
{%- set own_methods = methods | reject("in", inherited_members) | reject("equalto", "__init__") | list %}
{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

{% include "autosummary/_path.rst" %}

.. autoclass:: {{ objname }}
   :members:
   :undoc-members:
{%- if ph_bases is defined and fullname in ph_bases %}
   :show-inheritance:
{%- endif %}
{%- if exclude %}
   :exclude-members: {{ exclude | join(", ") }}
{%- endif %}
{% if own_attributes %}
   .. rubric:: Attributes
      :class: ph-api-members

   .. autosummary::
      :nosignatures:
{% for item in own_attributes %}
      ~{{ objname }}.{{ item }}
{%- endfor %}
{% endif %}
{%- if own_methods %}
   .. rubric:: Methods
      :class: ph-api-members

   .. autosummary::
      :nosignatures:
{% for item in own_methods %}
      ~{{ objname }}.{{ item }}
{%- endfor %}
{% endif %}
