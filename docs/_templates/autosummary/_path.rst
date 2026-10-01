.. container:: ph-api-path

   ``{{ fullname }}``
{%- if ph_also is defined and ph_also.get(fullname) %}

   .. rst-class:: ph-api-also-as

   Also available as {% for path in ph_also[fullname] %}``{{ path }}``{{ ", " if not loop.last }}{% endfor %}
{%- endif %}
