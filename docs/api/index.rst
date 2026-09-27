API Reference
=============

The supported API is grouped by lifecycle and scientific domain. These pages
are generated from the source docstrings. Internal helpers under
``petrophys._internal`` are intentionally excluded.

Common tasks
------------

These functions are also available as ``petrophys.read``,
``petrophys.plot``, and ``petrophys.interpret``.

.. autosummary::
   :toctree: generated

   petrophys.api

Data model and I/O
------------------

.. autosummary::
   :toctree: generated

   petrophys.model
   petrophys.model.log_data
   petrophys.io
   petrophys.datasets

Preprocessing
-------------

.. autosummary::
   :toctree: generated

   petrophys.preprocessing.standardization
   petrophys.preprocessing.normalization
   petrophys.preprocessing.quality
   petrophys.preprocessing.imputation

Conventional interpretation
---------------------------

.. autosummary::
   :toctree: generated

   petrophys.interpretation.temperature
   petrophys.interpretation.clay_volume
   petrophys.interpretation.lithology
   petrophys.interpretation.porosity
   petrophys.interpretation.porosity.hydrocarbon_correction
   petrophys.interpretation.electrical.rw
   petrophys.interpretation.electrical.rmf
   petrophys.interpretation.electrical.invasion
   petrophys.interpretation.electrical.cementation
   petrophys.interpretation.saturation
   petrophys.interpretation.permeability
   petrophys.interpretation.rock_physics

Specialist interpretation
-------------------------

.. autosummary::
   :toctree: generated

   petrophys.interpretation.nmr
   petrophys.interpretation.borehole_imaging
   petrophys.interpretation.geochemistry
   petrophys.interpretation.core_analysis
   petrophys.interpretation.saturation.height
   petrophys.interpretation.saturation.sonic

Reservoir evaluation
--------------------

.. autosummary::
   :toctree: generated

   petrophys.evaluation.bvw
   petrophys.evaluation.mhi
   petrophys.evaluation.net_pay
   petrophys.evaluation.rock_quality
   petrophys.evaluation.flow_units
   petrophys.evaluation.intervals
   petrophys.evaluation.summaries
   petrophys.evaluation.volumetrics
   petrophys.evaluation.diagnostics

Workflows, reporting, and visualization
---------------------------------------

.. autosummary::
   :toctree: generated

   petrophys.workflow
   petrophys.reporting
   petrophys.visualization
