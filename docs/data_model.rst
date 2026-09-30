Data and Depth
==============

Petrophys represents a logging run as a mutable :class:`petrophys.model.Well`.
The well owns one depth index, a mapping of mnemonic names to
:class:`petrophys.model.Curve` objects, header metadata, parameters, and
formation intervals. Scientific modules add their calculated curves to this
same object unless an API explicitly documents a separate return value.

Supported inputs
----------------

.. list-table::
   :header-rows: 1
   :widths: 17 28 55

   * - Input
     - Reader
     - Behavior
   * - LAS
     - :func:`petrophys.io.read_las`
     - Reads scalar curves, the LAS header, parameters, and the supplied depth
       index.
   * - DLIS
     - :func:`petrophys.io.read_dlis`
     - With the optional ``dlis`` dependency, merges compatible scalar
       channels from selected frames onto one depth grid by interpolation.
   * - CSV
     - :func:`petrophys.io.read_csv`
     - Builds one well from a wide table whose numeric columns share the
       supplied depth column.
   * - Multiwell CSV
     - :func:`petrophys.io.read_csv_wells`
     - Splits a long table by well identifier and builds one ``Well`` per
       group.
   * - Delimited ASCII
     - :func:`petrophys.io.read_ascii`
     - Reads a tabular file with a configurable separator, header, depth
       column, and null value.

What depth alignment means
--------------------------

Petrophys does **not** currently perform commercial log-depth matching. It
does not estimate tool-depth offsets, correlate repeated passes, or apply
stretch-and-squeeze corrections.

For LAS, CSV, and delimited ASCII inputs, curves remain associated with the
depth samples already supplied by the file. A curve must have the same number
of samples as the well depth index. Duplicate or changing-direction LAS depth
values raise a warning for review rather than being silently corrected.

When :func:`petrophys.io.read_dlis` merges scalar channels from multiple DLIS
frames, it converts compatible depth units and linearly resamples those
channels onto a common numerical grid. This is grid resampling, not physical
depth matching. Vector channels such as image or spectroscopy arrays are
handled through their specialist readers instead of being flattened into the
conventional scalar ``Well``.

If you need an explicit resampling operation for an independently sampled
array, use :func:`petrophys.model.log_data.align_to_depth` and keep the
provenance of that operation with your project data.

Curve identity and units
------------------------

Input readers preserve the values and metadata available in the source.
During dynamic-workflow normalization, recognized vendor mnemonics can be
mapped to canonical working names and supported physical units can be
converted. Unknown mnemonics and units are retained; they are not guessed into
an unrelated curve type.

By default, the dynamic workflow keeps vendor mnemonics and adds canonical
working aliases where needed. Set ``rename_input_curves=True`` only when the
input names themselves should be replaced.

.. code-block:: python

   import petrophys
   from petrophys.io import read_las
   from petrophys.preprocessing import standardize_curves

   well = read_las(petrophys.sample("30_6-5"))
   standardize_curves(well, rename_curves=False, convert_units=True)

   gr = well.get("GR")
   print(gr.mnemonic, gr.unit, gr.depth_unit)

Calculated curves
-----------------

``Well.add_curve`` associates a calculated array with the well's existing
depth index. The dynamic workflow uses this operation for products such as
``VCL``, ``PHIT``, ``PHIE``, ``SW``, ``PERM``, ``RQI``, ``FZI``, ``BVW``, and
reservoir flags. The returned workflow result refers to the same in-memory
well, so callers can inspect, plot, or export it after execution.

.. code-block:: python

   import numpy as np
   from petrophys.workflow import DynamicWorkflowConfig, run_dynamic_workflow

   result = run_dynamic_workflow(
       well,
       DynamicWorkflowConfig(rw=0.07, rw_meas_temp_f=230),
   )

   phie = result.well.get("PHIE")
   print(np.count_nonzero(np.isfinite(phie.data)), "valid PHIE samples")

Bundled samples
---------------

The installed package includes small public LAS samples and input templates,
listed by :func:`petrophys.list_samples` and found with :func:`petrophys.sample`;
this includes the Daydream CMR logs ``daydream1_cmr`` and
``daydream1_cmr_shallow``. The large native Daydream DLIS files are not
installed: they are Git LFS files in the source repository, found with
:func:`petrophys.datasets.integration_fixture` in a checkout.

.. code-block:: python

   import petrophys

   print(petrophys.list_samples())
   path = petrophys.sample("daydream1_cmr")
