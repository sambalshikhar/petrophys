Specialist Data
===============

NMR, borehole images, geochemistry, and laboratory measurements are all part
of the petrophysical data landscape, but their structures differ from ordinary
one-value-per-depth scalar curves. Petrophys keeps these APIs in specialist
interpretation packages while allowing their interpreted scalar products to
join the same ``Well`` used by the conventional workflow.

NMR logs
--------

``NMR`` is the generic measurement family; names such as CMR identify a
vendor tool or delivery. :class:`petrophys.interpretation.nmr.NMRData` handles
processed T2 distributions and associated scalar curves. It can calculate
logarithmic-mean T2, bulk-volume irreducible, free-fluid index, SDR
permeability, and Timur-Coates permeability.

.. code-block:: python

   import petrophys
   from petrophys.interpretation.nmr import NMRData

   nmr = NMRData.from_las(
       petrophys.sample("daydream1_cmr"),
       mphi_mnemonic="TCMR",
       mffi_mnemonic="CMFF",
   )

   swirr = nmr.swirr_nmr()                  # irreducible water saturation, v/v
   k_coates = nmr.permeability_coates()     # mD
   k_sdr = nmr.permeability_sdr()           # mD
   vendor = nmr.stored_permeability         # vendor KSDR/KTIM curves from the file

``nmr.t2_lm()`` needs T2 distribution bins or a stored T2 log-mean curve
(``t2lm_mnemonic``); this sample carries neither, so ``nmr.has_t2lm`` is False.

Use :func:`petrophys.visualization.create_nmr_panel` for an industry-style
depth panel that combines scalar NMR products and T2 distributions when those
channels are available. Petrophys does not reconstruct distributions from raw
echo trains.

Borehole images
---------------

Electrical and acoustic image deliveries commonly use multidimensional DLIS
channels, although compatible pad/flap arrays can also be represented in LAS.
The borehole-image readers retain array geometry and navigation rather than
forcing every image sample into a conventional scalar curve.

.. code-block:: python

   from petrophys.interpretation.borehole_imaging import (
       orient_to_azimuth,
       plot_fmi,
       read_fmi_dlis,
   )

   image = read_fmi_dlis("processed_image.dlis")
   oriented = orient_to_azimuth(image)
   axes = plot_fmi(oriented, equalize="dynamic")

The package provides static and dynamic equalization, azimuthal orientation,
navigation-aware depth filtering, coverage statistics, per-depth statistics,
and continuous image rendering. These are image preparation and visualization
tools, not a complete automatic fracture or structural interpretation system.

Spectral gamma and elemental spectroscopy
------------------------------------------

The dynamic workflow detects compatible spectral gamma and elemental channels
before conventional interpretation. It can derive spectral-gamma products,
standardized elemental fractions, oxide closure, mineral proxies, and
geochemical quality information. Raw detector spectra remain available as
specialist inputs; derived elemental curves should be preferred when a
validated vendor delivery already provides them.

For laboratory XRF tables, :class:`petrophys.interpretation.geochemistry.XRFData`
stores depth-indexed elemental measurements and supports ratios, normalization,
mineral proxies, lithology indicators, summary statistics, and export.
:func:`petrophys.visualization.create_xrf_panel` provides a compact depth panel.

Core and laboratory data
------------------------

:class:`petrophys.model.CoreData` and
``petrophys.interpretation.core_analysis`` support routine-core and related
laboratory workflows:

* porosity and permeability calibration;
* grain-density correction;
* porosity-permeability transforms;
* Archie cementation-exponent estimation;
* log interpolation at core depths;
* mechanical measurements and grain-size classes;
* digital facies and core-image color measurements.

Laboratory saturation is retained as calibration evidence rather than blindly
replacing the continuous log-derived saturation curve. Depth basis, recovery
conditions, fluid restoration, and SCAL methodology must be evaluated before
it is used as an interpretation constraint.

Saturation height
-----------------

``petrophys.interpretation.saturation.height`` includes capillary
pressure-height conversion, laboratory-to-reservoir conversion, Leverett J,
J-function fitting, Brooks-Corey fitting, and saturation prediction through a
log column. Use :func:`petrophys.visualization.create_saturation_height_plot`
to compare observed and modeled relationships.

Sonic saturation
----------------

``petrophys.interpretation.saturation.sonic`` provides a
resistivity-independent estimate based on Gassmann fluid substitution. The
module supports supplied compressional/shear sonic and density data, can use a
documented Vp-Vs relation when appropriate, and applies a detectability gate so
an unresolved fluid response is reported rather than presented as a confident
saturation estimate.

Optional dependencies and native fixtures
------------------------------------------

Install ``petrophys[dlis]`` for DLIS reading. Small scalar LAS samples ship in
the Python package. Larger native Daydream DLIS fixtures live in the repository
through Git LFS and are intended for integration tests and the borehole-image
tutorial; they are not part of a normal package installation.
