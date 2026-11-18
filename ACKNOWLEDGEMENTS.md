# Acknowledgements

Petrophys acknowledges the researchers, data providers and open-source
maintainers whose work supports the library and its examples.

## Methods and reference workflows

E. Craig Phillips ([Philliec459](https://github.com/Philliec459)) provides the
NMR, core and capillary-pressure reference examples and the COSTA carbonate
calibration approach used by Petrophys. The source repositories, pinned commits,
retained MIT notices and dataset limitations are recorded in
[THIRD_PARTY_DATA.md](https://github.com/sambalshikhar/petrophys/blob/main/THIRD_PARTY_DATA.md) and the collection manifests.
The COSTA dataset creators retain their separate attribution below.

Scientific methods are attributed in the relevant module docstrings and
tutorials. Cite the applicable method papers as well as Petrophys when using
those calculations in research. A software citation does not replace the
references for the equations or the data.

## Data providers

- Jorge Costa Gomes, Sebastian Geiger and Daniel Arnold, Heriot-Watt
  University: [COSTA model](https://doi.org/10.17861/6e36e28d-50d9-4e31-9790-18db4bce6e5d).
- GEOLINK-S2 and Lukas Mosser:
  [public GeoLink distribution](https://github.com/LukasMosser/geolink_dataset).
- QGC Pty Limited and the State of Queensland, Geological Survey of Queensland:
  Daydream 1 well-log deliveries.
- Kansas Geological Survey: Morgan No. 1-17 well logs and formation information.
- The Taranaki Basin Curated Well Logs dataset creators at IBM Research, with
  source data from New Zealand Petroleum & Minerals and GNS Science/Petlab:
  [dataset citation](https://doi.org/10.5281/zenodo.3832955).
- Norsk Hydro and the Norwegian Offshore Directorate: source reports for the
  North Sea formation-pressure examples.
- IODP Expedition 319 scientists, JAMSTEC/CDEX and the logging-data providers:
  C0009A pretest examples, [expedition proceedings](https://doi.org/10.2204/iodp.proc.319.2010).
- RMOTC/U.S. Department of Energy and the named log-acquisition providers:
  Teapot Dome examples.
- NLOG.NL, EBN and EpsLog: EDE-01 core-scanner products and matching
  composite well logs downloaded through the NLOG map viewer.
  Provenance and the terms of the separate data sources are recorded in
  [THIRD_PARTY_DATA.md](https://github.com/sambalshikhar/petrophys/blob/main/THIRD_PARTY_DATA.md).

## Software dependencies

Petrophys builds on NumPy, pandas, SciPy, Matplotlib, lasio, openpyxl and
defusedxml, with optional support from dlisio, scikit-learn and the MCP SDK.
Documentation and tutorials use Sphinx, Furo, MyST-NB and Jupyter tooling.
These projects retain their own licences and attribution requirements.

Acknowledgement is not a grant of redistribution rights or a claim of
endorsement. Institutional names and logos remain subject to their owners'
rights. Consult [THIRD_PARTY_DATA.md](https://github.com/sambalshikhar/petrophys/blob/main/THIRD_PARTY_DATA.md) for dataset terms,
modifications and provenance; the Petrophys software MIT licence does not
relicense third-party material.
