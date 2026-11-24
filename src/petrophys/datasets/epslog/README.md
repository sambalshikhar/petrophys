# EpsLog: EDE-01 core scanner and matching well logs

Source: **NLOG.NL**, downloaded per well through the
[NLOG map viewer](https://www.nlog.nl/nlog-mapviewer/) by the project maintainer.
Credit **NLOG.NL, EBN, EpsLog** and the acquisition providers named in the
original headers. The scanner delivery identifies EBN project P005W, EDE 01.
[NLOG reuse terms](https://www.nlog.nl/en/disclaimer) permit copying,
distribution and processing of website information with NLOG.NL attribution.
These data are separate from the software's MIT licence.

Two LAS files are bundled. Contents are unchanged; filenames are normalized.
`source_manifest.json` records original filenames and SHA-256 hashes.
Tutorial 12 executes the examples without external downloads.

| Sample | Contents | Coverage |
| --- | --- | --- |
| `epslog_ede_01_scanner` | XRF concentrations, merged grain-size P50, scanner strength, probe permeability, colour indices and clustering | 7,607 rows, 1031.01-1217.79 m |
| `ede_01_composite` | GR, RHOB, resistivity, DTC/DTS, cased-hole neutron and spectral gamma, caliper, bit size and acquisition flags | 17,162 rows, approximately 8.26-1315.92 m |

Scanner finite counts: 6,848 per concentration channel; 4,512 merged P50;
4,477 probe permeability; 6,263 strength. Missing measurements are preserved.

## Load and interpret

```python
import lasio
import numpy as np
import petrophys as pp
from petrophys.interpretation.geochemistry import XRFData
from petrophys.interpretation.core_analysis import GrainSizeData

path = pp.sample("epslog_ede_01_scanner")
las = lasio.read(path)
symbols = ["Al", "Ca", "Fe", "K", "Mn", "Rb", "S", "Si", "Ti"]
elements = {}
for symbol in symbols:
    curve = las.curves[f"{symbol.upper()} CONCENTRATION"]
    values = np.asarray(curve.data, dtype=float).copy()
    if curve.unit == "ppm":
        values /= 10000.0
    elif curve.unit != "%":
        raise ValueError(f"Unexpected XRF unit: {curve.unit}")
    elements[symbol] = values
xrf = XRFData(depth=las.index, elements=elements,
              units={symbol: "%" for symbol in symbols})
ratios = xrf.elemental_ratios()
proxies = xrf.mineral_proxies()
grain = GrainSizeData.from_las(path, d50_mnemonic="GS_P50___MERGED3",
                              d13_mnemonic=None, d87_mnemonic=None)
classes = grain.wentworth()
well = pp.read(pp.sample("ede_01_composite"))
```

Select XRF channels explicitly rather than passing the mixed scanner file
directly to `XRFData.from_las`, which treats every non-error curve as an element.
LE is an aggregate and is excluded. Convert Rb/Ti from ppm to percent before
calculating mass ratios. Proxies are relative indicators, not XRD fractions.
Mg is absent, so Mg-dependent ratios remain missing.

## Measurement limits

- Probe permeability is in mD, not laboratory plug Klinkenberg permeability.
- Merged P50 is in micrometres and combines laser and segmentation products.
  The delivered description includes `if(x1<2600;x1;x0*6/5)`.
  P13/P87 are absent; do not invent sorting percentiles.
- No measured core porosity is supplied, so a measured-porosity Kozeny-Carman
  calibration is not demonstrated. GeoLink plug calibration remains in Tutorial 12.
- Strength is a scanner product in MPa, not an independently documented plug
  UCS test. The report lists core VP/VS, but these are absent from the scanner
  LAS. Static Young's modulus and static Poisson ratio are absent.
- Composite RHOB is g/cm3, DTC/DTS are us/ft. Use the library's
  `elastic_properties` for dynamic well-log properties.
  Missing DTS must not be replaced with invented shear measurements.
- `NPHI_CH` and spectral curves suffixed `_CH` are cased-hole products.
  Preserve `CH_FLAG-DTC` and `CH_FLAG-DTS`; a missing flag does not establish
  open-hole acquisition. Sonic headers warn about washout effects on DTC/DTS.
- Depth indices are metres. Overlap does not establish core-to-log alignment;
  verify a depth tie before pointwise calibration or merging curves.
- Delivered cluster labels are processed classifications, not geological truth.

Large waveform DLIS files, the empty waveform LAS, density-image export,
duplicate grain-size export and alternate clustering files are not needed by
this tutorial. Reference reports are recorded by filename in the manifest;
figures are produced from LAS measurements rather than copied from PDFs.
