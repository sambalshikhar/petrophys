# Epslog: 15/12-20 S core-scanner data

Eight LAS files from the user-provided EpsLog delivery for Stratum Reservoir,
project P001W, cores C01/C02, plus a matching routine-core table from the
RealPore Norway compilation. The files are available through `petrophys.sample`
and `petrophys.list_samples("epslog")` in source checkouts and Python packages.

The LAS contents are unchanged; filenames are normalized. See
`source_manifest.json` for original filenames, SHA-256 hashes and the core-table
extraction provenance. The original high-resolution photographs, large PDF
reports, RGB curves, facies labels and alternative AI classification are not
needed for the XRF, grain-size or strength/sonic exercises and are not copied.

These are third-party measurements and derived products, not MIT-licensed
software. No standalone redistribution licence was supplied with this local
delivery. See `THIRD_PARTY_DATA.md` for attribution and the recorded terms.

## Inventory

All LAS depth indices are in metres. Counts below are finite measurements,
not the total number of rows on the depth grid.

| Sample name (without extension) | Contents | Valid C01 / C02 counts |
|---|---|---|
| `epslog_15_12_20_s_c01_xrf`, `epslog_15_12_20_s_c02_xrf` | 40 concentration channels including LE, 40 error channels, 6 ratios, 10 oxide equivalents | 582 / 1,117 depth samples |
| `epslog_15_12_20_s_c01_grain_segmentation`, `epslog_15_12_20_s_c02_grain_segmentation` | Area-weighted grain diameters from image segmentation: P20/P50/P80 in micrometres | 1,527 / 2,836 P50 estimates |
| `epslog_15_12_20_s_c01_grain_laser`, `epslog_15_12_20_s_c02_grain_laser` | Laser-derived grain-size proxy, `E_2DLAS_P50MSTD`, in micrometres | 1,673 / 3,174 estimates |
| `epslog_15_12_20_s_c01_strength_sonic`, `epslog_15_12_20_s_c02_strength_sonic` | Scanner strength at several scales, core ultrasonic Vp/Vs, dynamic Poisson ratio and confidence curves | 1,532 / 2,845 strength values at 1 cm; 277 / 200 paired Vp/Vs rows at 5 cm |
| `epslog_15_12_20_s_core` | CSV: RealPore same-well routine-core records, with source Excel rows | 356 records, 345 porosity, 243 horizontal and 66 vertical Klinkenberg permeability values, 306 grain-density values |

XRF covers 3815.02-3841.17 m and 3843.11-3897.74 m. The routine-core table
covers 3815.00-3897.75 m. The strength/sonic grids include padded intervals;
grid coverage does not mean all curves have valid measurements there.

## Loading the scanner files

```python
import petrophys as pp
from petrophys.interpretation.geochemistry import XRFData
from petrophys.interpretation.core_analysis import GrainSizeData, CoreMechanical
from petrophys.visualization import create_xrf_panel

print(pp.list_samples("epslog"))
prefix = "epslog_15_12_20_s_c01"  # use c02 for the second core

# lasio uppercases the delivered *_Er mnemonics to *_ER.
xrf = XRFData.from_las(pp.sample(prefix + "_xrf"), error_suffix="_ER")
ratios = xrf.elemental_ratios()
proxies = xrf.mineral_proxies()
fig = create_xrf_panel(xrf)

grain = GrainSizeData.from_las(
    pp.sample(prefix + "_grain_segmentation"),
    d50_mnemonic="P50GSDDML", d13_mnemonic=None, d87_mnemonic=None,
)
classes = grain.wentworth()

mechanics = CoreMechanical.from_las(
    pp.sample(prefix + "_strength_sonic"),
    ucs_mnemonic="E_STR001", vp_mnemonic="E_UVELP005",
    vs_mnemonic="E_UVELS005", poisson_mnemonic="E_PR005",
)
dynamic_nu = mechanics.computed_poisson()
```

The `ucs_mnemonic` argument places the delivered scanner-strength curve into
the existing reader; it does not establish that these are direct plug UCS
tests. The supplied Poisson ratio is dynamic. No static Young's modulus,
static Poisson ratio, or independent plug-test protocol is supplied.

XRF caveats:

- Use `error_suffix="_ER"`: the default `_ERR` would treat error channels as
  concentration channels. The remaining dictionary also contains delivered
  ratios and oxide equivalents; it is not a list of 56 measured elements.
- Concentrations and errors are declared as percent in the LAS, including
  Rb. Convert percent to ppm by multiplying by 10,000 when needed for a ppm
  display. Source report plots may use different display units.
- A zero can mean below detection limit. Mineral proxies are relative indices;
  oxide equivalents are not XRD mineral fractions. No XRD dataset is included.

Grain and mechanics caveats:

- P20/P80 are not P13/P87. Do not substitute them into the reader's sorting or
  skewness formulas. Both segmentation and laser curves are estimates, not
  sieve measurements. The laser description calls its result a mean grain-size
  estimate with a shale correction/scaling factor of 0.700, despite the P50
  mnemonic. The segmentation filename says V7 but its Other header says V5;
  the original metadata are preserved without resolving that version mismatch.
- Repeated `E_CONF005` mnemonics become `E_CONF005:1`, `:2`, `:3` in lasio;
  select the strength/P/S confidence series by its description. Sparse sonic
  values and nonphysical velocity combinations need QC; some computed Poisson
  ratios are outside the delivered valid range.
- Dynamic E/G/K additionally need bulk density. RealPore grain density cannot
  replace bulk density. Conventional wireline logs for this well were not found
  in the local collections inspected during this import.

## Matching routine core

```python
import pandas as pd
import petrophys as pp

core = pd.read_csv(pp.sample("epslog_15_12_20_s_core"))
core1 = core.loc[core["core"] == "CORE 1"].copy()
```

The CSV preserves the reported numeric values and missing entries. Column
`excel_row` is one-based including the header, and `source_file` identifies
`15_12-20_S__ROCK_AND_CORE__CORE_CONV_RAW_1.SPWLA`. The manifest records the
source workbook SHA-256, sheet and zero-based column mapping.

`phi_reported` is the workbook's "porosity best of available" column (2.265-29.6);
the verification example interprets it as percent and divides by 100 for the
Kozeny-Carman API. `kh_klinkenberg_reported` and `kv_klinkenberg_reported` retain
the original values because the selected workbook headers do not state their
units. Confirm the underlying SPWLA units before quantitative calibration.
`grain_density_gcc` comes from the explicitly labelled grain-density column.

There are 354 distinct depths among 356 records; repeated depths are preserved.
Joining within the same core to the nearest finite image P50 within 0.05 m
produces 214 complete P50/porosity/horizontal-permeability records. This is
nominal depth matching only: no independently verified depth correction or
gap interpolation is applied. The subset has not established permeability
prediction accuracy.

## Verification and tutorial scope

From the repository root:

```sh
python scripts/verify_epslog_data.py --report outputs/epslog-verification.json
```

Add `--source-dir PATH` to compare all eight LAS files with the original
Epslog folder. The verifier checks source hashes, sample discovery, reader
mappings, grain percentiles, dynamic Poisson calculations and nominal core
matches. Tutorial 12 now points to these inputs; the full notebook exercises
still need implementation and the calibration checks described above.

These files add no NMR T2 distributions, capillary-pressure curves, formation
pressure measurements, static elastic moduli or wireline logs.
