# Craig Phillips reference datasets

Six small source datasets from E. Craig Phillips
([Philliec459](https://github.com/Philliec459)), retained byte-for-byte under
descriptive filenames. These complement the Daydream summaries and COSTA wells.
They ship with the package and are available through `petrophys.sample()`.

`source_manifest.json` pins the upstream repository, commit, original path,
Git blob hash and SHA-256 of each copy. The two upstream MIT notices are retained
as `LICENSE-Craig-2020.txt` and `LICENSE-Craig-2022.txt`. Underlying COSTA data
also retain the creators' attribution and terms in `THIRD_PARTY_DATA.md`.
Repository licensing does not establish missing acquisition provenance.

## Inventory and scope

| Installed sample | Original filename | Contents | Supported use and limits |
|---|---|---|---|
| `phillips_nmr_bins` | `nmr.csv` | 51 depths, P1-P8, MPHI, MFFI, MBVI | Eight-bin distribution, T2 log-mean and cutoff sensitivity. Source well/acquisition are unspecified; this is an upstream teaching example, not a verified Daydream or independently validated field delivery. |
| `phillips_cmr_log` | `CMR.csv` | 573 depths, CMRP_3MS, CMFF, BVI | Scalar NMR log paired with the following core table. No T2 bins or documented cutoff. |
| `phillips_cmr_core` | `RSWC_CMR.csv` | 56 rows with CMR summaries, Kair, Cpor and depth | Core/NMR permeability calibration in this source example. These are not core measurements for the 51-row bin example or Daydream. |
| `phillips_gulfcoast_nmr` | `GulfCoast_NMR.xlsx` | 2,001 rows and 15 conventional/NMR columns on Sheet1 | Combined shaly-sand inputs. MPHI/MBVI are scalar curves, not bins. Preserve and mask the -999.25 null sentinel. |
| `costa_core_pc` | `data/Costa_core_data.xlsx` | 11,055 rows, 110 sample IDs, PcHg, PcOB, Sw, porosity and permeability | Craig's processed COSTA HPMI table. No plug-to-well/depth mapping, FWL or complete fluid/pressure-unit metadata is supplied in this workbook. |
| `costa_thomeer_fits` | `data/Thomeer_results_final.xlsx` | 109 sample IDs and fitted Thomeer parameters | Reference fitted results, not independent measurements. Sample 104 is absent; join by Sample rather than row position. |

## Load the T2-bin example

```python
import numpy as np
import petrophys as pp
from petrophys.interpretation.nmr import NMRData

nmr = NMRData.from_csv(
    pp.sample("phillips_nmr_bins"),
    t2_centers=np.array([4, 8, 16, 32, 64, 128, 256, 512], dtype=float),
    t2_cutoff=32.0,
)
t2_ms, bin_porosity_pu = nmr.t2_distribution(0)
t2lm_ms = nmr.t2_lm()
bvi_pu = nmr.bvi_from_bins(cutoff_ms=32.0, partial_bins=False)
ffi_pu = nmr.ffi_from_bins(cutoff_ms=32.0, partial_bins=False)
```

The eight times come from the exponential terms in Craig's
[`curve_fit_echo_train11_works.py`](https://github.com/Philliec459/NMR-Echo-Train-Inversion-to-created-a-typical-NMR-log/blob/d7d1b6810427a0b007e6651b5cb15539ec6aab2f/curve_fit_echo_train11_works.py#L40).
The CSV itself does not store an axis, units or a cutoff. Its amplitudes use the
porosity-unit convention (pu). P1-P3 sum to the supplied MBVI and P4-P8 to MFFI
within 0.002 pu rounding. Thus a sharp split **below 32 / at-or-above 32 ms**
reproduces its supplied summaries; ask for it with `partial_bins=False`, because
by default the bin that straddles the cutoff is split by its log-T2 width. The upstream plot title says 33 ms; using
33 as a strict cutoff on these discrete centres instead moves all of P4 to
bound fluid. Do not equate that title with a cutoff curve in the file.

The upstream program generates noisy echo trains *from* the supplied bins.
Those generated echoes are not acquired raw measurements. The source does not
identify how the original CSV bins were obtained. The file is suitable for
demonstrating array handling and checking arithmetic, without asserting that
it closes the need for a traceable field T2 delivery.

Changing the cutoff in `bvi_from_bins`/`ffi_from_bins` returns new calculations;
it does not overwrite the supplied MFFI/MBVI. Methods that use the stored
partition continue to use it. For a recalculated Coates result, construct a new
`NMRData` with the recalculated partition rather than relabelling the old one.

## Read the core and capillary-pressure tables

```python
import pandas as pd
import petrophys as pp

core = pd.read_csv(pp.sample("phillips_cmr_core"))
pc = pd.read_excel(pp.sample("costa_core_pc"), sheet_name="Sheet1")
plug = pc.loc[pc["Sample"] == 50, ["PcHg", "Sw", "Porosity", "Perm"]]
fits = pd.read_excel(pp.sample("costa_thomeer_fits"))
```

The CMR/core CSV porosities are fractional: multiply by 100 before supplying
them to `NMRData.from_dataframe`, which expects pu and does not infer CSV units.
The Gulf Coast workbook's MPHI/MBVI are also fractional; mask -999.25 before
converting them to pu.
`CMRP_3MS` excludes the short-T2 contribution and must not be described as
unqualified total porosity. Kair is the source's air-permeability reference,
not Klinkenberg-corrected permeability. Neither CSV documents a depth datum or
unit, so compare their common depth coordinate without assuming a well tie.

In the COSTA table porosity is in percent and Sw is fractional. Pressure-unit
and fluid-system details must be established before treating PcHg/PcOB as inputs
to reservoir-height calculations. In particular, do not assume PcOB is an
independent laboratory measurement or that the table supplies a measured FWL.
The fitted table is a separate processed product; its parameters and sample
metadata need not equal the source measurements exactly.

## Verification and selections

Run from a source checkout:

```bash
python scripts/verify_phillips_data.py --report outputs/phillips-verification.json
```

This checks file hashes, bin/summary closure, T2 arithmetic, the core/NMR pair,
all capillary-pressure sample curves, and the retained numerical values in all
17 existing COSTA LAS files. An optional `--costa-source-dir` pointing to Craig's
checkout repeats the numerical comparison against the original LAS files.

The existing COSTA well copies are already present in `datasets/costa`; the
manifest records their exact numerical correspondence, so they are not
duplicated here. `PHIE_CC` corresponds to upstream `PHIE`, `SW_INPUT` to `SW`,
and `PERM_CC` to `CORE_PERM`. Upstream `SO` and `CORE_POR` were omitted. This
mapping records file provenance; it does not establish that upstream PHIE is
an independent core measurement.

`Costa_core_data2.xlsx` is a second processed table and is not needed alongside
the fuller 110-sample table. Rosetta Stone tables describe another reference
collection and are not required for the present COSTA example. Model weights,
chartbook digitizations, full Geolog projects and Craig's application code are
not runtime dependencies of these data consumers and were not copied.

The scoped review does not establish sources for XRF/XRD, quantitative grain-size
or mechanical measurements. These files also do not establish a measured FWL
or a core-calibrated T2 cutoff for Daydream.
