# Third-party data and model notices

The MIT licence in `LICENSE` applies to Petrophys software. It does not replace
the terms of third-party well data or grant rights that their suppliers do not
grant. Preserve source attributions when redistributing data and derived files.
Source records below were checked on 15 September 2026, with the Phillips and
COSTA source-copy audit added on 1 October 2026; none implies supplier endorsement
of Petrophys or its interpretations.

## COSTA

Files: `src/petrophys/datasets/costa/*.las`.

Creators: Jorge Costa Gomes, Sebastian Geiger, and Daniel Arnold (2022).
Publisher: Heriot-Watt University. Dataset: *COSTA MODEL: Hierarchical carbonate
reservoir benchmarking case study*,
[DOI 10.17861/6e36e28d-50d9-4e31-9790-18db4bce6e5d](https://doi.org/10.17861/6e36e28d-50d9-4e31-9790-18db4bce6e5d).
The [publisher's record](https://researchportal.hw.ac.uk/en/datasets/costa-model-hierarchical-carbonate-reservoir-benchmarking-case-st/)
specifies **CC BY**, without a licence version in the displayed dataset entry.

Petrophys distributes a 17-well subset with selected/renamed curves and adapted
LAS headers. These are benchmark inputs, not independently acquired field truth.
Each file covers only the Kharaib-2 reservoir interval; `formations_costa.csv`
records that interval (first to last logged sample) for every well.
For publications, also cite Costa Gomes, Geiger, and Arnold, *The Design of an
Open-Source Carbonate Reservoir Model*, Petroleum Geoscience 28 (2022),
[DOI 10.1144/petgeo2021-067](https://doi.org/10.1144/petgeo2021-067).

The immediate source for all 17 LAS copies is E. Craig Phillips's
[COSTA workflow repository](https://github.com/Philliec459/Jupyter-Notebooks_for-Characterization-of-a-New-Open-Source-Carbonate-Reservoir-Benchmarking-Case-St),
commit `b2bf2df1e6a6b24c9187e3a982b975790a49108b`.
Retained values and null masks match his source files exactly; column selection,
renaming and header adaptations are documented in the dataset READMEs and
`datasets/phillips/source_manifest.json`. His repository MIT notice is retained
alongside the new source copies. Preserve the underlying COSTA attribution too.

## Craig Phillips reference data

Files: `src/petrophys/datasets/phillips/`.

Six data files are copied without content changes from E. Craig Phillips
([Philliec459](https://github.com/Philliec459)), under descriptive filenames:

- [NMR bin example](https://github.com/Philliec459/NMR-Echo-Train-Inversion-to-created-a-typical-NMR-log):
  `nmr.csv`, with the eight-bin axis documented by the companion program.
- [Paired CMR and core data](https://github.com/Philliec459/Estimate-Core-Permeability-from-NMR-data-using-either-Map-Inversion-or-kNN):
  `CMR.csv` and `RSWC_CMR.csv`.
- [Combined NMR and conventional logs](https://github.com/Philliec459/NEW-Shaley-Sand-Log-Analysis-Tutorial-using-Combined-NMR-and-Conventional-Logs):
  `GulfCoast_NMR.xlsx`.
- The COSTA workflow repository above: `Costa_core_data.xlsx` and
  `Thomeer_results_final.xlsx`; underlying COSTA attribution remains applicable.

Each source repository publishes an MIT notice. Copies of the original 2020 and
2022 notices are included as `LICENSE-Craig-2020.txt` and
`LICENSE-Craig-2022.txt`, including in Python distributions. Exact source commits,
paths, hashes and modifications are in `source_manifest.json`.
Petrophys does not claim authorship of these data or supplier endorsement.

The eight-bin example's original well/acquisition are not identified upstream.
Its companion program synthesizes echoes from bins; these must not be presented
as acquired raw echo data. The separate CMR/core example is not a calibration
for that bin example or Daydream. COSTA pressure/fluid metadata and plug-to-well
mapping are incomplete in the selected tables. See the collection README before
using these examples as field validation.

## Epslog and matching RealPore core measurements

Files: `src/petrophys/datasets/epslog/`.

The user supplied an EpsLog delivery for client **Stratum Reservoir**, project
**P001W**, well **15/12-20 S**, cores C01 and C02 (April 2024). Eight selected
LAS files retain their original contents and supplier headers. Only filenames
are normalized. Source filenames and SHA-256 hashes are in
`epslog/source_manifest.json`; the original photographs and large PDF reports
are not copied.

The accompanying 356-row routine-core CSV is a selected-column extract from
*RealPore Por Perm Lithology data 1240 Wells Norway public.xlsx*, sheet
*Combined all wells*. Its records identify the underlying source as
`15_12-20_S__ROCK_AND_CORE__CORE_CONV_RAW_1.SPWLA`. Excel row numbers, column
mapping and the workbook hash are retained. This table is from the RealPore
compilation, not an EpsLog measurement or a Craig Phillips dataset.

No standalone redistribution licence was found in the supplied Epslog delivery
or recorded for this workbook extract. Inclusion records the user's supplied
data and authorization to copy them into this checkout; it does not establish
a public-data licence. Suppliers retain their rights, and the Petrophys MIT
software licence does not relicense these measurements.

## Formation pressure: 35/11-13 and 35/11-10

Files: `src/petrophys/datasets/formation_pressure/`.

Originator: **Norsk Hydro**. Source: *Final Well Report, well 35/11-13*,
NH/OD-B-4522/05 Rev 1, dated 13 December 2005; Norwegian Offshore Directorate
wellbore ID **5063**. The user supplied the local North Sea Reports copy.
The collection retains four evidence pages (original PDF pages 44, 45, 54, 55)
and a rendering of the pressure plot, with source hashes and page references.

The CSV/LAS derivatives transcribe Table 3.7.4; interpreted fluid labels and
fit-selection flags are Petrophys additions identified separately from the
measurements. These data and report pages are not relicensed as MIT software.
The supplied copy does not provide a separate redistribution licence. No claim
of supplier endorsement or independent validation of the reported contact is made.

The second source is Norsk Hydro's *Final Well Report, well 35/11-10*, dated
12 December 1997 (wellbore ID **3042**), supplied in the same local collection.
Original PDF pages 34, 35 and 38 are retained. The Fensfjord CSV/LAS files contain
ten measured station markers digitized from Figure 8.2, not values sampled from
the drawn gradient lines. Fluid labels, pixel calibration, rounding and the
subset selection are documented in `35_11_10_source_manifest.json`. These
derivatives have the same attribution and licence qualification stated above.

## JAMSTEC C0020A native MDT pretests

Files: `src/petrophys/datasets/formation_pressure/c0020a_mdt_*`.
Originator: **Schlumberger**, acquired for **JAMSTEC**, IODP Expedition 337,
well C0020A, Run 4 XLD probe, September 2012. Public custodian: JAMSTEC's
[Sio7 logging archive](https://www.jamstec.go.jp/sio7/well-logging/337/C0020A/).

The 35 LAS traces retain original bytes and source headers; filenames alone
were shortened. The associated pretest report is unchanged. The reference CSV
is a Petrophys transcription of per-test formation pressures/statuses, with
null formation pressure for failed tests. Exact source URLs, original filenames,
report pages and SHA-256 checksums are recorded in the collection manifest.
These data are used for scientific/educational reproducibility. JAMSTEC
[identifies IODP policy as governing IODP cruise data](https://www.jamstec.go.jp/mare3/e/information/sampledata.html).
No separate software-style licence is embedded in the delivery; it is not
relicensed as MIT code, and no supplier endorsement of our picker is claimed.

## GeoLink

Files: `src/petrophys/datasets/geolink/*.las` and the associated 30/6-5
formation/lithology CSV tables. Data attribution: **GEOLINK-S2**.

The public [Lukas Mosser GeoLink distribution](https://github.com/LukasMosser/geolink_dataset)
identifies its data and analysis as
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
This attribution must accompany GeoLink samples and adaptations; they must not
be described as MIT-licensed data. The Petrophys examples contain curated LAS
headers, selected curves, and interpreted/core-related additions. Those additions
are not all attributable to the original GeoLink release.

The exact upstream archive and permission for those added fields have not been
recorded in this repository. Before a public package release, the maintainer must
confirm that the actual distributed copies and their additions are covered by
the cited licence or another documented permission. A licence on a related
public distribution is not proof of provenance for every local copy.

## Daydream 1

Files: `src/petrophys/datasets/daydream/` (bundled LAS samples and repository-only
LAS/DLIS fixtures). Originator: **QGC Pty Limited**. Public custodian:
**State of Queensland**, Geological Survey of Queensland.

Source: *ATP645 QGC PTY LIMITED DAYDREAM 1 WELL COMPLETION REPORT*, **CR078710**.
The [official metadata](https://geoscience.data.qld.gov.au/api/3/action/package_show?id=cr078710)
specifies `license_id: cc-by`, Creative Commons Attribution, and links to
<http://www.opendefinition.org/licenses/cc-by>. The record does not specify a
version number; no version is inferred here.

Scalar LAS fixtures are curated selections with converted units and aligned
depth grids. Native DLIS files retain the acquisition/processed delivery arrays.
File hashes and curve-selection details are in the collection README. Vendor
mnemonics identify supplied measurements and products, not Petrophys algorithms.

## Kansas Geological Survey

Files: `src/petrophys/datasets/kgs/1046123714.*` and the accompanying tops CSV.
Well: Morgan No. 1-17, Rawlins County, Kansas, KGS identifier 1046123714.
The [KGS terms of use](https://www.kgs.ku.edu/General/copyright.html) permit reuse
and modification with retained notices and this attribution:

> The source of this material is the Kansas Geological Survey website at http://www.kgs.ku.edu/. All Rights Reserved.

KGS materials are supplied without warranty. The companion metadata has
conflicting source/basin fields; see the collection README. Confirm provenance
of the companion metadata and formation picks before redistributing them as
KGS-authored material.

## Pretrained imputation model

File: `src/petrophys/preprocessing/imputation/pretrained/geolink_curve_imputation.joblib`.
This is a Petrophys-trained model derived from a 205-well GeoLink collection,
not a model supplied or endorsed by GEOLINK-S2. `validation.json` records the
training digest, dependencies, held-out-well metrics, and retained targets.

The model is not relicensed here. Its release requires confirmation of the
training-data provenance and any applicable derived-data/share-alike obligations
described under GeoLink above. The training code is covered by the software MIT
licence; the code licence alone does not establish rights to the model artifact.
