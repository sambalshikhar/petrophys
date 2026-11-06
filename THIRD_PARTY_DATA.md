# Third-party data and model notices

The MIT licence in `LICENSE` applies to Petrophys software. It does not replace
the terms of third-party well data or grant rights that their suppliers do not
grant. Preserve source attributions when redistributing data and derived files.
Source records below were checked in September and October 2026; none implies
supplier endorsement of Petrophys or its interpretations.

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
`src/petrophys/_internal/data/costa_interpretation_parameters.csv` (per-well
BHT, Rw at 68 degF and the
saturation, clay-volume and porosity methods) and
`src/petrophys/_internal/data/costa_reference_carbonate_volumes.csv` (calcite
and dolomite volumes of 15
wells from an earlier interpretation of these logs) are the remaining inputs
of the published COSTA context package, which the `costa` context profile
recomputes. Both are derived from the COSTA logs and remain subject to the
underlying COSTA attribution.
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

Files: `src/petrophys/datasets/formation_pressure/35_11_*_mdt.las`.

Originator: **Norsk Hydro**. MDT pressure stations of well 35/11-13 (run 1A,
2005; Norwegian Offshore Directorate wellbore ID **5063**) and well 35/11-10
(runs 1A-2E, June 1997; wellbore ID **3042**), as published by the Norwegian
Offshore Directorate. Each LAS holds the station depths, formation and mud
pressures, temperatures (35/11-10), mobility and test quality; no fluid
interpretation is included. No separate redistribution licence is stated for
these data, and they are not relicensed as MIT software. No claim of supplier
endorsement is made.

## Formation pressure: IODP Hole C0009A MDT pretests

Files: `src/petrophys/datasets/formation_pressure/319-C0009A_MDT_*LTP.DLIS`
and `c0009a_mdt_*.las`.

Data: Integrated Ocean Drilling Program (IODP) Expedition 319, Hole C0009A,
acquired by Schlumberger for CDEX/JAMSTEC in July 2009 and published in the
JAMSTEC CDEX well-logging archive
(<https://www.jamstec.go.jp/sio7/well-logging/319/C0009A/ORIGINAL/DATA/>).
The DLIS files are the original archive files, unchanged. Each LAS is a
Petrophys conversion of the matching DLIS (frame 600T); six channels were kept
and their values are unchanged. Cite: Saffer, D., McNeill, L., Byrne, T.,
Araki, E., Toczko, S., Eguchi, N., Takahashi, K., and the Expedition 319
Scientists, 2010. *Proc. IODP*, 319. doi:10.2204/iodp.proc.319.2010. The data
are not relicensed as MIT software.

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

## Taranaki Basin Curated Well Logs Dataset

Files: `src/petrophys/datasets/taranaki/` (Cheal-1, Tuhua-7 and their formation
intervals). Source: Taranaki Basin Curated Well Logs Dataset, IBM Developer Data
Asset eXchange, curated from the New Zealand Petroleum & Minerals Online
Exploration Database and Petlab (GNS Science). Licence: Community Data License
Agreement - Sharing 1.0, included as `LICENSE-CDLA-Sharing-1.0.txt`; these files
and data derived from them remain under that licence. The files are a modified
extract (two wells, four columns removed, written as LAS 2.0 with the dataset's
documented units, formation intervals derived); log values are unchanged. See
the collection README.

## Teapot Dome (NPR-3)

Files: `src/petrophys/datasets/teapot/` (Fenix WX #17-21 and Fenix MDX #44-10).
Source: Teapot Dome data set released by the Rocky Mountain Oilfield Testing
Center (RMOTC), U.S. Department of Energy, as non-proprietary data; logs
acquired by Schlumberger for Fenix & Scisson. The LAS files are the original
deliveries, renamed. Supplied without warranty.
