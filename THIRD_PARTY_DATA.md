# Third-party data and model notices

The MIT licence in `LICENSE` applies to Petrophys software. It does not replace
the terms of third-party well data or grant rights that their suppliers do not
grant. Preserve source attributions when redistributing data and derived files.
Source records below were checked on 15 September 2026; none implies supplier
endorsement of Petrophys or its interpretations.

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
For publications, also cite Costa Gomes, Geiger, and Arnold, *The Design of an
Open-Source Carbonate Reservoir Model*, Petroleum Geoscience 28 (2022),
[DOI 10.1144/petgeo2021-067](https://doi.org/10.1144/petgeo2021-067).

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
