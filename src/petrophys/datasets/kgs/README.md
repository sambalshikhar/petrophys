# KGS well sample

Source and redistribution terms are recorded in `THIRD_PARTY_DATA.md` in the
distribution, including the KGS attribution and companion-data provenance caveat.

This folder contains the Morgan No. 1-17 public well package used to exercise
legacy LAS metadata, generic formation tops, and input-aware interpretation.

| File | Contents |
|---|---|
| `1046123714.las` | LAS 2.0 log from -2.0 to 4695.5 ft at 0.5 ft sampling. |
| `1046123714_formation_tops.csv` | Eleven formation tops from 3782 to 4618 ft. |
| `1046123714_metadata.txt` | Source-supplied well metadata and provenance fields. |

The source package identifies the well as Morgan No. 1-17, KGS ID 1046123714,
in Rawlins County, Kansas. Its accompanying metadata labels the basin as the
Illinois Basin and the source as EDX/ISGS, which conflicts with the LAS header.
Those source fields are not used as interpretation constraints.

Several legacy curve headers have blank or incorrect units. In particular,
`CNLS` is described as compensated-neutron limestone porosity and `RHOC` as
density correction. The standardization layer resolves those mnemonics before
the dynamic workflow runs.
