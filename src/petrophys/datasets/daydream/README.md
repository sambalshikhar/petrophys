# Daydream 1 datasets

Source: QGC Pty Limited, Daydream 1 well completion report CR078710, distributed
by the State of Queensland. The official catalogue identifies Creative Commons
Attribution. See `THIRD_PARTY_DATA.md` in the distribution for the source record
and licence details.

All retained Daydream 1 files live in this collection folder. The two processed
CMR summary LAS files are small package samples. The native DLIS files and
paired scalar LAS validation files use Git LFS and are intentionally excluded
from wheels and source distributions. Run `git lfs pull` in a source checkout
before running the native-reader or FMI examples.

The repository fixtures are not runtime dependencies. Tutorials that use the
native deliveries require the downloaded files. The DLIS files preserve multi-frame and processed-image
coverage that cannot be represented faithfully by scalar LAS alone.

The files are public Queensland well-completion data for Daydream 1, ATP 645.
The repository filenames are normalized for portability; the original delivery
filenames and checksums are recorded below.

| Repository file | Original delivery file | Bytes | SHA-256 |
|---|---|---:|---|
| `daydream1_cmr.las` | `Daydream 1_Suite 2_Run 3_CAL-CMR_4140-3755m.las` | 3,031,938 | `2e14b8185c268cdb57b3bc83d619fc641b1285bf64c76e47b9dfec6e1a9004ad` |
| `daydream1_cmr_shallow.las` | `Daydream 1_Suite 2_Run 3_CAL-CMR_3050-2860m.las` | 1,637,660 | `20ad5535dd47df45d7ad9acfcdaa85b114aa571df1dfb90f4ed013b52b078b37` |
| `daydream1_fmi_processed_dynamic_image.dlis` | `Daydream 1_FMI_Processed_Dynamic_Image.dlis` | 111,779,520 | `1449d86411a22299d6bed85accb1e44259c7fd02594f154fcf0dd7598ba27b75` |
| `daydream1_deep_pex_aps_ppc_ecs.dlis` | `QGC_Daydream-1_Dual-PEX-APS-PPC-ECS.dlis` | 172,587,816 | `4e1943402d5abe7ce81a10bdefd8cd8c0a31c2a2b2bc22a0ef91997b3671af86` |
| `daydream1_deep_curated_scalar_logs.las` | Curated from `QGC_Daydream-1_Dual-PEX-APS-PPC-HNGS-ECS_Cust.las` | 7,317,316 | `566ec08e6c8cd63c2617496cb65f01632cd40053e2b79d5ca0f436009977627c` |
| `daydream1_shallow_aps_pex_hngs.dlis` | `QGC_Daydream1_R2-APS-PEX-HNGS_1860-835_Main.dlis` | 120,929,868 | `1285f048735bde98e9f254b0c8e6d43187eb303f900de4e83f11df45dca4b1df` |
| `daydream1_shallow_aps_pex_hngs.las` | `Daydream 1_Suite 1_Run 2_APS-PEX-HNGS.las` | 30,031,316 | `27c96207fec5296aa56dd2222ae1511412bd6fe7d5731eeab843e27dd86e972d` |

Only processed or scalar LAS products are retained: each CMR LAS has 15 curves,
the shallow conventional LAS has 28 curves, and the curated deep LAS has 121
curves. None contains flattened detector or time-gate arrays. The original
395,884,528-byte array-expanded deep LAS and the larger raw FMI acquisition
file are not retained.

The curated deep LAS preserves all 8,548 measured-depth rows, including the
complete 2844.8508-4147.4136 m interval and original sampling. Its little-endian
float64 depth-vector SHA-256 is
`0dbeb678b977e1a584ec0b1721b7c5e18940d9f9b2f3bb063c806b3f33a32190`.
It retains 121 scalar conventional, trajectory, spectral-gamma, elemental,
vendor-interpretation, mineral, and QC curves from the original 2,569 columns.
Raw detector-bin families (`S1SP`, `S2SP`, `C1SP`, `C2SP`, `ESSP`, and
`EWWS_20`) and neutron time-gate arrays (`A1TD`, `A2TD`, and `AFTD`) are omitted;
the native DLIS remains the fixture for those acquisition structures.

The source LAS SHA-256 is
`cc63cc99bcdfa12af4580d7993a0f0e530b7395180bc29ec3b55ab1a07914a42`.
The derivative was verified against that source for retained values,
missing-value masks, curve order, and the exact depth vector before inclusion.
