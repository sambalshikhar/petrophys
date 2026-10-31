# Daydream 1 datasets

Source: QGC Pty Limited, Daydream 1 well completion report CR078710, distributed
by the State of Queensland. The official catalogue identifies Creative Commons
Attribution. See `THIRD_PARTY_DATA.md` in the distribution for the source record
and licence details.

All retained Daydream 1 files live in this collection folder. The two processed
CMR summary LAS files are small package samples. The native DLIS files and
paired scalar LAS validation files use Git LFS and are excluded from wheels
and source distributions; `petrophys.datasets.integration_fixture` downloads
them on first use and checks them against the SHA-256 values below.

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
| `daydream1_shallow_aps_pex_hngs.las` | `Daydream 1_Suite 1_Run 2_APS-PEX-HNGS.las` | 30,031,316 | `27c96207fec5296aa56dd2222ae1511412bd6fe7d5731eeab843e27dd86e972d` |

Only processed or scalar LAS products are retained: each CMR LAS has 15 curves
and the shallow conventional LAS has 28 curves. None contains flattened detector
or time-gate arrays. The original 395,884,528-byte array-expanded deep LAS and
the larger raw FMI acquisition file are not retained.
