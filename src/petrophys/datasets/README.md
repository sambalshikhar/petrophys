# Bundled sample data

These datasets ship with `petrophys` so the tutorials, demos, and quick tries
work out of the box — after both `git clone` and `pip install`.

## How to use

```python
import petrophys
from petrophys.io import read_las

well = read_las(petrophys.sample("HW-3"))      # a carbonate well
well = read_las(petrophys.sample("30_6-5"))    # a clastic well

petrophys.list_samples()            # every bundled sample
petrophys.list_samples("costa")     # just the Costa wells
```

`petrophys.sample(name)` returns the file path (name is the file stem, extension
optional). `petrophys.list_samples(kind)` lists what's available.

## What's here

| Folder / file | Description |
|---------------|-------------|
| `costa/` (HW-3 … HW-32) | 17 wells from the **Costa** field — **carbonate** reservoir. Curves: GR, RHOB, NPHI, DT, RT, plus core PHIE_CC / SW_INPUT / PERM_CC and Archie params (a, m, n) in the header. |
| `geolink/30_6-5.las` | North Sea **clastic** well (NCS 30/6-5). Rich curve set (GR, RHOB, NPHI, PEF, DTC, multi-spacing resistivity, SP, CALI) **plus a vendor `LITHOLOGY_GEOLINK` log** and an operator `~Params` header block (RW, RMF, BHT, CLFL). The main tutorial well. |
| `geolink/16_7-1.las` | North Sea well with tight zones where deep resistivity goes **off-scale** (~20% of samples > 2000 ohm·m) — used to demo industrial **curve wrapping**. Carries a `~Params` header block (CLFL). |
| `daydream/daydream1_cmr.las` | Real Schlumberger **CMR** summary log, well **Daydream 1** (QGC, ATP 645, Bowen Basin - public Queensland well-completion data), 3755-4145 m. Curves: TCMR, CMFF, CMRP_3MS, KSDR, KTIM, T2CUTOFF, GR_EDTC, calipers. |
| `daydream/daydream1_cmr_shallow.las` | Second CMR section of the same well, 2860-3050 m. |
| `formations_30_6-5.csv` | Formation tops for 30/6-5 (formation_name, top, bottom, lithology). |
| `lithology_30_6-5.csv` | Per-interval lithology for 30/6-5 (top, bottom, lithology). |

## Notes

- The Costa wells are **carbonate**; the dynamic workflow auto-detects this from
  core porosity + apparent matrix density when `lithology="auto"` / `matrix_type="auto"`.
- A much larger geolink LAS collection (200+ wells) is used in development but is
  **not** bundled (too large); it lives outside the package and is gitignored.
