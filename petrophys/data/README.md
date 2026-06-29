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
| `geolink/30_6-5.las` | North Sea **clastic** well (NCS 30/6-5). Rich curve set (GR, RHOB, NPHI, PEF, DTC, multi-spacing resistivity, SP, CALI) **plus a vendor `LITHOLOGY_GEOLINK` log**. The main tutorial well. |
| `geolink/16_7-1.las` | North Sea well with tight zones where deep resistivity goes **off-scale** (~20% of samples > 2000 ohm·m) — used to demo industrial **curve wrapping**. |
| `nmr.csv` | NMR data (depth, MPHI, 8 T2 bins, MFFI, MBVI). |
| `formations_30_6-5.csv` | Formation tops for 30/6-5 (formation_name, top, bottom, lithology). |
| `lithology_30_6-5.csv` | Per-interval lithology for 30/6-5 (top, bottom, lithology). |

## Notes

- The Costa wells are **carbonate**; the dynamic workflow auto-detects this from
  core porosity + apparent matrix density when `lithology="auto"` / `matrix_type="auto"`.
- A much larger geolink LAS collection (200+ wells) is used in development but is
  **not** bundled (too large); it lives outside the package and is gitignored.
