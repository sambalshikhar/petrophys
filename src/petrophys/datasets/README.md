# Repository datasets

All repository-owned well data live below this directory.

Third-party data retain their suppliers' rights. See `THIRD_PARTY_DATA.md` in
the distribution and each collection's README for sources and terms.

- Small sample datasets ship with `petrophys`, so tutorials and demos work
  after both `git clone` and `pip install`.
- Larger native deliveries and validation LAS files live beside related
  samples in the same collection folder. They use Git LFS and are excluded
  from wheels and source distributions.

## Bundled samples

```python
import petrophys
from petrophys.io import read_las

well = read_las(petrophys.sample("HW-3"))
well = read_las(petrophys.sample("30_6-5"))

petrophys.list_samples()
petrophys.list_samples("costa")
```

`petrophys.sample(name)` returns a bundled file path. Application code can pass
any user-owned path directly to `read_las()` or `read_dlis()`.

| Folder or file | Purpose |
|---|---|
| `costa/` | The 17-well Costa carbonate field used by the multiwell field-study and screening notebooks. Each file covers only the Kharaib-2 reservoir interval, not the whole well. |
| `geolink/30_6-5.las` | Main North Sea clastic tutorial well with a rich curve suite and vendor lithology. Its header RW is the NPD formation-water value, 0.070 ohm.m at 230 degF (RWT); the GeoLink delivery had RW 0.7 without a temperature. |
| `geolink/16_7-1.las` | Tight-zone and industrial curve-wrapping example. |
| `geolink/15_9-14.las` | Mineral-solver tutorial well with vendor lithology. |
| `geolink/25_5-1.las` | Core-bearing North Sea example with PHIC, permeability, saturation, and grain-density curves. |
| `kgs/1046123714.las` | Morgan No. 1-17 legacy LAS example with generic CSV formation tops. |
| `daydream/daydream1_cmr.las` | Daydream 1 processed CMR summary, 3755-4145 m. |
| `daydream/daydream1_cmr_shallow.las` | Daydream 1 processed CMR summary, 2860-3050 m. |
| `phillips/` | Six small NMR, paired core and COSTA capillary-pressure reference datasets from Craig Phillips. See `phillips/README.md` for provenance, units and limitations; CSV and Excel paths are available through `sample()`. |
| `epslog/` | 15/12-20 S cores C01/C02: XRF, image/laser grain-size estimates, scanner strength and ultrasonic velocities, plus a same-well RealPore core-table extract. See `epslog/README.md` for loading mappings and unresolved calibration inputs; discover with `list_samples("epslog")`. |
| `formations_30_6-5.csv` | Formation tops for 30/6-5. |
| `formation_pressure/` | One canonical LAS station file for each of the 35/11-13 and 35/11-10 Fensfjord examples; Tutorial 15 calls the library reader, FWL fitter and plotter. |
| `formations_costa.csv` | The KHARAIB-2 interval of each Costa well (first to last logged sample, in ft), as a multi-well table: `pp.read(pp.sample("HW-3"), formations=pp.sample("formations_costa"))` selects the row for the well. |
| `lithology_30_6-5.csv` | Per-interval lithology for 30/6-5. |

## Repository fixtures

Large LAS/DLIS regression files use the same domain folders as the bundled
samples but are explicitly catalogued and resolved separately, so they are
never confused with installed samples:

```python
from petrophys.datasets import integration_fixture

path = integration_fixture("daydream1_fmi_processed_dynamic_image.dlis")
```

Run `git lfs pull` in a source checkout before using these files. Tests skip
cleanly when optional fixtures are unavailable. See `daydream/README.md` for
provenance, checksums, and validation commands.

The Costa wells are all retained intentionally: the public multiwell notebooks
load and compare the complete 17-well set. The larger external GeoLink
development collection is not a repository dependency.
