# Petrophys Tutorial Curriculum

A progressive, hands-on curriculum that teaches you how to leverage the **petrophys**
library end-to-end — from loading raw well logs to a full formation evaluation and the
library's advanced, differentiating capabilities.

Every notebook runs entirely on the **bundled sample data** (no downloads) via
`petrophys.sample(...)` and `petrophys.list_samples(...)`.

## How to use this curriculum

Work through the notebooks **in order** — each builds on concepts from the previous ones.
Every notebook follows the same rhythm:

1. **Learning goals** — what you'll be able to do afterwards
2. **Why it matters** — the physics / interpretation context
3. **Load data** → **step-by-step** with explanation
4. **Visualize** the result
5. **Pitfalls & parameter choices**
6. **Recap** + a **"Try it yourself"** exercise + **what's next**

## Datasets

| Dataset | Wells | Character | Best for |
|---------|-------|-----------|----------|
| **Geolink** (North Sea) | `30_6-5` (primary), `15_9-14`, `16_7-1`, `25_5-1` | Clastic, rich curve set, formation tops + per-interval lithology, core PHIC/KH_CORE/SWC | Full clastic formation evaluation, mineral solving, dynamic workflow |
| **Costa** | `HW-3` … `HW-32` (17 wells) | Carbonate, core PHIE_CC / SW_INPUT / PERM_CC, Archie header params | Carbonate Archie Sw, core calibration, multi-well/field studies |

```python
import petrophys
petrophys.list_samples()            # all bundled wells
petrophys.list_samples("geolink")   # ['15_9-14', '16_7-1', '25_5-1', '30_6-5']
petrophys.list_samples("costa")     # 17 carbonate wells
from petrophys.io import read_las
well = read_las(petrophys.sample("30_6-5"))
```

## Part A — Foundations & Core Science

| # | Notebook | You will learn |
|---|----------|----------------|
| 00 | `00_orientation_and_setup` | What petrophys is, install/extras, data access, the two datasets, how to read this curriculum |
| 01 | `01_loading_and_inspecting_data` | The `Well`/`Curve` model, headers, curve inventory, units |
| 02 | `02_qaqc_and_standardization` | Mnemonic aliasing, unit conversion, bad-hole/washout, value-range QC, normalization |
| 03 | `03_clay_volume` | Vcl from GR (linear/Larionov/Steiber/Clavier), SP, neutron-density; clay resistivity |
| 04 | `04_porosity` | Density/neutron/sonic/total/effective porosity; hydrocarbon correction |
| 05 | `05_lithology_and_mineral_solver` | N-D/PEF crossplots; 2- & 3-mineral least-squares solver; deterministic transparency |
| 06 | `06_formation_water_and_environment` | Rw (SP/salinity), Rmf, temperature profile, invasion |
| 07 | `07_water_saturation` | Archie + shaly-sand models; Pickett m/n; clastic vs carbonate model selection |
| 08 | `08_permeability` | Timur/Coates; core poro-perm calibration; impermeable flags |
| 09 | `09_reservoir_quality_and_net_pay` | RQI/FZI/flow units; BVW/MHI; 3-stage net pay; HCPV/OOIP/OGIP |

## Part B — Integrated & Advanced Workflows

| # | Notebook | You will learn |
|---|----------|----------------|
| 10 | `10_dynamic_workflow_glass_box` | Auto method selection, 10-module pipeline, step traces, overrides, zoned-by-formation |
| 11 | `11_formation_and_interval_analysis` | Tops CSV, zones/subzones, per-interval & formation-level summaries |
| 12 | `12_core_calibration_and_digital_core` | Core→log calibration; digital-core facies (K-means), XRF, grain size, geomechanics |
| 13 | `13_ml_curve_imputation` | Pretrained XGBoost geolink model; impute missing curves; confidence |
| 14 | `14_nmr_petrophysics` | NMR T2 bins; SDR & Coates permeability; BVI/FFI; NMR vs D-N porosity |
| 15 | `15_capillary_pressure_saturation_height` | J-function, Brooks–Corey fitting, Pc↔height, log-based Sw-height |
| 16 | `16_borehole_image_fmi` | FMI pads → oriented image, equalization, azimuthal coverage |
| 17 | `17_visualization_masterclass` | XML templates, adaptive viewer, all crossplots, specialist panels, batch export |
| 18 | `18_multiwell_field_study` | Costa 17-well batch, normalization, cross-well comparison, portfolio summary |

## Runtime

Notebooks are authored and verified against the **`xtts_env`** conda environment
(registered Jupyter kernel: **Python (xtts_env)**). Optional ML notebooks (12, 13) require
`scikit-learn`, `xgboost`, `joblib`.

> The previous notebook set is preserved unchanged under [`legacy/`](legacy/).
