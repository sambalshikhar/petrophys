# Petrophys notebooks

Worked examples for the `petrophys` library, in three groups. The **foundations** get you
loading data and running a full interpretation; the **per-property** notebooks each cover one
computation (read the one you need); the **deep dives** open up how the harder pieces actually
work. Every notebook runs on the sample wells bundled with the package
(`petrophys/data/geolink/`), so no external data is needed.

```bash
pip install -e .          # install petrophys from the repo root
pip install jupyter       # if you don't already have it
jupyter lab notebooks/
```

## Foundations

| Notebook | What it covers |
|----------|----------------|
| `00_index` | What the library is and a one-cell sanity check — start here |
| `01_loading_data` | Read a LAS/DLIS well, find curves, standardize names and units |
| `02_automated_workflow` | One call from raw logs to a full interpretation |

## One per property

| Notebook | What it covers |
|----------|----------------|
| `03_clay_volume` | Clay volume — gamma ray vs. neutron-density, and where they disagree |
| `04_porosity` | Density/neutron/sonic porosity; total vs. effective (clay-bound water) |
| `05_water_saturation` | Archie and the shaly-sand models; reading Rw and m from a Pickett plot |
| `06_permeability` | Log estimate vs. core-calibrated permeability |
| `07_lithology` | Mineral volumes and lithology from the multi-mineral solve |
| `08_net_pay` | Net rock → reservoir → pay, and the cutoffs behind them |

## Visualization

| Notebook | What it covers |
|----------|----------------|
| `09_plotting` | The standard displays — triple combo, crossplots, Pickett |

## Deep dives

How the harder pieces work under the hood — these go past usage into the method and the evidence.

| Notebook | What it covers |
|----------|----------------|
| `10_deterministic_mineral_solver` | The mineral solver as a linear system — the actual A·x=b at one depth |
| `11_glass_box_dynamic_workflow` | The workflow's decision trace, and how it reroutes on a sparser well |
| `12_petrophysics_for_llm_agents` | JSON traces, statistics, and QC facts as an agent backend |
| `13_core_calibration_and_validation` | Porosity and permeability against core, and validation vs. NPD ground truth |
| `14_fmi_pads_to_oriented_image` | Rotating FMI buttons to azimuth, and image equalization |

The deep-dive notebooks ship executed, with their figures embedded. `14` needs a processed FMI
DLIS (set `PETROPHYS_FMI_DLIS`); the rest run on the bundled wells as-is.
