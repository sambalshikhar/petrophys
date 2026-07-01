# Petrophys Case Studies

Where the tutorials teach one module at a time, these **case studies** are realistic,
end-to-end interpretation *projects*. Each one states a business question, works a real well
(or the whole field) through a chained multi-module workflow, and finishes with a concrete
deliverable — a completion recommendation, a rock-type profile, a field ranking, a caveated
interpretation, or a decision under uncertainty.

Every notebook follows the same spine: **the brief → the data → workflow plan → execution →
integrated result → interpretation & deliverable → limitations → reproduce**. All run on the
bundled Geolink + Costa data against the `xtts_env` kernel and are verified end-to-end under
nbconvert.

> Work the [Part A foundations](../part_a_foundations/) and [Part B advanced](../part_b_advanced/)
> tutorials first — the case studies reference those concepts rather than re-teaching them.

| # | Case study | Well(s) | The project | Headline result |
|---|-----------|---------|-------------|-----------------|
| **CS1** | [Full formation evaluation](cs1_full_formation_evaluation_30_6-5.ipynb) | Geolink **30/6-5** (clastic) | Raw LAS → complete evaluation → zone-by-zone completion recommendation, with the dynamic-workflow step trace as the audit and a per-formation roll-up | Perforate the **Brent (Ness+Etive)** oil sand first (net pay 50 m, Sw ~0.3, high perm); Statfjord as secondary |
| **CS2** | [Carbonate reservoir characterization & rock typing](cs2_carbonate_rock_typing_costa.ipynb) | Costa **HW-10** (carbonate) + core | Archie Sw with real header params, core-log calibration, RQI/FZI rock typing, saturation-height | **3 flow units**; RRT3 (median 42 mD) is the best reservoir. *Key lesson:* Rw must be temperature-corrected or Sw>1 everywhere |
| **CS3** | [Field-scale screening & well ranking](cs3_field_screening_costa_17wells.ipynb) | **All 17 Costa** wells | Batch-interpret the field, normalize GR cross-well, rank by hydrocarbon pore thickness, tier for development | Top-3 **HW-30 / HW-29 / HW-25** carry 57% of field HCPT; 7 Tier-1 candidates |
| **CS4** | [Making the most of an imperfect well](cs4_imperfect_well_recovery_16_7-1.ipynb) | Geolink **16/7-1** (data-poor) | QC bad-hole + off-scale resistivity, ML-impute the missing neutron log, validate by blank-and-restore, interpret with uncertainty flags | 33.4 m net pay but only **10.1 m firm** — 70% rests on imputed/flagged data. The honesty case study |
| **CS5** | [Saturation-model selection & sensitivity](cs5_saturation_model_sensitivity_30_6-5.ipynb) | Geolink **30/6-5** (shaly sand) | Run all 5 Sw models, Pickett m/n, sweep Rw/m/n/cutoff, quantify the spread | Recommend **Indonesia** model; HCPV band −40%/+78%, with **Rw** the dominant lever → acquire a measured Rw next |

## Runtime
`xtts_env` conda env (Jupyter kernel "Python (xtts_env)"). CS4 uses the optional ML stack
(`scikit-learn`, `xgboost`, `joblib`). All five execute top-to-bottom with zero errors.
