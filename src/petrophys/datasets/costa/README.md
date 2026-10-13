# COSTA samples

Source: Costa Gomes, Geiger, and Arnold (2022), Heriot-Watt University,
*COSTA MODEL: Hierarchical carbonate reservoir benchmarking case study*.
[Dataset DOI](https://doi.org/10.17861/6e36e28d-50d9-4e31-9790-18db4bce6e5d).
The publisher lists CC BY. Petrophys includes 17 selected wells with adapted LAS
headers and curve names. See `THIRD_PARTY_DATA.md` in the distribution for terms
and the requested publication citation.

The immediate source copies are Craig Phillips's
[COSTA workflow repository](https://github.com/Philliec459/Jupyter-Notebooks_for-Characterization-of-a-New-Open-Source-Carbonate-Reservoir-Benchmarking-Case-St),
commit `b2bf2df1e6a6b24c9187e3a982b975790a49108b`.
`../phillips/source_manifest.json` records the source hashes and verifies all
6,869 rows across the 17 wells. Retained values and null masks are unchanged:
`PHIE_CC` is upstream `PHIE`, `SW_INPUT` is `SW`, and `PERM_CC` is `CORE_PERM`.
The separate upstream `SO` and `CORE_POR` curves were omitted. A renamed curve
is not evidence of an additional measurement or recalibration.
The original MIT notice is retained in `../phillips/LICENSE-Craig-2022.txt`.
Related HPMI data and fitted parameters are available as
`sample("costa_core_pc")` and `sample("costa_thomeer_fits")`.

`petrophys/_internal/data/costa_interpretation_parameters.csv` and
`costa_reference_carbonate_volumes.csv` beside it hold the other inputs of the
published COSTA context package: per well, the
BHT, the Rw at 68 degF and the saturation, clay-volume and porosity methods
used, and, by depth for 15 wells, calcite (`vlim`) and dolomite (`vdol`)
volumes from an earlier interpretation of these logs, which the package took
where available. `pp.export_context(None, out, layout="category_formation_lithology",
context_profile="costa")` recomputes that package from these files.

For the carbonate interpretation, pass `lithology_domain="carbonate"` in
`DynamicWorkflowConfig`; the matrix is then resolved within the carbonate domain. Automatic mineral selection can
also use interpreted lithology or core-constrained matrix density. Sparse or
conflicting measurements cannot establish the mineral system uniquely.
