# COSTA samples

Source: Costa Gomes, Geiger, and Arnold (2022), Heriot-Watt University,
*COSTA MODEL: Hierarchical carbonate reservoir benchmarking case study*.
[Dataset DOI](https://doi.org/10.17861/6e36e28d-50d9-4e31-9790-18db4bce6e5d).
The publisher lists CC BY. Petrophys includes 17 selected wells with adapted LAS
headers and curve names. See `THIRD_PARTY_DATA.md` in the distribution for terms
and the requested publication citation.

For the carbonate interpretation, pass `lithology="carbonate"` and
`matrix_type="auto"` in `DynamicWorkflowConfig`. Automatic mineral selection can
also use interpreted lithology or core-constrained matrix density. Sparse or
conflicting measurements cannot establish the mineral system uniquely.
