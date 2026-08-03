# Validation

Reproducible checks against external or multi-well reference datasets live
here. Generated results are not committed. Large integration fixtures may be
stored separately with Git LFS and are excluded from package distributions.
Unit and regression tests that must run in CI remain under `tests/`.

`las_dlis_equivalence.py` compares matching shallow and deep Daydream 1 LAS and
DLIS deliveries at both native-delivery and matched-common-channel levels. The
deep LAS fixture is a full-depth curated scalar subset of its 396 MB vendor
delivery. The pair also verifies that uncalibrated LAS resistivity is preserved
without being used for saturation. See
`src/petrophys/datasets/daydream/README.md` for fixture provenance
and usage.
