# Validation

Reproducible checks against external or multi-well reference datasets live
here. Generated results are not committed. Large integration fixtures may be
stored separately with Git LFS and are excluded from package distributions.
Unit and regression tests that must run in CI remain under `tests/`.

`las_dlis_equivalence.py` compares matching shallow and deep Daydream 1 LAS and
DLIS deliveries at both native-delivery and matched-common-channel levels. The
deep pair also verifies that uncalibrated LAS resistivity is preserved without
being used for saturation. See `integration_data/daydream/README.md` for fixture
provenance and usage.
