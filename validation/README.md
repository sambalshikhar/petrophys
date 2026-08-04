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

`porosity_saturation.py` runs an empirical multiwell check of core-porosity
agreement and net-pay behavior for reference dry and gas wells. The bundled
30/6-5 sample runs without external data; the wider check accepts a portable
LAS directory through `--data-dir` or `PETROPHYS_GEOLINK_DIR`. Missing optional
wells and workflow warnings remain visible. Results can also be written as
structured JSON:

```powershell
python validation/porosity_saturation.py --well 30_6-5
python validation/porosity_saturation.py --data-dir path/to/las --output build/porosity_saturation.json
```
