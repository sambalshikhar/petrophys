# Formation-pressure example wells

This directory contains one normalized LAS station file per example well:

- `35_11_13_mdt.las` — 35/11-13, 29 MDT stations, TVD below MSL in metres and formation pressure in bar.
- `35_11_10_fensfjord_mdt.las` — 35/11-10 Fensfjord, digitized MDT stations, TVD below MSL in metres and formation pressure in bar.

The files are ready for the library reader:

```python
import petrophys as pp
from petrophys.interpretation.saturation.height import read_formation_pressure

data = read_formation_pressure(pp.sample("35_11_13_mdt"))
fit = data.fit_fwl()
print(fit.fwl_depth)
```

`DEPT` is the vertical depth used for the pressure fit. `MD` is retained as a
separate measured-depth curve. `PRES` is formation pressure, not mud pressure.
The LAS files contain the report-derived branch and quality curves needed for a
reproduction example; native MDT deliveries may require the user to provide
their own fluid or quality interpretation.

The public workflow also accepts a user-owned CSV or DataFrame. This repository
keeps only the two LAS examples so the packaged data stays small and each well
has one canonical input.

For unlabeled pressure stations, remove or ignore the interpretation curves and
call `infer_contacts()` after applying measurement-quality selection. Contact
inference remains conditional on a common datum, connected reservoir and
appropriate pressure units.
