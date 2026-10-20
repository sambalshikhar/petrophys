# Teapot Dome wells (Wyoming, USA)

Two wells from Teapot Dome oil field (Naval Petroleum Reserve No. 3), Natrona
County, Wyoming:

| File | Well | API number | Interval | Curves |
|---|---|---|---|---|
| `teapot_17-wx-21.las` | Fenix WX #17-21 (1978) | 49025107090000 | 0-5649.5 ft | GRD, NPHI, CALD, RHOB, DRHO, SPR, SFL, ILM, ILD, DT |
| `teapot_44-mdx-10.las` | Fenix MDX #44-10 (1977) | 49025104310000 | 470-3791.5 ft | DT, SPR, LLS, LLD, MSFL, NPOR, RHOB, CORR, GRD, CALD |

The files are the original LAS 2.0 deliveries, renamed. Both are logged in
feet with NULL -999.25 and carry mud, temperature, bit-size and casing values
in their headers. Casing is set to about 500 ft in both wells; readings above
the casing shoe are not open-hole formation measurements.

```python
import petrophys as pp

well = pp.read(pp.sample("teapot_17-wx-21"))
result = pp.interpret(well)
```

## Source

Teapot Dome (NPR-3) data set, released by the Rocky Mountain Oilfield Testing
Center (RMOTC), U.S. Department of Energy, as non-proprietary data for research,
education and software testing. Logs acquired by Schlumberger for Fenix &
Scisson. The data are provided without warranty.
