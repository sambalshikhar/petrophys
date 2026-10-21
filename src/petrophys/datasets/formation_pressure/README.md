# Formation-pressure example data

Two kinds of MDT (Schlumberger Modular Formation Dynamics Tester) data:

- **Pressure stations** for two North Sea wells: one row per test with the
  stabilized formation pressure, depth-indexed.
- **Raw pretest records** from IODP Hole C0009A: probe pressure against time
  through each test, as the original DLIS files and as LAS copies.

## Pressure stations (35/11-13, 35/11-10)

| File | Well | Stations |
|---|---|---|
| `35_11_13_mdt.las` | 35/11-13 (2005) | 29 pretests, MDT run 1A, 3071.6-3180.1 m TVD MSL |
| `35_11_10_mdt.las` | 35/11-10 (1997) | 88 pretests and sampling stations, MDT runs 1A-2E, 1949.9-2847.8 m TVD MSL |

The files hold measurements only: no fluid labels and no contacts. Fluid
contacts are inferred from the pressures (`infer_contacts()`).

`DEPT` is true vertical depth below MSL in metres (positive down) and `MD` the
measured depth from RKB. `PRES` is the quartz-gauge (CQG) formation pressure in
bar; `MUDPRE`/`MUDPOST` (35/11-13) and `HYD_BEFORE`/`HYD_AFTER` (35/11-10) are
mud pressures, not formation pressures. `QC` is the test quality: 0 marks a
valid formation pressure. In 35/11-10, `PRES_STRAIN` is the strain-gauge
reading of the same test, `SAMPLE` = 1 marks a sampling station (its pressure
is read during sampling, not a static pretest), and `RUN`/`TEST` give the test
number.

```python
import lasio
import petrophys as pp
from petrophys.interpretation.saturation.height import read_formation_pressure

las = lasio.read(pp.sample("35_11_13_mdt"))
las.append_curve("USE_FIT", (las["QC"] == 0).astype(int))   # valid pretests only
fit = read_formation_pressure(las).infer_contacts()
print(fit.status, fit.fwl_depth, [c.depth for c in fit.contacts])
```

Contact inference assumes one connected reservoir. 35/11-10 covers several
pressure regimes, so select the stations of one reservoir interval (by depth)
and the static pretests (`SAMPLE == 0`) before inferring contacts.

## Raw pretest records (IODP C0009A)

Three single-probe MDT pretest files from IODP Expedition 319, Hole C0009A
(Kumano Basin, offshore Kii Peninsula, Japan; July 2009), one tool setting per
file:

| Original DLIS | LAS copy | Behaviour |
|---|---|---|
| `319-C0009A_MDT_059LTP.DLIS` | `c0009a_mdt_059.las` | Stable buildups; pressure accepted (4102.13 psi) |
| `319-C0009A_MDT_067LTP.DLIS` | `c0009a_mdt_067.las` | Low mobility, buildup still rising; returned for review with a candidate |
| `319-C0009A_MDT_078LTP.DLIS` | `c0009a_mdt_078.las` | Unstable mud baseline; returned for review without a candidate |

The DLIS files are the originals, unchanged. Each LAS copy holds six channels
of the DLIS frame `600T`, values unchanged:

| Curve | Unit | Description |
|---|---|---|
| `TIME` | s | Elapsed acquisition time (index; ms in the DLIS) |
| `TDEP` | ft | Tool depth, measured depth below the drill floor (0.1 in in the DLIS) |
| `BQP1` | psi | Probe quartz gauge pressure |
| `BSG1` | psi | Probe strain gauge pressure |
| `BQT1` | degC | Quartz gauge temperature |
| `PTV1` | cm3 | Pretest volume |

```python
import petrophys as pp
from petrophys.interpretation.saturation.height import read_mdt_pretest

test = read_mdt_pretest(pp.sample("c0009a_mdt_059"), time_column="TIME",
                        pressure_column="BQP1", depth_column="TDEP")
# The original DLIS (needs the dlis extra) gives the same pick:
# test = read_mdt_pretest(pp.sample("319-C0009A_MDT_059LTP"), pressure_column="BQP1")
pick = test.pick_pressure()
print(pick.status, pick.pressure, pick.station_depth)   # accepted 4102.13 psi at 9229.1 ft
```

See `THIRD_PARTY_DATA.md` for attribution.
