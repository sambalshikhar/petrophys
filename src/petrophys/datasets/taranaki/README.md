# Taranaki Basin wells (New Zealand)

Two onshore wells from the Taranaki Basin Curated Well Logs Dataset:

| File | Well | Interval (MD) | Notes |
|---|---|---|---|
| `cheal-1.las` | Cheal-1 | 376.6-2847.9 m | Vertical; Urenui, Mount Messenger, Manganui and Moki formations |
| `tuhua-7.las` | Tuhua-7 | 590.6-2671.0 m | Deviated; includes the Tikorangi limestone. Formation labels repeat along the borehole |
| `formations_cheal-1.csv`, `formations_tuhua-7.csv` | | | Formation intervals from the dataset's `FORMATION` column |

Curves and units, as given by the dataset README: `BS`, `CALI` (in), `DENS`,
`DRHO` (g/cc), `DTC` (us/ft), `GR` (gAPI), `NEUT` (v/v), `PEF` (b/e),
`RESD`, `RESM`, `RESS` (ohm.m), `SP` (mV), `TEMP` (degC). Depth is measured
depth in metres.

```python
import petrophys as pp

well = pp.read(pp.sample("cheal-1"))
tops = pp.read_formations(pp.sample("formations_cheal-1"))
result = pp.interpret(well, formation_tops=tops)
```

## Source and licence

Taranaki Basin Curated Well Logs Dataset, IBM Developer Data Asset eXchange
(<https://developer.ibm.com/exchanges/data/all/taranaki-basin-curated-well-logs-data/>),
curated from the New Zealand Petroleum & Minerals Online Exploration Database
(data.nzpam.govt.nz) and Petlab (pet.gns.cri.nz).

Licensed under the Community Data License Agreement - Sharing, Version 1.0
(CDLA-Sharing-1.0); the full text is in `LICENSE-CDLA-Sharing-1.0.txt` and at
<https://cdla.dev/sharing-1-0/>. Redistribution of these files and of data
derived from them must remain under the same licence.

**Modifications** (CDLA-Sharing section 3.1(b)): these files are an extract of
the dataset's `logs.csv`. Two wells were selected; the columns `X`, `Y`, `Z` and
`TENS` were removed; each well was written as LAS 2.0 with the units listed
above and the coordinates from `coords.csv`; the formation intervals were
derived from the `FORMATION` column. Log values are unchanged.
