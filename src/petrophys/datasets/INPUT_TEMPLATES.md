# Input templates

Please provide one LAS and, when needed, one formation CSV and one lithology
CSV per well. Replace the synthetic example rows with your data.

| File | Columns | Use |
| --- | --- | --- |
| `formation_tops_template.csv` | `formation_name,top_depth` | Plot boundaries |
| `formation_intervals_template.csv` | `formation_name,top_depth,bottom_depth` | Bounded analysis; also plot tops |
| `lithology_intervals_template.csv` | `top,bottom,lithology` | Plot categorical rock intervals |
| `lithology_codes_template.csv` | `code,lithology` | Decode an existing numeric LAS lithology curve |

Use UTF-8, comma separators, decimal points, finite numeric depths, one header
row, and one well per file for these minimal templates. Optional metadata
columns are `well`, `depth_unit`, `depth_reference`, and `datum`.
Top must be less than bottom; lithology intervals may abut but not overlap.
Uncovered depths remain unfilled. Codebook assignments are illustrative and
must be replaced with your vendor's definitions.

State the well identity, units, depth coordinate (MD/TVD/TVDSS), datum, source,
and any core-to-log shift in accompanying delivery notes. Shared readers convert
declared metres/feet when joining to a well, but require matching depth reference
and datum. Missing metadata is recorded as an assumption, not verified alignment.
Distinguish formation groups/members from formation rank and categorical
lithology from percentage mixtures. No WITSML or PDF import is implied.

```python
import petrophys as pp

well = pp.read("my_well.las", formations="formations.csv", lithology="lithology.csv")
viewer = pp.plot(well)
viewer.show()
```

Both geology arguments are optional. CSV, TSV, Excel and DataFrames are
accepted; nonstandard headers can be mapped explicitly. Multiple Excel sheets
require selection. Attached lithology is used by both plotting and subsequent
`pp.interpret(well)`; passing it only to a viewer changes only that display.
Interpretation works on a copy by default. Top-only formation analysis requires
an explicit terminal depth; plotting tops does not require bottoms.
Read the full [input guide](https://petrophys.readthedocs.io/en/latest/input_templates.html)
(`docs/input_templates.rst` in the repository) for industry sources, vendor field
mapping, validation, saturation unit handling and a complete bundled-well example.
