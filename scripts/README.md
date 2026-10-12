# Project Scripts

This directory contains runnable utilities for context generation, workflow
trace export, model training, and figure generation. Run them from the
repository root. Tutorials belong in `examples/`.

`verify_phillips_data.py` checks the copied Craig Phillips NMR/core/SCAL files
against pinned source hashes and exercises their supported data paths. It also
checks all 17 existing COSTA well derivatives. Run
`python scripts/verify_phillips_data.py --report outputs/phillips-verification.json`.
Use `--costa-source-dir PATH` to compare against a local checkout of Craig's
COSTA workflow repository as well.

`verify_epslog_data.py` checks the eight Epslog LAS deliveries and the matching
RealPore core extract against recorded hashes, then exercises the XRF, grain
size and core-mechanics readers and nominal core-depth matching. Run
`python scripts/verify_epslog_data.py --report outputs/epslog-verification.json`.
Use `--source-dir PATH` to compare the LAS copies with the original Epslog
folder (containing `LAS/`). These checks do not establish a calibrated depth
tie or replace missing static mechanics and wireline measurements.

## MCP live evaluation

`mcp_live_eval.py` runs the scenarios in `mcp_eval_scenarios.json` against the
MCP server with a real OpenAI model (default `gpt-5.6-luna`) and records every
tool call, error, output size, token count and final answer. Use it after
changing tool descriptions or outputs. It needs `OPENAI_API_KEY`, costs API
tokens, and can target the in-process server, stdio, or a running HTTP server
(`--target http://127.0.0.1:8000/mcp`, token from `PETROPHYS_MCP_TOKEN`).

## Eval context benchmark

`generate_benchmark_context.py` regenerates the GeoLink eval context benchmark
package (`context_json_category_all_wells_fm_gp_mb_core_no_metrics_phiecc_sat_latest`).
One command reproduces the whole package from the raw LAS files:

```
python scripts/generate_benchmark_context.py --from-raw --input-dir PATH/eval --output-dir NEW_DIR
```

It interprets the raw logs into `NEW_DIR/processed_las`, then builds the
context from those files. The NPD tops workbook
(`NPD stratigraphic picks north sea.xlsx`) and the lithology codebook
(`Litho_Nomenclature_Kingdom.xlsx`) are read from the folder that holds `eval`;
`--tops-file` and `--lithology-mapping-file` name other files. Without
`--from-raw`, `--input-dir` is a folder of interpreted LAS files such as
`eval_processed`, and the raw LAS files are read from its sibling `eval` folder
(or `--raw-input-dir`).

The package was made in two runs with different calculations: its main run
interpreted 30 of the 32 wells, and a re-run interpreted `15_9-17` and
`34_10-23` and wrote the context of every well. The script's default,
`--calculation-profile benchmark`, keeps both sets of calculations: it
interprets `15_9-17` and `34_10-23` with the re-run's calculations and the
other wells with the main run's, and writes the context of every well with the
re-run's. The two wells are recognized by their UWI or WELL header or by their
file name. The context step also sets `reuse_input_resflag=True`, because the
package picked SWIRR_APP from the reservoir flag already on the interpreted LAS
rather than recomputing it. `--calculation-profile current` runs both steps
with the library's default calculations instead, which do not reproduce the
package.

ELECTROFACIES labels cannot be recovered: the package's k-means clustering was
unseeded. The script fixes Python's string-hash seed, so reruns into the same
output folder write byte-identical files. The one exception is a run in which
Python compiles the library afresh instead of loading it from its bytecode
cache, as it normally does: that run can order some lists of curve names
differently.

A full run was checked against the archived package. The 30 wells of the main
run reproduce every number (ELECTROFACIES aside) and every note, except that
the derived-log notes list some curves in another order. `15_9-17` and
`34_10-23` reproduce every number except the SO statistics in their archived
files, and some of their note wording differs. The command above regenerates
that run for all 32 wells: the interpreted LAS files are byte-identical, the
workflow traces identical apart from the output paths they record, and the
25,566 context files hold the same 10.5 million numbers and the same notes,
compared without regard to list order.

The interpreted LAS files are byte-identical to `eval_processed` for 25 wells.
Four more (`15_9-13`, `30_6-5`, `34_10-21`, `35_9-2`) differ only in the spaces
that align the `~Well` section: their field names contain a non-ASCII
character, such as the `Ø` of `OSEBERG ØST`, which the archived files count as
two characters when they align the columns. Their contexts are unaffected.
`15_9-17` and `34_10-23` hold the same data but write the standard unit labels
`API` and `v/v` where the archived files keep `gAPI` and `m3/m3`, and give the
temperature-profile parameters units and descriptions; `15_9-17` also has the
`~Well` spacing difference. `16_10-4` differs in the description of PHIE and at
three depths where the archived file has SW, SXO, BVW and MHI of 0.
