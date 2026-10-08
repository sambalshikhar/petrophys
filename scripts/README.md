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
Run
`python scripts/generate_benchmark_context.py --input-dir PATH/eval_processed --output-dir NEW_DIR`.
It reads the raw LAS files from the sibling `eval` folder and, by default, the
NPD tops workbook and lithology codebook next to `eval_processed`. Reruns are
byte-identical. ELECTROFACIES labels cannot be recovered (the original clustering was
unseeded).

`--from-raw` runs the whole package from the raw logs instead: it interprets
them into `<output-dir>/processed_las`, then builds the context from those
files.

The archived package is not one run: 30 wells were interpreted in April, while
`15_9-17` and `34_10-23` were reprocessed ten days later with changed
calculations. `--las-profile` sets the profile for every well and
`--well-profile` names the exceptions, so one command covers the package:

```
python scripts/generate_benchmark_context.py --from-raw \
  --input-dir PATH/eval --output-dir NEW_DIR \
  --well-profile 15_9-17 current --well-profile 34_10-23 current
```

The context step asks for `reuse_input_resflag=True`, because the package
picked SWIRR_APP from the reservoir flag already on the interpreted LAS rather
than recomputing it, which is no longer the default.

Checked against the archived package: 29 of the 32 wells reproduce every number
and every note. `16_10-4` matches numerically but orders two entries of one note
list differently, and `15_9-17` and `34_10-23` reproduce every number except the
SO statistics, which the later run added, plus some note wording.

The interpreted LAS files are byte-identical to `eval_processed` for 25 wells.
Four more (`15_9-13`, `30_6-5`, `34_10-21`, `35_9-2`) differ only in the field
name in the LAS header, where a non-ASCII character such as the `Ø` of
`OSEBERG ØST` does not survive the read/write round trip; their contexts are
unaffected. `16_10-4` also keeps PHIT and PHI_SEC, which the original export
dropped.
