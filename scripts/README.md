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
