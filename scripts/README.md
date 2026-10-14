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

`generate_benchmark_context.py` generates category context for GeoLink, COSTA
and Danish using each dataset's interpretation parameters, calibration and
grouping rules. It accepts one LAS file or a folder, identifies the collection
by well name, and retains interpreted LAS and workflow traces in
`OUTPUT/processed_las`. Danish runs also retain `OUTPUT/prepared_las`.

See the [benchmark context guide](../docs/benchmark_context.md) for required
inputs, complete commands and subset runs. For context from your own wells,
see the [general context guide](../docs/context.md).

```bash
python scripts/generate_benchmark_context.py --dataset costa --input-dir "src/petrophys/datasets/costa" --well HW-31 --output-dir "outputs/HW-31_benchmark_01"
python scripts/generate_benchmark_context.py --dataset geolink --from-raw --input-dir "PATH/eval" --tops-file "PATH/NPD stratigraphic picks north sea.xlsx" --lithology-mapping-file "PATH/Litho_Nomenclature_Kingdom.xlsx" --output-dir "outputs/geolink_benchmark_01"
python scripts/generate_benchmark_context.py --dataset danish --input-dir "PATH/danish/las" --tops-file "PATH/Danish_formation_tops_for_context.xlsx" --lithology-mapping-file "PATH/Litho_Nomenclature_Kingdom.xlsx" --output-dir "outputs/danish_benchmark_01"
```

For a GeoLink raw LAS folder, supply `--from-raw`. For an interpreted GeoLink
folder, omit it and provide `--raw-input-dir` to identify the original input
curves. Use a fresh output directory for each run.
