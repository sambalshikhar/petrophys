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
