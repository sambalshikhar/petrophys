---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# LLM agents (MCP)

A context gives a model the result of an interpretation ({doc}`context`); an MCP server lets the model run the interpretation itself.
Petrophys ships a [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server for LLM applications such as Claude Desktop, Claude Code, Cursor and VS Code.
Through it, a model loads wells, interprets them, reads the trace and plots logs on your machine. It uses the same workflow as the Python API.

```{code-cell} python
:tags: [remove-cell]
import pandas as pd

pd.set_option("display.max_colwidth", None)  # show table text in full
```

## What a model can call

Here the server runs in-process, and the MCP Python client sends the same requests a real client sends:

```{code-cell} python
import tempfile

import pandas as pd
from mcp import Client
from petrophys.mcp import create_server

workspace = tempfile.TemporaryDirectory()
server = create_server(roots=[workspace.name])  # the folders the model may use

async with Client(server) as client:
    tools = (await client.list_tools()).tools
pd.DataFrame([{"Tool": tool.name,
               "What it does": tool.description.splitlines()[0],
               "Changes state": "no" if tool.annotations.read_only_hint else "yes"}
              for tool in tools]).style.hide(axis="index")
```

The server also offers the resources `petrophys://guide`, `petrophys://settings` and `petrophys://workflow-steps`, and an `interpretation_report` prompt.

## Install

The server needs Python 3.10 or newer.

```bash
python -m pip install "petrophys[mcp]"
petrophys-mcp check --root /path/to/wells     # installation, folders and a test session
```

## Connect a client on your machine

Local clients start the server themselves over stdio. Let `petrophys-mcp config` write the exact settings: it uses absolute paths, so it works whatever the client's `PATH` is. `--root` names the folders the server may read wells from and write exports to (repeatable); the bundled sample wells are always readable.

::::{tab-set}

:::{tab-item} Claude Code
```bash
petrophys-mcp config claude-code --root /path/to/wells   # prints the `claude mcp add ...` command
```
:::

:::{tab-item} Claude Desktop
```bash
petrophys-mcp config claude-desktop --root /path/to/wells
```
Paste the printed `mcpServers` entry into `claude_desktop_config.json` and restart Claude Desktop.
:::

:::{tab-item} Cursor, VS Code, Codex
```bash
petrophys-mcp config cursor --root /path/to/wells    # .cursor/mcp.json
petrophys-mcp config vscode --root /path/to/wells    # .vscode/mcp.json
petrophys-mcp config codex --root /path/to/wells     # ~/.codex/config.toml
```
:::

::::

## Deploy over HTTP

Use HTTP when the model is not on your machine: a server shared by a team, a container, or a hosted model (OpenAI's Responses API, ChatGPT connectors) that reaches the server by URL. Clients send a bearer token; the server refuses to listen beyond localhost without one. One server process holds one set of loaded wells and interpretations: every client of an HTTP server sees and can use the wells and runs of the others, so run a server per user when that matters.

::::{tab-set}

:::{tab-item} Run it
```bash
petrophys-mcp serve --http --root /path/to/wells --token auto
# Petrophys MCP server (http) at http://127.0.0.1:8000/mcp   (the generated token is printed)
```
Add `--host 0.0.0.0` to accept other machines. Behind a proxy or tunnel, pass its domain with `--allowed-host`. `GET /healthz` answers without a token.
:::

:::{tab-item} Docker
```bash
export PETROPHYS_MCP_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
mkdir -p wells && docker compose up -d           # uses the repository's docker-compose.yml
```
The container serves `http://localhost:8000/mcp`, reads wells from `./wells` and writes exports there.
:::

:::{tab-item} OpenAI
A hosted model needs a public HTTPS URL, from a reverse proxy or a tunnel to the server. Then:

```bash
petrophys-mcp config openai --url https://<your-host>/mcp --token <TOKEN>
```

prints the `{"type": "mcp", ...}` entry for `tools=[...]` in the Responses API. Without a public URL, `examples/mcp_openai_agent.py` connects an OpenAI model to a local server: `python examples/mcp_openai_agent.py "Interpret sample well 30/6-5 ..."`.
:::

:::{tab-item} Clients and checks
```bash
petrophys-mcp config claude-code --url https://<your-host>/mcp --token <TOKEN>
petrophys-mcp config vscode --url https://<your-host>/mcp --token <TOKEN>
petrophys-mcp check --url https://<your-host>/mcp --token <TOKEN>
```
`check` opens a real session against the running server and reports a rejected token plainly.
:::

::::

Then ask, for example: *"Load the sample well 30_6-5, interpret it with Rw = 0.07 ohm.m at 230 degF, and summarize net pay in the Brent Group."*

## Pin the calculations

Interpretations use the library's `"current"` calculations unless the `settings` of a tool call set `calculation_profile`.
An evaluation harness that must reproduce the GeoLink eval package can make `"geolink"` the server's default ({ref}`calculation profiles <calculation-profiles>`):

```bash
petrophys-mcp serve --calculation-profile geolink --root /path/to/wells
PETROPHYS_CALCULATION_PROFILE=geolink petrophys-mcp serve --root /path/to/wells   # the same
```

For a server that a client starts, set `PETROPHYS_CALCULATION_PROFILE` in the environment of the client's server entry.
A `calculation_profile` in a tool call's settings still wins.
Only the MCP server reads the variable: `pp.interpret` and `petrophys-context` ignore it.
`petrophys-mcp check` reports the profile the variable sets, or `current`.

## A live session

Each call below opens its own short connection, because an open connection cannot span notebook cells. The server keeps loaded wells between calls.

```{code-cell} python
from base64 import b64decode

from IPython.display import Image


async def call(tool, **arguments):
    async with Client(server) as client:
        return await client.call_tool(tool, arguments)
```

**Load** a sample well with its formations and lithology:

```{code-cell} python
loaded = await call("load_sample_well", name="30_6-5", with_geology=True)
well = loaded.structured_content
print(well["well_id"], f"| {well['top']:.0f}-{well['bottom']:.0f} {well['depth_unit']}",
      f"| {len(well['curves'])} curves | {len(well['formations'])} formations")
```

**Interpret** with the Rw the user supplied, summarizing the Brent Group window:

```{code-cell} python
run = (await call("interpret_well", well_id="30_6-5",
                  settings={"rw": 0.07, "rw_meas_temp_f": 230},  # Rw (ohm.m) at 230 degF
                  summary_top=2841, summary_bottom=2958)).structured_content
print(run["interpretation_id"], "|", run["status"], "|", run["resolved"])
print("net thickness:", run["net_thickness"])
print(len(run["skipped_steps"]), "skipped steps and", len(run["warnings"]),
      "warnings, each with a reason")
```

`assumptions` says in plain words what the numbers rest on, and `review_notes` what to confirm with the user; this run has no review notes.
The net-pay lines give the cutoffs of each lithology family and the samples left out of net reservoir and pay (evaporite, coal, halite recognized by its log response, bad hole), with the net pay each removed over the whole well, not just the summary window:

```{code-cell} python
from textwrap import fill

for line in run["assumptions"] + run["review_notes"]:
    print(fill(f"- {line}", 88, subsequent_indent="  "))
```

A bad setting fails before anything runs, with a message the model can act on:

```{code-cell} python
bad = await call("interpret_well", well_id="30_6-5",
                 settings={"rw": 0.07, "hc_type": "gass"})
print(bad.content[0].text)
```

**Summarize** the Brent Group window:

```{code-cell} python
zones = (await call("summarize_zones", interpretation_id=run["interpretation_id"],
                    top=2841, bottom=2958)).structured_content
columns = ["zone_name", "gross_thickness", "net_pay", "avg_phie", "avg_sw", "avg_vcl"]
pd.DataFrame(zones["zones"])[columns].style.hide(axis="index").format(precision=3)
```

**Plot** the reservoir; `plot_well` returns a PNG the client shows to the model and the user:

```{code-cell} python
figure = await call("plot_well", target_id=run["interpretation_id"],
                    top=2841, bottom=2958, dpi=72,
                    tracks=["GAMMA / HOLE", "POROSITY", "SATURATION", "FLAGS"])
Image(data=b64decode(figure.content[0].data), width=640)
```

```{code-cell} python
:tags: [remove-cell]
workspace.cleanup()
```

Every response is compact JSON that carries the well's depth unit. Data-quality problems, such as a curve in the wrong unit, come back in `warnings`. The {doc}`MCP tutorial <tutorials/part_b/20_mcp_server>` adds `explain_step` and `export_results`.

## Good practice and troubleshooting

| Situation | What to do |
|---|---|
| Rw unknown | Rw rarely comes from logs alone. The server tells models to ask for it; give it, with its temperature, when you know it |
| Reviewing an answer | ask the model to report `warnings` and `skipped_steps`: they show which results rest on assumptions |
| Whole-well totals | ask for a reservoir window (`summary_top`/`summary_bottom`) or per-formation results: one Rw serves the whole well, so shallow overburden with fresher water, where no lithology code applies, can still be flagged as marginal pay |
| File access | keep `--root` to the folders the model may use; exports overwrite files of the same name there |
| `The Petrophys MCP server needs the MCP SDK` | `pip install "petrophys[mcp]"` (Python 3.10+) |
| `... is outside the folders this server may access` | add the folder with `--root` or `PETROPHYS_MCP_ROOTS` (`:`-separated on Linux and macOS, `;` on Windows) |
| The client shows no Petrophys tools | run `petrophys-mcp check --root <folder>`, then the configured command in a terminal: it should wait without errors. Logs go to standard error, which clients show in their MCP logs |
| HTTP clients get 401 | the client must send `Authorization: Bearer <token>`; `petrophys-mcp check --url ... --token ...` confirms the token |
| Hosted model cannot reach the server | it needs a public HTTPS URL (proxy or tunnel); pass that domain with `--allowed-host` |

## Next

The guide ends here. The {doc}`tutorials <tutorials/index>` go further: {doc}`tutorial 20 <tutorials/part_b/20_mcp_server>` runs a full MCP session (load, interpret, explain, summarize, plot, export), and the foundations lessons build the interpretation one physical quantity at a time.
