---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# LLM agents (MCP)

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
petrophys-mcp --help
```

`petrophys-mcp` (or `python -m petrophys.mcp`) serves over standard input and output; MCP clients start it for you.

## Connect a client

Give the server the folders that hold your wells with `--root` (repeatable). It reads and writes only inside them; the bundled sample wells are always readable.

::::{tab-set}

:::{tab-item} Claude Code
```bash
claude mcp add petrophys -- petrophys-mcp --root /path/to/wells
```
:::

:::{tab-item} Claude Desktop
In `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "petrophys": {
      "command": "petrophys-mcp",
      "args": ["--root", "/path/to/wells"]
    }
  }
}
```
:::

:::{tab-item} Cursor and VS Code
Cursor uses `.cursor/mcp.json` with the same content as Claude Desktop. VS Code uses `.vscode/mcp.json` with the key `servers`:

```json
{
  "servers": {
    "petrophys": {
      "command": "petrophys-mcp",
      "args": ["--root", "/path/to/wells"]
    }
  }
}
```
:::

::::

If `petrophys-mcp` is not on the client's `PATH`, use the Python that has Petrophys installed:
`"command": "/path/to/venv/bin/python", "args": ["-m", "petrophys.mcp", "--root", "/path/to/wells"]`.

Then ask, for example: *"Load the sample well 30_6-5, interpret it with Rw = 0.07 ohm.m at 230 degF, and summarize net pay in the Brent Group."*

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

**Interpret** with the Rw the user supplied, summarizing the Brent window:

```{code-cell} python
run = (await call("interpret_well", well_id="30_6-5",
                  settings={"rw": 0.07, "rw_meas_temp_f": 230},  # Rw (ohm.m) at 230 degF
                  summary_top=2841, summary_bottom=2958)).structured_content
print(run["interpretation_id"], "|", run["status"], "|", run["resolved"])
print("net thickness:", run["net_thickness"])
print(len(run["skipped_steps"]), "skipped steps and", len(run["warnings"]),
      "warnings, each with a reason")
```

A bad setting fails before anything runs, with a message the model can act on:

```{code-cell} python
bad = await call("interpret_well", well_id="30_6-5",
                 settings={"rw": 0.07, "hc_type": "gass"})
print(bad.content[0].text)
```

**Summarize** the Brent window:

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
| Whole-well totals | ask for a reservoir window (`summary_top`/`summary_bottom`); shallow overburden can pass default cutoffs |
| File access | keep `--root` to the folders the model may use; exports overwrite files of the same name there |
| `The Petrophys MCP server needs the MCP SDK` | `pip install "petrophys[mcp]"` (Python 3.10+) |
| `... is outside the folders this server may access` | add the folder with `--root` or `PETROPHYS_MCP_ROOTS` (`:`-separated on Linux and macOS, `;` on Windows) |
| The client shows no Petrophys tools | run the configured command in a terminal: it should wait without errors. Logs go to standard error, which clients show in their MCP logs |

## Go deeper

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Petrophys for LLM agents
:link: tutorials/part_b/20_mcp_server
:link-type: doc
A full session: load, interpret, explain, summarize, plot, export.
:::

:::{grid-item-card} Generate LLM context
:link: tutorials/part_b/19_llm_context_generation
:link-type: doc
Give a model computed JSON instead of tool access.
:::

:::{grid-item-card} The workflow as a glass box
:link: tutorials/part_b/10_dynamic_workflow_glass_box
:link-type: doc
The trace a model reads through `explain_step`.
:::

:::{grid-item-card} Complete tour
:link: tutorials/complete_tour
:link-type: doc
Every part of the library on real wells.
:::
::::
