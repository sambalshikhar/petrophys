LLM agents (MCP)
================

Petrophys ships a `Model Context Protocol <https://modelcontextprotocol.io>`_
(MCP) server, so an LLM application such as Claude Desktop, Claude Code,
Cursor or VS Code can load wells, run interpretations, read the step trace,
summarize zones, plot logs and export results on your machine. The model
works with the same validated workflow and trace as the Python API; it does
not reimplement any petrophysics.

Install
-------

The server needs Python 3.10 or newer:

.. code-block:: bash

   python -m pip install "petrophys[mcp]"
   petrophys-mcp --help

``petrophys-mcp`` (or ``python -m petrophys.mcp``) serves over standard
input/output; MCP clients start it for you.

Connect a client
----------------

Give the server the folders that hold your well files with ``--root``
(repeatable). It can read and write only inside these folders; the sample
wells bundled with Petrophys are always readable.

**Claude Code**

.. code-block:: bash

   claude mcp add petrophys -- petrophys-mcp --root /path/to/wells

**Claude Desktop** (``claude_desktop_config.json``)

.. code-block:: json

   {
     "mcpServers": {
       "petrophys": {
         "command": "petrophys-mcp",
         "args": ["--root", "/path/to/wells"]
       }
     }
   }

**Cursor** (``.cursor/mcp.json``) and **VS Code** (``.vscode/mcp.json``, key
``servers`` instead of ``mcpServers``) use the same command and arguments.

If ``petrophys-mcp`` is not on the client's ``PATH``, give the full path of
the Python that has Petrophys installed instead:
``"command": "/path/to/venv/bin/python", "args": ["-m", "petrophys.mcp", "--root", "/path/to/wells"]``.

Then ask, for example: *"Load the sample well 30_6-5, interpret it with
Rw = 0.07 ohm.m, and summarize net pay in the Brent Group."*

What the model can do
---------------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Tool
     - Purpose
   * - ``list_sample_wells``
     - Bundled sample wells and input templates.
   * - ``load_well``
     - Load a LAS, DLIS, CSV or TSV file (optionally with formation and
       lithology files); returns a well id, depth range, curves and header
       values.
   * - ``load_sample_well``
     - Load a bundled sample, optionally with its formations and lithology.
   * - ``list_loaded``
     - Wells and interpretations held in the session.
   * - ``describe_well``
     - Header, depth range, curves with units and coverage, attached geology.
   * - ``curve_statistics``
     - Valid count, min, P10, median, mean, P90 and max of curves in a window.
   * - ``curve_values``
     - Curve values versus depth, decimated to at most ``max_points`` samples.
   * - ``interpretation_settings``
     - Every interpretation setting with type, unit, default and accepted
       values, and the workflow steps.
   * - ``interpret_well``
     - Run the full interpretation; returns status, settings used, executed
       and skipped steps (with reasons), warnings, key results and net
       thickness.
   * - ``explain_step``
     - Method, input and output curves, parameters and decisions of one step.
   * - ``summarize_zones``
     - Gross, net rock/reservoir/pay and averages per formation or window.
   * - ``plot_well``
     - Multi-track log plot of a depth window, returned as a PNG image.
   * - ``export_results``
     - Write an interpretation to LAS (with the settings used), CSV or JSON.

The server also provides the resources ``petrophys://guide`` (how to use the
tools), ``petrophys://settings`` and ``petrophys://workflow-steps``, and an
``interpretation_report`` prompt that walks the model from loading a well to
a written report.

Every response is compact JSON with the well's ``depth_unit``. Invalid
settings, unknown ids and paths outside the allowed folders return a tool
error with an actionable message, for example::

   Invalid DynamicWorkflowConfig:
     - hc_type='gass' is not valid; use one of: oil, gas (did you mean 'gas'?)

Data-quality problems found while loading or interpreting (such as a curve
in the wrong unit) are returned in ``warnings``.

Example session
---------------

A typical exchange, abridged. The model loads the sample and reads its curve
list, then interprets with the Rw the user provided:

.. code-block:: json

   {"tool": "interpret_well",
    "arguments": {"well_id": "30_6-5", "settings": {"rw": 0.07},
                  "summary_top": 2841, "summary_bottom": 2958}}

.. code-block:: json

   {"interpretation_id": "30_6-5/run-1",
    "status": "interpreted",
    "resolved": {"matrix_type": "sandstone", "lithology": "clastic"},
    "settings_used": {"rw": 0.07, "matrix_type": "sandstone", "lithology": "clastic"},
    "skipped_steps": {"hc_correction": "apply_hc_correction=False (hydrocarbon correction is opt-in)"},
    "net_thickness": {"net_pay": 82.8, "depth_unit": "m"},
    "warnings": ["..."]}

It then calls ``explain_step`` for ``rw`` to report how Rw was
temperature-corrected, ``summarize_zones`` for per-formation net pay, and
``plot_well`` for a figure of the reservoir.

Use from Python
---------------

The server can be driven in-process, which is also how its tests work:

.. code-block:: python

   import asyncio

   from mcp import Client
   from petrophys.mcp import create_server


   async def main():
       async with Client(create_server(roots=["."])) as client:
           well = await client.call_tool("load_sample_well", {"name": "30_6-5"})
           run = await client.call_tool(
               "interpret_well", {"well_id": "30_6-5", "settings": {"rw": 0.07}}
           )
           print(run.structured_content["status"])


   asyncio.run(main())

The :doc:`MCP tutorial <tutorials/part_b/20_mcp_server>` walks through a full
session with outputs.

Good practice
-------------

* Formation water resistivity (``rw``) usually cannot be derived from logs
  alone. The server instructs models to ask for it rather than assume one;
  give it in your request when you know it.
* Ask the model to report ``warnings`` and ``skipped_steps``: they show which
  results rest on assumptions or incomplete data.
* Keep ``--root`` to the folders you want the model to access. Exports
  overwrite files of the same name inside those folders.

Troubleshooting
---------------

``The Petrophys MCP server needs the MCP SDK``
   Install the extra: ``pip install "petrophys[mcp]"`` (Python 3.10+).

``... is outside the folders this server may access``
   Add the folder with ``--root`` or ``PETROPHYS_MCP_ROOTS`` (separate
   several folders with ``:`` on Linux/macOS or ``;`` on Windows).

The client shows no Petrophys tools
   Run the configured command in a terminal; it should wait for input
   without printing errors. Log messages go to standard error, which most
   clients show in their MCP logs.
