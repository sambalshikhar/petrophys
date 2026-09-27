"""Model Context Protocol server exposing Petrophys to LLM clients.

Run ``petrophys-mcp`` (or ``python -m petrophys.mcp``) and register it with an
MCP client such as Claude Desktop, Claude Code, Cursor or VS Code. The server
keeps loaded wells and interpretation results in memory for the session and
returns compact JSON summaries; plots are returned as PNG images.
"""

from __future__ import annotations

import io
import json
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, Dict, Iterator, List, Literal, Optional, Sequence

import numpy as np
from pydantic import Field

from petrophys.mcp.serialize import (
    curve_statistics,
    decimated_values,
    depth_mask,
    interpretation_summary,
    json_object,
    json_safe,
    step_detail,
    well_summary,
)
from petrophys.mcp.session import Session, SessionError

try:
    from mcp.server.mcpserver import Image, MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "The Petrophys MCP server needs the MCP SDK (Python 3.10+). "
        "Install it with: pip install 'petrophys[mcp]'"
    ) from exc

SERVER_INSTRUCTIONS = """\
Petrophys interprets well logs (LAS, DLIS, CSV) into clay volume, porosity,
water saturation, permeability and net pay, recording every decision.

Typical session:
1. load_well(path) or load_sample_well(name); read the returned curve list.
2. interpret_well(well_id, settings) - pass values the logs cannot provide,
   above all formation water resistivity rw (ohm.m). Never invent rw: ask the
   user, or use a value documented in the file header or report.
3. Read the summary: status, warnings, skipped_steps (each with a reason) and
   key_results. Use explain_step for the method and parameters of a step,
   curve_statistics or summarize_zones for intervals, plot_well for a figure.
4. Report assumptions and data-quality warnings alongside results.

Settings are validated: an invalid value returns the accepted values.
interpretation_settings lists every setting with its unit and default.
Depths are in the well's depth unit (see depth_unit in every response).
"""

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
LOCAL_STATE = ToolAnnotations(
    read_only_hint=False, destructive_hint=False, open_world_hint=False
)

WellId = Annotated[
    str, Field(description="Id returned by load_well or load_sample_well.")
]
InterpretationId = Annotated[
    str, Field(description="Id returned by interpret_well, e.g. '30_6-5/run-1'.")
]
Top = Annotated[
    Optional[float],
    Field(description="Top of the depth window, in the well's depth unit."),
]
Bottom = Annotated[
    Optional[float],
    Field(description="Base of the depth window, in the well's depth unit."),
]


@contextmanager
def _captured_warnings() -> Iterator[List[str]]:
    """Collect warning messages (e.g. DataQualityWarning) raised inside the block."""
    messages: List[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield messages
    for item in caught:
        if issubclass(item.category, (UserWarning, RuntimeWarning)):
            text = str(item.message)
            if text not in messages:
                messages.append(text)


@contextmanager
def _tool_errors() -> Iterator[None]:
    """Turn user-facing library errors into MCP tool errors with their message."""
    try:
        yield
    except ToolError:
        raise
    except (
        SessionError,
        ValueError,
        TypeError,
        KeyError,
        FileNotFoundError,
        RuntimeError,
    ) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        raise ToolError(str(message)) from exc


def create_server(
    roots: Optional[Sequence[str | Path]] = None,
    *,
    session: Optional[Session] = None,
) -> MCPServer:
    """Build the Petrophys MCP server.

    Parameters
    ----------
    roots : sequence of path, optional
        Folders the server may read wells from and write exports to.
        Defaults to ``PETROPHYS_MCP_ROOTS`` or the current directory.
        Bundled sample wells are always readable.
    session : Session, optional
        Existing session to serve (mainly for tests).

    Returns
    -------
    MCPServer
        Call ``.run()`` to serve over stdio.
    """
    from petrophys.version import __version__

    state = session or Session(roots=[Path(root) for root in roots or []])
    server = MCPServer(
        "petrophys",
        title="Petrophys well-log interpretation",
        instructions=SERVER_INSTRUCTIONS,
        version=__version__,
        website_url="https://petrophys.netlify.app",
        log_level="WARNING",
    )

    # -- discovery -----------------------------------------------------------
    @server.tool(annotations=READ_ONLY)
    def list_sample_wells() -> dict[str, Any]:
        """List the sample wells and input templates bundled with Petrophys."""
        from petrophys.datasets import list_samples, sample

        wells = [
            name
            for name in list_samples()
            if Path(sample(name)).suffix.lower() in {".las", ".dlis"}
        ]
        return {
            "wells": wells,
            "templates": [name for name in list_samples() if name not in wells],
            "hint": "Load one with load_sample_well(name). '30_6-5' is a complete North Sea well.",
        }

    @server.tool(annotations=READ_ONLY)
    def interpretation_settings() -> dict[str, Any]:
        """Describe every interpretation setting: type, default, unit, accepted values."""
        from petrophys.workflow.schema import config_schema, workflow_steps

        return json_object(
            {"settings": config_schema(), "workflow_steps": workflow_steps()}
        )

    # -- loading -------------------------------------------------------------
    def _load(
        path: Path, source: str, well_id: Optional[str], **options: Any
    ) -> dict[str, Any]:
        import petrophys as pp

        with _captured_warnings() as messages:
            well = pp.read(path, **options)
        identifier = state.add_well(well, source, well_id)
        return {**well_summary(well, identifier, source), "warnings": messages}

    @server.tool(annotations=LOCAL_STATE)
    def load_well(
        path: Annotated[
            str,
            Field(
                description="LAS, DLIS, CSV or TSV file; relative paths start at the first allowed folder."
            ),
        ],
        depth_unit: Annotated[
            Optional[Literal["m", "ft"]],
            Field(description="Depth unit; required for CSV/TSV files."),
        ] = None,
        formations: Annotated[
            Optional[str],
            Field(description="Optional formation tops/intervals file (CSV/Excel)."),
        ] = None,
        lithology: Annotated[
            Optional[str],
            Field(
                description="Optional lithology intervals file (top, bottom, lithology)."
            ),
        ] = None,
        well_id: Annotated[
            Optional[str],
            Field(description="Id to store the well under; defaults to the well name."),
        ] = None,
    ) -> dict[str, Any]:
        """Load a well file and return its id, depth range, curves and header values."""
        with _tool_errors():
            resolved = state.resolve_path(path)
            options: Dict[str, Any] = {}
            if depth_unit is not None:
                options["depth_unit"] = depth_unit
            if formations is not None:
                options["formations"] = state.resolve_path(formations)
            if lithology is not None:
                options["lithology"] = state.resolve_path(lithology)
            return _load(resolved, str(resolved), well_id, **options)

    @server.tool(annotations=LOCAL_STATE)
    def load_sample_well(
        name: Annotated[
            str, Field(description="Sample name from list_sample_wells, e.g. '30_6-5'.")
        ],
        with_geology: Annotated[
            bool,
            Field(
                description="Attach the sample's formation and lithology files when bundled."
            ),
        ] = False,
    ) -> dict[str, Any]:
        """Load a bundled sample well (optionally with its formations and lithology)."""
        from petrophys.datasets import list_samples, sample

        with _tool_errors():
            path = Path(sample(name))
            options: Dict[str, Any] = {}
            if with_geology:
                for option, prefix in (
                    ("formations", "formations_"),
                    ("lithology", "lithology_"),
                ):
                    if prefix + name in list_samples():
                        options[option] = sample(prefix + name)
            return _load(path, f"sample:{name}", name, **options)

    @server.tool(annotations=READ_ONLY)
    def list_loaded() -> dict[str, Any]:
        """List wells and interpretations held in this session."""
        return {
            "wells": {well_id: state.sources[well_id] for well_id in state.wells},
            "interpretations": {
                run_id: {"well_id": owner, "status": result.status}
                for run_id, (owner, result) in state.interpretations.items()
            },
        }

    @server.tool(annotations=READ_ONLY)
    def describe_well(well_id: WellId) -> dict[str, Any]:
        """Return the header, depth range, curves (with units and coverage) and geology of a well."""
        with _tool_errors():
            return well_summary(state.well(well_id), well_id, state.sources[well_id])

    # -- data access -----------------------------------------------------------
    @server.tool(name="curve_statistics", annotations=READ_ONLY)
    def curve_statistics_tool(
        target_id: Annotated[
            str,
            Field(
                description="A well id, or an interpretation id to include calculated curves."
            ),
        ],
        curves: Annotated[
            Optional[List[str]],
            Field(description="Curve mnemonics; defaults to all curves."),
        ] = None,
        top: Top = None,
        bottom: Bottom = None,
    ) -> dict[str, Any]:
        """Summary statistics (valid count, min, P10, median, mean, P90, max) of curves in a depth window."""
        with _tool_errors():
            well = state.target_well(target_id)
            mask = depth_mask(well, top, bottom)
            names = curves or list(well.curves)
            stats: Dict[str, Any] = {}
            missing = []
            for name in names:
                curve = well.get(name)
                if curve is None:
                    missing.append(name)
                    continue
                stats[curve.mnemonic] = {
                    "unit": curve.unit,
                    **curve_statistics(np.asarray(curve.data)[mask]),
                }
            return json_object(
                {
                    "target_id": target_id,
                    "depth_unit": well.depth_unit,
                    "top": top,
                    "bottom": bottom,
                    "statistics": stats,
                    "missing_curves": missing,
                }
            )

    @server.tool(annotations=READ_ONLY)
    def curve_values(
        target_id: Annotated[
            str,
            Field(
                description="A well id, or an interpretation id to include calculated curves."
            ),
        ],
        curves: Annotated[
            List[str], Field(description="Curve mnemonics, e.g. ['GR', 'PHIE', 'SW'].")
        ],
        top: Top = None,
        bottom: Bottom = None,
        max_points: Annotated[
            int,
            Field(
                ge=10,
                le=2000,
                description="Maximum number of depth samples returned; the window is evenly decimated.",
            ),
        ] = 200,
    ) -> dict[str, Any]:
        """Curve values versus depth in a window, decimated to at most max_points samples."""
        with _tool_errors():
            well = state.target_well(target_id)
            return {
                "target_id": target_id,
                **decimated_values(
                    well, curves, depth_mask(well, top, bottom), max_points
                ),
            }

    # -- interpretation ----------------------------------------------------------
    @server.tool(annotations=LOCAL_STATE)
    def interpret_well(
        well_id: WellId,
        settings: Annotated[
            Optional[Dict[str, Any]],
            Field(
                description=(
                    "Interpretation settings, e.g. {'rw': 0.05, 'bht_f': 210, 'sw_model': 'auto'}. "
                    "See interpretation_settings for every name, unit and accepted value."
                )
            ),
        ] = None,
        summary_top: Annotated[
            Optional[float],
            Field(
                description="Top of the interval used for key_results (e.g. the reservoir)."
            ),
        ] = None,
        summary_bottom: Annotated[
            Optional[float],
            Field(description="Base of the interval used for key_results."),
        ] = None,
    ) -> dict[str, Any]:
        """Run the full interpretation on a loaded well and summarize the result.

        The input well is not modified. The summary lists the settings used
        (including resolved matrix and lithology), executed steps, skipped
        steps with reasons, warnings, key curve statistics and net thickness.
        """
        import petrophys as pp

        with _tool_errors():
            well = state.well(well_id)
            with _captured_warnings() as messages:
                result = pp.interpret(well, **(settings or {}))
            run_id = state.add_interpretation(well_id, result)
            return interpretation_summary(
                result,
                run_id,
                well_id,
                top=summary_top,
                bottom=summary_bottom,
                captured_warnings=messages,
            )

    @server.tool(annotations=READ_ONLY)
    def explain_step(
        interpretation_id: InterpretationId,
        step: Annotated[
            str, Field(description="Step name, e.g. 'rw', 'porosity', 'saturation'.")
        ],
    ) -> dict[str, Any]:
        """Method, input and output curves, parameters and decisions of one workflow step."""
        with _tool_errors():
            return step_detail(state.interpretation(interpretation_id), step)

    @server.tool(annotations=READ_ONLY)
    def summarize_zones(
        interpretation_id: InterpretationId,
        formations: Annotated[
            Optional[str],
            Field(
                description="Formation tops file; defaults to formations attached when the well was loaded."
            ),
        ] = None,
        top: Annotated[
            Optional[float],
            Field(description="Instead of formations: top of a single zone."),
        ] = None,
        bottom: Annotated[
            Optional[float],
            Field(description="Instead of formations: base of a single zone."),
        ] = None,
    ) -> dict[str, Any]:
        """Per-zone gross, net rock/reservoir/pay thickness and average PHIE, SW, VCL, PERM."""
        import pandas as pd

        from petrophys.evaluation.summaries.formation_analysis import analyze_formations
        from petrophys.evaluation.summaries.zone_summary import calculate_zone_summary
        from petrophys.model.formation_tops import FormationTops
        from petrophys.model.intervals import Formation

        with _tool_errors():
            result = state.interpretation(interpretation_id)
            well = result.well
            if top is not None and bottom is not None:
                zone = calculate_zone_summary(
                    well, top, bottom, zone_name=f"{top:g}-{bottom:g}"
                )
                return json_object({"depth_unit": well.depth_unit, "zones": [zone]})
            if formations is not None:
                tops = FormationTops.from_csv(
                    state.resolve_path(formations), terminal_depth=well.bottom
                )
            else:
                records = (well.get_parameter("_geology_context") or {}).get(
                    "formations"
                )
                if not records:
                    raise ToolError(
                        "No formations are attached to this well. Pass formations=<file>, "
                        "or top and bottom for a single zone."
                    )
                table = pd.DataFrame(records)
                if "bottom_depth" in table and table["bottom_depth"].isna().all():
                    table = table.drop(columns="bottom_depth")
                tops = FormationTops.from_dataframe(
                    table, name_col="formation_name", terminal_depth=well.bottom
                )
            zones = [
                interval
                for interval in tops.intervals
                if isinstance(interval, Formation)
            ]
            analysis = analyze_formations(well, zones)
            return json_object(
                {
                    "depth_unit": well.depth_unit,
                    "zones": analysis.to_dataframe().to_dict("records"),
                    "total_gross": analysis.total_gross,
                    "total_net_pay": analysis.total_net_pay,
                }
            )

    # -- outputs ---------------------------------------------------------------
    @server.tool(annotations=READ_ONLY)
    def plot_well(
        target_id: Annotated[
            str,
            Field(
                description="A well id, or an interpretation id to plot calculated curves."
            ),
        ],
        top: Top = None,
        bottom: Bottom = None,
        tracks: Annotated[
            Optional[List[str]],
            Field(
                description="Track names to show, e.g. ['GR', 'RESISTIVITY']; default all."
            ),
        ] = None,
        template: Annotated[
            str,
            Field(
                description="'auto', 'raw', 'petrophysics', 'minerals' or 'comprehensive'."
            ),
        ] = "auto",
        dpi: Annotated[int, Field(ge=50, le=200, description="Image resolution.")] = 90,
    ) -> Image:
        """Render a multi-track log plot of a depth window as a PNG image."""
        import petrophys as pp

        with _tool_errors():
            well = state.target_well(target_id)
            depth_range = None
            if top is not None or bottom is not None:
                start = top if top is not None else well.top
                end = bottom if bottom is not None else well.bottom
                if start is None or end is None:
                    raise ToolError("The well has no depth range to plot.")
                depth_range = (float(start), float(end))
            with _captured_warnings():
                viewer = pp.plot(
                    well, depth_range=depth_range, tracks=tracks, template=template
                )
            try:
                if viewer.fig is None:
                    raise ToolError("The plot could not be drawn.")
                buffer = io.BytesIO()
                viewer.fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
            finally:
                viewer.close()
            return Image(data=buffer.getvalue(), format="png")

    @server.tool(
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=True, open_world_hint=False
        )
    )
    def export_results(
        interpretation_id: InterpretationId,
        path: Annotated[
            str,
            Field(
                description="Output file inside an allowed folder; overwritten if it exists."
            ),
        ],
        format: Annotated[
            Literal["las", "csv", "json"],
            Field(
                description="'las' (curves + settings used), 'csv' (curves) or 'json' (full trace)."
            ),
        ] = "las",
    ) -> dict[str, Any]:
        """Write an interpretation to LAS, CSV or JSON and return the file path."""
        with _tool_errors():
            result = state.interpretation(interpretation_id)
            output = state.resolve_path(path, must_exist=False)
            output.parent.mkdir(parents=True, exist_ok=True)
            if format == "las":
                result.to_las(output)
            elif format == "csv":
                result.well.to_csv(output)
            else:
                result.to_json(output)
            return {
                "path": str(output),
                "format": format,
                "bytes": output.stat().st_size,
            }

    # -- resources and prompts -----------------------------------------------------
    @server.resource(
        "petrophys://guide", mime_type="text/markdown", title="How to use Petrophys"
    )
    def guide() -> str:
        """Usage guide for agents: workflow, conventions and pitfalls."""
        return "# Petrophys MCP guide\n\n" + SERVER_INSTRUCTIONS

    @server.resource(
        "petrophys://settings",
        mime_type="application/json",
        title="Interpretation settings",
    )
    def settings_resource() -> str:
        """Every interpretation setting with type, default, description and accepted values."""
        from petrophys.workflow.schema import config_schema

        return json.dumps(json_safe(config_schema()), indent=2)

    @server.resource(
        "petrophys://workflow-steps",
        mime_type="application/json",
        title="Workflow steps",
    )
    def steps_resource() -> str:
        """The workflow steps in execution order with a description each."""
        from petrophys.workflow.schema import workflow_steps

        return json.dumps(workflow_steps(), indent=2)

    @server.prompt(title="Interpret a well and write a report")
    def interpretation_report(
        well: Annotated[
            str,
            Field(
                description="Path to a well file, or a sample name such as '30_6-5'."
            ),
        ],
        rw: Annotated[
            Optional[str],
            Field(description="Formation water resistivity in ohm.m, if known."),
        ] = None,
    ) -> str:
        """Guide the model through loading, interpreting and reporting on a well."""
        rw_text = (
            f"Use rw={rw} ohm.m."
            if rw
            else "Ask me for the formation water resistivity (rw) before interpreting; do not assume one."
        )
        return (
            f"Interpret the well '{well}' with the Petrophys tools.\n"
            "1. Load it (load_sample_well for a sample name, otherwise load_well) and review the curves.\n"
            f"2. {rw_text}\n"
            "3. Run interpret_well, then check status, warnings and skipped_steps.\n"
            "4. Identify the reservoir interval(s) and use summarize_zones or curve_statistics there.\n"
            "5. Show a plot_well figure of the main reservoir.\n"
            "6. Write a short report: data used, settings and assumptions, key results "
            "(porosity, water saturation, net pay), data-quality warnings, and recommended next steps."
        )

    return server


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point: ``petrophys-mcp [--root DIR ...]``."""
    import argparse

    import matplotlib

    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(
        prog="petrophys-mcp",
        description="Serve Petrophys to MCP clients (Claude Desktop, Claude Code, Cursor, ...) over stdio.",
    )
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        metavar="DIR",
        help="Folder the server may read wells from and write exports to (repeatable). "
        "Defaults to $PETROPHYS_MCP_ROOTS or the current directory.",
    )
    args = parser.parse_args(argv)
    create_server(args.root).run()
    return 0
