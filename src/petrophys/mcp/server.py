"""Model Context Protocol server exposing Petrophys to LLM clients.

Run ``petrophys-mcp`` (or ``python -m petrophys.mcp``) and register it with an
MCP client such as Claude Desktop, Claude Code, Cursor or VS Code. The server
keeps loaded wells and interpretation results in memory for the session and
returns compact JSON summaries; plots are returned as PNG images.
"""

from __future__ import annotations

import copy
import io
import json
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, Dict, Iterator, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import Field

from petrophys.mcp.serialize import (
    compact,
    curve_statistics,
    decimated_values,
    depth_mask,
    interpretation_summary,
    json_safe,
    step_detail,
    well_summary,
    workflow_summary,
)
from petrophys.mcp.session import Session, SessionError, default_calculation_profile

try:
    from mcp.server.mcpserver import Image, MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import CallToolResult, TextContent, ToolAnnotations
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "The Petrophys MCP server needs the MCP SDK (Python 3.10+). "
        "Install it with: pip install 'petrophys[mcp]'"
    ) from exc

SERVER_INSTRUCTIONS = """\
Petrophys interprets well logs (LAS, DLIS, CSV) into clay volume, porosity,
water saturation, permeability and net pay, recording every decision.

Typical session:
1. load_sample_well(name) or load_well(path). Read header_parameters (with
   units) and header_notes. For the user's own file, run check_data_quality.
2. interpret_well(well_id, settings, summary_top, summary_bottom). Rw drives
   saturation and net pay: use the user's rw (ohm.m) with rw_meas_temp_f
   (degF). If the user gave no Rw, ask for it; a header RW without a
   temperature is not reliable. Give the reservoir as summary_top/bottom.
3. Report the result's assumptions, review_notes and warnings with the
   numbers. Use summarize_zones (per formation, rank='formation' or 'group'),
   explain_step (method and parameters of a step) and plot_well.
More: interpret_wells compares several wells in one call; run_workflow runs
a recipe you specify step by step (quick looks, fixed parameters);
spectral_gamma reads K-Th-U logs; export_results writes LAS/CSV/JSON.
Prefer these summaries to curve_values, which returns decimated samples.
Depths are in each well's depth unit (depth_unit in every response).
"""

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
LOCAL_STATE = ToolAnnotations(
    read_only_hint=False, destructive_hint=False, open_world_hint=False
)

#: Return type of tools that answer with JSON: compact text plus structured content.
JsonReply = Annotated[CallToolResult, dict[str, Any]]

WellId = Annotated[
    str,
    Field(
        description="Id from load_well/load_sample_well. A bundled sample name such as "
        "'30_6-5' also works and is loaded automatically."
    ),
]
InterpretationId = Annotated[
    str,
    Field(
        description="Id returned by interpret_well, interpret_wells or run_workflow, "
        "e.g. '30_6-5/run-1'."
    ),
]
TargetId = Annotated[
    str,
    Field(
        description="A well id, or an interpretation id to include calculated curves "
        "(VCL, PHIE, SW, PERM, NET_PAY, ...)."
    ),
]
Top = Annotated[
    Optional[float],
    Field(description="Top of the depth window, in the well's depth unit."),
]
Bottom = Annotated[
    Optional[float],
    Field(description="Base of the depth window, in the well's depth unit."),
]
SummaryTop = Annotated[
    Optional[float],
    Field(description="Top of the reservoir window for the summary (well depth unit); "
                      "default: whole well."),
]
SummaryBottom = Annotated[
    Optional[float],
    Field(description="Base of the reservoir window for the summary."),
]
Settings = Annotated[
    Optional[Dict[str, Any]],
    Field(
        description="Interpretation settings by name, e.g. {'rw': 0.07, 'rw_meas_temp_f': 230, "
        "'lithology_domain': 'carbonate'}; see interpret_well and interpretation_settings."
    ),
]

#: Settings that name files or session objects rather than plain values.
_PATH_SETTINGS = ("lithology_csv", "lithology_intervals")

#: Most values (depths x curves, depth included) curve_values returns.
_MAX_CURVE_VALUES = 1000

#: Track templates of plot_well.
PlotTemplate = Literal["auto", "raw", "petrophysics", "minerals", "comprehensive", "geolink_full"]


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


def _lean_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Drop generated titles and collapse ``anyOf [X, null]`` to ``X`` (optional anyway)."""
    lean = copy.deepcopy(schema)
    lean.pop("title", None)
    for spec in lean.get("properties", {}).values():
        spec.pop("title", None)
        if "default" in spec and spec["default"] is None:
            spec.pop("default")
        options = spec.get("anyOf")
        if isinstance(options, list) and len(options) == 2:
            kept = [option for option in options if option.get("type") != "null"]
            if len(kept) == 1:
                spec.pop("anyOf")
                for key, value in kept[0].items():
                    spec.setdefault(key, value)
    return lean


class _PetrophysServer(MCPServer):
    """MCPServer that advertises lean input schemas (fewer tokens for every model turn)."""

    async def list_tools(self):  # type: ignore[override]
        tools = await super().list_tools()
        for tool in tools:
            if isinstance(tool.input_schema, dict):
                tool.input_schema = _lean_schema(tool.input_schema)
        return tools


def create_server(
    roots: Optional[Sequence[str | Path]] = None,
    *,
    session: Optional[Session] = None,
    calculation_profile: Optional[str] = None,
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
    calculation_profile : str, optional
        Calculation profile of interpretations whose settings do not set
        ``calculation_profile``, e.g. ``"geolink"`` to pin an
        evaluation harness to the eval package's calculations. Defaults to
        ``PETROPHYS_CALCULATION_PROFILE``, else the library default
        (``"current"``). A profile given in a tool's settings wins. The
        environment variable is read here, by the MCP server only; library
        calls such as :func:`petrophys.interpret` ignore it.

    Returns
    -------
    MCPServer
        Call ``.run()`` to serve over stdio.

    Raises
    ------
    ValueError
        If the calculation profile is not valid.
    """
    import petrophys as pp
    from petrophys.mcp import catalog, reports
    from petrophys.version import __version__

    default_profile = default_calculation_profile(calculation_profile)
    state = session or Session(roots=[Path(root) for root in roots or []])
    server = _PetrophysServer(
        "petrophys",
        title="Petrophys well-log interpretation",
        instructions=SERVER_INSTRUCTIONS,
        version=__version__,
        website_url="https://petrophys.netlify.app",
        log_level="WARNING",
    )

    # -- helpers ---------------------------------------------------------------
    def redacted(value: Any) -> Any:
        if isinstance(value, str):
            return state.redact(value)
        if isinstance(value, dict):
            return {key: redacted(item) for key, item in value.items()}
        if isinstance(value, list):
            return [redacted(item) for item in value]
        return value

    def reply(data: Dict[str, Any]) -> CallToolResult:
        """Compact JSON text for the model plus the same data as structured content."""
        clean = redacted(compact(data))
        text = json.dumps(clean, separators=(",", ":"), ensure_ascii=False)
        return CallToolResult(content=[TextContent(type="text", text=text)],
                              structured_content=clean)

    @contextmanager
    def tool_errors() -> Iterator[None]:
        """Turn user-facing library errors into tool errors, without server paths."""
        try:
            yield
        except ToolError as exc:
            raise ToolError(state.redact(str(exc))) from exc
        except (SessionError, ValueError, TypeError, LookupError, OSError,
                RuntimeError) as exc:
            message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
            raise ToolError(state.redact(str(message))) from exc

    def load_file(path: Path, source: str, well_id: Optional[str], **options: Any
                  ) -> Dict[str, Any]:
        with _captured_warnings() as messages:
            well = pp.read(path, **options)
        identifier = state.add_well(well, source, well_id)
        return {**well_summary(well, identifier, source), "warnings": messages}

    def load_sample(name: str, with_geology: bool = True,
                    well_id: Optional[str] = None) -> Dict[str, Any]:
        from petrophys.datasets import sample

        resolved = catalog.resolve_sample(name)
        options: Dict[str, Any] = {}
        if with_geology:
            options.update(catalog.sample_geology(resolved))
        summary = load_file(Path(sample(resolved)), f"sample:{resolved}", well_id or resolved,
                            **options)
        if resolved != name:
            summary["note"] = f"{name!r} is the bundled sample {resolved!r}."
        return summary

    def ensure_well(well_id: str) -> Tuple[str, Optional[str]]:
        """Id of a loaded well; a bundled sample name is loaded on first use."""
        if well_id in state.wells:
            return well_id, None
        same = [known for known in state.wells
                if catalog.normalized_name(known) == catalog.normalized_name(well_id)]
        if len(same) == 1:
            return same[0], None
        if well_id in state.interpretations:
            raise SessionError(
                f"{well_id!r} is an interpretation id; give the well id "
                f"{state.interpretations[well_id][0]!r}."
            )
        try:
            resolved = catalog.resolve_sample(well_id)
        except ValueError:
            state.well(well_id)  # raises with the known ids
            raise
        if resolved in state.wells:
            return resolved, None
        loaded = load_sample(resolved)
        return loaded["well_id"], (
            f"Loaded the bundled sample {resolved!r} (with its formations/lithology when "
            "bundled)."
        )

    def prepare_settings(settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if settings is None:
            settings = {}
        if not isinstance(settings, dict):
            raise ToolError("settings must be a JSON object such as {'rw': 0.07}.")
        prepared = dict(settings)
        if default_profile is not None:
            prepared.setdefault("calculation_profile", default_profile)
        for name in _PATH_SETTINGS:
            if isinstance(prepared.get(name), str):
                prepared[name] = str(state.resolve_path(prepared[name]))
        if prepared.get("formation_tops") is not None:
            raise ToolError(
                "formation_tops cannot be passed over MCP; attach tops when loading "
                "(load_well formations=<file>, or load_sample_well) and they are used "
                "automatically."
            )
        peers = prepared.get("peer_wells")
        if peers is not None:
            if not isinstance(peers, list) or not all(isinstance(item, str) for item in peers):
                raise ToolError("peer_wells must be a list of loaded well ids.")
            prepared["peer_wells"] = [state.well(ensure_well(item)[0]) for item in peers]
        return prepared

    def run_interpretation(well_id: str, settings: Dict[str, Any]) -> Tuple[str, Any, List[str]]:
        well = state.well(well_id)
        settings = dict(settings)
        # COSTA is the one bundled collection whose lithology needs a field
        # calibration. Apply that calibration when the agent loads a COSTA
        # sample or a LAS carrying the COSTA field header. Explicit settings
        # still win, so users can run a generic carbonate or clastic check.
        field = str(getattr(getattr(well, "header", None), "field", "") or "").strip().lower()
        if not field:
            field = str(well.get_parameter("FLD", "") or "").strip().lower()
        source = str(state.sources.get(well_id, "")).strip().lower()
        is_costa = field == "costa" or source.startswith("sample:") and Path(
            pp.sample(source.split(":", 1)[1])
        ).parent.name.casefold() == "costa"
        if is_costa:
            settings.setdefault("lithology_domain", "carbonate")
            settings.setdefault("carbonate_calibration", "costa")
        with _captured_warnings() as messages:
            result = pp.interpret(well, **settings)
        return state.add_interpretation(well_id, result), result, messages

    def new_curves(well_id: str, result: Any) -> List[str]:
        before = set(state.well(well_id).curves)
        return [name for name in result.well.curves if name not in before]

    # -- discovery -----------------------------------------------------------
    @server.tool(annotations=READ_ONLY)
    def list_sample_wells() -> JsonReply:
        """List the bundled sample wells by collection, with what each is good for.

        Also lists other bundled files (formation tops, lithology, core tables,
        input templates). Sample names ignore case and separators, so '30/6-5'
        and 'hw3' work in load_sample_well and as well ids.
        """
        return reply(catalog.sample_catalog())

    @server.tool(annotations=READ_ONLY)
    def interpretation_settings(
        names: Annotated[
            Optional[List[str]],
            Field(description="Settings to describe in full, e.g. ['rw', 'sw_cutoff']."),
        ] = None,
        group: Annotated[
            Optional[str],
            Field(description="A settings group to describe in full, e.g. 'Net-pay cutoffs', "
                              "'Matrix properties', 'Mud properties'."),
        ] = None,
    ) -> JsonReply:
        """Look up interpret_well settings (name, default, unit, accepted values).

        Without arguments: the commonly used settings with their defaults and
        a list of every setting by group. names=[...] or group='...' returns full
        detail. The key settings are already described in interpret_well; call
        this only for others.
        """
        with tool_errors():
            return reply(catalog.settings_listing(names, group))

    # -- loading -------------------------------------------------------------
    @server.tool(annotations=LOCAL_STATE)
    def load_well(
        path: Annotated[
            str,
            Field(
                description="LAS, DLIS, CSV or TSV file inside the allowed folders; relative "
                "paths start in the server's folder (e.g. 'my_well.las')."
            ),
        ],
        depth_unit: Annotated[
            Optional[Literal["m", "ft"]],
            Field(description="Depth unit; required for CSV/TSV files."),
        ] = None,
        formations: Annotated[
            Optional[str],
            Field(description="Formation tops/intervals file (CSV/Excel) to attach; enables "
                              "per-formation summaries and tops on plots."),
        ] = None,
        lithology: Annotated[
            Optional[str],
            Field(description="Lithology intervals file (top, bottom, lithology) to attach."),
        ] = None,
        well_id: Annotated[
            Optional[str],
            Field(description="Id to store the well under; defaults to the well name."),
        ] = None,
    ) -> JsonReply:
        """Load a well file and summarize it.

        Returns well_id, depth range and unit, header_parameters with units (BHT,
        TD, RW with RWT, RMF, ...), header_notes (e.g. RW without temperature),
        curves as rows [mnemonic, unit, valid %, valid top, valid bottom,
        description], attached formations [name, top, bottom, rank] and warnings.
        Next: check_data_quality, then interpret_well.
        """
        with tool_errors():
            resolved = state.resolve_path(path)
            options: Dict[str, Any] = {}
            if depth_unit is not None:
                options["depth_unit"] = depth_unit
            if formations is not None:
                options["formations"] = state.resolve_path(formations)
            if lithology is not None:
                options["lithology"] = state.resolve_path(lithology)
            return reply(load_file(resolved, state.display_path(resolved), well_id, **options))

    @server.tool(annotations=LOCAL_STATE)
    def load_sample_well(
        name: Annotated[
            str,
            Field(description="Sample well name from list_sample_wells, e.g. '30_6-5' "
                              "(or '30/6-5'), 'HW-3', '25_5-1'."),
        ],
        with_geology: Annotated[
            bool,
            Field(
                description="Attach the sample's bundled formation tops and lithology "
                "intervals (30_6-5: tops and lithology; COSTA HW-*: the KHARAIB-2 interval; "
                "1046123714: tops)."
            ),
        ] = True,
    ) -> JsonReply:
        """Load a bundled sample well; returns the same summary as load_well."""
        with tool_errors():
            return reply(load_sample(name, with_geology))

    @server.tool(annotations=READ_ONLY)
    def list_loaded() -> JsonReply:
        """List the wells and interpretation runs held in this session, with their ids."""
        return reply({
            "wells": {well_id: state.sources[well_id] for well_id in state.wells},
            "interpretations": {
                run_id: {"well_id": owner, "status": result.status,
                         "kind": "run_workflow" if hasattr(result, "workflow") else
                         "interpret_well"}
                for run_id, (owner, result) in state.interpretations.items()
            },
        })

    @server.tool(annotations=READ_ONLY)
    def describe_well(well_id: WellId) -> JsonReply:
        """Summarize a loaded well again (same content as load_well) without reloading it."""
        with tool_errors():
            identifier, note = ensure_well(well_id)
            summary = well_summary(state.well(identifier), identifier, state.sources[identifier])
            if note:
                summary["note"] = note
            return reply(summary)

    # -- data access and quality ---------------------------------------------------
    @server.tool(annotations=READ_ONLY)
    def check_data_quality(
        well_id: WellId,
        curves: Annotated[
            Optional[List[str]], Field(description="Curves to check; default all.")
        ] = None,
        top: Top = None,
        bottom: Bottom = None,
    ) -> JsonReply:
        """Data-quality report with a recommended fix for every problem.

        Per curve: implausible values (and likely unit or scale errors), coverage
        and gaps, held (stuck or padded) readings, values pinned at a tool limit,
        duplicate curves. Also bad-hole intervals (caliper vs bit size, DRHO),
        sparse core data and header problems. Problem curves come first by
        severity (high: unusable as is). Use it instead of reading raw values.
        """
        with tool_errors():
            identifier, note = ensure_well(well_id)
            report = reports.data_quality_report(
                state.well(identifier), curves=curves, top=top, bottom=bottom
            )
            report = {"well_id": identifier, **report}
            if note:
                report["note"] = note
            return reply(report)

    @server.tool(name="curve_statistics", annotations=READ_ONLY)
    def curve_statistics_tool(
        target_id: TargetId,
        curves: Annotated[
            Optional[List[str]],
            Field(description="Curve mnemonics; defaults to all curves."),
        ] = None,
        top: Top = None,
        bottom: Bottom = None,
    ) -> JsonReply:
        """Statistics of curves over a depth window: valid count, min, P10, median, mean, P90, max.

        The cheap way to characterise an interval. Use an interpretation id to
        include calculated curves. Defaults: all curves, whole well.
        """
        with tool_errors():
            if target_id not in state.interpretations:
                target_id = ensure_well(target_id)[0]
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
                values = np.asarray(curve.data)
                try:
                    numbers = values.astype(float)
                except (TypeError, ValueError):
                    continue
                stats[curve.mnemonic] = {"unit": curve.unit, **curve_statistics(numbers[mask])}
                stats[curve.mnemonic].pop("samples", None)
            data = {
                "target_id": target_id,
                "depth_unit": well.depth_unit,
                "top": top,
                "bottom": bottom,
                "samples_in_window": int(np.count_nonzero(mask)),
                "statistics": stats,
                "missing_curves": missing,
            }
            if missing:
                data["available_curves"] = list(well.curves)
            return reply(data)

    @server.tool(annotations=READ_ONLY)
    def curve_values(
        target_id: TargetId,
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
                description="Most depths returned; the window is evenly decimated.",
            ),
        ] = 200,
    ) -> JsonReply:
        """Curve values versus depth, decimated, for inspecting a short interval.

        At most max_points depths and 1000 values in total; 'decimated' and
        'note' say what you got. For QC use check_data_quality; for interval
        averages curve_statistics or summarize_zones.
        """
        with tool_errors():
            if target_id not in state.interpretations:
                target_id = ensure_well(target_id)[0]
            well = state.target_well(target_id)
            return reply({
                "target_id": target_id,
                **decimated_values(
                    well, curves, depth_mask(well, top, bottom), max_points,
                    max_values=_MAX_CURVE_VALUES
                ),
            })

    # -- interpretation ----------------------------------------------------------
    @server.tool(annotations=LOCAL_STATE)
    def interpret_well(
        well_id: WellId,
        settings: Settings = None,
        summary_top: SummaryTop = None,
        summary_bottom: SummaryBottom = None,
    ) -> JsonReply:
        """Run the full interpretation (petrophys.interpret) on a well and summarize it.

        Temperature, clay volume, porosity, minerals, Rw, water saturation,
        permeability and net pay, each method chosen from the data. Returns
        interpretation_id (for summarize_zones, plot_well, explain_step,
        export_results), assumptions (Rw value, temperature and source, matrix,
        models, cutoffs), review_notes (read them), net_thickness, pay_averages
        (over net-pay samples) and key_results for the summary window, new_curves,
        skipped steps and warnings.

        Key settings (others: interpretation_settings):
        - rw (ohm.m) with rw_meas_temp_f (degF): formation water resistivity and
          its temperature; take both from the user. Without rw the header RW is
          used (at RWT, else at the median formation temperature): ask the user.
          Or salinity_ppm.
        - bht_f (degF), td_ft (ft): bottom-hole temperature; default header BHT.
        - lithology_domain: clastic (default)|carbonate|evaporite|auto (set
          'carbonate' for carbonates); matrix_type: auto|sandstone|limestone|dolomite
          (default sandstone in the clastic domain, auto in the others).
        - sw_model: auto|archie|simandoux|indonesia|dual_water|waxman_smits;
          a, m, n: Archie parameters (default by lithology). dual_water needs
          rwb (ohm.m), waxman_smits cec (meq/100 g) and b_coeff; ask the user,
          as they are not estimated.
        - vcl_cutoff, phie_cutoff, sw_cutoff (v/v): net-pay cutoffs; defaults by
          lithology family: clastic VCL < 0.40, PHIE > 0.08, SW < 0.60;
          carbonate VCL < 0.30, PHIE > 0.045, SW < 0.70. Evaporite and bad-hole
          samples are not net reservoir or pay (the "geolink" calculation
          profile keeps VCL < 0.3, SW < 0.7 and counts them).
        - gr_min, gr_max (API): GR clean/shale (default 20/100).
        - apply_hc_correction, hc_type: oil|gas; skip_steps: e.g. ['permeability'].
        Use run_workflow for a recipe with given methods (e.g. a quick look) and
        interpret_wells to compare wells.
        """
        with tool_errors():
            identifier, note = ensure_well(well_id)
            run_id, result, messages = run_interpretation(identifier, prepare_settings(settings))
            summary = interpretation_summary(
                result, run_id, identifier, top=summary_top, bottom=summary_bottom,
                captured_warnings=messages, new_curves=new_curves(identifier, result),
            )
            if note:
                summary["note"] = note
            return reply(summary)

    @server.tool(annotations=LOCAL_STATE)
    def interpret_wells(
        wells: Annotated[
            List[str],
            Field(description="Well ids or bundled sample names, e.g. ['HW-3', 'HW-4', 'HW-5'] "
                              "(samples are loaded automatically)."),
        ],
        settings: Settings = None,
        rw_from_header: Annotated[
            bool,
            Field(description="Use each well's header RW at its header RWT. Wells whose header "
                              "has no RW or no RWT are not run (the row says why)."),
        ] = False,
        summary_top: SummaryTop = None,
        summary_bottom: SummaryBottom = None,
    ) -> JsonReply:
        """Interpret several wells with the same settings and compare them in one call.

        One row per well: interpretation_id, status, interval, the Rw used, gross,
        net pay, NTG, mean PHIE/SW/VCL over the interval and PHIE/SW averaged over
        net-pay samples; plus rankings by net pay and mean PHIE. Each well covers
        its whole logged interval unless summary_top/bottom are given (one window
        for all). Settings as in interpret_well.
        """
        with tool_errors():
            if not wells:
                raise ToolError("Give at least one well id or sample name.")
            if len(wells) > 30:
                raise ToolError("At most 30 wells per call.")
            base = prepare_settings(settings)
            if rw_from_header and "rw" in base:
                raise ToolError("Give rw in settings or rw_from_header=true, not both.")
            rows: List[Dict[str, Any]] = []
            units = set()
            for name in wells:
                row: Dict[str, Any] = {"well": name}
                try:
                    identifier, _ = ensure_well(name)
                    row["well"] = identifier
                    well = state.well(identifier)
                    well_settings = dict(base)
                    if rw_from_header:
                        header_rw = _header_rw(well)
                        well_settings.update(header_rw)
                    run_id, result, messages = run_interpretation(identifier, well_settings)
                    summary = interpretation_summary(
                        result, run_id, identifier, top=summary_top, bottom=summary_bottom,
                        captured_warnings=messages,
                    )
                    row.update(_well_row(summary, result.well, summary_top, summary_bottom))
                    if rw_from_header:
                        row["rw"] = (f"{header_rw['rw']:g} ohm.m at "
                                     f"{header_rw['rw_meas_temp_f']:g} degF (header RW and RWT)")
                    units.add(result.well.depth_unit)
                except (ToolError, SessionError, ValueError, TypeError, LookupError, OSError,
                        RuntimeError) as exc:
                    row.update({"status": "not_run", "reason": state.redact(str(exc))})
                rows.append(row)
            done = [row for row in rows if row.get("status") not in {None, "not_run"}]
            data: Dict[str, Any] = {"settings": json_safe(settings or {}),
                                    "rw_from_header": rw_from_header, "wells": rows}
            if done:
                data["ranking"] = {
                    "by_net_pay": [row["well"] for row in sorted(
                        done, key=lambda row: -(row.get("net_pay") or 0))],
                    "by_mean_PHIE": [row["well"] for row in sorted(
                        done, key=lambda row: -(row.get("mean_PHIE") or 0))],
                }
            notes = []
            if len(units) > 1:
                notes.append("Wells use different depth units; thicknesses are in each well's "
                             "depth_unit.")
            if summary_top is None and summary_bottom is None:
                notes.append("Each row covers the well's whole logged interval; check it is the "
                             "reservoir (use summary_top/summary_bottom or summarize_zones).")
            if notes:
                data["notes"] = notes
            return reply(data)

    @server.tool(annotations=LOCAL_STATE)
    def run_workflow(
        well_id: WellId,
        steps: Annotated[
            List[Dict[str, Any]],
            Field(
                description="Ordered steps, each {'kind': <step kind>, <parameters>} (optional "
                "'name'), e.g. [{'kind': 'clay_volume', 'method': 'gr', 'gr_clean': 20, "
                "'gr_shale': 110}, {'kind': 'porosity', 'method': 'density', "
                "'matrix_density': 2.65}, {'kind': 'formation_water', 'rw': 0.07, "
                "'rw_temp_f': 230}, {'kind': 'saturation', 'model': 'archie'}, "
                "{'kind': 'net_pay', 'vcl_cutoff': 0.35, 'phie_cutoff': 0.1, "
                "'sw_cutoff': 0.6}]."
            ),
        ],
        name: Annotated[
            Optional[str], Field(description="Name of the workflow, shown in results and exports.")
        ] = None,
        summary_top: SummaryTop = None,
        summary_bottom: SummaryBottom = None,
    ) -> JsonReply:
        """Run your own recipe of steps (petrophys.Workflow) on a well.

        Use it when the user gives methods and parameters (a quick look, a
        company recipe) instead of letting interpret_well choose. Steps run in
        order on the curves of earlier steps. Main kinds and parameters:
        - clay_volume: method gr|sp|neutron_density, gr_clean, gr_shale -> VCL
        - porosity: method density|neutron_density|neutron|sonic, matrix,
          matrix_density (g/cc) -> PHIT, PHIE
        - formation_water: rw (ohm.m) with rw_temp_f (degF), or rw='header' -> RW
        - saturation: model archie|simandoux|indonesia|..., a, m, n -> SW
        - net_pay: vcl_cutoff, phie_cutoff, sw_cutoff (v/v) -> NET_PAY
        - also temperature, permeability, mud_filtrate, bvw, mhi, rock_quality,
          spectral_gamma, interpret (any interpret_well setting).
        workflow_step_kinds lists every parameter. Returns interpretation_id
        (usable with summarize_zones, plot_well, explain_step, export_results),
        each step's status, net_thickness, pay_averages and key_results for the
        summary window.
        """
        from petrophys.workflow.custom import Workflow

        with tool_errors():
            identifier, note = ensure_well(well_id)
            definition = _workflow_definition(steps, name, prepare_settings, state)
            workflow = Workflow.from_dict(definition)
            with _captured_warnings() as messages:
                run = workflow.run(state.well(identifier), strict=False)
            run_id = state.add_interpretation(identifier, run)
            summary = workflow_summary(run, run_id, identifier, top=summary_top,
                                       bottom=summary_bottom, captured_warnings=messages)
            summary["definition"] = {"name": workflow.name,
                                     "steps": workflow.to_dict(include_version=False)["steps"]}
            if note:
                summary["note"] = note
            return reply(summary)

    @server.tool(annotations=READ_ONLY)
    def workflow_step_kinds(
        kind: Annotated[
            Optional[str],
            Field(description="A step kind for full detail (parameter descriptions, required "
                              "and produced curves), e.g. 'saturation'."),
        ] = None,
    ) -> JsonReply:
        """List run_workflow step kinds with their parameters and defaults."""
        with tool_errors():
            return reply(catalog.step_kinds(kind))

    @server.tool(annotations=READ_ONLY)
    def explain_step(
        interpretation_id: InterpretationId,
        step: Annotated[
            str,
            Field(description="Step name, e.g. 'rw', 'temperature', 'vclay', 'porosity', "
                              "'saturation', 'net_pay' (see executed_steps of the run)."),
        ],
    ) -> JsonReply:
        """Method, input and output curves, parameters and decisions of one step of a run.

        Use it to answer how a value was obtained (e.g. how Rw was temperature
        corrected, which saturation model ran where, which cutoffs were applied)
        or why a step was skipped.
        """
        with tool_errors():
            return reply(step_detail(state.interpretation(interpretation_id), step))

    @server.tool(annotations=READ_ONLY)
    def summarize_zones(
        interpretation_id: InterpretationId,
        formations: Annotated[
            Optional[str],
            Field(
                description="A formation tops file (CSV/Excel) inside the allowed folders. Omit "
                "it to use the formations attached at load time."
            ),
        ] = None,
        top: Annotated[
            Optional[float],
            Field(description="Instead of formations: top of a single zone (well depth unit)."),
        ] = None,
        bottom: Annotated[
            Optional[float],
            Field(description="Instead of formations: base of a single zone."),
        ] = None,
        rank: Annotated[
            Optional[Literal["formation", "group"]],
            Field(description="Summarize one level only: 'formation' or 'group'. Default: every "
                              "zone in the tops, groups and formations mixed."),
        ] = None,
        names: Annotated[
            Optional[List[str]],
            Field(description="Only zones whose name contains one of these (case-insensitive), "
                              "e.g. ['Etive', 'Ness']."),
        ] = None,
    ) -> JsonReply:
        """Per-zone table of a run: top, bottom, gross, net reservoir, net pay, NTG, and PHIE,
        Sw, VCL averaged over net-pay samples, permeability (geometric mean).

        Zones are the attached formations (or a tops file), or one window (top,
        bottom). Tops often mix groups and their formations: rank='formation' or
        'group' gives one level with totals; names=[...] picks zones.
        """
        from petrophys.evaluation.summaries.zone_summary import calculate_zone_summary

        with tool_errors():
            result = state.interpretation(interpretation_id)
            well = result.well
            if top is not None or bottom is not None:
                start = float(top) if top is not None else float(well.top)
                end = float(bottom) if bottom is not None else float(well.bottom)
                zone = calculate_zone_summary(well, start, end, zone_name=f"{start:g}-{end:g}")
                keys = ("zone_name", "top_depth", "bottom_depth", "gross_thickness",
                        "net_rock", "net_reservoir", "net_pay", "ntg", "avg_phie", "avg_sw",
                        "avg_vcl", "avg_perm", "n_pay_samples")
                record = {key: getattr(zone, key) for key in keys}
                return reply({
                    "depth_unit": well.depth_unit,
                    "zones": [record],
                    "averages_basis": "avg_phie/avg_sw/avg_vcl are means over net-pay samples; "
                                      "avg_perm is a geometric mean",
                })
            source: Any = None
            note = None
            filters = list(names or [])
            if formations is not None:
                attached = (well.get_parameter("_geology_context") or {}).get("formations")
                try:
                    source = state.resolve_path(formations)
                except SessionError:
                    if not attached:
                        raise
                    labels = [str(row.get("formation_name")) for row in attached]
                    word = formations.strip().lower()
                    if any(word in label.lower() for label in labels):
                        filters.append(formations)
                        note = (f"formations={formations!r} is not a file; used it to select "
                                "attached formations by name (use names= for that).")
                    elif word in {"formations", "formation", "attached", "default", "all",
                                  "tops", "groups"}:
                        note = (f"formations={formations!r} is not a file; used the attached "
                                "formations (omit formations for that).")
                    else:
                        raise
            table = reports.zone_table(well, formations=source, rank=rank, names=filters or None)
            if note:
                table["note"] = note
            return reply(table)

    @server.tool(annotations=READ_ONLY)
    def spectral_gamma(
        well_id: WellId,
        top: Top = None,
        bottom: Bottom = None,
    ) -> JsonReply:
        """Interpret spectral gamma ray (K, Th, U): Th/U depositional environment, clay index.

        Returns where spectral data exist, statistics of K, Th, U, the share of
        samples per Th/U class (< 2 marine reducing, 2-7 marine, > 7 continental
        oxidising), the class per formation (else depth bins), uranium-rich beds,
        a thorium clay index and Th/K, U/K when potassium exists. Sample 25_5-1
        has THOR and URAN.
        """
        with tool_errors():
            identifier, note = ensure_well(well_id)
            report = reports.spectral_gamma_report(state.well(identifier), top=top,
                                                   bottom=bottom)
            report = {"well_id": identifier, **report}
            if note:
                report["note"] = note
            return reply(report)

    # -- outputs ---------------------------------------------------------------
    @server.tool(annotations=READ_ONLY)
    def plot_well(
        target_id: TargetId,
        top: Top = None,
        bottom: Bottom = None,
        tracks: Annotated[
            Optional[List[str]],
            Field(
                description="Tracks to show, in order; short names work ('GR', 'RES', 'SW'). "
                "Default: all."
            ),
        ] = None,
        template: Annotated[
            PlotTemplate,
            Field(
                description="Track layout and its track names. auto (from the curves present): "
                "LITHOLOGY, GAMMA / HOLE, RESISTIVITY, DENSITY / NEUTRON, POROSITY, "
                "SATURATION, PERMEABILITY, FLAGS. raw: GR / SP / CALI, RESISTIVITY, "
                "NEUTRON / DENSITY. petrophysics: LITHOLOGY, GR / VCL, RESISTIVITY, "
                "NEUTRON / DENSITY / SONIC, POROSITY, SATURATION, PERMEABILITY, FLAGS. "
                "minerals: MINERAL VOLUMES, POROSITY / SW. comprehensive: all."
            ),
        ] = "auto",
        zone: Annotated[
            Optional[str],
            Field(description="Plot an attached formation by name instead of top/bottom, e.g. "
                              "'Etive' (case-insensitive, partial match)."),
        ] = None,
        dpi: Annotated[int, Field(ge=50, le=200, description="Image resolution.")] = 90,
    ) -> List[Union[Image, TextContent]]:
        """Render a multi-track log plot (PNG) of a depth window, with formation tops.

        Use an interpretation id to show calculated curves. A short JSON text
        after the image lists the window and tracks drawn.
        """
        from petrophys.visualization.log_viewer import LogViewer

        with tool_errors():
            if target_id not in state.interpretations:
                target_id = ensure_well(target_id)[0]
            well = state.target_well(target_id)
            label = None
            if zone:
                label, start, end = reports.find_zone(well, zone)
                margin = 0.05 * (end - start)
                top, bottom = start - margin, end + margin
            depth_range = None
            if top is not None or bottom is not None:
                first = top if top is not None else well.top
                last = bottom if bottom is not None else well.bottom
                if first is None or last is None:
                    raise ToolError("The well has no depth range to plot.")
                depth_range = (float(first), float(last))
            selected = None
            with _captured_warnings():
                if tracks:
                    layout = LogViewer(well, template_defaults=template)
                    available = [track.display_name for track in layout.template.tracks]
                    selected = reports.resolve_tracks(tracks, available)
                    if "DEPTH" in available and "DEPTH" not in selected:
                        selected.insert(0, "DEPTH")
                viewer = pp.plot(well, depth_range=depth_range, tracks=selected,
                                 template=template)
            try:
                if viewer.fig is None:
                    raise ToolError("The plot could not be drawn.")
                buffer = io.BytesIO()
                from petrophys.visualization import save_figure

                save_figure(viewer.fig, buffer, format="png", dpi=dpi)
                drawn = [track.display_name for track in viewer.template.tracks]
                info = {
                    "target_id": target_id,
                    "depth_unit": well.depth_unit,
                    "window": list(depth_range) if depth_range else [well.top, well.bottom],
                    "zone": label,
                    "tracks": drawn,
                    "omitted_tracks": list(getattr(viewer, "omitted_tracks", []) or []),
                    "warnings": [text for text in getattr(viewer, "warnings", []) or []
                                 if "not independently verified" not in text][:5],
                }
            finally:
                viewer.close()
            text = json.dumps(redacted(compact(info)), separators=(",", ":"), ensure_ascii=False)
            return [Image(data=buffer.getvalue(), format="png"),
                    TextContent(type="text", text=text)]

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
                description="Output file inside the allowed folders, e.g. 'exports/30_6-5.las'; "
                "overwritten if it exists."
            ),
        ],
        format: Annotated[
            Literal["las", "csv", "json"],
            Field(
                description="'las' (curves + settings used), 'csv' (curves) or 'json' (full trace)."
            ),
        ] = "las",
    ) -> JsonReply:
        """Write an interpretation or workflow run to LAS, CSV or JSON.

        Returns the file's path relative to the allowed folder, its size and
        the number of curves. Give the user this relative path.
        """
        with tool_errors():
            result = state.interpretation(interpretation_id)
            output = state.resolve_path(path, must_exist=False)
            output.parent.mkdir(parents=True, exist_ok=True)
            if format == "las":
                result.to_las(output)
            elif format == "csv":
                result.well.to_csv(output)
            else:
                result.to_json(output)
            size = output.stat().st_size
            folder = next((label for root, label in zip(state.roots, state.root_labels)
                           if output.is_relative_to(root)), None)
            return reply({
                "path": state.display_path(output),
                "folder": folder,
                "format": format,
                "bytes": size,
                "size": _human_size(size),
                "curves": len(result.well.curves),
            })

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
        rw_temperature_f: Annotated[
            Optional[str],
            Field(description="Temperature (degF) at which rw was measured, if known."),
        ] = None,
    ) -> str:
        """Guide the model through loading, interpreting and reporting on a well."""
        if rw and rw_temperature_f:
            rw_text = f"Use rw={rw} ohm.m with rw_meas_temp_f={rw_temperature_f}."
        elif rw:
            rw_text = (
                f"Use rw={rw} ohm.m; ask me at what temperature it was measured "
                "(rw_meas_temp_f) before interpreting."
            )
        else:
            rw_text = (
                "Ask me for the formation water resistivity (rw) and its temperature "
                "before interpreting; do not assume them."
            )
        return (
            f"Interpret the well '{well}' with the Petrophys tools.\n"
            "1. Load it (load_sample_well for a sample name, otherwise load_well), review the "
            "curves and header_notes, and run check_data_quality.\n"
            f"2. {rw_text}\n"
            "3. Run interpret_well with the reservoir as summary_top/summary_bottom, then check "
            "status, assumptions, review_notes, warnings and skipped_steps.\n"
            "4. Use summarize_zones (rank='formation') or curve_statistics for the reservoir "
            "interval(s).\n"
            "5. Show a plot_well figure of the main reservoir.\n"
            "6. Write a short report: data used and data-quality problems, settings and "
            "assumptions, key results (porosity, water saturation, net pay), and recommended "
            "next steps."
        )

    return server


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("bytes", "KB", "MB"):
        if value < 1024:
            return f"{value:.0f} {unit}" if unit == "bytes" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _header_rw(well: Any) -> Dict[str, float]:
    """rw and rw_meas_temp_f (degF) from a well's header RW and RWT."""
    from petrophys.model.units import temperature_to_fahrenheit

    def number(name: str) -> Optional[float]:
        try:
            value = float(well.get_parameter(name))
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    rw, rwt = number("RW"), number("RWT")
    if rw is None:
        raise ValueError("the header has no RW; give rw and rw_meas_temp_f in settings")
    if rwt is None:
        raise ValueError(
            f"the header RW ({rw:g}) has no temperature (RWT); give rw and rw_meas_temp_f in "
            "settings instead"
        )
    unit = well.get_parameter("RWT__UNIT")
    return {"rw": rw, "rw_meas_temp_f": float(temperature_to_fahrenheit(rwt, unit or None))}


def _well_row(summary: Dict[str, Any], well: Any, top: Optional[float],
              bottom: Optional[float]) -> Dict[str, Any]:
    """One interpret_wells row from an interpretation summary."""
    stats = summary.get("key_results", {})
    thickness = summary.get("net_thickness", {})
    pay = summary.get("pay_averages", {})
    gross = thickness.get("gross")
    net_pay = thickness.get("net_pay")
    rw = next((text for text in summary.get("assumptions", []) if text.startswith("Rw")), "")
    row = {
        "interpretation_id": summary["interpretation_id"],
        "status": summary["status"],
        "depth_unit": well.depth_unit,
        "interval": [top if top is not None else well.top,
                     bottom if bottom is not None else well.bottom],
        "rw": rw.split("; ")[0],
        "matrix": summary.get("resolved", {}).get("matrix_type"),
        "domain": summary.get("resolved", {}).get("lithology_domain"),
        "gross": gross,
        "net_pay": net_pay,
        "ntg": (net_pay / gross) if gross and net_pay is not None else None,
        "mean_PHIE": stats.get("PHIE", {}).get("mean"),
        "mean_SW": stats.get("SW", {}).get("mean"),
        "mean_VCL": stats.get("VCL", {}).get("mean"),
        "pay_avg_PHIE": pay.get("PHIE"),
        "pay_avg_SW": pay.get("SW"),
        "warnings": len(summary.get("warnings", [])),
    }
    notes = [note for note in summary.get("review_notes", []) if "logged interval" not in note]
    if notes:
        row["review_notes"] = notes
    return row


def _workflow_definition(
    steps: Any, name: Optional[str], prepare_settings: Any, state: Session
) -> Dict[str, Any]:
    """Turn run_workflow's step list into a :meth:`Workflow.from_dict` definition."""
    from petrophys.workflow.steps import _REGISTRY

    if not isinstance(steps, list) or not steps:
        raise ToolError(
            "steps must be a non-empty list such as [{'kind': 'clay_volume', 'method': 'gr'}]."
        )
    built = []
    for number, item in enumerate(steps, start=1):
        if not isinstance(item, dict) or "kind" not in item:
            raise ToolError(
                f"Step {number} must be an object with 'kind' (one of {', '.join(_REGISTRY)}) "
                "and its parameters."
            )
        kind = str(item["kind"]).strip()
        if kind not in _REGISTRY:
            import difflib

            close = difflib.get_close_matches(kind, list(_REGISTRY), n=1, cutoff=0.5)
            raise ToolError(
                f"Step {number}: unknown kind {kind!r}"
                + (f" (did you mean {close[0]!r}?)" if close else "")
                + f". Kinds: {', '.join(_REGISTRY)}. Only built-in steps are available here."
            )
        params = item.get("params")
        if not isinstance(params, dict):
            params = {key: value for key, value in item.items() if key not in {"kind", "name"}}
        params = dict(params)
        if kind == "interpret":
            params = prepare_settings(params)
        if kind == "formation_summary" and isinstance(params.get("formations"), str):
            params["formations"] = str(state.resolve_path(params["formations"]))
        built.append({"kind": kind, "name": item.get("name"), "params": params})
    return {"name": name or "Custom workflow", "steps": built}


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
