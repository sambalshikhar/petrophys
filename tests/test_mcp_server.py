"""The MCP server exposes the interpretation workflow to LLM clients."""

import asyncio
import json
import shutil

import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402

import petrophys as pp  # noqa: E402
from petrophys.mcp import create_server  # noqa: E402

EXPECTED_TOOLS = {
    "list_sample_wells",
    "interpretation_settings",
    "load_well",
    "load_sample_well",
    "list_loaded",
    "describe_well",
    "check_data_quality",
    "curve_statistics",
    "curve_values",
    "interpret_well",
    "interpret_wells",
    "run_workflow",
    "workflow_step_kinds",
    "explain_step",
    "summarize_zones",
    "spectral_gamma",
    "plot_well",
    "export_results",
}

QUICK_LOOK = [
    {"kind": "clay_volume", "method": "gr", "gr_clean": 20, "gr_shale": 110},
    {"kind": "porosity", "method": "density", "matrix_density": 2.65},
    {"kind": "formation_water", "rw": 0.07, "rw_temp_f": 230},
    {"kind": "saturation", "model": "archie"},
    {"kind": "net_pay", "vcl_cutoff": 0.35, "phie_cutoff": 0.10, "sw_cutoff": 0.60},
]


def _run(root, script):
    """Run ``script(client)`` against a fresh in-process server rooted at ``root``."""

    async def main():
        async with Client(create_server([root])) as client:
            return await script(client)

    return asyncio.run(main())


def _data(result):
    assert not result.is_error, result.content[0].text
    return result.structured_content


def _error(result):
    assert result.is_error, result.content[0].text
    return result.content[0].text


def test_every_tool_is_described_for_the_model(tmp_path):
    async def script(client):
        return (await client.list_tools()).tools

    tools = _run(tmp_path, script)
    assert {tool.name for tool in tools} == EXPECTED_TOOLS
    for tool in tools:
        assert tool.description and len(tool.description) > 40, tool.name
        assert "title" not in tool.input_schema, tool.name
        for name, schema in tool.input_schema.get("properties", {}).items():
            assert schema.get("description"), f"{tool.name}.{name} has no description"
            assert "title" not in schema and "anyOf" not in schema, f"{tool.name}.{name}"
    described = {tool.name: tool.description for tool in tools}
    # The key settings, their units and the Rw rule are in interpret_well itself.
    for text in ("rw_meas_temp_f", "degF", "lithology_domain", "sw_cutoff", "ask the user"):
        assert text in described["interpret_well"], text
    # Plot track names per template are listed for the model.
    plot = next(tool for tool in tools if tool.name == "plot_well")
    assert "GR / VCL" in plot.input_schema["properties"]["template"]["description"]


def test_load_interpret_explain_and_export(tmp_path):
    async def script(client):
        loaded = _data(await client.call_tool("load_sample_well", {"name": "30_6-5"}))
        summary = _data(
            await client.call_tool(
                "interpret_well",
                {
                    "well_id": loaded["well_id"],
                    "settings": {"rw": 0.07},
                    "summary_top": 2841,
                    "summary_bottom": 2958,
                },
            )
        )
        step = _data(
            await client.call_tool(
                "explain_step",
                {"interpretation_id": summary["interpretation_id"], "step": "rw"},
            )
        )
        exported = _data(
            await client.call_tool(
                "export_results",
                {
                    "interpretation_id": summary["interpretation_id"],
                    "path": "out/run.las",
                },
            )
        )
        return loaded, summary, step, exported

    loaded, summary, step, exported = _run(tmp_path, script)
    assert loaded["well_id"] == "30_6-5"
    mnemonic = loaded["curve_columns"].index("mnemonic")
    assert any(row[mnemonic] == "GR" for row in loaded["curves"])
    assert summary["status"] == "interpreted"
    assert summary["settings_used"]["rw"] == 0.07
    assert summary["skipped_steps"]["hc_correction"].startswith(
        "apply_hc_correction=False"
    )
    assert 0.0 < summary["key_results"]["PHIE"]["median"] < 0.4
    assert step["parameters"]["rw"] == 0.07
    assert (tmp_path / "out" / "run.las").is_file()
    assert exported["format"] == "las"
    # rw without its temperature is flagged as an assumption to confirm.
    assert "ASSUMED" in summary["assumptions"][0]
    assert any("ASSUMED" in note for note in summary["review_notes"])


def test_responses_are_compact_json(tmp_path):
    async def script(client):
        return await client.call_tool("load_sample_well", {"name": "30_6-5"})

    result = _run(tmp_path, script)
    text = result.content[0].text
    assert "\n" not in text and ", " not in text.split('"curves"')[0][:200]
    assert json.loads(text) == result.structured_content
    # Header values carry their units; a degC BHT is converted for the degF settings.
    loaded = result.structured_content
    assert loaded["header_parameters"]["BHT"].startswith("126 degC (= 258.8 degF)")
    notes = " ".join(loaded["header_notes"])
    assert "Header RW = 0.07 ohm.m at RWT 230 degF" in notes
    assert "higher than RMF" not in notes          # RW 0.07 is below RMF 0.08


def test_brent_answer_with_assumptions(tmp_path):
    async def script(client):
        await client.call_tool("load_sample_well", {"name": "30/6-5"})
        brent = _data(await client.call_tool("interpret_well", {
            "well_id": "30_6-5",
            "settings": {"rw": 0.07, "rw_meas_temp_f": 230},
            "summary_top": 2841, "summary_bottom": 2958}))
        # The eval package's calculations, as an evaluation harness pins them.
        pinned = _data(await client.call_tool("interpret_well", {
            "well_id": "30_6-5",
            "settings": {"rw": 0.07, "rw_meas_temp_f": 230, "calculation_profile": "geolink"},
            "summary_top": 2841, "summary_bottom": 2958}))
        whole = _data(await client.call_tool("interpret_well", {
            "well_id": "30_6-5", "settings": {"rw": 0.07, "rw_meas_temp_f": 230}}))
        header = _data(await client.call_tool("interpret_well", {"well_id": "30_6-5"}))
        explicit = _data(await client.call_tool("interpret_well", {
            "well_id": "30_6-5", "settings": {"rw": 0.07, "rw_meas_temp_f": 230},
            "summary_top": 176.6, "summary_bottom": 3556.1}))
        reservoir_file = _data(await client.call_tool("interpret_well", {
            "well_id": "HW-3", "settings": {"lithology_domain": "carbonate"}}))
        costa_defaults = _data(await client.call_tool("interpret_well", {
            "well_id": "HW-3", "settings": {}}))
        return brent, pinned, whole, header, explicit, reservoir_file, costa_defaults

    brent, pinned, whole, header, explicit, reservoir_file, costa_defaults = _run(tmp_path, script)
    assert brent["net_thickness"]["net_pay"] == pytest.approx(23.0, abs=0.1)
    assert brent["pay_averages"]["PHIE"] == pytest.approx(0.211, abs=0.005)
    assert brent["assumptions"][0].startswith("Rw 0.07 ohm.m at 230 degF, given in settings")
    assert any(text.startswith("Net-pay cutoffs: clastic VCL < 0.4, PHIE > 0.08, SW < 0.6")
               for text in brent["assumptions"])
    assert any(text.startswith("Not net reservoir or pay: bad hole") for text in brent["assumptions"])
    # The eval package's calculations, as an evaluation harness pins them; 30/6-5
    # gets those of the package's main run.
    assert pinned["net_thickness"]["net_pay"] == pytest.approx(38.4, abs=0.1)
    assert pinned["pay_averages"]["PHIE"] == pytest.approx(0.202, abs=0.005)
    assert any(text.startswith("Net-pay cutoffs: VCL < 0.3") for text in pinned["assumptions"])
    assert pinned["settings_used"]["calculation_profile"] == "geolink"
    assert not any("whole logged interval" in note for note in brent["review_notes"])
    assert any("whole logged interval" in note for note in whole["review_notes"])
    # A window that spans the whole well is still flagged; a file holding one
    # zone (COSTA files are the KHARAIB-2 reservoir only) is not.
    assert any("logged interval" in note for note in explicit["review_notes"])
    assert not any("logged interval" in note for note in reservoir_file["review_notes"])
    assert "VCALC" in reservoir_file["new_curves"]
    assert "VILL" in costa_defaults["new_curves"]
    assert costa_defaults["settings_used"]["lithology_domain"] == "carbonate"
    assert costa_defaults["settings_used"]["carbonate_calibration"] == "costa"
    # Without rw the header RW is used at its header temperature, and said so.
    assert "LAS header RW" in header["assumptions"][0]
    assert "at 230 degF" in header["assumptions"][0]


def test_server_default_calculation_profile_and_override(tmp_path, monkeypatch):
    settings = {"rw": 0.07, "rw_meas_temp_f": 230}

    async def interpret(server, **extra):
        async with Client(server) as client:
            return _data(await client.call_tool("interpret_well", {
                "well_id": "30_6-5", "settings": {**settings, **extra}}))

    monkeypatch.delenv("PETROPHYS_CALCULATION_PROFILE", raising=False)
    pinned = create_server([tmp_path], calculation_profile="geolink")
    default = asyncio.run(interpret(pinned))
    override = asyncio.run(interpret(pinned, calculation_profile="current"))
    assert default["settings_used"]["calculation_profile"] == "geolink"
    assert any(text.startswith("Net-pay cutoffs: VCL < 0.3") for text in default["assumptions"])
    # A profile given in the tool settings wins over the server default.
    assert "calculation_profile" not in override["settings_used"]
    assert any(text.startswith("Net-pay cutoffs: clastic") for text in override["assumptions"])

    monkeypatch.setenv("PETROPHYS_CALCULATION_PROFILE", "GeoLink")
    from_env = asyncio.run(interpret(create_server([tmp_path])))
    assert from_env["settings_used"]["calculation_profile"] == "geolink"
    for value in ("may2026", "benchmark_rerun"):
        monkeypatch.setenv("PETROPHYS_CALCULATION_PROFILE", value)
        with pytest.raises(ValueError, match=f"PETROPHYS_CALCULATION_PROFILE='{value}'"):
            create_server([tmp_path])


def test_errors_are_actionable(tmp_path):
    async def script(client):
        await client.call_tool("load_sample_well", {"name": "30_6-5"})
        bad_setting = await client.call_tool(
            "interpret_well",
            {"well_id": "30_6-5", "settings": {"hc_type": "gass"}},
        )
        unknown_well = await client.call_tool("describe_well", {"well_id": "nope"})
        outside = await client.call_tool(
            "load_well", {"path": str(tmp_path.parent / "elsewhere.las")}
        )
        missing = await client.call_tool("load_well", {"path": "my_wel.las"})
        sneaky = await client.call_tool("interpret_well", {
            "well_id": "30_6-5", "settings": {"lithology_csv": "/etc/passwd"}})
        return bad_setting, unknown_well, outside, missing, sneaky

    (tmp_path / "my_well.las").write_text("~V\n")
    bad_setting, unknown_well, outside, missing, sneaky = _run(tmp_path, script)
    assert "did you mean 'gas'" in _error(bad_setting)
    assert "Known well ids: 30_6-5" in _error(unknown_well)
    text = _error(outside)
    assert "outside the folders this server may access" in text
    assert "Did you mean 'my_well.las'" in _error(missing)
    assert "outside the folders" in _error(sneaky)
    # Messages name folders by label, never by the server's absolute path.
    for result in (outside, missing, sneaky):
        assert str(tmp_path) not in result.content[0].text


def test_sample_names_are_normalized(tmp_path):
    async def script(client):
        a = _data(await client.call_tool("load_sample_well", {"name": "25/5-1"}))
        b = _data(await client.call_tool("load_sample_well", {"name": "hw3"}))
        miss = await client.call_tool("load_sample_well", {"name": "30/6-6"})
        # A sample name works as a well id and is loaded on first use.
        described = _data(await client.call_tool("describe_well", {"well_id": "HW-4"}))
        return a, b, miss, described

    a, b, miss, described = _run(tmp_path, script)
    assert a["well_id"] == "25_5-1" and b["well_id"] == "HW-3"
    assert b["formations"][0][0] == "KHARAIB-2"  # COSTA tops are attached
    assert "Closest: 30_6-5" in _error(miss)
    assert described["well_id"] == "HW-4" and "Loaded the bundled sample" in described["note"]


def test_files_curves_zones_and_plots(tmp_path):
    shutil.copy(pp.sample("30_6-5"), tmp_path / "my_well.las")

    async def script(client):
        loaded = _data(
            await client.call_tool(
                "load_well", {"path": "my_well.las", "well_id": "mine"}
            )
        )
        run = _data(
            await client.call_tool(
                "interpret_well", {"well_id": "mine", "settings": {"rw": 0.07}}
            )
        )
        values = _data(
            await client.call_tool(
                "curve_values",
                {
                    "target_id": run["interpretation_id"],
                    "curves": ["GR", "PHIE", "NOPE"],
                    "top": 2800,
                    "bottom": 3000,
                    "max_points": 25,
                },
            )
        )
        zone = _data(
            await client.call_tool(
                "summarize_zones",
                {
                    "interpretation_id": run["interpretation_id"],
                    "top": 2841,
                    "bottom": 2958,
                },
            )
        )
        plot = await client.call_tool(
            "plot_well",
            {"target_id": run["interpretation_id"], "top": 2800, "bottom": 3000},
        )
        return loaded, values, zone, plot

    loaded, values, zone, plot = _run(tmp_path, script)
    assert loaded["well_id"] == "mine"
    assert loaded["source"] == "my_well.las"  # relative to the allowed folder
    assert values["points"] == 25 and values["decimated"]
    assert set(values["curves"]) == {"GR", "PHIE"} and values["missing_curves"] == [
        "NOPE"
    ]
    assert zone["zones"][0]["net_pay"] > 0
    assert not plot.is_error and plot.content[0].mime_type == "image/png"


def test_curve_values_are_capped(tmp_path):
    curves = ["GR", "RHOB", "NPHI", "RDEP", "RMED", "RSHA", "DTC", "CALI", "BS", "SP"]

    async def script(client):
        return _data(await client.call_tool("curve_values", {
            "target_id": "30_6-5", "curves": curves, "max_points": 2000}))

    values = _run(tmp_path, script)
    total = values["points"] * (len(values["curves"]) + 1)
    assert total <= 1000 and values["decimated"]
    assert "Capped at" in values["note"]


def test_check_data_quality_reports_problems_and_fixes(tmp_path):
    shutil.copy(pp.sample("15_9-14"), tmp_path / "my_well.las")

    async def script(client):
        await client.call_tool("load_well", {"path": "my_well.las", "well_id": "mine"})
        return _data(await client.call_tool("check_data_quality", {"well_id": "mine"}))

    report = _run(tmp_path, script)
    by_curve = {entry["curve"]: entry for entry in report["problem_curves"]}
    assert report["problem_curves"][0]["curve"] == "DTE"
    assert by_curve["DTE"]["severity"] == "high" and by_curve["DTE"]["fixes"]
    assert any("identical to RSHA" in text for text in by_curve["RMED"]["issues"])
    assert any("base of the log" in text for text in by_curve["GR"]["issues"])
    assert report["bad_hole"]["flagged_%"] > 0
    assert "No BHT" in report["header_notes"][0]


def test_spectral_gamma(tmp_path):
    async def script(client):
        report = _data(await client.call_tool("spectral_gamma", {"well_id": "25/5-1"}))
        none = await client.call_tool("spectral_gamma", {"well_id": "30_6-5"})
        return report, none

    report, none = _run(tmp_path, script)
    assert set(report["inputs"]) == {"THORIUM", "URANIUM"}
    environment = report["th_u_environment"]
    assert sum(environment["share_%"].values()) == pytest.approx(100, abs=0.1)
    assert 2 < environment["median_th_u"] < 7
    first = environment["by_interval"][0]
    assert first["dominant"] == "marine reducing"
    assert first["top"] == pytest.approx(2905.7, abs=0.1)
    assert first["bottom"] == pytest.approx(2966.7, abs=0.5)
    assert any("held" in note for note in report["notes"])
    assert "no processed spectral gamma" in _error(none)


def test_run_workflow_is_a_registered_run(tmp_path):
    async def script(client):
        run = _data(await client.call_tool("run_workflow", {
            "well_id": "30_6-5", "steps": QUICK_LOOK, "name": "quick look",
            "summary_top": 2841, "summary_bottom": 2958}))
        run_id = run["interpretation_id"]
        step = _data(await client.call_tool(
            "explain_step", {"interpretation_id": run_id, "step": "formation_water"}))
        zones = _data(await client.call_tool(
            "summarize_zones", {"interpretation_id": run_id, "names": ["Etive"]}))
        exported = _data(await client.call_tool(
            "export_results", {"interpretation_id": run_id, "path": "ql.csv", "format": "csv"}))
        plot = await client.call_tool("plot_well", {"target_id": run_id, "zone": "Etive"})
        typo = await client.call_tool("run_workflow", {
            "well_id": "30_6-5", "steps": [{"kind": "porosty"}]})
        bad_param = await client.call_tool("run_workflow", {
            "well_id": "30_6-5", "steps": [{"kind": "porosity", "method": "densty"}]})
        imported = await client.call_tool("run_workflow", {
            "well_id": "30_6-5", "steps": [{"kind": "os:system"}]})
        kinds = _data(await client.call_tool("workflow_step_kinds", {"kind": "saturation"}))
        return run, step, zones, exported, plot, typo, bad_param, imported, kinds

    run, step, zones, exported, plot, typo, bad_param, imported, kinds = _run(tmp_path, script)
    assert run["status"] == "complete"
    assert run["net_thickness"]["net_pay"] == pytest.approx(20.1, abs=0.2)
    assert [entry["kind"] for entry in run["steps"]] == [item["kind"] for item in QUICK_LOOK]
    assert step["parameters"]["rw_used"] == 0.07
    assert zones["zones"][0]["Formation"] == "Etive Fm."
    assert exported["path"] == "ql.csv" and (tmp_path / "ql.csv").is_file()
    assert not plot.is_error
    assert "did you mean 'porosity'" in _error(typo)
    assert "did you mean 'density'" in _error(bad_param)
    assert "Only built-in steps" in _error(imported)
    assert "choices" in kinds["kinds"]["saturation"]["params"]["model"]


def test_interpret_wells_compares_in_one_call(tmp_path):
    async def script(client):
        costa = _data(await client.call_tool("interpret_wells", {
            "wells": ["HW-3", "HW-4", "HW-5"], "settings": {"lithology_domain": "carbonate"},
            "rw_from_header": True}))
        no_rwt = _data(await client.call_tool("interpret_wells", {
            "wells": ["30_6-5"], "rw_from_header": True}))
        zones = _data(await client.call_tool(
            "summarize_zones", {"interpretation_id": costa["wells"][0]["interpretation_id"]}))
        return costa, no_rwt, zones

    costa, no_rwt, zones = _run(tmp_path, script)
    assert costa["ranking"]["by_net_pay"][0] == "HW-3"
    first = costa["wells"][0]
    assert first["rw"] == "0.0682 ohm.m at 68 degF (header RW and RWT)"
    assert first["depth_unit"] == "ft" and first["net_pay"] > 100
    # 30/6-5's header carries RW with its temperature, so the header route runs.
    assert no_rwt["wells"][0]["status"] == "interpreted"
    assert zones["zones"][0]["Formation"] == "KHARAIB-2"


def test_zone_ranks_and_settings_listing(tmp_path):
    async def script(client):
        run = _data(await client.call_tool("interpret_well", {
            "well_id": "30_6-5", "settings": {"rw": 0.07, "rw_meas_temp_f": 230}}))
        run_id = run["interpretation_id"]
        mixed = _data(await client.call_tool("summarize_zones", {"interpretation_id": run_id}))
        formations = _data(await client.call_tool(
            "summarize_zones", {"interpretation_id": run_id, "rank": "formation"}))
        literal = _data(await client.call_tool(
            "summarize_zones", {"interpretation_id": run_id, "formations": "formations"}))
        compact = await client.call_tool("interpretation_settings", {})
        full = _data(await client.call_tool("interpretation_settings", {"names": ["sw_cutoff"]}))
        return mixed, formations, literal, compact, full

    mixed, formations, literal, compact, full = _run(tmp_path, script)
    assert "note" in mixed and "total" not in mixed
    assert {row["Rank"] for row in formations["zones"]} == {"formation"}
    assert formations["total"]["net_pay"] > 0
    assert "used the attached formations" in literal["note"]
    assert len(compact.content[0].text) < 7000
    names = [item["name"] for item in compact.structured_content["settings"]]
    assert {"rw", "rw_meas_temp_f", "bht_f", "sw_model", "sw_cutoff"} <= set(names)
    assert full["settings"][0]["group"].startswith("Net-pay cutoffs")


def test_plot_track_names_are_matched_loosely(tmp_path):
    async def script(client):
        good = await client.call_tool("plot_well", {
            "target_id": "30_6-5", "tracks": ["gr", "RES"], "template": "petrophysics",
            "top": 2800, "bottom": 2900, "dpi": 50})
        bad = await client.call_tool("plot_well", {
            "target_id": "30_6-5", "tracks": ["SPECTRAL_GAMMA"], "template": "raw"})
        return good, bad

    good, bad = _run(tmp_path, script)
    assert not good.is_error and good.content[0].mime_type == "image/png"
    info = json.loads(good.content[1].text)
    assert info["tracks"] == ["DEPTH", "GR / VCL", "RESISTIVITY"]
    assert "Valid tracks" in _error(bad) and "GR / SP / CALI" in _error(bad)


def test_resources_and_prompt(tmp_path):
    async def script(client):
        guide = await client.read_resource("petrophys://guide")
        settings = await client.read_resource("petrophys://settings")
        prompt = await client.get_prompt("interpretation_report", {"well": "30_6-5"})
        return guide, settings, prompt

    guide, settings, prompt = _run(tmp_path, script)
    assert "interpret_well" in guide.contents[0].text
    assert '"name": "rw"' in settings.contents[0].text
    assert "do not assume them" in prompt.messages[0].content.text


def test_documentation_lists_every_tool():
    import re
    from pathlib import Path

    # The guide builds its tool table from the running server at docs build
    # time, so every registered tool is listed; check it still does.
    page = Path(__file__).resolve().parents[1] / "docs" / "mcp.md"
    source = page.read_text()
    assert "execution_mode: force" in source
    assert re.search(r"await client\.list_tools\(\)", source)
