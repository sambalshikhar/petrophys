"""The MCP server exposes the interpretation workflow to LLM clients."""

import asyncio
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
    "curve_statistics",
    "curve_values",
    "interpret_well",
    "explain_step",
    "summarize_zones",
    "plot_well",
    "export_results",
}


def _run(root, script):
    """Run ``script(client)`` against a fresh in-process server rooted at ``root``."""

    async def main():
        async with Client(create_server([root])) as client:
            return await script(client)

    return asyncio.run(main())


def _data(result):
    assert not result.is_error, result.content[0].text
    return result.structured_content


def test_every_tool_is_described_for_the_model(tmp_path):
    async def script(client):
        return (await client.list_tools()).tools

    tools = _run(tmp_path, script)
    assert {tool.name for tool in tools} == EXPECTED_TOOLS
    for tool in tools:
        assert tool.description and len(tool.description) > 20, tool.name
        for name, schema in tool.input_schema.get("properties", {}).items():
            assert schema.get("description"), f"{tool.name}.{name} has no description"


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
    assert any(curve["mnemonic"] == "GR" for curve in loaded["curves"])
    assert summary["status"] == "interpreted"
    assert summary["settings_used"]["rw"] == 0.07
    assert summary["skipped_steps"]["hc_correction"].startswith(
        "apply_hc_correction=False"
    )
    assert 0.0 < summary["key_results"]["PHIE"]["median"] < 0.4
    assert step["parameters"]["rw"] == 0.07
    assert (tmp_path / "out" / "run.las").is_file()
    assert exported["format"] == "las"


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
        return bad_setting, unknown_well, outside

    bad_setting, unknown_well, outside = _run(tmp_path, script)
    assert bad_setting.is_error and "did you mean 'gas'" in bad_setting.content[0].text
    assert (
        unknown_well.is_error
        and "Known well ids: 30_6-5" in unknown_well.content[0].text
    )
    assert (
        outside.is_error
        and "outside the folders this server may access" in outside.content[0].text
    )


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
    assert values["points"] == 25 and values["decimated"]
    assert set(values["curves"]) == {"GR", "PHIE"} and values["missing_curves"] == [
        "NOPE"
    ]
    assert zone["zones"][0]["net_pay"] > 0
    assert not plot.is_error and plot.content[0].mime_type == "image/png"


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
