"""The MCP server answers a client end to end: tools, a sample well, an
interpretation, its trace, zone summaries, a plot, an export and refusals."""

import tempfile
import unittest
from pathlib import Path

from _support import HAS_MCP

if HAS_MCP:
    from mcp import Client

    from petrophys.mcp import create_server


@unittest.skipUnless(HAS_MCP, "the MCP server needs Python 3.10+ and pip install petrophys[mcp]")
class McpServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.server = create_server(roots=[self.folder.name])

    async def asyncTearDown(self):
        self.folder.cleanup()

    async def call(self, tool, **arguments):
        async with Client(self.server) as client:
            return await client.call_tool(tool, arguments)

    async def test_tools_are_listed_with_descriptions(self):
        async with Client(self.server) as client:
            tools = (await client.list_tools()).tools
        names = {tool.name for tool in tools}
        for name in ("load_sample_well", "load_well", "interpret_well", "explain_step",
                     "summarize_zones", "plot_well", "export_results", "interpretation_settings"):
            self.assertIn(name, names)
        self.assertTrue(all(tool.description for tool in tools))

    async def test_interpretation_end_to_end(self):
        well = await self.call("load_sample_well", name="30_6-5", with_geology=True)
        self.assertFalse(well.is_error, well.content)
        self.assertEqual(well.structured_content["well_id"], "30_6-5")

        run = await self.call("interpret_well", well_id="30_6-5", settings={"rw": 0.07, "rw_meas_temp_f": 230},
                              summary_top=2841, summary_bottom=2958)
        self.assertFalse(run.is_error, run.content)
        result = run.structured_content
        self.assertEqual(result["status"], "interpreted")
        self.assertGreater(result["net_thickness"]["net_pay"], 0)
        run_id = result["interpretation_id"]

        step = await self.call("explain_step", interpretation_id=run_id, step="rw")
        self.assertFalse(step.is_error, step.content)
        self.assertIn("method", step.structured_content)

        zones = await self.call("summarize_zones", interpretation_id=run_id)
        self.assertFalse(zones.is_error, zones.content)
        self.assertTrue(zones.structured_content["zones"])

        plot = await self.call("plot_well", target_id=run_id, top=2800, bottom=3000, dpi=60)
        self.assertFalse(plot.is_error, plot.content)
        self.assertEqual(plot.content[0].type, "image")

        export = await self.call("export_results", interpretation_id=run_id, path="exports/30_6-5.las")
        self.assertFalse(export.is_error, export.content)
        self.assertTrue((Path(self.folder.name) / "exports" / "30_6-5.las").is_file())

    async def test_bad_requests_are_refused_with_a_message(self):
        await self.call("load_sample_well", name="30_6-5")
        for tool, arguments in [("interpret_well", {"well_id": "30_6-5", "settings": {"hc_type": "gass"}}),
                                ("load_well", {"path": "/etc/passwd"}),
                                ("explain_step", {"interpretation_id": "30_6-5/run-9", "step": "rw"})]:
            with self.subTest(tool=tool):
                reply = await self.call(tool, **arguments)
                self.assertTrue(reply.is_error)
                self.assertTrue(reply.content[0].text.strip())


if __name__ == "__main__":
    unittest.main()
