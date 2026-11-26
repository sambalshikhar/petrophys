"""Every public function over MCP: found by what it does, described with how to give each
input, and called on a well with each input read as the kind the library declares.

A synthetic well checks each kind against the library called directly; a sweep calls every
function the sample well 30_6-5 can feed and requires an answer or a reason, never a crash;
a client checks the three tools end to end."""

import dataclasses
import json
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

import _support  # noqa: F401 - import path and plotting backend
from _support import HAS_MCP

import matplotlib.pyplot as plt

import petrophys as pp
from petrophys import function_schema
from petrophys.mcp.functions import FunctionCallError, LibraryFunctions, _blocked, _lookup, _matching_curves
from petrophys.mcp.session import Session, SessionError

if HAS_MCP:
    from mcp import Client

    from petrophys.mcp import create_server

N = 200


def synthetic_well():
    """200 samples at 0.5 m with logs, an interpreted set, a % porosity, core plugs every 5 m."""
    depth = np.arange(1000.0, 1100.0, 0.5)
    x = np.linspace(0.0, 1.0, N)
    well = pp.Well(name="SYN")
    well.depth = depth
    well.depth_unit = "m"
    for name, values, unit in [
        ("GR", 20 + 100 * x, "gAPI"), ("VCL", 0.6 * x, "v/v"), ("PHIE", 0.30 - 0.2 * x, "v/v"),
        ("PHID", 0.32 - 0.2 * x, "v/v"), ("NPHI", 0.28 - 0.1 * x, "v/v"),
        ("PHI_PCT", 100 * (0.30 - 0.2 * x), "%"), ("SW", 0.2 + 0.7 * x, "v/v"),
        ("RDEP", 50 - 45 * x, "ohm.m"), ("THOR", 2 + 14 * x, "ppm"),
        ("D_UM", np.geomspace(5, 2000, N), "um"), ("BAD", (np.arange(N) % 17 == 0).astype(float), ""),
    ]:
        well.add_curve(name, values, unit)
    plugs = np.full(N, np.nan)
    plugs[::10] = (0.30 - 0.2 * x)[::10] + 0.01
    well.add_curve("PHIC", plugs, "v/v")
    well.add_curve("KH_CORE", np.where(np.isfinite(plugs), 10 ** (12 * plugs - 1), np.nan), "mD")
    return well


def same(a, b):
    """Two results of the library are the same: arrays to the last bit, NaN where NaN."""
    if dataclasses.is_dataclass(a):
        return all(same(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    return a == b


class LibraryFunctionKinds(unittest.TestCase):
    """Each kind of input against the library called directly, on a synthetic well."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.session = Session(roots=[self.root])
        self.well = synthetic_well()
        self.session.add_well(self.well, "synthetic", "SYN")
        self.lib = LibraryFunctions(self.session)

    def tearDown(self):
        self.folder.cleanup()

    def call(self, name, target="SYN", **arguments):
        options = {key: arguments.pop(key) for key in ("top", "bottom", "store_as") if key in arguments}
        return self.lib.call(name, target, arguments, **options)

    def test_a_curve_by_name_is_read_over_the_window_and_the_result_written_as_a_curve(self):
        calculate_bvw = _lookup("calculate_bvw")
        reply, images = self.call("calculate_bvw", phie="PHIE", sw="SW", top=1020, bottom=1040)
        self.assertEqual(images, [])
        inside = (self.well.depth >= 1020) & (self.well.depth <= 1040)
        self.assertEqual(reply["result"]["curve"], "CALCULATE_BVW")
        self.assertEqual(reply["result"]["valid"], int(inside.sum()))
        self.assertEqual(reply["inputs"], {"phie": "curve PHIE (v/v)", "sw": "curve SW (v/v)"})
        written = np.asarray(self.well["CALCULATE_BVW"].data)
        expected = calculate_bvw(self.well["PHIE"].data, self.well["SW"].data)
        self.assertTrue(np.array_equal(written[inside], expected[inside]))
        self.assertTrue(np.isnan(written[~inside]).all())
        again, _ = self.call("calculate_bvw", phie="PHIE", sw="SW")
        self.assertEqual(again["result"]["curve"], "CALCULATE_BVW_2")          # never overwrites

    def test_a_number_is_used_at_every_sample_and_store_as_names_the_curve(self):
        archie = _lookup("archie")
        reply, _ = self.call("archie", rt="RDEP", rw=0.05, phi="PHIE", store_as="sw archie")
        self.assertEqual(reply["result"]["curve"], "SW_ARCHIE")
        self.assertEqual(reply["inputs"]["rw"], "0.05 at every sample")
        expected = archie(rt=self.well["RDEP"].data, rw=0.05, phi=self.well["PHIE"].data)
        self.assertTrue(np.array_equal(self.well["SW_ARCHIE"].data, expected, equal_nan=True))

    def test_a_curve_in_another_unit_than_declared_is_refused(self):
        with self.assertRaisesRegex(FunctionCallError, r"phi takes v/v; PHI_PCT is in %"):
            self.call("archie", rt="RDEP", rw=0.05, phi="PHI_PCT")

    def test_a_samplewise_function_is_applied_at_each_sample(self):
        wentworth_class = _lookup("wentworth_class")
        reply, _ = self.call("wentworth_class", d_um="D_UM")
        labels = [wentworth_class(d) for d in self.well["D_UM"].data]
        shares = reply["result"]["labels_per_sample"]
        self.assertEqual(set(shares), set(labels))
        self.assertEqual(self.lib.call("wentworth_class", None, {"d_um": 300})[0]["result"], wentworth_class(300))

    def test_a_mask_is_a_curve_or_a_condition_and_nothing_else(self):
        thorium_clay_index = _lookup("thorium_clay_index")
        reply, _ = self.call("thorium_clay_index", thorium_ppm="THOR", mask="VCL < 0.3 and not BAD")
        vcl, bad = np.asarray(self.well["VCL"].data), np.asarray(self.well["BAD"].data) > 0.5
        direct = thorium_clay_index(self.well["THOR"].data, mask=(vcl < 0.3) & ~bad)
        self.assertTrue(same(self.session.result(reply["result_id"]), direct))
        by_curve, _ = self.call("thorium_clay_index", thorium_ppm="THOR", mask="BAD")
        self.assertTrue(same(self.session.result(by_curve["result_id"]), thorium_clay_index(self.well["THOR"].data, mask=bad)))
        for condition, message in [("__import__('os').system('true')", "compares curves"),
                                   ("VCL < 0.3 and NOPE > 1", "'NOPE'"), ("VCL + 1", "compares curves"),
                                   ("VCL <", "neither a curve nor a condition")]:
            with self.subTest(condition=condition), self.assertRaisesRegex(FunctionCallError, message):
                self.call("thorium_clay_index", thorium_ppm="THOR", mask=condition)

    def test_choices_are_checked_and_switches_read(self):
        total_porosity, should_use_sp = _lookup("total_porosity"), _lookup("should_use_sp")
        reply, _ = self.call("total_porosity", nphi="NPHI", phi_d="PHID", method="AVERAGE")
        direct = total_porosity(self.well["NPHI"].data, self.well["PHID"].data, method="average")
        self.assertTrue(same(self.well[reply["result"]["curve"]].data, direct))
        with self.assertRaisesRegex(FunctionCallError, "method is one of adaptive, average, rms, not 'median'"):
            self.call("total_porosity", nphi="NPHI", phi_d="PHID", method="median")
        reply, _ = self.lib.call("should_use_sp", None, {"mud_type": "wbm", "has_kcl": "false", "salinity_ppm": 30000})
        self.assertTrue(same(self.session.result(reply["result_id"]),
                             should_use_sp(mud_type="WBM", has_kcl=False, salinity_ppm=30000)))

    def test_the_well_given_is_a_copy_and_curves_the_function_adds_are_kept_once(self):
        before = set(self.well.curves)
        reply, _ = self.call("add_volumetric_pay_flag", fluid_type="gas")
        self.assertEqual(reply["inputs"]["fluid_type"], "'gas' (text, not a curve)")
        self.assertEqual(reply["result"]["curve"], "PAY_FLAG_VOLUMETRIC")
        self.assertEqual([c["name"] for c in reply["curves_added"]], ["PAY_FLAG_VOLUMETRIC"])
        self.assertEqual(set(self.well.curves) - before, {"PAY_FLAG_VOLUMETRIC"})

    def test_a_function_given_the_whole_well_gets_whole_curves_beside_it(self):
        reply, _ = self.call("mask_bad_hole_curves", bad_hole_flag="BAD", curves_to_mask=["GR"],
                             top=1020, bottom=1040)
        self.assertIn("reads the whole well", reply["note"])
        masked = self.session.wells[reply["wells_added"][0]["well_id"]]
        self.assertTrue(np.isnan(np.asarray(masked["GR"].data)[np.asarray(self.well["BAD"].data) > 0.5]).all())

    def test_a_returned_well_is_a_new_well_named_after_its_source(self):
        first, _ = self.call("standardize_curves")
        second, _ = self.call("standardize_curves")
        self.assertEqual(first["result"]["well_id"], "SYN-standardize_curves")
        self.assertEqual(second["result"]["well_id"], "SYN-standardize_curves-2")
        self.assertIn("SYN-standardize_curves", self.session.wells)

    def test_results_are_inputs_to_later_calls_whole_or_by_part(self):
        core, _ = self.call("core_data")
        self.assertEqual(core["result"]["type"], "CoreData")
        calibration, _ = self.call("calibrate_porosity", core=core["result_id"])
        self.assertEqual(calibration["inputs"]["core"], f"the result {core['result_id']}")
        transform, _ = self.call("fit_log_poro_perm_transform", core=core["result_id"], porosity_curve="PHIE")
        applied, _ = self.call("apply_poro_perm", transform=transform["result_id"], porosity_curve="PHIE")
        self.assertIn("PERM_CC", self.well.curves)
        self.assertTrue(applied["curves_added"])
        porosity, label = self.lib._session_object(f"{core['result_id']}.porosity", [])
        self.assertEqual(label, f"the result {core['result_id']}.porosity")
        self.assertEqual(len(porosity), int(np.isfinite(self.well["PHIC"].data).sum()))
        with self.assertRaisesRegex(FunctionCallError, "no part 'nope'; its parts: .*porosity"):
            self.call("calibrate_porosity", core=f"{core['result_id']}.nope")

    def test_json_becomes_the_declared_dataclass_or_table(self):
        reply, _ = self.call("analyze_interval", interval={"name": "A", "top_depth": 1020, "bottom_depth": 1040})
        self.assertEqual(reply["result"]["type"], "IntervalSummary")
        with self.assertRaisesRegex(FunctionCallError, "Interval takes the fields name, top_depth, bottom_depth"):
            self.call("analyze_interval", interval={"name": "A", "top": 1020})
        rows = [{"SAMPLE": 1, "PC": pc, "SW": sw, "POROSITY": 0.2, "PERMEABILITY": 150.0}
                for pc, sw in [(1, 1.0), (5, 0.6), (20, 0.35), (80, 0.2)]]
        reply, _ = self.lib.call("read_capillary_pressure", None, {"source": rows})
        self.assertEqual(reply["result"]["type"], "CapillaryPressureData")

    def test_files_are_read_and_written_only_inside_the_allowed_folders(self):
        shutil.copy(Path(pp.__file__).parent / "datasets" / "phillips" / "costa_core_pc.xlsx", self.root / "pc.xlsx")
        reply, _ = self.lib.call("read_capillary_pressure", None, {"source": "pc.xlsx"})
        self.assertEqual(reply["inputs"]["source"], "the file pc.xlsx")
        plot, images = self.call("pickett_plot", rw=0.05)
        self.assertEqual(len(images), 1)
        self.lib.call("save_figure", None, {"figure": f"{plot['result_id']}.fig", "filename": "pickett.png"})
        self.assertTrue((self.root / "pickett.png").is_file())
        outside = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        self.addCleanup(Path(outside.name).unlink)
        for name, arguments in [("read_capillary_pressure", {"source": outside.name}),
                                ("read_capillary_pressure", {"source": "../escape.xlsx"}),
                                ("save_figure", {"figure": f"{plot['result_id']}.fig", "filename": "../out.png"}),
                                ("read_curve_aliases", {"source": outside.name})]:
            with self.subTest(name=name, arguments=arguments), self.assertRaisesRegex(SessionError, "outside the folders"):
                self.lib.call(name, None, arguments)
        with self.assertRaisesRegex(FunctionCallError, "outside the folders"):
            self.call("calculate_porosity", matrix=outside.name)       # text naming a file outside

    def test_figures_come_back_as_png_and_are_closed(self):
        before = set(plt.get_fignums())
        reply, images = self.call("pickett_plot", rw=0.05, top=1000, bottom=1050)
        self.assertEqual(reply["result"], "figure: image 1")
        self.assertTrue(images[0].startswith(b"\x89PNG"))
        self.assertEqual(set(plt.get_fignums()), before)

    def test_inputs_only_code_can_give_are_named_and_refused(self):
        described = self.lib.describe("monte_carlo")
        self.assertIn("a Python function", described["not_over_mcp"])
        self.assertNotIn("example", described)
        with self.assertRaisesRegex(FunctionCallError, "cannot be called over MCP: function takes a Python function"):
            self.lib.call("monte_carlo", None, {})

    def test_bad_requests_say_what_to_send(self):
        cases = [
            (("calculate_bwv", "SYN", {}), {}, "Did you mean 'calculate_bvw'"),
            (("calculate_bvw", "SYN", {"phi": "PHIE", "sw": "SW"}), {}, r"not 'phi' \(did you mean 'phie'\?\)"),
            (("calculate_bvw", "SYN", {"sw": "SW"}), {}, r"needs phie \(a curve name on the well \(PHIE\)"),
            (("calculate_bvw", "SYN", {"phie": "PHIX", "sw": "SW"}), {}, "curves that fit phie: PHIE"),
            (("calculate_bvw", None, {"phie": 0.2, "sw": 0.5}), {"top": 1000}, "need a target_id"),
            (("calculate_bvw", "SYN", {"phie": "PHIE", "sw": "SW"}), {"top": 5000, "bottom": 6000}, "No samples between"),
            (("calculate_bvw", "SYN", ["PHIE", "SW"]), {}, "must be a JSON object"),
            (("calculate_bvw", "NOWELL", {}), {}, "Unknown well id 'NOWELL'"),
        ]
        for (name, target, arguments), options, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.lib.call(name, target, arguments, **options)
        reply, _ = self.lib.call("calculate_bvw", "SYN", '{"phie": "PHIE", "sw": "SW"}')
        self.assertEqual(reply["result"]["curve"], "CALCULATE_BVW")

    def test_searches_find_what_a_petrophysicist_means(self):
        queries = {
            "bulk volume water": "calculate_bvw", "pickett plot": "pickett_plot",
            "archie water saturation": "archie", "flow zone indicator": "calculate_fzi",
            "rock quality index": "calculate_rqi", "moveable hydrocarbon index": "calculate_mhi",
            "lorenz coefficient": "lorenz_coefficient", "wentworth grain size class": "wentworth_class",
            "thorium clay index": "thorium_clay_index", "rw from salinity": "rw_from_salinity",
            "simandoux": "simandoux", "net to gross": "calculate_net_gross", "density porosity": "density_porosity",
            "timur permeability": "timur", "cementation exponent from core": "archie_m",
            "normal compaction trend": "fit_normal_compaction_trend",
        }
        for query, wanted in queries.items():
            with self.subTest(query=query):
                found = [hit["name"] for hit in self.lib.find(query, limit=3)["found"]]
                self.assertIn(wanted, found)
        areas = self.lib.find()["areas"]
        self.assertEqual({area.split(".")[0] for area in areas},
                         {"interpretation", "evaluation", "preprocessing", "visualization"})
        only = self.lib.find("porosity", package="visualization", limit=30)["found"]
        self.assertTrue(only and all(hit["module"].startswith("visualization") for hit in only))
        with self.assertRaisesRegex(FunctionCallError, "package is one of"):
            self.lib.find("porosity", package="io")


class EveryPublicFunction(unittest.TestCase):
    """Generated over the whole library: every function described, every one the sample well
    can feed called, each answering with JSON or a reason."""

    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.session = Session(roots=[cls.folder.name])
        well = pp.read(pp.sample("30_6-5"))
        cls.session.add_well(well, "sample:30_6-5", "30_6-5")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = pp.interpret(well, rw=0.07, rw_meas_temp="230 degF")
        cls.run_id = cls.session.add_interpretation("30_6-5", result)
        cls.lib = LibraryFunctions(cls.session)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_every_function_is_described_with_how_to_give_each_input(self):
        functions = function_schema.public_functions()
        for function in functions:
            with self.subTest(function=function.__name__):
                described = self.lib.describe(function.__name__, self.run_id)
                json.dumps(described)
                self.assertTrue(all(p["give"] for p in described["parameters"]))
                self.assertTrue(("example" in described) != ("not_over_mcp" in described))

    def test_every_function_the_well_can_feed_answers_or_says_why(self):
        """Inputs from the well's own curves (by the standard curve each declares), the first
        choice, a condition for a mask; a function needing other inputs is left out."""
        target = self.session.target_well(self.run_id)
        ran, refused = [], []
        for function in function_schema.public_functions():
            described = function_schema.describe(function)
            if _blocked(function, described):
                continue
            arguments, feedable = {}, True
            for p in described["parameters"]:
                if not p["required"] or p["kind"] in ("well", "depth", "window"):
                    continue
                here = _matching_curves(target, p["curve"]) if p.get("curve") else []
                if p["kind"] in ("curve", "curve_name") and here:
                    arguments[p["name"]] = here[:1] if p.get("many") else here[0]
                elif p["kind"] == "choice":
                    arguments[p["name"]] = p["choices"][0]
                elif p["kind"] == "switch":
                    arguments[p["name"]] = True
                elif p["kind"] == "mask":
                    arguments[p["name"]] = "VCL < 0.5"
                else:
                    feedable = False
                    break
            if not feedable:
                continue
            before = set(plt.get_fignums())
            with self.subTest(function=function.__name__):
                try:
                    reply, _ = self.lib.call(function.__name__, self.run_id, arguments, top=2841, bottom=2958, dpi=50)
                    json.dumps(reply)
                    ran.append(function.__name__)
                except (FunctionCallError, SessionError) as exc:
                    self.assertTrue(str(exc).strip())
                    refused.append(function.__name__)
                self.assertEqual(set(plt.get_fignums()), before, "a figure was left open")
        # 150 of the library's functions ran on this well when this test was written; far
        # fewer means the binding broke, not the library.
        self.assertGreaterEqual(len(ran), 120, f"ran {len(ran)}; refused: {refused}")


@unittest.skipUnless(HAS_MCP, "the MCP server needs Python 3.10+ and pip install petrophys[mcp]")
class LibraryFunctionTools(unittest.IsolatedAsyncioTestCase):
    """The three tools through an MCP client."""

    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.server = create_server(roots=[self.folder.name])

    async def asyncTearDown(self):
        self.folder.cleanup()

    async def test_find_describe_and_call_end_to_end(self):
        async with Client(self.server) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            for name in ("find_functions", "describe_function", "call_function"):
                self.assertTrue(tools[name].description)
            self.assertTrue(tools["call_function"].annotations.destructive_hint)
            found = await client.call_tool("find_functions", {"query": "bulk volume water", "limit": 3})
            self.assertEqual(found.structured_content["found"][0]["name"], "calculate_bvw")
            described = await client.call_tool("describe_function", {"name": "calculate_bvw", "target_id": "30_6-5"})
            self.assertEqual(described.structured_content["example"]["target_id"], "30_6-5")
            called = await client.call_tool("call_function", {
                "name": "calculate_bvw", "target_id": "30_6-5", "arguments": {"phie": "PHIC", "sw": "SWC"}})
            self.assertFalse(called.is_error, called.content)
            self.assertEqual(called.structured_content["result"]["curve"], "CALCULATE_BVW")
            plot = await client.call_tool("call_function", {
                "name": "pickett_plot", "target_id": "30_6-5", "arguments": {"phi": "PHIC", "rw": 0.05}, "dpi": 50})
            self.assertFalse(plot.is_error, plot.content)
            self.assertEqual([c.type for c in plot.content], ["image", "text"])
            loaded = await client.call_tool("list_loaded", {})
            self.assertEqual({r["function"] for r in loaded.structured_content["results"].values()},
                             {"calculate_bvw", "pickett_plot"})
            refused = await client.call_tool("call_function", {
                "name": "read_capillary_pressure", "arguments": {"source": "/etc/passwd"}})
            self.assertTrue(refused.is_error)
            self.assertIn("outside the folders", refused.content[0].text)
            self.assertNotIn(str(Path(self.folder.name).resolve()), refused.content[0].text)


if __name__ == "__main__":
    unittest.main()
