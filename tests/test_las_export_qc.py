"""Regression tests for LAS parameter export."""

import unittest

import lasio

from petrophys._internal.benchmark import historical_benchmark
from petrophys.io.export import _export_well_parameters
from petrophys.model.well import Well


class LasParameterExportTests(unittest.TestCase):
    def test_qc_report_is_not_exported_even_in_benchmark_mode(self):
        well = Well("test")
        report = [{"curve": "GR", "changed": 3}]
        well.set_parameter("VALUE_RANGE_QC_REPORT", report)
        well.set_parameter("temperature_profile_source", "fallback_geothermal_gradient")

        las = lasio.LASFile()
        with historical_benchmark():
            _export_well_parameters(las, well)

        self.assertNotIn("VALUE_RANGE_QC_REPORT", las.params.keys())
        self.assertEqual(las.params["temperature_profile_source"].value, "fallback_geothermal_gradient")
        self.assertEqual(well.get_parameter("VALUE_RANGE_QC_REPORT"), report)


if __name__ == "__main__":
    unittest.main()
