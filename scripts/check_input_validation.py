"""Regression checks for user-visible input/output diagnostics.

Run with ``python scripts/check_input_validation.py`` from a source checkout.
"""

import unittest
import warnings

import numpy as np
import pandas as pd

from petrophys._internal.benchmark import historical_benchmark
from petrophys.exceptions import DataQualityWarning


class InputValidationChecks(unittest.TestCase):
    def warns(self, function, *args, contains, **kwargs):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = function(*args, **kwargs)
        messages = [str(w.message) for w in captured if issubclass(w.category, DataQualityWarning)]
        self.assertTrue(any(contains in message for message in messages), messages)
        return result, messages

    def test_core_impossible_inputs_and_outputs(self):
        from petrophys.interpretation.core_analysis import CoreMechanical

        core = CoreMechanical(depth=np.arange(3), ucs=np.ones(3),
                              vp=np.array([1500., 2000., 2500.]), vs=np.array([3000., 4000., 5000.]))
        result, messages = self.warns(core.dynamic_moduli, 2.5, contains="Vp/Vs")
        self.assertTrue(any("Poisson ratio output" in m for m in messages))
        self.assertTrue(any("bulk modulus output" in m for m in messages))
        # A warning reports the existing result rather than silently repairing it.
        np.testing.assert_allclose(result["poisson_ratio"], 7 / 6)

    def test_core_valid_mechanics_and_missing_samples(self):
        from petrophys.interpretation.core_analysis import CoreMechanical

        core = CoreMechanical(depth=np.arange(2), ucs=np.ones(2),
                              vp=np.array([4000., np.nan]), vs=np.array([2000., np.nan]))
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = core.dynamic_moduli(2.5)
        self.assertFalse(captured)
        self.assertAlmostEqual(result["young_modulus"][0], 26.66666666666667)
        self.assertTrue(np.isnan(result["young_modulus"][1]))

    def test_nmr_negative_fraction_and_overflow(self):
        from petrophys.interpretation.nmr.permeability import sdr_permeability

        result, messages = self.warns(sdr_permeability, [-1., .2], [30., 50.], contains="SDR porosity")
        self.assertIn("1 of 2", messages[0])
        np.testing.assert_allclose(result, [3600., 16.])
        _, messages = self.warns(sdr_permeability, [1e100], [1e100], contains="permeability output")

    def test_nmr_coates_and_numeric_strings(self):
        from petrophys.interpretation.nmr.permeability import coates_permeability, sdr_permeability

        self.warns(coates_permeability, [-20.], [10.], [10.], contains="Coates porosity")
        with self.assertRaisesRegex(ValueError, "phi_frac.*numeric"):
            sdr_permeability(["bad"], [50.])
        with self.assertRaisesRegex(ValueError, "compatible sample shapes"):
            sdr_permeability([.1, .2], [10., 20., 30.])

    def test_bulk_water_direct_and_module(self):
        from petrophys.evaluation.bvw.bvw_module import BvwModule, calculate_bvw
        from petrophys.model.well import Well

        result, messages = self.warns(calculate_bvw, [-.1, -.2], [.3, .4], contains="BVW porosity")
        np.testing.assert_allclose(result, [-.03, -.08])
        self.assertTrue(any("BVW output" in m for m in messages))
        self.warns(BvwModule().calculate, Well(), phie=np.array([.2]), sw=np.array([1.5]),
                   contains="water saturation")
        self.assertIsInstance(calculate_bvw(.2, .4), float)
        self.assertAlmostEqual(calculate_bvw(.2, .4), .08)

    def test_core_text_coercion_is_recorded(self):
        from petrophys.io import read_core

        frame = pd.DataFrame({"DEPTH": [1000., 1001., 1002.], "CPOR": ["bad", "", .2]})
        core, messages = self.warns(read_core, frame, depth_unit="m", contains="CPOR")
        report = core.metadata["import_report"]
        self.assertEqual(report["invalid_numeric_cells"]["CPOR"], {"count": 1, "row_positions": [0]})
        self.assertTrue(report["warnings"])
        np.testing.assert_allclose(core.porosity, [np.nan, np.nan, .2], equal_nan=True)

    def test_core_missing_tokens_are_not_bad_text(self):
        from petrophys.io import read_core

        frame = pd.DataFrame({"DEPTH": [1., 2., 3.], "CPOR": [None, "N/A", "NaN"]})
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            core = read_core(frame, depth_unit="m")
        self.assertFalse(captured)
        self.assertNotIn("invalid_numeric_cells", core.metadata["import_report"])

    def test_archie_and_shaly_models_warn_before_clipping(self):
        from petrophys.interpretation.saturation.archie import archie
        from petrophys.interpretation.saturation.indonesia import indonesia
        from petrophys.interpretation.saturation.simandoux import simandoux

        for function, extra in ((archie, {}), (indonesia, {"vcl": .2, "rcl": 2.}),
                                (simandoux, {"vcl": .2, "rcl": 2.})):
            with self.subTest(function=function.__name__):
                self.warns(function, rt=[10., 20.], rw=.1, phi=[1.1, 2.],
                           contains="Saturation phi", **extra)
        self.warns(archie, rt=[-1., 20.], rw=.1, phi=[.2, .25], contains="resistivity")
        with self.assertRaisesRegex(ValueError, "finite positive resistivity"):
            archie(10., np.nan, .2)

    def test_gr_baselines_and_density_contrast(self):
        from petrophys.interpretation.clay_volume.gr_method import vcl_from_gr
        from petrophys.interpretation.porosity.density_porosity import density_porosity

        result, _ = self.warns(vcl_from_gr, np.array([20., 80.]), 20., 10., contains="baselines")
        np.testing.assert_array_equal(result, [0., 0.])
        result, _ = self.warns(vcl_from_gr, [20., 80.], 20., 20., contains="baselines")
        self.assertTrue(np.isnan(result).all())
        result, _ = self.warns(density_porosity, [2.3], rhob_fluid=2.65, contains="density contrast")
        self.assertTrue(np.isnan(result).all())

    def test_remaining_saturation_families(self):
        from petrophys.interpretation.saturation.dual_water import dual_water
        from petrophys.interpretation.saturation.north_sea import north_sea_shaly_sand
        from petrophys.interpretation.saturation.nigeria_schlumberger import nigeria_schlumberger
        from petrophys.interpretation.saturation.waxman_smits import waxman_smits
        from petrophys.interpretation.saturation.sxo import calculate_sxo

        cases = [
            (dual_water, dict(phi_t=[1.1], rt=[10.], rw=.1, rwb=.05, vcl=[.2], phi_cl=.1)),
            (north_sea_shaly_sand, dict(phi=[1.1], rt=[10.], rw=.1, vsh=[.2], rtsh=2.)),
            (nigeria_schlumberger, dict(phi=[1.1], rt=[10.], rw=.1, vcl=[.2], rsh=2.)),
            (waxman_smits, dict(phi_t=[1.1], rt=[10.], rw=.1, cec=.1, rho_ma=2.65)),
            (calculate_sxo, dict(phi=[1.1], rxo=[10.], rmf=.2)),
        ]
        for function, kwargs in cases:
            with self.subTest(function=function.__name__):
                self.warns(function, contains="Saturation phi", **kwargs)

    def test_timur_replacement_and_mhi_label_warning(self):
        from petrophys.interpretation.permeability.timur import timur
        from petrophys.evaluation.mhi.classification import classify_mhi

        _, messages = self.warns(timur, [.2], -.2, contains="replaced by 0.01")
        self.assertTrue(any("irreducible water saturation" in m for m in messages))
        self.warns(classify_mhi, np.array([-1., np.nan]), contains="MHI classification")

    def test_explicit_survey_shape_and_type_errors(self):
        from petrophys.io.survey_reader import minimum_curvature

        for kwargs, message in [
            (dict(md=[0, 100, 200], inclination=[0, 10], azimuth=[0, 10, 20]), "same number"),
            (dict(md=[0, 100], inclination=None, azimuth=[0, 0]), "one-dimensional"),
            (dict(md=[0, 100], inclination=[0, "bad"], azimuth=[0, 0]), "inclination.*numeric"),
            (dict(md=[0, 0], inclination=[0, 0], azimuth=[0, 0]), "strictly increasing"),
        ]:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                minimum_curvature(**kwargs)
        np.testing.assert_allclose(minimum_curvature([0., 100.], [0., 0.], [0., 0.])["tvd"], [0., 100.])

    def test_fluid_modulus_errors_name_the_parameter(self):
        from petrophys.interpretation.saturation.sonic.rock_physics import sw_from_fluid_modulus

        for value in [0., -1., np.nan, "bad"]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "k_brine"):
                sw_from_fluid_modulus(1., value, .1)
        with self.assertRaisesRegex(ValueError, "must differ"):
            sw_from_fluid_modulus(1., 2.5, 2.5)
        np.testing.assert_allclose(
            sw_from_fluid_modulus(np.array([.1, 2.5]), np.array([2.5, 2.5]), .1), [0., 1.])

    def test_ordinary_missing_values_do_not_warn(self):
        from petrophys.evaluation.bvw.bvw_module import calculate_bvw
        from petrophys.interpretation.nmr.permeability import sdr_permeability
        from petrophys.interpretation.saturation.archie import archie

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", DataQualityWarning)
            calculate_bvw([.2, np.nan], [.4, np.nan])
            sdr_permeability([.2, np.nan], [50., np.nan])
            archie([10., np.nan], .1, [.2, np.nan])
        self.assertFalse([w for w in captured if issubclass(w.category, DataQualityWarning)])

    def test_warnings_can_be_promoted_to_errors(self):
        from petrophys.evaluation.bvw.bvw_module import calculate_bvw

        with warnings.catch_warnings():
            warnings.simplefilter("error", DataQualityWarning)
            with self.assertRaises(DataQualityWarning):
                calculate_bvw(-.1, .5)

    def test_benchmark_warning_policy_is_unchanged(self):
        from petrophys.evaluation.bvw.bvw_module import calculate_bvw
        from petrophys.interpretation.permeability.timur import timur

        with historical_benchmark(), warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", DataQualityWarning)
            calculate_bvw(-.1, .5)
            timur(.2, -.2)
        self.assertFalse([w for w in captured if issubclass(w.category, DataQualityWarning)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
