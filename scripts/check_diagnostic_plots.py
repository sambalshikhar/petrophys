"""Scientific and API regression checks: python scripts/check_diagnostic_plots.py."""

import unittest
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import petrophys as pp
from petrophys.exceptions import DataQualityWarning
from petrophys.visualization import crossplot, xrf_panel
from petrophys.visualization.lithology_crossplots import (
    _buckles_chart, _hingle_chart, _mid_chart, _mn_chart, _nd_chart,
)
from petrophys.visualization.pickett import PickettPlot, _pickett_chart
from petrophys.visualization.crossplot_viewer import CrossPlot
from petrophys.visualization.analytics import histograms


def control():
    w = pp.Well("plot control"); w.depth = np.arange(4.); w.depth_unit = "m"
    phi = np.array([.05, .1, .2, .3])
    for name, values, unit in [
        ("PHIE", phi, "v/v"), ("PHIT", phi, "v/v"), ("NPHI", phi, "v/v"),
        ("RHOB", 2.71-1.71*phi, "g/cm3"), ("DTC", 47.6+(189-47.6)*phi, "us/ft"),
        ("RDEP", .096/(phi**1.85*.5**2.2), "ohm.m"),
        ("SW", [.2, .4, .6, .8], "v/v"), ("GR", [20., 40., 60., 80.], "gAPI"),
    ]:
        w.add_curve(name, np.array(values, dtype=float, copy=True), unit)
    return w


def point_coordinates(figure):
    return np.asarray(figure.axes[0].collections[0].get_offsets(), dtype=float)


class DiagnosticPlotChecks(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_archie_contours_with_nondefault_exponents(self):
        a, rw, m, n = 1.2, .08, 1.85, 2.2
        p = _pickett_chart(control(), rw, a=a, m=m, n=n)
        for line, sw in zip(p.ax.lines, [.2, .4, .6, .8, 1.]):
            phi, rt = line.get_data()
            np.testing.assert_allclose((a*rw/(phi**m*rt))**(1/n), sw)
        h = _hingle_chart(control(), rw, a=a, m=m, n=n)
        for line, sw in zip(h.axes[0].lines, [1., .8, .6, .4, .2]):
            phi, transformed = line.get_data(); ok = phi > 0
            rt = transformed[ok]**(-m)
            np.testing.assert_allclose((a*rw/(phi[ok]**m*rt))**(1/n), sw)

    def test_water_line_parameter_recovery(self):
        phi = np.array([.05, .1, .2, .3]); rt = 1.2*.08/phi**1.85
        p = PickettPlot(); p.create(phi, rt, .08)
        np.testing.assert_allclose(p.add_trend_line(phi, rt), [1.85, np.log10(1.2*.08)])

    def test_pickett_and_hingle_convert_units_without_mutation(self):
        first, second = control(), control()
        second.get("PHIE").data *= 100; second.get("PHIE").unit = "%"
        second.get("RDEP").data /= .3048; second.get("RDEP").unit = "ohm.ft"
        before = second.to_dataframe().copy()
        for function in [_pickett_chart, _hingle_chart]:
            with self.subTest(function=function.__name__):
                base, converted = function(first, .08), function(second, .08)
                fa = base.fig if hasattr(base, "fig") else base
                fb = converted.fig if hasattr(converted, "fig") else converted
                np.testing.assert_allclose(point_coordinates(fa), point_coordinates(fb))
        np.testing.assert_allclose(second.to_dataframe().to_numpy(), before.to_numpy(), equal_nan=True)
        self.assertEqual(second.get("PHIE").unit, "%")

    def test_invalid_archie_parameters_are_explicit_errors(self):
        for function in [_pickett_chart, _hingle_chart]:
            for parameter in ["rw", "a", "m", "n"]:
                for value in [0., -1., np.nan, np.inf, "bad", None]:
                    with self.subTest(function=function.__name__, parameter=parameter, value=value):
                        with self.assertRaisesRegex(ValueError, parameter):
                            function(control(), **{"rw": .08, parameter: value})

    def test_invalid_shapes_and_saturation_lines(self):
        with self.assertRaisesRegex(ValueError, "matching lengths"):
            PickettPlot().create(np.ones(3), np.ones(2), .08)
        for lines in [[], [0., .5], [1.2], [np.nan], [[.2, .5]]]:
            with self.subTest(lines=lines), self.assertRaisesRegex(ValueError, "sw_lines"):
                PickettPlot().create(np.array([.1]), np.array([10.]), .08, sw_lines=lines)

    def test_buckles_curves_preserve_constant_bvw(self):
        fig = _buckles_chart(control())
        for line, bvw in zip(fig.axes[0].lines, [.02, .04, .06, .08, .1, .15]):
            phi, sw = line.get_data()
            self.assertTrue(np.all(sw <= 1))
            np.testing.assert_allclose(phi*sw, bvw)

    def test_sparse_mixed_unit_fallback_and_label(self):
        w = control()
        w.get("PHIE").data = np.array([5., np.nan, 20., 30.]); w.get("PHIE").unit = "%"
        fig = _buckles_chart(w)
        np.testing.assert_allclose(point_coordinates(fig)[:, 0], [.05, .1, .2, .3])
        self.assertIn("effective and total", fig.axes[0].get_xlabel())
        w.remove_curve("PHIE")
        self.assertIn("Total", _buckles_chart(w).axes[0].get_xlabel())
        self.assertIn("Total", _hingle_chart(w, rw=0.05).axes[0].get_xlabel())

    def test_mid_pure_endmembers_with_and_without_total_porosity(self):
        for use_phit in [True, False]:
            for rho, neutron, dt in [(2.65, -.02, 55.5), (2.71, 0., 47.6), (2.87, .02, 43.5)]:
                with self.subTest(use_phit=use_phit, rho=rho):
                    w = control(); phi = np.array([0., .1, .2, .4])
                    w.get("PHIT").data = phi
                    w.get("RHOB").data = rho*(1-phi)+phi
                    w.get("NPHI").data = neutron*(1-phi)+phi
                    w.get("DTC").data = dt*(1-phi)+189*phi
                    if not use_phit: w.remove_curve("PHIT")
                    xy = point_coordinates(_mid_chart(w))
                    np.testing.assert_allclose(xy, np.tile([dt, rho], (4, 1)), atol=1e-10)

    def test_binary_nd_porosity_and_unsupported_samples(self):
        from petrophys.interpretation.lithology import nd_crossplot_porosity
        for first, second in [((2.65, -.02), (2.71, 0.)), ((2.71, 0.), (2.87, .02))]:
            for fraction in [.2, .5, .8]:
                rho = first[0]*fraction + second[0]*(1-fraction)
                neutron = first[1]*fraction + second[1]*(1-fraction)
                phi = np.array([0., .1, .2, .4])
                np.testing.assert_allclose(nd_crossplot_porosity(rho*(1-phi)+phi, neutron*(1-phi)+phi), phi, atol=1e-10)
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter("always")
            result = nd_crossplot_porosity([2.9], [.5])
        self.assertTrue(np.isnan(result).all())
        self.assertTrue(any(issubclass(n.category, DataQualityWarning) for n in notices))

    def test_mid_explicit_porosity_needs_no_neutron(self):
        w = control(); w.remove_curve("NPHI"); w.rename_curve("PHIT", "TOTAL_PHI")
        np.testing.assert_allclose(point_coordinates(_mid_chart(w, porosity_curve="TOTAL_PHI")),
                                   np.tile([47.6, 2.71], (4, 1)))
        with self.assertRaisesRegex(ValueError, "MISSING"):
            _mid_chart(w, porosity_curve="MISSING")

    def test_crossplot_draws_the_density_neutron_chart(self):
        w = control(); w.get("NPHI").data *= 100; w.get("NPHI").unit = "%"
        w.get("RHOB").data *= 1000; w.get("RHOB").unit = "kg/m3"
        first, second = _nd_chart(w), crossplot(w, "RHOB", "NPHI")
        self.assertIsInstance(second, CrossPlot)
        np.testing.assert_allclose(point_coordinates(first), point_coordinates(second.fig))
        self.assertEqual(len(first.axes[0].lines), len(second.ax.lines))

    def test_unknown_and_missing_chart_units(self):
        w = control(); w.get("NPHI").unit = "banana"
        with self.assertRaisesRegex(ValueError, "NPHI.*banana"):
            _nd_chart(w)
        w.get("NPHI").unit = ""
        with self.assertWarnsRegex(DataQualityWarning, "assuming v/v"):
            _nd_chart(w)

    def test_histograms_normalize_units_and_reject_incompatible_units(self):
        first, second = control(), control()
        second.get("PHIE").data *= 100; second.get("PHIE").unit = "%"
        ax = histograms([first, second], curve="PHIE", bins=10).ax
        np.testing.assert_array_equal([p.get_height() for p in ax.containers[0]],
                                      [p.get_height() for p in ax.containers[1]])
        self.assertIn("v/v", ax.get_xlabel())
        second.get("PHIE").unit = "g/cm3"
        with self.assertRaisesRegex(ValueError, "cannot be converted"):
            histograms([first, second], curve="PHIE")

    def test_xrf_concentrations_and_proxies_use_common_mass_units(self):
        from petrophys.interpretation.geochemistry import XRFData
        xrf = XRFData(depth=np.arange(3.), elements={"Al": np.array([3.,4.,5.]), "Ti": np.array([1000.,2000.,3000.])},
                      units={"Al": "%", "Ti": "ppm"})
        fig = xrf_panel(xrf)
        np.testing.assert_allclose(fig.axes[1].lines[1].get_xdata(), [.1,.2,.3])
        self.assertIn("%", fig.axes[1].get_xlabel())
        np.testing.assert_array_equal(xrf.elements["Ti"], [1000.,2000.,3000.])
        self.assertEqual(xrf.units["Ti"], "ppm")
        ppm = xrf_panel(xrf, concentration_unit="ppm")
        np.testing.assert_allclose(ppm.axes[1].lines[0].get_xdata(), [30000.,40000.,50000.])

    def test_off_axis_disclosure_and_configurable_limits(self):
        w = control(); w.get("RDEP").data[0] = 5000.
        default = _pickett_chart(w, .08)
        self.assertTrue(any("outside" in t.get_text() for t in default.ax.texts))
        expanded = _pickett_chart(w, .08, ylim=(.1, 10000.))
        self.assertFalse(any("outside" in t.get_text() for t in expanded.ax.texts))

    def test_colour_gaps_do_not_drop_data(self):
        w = control(); w.get("GR").data[1] = np.nan
        fig = _nd_chart(w)
        self.assertEqual(len(fig.axes[0].collections[0].get_offsets()), 4)

    def test_plot_labels_are_inside_axes(self):
        fig = _mn_chart(control()); fig.canvas.draw()
        quartz = next(t for t in fig.axes[0].texts if t.get_text() == "Quartz")
        self.assertLessEqual(quartz.get_window_extent().xmax, fig.axes[0].bbox.xmax)
        p = _pickett_chart(control(), .08); p.fig.canvas.draw()
        for text in p.ax.texts:
            if text.get_text().startswith("Sw="):
                self.assertLessEqual(text.get_window_extent().ymax, p.ax.bbox.ymax)


if __name__ == "__main__":
    unittest.main(verbosity=2)
