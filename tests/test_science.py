"""The textbook equations against values worked by hand, and the rules that pick
an interpretation's inputs: each test runs in well under a second, so a change
that moves a result is caught before the reference suite runs."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import _support  # noqa: F401 - import path and plotting backend

import petrophys as pp
from petrophys.evaluation.rock_quality.rock_quality_module import calculate_fzi, calculate_phi_z, calculate_rqi
from petrophys.interpretation.clay_volume import vcl_from_gr
from petrophys.interpretation.electrical.rw.salinity_rw import rw_from_salinity
from petrophys.interpretation.permeability import timur
from petrophys.interpretation.permeability.coates_denoo import coates_denoo
from petrophys.interpretation.porosity.density_porosity import density_porosity
from petrophys.interpretation.porosity.sonic_porosity import raymer_hunt_gardner, wyllie_time_average
from petrophys.interpretation.saturation import archie, indonesia, simandoux
from petrophys.interpretation.temperature.temp_corrections import arps_equation
from petrophys.preprocessing import depth_match, find_log_depth_shift, shift_curves
from petrophys.preprocessing.quality.depth_match import _moved


class SaturationEquations(unittest.TestCase):
    def test_archie(self):
        # Sw = (a Rw / (phi^m Rt))^(1/n)
        self.assertAlmostEqual(archie(rt=20.0, rw=0.05, phi=0.2), 0.25, places=12)
        self.assertAlmostEqual(archie(rt=20.0, rw=0.05, phi=0.2, a=0.81), math.sqrt(0.81 * 0.05 / (0.04 * 20)), places=12)
        self.assertAlmostEqual(archie(rt=20.0, rw=0.05, phi=0.2, n=2.5), 0.0625 ** (1 / 2.5), places=12)

    def test_shaly_sand_models_reduce_to_archie_in_clean_rock(self):
        clean = archie(rt=20.0, rw=0.05, phi=0.2)
        self.assertAlmostEqual(simandoux(rt=20.0, rw=0.05, phi=0.2, vcl=0.0, rcl=2.0), clean, places=9)
        self.assertAlmostEqual(indonesia(rt=20.0, rw=0.05, phi=0.2, vcl=0.0, rcl=2.0), clean, places=9)

    def test_clay_conduction_lowers_sw(self):
        clean = archie(rt=20.0, rw=0.05, phi=0.2)
        self.assertLess(simandoux(rt=20.0, rw=0.05, phi=0.2, vcl=0.3, rcl=2.0), clean)
        self.assertLess(indonesia(rt=20.0, rw=0.05, phi=0.2, vcl=0.3, rcl=2.0), clean)


class Porosity(unittest.TestCase):
    def test_density_porosity(self):
        # (rho_ma - rho_b) / (rho_ma - rho_fl)
        self.assertAlmostEqual(density_porosity(2.32, rhob_matrix=2.65, rhob_fluid=1.0), 0.2, places=12)

    def test_sonic_porosity(self):
        # Wyllie: (dt - dt_ma) / (dt_fl - dt_ma); Raymer-Hunt-Gardner approximation: 0.625 (dt - dt_ma) / dt
        self.assertAlmostEqual(wyllie_time_average(88.8), 33.3 / 133.5, places=12)
        self.assertAlmostEqual(raymer_hunt_gardner(88.8, approximate=True), 0.625 * 33.3 / 88.8, places=12)


class ClayVolume(unittest.TestCase):
    def test_gamma_ray_index_and_larionov(self):
        np.testing.assert_allclose(vcl_from_gr(np.array([20.0, 60.0, 100.0]), gr_min=20, gr_max=100), [0.0, 0.5, 1.0])
        # Larionov (1969): older rocks 0.33 (2^(2 IGR) - 1), Tertiary 0.083 (2^(3.7 IGR) - 1)
        self.assertAlmostEqual(vcl_from_gr(60.0, 20, 100, method="larionov_older"), 0.33, places=12)
        self.assertAlmostEqual(vcl_from_gr(60.0, 20, 100, method="larionov_tertiary"), 0.083 * (2 ** 1.85 - 1), places=12)


class Permeability(unittest.TestCase):
    def test_coates_denoo(self):
        # k = 10000 phi^4 ((1 - Swi) / Swi)^2
        self.assertAlmostEqual(coates_denoo(0.25, 0.2), 625.0, places=9)

    def test_timur(self):
        # k = 0.136 (100 phi)^4.4 / (100 Swi)^2, porosity and Swi in percent
        self.assertAlmostEqual(timur(0.25, 0.2), 0.136 * 25 ** 4.4 / 20 ** 2, delta=1e-9 * 0.136 * 25 ** 4.4 / 400)

    def test_rock_quality_index_and_flow_zone_indicator(self):
        # RQI = 0.0314 sqrt(k / phi), phi_z = phi / (1 - phi), FZI = RQI / phi_z
        rqi = calculate_rqi(100.0, 0.2)
        self.assertAlmostEqual(rqi, 0.0314 * math.sqrt(500.0), places=12)
        self.assertAlmostEqual(calculate_phi_z(0.2), 0.25, places=12)
        self.assertAlmostEqual(calculate_fzi(rqi, 0.25), rqi / 0.25, places=12)


class FormationWater(unittest.TestCase):
    def test_arps(self):
        # R2 = R1 (T1 + 6.77) / (T2 + 6.77) in degF; (T + 21.5) in degC
        self.assertAlmostEqual(arps_equation(0.1, 75.0, 200.0), 0.1 * 81.77 / 206.77, places=12)
        self.assertAlmostEqual(arps_equation(0.1, 25.0, 90.0, temp_unit="C"), 0.1 * 46.5 / 111.5, places=12)

    def test_brine_resistivity_from_salinity(self):
        # 35,000 ppm NaCl reads about 0.18 ohm.m at 75 degF on the Schlumberger Gen-9 chart.
        self.assertAlmostEqual(float(rw_from_salinity(35000.0, 75.0)), 0.18, delta=0.005)
        # Saltier water conducts better; hotter water too.
        self.assertLess(float(rw_from_salinity(200000.0, 75.0)), float(rw_from_salinity(35000.0, 75.0)))
        self.assertLess(float(rw_from_salinity(35000.0, 200.0)), float(rw_from_salinity(35000.0, 75.0)))


class RwSourceOrder(unittest.TestCase):
    """An Rw you give beats a salinity you give, which beats the header's Rw (30/6-5: 0.07 at 230 degF)."""

    @classmethod
    def setUpClass(cls):
        cls.well = pp.read(pp.sample("30_6-5"))

    def median_rw(self, **settings):
        return float(np.nanmedian(pp.interpret(self.well, **settings).well["RW"].data))

    def test_order(self):
        header = self.median_rw()
        given = self.median_rw(rw=0.035, rw_meas_temp="230 degF", salinity_ppm=10000)
        salinity = self.median_rw(salinity_ppm=150000)
        # The header value at formation temperature; half of it when Rw is given at half.
        self.assertAlmostEqual(given / header, 0.5, places=6)
        # A salinity you give displaces the header Rw: 150,000 ppm is far saltier than 0.07 ohm.m at 230 degF.
        self.assertLess(salinity, 0.5 * header)


class CarbonatePermeabilityWithoutCore(unittest.TestCase):
    """HW-3 carries core-corrected PERM_CC; without it a carbonate run warns that PERM has no core behind it."""

    @staticmethod
    def warned(well):
        result = pp.interpret(well, lithology="carbonate")
        return any("carbonate well is" in str(message) for message in result.warnings)

    def test_warns_only_without_core(self):
        well = pp.read(pp.sample("HW-3"))
        self.assertFalse(self.warned(well))
        for name in ("PERM_CC", "PHIE_CC"):
            well.remove_curve(name)
        self.assertTrue(self.warned(well))


class DepthAlignment(unittest.TestCase):
    def well_with_lag(self, lag):
        rng = np.random.default_rng(0)
        depth = np.arange(1000.0, 1500.0, 0.5)
        signal = np.convolve(rng.normal(size=depth.size + 40), np.ones(9) / 9, mode="same")[:depth.size]
        well = pp.Well(name="synthetic")
        well.depth = depth
        well.depth_unit = "m"
        well.add_curve("NPHI", 0.25 + 0.05 * signal, "v/v")
        # Density mirrors neutron, recorded `lag` samples too deep.
        well.add_curve("RHOB", 2.45 - 0.2 * np.roll(signal, lag), "g/cc")
        return well

    def test_finds_and_removes_a_shift(self):
        well = self.well_with_lag(20)  # 10 m too deep
        found = find_log_depth_shift(well, "RHOB", "NPHI")
        self.assertAlmostEqual(found.shift, -10.0)
        self.assertLess(found.correlation, -0.95)
        self.assertTrue(found.off_depth)
        aligned = shift_curves(well, ["RHOB"], found.shift)
        self.assertAlmostEqual(find_log_depth_shift(aligned, "RHOB", "NPHI").shift, 0.0)
        np.testing.assert_array_equal(well["RHOB"].data[:5], self.well_with_lag(20)["RHOB"].data[:5])  # copy

    def test_curves_on_depth_are_left(self):
        found = find_log_depth_shift(self.well_with_lag(0), "RHOB", "NPHI")
        self.assertEqual(found.shift, 0.0)
        self.assertFalse(found.off_depth)


class DepthMatchVaryingShift(unittest.TestCase):
    """A second pass of the 30/6-5 gamma ray, spliced and stretched by known amounts, is matched back."""

    def test_splice_and_stretch_are_recovered(self):
        well = pp.read(pp.sample("30_6-5"))
        depth = np.asarray(well.depth, dtype=float)
        gr = np.asarray(well["GR"].data, dtype=float)
        # On depth above 2500 m, 1.5 m too deep below, and stretching to 2.5 m at the base.
        true = np.where(depth < 2500, 0.0, np.where(depth < 2800, -1.5, -1.5 - (depth - 2800) / (depth.max() - 2800)))
        noise = np.random.default_rng(1).normal(0.0, 3.0, depth.size)
        well.add_curve("GR_PASS2", _moved(depth, gr, -true) + noise, "API")
        well.add_curve("GR_SAME", gr + noise, "API")
        match = depth_match(well, "GR_PASS2", "GR")
        logged = np.isfinite(gr) & (depth > 600)
        error = np.abs(match.shift - true)[logged]
        step = float(np.median(np.diff(depth)))
        self.assertGreater(np.mean(error <= step + 1e-9), 0.85)   # within one sample at most depths
        self.assertLess(np.percentile(error, 95), 0.5)
        self.assertGreater(match.correlation_after, match.correlation_before)
        self.assertEqual(depth_match(well, "GR_SAME", "GR").max_shift, 0.0)  # on depth: left alone
        moved = shift_curves(well, ["GR_PASS2"], match)
        self.assertGreater(np.corrcoef(*[np.nan_to_num(v) for v in (moved["GR_PASS2"].data[logged], gr[logged])])[0, 1],
                           np.corrcoef(*[np.nan_to_num(v) for v in (well["GR_PASS2"].data[logged], gr[logged])])[0, 1])


class PlotObject(unittest.TestCase):
    def test_every_plot_saves_and_exposes_its_axes(self):
        chart = pp.bar_chart(["A", "B"], [3.0, 5.0])
        self.assertIs(chart.ax, chart.axes[0])
        with tempfile.TemporaryDirectory() as folder:
            chart.save(Path(folder) / "chart.png")
            self.assertTrue((Path(folder) / "chart.png").stat().st_size > 0)
        chart.close()

    def test_top_and_bottom_set_the_window(self):
        plot = pp.zone_flags_plot(pp.interpret(pp.read(pp.sample("HW-3"))), ["NET_PAY"], top=7850, bottom=7900)
        self.assertEqual(sorted(plot.ax.get_ylim()), [7850.0, 7900.0])
        plot.close()


if __name__ == "__main__":
    unittest.main()
