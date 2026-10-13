"""Profile-dependent defaults, net-pay cutoffs and non-reservoir samples.

Under the default ``"current"`` profile, unset net-pay cutoffs take the
standard of each sample's lithology family (STANDARD_CUTOFFS: clastic
VCL < 0.40, PHIE > 0.08, SW < 0.60; carbonate VCL < 0.30, PHIE > 0.045,
SW < 0.70), and evaporite and bad-hole samples are not net reservoir or pay.
The "geolink" profile keeps the eval package's VCL < 0.3, SW < 0.7 and counts
every sample that passes. The expected flags below are worked by hand from
those cutoffs; a sample exactly at a cutoff fails it.
"""
import dataclasses
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import pytest

import petrophys as pp
from petrophys import Well
from petrophys.evaluation.net_pay import generate_net_flags, summarize_pay
from petrophys.evaluation.sensitivity import cutoff_sensitivity
from petrophys.workflow import DynamicWorkflow, DynamicWorkflowConfig, Workflow, WorkflowStep, steps
from petrophys.workflow.dynamic_workflow import (
    _halite_log_signature,
    _left_at_default,
    _profile_setting,
    _rerun_calculations,
)
from petrophys.workflow.schema import describe_setting
from petrophys.workflow.zoned_batch import run_dynamic_workflow_with_zones

SANDSTONE, LIMESTONE, HALITE, ANHYDRITE, GAP = 2.0, 11.0, 21.0, 22.0, 26.0


def _flag_well(vcl, phie, sw, codes, bad_hole=None, step=0.5) -> Well:
    """A well carrying VCL, PHIE and SW, as the net-pay step reads them."""
    n = len(vcl)
    well = Well("NET-PAY")
    well.depth = 2000.0 + step * np.arange(n)
    well.depth_unit = "m"
    well.add_curve("VCL", np.asarray(vcl, dtype=float), "v/v", "Clay volume")
    well.add_curve("PHIE", np.asarray(phie, dtype=float), "v/v", "Effective porosity")
    well.add_curve("SW", np.asarray(sw, dtype=float), "v/v", "Water saturation")
    well.add_curve("LITHOLOGY_GEOLINK", np.asarray(codes, dtype=float), "", "Lithology")
    if bad_hole is not None:
        well.add_curve("BAD_HOLE", np.asarray(bad_hole, dtype=float), "", "Bad Hole Flag")
    return well


def _net_pay(well: Well, profile: str, **settings):
    config = DynamicWorkflowConfig(calculation_profile=profile, **settings)
    result = DynamicWorkflow(config).run_selected_steps(well, [WorkflowStep.NET_PAY])
    result.raise_for_errors()
    step = next(step for step in result.steps_executed if step.step == "net_pay")
    flags = {name: well.get(name).data.astype(int).tolist() for name in ("NET_ROCK", "NET_RES", "NET_PAY")}
    return flags, step


# Six sandstone (clastic) and six limestone (carbonate) samples, several of
# them exactly at a cutoff.
VCL = [0.10, 0.35, 0.40, 0.10, 0.10, 0.10, 0.10, 0.30, 0.25, 0.10, 0.20, 0.35]
PHIE = [0.20, 0.20, 0.20, 0.08, 0.20, 0.20, 0.06, 0.10, 0.10, 0.045, 0.15, 0.15]
SW = [0.30, 0.30, 0.30, 0.30, 0.60, 0.65, 0.65, 0.20, 0.70, 0.20, 0.50, 0.50]
CODES = [SANDSTONE] * 6 + [LIMESTONE] * 6


def _mixed_well() -> Well:
    return _flag_well(VCL, PHIE, SW, CODES)


def test_unset_cutoffs_follow_each_samples_lithology_family():
    flags, step = _net_pay(_mixed_well(), "current")

    # Sandstone: 0.40 / 0.08 / 0.60; limestone: 0.30 / 0.045 / 0.70.
    assert flags["NET_ROCK"] == [1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0]
    assert flags["NET_RES"] == [1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0]
    assert flags["NET_PAY"] == [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
    params = step.parameters
    assert params["cutoffs_by_family"] == {
        "clastic": {"vcl_cutoff": 0.40, "phie_cutoff": 0.08, "sw_cutoff": 0.60, "samples": 6},
        "carbonate": {"vcl_cutoff": 0.30, "phie_cutoff": 0.045, "sw_cutoff": 0.70, "samples": 6},
    }
    assert params["vcl_cutoff"] == "lithology_dependent"
    assert set(params["cutoff_source"].values()) == {"lithology_family_standard"}
    assert step.decisions[0].startswith("NET_ROCK: VCL < ")
    assert "0.4 clastic" in step.decisions[0] and "0.3 carbonate" in step.decisions[0]
    assert step.decisions[0].endswith("-> 9 samples (4.5 m)")
    assert step.decisions[1].endswith("-> 7 samples (3.5 m)")
    assert step.decisions[2].endswith("-> 4 samples (2.0 m)")


# The eval package's main run and its re-run (its re-run wells and context).
@pytest.mark.parametrize("calculations", ["main_run", "rerun"])
def test_benchmark_keeps_the_eval_package_cutoffs(calculations):
    with _rerun_calculations(calculations == "rerun"):
        flags, step = _net_pay(_mixed_well(), "geolink")

    # VCL < 0.3, PHIE > 0.08 (limestone is not the label "carbonate"), SW < 0.7.
    assert flags["NET_ROCK"] == [1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0]
    assert flags["NET_RES"] == [1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0]
    assert flags["NET_PAY"] == [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0]
    assert step.parameters == {
        "vcl_cutoff": 0.3, "phie_cutoff": None, "sw_cutoff": 0.7,
        "lithology": "per-depth", "seal_vcl_cutoff": 0.8,
    }
    assert step.decisions[:3] == [
        "NET_ROCK: VCL < 0.3 -> 8 samples (4.0 m)",
        "NET_RES:  + PHIE > 0.08 (0.045 where carbonate) -> 5 samples (2.5 m)",
        "NET_PAY:  + SW < 0.7 -> 4 samples (2.0 m)",
    ]


def test_cutoffs_set_in_the_config_apply_to_every_sample():
    flags, step = _net_pay(_mixed_well(), "current", vcl_cutoff=0.35, sw_cutoff=0.5)

    # VCL < 0.35 and SW < 0.5 everywhere; PHIE still 0.08 / 0.045 by family.
    assert flags["NET_PAY"] == [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    assert step.parameters["cutoff_source"] == {
        "vcl_cutoff": "config", "phie_cutoff": "lithology_family_standard", "sw_cutoff": "config",
    }
    assert step.parameters["vcl_cutoff"] == 0.35
    assert step.parameters["phie_cutoff"] == "lithology_dependent"


def test_evaporite_and_bad_hole_samples_are_not_net_reservoir_or_pay():
    codes = [SANDSTONE, SANDSTONE, SANDSTONE, SANDSTONE, HALITE, HALITE, ANHYDRITE, SANDSTONE]
    bad_hole = [0, 1, 1, 0, 1, 0, 0, 0]

    def well():
        n = len(codes)
        return _flag_well([0.1] * n, [0.2] * n, [0.2] * n, codes, bad_hole)

    current, step = _net_pay(well(), "current")
    benchmark, _ = _net_pay(well(), "geolink")

    assert current["NET_ROCK"] == [1] * 8
    assert current["NET_RES"] == current["NET_PAY"] == [1, 0, 0, 1, 0, 0, 0, 1]
    assert benchmark["NET_PAY"] == [1] * 8
    # Evaporites first (three samples), then bad hole among the rest (two more).
    assert step.parameters["excluded_from_net_reservoir_and_pay"] == {
        "evaporite": {"samples": 3, "net_reservoir_samples_removed": 3, "net_pay_samples_removed": 3,
                      "net_reservoir_removed": 1.5, "net_pay_removed": 1.5, "depth_unit": "m"},
        "bad_hole": {"samples": 3, "net_reservoir_samples_removed": 2, "net_pay_samples_removed": 2,
                     "net_reservoir_removed": 1.0, "net_pay_removed": 1.0, "depth_unit": "m"},
    }
    assert any(text.startswith("Bad hole (BAD_HOLE = 1)") and "1.0 m of net pay" in text
               for text in step.decisions)


def test_swirr_samples_are_picked_in_clean_porous_rock():
    well = _flag_well(VCL, PHIE, SW, CODES, bad_hole=[0] * 10 + [1, 0])
    well.add_curve("RDEP", np.full(well.n_samples, 20.0), "ohm.m", "Deep resistivity")
    result = DynamicWorkflow(DynamicWorkflowConfig()).run_selected_steps(well, [WorkflowStep.PERMEABILITY])
    params = next(step for step in result.steps_executed if step.step == "permeability").parameters

    # VCL < 0.3 and PHIE > 0.08 in every lithology (the limestone at PHIE 0.06
    # is net reservoir, but not a Swirr sample), without the bad-hole sample.
    assert well.get("RESFLAG").data.astype(int).tolist() == [1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0]
    flag = params["pre_permeability_resflag"]
    assert (flag["vcl_cutoff"], flag["phie_cutoff"]) == (0.3, 0.08)
    assert flag["excluded_samples"] == {"bad_hole": 1}


def test_the_lithology_domain_alone_excludes_nothing():
    n = 4
    well = Well("DOMAIN")
    well.depth = 1000.0 + 0.5 * np.arange(n)
    well.depth_unit = "m"
    for name, value in (("VCL", 0.1), ("PHIE", 0.2), ("SW", 0.2)):
        well.add_curve(name, np.full(n, value), "v/v", name)

    flags, step = _net_pay(well, "current", lithology_domain="evaporite")

    assert flags["NET_PAY"] == [1] * n
    assert not step.parameters.get("excluded_from_net_reservoir_and_pay")


def test_a_geolink_gap_is_not_an_evaporite():
    # GeoLink code 26 is "Gap" (no lithology recorded) and has no lithology
    # group under "current"; only the halite and anhydrite codes are left out.
    codes = [SANDSTONE, GAP, GAP, HALITE]
    n = len(codes)
    flags, step = _net_pay(_flag_well([0.1] * n, [0.2] * n, [0.2] * n, codes), "current")

    assert flags["NET_PAY"] == [1, 1, 1, 0]
    assert step.parameters["excluded_from_net_reservoir_and_pay"]["evaporite"]["samples"] == 1


def test_evaporite_intervals_are_not_net_reservoir_or_pay():
    n = 4
    well = Well("INTERVALS")
    well.depth = 2000.0 + 0.5 * np.arange(n)
    well.depth_unit = "m"
    for name, value in (("VCL", 0.1), ("PHIE", 0.2), ("SW", 0.2)):
        well.add_curve(name, np.full(n, value), "v/v", name)
    intervals = pd.DataFrame({"top": [2000.0, 2001.0], "bottom": [2001.0, 2001.5],
                              "lithology": ["Sandstone", "Evaporite"]})

    flags, step = _net_pay(well, "current", lithology_intervals=intervals)

    assert flags["NET_PAY"] == [1, 1, 0, 0]
    assert step.parameters["excluded_from_net_reservoir_and_pay"]["evaporite"]["net_pay_removed"] == 1.0


@pytest.mark.parametrize("profile, expected", [
    ("current", (True, None, None)),
    ("geolink", (False, 0.3, 0.7)),
])
def test_the_trace_shows_the_settings_each_profile_used(profile, expected):
    config = DynamicWorkflowConfig(calculation_profile=profile)
    result = DynamicWorkflow(config).run_selected_steps(_mixed_well(), [WorkflowStep.NET_PAY])

    settings = result.to_trace_dict()["config"]
    assert (settings["standardize_input"], settings["vcl_cutoff"], settings["sw_cutoff"]) == expected
    assert result.config.standardize_input is None  # the run config itself stays unset


def _logged_well(n: int = 60, bit_size: Optional[float] = 8.5, drho: Optional[float] = None) -> Well:
    well = Well("PROFILE-TEST")
    well.depth = np.arange(2000.0, 2000.0 + 0.5 * n, 0.5)
    well.depth_unit = "m"
    well.add_curve("GR", np.linspace(20.0, 130.0, n), "API", "Gamma Ray")
    well.add_curve("RHOB", np.linspace(2.20, 2.62, n), "g/cc", "Bulk Density")
    well.add_curve("NPHI", np.linspace(0.30, 0.05, n), "v/v", "Neutron")
    well.add_curve("PEF", np.linspace(2.0, 4.5, n), "b/e", "PEF")
    well.add_curve("RDEP", np.linspace(40.0, 2.0, n), "ohm.m", "Deep Resistivity")
    caliper = np.full(n, 8.7)
    caliper[5:15] = 11.0  # a washout over the cleanest, most porous rock
    well.add_curve("CALI", caliper, "in", "Caliper")
    if bit_size is not None:
        well.add_curve("BS", np.full(n, bit_size), "in", "Bit size")
    if drho is not None:
        well.add_curve("DRHO", np.full(n, drho), "g/cm3", "Density correction")
    return well


def _run(profile: str, bit_size: Optional[float] = 8.5, drho: Optional[float] = None, **settings):
    config = DynamicWorkflowConfig(
        rw=0.05, rw_meas_temp_f=150.0, bht_f=180.0, skip_steps=["rmf"],
        calculation_profile=profile, **settings,
    )
    result = DynamicWorkflow(config).run(_logged_well(bit_size=bit_size, drho=drho))
    assert result.success, result.errors
    return result


def _step(result, name):
    return next(step for step in result.steps_executed if step.step == name)


def test_swirr_is_picked_in_clean_reliable_net_reservoir():
    current = _step(_run("current"), "permeability").parameters["pre_permeability_resflag"]
    benchmark = _step(_run("geolink"), "permeability").parameters["pre_permeability_resflag"]

    # Net reservoir's PHIE cutoff and left-out samples, with the clean-rock
    # (saturation-routing) clay cutoff 0.3 rather than the 0.40 net-rock one.
    assert (current["vcl_cutoff"], current["phie_cutoff"], current["sw_cutoff"]) == (0.3, 0.08, 0.6)
    assert current["excluded_samples"] == {"bad_hole": 10}
    assert (benchmark["vcl_cutoff"], benchmark["phie_cutoff"], benchmark["sw_cutoff"]) == (0.3, None, 0.7)
    assert "excluded_samples" not in benchmark


def test_assumed_values_are_warned_about_under_current_only():
    current = _run("current", bit_size=None, drho=0.02)
    benchmark = _run("geolink", bit_size=None, drho=0.02)

    # Neither well has a bit size. The benchmark assumes 8.5 in for the
    # caliper test and says nothing; "current" assumes none and says where
    # the caliper goes untested.
    assert _step(benchmark, "bad_hole").parameters["bit_size_source"] == "fallback_8.5"
    assert _step(current, "bad_hole").parameters["bit_size_source"] == "none"
    assert any(warning.startswith("No bit size") for warning in current.warnings)
    assert not any("bit size" in warning for warning in benchmark.warnings)

    bare = DynamicWorkflowConfig(rw=0.05, skip_steps=["rmf"])
    result = DynamicWorkflow(bare).run(_logged_well())
    assert any("rw_meas_temp_f" in warning for warning in result.warnings)
    assert any("30 C/km" in warning for warning in result.warnings)
    quiet = DynamicWorkflow(
        DynamicWorkflowConfig(rw=0.05, skip_steps=["rmf"], calculation_profile="geolink")
    ).run(_logged_well())
    assert not any("rw_meas_temp_f" in w or "30 C/km" in w for w in quiet.warnings)


def test_without_a_bit_size_bad_hole_comes_from_drho_alone():
    # In-gauge 12.25 in hole (CALI 12.5 in) with two samples of poor pad
    # contact (DRHO 0.25 g/cm3). With no bit size "current" runs no caliper
    # test, so BAD_HOLE flags only those two samples, the same two it flags
    # when the 12.25 in bit is known, and net reservoir and pay lose only
    # them. The benchmark assumes an 8.5 in bit: CALI - 8.5 = 4 in > 1.5 in
    # flags every sample, and its net pay keeps them all.
    n = 8
    drho = np.full(n, 0.02)
    drho[[2, 5]] = 0.25
    steps = [WorkflowStep.BAD_HOLE, WorkflowStep.NET_PAY]

    def interpret(bit_size, profile="current"):
        well = _flag_well([0.1] * n, [0.2] * n, [0.2] * n, [SANDSTONE] * n)
        well.add_curve("CALI", np.full(n, 12.5), "in", "Caliper")
        well.add_curve("DRHO", drho, "g/cm3", "Density correction")
        if bit_size is not None:
            well.add_curve("BS", np.full(n, bit_size), "in", "Bit size")
        result = DynamicWorkflow(DynamicWorkflowConfig(calculation_profile=profile)).run_selected_steps(
            well, steps
        )
        excluded = _step(result, "net_pay").parameters.get("excluded_from_net_reservoir_and_pay")
        return well, result, excluded

    well, result, excluded = interpret(bit_size=None)
    assert well.get("BAD_HOLE").data.astype(int).tolist() == [0, 0, 1, 0, 0, 1, 0, 0]
    assert well.get("NET_PAY").data.astype(int).tolist() == [1, 1, 0, 1, 1, 0, 1, 1]
    assert list(excluded) == ["bad_hole"] and excluded["bad_hole"]["samples"] == 2
    params = _step(result, "bad_hole").parameters
    assert (params["bit_size_source"], params["bit_size_used"], params["bit_size_unique_values"]) == (
        "none", None, []
    )
    assert (params["no_bit_size_samples"], params["caliper_test_samples"], params["not_assessed_samples"]) == (
        8, 0, 0
    )
    assert result.warnings == [
        "No bit size (BS curve or header) at 8 samples with a caliper reading (2000.0-2003.5 m): "
        "the caliper washout test (CALI - BS > 1.5 in) and the large-hole test (BS > 17.5 in) "
        "do not run there, and BAD_HOLE there flags only |DRHO| > 0.15 g/cm3."
    ]

    well, result, excluded = interpret(bit_size=12.25)
    assert well.get("BAD_HOLE").data.astype(int).tolist() == [0, 0, 1, 0, 0, 1, 0, 0]
    assert well.get("NET_PAY").data.astype(int).tolist() == [1, 1, 0, 1, 1, 0, 1, 1]
    assert list(excluded) == ["bad_hole"]
    assert _step(result, "bad_hole").parameters["caliper_test_samples"] == 8
    assert not result.warnings

    well, result, _ = interpret(bit_size=None, profile="geolink")
    assert well.get("BAD_HOLE").data.tolist() == [1.0] * n
    params = _step(result, "bad_hole").parameters
    assert (params["bit_size_source"], params["bit_size_used"]) == ("fallback_8.5", 8.5)
    assert well.get("NET_PAY").data.astype(int).tolist() == [1] * n
    assert not result.warnings


def test_las_export_lists_warnings_under_current_only(tmp_path):
    current = _run("current", bit_size=None, drho=0.02)
    benchmark = _run("geolink", bit_size=None, drho=0.02)
    # The "geolink" profile keeps the eval package's ~Other section, warnings or not.
    benchmark.warnings.append("Rw could not be determined. Using default 0.05 ohm.m")

    current_text = current.to_las(tmp_path / "current.las").read_text(encoding="utf-8")
    benchmark_text = benchmark.to_las(tmp_path / "benchmark.las").read_text(encoding="utf-8")
    assert "Warning: No bit size (BS curve or header)" in current_text
    assert "Warning:" not in benchmark_text


def test_profile_defaults_resolve_in_one_place():
    current = DynamicWorkflowConfig()
    benchmark = DynamicWorkflowConfig(calculation_profile="geolink")

    assert (current.standardize_input, current.vcl_cutoff, current.sw_cutoff) == (None, None, None)
    assert _profile_setting(current, "standardize_input") is True
    assert _profile_setting(current, "vcl_cutoff") is None  # resolved per lithology family
    assert _profile_setting(benchmark, "standardize_input") is False
    assert (_profile_setting(benchmark, "vcl_cutoff"), _profile_setting(benchmark, "sw_cutoff")) == (0.3, 0.7)
    # Explicit values always win.
    explicit = DynamicWorkflowConfig(standardize_input=False, vcl_cutoff=0.35)
    assert _profile_setting(explicit, "standardize_input") is False
    assert _profile_setting(explicit, "vcl_cutoff") == 0.35
    with pytest.raises(ValueError, match="standardize_input must be True or False"):
        DynamicWorkflowConfig(standardize_input="yes")
    assert describe_setting("standardize_input")["profile_defaults"] == {
        "current": True, "geolink": False,
    }
    assert describe_setting("vcl_cutoff")["profile_defaults"] == {"current": None, "geolink": 0.3}


def test_header_cutoffs_fill_settings_left_unset_under_every_profile():
    for profile in ("current", "geolink"):
        well = _flag_well(VCL, PHIE, SW, CODES)
        well.set_parameter("VCL_CUTOFF", 0.25)
        config = DynamicWorkflowConfig(calculation_profile=profile)
        result = run_dynamic_workflow_with_zones(well, config)
        assert result.global_overrides["vcl_cutoff"] == 0.25
        assert _step(result.base_result, "net_pay").parameters["vcl_cutoff"] == 0.25


def _kg_m3_density_well() -> Well:
    depth = np.arange(1000.0, 1050.0, 1.0)
    well = Well("DENSITY")
    well.depth = depth
    well.depth_unit = "m"
    well.add_curve("GR", np.full(depth.size, 60.0), "API", "Gamma ray")
    well.add_curve("RHOZ", np.full(depth.size, 2400.0), "g/cc", "Density")
    return well


def test_input_is_standardized_by_default_but_not_under_the_benchmark_profile():
    with pytest.warns(pp.DataQualityWarning, match="unit label"):
        current = pp.interpret(_kg_m3_density_well(), rw=0.07)
    assert any(step.step == "normalization" for step in current.steps_executed)

    with warnings.catch_warnings():
        warnings.simplefilter("error", pp.DataQualityWarning)
        benchmark = pp.interpret(_kg_m3_density_well(), rw=0.07, calculation_profile="geolink")
    assert benchmark.specific_steps_skipped["normalization"] == "standardize_input=False"


def test_net_flags_and_pay_summary_fail_a_sample_exactly_at_a_cutoff():
    vcl = np.array([0.40, 0.10, 0.10, 0.10])
    phie = np.array([0.20, 0.08, 0.20, 0.20])
    sw = np.array([0.30, 0.30, 0.60, 0.30])

    flags = generate_net_flags(vcl, phie, sw, 0.40, 0.08, 0.60)
    assert flags["net_rock"].tolist() == [False, True, True, True]
    assert flags["net_reservoir"].tolist() == [False, False, True, True]
    assert flags["net_pay"].tolist() == [False, False, False, True]

    pay = summarize_pay(vcl=vcl, phie=phie, sw=sw, depth=np.arange(4.0), depth_unit="m")
    # Clastic standards 0.40 / 0.08 / 0.60: only the last sample is pay.
    assert (pay.net_rock_thickness, pay.net_reservoir_thickness, pay.net_pay_thickness) == (3.0, 2.0, 1.0)
    assert pay.hcpt == pytest.approx(0.20 * (1 - 0.30))


def test_cutoff_sensitivity_example_counts_strictly():
    depth = np.arange(1000.0, 1010.0, 1.0)
    vcl = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.1, 0.15, 0.25, 0.6, 0.1])
    sw = np.array([0.2, 0.3, 0.4, 0.5, 0.9, 0.55, 0.65, 0.7, 1.0, 0.35])
    grid = cutoff_sensitivity(vcl=vcl, phie=np.full(10, 0.2), sw=sw, depth=depth,
                              vcl_cutoffs=[0.2, 0.4], sw_cutoffs=[0.5, 0.7])
    net = grid.pivot(index="vcl_cutoff", columns="sw_cutoff", values="net_pay_thickness")
    assert net.to_dict("index") == {0.2: {0.5: 3.0, 0.7: 5.0}, 0.4: {0.5: 4.0, 0.7: 7.0}}


# ---------------------------------------------------------------------------
# Halite known only from its logs, coal, implausible header bit sizes
# ---------------------------------------------------------------------------
COAL = 18.0


def _halite_logs(well: Well, rhob, gr, rdep) -> Well:
    well.add_curve("RHOB", np.asarray(rhob, dtype=float), "g/cc", "Bulk density")
    well.add_curve("GR", np.asarray(gr, dtype=float), "API", "Gamma ray")
    well.add_curve("RDEP", np.asarray(rdep, dtype=float), "ohm.m", "Deep resistivity")
    return well


def test_halite_log_signature_reads_halite_and_nothing_like_it():
    # One sample per case; NPHI and PEF are logged only where a case needs them.
    nan = np.nan
    well = Well("SIGNATURE")
    well.depth = 1000.0 + 0.5 * np.arange(10)
    _halite_logs(
        well,
        rhob=[2.05, 2.05, 2.20, 2.05, 2.05, 2.05, 2.05, 2.05, 2.05, 2.05],
        gr=[5, 5, 5, 20, 5, 5, 5, 5, 5, 5],
        rdep=[3000, 100, 3000, 3000, nan, nan, 3000, 3000, 3000, 3000],
    )
    well.add_curve("CALI", np.r_[np.full(4, 8.5), 7.9, 8.1, np.full(4, 8.5)], "in", "Caliper")
    well.add_curve("BS", np.full(10, 8.5), "in", "Bit size")
    well.add_curve("NPHI", np.array([nan] * 6 + [0.20, -0.02, nan, nan]), "v/v", "Neutron")
    well.add_curve("PEF", np.array([nan] * 8 + [1.8, 4.6]), "b/e", "PEF")

    halite = _halite_log_signature(well)

    # 0 halite; 1 RT 100 ohm.m; 2 RHOB 2.20; 3 GR 20 API; 4 no RT and the hole
    # 0.6 in under gauge (salt creep); 5 only 0.4 in under; 6 NPHI 0.20 (porous
    # rock); 7 NPHI -0.02; 8 PEF 1.8 (quartz); 9 PEF 4.6.
    assert halite.tolist() == [True, False, False, False, True, False, False, True, False, True]


def test_halite_on_the_logs_is_not_net_reservoir_or_pay():
    # Four uncoded samples read like halite (as below the end of the 16/7-1
    # code log); their PHIE and SW, as computed on a sandstone matrix, pass
    # every cutoff. Two sandstone samples and two halite-coded samples.
    nan = np.nan
    codes = [SANDSTONE, nan, nan, nan, nan, HALITE, HALITE, SANDSTONE]
    n = len(codes)

    def well():
        return _halite_logs(
            _flag_well([0.05] * n, [0.25] * n, [0.1] * n, codes),
            rhob=[2.30, 2.06, 2.06, 2.06, 2.06, 2.06, 2.06, 2.30],
            gr=[30, 5, 5, 5, 5, 5, 5, 30],
            rdep=[50, 3000, 3000, 3000, 3000, 3000, 3000, 50],
        )

    current, step = _net_pay(well(), "current")
    benchmark, _ = _net_pay(well(), "geolink")

    assert current["NET_PAY"] == [1, 0, 0, 0, 0, 0, 0, 1]
    assert benchmark["NET_PAY"] == [1] * n
    excluded = step.parameters["excluded_from_net_reservoir_and_pay"]
    assert excluded["evaporite"]["samples"] == 2
    assert excluded["halite_log_signature"] == {
        "samples": 4, "net_reservoir_samples_removed": 4, "net_pay_samples_removed": 4,
        "net_reservoir_removed": 2.0, "net_pay_removed": 2.0, "depth_unit": "m",
    }
    assert any(text.startswith("Halite on the logs where no lithology group") for text in step.decisions)

    # The same samples stay out of the reservoir flag that SWIRR_APP is picked from.
    swirr_well = well()
    result = DynamicWorkflow(DynamicWorkflowConfig()).run_selected_steps(swirr_well, [WorkflowStep.PERMEABILITY])
    flag = _step(result, "permeability").parameters["pre_permeability_resflag"]
    assert flag["excluded_samples"] == {"evaporite": 2, "halite_log_signature": 4}
    assert swirr_well.get("RESFLAG").data.astype(int).tolist() == [1, 0, 0, 0, 0, 0, 0, 1]


def test_halite_on_the_logs_takes_the_halite_density_endpoint():
    # No lithology log. RHOB 2.00 g/cm3 halite: PHID = (2.032 - 2.00)/(2.032 - 1)
    # = 0.0310 on the halite log density, 0.3939 on the sandstone grain density
    # (2.65 - 2.00)/(2.65 - 1). The sandstone at 2.32 g/cm3 reads 0.2 either way.
    def well():
        logs = Well("SALT")
        logs.depth = 1500.0 + 0.5 * np.arange(4)
        logs.depth_unit = "m"
        return _halite_logs(logs, rhob=[2.32, 2.00, 2.00, 2.32], gr=[30, 5, 5, 30], rdep=[20, 4000, 4000, 20])

    def porosity(profile):
        logs = well()
        config = DynamicWorkflowConfig(calculation_profile=profile, matrix_type="sandstone")
        result = DynamicWorkflow(config).run_selected_steps(logs, [WorkflowStep.POROSITY])
        return logs.get("PHID").data, _step(result, "porosity")

    phid, step = porosity("current")
    np.testing.assert_allclose(phid, [0.2, 0.032 / 1.032, 0.032 / 1.032, 0.2], rtol=1e-9)
    endpoint = step.parameters["matrix_endpoints"][-1]
    assert (endpoint["lithology_group"], endpoint["rho_ma"], endpoint["samples"]) == (
        "HALITE (LOG SIGNATURE)", 2.032, 2,
    )
    assert any(text.startswith("2 samples read as halite on the logs") for text in step.decisions)

    phid, step = porosity("geolink")
    np.testing.assert_allclose(phid, [0.2, 0.65 / 1.65, 0.65 / 1.65, 0.2], rtol=1e-9)
    assert step.parameters["matrix_endpoints"] == []


def test_coal_is_not_net_reservoir_or_pay():
    codes = [SANDSTONE, COAL, COAL, SANDSTONE]
    n = len(codes)

    current, step = _net_pay(_flag_well([0.1] * n, [0.3] * n, [0.1] * n, codes), "current")
    benchmark, _ = _net_pay(_flag_well([0.1] * n, [0.3] * n, [0.1] * n, codes), "geolink")

    assert current["NET_PAY"] == [1, 0, 0, 1]
    assert benchmark["NET_PAY"] == [1] * n
    assert step.parameters["excluded_from_net_reservoir_and_pay"]["coal"]["net_pay_removed"] == 1.0
    assert any(text.startswith("Coal is not reservoir rock") for text in step.decisions)


@pytest.mark.parametrize("header_bit_size", [-999.25, 311.15])
def test_an_implausible_header_bit_size_is_ignored(header_bit_size):
    # A null or a 12.25 in bit written in mm. As given, it flags the whole
    # in-gauge 12.25 in hole as bad hole (CALI - BS > 1.5 in, or BS > 17.5 in);
    # "current" ignores it, and with no bit size BAD_HOLE flags only the two
    # samples with |DRHO| > 0.15 g/cm3.
    n = 8
    drho = np.full(n, 0.02)
    drho[[2, 5]] = 0.25

    def interpret(profile):
        well = _flag_well([0.1] * n, [0.2] * n, [0.2] * n, [SANDSTONE] * n)
        well.add_curve("CALI", np.full(n, 12.5), "in", "Caliper")
        well.add_curve("DRHO", drho, "g/cm3", "Density correction")
        well.header.bit_size = header_bit_size
        config = DynamicWorkflowConfig(calculation_profile=profile)
        result = DynamicWorkflow(config).run_selected_steps(well, [WorkflowStep.BAD_HOLE, WorkflowStep.NET_PAY])
        return well, result

    well, result = interpret("current")
    params = _step(result, "bad_hole").parameters
    assert (params["bit_size_source"], params["bit_size_used"]) == ("none", None)
    assert well.get("BAD_HOLE").data.astype(int).tolist() == [0, 0, 1, 0, 0, 1, 0, 0]
    assert well.get("NET_PAY").data.astype(int).tolist() == [1, 1, 0, 1, 1, 0, 1, 1]
    message = f"Header bit size {header_bit_size} is not a hole size in inches (3-36 in)"
    assert any(warning.startswith(message) for warning in result.warnings)
    assert any(text.startswith(message) for text in _step(result, "bad_hole").decisions)

    well, result = interpret("geolink")
    params = _step(result, "bad_hole").parameters
    assert (params["bit_size_source"], params["bit_size_used"]) == ("well_header", header_bit_size)
    assert well.get("BAD_HOLE").data.tolist() == [1.0] * n
    assert not result.warnings


def test_swirr_samples_follow_the_saturation_routing_clay_cutoff():
    result = _run("current", sw_routing_vcl_cutoff=0.15)
    assert _step(result, "saturation").parameters["vcl_cutoff_for_routing"] == 0.15
    assert _step(result, "permeability").parameters["pre_permeability_resflag"]["vcl_cutoff"] == 0.15


def test_benchmark_keeps_a_configs_cutoffs_over_header_values():
    # pp.interpret(config=...) under the "geolink" profile: the unset cutoffs
    # hold 0.3 and 0.7 there, which a header VCL_CUTOFF does not replace.
    def cutoff(config, **options):
        well = _flag_well(VCL, PHIE, SW, CODES)
        well.set_parameter("VCL_CUTOFF", 0.25)
        result = pp.interpret(well, config=config, **options)
        return _step(result, "net_pay").parameters["vcl_cutoff"]

    assert cutoff(DynamicWorkflowConfig(calculation_profile="geolink")) == 0.3
    assert cutoff(DynamicWorkflowConfig(), calculation_profile="geolink") == 0.3
    assert cutoff(DynamicWorkflowConfig()) == 0.25

    # In zoned batch runs a benchmark config's 0.3 counts as left at the default.
    benchmark = DynamicWorkflowConfig(calculation_profile="geolink", vcl_cutoff=0.3)
    assert _left_at_default(benchmark, "vcl_cutoff")
    assert not _left_at_default(dataclasses.replace(benchmark, vcl_cutoff=0.25), "vcl_cutoff")
    assert not _left_at_default(DynamicWorkflowConfig(vcl_cutoff=0.3), "vcl_cutoff")
    well = _flag_well(VCL, PHIE, SW, CODES)
    well.set_parameter("VCL_CUTOFF", 0.25)
    zoned = run_dynamic_workflow_with_zones(well, benchmark)
    assert _step(zoned.base_result, "net_pay").parameters["vcl_cutoff"] == 0.25


def test_summary_thickness_counts_samples_under_the_benchmark_profile():
    # Depths 2000, 2000.5, 2001, 2002, 2002.5 m: step 0.5 m (the median), so
    # 5 x 0.5 = 2.5 m as the trace counts it; the midpoint cells add up to
    # 0.5 + 0.5 + 0.75 + 0.75 + 0.5 = 3.0 m.
    def thickness(profile):
        n = 5
        well = _flag_well([0.1] * n, [0.2] * n, [0.2] * n, [SANDSTONE] * n)
        well.depth = np.array([2000.0, 2000.5, 2001.0, 2002.0, 2002.5])
        config = DynamicWorkflowConfig(calculation_profile=profile)
        result = DynamicWorkflow(config).run_selected_steps(well, [WorkflowStep.NET_PAY])
        return result.summary()["net_thickness"]["net_pay"]

    assert thickness("geolink") == 2.5
    assert thickness("current") == 3.0


# ---------------------------------------------------------------------------
# steps.net_pay: the same cutoffs and exclusions as the workflow
# ---------------------------------------------------------------------------
def _steps_net_pay(well: Well, **settings):
    run = Workflow("net pay", [steps.net_pay(**settings)]).run(well)
    flags = {name: run.well.get(name).data.astype(int).tolist() for name in ("NET_ROCK", "NET_RES", "NET_PAY")}
    return flags, run.steps[0]


def test_steps_net_pay_takes_each_samples_family_cutoffs():
    flags, step = _steps_net_pay(_mixed_well())
    workflow_flags, _ = _net_pay(_mixed_well(), "current")

    assert flags == workflow_flags
    assert flags["NET_PAY"] == [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
    assert step.parameters["vcl_cutoff_used"] == "lithology_dependent"
    assert step.decisions[0] == "NET_ROCK: VCL < 0.3 carbonate, 0.4 clastic -> 9 samples (4.5 m)"


def test_steps_net_pay_leaves_out_what_the_workflow_leaves_out():
    codes = [SANDSTONE, SANDSTONE, SANDSTONE, SANDSTONE, HALITE, COAL, ANHYDRITE, SANDSTONE]
    bad_hole = [0, 1, 1, 0, 1, 0, 0, 0]
    n = len(codes)

    def well(bit_size=True):
        logs = _flag_well([0.1] * n, [0.2] * n, [0.2] * n, codes, bad_hole)
        if bit_size:
            logs.add_curve("BS", np.full(n, 8.5), "in", "Bit size")
        return logs

    flags, step = _steps_net_pay(well())
    workflow_flags, workflow_step = _net_pay(well(), "current")
    plain, _ = _steps_net_pay(well(), exclude_non_reservoir=False)

    assert flags["NET_PAY"] == workflow_flags["NET_PAY"] == [1, 0, 0, 1, 0, 0, 0, 1]
    assert plain["NET_PAY"] == [1] * n
    assert (
        step.parameters["excluded_from_net_reservoir_and_pay"]
        == workflow_step.parameters["excluded_from_net_reservoir_and_pay"]
    )
    assert list(step.parameters["excluded_from_net_reservoir_and_pay"]) == ["evaporite", "coal", "bad_hole"]
    assert any(text.startswith("Bad hole (BAD_HOLE = 1)") for text in step.decisions)

    # Without a bit size BAD_HOLE leaves as given: the "current" bad-hole step
    # assumes no bit size, so the flag does not rest on an assumed one.
    no_bit_size = well(bit_size=False)
    no_bit_size.add_curve("DRHO", np.array([0.0, 0.2, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0]), "g/cm3", "DRHO")
    flags, step = _steps_net_pay(no_bit_size)
    assert flags["NET_PAY"] == [1, 0, 0, 1, 0, 0, 0, 1]
    assert step.parameters["excluded_from_net_reservoir_and_pay"]["bad_hole"]["samples"] == 3
