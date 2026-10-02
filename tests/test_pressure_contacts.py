"""Unlabeled pressure interpretation: identifiability and failure modes."""
import numpy as np
import pandas as pd
import pytest
import lasio

from petrophys import sample
from petrophys.interpretation.saturation.height import (
    infer_pressure_contacts, read_formation_pressure,
)


def column(densities=(0.7, 1.1), contacts=(1200,)):
    z = np.arange(1000, 1401, 10, dtype=float)
    p = 100 + densities[0]*0.0980665*(z-1000)
    for contact, contrast in zip(contacts, np.diff(densities)):
        p += contrast*0.0980665*np.maximum(z-contact, 0)
    return z, p


@pytest.mark.parametrize("densities,contacts,sequence", [
    ((0.7, 1.1), (1200,), ("oil", "water")),
    ((0.2, 1.1), (1200,), ("gas", "water")),
    ((0.2, 0.7, 1.1), (1100, 1200), ("gas", "oil", "water")),
    ((0.2, 0.7), (1200,), ("gas", "oil")),
])
def test_known_contacts_without_labels(densities, contacts, sequence):
    z, p = column(densities, contacts)
    # Scramble acquisition order: inferred assignments must align with input.
    order = np.random.default_rng(1).permutation(len(z))
    result = infer_pressure_contacts(z[order], p[order])
    assert result.status == "interpreted"
    assert result.fluid_sequences == (sequence,)
    np.testing.assert_allclose([b.density_g_cc for b in result.branches], densities)
    np.testing.assert_allclose([c.depth for c in result.contacts], contacts, atol=1e-8)
    for c in result.contacts:
        assert c.depth_interval[0] < c.depth < c.depth_interval[1]
    for i, branch in enumerate(result.branches):
        zz = z[order][result.station_branch == i]
        assert zz.min() == pytest.approx(branch.depth_min)
    assert result.fwl_depth == (pytest.approx(1200) if sequence[-1] == "water" else None)


def test_single_gradient_missing_values_qc_and_suspect_outlier():
    z, p = column((1.05,), ())
    p[8] = np.nan
    use = np.ones(len(z), dtype=bool)
    use[3] = False
    result = infer_pressure_contacts(z, p, use=use)
    assert result.status == "single_gradient"
    assert result.fwl_depth is None and not result.contacts
    assert (result.station_branch[[3, 8]] == -1).all()
    p[20] += 5
    result = infer_pressure_contacts(z, p, use=use)
    assert result.fwl_depth is None  # an isolated bad pick is not a water leg
    assert result.suspect[20]


def test_compartment_pressure_jump_is_not_a_fluid_contact():
    z, p = column((0.7,), ())
    p[z >= 1200] += 3
    result = infer_pressure_contacts(z, p)
    assert result.status in {"ambiguous", "inconsistent"}
    assert result.fwl_depth is None
    assert any("compartment" in m or "hydraulic" in m for m in result.messages)


def test_overlapping_density_hypotheses_need_external_phase_evidence():
    z, p = column((0.2, 0.97), (1200,))
    result = infer_pressure_contacts(z, p)
    assert result.status == "ambiguous"
    assert result.fwl_depth is None
    assert set(result.fluid_sequences) == {("gas", "oil"), ("gas", "water")}
    constrained = infer_pressure_contacts(
        z, p, fluid_density_ranges={"gas": (0.1, 0.4), "water": (0.94, 1.1)})
    assert constrained.status == "interpreted"
    assert constrained.fwl_depth == pytest.approx(1200)


def test_units_and_datum_invariance():
    z, p = column()
    base = infer_pressure_contacts(z, p)
    shifted = infer_pressure_contacts(z/0.3048+50000, p/0.06894757293168+3000,
                                      depth_unit="ft", pressure_unit="psi",
                                      depth_reference="TVD_RKB")
    assert shifted.status == base.status
    assert (shifted.fwl_depth-50000)*0.3048 == pytest.approx(base.fwl_depth)
    np.testing.assert_allclose([b.density_g_cc for b in shifted.branches],
                               [b.density_g_cc for b in base.branches])


def test_real_unlabeled_csv_las_and_label_independence(tmp_path):
    source = pd.read_csv(sample("35_11_13_mdt_points"))
    points = source.drop(columns=["FLUID", "USE_FIT"])
    # Selection comes only from the original measurement remarks. Gas stations
    # previously excluded by the labeled example are now included.
    points["USE_FIT"] = points.REMARKS.eq("ok")
    path = tmp_path / "stations.csv"
    points.to_csv(path, index=False)
    csv = read_formation_pressure(path)
    assert csv.points.use_fit.sum() == 21
    assert csv.points.fluid.eq("unknown").all()
    with pytest.raises(ValueError, match="infer_contacts"):
        csv.fit_fwl()
    fit = csv.infer_contacts()
    assert fit.status == "interpreted"
    assert fit.fwl_depth == pytest.approx(3113.1740677202133)
    assert abs(fit.fwl_depth-3112.6) < 2  # external check, never a model input
    # Inference itself must not read any label, even when a table contains one.
    csv.points["fluid"] = "deliberately-wrong-label"
    assert csv.infer_contacts().fwl_depth == fit.fwl_depth
    las = lasio.read(sample("35_11_13_mdt"))
    las.delete_curve("FLUID")
    del las.params["FLDMAP"]
    las["USE_FIT"] = points.USE_FIT.astype(int).to_numpy()
    las_path = tmp_path / "unlabeled.las"
    las.write(str(las_path), version=2)
    from_las = read_formation_pressure(las_path).infer_contacts()
    assert from_las.fwl_depth == pytest.approx(fit.fwl_depth)


def test_digitized_second_well_returns_ambiguity_without_labels():
    p = pd.read_csv(sample("35_11_10_fensfjord_mdt_points")).drop(columns=["FLUID", "USE_FIT"])
    data = read_formation_pressure(p)
    result = data.infer_contacts()
    assert result.status == "ambiguous"
    assert result.fwl_depth is None
    assert abs(result.contacts[0].depth-2290) < 5
    assert len(result.fluid_sequences) > 1
    from petrophys.visualization import create_pressure_contact_plot
    import matplotlib.pyplot as plt
    fig = create_pressure_contact_plot(data, result)
    assert fig.axes[0].yaxis_inverted()
    assert "ambiguous" in fig.axes[0].get_title()
    assert any("Candidate" in text for text in fig.axes[0].get_legend_handles_labels()[1])
    plt.close(fig)


def test_raw_time_based_mdt_las_is_not_treated_as_station_pressures():
    # Mnemonics/units match the inspected JAMSTEC C0020A raw pretest delivery.
    las = lasio.LASFile()
    las.append_curve("TIME", [1, 2, 3], unit="S")
    las.append_curve("PQDVP1", [2696.99]*3, unit="M")
    las.append_curve("PQQP1", [4358, 4250, 4300], unit="PSI")
    with pytest.raises(ValueError, match="pretest"):
        read_formation_pressure(las, depth_column="PQDVP1", pressure_column="PQQP1",
                                depth_reference="TVDSS")


def test_insufficient_distinct_depths_and_invalid_settings():
    result = infer_pressure_contacts([100, 100, 101], [10, 10, 10.1])
    assert result.status == "insufficient_data"
    assert result.fwl_depth is None
    z, p = column()
    for kwargs in [{"pressure_noise": 0}, {"min_points": 1}, {"max_branches": 4},
                   {"depth_reference": "MD"}, {"score_margin": np.nan},
                   {"fluid_density_ranges": {"oil": (1, 0.5)}},
                   {"use": ["False"]*len(z)}]:
        with pytest.raises(ValueError):
            infer_pressure_contacts(z, p, **kwargs)
