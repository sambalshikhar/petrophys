import numpy as np
import pandas as pd
import pytest

from petrophys import Well
from petrophys.preprocessing.imputation import PretrainedCurveImputer
from petrophys.io.export import export_las


class MeanFeatureRegressor:
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        frame = pd.DataFrame(X)
        return frame.mean(axis=1).to_numpy(dtype=float)


def _test_pack():
    feature_curves = ["GR", "NPHI", "DTC"]
    return {
        "schema_version": 1,
        "name": "test-pack.joblib",
        "feature_curves": feature_curves,
        "models": {
            "rf": {
                "RHOB": {
                    "estimator": MeanFeatureRegressor(),
                    "features": feature_curves,
                }
            }
        },
    }


def test_pretrained_imputer_only_fills_missing_target_samples():
    well = Well("IMPUTE")
    well.depth = np.array([100.0, 101.0, 102.0, 103.0])
    well.add_curve("RHOB", np.array([2.10, np.nan, 2.40, np.nan]), "g/cm3")
    well.add_curve("GR", np.array([60.0, 70.0, 80.0, np.nan]), "API")
    well.add_curve("NPHI", np.array([0.20, 0.30, 0.25, np.nan]), "v/v")
    well.add_curve("DTC", np.array([90.0, 100.0, 110.0, np.nan]), "us/ft")

    result = PretrainedCurveImputer(model_pack=_test_pack()).impute_missing(
        well,
        "RHOB",
        model_name="rf",
        min_available_features=2,
    )

    assert result.imputed_count == 1
    assert result.skipped_count == 1
    np.testing.assert_allclose(
        well["RHOB"].data,
        np.array([2.10, (70.0 + 0.30 + 100.0) / 3.0, 2.40, np.nan]),
        equal_nan=True,
    )
    assert "ML_IMPUTATION:" in result.note
    assert "RHOB" in result.note
    assert well.get_parameter("ML_IMPUTATION_NOTES") == [result.note]


def test_pretrained_imputer_does_not_create_absent_target_curve():
    well = Well("ABSENT")
    well.depth = np.array([0.0, 1.0])
    well.add_curve("GR", np.array([50.0, 60.0]), "API")

    with pytest.raises(KeyError, match="only fills null samples"):
        PretrainedCurveImputer(model_pack=_test_pack()).impute_missing(well, "RHOB")

    assert "RHOB" not in well.curves


def test_export_las_writes_ml_imputation_note_to_other_section(tmp_path):
    well = Well("EXPORT-NOTE")
    well.depth = np.array([100.0, 101.0])
    well.add_curve("RHOB", np.array([2.1, 2.2]), "g/cm3")
    note = "ML_IMPUTATION: RHOB missing samples were imputed with pretrained rf model."
    well.set_parameter("ML_IMPUTATION_NOTES", [note])

    output = tmp_path / "imputed.las"
    export_las(well, output)

    text = output.read_text()
    assert "~Other" in text
    assert note in text
    assert "ML_IMPUTATION_NOTES" not in text


def test_validation_filter_removes_estimators_but_preserves_companions():
    from petrophys.preprocessing.imputation.training import filter_model_pack

    pack = _test_pack()
    pack['models']['knn'] = {'RHOB': pack['models']['rf']['RHOB']}
    pack['models']['rf']['GR'] = pack['models']['rf']['RHOB']
    pack['models']['rf']['SP'] = pack['models']['rf']['RHOB']
    pack['metadata'] = {'targets': {
        'RHOB': {'models': {
            'rf': {'r2': 0.7, 'evaluation': 'held_out_wells'},
            'knn': {'r2': 0.1, 'evaluation': 'held_out_wells'},
        }},
        'GR': {'models': {'rf': {'r2': 0.14, 'evaluation': 'held_out_wells'}}},
        'SP': {'models': {'rf': {'r2': 0.99, 'evaluation': 'in_sample_single_well'}}},
    }}
    filtered = filter_model_pack(pack)
    assert filtered['target_curves'] == ['RHOB']
    assert list(filtered['models']) == ['rf']
    assert list(filtered['models']['rf']) == ['RHOB']
    assert 'GR' in filtered['models']['rf']['RHOB']['features']
    assert 'GR' in pack['models']['rf']
    assert 'knn' in pack['metadata']['targets']['RHOB']['models']


def test_auto_selection_rejects_weak_held_out_score_by_default():
    well = Well('WEAK-MODEL')
    well.depth = np.array([1.0, 2.0])
    well.add_curve('RHOB', [np.nan, 2.4], 'g/cc')
    pack = _test_pack()
    pack['metadata'] = {'targets': {'RHOB': {'models': {
        'rf': {'r2': 0.14, 'evaluation': 'held_out_wells'},
    }}}}
    with pytest.raises(ValueError, match='R2 >= 0.500'):
        PretrainedCurveImputer(model_pack=pack).impute_missing(well, 'RHOB')
    assert np.isnan(well['RHOB'].data[0])


def test_incompatible_model_pack_raises_model_compatibility_error(monkeypatch, tmp_path):
    joblib = pytest.importorskip("joblib")

    from petrophys.preprocessing.imputation import curve_imputer

    def broken_load(path):
        raise KeyError(121)

    monkeypatch.setattr(joblib, "load", broken_load)
    with pytest.raises(curve_imputer.ModelCompatibilityError, match="require scikit-learn 1.8.x"):
        curve_imputer._load_serialized_model_pack(tmp_path / "pack.joblib")
