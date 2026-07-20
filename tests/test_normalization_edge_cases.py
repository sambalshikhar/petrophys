import numpy as np

from petrophys.preprocessing.normalization.histogram_matching import histogram_matching
from petrophys.preprocessing.normalization.normalization_module import (
    linear_normalization,
    normalize_log,
)


def test_normalize_log_accepts_numpy_arrays_without_treating_data_buffer_as_curve():
    normalized = normalize_log(np.array([5.0, 5.0]), log_min=0.0, log_max=150.0)

    np.testing.assert_allclose(normalized, [75.0, 75.0])


def test_linear_normalization_preserves_nan_for_constant_source_range():
    normalized = linear_normalization(
        np.array([5.0, np.nan]),
        log_min=0.0,
        log_max=150.0,
        log_low=5.0,
        log_high=5.0,
    )

    assert normalized[0] == 75.0
    assert np.isnan(normalized[1])


def test_histogram_matching_keeps_identical_constant_arrays_unchanged():
    source = np.array([5.0, 5.0, np.nan])
    matched = histogram_matching(source, np.array([5.0, 5.0]))

    np.testing.assert_allclose(matched[:2], [5.0, 5.0])
    assert np.isnan(matched[2])
