"""
MHI classification.
"""

from typing import Union
import numpy as np

from petrophys.model.classifications import MhiClassification


def classify_mhi(mhi: Union[float, np.ndarray]) -> Union[MhiClassification, np.ndarray]:
    """Classify MHI values."""
    return MhiClassification.classify(mhi)
