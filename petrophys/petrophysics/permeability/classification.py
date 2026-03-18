"""
Permeability classification.
"""

from typing import Union
import numpy as np

from petrophys.base.classifications import PermeabilityClassification


def classify_permeability(
    perm: Union[float, np.ndarray]
) -> Union[PermeabilityClassification, np.ndarray]:
    """Classify permeability values."""
    return PermeabilityClassification.classify(perm)
