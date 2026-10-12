"""
Permeability classification.
"""

from typing import Union
import numpy as np

from petrophys.model.classifications import PermeabilityClassification


def classify_permeability(
    perm: Union[float, np.ndarray]
) -> Union[PermeabilityClassification, np.ndarray]:
    """Classify permeability values in mD (``None`` where a value is not finite)."""
    return PermeabilityClassification.classify(perm)
