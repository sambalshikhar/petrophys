"""
Impermeability detection module.

Flags zones where porosity relationships indicate impermeability.
"""

from typing import Union
import numpy as np


def flag_impermeable(
    phi_s: Union[float, np.ndarray],
    phi_d: Union[float, np.ndarray],
    phi_n: Union[float, np.ndarray]
) -> Union[bool, np.ndarray]:
    """
    Flag zones where porosity relationships indicate impermeability.

    A zone is flagged as impermeable if either condition is met:
        Condition 1: phi_S > 1.5 * phi_D  (sonic porosity >> density porosity)
        Condition 2: phi_N > phi_D + 0.15  (neutron porosity >> density porosity)

    Parameters
    ----------
    phi_s : float or np.ndarray
        Sonic porosity (fraction)
    phi_d : float or np.ndarray
        Density porosity (fraction)
    phi_n : float or np.ndarray
        Neutron porosity (fraction)

    Returns
    -------
    bool or np.ndarray
        True where zone is flagged as impermeable
    """
    return (phi_s > 1.5 * phi_d) | (phi_n > phi_d + 0.15)
