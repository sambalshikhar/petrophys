"""
Data type definitions for petrophysics calculations.

This module defines the data type classification system used throughout
the library to categorize inputs and outputs of each module.
"""

from enum import Enum


class DataType(Enum):
    """
    Data type classification per I/O specification.

    Attributes
    ----------
    CURVE : str
        Continuous log data (depth-indexed arrays)
    CONSTANT : str
        Fixed physical values (single values, depth-independent)
    DEFAULT : str
        User-adjustable parameters with default values
    HEADER : str
        Values from log header metadata
    COMPUTED : str
        Values derived from other modules
    """

    CURVE = "curve"
    CONSTANT = "constant"
    DEFAULT = "default"
    HEADER = "header"
    COMPUTED = "computed"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, value: str) -> "DataType":
        """
        Create DataType from string value.

        Parameters
        ----------
        value : str
            String representation of the data type

        Returns
        -------
        DataType
            Corresponding DataType enum member

        Raises
        ------
        ValueError
            If value does not match any DataType
        """
        value_lower = value.lower()
        for member in cls:
            if member.value == value_lower:
                return member
        raise ValueError(f"Unknown data type: {value}")
