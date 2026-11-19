"""Warning and exception classes raised by Petrophys.

Filter Petrophys data-quality warnings like any other warning category::

    import warnings
    from petrophys.exceptions import DataQualityWarning

    warnings.simplefilter("error", DataQualityWarning)  # fail fast
"""

__all__ = ["DataQualityWarning"]


class DataQualityWarning(UserWarning):
    """An input or calculated result is outside its expected physical domain.

    Raised, for example, when most samples of a curve fall outside the
    physically plausible range for its declared unit and are removed, which
    usually means the unit label is wrong. The message states whether values
    were removed, replaced or retained. Ordinary missing NaN samples do not
    by themselves trigger this warning.
    """
