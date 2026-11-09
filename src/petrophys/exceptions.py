"""Warning and exception classes raised by Petrophys.

Filter Petrophys data-quality warnings like any other warning category::

    import warnings
    from petrophys.exceptions import DataQualityWarning

    warnings.simplefilter("error", DataQualityWarning)  # fail fast
"""

__all__ = ["DataQualityWarning"]


class DataQualityWarning(UserWarning):
    """Input data looks wrong and was changed or ignored.

    Raised, for example, when most samples of a curve fall outside the
    physically plausible range for its declared unit and are removed, which
    usually means the unit label is wrong.
    """
