"""
One-curve classification bands, as the diagnostics analyzers apply them.

A section that grades one curve assigns every valid sample a base label and
then relabels, step by step and in order, the samples each threshold holds;
later steps win. :class:`Bands` holds those steps as data, so the analyzer that
classifies with them and anyone reading its classes (which label a value takes,
the edges between classes) use the same definition, including on which side of
each edge a value that equals it falls.

:class:`Bands` and :func:`bands_of` live in :mod:`petrophys.model.class_tables`,
next to the canonical class table of each property; this module re-exports
them.
"""

from petrophys.model.class_tables import Bands, bands_of

__all__ = ["Bands", "bands_of"]
