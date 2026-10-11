"""
One-curve classification bands, as the diagnostics analyzers apply them.

A section that grades one curve assigns every valid sample a base label and
then relabels, step by step and in order, the samples each threshold holds;
later steps win. :class:`Bands` holds those steps as data, so the analyzer that
classifies with them and anyone reading its classes (which label a value takes,
the edges between classes) use the same definition, including on which side of
each edge a value that equals it falls.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

_OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}


@dataclass(frozen=True)
class Bands:
    """A one-curve classification: ``base`` for every valid sample, then each
    ``(op, threshold, label)`` step in order (``op`` one of ``>``, ``>=``,
    ``<``, ``<=``).

    Examples
    --------
    >>> b = Bands("POOR", ((">", 1.0, "FAIR"), (">", 10.0, "GOOD")))
    >>> list(b.classify([0.5, 1.0, 5.0, 50.0])), b.edges, b.labels
    (['POOR', 'POOR', 'FAIR', 'GOOD'], (1.0, 10.0), ('POOR', 'FAIR', 'GOOD'))
    """

    base: str
    steps: Tuple[Tuple[str, float, str], ...]

    def __post_init__(self):
        bad = [op for op, _, _ in self.steps if op not in _OPS]
        if bad:
            raise ValueError(f"unknown comparison(s) {bad}; use one of {sorted(_OPS)}")

    def classify(self, values, valid: Optional[np.ndarray] = None, missing: str = "NO_DATA") -> np.ndarray:
        """The label of every sample (``missing`` where the value is not valid; default validity: finite)."""
        x = np.asarray(values, dtype=float)
        ok = np.isfinite(x) if valid is None else np.asarray(valid, dtype=bool)
        out = np.full(x.shape, missing, dtype=object)
        out[ok] = self.base
        with np.errstate(invalid="ignore"):
            for op, threshold, label in self.steps:
                out[ok & _OPS[op](x, threshold)] = label
        return out

    def label_of(self, value: float) -> Optional[str]:
        """The label of one value (None when it is not finite)."""
        if value is None or not np.isfinite(value):
            return None
        return str(self.classify(np.array([float(value)]))[0])

    @property
    def edges(self) -> Tuple[float, ...]:
        """The thresholds, ascending."""
        return tuple(sorted({float(t) for _, t, _ in self.steps}))

    @property
    def labels(self) -> Tuple[str, ...]:
        """The labels from the lowest to the highest values."""
        e = self.edges
        if not e:
            return (self.base,)
        gaps = [e[0] - 1.0] + [(a + b) / 2 for a, b in zip(e, e[1:])] + [e[-1] + 1.0]
        order = []
        for v in gaps:
            label = self.label_of(v)
            if label not in order:
                order.append(label)
        return tuple(order)


def bands_of(analyzer: type, name: str) -> Bands:
    """The :class:`Bands` an analyzer class publishes under ``name`` (e.g. ``"DISTRIBUTION_BANDS"``)."""
    bands = getattr(analyzer, name, None)
    if not isinstance(bands, Bands):
        raise ValueError(f"{getattr(analyzer, '__name__', analyzer)} has no Bands named {name!r}")
    return bands


__all__ = ["Bands", "bands_of"]
