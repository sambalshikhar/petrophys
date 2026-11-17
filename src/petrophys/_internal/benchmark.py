"""Whether the code runs to reproduce a benchmark context package.

The GeoLink, COSTA and Danish context profiles apply dataset-specific
calculation rules within :func:`historical_benchmark`. Outside that context,
the library uses its default calculation rules.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_HISTORICAL: ContextVar[bool] = ContextVar("petrophys_historical_benchmark", default=False)


def reproducing_benchmark() -> bool:
    """True inside :func:`historical_benchmark`."""
    return _HISTORICAL.get()


@contextmanager
def historical_benchmark() -> Iterator[None]:
    """Run the block with the calculations of the benchmark packages."""
    token = _HISTORICAL.set(True)
    try:
        yield
    finally:
        _HISTORICAL.reset(token)
