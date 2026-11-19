"""A message that names the inputs it asks for, as data.

Petrophys's advice reads ``pass formations=...`` or ``add steps.mud_filtrate(rmf=..., rmf_temp_f=...)``: written for a
Python caller. :class:`Advice` is that message as a ``str`` (every reader of messages reads it unchanged) that also
carries the names of the inputs it asks for (``.inputs``), so a caller with its own calls (an agent's tools, a UI) can
offer them without parsing the sentence. Joining messages with :func:`joined` keeps their inputs.
"""
from __future__ import annotations

from typing import Iterable, Tuple


class Advice(str):
    """A message (a ``str``) and the inputs it asks for (``inputs``, parameter names in order)."""

    inputs: Tuple[str, ...]

    def __new__(cls, text: str, inputs: Iterable[str] = ()) -> "Advice":
        obj = super().__new__(cls, text)
        obj.inputs = tuple(dict.fromkeys(str(i) for i in inputs))
        return obj

    def __reduce__(self):
        return (Advice, (str(self), self.inputs))


def inputs_of(message: object) -> Tuple[str, ...]:
    """The inputs a message asks for: an :class:`Advice`'s, or one carried as an exception's or a warning's first
    argument; ``()`` for any other message."""
    found = getattr(message, "inputs", None)
    if isinstance(found, tuple):
        return found
    args = getattr(message, "args", None)
    if args:
        return inputs_of(args[0]) if args[0] is not message else ()
    return ()


def joined(parts: Iterable[str], sep: str = "; ") -> str:
    """Messages joined with ``sep``; an :class:`Advice` when any part asks for inputs (their inputs together)."""
    parts = list(parts)
    text = sep.join(str(p) for p in parts)
    inputs = [i for p in parts for i in inputs_of(p)]
    return Advice(text, inputs) if inputs else text


def restated(text: str, *originals: object) -> str:
    """``text`` (a message reworded: prefixed, capitalised) keeping the inputs its originals ask for."""
    inputs = [i for o in originals for i in inputs_of(o)]
    return Advice(text, inputs) if inputs else text
