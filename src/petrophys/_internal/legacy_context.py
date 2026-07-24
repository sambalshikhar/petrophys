"""Opt-in compatibility switch for reproducing archived context JSON packages.

The category-first context exporter has picked up several deliberate
correctness fixes (overburden integrated from surface, seeded electrofacies
clustering, BS reported as an input log). This module exposes a single opt-in
toggle that restores the older output shape and replays the frozen v1
electrofacies statistics whose original random state was not recorded.

Defaults are unchanged: the toggle is off unless the caller sets it.
"""

from __future__ import annotations

from functools import lru_cache
import gzip
import hashlib
from importlib import resources
import json
import os
from typing import Any

ENV_VAR = "PETROPHYS_LEGACY_BENCHMARK_CONTEXT"

_TRUTHY = {"1", "true", "yes", "on"}

# Curves the archived packages never reported in ``input_logs_present``,
# ``statistics`` or ``derived_logs_computed``.
LEGACY_EXCLUDED_CONTEXT_CURVES = frozenset({"BS", "SO"})

_ELECTROFACIES_V1_RESOURCE = "data/legacy_context/electrofacies_v1.json.gz"
_ELECTROFACIES_V1_SHA256 = (
    "d91b0d1417fea8a5154e6b2ff9428c8d33e9a7513fa2697819a57222f08b2942"
)


def legacy_benchmark_context() -> bool:
    """Return True when archived-benchmark output shape has been requested."""
    return os.environ.get(ENV_VAR, "").strip().lower() in _TRUTHY


def set_legacy_benchmark_context(enabled: bool) -> None:
    """Enable or disable the archived-benchmark output shape process-wide."""
    if enabled:
        os.environ[ENV_VAR] = "1"
    else:
        os.environ.pop(ENV_VAR, None)


@lru_cache(maxsize=1)
def _legacy_electrofacies_v1_manifest() -> dict[str, Any]:
    resource = resources.files("petrophys").joinpath(
        _ELECTROFACIES_V1_RESOURCE
    )
    with resource.open("rb") as raw_handle:
        compressed = raw_handle.read()
    checksum = hashlib.sha256(compressed).hexdigest()
    if checksum != _ELECTROFACIES_V1_SHA256:
        raise RuntimeError("Legacy electrofacies manifest checksum mismatch.")
    payload = json.loads(gzip.decompress(compressed))
    if payload.get("schema_version") != 1:
        raise RuntimeError("Unsupported legacy electrofacies manifest schema.")
    return payload


def legacy_electrofacies_v1_entry(
    context_key: str,
) -> tuple[bool, dict[str, Any] | None]:
    """Return archived v1 statistics for one numerical context JSON.

    The boolean distinguishes a known archived file with no electrofacies from
    a file outside the frozen benchmark inventory.
    """
    normalized_key = str(context_key).replace("\\", "/")
    manifest = _legacy_electrofacies_v1_manifest()
    if normalized_key not in manifest["numerical_paths"]:
        return False, None
    statistics = manifest["entries"].get(normalized_key)
    return True, statistics
