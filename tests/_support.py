"""Shared setup for the test suite: import path, plotting backend, optional packages."""

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
# Test the checkout the suite belongs to, installed or not.
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def has(module: str) -> bool:
    """Whether an optional package is installed."""
    return importlib.util.find_spec(module) is not None


HAS_SKLEARN = has("sklearn")
HAS_DLISIO = has("dlisio")
HAS_MCP = sys.version_info >= (3, 10) and has("mcp")
