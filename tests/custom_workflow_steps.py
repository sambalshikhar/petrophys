"""Importable custom workflow steps for ``test_custom_workflow.py``.

Saved workflows store custom steps by import path, so the save/load tests
need steps defined in a module rather than inside a test function.
"""

import numpy as np
import pandas as pd

import petrophys as pp


@pp.workflow_step(requires=["GR"], produces={"VSH_LIN": "v/v"})
def linear_vsh(well, gr_clean=20.0, gr_shale=120.0):
    """Linear shale volume from gamma ray."""
    gr = well.get("GR").data
    return {"VSH_LIN": np.clip((gr - gr_clean) / (gr_shale - gr_clean), 0.0, 1.0)}


@pp.workflow_step(requires=["PHIE", "SW"], produces={"HCPV_FRAC": "v/v"})
def hydrocarbon_fraction(well):
    """Hydrocarbon-filled porosity, PHIE * (1 - SW)."""
    return {"HCPV_FRAC": well.get("PHIE").data * (1.0 - well.get("SW").data)}


@pp.workflow_step(requires=["GR"], produces={"BROKEN": ""})
def failing_step(well):
    """Always fails, to exercise error handling."""
    raise RuntimeError("deliberate failure")


@pp.workflow_step(requires=["GR"])
def gr_histogram(well, bins=4):
    """Gamma-ray histogram as a table artifact."""
    counts, edges = np.histogram(well.get("GR").data[np.isfinite(well.get("GR").data)], bins=bins)
    return pd.DataFrame({"low": edges[:-1], "high": edges[1:], "count": counts})


def not_a_step(well):
    """A plain function, not decorated with workflow_step."""
    return None
