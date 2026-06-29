"""Tests for the XRF mineral_proxies fix and the geochemistry panel."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from petrophys.core import XRFData
from petrophys.visualization import create_xrf_panel


def _xrf(n=80):
    rng = np.random.RandomState(0)
    elements = {
        "SI": rng.uniform(30, 45, n), "AL": rng.uniform(0.5, 8, n),
        "CA": rng.uniform(0, 12, n), "FE": rng.uniform(0, 5, n),
        "K": rng.uniform(0, 1.5, n), "MG": rng.uniform(0, 1, n),
        "S": rng.uniform(0, 6, n), "MN": rng.uniform(0, 0.2, n),
        "TI": rng.uniform(0, 0.5, n),
    }
    return XRFData(depth=np.linspace(3815, 3841, n), elements=elements)


def test_xrfdata_get_does_not_raise_on_arrays():
    # regression: `a or b` on numpy arrays raised "truth value ambiguous"
    x = _xrf()
    assert x.get("Si") is not None      # case-insensitive
    assert x.get("SI") is not None
    assert x.get("NOPE") is None


def test_mineral_proxies_returns_dict():
    mp = _xrf().mineral_proxies()
    assert isinstance(mp, dict)
    assert "quartz_index" in mp
    assert np.isfinite(np.asarray(mp["quartz_index"], dtype=float)).any()


def test_create_xrf_panel_renders():
    fig = create_xrf_panel(_xrf())
    titles = [ax.get_title() for ax in fig.axes]
    assert any("Clastic" in t for t in titles)
    assert any("Carbonate" in t for t in titles)
    assert any("Mineral proxies" in t for t in titles)
    plt.close(fig)
