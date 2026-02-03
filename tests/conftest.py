"""
Pytest configuration and fixtures for petrophys tests.
"""

import pytest
import numpy as np


@pytest.fixture
def sample_depth():
    """Sample depth array."""
    return np.arange(8000, 8100, 0.5)


@pytest.fixture
def sample_gr():
    """Sample GR log data."""
    np.random.seed(42)
    return np.random.uniform(20, 120, 200)


@pytest.fixture
def sample_rhob():
    """Sample bulk density data."""
    np.random.seed(42)
    return np.random.uniform(2.2, 2.7, 200)


@pytest.fixture
def sample_nphi():
    """Sample neutron porosity data."""
    np.random.seed(42)
    return np.random.uniform(0.05, 0.35, 200)


@pytest.fixture
def sample_rt():
    """Sample resistivity data."""
    np.random.seed(42)
    return np.random.uniform(1, 100, 200)


@pytest.fixture
def sample_well(sample_depth, sample_gr, sample_rhob, sample_nphi, sample_rt):
    """Create a sample Well object for testing."""
    from petrophys import Well

    well = Well()
    well.header = {
        "WELL": "TEST-1",
        "COMP": "Test Company",
        "FLD": "Test Field",
    }

    well.add_curve("DEPT", sample_depth, "ft", "Depth")
    well.add_curve("GR", sample_gr, "API", "Gamma Ray")
    well.add_curve("RHOB", sample_rhob, "g/cc", "Bulk Density")
    well.add_curve("NPHI", sample_nphi, "v/v", "Neutron Porosity")
    well.add_curve("RT", sample_rt, "ohm.m", "True Resistivity")

    return well
