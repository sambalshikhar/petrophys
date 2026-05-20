"""
NMR (Nuclear Magnetic Resonance) log analysis.

Handles pre-processed NMR logs with T2 bin distributions (P1–P8),
total porosity (MPHI), Free Fluid Index (MFFI), and Bulk Volume
Irreducible (MBVI).

Typical workflow::

    from petrophys.nmr import NMRData

    nmr = NMRData.from_csv("nmr.csv")
    t2lm  = nmr.t2_lm()
    k_sdr = nmr.permeability_sdr()       # SDR model
    k_tc  = nmr.permeability_coates()    # Timur-Coates model
    r_pore = nmr.pore_size()             # modal pore radius (µm)
    df = nmr.to_dataframe()              # export all results

Default T2 bin centres: [4, 8, 16, 32, 64, 128, 256, 512] ms.
Default T2 cutoff: 32 ms (clastics).  Use T2_CUTOFF_CARBONATE = 92 ms
for carbonates.
"""

from petrophys.nmr.nmr_data import (
    NMRData,
    DEFAULT_T2_CENTERS,
    T2_CUTOFF_CLASTIC,
    T2_CUTOFF_CARBONATE,
)
from petrophys.nmr.permeability import (
    sdr_permeability,
    coates_permeability,
    t2lm_from_bins,
    bvi_from_bins,
    ffi_from_bins,
)

__all__ = [
    # Data container
    "NMRData",
    # Constants
    "DEFAULT_T2_CENTERS",
    "T2_CUTOFF_CLASTIC",
    "T2_CUTOFF_CARBONATE",
    # Standalone permeability functions
    "sdr_permeability",
    "coates_permeability",
    "t2lm_from_bins",
    "bvi_from_bins",
    "ffi_from_bins",
]
