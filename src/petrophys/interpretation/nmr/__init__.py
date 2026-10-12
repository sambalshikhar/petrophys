"""
NMR (Nuclear Magnetic Resonance) log analysis.

Handles pre-processed NMR logs with T2 bin distributions (P1–P8),
total porosity (MPHI), Free Fluid Index (MFFI), and Bulk Volume
Irreducible (MBVI).

Typical workflow::

    from petrophys.interpretation.nmr import NMRData

    nmr = NMRData.from_las(petrophys.sample("daydream1_cmr"),
                       mphi_mnemonic="TCMR", mffi_mnemonic="CMFF")
    t2lm  = nmr.t2_lm()
    k_sdr = nmr.permeability_sdr()       # SDR model
    k_tc  = nmr.permeability_coates()    # Timur-Coates model
    r_pore = nmr.pore_size()             # modal pore radius (µm)
    df = nmr.to_dataframe()              # export all results

Default T2 bin centres: [4, 8, 16, 32, 64, 128, 256, 512] ms.
Default T2 cutoff: 33 ms (clastics).  Use T2_CUTOFF_CARBONATE = 92 ms
for carbonates.  The bin that straddles the cutoff is split by log-T2
width; pass ``partial_bins=False`` to count whole bins by their centre.
"""

from petrophys.interpretation.nmr.nmr_data import (
    NMRData,
    DEFAULT_T2_CENTERS,
    T2_CUTOFF_CLASTIC,
    T2_CUTOFF_CARBONATE,
)
from petrophys.interpretation.nmr.calibration import NMRCalibration
from petrophys.interpretation.nmr.permeability import (
    sdr_permeability,
    coates_permeability,
    t2lm_from_bins,
    bvi_from_bins,
    ffi_from_bins,
    t2_fraction_below_cutoff,
)

__all__ = [
    # Data container
    "NMRData",
    # Calibration
    "NMRCalibration",
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
    "t2_fraction_below_cutoff",
]
