"""Canonical saturation calculations.

The canonical mineral inversion is
:class:`~petrophys.interpretation.lithology.MultiMineralSolverV1`.
"""

import numpy as np

from petrophys.interpretation.saturation.sw_module import (
    SaturationModule,
    SwModel,
)


CALCULATION_REVISION = "canonical"


def canonical_indonesia(rt, rw, phi, vcl, rcl, a=1.0, m=2.0, n=2.0):
    """Calculate Indonesia water saturation in the canonical operation order."""
    rt = np.asarray(rt, dtype=float)
    phi = np.asarray(phi, dtype=float)
    rt = np.where(rt <= 0, 1e-10, rt)
    phi = np.where(phi <= 0, 1e-10, phi)
    rcl = max(float(rcl), 1e-10)
    rw = np.asarray(rw, dtype=float)
    if rw.ndim == 0:
        rw = np.full(rt.shape, float(rw), dtype=float)
    else:
        rw = np.broadcast_to(rw, rt.shape).astype(float)
    rw = np.where(rw <= 0, 1e-10, rw)
    vcl = np.clip(vcl, 0.0, 0.99)

    clay_term = np.power(vcl, 1.0 - vcl / 2.0) / np.sqrt(rcl)
    sand_term = np.power(phi, m / 2.0) / np.sqrt(a * rw)
    lhs = np.maximum(1.0 / np.sqrt(rt) - clay_term, 0.0)
    sand_term = np.where(sand_term <= 0, 1e-10, sand_term)
    return np.clip(np.power(lhs / sand_term, 2.0 / n), 0.0, 1.0)


def canonical_dual_water(phi_t, rt, rw, rwb, vcl, phi_cl, a=1.0, m=2.0, n=2.0):
    """Calculate Dual Water saturation in the canonical operation order."""
    phi_bw = vcl * phi_cl
    phi_t_safe = np.where(phi_t <= 0, 1e-10, phi_t)
    swb = phi_bw / phi_t_safe
    rwt = 1.0 / (swb / rwb + (1.0 - swb) / rw)
    swt = ((a * rwt) / (phi_t_safe ** m * rt)) ** (1.0 / n)
    swb_safe = np.where(swb >= 1.0, 0.999, swb)
    return np.clip((swt - swb_safe) / (1.0 - swb_safe), 0.0, 1.0)


class CanonicalSaturationModule(SaturationModule):
    """Use canonical Indonesia and Dual Water equations within current routing."""

    def _run_model(
        self,
        model,
        rt,
        rw,
        rcl,
        vcl,
        phie,
        phit,
        cec,
        b_coeff,
        rho_ma,
        m_star,
        rwb,
        phi_cl,
        rtsh,
        vsh_exp,
        shale_exp,
        a_value=None,
        m_value=None,
        n_value=None,
    ):
        a_value = self.a if a_value is None else a_value
        m_value = self.m if m_value is None else m_value
        n_value = self.n if n_value is None else n_value
        if model == SwModel.INDONESIA:
            return canonical_indonesia(
                rt, rw, phie, vcl, rcl, a_value, m_value, n_value
            )
        if model == SwModel.DUAL_WATER:
            if rwb is None:
                rwb = rw * 0.3
            return canonical_dual_water(
                phit, rt, rw, rwb, vcl, phi_cl, a_value, m_value, n_value
            )
        return super()._run_model(
            model,
            rt,
            rw,
            rcl,
            vcl,
            phie,
            phit,
            cec,
            b_coeff,
            rho_ma,
            m_star,
            rwb,
            phi_cl,
            rtsh,
            vsh_exp,
            shale_exp,
            a_value,
            m_value,
            n_value,
        )
