"""Canonical saturation calculations of the "geolink" calculation profile.

The ``"geolink"`` calculation profile reproduces the GeoLink eval package,
whose shaly-sand saturation uses the Indonesia and Dual Water variants below.
The ``"current"`` profile uses the published forms,
:func:`~petrophys.interpretation.saturation.indonesia` (Poupon and Leveaux,
1971) and :func:`~petrophys.interpretation.saturation.dual_water`.

The profile's mineral solve is
:class:`~petrophys.interpretation.lithology.MultiMineralSolver` with
unweighted rows.
"""

import numpy as np

from petrophys.interpretation.saturation.sw_module import (
    SaturationModule,
    SwModel,
)


CALCULATION_REVISION = "canonical"


def canonical_indonesia(rt, rw, phi, vcl, rcl, a=1.0, m=2.0, n=2.0):
    """Indonesia variant of the "geolink" profile, with Sw on the sand term only.

    Solves ``1/sqrt(Rt) = Vcl^(1-Vcl/2)/sqrt(Rcl) + phi^(m/2)/sqrt(a*Rw) * Sw^(n/2)``:
    the clay term carries no ``Sw^(n/2)``, so it is subtracted from
    ``1/sqrt(Rt)`` and Sw is 0 wherever the clay term alone reaches
    ``1/sqrt(Rt)``. This is the Nigeria-Schlumberger form with shale exponent
    ``1 - Vcl/2``, not the Indonesia equation of Poupon and Leveaux (1971),
    which multiplies both terms by ``Sw^(n/2)``
    (:func:`~petrophys.interpretation.saturation.indonesia`). In shaly rock it
    gives a lower Sw: 0.069 against 0.275 at Rt=10, Rw=0.05, phi=0.2, Vcl=0.3,
    Rcl=2 and a=1, m=n=2.
    """
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
    """Dual Water variant of the "geolink" profile.

    :func:`~petrophys.interpretation.saturation.dual_water` with the apparent
    water resistivity mixed from the raw bound-water saturation
    ``Swb = Vcl * phi_cl / phi_t``, which exceeds 1 where
    ``Vcl * phi_cl > phi_t``; Swb is capped below 1 only for the final
    ``(Swt - Swb) / (1 - Swb)``.
    """
    phi_bw = vcl * phi_cl
    phi_t_safe = np.where(phi_t <= 0, 1e-10, phi_t)
    swb = phi_bw / phi_t_safe
    rwt = 1.0 / (swb / rwb + (1.0 - swb) / rw)
    swt = ((a * rwt) / (phi_t_safe ** m * rt)) ** (1.0 / n)
    swb_safe = np.where(swb >= 1.0, 0.999, swb)
    return np.clip((swt - swb_safe) / (1.0 - swb_safe), 0.0, 1.0)


class CanonicalSaturationModule(SaturationModule):
    """Saturation routing of the "geolink" profile.

    :class:`~petrophys.interpretation.saturation.SaturationModule` with the
    Indonesia and Dual Water variants of this module. Dual Water takes
    Rwb = 0.3 Rw when no Rwb is given, and Waxman-Smits returns the
    total-porosity Sw with CEC in meq/g.
    """

    def __init__(self, *args, **kwargs):
        kwargs.update(rwb_fallback=0.3, effective_waxman_smits=False)
        super().__init__(*args, **kwargs)

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
        clip_counts=None,
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
            clip_counts=clip_counts,
        )
