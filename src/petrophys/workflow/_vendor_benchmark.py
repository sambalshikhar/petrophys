"""May 2026 vendor-lithology benchmark compatibility calculations.

This module is intentionally private.  It preserves the numerical behavior
used to create the historical context benchmark at commit ``3b8051c`` while
the public interpretation modules continue to use the current, more general
algorithms.  Only :class:`petrophys.workflow.DynamicWorkflow` selects these
calculations, and only for the explicit vendor benchmark profile.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import nnls

from petrophys.interpretation.lithology.matrix_properties import (
    get_mineral_properties,
)
from petrophys.interpretation.saturation.sw_module import (
    SaturationModule,
    SwModel,
)


BENCHMARK_PROFILE_VERSION = "may_2026"


def _mineral_endpoints(name: str) -> Dict[str, float]:
    properties = get_mineral_properties(name)
    return {
        "density": properties["density"],
        "nphi": properties["nphi"],
        "pef": properties["pef"],
        "u": properties["u"],
        "dt": properties["slowness"],
    }


def _constrain_mineral_fractions(
    fractions: np.ndarray,
    available_fraction: float,
) -> np.ndarray:
    constrained = np.asarray(fractions, dtype=float).copy()
    constrained = np.clip(constrained, 0.0, max(float(available_fraction), 0.0))
    total = float(np.nansum(constrained))
    if total > available_fraction and total > 0.0:
        constrained *= float(available_fraction) / total
    return constrained


class May2026BenchmarkMultiMineralSolver:
    """Reproduce the mineral inversion used by the May 2026 benchmark.

    In particular, this keeps the historical equation selection and the
    absolute 0.05 closure gate.  The generic solver deliberately does not use
    this implementation because it supports missing-neutron alternatives and
    improved QC behavior.
    """

    def __init__(
        self,
        minerals: Optional[list] = None,
        minerals_no_pef: Optional[list] = None,
        clay_mineral: str = "clay",
        fluid: str = "water",
    ):
        self.minerals = minerals or ["quartz", "calcite", "dolomite", "anhydrite"]
        self.minerals_no_pef = minerals_no_pef or self.minerals[:3]
        all_minerals = list(dict.fromkeys(self.minerals + self.minerals_no_pef))
        self.mineral_props = {
            mineral: _mineral_endpoints(mineral) for mineral in all_minerals
        }
        self.fluid_props = _mineral_endpoints(fluid)
        self.clay_props = _mineral_endpoints(clay_mineral)

    def _solve_sample(
        self,
        rhob: float,
        nphi: float,
        vcl: float,
        phie: float,
        pef: Optional[float],
        dt: Optional[float],
    ) -> Tuple[list, Tuple[float, ...]]:
        vcl = float(np.clip(vcl, 0.0, 1.0))
        phie = float(np.clip(phie, 0.0, 1.0))
        active_minerals = (
            self.minerals if pef is not None or dt is not None else self.minerals_no_pef
        )

        mineral_props = [self.mineral_props[name] for name in active_minerals]
        matrix_rows = []
        response = []

        def add_row(mineral_values, clay_value, fluid_value, log_value):
            matrix_rows.append(mineral_values)
            response.append(log_value - vcl * clay_value - phie * fluid_value)

        add_row([1.0] * len(active_minerals), 1.0, 1.0, 1.0)
        add_row(
            [item["density"] for item in mineral_props],
            self.clay_props["density"],
            self.fluid_props["density"],
            rhob,
        )
        add_row(
            [item["nphi"] for item in mineral_props],
            self.clay_props["nphi"],
            self.fluid_props["nphi"],
            nphi,
        )
        if pef is not None:
            add_row(
                [item["u"] for item in mineral_props],
                self.clay_props["u"],
                self.fluid_props["u"],
                pef * rhob,
            )
        if dt is not None:
            add_row(
                [item["dt"] for item in mineral_props],
                self.clay_props["dt"],
                self.fluid_props["dt"],
                dt,
            )

        fractions, _ = nnls(np.asarray(matrix_rows), np.asarray(response))
        available_fraction = 1.0 - vcl - phie
        if abs(float(fractions.sum()) - available_fraction) > 0.05:
            return active_minerals, tuple(np.nan for _ in active_minerals)
        fractions = _constrain_mineral_fractions(fractions, available_fraction)
        return active_minerals, tuple(float(value) for value in fractions)

    def solve(
        self,
        rhob,
        nphi,
        vcl,
        phie,
        pef=None,
        dt=None,
        core_phi=None,
    ) -> Dict[str, np.ndarray]:
        del core_phi
        rhob = np.atleast_1d(np.asarray(rhob, dtype=float))
        nphi = np.atleast_1d(np.asarray(nphi, dtype=float))
        vcl = np.atleast_1d(np.asarray(vcl, dtype=float))
        phie = np.atleast_1d(np.asarray(phie, dtype=float))
        pef_values = (
            np.atleast_1d(np.asarray(pef, dtype=float)) if pef is not None else None
        )
        dt_values = (
            np.atleast_1d(np.asarray(dt, dtype=float)) if dt is not None else None
        )

        all_minerals = list(dict.fromkeys(self.minerals + self.minerals_no_pef))
        result = {mineral: np.zeros(len(rhob)) for mineral in all_minerals}
        result["vcl"] = vcl.copy()
        result["phie"] = phie.copy()
        result["qc_flag"] = np.zeros(len(rhob), dtype=bool)

        for index in range(len(rhob)):
            if np.any(np.isnan([rhob[index], nphi[index], vcl[index], phie[index]])):
                for mineral in all_minerals:
                    result[mineral][index] = np.nan
                result["qc_flag"][index] = True
                continue

            pef_value = float(pef_values[index]) if pef_values is not None else None
            dt_value = float(dt_values[index]) if dt_values is not None else None
            if pef_value is not None and np.isnan(pef_value):
                pef_value = None
            if dt_value is not None and np.isnan(dt_value):
                dt_value = None

            active_minerals, fractions = self._solve_sample(
                rhob[index],
                nphi[index],
                vcl[index],
                phie[index],
                pef_value,
                dt_value,
            )
            for mineral, value in zip(active_minerals, fractions):
                result[mineral][index] = value
            if np.any(np.isnan(fractions)):
                result["qc_flag"][index] = True

        result["solved"] = ~result["qc_flag"]
        return result


def may_2026_indonesia(rt, rw, phi, vcl, rcl, a=1.0, m=2.0, n=2.0):
    """Historical Indonesia implementation retained for benchmark parity."""
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


def may_2026_dual_water(phi_t, rt, rw, rwb, vcl, phi_cl, a=1.0, m=2.0, n=2.0):
    """Historical Dual Water operation order retained for benchmark parity."""
    phi_bw = vcl * phi_cl
    phi_t_safe = np.where(phi_t <= 0, 1e-10, phi_t)
    swb = phi_bw / phi_t_safe
    rwt = 1.0 / (swb / rwb + (1.0 - swb) / rw)
    swt = ((a * rwt) / (phi_t_safe ** m * rt)) ** (1.0 / n)
    swb_safe = np.where(swb >= 1.0, 0.999, swb)
    return np.clip((swt - swb_safe) / (1.0 - swb_safe), 0.0, 1.0)


class May2026BenchmarkSaturationModule(SaturationModule):
    """Use historical Indonesia/Dual Water equations within current routing."""

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
            return may_2026_indonesia(
                rt, rw, phie, vcl, rcl, a_value, m_value, n_value
            )
        if model == SwModel.DUAL_WATER:
            if rwb is None:
                rwb = rw * 0.3
            return may_2026_dual_water(
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
