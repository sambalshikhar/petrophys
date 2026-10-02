"""Candidate fluid contacts from unlabeled, QC-reviewed station pressures.

This implements gradient analysis, not interpretation of raw pretest time
traces. Segmentation and phase assignment are separate; a pressure discontinuity
or overlapping density hypotheses must not silently become a free-water level.
"""
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .formation_pressure import PressureGradient


# Broad screening assumptions, not universal PVT bounds or vendor defaults.
# Overlap is intentional: pressure density cannot always identify a fluid.
DEFAULT_FLUID_DENSITIES = {"gas": (0.02, 0.50), "oil": (0.45, 1.05),
                           "water": (0.90, 1.30)}
_DEPTH_M = {"m": 1.0, "ft": 0.3048}
_PRESSURE_BAR = {"bar": 1.0, "psi": 0.06894757293168, "kPa": 0.01, "MPa": 10.0}
_G = 0.0980665  # bar/m per g/cm3, standard gravity


@dataclass(frozen=True)
class PressureBranch:
    """Inferred depth segment; fluid candidates are hypotheses, not samples."""

    line: PressureGradient
    depth_min: float
    depth_max: float
    density_g_cc: float
    density_interval_g_cc: tuple
    fluid_candidates: tuple


@dataclass(frozen=True)
class PressureContact:
    """Pressure-line intersection and conditional +/-2-sigma sensitivity.

    This is not a saturation-defined contact. The interval excludes uncertainty
    in segment choice, fluid priors, connectivity and depth measurements.
    """

    kind: str
    depth: float
    pressure: float
    depth_interval: tuple


@dataclass(frozen=True)
class PressureContactResult:
    """Unlabeled inference with explicit ambiguity and original-row alignment.

    ``station_branch`` is -1 for unused rows, otherwise a zero-based branch.
    ``suspect`` flags large residuals or implausible branches for review;
    measurements are not deleted.
    Model scores are penalized Huber losses, not probabilities or formal BIC.
    """

    status: str
    branches: tuple
    contacts: tuple
    fluid_sequences: tuple
    station_branch: np.ndarray
    suspect: np.ndarray
    model_scores: dict
    messages: tuple
    depth_unit: str
    pressure_unit: str
    depth_reference: str

    @property
    def fwl_depth(self):
        """Return an FWL only for an unambiguous, physically consistent model."""
        if self.status != "interpreted":
            return None
        return next((c.depth for c in self.contacts
                     if c.kind in {"oil_water", "gas_water"}), None)


@dataclass
class _Segment:
    start: int
    stop: int
    slope: float
    z0: float
    p0: float
    covariance: np.ndarray
    residual: np.ndarray
    loss: float

    def pressure_at(self, z):
        return self.p0 + self.slope * (z - self.z0)

    def variance_at(self, z):
        v = np.array([1.0, z - self.z0])
        return float(v @ self.covariance @ v)


def infer_pressure_contacts(
    depth, pressure, *, use=None, depth_unit="m", pressure_unit="bar",
    depth_reference="TVDSS", max_branches=3, min_points=3,
    min_depth_span=None, pressure_noise=None, fluid_density_ranges=None,
    score_margin=6.0,
):
    """Infer candidate gradients, fluid types and contacts without fluid codes.

    Supply equilibrated formation-pressure picks on a positive-down TVD datum,
    from one prospective connected reservoir with a common pressure reference.
    ``use`` is an optional Boolean QC mask, never a fluid-interval mask.
    Nonfinite measurements are retained as unused in the row-aligned result.

    Sort by depth, enumerate contiguous one/two/three-branch partitions and fit
    each branch by Huber IRLS. Minimize ``2*Huber_loss + (3*k-1)*log(n)``: the
    penalty counts two line parameters per branch and k-1 split positions.
    The winning segmentation is chosen BEFORE density interpretation; the
    algorithm does not force the lowest branch to be water.

    Convert dP/dTVD to fluid density and enumerate allowed gas/oil/water orders.
    Default screening ranges (g/cm3) are gas .02-.50, oil .45-1.05 and water
    .90-1.30. These overlapping, configurable heuristics are not universal fluid
    properties. Supply reservoir-condition PVT ranges where available; a
    subset of fluid names can express known phase presence/absence.

    At least ``min_points`` distinct depths and ``min_depth_span`` per branch
    are required (defaults: 3 and 1 metre equivalent). ``pressure_noise`` is a
    positive assumed pressure error scale in the input unit (default .05 bar
    equivalent, NOT an instrument accuracy claim). It controls robust fitting,
    model selection, suspect flags and conditional sensitivity intervals.

    Close scores across branch counts (difference < score_margin), overlapping
    phase hypotheses or weak gradient separation return ambiguity. Nonphysical
    gradients or intersections outside the station gap beyond their conditional
    sensitivity return inconsistency, possibly indicating separate compartments.
    Single gradients never produce FWL. Sampling/depletion/supercharging can
    mimic contacts; this is a reviewable interpretation, not proof of fluid
    identity or reservoir connectivity. No report contacts are accepted.
    """
    if depth_unit not in _DEPTH_M or pressure_unit not in _PRESSURE_BAR:
        raise ValueError("Use depth_unit m/ft and pressure_unit bar/psi/kPa/MPa")
    if depth_reference not in {"TVDSS", "TVD_MSL", "TVD_RKB", "TVD_RT"}:
        raise ValueError("Use an explicit positive-down TVD datum, not MD")
    if (not isinstance(max_branches, int) or max_branches not in {1, 2, 3}
            or not isinstance(min_points, int) or min_points < 2):
        raise ValueError("max_branches must be 1-3 and min_points an integer >=2")
    if not np.isfinite(score_margin) or score_margin < 0:
        raise ValueError("score_margin must be finite and nonnegative")
    dz, dp = _DEPTH_M[depth_unit], _PRESSURE_BAR[pressure_unit]
    span = 1.0 if min_depth_span is None else float(min_depth_span) * dz
    noise = 0.05 if pressure_noise is None else float(pressure_noise) * dp
    if not np.isfinite(span) or span <= 0 or not np.isfinite(noise) or noise <= 0:
        raise ValueError("min_depth_span and pressure_noise must be finite and positive")
    ranges = dict(DEFAULT_FLUID_DENSITIES if fluid_density_ranges is None else fluid_density_ranges)
    if not ranges or set(ranges) - {"gas", "oil", "water"}:
        raise ValueError("fluid_density_ranges must map gas/oil/water to positive bounds")
    for bounds in ranges.values():
        if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or not 0 < bounds[0] < bounds[1]:
            raise ValueError("Fluid density bounds must be finite, positive and increasing")
    z_all, p_all = np.asarray(depth, dtype=float), np.asarray(pressure, dtype=float)
    if z_all.ndim != 1 or p_all.shape != z_all.shape:
        raise ValueError("depth and pressure must be matching 1-D arrays")
    keep = np.ones(z_all.size, dtype=bool) if use is None else np.asarray(use)
    if keep.shape != z_all.shape or keep.dtype.kind != "b":
        raise ValueError("use must be a matching Boolean QC mask")
    good = keep & np.isfinite(z_all) & np.isfinite(p_all)
    rows = np.flatnonzero(good)
    rows = rows[np.argsort(z_all[rows], kind="stable")]
    z, p = z_all[rows] * dz, p_all[rows] * dp
    assignments = np.full(z_all.size, -1, dtype=int)
    suspect = np.zeros(z_all.size, dtype=bool)
    n = len(z)
    messages = ["Density ranges and pressure error scale are interpretation assumptions."]
    if len(rows) < len(z_all):
        messages.append(f"Excluded {len(z_all)-len(rows)} nonfinite or QC-unselected stations.")

    def result(status, branches=(), contacts=(), sequences=(), scores=None):
        return PressureContactResult(status, tuple(branches), tuple(contacts), tuple(sequences),
                                     assignments, suspect, scores or {}, tuple(messages),
                                     depth_unit, pressure_unit, depth_reference)

    if n < min_points or np.unique(z).size < min_points or np.ptp(z) < span:
        messages.append("Insufficient distinct station depths or depth span for a gradient.")
        return result("insufficient_data")
    if n > 300:
        raise ValueError("Limit analysis to 300 QC-reviewed pressure stations per reservoir; not raw time samples")

    cache = {}

    def segment(a, b):
        key = (a, b)
        if key in cache:
            return cache[key]
        zz, pp = z[a:b], p[a:b]
        if len(zz) < min_points or np.unique(zz).size < min_points or np.ptp(zz) < span:
            cache[key] = None
            return None
        z0, base = float(zz.mean()), float(pp.mean())
        x = np.column_stack([np.ones(len(zz)), zz-z0])
        target = pp-base
        beta = np.linalg.lstsq(x, target, rcond=None)[0]
        for _ in range(30):
            residual = target - x @ beta
            weights = np.minimum(1.0, 1.345*noise / np.maximum(np.abs(residual), 1e-30))
            root_w = np.sqrt(weights)
            updated = np.linalg.lstsq(x*root_w[:, None], target*root_w, rcond=None)[0]
            if np.max(np.abs(x @ (updated-beta))) < noise*1e-6:
                beta = updated
                break
            beta = updated
        residual = target-x @ beta
        weights = np.minimum(1.0, 1.345*noise / np.maximum(np.abs(residual), 1e-30))
        scaled = np.abs(residual)/noise
        loss = float(np.where(scaled <= 1.345, 0.5*scaled**2,
                              1.345*(scaled-0.5*1.345)).sum())
        scatter = max(noise, 1.4826 * float(np.median(np.abs(residual-np.median(residual)))))
        cov = scatter**2 * np.linalg.inv(x.T @ (weights[:, None]*x))
        value = _Segment(a, b, float(beta[1]), z0, base+float(beta[0]), cov, residual, loss)
        cache[key] = value
        return value

    best_by_count = {}
    # Equal-depth repeats stay together. Different branches need distinct TVDs.
    cuts = [i for i in range(min_points, n-min_points+1) if z[i] > z[i-1]]
    for k in range(1, max_branches+1):
        for split in combinations(cuts, k-1):
            edges = (0,)+split+(n,)
            if any(b-a < min_points for a, b in zip(edges[:-1], edges[1:])):
                continue
            parts = [segment(a, b) for a, b in zip(edges[:-1], edges[1:])]
            if any(s is None for s in parts):
                continue
            score = 2*sum(s.loss for s in parts)+(3*k-1)*np.log(n)
            if k not in best_by_count or score < best_by_count[k][0]:
                best_by_count[k] = (float(score), parts)
    ranked = sorted(best_by_count.values(), key=lambda item: item[0])
    score, parts = ranked[0]
    scores = {k: value[0] for k, value in best_by_count.items()}
    branches = []
    for i, seg in enumerate(parts):
        density = seg.slope/_G
        delta = 2*np.sqrt(seg.covariance[1, 1])/_G
        interval = (float(density-delta), float(density+delta))
        fluids = tuple(f for f in ("gas", "oil", "water") if f in ranges and
                       interval[1] >= ranges[f][0] and interval[0] <= ranges[f][1])
        line = PressureGradient(seg.slope*dz/dp, seg.z0/dz, seg.p0/dp,
                                seg.stop-seg.start, float(np.sqrt(np.mean(seg.residual**2)))/dp)
        branches.append(PressureBranch(line, z[seg.start]/dz, z[seg.stop-1]/dz,
                                       float(density), interval, fluids))
        assignments[rows[seg.start:seg.stop]] = i
        suspect[rows[seg.start:seg.stop]] = np.abs(seg.residual) > 3*noise
        if not fluids or seg.slope <= 0:
            suspect[rows[seg.start:seg.stop]] = True
    if any(s.slope <= 0 for s in parts):
        messages.append("Nonpositive gradient: review pressure picks, datum and compartments.")
        return result("inconsistent", branches, scores=scores)
    if len(parts) == 1:
        messages.append("Only one supported pressure gradient; no contact can be determined.")
        return result("single_gradient", branches, scores=scores)
    k = len(parts)
    sequences = [seq for seq in combinations(("gas", "oil", "water"), k)
                 if all(f in b.fluid_candidates for f, b in zip(seq, branches))]
    intersections = []
    weak = False
    for upper, lower in zip(parts[:-1], parts[1:]):
        contrast = lower.slope-upper.slope
        separation = 2*np.sqrt(upper.covariance[1, 1]+lower.covariance[1, 1])
        if contrast <= 0:
            messages.append("Gradients do not increase downwards; possible compartments or bad pressure picks.")
            return result("inconsistent", branches, sequences=sequences, scores=scores)
        weak |= contrast <= separation
        center = (z[upper.stop-1]+z[lower.start])/2
        contact = center+(upper.pressure_at(center)-lower.pressure_at(center))/contrast
        delta = 2*np.sqrt(upper.variance_at(contact)+lower.variance_at(contact))/contrast
        if contact+delta < z[upper.stop-1] or contact-delta > z[lower.start]:
            messages.append("Line intersection is inconsistent with the station gap; possible pressure compartment.")
            return result("inconsistent", branches, sequences=sequences, scores=scores)
        intersections.append((contact, lower.pressure_at(contact), delta))
    ambiguous = False
    if len(sequences) != 1:
        ambiguous = True
        messages.append("Fluid identities are not unique under the density ranges; supply PVT/phase evidence.")
    if weak:
        ambiguous = True
        messages.append("Gradient separation is small relative to the assumed pressure error.")
    if len(ranked) > 1 and ranked[1][0]-score < score_margin:
        ambiguous = True
        messages.append("Competing branch counts have similar scores; contact interpretation is ambiguous.")
    if suspect[rows].sum() > 0.2*n:
        ambiguous = True
        messages.append("More than 20% of selected measurements have large residuals; review QC.")
    if not sequences:
        messages.append("No gas/oil/water sequence fits these gradients; consider separate hydraulic zones.")
    contacts = []
    for i, (zz, pp, delta) in enumerate(intersections):
        kind = "_".join(sequences[0][i:i+2]) if len(sequences) == 1 else "unresolved"
        contacts.append(PressureContact(kind, float(zz/dz), float(pp/dp),
                                        (float((zz-delta)/dz), float((zz+delta)/dz))))
    messages.append("Contact intervals are conditional sensitivity only; fluid and segment choice remain uncertain.")
    return result("ambiguous" if ambiguous else "interpreted", branches, contacts, sequences, scores)
