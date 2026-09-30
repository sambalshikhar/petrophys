"""Thorium and uranium indicators from processed spectral gamma ray.

Thorium is insoluble and travels with detrital clay and heavy minerals.
Uranium dissolves in oxidising water and precipitates where conditions turn
reducing, typically with organic matter. Three standard products follow:

- :func:`classify_th_u_environment` reads the Th/U ratio as a
  depositional-environment indicator (Adams and Weaver, 1958; Fertl, 1979).
- :func:`thorium_clay_index` is a linear clay indicator from thorium, which
  uranium cannot disturb.
- :func:`uranium_stripped_gamma_ray` removes uranium's contribution from a
  total gamma ray, approximating the uranium-free gamma ray (CGR) when no
  potassium curve is available.

Concentrations are thorium and uranium in ppm; gamma rays are in API units.

References
----------
Adams, J.A.S. and Weaver, C.E. (1958). Thorium-to-uranium ratios as
indicators of sedimentary processes: example of concept of geochemical facies.
*AAPG Bulletin* 42(2), 387-430.

Fertl, W.H. (1979). Gamma ray spectral data assists in complex formation
evaluation. *The Log Analyst* 20(5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from petrophys.interpretation.geochemistry.spectral_gamma import DEFAULT_API_COEFFICIENTS

#: Th/U class boundaries of Adams and Weaver (1958), as popularised for well
#: logs by Fertl (1979): below 2 marine reducing (black, organic-rich shales),
#: 2 to 7 marine (grey and green shales, greywackes), above 7 continental
#: oxidising (weathered, terrestrial sediments with uranium leached out).
TH_U_ENVIRONMENT_THRESHOLDS: Tuple[float, float] = (2.0, 7.0)

#: Environment labels, in order of increasing Th/U.
TH_U_ENVIRONMENTS: Tuple[str, str, str] = ("marine reducing", "marine", "continental oxidising")


def _values(data) -> np.ndarray:
    return np.asarray(getattr(data, "data", data), dtype=float)


@dataclass(repr=False)
class ThUEnvironment:
    """Th/U ratio and depositional-environment class per sample.

    Attributes
    ----------
    th_u_ratio : numpy.ndarray
        Thorium / uranium, NaN where the sample is not rated (uranium below
        ``uranium_floor_ppm``, negative thorium or a missing value).
    environment : numpy.ndarray of object
        One of :data:`TH_U_ENVIRONMENTS` per rated sample, ``""`` elsewhere.
    uranium_floor_ppm : float
        Minimum uranium (ppm) for a sample to be rated.
    thresholds : tuple of float
        Class boundaries used (Th/U).
    """

    th_u_ratio: np.ndarray
    environment: np.ndarray
    uranium_floor_ppm: float
    thresholds: Tuple[float, float] = TH_U_ENVIRONMENT_THRESHOLDS

    @property
    def rated(self) -> np.ndarray:
        """Boolean mask of the samples that received a class."""
        return np.isfinite(self.th_u_ratio)

    def summary(self, groups: Optional[Mapping[str, np.ndarray]] = None):
        """Tabulate the environment classes, optionally for several groups.

        Parameters
        ----------
        groups : mapping of str to boolean array, optional
            Named sample selections, e.g. ``{"reservoir": in_reservoir}``.
            Only rated samples inside each selection count. Defaults to all
            samples.

        Returns
        -------
        pandas.DataFrame
            One row per group: ``samples`` (rated samples), ``median Th/U``
            and the percentage of rated samples in each environment
            (``"<environment> (%)"``).
        """
        import pandas as pd

        if groups is None:
            groups = {"all samples": np.ones(self.th_u_ratio.shape, dtype=bool)}
        rows = {}
        for name, selection in groups.items():
            selection = np.asarray(selection, dtype=bool).reshape(-1)
            if selection.shape != self.th_u_ratio.shape:
                raise ValueError(
                    f"group {name!r} has {selection.size} values for "
                    f"{self.th_u_ratio.size} samples"
                )
            chosen = selection & self.rated
            count = int(np.count_nonzero(chosen))
            row = {
                "samples": count,
                "median Th/U": float(np.median(self.th_u_ratio[chosen])) if count else np.nan,
            }
            for label in TH_U_ENVIRONMENTS:
                matches = np.count_nonzero(self.environment[chosen] == label)
                row[f"{label} (%)"] = 100.0 * matches / count if count else np.nan
            rows[str(name)] = row
        return pd.DataFrame.from_dict(rows, orient="index")

    def __repr__(self) -> str:
        rated = self.rated
        count = int(np.count_nonzero(rated))
        if not count:
            return f"ThUEnvironment(no samples rated; uranium floor {self.uranium_floor_ppm} ppm)"
        shares = ", ".join(
            f"{label} {100.0 * np.count_nonzero(self.environment == label) / count:.0f}%"
            for label in TH_U_ENVIRONMENTS
        )
        return (
            f"ThUEnvironment({count} of {rated.size} samples rated (U >= "
            f"{self.uranium_floor_ppm:g} ppm), median Th/U "
            f"{np.median(self.th_u_ratio[rated]):.1f}: {shares})"
        )


def classify_th_u_environment(
    thorium_ppm,
    uranium_ppm,
    *,
    uranium_floor_ppm: float = 0.5,
    thresholds: Sequence[float] = TH_U_ENVIRONMENT_THRESHOLDS,
) -> ThUEnvironment:
    """Classify depositional environment from the thorium/uranium ratio.

    Classes (Adams and Weaver, 1958; Fertl, 1979), with each boundary
    belonging to the lower class:

    ==========  =========================  ========================================
    Th/U        Environment                Typical rocks
    ==========  =========================  ========================================
    < 2         marine reducing            black, organic-rich shales that fix U
    2 - 7       marine                     grey and green shales, greywackes
    > 7         continental oxidising      weathered, terrestrial sediments
    ==========  =========================  ========================================

    Samples with uranium below ``uranium_floor_ppm`` are not rated: at low
    concentrations the stripped uranium reading is mostly counting noise (it
    can even go negative), and a ratio built on it means nothing. Read the
    classes over intervals rather than sample by sample, and confirm them
    with tops, lithology or core.

    Parameters
    ----------
    thorium_ppm, uranium_ppm : array-like or Curve
        Processed thorium and uranium concentrations in ppm (see
        :func:`~petrophys.interpretation.geochemistry.interpret_spectral_gamma`
        for unit standardisation).
    uranium_floor_ppm : float, default 0.5
        Minimum uranium, in ppm, for a sample to be rated.
    thresholds : pair of float, default (2.0, 7.0)
        Th/U class boundaries.

    Returns
    -------
    ThUEnvironment
        Per-sample Th/U and class, with ``summary()`` for tables.

    Raises
    ------
    ValueError
        If the arrays differ in length, the floor is negative, or the
        thresholds are not two increasing positive numbers.

    Examples
    --------
    >>> env = classify_th_u_environment([12.0, 11.0, 2.4, 9.0], [1.2, 6.0, 0.2, 2.0])
    >>> env.environment.tolist()
    ['continental oxidising', 'marine reducing', '', 'marine']
    >>> env.th_u_ratio.round(2).tolist()
    [10.0, 1.83, nan, 4.5]
    """
    thorium = _values(thorium_ppm).reshape(-1)
    uranium = _values(uranium_ppm).reshape(-1)
    if thorium.shape != uranium.shape:
        raise ValueError(
            f"thorium_ppm has {thorium.size} values but uranium_ppm has {uranium.size}"
        )
    floor = float(uranium_floor_ppm)
    if not np.isfinite(floor) or floor < 0:
        raise ValueError("uranium_floor_ppm must be a non-negative number")
    bounds = np.asarray(thresholds, dtype=float).reshape(-1)
    if bounds.size != 2 or not np.all(np.isfinite(bounds)) or not (0 < bounds[0] < bounds[1]):
        raise ValueError("thresholds must be two increasing positive numbers")

    with np.errstate(invalid="ignore"):
        rated = (
            np.isfinite(thorium) & np.isfinite(uranium) & (uranium >= floor)
            & (uranium > 0) & (thorium >= 0)
        )
    ratio = np.full(thorium.shape, np.nan)
    ratio[rated] = thorium[rated] / uranium[rated]
    labels = np.array(TH_U_ENVIRONMENTS, dtype=object)
    environment = np.full(thorium.shape, "", dtype=object)
    environment[rated] = labels[np.digitize(ratio[rated], bounds, right=True)]
    return ThUEnvironment(
        th_u_ratio=ratio,
        environment=environment,
        uranium_floor_ppm=floor,
        thresholds=(float(bounds[0]), float(bounds[1])),
    )


@dataclass(repr=False)
class ThoriumClayIndex:
    """Linear thorium clay index and the baselines behind it.

    Attributes
    ----------
    index : numpy.ndarray
        ``(Th - clean) / (shale - clean)`` clipped to 0-1 (NaN where thorium
        is missing).
    clean_ppm, shale_ppm : float
        Clean and shale thorium baselines (ppm).
    """

    index: np.ndarray
    clean_ppm: float
    shale_ppm: float

    def __repr__(self) -> str:
        finite = self.index[np.isfinite(self.index)]
        median = f"{np.median(finite):.2f}" if finite.size else "n/a"
        return (
            f"ThoriumClayIndex(clean {self.clean_ppm:.2f} ppm, shale {self.shale_ppm:.2f} ppm, "
            f"median index {median})"
        )


def thorium_clay_index(
    thorium_ppm,
    *,
    clean_ppm: Optional[float] = None,
    shale_ppm: Optional[float] = None,
    percentiles: Tuple[float, float] = (5.0, 95.0),
    mask=None,
) -> ThoriumClayIndex:
    """Linear clay index from thorium, which uranium cannot disturb.

    ``ITh = (Th - Th_clean) / (Th_shale - Th_clean)``, clipped to 0-1. It is
    the thorium analogue of the linear gamma-ray index IGR and, like IGR, an
    upper bound on clay volume before any non-linear correction. Compare it
    with IGR to find where uranium makes the GR clay volume read high (in
    uranium-rich beds) or, through a uranium-inflated shale baseline, low
    everywhere else.

    Parameters
    ----------
    thorium_ppm : array-like or Curve
        Thorium concentration in ppm.
    clean_ppm, shale_ppm : float, optional
        Clean and shale baselines in ppm. Unset baselines are picked as the
        ``percentiles`` of thorium inside ``mask``.
    percentiles : pair of float, default (5, 95)
        Percentiles for the clean and shale baselines. Pick the GR baselines
        of a comparison IGR with the same rule over the same window.
    mask : boolean array, optional
        Samples used to pick the baselines (for example the usable spectral
        window). Defaults to every finite sample. The index itself is
        computed for all samples.

    Returns
    -------
    ThoriumClayIndex
        The index and the baselines used.

    Raises
    ------
    ValueError
        If no finite thorium lies in ``mask``, or the shale baseline is not
        above the clean baseline.

    Examples
    --------
    >>> ith = thorium_clay_index([2.0, 6.0, 10.0, 14.0], clean_ppm=2.0, shale_ppm=14.0)
    >>> ith.index.round(3).tolist()
    [0.0, 0.333, 0.667, 1.0]
    """
    thorium = _values(thorium_ppm).reshape(-1)
    if clean_ppm is None or shale_ppm is None:
        selection = np.isfinite(thorium)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool).reshape(-1)
            if mask.shape != thorium.shape:
                raise ValueError(f"mask has {mask.size} values for {thorium.size} samples")
            selection &= mask
        if not np.any(selection):
            raise ValueError("No finite thorium values to pick baselines from")
        low, high = np.percentile(thorium[selection], [float(p) for p in percentiles])
        clean_ppm = float(low) if clean_ppm is None else clean_ppm
        shale_ppm = float(high) if shale_ppm is None else shale_ppm
    clean_ppm = float(clean_ppm)
    shale_ppm = float(shale_ppm)
    if not (np.isfinite(clean_ppm) and np.isfinite(shale_ppm)) or shale_ppm <= clean_ppm:
        raise ValueError(
            f"The shale baseline ({shale_ppm} ppm) must be above the clean baseline "
            f"({clean_ppm} ppm)"
        )
    index = np.clip((thorium - clean_ppm) / (shale_ppm - clean_ppm), 0.0, 1.0)
    return ThoriumClayIndex(index=index, clean_ppm=clean_ppm, shale_ppm=shale_ppm)


def uranium_stripped_gamma_ray(
    gamma_ray_api,
    uranium_ppm,
    *,
    api_per_ppm_uranium: float = DEFAULT_API_COEFFICIENTS["URANIUM"],
) -> np.ndarray:
    """Remove uranium's contribution from a total gamma ray.

    ``GR - k_U * U``, with ``k_U`` the uranium sensitivity (default
    8.09 API per ppm U). Without a potassium curve there is no computed
    uranium-free gamma ray (``CGR = 16.32 K + 3.93 Th``), and this
    difference approximates it. Pick GR clay-volume baselines on it so that
    uranium-rich beds do not set the shale baseline.

    It assumes the gamma ray and the spectral uranium share one calibration;
    check that with a least-squares fit of GR on Th and U before relying on
    it. Where uranium supplies most of the gamma ray, a small sensitivity
    error dominates the result; prefer a thorium clay index there.

    Parameters
    ----------
    gamma_ray_api : array-like or Curve
        Total gamma ray in API units.
    uranium_ppm : array-like or Curve
        Uranium concentration in ppm.
    api_per_ppm_uranium : float, default 8.09
        Uranium sensitivity (API per ppm).

    Returns
    -------
    numpy.ndarray
        Uranium-stripped gamma ray in API units (NaN where an input is
        missing).

    Raises
    ------
    ValueError
        If the arrays differ in length or the sensitivity is not positive.

    Examples
    --------
    >>> uranium_stripped_gamma_ray([80.0, 150.0], [2.0, 10.0]).round(2).tolist()
    [63.82, 69.1]
    """
    gamma_ray = _values(gamma_ray_api).reshape(-1)
    uranium = _values(uranium_ppm).reshape(-1)
    if gamma_ray.shape != uranium.shape:
        raise ValueError(
            f"gamma_ray_api has {gamma_ray.size} values but uranium_ppm has {uranium.size}"
        )
    coefficient = float(api_per_ppm_uranium)
    if not np.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("api_per_ppm_uranium must be a positive number")
    return gamma_ray - coefficient * uranium


__all__ = [
    "TH_U_ENVIRONMENTS",
    "TH_U_ENVIRONMENT_THRESHOLDS",
    "ThUEnvironment",
    "ThoriumClayIndex",
    "classify_th_u_environment",
    "thorium_clay_index",
    "uranium_stripped_gamma_ray",
]
