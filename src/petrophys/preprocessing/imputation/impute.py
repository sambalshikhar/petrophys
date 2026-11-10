"""
Fill the missing samples of a log curve with a model trained on the fly.

A gradient-boosted tree model (scikit-learn ``HistGradientBoostingRegressor``)
learns the curve from the other logs where it was measured: in the well
itself, and in any training wells you pass. It then predicts the curve where
it is missing. Nothing is stored between calls.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from petrophys.preprocessing.standardization.alias_map import (
    get_aliases,
    get_standard_mnemonic,
    get_standard_unit,
)
from petrophys.preprocessing.standardization.curve_standardizer import (
    CURVE_PLAUSIBLE_RANGES,
    POROSITY_FRACTION_CURVES,
)
from petrophys.preprocessing.standardization.unit_converter import get_conversion_factor
from petrophys.preprocessing.standardization.user_aliases import user_standard_name

if TYPE_CHECKING:
    from petrophys.model.well import Well

PathLike = Union[str, Path]

#: Logs the model can learn from; whichever of them a well has are used.
FEATURE_CURVES = (
    "GR", "SP", "CALI", "BS", "RHOB", "DRHO", "NPHI", "PEF", "DTC", "DTS",
    "RDEP", "RMED", "RSHA", "RXO",
)
#: Curves modelled as log10, since they span decades.
_LOG_CURVES = {"RDEP", "RMED", "RSHA", "RXO", "RT", "RXOZ", "PERM"}
#: Fewest samples with the curve measured needed to train.
MIN_TRAINING_SAMPLES = 200
_MAX_TRAINING_ROWS = 200_000


def _standard_name(well: Any, mnemonic: str) -> str:
    name = user_standard_name(well, mnemonic) or get_standard_mnemonic(str(mnemonic).strip().upper())
    return str(name).upper()


def _find_curve(well: "Well", standard: str) -> Optional[str]:
    """The well's mnemonic for a standard curve name, or None."""
    standard = standard.upper()
    for name in (standard, *(alias.upper() for alias in get_aliases(standard))):
        if well.has_curve(name):
            return name
    for name in well.curves:
        if _standard_name(well, name) == standard:
            return str(name)
    return None


def _standard_values(well: "Well", mnemonic: str, standard: str) -> Tuple[np.ndarray, float]:
    """Curve values in the standard unit, with nulls and implausible values as NaN.

    Returns the values and the factor from the curve's unit to the standard
    unit, so predictions can be written back in the curve's own unit.
    """
    curve = well[mnemonic]
    data = np.asarray(curve.data, dtype=float).reshape(-1).copy()
    data[~np.isfinite(data) | (data <= -900.0)] = np.nan
    factor = 1.0
    unit = str(getattr(curve, "unit", "") or "").strip()
    standard_unit = str(get_standard_unit(standard) or "")
    if unit and standard_unit:
        conversion = get_conversion_factor(unit, standard_unit)
        if conversion is None:
            raise ValueError(
                f"{well.name}: {mnemonic} is in {unit!r}, which cannot be converted to {standard_unit!r}"
            )
        factor = float(conversion)
        data *= factor
    if standard in POROSITY_FRACTION_CURVES:
        finite = data[np.isfinite(data)]
        if finite.size and float(np.median(finite)) > 1.0:
            data /= 100.0
            factor *= 0.01
    limits = CURVE_PLAUSIBLE_RANGES.get(standard)
    if limits is not None:
        data[(data < limits[0]) | (data > limits[1])] = np.nan
    return data, factor


def _well_table(well: "Well", names: Sequence[str]) -> Dict[str, np.ndarray]:
    table = {}
    for standard in names:
        mnemonic = _find_curve(well, standard)
        if mnemonic is not None:
            table[standard] = _standard_values(well, mnemonic, standard)[0]
    return table


def _as_well(source: Any) -> "Well":
    from petrophys.model.well import Well

    if isinstance(source, Well):
        return source
    if isinstance(getattr(source, "well", None), Well):  # an interpretation result
        return source.well
    from petrophys.api import read

    return read(source)


def _store_result(well: "Well", base: str, values: np.ndarray, unit: str,
                  output: Optional[PathLike]) -> "Well":
    name = f"{base}_IMPUTED"
    if well.has_curve(name):
        well.remove_curve(name)
    well.add_curve(name, values, unit=unit, description=f"{base}, with its gaps imputed")
    if output is not None:
        well.to_las(output)
    return well


def impute(
    well: Any,
    curve: str,
    *,
    train_wells: Optional[Sequence[Any]] = None,
    output: Optional[PathLike] = None,
) -> Any:
    """Reconstruct the missing samples of a curve.

    A gradient-boosted tree model learns ``curve`` from the other logs
    (gamma ray, density, neutron, sonic, resistivity and so on) wherever it
    was measured, in this well and in ``train_wells``, and predicts it where
    it is missing. The result is added as ``<CURVE>_IMPUTED``, named after
    the well's own mnemonic for the curve (``ZDEN_IMPUTED`` for a ``ZDEN``
    density asked for as ``"RHOB"``): the measured values where the curve
    has them and the predictions in its gaps. The original curve is left as
    it is.

    Parameters
    ----------
    well : Well, DynamicWorkflowResult or path
        The well, an interpretation result (its interpreted well), or a LAS,
        DLIS, CSV or TSV file.
    curve : str
        The curve to reconstruct, by its mnemonic or standard name
        (``"RHOB"``, ``"DT"``, ``"NPHI"`` ...).
    train_wells : Well, DynamicWorkflowResult or path, or a sequence of them, optional
        Other wells to learn from, typically from the same field. Needed when
        the well lacks the curve altogether, and useful when its measured
        part is short.
    output : path, optional
        Write the well, with the new curve, to this LAS file.

    Returns
    -------
    Well or DynamicWorkflowResult
        What was passed, with ``<CURVE>_IMPUTED`` added to its well: the
        result for a result, else the well.

    Raises
    ------
    ValueError
        If fewer than 200 samples with the curve measured are available to
        learn from, or no log the model can use is present where the curve
        is missing.

    Notes
    -----
    Tree models reproduce the relations seen in training and do not
    extrapolate: where the input logs fall outside the range they had in the
    training data (the 1st to 99th percentile, widened by a tenth of that
    span on each side), a warning gives the number of such samples. Depth is not
    used as an input. Imputation needs scikit-learn (``pip install
    "petrophys[ml]"``).

    Examples
    --------
    >>> well = pp.impute("16_7-1.las", "RHOB", output="16_7-1_imputed.las")
    >>> well = pp.impute(target, "DT", train_wells=["HW-3.las", "HW-4.las"])
    """
    given = well
    well = _as_well(well)
    returned = given if well is not given and hasattr(given, "steps_executed") else None
    target = _standard_name(well, curve)
    target_mnemonic = _find_curve(well, target)

    n = well.n_samples
    if target_mnemonic is not None:
        measured, factor = _standard_values(well, target_mnemonic, target)
        unit = str(getattr(well[target_mnemonic], "unit", "") or "")
        recorded = np.asarray(well[target_mnemonic].data, dtype=float).reshape(-1).copy()
        recorded[~np.isfinite(recorded) | (recorded <= -900.0)] = np.nan
    else:
        measured, factor = np.full(n, np.nan), 1.0
        unit = str(get_standard_unit(target) or "")
        recorded = measured.copy()
    # Only null samples are filled; recorded values, plausible or not, stay.
    missing = ~np.isfinite(recorded)
    if not np.any(missing):
        warnings.warn(f"{well.name}: {curve} has no missing samples; nothing to impute", stacklevel=2)
        _store_result(well, target_mnemonic or str(curve).strip().upper(), recorded, unit, output)
        return returned if returned is not None else well

    if isinstance(train_wells, (str, Path)) or hasattr(train_wells, "n_samples") or hasattr(train_wells, "well"):
        train_wells = [train_wells]  # one well, file or result
    others = [_as_well(source) for source in (train_wells or [])]
    if target_mnemonic is None and not any(_find_curve(other, target) is not None for other in others):
        raise ValueError(
            f"{well.name} has no {curve} curve; pass train_wells= with wells that have it"
        )

    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError('pp.impute needs scikit-learn: pip install "petrophys[ml]"') from exc

    own = _well_table(well, [name for name in FEATURE_CURVES if name != target])
    # Inputs: logs present where the curve is missing and in every training source.
    features = [name for name, values in own.items() if np.any(np.isfinite(values[missing]))]
    tables = [(own, measured)] if np.any(~missing) else []
    for other in others:
        mnemonic = _find_curve(other, target)
        if mnemonic is None:
            continue
        table = _well_table(other, features)
        features = [name for name in features if name in table]
        tables.append((table, _standard_values(other, mnemonic, target)[0]))
    if np.any(missing) and not features:
        raise ValueError(
            f"{well.name}: no input log is present both where {curve} is missing and in the "
            f"training data (candidates: {', '.join(FEATURE_CURVES)})"
        )

    x_parts, y_parts = [], []
    for table, values in tables:
        rows = np.isfinite(values)
        x_parts.append(np.column_stack([table[name][rows] for name in features]))
        y_parts.append(values[rows])
    x_train = np.vstack(x_parts) if x_parts else np.empty((0, len(features)))
    y_train = np.concatenate(y_parts) if y_parts else np.empty(0)
    keep = np.any(np.isfinite(x_train), axis=1)
    log_target = target in _LOG_CURVES
    if log_target:
        keep &= y_train > 0
    x_train, y_train = x_train[keep], y_train[keep]
    # A column present only in the target's gaps cannot teach the model a
    # relationship. Do not count it as evidence that a gap is predictable.
    learned_features = np.any(np.isfinite(x_train), axis=0)
    features = [name for name, learned in zip(features, learned_features) if learned]
    x_train = x_train[:, learned_features]
    if not features:
        raise ValueError(
            f"{well.name}: no input log is present both where {curve} is missing and in the "
            f"training data (candidates: {', '.join(FEATURE_CURVES)})"
        )
    if y_train.size < MIN_TRAINING_SAMPLES:
        raise ValueError(
            f"{curve}: {y_train.size} measured samples to learn from, at least "
            f"{MIN_TRAINING_SAMPLES} are needed; pass train_wells= with more data"
        )
    if y_train.size > _MAX_TRAINING_ROWS:
        rows = np.random.default_rng(0).choice(y_train.size, _MAX_TRAINING_ROWS, replace=False)
        x_train, y_train = x_train[rows], y_train[rows]

    if log_target:
        y_train = np.log10(y_train)

    imputed = np.full(n, np.nan)
    x_gap = np.column_stack([own[name][missing] for name in features]) if features else None
    predictable = np.zeros(n, dtype=bool)
    if x_gap is not None:
        has_input = np.any(np.isfinite(x_gap), axis=1)
        predictable[np.flatnonzero(missing)[has_input]] = True
        if np.any(has_input):
            model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                  early_stopping=False, random_state=0)
            model.fit(x_train, y_train)
            predicted = model.predict(x_gap[has_input])
            imputed[predictable] = 10.0 ** predicted if log_target else predicted
            low = np.nanpercentile(x_train, 1, axis=0)
            high = np.nanpercentile(x_train, 99, axis=0)
            span = high - low
            with np.errstate(invalid="ignore"):
                outside = np.any((x_gap[has_input] < low - 0.1 * span)
                                 | (x_gap[has_input] > high + 0.1 * span), axis=1)
            if np.any(outside):
                warnings.warn(
                    f"{well.name}: {int(outside.sum())} of {int(has_input.sum())} imputed {curve} "
                    "samples have input logs outside the range of the training data, where the "
                    "model cannot extrapolate",
                    stacklevel=2,
                )

    base = target_mnemonic or str(curve).strip().upper()
    values = np.where(missing, imputed / factor, recorded)
    _store_result(well, base, values, unit, output)
    return returned if returned is not None else well
