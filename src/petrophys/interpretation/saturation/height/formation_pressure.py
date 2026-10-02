"""Free-water level from two equilibrated formation-pressure gradients."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PressureGradient:
    """Linear pressure(depth), in the input pressure and depth units."""

    gradient: float
    reference_depth: float
    reference_pressure: float
    n: int
    rmse: float

    def pressure_at(self, depth):
        return self.reference_pressure + self.gradient * (
            np.asarray(depth, dtype=float) - self.reference_depth
        )


@dataclass(frozen=True)
class PressureFWLResult:
    """Gradient intersection on the caller's positive-down TVD datum."""

    fwl_depth: float
    fwl_pressure: float
    water: PressureGradient
    hydrocarbon: PressureGradient
    depth_unit: str
    pressure_unit: str
    depth_reference: str
    hydrocarbon_fluid: str = "oil"


def fit_pressure_fwl(
    depth, pressure, fluid, *, hydrocarbon="oil", depth_unit="m",
    pressure_unit="bar", depth_reference="TVDSS",
):
    """Fit water/hydrocarbon pressure lines and return their intersection.

    Supply QC-selected, equilibrated measurements from one connected reservoir,
    with positive-down true vertical depths on one datum and a common pressure
    reference. MD is not accepted. ``fluid`` must contain only ``water`` and
    the selected ``hydrocarbon`` (``oil`` or ``gas``); filter other fluids,
    dry tests and suspect/supercharged points explicitly before calling.

    Units are m/ft and bar/psi/kPa/MPa. Outputs retain the supplied units;
    ``gradient`` is pressure-unit/depth-unit. Fits are unweighted ordinary
    least squares, with at least two distinct depths per fluid. The result is
    an extrapolated equilibrium estimate, not a saturation-defined OWC or an
    uncertainty interval. Hydraulic connectivity and fluid labels are supplied
    interpretations; this function cannot establish them from numbers alone.
    """
    depth_factors = {"m": 1.0, "ft": 0.3048}
    pressure_factors = {"bar": 1.0, "psi": 0.06894757293168,
                        "kPa": 0.01, "MPa": 10.0}
    if depth_unit not in depth_factors or pressure_unit not in pressure_factors:
        raise ValueError("Use depth_unit m/ft and pressure_unit bar/psi/kPa/MPa")
    if depth_reference not in {"TVDSS", "TVD_RKB", "TVD_RT", "TVD_MSL"}:
        raise ValueError("Use positive-down TVDSS, TVD_MSL, TVD_RKB or TVD_RT; not MD")
    if hydrocarbon not in {"oil", "gas"}:
        raise ValueError("hydrocarbon must be oil or gas")
    z = np.asarray(depth, dtype=float)
    p = np.asarray(pressure, dtype=float)
    labels = np.asarray(fluid, dtype=str)
    if z.ndim != 1 or p.shape != z.shape or labels.shape != z.shape:
        raise ValueError("depth, pressure and fluid must be matching 1-D arrays")
    if not np.all(np.isfinite(z) & np.isfinite(p)):
        raise ValueError("Filter missing/nonfinite pressure tests before fitting")
    if not np.all(np.isin(labels, ["water", hydrocarbon])):
        raise ValueError("Filter other/unknown fluids before fitting")

    def fit(label):
        zz, pp = z[labels == label], p[labels == label]
        if zz.size < 2 or np.unique(zz).size < 2:
            raise ValueError(f"{label} needs at least two distinct TVD depths")
        zm, pm = float(zz.mean()), float(pp.mean())
        centered = zz - zm
        slope = float(np.dot(centered, pp - pm) / np.dot(centered, centered))
        residual = pp - (pm + slope * centered)
        return PressureGradient(slope, zm, pm, len(zz),
                                float(np.sqrt(np.mean(residual ** 2))))

    water, hc = fit("water"), fit(hydrocarbon)
    contrast = water.gradient - hc.gradient
    if min(water.gradient, hc.gradient) <= 0:
        raise ValueError("Both fluid pressure gradients must increase with TVD")
    if contrast <= 1e-6 * max(water.gradient, hc.gradient):
        raise ValueError("Water gradient must be steeper; parallel/reversed gradients give no stable FWL")
    # Centered calculation avoids subtracting large intercepts.
    reference = hc.reference_depth
    fwl = reference + (hc.pressure_at(reference) - water.pressure_at(reference)) / contrast
    return PressureFWLResult(float(fwl), float(water.pressure_at(fwl)),
                             water, hc, depth_unit, pressure_unit, depth_reference, hydrocarbon)


@dataclass
class FormationPressureData:
    """Stations with normalized columns and explicit TVD metadata.

    Original columns remain available for QC. Missing measurements and
    unselected fluid branches are retained but excluded from fitting.
    """

    points: pd.DataFrame
    depth_unit: str
    pressure_unit: str
    depth_reference: str

    def fit_mask(self, hydrocarbon="oil"):
        """Stations selected by USE_FIT, finite values and chosen fluid pair."""
        if hydrocarbon not in {"oil", "gas"}:
            raise ValueError("hydrocarbon must be oil or gas")
        p = self.points
        return (p.use_fit & np.isfinite(p.depth) & np.isfinite(p.pressure)
                & p.fluid.isin(["water", hydrocarbon]))

    def fit_fwl(self, hydrocarbon="oil"):
        """Fit the selected fluid pair; report contacts are never inputs."""
        if not self.points.fluid.isin(["oil", "gas", "water"]).any():
            raise ValueError("No fluid labels supplied; use infer_contacts() for unlabeled station pressures")
        p = self.points.loc[self.fit_mask(hydrocarbon)]
        return fit_pressure_fwl(p.depth, p.pressure, p.fluid,
                                hydrocarbon=hydrocarbon,
                                depth_unit=self.depth_unit,
                                pressure_unit=self.pressure_unit,
                                depth_reference=self.depth_reference)

    def infer_contacts(self, **kwargs):
        """Infer pressure branches and candidate contacts, ignoring fluid labels.

        USE_FIT is respected as a measurement-QC mask. Do not reuse a mask
        based on previously interpreted fluid intervals for an unlabeled check.
        See infer_pressure_contacts for assumptions and ambiguity handling.
        """
        from .pressure_contacts import infer_pressure_contacts
        return infer_pressure_contacts(
            self.points.depth, self.points.pressure, use=self.points.use_fit,
            depth_unit=self.depth_unit, pressure_unit=self.pressure_unit,
            depth_reference=self.depth_reference, **kwargs,
        )


def read_formation_pressure(
    source, *, depth_column=None, pressure_column=None, fluid_column=None,
    use_column=None, depth_unit=None, pressure_unit=None, depth_reference=None,
    fluid_map=None, csv_kwargs=None,
):
    """Read a CSV, LAS path, lasio.LASFile, or pandas DataFrame of stations.

    Unambiguous column names such as TVDSS_M/PRES_BAR carry metadata. LAS
    curve units and a DREF parameter are also read. Otherwise supply column
    names, units and a positive-down TVD datum explicitly. DEPT alone never
    implies TVD; this reader does not convert MD to TVD.

    FLUID is optional: without it, use the result's ``infer_contacts()`` method.
    If present it contains oil/gas/water (case insensitive), or numeric codes mapped
    by ``fluid_map={1: 'gas', 2: 'oil', 3: 'water'}``. LAS may instead provide
    ``FLDMAP`` as ``1=gas;2=oil;3=water``. Codes are not assumed universal.
    USE_FIT, if present, accepts true/false or 1/0. Without it, every finite
    station of the selected fluid pair is used. Unknown labels are rejected.
    Apply reservoir selection and measurement QC before fitting. The reader
    does not infer fluid identities; ``infer_contacts()`` is an explicit,
    separate interpretation step. Raw time-indexed pretests are not station
    pressure measurements: use read_mdt_pretest().pick_pressure() first, then
    formation_pressures_from_mdt() with explicit TVD depths and datum.

    ``csv_kwargs`` passes delimiter, null-value and other options to read_csv.
    Inputs are copied, never modified. Units are retained; pressures must
    share a common absolute/gauge basis. Read a single connected reservoir.
    """
    import lasio

    las = None
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    else:
        if isinstance(source, lasio.LASFile):
            las = source
        else:
            path = Path(source)
            if path.suffix.lower() == ".las":
                las = lasio.read(str(path))
            elif path.suffix.lower() == ".csv":
                frame = pd.read_csv(path, **(csv_kwargs or {}))
            else:
                raise ValueError("Supply a .csv/.las file, DataFrame or lasio.LASFile")
        if las is not None:
            if las.curves[0].mnemonic.upper() in {"TIME", "ETIM", "TIME_1900"} or las.curves[0].unit.upper() in {"S", "SEC", "MS", "SECOND"}:
                raise ValueError("Raw time-indexed MDT pretest: use read_mdt_pretest().pick_pressure(), then formation_pressures_from_mdt() with TVD depths before gradient analysis")
            frame = las.df().reset_index()
    columns = {str(c).upper(): c for c in frame.columns}
    if len(columns) != len(frame.columns):
        raise ValueError("Column names must be unique (case insensitive)")

    def column(explicit, candidates, purpose):
        if explicit is not None:
            if str(explicit).upper() not in columns:
                raise ValueError(f"Missing {purpose} column: {explicit}")
            return columns[str(explicit).upper()]
        matches = [columns[c] for c in candidates if c in columns]
        if len(matches) != 1:
            raise ValueError(f"Specify {purpose}_column; found candidates {matches}")
        return matches[0]

    depth_names = [f"{datum}_{unit}" for datum in
                   ("TVDSS", "TVD_MSL", "TVD_RKB", "TVD_RT") for unit in ("M", "FT")]
    dc = column(depth_column, depth_names + ["TVDSS", "TVD", "DEPT", "DEPTH"], "depth")
    pc = column(pressure_column, ["PRES", "PRESSURE", "PRES_BAR", "PRES_PSI",
                                 "PRES_KPA", "PRES_MPA"], "pressure")
    fc = column(fluid_column, [], "fluid") if fluid_column is not None else columns.get("FLUID")
    uc = column(use_column, [], "use") if use_column is not None else columns.get("USE_FIT")
    dname, pname = str(dc).upper(), str(pc).upper()
    if depth_unit is None:
        depth_unit = (las.curves[dc].unit if las is not None else
                      dname.rsplit("_", 1)[-1] if dname in depth_names else None)
    if pressure_unit is None:
        pressure_unit = (las.curves[pc].unit if las is not None else
                         pname[5:] if pname.startswith("PRES_") else None)
    if depth_reference is None:
        if dname in depth_names:
            depth_reference = dname.rsplit("_", 1)[0]
        elif dname == "TVDSS":
            depth_reference = "TVDSS"
        elif las is not None and "DREF" in las.params:
            depth_reference = str(las.params["DREF"].value)
    units = {"m": "m", "ft": "ft", "bar": "bar", "psi": "psi", "kpa": "kPa", "mpa": "MPa"}
    du = units.get(str(depth_unit).strip().lower())
    pu = units.get(str(pressure_unit).strip().lower())
    dr = str(depth_reference).strip().upper().replace(" ", "_")
    if du not in {"m", "ft"} or pu not in {"bar", "psi", "kPa", "MPa"}:
        raise ValueError("Specify depth_unit m/ft and pressure_unit bar/psi/kPa/MPa")
    if dr not in {"TVDSS", "TVD_MSL", "TVD_RKB", "TVD_RT"}:
        raise ValueError("Specify positive-down depth_reference TVDSS/TVD_MSL/TVD_RKB/TVD_RT; not MD")
    if dname == "MD" or dname.startswith("MD_"):
        raise ValueError("MD cannot be used as true vertical depth")
    if "WELL" in columns and frame[columns["WELL"]].dropna().nunique() > 1:
        raise ValueError("Select one well before reading formation pressures")
    if fluid_map is None and las is not None and "FLDMAP" in las.params:
        fluid_map = dict(part.strip().split("=", 1)
                         for part in str(las.params["FLDMAP"].value).split(";"))

    def fluid_key(value):
        try:
            return str(float(value))
        except (ValueError, TypeError):
            return str(value).strip().lower()

    if fc is None:
        labels = pd.Series("unknown", index=frame.index)
    elif fluid_map is not None:
        mapping = {fluid_key(k): str(v).strip().lower() for k, v in fluid_map.items()}
        labels = frame[fc].map(lambda v: mapping.get(fluid_key(v)))
    else:
        labels = frame[fc].astype(str).str.strip().str.lower()
    if fc is not None and not labels.isin(["oil", "gas", "water"]).all():
        raise ValueError("FLUID must identify oil/gas/water; supply fluid_map for coded curves")
    selected = pd.Series(True, index=frame.index)
    if uc is not None:
        flags = frame[uc].astype(str).str.strip().str.lower()
        flag_map = {"true": True, "false": False, "1": True, "0": False,
                    "1.0": True, "0.0": False}
        if not flags.isin(flag_map).all():
            raise ValueError("USE_FIT must contain only true/false or 1/0")
        selected = flags.map(flag_map)
    frame["depth"] = pd.to_numeric(frame[dc], errors="raise")
    frame["pressure"] = pd.to_numeric(frame[pc], errors="raise")
    frame["fluid"] = labels
    frame["use_fit"] = selected
    return FormationPressureData(frame, du, pu, dr)
