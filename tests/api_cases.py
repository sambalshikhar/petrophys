"""Cases for every public entry point with each documented input form.

``run_all`` exercises reading, the Well model, interpretation settings, results,
plotting, imputation and context export on the bundled samples, including the
inputs that must be refused. Each case is ``(group, name, status, detail)``,
with status ``ok``, ``expected-error`` (the documented error was raised),
``FAIL`` (an error was raised that the case did not expect) or
``UNEXPECTED-OK`` (input that should be refused was accepted).
"""

import json
import shutil
import tempfile
import traceback
import warnings
from pathlib import Path

from _support import HAS_DLISIO, HAS_SKLEARN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import petrophys as pp


def run_all(only=None):
    """Run the cases (all, or the groups in ``only``) and return their outcomes."""
    TMP = Path(tempfile.mkdtemp(prefix="petrophys_api_"))
    RESULTS = []
    try:
        _cases(TMP, RESULTS, set(only or ()))
    finally:
        plt.close("all")
        shutil.rmtree(TMP, ignore_errors=True)
    return RESULTS


def _cases(TMP, RESULTS, ONLY):
    def case(group, name, fn, expect=None):
        if ONLY and group not in ONLY:
            return None
        if group == "impute" and not HAS_SKLEARN:
            return None
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                value = fn()
                if expect is not None:
                    RESULTS.append((group, name, "UNEXPECTED-OK", f"expected {getattr(expect, '__name__', expect)}"))
                else:
                    RESULTS.append((group, name, "ok", ""))
                plt.close("all")
                return value
            except Exception as error:  # noqa: BLE001
                plt.close("all")
                message = f"{type(error).__name__}: {error}"
                if expect is not None and isinstance(error, expect):
                    RESULTS.append((group, name, "expected-error", message[:300]))
                else:
                    lines = traceback.format_exc().strip().splitlines()
                    where = [line.strip() for line in lines if "petrophys" in line and "File" in line][-1:]
                    RESULTS.append((group, name, "FAIL", message[:400] + " @ " + " ".join(where)))
                return None

    S = pp.sample
    W305 = S("30_6-5")
    TOPS = S("formations_30_6-5")
    LITH = S("lithology_30_6-5")

    # ---------------------------------------------------------------- inputs on disk
    well305 = pp.read(W305)
    csv_path = TMP / "well.csv"
    well305.depth_slice(2800, 3000).to_csv(csv_path)
    tsv_path = TMP / "well.tsv"
    pd.read_csv(csv_path).to_csv(tsv_path, sep="\t", index=False)
    xlsx_tops = TMP / "tops.xlsx"
    pd.read_csv(TOPS).to_excel(xlsx_tops, index=False)
    xlsx_lith = TMP / "lith.xlsx"
    pd.read_csv(LITH).to_excel(xlsx_lith, index=False)
    odd_tops = TMP / "odd_tops.csv"
    pd.DataFrame({"Surface": ["Ness", "Etive"], "MD": [2841.0, 2865.0], "Base": [2865.0, 2958.0]}).to_csv(odd_tops, index=False)
    multi_tops = TMP / "multi_tops.csv"
    pd.DataFrame({"well": ["30/6-5", "30/6-5", "15/9-14"], "formation_name": ["Ness", "Etive", "X"],
                  "top_depth": [2841.0, 2865.0, 3000.0]}).to_csv(multi_tops, index=False)
    alias_csv = TMP / "aliases.csv"
    pd.DataFrame({"curve": ["GAMMA_RAY_1"], "standard": ["GR"]}).to_csv(alias_csv, index=False)
    codes_csv = S("lithology_codes_template")
    zones_df = pd.DataFrame({"zone": ["Ness", "Etive"], "top": [2841.0, 2865.0], "bottom": [2865.0, 2958.0],
                             "rw": [0.07, 0.08]})
    zones_xlsx = TMP / "zones.xlsx"
    zones_df.to_excel(zones_xlsx, index=False)
    zones_csv = TMP / "zones.csv"
    zones_df.to_csv(zones_csv, index=False)

    # ---------------------------------------------------------------- samples
    case("samples", "list_samples()", lambda: pp.list_samples())
    for kind in ["logs", "nmr", "pressure", "core", "geology", "templates", "costa", "geolink", "LOGS"]:
        case("samples", f"list_samples({kind!r})", lambda kind=kind: pp.list_samples(kind) or (_ for _ in ()).throw(AssertionError("empty")))
    case("samples", "list_samples('nope')", lambda: pp.list_samples("nope"), ValueError)
    case("samples", "sample('30_6-5.las')", lambda: pp.sample("30_6-5.las"))
    case("samples", "sample(' 30_6-5 ')", lambda: pp.sample(" 30_6-5 "))
    case("samples", "sample('30/6-5')", lambda: pp.sample("30/6-5"))
    case("samples", "sample('hw-3') lower case", lambda: pp.sample("hw-3"))
    case("samples", "sample('nope')", lambda: pp.sample("nope"), FileNotFoundError)
    case("samples", "sample('30_6_5') suggests", lambda: pp.sample("30_6_5"), FileNotFoundError)

    # ---------------------------------------------------------------- read
    case("read", "str path", lambda: pp.read(str(W305)))
    case("read", "Path", lambda: pp.read(Path(W305)))
    case("read", "formations csv", lambda: pp.read(W305, formations=TOPS))
    case("read", "formations xlsx", lambda: pp.read(W305, formations=xlsx_tops))
    case("read", "formations DataFrame", lambda: pp.read(W305, formations=pd.read_csv(TOPS)))
    case("read", "formations dict", lambda: pp.read(W305, formations={"Ness": 2841, "Etive": 2865}))
    case("read", "formations pairs", lambda: pp.read(W305, formations=[("Ness", 2841), ("Etive", 2865)]))
    case("read", "formations list of dicts", lambda: pp.read(W305, formations=[{"formation_name": "Ness", "top_depth": 2841}]))
    case("read", "formations rank=formation", lambda: pp.read(W305, formations=TOPS, formation_options={"rank": "formation"}))
    case("read", "formations odd columns", lambda: pp.read(W305, formations=odd_tops, formation_options={
        "columns": {"formation_name": "Surface", "top_depth": "MD", "bottom_depth": "Base"}}))
    case("read", "formations odd columns, no mapping", lambda: pp.read(W305, formations=odd_tops))
    case("read", "multiwell tops + well_name", lambda: pp.read(W305, formations=multi_tops, formation_options={"well_name": "30/6-5"}))
    case("read", "multiwell tops, no well_name (matched by well name)", lambda: pp.read(W305, formations=multi_tops))
    case("read", "lithology csv", lambda: pp.read(W305, lithology=LITH))
    case("read", "lithology xlsx", lambda: pp.read(W305, lithology=xlsx_lith))
    case("read", "lithology DataFrame", lambda: pp.read(W305, lithology=pd.read_csv(LITH)))
    case("read", "formations + lithology", lambda: pp.read(W305, formations=TOPS, lithology=LITH))
    case("read", "curve_aliases dict", lambda: pp.read(W305, curve_aliases={"GAMMA_RAY_1": "GR"}))
    case("read", "curve_aliases csv", lambda: pp.read(W305, curve_aliases=alias_csv))
    case("read", "survey vertical", lambda: pp.read(W305, survey="vertical"))
    case("read", "depth_reference MD datum KB", lambda: pp.read(W305, depth_reference="MD", datum="KB"))
    case("read", "csv depth_unit m", lambda: pp.read(csv_path, depth_unit="m"))
    case("read", "csv depth_unit 'M'", lambda: pp.read(csv_path, depth_unit="M"))
    case("read", "csv depth_unit 'meters'", lambda: pp.read(csv_path, depth_unit="meters"))
    case("read", "csv depth_unit 'feet'", lambda: pp.read(csv_path, depth_unit="feet"))
    case("read", "csv no depth_unit", lambda: pp.read(csv_path), ValueError)
    case("read", "csv depth_unit 'yards'", lambda: pp.read(csv_path, depth_unit="yards"), ValueError)
    case("read", "tsv", lambda: pp.read(tsv_path, depth_unit="m"))
    case("read", "csv + formations", lambda: pp.read(csv_path, depth_unit="m", formations=TOPS))
    case("read", "missing file", lambda: pp.read(TMP / "nope.las"), FileNotFoundError)
    case("read", "unsupported .txt", lambda: pp.read(TMP / "x.txt"), ValueError)
    case("read", "sample name not path", lambda: pp.read("30_6-5"), (ValueError, FileNotFoundError))
    case("read", "unknown reader option", lambda: pp.read(W305, foo=1), TypeError)
    for name in pp.list_samples("logs") + pp.list_samples("nmr") + pp.list_samples("pressure"):
        path = S(name)
        if path.suffix.lower() == ".las" or (path.suffix.lower() == ".dlis" and HAS_DLISIO):
            case("read", f"sample {name}", lambda path=path: pp.read(path), ValueError if name.startswith("319-C0009A_MDT_") else None)

    # ---------------------------------------------------------------- read_formations
    case("read_formations", "csv", lambda: pp.read_formations(TOPS))
    case("read_formations", "xlsx", lambda: pp.read_formations(xlsx_tops))
    case("read_formations", "dict", lambda: pp.read_formations({"A": 100, "B": 200}))
    case("read_formations", "pairs", lambda: pp.read_formations([("A", 100), ("B", 200)]))
    case("read_formations", "rank", lambda: pp.read_formations(TOPS, rank="formation"))
    case("read_formations", "well= converts", lambda: pp.read_formations(TOPS, well=well305))
    case("read_formations", "costa multiwell + well_name", lambda: pp.read_formations(S("formations_costa"), well_name="HW-3"))
    case("read_formations", "costa multiwell no well_name", lambda: pp.read_formations(S("formations_costa")), ValueError)
    case("read_formations", "kgs tops", lambda: pp.read_formations(S("1046123714_formation_tops")))
    case("read_formations", "template tops", lambda: pp.read_formations(S("formation_tops_template")))
    case("read_formations", "template intervals", lambda: pp.read_formations(S("formation_intervals_template")))
    case("read_formations", "empty dict", lambda: pp.read_formations({}), ValueError)
    case("read_formations", "top below bottom", lambda: pp.read_formations(
        pd.DataFrame({"formation_name": ["A"], "top_depth": [200], "bottom_depth": [100]})), ValueError)

    # ---------------------------------------------------------------- Well / config
    def build_well():
        w = pp.Well("demo")
        depth = np.arange(1000, 1100, 0.5)
        w.set_depth(depth, unit="m") if hasattr(w, "set_depth") else None
        w.add_curve("GR", np.linspace(20, 120, depth.size), unit="gAPI")
        return w
    case("Well", "Well() + add_curve", build_well)
    case("Well", "Well.from_dataframe?", lambda: pp.Well.from_dataframe(pd.read_csv(csv_path), depth_column="DEPTH", depth_unit="m")
         if hasattr(pp.Well, "from_dataframe") else (_ for _ in ()).throw(AttributeError("no Well.from_dataframe")))
    case("Well", "depth_slice reversed", lambda: well305.depth_slice(3000, 2800))
    case("Well", "depth_slice outside", lambda: well305.depth_slice(9000, 9100))
    case("Well", "well['NOPE']", lambda: well305["NOPE"], KeyError)
    case("Well", "to_las/to_csv", lambda: (well305.to_las(TMP / "w.las"), well305.to_csv(TMP / "w2.csv")))
    case("Well", "describe", lambda: well305.describe())
    case("config", "DynamicWorkflowConfig()", lambda: pp.DynamicWorkflowConfig())
    case("config", "kwargs", lambda: pp.DynamicWorkflowConfig(rw=0.07, sw_model="indonesia"))
    case("config", "bad kwarg", lambda: pp.DynamicWorkflowConfig(rww=0.07), TypeError)

    # ---------------------------------------------------------------- interpret
    def run(*args, **kwargs):
        result = pp.interpret(*args, **kwargs)
        assert result.status in {"interpreted", "no_interpretation", "failed"}, result.status
        if result.status == "failed":
            raise RuntimeError(f"status failed: {result.errors}")
        return result

    r_base = case("interpret", "Well", lambda: run(well305, rw=0.07, rw_meas_temp="230 degF"))
    case("interpret", "str path", lambda: run(str(W305)))
    case("interpret", "Path", lambda: run(Path(W305)))
    case("interpret", "csv path + depth_unit", lambda: run(csv_path, depth_unit="m"))
    case("interpret", "csv path without depth_unit", lambda: run(csv_path), ValueError)
    case("interpret", "depth_unit with Well", lambda: run(well305, depth_unit="m"), ValueError)
    case("interpret", "config object", lambda: run(well305, pp.DynamicWorkflowConfig(rw=0.07)))
    case("interpret", "config + override", lambda: run(well305, pp.DynamicWorkflowConfig(rw=0.07), sw_model="simandoux"))
    case("interpret", "config as keyword", lambda: run(well305, config=pp.DynamicWorkflowConfig(rw=0.07)))
    case("interpret", "dict as config", lambda: run(well305, {"rw": 0.07}))
    case("interpret", "dict config + same keyword", lambda: run(well305, {"rw": 0.07}, rw=0.08), TypeError)
    case("interpret", "config wrong type", lambda: run(well305, 0.07), TypeError)
    case("interpret", "typo option", lambda: run(well305, rww=0.07), TypeError)
    case("interpret", "calculation_profile='costa'", lambda: run(well305, calculation_profile="costa"), ValueError)
    case("interpret", "calculation_profile='geolink'", lambda: run(well305, calculation_profile="geolink"))
    case("interpret", "formations csv", lambda: run(well305, formations=TOPS))
    case("interpret", "formations dict", lambda: run(well305, formations={"Ness": 2841, "Etive": 2865}))
    case("interpret", "lithology intervals csv", lambda: run(well305, lithology=LITH))
    case("interpret", "lithology intervals xlsx", lambda: run(well305, lithology=xlsx_lith))
    case("interpret", "lithology DataFrame", lambda: run(well305, lithology=pd.read_csv(LITH)))
    case("interpret", "lithology code curve name", lambda: run(well305, lithology="LITHOLOGY_GEOLINK"))
    case("interpret", "lithology code curve lower case", lambda: run(well305, lithology="lithology_geolink"))
    for domain in ["clastic", "carbonate", "evaporite", "auto", "Clastic"]:
        case("interpret", f"lithology domain {domain}", lambda domain=domain: run(well305, lithology=domain))
    case("interpret", "lithology nonsense word", lambda: run(well305, lithology="granite"), ValueError)
    case("interpret", "lithology missing file", lambda: run(well305, lithology=str(TMP / "nope.csv")), (FileNotFoundError, ValueError))
    case("interpret", "lithology + lithology_domain", lambda: run(well305, lithology="clastic", lithology_domain="carbonate"), TypeError)
    case("interpret", "output las", lambda: run(well305, output=TMP / "out.las"))
    case("interpret", "output to missing folder", lambda: run(well305, output=TMP / "no" / "out.las"))
    case("interpret", "curve_aliases", lambda: run(well305, curve_aliases={"GAMMA_RAY_1": "GR"}))

    def check_inplace():
        w = pp.read(W305)
        n = w.n_curves
        run(w)
        assert w.n_curves == n, "copy changed caller's well"
        run(w, inplace=True)
        assert w.n_curves > n, "inplace did not add curves"
    case("interpret", "inplace False/True", check_inplace)
    for model in ["archie", "simandoux", "indonesia", "waxman_smits", "dual_water", "nigeria", "modified_simandoux", "auto", "bogus"]:
        case("interpret", f"sw_model={model}", lambda model=model: run(well305, rw=0.07, sw_model=model),
             ValueError if model in {"bogus", "nigeria"} else RuntimeError if model in {"waxman_smits", "dual_water"} else None)
    for method in ["timur", "coates", "coates_denoo", "wyllie_rose", "bogus"]:
        case("interpret", f"perm_method={method}", lambda method=method: run(well305, perm_method=method),
             ValueError if method == "bogus" else None)
    for matrix in ["sandstone", "limestone", "dolomite", "auto", "bogus"]:
        case("interpret", f"matrix_type={matrix}", lambda matrix=matrix: run(well305, matrix_type=matrix),
             ValueError if matrix == "bogus" else None)
    case("interpret", "salinity_ppm", lambda: run(well305, salinity_ppm=30000))
    case("interpret", "rw negative", lambda: run(well305, rw=-1), ValueError)
    case("interpret", "rw as string", lambda: run(well305, rw="0.07"), (TypeError, ValueError))
    case("interpret", "a m n", lambda: run(well305, a=1, m=2, n=2))
    case("interpret", "cutoffs", lambda: run(well305, vcl_cutoff=0.4, sw_cutoff=0.6, phie_cutoff=0.08))
    case("interpret", "cutoff in percent", lambda: run(well305, vcl_cutoff=40))
    case("interpret", "cutoff > 100", lambda: run(well305, vcl_cutoff=140), ValueError)
    case("interpret", "gr_min/gr_max", lambda: run(well305, gr_min=30, gr_max=110))
    case("interpret", "gr_min > gr_max", lambda: run(well305, gr_min=110, gr_max=30), ValueError)
    case("interpret", "bad_hole", lambda: run(well305, bad_hole={"cali": 2.0, "drho": 0.2}))
    case("interpret", "bad_hole bad key", lambda: run(well305, bad_hole={"caliper": 2.0}), ValueError)
    case("interpret", "bad_hole not mapping", lambda: run(well305, bad_hole=1.5), (TypeError, ValueError))
    case("interpret", "perm_coefficients", lambda: run(well305, perm_coefficients={"a": 8581, "b": 4.4, "c": 2}))
    case("interpret", "perm_coefficients bad key", lambda: run(well305, perm_coefficients={"z": 1}), ValueError)
    case("interpret", "mineral_endpoints", lambda: run(well305, mineral_endpoints={"quartz": {"rhob": 2.65}, "water": {"rhob": 1.1}}))
    case("interpret", "mineral_endpoints bad mineral", lambda: run(well305, mineral_endpoints={"granite": {"rhob": 2.7}}), ValueError)
    case("interpret", "mineral_endpoints bad property", lambda: run(well305, mineral_endpoints={"quartz": {"density": 2.65}}), ValueError)
    case("interpret", "settings under geolink", lambda: run(well305, calculation_profile="geolink", bad_hole={"cali": 2}), ValueError)
    case("interpret", "skip_steps", lambda: run(well305, skip_steps=["permeability"]))
    case("interpret", "skip_steps bogus", lambda: run(well305, skip_steps=["bogus"]), ValueError)
    case("interpret", "zones DataFrame", lambda: run(well305, zones=zones_df))
    case("interpret", "zones xlsx", lambda: run(well305, zones=zones_xlsx))
    case("interpret", "zones csv", lambda: run(well305, zones=zones_csv))
    case("interpret", "zones list of dicts", lambda: run(well305, zones=zones_df.to_dict("records")))
    case("interpret", "zones name-only", lambda: run(well305, zones=pd.DataFrame({"zone": ["Ness", "Etive"]}), formations=TOPS), ValueError)
    case("interpret", "zones with bad_hole column", lambda: run(well305, zones=zones_df.assign(m=[2.0, 1.9])))
    case("interpret", "zones nested setting", lambda: run(well305, zones=[{"zone": "Ness", "top": 2841, "bottom": 2865,
                                                                         "perm_coefficients": {"a": 9000}}]))
    case("interpret", "zones overlapping", lambda: run(well305, zones=pd.DataFrame(
        {"zone": ["A", "B"], "top": [2841, 2850], "bottom": [2865, 2900]})), ValueError)
    case("interpret", "zones unknown column", lambda: run(well305, zones=zones_df.assign(rww=1)), ValueError)
    case("interpret", "zones outside well", lambda: run(well305, zones=pd.DataFrame({"zone": ["X"], "top": [9000], "bottom": [9100]})))
    case("interpret", "context_profile costa", lambda: run(S("HW-3"), context_profile="costa"))
    case("interpret", "context_profile with Well", lambda: run(pp.read(S("HW-3")), context_profile="costa"), TypeError)
    case("interpret", "context_profile + option", lambda: run(S("HW-3"), context_profile="costa", rw=0.1), TypeError)
    case("interpret", "context_profile bogus", lambda: run(S("HW-3"), context_profile="bogus"), ValueError)
    case("interpret", "not a well", lambda: run(42), TypeError)
    case("interpret", "DataFrame as well", lambda: run(pd.read_csv(csv_path)), TypeError)

    r_zones = None
    if r_base is not None:
        case("result", "summary()", lambda: r_base.summary())
        case("result", "summary(top,bottom)", lambda: r_base.summary(top=2841, bottom=2958))
        case("result", "summary(reversed)", lambda: r_base.summary(top=2958, bottom=2841))
        case("result", "summary(outside)", lambda: r_base.summary(top=9000, bottom=9100))
        case("result", "zone_summary() without zones", lambda: r_base.zone_summary(), ValueError)
        case("result", "print_trace", lambda: r_base.print_trace())
        case("result", "to_dataframe", lambda: r_base.to_dataframe())
        case("result", "to_trace_dict json-able", lambda: json.dumps(r_base.to_trace_dict()))
        case("result", "to_las", lambda: r_base.to_las(TMP / "r.las"))
        case("result", "to_json", lambda: r_base.to_json(TMP / "r.json"))
        case("result", "plot()", lambda: r_base.plot())
        case("result", "plot(top,bottom,template)", lambda: r_base.plot(top=2841, bottom=2958, template="petrophysics"))
        case("result", "raise_for_errors", lambda: r_base.raise_for_errors())
        case("result", "repr", lambda: repr(r_base))
        case("result", "las round trip interpret", lambda: run(TMP / "r.las"))
    r_zones = case("interpret", "zones for summary", lambda: run(well305, zones=zones_df))
    if r_zones is not None:
        case("result", "zone_summary()", lambda: r_zones.zone_summary())
        case("result", "zone_intervals", lambda: r_zones.zone_intervals)

    # ---------------------------------------------------------------- plot
    case("plot", "Well", lambda: pp.plot(well305))
    case("plot", "result", lambda: pp.plot(r_base))
    case("plot", "str path", lambda: pp.plot(str(W305), top=2800, bottom=3000))
    case("plot", "csv path", lambda: pp.plot(csv_path), ValueError)
    case("plot", "top only", lambda: pp.plot(well305, top=2800))
    case("plot", "bottom only", lambda: pp.plot(well305, bottom=500))
    case("plot", "reversed window", lambda: pp.plot(well305, top=3000, bottom=2800))
    case("plot", "window outside well", lambda: pp.plot(well305, top=9000, bottom=9100))
    case("plot", "top == bottom", lambda: pp.plot(well305, top=2800, bottom=2800), ValueError)
    for template in ["auto", "raw", "petrophysics", "minerals", "comprehensive", "geolink_full", "PETROPHYSICS"]:
        case("plot", f"template {template} (well)", lambda template=template: pp.plot(well305, top=2800, bottom=3000, template=template))
        case("plot", f"template {template} (result)", lambda template=template: pp.plot(r_base, top=2800, bottom=3000, template=template))
    case("plot", "template bogus", lambda: pp.plot(well305, template="bogus"), ValueError)
    case("plot", "tracks", lambda: pp.plot(well305, top=2800, bottom=3000, tracks=["RESISTIVITY", "GAMMA / HOLE"]))
    case("plot", "tracks bogus", lambda: pp.plot(well305, tracks=["BOGUS"]), ValueError)
    case("plot", "tracks as string", lambda: pp.plot(well305, tracks="RESISTIVITY"))
    case("plot", "curves", lambda: pp.plot(well305, curves=["GR", "RHOB"]))
    case("plot", "curves lower case", lambda: pp.plot(well305, curves=["gr", "rhob"]))
    case("plot", "curves bogus", lambda: pp.plot(well305, curves=["BOGUS"]), ValueError)
    case("plot", "curves + tracks", lambda: pp.plot(well305, curves=["GR", "RDEP"], tracks=["RESISTIVITY"]), ValueError)
    case("plot", "formations csv", lambda: pp.plot(well305, top=2800, bottom=3000, formations=TOPS))
    case("plot", "formations dict", lambda: pp.plot(well305, top=2800, bottom=3000, formations={"Ness": 2841}))
    case("plot", "lithology csv", lambda: pp.plot(well305, lithology=LITH))
    case("plot", "lithology_codes dict", lambda: pp.plot(well305, lithology_codes={65000: "Shale"}))
    case("plot", "lithology_codes csv", lambda: pp.plot(well305, lithology_codes=codes_csv))
    case("plot", "formation_name", lambda: pp.plot(well305, formation_name="Brent Group"))
    case("plot", "figsize", lambda: pp.plot(well305, figsize=(8, 6)))
    case("plot", "strict raw (missing curves)", lambda: pp.plot(pp.read(S("HW-3")), template="petrophysics", strict=True), ValueError)
    case("plot", "curve_styles", lambda: pp.plot(well305, curve_styles={"GR": {"color": "black", "right": 200}}))
    case("plot", "curve_styles bad key", lambda: pp.plot(well305, curve_styles={"GR": {"colour": "black"}}), ValueError)
    case("plot", "curve_styles unknown curve", lambda: pp.plot(well305, curve_styles={"BOGUS": {"color": "k"}}))
    case("plot", "curve_aliases", lambda: pp.plot(well305, curve_aliases={"GAMMA_RAY_1": "GR"}))
    case("plot", "save png", lambda: pp.plot(well305, top=2800, bottom=3000).save(TMP / "p.png"))
    case("plot", "save pdf", lambda: pp.plot(well305, top=2800, bottom=3000).save(TMP / "p.pdf"))
    case("plot", "not a well", lambda: pp.plot(42), TypeError)
    for name in pp.list_samples("logs") + ["daydream1_cmr"]:
        case("plot", f"sample {name}", lambda name=name: pp.plot(S(name)))

    # ---------------------------------------------------------------- impute
    case("impute", "Well RHOB", lambda: pp.impute(pp.read(W305), "RHOB"))
    case("impute", "lower case curve", lambda: pp.impute(pp.read(W305), "rhob"))
    case("impute", "path", lambda: pp.impute(W305, "DTC"))
    case("impute", "result", lambda: pp.impute(run(W305), "RHOB"))
    case("impute", "standard name for vendor curve", lambda: pp.impute(pp.read(S("cheal-1")), "RHOB"))
    case("impute", "absent curve no train", lambda: pp.impute(pp.read(S("16_7-1")), "NPHI"), ValueError)
    case("impute", "absent curve train_wells paths", lambda: pp.impute(pp.read(S("16_7-1")), "NPHI", train_wells=[S("15_9-14"), S("30_6-5")]))
    case("impute", "train_wells Well objects", lambda: pp.impute(pp.read(S("16_7-1")), "NPHI", train_wells=[pp.read(S("15_9-14"))]))
    case("impute", "train_wells single path not list", lambda: pp.impute(pp.read(S("16_7-1")), "NPHI", train_wells=S("15_9-14")))
    case("impute", "output", lambda: pp.impute(pp.read(W305), "RHOB", output=TMP / "imp.las"))
    case("impute", "bogus curve", lambda: pp.impute(pp.read(W305), "BOGUS"), ValueError)
    case("impute", "csv path", lambda: pp.impute(csv_path, "RHOB"), ValueError)

    # ---------------------------------------------------------------- export_context
    case("context", "result -> dict", lambda: pp.export_context(r_base))
    case("context", "Well -> dict", lambda: pp.export_context(well305))
    case("context", "path -> dict", lambda: pp.export_context(W305))
    case("context", "result -> json file", lambda: pp.export_context(r_base, TMP / "ctx.json"))
    case("context", "json file exists, no overwrite", lambda: pp.export_context(r_base, TMP / "ctx.json"), FileExistsError)
    case("context", "overwrite", lambda: pp.export_context(r_base, TMP / "ctx.json", overwrite=True))
    case("context", "category", lambda: pp.export_context(r_base, category="saturation"))
    case("context", "category bogus", lambda: pp.export_context(r_base, category="bogus"), ValueError)
    case("context", "zoned result -> dict", lambda: pp.export_context(r_zones))
    case("context", "layout bogus", lambda: pp.export_context(r_base, layout="bogus"), ValueError)
    case("context", "layout package, no output", lambda: pp.export_context(r_base, layout="package"), (TypeError, ValueError))
    case("context", "old name tops_file", lambda: pp.export_context(r_base, tops_file=TOPS), TypeError)

    pkg_in = TMP / "pkg_in"
    pkg_in.mkdir()
    for name in ["HW-3", "HW-4"]:
        (pkg_in / f"{name}.las").write_bytes(S(name).read_bytes())
    costa_tops = S("formations_costa")
    case("context", "package defaults", lambda: pp.export_context(pkg_in, TMP / "pkg1", layout="package"))
    case("context", "package output exists", lambda: pp.export_context(pkg_in, TMP / "pkg1", layout="package"), FileExistsError)
    case("context", "package + formations", lambda: pp.export_context(pkg_in, TMP / "pkg2", layout="package", formations=costa_tops))
    case("context", "package formations no split", lambda: pp.export_context(pkg_in, TMP / "pkg3", layout="package",
                                                                          formations=costa_tops, split_by_lithology=False))
    costa_zones = pd.DataFrame({"well": ["HW-3", "HW-3", "HW-4"], "zone": ["Upper", "Lower", "All"],
                                "top": [None, None, None], "bottom": [None, None, None]})
    w3 = pp.read(S("HW-3"))
    w4 = pp.read(S("HW-4"))
    mid3 = float(np.nanmedian(w3.depth))
    costa_zones = pd.DataFrame({"well": ["HW-3", "HW-3", "HW-4"], "zone": ["Upper", "Lower", "All"],
                                "top": [w3.top, mid3, w4.top], "bottom": [mid3, w3.bottom, w4.bottom]})
    case("context", "package zones", lambda: pp.export_context(pkg_in, TMP / "pkg4", layout="package", zones=costa_zones))
    case("context", "package zones no split", lambda: pp.export_context(pkg_in, TMP / "pkg5", layout="package", zones=costa_zones,
                                                                      split_by_lithology=False))
    case("context", "package zones xlsx", lambda: (costa_zones.to_excel(TMP / "cz.xlsx", index=False),
                                                   pp.export_context(pkg_in, TMP / "pkg6", layout="package", zones=TMP / "cz.xlsx")))
    case("context", "package workflow_config", lambda: pp.export_context(pkg_in, TMP / "pkg7", layout="package",
                                                                       workflow_config=pp.DynamicWorkflowConfig(rw=0.03)))
    case("context", "package settings as kwargs", lambda: pp.export_context(pkg_in, TMP / "pkg8", layout="package", rw=0.03))
    case("context", "package typo option", lambda: pp.export_context(pkg_in, TMP / "pkgT", layout="package", rww=0.03), TypeError)
    case("context", "package settings + workflow_config", lambda: pp.export_context(pkg_in, TMP / "pkgS", layout="package",
         workflow_config=pp.DynamicWorkflowConfig(rw=0.03), sw_model="indonesia"))
    case("context", "well layout with a setting", lambda: pp.export_context(r_base, rw=0.03), TypeError)
    case("context", "profile with a setting", lambda: pp.export_context(pkg_in, TMP / "pkgP", layout="package", context_profile="costa", rw=0.1), ValueError)
    def zone_warning():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pp.export_context(pkg_in, TMP / "pkgW", layout="package",
                              zones=pd.DataFrame({"well": ["HW-3"], "zone": ["X"], "top": [100.0], "bottom": [200.0]}))
        assert any("lies outside the logged depths" in str(item.message) for item in caught), [str(i.message) for i in caught]
    case("context", "zone outside well warns", zone_warning)
    case("context", "package empty folder", lambda: (TMP / "empty").mkdir() or pp.export_context(TMP / "empty", TMP / "pkg9", layout="package"),
         (ValueError, FileNotFoundError))
    case("context", "package from a file not folder", lambda: pp.export_context(S("HW-3"), TMP / "pkg10", layout="package"),
         (ValueError, NotADirectoryError))
    case("context", "context_profile costa", lambda: pp.export_context(pkg_in, TMP / "pkg11", layout="package", context_profile="costa"))

    # ---------------------------------------------------------------- whole sample sweep
    for name in pp.list_samples("logs"):
        case("sweep", f"interpret {name}", lambda name=name: run(S(name)))
        case("sweep", f"interpret+plot {name}", lambda name=name: pp.plot(run(S(name)), template="petrophysics"))
        case("sweep", f"context {name}", lambda name=name: pp.export_context(run(S(name))))
    for name in ["daydream1_cmr", "35_11_10_mdt"]:
        case("sweep", f"interpret non-log {name}", lambda name=name: run(S(name)))

