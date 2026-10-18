"""Regression tests for MCP server, CLI and packaging fixes."""

import asyncio
import copy
import subprocess
import sys
import warnings

import numpy as np
import pytest

import petrophys as pp


@pytest.fixture(scope="module")
def well_30_6_5():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pp.read(pp.sample("30_6-5"))


def test_data_quality_bad_hole_converts_mm_caliper(well_30_6_5):
    from petrophys.mcp import reports

    metric = copy.deepcopy(well_30_6_5)
    caliper = metric.get("CALI")
    caliper.data = caliper.data * 25.4
    caliper.unit = "mm"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inch = reports.data_quality_report(well_30_6_5, curves=["CALI"])["bad_hole"]
        mm = reports.data_quality_report(metric, curves=["CALI"])["bad_hole"]
    assert mm["flagged_%"] < 90.0
    assert abs(mm["flagged_%"] - inch["flagged_%"]) < 1.0


def test_data_quality_bad_hole_uses_drho_without_caliper(well_30_6_5):
    from petrophys.mcp import reports

    drho_only = copy.deepcopy(well_30_6_5)
    for name in ("CALI", "BS"):
        drho_only.remove_curve(name)
    drho_only.header.bit_size = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bad_hole = reports.data_quality_report(drho_only, curves=["DRHO"])["bad_hole"]
    assert bad_hole["flagged_%"] > 0.0


def test_zone_table_does_not_total_nested_zones():
    from petrophys.mcp import catalog, reports

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        well = pp.read(pp.sample("30_6-5"), **catalog.sample_geology("30_6-5"))
        result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230)
    nested = reports.zone_table(result.well, names=["BAAT", "Etive"])
    assert "total" not in nested
    assert "overlap" in nested["note"]
    side_by_side = reports.zone_table(result.well, names=["Etive", "Ness"])
    assert side_by_side["total"]["gross"] == pytest.approx(
        sum(row["Gross (m)"] for row in side_by_side["zones"])
    )


def test_redact_hides_repr_of_absolute_path(tmp_path):
    from petrophys.mcp.session import Session

    session = Session(roots=[tmp_path])
    message = str(PermissionError(13, "Permission denied", str(tmp_path / "adir")))
    redacted = session.redact(message)
    assert str(tmp_path) not in redacted
    assert str(tmp_path).replace("\\", "\\\\") not in redacted
    assert "adir" in redacted


def test_add_well_refuses_other_file_under_used_id(well_30_6_5):
    from petrophys.mcp.session import Session, SessionError

    session = Session()
    assert session.add_well(well_30_6_5, "sample:25_5-1", "25_5-1") == "25_5-1"
    # Reloading the same source replaces it.
    assert session.add_well(well_30_6_5, "sample:25_5-1", "25_5-1") == "25_5-1"
    with pytest.raises(SessionError, match="already used"):
        session.add_well(well_30_6_5, "w.las", "25_5-1")
    assert session.sources["25_5-1"] == "sample:25_5-1"


def _status(middleware, header):
    sent = []

    async def app(scope, receive, send):
        sent.append({"type": "app"})

    async def send(message):
        sent.append(message)

    middleware.app = app
    scope = {"type": "http", "path": "/mcp", "headers": [(b"authorization", header)]}
    asyncio.run(middleware(scope, None, send))
    return 200 if sent[0]["type"] == "app" else sent[0]["status"]


def test_token_auth_non_ascii_header_and_empty_token():
    from petrophys.mcp.http import TokenAuthMiddleware

    guarded = TokenAuthMiddleware(None, "sekret", "x")
    assert _status(guarded, "Bearer sékret".encode("latin-1")) == 401
    assert _status(guarded, b"Bearer sekret") == 200
    assert TokenAuthMiddleware(None, "", "x").token is None


def test_missing_mcp_extra_gives_install_hint():
    code = (
        "import sys; sys.modules['pydantic'] = None; sys.modules['mcp'] = None\n"
        "from petrophys.mcp import main\n"
        "sys.exit(main(['serve', '--root', '.']))\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode != 0
    assert "petrophys[mcp]" in done.stderr
    assert "Traceback" not in done.stderr


def test_version_info_parses_pre_release_strings():
    import re

    from petrophys import version

    assert version.__version_info__[:3] == tuple(int(x) for x in version.__version__.split(".")[:3])
    pattern = r"\d+(?:\.\d+)*"
    for text, expected in (("3.0.0rc1", (3, 0, 0)), ("3.0.0.post1", (3, 0, 0)),
                           ("3.1.0.dev0", (3, 1, 0))):
        assert tuple(int(x) for x in re.match(pattern, text).group(0).split(".")) == expected


def test_context_cli_names_the_unmatched_pattern_and_well(tmp_path):
    from petrophys.reporting.context import export_context

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "30_6-5.las").write_bytes(open(pp.sample("30_6-5"), "rb").read())
    with pytest.raises(ValueError, match=r"no input file matches '\*\.dlis'"):
        export_context(logs, tmp_path / "a", layout="category_formation_lithology",
                       pattern="*.dlis")
    with pytest.raises(ValueError, match="NOSUCHWELL"):
        export_context(logs, tmp_path / "b", layout="category_formation_lithology",
                       well_names=["NOSUCHWELL"])


def test_create_server_selects_headless_backend():
    pytest.importorskip("mcp")
    code = (
        "import matplotlib\n"
        "from petrophys.mcp import create_server\n"
        "create_server()\n"
        "print(matplotlib.get_backend().lower())\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.stdout.strip().endswith("agg"), done.stderr
