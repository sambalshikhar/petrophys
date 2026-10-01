"""Tests for the petrophys-mcp command line and HTTP deployment."""

import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

pytest.importorskip("mcp")

from petrophys.mcp import cli  # noqa: E402
from petrophys.mcp.http import TOKEN_ENV_VAR, build_app, is_loopback  # noqa: E402


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def http_server(tmp_path):
    """Run a token-protected HTTP server in a background thread."""
    import uvicorn

    port = _free_port()
    token = "test-token-123"
    app = build_app([str(tmp_path)], token=token)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}", token, tmp_path
    server.should_exit = True
    thread.join(timeout=10)


def test_health_needs_no_token_but_mcp_does(http_server):
    base, _, _ = http_server
    with urllib.request.urlopen(f"{base}/healthz", timeout=10) as reply:
        assert json.loads(reply.read())["status"] == "ok"
    request = urllib.request.Request(f"{base}/mcp", data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=10)
    assert error.value.code == 401
    assert "Bearer" in error.value.headers.get("WWW-Authenticate", "")


def test_check_connects_with_the_token_and_reports_a_wrong_one(http_server):
    base, token, root = http_server
    results = cli.run_checks([str(root)], url=f"{base}/mcp", token=token)
    assert all(ok for ok, _, _ in results), results
    remote = results[-1]
    assert remote[1].startswith("server at") and "tools" in remote[2]

    failed = cli.run_checks([str(root)], url=f"{base}/mcp", token="wrong")[-1]
    assert failed[0] is False and "rejected the token" in failed[2]


def test_public_interface_requires_a_token(monkeypatch, capsys):
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
    assert cli.main(["serve", "--http", "--host", "0.0.0.0"]) == 2
    assert "refusing to listen on 0.0.0.0 without a token" in capsys.readouterr().err


def test_bare_root_argument_still_serves_stdio(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(cli, "_serve", lambda args: seen.update(vars(args)) or 0)
    assert cli.main(["--root", str(tmp_path)]) == 0
    assert seen["transport"] == "stdio" and seen["root"] == [str(tmp_path)]


@pytest.mark.parametrize("client", cli.CLIENTS)
def test_config_for_every_client(client, tmp_path):
    local = cli.config_text(client, [str(tmp_path)])
    remote = cli.config_text(client, url="https://wells.example.com/mcp", token="s3cret")
    assert "petrophys" in local
    if client != "openai":
        assert sys.executable in local and str(tmp_path) in local
        assert "https://wells.example.com/mcp" in remote
    if client == "codex":  # Codex reads the token from an environment variable
        assert "bearer_token_env_var" in remote
    else:
        assert "Bearer s3cret" in remote
        assert "<TOKEN>" in cli.config_text(client, url="https://wells.example.com/mcp")


def test_check_in_process(tmp_path):
    results = cli.run_checks([str(tmp_path)])
    assert all(ok for ok, _, _ in results), results
    assert any(label == "server (in-process)" for _, label, _ in results)


def test_loopback_detection():
    assert is_loopback("127.0.0.1") and is_loopback("localhost") and is_loopback("::1")
    assert not is_loopback("0.0.0.0") and not is_loopback("10.1.2.3")
