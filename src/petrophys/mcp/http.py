"""Serve the Petrophys MCP server over HTTP (streamable HTTP or SSE).

Local clients normally start the server themselves over stdio. HTTP is for
everything else: a server shared by a team, a container, or a hosted LLM
(OpenAI's Responses API, ChatGPT connectors) that reaches the server by URL.

Requests are authenticated with a bearer token whenever one is configured, and
a token is required as soon as the server listens on anything but localhost.
``GET /healthz`` answers without authentication, for container health checks.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import secrets
from typing import Any, Optional, Sequence

TOKEN_ENV_VAR = "PETROPHYS_MCP_TOKEN"
HEALTH_PATH = "/healthz"

Scope = Any
Receive = Any
Send = Any
ASGIApp = Any  # any ASGI application (Starlette apps are typed differently)


def is_loopback(host: str) -> bool:
    """Return True when ``host`` only accepts connections from this machine."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def generate_token() -> str:
    """Return a random URL-safe token suitable for ``--token``."""
    return secrets.token_urlsafe(32)


class TokenAuthMiddleware:
    """Require ``Authorization: Bearer <token>`` on every HTTP request.

    ``GET /healthz`` is always answered, without a token, with the server
    version. WebSocket and lifespan events pass through unchanged.
    """

    def __init__(self, app: ASGIApp, token: Optional[str], version: str) -> None:
        self.app = app
        self.token = token
        self.version = version

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        if scope.get("path") == HEALTH_PATH:
            await _json_response(send, 200, {"status": "ok", "server": "petrophys",
                                             "version": self.version})
            return
        if self.token is not None and not self._authorized(scope):
            await _json_response(
                send, 401,
                {"error": "unauthorized",
                 "message": "Send the server token as 'Authorization: Bearer <token>'."},
                extra_headers=[(b"www-authenticate", b'Bearer realm="petrophys"')],
            )
            return
        await self.app(scope, receive, send)

    def _authorized(self, scope: Scope) -> bool:
        for name, value in scope.get("headers") or []:
            if name.lower() == b"authorization":
                scheme, _, supplied = value.decode("latin-1").partition(" ")
                if scheme.lower() == "bearer":
                    return hmac.compare_digest(supplied.strip(), self.token or "")
        return False


async def _json_response(send: Send, status: int, body: dict,
                         extra_headers: Sequence[tuple] = ()) -> None:
    payload = json.dumps(body).encode()
    headers = [(b"content-type", b"application/json"),
               (b"content-length", str(len(payload)).encode()), *extra_headers]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": payload})


def build_app(
    roots: Optional[Sequence[str]] = None,
    *,
    transport: str = "http",
    path: str = "/mcp",
    host: str = "127.0.0.1",
    token: Optional[str] = None,
    allowed_hosts: Sequence[str] = (),
    stateless: bool = False,
    calculation_profile: Optional[str] = None,
) -> Any:
    """Build the ASGI application for an HTTP deployment.

    Parameters
    ----------
    roots : sequence of str, optional
        Folders the server may read wells from and write exports to.
    transport : {"http", "sse"}
        Streamable HTTP (the current MCP standard) or the older SSE transport.
    path : str
        URL path of the MCP endpoint (streamable HTTP only).
    host : str
        Interface the server will listen on; decides the default Host-header
        checks against DNS rebinding.
    token : str, optional
        Bearer token required on every request (except ``/healthz``).
    allowed_hosts : sequence of str
        Host headers to accept, e.g. a tunnel or proxy domain. With a token
        and no list, Host headers are not restricted (the token protects the
        server); without a token, only localhost names are accepted.
    stateless : bool
        Serve each request without a session (for load-balanced deployments).
        Loaded wells and interpretations then do not persist between calls,
        so keep it off unless a proxy cannot keep sessions.
    calculation_profile : str, optional
        Calculation profile of interpretations whose settings do not set one
        (see :func:`petrophys.mcp.server.create_server`).

    Returns
    -------
    ASGI application
    """
    from mcp.server.transport_security import TransportSecuritySettings

    from petrophys.mcp.server import create_server
    from petrophys.version import __version__

    server = create_server(roots, calculation_profile=calculation_profile)
    if allowed_hosts:
        security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=list(allowed_hosts),
            allowed_origins=[f"https://{name}" for name in allowed_hosts]
            + [f"http://{name}" for name in allowed_hosts],
        )
    elif token is not None or not is_loopback(host):
        security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    else:
        security = None  # SDK default: localhost names only
    if transport == "sse":
        app = server.sse_app(transport_security=security, host=host)
    elif transport == "http":
        app = server.streamable_http_app(
            streamable_http_path=path, stateless_http=stateless,
            transport_security=security, host=host,
        )
    else:
        raise ValueError(f"transport must be 'http' or 'sse', not {transport!r}")
    return TokenAuthMiddleware(app, token, __version__)


def serve(
    roots: Optional[Sequence[str]] = None,
    *,
    transport: str = "http",
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = "/mcp",
    token: Optional[str] = None,
    allowed_hosts: Sequence[str] = (),
    stateless: bool = False,
    log_level: str = "warning",
    calculation_profile: Optional[str] = None,
) -> None:
    """Run the HTTP server until interrupted (see :func:`build_app`)."""
    import uvicorn

    app = build_app(roots, transport=transport, path=path, host=host, token=token,
                    allowed_hosts=allowed_hosts, stateless=stateless,
                    calculation_profile=calculation_profile)
    uvicorn.run(app, host=host, port=port, log_level=log_level)
