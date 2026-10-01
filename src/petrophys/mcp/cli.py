"""Command line for the Petrophys MCP server.

``petrophys-mcp``                      serve over stdio (what MCP clients launch)
``petrophys-mcp serve --http``          serve over HTTP for remote clients
``petrophys-mcp config <client>``       print ready-to-paste client configuration
``petrophys-mcp check``                 check the installation, folders and a server

``petrophys-mcp --root DIR`` without a sub-command still serves over stdio.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from petrophys.mcp.http import TOKEN_ENV_VAR, generate_token, is_loopback
from petrophys.mcp.session import ROOTS_ENV_VAR

CLIENTS = ("claude-code", "claude-desktop", "cursor", "vscode", "openai", "codex")
COMMANDS = ("serve", "config", "check")


def _add_root_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root", action="append", default=[], metavar="DIR",
        help="Folder the server may read wells from and write exports to (repeatable). "
             f"Defaults to ${ROOTS_ENV_VAR} or the current directory.",
    )


def _parser() -> argparse.ArgumentParser:
    from petrophys.version import __version__

    parser = argparse.ArgumentParser(
        prog="petrophys-mcp",
        description="Petrophys MCP server: let LLM clients (Claude, ChatGPT/OpenAI, Cursor, "
                    "VS Code, Codex) load wells, interpret them, plot and export results.",
        epilog="Examples:\n"
               "  petrophys-mcp --root ~/wells                    stdio, for local clients\n"
               "  petrophys-mcp serve --http --root ~/wells       http://127.0.0.1:8000/mcp\n"
               "  petrophys-mcp config claude-code --root ~/wells\n"
               "  petrophys-mcp check --root ~/wells",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"petrophys {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="{serve,config,check}")

    serve = sub.add_parser("serve", help="Run the server (stdio by default, or --http/--sse).",
                           description="Run the Petrophys MCP server.")
    _add_root_argument(serve)
    mode = serve.add_mutually_exclusive_group()
    mode.add_argument("--http", dest="transport", action="store_const", const="http",
                      help="Serve streamable HTTP at http://HOST:PORT/mcp (remote clients).")
    mode.add_argument("--sse", dest="transport", action="store_const", const="sse",
                      help="Serve the older SSE transport at http://HOST:PORT/sse.")
    serve.add_argument("--host", default="127.0.0.1",
                       help="Interface to listen on (default 127.0.0.1; use 0.0.0.0 in a container).")
    serve.add_argument("--port", type=int, default=8000, help="Port (default 8000).")
    serve.add_argument("--path", default="/mcp", help="MCP endpoint path for --http (default /mcp).")
    serve.add_argument(
        "--token", default=None,
        help=f"Bearer token clients must send. Defaults to ${TOKEN_ENV_VAR}; 'auto' generates one. "
             "Required unless the server listens on localhost only.",
    )
    serve.add_argument("--allowed-host", action="append", default=[], metavar="NAME",
                       help="Host header to accept, e.g. your tunnel or proxy domain (repeatable).")
    serve.add_argument("--stateless", action="store_true",
                       help="No sessions (for load balancers). Loaded wells do not persist between calls.")
    serve.add_argument("--insecure", action="store_true",
                       help="Allow listening on a non-local interface without a token. Not recommended.")
    serve.add_argument("--log-level", default="warning",
                       choices=("critical", "error", "warning", "info", "debug"))
    serve.set_defaults(transport="stdio")

    config = sub.add_parser("config", help="Print configuration for an MCP client.",
                            description="Print ready-to-paste configuration for an MCP client.")
    config.add_argument("client", choices=CLIENTS)
    _add_root_argument(config)
    config.add_argument("--url", help="Configure a remote HTTP server (e.g. https://host/mcp) "
                                      "instead of a local stdio server.")
    config.add_argument("--token", default=None,
                        help="Token to write into the configuration (a placeholder is used otherwise).")

    check = sub.add_parser("check", help="Check the installation, data folders and a server.",
                           description="Check that the server can start and use its folders.")
    _add_root_argument(check)
    check.add_argument("--url", help="Also connect to a running HTTP server at this URL.")
    check.add_argument("--token", default=None, help=f"Token for --url (default ${TOKEN_ENV_VAR}).")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point of ``petrophys-mcp``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or (argv[0] not in COMMANDS and argv[0] not in {"-h", "--help", "--version"}):
        argv = ["serve", *argv]  # `petrophys-mcp --root DIR` keeps serving over stdio
    args = _parser().parse_args(argv)
    if args.command == "config":
        return _config(args)
    if args.command == "check":
        return _check(args)
    return _serve(args)


# -- serve ---------------------------------------------------------------------
def _serve(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    if args.transport == "stdio":
        from petrophys.mcp.server import create_server

        create_server(args.root or None).run()
        return 0

    from petrophys.mcp.http import serve

    token = args.token if args.token is not None else os.environ.get(TOKEN_ENV_VAR)
    generated = token == "auto"
    if generated:
        token = generate_token()
    if not token and not is_loopback(args.host) and not args.insecure:
        print(f"petrophys-mcp: refusing to listen on {args.host} without a token. Pass --token "
              f"<secret> (or --token auto), or set {TOKEN_ENV_VAR}.", file=sys.stderr)
        return 2
    endpoint = args.path if args.transport == "http" else "/sse"
    shown_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{shown_host}:{args.port}{endpoint}"
    roots = args.root or [os.environ.get(ROOTS_ENV_VAR) or os.getcwd()]
    print(f"Petrophys MCP server ({args.transport}) at {url}", file=sys.stderr)
    print(f"  folders : {', '.join(str(Path(r).expanduser().resolve()) for r in roots)}", file=sys.stderr)
    if token:
        print("  auth    : bearer token required" + (f" -> {token}" if generated else ""), file=sys.stderr)
    else:
        print("  auth    : none (localhost only)", file=sys.stderr)
    print(f"  health  : http://{shown_host}:{args.port}/healthz", file=sys.stderr)
    print(f"  clients : petrophys-mcp config <client> --url {url}", file=sys.stderr)
    serve(args.root or None, transport=args.transport, host=args.host, port=args.port,
          path=args.path, token=token or None, allowed_hosts=args.allowed_host,
          stateless=args.stateless, log_level=args.log_level)
    return 0


# -- config --------------------------------------------------------------------
def _stdio_command(roots: List[str]) -> List[str]:
    folders = [str(Path(root).expanduser().resolve()) for root in roots] or [os.getcwd()]
    command = [sys.executable, "-m", "petrophys.mcp"]
    for folder in folders:
        command += ["--root", folder]
    return command


def config_text(client: str, roots: Sequence[str] = (), url: Optional[str] = None,
                token: Optional[str] = None) -> str:
    """Return configuration for ``client`` as text to paste.

    Local configurations launch this Python interpreter with absolute paths,
    so they work regardless of the client's PATH. Remote ones (``url``) send
    the bearer token in an Authorization header.
    """
    secret = token or "<TOKEN>"
    auth = {"Authorization": f"Bearer {secret}"}
    if url is None:
        command = _stdio_command(list(roots))
        if client == "claude-code":
            return "claude mcp add petrophys -- " + " ".join(shlex.quote(p) for p in command)
        if client == "codex":
            return ("# ~/.codex/config.toml\n[mcp_servers.petrophys]\n"
                    f"command = {json.dumps(command[0])}\nargs = {json.dumps(command[1:])}")
        if client == "vscode":
            return "// .vscode/mcp.json\n" + json.dumps(
                {"servers": {"petrophys": {"type": "stdio", "command": command[0],
                                           "args": command[1:]}}}, indent=2)
        if client in {"claude-desktop", "cursor"}:
            where = ("claude_desktop_config.json" if client == "claude-desktop"
                     else ".cursor/mcp.json (or ~/.cursor/mcp.json)")
            return f"// {where}\n" + json.dumps(
                {"mcpServers": {"petrophys": {"command": command[0], "args": command[1:]}}},
                indent=2)
        if client == "openai":
            return ("# OpenAI's hosted MCP tool needs a public HTTPS URL. Start the server with\n"
                    "#   petrophys-mcp serve --http --token auto\n"
                    "# expose it (reverse proxy or tunnel), then run:\n"
                    "#   petrophys-mcp config openai --url https://<your-host>/mcp --token <TOKEN>\n"
                    "# For a local model loop without a public URL, see examples/mcp_openai_agent.py.")
    else:
        if client == "claude-code":
            return (f"claude mcp add --transport http petrophys {shlex.quote(url)} "
                    f"--header {shlex.quote('Authorization: Bearer ' + secret)}")
        if client == "codex":
            return ("# ~/.codex/config.toml\n[mcp_servers.petrophys]\n"
                    f"url = {json.dumps(url)}\nbearer_token_env_var = \"{TOKEN_ENV_VAR}\"")
        if client == "vscode":
            return "// .vscode/mcp.json\n" + json.dumps(
                {"servers": {"petrophys": {"type": "http", "url": url, "headers": auth}}}, indent=2)
        if client in {"claude-desktop", "cursor"}:
            where = ("claude_desktop_config.json" if client == "claude-desktop"
                     else ".cursor/mcp.json (or ~/.cursor/mcp.json)")
            entry = ({"url": url, "headers": auth} if client == "cursor" else
                     {"command": "npx", "args": ["-y", "mcp-remote", url, "--header",
                                                 f"Authorization: Bearer {secret}"]})
            return f"// {where}\n" + json.dumps({"mcpServers": {"petrophys": entry}}, indent=2)
        if client == "openai":
            tool = {"type": "mcp", "server_label": "petrophys", "server_url": url,
                    "headers": auth, "require_approval": "never"}
            return ("# Responses API: pass this in tools=[...]\n" + json.dumps(tool, indent=2))
    raise ValueError(f"unknown client {client!r}; choose one of {', '.join(CLIENTS)}")


def _config(args: argparse.Namespace) -> int:
    print(config_text(args.client, args.root, args.url, args.token))
    return 0


# -- check ---------------------------------------------------------------------
def run_checks(roots: Sequence[str] = (), url: Optional[str] = None,
               token: Optional[str] = None) -> List[tuple]:
    """Run installation checks; return ``(ok, label, detail)`` tuples."""
    import asyncio
    import platform
    import tempfile

    from petrophys.version import __version__

    results: List[tuple] = []
    results.append((True, "petrophys", f"{__version__} on Python {platform.python_version()}"))
    try:
        from importlib.metadata import version

        results.append((True, "MCP SDK", version("mcp")))
    except Exception as exc:  # pragma: no cover - mcp is required to get here
        results.append((False, "MCP SDK", f"not importable ({exc}); pip install 'petrophys[mcp]'"))
        return results

    folders = [Path(r).expanduser() for r in roots] or [
        Path(os.environ.get(ROOTS_ENV_VAR) or os.getcwd())]
    for folder in folders:
        if not folder.is_dir():
            results.append((False, f"folder {folder}", "does not exist"))
            continue
        try:
            with tempfile.NamedTemporaryFile(dir=folder, prefix=".petrophys-check-"):
                pass
            writable = "readable and writable"
        except OSError:
            writable = "readable, NOT writable (exports will fail)"
        n_files = sum(1 for p in folder.iterdir() if p.suffix.lower() in {".las", ".dlis", ".csv", ".tsv"})
        results.append((True, f"folder {folder.resolve()}", f"{writable}; {n_files} well/table files"))

    async def probe(target) -> str:
        from mcp import Client

        async with Client(target) as client:
            tools = (await client.list_tools()).tools
            loaded = await client.call_tool("load_sample_well", {"name": "30_6-5"})
            if loaded.is_error:
                raise RuntimeError("load_sample_well failed")
            return f"{len(tools)} tools; sample well 30_6-5 loads"

    try:
        from petrophys.mcp.server import create_server

        results.append((True, "server (in-process)",
                        asyncio.run(probe(create_server([str(f) for f in folders])))))
    except Exception as exc:
        results.append((False, "server (in-process)", str(exc)))

    if url:
        secret = token or os.environ.get(TOKEN_ENV_VAR)
        try:
            import httpx2
            from mcp.client.streamable_http import streamable_http_client

            headers = {"Authorization": f"Bearer {secret}"} if secret else {}

            async def remote() -> str:
                async with httpx2.AsyncClient(headers=headers, timeout=60) as http_client:
                    reply = await http_client.post(url, json={})
                    if reply.status_code == 401:
                        raise PermissionError("the server rejected the token (HTTP 401); pass "
                                              f"--token or set {TOKEN_ENV_VAR}")
                    transport = streamable_http_client(url, http_client=http_client)
                    return await probe(transport)

            results.append((True, f"server at {url}", asyncio.run(remote())))
        except Exception as exc:
            cause = _root_cause(exc)
            results.append((False, f"server at {url}", f"{type(cause).__name__}: {cause}"))
    return results


def _root_cause(exc: BaseException) -> BaseException:
    """Unwrap exception groups raised by async task groups to the first real error."""
    while True:
        inner = getattr(exc, "exceptions", None)
        if not inner:
            return exc
        exc = inner[0]


def _check(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    results = run_checks(args.root, args.url, args.token)
    for ok, label, detail in results:
        print(f"{'ok  ' if ok else 'FAIL'}  {label}: {detail}")
    failed = [r for r in results if not r[0]]
    print("All checks passed." if not failed else f"{len(failed)} check(s) failed.")
    return 1 if failed else 0
