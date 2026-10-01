"""Model Context Protocol (MCP) server for Petrophys.

Lets any MCP-capable LLM client (Claude Desktop, Claude Code, Cursor, VS Code,
Codex, OpenAI/ChatGPT) load wells, run interpretations, inspect traces,
summarize zones, plot logs and export results. Install with
``pip install "petrophys[mcp]"`` and run ``petrophys-mcp`` (stdio) or
``petrophys-mcp serve --http`` (remote clients); ``petrophys-mcp config`` and
``petrophys-mcp check`` help set it up. See the MCP guide in the documentation.
"""

from petrophys.mcp.session import ROOTS_ENV_VAR, Session, SessionError

__all__ = ["ROOTS_ENV_VAR", "Session", "SessionError", "create_server", "main"]


def create_server(*args, **kwargs):
    """Build the MCP server; see :func:`petrophys.mcp.server.create_server`."""
    from petrophys.mcp.server import create_server as _create_server

    return _create_server(*args, **kwargs)


def main(argv=None):
    """Run the ``petrophys-mcp`` command line; see :mod:`petrophys.mcp.cli`."""
    from petrophys.mcp.cli import main as _main

    return _main(argv)
