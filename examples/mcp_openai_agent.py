"""Talk to your wells through an OpenAI model and the Petrophys MCP server.

The model gets the server's tools as function tools and its instructions as
context; every tool call runs through a real MCP client. Nothing needs a public
URL: by default the server runs on this machine over stdio.

    pip install "petrophys[mcp]" openai
    export OPENAI_API_KEY=...
    python examples/mcp_openai_agent.py \
        "Interpret sample well 30/6-5 with Rw 0.07 ohm.m at 230 degF and summarize the Brent"

Options:
    --root DIR       folder with your LAS/DLIS/CSV files (default: current folder)
    --url URL        use a running HTTP server instead (petrophys-mcp serve --http);
                     the token comes from PETROPHYS_MCP_TOKEN
    --model NAME     OpenAI model (default: gpt-5.6-luna)

Run without a question for an interactive session (type 'exit' to quit).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys

from openai import OpenAI

ROLE = (
    "You are a petrophysics assistant. Use the Petrophys tools instead of guessing numbers. "
    "If a value you need cannot come from the data (formation water resistivity above all), "
    "ask for it. Answer concisely with key numbers and units, and state your assumptions."
)


def _function_tools(tools) -> list:
    """MCP tool definitions -> OpenAI Responses function tools."""
    return [{"type": "function", "name": tool.name, "description": tool.description or "",
             "parameters": tool.input_schema or {"type": "object", "properties": {}},
             "strict": False} for tool in tools]


def _tool_output(result):
    """MCP tool result -> Responses function_call_output (text, plus images)."""
    texts = [item.text for item in result.content or [] if item.type == "text"]
    images = [f"data:{item.mime_type};base64,{item.data}"
              for item in result.content or [] if item.type == "image"]
    text = "\n".join(texts) or json.dumps(result.structured_content or {})
    if not images:
        return text
    return [{"type": "input_text", "text": text}] + [
        {"type": "input_image", "image_url": url} for url in images]


async def ask(question: str, mcp_client, openai_client: OpenAI, model: str,
              instructions: str, tools: list, previous_id: str | None) -> tuple[str, str]:
    """Run one user turn to completion; return (answer, response id)."""
    pending = [{"role": "user", "content": question}]
    while True:
        response = await asyncio.to_thread(
            openai_client.responses.create, model=model, instructions=instructions,
            tools=tools, input=pending, previous_response_id=previous_id)
        previous_id = response.id
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            return response.output_text, previous_id
        pending = []
        for call in calls:
            arguments = json.loads(call.arguments or "{}")
            print(f"  -> {call.name}({json.dumps(arguments)[:120]})", file=sys.stderr)
            result = await mcp_client.call_tool(call.name, arguments)
            pending.append({"type": "function_call_output", "call_id": call.call_id,
                            "output": _tool_output(result)})


async def main_async(args: argparse.Namespace) -> None:
    from mcp import Client, StdioServerParameters

    async with contextlib.AsyncExitStack() as stack:
        if args.url:
            import httpx2
            from mcp.client.streamable_http import streamable_http_client

            token = os.environ.get("PETROPHYS_MCP_TOKEN", "")
            http_client = await stack.enter_async_context(httpx2.AsyncClient(
                headers={"Authorization": f"Bearer {token}"}, timeout=120))
            target = streamable_http_client(args.url, http_client=http_client)
        else:
            target = StdioServerParameters(
                command=sys.executable,
                args=["-m", "petrophys.mcp", "--root", os.path.abspath(args.root)])
        mcp_client = await stack.enter_async_context(Client(target))
        tools = _function_tools((await mcp_client.list_tools()).tools)
        server_instructions = mcp_client.instructions if isinstance(
            getattr(mcp_client, "instructions", None), str) else ""
        instructions = ROLE + ("\n\nPetrophys server instructions:\n" + server_instructions
                               if server_instructions else "")
        openai_client = OpenAI()
        previous_id = None
        questions = [args.question] if args.question else None
        while True:
            question = questions.pop(0) if questions else (
                None if questions is not None else input("\nyou> ").strip())
            if not question or question.lower() in {"exit", "quit"}:
                break
            answer, previous_id = await ask(question, mcp_client, openai_client, args.model,
                                            instructions, tools, previous_id)
            print(f"\n{answer}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question", nargs="?", help="Ask once and exit (omit for a chat).")
    parser.add_argument("--root", default=".", help="Folder with your well files.")
    parser.add_argument("--url", help="URL of a running Petrophys HTTP server.")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.6-luna"))
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
