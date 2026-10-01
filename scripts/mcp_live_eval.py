"""Live evaluation of the Petrophys MCP server with an OpenAI model (maintainers).

The model sees the server's MCP tools as function tools (schemas and
descriptions straight from list_tools) and the server's instructions, exactly
as an MCP host passes them. Every tool call goes through a real MCP client.

Usage (needs OPENAI_API_KEY; costs API tokens):
    python scripts/mcp_live_eval.py --out mcp-eval/run1 [--only 02_interpret_brent ...]
        [--target inprocess|stdio|http://127.0.0.1:8000/mcp] [--model gpt-5.6-luna]

Scenarios live in scripts/mcp_eval_scenarios.json. Each run writes one JSON
transcript per scenario (tool calls, arguments, errors, output sizes, tokens,
final answer) and a summary.json; read the finals to judge correctness.
For an HTTP target the token comes from PETROPHYS_MCP_TOKEN.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

HOST_PROMPT = (
    "You are a petrophysics assistant working for a geoscientist. You can use the "
    "Petrophys tools to load and interpret well data. Use the tools rather than "
    "guessing numbers. When the request lacks information that you need and must "
    "not assume, ask for it in your final answer instead of inventing it. "
    "Answer concisely, with the key numbers and their units."
)


def _key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Set OPENAI_API_KEY to run the live evaluation.")
    return key


def _target(spec: str, root: str):
    from mcp import StdioServerParameters

    if spec == "inprocess":
        from petrophys.mcp import create_server

        return create_server(roots=[root])
    if spec == "stdio":
        return StdioServerParameters(
            command=sys.executable, args=["-m", "petrophys.mcp", "--root", root]
        )
    return spec  # URL for streamable HTTP


def _tool_specs(tools) -> list:
    specs = []
    for tool in tools:
        specs.append({
            "type": "function",
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema or {"type": "object", "properties": {}},
            "strict": False,
        })
    return specs


def _result_to_output(result) -> tuple[object, dict]:
    """Turn an MCP CallToolResult into a Responses function_call_output."""
    texts, images = [], []
    for item in result.content or []:
        kind = getattr(item, "type", "")
        if kind == "text":
            texts.append(item.text)
        elif kind == "image":
            images.append(f"data:{item.mime_type};base64,{item.data}")
    text = "\n".join(texts)
    if not text and result.structured_content is not None:
        text = json.dumps(result.structured_content)
    meta = {"is_error": bool(result.is_error), "chars": len(text), "images": len(images)}
    if images:
        parts = [{"type": "input_text", "text": text or "(image)"}]
        parts += [{"type": "input_image", "image_url": url} for url in images]
        return parts, meta
    return text, meta


async def run_scenario(scenario: dict, *, model: str, target_spec: str, root: str,
                       max_turns: int = 20) -> dict:
    from mcp import Client

    client_ai = OpenAI(api_key=_key())
    record = {"id": scenario["id"], "prompt": scenario["prompt"], "calls": [],
              "usage": {"input": 0, "cached": 0, "output": 0, "reasoning": 0}, "turns": 0}
    started = time.time()
    import contextlib

    stack = contextlib.AsyncExitStack()
    target = _target(target_spec, root)
    if isinstance(target, str):  # remote server: send the bearer token
        import httpx2
        from mcp.client.streamable_http import streamable_http_client

        token = os.environ.get("PETROPHYS_MCP_TOKEN", "")
        http_client = await stack.enter_async_context(
            httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"}, timeout=120))
        target = streamable_http_client(target, http_client=http_client)
    async with stack, Client(target) as mcp_client:
        listed = await mcp_client.list_tools()
        tools = _tool_specs(listed.tools)
        instructions = HOST_PROMPT
        server_instructions = getattr(mcp_client, "instructions", None)
        if callable(server_instructions):
            server_instructions = None
        if server_instructions:
            instructions += "\n\nInstructions from the Petrophys MCP server:\n" + server_instructions
        record["n_tools"] = len(tools)
        record["instructions_chars"] = len(instructions)

        previous_id = None
        next_input = [{"role": "user", "content": scenario["prompt"]}]
        final_text = ""
        for turn in range(max_turns):
            record["turns"] = turn + 1
            kwargs = dict(model=model, instructions=instructions, tools=tools, input=next_input)
            if previous_id:
                kwargs["previous_response_id"] = previous_id
            response = await asyncio.to_thread(client_ai.responses.create, **kwargs)
            previous_id = response.id
            usage = response.usage
            if usage is not None:
                record["usage"]["input"] += usage.input_tokens or 0
                input_details = getattr(usage, "input_tokens_details", None)
                record["usage"]["cached"] += getattr(input_details, "cached_tokens", 0) or 0
                record["usage"]["output"] += usage.output_tokens or 0
                details = getattr(usage, "output_tokens_details", None)
                record["usage"]["reasoning"] += getattr(details, "reasoning_tokens", 0) or 0
            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:
                final_text = response.output_text
                break
            next_input = []
            for call in calls:
                try:
                    arguments = json.loads(call.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                t0 = time.time()
                try:
                    result = await mcp_client.call_tool(call.name, arguments)
                    output, meta = _result_to_output(result)
                except Exception as exc:  # transport-level failure
                    output, meta = f"Tool call failed: {exc}", {"is_error": True, "chars": 0, "images": 0}
                meta.update({"tool": call.name, "arguments": arguments,
                             "seconds": round(time.time() - t0, 2),
                             "preview": (output if isinstance(output, str) else output[0]["text"])[:600]})
                record["calls"].append(meta)
                next_input.append({"type": "function_call_output", "call_id": call.call_id,
                                   "output": output})
        else:
            final_text = "(max turns reached)"
    record["final"] = final_text
    record["seconds"] = round(time.time() - started, 1)
    record["errors"] = sum(1 for call in record["calls"] if call["is_error"])
    return record


async def main_async(args) -> None:
    scenarios = json.loads(Path(args.scenarios).read_text())
    if args.only:
        scenarios = [s for s in scenarios if s["id"] in args.only]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(scenario):
        async with semaphore:
            try:
                record = await run_scenario(scenario, model=args.model, target_spec=args.target,
                                            root=args.root)
            except Exception as exc:
                record = {"id": scenario["id"], "prompt": scenario["prompt"], "crash": repr(exc)}
            (out / f"{scenario['id']}.json").write_text(json.dumps(record, indent=1, default=str))
            print(f"{scenario['id']:28} calls={len(record.get('calls', [])):2} "
                  f"errors={record.get('errors', '-')} turns={record.get('turns', '-')} "
                  f"secs={record.get('seconds', '-')} "
                  f"tokens_in={record.get('usage', {}).get('input', '-')}"
                  + (f" CRASH {record['crash'][:120]}" if "crash" in record else ""), flush=True)
            return record

    records = await asyncio.gather(*(one(s) for s in scenarios))
    (out / "summary.json").write_text(json.dumps(
        [{k: r.get(k) for k in ("id", "turns", "errors", "seconds", "usage", "crash")} for r in records],
        indent=1, default=str))


def _default_root() -> str:
    import shutil
    import tempfile

    import petrophys

    folder = Path(tempfile.mkdtemp(prefix="petrophys-mcp-eval-"))
    shutil.copy(petrophys.sample("15_9-14"), folder / "my_well.las")
    return str(folder)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default=str(Path(__file__).with_name("mcp_eval_scenarios.json")))
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--target", default="inprocess")
    parser.add_argument("--root", help="Folder the server may use (default: a temporary folder "
                        "holding my_well.las, a copy of sample 15_9-14, as scenario 07 expects)")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    args.root = args.root or _default_root()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
