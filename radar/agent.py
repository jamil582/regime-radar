""""Ask the Desk" — a multi-step tool-use agent on Command A+.

The model plans, calls our local tools (vector search over Embed 5 indexes, price analytics,
theme exposure, 10-K risk drift), and writes a final answer whose citations point back to the
exact tool outputs it relied on.
"""
from __future__ import annotations

import json
from typing import Any

from .cohere_api import message_text, message_thinking
from .grounding import normalise_citations, tool_document

MAX_STEPS = 6

SYSTEM = (
    "You are the research desk for a cross-asset portfolio manager in October 2026. "
    "Use the tools to gather evidence before answering; prefer several targeted searches over one vague one. "
    "Quote concrete numbers. Distinguish clearly between facts from tools and your own inference. "
    "Keep answers under 250 words unless asked for more."
)

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_market_intel",
            "description": "Semantic search over recent news headlines and 10-K risk-factor sections (Embed 5).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language search query"},
                    "source": {"type": "string", "enum": ["news", "filings", "all"], "description": "Which index to search"},
                    "ticker": {"type": "string", "description": "Optional ticker to restrict results to"},
                    "k": {"type": "integer", "description": "Number of results, default 6"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price_stats",
            "description": "Latest price, 1d/5d/1m/3m % change, annualised volatility and max drawdown for a ticker.",
            "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_assets",
            "description": "3-month daily-return correlation and relative performance between two tickers.",
            "parameters": {
                "type": "object",
                "properties": {"ticker_a": {"type": "string"}, "ticker_b": {"type": "string"}},
                "required": ["ticker_a", "ticker_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "theme_exposure",
            "description": "How strongly a ticker's recent news flow maps onto each 2026 macro regime theme.",
            "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "risk_factor_changes",
            "description": "New and dropped risk factors between a company's two most recent 10-K filings.",
            "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]},
        },
    },
]


def _run_tool(radar, name: str, args: dict[str, Any]) -> Any:
    if name == "search_market_intel":
        src = args.get("source", "all")
        kinds = {"news": ("news",), "filings": ("filing",)}.get(src, ("news", "filing"))
        hits = radar.search(args["query"], k=int(args.get("k") or 6), kinds=kinds, ticker=args.get("ticker"))
        return [dict(radar.public_doc(h.doc), relevance=round(h.score, 3)) for h in hits]
    if name == "get_price_stats":
        return radar.price_stats(args["ticker"])
    if name == "compare_assets":
        return radar.compare(args["ticker_a"], args["ticker_b"])
    if name == "theme_exposure":
        return radar.theme_exposure(args["ticker"])
    if name == "risk_factor_changes":
        return radar.risk_drift(args["ticker"], summarise=False)
    return {"error": f"unknown tool {name}"}


def ask(client, radar, question: str, history: list[dict[str, Any]] | None = None,
        thinking_budget: int = 1024) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}]
    messages += history or []
    messages.append({"role": "user", "content": question})
    trace: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}

    for step in range(MAX_STEPS):
        body = client.chat(messages, tools=TOOLS, thinking_budget=thinking_budget, temperature=0.2,
                           feature="agent (tool use)")
        msg = body.get("message") or {}
        calls = msg.get("tool_calls") or []
        if not calls:
            return {
                "answer": message_text(body),
                "citations": normalise_citations(body),
                "thinking": message_thinking(body),
                "trace": trace,
                "sources": sources,
                "steps": step + 1,
            }
        messages.append({"role": "assistant", "tool_plan": msg.get("tool_plan", ""), "tool_calls": calls})
        for call in calls:
            fn = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            try:
                result = _run_tool(radar, fn, args)
            except Exception as exc:  # surface tool errors to the model instead of crashing
                result = {"error": str(exc)}
            trace.append({"step": step + 1, "plan": msg.get("tool_plan", ""), "tool": fn, "args": args,
                          "result_preview": json.dumps(result, default=str)[:400]})
            content = []
            if isinstance(result, list):
                for item in result:
                    did = item.get("id") or f"{call['id']}-{len(content)}"
                    sources[did] = item
                    content.append(tool_document(did, item))
            else:
                did = f"{fn}:{json.dumps(args, sort_keys=True)}"
                sources[did] = {"id": did, "title": f"{fn}({', '.join(f'{k}={v}' for k, v in args.items())})",
                                "kind": "tool", "data": result}
                content.append(tool_document(did, result))
            if not content:
                content.append(tool_document(f"{call['id']}-empty", {"result": "no results"}))
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})

    return {"answer": "I hit the tool-step limit before finishing. Try a narrower question.",
            "citations": [], "thinking": "", "trace": trace, "sources": sources, "steps": MAX_STEPS}
