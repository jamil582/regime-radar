"""Headline enrichment with Command A+ Structured Outputs.

A JSON Schema constrains the model's output, so every headline comes back with machine-usable
fields (sentiment, event type, materiality, tickers) we can chart and filter on — no regex parsing.
"""
from __future__ import annotations

import json
from typing import Any

from .cohere_api import message_text, parse_json
from .themes import THEMES

EVENT_TYPES = ["earnings", "guidance", "macro_data", "central_bank", "geopolitics", "m_and_a",
               "financing", "regulation", "product", "labor", "commodity", "other"]

SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "sentiment", "event_type", "materiality", "tickers", "takeaway"],
                "properties": {
                    "id": {"type": "string"},
                    "sentiment": {"type": "number", "description": "-1 very bearish .. +1 very bullish for the affected assets"},
                    "event_type": {"type": "string", "enum": EVENT_TYPES},
                    "materiality": {"type": "integer", "description": "1 noise .. 5 market-moving"},
                    "tickers": {"type": "array", "items": {"type": "string"}},
                    "takeaway": {"type": "string", "description": "<= 18 words, what a PM should know"},
                },
            },
        }
    },
}

SYSTEM = (
    "You are a buy-side news analyst in October 2026. The macro regime: the Fed resumed hiking, "
    "10-year Treasury yields are above 5%, Brent is near $100 on Middle East tensions, gold is above $4,000, "
    "and AI infrastructure spending is enormous. Score each headline strictly from its text. "
    "Use US ticker symbols only when the company or ETF is clearly identified."
)


def enrich_headlines(client, headlines: list[dict[str, Any]], batch_size: int = 15) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for i in range(0, len(headlines), batch_size):
        batch = headlines[i:i + batch_size]
        payload = [{"id": h["id"], "headline": h["title"], "summary": h.get("summary", "")[:400]} for h in batch]
        body = client.chat(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Analyse these headlines. Return one item per id.\n" + json.dumps(payload)},
            ],
            json_schema=SCHEMA,
            temperature=0.0,
            feature="enrich (structured output)",
        )
        try:
            for item in parse_json(message_text(body)).get("items", []):
                if item.get("id"):
                    item["sentiment"] = max(-1.0, min(1.0, float(item.get("sentiment", 0))))
                    item["materiality"] = int(max(1, min(5, int(item.get("materiality", 1)))))
                    results[item["id"]] = item
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"[enrich] could not parse batch {i}: {exc}")
    return results


def theme_names() -> str:
    return ", ".join(t["name"] for t in THEMES)
