"""The morning regime briefing: retrieval-augmented, reasoning-enabled, fully cited."""
from __future__ import annotations

from typing import Any

from .cohere_api import message_text, message_thinking
from .grounding import as_document, normalise_citations

PROMPT = """Write today's regime briefing for a cross-asset portfolio manager.

Structure (markdown, no preamble):
## Bottom line
Two sentences: what regime are we in and what changed.
## Regime scorecard
For each of the themes below, one bullet: **theme name** — direction (heating / cooling / steady) and the evidence.
{themes}
## Watchlist implications
One bullet per ticker that the documents give real evidence for: {tickers}.
## What would change my mind
Two or three falsifiable signposts to watch next week.

Only use facts from the provided documents. Be specific with numbers. If evidence is thin, say so."""


def build_briefing(client, radar, thinking_budget: int = 2048, per_theme: int = 5) -> dict[str, Any]:
    docs, seen = [], set()
    for theme in radar.theme_summary():
        for hit in radar.search_by_vector(theme["vector"], k=per_theme, kinds=("news",)):
            if hit.doc["id"] not in seen:
                seen.add(hit.doc["id"])
                docs.append(hit.doc)
    for hit in radar.search("new or emerging risk factors", k=4, kinds=("filing",)):
        if hit.doc["id"] not in seen:
            seen.add(hit.doc["id"])
            docs.append(hit.doc)

    theme_lines = "\n".join(
        f"- {t['name']} (heat {t['heat']:.1f}, avg sentiment {t['sentiment']:+.2f}, {t['count']} items)"
        for t in radar.theme_summary()
    )
    body = client.chat(
        [{"role": "user", "content": PROMPT.format(themes=theme_lines, tickers=", ".join(radar.watchlist))}],
        documents=[as_document(d, radar.enrichment.get(d["id"])) for d in docs],
        thinking_budget=thinking_budget,
        temperature=0.2,
        feature="briefing (RAG + reasoning)",
    )
    text = message_text(body)
    return {
        "markdown": text,
        "citations": normalise_citations(body),
        "thinking": message_thinking(body),
        "sources": {d["id"]: radar.public_doc(d) for d in docs},
        "n_documents": len(docs),
    }
