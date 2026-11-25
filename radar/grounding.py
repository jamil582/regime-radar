"""Helpers for Command A+ grounded generation: building documents and normalising citations."""
from __future__ import annotations

import json
from typing import Any


def as_document(doc: dict[str, Any], enrichment: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convert an internal record into a v2 `documents` entry. Values are kept as short strings."""
    data: dict[str, str] = {"title": doc.get("title", "")}
    if doc.get("kind") == "filing":
        data["snippet"] = doc.get("text", "")[:1500]
        data["source"] = f"{doc.get('ticker')} 10-K FY{doc.get('fiscal_year')} - Item 1A Risk Factors"
    else:
        data["snippet"] = doc.get("summary") or doc.get("title", "")
        data["source"] = doc.get("publisher", "")
        if doc.get("published"):
            data["published"] = doc["published"][:10]
    if enrichment:
        data["analyst_tags"] = (f"sentiment={enrichment.get('sentiment')}, event={enrichment.get('event_type')}, "
                                f"materiality={enrichment.get('materiality')}")
    return {"id": doc["id"], "data": data}


def tool_document(doc_id: str, payload: Any) -> dict[str, Any]:
    """Tool results are returned to the model as documents so the final answer can cite them."""
    return {"type": "document", "document": {"id": doc_id, "data": json.dumps(payload, default=str)}}


def normalise_citations(body: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for c in (body.get("message") or {}).get("citations") or []:
        ids = []
        for s in c.get("sources") or []:
            sid = s.get("id") or (s.get("document") or {}).get("id")
            if sid and sid not in ids:
                ids.append(sid)
        if ids:
            out.append({"start": c.get("start"), "end": c.get("end"), "text": c.get("text"), "sources": ids})
    return out
