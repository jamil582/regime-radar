"""Deterministic offline stand-in for the Cohere API.

It returns responses in exactly the v2 wire format (embeddings.float, message.content,
tool_calls, citations...) so every code path — indexing, structured outputs, the tool-use loop,
citation rendering — runs without a network or key. Used for the demo mode and the test suite.
It is *not* a model: embeddings are hashed bag-of-words and text is template-generated.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

import numpy as np

from .cohere_api import CallRecord, _l2, telemetry

_STOP = set("a an the of to in on for and or as at by with is are was be from that this its it after amid "
            "than more most into over up down new not but has have had will could may their our we us".split())
_SYN = {
    "hike": "rate", "hikes": "rate", "hiking": "rate", "fomc": "fed", "warsh": "fed", "treasury": "yield",
    "yields": "yield", "bond": "yield", "brent": "oil", "wti": "oil", "crude": "oil", "diesel": "oil",
    "iran": "middleeast", "troops": "middleeast", "lng": "energy", "dram": "memory", "nand": "memory",
    "hbm": "memory", "openai": "ai", "anthropic": "ai", "gpu": "ai", "accelerators": "ai", "datacenter": "ai",
    "bullion": "gold", "euro": "dollar", "layoffs": "consumer", "cuts": "consumer", "payrolls": "jobs",
    "spending": "consumer", "retail": "consumer", "apparel": "consumer",
}
_POS = "beat beats strong gains rally rallies surges surge jumps rises climbs record raised raise up wins".split()
_NEG = "miss misses falls drops slides cuts layoffs decline weak selloff sheds slump down lower".split()
TICKERS = ["NVDA", "MU", "AVGO", "XOM", "GLD", "TLT", "NKE", "ACN", "SPY", "NWND", "HBRO"]


def _tokens(text: str) -> list[str]:
    words = [w for w in re.findall(r"[a-z0-9$%]+", text.lower()) if w not in _STOP and len(w) > 1]
    out = []
    for w in words:
        out.append(w[:6])
        if w in _SYN:
            out.append(_SYN[w])
    return out


def _hash_vec(text: str, dim: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    toks = _tokens(text)
    for t in toks + [a + "_" + b for a, b in zip(toks, toks[1:])]:
        h = hashlib.md5(t.encode()).digest()
        idx = int.from_bytes(h[:4], "little")
        # Spread each token across a few dims, biased toward the prefix (Matryoshka-style).
        for k in range(3):
            j = (idx >> (k * 7)) % (dim if k else min(dim, 256))
            v[j] += 1.0 if h[4 + k] & 1 else -1.0
    return v


class MockCohereClient:
    def _rec(self, endpoint, model, feature, items=0, inp=0, out=0):
        telemetry.add(CallRecord(time.time(), endpoint, model, feature, latency_ms=0, input_tokens=inp,
                                 output_tokens=out, items=items))

    # ------------------------------------------------------------------ embed
    def embed(self, texts, *, model, input_type, output_dimension=None, feature="embed"):
        texts = list(texts)
        dim = output_dimension or 2048
        self._rec("/v2/embed", model, feature, items=len(texts), inp=sum(len(t.split()) for t in texts))
        if not texts:
            return np.zeros((0, dim), dtype=np.float32)
        return _l2(np.stack([_hash_vec(t, dim) for t in texts]))

    # ------------------------------------------------------------------ chat
    def chat(self, messages, *, model=None, documents=None, tools=None, json_schema=None,
             thinking_budget=None, temperature=0.3, max_tokens=None, feature="chat"):
        model = model or "command-a-plus-05-2026"
        if json_schema:
            body = self._structured(messages)
        elif tools:
            body = self._agent(messages)
        else:
            body = self._grounded(messages, documents or [])
        n_in = sum(len(json.dumps(m)) for m in messages) // 4 + sum(len(json.dumps(d)) for d in documents or []) // 4
        n_out = sum(len(b.get("text", "")) for b in body["message"]["content"]) // 4
        self._rec("/v2/chat", model, feature, inp=n_in, out=n_out)
        body["usage"] = {"billed_units": {"input_tokens": n_in, "output_tokens": n_out}}
        return body

    # structured outputs -------------------------------------------------
    def _structured(self, messages):
        payload = json.loads(messages[-1]["content"].split("\n", 1)[1])
        items = []
        for h in payload:
            text = (h["headline"] + " " + h.get("summary", "")).lower()
            words = set(re.findall(r"[a-z]+", text))
            score = 0.25 * (len(words & set(_POS)) - len(words & set(_NEG)))
            event = ("central_bank" if re.search(r"\bfed\b|fomc|warsh|dot plot", text) else
                     "macro_data" if re.search(r"cpi|pce|payroll|inflation|spending", text) else
                     "earnings" if re.search(r"revenue|quarter|eps|results", text) else
                     "financing" if re.search(r"lend|installment|ipo|valuation|round", text) else
                     "geopolitics" if re.search(r"middle east|iran|troops", text) else
                     "commodity" if re.search(r"oil|crude|gold|brent|lng", text) else
                     "m_and_a" if re.search(r"merger|acqui|takeover", text) else "other")
            mat = 5 if event in ("central_bank",) else 4 if re.search(r"record|highest|\$\d+ billion", text) else 3
            items.append({"id": h["id"], "sentiment": round(max(-1, min(1, score)), 2), "event_type": event,
                          "materiality": mat, "tickers": [t for t in TICKERS if t.lower() in words],
                          "takeaway": h["headline"][:110]})
        return _body(json.dumps({"items": items}))

    # tool use -----------------------------------------------------------
    def _agent(self, messages):
        last = messages[-1]
        question = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        if last["role"] == "user":
            tickers = [t for t in TICKERS if re.search(rf"\b{t}\b", question.upper())]
            calls = [_call("search_market_intel", {"query": question, "source": "all", "k": 5})]
            for t in tickers[:2]:
                calls.append(_call("get_price_stats", {"ticker": t}))
                if t in ("NWND", "HBRO") or "risk" in question.lower():
                    calls.append(_call("risk_factor_changes", {"ticker": t}))
            plan = f"I will search the news and filings index for '{question[:60]}'" + (
                f" and pull analytics for {', '.join(tickers)}." if tickers else ".")
            return {"id": "mock", "finish_reason": "TOOL_CALL",
                    "message": {"role": "assistant", "tool_plan": plan, "tool_calls": calls, "content": []}}
        # Final answer grounded on the tool documents returned so far.
        docs = []
        for m in messages:
            if m["role"] == "tool":
                for c in m["content"]:
                    docs.append(c["document"])
        parsed = [{"id": d["id"], "data": json.loads(d["data"])} for d in docs]
        # Put analytics-tool outputs (prices, risk drift) ahead of raw search hits.
        parsed.sort(key=lambda d: 0 if isinstance(d["data"], dict) and ("last" in d["data"] or "added" in d["data"]) else 1)
        return self._grounded([{"role": "user", "content": question}], parsed, agent=True)

    # grounded generation with citations ---------------------------------------
    def _grounded(self, messages, documents, agent=False):
        q = messages[-1]["content"]
        parts, cites = [], []

        def add(text, doc_ids=None):
            start = sum(len(p) for p in parts)
            parts.append(text)
            if doc_ids:
                cites.append({"start": start, "end": start + len(text), "text": text,
                              "sources": [{"type": "document", "id": i} for i in doc_ids]})

        def title(d):
            data = d["data"]
            if isinstance(data, dict):
                return data.get("title") or data.get("ticker") or json.dumps(data)[:80]
            return str(data)[:80]

        if not documents:
            add("Not enough evidence in the index to answer that.")
            return _body("".join(parts), cites)
        if "## Bottom line" in q:
            add("## Bottom line\n")
            add("*Offline demo — this briefing is template-generated from retrieved documents, not written by Command A+.*\n\n")
            add(title(documents[0]), [documents[0]["id"]])
            add(". ")
            if len(documents) > 1:
                add(title(documents[1]), [documents[1]["id"]])
                add(".\n\n")
            add("## Regime scorecard\n")
            for d in documents[2:9]:
                add("- ")
                add(title(d), [d["id"]])
                add("\n")
            add("\n## What would change my mind\n- September CPI and payrolls ahead of the Oct 27-28 FOMC.\n"
                "- Brent breaking back below $90.\n")
        else:
            add("*Offline demo answer (template, not Command A+):* " if agent else "")
            add("The most relevant evidence: ")
            for i, d in enumerate(documents[:5]):
                data = d["data"]
                if isinstance(data, dict) and "last" in data:
                    add(f"{data.get('ticker')} last {data.get('last')} ({data.get('chg_1m_pct')}% 1m, "
                        f"vol {data.get('vol_ann_pct')}%)", [d["id"]])
                elif isinstance(data, dict) and "added" in data:
                    new = [a.get("title", "") for a in data["added"][:2]]
                    add(f"new risk language in {data.get('ticker')}: " + "; ".join(new), [d["id"]])
                else:
                    add(title(d), [d["id"]])
                add("; " if i < min(4, len(documents) - 1) else ".")
        return _body("".join(parts), cites)


def _call(name, args):
    return {"id": f"call_{hashlib.md5((name + json.dumps(args)).encode()).hexdigest()[:8]}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _body(text, citations=None):
    return {"id": "mock", "finish_reason": "COMPLETE",
            "message": {"role": "assistant", "content": [{"type": "text", "text": text}],
                        "citations": citations or []}}
