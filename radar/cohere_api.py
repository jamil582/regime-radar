"""A thin, dependency-light client for Cohere's v2 REST API.

Only `requests` is needed. Every call is recorded in `telemetry` so the dashboard can show
exactly which model served which feature, with latency and billed tokens.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import Any, Iterable

import numpy as np

from .config import settings

EMBED_BATCH = 96  # API limit: max 96 texts/inputs per embed call


class CohereError(RuntimeError):
    pass


@dataclass
class CallRecord:
    ts: float
    endpoint: str
    model: str
    feature: str
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    items: int = 0
    ok: bool = True


class Telemetry:
    def __init__(self, maxlen: int = 500):
        self._calls: deque[CallRecord] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, rec: CallRecord) -> None:
        with self._lock:
            self._calls.append(rec)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            calls = list(self._calls)
        by_model: dict[str, dict[str, Any]] = {}
        for c in calls:
            m = by_model.setdefault(c.model, {"calls": 0, "items": 0, "input_tokens": 0,
                                              "output_tokens": 0, "latency_ms": [], "features": set()})
            m["calls"] += 1
            m["items"] += c.items
            m["input_tokens"] += c.input_tokens
            m["output_tokens"] += c.output_tokens
            m["latency_ms"].append(c.latency_ms)
            m["features"].add(c.feature)
        for m in by_model.values():
            lat = sorted(m.pop("latency_ms"))
            m["p50_ms"] = lat[len(lat) // 2] if lat else 0
            m["features"] = sorted(m["features"])
        return {"models": by_model, "recent": [asdict(c) for c in calls[-25:]][::-1]}


telemetry = Telemetry()


class CohereClient:
    """Wraps /v2/embed and /v2/chat."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, timeout: int = 120):
        import requests  # imported lazily so offline mode never needs it

        self._requests = requests
        self.api_key = api_key or settings.api_key
        self.base_url = (base_url or settings.base_url).rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Client-Name": "regime-radar",
        })

    # ------------------------------------------------------------------ transport
    def _post(self, path: str, payload: dict[str, Any], feature: str, items: int = 0) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        delay = 1.5
        for attempt in range(5):
            t0 = time.perf_counter()
            resp = self.session.post(url, data=json.dumps(payload), timeout=self.timeout)
            latency = int((time.perf_counter() - t0) * 1000)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                retry_after = resp.headers.get("Retry-After")
                time.sleep(float(retry_after) if retry_after else delay)
                delay *= 2
                continue
            if not resp.ok:
                telemetry.add(CallRecord(time.time(), path, payload.get("model", "?"), feature, latency,
                                         items=items, ok=False))
                raise CohereError(f"{path} -> HTTP {resp.status_code}: {resp.text[:500]}")
            body = resp.json()
            billed = (body.get("meta") or {}).get("billed_units") or (body.get("usage") or {}).get("billed_units") or {}
            telemetry.add(CallRecord(
                time.time(), path, payload.get("model", "?"), feature, latency,
                input_tokens=int(billed.get("input_tokens", 0) or 0),
                output_tokens=int(billed.get("output_tokens", 0) or 0),
                items=items,
            ))
            return body
        raise CohereError(f"{path}: retries exhausted")

    # ------------------------------------------------------------------ embed
    def embed(
        self,
        texts: Iterable[str],
        *,
        model: str,
        input_type: str,
        output_dimension: int | None = None,
        feature: str = "embed",
    ) -> np.ndarray:
        """Embed texts in batches of 96. Returns an (n, d) float32 matrix, L2-normalised."""
        texts = list(texts)
        out: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i:i + EMBED_BATCH]
            payload: dict[str, Any] = {
                "model": model,
                "input_type": input_type,
                "texts": batch,
                "embedding_types": ["float"],
                "truncate": "END",
            }
            if output_dimension:
                payload["output_dimension"] = output_dimension
            body = self._post("/v2/embed", payload, feature, items=len(batch))
            out.extend(body["embeddings"]["float"])
        mat = np.asarray(out, dtype=np.float32)
        return _l2(mat)

    # ------------------------------------------------------------------ chat
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        documents: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        json_schema: dict[str, Any] | None = None,
        thinking_budget: int | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        feature: str = "chat",
    ) -> dict[str, Any]:
        """Call /v2/chat. Returns the raw response body (message, citations, usage...)."""
        payload: dict[str, Any] = {
            "model": model or settings.chat_model,
            "messages": messages,
            "temperature": temperature,
        }
        if documents:
            payload["documents"] = documents
        if tools:
            payload["tools"] = tools
        if json_schema:
            payload["response_format"] = {"type": "json_object", "json_schema": json_schema}
        if thinking_budget is None:
            # Extraction-style calls don't benefit from reasoning; keep them fast and cheap.
            payload["thinking"] = {"type": "disabled"}
        elif thinking_budget > 0:
            payload["thinking"] = {"type": "enabled", "token_budget": thinking_budget}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        return self._post("/v2/chat", payload, feature)


# ---------------------------------------------------------------------- helpers
def _l2(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


def message_text(body: dict[str, Any]) -> str:
    """Concatenate the text blocks of a v2 chat response (skips `thinking` blocks)."""
    parts = []
    for block in (body.get("message") or {}).get("content") or []:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def message_thinking(body: dict[str, Any]) -> str:
    parts = []
    for block in (body.get("message") or {}).get("content") or []:
        if block.get("type") == "thinking":
            parts.append(block.get("thinking", ""))
    return "".join(parts)


def parse_json(text: str) -> Any:
    """Structured outputs return JSON text; be tolerant of stray code fences anyway."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("\n") + 1:] if "\n" in text else text
    return json.loads(text)


def get_client():
    """Real client when a key is configured, otherwise the deterministic offline mock."""
    if settings.offline:
        from .mock_cohere import MockCohereClient
        return MockCohereClient()
    return CohereClient()
