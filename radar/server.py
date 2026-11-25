"""Zero-dependency HTTP server for the dashboard (stdlib http.server + JSON API)."""
from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import agent
from .cohere_api import telemetry
from .config import ROOT, settings
from .pipeline import Radar

WEB = ROOT / "web"
radar: Radar | None = None


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    return str(o)


class Handler(BaseHTTPRequestHandler):
    server_version = "RegimeRadar/1.0"

    def log_message(self, fmt, *args):  # quieter console
        if "/api/" in (args[0] if args else ""):
            super().log_message(fmt, *args)

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, default=_json_default).encode(), "application/json")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/api/snapshot":
                return self._json(radar.snapshot())
            if url.path == "/api/briefing":
                return self._json(radar.get_briefing(force=q.get("force") == "1"))
            if url.path == "/api/drift":
                return self._json(radar.risk_drift(q.get("ticker", "")))
            if url.path == "/api/search":
                kinds = {"news": ("news",), "filing": ("filing",)}.get(q.get("kind", ""), ("news", "filing"))
                hits = radar.search(q.get("q", ""), k=int(q.get("k", 10)), kinds=kinds)
                return self._json([dict(radar.public_doc(h.doc), score=round(h.score, 4)) for h in hits])
            if url.path == "/api/telemetry":
                return self._json(telemetry.summary())
            if url.path == "/api/exposure":
                return self._json(radar.theme_exposure(q.get("ticker", "")))
            return self._static(url.path)
        except Exception as exc:
            traceback.print_exc()
            return self._json({"error": str(exc)}, 500)

    def do_POST(self):
        url = urlparse(self.path)
        try:
            if url.path == "/api/refresh":
                return self._json(radar.refresh())
            if url.path == "/api/ask":
                b = self._body()
                return self._json(agent.ask(radar.client, radar, b.get("question", ""), b.get("history") or []))
            return self._json({"error": "not found"}, 404)
        except Exception as exc:
            traceback.print_exc()
            return self._json({"error": str(exc)}, 500)

    def _static(self, path: str):
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        f = (WEB / rel).resolve()
        if not str(f).startswith(str(WEB.resolve())) or not f.is_file():
            return self._send(404, b"not found", "text/plain")
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        return self._send(200, f.read_bytes(), ctype)


def main() -> None:
    global radar
    ap = argparse.ArgumentParser(description="Regime Radar dashboard")
    ap.add_argument("--demo", action="store_true", help="offline mode: mock models + bundled snapshot")
    ap.add_argument("--port", type=int, default=settings.port)
    ap.add_argument("--host", default=settings.host)
    args = ap.parse_args()
    if args.demo:
        settings.demo = True
    radar = Radar()
    if not radar.offline:
        radar.load_cache()
    print(f"Regime Radar · mode={'DEMO (offline mock)' if radar.offline else 'LIVE (Cohere API)'}")
    threading.Thread(target=radar.refresh, daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Dashboard → http://{args.host}:{args.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
