"""Radar: the stateful core that ingests data, embeds it, scores themes and answers queries."""
from __future__ import annotations

import json
import math
import threading
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np

from . import briefing as briefing_mod
from . import risk_drift as drift_mod
from .cohere_api import get_client
from .config import CACHE_DIR, DATA_DIR, settings
from .enrich import enrich_headlines
from .sources import prices as prices_mod
from .themes import THEMES
from .vectorstore import Hit, VectorIndex


class Radar:
    def __init__(self, client=None):
        self.client = client or get_client()
        self.offline = settings.offline
        self.watchlist = list(settings.watchlist)
        self.news = VectorIndex("news · embed-v5.0-fast", settings.coarse_dim)
        self.filings = VectorIndex("filings · embed-v5.0-pro", settings.coarse_dim)
        self.theme_vecs: np.ndarray | None = None
        self.enrichment: dict[str, dict[str, Any]] = {}
        self.prices: dict[str, list[dict[str, Any]]] = {}
        self.briefing: dict[str, Any] | None = None
        self._drift_cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self.last_refresh: str | None = None
        self.status: list[str] = []

    # ================================================================ ingest
    def _log(self, msg: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.status.append(f"{stamp} {msg}")
        self.status = self.status[-40:]
        print(f"[radar] {msg}")

    def _cache_key(self) -> str:
        return "mock" if self.offline else settings.embed_fast.replace(".", "_")

    def load_cache(self) -> None:
        base = CACHE_DIR / self._cache_key()
        if self.news.load(base / "news"):
            self._log(f"loaded {len(self.news)} cached news vectors")
        if self.filings.load(base / "filings"):
            self._log(f"loaded {len(self.filings)} cached filing vectors")
        enr = base / "enrichment.json"
        if enr.exists():
            self.enrichment = json.loads(enr.read_text())

    def save_cache(self) -> None:
        base = CACHE_DIR / self._cache_key()
        self.news.save(base / "news")
        self.filings.save(base / "filings")
        (base / "enrichment.json").write_text(json.dumps(self.enrichment))

    def refresh(self) -> dict[str, Any]:
        with self._lock:
            t0 = time.time()
            self._embed_themes()
            headlines, filings = (self._sample_data() if self.offline else self._live_data())

            new_news = [h for h in headlines if not self.news.has(h["id"])]
            if new_news:
                self._log(f"embedding {len(new_news)} new headlines with {settings.embed_fast}")
                vecs = self.client.embed([self._news_text(h) for h in new_news], model=settings.embed_fast,
                                         input_type="search_document", feature="index news (fast)")
                self.news.add(new_news, vecs)

            new_filings = [f for f in filings if not self.filings.has(f["id"])]
            if new_filings:
                self._log(f"embedding {len(new_filings)} 10-K chunks with {settings.embed_pro}")
                vecs = self.client.embed([f["text"] for f in new_filings], model=settings.embed_pro,
                                         input_type="search_document", feature="index filings (pro)")
                self.filings.add(new_filings, vecs)

            todo = [h for h in self.news.docs if h["id"] not in self.enrichment]
            if todo:
                self._log(f"enriching {len(todo)} headlines with {settings.chat_model} structured outputs")
                self.enrichment.update(enrich_headlines(self.client, todo))

            self._load_prices()
            self.briefing = None
            self._drift_cache.clear()
            if not self.offline:
                self.save_cache()
            self.last_refresh = datetime.now().isoformat(timespec="seconds")
            self._log(f"refresh done in {time.time() - t0:.1f}s")
            return {"news": len(self.news), "filings": len(self.filings), "new_news": len(new_news),
                    "new_filings": len(new_filings)}

    @staticmethod
    def _news_text(h: dict[str, Any]) -> str:
        return f"{h['title']}. {h.get('summary', '')}".strip()

    def _embed_themes(self) -> None:
        if self.theme_vecs is None:
            # Embedded once with Fast as *queries*; because Fast and Pro share an embedding space,
            # the same vectors score news (Fast) and filings (Pro).
            self.theme_vecs = self.client.embed([t["description"] for t in THEMES], model=settings.embed_fast,
                                                input_type="search_query", feature="theme anchors")

    def _sample_data(self) -> tuple[list[dict], list[dict]]:
        raw = json.loads((DATA_DIR / "sample" / "headlines.json").read_text())["items"]
        headlines = []
        for i, h in enumerate(raw):
            headlines.append({"id": f"demo-news-{i:03d}", "kind": "news", "title": h["title"],
                              "summary": h.get("summary", ""), "url": h.get("url", ""),
                              "published": h["date"] + "T12:00:00+00:00", "publisher": h.get("publisher", ""),
                              "tickers": h.get("tickers", [])})
        filings = []
        rf = json.loads((DATA_DIR / "sample" / "risk_factors.json").read_text())["filings"]
        for f in rf:
            filings.extend(self._filing_docs(f["ticker"], f["fiscal_year"], f["filed"], "", f["chunks"],
                                             company=f.get("company")))
        return headlines, filings

    def _live_data(self) -> tuple[list[dict], list[dict]]:
        from .sources import news as news_mod
        from .sources import sec as sec_mod

        self._log("fetching RSS headlines")
        headlines = news_mod.fetch_headlines(self.watchlist)
        filings: list[dict] = []
        if settings.sec_user_agent:
            for t in settings.filings_tickers:
                if any(d.get("ticker") == t for d in self.filings.docs):
                    continue
                try:
                    self._log(f"pulling {t} 10-K risk factors from EDGAR")
                    for f in sec_mod.latest_10k_risk_factors(t):
                        filings.extend(self._filing_docs(t, f["fiscal_year"], f["filed"], f["url"], f["chunks"]))
                except Exception as exc:
                    self._log(f"EDGAR {t} failed: {exc}")
        else:
            self._log("SEC_USER_AGENT not set - skipping 10-K risk factors")
        return headlines, filings

    @staticmethod
    def _filing_docs(ticker, fy, filed, url, chunks, company=None) -> list[dict[str, Any]]:
        docs = []
        for i, text in enumerate(chunks):
            first = text.split(". ")[0]
            docs.append({"id": f"10k-{ticker}-{fy}-{i:03d}", "kind": "filing", "ticker": ticker,
                         "company": company or ticker, "fiscal_year": fy, "filed": filed, "url": url,
                         "title": f"{ticker} FY{fy} · {first[:90]}", "text": text,
                         "published": f"{filed}T00:00:00+00:00", "tickers": [ticker]})
        return docs

    def _load_prices(self) -> None:
        if self.offline:
            self.prices = {t: _synthetic_prices(t) for t in self.watchlist}
            return
        for t in self.watchlist:
            try:
                self.prices[t] = prices_mod.fetch_history(t)
            except Exception as exc:
                self._log(f"price fetch {t} failed: {exc}")

    # ================================================================ retrieval
    def embed_query(self, text: str) -> np.ndarray:
        return self.client.embed([text], model=settings.embed_fast, input_type="search_query",
                                 feature="query")[0]

    def search(self, query: str, k: int = 8, kinds=("news", "filing"), ticker: str | None = None) -> list[Hit]:
        return self.search_by_vector(self.embed_query(query), k=k, kinds=kinds, ticker=ticker)

    def search_by_vector(self, vec, k=8, kinds=("news", "filing"), ticker=None) -> list[Hit]:
        where = (lambda d: ticker.upper() in [x.upper() for x in d.get("tickers", [])]) if ticker else None
        hits: list[Hit] = []
        if "news" in kinds:
            hits += self.news.search(vec, k=k, where=where)
        if "filing" in kinds:
            hits += self.filings.search(vec, k=k, where=where)
        hits.sort(key=lambda h: -h.score)
        return hits[:k]

    def public_doc(self, d: dict[str, Any]) -> dict[str, Any]:
        out = {k: d.get(k) for k in ("id", "kind", "title", "url", "published", "publisher", "tickers",
                                     "ticker", "fiscal_year")}
        if d.get("kind") == "filing":
            out["text"] = d.get("text", "")[:700]
        else:
            out["summary"] = d.get("summary", "")
        if d["id"] in self.enrichment:
            out["ai"] = self.enrichment[d["id"]]
        return {k: v for k, v in out.items() if v not in (None, "", [])}

    # ================================================================ themes
    def _theme_matrix(self) -> np.ndarray:
        vecs = self.news.full_vectors()
        if not len(vecs) or self.theme_vecs is None:
            return np.zeros((0, len(THEMES)))
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        return vecs @ self.theme_vecs.T

    def theme_assignments(self) -> list[tuple[int, float]]:
        """Primary theme per headline via per-theme z-scores (robust to each anchor's baseline similarity)."""
        sims = self._theme_matrix()
        if not len(sims):
            return []
        z = (sims - sims.mean(axis=0)) / (sims.std(axis=0) + 1e-6)
        out = []
        for row in z:
            j = int(row.argmax())
            out.append((j, float(row[j])) if row[j] > 0.5 else (-1, float(row[j])))
        return out

    def theme_summary(self) -> list[dict[str, Any]]:
        assigns = self.theme_assignments()
        docs = self.news.docs
        latest = max((_d(d) for d in docs), default=date.today())
        start = latest - timedelta(days=20)
        days = [(start + timedelta(days=i)).isoformat() for i in range(21)]
        out = []
        for j, theme in enumerate(THEMES):
            members = [(docs[i], z) for i, (tj, z) in enumerate(assigns) if tj == j]
            heat, sent, timeline = 0.0, [], defaultdict(float)
            for d, z in members:
                e = self.enrichment.get(d["id"], {})
                w = e.get("materiality", 2) * math.exp(-(latest - _d(d)).days / 7)
                heat += w
                timeline[_d(d).isoformat()] += e.get("materiality", 2)
                if "sentiment" in e:
                    sent.append(e["sentiment"])
            recent = sum(v for k, v in timeline.items() if k > (latest - timedelta(days=3)).isoformat())
            prior = sum(v for k, v in timeline.items() if k <= (latest - timedelta(days=3)).isoformat())
            out.append({
                **{k: theme[k] for k in ("id", "name", "slot", "description")},
                "vector": None if self.theme_vecs is None else self.theme_vecs[j],
                "heat": round(heat, 2),
                "count": len(members),
                "sentiment": round(float(np.mean(sent)), 2) if sent else 0.0,
                "momentum": ("heating" if recent > max(prior, 1) * 0.8 else "cooling" if recent < prior * 0.4 else "steady"),
                "timeline": [timeline.get(day, 0) for day in days],
                "days": days,
                "top": [self.public_doc(d) for d, _ in sorted(members, key=lambda m: -m[1])[:4]],
            })
        return out

    def theme_exposure(self, ticker: str) -> dict[str, Any]:
        sims = self._theme_matrix()
        idx = [i for i, d in enumerate(self.news.docs) if ticker.upper() in d.get("tickers", [])]
        if not idx:
            return {"ticker": ticker, "note": "no recent news for this ticker"}
        rel = sims[idx].mean(axis=0) - sims.mean(axis=0)
        ranked = sorted(zip(THEMES, rel), key=lambda x: -x[1])
        return {"ticker": ticker.upper(), "n_headlines": len(idx),
                "exposure": [{"theme": t["name"], "score": round(float(s) * 100, 1)} for t, s in ranked]}

    # ================================================================ prices
    def price_stats(self, ticker: str) -> dict[str, Any]:
        rows = self.prices.get(ticker.upper())
        if rows is None and not self.offline:
            try:
                rows = self.prices[ticker.upper()] = prices_mod.fetch_history(ticker.upper())
            except Exception as exc:
                return {"ticker": ticker, "error": str(exc)}
        if not rows:
            return {"ticker": ticker, "error": "no price data"}
        stats = {"ticker": ticker.upper(), **prices_mod.price_stats(rows)}
        if self.offline:
            stats["note"] = "SYNTHETIC demo series indexed to 100 - not real prices"
        return stats

    def compare(self, a: str, b: str) -> dict[str, Any]:
        ra, rb = self.prices.get(a.upper()), self.prices.get(b.upper())
        if not ra or not rb:
            return {"error": "price data missing for one of the tickers"}
        sa, sb = prices_mod.price_stats(ra), prices_mod.price_stats(rb)
        out = {"pair": f"{a.upper()}/{b.upper()}", "corr_3m": prices_mod.correlation(ra, rb),
               "rel_perf_3m_pct": None if sa.get("chg_3m_pct") is None or sb.get("chg_3m_pct") is None
               else round(sa["chg_3m_pct"] - sb["chg_3m_pct"], 2)}
        if self.offline:
            out["note"] = "SYNTHETIC demo series - not real prices"
        return out

    # ================================================================ risk drift
    def filing_tickers(self) -> list[str]:
        return sorted({d["ticker"] for d in self.filings.docs})

    def risk_drift(self, ticker: str, summarise: bool = True) -> dict[str, Any]:
        ticker = ticker.upper()
        if ticker in self._drift_cache and (not summarise or "summary" in self._drift_cache[ticker]):
            return self._drift_cache[ticker]
        docs = self.filings.docs
        years = sorted({d["fiscal_year"] for d in docs if d["ticker"] == ticker}, reverse=True)
        if len(years) < 2:
            return {"ticker": ticker, "error": "need two 10-K filings for this ticker"}
        vecs = self.filings.full_vectors()
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        ni = [i for i, d in enumerate(docs) if d["ticker"] == ticker and d["fiscal_year"] == years[0]]
        oi = [i for i, d in enumerate(docs) if d["ticker"] == ticker and d["fiscal_year"] == years[1]]
        drift = drift_mod.compute_drift(vecs[ni], vecs[oi], [docs[i] for i in ni], [docs[i] for i in oi])
        drift.update({"ticker": ticker, "new_year": years[0], "old_year": years[1]})
        if summarise:
            drift["summary"] = drift_mod.summarise_drift(self.client, ticker, drift)
        self._drift_cache[ticker] = drift
        return drift

    # ================================================================ briefing
    def get_briefing(self, force: bool = False) -> dict[str, Any]:
        with self._lock:
            if self.briefing is None or force:
                self.briefing = briefing_mod.build_briefing(self.client, self)
                self.briefing["generated"] = datetime.now().isoformat(timespec="seconds")
            return self.briefing

    # ================================================================ dashboard payload
    def snapshot(self) -> dict[str, Any]:
        assigns = self.theme_assignments()
        feed = []
        for i, d in enumerate(self.news.docs):
            doc = self.public_doc(d)
            tj = assigns[i][0] if i < len(assigns) else -1
            doc["theme"] = THEMES[tj]["id"] if tj >= 0 else None
            feed.append(doc)
        feed.sort(key=lambda d: (d.get("published") or "", d.get("ai", {}).get("materiality", 0)), reverse=True)
        tiles = []
        for t in self.watchlist:
            s = self.price_stats(t)
            rows = self.prices.get(t) or []
            tiles.append({**s, "spark": [r["close"] for r in rows[-60:]]})
        themes = [{k: v for k, v in t.items() if k != "vector"} for t in self.theme_summary()]
        return {
            "mode": "demo" if self.offline else "live",
            "models": {"embed_fast": settings.embed_fast, "embed_pro": settings.embed_pro,
                       "chat": settings.chat_model},
            "last_refresh": self.last_refresh,
            "themes": themes,
            "feed": feed[:120],
            "tiles": tiles,
            "filing_tickers": self.filing_tickers(),
            "index_stats": [self.news.stats(), self.filings.stats()],
            "status": self.status[-8:],
        }


def _d(doc: dict[str, Any]) -> date:
    p = doc.get("published")
    try:
        return datetime.fromisoformat(p).date() if p else date.today()
    except ValueError:
        return date.today()


def _synthetic_prices(ticker: str, n: int = 130) -> list[dict[str, Any]]:
    """Deterministic random walk indexed to 100. Clearly labelled synthetic in the UI and tool output."""
    rng = np.random.default_rng(sum(map(ord, ticker)) * 7919)
    drift = {"XOM": 0.0015, "GLD": 0.0012, "MU": 0.002, "NVDA": 0.0012, "AVGO": 0.001, "TLT": -0.0009,
             "NKE": -0.0012}.get(ticker, 0.0003)
    vol = {"MU": 0.03, "NVDA": 0.028, "AVGO": 0.025, "TLT": 0.011, "GLD": 0.012, "SPY": 0.01}.get(ticker, 0.018)
    r = rng.normal(drift, vol, n)
    c = 100 * np.exp(np.cumsum(r) - r[0])  # starts at 100
    end = date(2026, 10, 2)
    days, d = [], end
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    days.reverse()
    return [{"date": day.isoformat(), "close": round(float(v), 2)} for day, v in zip(days, c)]
