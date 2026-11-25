"""Live headline ingestion from public RSS feeds (no API keys needed)."""
from __future__ import annotations

import hashlib
import html
import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus

UA = "Mozilla/5.0 (compatible; RegimeRadar/1.0; +https://github.com)"

FEEDS = {
    "yahoo": "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
    "google": "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en",
}

MACRO_QUERIES = [
    "Federal Reserve rate hike Treasury yields",
    "oil prices Middle East Brent",
    "AI data center spending chips",
    "gold price record",
    "consumer spending layoffs retail earnings",
    "dollar euro currency markets",
]

_TAG = re.compile(r"<[^>]+>")


def _clean(text: str | None) -> str:
    return html.unescape(_TAG.sub(" ", text or "")).replace("\xa0", " ").strip()


def _doc_id(title: str, link: str) -> str:
    return "news-" + hashlib.sha1(f"{title}|{link}".encode()).hexdigest()[:12]


def parse_rss(xml_text: str, *, source: str, ticker: str | None = None) -> list[dict[str, Any]]:
    items = []
    root = ET.fromstring(xml_text)
    for it in root.iter("item"):
        title = _clean(it.findtext("title"))
        if not title:
            continue
        link = (it.findtext("link") or "").strip()
        summary = _clean(it.findtext("description"))
        pub = it.findtext("pubDate")
        try:
            published = parsedate_to_datetime(pub).isoformat() if pub else None
        except (TypeError, ValueError):
            published = None
        publisher = _clean(it.findtext("source")) or source
        items.append({
            "id": _doc_id(title, link),
            "kind": "news",
            "title": title,
            "summary": summary if summary and summary != title else "",
            "url": link,
            "published": published,
            "publisher": publisher,
            "tickers": [ticker] if ticker else [],
        })
    return items


def fetch_headlines(watchlist: list[str], per_feed: int = 15, timeout: int = 15) -> list[dict[str, Any]]:
    import requests

    out: dict[str, dict[str, Any]] = {}
    s = requests.Session()
    s.headers["User-Agent"] = UA
    jobs = [("yahoo", FEEDS["yahoo"].format(ticker=t), t) for t in watchlist]
    jobs += [("google", FEEDS["google"].format(query=quote_plus(q + " when:7d")), None) for q in MACRO_QUERIES]
    for source, url, ticker in jobs:
        try:
            r = s.get(url, timeout=timeout)
            r.raise_for_status()
            for item in parse_rss(r.text, source=source, ticker=ticker)[:per_feed]:
                if item["id"] in out:
                    out[item["id"]]["tickers"] = sorted(set(out[item["id"]]["tickers"]) | set(item["tickers"]))
                else:
                    out[item["id"]] = item
        except Exception as exc:  # one bad feed shouldn't kill the refresh
            print(f"[news] {source} {ticker or ''} failed: {exc}")
    return list(out.values())
