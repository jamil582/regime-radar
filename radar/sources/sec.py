"""Pull the Item 1A "Risk Factors" section of a company's last two 10-K filings from SEC EDGAR.

EDGAR is free but requires a descriptive User-Agent (set SEC_USER_AGENT="Your Name you@example.com").
"""
from __future__ import annotations

import html
import re
from typing import Any

from ..config import settings

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

_cik_cache: dict[str, int] = {}


def _session():
    import requests

    if not settings.sec_user_agent:
        raise RuntimeError("Set SEC_USER_AGENT (e.g. 'Jane Doe jane@example.com') to use SEC EDGAR.")
    s = requests.Session()
    s.headers.update({"User-Agent": settings.sec_user_agent, "Accept-Encoding": "gzip, deflate"})
    return s


def cik_for(ticker: str) -> int:
    if not _cik_cache:
        data = _session().get(TICKERS_URL, timeout=20).json()
        for row in data.values():
            _cik_cache[row["ticker"].upper()] = int(row["cik_str"])
    return _cik_cache[ticker.upper()]


def html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</tr>|</li>|</h\d>", "\n", raw)
    raw = re.sub(r"<[^>]+>", " ", raw)
    text = html.unescape(raw).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text)


def extract_risk_factors(text: str) -> str:
    """Return the body of Item 1A, skipping the table-of-contents occurrence."""
    starts = [m.start() for m in re.finditer(r"(?i)item\s*1a\.?\s*[\-–—:]?\s*risk\s+factors", text)]
    if not starts:
        return ""
    best = ""
    for st in starts:
        end_m = re.search(r"(?i)item\s*1b\.?|item\s*1c\.?\s*[\-–—:]?\s*cybersecurity|item\s*2\.?\s*[\-–—:]?\s*properties",
                          text[st + 50:])
        chunk = text[st: st + 50 + end_m.start()] if end_m else text[st: st + 200_000]
        if len(chunk) > len(best):
            best = chunk
    return best


def chunk_paragraphs(section: str, min_chars: int = 220, max_chars: int = 1800) -> list[str]:
    """Group paragraphs into retrieval-sized chunks; risk factors are usually one bold header + body."""
    paras = [p.strip() for p in section.split("\n\n") if len(p.strip()) > 40]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) > max_chars and len(buf) >= min_chars:
            chunks.append(buf.strip())
            buf = ""
        buf += p + "\n"
    if len(buf.strip()) >= min_chars:
        chunks.append(buf.strip())
    return chunks


def latest_10k_risk_factors(ticker: str, n_filings: int = 2) -> list[dict[str, Any]]:
    """Return [{fiscal_year, filed, url, chunks:[...]}, ...] newest first."""
    s = _session()
    cik = cik_for(ticker)
    sub = s.get(SUBMISSIONS_URL.format(cik=cik), timeout=20).json()
    recent = sub["filings"]["recent"]
    out = []
    for form, acc, doc, filed, period in zip(recent["form"], recent["accessionNumber"], recent["primaryDocument"],
                                            recent["filingDate"], recent["reportDate"]):
        if form != "10-K":
            continue
        url = DOC_URL.format(cik=cik, acc=acc.replace("-", ""), doc=doc)
        text = html_to_text(s.get(url, timeout=60).text)
        section = extract_risk_factors(text)
        out.append({"ticker": ticker.upper(), "filed": filed, "period": period,
                    "fiscal_year": period[:4], "url": url, "chunks": chunk_paragraphs(section)})
        if len(out) >= n_filings:
            break
    return out
