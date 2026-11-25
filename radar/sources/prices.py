"""Daily price history from Yahoo Finance's public chart endpoint (no key)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range={range}&interval=1d"
UA = "Mozilla/5.0 (compatible; RegimeRadar/1.0)"


def fetch_history(ticker: str, rng: str = "6mo", timeout: int = 15) -> list[dict[str, Any]]:
    import requests

    r = requests.get(CHART_URL.format(ticker=ticker, range=rng), headers={"User-Agent": UA}, timeout=timeout)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    closes = res["indicators"]["quote"][0]["close"]
    rows = []
    for ts, c in zip(res["timestamp"], closes):
        if c is None:
            continue
        rows.append({"date": datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat(), "close": round(c, 4)})
    return rows


def price_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summary stats the agent tool and KPI tiles use."""
    if len(rows) < 2:
        return {}
    c = np.array([r["close"] for r in rows], dtype=float)
    rets = np.diff(np.log(c))

    def chg(n: int) -> float | None:
        return None if len(c) <= n else round(float(c[-1] / c[-1 - n] - 1) * 100, 2)

    peak = np.maximum.accumulate(c)
    return {
        "last": round(float(c[-1]), 2),
        "as_of": rows[-1]["date"],
        "chg_1d_pct": chg(1),
        "chg_5d_pct": chg(5),
        "chg_1m_pct": chg(21),
        "chg_3m_pct": chg(63),
        "vol_ann_pct": round(float(rets[-63:].std() * np.sqrt(252) * 100), 1),
        "max_drawdown_pct": round(float((c / peak - 1).min() * 100), 1),
    }


def correlation(a: list[dict[str, Any]], b: list[dict[str, Any]], window: int = 63) -> float | None:
    da = {r["date"]: r["close"] for r in a}
    db = {r["date"]: r["close"] for r in b}
    dates = sorted(set(da) & set(db))[-window - 1:]
    if len(dates) < 10:
        return None
    x = np.diff(np.log([da[d] for d in dates]))
    y = np.diff(np.log([db[d] for d in dates]))
    return round(float(np.corrcoef(x, y)[0, 1]), 2)
