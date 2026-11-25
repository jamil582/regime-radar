"""Retrieval eval: how much quality do Matryoshka truncation, quantisation and Fast/Pro mixing cost?

Uses the bundled snapshot plus hand-labelled queries. Run with a key for real numbers:

    COHERE_API_KEY=... python -m scripts.eval_retrieval
    RADAR_DEMO=1 python -m scripts.eval_retrieval   # offline smoke run (mock embeddings)

Configurations compared (recall@5 and MRR):
  A  fast→fast, float32 2048-d brute force      (quality ceiling for Fast)
  B  fast→fast, 256-bit binary shortlist → int8 rescoring   (what the dashboard runs)
  C  fast→fast, 256-d Matryoshka prefix only
  D  fast query → pro documents (shared embedding space)
  E  pro→pro, float32 2048-d
"""
from __future__ import annotations

import json
import time

import numpy as np

from radar.cohere_api import get_client
from radar.config import DATA_DIR, settings
from radar.vectorstore import VectorIndex, matryoshka_prefix

# query -> substrings identifying relevant headlines in data/sample/headlines.json
QUERIES = {
    "central bank tightening monetary policy": ["Fed raises", "Dot plot", "Warsh"],
    "long-dated government bond selloff": ["10-year Treasury yield briefly", "Futures rise as Fed"],
    "energy supply disruption from conflict": ["Brent jumps", "Diesel tops", "WTI crude drops"],
    "who is funding frontier AI labs": ["Broadcom agrees", "Nvidia pays final", "Anthropic lines up"],
    "memory chip pricing power": ["Micron posts"],
    "safe-haven precious metals demand": ["Gold climbs", "Gold edges"],
    "weak consumer and job cuts at brands": ["Nike fiscal Q1", "Nike slides", "Spending outpaces"],
    "currency markets and the greenback": ["Dollar index rises"],
    "real estate hurt by higher rates": ["REITs and net-lease"],
    "inflation data releases": ["August CPI", "August PCE"],
    "defense spending on missile interceptors": ["RTX wins"],
    "long-term natural gas export contracts": ["Venture Global"],
}


def metrics(rank_lists: list[list[int]], gold: list[set[int]], k: int = 5) -> dict[str, float]:
    rec, mrr = [], []
    for ranks, g in zip(rank_lists, gold):
        rec.append(len(set(ranks[:k]) & g) / len(g))
        rr = next((1 / (i + 1) for i, r in enumerate(ranks) if r in g), 0.0)
        mrr.append(rr)
    return {"recall@5": round(float(np.mean(rec)), 3), "mrr": round(float(np.mean(mrr)), 3)}


def main() -> None:
    client = get_client()
    items = json.loads((DATA_DIR / "sample" / "headlines.json").read_text())["items"]
    texts = [f"{h['title']}. {h['summary']}" for h in items]
    gold = [{i for i, h in enumerate(items) if any(s in h["title"] for s in subs)} for subs in QUERIES.values()]
    docs = [{"id": str(i)} for i in range(len(items))]

    t = time.time()
    d_fast = client.embed(texts, model=settings.embed_fast, input_type="search_document", feature="eval")
    d_pro = client.embed(texts, model=settings.embed_pro, input_type="search_document", feature="eval")
    q_fast = client.embed(list(QUERIES), model=settings.embed_fast, input_type="search_query", feature="eval")
    q_pro = client.embed(list(QUERIES), model=settings.embed_pro, input_type="search_query", feature="eval")
    print(f"embedded {len(texts)} docs x2 models + {len(QUERIES)} queries x2 in {time.time() - t:.1f}s\n")

    def brute(q, d):
        return [list(np.argsort(-(d @ qv))) for qv in q]

    idx = VectorIndex("fast", settings.coarse_dim)
    idx.add(docs, d_fast)
    two_stage = [[int(h.doc["id"]) for h in idx.search(qv, k=10, shortlist=16)] for qv in q_fast]
    pre_q, pre_d = matryoshka_prefix(q_fast, settings.coarse_dim), matryoshka_prefix(d_fast, settings.coarse_dim)

    rows = {
        "A fast float32 2048d": brute(q_fast, d_fast),
        f"B fast binary{settings.coarse_dim}->int8 (dashboard)": two_stage,
        f"C fast prefix {settings.coarse_dim}d only": brute(pre_q, pre_d),
        "D fast query -> pro docs": brute(q_fast, d_pro),
        "E pro float32 2048d": brute(q_pro, d_pro),
    }
    print(f"{'configuration':42} {'recall@5':>9} {'MRR':>6}")
    for name, ranks in rows.items():
        m = metrics(ranks, gold)
        print(f"{name:42} {m['recall@5']:>9.3f} {m['mrr']:>6.3f}")
    n, dim = d_fast.shape
    print(f"\nstorage per doc: float32 {dim * 4} B · int8 {dim} B · binary prefix {settings.coarse_dim // 8} B")
    if settings.offline:
        print("\n(offline mock embeddings - numbers are a smoke test only; set COHERE_API_KEY for real results)")


if __name__ == "__main__":
    main()
