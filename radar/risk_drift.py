"""10-K risk-factor drift: what did management start (or stop) worrying about this year?

Both years' Item 1A chunks are embedded with Embed 5 *Pro* (long, dense legal prose). For each
chunk in the new filing we take its best match in the prior filing; chunks with no close
counterpart are new risks, and prior-year chunks with no counterpart were dropped.
Command A+ then explains the drift, citing the exact paragraphs.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .cohere_api import message_text
from .grounding import as_document, normalise_citations


def compute_drift(new_vecs: np.ndarray, old_vecs: np.ndarray, new_chunks: list[dict], old_chunks: list[dict],
                  novelty_threshold: float = 0.80, top: int = 5) -> dict[str, Any]:
    sims = new_vecs @ old_vecs.T                    # (new, old) cosine matrix
    best_new = sims.max(axis=1)                      # how well each new chunk is covered last year
    best_old = sims.max(axis=0)                      # how well each old chunk survives this year
    new_order = np.argsort(best_new)
    old_order = np.argsort(best_old)
    added = [dict(new_chunks[i], novelty=round(1 - float(best_new[i]), 3),
                  closest_prior=old_chunks[int(sims[i].argmax())]["title"])
             for i in new_order[:top] if best_new[i] < novelty_threshold or i in new_order[:2]]
    dropped = [dict(old_chunks[j], novelty=round(1 - float(best_old[j]), 3))
               for j in old_order[:3] if best_old[j] < novelty_threshold]
    return {
        "stability": round(float(np.median(best_new)), 3),
        "n_new_chunks": len(new_chunks),
        "n_old_chunks": len(old_chunks),
        "added": added,
        "dropped": dropped,
        "similarity_histogram": np.histogram(best_new, bins=10, range=(0.4, 1.0))[0].tolist(),
    }


def summarise_drift(client, ticker: str, drift: dict[str, Any]) -> dict[str, Any]:
    docs = [as_document(d) for d in drift["added"] + drift["dropped"]]
    if not docs:
        return {"markdown": "No material change in risk factors.", "citations": []}
    body = client.chat(
        [{"role": "user", "content": (
            f"These are the most novel risk-factor passages in {ticker}'s latest 10-K versus the prior year, "
            "plus passages that were dropped. In <=120 words, explain what management is newly worried about "
            "and how it connects to the 2026 macro backdrop (rates, oil, AI capex, consumer). Cite passages.")}],
        documents=docs,
        thinking_budget=512,
        temperature=0.2,
        feature="risk drift summary (RAG)",
    )
    return {"markdown": message_text(body), "citations": normalise_citations(body)}
