"""Offline test suite — runs against the mock client, no network or API key required.

    RADAR_DEMO=1 python -m unittest discover -s tests -v
"""
import json
import os
import unittest

os.environ["RADAR_DEMO"] = "1"

import numpy as np  # noqa: E402

from radar import agent  # noqa: E402
from radar.config import settings  # noqa: E402
from radar.grounding import normalise_citations  # noqa: E402
from radar.pipeline import Radar  # noqa: E402
from radar.risk_drift import compute_drift  # noqa: E402
from radar.sources.news import parse_rss  # noqa: E402
from radar.sources.sec import chunk_paragraphs, extract_risk_factors  # noqa: E402
from radar.vectorstore import VectorIndex, matryoshka_prefix, quantize_int8  # noqa: E402

settings.demo = True


def _rand(n, d=2048, seed=0):
    v = np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class VectorStoreTests(unittest.TestCase):
    def test_int8_roundtrip_preserves_cosine(self):
        v = _rand(50)
        q, s = quantize_int8(v)
        back = q.astype(np.float32) * s
        back /= np.linalg.norm(back, axis=1, keepdims=True)
        self.assertGreater(float(np.min(np.sum(v * back, axis=1))), 0.999)

    def test_prefix_is_normalised(self):
        p = matryoshka_prefix(_rand(5), 256)
        self.assertEqual(p.shape, (5, 256))
        np.testing.assert_allclose(np.linalg.norm(p, axis=1), 1.0, rtol=1e-5)

    def test_two_stage_finds_exact_match(self):
        v = _rand(300)
        idx = VectorIndex("t")
        idx.add([{"id": str(i)} for i in range(300)], v)
        for i in (0, 123, 299):
            self.assertEqual(idx.search(v[i], k=1)[0].doc["id"], str(i))

    def test_filter_and_persistence(self, tmp=None):
        import tempfile
        from pathlib import Path

        v = _rand(10)
        idx = VectorIndex("t")
        idx.add([{"id": str(i), "tickers": ["X"] if i % 2 else []} for i in range(10)], v)
        hits = idx.search(v[0], k=10, where=lambda d: "X" in d["tickers"])
        self.assertTrue(all(int(h.doc["id"]) % 2 for h in hits))
        with tempfile.TemporaryDirectory() as d:
            idx.save(Path(d) / "i")
            idx2 = VectorIndex("t")
            self.assertTrue(idx2.load(Path(d) / "i"))
            self.assertEqual(idx2.search(v[3], k=1)[0].doc["id"], "3")


class SourceParsingTests(unittest.TestCase):
    def test_parse_rss(self):
        xml = """<rss><channel><item><title>Fed hikes &amp; yields jump</title><link>https://x/1</link>
        <pubDate>Wed, 16 Sep 2026 18:00:00 GMT</pubDate><description>&lt;b&gt;Body&lt;/b&gt;</description></item></channel></rss>"""
        items = parse_rss(xml, source="test", ticker="TLT")
        self.assertEqual(items[0]["title"], "Fed hikes & yields jump")
        self.assertEqual(items[0]["summary"], "Body")
        self.assertTrue(items[0]["published"].startswith("2026-09-16"))

    def test_extract_risk_factors_skips_toc(self):
        text = ("Item 1A. Risk Factors 12\nItem 1B. Unresolved 30\n" + "filler " * 50 +
                "\nItem 1A. Risk Factors\n\n" + ("Our business faces risk from rates and energy. " * 10 + "\n\n") * 4 +
                "Item 1B. Unresolved Staff Comments")
        sec = extract_risk_factors(text)
        self.assertIn("Our business faces risk", sec)
        self.assertGreaterEqual(len(chunk_paragraphs(sec)), 1)


class CitationTests(unittest.TestCase):
    def test_normalise(self):
        body = {"message": {"citations": [{"start": 0, "end": 5, "text": "Hello",
                                           "sources": [{"type": "document", "id": "a"},
                                                       {"type": "tool", "id": "b"}]}]}}
        self.assertEqual(normalise_citations(body)[0]["sources"], ["a", "b"])


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.radar = Radar()
        cls.stats = cls.radar.refresh()

    def test_indexes_built(self):
        self.assertGreater(self.stats["news"], 20)
        self.assertGreater(self.stats["filings"], 10)
        self.assertEqual(len(self.radar.enrichment), self.stats["news"])

    def test_themes_cover_headlines(self):
        themes = self.radar.theme_summary()
        self.assertEqual(len(themes), 7)
        self.assertGreater(sum(t["count"] for t in themes), 15)

    def test_semantic_search_crosses_indexes(self):
        hits = self.radar.search("energy costs Middle East conflict", k=6)
        kinds = {h.doc["kind"] for h in hits}
        self.assertIn("filing", kinds)
        self.assertIn("news", kinds)

    def test_risk_drift_finds_new_risks(self):
        d = self.radar.risk_drift("NWND")
        titles = " ".join(a["title"] for a in d["added"])
        self.assertTrue("Vendor financing" in titles or "Energy costs" in titles, titles)
        self.assertIn("summary", d)

    def test_agent_loop_and_citations(self):
        out = agent.ask(self.radar.client, self.radar, "What new risks did NWND add this year?")
        self.assertGreaterEqual(len(out["trace"]), 2)
        self.assertTrue(out["citations"])
        for c in out["citations"]:
            for s in c["sources"]:
                self.assertIn(s, out["sources"])

    def test_briefing_is_cited(self):
        b = self.radar.get_briefing()
        self.assertTrue(b["citations"])
        self.assertTrue(set(s for c in b["citations"] for s in c["sources"]) <= set(b["sources"]))

    def test_snapshot_is_json_serialisable(self):
        json.dumps(self.radar.snapshot(), default=float)


if __name__ == "__main__":
    unittest.main()
