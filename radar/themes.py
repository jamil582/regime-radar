"""The 2026 macro regimes the radar tracks.

Each theme is described in natural language rather than by keyword lists. The description is
embedded once (input_type="search_query") and every headline/filing chunk is scored against it,
so a headline like "Refiners scramble as Gulf tanker insurance doubles" lands in the oil-shock
theme without ever mentioning "oil" or "Iran".
"""
from __future__ import annotations

THEMES: list[dict[str, str]] = [
    {
        "id": "rates_up",
        "name": "Higher-for-longer, again",
        "slot": 1,
        "description": (
            "The Federal Reserve is hiking interest rates again; Treasury yields at multi-decade highs; "
            "10-year yield above 5%; bond market selloff; rising mortgage rates; hawkish FOMC dot plot; "
            "pressure on rate-sensitive assets like REITs, utilities and long-duration growth stocks."
        ),
    },
    {
        "id": "oil_shock",
        "name": "Middle East oil shock",
        "slot": 2,
        "description": (
            "Geopolitical conflict in the Middle East disrupting energy supply; Brent crude above $100; "
            "US-Iran tensions, troop deployments, shipping and tanker risk in the Strait of Hormuz; "
            "energy stocks rallying; fuel costs feeding inflation."
        ),
    },
    {
        "id": "ai_capex",
        "name": "AI infrastructure super-cycle",
        "slot": 3,
        "description": (
            "Massive spending on AI data centers, GPUs and accelerators; hyperscaler capex; mega funding "
            "rounds and chip-leasing financing for AI labs; AI company IPOs; power demand from data centers."
        ),
    },
    {
        "id": "memory",
        "name": "Memory & semis pricing power",
        "slot": 4,
        "description": (
            "DRAM, NAND and high-bandwidth memory (HBM) prices rising; memory shortage; long-term supply "
            "agreements; semiconductor earnings beats and raised guidance; chip supply chain constraints."
        ),
    },
    {
        "id": "hard_assets",
        "name": "Gold & hard-asset bid",
        "slot": 5,
        "description": (
            "Gold at record highs above $4,000 an ounce; central bank gold buying; investors hedging "
            "inflation, debt and geopolitical risk with commodities and real assets."
        ),
    },
    {
        "id": "consumer_squeeze",
        "name": "Consumer squeeze",
        "slot": 6,
        "description": (
            "Slowing consumer spending; retailers and apparel brands missing revenue estimates; layoffs "
            "and job cuts; low-hire low-fire labor market; weak payroll growth; tariffs raising prices."
        ),
    },
    {
        "id": "strong_dollar",
        "name": "King dollar / FX stress",
        "slot": 7,
        "description": (
            "US dollar strengthening; euro, yen and emerging-market currencies weakening; European "
            "political uncertainty; capital flowing into US assets; FX headwinds for multinationals."
        ),
    },
]

THEME_BY_ID = {t["id"]: t for t in THEMES}
