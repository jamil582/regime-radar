"""Runtime configuration, read from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader so the project has no hard dependency on python-dotenv."""
    if not path.exists():
        return
    raw = path.read_bytes()
    # Windows editors/PowerShell often write UTF-16 or UTF-8 with a BOM; handle all of them.
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8-sig", errors="replace")
    for line in text.splitlines():
        line = line.strip().lstrip("\ufeff")
        if line.lower().startswith("export "):
            line = line[7:]
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if value and not os.environ.get(key):  # also override an empty pre-existing variable
            os.environ[key] = value


_load_dotenv(ROOT / ".env")


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    api_key: str = field(default_factory=lambda: os.getenv("COHERE_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("COHERE_BASE_URL", "https://api.cohere.com"))

    # Model routing. Embed 5 Fast and Pro share one embedding space, so the high-volume
    # news stream goes through Fast while long-form SEC filings get Pro — and a single
    # query vector can search both indexes.
    embed_fast: str = field(default_factory=lambda: os.getenv("EMBED_MODEL_FAST", "embed-v5.0-fast"))
    embed_pro: str = field(default_factory=lambda: os.getenv("EMBED_MODEL_PRO", "embed-v5.0-pro"))
    chat_model: str = field(default_factory=lambda: os.getenv("CHAT_MODEL", "command-a-plus-05-2026"))

    # Matryoshka: we store full 2048-d vectors and use a truncated prefix for a cheap first pass.
    embed_dim: int = field(default_factory=lambda: int(os.getenv("EMBED_DIM", "2048")))
    coarse_dim: int = field(default_factory=lambda: int(os.getenv("COARSE_DIM", "256")))

    watchlist: list[str] = field(
        default_factory=lambda: [
            t.strip().upper()
            for t in os.getenv("WATCHLIST", "NVDA,MU,AVGO,XOM,GLD,TLT,NKE,ACN,SPY").split(",")
            if t.strip()
        ]
    )
    # Operating companies whose 10-K risk factors get indexed (ETFs don't file 10-Ks).
    filings_tickers: list[str] = field(
        default_factory=lambda: [
            t.strip().upper() for t in os.getenv("FILINGS_TICKERS", "NVDA,MU,XOM,NKE").split(",") if t.strip()
        ]
    )
    # SEC asks automated clients to identify themselves: "Company Name contact@example.com".
    sec_user_agent: str = field(default_factory=lambda: os.getenv("SEC_USER_AGENT", ""))

    demo: bool = field(default_factory=lambda: _bool("RADAR_DEMO"))
    host: str = field(default_factory=lambda: os.getenv("RADAR_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("RADAR_PORT", "8787")))

    @property
    def offline(self) -> bool:
        """Run fully offline (mock models + bundled snapshot) when demo is on or no key is set."""
        return self.demo or not self.api_key


settings = Settings()
