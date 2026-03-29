"""
get_tickers.py  -  NASDAQ 100 stock list (Finviz hardened)
==========================================================
Improvements:
  1. Retry + Exponential Backoff with Jitter
  2. Session reuse + rotating User-Agent
  3. Hard timeout per request (20s)
  4. Result validation (< 90 tickers = retry)
  5. Hardcoded fallback (used only if ALL retries fail)
"""

import time
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── Config ────────────────────────────────────────────────────
TIMEOUT       = 20
MAX_RETRIES   = 5
BASE_BACKOFF  = 5        # seconds; doubles each attempt: 5, 10, 20, 40 ...
JITTER_RANGE  = (1, 4)   # random extra seconds added to each wait
MIN_TICKERS   = 90       # fewer than this = treat as failed response

USER_AGENTS = [
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
     "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"),
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
     "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0"),
    ("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 "
     "(KHTML, like Gecko) Version/17.3 Safari/605.1.15"),
    ("Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0"),
]

_HARDCODED = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "NFLX","AMD","ADBE","QCOM","PEP","AMAT","CSCO","TXN","INTU","AMGN",
    "CMCSA","HON","INTC","VRTX","BKNG","PANW","LRCX","SBUX","KLAC","MELI",
    "REGN","ADI","MDLZ","SNPS","CDNS","ASML","MU","CTAS","ORLY","ABNB",
    "MAR","FTNT","MRVL","CEG","PCAR","PYPL","CPRT","ROST","ADP","KDP",
    "DXCM","BIIB","IDXX","CHTR","ILMN","ODFL","VRSK","ANSS","FAST","FANG",
    "EXC","XEL","CTSH","SIRI","MNST","PAYX","GEHC","ON","TEAM","ZS",
    "CRWD","DDOG","SNOW","OKTA","TTD","DOCU","ZM","UBER","DASH","RBLX",
    "COIN","HOOD","SOFI","AFRM","RIVN","PDD","BIDU","JD","NTES","BABA",
    "SE","GRAB","LAZR","SMCI","ARM","PLTR","APP","MSTR","COIN","HOOD",
]


# ── Build a hardened requests Session ────────────────────────
def _build_session():
    session = requests.Session()

    # urllib3 transport-level retry (connection errors, 5xx)
    adapter = HTTPAdapter(max_retries=Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    ))
    session.mount("https://", adapter)
    session.mount("http://",  adapter)

    session.headers.update({
        "User-Agent"     : random.choice(USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept"         : "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection"     : "keep-alive",
    })
    return session


# ── Monkeypatch requests.get so finvizfinance uses our session ─
def _patch_requests(session):
    import requests as req_module
    original = req_module.get

    def patched(url, **kwargs):
        kwargs.setdefault("timeout", TIMEOUT)
        hdrs = dict(session.headers)
        hdrs.update(kwargs.pop("headers", {}))
        return session.get(url, headers=hdrs, **kwargs)

    req_module.get = patched
    return original


def _restore_requests(original):
    import requests as req_module
    req_module.get = original


# ── Single fetch attempt ──────────────────────────────────────
def _fetch_once(session):
    from finvizfinance.screener.overview import Overview
    original = _patch_requests(session)
    try:
        screen = Overview()
        screen.set_filter(filters_dict={
            "Exchange": "NASDAQ",
            "Index"   : "NASDAQ 100",
        })
        df      = screen.screener_view()
        tickers = df["Ticker"].dropna().str.strip().tolist()
    finally:
        _restore_requests(original)

    if len(tickers) < MIN_TICKERS:
        raise ValueError(
            f"Only {len(tickers)} tickers (expected >= {MIN_TICKERS})"
        )
    return tickers


# ── Retry loop with exponential backoff + jitter ─────────────
def _fetch_with_retry():
    """
    Wait schedule (seconds):
      attempt 1 fail -> 5  + jitter
      attempt 2 fail -> 10 + jitter
      attempt 3 fail -> 20 + jitter
      attempt 4 fail -> 40 + jitter
      attempt 5 fail -> raise
    """
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        session = _build_session()   # fresh session + new UA each time
        try:
            print(f"  [Finviz] Attempt {attempt}/{MAX_RETRIES} ...",
                  end=" ", flush=True)
            tickers = _fetch_once(session)
            print(f"OK ({len(tickers)} tickers)")
            return tickers

        except Exception as e:
            last_err = e
            print(f"FAILED: {e}")
            if attempt < MAX_RETRIES:
                wait = BASE_BACKOFF * (2 ** (attempt - 1)) + random.uniform(*JITTER_RANGE)
                print(f"  [Finviz] Retrying in {wait:.1f}s ...")
                time.sleep(wait)

    raise RuntimeError(
        f"All {MAX_RETRIES} Finviz attempts failed. Last: {last_err}"
    )


# ══════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════
def get_nasdaq100_tickers(verbose=True):
    """
    Returns NASDAQ 100 constituent tickers as list[str].
    Tries Finviz up to MAX_RETRIES times; falls back to
    hardcoded list only if every attempt fails.
    """
    if verbose:
        print(f"[Tickers] Fetching NASDAQ 100 from Finviz "
              f"(max {MAX_RETRIES} retries, timeout={TIMEOUT}s) ...")
    try:
        tickers = _fetch_with_retry()
        tickers = list(dict.fromkeys(
            t.strip().upper() for t in tickers
            if isinstance(t, str) and t.strip()
        ))
        if verbose:
            print(f"[Tickers] Done: {len(tickers)} tickers")
        return tickers

    except Exception as e:
        if verbose:
            print(f"[Tickers] Finviz completely failed: {e}")
            print(f"[Tickers] Using hardcoded fallback ({len(_HARDCODED)} tickers)")
        return list(_HARDCODED)


if __name__ == "__main__":
    t = get_nasdaq100_tickers()
    print(f"\n{len(t)} tickers:")
    for i in range(0, len(t), 10):
        print("  " + "  ".join(f"{x:<6}" for x in t[i:i+10]))