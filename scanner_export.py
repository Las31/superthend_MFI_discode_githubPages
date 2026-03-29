"""
scanner_export.py
=================
把掃描結果輸出成 data.json，供 GitHub Pages 的 index.html 讀取。
在本機或 GitHub Actions 裡執行都可以。

用法：
  python scanner_export.py
  python scanner_export.py --atr 10 --mult 3 --mfi 14 --days 3

輸出：
  data.json   （放在 repo 根目錄，Pages 直接讀取）
"""

import sys, os, json, datetime, argparse, warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--atr",    type=int,   default=10)
parser.add_argument("--mult",   type=float, default=3.0)
parser.add_argument("--mfi",    type=int,   default=14)
parser.add_argument("--days",   type=int,   default=3)
parser.add_argument("--period", type=str,   default="6mo")
args = parser.parse_args()

# ── 把同資料夾的模組加入路徑 ─────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

missing = []
for pkg in ["finvizfinance", "yfinance", "pandas"]:
    try: __import__(pkg.replace("-","_"))
    except ImportError: missing.append(pkg)
if missing:
    print(f"[ERROR] pip install {' '.join(missing)}")
    sys.exit(1)

import pandas as pd
import yfinance as yf

# ── 覆寫全域參數再 import scanner 函式 ───────────────────────
import supertrend_scanner as sc
sc.ATR_PERIOD   = args.atr
sc.BASE_MULT    = args.mult
sc.MFI_PERIOD   = args.mfi
sc.FLIP_DAYS    = args.days
sc.DATA_PERIOD  = args.period

from supertrend_scanner import (
    get_nasdaq100_tickers,
    scan_flip_bullish,
    compute_supertrend_dynamic,
    backtest_weight,
    WEIGHT_CANDIDATES,
)

# ── 掃描 ─────────────────────────────────────────────────────
print("[scanner_export] Starting scan...")
tickers   = get_nasdaq100_tickers()
result_df = scan_flip_bullish(tickers)

# ── 每支股票補上近 3 個月圖表資料 ────────────────────────────
def get_chart_data(ticker: str) -> dict:
    try:
        raw = yf.download(ticker, period="3mo", interval="1d",
                          progress=False, auto_adjust=True)
        if raw.empty:
            return {}
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.dropna()

        best_w = backtest_weight(raw, args.atr, args.mult, args.mfi, WEIGHT_CANDIDATES)
        df = compute_supertrend_dynamic(raw, args.atr, args.mult, args.mfi, best_w)

        def clean(series):
            return [round(float(v), 2) if pd.notna(v) else None for v in series]

        return {
            "dates"     : [str(d)[:10] for d in df.index],
            "close"     : clean(df["Close"]),
            "supertrend": clean(df["SuperTrend"]),
            "mfi"       : [round(float(v), 1) if pd.notna(v) else None for v in df["MFI"]],
        }
    except Exception as e:
        print(f"  [chart] {ticker} failed: {e}")
        return {}

rows = []
if not result_df.empty:
    total = len(result_df)
    for i, (_, row) in enumerate(result_df.iterrows(), 1):
        ticker = row["Ticker"]
        print(f"  [chart {i}/{total}] {ticker}")
        chart = get_chart_data(ticker)

        rows.append({
            "ticker"      : ticker,
            "flip_date"   : str(row["Flip Date"])[:10],
            "flip_price"  : round(float(row["Flip Price"]), 2),
            "last_price"  : round(float(row["Last Price"]), 2),
            "change_pct"  : round(float(row["Change %"]), 2),
            "mfi"         : round(float(row["MFI"]), 1),
            "mfi_grade"   : row["MFI Grade"],
            "dyn_mult"    : round(float(row["Dyn Mult"]), 3),
            "best_weight" : float(row["Best Weight"]),
            "supertrend"  : round(float(row["SuperTrend"]), 2),
            "atr"         : round(float(row["ATR"]), 2),
            "chart"       : chart,
        })

# ── 組裝最終 JSON ─────────────────────────────────────────────
output = {
    "meta": {
        "scan_time"  : datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "atr"        : args.atr,
        "mult"       : args.mult,
        "mfi_period" : args.mfi,
        "flip_days"  : args.days,
        "total"      : len(rows),
        "strong"     : sum(1 for r in rows if r["mfi_grade"] == "Strong"),
        "normal"     : sum(1 for r in rows if r["mfi_grade"] == "Normal"),
        "weak"       : sum(1 for r in rows if r["mfi_grade"] == "Weak"),
    },
    "rows": rows,
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, separators=(",", ":"))

print(f"\n[scanner_export] Done — {len(rows)} stocks -> {out_path}")
