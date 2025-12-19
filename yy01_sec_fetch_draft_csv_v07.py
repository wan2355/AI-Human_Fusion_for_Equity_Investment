#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## uasge
#{{{
"""
python3 yy01_sec_fetch_draft_csv_v07.py \
  -t LITE MU PLTR NVDA \
  -y 8 \
  -o sec_out_3 \
  -u "your_project/0.1 (you@example.com)" \
  --tv-csv tv_manual.csv \
  --tv-unit B \
  --tv-mode auto \
  --tv-threshold 0.25 \
  --add-market

---------------------

Semi-automatic fundamentals collector (public source, no paid API).
- Input: tickers (US listed), one per line, or via --tickers AAPL MSFT ...
- Fetches SEC EDGAR XBRL "companyfacts" JSON per company (requires internet).
- Extracts annual (FY) series for:
    Revenue  -> us-gaap:Revenues
    EBIT     -> us-gaap:OperatingIncomeLoss (proxy for EBIT)
    CFO      -> us-gaap:NetCashProvidedByUsedInOperatingActivities
    CapEx    -> us-gaap:PaymentsToAcquirePropertyPlantAndEquipment (positive outflow)
    SBC      -> us-gaap:ShareBasedCompensation
    FCF      -> CFO - CapEx
- Output: draft_input.csv compatible with financial_eval_04k.py.
Notes:
- Many companies have missing tags / restatements; this script aims for "good enough draft".
- You will still want to spot-check and manually fix the CSV for edge cases.

Usage examples:
  python3 sec_fetch_draft_csv.py --tickers NVDA MU PLTR LITE --years 8 --outdir sec_out
  python3 sec_fetch_draft_csv.py --ticker-file tickers.txt --years 10 --outdir sec_out

==============================================================
PROGRAM 1 : yy01_sec_fetch_draft_csv_v03.py
================================================================================
目的
  SEC（companyfacts）から財務データを取得し、下流分析用の下書きCSVを作成する。
  ※ 金額単位はすべて USD（1ドル単位）

引数一覧
  -t  --tickers        ティッカーを直接指定（空白区切り）
  -f  --ticker-file    ティッカー一覧ファイル（1行1銘柄）
  -y  --years          取得する直近年数（default: 8）
  -o  --outdir         出力ディレクトリ（default: sec_out）
  -u  --user-agent     SEC用User-Agent（推奨：連絡先入り）o

--add-market : yfinanceで最新 Price/MarketCap を追加（取れる範囲で）

--market-out : 市場データCSV名（デフォルト market_snapshot.csv）

--tv-csv : TradingView/手動CSV（任意）

--tv-unit B|USD : tv-csvのRevenue単位（デフォルトB）

--tv-mode auto|always|never : Revenue採用ルール

auto : 基本SEC、乖離が大きい年だけTVに差替え（+SEC欠損はTV）

always: TVがあればTVを採用

never : SEC固定

--tv-threshold : auto時の乖離閾値（デフォルト0.25）

"""
# }}}

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests

# Optional market data (price/market cap)
try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None

SEC_HEADERS = {
    # IMPORTANT: SEC requests a descriptive User-Agent with contact info.
    # NOTE: Do NOT set the "Host" header manually.
    # If you set Host=data.sec.gov while requesting https://www.sec.gov/..., some environments will
    # receive a 404 (because the Host header and URL host disagree).
    "User-Agent": "kana-fundamental-playground/0.1 (contact: you@example.com)",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

# Optional: let the user override the User-Agent from CLI without editing code.
# Example:
#   python3 sec_fetch_draft_csv.py --user-agent "YourProject/0.1 (email@example.com)" ...

TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
TICKER_CIK_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"

# XBRL tags we try (first match wins)
TAG_REVENUE = [
    ("us-gaap", "Revenues"),
    ("us-gaap", "SalesRevenueNet"),
    ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
    ("us-gaap", "SalesRevenueGoodsNet"),
    ("us-gaap", "SalesRevenueServicesNet"),
    ("ifrs-full", "Revenue"),
]

TAG_EBIT = [
    ("us-gaap", "OperatingIncomeLoss"),
    ("us-gaap", "PretaxIncomeLoss"),
    ("us-gaap", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"),
    ("ifrs-full", "OperatingProfitLoss"),
]

TAG_CFO = [
    ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
    ("ifrs-full", "NetCashFlowsFromUsedInOperatingActivities"),
]

TAG_CAPEX = [
    ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
    ("us-gaap", "CapitalExpenditures"),
    ("us-gaap", "PaymentsToAcquireProductiveAssets"),
    ("ifrs-full", "PurchaseOfPropertyPlantAndEquipment"),
]

TAG_SBC = [
    ("us-gaap", "ShareBasedCompensation"),
    ("us-gaap", "ShareBasedCompensationArrangementByShareBasedPaymentAwardExpenseRecognized"),
    ("ifrs-full", "SharebasedPaymentArrangementBySharebasedPaymentAwardExpenseRecognized"),
]



def _sleep_polite():
    # SEC rate limits; keep it gentle.
    time.sleep(0.25)


def _download_json(url: str, cache_path: Path) -> dict:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    r = requests.get(url, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    _sleep_polite()
    return data


def _load_ticker_cik_map(cache_dir: Path) -> Dict[str, str]:
    """Load mapping TICKER -> CIK (no zero padding).

    Primary source: company_tickers.json
    Fallback: company_tickers_exchange.json

    Why fallback?
      - Some environments get a misleading 404 when request headers are malformed.
      - SEC occasionally changes or redirects endpoints.
    """

    def _parse_mapping(data: object) -> Dict[str, str]:
        m: Dict[str, str] = {}
        # company_tickers.json: dict keyed by integer strings
        if isinstance(data, dict):
            items = data.values()
        # company_tickers_exchange.json: sometimes list, sometimes dict with 'data'
        elif isinstance(data, list):
            items = data
        else:
            items = []

        for v in items:
            if not isinstance(v, dict):
                continue
            t = str(v.get("ticker", v.get("Ticker", ""))).upper().strip()
            cik = v.get("cik_str", v.get("cik", v.get("CIK", "")))
            cik = str(cik).strip()
            if t and cik:
                m[t] = cik
        return m

    # Try primary
    try:
        cache_path = cache_dir / "company_tickers.json"
        data = _download_json(TICKER_CIK_URL, cache_path)
        m = _parse_mapping(data)
        if m:
            return m
    except requests.HTTPError as e:
        print(f"[WARN] Failed to fetch company_tickers.json ({e}). Trying exchange file...")

    # Fallback
    cache_path2 = cache_dir / "company_tickers_exchange.json"
    data2 = _download_json(TICKER_CIK_EXCHANGE_URL, cache_path2)
    m2 = _parse_mapping(data2)
    if not m2:
        raise SystemExit(
            "Could not build ticker->CIK mapping from SEC. "
            "Tip: set --user-agent with a real contact, and check outbound access to sec.gov."
        )
    return m2


def _pick_fact_series(facts: dict, namespace: str, tag: str) -> Optional[List[dict]]:
    try:
        node = facts["facts"][namespace][tag]["units"]
    except Exception:
        return None
    # prefer USD
    for unit in ["USD", "usd"]:
        if unit in node:
            return node[unit]
    # fallback: first unit
    for _, arr in node.items():
        return arr
    return None


def _extract_annual(series: List[dict], *, allow_quarterly_fallback: bool = True) -> pd.DataFrame:
    """
    Convert companyfacts array into annual rows (Year, value).

    Priority:
      1) fp == "FY" (annual) if available
      2) annual forms (10-K / 20-F / 40-F) if available
      3) (optional) fallback: synthesize annual values from quarterly (Q1..Q4)
         - if filings look cumulative (Q4 >= Q3 >= Q2 >= Q1), use Q4 as FY
         - else, if all 4 quarters exist, sum Q1..Q4

    Notes:
      - This is a *best-effort* converter. SEC companyfacts sometimes lacks FY rows
        for certain tags/issuers, especially for newer tickers or when a tag is
        reported only in quarterly form.
    """
    if not series:
        return pd.DataFrame(columns=["Year", "value"])

    df0 = pd.DataFrame(series)

    # normalize columns we need
    for c in ["val", "fy", "fp", "form", "end", "start"]:
        if c not in df0.columns:
            df0[c] = None

    df0["val"] = pd.to_numeric(df0["val"], errors="coerce")
    df0 = df0.dropna(subset=["val"]).copy()
    if df0.empty:
        return pd.DataFrame(columns=["Year", "value"])

    df0["fp"] = df0["fp"].astype(str).str.upper()
    df0["form"] = df0["form"].astype(str).str.upper()
    df0["end_dt"] = pd.to_datetime(df0["end"], errors="coerce")

    def _derive_year(dfx: pd.DataFrame) -> pd.Series:
        if dfx["fy"].notna().any():
            return pd.to_numeric(dfx["fy"], errors="coerce")
        return pd.to_datetime(dfx["end"], errors="coerce").dt.year

    # -------- prefer annual rows
    df = df0.copy()

    df_fy = df[df["fp"] == "FY"].copy()
    if not df_fy.empty:
        df = df_fy

    annual_forms = {"10-K", "20-F", "40-F"}
    df_form = df[df["form"].isin(annual_forms)].copy()
    if not df_form.empty:
        df = df_form

    df["Year"] = _derive_year(df)
    df = df.dropna(subset=["Year"])
    if not df.empty:
        df["Year"] = df["Year"].astype(int)
        # in case of duplicates per year, keep the latest 'end'
        df = df.sort_values(["Year", "end_dt"])
        df = df.groupby("Year", as_index=False).tail(1)
        return df[["Year", "val"]].rename(columns={"val": "value"})

    # -------- fallback: build annual from quarterly
    if not allow_quarterly_fallback:
        return pd.DataFrame(columns=["Year", "value"])

    qdf = df0[df0["fp"].isin({"Q1", "Q2", "Q3", "Q4"})].copy()
    if qdf.empty:
        return pd.DataFrame(columns=["Year", "value"])

    qdf["Year"] = _derive_year(qdf)
    qdf = qdf.dropna(subset=["Year"])
    if qdf.empty:
        return pd.DataFrame(columns=["Year", "value"])
    qdf["Year"] = qdf["Year"].astype(int)

    out_rows = []
    for year, g in qdf.groupby("Year"):
        # pick the latest value per quarter
        q_latest = {}
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            gg = g[g["fp"] == q].sort_values("end_dt")
            if not gg.empty:
                q_latest[q] = float(gg.iloc[-1]["val"])

        if "Q4" in q_latest:
            # detect cumulative pattern (very common)
            seq = [q_latest.get(q) for q in ["Q1", "Q2", "Q3", "Q4"]]
            # keep only non-None up to Q4
            ok = True
            prev = None
            for v in seq:
                if v is None:
                    continue
                if prev is not None and v < prev:
                    ok = False
                    break
                prev = v
            if ok:
                out_rows.append({"Year": year, "value": q_latest["Q4"]})
                continue

        if all(q in q_latest for q in ["Q1", "Q2", "Q3", "Q4"]):
            out_rows.append({"Year": year, "value": q_latest["Q1"] + q_latest["Q2"] + q_latest["Q3"] + q_latest["Q4"]})

    if not out_rows:
        return pd.DataFrame(columns=["Year", "value"])

    out = pd.DataFrame(out_rows).sort_values("Year")
    return out


def _get_tag_value(facts: dict, tag_list: List[Tuple[str, str]]) -> pd.DataFrame:
    for ns, tag in tag_list:
        series = _pick_fact_series(facts, ns, tag)
        if series:
            return _extract_annual(series)
    return pd.DataFrame(columns=["Year", "value"])


def build_draft_for_ticker(ticker: str, cik: str, cache_dir: Path) -> pd.DataFrame:
    cik10 = str(cik).zfill(10)
    cache_path = cache_dir / f"companyfacts_{ticker}_{cik10}.json"
    facts = _download_json(COMPANYFACTS_URL.format(cik10=cik10), cache_path)

    company = facts.get("entityName", ticker)

    rev = _get_tag_value(facts, TAG_REVENUE).rename(columns={"value": "Revenue"})
    ebit = _get_tag_value(facts, TAG_EBIT).rename(columns={"value": "EBIT"})
    cfo = _get_tag_value(facts, TAG_CFO).rename(columns={"value": "CFO"})
    capex = _get_tag_value(facts, TAG_CAPEX).rename(columns={"value": "CapEx"})
    sbc = _get_tag_value(facts, TAG_SBC).rename(columns={"value": "SBC"})

    # merge on Year
    out = rev.merge(ebit, on="Year", how="outer")
    out = out.merge(cfo, on="Year", how="outer")
    out = out.merge(capex, on="Year", how="outer")
    out = out.merge(sbc, on="Year", how="outer")

    # CapEx in SEC is usually a positive cash outflow number. Keep positive here.
    # Compute FCF = CFO - CapEx
    capex_num = pd.to_numeric(out["CapEx"], errors="coerce")
    cfo_num = pd.to_numeric(out["CFO"], errors="coerce")
    # SEC cash flow tags are sometimes negative (cash outflow). Normalize CapEx to positive.
    capex_pos = capex_num.abs()
    out["CapEx"] = capex_pos
    out["FCF"] = cfo_num - capex_pos

    out.insert(0, "Company", company)
    out.insert(0, "Ticker", ticker)

    # Optional columns that financial_eval can accept (left blank for manual fill)
    for c in ["Price", "MktCap_B", "SharesOut_B"]:
        out[c] = pd.NA

    # order columns
    cols = ["Ticker", "Company", "Year", "Revenue", "EBIT", "FCF", "CapEx", "SBC", "Price", "MktCap_B", "SharesOut_B"]
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[cols].sort_values("Year")

    return out



def fetch_market_snapshot(tickers: List[str]) -> pd.DataFrame:
    """
    Fetch latest market snapshot using yfinance (best-effort).
    Output columns: Ticker, Date, Price, MarketCap
      - Price: latest close/last price (USD)
      - MarketCap: market cap (USD)
    """
    if yf is None:
        raise RuntimeError("yfinance is not available. Install: pip install yfinance")
    rows = []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            # Try fast_info first (lighter), fallback to info/history
            price = None
            mcap = None
            dt = None

            try:
                fi = getattr(tk, "fast_info", None)
                if fi:
                    price = fi.get("last_price") or fi.get("lastPrice") or fi.get("regularMarketPrice")
            except Exception:
                pass

            try:
                info = tk.info or {}
                mcap = info.get("marketCap")
                if price is None:
                    price = info.get("regularMarketPrice") or info.get("currentPrice")
            except Exception:
                info = {}

            if price is None:
                try:
                    h = tk.history(period="5d")
                    if h is not None and len(h) > 0:
                        last = h.iloc[-1]
                        price = float(last.get("Close"))
                        dt = str(last.name.date())
                except Exception:
                    pass

            if dt is None:
                dt = pd.Timestamp.utcnow().strftime("%Y-%m-%d")

            rows.append({
                "Ticker": t,
                "Date": dt,
                "Price": float(price) if price is not None else pd.NA,
                "MarketCap": float(mcap) if mcap is not None else pd.NA,
            })
        except Exception as e:
            rows.append({"Ticker": t, "Date": pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
                         "Price": pd.NA, "MarketCap": pd.NA, "Error": str(e)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('-t',"--tickers", nargs="*", default=[])
    #ap.add_argument('-t',"--ticker-file", default="yy00_my_tickers.txt", help="One ticker per line")
    ap.add_argument('-f',"--ticker-file", default=None, help="One ticker per line")
    ap.add_argument('-y',"--years", type=int, default=8, help="Keep last N years")
    ap.add_argument('-o',"--outdir", default="sec_out")
    ap.add_argument('-u',"--user-agent", default=None, help="Override SEC User-Agent header")
    ap.add_argument("--add-market", action="store_true", help="Also fetch latest market data via yfinance (Price, MarketCap)")
    ap.add_argument("--market-out", default="market_snapshot.csv", help="Output CSV for market snapshot (Ticker,Date,Price,MarketCap)")
    ap.add_argument("--tv-csv", default=None, help="Optional TradingView/manual CSV to merge (Yearly Revenue/Price/MarketCap).")
    ap.add_argument("--tv-unit", default="B", choices=["B","USD"], help="Unit for Revenue/EBIT/FCF/CapEx/SBC in --tv-csv: B=Billions USD (TradingView), USD=1 USD.")
    ap.add_argument("--tv-mode", default="auto", choices=["auto","always","never"], help="How to pick Revenue when both SEC and TV exist: auto=SEC default but replace when divergence is large; always=use TV when present; never=SEC only.")
    ap.add_argument("--tv-threshold", type=float, default=0.25, help="Divergence threshold for auto mode: abs(TV-SEC)/SEC > threshold => use TV.")
    args = ap.parse_args()

    if args.user_agent:
        SEC_HEADERS["User-Agent"] = args.user_agent

    tickers: List[str] = [t.upper() for t in args.tickers if t.strip()]
    if args.ticker_file:
        lines = Path(args.ticker_file).read_text(encoding="utf-8").splitlines()
        tickers += [ln.strip().upper() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    tickers = sorted(set(tickers))
    if not tickers:
        raise SystemExit("No tickers provided. Use --tickers or --ticker-file.")

    outdir = Path(args.outdir)
    cache_dir = outdir / "sec_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    t2c = _load_ticker_cik_map(cache_dir)

    frames = []
    missing = []
    for t in tickers:
        cik = t2c.get(t)
        if not cik:
            missing.append(t)
            continue
        try:
            df = build_draft_for_ticker(t, cik, cache_dir)
            if args.years and df["Year"].notna().any():
                maxy = int(df["Year"].max())
                df = df[df["Year"] >= maxy - (args.years - 1)]
            frames.append(df)
        except Exception as e:
            print(f"[WARN] {t}: {e}")
            continue

    if missing:
        print("[WARN] No CIK found for:", ", ".join(missing))

    if not frames:
        raise SystemExit("No data fetched.")

    #out = pd.concat(frames, ignore_index=True)

    ##251219
    # Drop empty frames and all-NA columns to avoid FutureWarning on concat dtype inference
    clean = []
    for df in frames:
        if df is None or df.empty:
            continue
        df2 = df.dropna(axis=1, how="all")
        ## if it became empty (all columns were NA), skip it
        if df2.empty:
            continue
        clean.append(df2)

    if clean:
        out = pd.concat(clean, ignore_index=True)
    else:
        out = pd.DataFrame()
    #---------------------------------------


    
    # ---- Optional: fetch and merge latest market snapshot (Price, MarketCap) ----

    # ---- Optional merge: TradingView/manual CSV (Revenue/Price/MarketCap)
    # Expected columns (flexible names supported):
    #   Ticker, Year, Revenue, Price, MarketCap  (Revenue unit determined by --tv-unit)
    # Output columns:
    #   Revenue_SEC, Revenue_TV, Revenue_Source, Revenue (final)
    if args.tv_csv:
        tv_path = Path(args.tv_csv)
        if not tv_path.exists():
            raise SystemExit(f"--tv-csv not found: {tv_path}")
        tv = pd.read_csv(tv_path)
        # Canonize TV columns
        col_map = {}
        for c in tv.columns:
            cl = c.lower()
            if c == "Ticker": col_map[c] = "Ticker"
            elif c == "Year": col_map[c] = "Year"
            elif "revenue" in cl or cl in ("rev","sales"): col_map[c] = "Revenue_TV_raw"
            elif "price" in cl: col_map[c] = "Price_TV"
            elif "marketcap" in cl or "mktcap" in cl or "時価総額" in c: col_map[c] = "MktCap_TV_raw"
        tv = tv.rename(columns=col_map)
        for need in ["Ticker","Year"]:
            if need not in tv.columns:
                raise SystemExit(f"--tv-csv must contain column: {need}")
        tv["Ticker"] = tv["Ticker"].astype(str).str.upper().str.strip()
        tv["Year"] = pd.to_numeric(tv["Year"], errors="coerce").astype("Int64")

        # Convert units to USD (1 USD)
        if "Revenue_TV_raw" in tv.columns:
            tv["Revenue_TV"] = pd.to_numeric(tv["Revenue_TV_raw"], errors="coerce")
            if args.tv_unit == "B":
                tv["Revenue_TV"] = tv["Revenue_TV"] * 1e9
        else:
            tv["Revenue_TV"] = np.nan

        if "MktCap_TV_raw" in tv.columns:
            tv["MktCap_USD"] = pd.to_numeric(tv["MktCap_TV_raw"], errors="coerce")
            # TradingView sometimes stores MarketCap in billions; detect by magnitude if user forgot
            # (If the user wants strictness, they can pre-convert to USD.)
            # Here: if values look like 10~5000, treat as billions.
            med = tv["MktCap_USD"].dropna().median()
            if med is not None and med > 0 and med < 1e6:
                tv["MktCap_USD"] = tv["MktCap_USD"] * 1e9
        else:
            tv["MktCap_USD"] = np.nan

        if "Price_TV" in tv.columns:
            tv["Price"] = pd.to_numeric(tv["Price_TV"], errors="coerce")
        else:
            tv["Price"] = np.nan

        tv_keep = tv[["Ticker","Year","Revenue_TV","Price","MktCap_USD"]].copy()
        out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
        out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")

        # Keep SEC revenue as Revenue_SEC, and keep existing Revenue as SEC value (if already computed)
        if "Revenue_SEC" not in out.columns:
            out = out.rename(columns={"Revenue":"Revenue_SEC"})
        # Merge TV
        out = out.merge(tv_keep, on=["Ticker","Year"], how="left", suffixes=("","_TV"))

        # Decide final Revenue (USD)
        out["Revenue_Source"] = "SEC"
        sec = pd.to_numeric(out["Revenue_SEC"], errors="coerce")
        tvv = pd.to_numeric(out["Revenue_TV"], errors="coerce")

        if args.tv_mode == "always":
            use_tv = tvv.notna()
        elif args.tv_mode == "never":
            use_tv = pd.Series(False, index=out.index)
        else:
            # auto: SEC default, but replace when divergence is large or SEC missing
            with np.errstate(divide="ignore", invalid="ignore"):
                div = (tvv - sec).abs() / sec.abs()
            use_tv = tvv.notna() & (sec.isna() | (div > float(args.tv_threshold)))

        out["Revenue"] = sec
        out.loc[use_tv, "Revenue"] = tvv.loc[use_tv]
        out.loc[use_tv, "Revenue_Source"] = "TV(manual)"

        # Market cap (USD -> B) for compatibility
        if "MktCap_B" not in out.columns:
            out["MktCap_B"] = np.nan
        mcap_usd = pd.to_numeric(out.get("MktCap_USD", np.nan), errors="coerce")
        fill_b = out["MktCap_B"].isna() & mcap_usd.notna()
        out.loc[fill_b, "MktCap_B"] = mcap_usd.loc[fill_b] / 1e9

    ##
    if args.add_market:
        snap = fetch_market_snapshot(tickers)
        (outdir / args.market_out).write_text(snap.to_csv(index=False, encoding="utf-8", lineterminator="\n"), encoding="utf-8")
        # Fill Price and MktCap_B in the latest Year row per ticker (best-effort).
        if not snap.empty and "Ticker" in out.columns and "Year" in out.columns:
            snap2 = snap.copy()
            snap2["Ticker"] = snap2["Ticker"].astype(str).str.upper()
            snap2 = snap2.set_index("Ticker")
            for t in out["Ticker"].astype(str).unique().tolist():
                if t not in snap2.index:
                    continue
                maxy = out.loc[out["Ticker"] == t, "Year"].max()
                idx = out.index[(out["Ticker"] == t) & (out["Year"] == maxy)]
                if len(idx) == 0:
                    continue
                price = snap2.loc[t].get("Price", pd.NA)
                mcap = snap2.loc[t].get("MarketCap", pd.NA)
                # Note: units are USD (1 dollar), consistent with SEC companyfacts.
                if "Price" in out.columns:
                    out.loc[idx, "Price"] = price
                if "MktCap_B" in out.columns:
                    out.loc[idx, "MktCap_B"] = mcap
        print(f"OK: wrote {outdir/args.market_out} (market snapshot)")

    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / "draft_input.csv", index=False, encoding="utf-8", lineterminator="\n")
    print(f"OK: wrote {outdir/'draft_input.csv'}")


if __name__ == "__main__":
    main()


## mode lline : 折りたたみの設定でmarker {{{ }}} を使う。
# vim:set foldmethod=marker:
