#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## usage
#{{{
"""
usage:

python3 financial_eval.py --input input.csv --window-years 5 --outdir out


目的:
- 入力CSV（年次データ）から、複数年（WindowYears=サイクル長）で
  平均/最悪/成長/バリュエーションを計算し、3軸(CYCLE/SW/HW)でスコア化します。

出力（ユーザー要望に合わせ「prettyのみ」）:
- metrics_pretty.csv
- scorecards_pretty.csv（Scoreは整数）
- summary_A4.csv / summary_A4.md（重要指標だけ）
- charts/*.png（重要指標の比較グラフ）
- glossary_full.md（出力列の完全用語集：日本語/英語）
- README.md

入力CSV（UTF-8/LF推奨）
必須列:
- Ticker, Company, Year, Revenue, EBIT, FCF

任意列:
- CapEx, SBC, Price, MktCap_B, SharesOut_B

単位:
- Revenue/EBIT/FCF/CapEx/SBC/MktCap_B: Billion USD（B）
- Price: USD/株
- SharesOut_B: Billion shares（B株）※任意

重要: Pとは何か
- 本ツールの基準は **時価総額（Market Cap / MktCap）**
  MktCap = Price * SharesOut
  （TradingViewのMarket Capを MktCap_B として直接入力するのが簡単）

定義（Latest=最新年）
- PS_Latest        = MktCap / Rev.（売上）
- PFCF_Latest      = MktCap / FCF（FCF<=0 は意味が壊れるので空欄）
- POwnerFCF_Latest = MktCap / (FCF - SBC)（<=0は空欄）


===============================================================================
PROGRAM 2 : yy02_financial_eval.py
===============================================================================
目的
  yy01で取得したCSVを使い、各種指標・スコア・可視化を生成する。
  比較の公平性のため、年数の揃え方を制御できる。

引数一覧
  -i  --input           入力CSV（必須）
  -w  --window-years    使用する直近年数（default: 5）
      --window-mode     年数の揃え方
                         intersection : 全銘柄で共通の最短年数に揃える
                         per-ticker   : 各銘柄ごとに直近-w年を使用
                         （default: intersection）
  -o  --outdir          出力ディレクトリ
      --color-by        Factor Mapの色軸（CYCLE | SW | HW）

出力列
  ActualWindowYears
    実際に計算に使われた年数
    ・intersection : 全銘柄で同一
    ・per-ticker   : 銘柄ごとに異なる

使用例（推奨・公平比較）
  python3 yy02_financial_eval_04s.py \
    -i input_data.csv \
    -w 5 \
    --window-mode intersection \
    -o out_eval \
    --color-by SW

使用例（最大履歴を使う場合）
  python3 yy02_financial_eval_04s.py \
    -i input_data.csv \
    -w 8 \
    --window-mode per-ticker \
    -o out_eval \
    --color-by CYCLE

"""
# }}}


from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import textwrap


# -----------------------------
# Fixed ordering for percentile bars (meaning-first).
# Percentile is cross-sectional rank among tickers in the current run (0-100).
# -----------------------------
PERCENTILE_GROUPS = [
    ("Valuation", [
        ("MC/FCF", "MC/FCF (cheaper is better)", False),
        ("MC/OwnerFCF", "MC/OwnerFCF (cheaper is better)", False),
        ("PEG_PS_Latest", "PEG-like: (MC/Rev)/RevGrowth% (lower is better)", False),
    ]),
    ("Profitability", [
        ("FCF/Rev_Avg", "FCF/Rev (Avg) (higher is better)", True),
        ("EBIT/Rev_Avg", "EBIT/Rev (Avg) (higher is better)", True),
    ]),
    ("Efficiency", [
        ("CapEx/Rev_Avg", "CapEx/Rev (Avg) (higher = heavier)", None),
        ("SBC/Rev_Avg", "SBC/Rev (Avg) (higher = more SBC)", None),
    ]),
    ("Growth", [
        ("Rev_CAGR", "Rev CAGR (higher is better)", True),
    ]),
    ("Scale", [
        ("MktCap_B_Latest", "Market Cap (B USD)", None),
    ]),
]

def _render_defs_md() -> str:
    """Generate markdown snippet that defines factors/axes and chart conventions.
    This is the single source used to sync README/glossary sections."""
    lines = []
    lines.append("## Definitions (auto-synced)")
    lines.append("")
    lines.append("### Score axes (0-100)")
    lines.append("- Score_SW: structure tilt toward software/human intensity (not 'good/bad').")
    lines.append("  - Higher means: lower CapEx/Rev (lighter), higher SBC/Rev (more human intensity), higher Rev CAGR (growth tilt).")
    lines.append("- Score_HW: structure tilt toward hardware/capital intensity (not 'good/bad').")
    lines.append("  - Higher means: higher CapEx/Rev (heavier) plus resilience checks (EBIT/Rev_Min, FCF/Rev_Min).")
    lines.append("- Score_CYCLE: cyclicality / stability score (higher = more stable / less cyclical).")
    lines.append("")
    lines.append("### Factor Map")
    lines.append("- X = Valuation (Expensive) 0-100: higher means 'more expensive' (NOT better).")
    lines.append("  - Components: PS_Latest (MC/Rev), MC/FCF, MC/OwnerFCF, PEG-like.")
    lines.append("- Y = Business Strength 0-100: higher means stronger profitability/resilience.")
    lines.append("  - Components: EBIT/Rev & FCF/Rev (Avg & Min).")
    lines.append("- Bubble size ~ Market Cap (B USD).")
    lines.append("- Color axis selectable: CYCLE / SW / HW.")
    lines.append("")
    lines.append("### Raw percentile bars (per ticker)")
    lines.append("- Each bar is a cross-sectional percentile rank among tickers in this run (0-100).")
    lines.append("- It is NOT a 'contribution'. Contribution is shown in the score breakdown charts.")
    lines.append("- Fixed group order: Valuation → Profitability → Efficiency → Growth → Scale.")
    lines.append("")
    return "\n".join(lines)

def _sync_md_autosection(path: Path, begin: str, end: str, body: str) -> None:
    """Replace or append an auto section delimited by markers."""
    begin_m = f"<!--{begin}-->"
    end_m = f"<!--{end}-->"
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    if begin_m in content and end_m in content:
        pre = content.split(begin_m)[0]
        post = content.split(end_m)[1]
        newc = pre + begin_m + "\n\n" + body.strip() + "\n\n" + end_m + post
    else:
        if content and not content.endswith("\n"):
            content += "\n"
        newc = content + "\n" + begin_m + "\n\n" + body.strip() + "\n\n" + end_m + "\n"
    write_utf8_lf(path, newc)


def _norm_col(c: str) -> str:
    c = c.replace("\r\n", "\n").replace("\r", "\n")
    c = c.replace("\n", " ").strip()
    c = re.sub(r"\s+", " ", c)
    return c


def _canonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to the canonical schema used by this script.

    Goal:
      - tolerate mixed sources (SEC / TradingView manual / Yahoo finance)
      - keep units explicit:
          Revenue_* : USD (1.0 = $1)
          Price_USD : USD
          MktCap_USD: USD
          MktCap_B  : billions of USD (only for convenience / display)
          SharesOut_B : billions of shares (if provided)
    """
    if df is None or df.empty:
        return df

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", str(s)).strip()

    # Build rename map
    rename: dict[str, str] = {}
    for col in df.columns:
        c0 = _norm(col)
        cl = c0.lower()
        cu = c0.upper()

        # --- core identity ---
        if cu in ("TICKER", "SYMBOL"):
            rename[col] = "Ticker"; continue
        if cu in ("COMPANY", "NAME"):
            rename[col] = "Company"; continue
        if cu in ("YEAR", "FY", "FISCALYEAR"):
            rename[col] = "Year"; continue

        # --- revenue (dual source) ---
        if cl in ("revenue_sec", "rev_sec", "sales_sec"):
            rename[col] = "Revenue_SEC"; continue
        if cl in ("revenue_tv", "rev_tv", "sales_tv"):
            rename[col] = "Revenue_TV"; continue
        if cl in ("revenue_source", "rev_source"):
            rename[col] = "Revenue_Source"; continue

        # fallback revenue columns (single source)
        if cl in ("revenue", "rev", "sales", "total revenue", "totalrevenues", "売上", "売上高"):
            # keep as Revenue_SEC by default (so later override logic can still work)
            rename[col] = "Revenue_SEC"; continue

        # --- operating / cash flow ---
        if cl in ("ebit", "operating income", "operatingincome", "営業利益"):
            rename[col] = "EBIT"; continue
        if cl in ("fcf", "free cash flow", "freecashflow", "フリーキャッシュフロー"):
            rename[col] = "FCF"; continue
        if cl in ("capex", "capital expenditures", "capitalexpenditures", "設備投資"):
            rename[col] = "CapEx"; continue
        if cl in ("sbc", "stock based compensation", "stockbasedcompensation", "株式報酬"):
            rename[col] = "SBC"; continue

        # --- market data ---
        # IMPORTANT: detect explicit USD before generic "market cap" bucket
        if "mktcap_usd" in cl or "marketcap_usd" in cl or "market cap usd" in cl:
            rename[col] = "MktCap_USD"; continue
        if "price_usd" in cl:
            rename[col] = "Price"; continue

        # common price columns from finance sources
        if cl in ("price", "close", "adjclose", "adj close", "last", "株価", "終値"):
            rename[col] = "Price"; continue

        # market cap: either USD or B
        if "mktcap" in cl or "market cap" in cl or "marketcap" in cl:
            # if it looks like already in billions (contains (b), 'billion', etc.)
            if "(b" in cl or "b)" in cl or "billion" in cl or "bil" in cl:
                rename[col] = "MktCap_B"
            else:
                rename[col] = "MktCap_USD"
            continue

        # shares outstanding
        if "shares" in cl and ("out" in cl or "outstanding" in cl) or "発行" in cl:
            rename[col] = "SharesOut"; continue

        # date columns (optional)
        if cl in ("date", "asof", "as of"):
            rename[col] = "Date"; continue

    if rename:
        df = df.rename(columns=rename)

    # Ensure required columns exist
    required = ["Ticker", "Company", "Year", "EBIT", "FCF"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    return df


def read_input_csv(path: Path) -> pd.DataFrame:
    """
    Read input CSV robustly.

    Handles:
      - UTF-8 / CP932
      - column normalization + canonization
      - duplicated column names (consolidates by first non-null)
      - numeric coercion
      - Revenue synthesis (Revenue_TV preferred over Revenue_SEC)
      - MarketCap synthesis (USD/B) and fallback from Price*SharesOut

    Expected (minimum) columns:
      Ticker, Year
    Optional:
      Revenue_SEC, Revenue_TV, Revenue, Revenue_Source,
      EBIT, FCF, CapEx, SBC,
      Price, MktCap_USD, MktCap_B, SharesOut_B
    """
    try:
        df = pd.read_csv(path, sep=",", engine="python", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=",", engine="python", encoding="cp932")

    # Normalize/Canonize column names
    df.columns = [_norm_col(c) for c in df.columns]
    df = _canonize_columns(df)

    # Consolidate duplicated column names safely.
    # Pandas returns a DataFrame for df["X"] when columns are duplicated, which breaks downstream
    # numeric coercion. We coalesce duplicates left-to-right: first non-NA wins.
    if df.columns.duplicated().any():
        cols = list(df.columns)
        first_order = []
        seen = set()
        for c in cols:
            if c not in seen:
                first_order.append(c)
                seen.add(c)

        out = {}
        for c in first_order:
            idxs = [i for i, cc in enumerate(cols) if cc == c]
            # start with the left-most occurrence as a Series
            s = df.iloc[:, idxs[0]]
            for j in idxs[1:]:
                s2 = df.iloc[:, j]
                s = s.combine_first(s2)
            out[c] = s
        df = pd.DataFrame(out)


    # Ensure required columns exist
    if "Ticker" not in df.columns:
        raise ValueError("Input CSV must contain 'Ticker' column.")
    if "Year" not in df.columns:
        raise ValueError("Input CSV must contain 'Year' column.")

    # Create optional columns if missing (keeps downstream logic simple)
    for c in ["Revenue", "Revenue_SEC", "Revenue_TV", "EBIT", "FCF", "CapEx", "SBC",
              "Price", "MktCap_USD", "MktCap_B", "SharesOut_B"]:
        if c not in df.columns:
            df[c] = np.nan
    if "Revenue_Source" not in df.columns:
        df["Revenue_Source"] = ""

    # Coerce Year first
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    # Coerce numeric columns (only Series now)
    num_cols = ["Revenue", "Revenue_SEC", "Revenue_TV", "EBIT", "FCF", "CapEx", "SBC",
                "Price", "MktCap_B", "MktCap_USD", "SharesOut_B"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop invalid rows
    df = df.dropna(subset=["Ticker", "Year"]).copy()
    df["Year"] = df["Year"].astype(int)

    # ---- Revenue synthesis
    # Prefer TV/manual if present; else SEC.
    tv = df["Revenue_TV"]
    sec = df["Revenue_SEC"]
    use_tv = tv.notna()
    use_sec = (~use_tv) & sec.notna()

    df.loc[use_tv, "Revenue"] = tv[use_tv]
    df.loc[use_sec, "Revenue"] = sec[use_sec]

    # Source label (only set if empty)
    empty_src = df["Revenue_Source"].fillna("").astype(str).str.len() == 0
    df.loc[empty_src & use_tv, "Revenue_Source"] = "TV(manual)"
    df.loc[empty_src & use_sec, "Revenue_Source"] = "SEC"

    # ---- MarketCap synthesis
    # If only one is present, derive the other.
    # Prefer USD as canonical, B as convenience.
    if df["MktCap_USD"].isna().all() and df["MktCap_B"].notna().any():
        # assume MktCap_B is in billions USD
        df["MktCap_USD"] = df["MktCap_B"] * 1e9
    if df["MktCap_B"].isna().all() and df["MktCap_USD"].notna().any():
        df["MktCap_B"] = df["MktCap_USD"] / 1e9

    # Fallback: Price * SharesOut_B (billions shares) -> market cap (billions USD)
    miss = df["MktCap_B"].isna() & df["Price"].notna() & df["SharesOut_B"].notna()
    df.loc[miss, "MktCap_B"] = df.loc[miss, "Price"] * df.loc[miss, "SharesOut_B"]
    miss_usd = df["MktCap_USD"].isna() & df["MktCap_B"].notna()
    df.loc[miss_usd, "MktCap_USD"] = df.loc[miss_usd, "MktCap_B"] * 1e9

    return df

def _compute_available_years(df: pd.DataFrame, ticker: str) -> List[int]:
    yrs = df.loc[df["Ticker"] == ticker, "Year"].dropna().unique().tolist()
    out: List[int] = []
    for y in yrs:
        try:
            out.append(int(y))
        except Exception:
            continue
    out.sort()
    return out


def cagr(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 2:
        return np.nan
    start = float(s.iloc[0])
    end = float(s.iloc[-1])
    n = len(s) - 1
    if n <= 0 or start <= 0 or end <= 0:
        return np.nan
    return (end / start) ** (1.0 / n) - 1.0


def compute_metrics(df: pd.DataFrame, window_years: int, window_years_by_ticker: Optional[Dict[str,int]] = None) -> pd.DataFrame:
    out = []
    for tkr, g in df.groupby("Ticker"):
        wy = int(window_years_by_ticker.get(tkr, window_years)) if window_years_by_ticker else int(window_years)
        g = g.sort_values("Year")
        max_year = int(g["Year"].max())
        win_start = max_year - (wy - 1)
        gw = g[g["Year"] >= win_start].copy()

        def safe_mean(x: pd.Series) -> float:
            return float(x.mean()) if x.notna().any() else np.nan

        rev_avg = safe_mean(gw["Revenue"])
        ebit_avg = safe_mean(gw["EBIT"])
        fcf_avg = safe_mean(gw["FCF"])

        ebit_margin = (gw["EBIT"] / gw["Revenue"]).replace([np.inf, -np.inf], np.nan)
        fcf_margin = (gw["FCF"] / gw["Revenue"]).replace([np.inf, -np.inf], np.nan)
        capex_ratio = (gw["CapEx"] / gw["Revenue"]).replace([np.inf, -np.inf], np.nan)
        sbc_ratio = (gw["SBC"] / gw["Revenue"]).replace([np.inf, -np.inf], np.nan)

        ebit_margin_avg = safe_mean(ebit_margin)
        fcf_margin_avg = safe_mean(fcf_margin)
        capex_ratio_avg = safe_mean(capex_ratio)
        sbc_ratio_avg = safe_mean(sbc_ratio)

        ebit_margin_min = float(ebit_margin.min()) if ebit_margin.notna().any() else np.nan
        fcf_margin_min = float(fcf_margin.min()) if fcf_margin.notna().any() else np.nan

        rev_cagr = cagr(gw.set_index("Year")["Revenue"])
        fcf_cagr = cagr(gw.set_index("Year")["FCF"])
        owner_fcf_series = (gw.set_index("Year")["FCF"] - gw.set_index("Year")["SBC"]) if ("FCF" in gw and "SBC" in gw) else None
        owner_fcf_cagr = cagr(owner_fcf_series) if owner_fcf_series is not None else np.nan


        gl = g[g["Year"] == max_year].iloc[-1]
        mktcap = gl["MktCap_B"]
        price = gl["Price"]

        owner_fcf = (gl["FCF"] - gl["SBC"]) if pd.notna(gl["FCF"]) and pd.notna(gl["SBC"]) else np.nan

        ps = (mktcap / gl["Revenue"]) if pd.notna(mktcap) and pd.notna(gl["Revenue"]) and gl["Revenue"] != 0 else np.nan
        pfcf = (mktcap / gl["FCF"]) if pd.notna(mktcap) and pd.notna(gl["FCF"]) and gl["FCF"] > 0 else np.nan
        p_owner_fcf = (mktcap / owner_fcf) if pd.notna(mktcap) and pd.notna(owner_fcf) and owner_fcf > 0 else np.nan
        peg_ps_latest = (ps / (rev_cagr*100.0)) if pd.notna(ps) and pd.notna(rev_cagr) and rev_cagr > 0 else np.nan
        peg_ownerfcf_latest = (p_owner_fcf / (owner_fcf_cagr*100.0)) if pd.notna(p_owner_fcf) and pd.notna(owner_fcf_cagr) and owner_fcf_cagr > 0 else np.nan

        out.append({
            "Ticker": tkr,
            "Company": str(g["Company"].iloc[0]),
            "MaxYear": max_year,
            "WindowYears": wy,
            "ActualWindowYears": int(gw["Year"].nunique()),
            "WindowStartYear": win_start,

            "Revenue_Avg": rev_avg,
            "EBIT_Avg": ebit_avg,
            "FCF_Avg": fcf_avg,

            "EBITMargin_Avg": ebit_margin_avg,
            "EBITMargin_Min": ebit_margin_min,
            "FCFMargin_Avg": fcf_margin_avg,
            "FCFMargin_Min": fcf_margin_min,

            "CapExOverRev_Avg": capex_ratio_avg,
            "SBCOverRev_Avg": sbc_ratio_avg,

            "Rev_CAGR": rev_cagr,
            "FCF_CAGR": fcf_cagr,
            "OwnerFCF_CAGR": owner_fcf_cagr,
            "PEG_PS_Latest": peg_ps_latest,
            "PEG_OwnerFCF_Latest": peg_ownerfcf_latest,
            "OwnerFCF_CAGR": owner_fcf_cagr,
            "PEG_PS_Latest": peg_ps_latest,
            "PEG_OwnerFCF_Latest": peg_ownerfcf_latest,

            "Price_Latest": price,
            "MktCap_B_Latest": mktcap,

            "PS_Latest": ps,
            "PFCF_Latest": pfcf,
            "POwnerFCF_Latest": p_owner_fcf,
        })

    return pd.DataFrame(out).sort_values("Ticker").reset_index(drop=True)


def score_1to5(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.Series(values, dtype="float64")
    mask = s.notna()
    if mask.sum() <= 1:
        return pd.Series(np.where(mask, 3.0, np.nan), index=s.index, dtype="float64")
    if higher_is_better:
        r = s[mask].rank(ascending=False, method="min")
    else:
        r = s[mask].rank(ascending=True, method="min")
    n = int(mask.sum())
    score = 1 + 4 * (1 - (r - 1) / (n - 1))
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out.loc[mask] = score
    return out


def build_scorecard(metrics: pd.DataFrame, profile: str) -> pd.DataFrame:
    profile = profile.lower()

    if profile == "cycle":
        defs: List[Tuple[str, bool, float]] = [
            ("FCFMargin_Avg", True, 0.25),
            ("FCFMargin_Min", True, 0.20),
            ("EBITMargin_Avg", True, 0.15),
            ("CapExOverRev_Avg", False, 0.10),
            ("PS_Latest", False, 0.10),
            ("PFCF_Latest", False, 0.10),
            ("Rev_CAGR", True, 0.10),
        ]
    elif profile == "sw":
        # SWは「良し悪し」よりも「軽装備・人（ソフト）寄りの体質」を表す意図。
        # そのため 利益率（EBIT/Rev, FCF/Rev）は結果に寄りやすいので、ここでは中心に置かない。
        # 代わりに「装備の軽さ（CapEx/Revが低い）」「人件費/SBCの重さ（SBC/Revが高い）」
        # 「スケール余地（Rev_CAGR）」を主に見て、相対評価する。
        defs = [
            ("CapExOverRev_Avg", False, 0.40),  # 軽装備ほどSW寄り
            ("SBCOverRev_Avg", True, 0.30),     # 人（株式報酬）に寄るほどSW寄り（良し悪しではなく体質）
            ("Rev_CAGR", True, 0.20),           # スケール余地（補助）
            ("PS_Latest", False, 0.10),         # 割高さのチェック（補助）
        ]
    elif profile == "hw":
        # HWは「良し悪し」よりも「重装備(ハード寄り)の体質」を表す意図。
        # CapEx/Rev が高いほどHW寄りとして扱う。
        # ただし「重いだけ」にならないよう、耐性（EBIT/Rev_Min など）を併せて見る。
        defs = [
            ("CapExOverRev_Avg", True, 0.45),   # 装備の重さ（高いほどHW寄り）
            ("EBITMargin_Min", True, 0.25),     # 不況時の耐性（最悪年）
            ("FCFMargin_Min", True, 0.15),      # 現金化の耐性（最悪年）
            ("PFCF_Latest", False, 0.15),       # 割高さのチェック（補助）
        ]
    else:
        raise ValueError("profile must be one of: cycle, sw, hw")

    m = metrics.copy().reset_index(drop=True)
    for col, hib, w in defs:
        m[f"S_{col}"] = score_1to5(m[col], higher_is_better=hib)
        m[f"W_{col}"] = w

    s_cols = [f"S_{col}" for col, _, _ in defs]
    w_cols = [f"W_{col}" for col, _, _ in defs]

    S = m[s_cols].to_numpy(dtype=float)
    W = m[w_cols].to_numpy(dtype=float)
    mask = ~np.isnan(S)

    num = np.nansum(S * W, axis=1)
    denom = np.nansum(W * mask, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        m["Score_1to5"] = np.where(denom > 0, num / denom, np.nan)
    m["Score_0to100"] = m["Score_1to5"] * 20

    def rating(x: float) -> str:
        if pd.isna(x):
            return ""
        if x >= 85: return "A"
        if x >= 70: return "B"
        if x >= 55: return "C"
        if x >= 40: return "D"
        return "E"

    m["Rating"] = m["Score_0to100"].apply(rating)
    m["Profile"] = profile.upper()

    core = ["Profile", "Ticker", "Company", "MaxYear", "WindowYears", "WindowStartYear", "Score_0to100", "Rating"]
    metrics_cols = [col for col, _, _ in defs]
    score_cols = [f"S_{col}" for col, _, _ in defs]
    return m[core + metrics_cols + score_cols]


def _fmt_num_general(v):
    if pd.isna(v):
        return ""
    try:
        x = float(v)
    except Exception:
        return str(v)
    if abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    return f"{x:.3f}"


def _fmt_score_int(v):
    if pd.isna(v):
        return ""
    try:
        return str(int(round(float(v))))
    except Exception:
        return ""


def df_pretty_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(_fmt_num_general)
    return out


def df_pretty_scorecards(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c == "Score_0to100":
            out[c] = out[c].map(_fmt_score_int)
        elif pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(_fmt_num_general)
    return out



# ----------------------------
# Naming simplification layer
# ----------------------------
# 目的:
# - pandas内部の列名（例: FCFMargin_Avg）を壊さず、
#   出力/表示では数式表記（例: FCF/Rev_Avg）に統一する。
# - 「margin」という単語は出力側で使わない。

# pandas列名(旧/alias) -> 出力列名(数式/canonical)
OUTPUT_COL_RENAME = {
    # valuation
    "PFCF_Latest": "MC/FCF",
    "POwnerFCF_Latest": "MC/OwnerFCF",

    # ratios (avoid 'margin' wording in outputs)
    "EBITMargin_Avg": "EBIT/Rev_Avg",
    "EBITMargin_Min": "EBIT/Rev_Min",
    "FCFMargin_Avg": "FCF/Rev_Avg",
    "FCFMargin_Min": "FCF/Rev_Min",
    "CapExOverRev_Avg": "CapEx/Rev_Avg",
    "SBCOverRev_Avg": "SBC/Rev_Avg",

    # keep these readable too
    "PS_Latest": "MC/Rev",
}

# 軸ラベル/凡例用（必要ならさらに短縮）
OUTPUT_LABEL_MAP = {
    "MC/FCF": "MC/FCF",
    "MC/OwnerFCF": "MC/OwnerFCF",
    "MC/Rev": "MC/Rev",
    "EBITMargin_Avg": "EBIT/Rev (Avg)",
    "EBITMargin_Min": "EBIT/Rev (Min)",
    "FCFMargin_Avg": "FCF/Rev (Avg)",
    "FCFMargin_Min": "FCF/Rev (Min)",
    "CapExOverRev_Avg": "CapEx/Rev (Avg)",
    "SBCOverRev_Avg": "SBC/Rev (Avg)",
    "PS_Latest": "MC/Rev",
    "PFCF_Latest": "MC/FCF",
    "POwnerFCF_Latest": "MC/OwnerFCF",
}

def rename_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """出力ファイル用に列名を数式表記へ統一（内部計算の列名は維持）。"""
    return df.rename(columns=OUTPUT_COL_RENAME)


def write_utf8_lf(path: Path, text: str) -> None:
    path.write_text(text.replace("\r\n", "\n").replace("\r", "\n"), encoding="utf-8", newline="\n")


A4_COLUMNS = [
    "Ticker","Company","MaxYear","WindowYears","ActualWindowYears",
    "Score_CYCLE","Rating_CYCLE","Score_SW","Rating_SW","Score_HW","Rating_HW",
    "EBITMargin_Avg","EBITMargin_Min","FCFMargin_Avg","FCFMargin_Min",
    "CapExOverRev_Avg","SBCOverRev_Avg","Rev_CAGR",
    #"MktCap_B_Latest","PS_Latest","PFCF_Latest","POwnerFCF_Latest",
    "MktCap_B_Latest","PS_Latest","MC/FCF","MC/OwnerFCF",
    "PEG_PS_Latest","PEG_OwnerFCF_Latest",
]

SUMMARY2_COLUMNS = [
    "Ticker",
    "Rating_CYCLE","Score_CYCLE","Score_SW","Score_HW",
    "PEG_PS_Latest","PEG_OwnerFCF_Latest",
    #"PFCF_Latest","POwnerFCF_Latest",
    "MC/FCF","MC/OwnerFCF",
    "EBITMargin_Avg","FCFMargin_Avg",
    "CapExOverRev_Avg","SBCOverRev_Avg",
    "Rev_CAGR",
    "MaxYear","Company",
]


def build_summary_A4(metrics: pd.DataFrame, scorecards: pd.DataFrame) -> pd.DataFrame:
    sc = scorecards[["Profile","Ticker","Score_0to100","Rating"]].copy()
    sc_piv = sc.pivot(index="Ticker", columns="Profile", values=["Score_0to100","Rating"])
    sc_piv.columns = [f"{a}_{b}" for a,b in sc_piv.columns]
    sc_piv = sc_piv.reset_index()

    merged = metrics.merge(sc_piv, on="Ticker", how="left")
    merged = merged.rename(columns={
        "Score_0to100_CYCLE":"Score_CYCLE","Rating_CYCLE":"Rating_CYCLE",
        "Score_0to100_SW":"Score_SW","Rating_SW":"Rating_SW",
        "Score_0to100_HW":"Score_HW","Rating_HW":"Rating_HW",
    })

    OUT_COL_RENAME = {"PFCF_Latest":"MC/FCF","POwnerFCF_Latest":"MC/OwnerFCF"}
    merged = merged.rename(columns=OUT_COL_RENAME)
    cols = [c for c in A4_COLUMNS if c in merged.columns]
    return merged[cols].sort_values(["Score_CYCLE","Score_SW","Score_HW"], ascending=False)

def build_summary2(metrics: pd.DataFrame, scorecards: pd.DataFrame) -> pd.DataFrame:
    """
    summary2: 比較・投資判断向け（重要指標を厳選）
    - 並びはユーザー指定順
    - Score列は整数表示（pretty側で整形）
    """
    sc = scorecards[["Profile","Ticker","Score_0to100","Rating"]].copy()
    sc_piv = sc.pivot(index="Ticker", columns="Profile", values=["Score_0to100","Rating"])
    sc_piv.columns = [f"{a}_{b}" for a,b in sc_piv.columns]
    sc_piv = sc_piv.reset_index()

    merged = metrics.merge(sc_piv, on="Ticker", how="left")
    merged = merged.rename(columns={
        "Score_0to100_CYCLE":"Score_CYCLE","Rating_CYCLE":"Rating_CYCLE",
        "Score_0to100_SW":"Score_SW","Rating_SW":"Rating_SW",
        "Score_0to100_HW":"Score_HW","Rating_HW":"Rating_HW",
    })

    OUT_COL_RENAME = {"PFCF_Latest":"MC/FCF","POwnerFCF_Latest":"MC/OwnerFCF"}
    merged = merged.rename(columns=OUT_COL_RENAME)
    cols = [c for c in SUMMARY2_COLUMNS if c in merged.columns]
    # Score優先で並べる（空欄でも落ちないように）
    sort_cols = [c for c in ["Score_CYCLE","Score_SW","Score_HW"] if c in cols]
    if sort_cols:
        merged = merged.sort_values(sort_cols, ascending=False)
    return merged[cols]




def save_scores_2d(df: pd.DataFrame, outpath: Path) -> None:
    """2D散布図: x=Score_SW, y=Score_HW, color=Score_CYCLE, label=Ticker"""
    need = ["Ticker", "Score_SW", "Score_HW", "Score_CYCLE"]
    for c in need:
        if c not in df.columns:
            return

    d = df[need].copy()
    for c in ["Score_SW", "Score_HW", "Score_CYCLE"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(d["Score_SW"], d["Score_HW"], s=220, c=d["Score_CYCLE"])

    ax.set_xlabel("Score_SW (points)")
    ax.set_ylabel("Score_HW (points)")
    ax.set_title("Scores on a plane: x=SW, y=HW, color=CYCLE", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(True)

    # label
    for _, r in d.iterrows():
        ax.text(r["Score_SW"] + 0.8, r["Score_HW"] + 0.3, str(r["Ticker"]), va="center")

    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Score_CYCLE (points)")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)



def _percentile_rank(series: pd.Series) -> pd.Series:
    """Return percentile rank 0-100 (higher value => higher percentile)."""
    s = pd.to_numeric(series, errors="coerce")
    # rank(pct=True) gives 0..1; convert to 0..100
    return (s.rank(pct=True) * 100.0)


def save_raw_percentile_bars(df: pd.DataFrame, outdir: Path) -> None:
    """Create per-ticker percentile bar charts for key raw metrics (0-100).

    Notes:
      - Percentile is cross-sectional rank among tickers in *this run* (0-100).
      - For valuation multiples where 'lower is better', we invert to show 'goodness percentile'.
      - For structure metrics (CapEx/Rev, SBC/Rev, MarketCap), we keep raw percentile (no good/bad).
      - Ordering is fixed by meaning: Valuation → Profitability → Efficiency → Growth → Scale.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    d = df.copy()
    if "Ticker" not in d.columns:
        return

    # Build percentile dataframe in fixed order
    p = pd.DataFrame({"Ticker": d["Ticker"].astype(str)})
    label_map: Dict[str, str] = {}
    hib_map: Dict[str, Optional[bool]] = {}

    ordered_cols: List[str] = []
    group_spans: List[Tuple[str, int, int]] = []  # (group, start_idx, end_idx) indices in ordered_cols

    cur = 0
    for gname, items in PERCENTILE_GROUPS:
        start = cur
        for col, label, hib in items:
            if col not in d.columns:
                continue
            pr = _percentile_rank(d[col])
            if hib is False:
                pr = 100.0 - pr  # convert to goodness percentile for "cheaper is better"
            p[col] = pr
            label_map[col] = label
            hib_map[col] = hib
            ordered_cols.append(col)
            cur += 1
        end = cur
        if end > start:
            group_spans.append((gname, start, end))

    if not ordered_cols:
        return

    # One chart per ticker (small N; easier to read)
    for _, r in p.iterrows():
        t = r["Ticker"]
        vals = [float(r[c]) for c in ordered_cols]

        fig = plt.figure(figsize=(11.5, 6.5))
        ax = fig.add_subplot(111)
        y = list(range(len(ordered_cols)))
        ax.barh(y, vals)

        ax.set_yticks(y)
        ax.set_yticklabels([textwrap.fill(str(label_map.get(c, c)), width=22) for c in ordered_cols])
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='x', labelsize=8)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Percentile (0-100)")
        ax.set_title(f"Raw-metric percentiles: {t}", fontsize=10)
        ax.grid(True, axis="x")

        # Group separators + labels (no custom colors)
        for gname, s, e in group_spans:
            # draw a horizontal separator at the boundary (except top)
            if s > 0:
                ax.axhline(s - 0.5, linewidth=1)
            # put group name on the left margin (axes coords)
            ymid = (s + e - 1) / 2.0
            ax.text(-0.28, ymid, gname, va="center", ha="right",
                    transform=ax.get_yaxis_transform(), fontsize=9, fontweight="bold")

        # Extra left margin for wrapped y-labels and group headings
        fig.subplots_adjust(left=0.48, right=0.97, top=0.90, bottom=0.12)
        fig.tight_layout()
        fig.savefig(outdir / f"20_percentiles_{t}.png", dpi=200)
        plt.close(fig)


def save_score_breakdown(metrics: pd.DataFrame, outdir: Path) -> None:
    """Visualize how SW/HW/CYCLE scores are composed (weighted components)."""
    outdir.mkdir(parents=True, exist_ok=True)

    # We rebuild scorecards here to access component columns S_* and weights W_*
    for profile in ["sw", "hw", "cycle"]:
        try:
            sc = build_scorecard(metrics, profile=profile)
        except Exception:
            continue

        if "Ticker" not in sc.columns or "Score" not in sc.columns:
            continue

        # Identify component score/weight columns
        s_cols = [c for c in sc.columns if c.startswith("S_")]
        w_cols = [c for c in sc.columns if c.startswith("W_")]
        if not s_cols or not w_cols:
            continue

        # Map S_ col -> W_ col and compute contributions
        contrib = pd.DataFrame({"Ticker": sc["Ticker"].astype(str)})
        for s_col in s_cols:
            base = s_col[2:]
            w_col = f"W_{base}"
            if w_col not in sc.columns:
                continue
            contrib[base] = pd.to_numeric(sc[s_col], errors="coerce") * pd.to_numeric(sc[w_col], errors="coerce")

        # Normalize to 0-100 per ticker
        comp_cols = [c for c in contrib.columns if c != "Ticker"]
        total = contrib[comp_cols].sum(axis=1)
        for c in comp_cols:
            contrib[c] = np.where(total > 0, contrib[c] / total * 100.0, np.nan)

        # Plot stacked horizontal bars
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)

        y_pos = np.arange(len(contrib))
        left = np.zeros(len(contrib))
        for c in comp_cols:
            vals = contrib[c].fillna(0.0).to_numpy()
            ax.barh(y_pos, vals, left=left, label=c)
            left += vals

        ax.set_yticks(y_pos)
        ax.set_yticklabels([textwrap.fill(str(x), width=28) for x in contrib["Ticker"].tolist()])
        ax.set_xlim(0, 100)
        ax.set_xlabel("Contribution to score (percent of weighted sum)")
        ax.set_title(f"Score breakdown ({profile.upper()}): component contributions", fontsize=10)
        ax.grid(True, axis="x")
        ax.legend(loc="lower right", fontsize=8)

        # Extra left margin for wrapped y-labels and group headings
        fig.subplots_adjust(left=0.48, right=0.97, top=0.90, bottom=0.12)
        fig.tight_layout()
        fig.savefig(outdir / f"30_score_breakdown_{profile}.png", dpi=200)
        plt.close(fig)


def _scale_0_100(series: pd.Series, higher_is_better: bool = True,
                 q_low: float = 0.05, q_high: float = 0.95) -> pd.Series:
    """
    Robust 0-100 scaling with percentile clipping.
    - higher_is_better=True: larger -> closer to 100
    - higher_is_better=False: smaller -> closer to 100
    """
    s = pd.to_numeric(series, errors="coerce").astype(float)
    lo = s.quantile(q_low)
    hi = s.quantile(q_high)
    if pd.isna(lo) or pd.isna(hi) or lo == hi:
        # fallback: min-max without clipping
        lo = s.min()
        hi = s.max()
        if pd.isna(lo) or pd.isna(hi) or lo == hi:
            return pd.Series([np.nan]*len(s), index=s.index)

    s_clip = s.clip(lower=lo, upper=hi)
    norm = (s_clip - lo) / (hi - lo)
    if not higher_is_better:
        norm = 1.0 - norm
    return (norm * 100.0).clip(0.0, 100.0)


def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build compact factor view (0-100) from summary_A4-like dataframe.

    Factors:
    - Quality: profitability + resilience (EBIT/Rev & FCF/Rev, avg & min)
    - Growth: Rev_CAGR
    - Valuation (Expensive): MC/Rev, MC/FCF, MC/OwnerFCF, PEG-like
    - Capital_Intensity_HW: CapEx/Rev (high => HW-ish)
    - Human_Intensity_SW: SBC/Rev (high => SW-ish)
    """
    out = df[["Ticker"]].copy()

    # Quality sub-scores (higher is better)
    q_parts = []
    for c in ["EBITMargin_Avg", "FCFMargin_Avg", "EBITMargin_Min", "FCFMargin_Min"]:
        if c in df.columns:
            q_parts.append(_scale_0_100(df[c], higher_is_better=True))
    out["F_Quality"] = pd.concat(q_parts, axis=1).mean(axis=1) if q_parts else np.nan

    # Growth (higher is better)
    out["F_Growth"] = _scale_0_100(df["Rev_CAGR"], higher_is_better=True) if "Rev_CAGR" in df.columns else np.nan

    # Valuation (Expensive): higher means more expensive (not "better")
    v_parts = []
    for c in ["PS_Latest", "MC/FCF", "MC/OwnerFCF", "PEG_PS_Latest", "PEG_OwnerFCF_Latest"]:
        if c in df.columns:
            v_parts.append(_scale_0_100(df[c], higher_is_better=True))
    out["F_Valuation_Expensive"] = pd.concat(v_parts, axis=1).mean(axis=1) if v_parts else np.nan

    # Capital / Human intensity as "structure" axes (not good/bad)
    out["F_Capital_Intensity_HW"] = _scale_0_100(df["CapExOverRev_Avg"], higher_is_better=True) if "CapExOverRev_Avg" in df.columns else np.nan
    out["F_Human_Intensity_SW"] = _scale_0_100(df["SBCOverRev_Avg"], higher_is_better=True) if "SBCOverRev_Avg" in df.columns else np.nan

    # Attach raw reference values (useful for debugging / trust)
    for c in ["Score_CYCLE","Score_SW","Score_HW","MktCap_B_Latest",
              "EBITMargin_Avg","FCFMargin_Avg","CapExOverRev_Avg","SBCOverRev_Avg",
              "PS_Latest","MC/FCF","MC/OwnerFCF","PEG_PS_Latest","Rev_CAGR"]:
        if c in df.columns and c not in out.columns:
            out[c] = df[c]

    
    # ------------------------------------------------------------------
    # Robust fallback for sparse inputs (SEC-only drafts often miss FCF/CapEx/SBC).
    # If a factor cannot be computed, fall back to a simpler proxy and mark it.
    # This keeps plots from becoming empty while making the limitation explicit.
    # ------------------------------------------------------------------
    out["_ImputedFlags"] = ""

    # Business Strength / Quality: if missing, fall back to EBIT margin percentile (profitability proxy)
    if "F_Quality" in out.columns:
        miss = out["F_Quality"].isna()
        if miss.any() and ("EBITMargin_Avg" in out.columns):
            proxy_arr = _scale_0_100(out["EBITMargin_Avg"], higher_is_better=True)
            proxy = pd.Series(proxy_arr, index=out.index)
            out.loc[miss, "F_Quality"] = proxy.loc[miss]
            out.loc[miss, "_ImputedFlags"] = out.loc[miss, "_ImputedFlags"] + "Quality<-EBITMargin;"

    # Scores: if missing, put 50 (neutral) so 2D map still shows the ticker.
    for sc in ["Score_SW", "Score_HW", "Score_CYCLE"]:
        if sc in out.columns:
            miss = out[sc].isna()
            if miss.any():
                out.loc[miss, sc] = 50.0
                out.loc[miss, "_ImputedFlags"] = out.loc[miss, "_ImputedFlags"] + f"{sc}<-50;"

    # Note: valuation stays NaN if market cap is missing (factor-map handles this via fallback).

    return out




def save_factor_map(df: pd.DataFrame, outpath: Path, color_by: str = "CYCLE") -> None:
    """Bubble map for quick positioning.

    Preferred (needs MarketCap):
      x = Valuation (Expensive, 0-100)  [requires market cap]
      y = Business Strength (0-100)
      color = Score_* chosen by --color-by
      size  = MarketCap

    Fallback (when valuation is unavailable for all tickers):
      x = Growth (Rev CAGR, 0-100)      [no market cap needed]
      y = Business Strength (0-100)
      color = Score_* chosen by --color-by
      size  = Revenue (latest), if available, else constant
    """
    d = df.copy()
    if d is None or d.empty:
        return

    # Need at least these
    for r in ["Ticker", "F_Quality"]:
        if r not in d.columns:
            return

    y = pd.to_numeric(d["F_Quality"], errors="coerce")

    # x: valuation if available, else growth
    x_val = pd.to_numeric(d.get("F_Valuation_Expensive", np.nan), errors="coerce")
    use_growth_x = bool(np.isfinite(x_val).sum() == 0)
    if use_growth_x:
        x = pd.to_numeric(d.get("F_Growth", np.nan), errors="coerce")
        x_label = "Growth (Rev CAGR) 0-100 (higher = faster growth)"
        title_prefix = "Factor Map (fallback): x=Growth, y=Business Strength"
    else:
        x = x_val
        x_label = "Valuation (Expensive) 0-100 (higher = more expensive)"
        title_prefix = "Factor Map: x=Valuation(Expensive), y=Business Strength"

    # color
    cb = (color_by or "CYCLE").upper()
    if cb == "SW" and "Score_SW" in d.columns:
        cvals = pd.to_numeric(d["Score_SW"], errors="coerce")
        c_label = "Score_SW (points)"
        title = f"{title_prefix}, color=SW"
    elif cb == "HW" and "Score_HW" in d.columns:
        cvals = pd.to_numeric(d["Score_HW"], errors="coerce")
        c_label = "Score_HW (points)"
        title = f"{title_prefix}, color=HW"
    elif cb == "CYCLE" and "Score_CYCLE" in d.columns:
        cvals = pd.to_numeric(d["Score_CYCLE"], errors="coerce")
        c_label = "Score_CYCLE (points)"
        title = f"{title_prefix}, color=CYCLE"
    else:
        cvals = pd.to_numeric(d.get("F_Growth", np.nan), errors="coerce")
        c_label = "F_Growth (0-100)"
        title = f"{title_prefix}, color=Growth"

    # size: market cap preferred; fallback to revenue/latest; else constant
    sizes = None
    if "MktCap_USD_Latest" in d.columns:
        mc = pd.to_numeric(d["MktCap_USD_Latest"], errors="coerce")
        if np.isfinite(mc).any():
            sizes = mc
    if sizes is None and "Revenue_Latest" in d.columns:
        rv = pd.to_numeric(d["Revenue_Latest"], errors="coerce")
        if np.isfinite(rv).any():
            sizes = rv
    if sizes is None:
        sizes = pd.Series([1.0] * len(d), index=d.index)

    # normalize sizes to a nice bubble range
    s = pd.to_numeric(sizes, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    if np.isfinite(s).any():
        smin, smax = float(np.nanmin(s)), float(np.nanmax(s))
        if smax > smin:
            s_norm = 200 + 800 * (s - smin) / (smax - smin)
        else:
            s_norm = pd.Series([500.0] * len(s), index=s.index)
    else:
        s_norm = pd.Series([500.0] * len(d), index=d.index)

    # filter rows with x/y present
    plot_df = d.copy()
    plot_df["_x"] = x
    plot_df["_y"] = y
    plot_df["_c"] = cvals
    plot_df["_s"] = s_norm
    plot_df = plot_df[np.isfinite(plot_df["_x"]) & np.isfinite(plot_df["_y"])].copy()
    if plot_df.empty:
        # Create an explanatory placeholder so the output is never silently blank.
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)
        ax.axis("off")
        msg = (
            "Factor map could not be plotted because required inputs are missing for all tickers.\n\n"
            "Needed at minimum:\n"
            "  - Revenue (latest) and EBIT (to compute profitability proxy)\n"
            "  - MarketCap (if you want Valuation-based x-axis)\n\n"
            "Fix options:\n"
            "  1) Provide Revenue_TV and Price/MarketCap via yy01 (--tv-csv / --add-market)\n"
            "  2) Or manually fill Revenue / Price / MarketCap in the input CSV\n"
        )
        ax.text(0.02, 0.98, msg, va="top", ha="left", fontsize=12)
        fig.tight_layout()
        fig.savefig(outpath, dpi=160)
        plt.close(fig)
        return

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    sc = ax.scatter(plot_df["_x"], plot_df["_y"], s=plot_df["_s"], c=plot_df["_c"])

    # labels
    for _, r in plot_df.iterrows():
        ax.text(float(r["_x"]) + 0.3, float(r["_y"]) + 0.3, str(r["Ticker"]), fontsize=10)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Business Strength 0-100 (profitability + resilience)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(c_label)

    # bubble note: place outside plot area (top-right), avoids overlap
    note = "Bubble size ~ Market Cap" if ("MktCap_USD_Latest" in d.columns and np.isfinite(pd.to_numeric(d["MktCap_USD_Latest"], errors="coerce")).any()) else "Bubble size ~ Revenue (fallback)"
    ax.text(1.02, 1.02, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)



def save_barh(df: pd.DataFrame, value_col: str, title: str, unit: str, outpath: Path):
    d = df[["Ticker", value_col]].copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna().sort_values(value_col, ascending=True)
    if d.empty:
        return
    plt.figure(figsize=(8.27, 11.69/3.2))
    plt.barh(d["Ticker"], d[value_col])
    plt.title(title, fontsize=10)
    #label_map = {"PFCF_Latest":"MC/FCF","POwnerFCF_Latest":"MC/OwnerFCF"}
    xlab = OUTPUT_LABEL_MAP.get(value_col, value_col)
    if unit:
        plt.xlabel(f"{xlab} ({unit})")
    else:
        plt.xlabel(xlab)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))
    plt.savefig(outpath, dpi=220)
    plt.close()


def write_glossary_full(outdir: Path):
    rows = []
    def add(jp, en, desc): rows.append((jp,en,desc))

    add("時価総額（MC）","Market Cap (MC)","企業全体の値段。MC=株価×発行株式数（本ツールではMktCap_Bを使用）")
    add("MC/Rev","MC/Rev (Market Cap to Revenue)","MC/Rev=MC/Revenue。売上1ドルをいくらで買っているか（alias: PS, PS_Latest）")
    add("PFCF","MC/FCF (Price to Free Cash Flow)","PFCF=MC/FCF。FCFが0以下なら倍率の意味が崩れるので空欄")
    add("POwnerFCF","MC/OwnerFCF (Price to Owner FCF)","POwnerFCF=MC/(FCF-SBC)。株主に残る現金で評価。<=0は空欄")

    add("Ticker","Ticker","銘柄コード")
    add("Company","Company","会社名")
    add("Year","Year","年度（FY）")
    add("MaxYear","MaxYear","入力データの最新年")
    add("WindowYears","WindowYears","1サイクル=この年数。平均/最悪/成長をこの窓で計算")
    add("ActualWindowYears","ActualWindowYears","実際に計算に使われた年数（intersectionなら全銘柄同一、per-tickerなら銘柄ごとに異なる）")
    add("WindowStartYear","WindowStartYear","窓の開始年（MaxYear-(WindowYears-1)）")

    add("Revenue","Revenue","売上（Billion USD）")
    add("EBIT","EBIT","利息・税引前利益（本業利益に近い）")
    add("FCF","FCF (Free Cash Flow)","FCF=営業CF−設備投資。自由に使える現金の増減")
    add("CapEx","CapEx (Capital Expenditures)","設備投資（現金支出）")
    add("SBC","SBC (Stock-Based Compensation)","株式報酬（株主の取り分が薄まるコスト）")
    add("OwnerFCF","Owner FCF","OwnerFCF=FCF−SBC（株主に実質残る現金）")

    add("Revenue_Avg","Revenue_Avg","WindowYears内の売上平均")
    add("EBIT_Avg","EBIT_Avg","WindowYears内のEBIT平均")
    add("FCF_Avg","FCF_Avg","WindowYears内のFCF平均")

    add("EBIT/Rev_Avg","EBIT/Rev_Avg","EBIT/Rev（平均）。本ツール内部列名 alias: EBITMargin_Avg")
    add("EBIT/Rev_Min","EBIT/Rev_Min","EBIT/Rev（最悪）。不況耐性の目安。alias: EBITMargin_Min")
    add("FCF/Rev_Avg","FCF/Rev_Avg","FCF/Rev（平均）。alias: FCFMargin_Avg")
    add("FCF/Rev_Min","FCF/Rev_Min","FCF/Rev（最悪）。不況耐性の目安。alias: FCFMargin_Min")

    add("CapEx/Rev_Avg","CapEx/Rev_Avg","CapEx/Rev（平均）。装備コストの重さの目安。alias: CapExOverRev_Avg")
    add("SBC/Rev_Avg","SBC/Rev_Avg","SBC/Rev（平均）。株式報酬の重さの目安。alias: SBCOverRev_Avg")

    add("Rev_CAGR","Rev_CAGR","売上CAGR（start/end>0のみ）")
    add("FCF_CAGR","FCF_CAGR","FCF CAGR（start/end>0のみ）")

    add("Price_Latest","Price_Latest","最新年の株価（任意入力）")
    add("MktCap_B_Latest","MktCap_B_Latest","最新年の時価総額（Billion）。TradingViewのMarket Capを推奨")
    add("PS_Latest","PS_Latest","最新年: MC/Rev（出力表記はMC/Rev。alias吸収）")
    add("PFCF_Latest","PFCF_Latest","最新年: MC/FCF（出力表記はMC/FCF。FCF<=0は空欄）")
    add("POwnerFCF_Latest","POwnerFCF_Latest","最新年: MC/OwnerFCF=MC/(FCF-SBC)（出力表記はMC/OwnerFCF。<=0は空欄）")

    add("Profile","Profile","評価軸（CYCLE / SW / HW）")
    add("CYCLE","CYCLE profile","景気循環（波）の中で戦いやすい体質かを見る軸。耐性（EBIT/Rev_Min）や現金化（FCF/Rev）などを総合して相対評価する（投資判断そのものではない）")
    add("SW","SW profile","ソフト寄り（軽装備・人/ソフトでスケールしやすい）体質を見る軸。CapEx/Rev（低いほどSW寄り）、SBC/Rev（高いほどSW寄り）と成長（Rev_CAGR）を中心に相対評価する。※良し悪しではなく体質の可視化")
    add("HW","HW profile","ハード寄り（重装備・資本集約）体質を見る軸。CapEx/Rev（高いほどHW寄り）を中心に、不況時の耐性（EBIT/Rev_Min, FCF/Rev_Min）も合わせて相対評価する。※良し悪しではなく体質の可視化")

    # ---- Factor Map (案A) ----
    add("Valuation(Expensive)","Valuation (Expensive)","割高さの軸（0-100）。本ツールでは『PS(MC/Rev)』『PFCF(MC/FCF)』『PEG_PS』などを0-100に正規化して合成（高い=割高寄り）。値は“優劣”ではなく、価格が織り込んでいる期待の強さを表す")
    add("Quality","Quality", "稼ぐ力+耐性の軸（0-100）。本ツールでは『EBIT/Rev_Avg』『FCF/Rev_Avg』『EBIT/Rev_Min』『FCF/Rev_Min』を0-100に正規化して合成（高い=収益性/耐性が強い）")
    add("Factor Map","Factor Map","散布図: x=Valuation(Expensive), y=Quality。色=第3軸（デフォルトはCYCLE）、点サイズ=時価総額。『割高だが強い/安いが弱い/体質違い』を1枚で俯瞰する")
    add("Score_0to100","Score_0to100","総合点（0-100）。出力では整数")
    add("Rating","Rating","A〜E評価（Scoreを丸め）")
    add("S_xxx","S_xxx","各指標の順位スコア（1-5）")

    md = ["# Glossary (Full)\n",
          "| 日本語/Key | English | 説明 |",
          "|---|---|---|"]
    for jp,en,desc in rows:
        md.append(f"| {jp} | {en} | {desc} |")
    write_utf8_lf(outdir/"a2_glossary.md", "\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--input", required=True)
    ap.add_argument("-w","--window-years", type=int, default=5)
    ap.add_argument("--window-mode", default="intersection", choices=["intersection","per-ticker"], help="Window alignment: intersection (common shortest history) or per-ticker (use -w per ticker)")
    ap.add_argument("-o","--outdir", default="res_stock_funda")
    ap.add_argument("--color-by", default="CYCLE", choices=["CYCLE","SW","HW"], help="Color axis for factor map")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir/"charts").mkdir(parents=True, exist_ok=True)

    df = read_input_csv(Path(args.input))

    # ---- Window alignment (intersection / per-ticker)
    # Available history length per ticker (count of distinct Year values)
    available_years = {}
    for tkr, g in df.groupby("Ticker"):
        yrs = [int(y) for y in g["Year"].dropna().unique().tolist()]
        yrs = sorted(set(yrs))
        available_years[tkr] = len(yrs)

    if len(available_years) == 0:
        raise SystemExit("No tickers found in input data.")

    if args.window_mode == "intersection":
        # True intersection: use the common calendar years shared by all tickers
        year_sets = {t: set(_compute_available_years(df, t)) for t in available_years.keys()}
        common_years = sorted(set.intersection(*year_sets.values())) if year_sets else []
        if len(common_years) == 0:
            raise SystemExit("No common (intersection) years found across tickers.")
        selected_years = common_years[-int(args.window_years):]
        df = df[df["Year"].isin(selected_years)].copy()
        common_wy = int(len(selected_years))
        window_years_by_ticker = {t: common_wy for t in available_years.keys()}
    else:
        # Per-ticker window = min(--window-years, available years for that ticker)
        window_years_by_ticker = {t: int(min(args.window_years, n)) for t, n in available_years.items()}
    metrics = compute_metrics(df, window_years=int(args.window_years), window_years_by_ticker=window_years_by_ticker)

    sc_cycle = build_scorecard(metrics, "cycle")
    sc_sw = build_scorecard(metrics, "sw")
    sc_hw = build_scorecard(metrics, "hw")
    scorecards = pd.concat([sc_cycle, sc_sw, sc_hw], ignore_index=True)

    # pretty only
    metrics_pretty = df_pretty_metrics(metrics)
    scorecards_pretty = df_pretty_scorecards(scorecards)
    metrics_pretty = rename_output_columns(metrics_pretty)
    scorecards_pretty = rename_output_columns(scorecards_pretty)
    metrics_pretty.to_csv(outdir/"metrics_pretty.csv", index=False, encoding="utf-8", lineterminator="\n")
    scorecards_pretty.to_csv(outdir/"scorecards_pretty.csv", index=False, encoding="utf-8", lineterminator="\n")

    # summary A4
    summary = build_summary_A4(metrics, scorecards)
    summary_pretty = df_pretty_metrics(summary.copy())
    for c in ["Score_CYCLE","Score_SW","Score_HW"]:
        if c in summary_pretty.columns:
            summary_pretty[c] = summary_pretty[c].map(lambda x: _fmt_score_int(x) if x!="" else "")
    summary_pretty = rename_output_columns(summary_pretty)

    summary_pretty.to_csv(outdir/"summary_A4.csv", index=False, encoding="utf-8", lineterminator="\n")
    write_utf8_lf(outdir/"summary_A4.md", summary_pretty.to_markdown(index=False))

    ###--- debug ---
    #print('\n debug \n')
    #print("summary cols:", list(summary.columns))

    # summary2 (重要指標のみ・比較向け)
    summary2 = build_summary2(metrics, scorecards)
    summary2_pretty = df_pretty_metrics(summary2.copy())
    for c in ["Score_CYCLE","Score_SW","Score_HW"]:
        if c in summary2_pretty.columns:
            summary2_pretty[c] = summary2_pretty[c].map(lambda x: _fmt_score_int(x) if x!="" else "")
    summary2_pretty = rename_output_columns(summary2_pretty)

    summary2_pretty.to_csv(outdir/"00_summary.csv", index=False, encoding="utf-8", lineterminator="\n")
    write_utf8_lf(outdir/"00_summary.md", summary2_pretty.to_markdown(index=False))

    ## charts
    charts_base = summary.copy()

    save_barh(charts_base, "Score_CYCLE", "Score (CYCLE)", "points", outdir/"charts"/"00_score_cycle.png")
    save_barh(charts_base, "MktCap_B_Latest", "Market_Cap", "B USD", outdir/"charts"/"mc_marcket_cap.png")
    save_barh(charts_base, "FCFMargin_Avg", "FCF/Rev (Avg)", "ratio", outdir/"charts"/"fcf_rev_ratio.png")
    save_barh(charts_base, "CapExOverRev_Avg", "CapEx/Rev (Avg)", "ratio", outdir/"charts"/"capex_over_rev.png")
    save_barh(charts_base, "PS_Latest", "(MC / Rev.)","ratio" , outdir/"charts"/"ps_MC_Rev_ratio.png")
    save_barh(charts_base, "PEG_PS_Latest", "PEG-like: (MC / Rev.) / RevGrowth%", "ratio", outdir/"charts"/"01_peg_ps.png")
    save_barh(charts_base, "MC/FCF", "MC/FCF (x)", "ratio", outdir/"charts"/"02_pfcf_latest.png")
    save_barh(charts_base, "MC/OwnerFCF", "MC/OwnerFCF (x)", "ratio", outdir/"charts"/"03_MC_fcf_ratio.png")

    ## 
    #def save_scores_2d(df: pd.DataFrame, outpath: Path) -> None:
    save_scores_2d(charts_base, outdir/"charts"/"00_2d_score.png")


    # --- Factor aggregation (Plan A) ---
    factors = compute_factors(charts_base)
    factors.to_csv(outdir/"00_factors.csv", index=False, encoding="utf-8", lineterminator="\n")
    save_factor_map(factors, outdir/"charts"/"00_factor_map.png", color_by=args.color_by)

    # --- Raw metrics (percentile bars) and score breakdown ---
    save_raw_percentile_bars(summary, outdir/"charts")
    save_score_breakdown(metrics, outdir/"charts")

    ## glossary + README
    write_glossary_full(outdir)

    ## 以下でglosarry.md, readme.md を作成している。
    ## このままで修正しないこと。
    readme = f"""# financial_eval.py (COMPLETE v3)

    ## 何がわかるか（重要）
    - **summary_A4.csv** を見れば、主要指標だけで比較できます（A4相当）
    - **scorecards_pretty.csv** で、CYCLE/SW/HWのスコア（0-100）を比較できます
    - **charts/** で、重要指標の横棒グラフ比較ができます

    ## 入力CSV
    必須: Ticker, Company, Year, Revenue, EBIT, FCF
    任意: CapEx, SBC, Price, MktCap_B, SharesOut_B

    単位:
    - Revenue/EBIT/FCF/CapEx/SBC/MktCap_B: Billion USD（B）
    - Price: USD/株
    - SharesOut_B: Billion shares（B株）

    ## P（Price）とは何か
    本ツールの **P は株価ではなく時価総額（Market Cap）** です。
    - MC = Market Cap = Price * SharesOut
    （TradingViewのMarket Capを MktCap_B に入れる運用が最も簡単です）

    ## 定義（Latest=最新年）
    - PS_Latest        = MC / Rev.（売上）
    - PFCF_Latest      = MC / FCF（FCF<=0は空欄）
    - POwnerFCF_Latest = MC / (FCF - SBC)（<=0は空欄）

    ## 実行
    ```bash
    python3 financial_eval.py --input input.csv --window-years {args.window_years} --outdir out
    ```

    出力:
    - out/metrics_pretty.csv
    - out/scorecards_pretty.csv
    - out/summary_A4.csv
    - out/charts/*.png
    - out/glossary_full.md
    """
    write_utf8_lf(outdir/"00_README.md", readme)

    # Auto-sync definitions into README and glossary
    defs_md = _render_defs_md()
    _sync_md_autosection(outdir/"00_README.md", "AUTO:DEFS_BEGIN", "AUTO:DEFS_END", defs_md)
    _sync_md_autosection(outdir/"a2_glossary.md", "AUTO:DEFS_BEGIN", "AUTO:DEFS_END", defs_md)


    print(f"OK: outputs written to {outdir.resolve()}")


if __name__ == "__main__":
    main()


## mode lline : 折りたたみの設定でmarker {{{ }}} を使う。
# vim:set foldmethod=marker:
