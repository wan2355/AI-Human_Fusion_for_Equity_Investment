#!/usr/bin/env python3
# -*- coding: utf-8 -*-


## usage
#{{{
"""
v16i
Stress ranking から VIX を分離

変更後のロジック（要点）

まず 全銘柄分の df_total_all を作る
そこから VIX 行だけをマスク (ticker='^VIX' or name に「恐怖指数」)

VIX 以外を stress_v 降順でソート → ランキング表示

VIX は 別枠で 1 行＋“VIXレベル”コメントを表示
CSV 出力 (00_stress_rank.csv) では、
メイン銘柄を stress_v 降順
最後に VIX 行を付け足す（順位表からは分離）

VIX レベル分類（例）
if vix_close < 13:
    vix_level = "VIX_LOW_CALM"
elif vix_close < 20:
    vix_level = "VIX_NORMAL"
elif vix_close < 30:
    vix_level = "VIX_ELEVATED"
elif vix_close < 40:
    vix_level = "VIX_HIGH_RISK"
else:
    vix_level = "VIX_PANIC"

--------------------------
統合版 eq ダッシュボード
- モメンタムダッシュボード
- BUYシグナルスキャン
- ボリンジャーアラート＋チャート
を 1 本にまとめた版。

出力先:
  res_eq_check_all/{now}_eq_check_all/
    ├─ res_ticker_dashboard/
    ├─ res_bb/
    ├─ res_gold/           (別スクリプトから出力)
    └─ res_market_regime/  (別スクリプトから出力)


# ---------------------------------------------------------
# 評価基準サマリ
#
#
# 総合評価 stress_v (0.0〜1.0) を計算する。
#     使用する要素:
#       - stress_v:   0.0〜1.0 に正規化済みの市場ストレス指標
#       - zscore:     価格の偏り (3σ で 1.0 に正規化)
#       - dahs_v:     モメンタムダッシュボードのスコア (0〜5 → 0〜1)
#       - regime_v:   レジームスコア (0〜5 → 0〜1)
#       - shk_F:      ショックフラグ (NORMAL / WARN / SHOCK など)
#     これらを 0.0〜1.0 のスケールに揃え、欠損を除いて平均する。
#
#--------------------------------------------------------
# 1) stress_v (0.0〜1.0)
#    0.00〜0.20: CALM          … 非常に静かな地盤。通常運用。
#    0.20〜0.40: MILD_STRESS   … 軽いストレス。サイズ抑えめで新規可。
#    0.40〜0.60: ELEVATED      … 前震〜余震ゾーン。リスク縮小検討。
#    0.60〜0.80: HIGH_STRESS   … 明確なストレス。ポジ削減を優先。
#    0.80〜1.00: EXTREME_PANIC … パニック域。新規リスクオン原則禁止。
#
# 2) shk_act
#    "hold"       … 現状維持推奨。行動なし。
#    "take_profit"… 利確推奨。高値圏のとき多い。
#    "reduce"     … ポジ減らし推奨。
#    "watch"      … 経過観察（注意）。
#
# 3) shk_F
#    "NORMAL"     … ショックフラグなし。平常状態。
#    "ALERT"/"WARN" … 変動拡大。要注意。
#    "SHOCK"      … 大きなショック。異常揺れ。
#
# 4) shk_stat
#    "OK"             … 特段の異常なし。
#    "WATCH"          … 怪しい揺れ。数日監視。
#    "SHEDDING_IN_PROGRESS" … 需給悪化中。売り優勢。
#
# 5) zscore
#    |zscore| < 1.0 … 通常の揺れ。
#    1.0〜2.0      … やや偏った価格帯。利確・押し目検討帯。
#    2.0〜3.0      … かなりの偏り。逆張り候補 or クラッシュ予備軍。
#    >= 3.0        … 極端。大きなリバーサル／崩壊候補。
#
# 6) signal
#    "BUY"         … 順張り or 逆張りの買い候補。
#    "TAKE_PROFIT" … 高値圏。利確候補。
#    "HOLD"        … 特にアクション不要。
#    "AVOID"       … エントリ非推奨。
#
# 7) dahs_v
#    0〜5 の整数（大きいほど「シグナル強い」）。
#    4〜5 … 強いシグナル。注意度高。
#    2〜3 … 中立〜やや強い。
#    0〜1 … シグナル弱い／ほぼノイズ。
#
# 8) regime_v
#    0〜5 の想定（market regime ダッシュボードから）。
#    4〜5 … 明確なトレンド（強気 or 弱気）。
#    2〜3 … 過渡期。ノイズ多め。
#    0〜1 … レジーム不明瞭 or ボラ高・混沌。
# ---------------------------------------------------------

"""
#}}}

import os
import json
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from yy_yf_cache import fetch_yf_cached
import math

##  stpo font warnings
#
#{{{
import matplotlib.font_manager as fm
import matplotlib as mpl

# --- フォント設定（Linux / mac 共通のフォールバック付き） ---
try:
    path = "/usr/share/fonts/OTF/ipag.ttf"   # Linux 環境向け（存在しない場合もある）
    if os.path.exists(path):
        prop = fm.FontProperties(fname=path)
        mpl.rcParams['font.family'] = prop.get_name()
    else:
        raise FileNotFoundError(path)
except Exception:
    # mac やフォント未インストール環境では、一般的な日本語フォントにフォールバック
    mpl.rcParams['font.family'] = ['Hiragino Sans', 'IPAexGothic', 'IPAGothic', 'Arial Unicode MS', 'sans-serif']

#### mac (旧コードは上記 try/except に統合)
#{{{
## comment follows when use on Linux.
#mpl.rcParams['font.family'] = ['Hiragino Sans', 'Times New Roman']

## ref
### Trutype font
#mpl.rcParams["pdf.fonttype"] = 42  # TrueTypeベース埋め込み
#mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]  # 推奨順
##mpl.rcParams["font.family"] = "sans-serif"
##mpl.rcParams["font.family"] = "DejaVu Sans"
## }}}


#import matplotlib as mpl
import warnings
#warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=UserWarning)

## stop font waring of follows.
##  "RuntimeWarning: Glyph 19975 missing from current font."
import re
def ignore_specific_runtime_warning(message, category, filename, lineno, file=None, line=None):
    ## Customize the pattern as needed
    #if "Glyph" in str(message) and "missing from current font" in str(message):
    if "IPAGothic" in str(message) and "not found" in str(message):
        return
    # Else, process as usual
    return warnings.defaultaction

warnings.filterwarnings("ignore", category=RuntimeWarning)

## }}}


# =========================================================
# 共通ヘルパー（既存コードからのコピー）
# =========================================================

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(length, min_periods=length).mean()
    avg_loss = loss.rolling(length, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr_from_hlc(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_c = close.shift(1)
    tr = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def bollinger_from_close(close: pd.Series, window: int = 20):
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    up2 = ma + 2 * std
    dn2 = ma - 2 * std
    up3 = ma + 3 * std
    dn3 = ma - 3 * std
    return ma, up2, dn2, up3, dn3, std

def normalized_slope(series: pd.Series, bars: int) -> float:
    if len(series) < bars + 1:
        return np.nan
    end = series.iloc[-1]
    start = series.iloc[-(bars + 1)]
    base = series.iloc[-1]
    if pd.isna(end) or pd.isna(start) or pd.isna(base) or base == 0:
        return np.nan
    # 元スクリプトの考え方に合わせた %/bar
    return float(end - start) / float(base) * 100.0 / bars

def momentum_score(latest: dict) -> int:
    """yy_ticker_dashboard_05.py のロジックをそのまま流用"""
    score = 0
    if latest.get("ema10", np.nan) > latest.get("ema20", np.nan):
        score += 1
    if latest.get("slope10", 0) > 0:
        score += 1
    if latest.get("slope20", 0) > 0:
        score += 1
    if latest.get("rsi", 0) >= 55:
        score += 1
    if latest.get("rvol20", 0) >= 1.10:
        score += 1
    return score

def decide_signal(last: dict) -> tuple[str, str]:
    """
    yy_ticker_dashboard_05.py の decide_signal を簡略移植
    """
    c  = last.get('close')
    e10 = last.get('ema10'); e20 = last.get('ema20')
    r  = last.get('rsi');   rv = last.get('rvol20')
    s10= last.get('slope10'); s20 = last.get('slope20')
    up = last.get('bb_up'); dn = last.get('bb_dn')
    pr = last.get('prev_rsi')

    # LOSS_CUT
    if (e10 is not None and e20 is not None and e10 < e20) and (c < e20) \
       and (r is not None and r <= 40) and (rv is not None and rv >= 1.20) \
       and (s10 is not None and s10 < 0) and (s20 is not None and s20 < 0):
        return "LOSS_CUT", "ema10<ema20 & close<ema20 & rsi<=40 & rvol>=1.2 & slopes<0"

    # TAKE_PROFIT
    if (up is not None and not pd.isna(up) and c >= up*0.995) or \
       (r is not None and r >= 70 and rv is not None and rv >= 1.10):
        return "TAKE_PROFIT", "near +2sigma or rsi>=70 with vol"

    # BUY（順張り/押し目回復）
    if (e10 is not None and e20 is not None and e10 > e20) \
       and (s10 is not None and s10 > 0) and (s20 is not None and s20 > 0) \
       and (r is not None and r >= 45) and (c >= e20):
        return "BUY", "ema10>ema20 & slopes>0 & rsi>=45 & close>=ema20"
    if (r is not None and r >= 40) and (pr is not None and pr < 40) \
       and (e20 is not None and c >= e20*0.995):
        return "BUY", "rsi cross up from <40 & close~ema20"

    return "HOLD", "no clear edge; wait"

def classify_bb_status(last_close: float, ma20: float, std: float):
    """
    yy_bollinger_alert_report_03.py の classify_status を簡略移植
    """
    if not np.isfinite(last_close) or not np.isfinite(ma20) or not np.isfinite(std) or std == 0:
        return ("N/A", "hold", np.nan)
    z = (last_close - ma20) / std
    if z <= -3.0:
        return ("HARD(-3σ)", "trim_50", z)
    if z <= -2.8:
        return ("ACT(-2.8σ)", "trim_25_to_50", z)
    if z <= -2.0:
        return ("WARN(-2σ)", "monitor", z)
    return ("OK", "hold", z)


#----------------------------------
## detect_shock
#----------------------------------
def detect_shock(
    last_close: float,
    prev_close: float,
    last_atr: float,
    crash_pct: float = -0.08,
    crash_atr_mult: float = 2.0,
):
    """
    1日変化率とATRから「急変」を検出する簡易ロジック。
    戻り値: (shock_flag, shock_reason, pct_1d)

      shock_flag:
        "CRASH"  : 急落
        "NORMAL" : 急変なし
    """
    if not np.isfinite(last_close) or not np.isfinite(prev_close) or prev_close <= 0:
        return "NORMAL", "n/a", np.nan

    pct = (last_close / prev_close) - 1.0
    reasons = []

    # 絶対パーセント判定
    if pct <= crash_pct:
        reasons.append(f"pct {pct:.1%} <= {crash_pct:.1%}")

    # ATR倍率判定
    if np.isfinite(last_atr) and last_close > 0:
        atr_pct = last_atr / last_close
        if pct <= -crash_atr_mult * atr_pct:
            reasons.append(
                f"pct {pct:.1%} <= -{crash_atr_mult:.1f} * ATR% ({atr_pct:.1%})"
            )

    if reasons:
        return "CRASH", " & ".join(reasons), pct
    else:
        return "NORMAL", "no_shock", pct



# =========================================================
# BUYシグナル judge のミニ版（yy_scan_buy-signal.py より）
# =========================================================

# 固定パラメータ（必要なら config に逃がせます）
EMA_LEN = 20
SMA_LEN = 50
RSI_LEN = 14
PH_LEN = 20
ATR_BREAK = 0.5
VOL_MULT = 1.3
RSI_FLOOR = 45
NEED_SCORE = 3
NEED_SCORE_HB = 4

def build_daily_for_judge(df_daily: pd.DataFrame) -> pd.DataFrame:
    df = df_daily.copy()
    df["EMA20"] = ema(df["Close"], EMA_LEN)
    df["SMA50"] = sma(df["Close"], SMA_LEN)
    df["RSI14"] = rsi(df["Close"], RSI_LEN)
    df["VMA20"] = sma(df["Volume"], 20)
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["PH"] = df["High"].rolling(PH_LEN).max()
    df["ATR14"] = atr_from_hlc(df["High"], df["Low"], df["Close"], 14)
    return df.dropna()

def judge_buy_signal(df_daily: pd.DataFrame, symbol: str) -> dict:
    df = build_daily_for_judge(df_daily)
    if df.empty:
        return {"OK": False, "Score": 0}

    c     = float(df["Close"].iloc[-1])
    v     = float(df["Volume"].iloc[-1])
    ema20 = float(df["EMA20"].iloc[-1])
    sma50 = float(df["SMA50"].iloc[-1])
    rsi14 = float(df["RSI14"].iloc[-1])
    vma20 = float(df["VMA20"].iloc[-1])
    sma200 = float(df["SMA200"].iloc[-1])
    ph     = float(df["PH"].iloc[-1])
    atr    = float(df["ATR14"].iloc[-1])
    sma50_prev = float(df["SMA50"].iloc[-6]) if len(df) >= 6 else sma50

    c1 = (c > ema20) and (c > sma50)
    c2 = (v > vma20 * VOL_MULT)
    c3 = (rsi14 >= RSI_FLOOR)
    c4 = (sma50 > sma50_prev) and (c > sma200)
    c5 = (c > ph + atr * ATR_BREAK)

    score = int(c1) + int(c2) + int(c3) + int(c4) + int(c5)
    high_beta = symbol in ("PLTR", "HOOD", "IREN", "METC")
    need = NEED_SCORE_HB if high_beta else NEED_SCORE
    ok = (score >= need)

    return {
        "Price": int(c),
        "EMA20": int(ema20),
        "SMA50": int(sma50),
        "RSI14": int(rsi14),
        "Vol": int(v),
        "VMA20": int(vma20),
        "C1": bool(c1),
        "C2": bool(c2),
        "C3": bool(c3),
        "C4": bool(c4),
        "C5": bool(c5),
        "Score": int(score),
        "OK": bool(ok),
    }

# =========================================================
# 設定・共通ルート
# =========================================================

DEFAULT_CONFIG_PATH = Path("config/config_eq_dashboard.json")
#DEFAULT_CONFIG_PATH = Path("config_eq_dashboard.json")

def load_or_init_config(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    # なければ ticker_dashboard_05 のデフォルト相当で生成
    cfg = {
        "fetch": {
            "interval": "1h",
            "period": "1mo",
            "bb_months": 3
        },
        "tickers": {
            "5713.T": "住友鉱山",
            "6339.T": "新東工業",
            "^N225": "日経225先物",
            "^GSPC": "SP500",
            "SPY": "SPY",
            "^NDX": "NASDAQ100",
            "^RUT": "Russell_2000",
            "GLDM": "Gold",
            "1540.T": "Gold",
            "2914.T": "JT",
            "8058.T": "三菱商事",
            "8316.T": "三井住友F",
            "1496.T": "USD投資適格社債ETF(H)",
            "2866.T": "GX米国優先証券",
            "1329.T": "日経225ETF",
            "1489.T": "日経平均高配当ETF",
            "IREN": "IREN",
            "JEPQ": "JEPQ",
            "NVDA": "Nvidia",
            "MSFT": "MSFT"
        },
        "scan_buy_signal": {"enabled": True},
        "bollinger": {"enabled": True, "rsi_warn": 35, "rsi_hard": 30, "earnings_window": 5},
        "hy_vix": {"enabled": True}
    }
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INIT] Default config saved to {path}")
    return cfg

def make_root_dirs(now: str):
    root = Path("res_eq_check_all") / f"{now}_eq_check_all"
    dir_td = root / "res_ticker_dashboard"
    dir_bb = root / "res_bb"
    dir_gold = root / "res_gold"
    dir_mr = root / "res_market_regime"
    for d in (dir_td, dir_bb, dir_gold, dir_mr):
        d.mkdir(parents=True, exist_ok=True)
    return root, dir_td, dir_bb, dir_gold, dir_mr

# =========================================================
# 各ティッカーの処理
# =========================================================
# {{{
def fetch_intraday_df(ticker: str, interval: str, period: str) -> pd.DataFrame:
    df = fetch_yf_cached(ticker, period=period, interval=interval, max_age_days=1)
    if df is None or df.empty:
        raise RuntimeError(f"{ticker}: no data (intraday)")
    # 既に yy_yf_cache 側で index を tz-naive DatetimeIndex にしている前提
    df = df.rename(columns=str.lower)
    # 必須列チェック
    for col in ("close", "high", "low", "volume", "open"):
        if col not in df.columns:
            raise KeyError(f"{ticker}: '{col}' 列不足 (intraday)")
    return df

def fetch_daily_df(ticker: str, months: int = 6) -> pd.DataFrame:
    # だいたい months*31 日分
    period_days = int(months * 31)
    df = fetch_yf_cached(ticker, period=f"{period_days}d", interval="1d", max_age_days=1)
    if df is None or df.empty:
        raise RuntimeError(f"{ticker}: no data (daily)")
    # ボリンジャー/BUY用に Yahoo形式の列名をそのまま使う
    return df

def compute_momentum_block(df_intraday: pd.DataFrame) -> dict:
    df = df_intraday.copy()
    close = df["close"]
    vol = df["volume"]
    df["ema10"] = ema(close, 10)
    df["ema20"] = ema(close, 20)
    df["rsi14"] = rsi(close, 14)
    vma20 = sma(vol, 20)
    df["rvol20"] = vol / vma20
    ma, up2, dn2, _, _, _ = bollinger_from_close(close, 20)
    df["bb_up"] = up2
    df["bb_dn"] = dn2
    df["slope10"] = ema(close, 10).rolling(11).apply(lambda s: normalized_slope(s, 10), raw=False)
    df["slope20"] = ema(close, 20).rolling(21).apply(lambda s: normalized_slope(s, 20), raw=False)
    df["prev_rsi"] = df["rsi14"].shift(1)

    last = df.iloc[-1]
    latest = {
        "close": float(last["close"]),
        "ema10": float(last["ema10"]),
        "ema20": float(last["ema20"]),
        "slope10": float(last["slope10"]),
        "slope20": float(last["slope20"]),
        "rsi": float(last["rsi14"]),
        "rvol20": float(last["rvol20"]),
        "bb_up": float(last["bb_up"]) if not np.isnan(last["bb_up"]) else np.nan,
        "bb_dn": float(last["bb_dn"]) if not np.isnan(last["bb_dn"]) else np.nan,
        "prev_rsi": float(last["prev_rsi"]) if not np.isnan(last["prev_rsi"]) else np.nan,
    }
    sig, reason = decide_signal(latest)
    latest["signal"] = sig
    latest["reason"] = reason
    latest["score"] = momentum_score(latest)
    return df, latest

def plot_momentum_ticker(df: pd.DataFrame, ticker: str, outdir: Path):
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df.index, df["close"], label="close", linewidth=1.2)
    if "ema10" in df.columns:
        ax1.plot(df.index, df["ema10"], label="ema10", linestyle="--", linewidth=1.0)
    if "ema20" in df.columns:
        ax1.plot(df.index, df["ema20"], label="ema20", linestyle="--", linewidth=1.0)
    ax1.legend(loc="upper left")
    ax1.set_title(ticker)
    fig.tight_layout()
    fn = outdir / f"{ticker}_momentum.png"
    fig.savefig(fn)
    plt.close(fig)

def plot_bollinger_chart(df_daily: pd.DataFrame, symbol: str, label: str, outdir: Path,
                         last_ma: float, last_std: float, last_rsi: float, z: float):
    close = df_daily["Close"]
    ma20, up2, dn2, up3, dn3, _ = bollinger_from_close(close, 20)

    fig = plt.subplots(figsize=(11, 6))[0]
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(close.index, close.values, label=f"{symbol} Close", linewidth=1.4)
    ax1.plot(ma20.index, ma20.values, linestyle="--", linewidth=1.2, label="MA20")
    ax1.plot(up2.index, up2.values, linestyle=":", linewidth=1.0, label="+2σ")
    ax1.plot(dn2.index, dn2.values, linestyle=":", linewidth=1.0, label="-2σ")
    ax1.plot(up3.index, up3.values, linestyle="--", linewidth=0.8, label="+3σ")
    ax1.plot(dn3.index, dn3.values, linestyle="--", linewidth=0.8, label="-3σ")
    ax1.legend(loc="upper left")
    ax1.set_title(f"{symbol} ({label})")

    ax2 = fig.add_subplot(2, 1, 2)
    rsi14 = rsi(close, 14)
    ax2.plot(rsi14.index, rsi14.values, label="RSI(14)", linewidth=1.0)
    ax2.axhline(70, color="gray", linestyle="--", linewidth=0.8)
    ax2.axhline(30, color="gray", linestyle="--", linewidth=0.8)
    ax2.legend(loc="upper left")

    fig.tight_layout()
    fn = outdir / f"{symbol}_bb.png"
    fig.savefig(fn)
    plt.close(fig)
# }}}



#----------------------------------
## data merge with sashboard
#----------------------------------
#{{{
# =========================================================
# stress_v 評価ヘルパ
#
# stress_v:
#   zscore, rsi, rvol20, regime_v から 0.0〜1.0 に正規化した
#   「市場ストレス」の指標。
#
# stress_v_rank:
#   stress_v に加えて、
#   zscore, dahs_v, regime_v, shk_F などを組み合わせた
#   総合評価（0.0〜1.0）。
# =========================================================

def compute_stress_score(row: pd.Series) -> float:
    vals = []

    # 1) ボリンジャー zscore（下振れのみストレスとして評価）
    z = row.get("zscore", np.nan)
    if np.isfinite(z) and z < 0:
        vals.append(min(abs(z) / 3.0, 1.0))

    # 2) RSI の偏り（50 からの乖離）
    rsi_val = row.get("rsi", np.nan)
    if np.isfinite(rsi_val):
        vals.append(min(abs(rsi_val - 50.0) / 50.0, 1.0))

    # 3) 出来高の相対倍率 rvol20
    rvol = row.get("rvol20", np.nan)
    if np.isfinite(rvol) and rvol > 1.0:
        vals.append(min((rvol - 1.0) / 2.0, 1.0))

    # 4) regime スコア
    sc_reg = row.get("regime_v", np.nan)
    if np.isfinite(sc_reg):
        vals.append(min(max(sc_reg, 0.0) / 5.0, 1.0))

    if not vals:
        return np.nan
    return float(sum(vals) / len(vals))


def merge_with_regime(now_ts: str, root: Path):
    td_dir = root / "res_ticker_dashboard"
    mr_dir = root / "res_market_regime"
    bb_dir = root / "res_bb"

    # 各CSVのパス
    dash_csv = td_dir / f"00_momentum_summary_{now_ts}.csv"
    reg_csv  = mr_dir / f"00_res_market_regime_{now_ts}.csv"
    bb_csv   = bb_dir / f"00_bb_alerts_{now_ts}.csv"

    if not dash_csv.exists():
        print(f"[WARN] merge: {dash_csv} が見つかりません。")
        return
    if not reg_csv.exists():
        print(f"[WARN] merge: {reg_csv} が見つかりません。")
        return

    dash = pd.read_csv(dash_csv)
    reg  = pd.read_csv(reg_csv)

    # ---- カラム名を標準化（dashboard側）----
    dash = dash.rename(columns={
        "score": "dahs_v",
        "bb_up": "bb_2sigma_up",
        "bb_dn": "bb_2sigma_dn",
    })

    # ---- regime 側 ----
    if "ticker" not in reg.columns:
        reg = reg.reset_index()

    reg = reg.rename(columns={
        "score": "regime_v",
        "last":  "close_regime",   # 必要なら使う
    })

    # regime_v (score) の降順に並べ替え
    if "regime_v" in reg.columns:
        reg = reg.sort_values("regime_v", ascending=False).reset_index(drop=True)

    # ---- regime サブセット ----
    reg_cols = [
        "ticker",
        "regime_v",
        "vwap20",
        "vp_poc",
        "rsi_zone",
        "atr14",
        #"ema10",
        #"ema20",
        "sma50",
        "sma200",
        # "close_regime",
    ]
    reg_sub = reg[[c for c in reg_cols if c in reg.columns]].copy()

    # dashboard × regime
    merged = dash.merge(reg_sub, on="ticker", how="left")

    # ---- ボリンジャー急変情報（shk_*）をマージ ----
    if bb_csv.exists():
        bb = pd.read_csv(bb_csv)

        # ticker, status, action, zscore だけ利用
        bb = bb.rename(columns={
            #"symbol": "ticker",
            "shock_flag": "shk_F",
            "status": "shk_stat",
            "action": "shk_act",
            "shock_reason": "shk_R",
            "pct_1d": "pct_1d",
        })
        #bb_cols = ["ticker", "shk_F", "shk_stat", "shk_act", "zscore"]
        bb_cols = ["ticker", "shk_R", "shk_F", "shk_stat", "shk_act", "zscore", "pct_1d" ]
        bb_sub = bb[[c for c in bb_cols if c in bb.columns]].copy()

        merged = merged.merge(bb_sub, on="ticker", how="left")
    else:
        print(f"[WARN] merge: {bb_csv} が見つかりません（shk_* は NaN のまま）。")

    # ---- stress_score / stress_v_rank を計算 ----
    try:
        merged["stress_v"] = merged.apply(compute_stress_score, axis=1)
    except Exception as e:
        print(f"[WARN] stress_v calc failed: {e}")
        merged["stress_v"] = np.nan

    # ---- 最終的な列順を揃える ----
    final_cols = [
        "ticker",
        "stress_v",
        "shk_act",   # ← 追加
        "shk_F",   # ← 追加
        "shk_stat",   # ← 追加
        "zscore",       # ← 追加
        "signal",
        "dahs_v",
        "regime_v",
        "close",
        # "close_regime",
        "vwap20",
        "vp_poc",
        "pct_1d",
        "rsi",
        "rsi_zone",
        "atr14",
        "ema10",
        "ema20",
        "slope10",
        "slope20",
        "sma50",
        "sma200",
        "rvol20",
        "bb_2sigma_up",
        "bb_2sigma_dn",
        "reason",
        'shk_R',
        "name",
    ]
    merged = merged.reindex(columns=final_cols)

    out_csv = td_dir / f"02_res_merge__dashB_regime_{now_ts}.csv"
    merged_out = merged.copy()
    if "stress_v" in merged_out.columns:
        merged_out["stress_v"] = merged_out["stress_v"].round(4)
    merged_out.to_csv(out_csv, index=False, encoding="utf-8")

    # ---- 画面表示用（Unified view short）----
    # ご指定どおり「status まで」＝ shk_status だけ表示
    view_cols = [
        "ticker",
        "shk_F",   # ← 追加
        "shk_act",   # ← 追加
        "shk_stat",   # ← ここだけ出す
        "signal",
        #"dahs_v",
        "pct_1d",
        #"regime_v",
        #"close",
        #"rsi",
        #"atr14",
        #"ema10",
        #"ema20",
        #"slope10",
        #"slope20",
        #"sma50",
        #"sma200",
        #"rvol20",
        #"bb_2sigma_up",
        #"bb_2sigma_dn",
        "reason",
        'shk_R',
        "name",
    ]
    print("\n=== Unified view (check shock) ===\n")
    #print()
    print(merged[view_cols].to_string(index=False))
    print()

    #-----------------------------------------------------
    # ---- 画面表示用2（Unified view）----
    # ご指定どおり「status まで」＝ shk_status だけ表示
    view_cols2 = [
        "ticker",
        #"shk_F",   # ← 追加
        #"shk_act",   # ← 追加
        "shk_stat",   # ← ここだけ出す
        "signal",
        "dahs_v",
        "regime_v",
        "close",
        "rsi",
        #"atr14",
        #"ema10",
        "ema20",
        #"slope10",
        #"slope20",
        "sma50",
        #"sma200",
        "rvol20",
        #"bb_2sigma_up",
        #"bb_2sigma_dn",
        #"reason",
        #'shk_R',
        "name",
    ]
    print("\n=== Unified view (check normal status) ===\n")
    #print()
    print(merged[view_cols2].to_string(index=False))
    print()

    #-----------------------------------------------------
    # ---- 総合評価ビュー（stress_v_rank） ----
    total_cols = [
        "ticker",
        "close",
        "stress_v",
        "shk_act",
        "shk_F",
        "shk_stat",
        "zscore",
        "signal",
        "dahs_v",
        "regime_v",
        "vwap20",
        "vp_poc",
        "pct_1d",
        "rsi",
        #"rsi_zone",
        "name",
    ]
    # VIX を他銘柄と分離して扱うため、まず全体をコピー
    df_total_all = merged[total_cols].copy()

    # VIX 行（ticker='^VIX' または name に「恐怖指数」を含む）を特定
    if "ticker" in df_total_all.columns:
        vix_mask_total = df_total_all["ticker"].astype(str).str.upper().eq("^VIX")
    else:
        vix_mask_total = pd.Series(False, index=df_total_all.index)
    if "name" in df_total_all.columns:
        vix_mask_total = vix_mask_total | df_total_all["name"].astype(str).str.contains("恐怖指数", na=False)

    df_vix = df_total_all[vix_mask_total].copy()
    df_total = df_total_all[~vix_mask_total].copy()

    # stress_v の大きい順にソート（VIX 以外）
    if "stress_v" in df_total.columns:
        df_total = df_total.sort_values("stress_v", ascending=False)

    # 画面表示は stress_v=4桁に整形
    if "stress_v" in df_total.columns:
        df_total["stress_v"] = df_total["stress_v"].map(
            lambda x: f"{x:.4f}" if pd.notna(x) else ""
        )

    print("\n=== Stress ranking (summary, excl. VIX) ===\n")
    print(df_total.to_string(index=False))
    print()

    # VIX は別枠で表示し、レベルを簡易分類
    if not df_vix.empty:
        vix_row = df_vix.iloc[0]
        vix_close = vix_row.get("close", float("nan"))
        vix_level = ""
        if isinstance(vix_close, (int, float)) and math.isfinite(vix_close):
            # VIX フェーズ定義（16 / 20 / 25 / 30）
            #   <16      : VIX_LOW_CALM
            #   16–20    : VIX_NORMAL
            #   20–25    : VIX_ELEVATED
            #   25–30    : VIX_HIGH_RISK
            #   >=30     : VIX_PANIC
            if vix_close < 16:
                vix_level = "VIX_LOW_CALM"
            elif vix_close < 20:
                vix_level = "VIX_NORMAL"
            elif vix_close < 25:
                vix_level = "VIX_ELEVATED"
            elif vix_close < 30:
                vix_level = "VIX_HIGH_RISK"
            else:
                vix_level = "VIX_PANIC"

        # VIX の stress_v も他銘柄と同様に小数第4位で丸める
        if "stress_v" in df_vix.columns:
            df_vix["stress_v"] = df_vix["stress_v"].round(4)

        view_cols_vix = [c for c in ["ticker","close","stress_v","shk_act","shk_F","shk_stat","zscore","signal","pct_1d","rsi","name"] if c in df_vix.columns]
        # VIX の画面表示は関数の最後でまとめて行う
        lines = []
        lines.append("\n=== VIX status (separate) ===\n")
        lines.append(df_vix[view_cols_vix].to_string(index=False) + "\n")
        if vix_level:
            try:
                vc_str = f"{float(vix_close):.2f}"
            except Exception:
                vc_str = str(vix_close)
            lines.append(f"\n[VIX_LEVEL] {vix_level} (close={vc_str})\n")
        vix_status_text = "".join(lines)

        # VIX 結果を個別 CSV として保存
        try:
            vix_out = df_vix.copy()
            if vix_level:
                vix_out["vix_level"] = vix_level
            vix_csv = root / f"vix_{now_ts}.csv"
            vix_out.to_csv(vix_csv, index=False, encoding="utf-8")
        except Exception as e:
            print(f"[WARN] cannot write VIX csv: {e}")


    # CSV 00_stress_rank.csv として root 直下に保存
    out_total = root / "00_stress_rank.csv"
    df_csv_all = merged[total_cols].copy()

    if "ticker" in df_csv_all.columns:
        vix_mask_csv = df_csv_all["ticker"].astype(str).str.upper().eq("^VIX")
    else:
        vix_mask_csv = pd.Series(False, index=df_csv_all.index)
    if "name" in df_csv_all.columns:
        vix_mask_csv = vix_mask_csv | df_csv_all["name"].astype(str).str.contains("恐怖指数", na=False)

    df_csv_main = df_csv_all[~vix_mask_csv].copy()
    df_csv_vix  = df_csv_all[vix_mask_csv].copy()

    # stress_v の大きい順にソート（VIX 以外）
    if "stress_v" in df_csv_main.columns:
        df_csv_main = df_csv_main.sort_values("stress_v", ascending=False)
        df_csv_main["stress_v"] = df_csv_main["stress_v"].round(4)

    # VIX 行は最後に付ける（順位比較から分離）
    df_csv = pd.concat([df_csv_main, df_csv_vix], ignore_index=True)
    df_csv.to_csv(out_total, index=False, encoding="utf-8")

    # -----------------------------------------------------
    # market_regime と同じ銘柄だけを対象にした 3 つのCSVを作成
    try:
        mr_dir = root / "res_market_regime"
        reg_csv = mr_dir / f"00_res_market_regime_{now_ts}.csv"
        if reg_csv.exists():
            df_reg_all = pd.read_csv(reg_csv)

            # regime 側のティッカー
            reg_tickers = set()
            if "ticker" in df_reg_all.columns:
                reg_tickers = set(df_reg_all["ticker"].dropna().astype(str))

            # stress_v 側から該当銘柄のみ抽出
            df_total_sub = df_csv[df_csv["ticker"].astype(str).isin(reg_tickers)].copy()

            # regime 側も同じ銘柄に絞る
            df_reg_sub = df_reg_all[df_reg_all["ticker"].astype(str).isin(reg_tickers)].copy()
            # score の降順に並べ替え（regime の強い順）
            if "score" in df_reg_sub.columns:
                df_reg_sub = df_reg_sub.sort_values("score", ascending=False).reset_index(drop=True)

            # 画面表示用: regime スコアランキング（簡易ビュー）
            try:
                view_cols = [c for c in ["ticker", "role", "trend", "score", "pct", "last"] if c in df_reg_sub.columns]
                if view_cols:
                    print("\n=== Market regime (score ranking) ===\n")
                    print(df_reg_sub[view_cols].to_string(index=False))
                    print()
            except Exception as e:
                print(f"[WARN] cannot print market_regime view: {e}")

            # ダッシュボード元データ（最新 momentum summary）も同じ銘柄を抽出
            td_dir = root / "res_ticker_dashboard"
            dash_csv = td_dir / f"00_momentum_summary_{now_ts}.csv"
            if dash_csv.exists():
                df_dash_all = pd.read_csv(dash_csv)
                # rename 後と対応させるため ticker カラムがあればそのまま利用
                if "ticker" in df_dash_all.columns:
                    df_dash_sub = df_dash_all[df_dash_all["ticker"].astype(str).isin(reg_tickers)].copy()
                else:
                    df_dash_sub = df_dash_all.copy()
            else:
                df_dash_sub = pd.DataFrame()

            # 出力先: res_eq_check_all/{now}_eq_check_all/ 配下
            out_market_total = root / "00_market_stress_rank.csv"
            out_market_reg   = root / "01_market_regime.csv"
            out_market_dash  = root / "02_dashboard.csv"

            df_total_sub.to_csv(out_market_total, index=False, encoding="utf-8")
            df_reg_sub.to_csv(out_market_reg,   index=False, encoding="utf-8")
            if not df_dash_sub.empty:
                df_dash_sub.to_csv(out_market_dash, index=False, encoding="utf-8")

            # 3つの曲線 (regime total_score, stress_v平均, dashboardスコア合計) を 1 枚のグラフに
            try:
                # stress_v 側: 全銘柄の平均 stress_v を1点として扱う（VIX は除外）
                if not df_total_sub.empty:
                    df_avg_src = df_total_sub.copy()
                    try:
                        mask_vix = df_avg_src["ticker"].astype(str).str.upper().eq("^VIX")
                        if mask_vix.any():
                            df_avg_src = df_avg_src[~mask_vix]
                    except Exception:
                        pass
                    avg_stress_v = df_avg_src["stress_v"].mean()
                else:
                    avg_stress_v = float("nan")

                # regime 側: 可能なカラムから代表スコアを計算
                regime_score = float("nan")
                # 1) total_score カラムがあれば最優先（全体レジームスコア）
                if "total_score" in df_reg_all.columns:
                    try:
                        regime_score = float(df_reg_all["total_score"].iloc[0])
                    except Exception:
                        pass
                # 2) なければ score カラムの平均（regime 対象銘柄のスコア平均）
                if not math.isfinite(regime_score):
                    if "score" in df_reg_sub.columns:
                        try:
                            regime_score = float(df_reg_sub["score"].mean())
                        except Exception:
                            pass
                # 3) それも無ければ regime_v の平均（将来の拡張用）
                if not math.isfinite(regime_score) and "regime_v" in df_reg_sub.columns:
                    try:
                        regime_score = float(df_reg_sub["regime_v"].mean())
                    except Exception:
                        pass

                # dashboard 側: dahs_v の平均（regime対象銘柄のモメンタム強度）
                if not df_dash_sub.empty and "score" in df_dash_sub.columns:
                    dash_score = df_dash_sub["score"].mean()
                elif not df_total_sub.empty and "dahs_v" in df_total_sub.columns:
                    dash_score = df_total_sub["dahs_v"].mean()
                else:
                    dash_score = float("nan")

                # 単一時点なので x 軸は [0] とする
                x = [0]
                plt.figure()
                plt.plot(x, [avg_stress_v], marker="o", label="stress_v_avg")
                plt.plot(x, [regime_score], marker="s", label="regime_score")
                plt.plot(x, [dash_score], marker="^", label="dashboard_score")

                plt.xlabel("time (this run)")
                plt.ylabel("score")
                plt.legend()
                plt.title(f"Market summary (now_ts={now_ts})")

                out_png = root / "00_market_stress_v_rank.png"
                plt.savefig(out_png, bbox_inches="tight")
                plt.close()

                # ---- 3指標の履歴を保存（stress_v_avg / regime_score / dashboard_score）----
                try:
                    hist_dir = root.parent / "history"
                    #hist_dir.mkdir(exist_ok=True)
                    hist_dir.mkdir(parents=True, exist_ok=True)

                    # 1) regime の銘柄別履歴（long 形式）
                    reg_hist_file = hist_dir / "market_regime_history_long.csv"
                    if not df_reg_all.empty and "ticker" in df_reg_all.columns:
                        df_reg_hist = df_reg_all.copy()
                        df_reg_hist["timestamp"] = now_ts
                        df_reg_hist.to_csv(
                            reg_hist_file,
                            mode="a",
                            index=False,
                            header=not reg_hist_file.exists(),
                            encoding="utf-8",
                        )

                    # 2) run ごとのサマリ履歴（3指標）
                    sum_hist_file = hist_dir / "market_summary_history.csv"
                    df_sum = pd.DataFrame(
                        {
                            "timestamp": [now_ts],
                            "stress_v_avg": [avg_stress_v],
                            "regime_score": [regime_score],
                            "dashboard_score": [dash_score],
                        }
                    )

                    ###251125
                    ### 既存ファイルがあれば読み込んで下に連結 → w で上書き保存
                    #if sum_hist_file.exists():
                    #    try:
                    #        df_old = pd.read_csv(sum_hist_file)
                    #        df_sum = pd.concat([df_old, df_sum], ignore_index=True)
                    #    except Exception:
                    #        # 壊れていた場合は今回分だけで作り直す
                    #        pass


                    df_sum.to_csv(
                        sum_hist_file,
                        mode="a",
                        #mode="w",
                        header=False,
                        index=False,
                        #header=not sum_hist_file.exists(),
                        encoding="utf-8",
                    )

                    # 3) 過去30回分の 3本線グラフを作成
                    try:
                        hist_df = pd.read_csv(sum_hist_file)
                        if not hist_df.empty and "timestamp" in hist_df.columns:
                            # 古い→新しい順に並べて、末尾30件
                            hist_df = hist_df.sort_values("timestamp")
                            hist_df = hist_df.tail(30)

                            ts = hist_df["timestamp"].tolist()
                            x_idx = list(range(len(ts)))

                            plt.figure()
                            plt.plot(
                                x_idx,
                                hist_df["stress_v_avg"],
                                marker="o",
                                linestyle="-",
                                label="stress_v_avg",
                            )
                            plt.plot(
                                x_idx,
                                hist_df["regime_score"],
                                marker="s",
                                linestyle="-",
                                label="regime_score",
                            )
                            plt.plot(
                                x_idx,
                                hist_df["dashboard_score"],
                                marker="^",
                                linestyle="-",
                                label="dashboard_score",
                            )
                            plt.xticks(x_idx, ts, rotation=45, ha="right")
                            plt.xlabel("timestamp (last 30 runs)")
                            plt.ylabel("score")
                            plt.title("Market summary (last 30 runs)")
                            plt.legend()
                            plt.tight_layout()
                            #out_hist_png = root / "01_market_summary_30runs.png"
                            out_hist_png = hist_dir / "01_market_summary_30runs.png"
                            plt.savefig(out_hist_png, bbox_inches="tight")
                            plt.close()
                            if vix_status_text:
                                print(vix_status_text)


                    except Exception as e:
                        print(f"[WARN] cannot generate 30-run history chart: {e}")

                except Exception as e:
                    print(f"[WARN] cannot update history files: {e}")

            except Exception as e:
                print(f"[WARN] cannot create market summary plot: {e}")

    except Exception as e:
        print(f"[WARN] cannot create market CSVs: {e}")

# }}}


# =========================================================
# メイン処理
# =========================================================

#{{{
def main():

    ap = argparse.ArgumentParser()
    ap.add_argument('-c',"--config", default=str(DEFAULT_CONFIG_PATH), help="config_eq_dashboard.json")
    args = ap.parse_args()

    cfg = load_or_init_config(Path(args.config))

    ## ここで共通タイムスタンプを決定して環境変数にも設定
    # すでに EQ_RUN_TS が環境変数で指定されていればそれを優先し、
    # 無ければ現在時刻から生成する。
    env_ts = os.environ.get("EQ_RUN_TS")
    if env_ts:
        now = env_ts
    else:
        now = dt.datetime.now().strftime("%y%m%d_%H%M")

    os.environ["EQ_RUN_TS"] = now

    root, dir_td, dir_bb, dir_gold, dir_mr = make_root_dirs(now)

    ##2511120
    ## --- この run の設定をルートにもコピーして残す ---
    cfg_copy_path = root / "config_eq_dashboard.json"
    try:
        cfg_copy_path.write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[WARN] cannot write config copy: {e}")


    print()
    print("-" * 80)
    print(now)
    print("Unified eq dashboard")
    print("-" * 80)
    print()

    interval = cfg.get("fetch", {}).get("interval", "1h")
    period   = cfg.get("fetch", {}).get("period", "1mo")
    bb_months = int(cfg.get("fetch", {}).get("bb_months", 3))

    tickers = cfg.get("tickers", {})

    rows_mom = []
    rows_buy = []
    rows_bb  = []

    for tk, name in tickers.items():
        #print(f"[FETCH] {tk} ({name}) interval={interval} period={period}")
        try:
            df_intraday = fetch_intraday_df(tk, interval, period)
        except Exception as e:
            print(f"[WARN] intraday {tk}: {e}")
            continue

        # --- モメンタム ---
        try:
            df_mom, latest = compute_momentum_block(df_intraday)
        except Exception as e:
            print(f"[WARN] momentum {tk}: {e}")
            continue

        plot_momentum_ticker(df_mom.tail(300), tk, dir_td)
        row_m = {"ticker": tk, "name": name}
        row_m.update(latest)
        rows_mom.append(row_m)

        # --- BUYシグナル（日足） ---
        if cfg.get("scan_buy_signal", {}).get("enabled", True):
            try:
                df_daily = fetch_daily_df(tk, months=6)
                sig = judge_buy_signal(df_daily, tk)
                row_b = {"ticker": tk, "name": name}
                row_b.update(sig)
                rows_buy.append(row_b)
            except Exception as e:
                print(f"[WARN] buy-signal {tk}: {e}")

        # --- ボリンジャー（日足） ---
        if cfg.get("bollinger", {}).get("enabled", True):
            try:
                df_daily_bb = fetch_daily_df(tk, months=bb_months)
                close = df_daily_bb["Close"]
                high  = df_daily_bb["High"]   # ← shock 判定に必要
                low   = df_daily_bb["Low"]    # ← shock 判定に必要
        
                ma20, _, _, _, _, std = bollinger_from_close(close, 20)
                last_close = float(close.iloc[-1])
                last_ma = float(ma20.iloc[-1])
                last_std = float(std.iloc[-1])
                last_rsi = float(rsi(close, 14).iloc[-1])
        
                # --- 前日終値と ATR 取得（shock 判定用）---
                prev_close = float(close.iloc[-2]) if len(close) >= 2 else np.nan
                atr14 = atr_from_hlc(high, low, close, 14)
                last_atr = float(atr14.iloc[-1]) if len(atr14) > 0 else np.nan
        
                # --- ボリンジャー判定 ---
                status, action, z = classify_bb_status(last_close, last_ma, last_std)
        
                # --- ★ ここだけ追加：shock 判定（関数名は既存の detect_shock）---
                shock_flag, shock_reason, pct_1d = detect_shock(
                    last_close=last_close,
                    prev_close=prev_close,
                    last_atr=last_atr,
                    crash_pct=-0.08,        # 閾値は必要に応じ調整可能
                    crash_atr_mult=2.0,     # 同上
                )

                ##251121 ここを追加
                if shock_flag == "CRASH" and status == "OK":
                    status = "CRASH"
                    action = "CRASH"  # ← holdなどから強制的に引き上げる
                    # action はとりあえずそのままでも良いですが、
                    # 気になるようなら "check" や "monitor" に変える余地もあります

                rows_bb.append({
                    "ticker": tk,
                    "name": name,
                    "action": action,
                    "shock_flag": shock_flag,      # ← 追加
                    "status": status,
                    "zscore": round(z, 3),
                    "shock_reason": shock_reason,  # ← 追加（必要でなければ省略も可）
                    "close": round(last_close, 1),
                    "ma20": round(last_ma, 1),
                    "std": round(last_std, 2),
                    "rsi14": round(last_rsi, 1),
                    "pct_1d": round(pct_1d, 4) if np.isfinite(pct_1d) else np.nan,
                    "atr14":  round(last_atr, 3) if np.isfinite(last_atr) else np.nan,
                    #"pct_1d": pct_1d,              # ← 追加
                    #"atr14": last_atr,              # ← 参考まで
                })

                plot_bollinger_chart(df_daily_bb, tk, name, dir_bb, last_ma, last_std, last_rsi, z)

            except Exception as e:
                print(f"[WARN] bollinger {tk}: {e}")




    # --- モメンタム summary 出力 ---
    if rows_mom:
        df_m = pd.DataFrame(rows_mom)[
            ["ticker","signal","score","close","ema10","ema20","slope10","slope20","rsi","rvol20","bb_up","bb_dn","reason","name"]
        ].copy()

        order = ['TAKE_PROFIT','BUY','HOLD','LOSS_CUT']
        df_m["signal"] = pd.Categorical(df_m["signal"], categories=order, ordered=True)
        df_m = df_m.sort_values(["signal","score"], ascending=[True, False])

        # 丸め処理
        float3 = ["slope10","slope20","rvol20"]          # 小数 3 桁
        float2 = ["close","rsi","ema10","ema20","bb_up","bb_dn"]     # 小数 2 桁

        for c in df_m.columns:
            if c in float3:
                if pd.api.types.is_numeric_dtype(df_m[c]):
                    df_m[c] = pd.to_numeric(df_m[c], errors="coerce").astype(float).round(3)

            elif c in float2:
                if pd.api.types.is_numeric_dtype(df_m[c]):
                    df_m[c] = pd.to_numeric(df_m[c], errors="coerce").astype(float).round(2)

            else:
                if pd.api.types.is_numeric_dtype(df_m[c]):
                    df_m[c] = pd.to_numeric(df_m[c], errors="coerce").round(0)
                    try:
                        df_m[c] = df_m[c].astype("Int64")
                    except Exception:
                        pass

        # 丸め処理のあとにカラム名を標準化
        #df_m = df_m.rename(columns={
        #    "score": "score_dashboard",
        #})

        out_csv = dir_td / f"00_momentum_summary_{now}.csv"
        df_m.to_csv(out_csv, index=False, float_format="%.4f")
        #print(f"[OUT] {out_csv}")

        # コンソール短縮ビュー
        #print()
        #print("ticker momentum (short view)")
        print("-"*80)
        try:
            _view = df_m[["ticker","signal","score","reason","name"]]
        except Exception:
            _view = df_m
        wr = _view["reason"].astype(str).str.len().max()
        wn = _view["name"].astype(str).str.len().max()
        print(_view.to_string(
            index=False,
            formatters={'reason': ('{:<%d}' % wr).format, 'name': ('{:<%d}' % wn).format},
            col_space={'signal':13,'reason': 37, 'name':17}
        ))

    # --- BUYシグナル summary ---
    if rows_buy:
        df_b = pd.DataFrame(rows_buy)
        out_csv = dir_td / f"01_buy_signals_{now}.csv"
        df_b.to_csv(out_csv, index=False)
        #print(f"[OUT] {out_csv}")

    # --- ボリンジャー summary ---
    if rows_bb:
        df_bb = pd.DataFrame(rows_bb)
        out_csv = dir_bb / f"00_bb_alerts_{now}.csv"
        df_bb.to_csv(out_csv, index=False)
        #print(f"[OUT] {out_csv}")

    ##251120
    ## --- マーケット・レジーム長期ダッシュボード ---
    if cfg.get("market_regime", {}).get("enabled", True):
        try:
            from xx_market_regime_dashboard import run as regime_run
            regime_run(do_backtest=False)  # 内部で EQ_RUN_TS を利用させる

            # 実行直後に、この run の market_regime CSV を score順で並べ替え
            try:
                mr_dir = root / "res_market_regime"
                reg_csv = mr_dir / f"00_res_market_regime_{now}.csv"
                if reg_csv.exists():
                    df_reg = pd.read_csv(reg_csv)
                    # "score" カラムがあればそれで降順ソート
                    if "score" in df_reg.columns:
                        df_reg = df_reg.sort_values("score", ascending=False)
                    # rename 済みの "regime_v" カラムがある場合はこちらを利用
                    elif "regime_v" in df_reg.columns:
                        df_reg = df_reg.sort_values("regime_v", ascending=False)
                    df_reg.to_csv(reg_csv, index=False, encoding="utf-8")
            except Exception as e:
                print(f"[WARN] cannot resort market_regime csv: {e}")
        except Exception as e:
            print(f"[WARN] market_regime script failed: {e}")

    # --- Gold アラートログ ---
    if cfg.get("gold_alert", {}).get("enabled", True):
        try:
            from yy_gold_1540T_alert_premium_monitor_stooq import main as gold_main
            gold_main()  # 内部で EQ_RUN_TS を見て res_gold にログ保存
        except Exception as e:
           print(f"[WARN] gold alert script failed: {e}")
 
    # --- HY / VIX / VXN / SPX (外部スクリプト呼び出し) ---
    if cfg.get("hy_vix", {}).get("enabled", True):
        try:
            from yy_hy_vix_vxn_spx_updater_csvdefault_cal_US import main as hy_main
            print()
            print("[INFO] Run HY/VIX/VXN/SPX updater...")
            print()
            hy_main()
        except Exception as e:
            print(f"[WARN] HY/VIX updater: {e}")

    
    ##251120: merger with dashboard data
    print('\nmerge data with dashboard & regime\n')
    merge_with_regime(now, root)


    print()
    print("-"*80)
    print("Finished unified eq dashboard")
    print('short term evaluation')
    print("-"*80)
    print()
## }}}

if __name__ == "__main__":
    main()


## memo
## about VIX
# {{{
'''
VIX の扱いについての件、整理しながらコード化しておきました。

結論としては：

* **VIX は他銘柄と同じ stress_v ランキングに乗せない**
* **VIX 専用の「地震計レベル」を別枠で表示**
* **全体平均（stress_v_avg）を計算するときも VIX は除外**

という方針で、既存スクリプトを修正した版を作成しています。

---

## 1️⃣ 修正済みスクリプト（VIX 分離版）

`ww01_ticker_dashboard_16d_eqrun_env.py` をベースにして、
VIX の扱いだけを分離した版：

> **`ww01_ticker_dashboard_16e_vixsplit.py`**

を作りました。

📦 ダウンロード：

**[Download: ww01_ticker_dashboard_16e_vixsplit.py](sandbox:/mnt/data/ww01_ticker_dashboard_16e_vixsplit.py)** 

中身の主な変更点だけ説明します。

---

## 2️⃣ Stress ranking から VIX を分離

### 変更前（概念）

```python
total_cols = [...]
df_total = merged[total_cols].copy()
# 全銘柄を stress_v 降順にソート
df_total = df_total.sort_values("stress_v", ascending=False)
print("=== Stress ranking (summary) ===")
print(df_total)
...
df_csv = merged[total_cols].copy()
df_csv = df_csv.sort_values("stress_v", ascending=False)
df_csv.to_csv("00_stress_rank.csv")
```

→ これだと **^VIX も他銘柄と同じ並び**になり、
CRASH でも stress_v の数値次第で下の方に来てしまう、という違和感が出ていました。

---

### 変更後のロジック（要点）

1. まず **全銘柄分の df_total_all** を作る
2. そこから **VIX 行だけをマスク (`ticker='^VIX'` or name に「恐怖指数」)**
3. **VIX 以外を stress_v 降順でソート → ランキング表示**
4. VIX は **別枠で 1 行＋“VIXレベル”コメントを表示**
5. CSV 出力 (`00_stress_rank.csv`) では、

   * メイン銘柄を stress_v 降順
   * 最後に VIX 行を付け足す（順位表からは分離）

### 実装イメージ（抜粋）

```python
total_cols = [
    "ticker", "stress_v", "shk_act", "shk_F", "shk_stat",
    "zscore", "signal", "dahs_v", "regime_v",
    "close", "vwap20", "vp_poc", "pct_1d", "rsi", "name",
]

# 全体
df_total_all = merged[total_cols].copy()

# VIX 行を特定（ticker='^VIX' または name に「恐怖指数」）
if "ticker" in df_total_all.columns:
    vix_mask_total = df_total_all["ticker"].astype(str).str.upper().eq("^VIX")
else:
    vix_mask_total = pd.Series(False, index=df_total_all.index)
if "name" in df_total_all.columns:
    vix_mask_total |= df_total_all["name"].astype(str).str.contains("恐怖指数", na=False)

df_vix   = df_total_all[vix_mask_total].copy()
df_total = df_total_all[~vix_mask_total].copy()

# VIX 以外を stress_v 降順でソート
if "stress_v" in df_total.columns:
    df_total = df_total.sort_values("stress_v", ascending=False)

# 表示用に stress_v を 4 桁文字列に整形
if "stress_v" in df_total.columns:
    df_total["stress_v"] = df_total["stress_v"].map(
        lambda x: f"{x:.4f}" if pd.notna(x) else ""
    )

print("\n=== Stress ranking (summary, excl. VIX) ===\n")
print(df_total.to_string(index=False))
print()
```

---

## 3️⃣ VIX 専用のステータス表示 + 簡易レベル分類

VIX については、stress_v ランクに乗せず、
別枠で「今の VIX はどのくらい危険か」をコメント表示させています。

### VIX レベル分類（例）

```python
if vix_close < 13:
    vix_level = "VIX_LOW_CALM"
elif vix_close < 20:
    vix_level = "VIX_NORMAL"
elif vix_close < 30:
    vix_level = "VIX_ELEVATED"
elif vix_close < 40:
    vix_level = "VIX_HIGH_RISK"
else:
    vix_level = "VIX_PANIC"
```

### 表示例（イメージ）

```text
=== Stress ranking (summary, excl. VIX) ===
  2914.T  0.3744  ... JT
  6339.T  0.1933  ... 新東工業
  8316.T  0.1476  ... 三井住友F
  ...

=== VIX status (separate) ===
 ticker stress_v  shk_act shk_F shk_stat zscore signal   pct_1d   rsi        name
   ^VIX   0.1472   CRASH CRASH   CRASH   1.301   HOLD  -0.1132  35.28  恐怖指数（VIX）

[VIX_LEVEL] VIX_ELEVATED (close=23.47)
```

こうすることで、

* 個別株・指数 = 「各都市の震度ランク」
* VIX = 「震源のストレスレベル」

という役割分担がよりハッキリします。

---

## 4️⃣ CSV でも VIX を末尾に退避

`00_stress_rank.csv` についても、
**VIX 以外を stress_v 降順で並べた後、最後に VIX 行を足す**形に変更しています。

```python
out_total = root / "00_stress_rank.csv"
df_csv_all = merged[total_cols].copy()

# VIX 行のマスク
...
df_csv_main = df_csv_all[~vix_mask_csv].copy()
df_csv_vix  = df_csv_all[vix_mask_csv].copy()

if "stress_v" in df_csv_main.columns:
    df_csv_main = df_csv_main.sort_values("stress_v", ascending=False)
    df_csv_main["stress_v"] = df_csv_main["stress_v"].round(4)

# メイン + VIX を結合（VIX は最後）
df_csv = pd.concat([df_csv_main, df_csv_vix], ignore_index=True)
df_csv.to_csv(out_total, index=False, encoding="utf-8")
```

---

## 5️⃣ 3本線グラフの stress_v_avg からも VIX を除外

`00_market_stress_v_rank.png` や
`history/market_summary_history.csv` 用の

```python
avg_stress_v = df_total_sub["stress_v"].mean()
```

も、VIX を含めないように修正しています。

```python
# stress_v 側: 全銘柄の平均 stress_v を1点として扱う（VIX は除外）
if not df_total_sub.empty:
    df_avg_src = df_total_sub.copy()
    try:
        mask_vix = df_avg_src["ticker"].astype(str).str.upper().eq("^VIX")
        if mask_vix.any():
            df_avg_src = df_avg_src[~mask_vix]
    except Exception:
        pass
    avg_stress_v = df_avg_src["stress_v"].mean()
else:
    avg_stress_v = float("nan")
```

これで、

* **「地震の揺れ（各市場のストレス）」**
* **「震源の圧力（VIX）」**

がごちゃ混ぜにならず、
歴史グラフの平均も「各市場ベース」の意味を保ちます。

---

## 6️⃣ 今後の微調整候補

* VIX レベルのしきい値（13 / 20 / 30 / 40）は、実運用で見ながら調整可能です。
* 将来的には「VIX と HY スプレッド」を組み合わせた **“震源オシレーター”** を別途1本定義してもよいと思います。

---

## 7️⃣ 次のステップ

1. ダウンロードした `ww01_ticker_dashboard_16e_vixsplit.py` を
   既存の `16d_eqrun_env` と差し替えるか、
   動作確認用に別名で実行してみてください。
2. 出力の

   * `=== Stress ranking (summary, excl. VIX) ===`
   * `=== VIX status (separate) ===`
   * `00_stress_rank.csv`

   の並びに違和感がないか、実データで一度ご覧いただくのがよろしいかと思います。

'''
# }}}

## mode lline : 折りたたみの設定でmarker {{{ }}} を使う。
# vim:set foldmethod=marker:
