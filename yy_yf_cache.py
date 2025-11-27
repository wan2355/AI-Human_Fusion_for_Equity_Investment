#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yfinance 用のシンプル共通キャッシュ

- ティッカー + period + interval ごとに CSV を .yf_cache に保存
- 有効期限を過ぎていなければ再利用する
"""

import os
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


def _cache_dir() -> Path:
    d = Path(".yf_cache")
    d.mkdir(exist_ok=True)
    return d


def _cache_key(ticker: str, period: str, interval: str) -> Path:
    # 記号をファイル名として安全な形に変換
    safe = ticker.replace("^", "IDX_").replace("=", "_").replace("/", "_")
    return _cache_dir() / f"{safe}_{period}_{interval}.csv"


def _apply_eq_run_ts_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    EQ_RUN_TS 環境変数が指定されている場合、その時刻までのデータだけにクリップする。
    - EQ_RUN_TS 形式: "%y%m%d_%H%M"（例: "241001_1600"）
    - インデックスが DatetimeIndex でない場合や、パースに失敗した場合はそのまま返す。
    """
    ts = os.environ.get("EQ_RUN_TS")
    if not ts or df is None or df.empty:
        return df

    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        return df

    try:
        # "241001_1600" → 2024-10-01 16:00
        dt_cut = datetime.strptime(ts, "%y%m%d_%H%M")
    except Exception:
        # 形式が違う場合はフィルタせずそのまま返す
        return df

    # 指定時刻以下のバーだけ残す（該当がなければ空でもよい）
    return df[df.index <= dt_cut]


def fetch_yf_cached(
    ticker: str,
    #period: str = "6mo",
    period: str = "400d",
    interval: str = "1d",
    max_age_days: int = 1,
    #verbose: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    yfinance をキャッシュ付きで呼び出すラッパ。

    - まず .yf_cache 配下の CSV を確認し、有効期限内ならそれを読む
    - ダメなら yfinance から取得して CSV を上書き保存する
    """

    cache_path = _cache_key(ticker, period, interval)

    # ------------------------------
    # キャッシュ利用判定
    # ------------------------------
    if cache_path.exists():
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - mtime <= timedelta(days=max_age_days):
            try:
                # 新形式キャッシュ: "Date" 列をインデックスとして読み込む
                df = pd.read_csv(cache_path, index_col="Date", parse_dates=["Date"])

                ##251120
                if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                    # ★ここでタイムゾーンを剥がす
                    df.index = df.index.tz_localize(None)

                if df is not None and not df.empty:
                    if verbose:
                        print(f"[CACHE] {ticker} {period} {interval} -> {cache_path}")
                    df = _apply_eq_run_ts_filter(df)
                    return df
                else:
                    if verbose:
                        print(f"[CACHE] {ticker}: cached df is empty, re-download")
            except Exception as e:
                # 旧形式（ヘッダがおかしい等）はここで落ちるので、素直に再取得へ
                if verbose:
                    print(f"[CACHE] {ticker}: failed to read cache ({e}), re-download")

    # ------------------------------
    # ここまで来たら新規ダウンロード
    # ------------------------------
    if verbose:
        print(f"[YF]    download {ticker} {period} {interval} (history)")

    df = pd.DataFrame()

    # 1) 推奨: Ticker.history を使う
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, interval=interval, auto_adjust=False)
    except Exception as e:
        if verbose:
            print(f"[YF]    history() failed for {ticker}: {e}")

    # 2) 保険として yf.download にフォールバック
    if df is None or df.empty:
        if verbose:
            print(f"[YF]    fallback to yf.download for {ticker}")
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )

    if df is None or df.empty:
        raise RuntimeError(f"{ticker}: yfinance 取得失敗")

    # インデックス名を揃え、日付順にソート
    df = df.copy()
    
    ##251120
    ## タイムゾーンが付いていたら剥がす
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    ##251120
    df.index = pd.to_datetime(df.index)

    df.index.name = "Date"
    df = df.sort_index()



    # CSVとして保存（新形式: 先頭行に "Date" ヘッダ）
    df.to_csv(cache_path)

    if verbose:
        print(f"[SAVE ] {ticker} -> {cache_path}")

    # EQ_RUN_TS が指定されていれば、その時刻までにクリップしたものを返す
    df = _apply_eq_run_ts_filter(df)
    return df


def clear_cache(older_than_days: int | None = None) -> int:
    """
    .yf_cache 以下を削除。
    older_than_days が None の場合は全削除。
    それ以外は、指定日数より古いものだけ削除。

    戻り値: 削除したファイル数
    """
    d = _cache_dir()
    if not d.exists():
        return 0

    removed = 0
    now = datetime.now()

    for p in d.glob("*.csv"):
        if older_than_days is None:
            p.unlink(missing_ok=True)
            removed += 1
        else:
            mtime = datetime.fromtimestamp(p.stat().st_mtime)
            if now - mtime > timedelta(days=older_than_days):
                p.unlink(missing_ok=True)
                removed += 1

    return removed


