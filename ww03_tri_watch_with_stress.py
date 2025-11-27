#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
yy_tri_watch_with_stress_01.py

- TRIウォッチリスト (00_tri_watchlist_xxx.csv)
- ダッシュボードの stress ランク (00_stress_rank.csv)

をマージし、「TRI的にも良さそう かつ 市場状態も許容範囲」の
今日見るべき銘柄リストをコンパクトに表示する。

使い方例:

python yy_tri_watch_with_stress_01.py \
  --tri res_tri_watch/00_tri_watchlist_251127.csv \
  --stress res_eq_check_all/251127_0830_eq_check_all/00_stress_rank.csv \
  --max-stress 0.60


## 使い方
いつも通り ww01_ticker_dashboard_16i_vix_csv_hist.py を実行
→ res_eq_check_all/{EQ_RUN_TS}_eq_check_all/00_stress_rank.csv ができる

yy_tri_watchlist_01.py を実行
→ res_tri_watch/00_tri_watchlist_YYMMDD.csv ができる

下記の「マージ用スクリプト」で watch_flag=True かつ stress_v や signal 情報をまとめて表示
"""

import argparse
import pandas as pd
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tri", type=str, required=True, help="00_tri_watchlist_xxx.csv")
    ap.add_argument("--stress", type=str, required=True, help="00_stress_rank.csv")
    ap.add_argument(
        "--max-stress",
        type=float,
        default=0.60,
        help="stress_v の上限（これ以上は除外）",
    )
    ap.add_argument(
        "--allow-shock",
        action="store_true",
        help="SHEDDING_IN_PROGRESS や CRASH も含める場合に指定",
    )
    args = ap.parse_args()

    tri_path = Path(args.tri)
    st_path = Path(args.stress)

    if not tri_path.exists():
        raise FileNotFoundError(f"TRI watchlist not found: {tri_path}")
    if not st_path.exists():
        raise FileNotFoundError(f"stress_rank not found: {st_path}")

    df_tri = pd.read_csv(tri_path)
    df_st = pd.read_csv(st_path)

    # 必要列だけ使う
    # 00_stress_rank.csv 側の想定列: ticker, stress_v, signal, shk_stat, name など
    cols_keep = []
    for c in ["ticker", "stress_v", "signal", "shk_stat", "name"]:
        if c in df_st.columns:
            cols_keep.append(c)
    df_st = df_st[cols_keep].copy()

    # マージ
    df = pd.merge(df_tri, df_st, on="ticker", how="left")

    # 型・フィルタ
    if "stress_v" in df.columns:
        df["stress_v"] = pd.to_numeric(df["stress_v"], errors="coerce")

    # TRI watch フラグがない場合は全銘柄 True とみなす
    if "watch_flag" not in df.columns:
        df["watch_flag"] = True

    cond = df["watch_flag"] == True

    if "stress_v" in df.columns:
        cond = cond & (df["stress_v"] <= args.max_stress)

    if not args.allow_shock and "shk_stat" in df.columns:
        cond = cond & ~df["shk_stat"].fillna("").isin(
            ["SHEDDING_IN_PROGRESS", "CRASH"]
        )

    df_sel = df[cond].copy()

    # ソート: combined, stress_v など
    sort_keys = []
    if "combined" in df_sel.columns:
        sort_keys.append(("combined", False))
    if "stress_v" in df_sel.columns:
        sort_keys.append(("stress_v", True))

    if sort_keys:
        by = [k for k, _ in sort_keys]
        asc = [a for _, a in sort_keys]
        df_sel = df_sel.sort_values(by=by, ascending=asc)

    # 表示用の整形
    disp_cols = []
    for c in ["ticker", "tri5", "tri6", "combined", "stress_v", "signal", "shk_stat", "name"]:
        if c in df_sel.columns:
            disp_cols.append(c)

    print()
    print("=== TRI × stress_v 統合: 今日じっくり見るべき候補 ===")
    if df_sel.empty:
        print("該当なし（閾値や max-stress を少し緩めると候補が出るかもしれません）")
    else:
        df_disp = df_sel[disp_cols].copy()

        for c in ["tri5", "tri6", "combined", "stress_v"]:
            if c in df_disp.columns:
                df_disp[c] = df_disp[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")

        print(df_disp.to_string(index=False))

    print()


if __name__ == "__main__":
    main()

