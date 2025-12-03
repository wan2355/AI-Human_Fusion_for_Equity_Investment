#!/usr/bin/env python3
# -*- coding: utf-8 -*-


## usage
#{{{
"""
auto_fill_candidate_tri6.py

- candidate_tri6.csv に対して：
    * 既存の手入力はそのまま維持
    * 空欄セルだけを「カテゴリ別デフォルト」で自動補完
- ww01 の dash_csv（例: 00_stress_rank.csv）と銘柄を揃える：
    * dash_csv 側に存在するが candidate に無い ticker 行を追加
- hi-vola 含む category を自動推定しつつ、手入力の category があればそれを優先

使い方例:

    python vv03_auto_fill_candidate_tri6.py \
        --candidate_csv candidate_tri6.csv \
        --dash_csv res_eq_check_all/251203_1730_eq_check_all/00_stress_rank.csv \
        --out_csv candidate_tri6_filled.csv


#----------------------
様整理（このスクリプトがやること）

入力：

--candidate_csv : 既存の candidate_tri6.csv

--dash_csv : ww01 の 00_stress_rank.csv（tickerユニバース用。中身の列は使わなくてもOK）

--out_csv : 補完後の出力ファイル（省略時は candidate_tri6_filled.csv）

処理：

ticker のユニバースを
candidate と dash_csv の 和集合 にする

candidate に無い ticker は新規行として追加（marketは ticker末尾 .T → jp / それ以外 us）

各行に category を付与

既に列があればそれを優先

無ければ ticker と comment から自動推定

INDEX / GOLD / JP_HIGH_DIV / MEGATECH / HI_VOLA / OTHER

HI_VOLA には中小型や「高ボラ」「攻め枠」「BTC」系を寄せる

各 category ごとに

もしそのカテゴリで 手入力済みの値 があれば「平均値」を計算し、
それをそのカテゴリのデフォルトとする

それも無い場合は、スクリプト内の 固定デフォルト表 を使う

対象列：

prep_time_min

period_months

loss_amount

win_prob

rr_ratio

risk_span

max_dd_ratio

これらの 空欄だけ を category デフォルトで補完
（手入力済みのセルは一切いじらない）

出力：

元の列に加えて

category

auto_filled（1なら自動補完した行／0ならほぼ手入力）

形式：UTF-8、ヘッダ付き CSV
"""
# }}}

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import math


# ----------------------------------------------------------
# 共通ユーティリティ
# ----------------------------------------------------------

def safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    if val is None:
        return default
    s = str(val).strip()
    if s == "":
        return default
    try:
        return float(s)
    except ValueError:
        return default


def is_empty(val: Any) -> bool:
    if val is None:
        return True
    s = str(val).strip()
    return s == "" or s.upper() == "NA" or s.upper() == "NAN"


# ----------------------------------------------------------
# ファイル読み込み
# ----------------------------------------------------------

def read_csv_to_dicts(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))
    return rows


def write_csv_from_dicts(path: Path, rows: List[Dict[str, Any]]) -> None:
    # 全列集合
    fieldnames: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    ##----251203
    ## 一時ファイルに書き出し
    tmp_path = path.with_suffix('.tmp')
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    ## 改行コードをLFに統一して本ファイルへ
    with tmp_path.open("r", encoding="utf-8") as fin, path.open("w", newline="", encoding="utf-8") as fout:
        for line in fin:
            fout.write(line.replace('\r\n', '\n'))
    tmp_path.unlink()



    ##----org dame : DOSになる。
    ##with path.open("w", newline="", encoding="utf-8-sig") as f: ##<< これではdosになる.
    ### 改行コードをLF固定、BOMなしUTF-8で出力
    ##with path.open("w", newline="\n", encoding="utf-8") as f:
    #with path.open("w", newline="", encoding="utf-8") as f:
    #    writer = csv.DictWriter(f, fieldnames=fieldnames)
    #    writer.writeheader()
    #    for r in rows:
    #        writer.writerow(r)


# ----------------------------------------------------------
# category 推定ロジック（簡易版）
# ----------------------------------------------------------

def infer_category(ticker: str, market: str, comment: str) -> str:
    t = (ticker or "").upper()
    m = (market or "").lower()
    c = (comment or "")

    # まずはコメントの日本語キーワードで判定
    if "高ボラ" in c or "攻め枠" in c or "値動き大きい" in c or "BTC" in c:
        return "HI_VOLA"

    if "高配当" in c:
        return "JP_HIGH_DIV"

    if "純金" in c or "ゴールド" in c:
        return "GOLD"

    if "ETF" in c or "指数" in c or "日経" in c:
        return "INDEX"

    if "AI" in c or "NASDAQ" in c or "メガテック" in c:
        return "MEGATECH"

    # tickerベースの簡易判定
    if t.endswith(".T") and "ETF" in c:
        return "INDEX"

    if t in ("GLDM", "1540.T"):
        return "GOLD"

    if t in ("JEPI", "JEPQ"):
        # 分類の仕方はお好みで変更可
        return "JP_HIGH_DIV"

    if t in ("MSFT", "NVDA", "AMZN", "META", "GOOGL", "LITE", "QQQI"):
        return "MEGATECH"

    if t in ("IREN", "MARA", "RIOT"):
        return "HI_VOLA"

    # 日本株で特に特徴がなければ高配当寄せでも良いが、
    # ここでは一旦 OTHER にしておく
    return "OTHER"


# ----------------------------------------------------------
# category 別の静的デフォルト（初期叩き台）
#   ※あとで candidate 側のデータから動的に上書きする
# ----------------------------------------------------------

STATIC_DEFAULTS = {
    "INDEX": {
        "prep_time_min": 30,
        "period_months": 1,
        "loss_amount": 25000,
        "win_prob": 0.40,
        "rr_ratio": 1.8,
        "risk_span": 20000,
        "max_dd_ratio": 0.20,
    },
    "GOLD": {
        "prep_time_min": 40,
        "period_months": 4,
        "loss_amount": 30000,
        "win_prob": 0.35,
        "rr_ratio": 2.0,
        "risk_span": 30000,
        "max_dd_ratio": 0.30,
    },
    "JP_HIGH_DIV": {
        "prep_time_min": 45,
        "period_months": 6,
        "loss_amount": 30000,
        "win_prob": 0.45,
        "rr_ratio": 2.0,
        "risk_span": 30000,
        "max_dd_ratio": 0.30,
    },
    "MEGATECH": {
        "prep_time_min": 60,
        "period_months": 3,
        "loss_amount": 30000,
        "win_prob": 0.35,
        "rr_ratio": 2.2,
        "risk_span": 60000,
        "max_dd_ratio": 0.40,
    },
    "HI_VOLA": {
        "prep_time_min": 70,
        "period_months": 1,
        "loss_amount": 30000,
        "win_prob": 0.30,
        "rr_ratio": 2.5,
        "risk_span": 80000,
        "max_dd_ratio": 0.50,
    },
    "OTHER": {
        "prep_time_min": 50,
        "period_months": 3,
        "loss_amount": 30000,
        "win_prob": 0.35,
        "rr_ratio": 2.0,
        "risk_span": 40000,
        "max_dd_ratio": 0.30,
    },
}


FIELDS_TO_FILL = [
    "prep_time_min",
    "period_months",
    "loss_amount",
    "win_prob",
    "rr_ratio",
    "risk_span",
    "max_dd_ratio",
]


# ----------------------------------------------------------
# candidate から category 別の動的デフォルトを作る
# ----------------------------------------------------------

def build_dynamic_defaults(candidate_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    # まず category が無ければ推定して付与
    for r in candidate_rows:
        cat = r.get("category", "").strip()
        if cat == "":
            ticker = r.get("ticker", "")
            market = r.get("market", "")
            comment = r.get("comment", "")
            r["category"] = infer_category(ticker, market, comment)

    # category × field の平均値を計算
    agg: Dict[str, Dict[str, List[float]]] = {}
    for r in candidate_rows:
        cat = r.get("category", "").strip() or "OTHER"
        if cat not in agg:
            agg[cat] = {f: [] for f in FIELDS_TO_FILL}
        for f in FIELDS_TO_FILL:
            v = safe_float(r.get(f), None)
            if v is not None:
                agg[cat][f].append(v)

    dyn_defaults: Dict[str, Dict[str, float]] = {}
    for cat, fdict in agg.items():
        dyn_defaults[cat] = {}
        for f in FIELDS_TO_FILL:
            vals = fdict.get(f, [])
            if vals:
                dyn_defaults[cat][f] = sum(vals) / len(vals)

    return dyn_defaults


def get_default_for_category(cat: str, field: str,
                             dyn_defaults: Dict[str, Dict[str, float]]) -> float:
    cat = cat or "OTHER"
    # 1. 動的デフォルト（candidate からの平均）を優先
    if cat in dyn_defaults and field in dyn_defaults[cat]:
        return dyn_defaults[cat][field]
    # 2. 静的デフォルト
    if cat in STATIC_DEFAULTS and field in STATIC_DEFAULTS[cat]:
        return STATIC_DEFAULTS[cat][field]
    # 3. OTHER の静的デフォルト
    if "OTHER" in STATIC_DEFAULTS and field in STATIC_DEFAULTS["OTHER"]:
        return STATIC_DEFAULTS["OTHER"][field]
    # 4. 最後の保険
    if field in ("win_prob",):
        return 0.35
    if field in ("rr_ratio",):
        return 2.0
    if field in ("max_dd_ratio",):
        return 0.30
    if field in ("period_months",):
        return 3
    if field in ("prep_time_min",):
        return 50
    if field in ("loss_amount", "risk_span"):
        return 30000
    return 0.0


# ----------------------------------------------------------
# メイン処理
# ----------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('-c',"--candidate_csv", default='candidate_tri6.csv')
    #ap.add_argument('-c',"--candidate_csv", required=True)
    ap.add_argument('-d',"--dash_csv", default='00_market_stress_rank.csv',
                    help="ww01 の 00_stress_rank.csv など。ticker ユニバース用")
    #ap.add_argument('-d',"--dash_csv", required=True,
    #                help="ww01 の 00_stress_rank.csv など。ticker ユニバース用")
    ap.add_argument("--out_csv", default="",
                    help="省略時は candidate_tri6_filled.csv")
    args = ap.parse_args()

    candidate_path = Path(args.candidate_csv)
    dash_path = Path(args.dash_csv)

    if not candidate_path.exists():
        raise SystemExit(f"[ERROR] candidate_csv not found: {candidate_path}")
    if not dash_path.exists():
        raise SystemExit(f"[ERROR] dash_csv not found: {dash_path}")

    cand_rows = read_csv_to_dicts(candidate_path)
    dash_rows = read_csv_to_dicts(dash_path)

    # 既存 candidate の ticker -> row マップ
    cand_map: Dict[str, Dict[str, Any]] = {}
    for r in cand_rows:
        t = (r.get("ticker") or "").strip()
        if t:
            cand_map[t] = r

    # dash 側 ticker 集合
    dash_tickers = set()
    for r in dash_rows:
        t = (r.get("ticker") or "").strip()
        if t:
            dash_tickers.add(t)

    # ユニバース = candidate ∪ dash
    all_tickers = sorted(set(list(cand_map.keys()) + list(dash_tickers)))

    # candidate_rows を category 推定込みで動的デフォルト用に処理
    # （既存の cand_rows をそのまま使う）
    dyn_defaults = build_dynamic_defaults(cand_rows)

    out_rows: List[Dict[str, Any]] = []

    for t in all_tickers:
        if t in cand_map:
            row = dict(cand_map[t])  # コピーして使う
        else:
            # 新規銘柄: 空の行を作成
            row = {
                "ticker": t,
                "market": "jp" if t.endswith(".T") else "us",
                "comment": "",
            }

        # category 確定（既存値優先）
        cat = (row.get("category") or "").strip()
        if cat == "":
            cat = infer_category(row.get("ticker", ""), row.get("market", ""), row.get("comment", ""))
        row["category"] = cat

        # auto_filled フラグ（1 = 何かしら補完した）
        auto_filled = 0

        # 空欄セルだけ埋める
        for f in FIELDS_TO_FILL:
            if is_empty(row.get(f)):
                v_def = get_default_for_category(cat, f, dyn_defaults)
                row[f] = v_def
                auto_filled = 1

        row["auto_filled"] = auto_filled
        out_rows.append(row)

    # 出力パス
    if args.out_csv:
        out_path = Path(args.out_csv)
    else:
        out_path = Path("candidate_tri6_filled.csv")

    write_csv_from_dicts(out_path, out_rows)
    print(f"[OK] auto-filled candidate saved to: {out_path}")
    print(f"[INFO] total tickers: {len(all_tickers)}")


if __name__ == "__main__":
    main()


## mode lline : 折りたたみの設定でmarker {{{ }}} を使う。
# vim:set foldmethod=marker:
