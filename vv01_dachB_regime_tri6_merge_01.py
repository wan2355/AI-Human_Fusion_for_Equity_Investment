#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
## usage

python vv_dachB_regime_tri6_merge_01.py --candidate_csv candidate_tri6.csv

1. ww01_ticker_dashboard_16i_vix_csv_hist.py を実行して
   res_eq_check_all/XXXX_eq_check_all/00_stress_rank.csv を更新
2. res_eq_check_all 配下から「一番新しい XXXX_eq_check_all」を自動検出
3. eval_tri6_rhi.py を呼び出して TRI3/TRI6/RHI6 を計算

という一連の処理を 1 コマンドで実行するランチャー。
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional


def find_latest_stress_csv(eq_root: Path) -> Optional[Path]:
    """
    res_eq_check_all/XXXX_eq_check_all/00_stress_rank.csv のうち、
    一番新しい XXXX_eq_check_all を自動検出して、その CSV パスを返す。
    """
    if not eq_root.exists():
        print(f"[ERROR] eq_root が見つかりません: {eq_root}", file=sys.stderr)
        return None

    subdirs = [
        d for d in eq_root.iterdir()
        if d.is_dir() and d.name.endswith("_eq_check_all")
    ]
    if not subdirs:
        print(f"[ERROR] {eq_root} 配下に *_eq_check_all ディレクトリがありません", file=sys.stderr)
        return None

    subdirs_sorted = sorted(subdirs, key=lambda p: p.name)
    latest = subdirs_sorted[-1]
    csv_path = latest / "00_stress_rank.csv"
    if not csv_path.exists():
        print(f"[ERROR] 00_stress_rank.csv が見つかりません: {csv_path}", file=sys.stderr)
        return None

    print(f"[INFO] 最新の stress CSV: {csv_path}")
    print()
    return csv_path


def run_ww01(script_path: Path, extra_args: str) -> int:
    """
    ww01_ticker_dashboard_16i_vix_csv_hist.py をサブプロセスで実行。
    extra_args には "--config xxx.yaml" など、普段使っている引数を渡せる。
    """
    cmd = ["python", str(script_path)]
    if extra_args:
        # 空白区切りでそのまま渡す簡易実装
        cmd.extend(extra_args.split())

    print(f"[INFO] Run ww01: {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    return proc.returncode


def run_eval_tri6_rhi(
    script_path: Path,
    candidate_csv: Path,
    dash_csv: Path,
    default_win_prob: float,
    jp_adjust: float,
    default_rr: float,
    tri6_threshold: float,
    out_csv: Optional[Path],
) -> int:
    """
    eval_tri6_rhi.py をサブプロセスで実行。
    """
    cmd = [
        "python",
        str(script_path),
        "--candidate_csv", str(candidate_csv),
        "--dash_csv", str(dash_csv),
        "--default_win_prob", str(default_win_prob),
        "--jp_adjust", str(jp_adjust),
        "--default_rr", str(default_rr),
        "--tri6_threshold", str(tri6_threshold),
    ]

    if out_csv is not None:
        cmd.extend(["--out_csv", str(out_csv)])

    print(f"[INFO] Run vv02_eval_rhi_tri6_01: {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    return proc.returncode


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    # 必須
    ap.add_argument(
        '-c',"--candidate_csv",
        default='candidate_tri6.csv',
        #required=True,
        help="TRI パラメータを記載した candidate_tri6.csv",
    )

    # ww01 側
    ap.add_argument(
        '-w',"--ww01_script",
        default="ww01_ticker_dashboard_16i_vix_csv_hist.py",
        help="ww01 スクリプトのパス (デフォルト: ww01_ticker_dashboard_16i_vix_csv_hist.py)",
    )
    ap.add_argument(
        "--ww01_args",
        default="",
        help="ww01 にそのまま渡す追加引数 (例: \"--config config_eq.yaml\")",
    )
    ap.add_argument(
        "--skip_ww01",
        action="store_true",
        help="ダッシュボード更新(ww01)をスキップし、既存の res_eq_check_all だけで評価する",
    )
    ap.add_argument(
        "--eq_root",
        default="res_eq_check_all",
        help="ww01 の出力ルート (デフォルト: res_eq_check_all)",
    )

    # eval_tri6_rhi 側
    ap.add_argument(
        '-t',"--eval_script",
        #default="eval_tri6_06.py",
        default="vv02_eval_rhi_tri6_01.py",
        help="vv02_eval_rhi_tri6_01.py のパス",
    )
    ap.add_argument(
        "--default_win_prob",
        type=float,
        default=0.35,
        help="ベース勝率 (例: 0.35)",
    )
    ap.add_argument(
        "--jp_adjust",
        type=float,
        default=0.05,
        help="日本株の勝率上乗せ (例: 0.05)",
    )
    ap.add_argument(
        "--default_rr",
        type=float,
        default=2.0,
        help="RR 比 (利確幅 : 損切り幅)。例: 2.0",
    )
    ap.add_argument(
        "--tri6_threshold",
        type=float,
        default=0.0,
        help="RHI6_score の flag_pick 閾値 (例: 0.0)",
    )
    ap.add_argument(
        '-o',"--out_csv",
        default="",
        help="最終結果の CSV パス。未指定なら eval_tri6 側のデフォルトに任せる",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    candidate_csv = Path(args.candidate_csv)
    if not candidate_csv.exists():
        print(f"[ERROR] candidate_csv が見つかりません: {candidate_csv}", file=sys.stderr)
        sys.exit(1)

    ww01_script = Path(args.ww01_script)
    eval_script = Path(args.eval_script)
    eq_root = Path(args.eq_root)

    # 1) ww01 実行（スキップ指定がなければ）
    if not args.skip_ww01:
        if not ww01_script.exists():
            print(f"[ERROR] ww01_script が見つかりません: {ww01_script}", file=sys.stderr)
            sys.exit(1)

        rc = run_ww01(ww01_script, args.ww01_args)
        if rc != 0:
            print(f"[ERROR] ww01 実行に失敗しました (returncode={rc})", file=sys.stderr)
            sys.exit(rc)
    else:
        print("[INFO] --skip_ww01 が指定されているため、ww01 の実行をスキップします。")

    # 2) 最新の 00_stress_rank.csv を自動検出
    dash_csv = find_latest_stress_csv(eq_root)
    if dash_csv is None:
        sys.exit(1)

    # 3) eval_tri6_rhi を実行
    if not eval_script.exists():
        print(f"[ERROR] eval_script が見つかりません: {eval_script}", file=sys.stderr)
        sys.exit(1)

    if args.out_csv:
        out_csv = Path(args.out_csv)
    else:
        ## candidate_tri6.csv → candidate_tri6_tri6_rhi.csv のような名前に
        #stem = candidate_csv.stem
        #out_csv = candidate_csv.with_name(f"{stem}_tri6_rhi.csv")

        # 00_market_stress_rank.csv と同じディレクトリに出力
        # res_eq_check_all/{now}_eq_check_all/00_candidate_tri6_filled_tri6_rhi.csv
        out_dir = dash_csv.parent
        out_csv = out_dir / "000_res_rhi_tri6.csv"


    rc = run_eval_tri6_rhi(
        script_path=eval_script,
        candidate_csv=candidate_csv,
        dash_csv=dash_csv,
        default_win_prob=args.default_win_prob,
        jp_adjust=args.jp_adjust,
        default_rr=args.default_rr,
        tri6_threshold=args.tri6_threshold,
        out_csv=out_csv,
    )
    if rc != 0:
        print(f"[ERROR] eval_tri6_rhi 実行に失敗しました (returncode={rc})", file=sys.stderr)
        sys.exit(rc)

    print()
    print(f"[INFO] パイプライン完了。結果: {out_csv}")
    print()


if __name__ == "__main__":
    main()


## mode lline : 折りたたみの設定でmarker {{{ }}} を使う。
# vim:set foldmethod=marker:
