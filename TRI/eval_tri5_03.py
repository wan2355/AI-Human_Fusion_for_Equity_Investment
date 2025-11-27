#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
銘柄候補リスト (candidate_tri5.csv など) から
期待値・時間効率 (TRI3)・安定性補正付きスコア (TRI5) を計算し、
timestamp 付きディレクトリに結果 CSV と config.json を保存するスクリプト。

使い方例:

    python eval_tri5.py \
        --input candidate_tri5.csv \
        --default-win-prob 0.35 \
        --jp-adjust 0.05 \
        --rr-ratio 2.0 \
        --tri5-threshold 500 \
        -z first_try
"""

import argparse
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="銘柄候補の期待値・TRI3/TRI5を計算するツール"
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="candidate_tri5.csv",
        help="入力CSVファイルパス (デフォルト: candidate_tri5.csv)",
    )

    # --default_win_prob / --default-win-prob の両方OK
    parser.add_argument(
        "--default_win_prob",
        "--default-win-prob",
        dest="default_win_prob",
        type=float,
        default=0.35,
        help="デフォルト勝率 (0〜1、例: 0.35)",
    )

    # --jp_adjust / --jp-adjust の両方OK
    parser.add_argument(
        "--jp_adjust",
        "--jp-adjust",
        dest="jp_adjust",
        type=float,
        default=0.05,
        help="market が jp のとき default_win_prob に加算する値 (例: 0.05)",
    )

    parser.add_argument(
        "--rr-ratio",
        type=float,
        default=2.0,
        help="利益:損失 の比率 (R/R, デフォルト 2.0 = 2:1)",
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default="res_qg_selection",
        help="結果を保存するベースディレクトリ (デフォルト: res_qg_selection)",
    )

    parser.add_argument(
        "-z",
        "--suffix",
        type=str,
        default="",
        help="結果ディレクトリ名の末尾に付与するコメント (例: LITE_test)",
    )

    parser.add_argument(
        "-c","--config-only",
        action="store_true",
        help="計算せずに config.json のみ生成するテスト用フラグ",
    )

    parser.add_argument(
        "-t","--tri5-threshold",
        type=float,
        default=0.0,
        help=(
            "TRI5 がこの値以上の銘柄に flag_pick='PICK' を立てる "
            "(デフォルト: 0 = 全銘柄フラグなし)"
        ),
    )

    return parser.parse_args()


def _clamp(x: float, lo: float = 0.0, hi: float = 0.99) -> float:
    return max(lo, min(hi, x))


def _is_jp_market(market: str) -> bool:
    m = (market or "").lower()
    return m in {"jp", "jpn", "jp_stock", "jp-equity", "japan"}


def calc_metrics(
    *,
    prep_time_min: float,
    period_months: float,
    base_loss_amount: float,
    default_win_prob: float,
    jp_adjust: float,
    rr_ratio_global: float,
    market: str,
    win_prob_override: Optional[float] = None,
    rr_ratio_override: Optional[float] = None,
    win_amount_override: Optional[float] = None,
    risk_span: Optional[float] = None,
) -> Dict[str, Any]:
    """
    1銘柄分の各種指標を計算する。
    """

    # 勝率を決定
    if win_prob_override is not None:
        win_prob = _clamp(win_prob_override, 0.0, 0.99)
    else:
        win_prob = default_win_prob
        if _is_jp_market(market):
            win_prob += jp_adjust
        win_prob = _clamp(win_prob, 0.0, 0.99)

    loss_prob = 1.0 - win_prob

    # R/R を決定
    rr_ratio = rr_ratio_override if rr_ratio_override is not None else rr_ratio_global
    if rr_ratio <= 0:
        rr_ratio = rr_ratio_global

    # 損失額と利益額
    loss_amount = base_loss_amount
    if loss_amount <= 0:
        # 損失額が不正な場合は計算対象外
        return {
            "valid": False,
            "reason": "loss_amount <= 0",
        }

    if win_amount_override is not None and win_amount_override > 0:
        win_amount = win_amount_override
    else:
        win_amount = loss_amount * rr_ratio

    # 期待値 (1トレードあたり)
    expected_profit = win_prob * win_amount - loss_prob * loss_amount

    # 年率換算期待値と TRI3
    if period_months <= 0:
        expected_profit_annualized = None
        tri3 = None
    else:
        expected_profit_annualized = expected_profit * (12.0 / period_months)
        if prep_time_min > 0:
            tri3 = expected_profit_annualized / prep_time_min
        else:
            tri3 = None

    # 安定性補正 (risk_span があれば使う。なければ補正なしで TRI5=TRI3)
    stability_factor = None
    tri5 = tri3
    if risk_span is not None and risk_span > 0 and tri3 is not None:
        # シンプルな近似: 利益額 / 標準偏差
        stability_factor = win_amount / risk_span
        tri5 = tri3 * stability_factor

    return {
        "valid": True,
        "win_prob": win_prob,
        "loss_prob": loss_prob,
        "rr_ratio_effective": rr_ratio,
        "win_amount": win_amount,
        "loss_amount": loss_amount,
        "expected_profit": expected_profit,
        "expected_profit_annualized": expected_profit_annualized,
        "TRI3_yen_per_min_per_year": tri3,
        "stability_factor": stability_factor,
        "TRI5_score": tri5,
    }


def main() -> None:
    args = parse_args()

    # タイムスタンプ
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 出力ディレクトリ作成
    base_dir = Path(args.base_dir)
    suffix_part = f"_{args.suffix}" if args.suffix else ""
    result_dir = base_dir / f"res_eg_selc_{now}{suffix_part}"
    result_dir.mkdir(parents=True, exist_ok=True)

    # config.json (条件の記録) 準備
    config_path = result_dir / "config.json"
    config_data: Dict[str, Any] = {
        "timestamp": now,
        "input_csv": str(Path(args.input).resolve()),
        "default_win_prob": args.default_win_prob,
        "jp_adjust": args.jp_adjust,
        "rr_ratio_global": args.rr_ratio,
        "tri5_threshold": args.tri5_threshold,
        "base_dir": str(base_dir.resolve()),
        "result_dir": str(result_dir.resolve()),
        "note": (
            "win_prob は CSV の win_prob 列が優先。"
            "なければ default_win_prob + (market が jp のとき jp_adjust)。"
            "R/R は rr_ratio 列が優先。なければ rr_ratio_global を使用。"
        ),
        "required_columns": [
            "ticker",
            "market",
            "prep_time_min",
            "period_months",
            "loss_amount",
        ],
        "optional_columns": [
            "win_prob",
            "rr_ratio",
            "win_amount",
            "risk_span",
            "comment",
        ],
        "output_files": [],
    }

    # config だけ試し出力するモード
    if args.config_only:
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        print(f"[INFO] config_only モード: config.json を出力しました: {config_path}")
        return

    # 入力CSVの存在確認
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] 入力CSVが見つかりません: {input_path}")
        return

    rows_out: List[Dict[str, Any]] = []

    # 入力CSV読み込み
    with input_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 必須項目
            try:
                ticker = row.get("ticker", "").strip()
                market = row.get("market", "").strip()
                prep_time_min = float(row.get("prep_time_min", "0").strip())
                period_months = float(row.get("period_months", "0").strip())
                loss_amount = float(row.get("loss_amount", "0").strip())
            except Exception as e:
                print(f"[WARN] 行をスキップします (必須項目のパース失敗): {e}, row={row}")
                continue

            # オプション項目
            win_prob_override = None
            if "win_prob" in row and row["win_prob"].strip() != "":
                try:
                    win_prob_override = float(row["win_prob"].strip())
                except ValueError:
                    win_prob_override = None

            rr_ratio_override = None
            if "rr_ratio" in row and row["rr_ratio"].strip() != "":
                try:
                    rr_ratio_override = float(row["rr_ratio"].strip())
                except ValueError:
                    rr_ratio_override = None

            win_amount_override = None
            if "win_amount" in row and row["win_amount"].strip() != "":
                try:
                    win_amount_override = float(row["win_amount"].strip())
                except ValueError:
                    win_amount_override = None

            risk_span = None
            if "risk_span" in row and row["risk_span"].strip() != "":
                try:
                    risk_span = float(row["risk_span"].strip())
                except ValueError:
                    risk_span = None

            # 指標計算
            metrics = calc_metrics(
                prep_time_min=prep_time_min,
                period_months=period_months,
                base_loss_amount=loss_amount,
                default_win_prob=args.default_win_prob,
                jp_adjust=args.jp_adjust,
                rr_ratio_global=args.rr_ratio,
                market=market,
                win_prob_override=win_prob_override,
                rr_ratio_override=rr_ratio_override,
                win_amount_override=win_amount_override,
                risk_span=risk_span,
            )

            if not metrics.get("valid", False):
                print(
                    f"[WARN] 行をスキップします (指標計算不可: {metrics.get('reason')}): "
                    f"ticker={ticker}"
                )
                continue

            # 出力行作成 (元の列 + 指標列)
            out_row: Dict[str, Any] = dict(row)

            # 勝率・R/R・損益などのフォーマット
            out_row["win_prob_effective"] = f"{metrics['win_prob']:.2f}"
            out_row["loss_prob_effective"] = f"{metrics['loss_prob']:.2f}"
            out_row["rr_ratio_effective"] = f"{metrics['rr_ratio_effective']:.2f}"
            out_row["win_amount_effective"] = f"{metrics['win_amount']:.2f}"
            out_row["loss_amount_effective"] = f"{metrics['loss_amount']:.2f}"
            out_row["expected_profit"] = f"{metrics['expected_profit']:.2f}"

            if metrics["expected_profit_annualized"] is not None:
                out_row["expected_profit_annualized"] = (
                    f"{metrics['expected_profit_annualized']:.2f}"
                )
            else:
                out_row["expected_profit_annualized"] = ""

            if metrics["TRI3_yen_per_min_per_year"] is not None:
                out_row["TRI3_yen_per_min_per_year"] = (
                    f"{metrics['TRI3_yen_per_min_per_year']:.2f}"
                )
            else:
                out_row["TRI3_yen_per_min_per_year"] = ""

            # 安定性係数は小さい値なので 3桁表示
            if metrics["stability_factor"] is not None:
                out_row["stability_factor"] = f"{metrics['stability_factor']:.3f}"
            else:
                out_row["stability_factor"] = ""

            # TRI5 と flag_pick
            if metrics["TRI5_score"] is not None:
                tri5_val = metrics["TRI5_score"]
                out_row["TRI5_score"] = f"{tri5_val:.2f}"
                if args.tri5_threshold > 0 and tri5_val >= args.tri5_threshold:
                    out_row["flag_pick"] = "PICK"
                else:
                    out_row["flag_pick"] = ""
            else:
                out_row["TRI5_score"] = ""
                out_row["flag_pick"] = ""

            rows_out.append(out_row)

    # ソート: TRI5 降順 (なければ TRI3、それもなければ 0)
    def sort_key(r: Dict[str, Any]) -> float:
        v_tri5 = r.get("TRI5_score", "")
        try:
            return float(v_tri5)
        except ValueError:
            v_tri3 = r.get("TRI3_yen_per_min_per_year", "")
            try:
                return float(v_tri3)
            except ValueError:
                return 0.0

    rows_out_sorted = sorted(rows_out, key=sort_key, reverse=True)

    # CSV 書き出し
    if rows_out_sorted:
        result_csv = result_dir / f"result_{now}.csv"

        # もとの順序を取得
        original_fields: List[str] = list(rows_out_sorted[0].keys())

        # ticker の右隣に TRI5_score を配置
        preferred_front = ["ticker", "TRI5_score"]
        ordered_fields: List[str] = []

        for k in preferred_front:
            if k in original_fields and k not in ordered_fields:
                ordered_fields.append(k)

        for k in original_fields:
            if k not in ordered_fields:
                ordered_fields.append(k)

        fieldnames = ordered_fields

        with result_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_out_sorted)

        print(f"[INFO] 結果を書き出しました: {result_csv.resolve()}")
        config_data["output_files"].append(str(result_csv.resolve()))
    else:
        print("[WARN] 有効な行がなかったため、結果CSVは出力されませんでした。")

    # config.json を保存
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] config.json を書き出しました: {config_path.resolve()}")


if __name__ == "__main__":
    main()


