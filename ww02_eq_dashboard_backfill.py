#!/usr/bin/env python3
"""
ww_eq_dashboard_backfill_v01.py

既存の ww_ticker_dashboard_16*.py を「過去日付でまとめて実行」し、
res_eq_check_all/{ts}_eq_check_all/ 以下と history を一気に埋めるための
バックフィル用ドライバースクリプト。

使い方:
    python ww02_eq_dashboard_backfill_v01.py \
        --start 2024-10-01 --end 2024-11-23 \
        --config config_eq_dashboard.json \
        --time 16:00

前提:
  - 同じディレクトリに ww_ticker_dashboard_16d_eqrun_env.py があること
  - その中で EQ_RUN_TS を環境変数から読むように修正済みであること
"""

import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD 形式の開始日")
    ap.add_argument("--end", help="YYYY-MM-DD 形式の終了日（省略時は start と同じ）")
    ap.add_argument("--config", "-c", default="config_eq_dashboard.json",
                    help="ww_ticker_dashboard 用の config ファイルパス")
    ap.add_argument("--time", default="16:00",
                    help="EQ_RUN_TS に付与する時刻 (HH:MM, default=16:00)")
    ap.add_argument("--script", default="ww01_ticker_dashboard.py",
                    help="実行するダッシュボードスクリプト名")
    return ap.parse_args()


def iter_dates(start: dt.date, end: dt.date):
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)


def main():
    args = parse_args()

    ## clear cache
    from yy_yf_cache_eqrun_ts import clear_cache
    clear_cache(older_than_days=0)

    start_date = dt.datetime.strptime(args.start, "%Y-%m-%d").date()
    if args.end:
        end_date = dt.datetime.strptime(args.end, "%Y-%m-%d").date()
    else:
        end_date = start_date

    # HH:MM を "_HHMM" に変換
    hh, mm = args.time.split(":")
    time_suffix = f"_{int(hh):02d}{int(mm):02d}"

    script_path = Path(args.script).resolve()
    if not script_path.exists():
        print(f"[ERROR] script not found: {script_path}", file=sys.stderr)
        sys.exit(1)

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"[WARN] config not found: {config_path} (スクリプト側のデフォルトを使用します)")

    for d in iter_dates(start_date, end_date):
        ts = d.strftime("%y%m%d") + time_suffix  # 例: "251123_1600"
        env = os.environ.copy()
        env["EQ_RUN_TS"] = ts

        print("-" * 80)
        print(f"[BACKFILL] date={d.isoformat()}  EQ_RUN_TS={ts}")
        print("-" * 80)

        cmd = [sys.executable, str(script_path)]
        if config_path.exists():
            cmd.extend(["-c", str(config_path)])

        try:
            ret = subprocess.run(cmd, env=env, check=False)
        except KeyboardInterrupt:
            print("\n[BACKFILL] interrupted by user")
            break
        except Exception as e:
            print(f"[BACKFILL] error while running {script_path.name} for {ts}: {e}")
            continue

        if ret.returncode != 0:
            print(f"[BACKFILL] script exited with code {ret.returncode} for {ts}")

    print("\n[BACKFILL] done.")


if __name__ == "__main__":
    main()
