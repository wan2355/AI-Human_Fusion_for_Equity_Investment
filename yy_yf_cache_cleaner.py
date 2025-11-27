#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from yy_yf_cache import clear_cache

def main():
    ap = argparse.ArgumentParser(description="yfinance キャッシュ(.yf_cache)の管理")
    ap.add_argument(
        "--clear",
        action="store_true",
        help="キャッシュを全削除する（指定が無い場合は何もしない）",
    )
    ap.add_argument(
        "--older-than-days",
        type=int,
        default=None,
        help="この日数より古いキャッシュだけ削除（--clear と併用）",
    )
    args = ap.parse_args()

    if not args.clear:
        print("[INFO] --clear が指定されていないので、キャッシュ削除は行いません。")
        return

    removed = clear_cache(older_than_days=args.older_than_days)
    if args.older_than_days is None:
        print(f"[OK] .yf_cache 内のファイルを {removed} 個削除しました。")
    else:
        print(f"[OK] {args.older_than_days}日より古いキャッシュを {removed} 個削除しました。")

if __name__ == "__main__":
    main()

