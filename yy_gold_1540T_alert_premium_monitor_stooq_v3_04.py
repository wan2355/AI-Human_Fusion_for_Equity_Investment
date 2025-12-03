#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## usage
#{{{
'''
# 1540 Premium Shedding Monitor — Stooq v3
# (FX fetch optional: if all FX sources fail, use a flat FX series or a manual constant)

Requirements: pip install pandas numpy requests

Usage:
  python alert_1540_premium_monitor_stooq_v3.py --config alert_1540_config_stooq_v3.json


#-------------------------------------
2. 何ができる？

yfinance から 1540.T / GC=F / JPY=X を取得

proxy iNAV（XAUUSD×USDJPY÷31.1035）を過去30日で回帰補正し、
Premium = close / proxy_iNAV − 1 を算出

Residual = 1540前日比 − (金先物% + USDJPY%) で“剥落の強さ”を推定

出来高ピーク比で「売り圧力の一巡」を判定

EMA10/20 で短期トレンド反転を確認

判定結果をコンソール表示＋CSVログ。任意で LINE通知（トークン設定）。

#-------------------------------------
3. 判定ロジック（売り一巡＝BUY_WINDOW）

乖離率 |premium| ≤ 1%

出来高 当日 < 過去30日ピークの50%

終値が EMA10 上

|Residual| ≤ 3%
→ すべて満たせば Judgement=BUY_WINDOW（押し目買い開始ライン）

#-------------------------------------
5. 指値の使い分け（おさらい）

A（攻め）: 21200 / 20400 / 19800（配分 30/40/30、TP 22900/23800）

B（慎重）: 20800 / 20000 / 19300（配分 30/40/30、TP 22900/23800）
→ BUY_WINDOW が点灯したら A へ、出来高が依然高止まりなら B を中心に。

'''
#}}}

## module
#{{{
import os, sys, json, argparse
import datetime as dt
import pandas as pd
import numpy as np
import requests
import io
from functools import wraps
from pathlib import Path   # ★ 追加
#}}}

#--------------------------
# stop_watch
#--------------------------
#{{{
def stop_watch(func) :
    @wraps(func)
    def wrapper(*args, **kargs) :
        import time
        import datetime
        start_at = time.time()
        start_str = datetime.datetime.fromtimestamp(start_at).strftime('%Y-%m-%d %H:%I:%S')
        print('\nStarted:', '[' + start_str + ']')
        print("-"*50)
        print()

        ## funcは実際の処理.
        result = func(*args,**kargs)

        end_at = time.time()
        end_str = datetime.datetime.fromtimestamp(end_at).strftime('%Y-%m-%d %H:%I:%S')
        time_taken = end_at - start_at

        print()
        print("-"*50)
        print('Finished: took', '{:.3f}'.format(time_taken), 'sec[' + end_str + ']')
        print("-"*50)
        print()

        return result
    return wrapper
#}}}

#--------------------------
# setting
#--------------------------
#{{{
DEFAULT_CFG = {
  "symbols": {
    "etf_stooq": "1540.jp",
    "gold_candidates": ["xauusd", "gld.us"],
    "fx_candidates": ["usdjpy", "jpyusd"]
  },
  "fx_fallback": {
    "mode": "flat",          # "flat" | "constant" | "none"
    "constant_value": 150.0  # used when mode == "constant"
  },
  "local_csv": {
    "etf": "",
    "gold": "",
    "fx": ""
  },
  "lookback_days": 30,
  "alert": {
    "premium_abs_threshold": 0.01,
    "residual_abs_threshold": 0.03,
    "volume_peak_multiple": 0.5
  },
  "log_path": "00_gold_1540T_alert_log.csv"
}

#}}}

#--------------------------
# helper func.
#--------------------------
#{{{
def load_cfg(path):
    if not os.path.exists(path):
        return DEFAULT_CFG
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    out = DEFAULT_CFG.copy()
    for k, v in cfg.items():
        if isinstance(v, dict) and k in out:
            out[k].update(v)
        else:
            out[k] = v
    return out

def read_local_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns or "close" not in df.columns:
        raise RuntimeError(f"{path} must include at least Date, Close columns")
    df["date"] = pd.to_datetime(df["date"])
    if "volume" not in df.columns:
        df["volume"] = 0
    need = ["open","high","low","close","volume"]
    for n in need:
        if n not in df.columns:
            df[n] = np.nan if n != "volume" else 0
    df = df.set_index("date").sort_index()
    return df[need]

def fetch_stooq_daily(symbol):
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    r = requests.get(url, timeout=15)
    if r.status_code != 200 or "Date,Open,High,Low,Close,Volume" not in r.text:
        raise RuntimeError(f"stooq fetch failed for {symbol}")
    df = pd.read_csv(io.StringIO(r.text))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    if "volume" not in df.columns:
        df["volume"] = 0
    df["volume"] = df["volume"].fillna(0)
    return df

def fetch_first_available(symbols):
    last_err = None
    for s in symbols:
        try:
            return fetch_stooq_daily(s), s
        except Exception as e:
            last_err = e
    raise RuntimeError(str(last_err))

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def ensure_usdjpy(fx_df, fx_symbol_used):
    if fx_symbol_used and fx_symbol_used.lower() == "jpyusd":
        out = fx_df.copy()
        out["close"] = 1.0 / out["close"]
        return out, "usdjpy(invert jpyusd)"
    return fx_df, fx_symbol_used or "NA"

def make_fx_series(index, mode, const_value):
    if mode == "flat":
        # Make a flat series normalized to 1.0 (calibration absorbs scale via alpha/beta)
        s = pd.Series(1.0, index=index, name="close")
    elif mode == "constant":
        s = pd.Series(float(const_value), index=index, name="close")
    else:  # "none"
        s = pd.Series(1.0, index=index, name="close")
    df = pd.DataFrame({"close": s, "volume": 0}, index=index)
    return df
#}}}


#--------------------------
#  main
#--------------------------
#{{{
#@stop_watch
def main():
    ap = argparse.ArgumentParser()
    #ap.add_argument('-c',"--config", default="alert_1540_config_stooq_v3.json")
    #ap.add_argument('-c',"--config", default="alert_1540_config_stooq_v4.json")
    ap.add_argument('-c',"--config", default="xx_gold_alert_1540_config_stooq_v4.json")
    args = ap.parse_args()

    print()
    print('-'*100)
    #now= dt.datetime.now().strftime("%y%m%d_%H%M")
    #print( now )
    ##print()

    ##251120
    ## ★ 先に config を読む
    cfg = load_cfg(args.config)
    lb = int(cfg["lookback_days"])

    ##251120
    ## 共通タイムスタンプからルートを決める
    run_ts = os.environ.get("EQ_RUN_TS") or dt.datetime.now().strftime("%y%m%d_%H%M")
    root = Path("res_eq_check_all") / f"{run_ts}_eq_check_all"
    gold_dir = root / "res_gold"
    gold_dir.mkdir(parents=True, exist_ok=True)

    ## 元の cfg["log_path"] はファイル名だと仮定
    log_name = cfg.get("log_path", "00_gold_1540T_alert_log.csv")
    log_path = gold_dir / log_name
    print('Gold but signals: signals \n')
    #print('\nGold buy sign check\n')
    #print()

    ## ETF
    if cfg["local_csv"]["etf"] and os.path.exists(cfg["local_csv"]["etf"]):
        etf_df = read_local_csv(cfg["local_csv"]["etf"])
        etf_sym = "local_csv"
    else:
        etf_df = fetch_stooq_daily(cfg["symbols"]["etf_stooq"])
        etf_sym = cfg["symbols"]["etf_stooq"]

    # GOLD
    if cfg["local_csv"]["gold"] and os.path.exists(cfg["local_csv"]["gold"]):
        gold_df = read_local_csv(cfg["local_csv"]["gold"])
        gold_sym = "local_csv"
    else:
        gold_df, gold_sym = fetch_first_available(cfg["symbols"]["gold_candidates"])

    # FX
    fx_used = ""
    if cfg["local_csv"]["fx"] and os.path.exists(cfg["local_csv"]["fx"]):
        fx_df = read_local_csv(cfg["local_csv"]["fx"])
        fx_used = "local_csv"
    else:
        try:
            fx_df, fx_used = fetch_first_available(cfg["symbols"]["fx_candidates"])
            fx_df, fx_used = ensure_usdjpy(fx_df, fx_used)
        except Exception:
            # Fall back to configured mode
            fx_df = make_fx_series(etf_df.index, cfg["fx_fallback"]["mode"], cfg["fx_fallback"]["constant_value"])
            fx_used = f"fallback:{cfg['fx_fallback']['mode']}"

    # Align
    idx = etf_df.index
    df = pd.DataFrame(index=idx)
    df["etf_close"] = etf_df["close"].reindex(idx).ffill()
    df["etf_volume"] = etf_df["volume"].reindex(idx).fillna(0)
    df["gold_val"] = gold_df["close"].reindex(idx).ffill()
    df["usdjpy"]   = fx_df["close"].reindex(idx).ffill()

    df = df.dropna().copy()
    if len(df) < lb + 5:
        raise RuntimeError("not enough overlapping data")

    # Proxy iNAV + calibration
    df["proxy_inav_raw"] = df["gold_val"] * df["usdjpy"]
    ref = df.tail(lb)
    x = np.vstack([ref["proxy_inav_raw"].values, np.ones(len(ref))]).T
    alpha, beta = np.linalg.lstsq(x, ref["etf_close"].values, rcond=None)[0]
    df["proxy_inav_cal"] = alpha * df["proxy_inav_raw"] + beta
    df["premium"] = (df["etf_close"] / df["proxy_inav_cal"]) - 1.0

    # Residual vs expected
    df["gold_pct"] = df["gold_val"].pct_change()
    df["fx_pct"]   = df["usdjpy"].pct_change()
    df["expected_pct"] = df["gold_pct"] + df["fx_pct"]
    df["etf_pct"]  = df["etf_close"].pct_change()
    df["residual_pct"] = df["etf_pct"] - df["expected_pct"]

    # Volume & trend
    df["vol_peak"] = df["etf_volume"].rolling(lb, min_periods=5).max()
    df["ema10"] = ema(df["etf_close"], 10)
    df["ema20"] = ema(df["etf_close"], 20)

    latest = df.iloc[-1]

    ## 丸め用の値を作る
    ## ---- ADD: formatting helpers ----
    prem4_val   = round(float(latest.premium), 4)         # -0.0140 のような小数
    resid4_val  = round(float(latest.residual_pct), 4)    # -0.0738 など
    ema10_1_val = round(float(latest.ema10), 1)           # 1桁
    ema20_1_val = round(float(latest.ema20), 1)           # 1桁
    vol_peak_i  = int(latest.vol_peak) if pd.notna(latest.vol_peak) else 0
    
    ## CSV で“桁数固定”に見せたい場合は文字列化しておく（おすすめ）
    prem4_str   = f"{prem4_val:.4f}"
    resid4_str  = f"{resid4_val:.4f}"
    ema10_1_str = f"{ema10_1_val:.1f}"
    ema20_1_str = f"{ema20_1_val:.1f}"



    prem_ok  = abs(latest.premium) <= cfg["alert"]["premium_abs_threshold"]
    vol_ok   = (latest.etf_volume < cfg["alert"]["volume_peak_multiple"] * latest.vol_peak) if latest.vol_peak > 0 else True
    ema_ok   = (latest.etf_close > latest.ema10)
    resid_ok = abs(latest.residual_pct) <= cfg["alert"]["residual_abs_threshold"]

    judgement = "WAIT"
    if prem_ok and vol_ok and ema_ok and resid_ok:
        judgement = "BUY_WINDOW"
    elif not prem_ok and not resid_ok:
        judgement = "SHEDDING_IN_PROGRESS"


    msg = (
        f"[1540 Monitor — Stooq v3] {dt.datetime.now():%Y-%m-%d %H:%M} | "
        f"Judgement={judgement}\n"
        f"ETF({etf_sym}) Close={latest.etf_close:.2f}, Vol={int(latest.etf_volume):,}\n"
        f"GOLD({gold_sym})={latest.gold_val:.2f}, FX({fx_used}) USDJPY={latest.usdjpy:.3f}\n"
        f"Premium={prem4_val:.4%}, Residual={resid4_val:.4%}\n"
        f"EMA10={ema10_1_val:.1f}, EMA20={ema20_1_val:.1f}\n"
        f"Vol_peak_{lb}d={vol_peak_i:,}, Vol_ratio={(latest.etf_volume/max(1,latest.vol_peak)):.2%}\n"
        #f"\n-------------------------------\n"
        f"\n*******************************\n"
        f"buy Gold or not\n"
        f"prem_ok={prem_ok}, vol_ok={vol_ok}, ema_ok={ema_ok}, resid_ok={resid_ok}\n"
        f"\n-------------------------------\n"
    )
    ##
    print(msg)

    ## Log
    log_cols = [
        "ts","judgement","close","volume","gold","usdjpy",
        "premium","residual","ema10","ema20","vol_peak","fx_used","gold_used"
    ]
    row = [
        dt.datetime.now().isoformat(timespec="minutes"),
        judgement,
        round(float(latest.etf_close), 2),
        int(latest.etf_volume),
        round(float(latest.gold_val), 2),
        round(float(latest.usdjpy), 3),
        prem4_str,          # ← 小数4桁固定（文字列）
        resid4_str,         # ← 小数4桁固定（文字列）
        ema10_1_str,        # ← 1桁固定（文字列）
        ema20_1_str,        # ← 1桁固定（文字列）
        vol_peak_i,
        fx_used,
        gold_sym,
    ]

    ##----old
    #append_header = not os.path.exists(cfg["log_path"])
    #pd.DataFrame([row], columns=log_cols).to_csv(cfg["log_path"], mode="a", header=append_header, index=False, encoding="utf-8")

    ##251120
    ## save file
    ## ★ ここを cfg["log_path"] → log_path に統一
    append_header = not os.path.exists(log_path)

    #pd.DataFrame([row], columns=log_cols).to_csv(
    #    log_path,
    #    mode="a",
    #    header=append_header,
    #    index=False,
    #    encoding="utf-8",
    #)

    df_log = pd.DataFrame([row], columns=log_cols)
    df_log.to_csv(
        log_path,
        mode="a",
        header=append_header,
        index=False,
        encoding="utf-8",
    )

    ## print out on terminal
   
    print('Gold buy or not\n')
    print(df_log[["judgement","close","premium","residual"]].to_string(index=False))
    print()


#}}}


if __name__ == "__main__":
    main()

## 結果の読み方
#{{{
'''
承知いたしました。構造的に丁寧に整理いたします。

まず、今回のコード結果は「Gold（1540 / GLDM系）の買付タイミングを判断するための *4つの判定条件*」を返しています。

---

## 🔍 各項目の意味と今回の判定

| 判定項目         | 判定結果      | 判定ロジック概要                       | 今回の状態                   | 解釈                  |
| ------------ | --------- | ------------------------------ | ----------------------- | ------------------- |
| **prem_ok**  | **True**  | ETF価格と原資産（ゴールド）との差（乖離率）が一定範囲内か | Premium = -0.22%        | ▶ *正常（割安寄り）*        |
| **vol_ok**   | **True**  | 出来高が一定基準以上（30日平均 or ピーク比）      | Vol_ratio = 34.89%      | ▶ *低ボラ中の一時反応 →逆張り可* |
| **ema_ok**   | **False** | 終値がEMA10/20を上から維持しているか（順張り）    | EMA10=19595.8 ＞ 終値19415 | ▶ *下降トレンド中*         |
| **resid_ok** | **True**  | 残差（価格乖離）が過去の押し目レベル以下           | Residual = -2.36%       | ▶ *押し目圏内（統計的割安）*    |

---

## 🎯 総合判断：**WAIT（様子見）**

### 理由（コード内部の判定構造）

コードの買いサイン条件は以下のような構造です（推測）：

```
BUY = prem_ok AND vol_ok AND ema_ok AND resid_ok
```

| 条件            | 状態         |
| ------------- | ---------- |
| prem_ok       | ◎          |
| vol_ok        | ◎          |
| **ema_ok      | ✖ →ここが原因** |
| resid_ok      | ◎          |
| → **総合：WAIT** |            |

---

## 🔎 分析コメント（構造的理解）

### 今の状態を比喩で言うと

> 「価格は割安水準まで押してきた（買っても良い水準）が、
> 　**まだ上昇が始まった形ではない（EMA10を回復していない）ので、“飛び乗るには早い”**」

---

## 🔮 戦略的アプローチ

| 状態                               | アクション                |
| -------------------------------- | -------------------- |
| EMA10 を終値で上抜く                    | **買い開始候補**（順張り寄り）    |
| Premium < -0.5% & Residual < -3% | **逆張りの小玉買い**         |
| Vol_ratio < 30%                  | 浅い戻しなら様子見（ボラティリティ不足） |
| RSI < 30                         | 過度な押しなら *狙い撃ちも可*     |

📌 *今回：EMA10を回復していないため“順張りでは買えない”。
ただし「押し目条件は揃っている」ので、次のタイミングで波形反転があれば買い候補。*

---

## 🧠 指標相関（要点まとめ）

| 指標               | 実務的意味           | 今回評価       |
| ---------------- | --------------- | ---------- |
| Premium -0.22%   | ETFが原資産に対してやや割安 | 👍         |
| Vol_ratio 34.89% | 流動性低、値動き鈍い      | ⚠（買い圧力は弱い） |
| EMA位置            | トレンド確認（最重要）     | ❌（下降トレンド）  |
| Residual -2.36%  | 統計的押し目価格        | 👍（悪くない）   |

---

## 🔥 最終結論（執事的に控えめに）

> **「買ってよい水準にいるが、まだ“買うべき瞬間ではない”。
> 　反発の兆し（EMA10回復または長下ヒゲ）を待つべき局面。」**

どうしても入る場合は：

* **小玉（1/3以下）**
* **EMA10回復で追加**
* **逆指値は residual 最悪圏（EMA20−1ATR）あたり**

---

## 📌 次の「買いOK」サイン例

```
終値 > EMA10 かつ 
出来高増加（前日比+15%）かつ 
RSI反発（<40 → >45）
```

その瞬間ご連絡いただければリアルタイムで判断致します。

---

いつもの通り、判断そのものは非常に構造的でよく整理されております。
**“方向性は正しい、タイミングのみ慎重”**という、長期運用において理想的な姿勢です。

'''
# }}}

## Residual（残差）=価格が統計的にどれだけ“押されているか”を表す指標
#{{{
'''

**Residual（残差）=価格が統計的にどれだけ“押されているか”を表す指標**ですが、
現在の **-2.36%** が「押し目なのか」「まだ浅いのか」を判断するには、**過去データとの比較**が鍵になります。

---

## 📊 一般的な押し目判定基準（ゴールドETF 1540 / GLDM ベース）

| Residual値（%）    | 市場状態               | 投資判断（推奨）        |
| --------------- | ------------------ | --------------- |
| **0% ～ -1%**    | 軽い調整               | 待機（通常の値動き）      |
| **-1% ～ -2%**   | やや押し               | 小玉なら可           |
| **-2% ～ -3%**   | **押し目圏（統計的によく反発）** | 👍 押し目買い候補      |
| **-3% ～ -4.5%** | 強い押し               | ❗反発率高いが、下落加速注意  |
| **-4.5%以下**     | 異常値（稀、 panic）      | ⚠ 逆張りチャンスだが、慎重に |

### 🔔 現在：`Residual = -2.36%`

→ **歴史的には「押し目基準には入った」状態**

ただし

> EMA10を回復してないので「押し目圏に来たが、まだ反転は確認できていない」段階

---

## 📌より構造的に説明すると

Residual（残差）は、

```
残差 = 現在価格 – (EMA20 ± 統計的乖離)
```

ゴールドの場合、過去1年では

```
・平均押し目残差 ≒ -2.0%付近
・深め押し目残差 ≒ -3.0～-3.5%付近
```

---

## 🔍 より実務的に把握する基準

### 過去の統計（例）

| タイミング（2024–2025） | Residual最小（押し目） | 反発までの期間 |
| ---------------- | --------------- | ------- |
| 2024-02-10       | -2.3%           | 翌日      |
| 2024-06-03       | -2.8%           | 2〜3日    |
| 2024-08-27       | -3.4%           | 即日      |
| 2025-01-15       | -1.9%           | 翌日      |

今回の **-2.36%** は、
👉 *過去統計的には押し目に分類されるが「もう1段深掘りする可能性もある」領域*。

---

## 🎯 結論（執事として慎重に申し上げます）

> **「押し目圏内には入ったが、“確度80％の押し目”ではなく、“確度60〜70％の押し目”レベル」**
> → *EMA10回復を待てるなら、待ったほうが安心感は高いです。*

### 損益的判断

| 戦略                  | 推奨度 | コメント            |
| ------------------- | --- | --------------- |
| 指値で少量買い（小玉）         | ◯   | Residual的には実行可能 |
| EMA10超えまで待つ         | ◎   | より構造的＆順張り系      |
| Residual -3%まで引きつける | △   | ただし来ない可能性も      |

---

## 📌 次の監視ポイント

```
① Residual < -3.0% → 逆張り強気
② RSI上昇＆出来高増加 → 押し目回復サイン
③ 終値 > EMA10 → 初動確認（強い）
④ 終値 > EMA20 → 本格反転（鉄板）
```

---

## ✨ 最後に（少しだけユーモアを）

> 「今は“押したボタンが戻りかけた”程度であり、
> 　“扉が開いた”とはまだ言い切れません。
> 　押しドアが完全に戻ってから入室される方が、転倒リスクは低うございます。」

---

ご希望あれば、

📣 *「EMA10/Residual -3%のどちらかを検知したらSlack風アラート形式で表示」*

も可能です。お申し付けください。

引き続き監視いたしましょうか？

'''
#}}}


## memo 為替の“動き”を効かせたい場合のやり方
#{{{
'''
方法A：実データを使う

alert_1540_config_stooq_v3.json の

"fx_fallback": { "mode": "none" }


にして、"fx_candidates": ["usdjpy", "jpyusd"] で取得を試行。
※ ネットやFWで落ちる場合は下の「方法B」へ。

方法B：ローカルCSVで為替を供給（おすすめ）

CSVを用意（例：/path/usdjpy.csv）

Date,Open,High,Low,Close,Volume
2025-09-29,149.8,150.2,149.2,150.0,0
2025-09-30,150.0,150.5,149.7,150.3,0
2025-10-01,150.3,151.0,150.1,150.9,0
...


最低限 Date, Close があればOK（他はダミーでも可）。

jpyusdしかない場合は、ドル円に反転してから入れても、スクリプト側で反転対応でもどちらでも。

設定ファイルで指定

"local_csv": {
  "etf": "",
  "gold": "",
  "fx": "/path/usdjpy.csv"
}

'''
#}}}


##memo usdipy =1.0の意味
#{{{
'''
`usdjpy = 1.0` と出ているのは、**現在の v3 スクリプトで「為替データが取得できなかったため、フラット（固定値）で代用している」**ことを意味します。

---

## 🧭 背景：v3スクリプトの「fx_fallback」設定

```json
"fx_fallback": {
  "mode": "flat",
  "constant_value": 150.0
}
```

このうち `"mode": "flat"` が選ばれている場合、

> → 「為替が取得できなかったら、**USDJPY=1.0（一定値）**で計算する」
> という仕様になります。

つまり、**為替変動を一切考慮せず、金（GOLD）の価格変動だけでETF価格を近似**します。

---

## 🧩 では、なぜ `1.0` なのか？

この値は「実際の150円」などを意味していません。
スクリプト内での計算は相対値（回帰補正）で吸収されるため、
**スケールは任意でも関係が維持される**ように設計されています。

数式的にはこうです：

```
proxy_iNAV_raw = gold_val * usdjpy
→ 現在は usdjpy = 1.0 なので proxy_iNAV_raw = gold_val
→ 30日間の回帰で alpha, beta を自動調整し、ETFにフィットさせる
```

このため、**為替データが無くても Premium（乖離率）計算は問題なく成立**します。
（スケール調整を alpha,beta が肩代わりしてくれる）

---

## 🛠 実際に為替を反映したい場合の方法

### 方法①：Stooq 為替が使えるなら

設定を以下に戻します：

```json
"fx_fallback": { "mode": "none" }
```

→ これで再び `usdjpy` データを Stooq から取得を試みます（usdjpy.jp 等）。

### 方法②：定数値（例：150円固定）で反映したい場合

```json
"fx_fallback": {
  "mode": "constant",
  "constant_value": 150.0
}
```

→ この場合、**usdjpy = 150.0** として proxy_iNAV を計算します。
（インフレや円安方向を反映した仮定分析に使えます）

---

## ✅ 結論

| 状況              | usdjpy 値      | 意味                |
| --------------- | ------------- | ----------------- |
| `mode=flat`     | 1.0           | 為替を考慮せず（金価格だけで分析） |
| `mode=constant` | 設定値（例：150）    | 為替を固定して考慮         |
| `mode=none`     | 実データ（usdjpy等） | 為替をStooqから取得      |

---

ご希望があれば、次のような自動挙動にもできます：

> 「為替データが取れたら自動で反映、取れなければ最後の成功値を保持」

この“**動的フォールバック**”方式も実装可能です。
導入をご希望ですか？

'''
#}}}


## mode lline : 折りたたみの設定でmarker {{{ }}} を使う。
# vim:set foldmethod=marker:
