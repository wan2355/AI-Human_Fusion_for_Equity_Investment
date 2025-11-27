# monte_carlo_portfolio

 **Disclaimer / 免責**  
 **EN:** The author cannot take responsibility for any results produced by using this software. Use at your own risk.  
 **JA:** 本ソフトウェアの利用により生じたいかなる結果についても、作者は一切の責任を負いません。自己責任でご利用ください。

・作者は怠け者ですので、質問等に対する回答はいつできるかわかりません。ご容赦ください。

---

## Table of Contents / 目次
- [Overview / 概要](#overview--概要)
- [Features / 特長](#features--特長)
- [Repository Layout / 構成](#repository-layout--構成)
- [Requirements / 必要要件](#requirements--必要要件)
- [Installation / インストール](#installation--インストール)
- [Quickstart / クイックスタート](#quickstart--クイックスタート)
- [Configuration (`portfolio.json`) / 設定](#configuration-portfoliojson--設定)
- [Usage / 使い方](#usage--使い方)
- [Outputs / 生成物](#outputs--生成物)
- [Reproducibility / 再現性](#reproducibility--再現性)
- [License / ライセンス](#license--ライセンス)
- [No Redistribution Policy / 再配布禁止ポリシー](#no-redistribution-policy--再配布禁止ポリシー)
- [AI Assistance / AI利用の明示](#ai-assistance--ai利用の明示)
- [Contact & Commercial License / 連絡先・商用ライセンス](#contact--commercial-license--連絡先商用ライセンス)
- [Contributing / コントリビュート](#contributing--コントリビュート)
- [Acknowledgements / 謝辞](#acknowledgements--謝辞)

---

## Overview / 概要
- **EN:** Monte Carlo–based portfolio analysis and visualization. It explores scenarios (e.g., inflation ranges, FX shocks, t-distribution tails) and plots result distributions.
- **JA:** モンテカルロ法を用いたポートフォリオ解析・可視化ツール。インフレ範囲、為替ショック、t分布の厚い尾などのシナリオを走らせ、結果分布を可視化します。

---

## Features / 特長
- **EN:**  
  - Scenario sweeps across inflation / FX / distribution params  
  - Portfolio simulation over configurable horizon and steps  
  - Plots for return distribution / drawdown / percentile bands  
  - Config-driven (JSON), scriptable runs
- **JA:**  
  - インフレ・為替・分布パラメータの一括スイープ  
  - 期間・ステップ数を設定可能なシミュレーション  
  - 収益分布の可視化(Histogram etc)  

---

## Repository Layout / 構成
```
.
├─ README.md / README.ja.md (optional)
├─ LICENSE
├─ portfolio.json                # 設定ファイル（例）
├─ yy1_plot_sweep_multi.py       # 可視化・プロット
├─ xx_monte_carlo_simulation_*.py   # シミュレーション本体（例）
└─ result/                          # 図・CSV 等の出力先（任意）
```

- **EN:** File names may differ; adapt commands accordingly.  
- **JA:** 実ファイル名が異なる場合は、読み替えてください。

---

## Requirements / 必要要件
```
- **EN:** Python 3.10.x above ; typical libs: `numpy`, `pandas`, `matplotlib`, `scipy` (and others if your scripts import them).  
- **JA:** python 3.10.x 以上。一般的な依存：`numpy`、`pandas`、`matplotlib`、`scipy`（スクリプトが `import` するライブラリに合わせて追加）。
```

---

## Installation / インストール
```bash
pip install -U pip
pip install numpy pandas matplotlib scipy
## you might need more module for your env.
````

---

## Quickstart / クイックスタート

```bash
## inflation sweep mode
## set inflation rate 0.02, 0.03, and compare result.
python xx1_monte_carlo_simulation_inflation_range_fx_adjust_t-dist.py \
    -i 10_000_000 -y 30 -p portfolio.json \
    --withdraw_start_years 5 -w 500_000 \
    --inflation_min 0.02 --inflation_max 0.03 --inflation_sweep_steps 2 \
    --base_fx_rate 147 --fx_annual_vol 0.1 \
    --t_df 6 --fx_t_df 6 \
    -z wd3_comp_if2-3_wd50man_i1000man_fx_adjust0.10


## inflation fix mode, and get histogram snapshot every year.
python xx1_monte_carlo_simulation_inflation_range_fx_adjust_t-dist.py \
   -i 10_000_000  -p portfolio.json \
   --withdraw_start_years 10 -w 500_000 \
   --inflation_rate 0.03 \
   --base_fx_rate 147 --fx_annual_vol 0.1 \
   --t_df 6 --fx_t_df 6 \
   -z wd10_comp_if3_wd50man_i1000man_fx_adjust0.10
   
```
* **JA:** 上は実行例です。実際の引数や挙動はスクリプト内のコメントを参照してください。

---

## Configuration (`portfolio.json`) / 設定

> **EN:** Example template (adjust to your data).
> **JA:** テンプレ（実データに合わせて修正）。
>
> EN: add  usd or  foreign for  fx_adjustment
>
> JA: 外貨投資は usd or foreign をつける.

```json
{
    "usd: all_country": {
        "weight": 0.3,
        "mean_return": 0.075,
        "std_dev": 0.15
    },
    "usd: S&P500": {
        "weight": 0.2,
        "mean_return": 0.09,
        "std_dev": 0.18
    },
    "jpy: balanced_fund": {
        "weight": 0.3,
        "mean_return": 0.06,
        "std_dev": 0.0914
    },
    "jpy: ニッセイJ-REIT A": {
        "weight": 0.2,
        "mean_return": 0.0563,
        "std_dev": 0.1201
    }
}
```

---

## Usage / 使い方

### A. CLI examples / コマンド例

```bash
## 1) Sweep + auto overlay plot. : infaltionをいくつかのバージョンに分けて比較。
##    ただし、inflationは固定。
## inflation_min ~ max の間を inflation_sweep_stepで分割して、その値のinflationで固定して最後まで計算.
##
## eg) step 5 ならば,以下のinflationで計算。
##  0.02 0.0225 0.025 0.0275 0.03

## t分布 : for conte_carlo
##  t_df : jsonで設定した各資産の標準偏差を使った月初の資産のゆらぎ
##  fx_t_df : 為替のゆらぎ。
##   use 5~8 , lower num is strong.

## 1年ごとのhistgram の snapshotなし。
## add comment  -z

sample)
----------
## intial asset : 10_000_000
## running 30yrs
## asset info : portfolio.json_custom, (defalt : portfolio.json)
## start withdraw 3years later
## annual_withdraw=50万/year
## set inflation rate 0.02, 0.03/yr, and calc & compare.
## use t-dist for asset & fx for monte carlo.
##  --t_df : 5~8 , lower num is strong.

python xx1_monte_carlo_simulation_inflation_range_fx_adjust_t-dist.py \
    -i 10_000_000 -y 35 -p portfolio.json \
    --withdraw_start_years 5 -w 500_000 \
    --inflation_min 0.02 --inflation_max 0.03 --inflation_sweep_steps 2 \ 
    --base_fx_rate 147 --fx_annual_vol 0.1 \
    --t_df 6 --fx_t_df 6 \
    -z wd5_comp_if2-3_wd50man_i1000man_fx_adjust0.10


## 2) Single run with yearly snapshot histograms every year, using random num(t-dist).
##    inflationを変動させて引用してみる。
##    --inflation_min 0.02 --inflation_max 0.03 は「0.02〜0.03の乱数」で1回/yrの実行を行う.
##
## intial asset : 10_000_000
## running 35yrs
## asset info : portfolio.json_custom, (defalt : portfolio.json)
## start withdraw 3years later
## annual_withdraw=50万/year
## set inflation rate 0.02, 0.03/yr, but 「0.02〜0.03の乱数」で1回/yrの実行を行う.
## use t-dist for asset & fx for monte carlo.
##  --t_df : 5~8 , lower num is strong.
##

python xx1_monte_carlo_simulation_inflation_range_fx_adjust_t-dist.py \
  -i 10_000_000 -y 35 -p portfolio.json \
  --inflation_mode per_year --inflation_min 0.02 --inflation_max 0.03 \
  --withdraw_start_years 5 --annual_withdrawal 500_000 \
  --save_snapshot_hists --mode_interval_years 1 \
  --base_fx_rate 147 --fx_annual_vol 0.1 \
  -z wd5_if2-3_wd50man_i1000man_fx_adjust0.10 \
  --t_df 6


# 3) fix_inflation rate and save hist every year.
##  inflation 固定.
##  --inflation_mode fixed --inflation_rate X で Xで固定、実行。
##   1年ごとのhistgramのsnapshot可能。
##
## intial asset : 10_000_000
## running 35yrs
## asset info : portfolio.json_custom, (defalt : portfolio.json)
## start withdraw 3years later
## annual_withdraw=50万/year
## set inflation rate 0.03/yr
## set histogram x-range -2*10^7 ~1.5*10^8 yen to compare hist.
## use t-dist for asset & fx for monte carlo.
##  --t_df : 5~8 , lower num is strong.

python xx1_monte_carlo_simulation_inflation_range_fx_adjust_t-dist.py \
  -i 10_000_000 -y 35 -p portfolio.json \
  --withdraw_start_years 10 --annual_withdrawal 500_000 \
  --save_snapshot_hists --mode_interval_years 1 \
  --inflation_rate 0.03 \
  --base_fx_rate 147 --fx_annual_vol 0.1 \
  --fx_annual_vol 0.1 \
   --x_min -20000000 --x_max 150_000_000 \
  --t_df 6 \
  --fx_t_df 6 \
  -z wd10_if3.0_wd50man_fx_adjust0.10

## 以下は　t分布を使用せず、正規分布で計算(default).
python xx1_monte_carlo_simulation_inflation_range_fx_adjust_t-dist.py \
  -i 10_000_000 -y 35 -p portfolio.json \
  --withdraw_start_years 10 --annual_withdrawal 500_000 \
  --save_snapshot_hists --mode_interval_years 1 \
  --inflation_rate 0.03 \
  --base_fx_rate 147 --fx_annual_vol 0.1 \
  --fx_annual_vol 0.1

```

### B. Typical parameters / 主なパラメータ
```
引数の説明

## コア・実行制御

--simulation, -s      : int（default 10000） モンテカルロ試行数
--initial_asset, -i   : float           初期資産（JPY）
--years, -y           : int             運用年数
--portfolio, -p       : str             ポートフォリオJSONのパス
--seed                : int, 42(default)  乱数シード（再現性）
--early_terminate_if_ruined : flag      破綻（資産<=0）でその経路を打ち切り
---------------------------------------------------------------

## 取り崩し・スナップショット

--annual_withdrawal, -w : float   初年度取り崩し額（名目, JPY）
--withdraw_start_years  : int     取り崩し開始年（0=初年）
--save_snapshot_hists   : flag    途中年ヒストグラムを年次出力
--mode_interval_years   : int     ヒスト/モード抽出間隔（年, 既定=1）
--inflation_sweep_steps : int     スイープ分割数（等間隔）

 備考: 取り崩し額は毎年「前年取り崩し×(1+inflation)」で増額。
--------------------------------------------------------

## インフレ（inflation）

--inflation_rate        : float   単発実行時の年率インフレ
--inflation_min/max     : float   スイープの下限/上限（年率）
--inflation_sweep_steps : int     スイープ分割数（等間隔）

 備考: 取り崩し額は毎年「前年取り崩し×(1+inflation)」で増額。
 資産は実質換算（インフレ影響を調整）。
---------------------------

## 資産リターンの分布（t分布）

--t_df : float or None  df>2 で Student-t を使用（df 小ほど裾が厚い）。
 指定/<=2 は正規分布。
-------------

## 為替（FX：毎月変動＋初期レベル差）

--base_fx_rate          : float   基準レート（JPY/USD想定, 例 147）
--fx_min / --fx_max     : float   経路ごとに初期FXレートを一様分布で1回サンプル
--fx_annual_vol         : float   年率ボラ（月次は /√12）
--fx_drift_annual       : float   年率ドリフト（月次は 12乗根で換算）
--fx_t_df               : float   月次FXにも t分布を使用（df>2）。未指定は正規

 備考: 外貨判定は資産名にportfolio.json中に "foreign" または "usd" を含む場合に適用。
-------------------------------------------

## 結果保存・後処理

（自動）結果ディレクトリ : results/res_{years}yr_{YYMMDD_HHMM} 形式で作成
-z                        : str     保存ディレクトリ名に任意タグ付与
--post_plot_multi         : flag    スイープ後に外部プロッタを自動起動（既定ON）
--no-post-plot-multi      : flag    上記の無効化
--post_plot_call_multi    : str     図スクリプトのパス（既定: yy1_plot_sweep_multi_v5.py）
--post_plot_errorbars     : flag    エラーバー描画をプロッタへ指示（既定ON）
```

---

## Outputs / 生成物

* **EN:** CSV, txt (summary etc), PNG (histogram).
* **JA:** CSV, txt（試行結果の統計）、PNG（ヒストグラム）。

---

## License / ライセンス

* **EN:** Licensed under **Noncommercial No-Redistribution License 1.0 (NNRL-1.0)**. See **LICENSE** for full text.
  * Personal / noncommercial use & modification: **allowed** (for your own devices/accounts only)
  * **Redistribution:** **prohibited** without prior written permission
  * **Commercial use:** **prohibited**
  * **No warranty**
* **JA:** 本プロジェクトのライセンスは **NNRL-1.0** です（全文は `LICENSE`）。
  * 個人・非商用での利用・改変：**可**（自身の端末・アカウント内に限る）
  * **再配布：**事前の**書面許可なしでは禁止**
  * \*\*商用利用** ：**禁止**
  * **無保証**

---

## No Redistribution Policy / 再配布禁止ポリシー

* **EN:** Even though the repo is public, **forking, re-hosting, packaging, or sharing code/files is not permitted** without written permission.
* **JA:** リポジトリが公開でも、**フォーク・再ホスト・パッケージ化・コード共有は許可なく行えません**。

---

## AI Assistance / AI利用の明示

* **EN:** Developed with assistance from **GitHub Copilot** and **OpenAI ChatGPT (GPT-5 Thinking)**. The author reviewed, edited, and **takes responsibility** for all content.
* **JA:** **GitHub Copilot** と **OpenAI ChatGPT（GPT-5 Thinking）** の支援を受けて作成。**最終責任は作者**にあります。

---

## Contact & Commercial License / 連絡先・商用ライセンス

* **Email:** **[mygit@healing.sakura.ne.jp](mailto:mygit@healing.sakura.ne.jp)**
* **Issues:** **[https://github.com/wan2355/monte\_carlo\_portfolio/issues](https://github.com/wan2355/monte_carlo_portfolio/issues)**

---

## Contributing / コントリビュート

* **EN:**  Issues for bug reports or permission inquiries are welcome.
* **JA:** バグ報告・許諾相談の Issue は歓迎します。

---

## Acknowledgements / 謝辞

* **EN:** Thanks to open-source communities behind Python scientific libraries.
* **JA:** Python 科学技術系ライブラリのコミュニティに感謝します。

````
