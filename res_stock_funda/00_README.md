# financial_eval.py (COMPLETE v3)

    ## 何がわかるか（重要）
    - **summary_A4.csv** を見れば、主要指標だけで比較できます（A4相当）
    - **scorecards_pretty.csv** で、CYCLE/SW/HWのスコア（0-100）を比較できます
    - **charts/** で、重要指標の横棒グラフ比較ができます

    ## 入力CSV
    必須: Ticker, Company, Year, Revenue, EBIT, FCF
    任意: CapEx, SBC, Price, MktCap_B, SharesOut_B

    単位:
    - Revenue/EBIT/FCF/CapEx/SBC/MktCap_B: Billion USD（B）
    - Price: USD/株
    - SharesOut_B: Billion shares（B株）

    ## P（Price）とは何か
    本ツールの **P は株価ではなく時価総額（Market Cap）** です。
    - MC = Market Cap = Price * SharesOut
    （TradingViewのMarket Capを MktCap_B に入れる運用が最も簡単です）

    ## 定義（Latest=最新年）
    - PS_Latest        = MC / Rev.（売上）
    - PFCF_Latest      = MC / FCF（FCF<=0は空欄）
    - POwnerFCF_Latest = MC / (FCF - SBC)（<=0は空欄）

    ## 実行
    ```bash
    python3 financial_eval.py --input input.csv --window-years 5 --outdir out
    ```

    出力:
    - out/metrics_pretty.csv
    - out/scorecards_pretty.csv
    - out/summary_A4.csv
    - out/charts/*.png
    - out/glossary_full.md
    

<!--AUTO:DEFS_BEGIN-->

## Definitions (auto-synced)

### Score axes (0-100)
- Score_SW: structure tilt toward software/human intensity (not 'good/bad').
  - Higher means: lower CapEx/Rev (lighter), higher SBC/Rev (more human intensity), higher Rev CAGR (growth tilt).
- Score_HW: structure tilt toward hardware/capital intensity (not 'good/bad').
  - Higher means: higher CapEx/Rev (heavier) plus resilience checks (EBIT/Rev_Min, FCF/Rev_Min).
- Score_CYCLE: cyclicality / stability score (higher = more stable / less cyclical).

### Factor Map
- X = Valuation (Expensive) 0-100: higher means 'more expensive' (NOT better).
  - Components: PS_Latest (MC/Rev), MC/FCF, MC/OwnerFCF, PEG-like.
- Y = Business Strength 0-100: higher means stronger profitability/resilience.
  - Components: EBIT/Rev & FCF/Rev (Avg & Min).
- Bubble size ~ Market Cap (B USD).
- Color axis selectable: CYCLE / SW / HW.

### Raw percentile bars (per ticker)
- Each bar is a cross-sectional percentile rank among tickers in this run (0-100).
- It is NOT a 'contribution'. Contribution is shown in the score breakdown charts.
- Fixed group order: Valuation → Profitability → Efficiency → Growth → Scale.

<!--AUTO:DEFS_END-->
