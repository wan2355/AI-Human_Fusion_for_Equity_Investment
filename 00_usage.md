
# PROGRAM 1: yy01_sec_fetch_draft_csv_v03.py

PURPOSE
Fetch SEC companyfacts-based fundamentals (USD, 1 dollar units) and write a
draft CSV you can manually reshape into your analysis input.

ALL ARGUMENTS
Short  Long            Type        Default     Required  Description

---

-t     --tickers        list[str]   []          No        Tickers on CLI (space-separated)
-f     --ticker-file    path        None        No        Text file: one ticker per line (# ok)
-y     --years          int         8           No        Keep last N years per ticker (latest-based)
-o     --outdir         path        sec_out     No        Output directory
-u     --user-agent     str         None        No        Override SEC User-Agent header (recommended)

NOTES

* You can use either --tickers or --ticker-file (or both). If both are given,
  they are merged.
* SEC requests work best when User-Agent includes contact info.

USAGE (ALL ARGS EXAMPLE: CLI tickers)
python3 yy01_sec_fetch_draft_csv_v03.py 
-t LITE MU PLTR NVDA 
-y 8 
-o sec_out 
-u "YourProject/0.1 ([your_email@example.com](mailto:your_email@example.com))"

USAGE (ALL ARGS EXAMPLE: ticker file + extra CLI tickers)
python3 yy01_sec_fetch_draft_csv_v03.py 
-f tickers.txt 
-t NVDA 
-y 8 
-o sec_out 
-u "YourProject/0.1 ([your_email@example.com](mailto:your_email@example.com))"

# PROGRAM 2: yy02_financial_eval_04t.py

PURPOSE
Compute metrics/scores and generate charts (SW/HW/CYCLE, factor map, percentiles,
breakdowns). Adds window alignment and ActualWindowYears for fairness.

ALL ARGUMENTS
Short  Long              Type       Default         Required  Description

---

-i     --input            path       (none)          Yes       Input CSV
-w     --window-years     int        5               No        Trailing window length (years)
--window-mode      str        intersection    No        Window alignment:
intersection = common window across tickers
(min available, capped by -w)
per-ticker   = each ticker uses its latest -w years
-o     --outdir           path       res_stock_funda  No        Output directory
--color-by         str        CYCLE           No        Factor-map color axis: CYCLE | SW | HW

OUTPUT FIELD
ActualWindowYears
The number of years actually used for each ticker’s trailing-window metrics.
- window-mode=intersection: same for all tickers (common minimum)
- window-mode=per-ticker  : can differ by ticker

USAGE (ALL ARGS EXAMPLE: recommended, fair comparison)
python3 yy02_financial_eval_04t.py 
-i input_data.csv 
-w 5 
--window-mode intersection 
-o out_eval 
--color-by SW

USAGE (ALL ARGS EXAMPLE: per-ticker, max history where available)
python3 yy02_financial_eval_04t.py 
-i input_data.csv 
-w 8 
--window-mode per-ticker 
-o out_eval 
--color-by CYCLE

