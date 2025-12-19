# Glossary (Full)

| 日本語/Key | English | 説明 |
|---|---|---|
| 時価総額（MC） | Market Cap (MC) | 企業全体の値段。MC=株価×発行株式数（本ツールではMktCap_Bを使用） |
| MC/Rev | MC/Rev (Market Cap to Revenue) | MC/Rev=MC/Revenue。売上1ドルをいくらで買っているか（alias: PS, PS_Latest） |
| PFCF | MC/FCF (Price to Free Cash Flow) | PFCF=MC/FCF。FCFが0以下なら倍率の意味が崩れるので空欄 |
| POwnerFCF | MC/OwnerFCF (Price to Owner FCF) | POwnerFCF=MC/(FCF-SBC)。株主に残る現金で評価。<=0は空欄 |
| Ticker | Ticker | 銘柄コード |
| Company | Company | 会社名 |
| Year | Year | 年度（FY） |
| MaxYear | MaxYear | 入力データの最新年 |
| WindowYears | WindowYears | 1サイクル=この年数。平均/最悪/成長をこの窓で計算 |
| ActualWindowYears | ActualWindowYears | 実際に計算に使われた年数（intersectionなら全銘柄同一、per-tickerなら銘柄ごとに異なる） |
| WindowStartYear | WindowStartYear | 窓の開始年（MaxYear-(WindowYears-1)） |
| Revenue | Revenue | 売上（Billion USD） |
| EBIT | EBIT | 利息・税引前利益（本業利益に近い） |
| FCF | FCF (Free Cash Flow) | FCF=営業CF−設備投資。自由に使える現金の増減 |
| CapEx | CapEx (Capital Expenditures) | 設備投資（現金支出） |
| SBC | SBC (Stock-Based Compensation) | 株式報酬（株主の取り分が薄まるコスト） |
| OwnerFCF | Owner FCF | OwnerFCF=FCF−SBC（株主に実質残る現金） |
| Revenue_Avg | Revenue_Avg | WindowYears内の売上平均 |
| EBIT_Avg | EBIT_Avg | WindowYears内のEBIT平均 |
| FCF_Avg | FCF_Avg | WindowYears内のFCF平均 |
| EBIT/Rev_Avg | EBIT/Rev_Avg | EBIT/Rev（平均）。本ツール内部列名 alias: EBITMargin_Avg |
| EBIT/Rev_Min | EBIT/Rev_Min | EBIT/Rev（最悪）。不況耐性の目安。alias: EBITMargin_Min |
| FCF/Rev_Avg | FCF/Rev_Avg | FCF/Rev（平均）。alias: FCFMargin_Avg |
| FCF/Rev_Min | FCF/Rev_Min | FCF/Rev（最悪）。不況耐性の目安。alias: FCFMargin_Min |
| CapEx/Rev_Avg | CapEx/Rev_Avg | CapEx/Rev（平均）。装備コストの重さの目安。alias: CapExOverRev_Avg |
| SBC/Rev_Avg | SBC/Rev_Avg | SBC/Rev（平均）。株式報酬の重さの目安。alias: SBCOverRev_Avg |
| Rev_CAGR | Rev_CAGR | 売上CAGR（start/end>0のみ） |
| FCF_CAGR | FCF_CAGR | FCF CAGR（start/end>0のみ） |
| Price_Latest | Price_Latest | 最新年の株価（任意入力） |
| MktCap_B_Latest | MktCap_B_Latest | 最新年の時価総額（Billion）。TradingViewのMarket Capを推奨 |
| PS_Latest | PS_Latest | 最新年: MC/Rev（出力表記はMC/Rev。alias吸収） |
| PFCF_Latest | PFCF_Latest | 最新年: MC/FCF（出力表記はMC/FCF。FCF<=0は空欄） |
| POwnerFCF_Latest | POwnerFCF_Latest | 最新年: MC/OwnerFCF=MC/(FCF-SBC)（出力表記はMC/OwnerFCF。<=0は空欄） |
| Profile | Profile | 評価軸（CYCLE / SW / HW） |
| CYCLE | CYCLE profile | 景気循環（波）の中で戦いやすい体質かを見る軸。耐性（EBIT/Rev_Min）や現金化（FCF/Rev）などを総合して相対評価する（投資判断そのものではない） |
| SW | SW profile | ソフト寄り（軽装備・人/ソフトでスケールしやすい）体質を見る軸。CapEx/Rev（低いほどSW寄り）、SBC/Rev（高いほどSW寄り）と成長（Rev_CAGR）を中心に相対評価する。※良し悪しではなく体質の可視化 |
| HW | HW profile | ハード寄り（重装備・資本集約）体質を見る軸。CapEx/Rev（高いほどHW寄り）を中心に、不況時の耐性（EBIT/Rev_Min, FCF/Rev_Min）も合わせて相対評価する。※良し悪しではなく体質の可視化 |
| Valuation(Expensive) | Valuation (Expensive) | 割高さの軸（0-100）。本ツールでは『PS(MC/Rev)』『PFCF(MC/FCF)』『PEG_PS』などを0-100に正規化して合成（高い=割高寄り）。値は“優劣”ではなく、価格が織り込んでいる期待の強さを表す |
| Quality | Quality | 稼ぐ力+耐性の軸（0-100）。本ツールでは『EBIT/Rev_Avg』『FCF/Rev_Avg』『EBIT/Rev_Min』『FCF/Rev_Min』を0-100に正規化して合成（高い=収益性/耐性が強い） |
| Factor Map | Factor Map | 散布図: x=Valuation(Expensive), y=Quality。色=第3軸（デフォルトはCYCLE）、点サイズ=時価総額。『割高だが強い/安いが弱い/体質違い』を1枚で俯瞰する |
| Score_0to100 | Score_0to100 | 総合点（0-100）。出力では整数 |
| Rating | Rating | A〜E評価（Scoreを丸め） |
| S_xxx | S_xxx | 各指標の順位スコア（1-5） |

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
