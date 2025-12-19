# Equity-fundamental-playground-yumizen

株式ファンダメンタル分析を  
**数式ベース・構造理解重視**で行うための実験用リポジトリです。

本プロジェクトは、
銘柄の優劣や売買判断を自動化するものではありません。

---

## What this project does

- Collects and normalizes fundamental metrics
- Visualizes ratios such as:
  - FCF/Rev
  - CapEx/Rev
  - MC/Rev
  - MC/FCF
- Generates summary tables and scores based on:
  - CYCLE
  - SW (Software-like)
  - HW (Hardware-like)

---

## What this project does NOT do

- It does NOT provide buy/sell signals
- It does NOT predict stock prices
- It does NOT recommend specific investments

All outputs are for **educational and analytical purposes only**.


---

## Example Interpretation（考え方の例）

ある企業が次のような特徴を持っているとする。：

- SW が高い
- CYCLE は中程度
- MC/Rev・MC/FCF が非常に高い
- PEG-like が同業より突出している

単純に見ると「割高」に見えるかもしれない。

しかし本来の解釈は：
「市場が将来の選択肢価値を先に評価しており、
現在の成長やCFがまだ追いついていない状態」

これは、
- 投資不適格
- 評価が間違っている

という意味ではない。

**時間軸のズレ**をどう扱うか、という問題である。

---

## How to read the data

See:  
`dataの読み方.md`

---

## License

MIT License

This project is intended for educational and analytical use.

