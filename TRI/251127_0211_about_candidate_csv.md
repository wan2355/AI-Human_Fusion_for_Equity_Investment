承知いたしました。
ではご指示どおり **A → B** の順でまとめます。

---

## A. candidate_tri5.csv（TRI5 用・売買候補リスト）

### 1. 方針

* **TRI5 で評価するのは「売買して増やしたい銘柄」だけ**。
* 配当メインの 9286.T / JEPI / JEPQ は **ここには入れない**（Bで別管理）。
* 今回は、最近よく話題に出ている銘柄＋スクショに写っている銘柄を中心に
  「まずのたたき台」として数値を置いています。
  （あとでご自身の感覚に合わせて上書きしていただく前提です）

### 2. candidate_tri5.csv サンプル

```csv
ticker,market,prep_time_min,period_months,loss_amount,win_prob,rr_ratio,vol_std,comment
1329.T,jp,30,1,25000,0.35,1.8,20000,"日経225ETF 短期〜1ヶ月スイング"
1540.T,jp,40,4,30000,0.35,1.8,25000,"純金ETF 中期リスクヘッジ＋値幅取り"
2914.T,jp,45,6,30000,0.40,2.0,15000,"JT 高配当＋半年スイング候補"
4063.T,jp,55,4,35000,0.35,2.0,28000,"信越化学 成長株・中期スイング"
5713.T,jp,60,3,40000,0.35,2.0,30000,"住友金属鉱山 資源株・ボラ高め"
6339.T,jp,60,3,30000,0.35,2.0,32000,"新東工業 攻め枠・値動き大きい"
8058.T,jp,40,6,25000,0.40,2.0,18000,"三菱商事 商社株・半年スイング"
BE,us,40,3,25000,0.30,2.0,35000,"BE クリーンエネ・中期トレード"
GLDM,us,40,3,30000,0.33,1.8,25000,"GLDM ゴールドETF・為替込み"
IREN,us,70,1,30000,0.30,2.5,70000,"IREN BTC連動・高ボラ短期"
QQQI,us,35,3,25000,0.33,1.8,30000,"QQQI NASDAQカバコ ETF"
LITE,us,60,1,30000,0.35,2.0,60000,"LITE AI光学関連・短期スイング"
MSFT,us,50,3,35000,0.33,2.0,40000,"MSFT コアAI・中期スイング"
```

#### 各列のイメージ

* prep_time_min
  その銘柄に「1トレードあたり」割く分析＋場中時間の平均イメージ
* period_months
  その銘柄を **典型的にどれくらいの期間持つ戦略か**
* loss_amount
  1トレードで「これくらいまでは許容する」損失額（円）
* win_prob
  その戦略のざっくり勝率

  * 日本株：0.40
  * 米国株：0.30〜0.35（やや低め）
* rr_ratio
  利益：損失 の比率。デフォルト 2.0（2:1）
* vol_std
  価格 or 損益の「ざっくり標準偏差イメージ」
  → ボラが高い銘柄ほど大きく

すべて**「まずはこのくらいで置いて、回しながら微調整」**で大丈夫です。

---

### 3. TRI5 一定値以上を自動フラグするためのコード修正

ご要望の

> 「TRI5 が一定値を超えた銘柄だけ、自動でフラグ」

を入れるために、`eval_tri5.py` に **閾値オプション＋フラグ列** を追加します。

#### 3-1. 追加する CLI オプション

```bash
python eval_tri5.py \
  --input candidate_tri5.csv \
  --default-win-prob 0.35 \
  --jp-adjust 0.05 \
  --rr-ratio 2.0 \
  --tri5-threshold 500 \
  -z first_try
```

ここで `--tri5-threshold 500` が
「TRI5 が 500 以上ならフラグ」というイメージです。

#### 3-2. コード修正（差分）

**(1) 引数パーサ部に閾値を追加**

```python
# 修正前
parser.add_argument(
    "--config-only",
    action="store_true",
    help="計算せずに、config.json のみ試し生成するテスト用フラグ",
)

# 修正後
parser.add_argument(
    "--config-only",
    action="store_true",
    help="計算せずに、config.json のみ試し生成するテスト用フラグ",
)
parser.add_argument(
    "--tri5-threshold",
    type=float,
    default=0.0,
    help="TRI5 がこの値以上の銘柄に flag_pick を立てる (デフォルト: 0 = 全銘柄)",
)
```

**(2) config.json にも記録**

```python
config_data: Dict[str, Any] = {
    ...
    "rr_ratio_global": args.rr_ratio,
    "tri5_threshold": args.tri5_threshold,  # ← 追加
    ...
}
```

**(3) 出力行に flag_pick 列を追加**

```python
# 修正前（TRI5_score の列を入れているあたり）
if metrics["TRI5_score"] is not None:
    out_row["TRI5_score"] = f"{metrics['TRI5_score']:.4f}"
else:
    out_row["TRI5_score"] = ""

rows_out.append(out_row)
```

```python
# 修正後
if metrics["TRI5_score"] is not None:
    tri5_val = metrics["TRI5_score"]
    out_row["TRI5_score"] = f"{tri5_val:.4f}"
    # 閾値を超えたらフラグ
    if args.tri5_threshold > 0 and tri5_val >= args.tri5_threshold:
        out_row["flag_pick"] = "PICK"
    else:
        out_row["flag_pick"] = ""
else:
    out_row["TRI5_score"] = ""
    out_row["flag_pick"] = ""

rows_out.append(out_row)
```

これで、`result_*.csv` を Excel で開けば

* flag_pick = "PICK" だけフィルタすれば
  「手間をかける価値がある銘柄だけ」が一発で抽出できる、という流れになります。

---

## B. income_stable_holdings.csv（配当・安定枠）

こちらは TRI5 ではなく、**配当と安全性を中心に見る「第2層」用の表**です。

### 1. 方針

* 「基本放置・リバランス時だけ触る」ものをまとめる。
* TRI のような時間効率ではなく

  * 配当利回り
  * 分散効果
  * 為替リスク
    を軸に管理。

### 2. income_stable_holdings.csv サンプル

```csv
ticker,market,units,avg_price_jpy,div_yield_now,target_yield,fx_risk,role,comment
9286.T,jp,7,85900,0.07,0.07,low,L2_core,"インフラファンド・2030頃まで基本ホールド"
JEPI,us,32,9000,0.08,0.07,med,L2_core,"毎月分配・為替リスク中・下落時買い増し候補"
JEPQ,us,53,9200,0.09,0.08,med_high,L2_core_plus,"配当高め・価格変動やや大きい"
GLDM,us,126,13000,0.00,0.00,med,L1_hedge,"金ETF・為替込みヘッジ。配当はゼロ"
QQQI,us,1,8000,0.08,0.07,med,L2_satellite,"小口カバコETF・実験枠"
```

※ avg_price_jpy はざっくりで構いません（実値に直すのは後からでもOKです）

#### 各列の意味

* units
  保有株数
* div_yield_now
  現在利回り（ざっくりでOK）
* target_yield
  「このくらいまで落ちてきたら買いたい」目安利回り
* fx_risk
  low / med / med_high などざっくり
* role
  L2_core, L2_core_plus, L1_hedge など「ポートフォリオの役割」
* comment
  自由メモ

TRI 系とは完全に別レイヤーとして、
「ここは触らなくていい（or 年数回だけ見る）」と決めてしまう設計です。

---

## さいごに

* A で **「今後、銘柄研究する価値がある “攻め枠” だけを TRI5 でふるいにかける仕組み** ができました。
* B で **「配当・安定用の“触らない枠”」を分離** したことで、
  **“考えるべき対象” と “放置でよい資産” が明確に分かれてきた** と思います。

次の一手としては：

1. 実際に `candidate_tri5.csv` を作って `eval_tri5.py` を回してみる
2. `result_*.csv` を見ながら

   * TRI5 閾値（例: 400 / 600 / 800）をどこに置くか
   * どの銘柄を「本当に研究する候補」に残すか
     を一緒に調整する

という流れがよろしいかと存じます。

もしよろしければ次は、

* 実際に「JT / 8058 / 5713 / LITE あたりの TRI5 値がどう並ぶか」を
  こちらでざっくり計算して、**ランキング表形式** でお見せすることも可能です。
;
