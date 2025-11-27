


3. *_effective / vol_std / stability_factor の意味
(1) *_effective 系

すべて「最終的に計算に使われた値」を示す列です。

win_prob_effective
実際に使った勝率。

CSV に win_prob が書いてあればそれを優先

なければ
default_win_prob ＋（market が jp のとき jp_adjust）を使い、0〜0.99 にクランプ

loss_prob_effective
1 - win_prob_effective の値。
敗北確率です。

rr_ratio_effective
利益:損失 の比率。

CSV に rr_ratio があればそれを優先

なければグローバル引数 --rr-ratio を使用

win_amount_effective
1トレードあたりの 期待利益額（勝ち側の幅）。

CSV に win_amount があればそれを優先

なければ loss_amount × rr_ratio_effective

loss_amount_effective
1トレードあたりの 想定損失額。

CSV の loss_amount をそのまま使用
（マイナスではなく「絶対値」を入れておく前提）

(2) vol_std >> risk_span

~「volatility standard deviation（ボラティリティの標準偏差）」の略で、その戦略の損益や価格変動のブレの大きさ を表すために使う入力列です。~

1トレードあたりの典型的な損益変動幅（概算）
実際には標準偏差の厳密計算というより「体感的ボラ尺度（リスク許容範囲）に近い」


厳密な統計値でなくて構いません。
たとえば

JT のように穏やかな銘柄 → 15000

LITE や IREN のような高ボラ → 60000〜70000

といった “ざっくりのスケール感” を入れておく前提です。

入っていない場合は、安定性補正を行わず TRI5 = TRI3 になります。


sample 

| 銘柄   | win_amount | risk_span | stability_factor | コメント      |
| ---- | ---------- | --------- | ---------------- | --------- |
| JT   | +100,000   | 15,000    | 6.67             | 安定的で優秀    |
| LITE | +60,000    | 60,000    | 1.00             | 利益幅 ≒ 変動幅 |
| IREN | +75,000    | 70,000    | 1.07             | ほぼルーレット型  |
| MSFT | +80,000    | 40,000    | 2.00             | 比較的安定     |


(3) stability_factor

安定度（ボラに対する利益幅）を表す係数で、
計算式は

stability_factor = win_amount_effective / risk_span

イメージとしては

「ブレ 1円あたり、どれくらい利益を狙っているか」

という 簡易シャープレシオ のようなものです。

TRI5 の計算では

TRI5 = TRI3 × stability_factor


として、この係数で TRI3 を重み付けしています。

同じ TRI3 でも

安定的に取れる銘柄（vol_std 小） → stability_factor 大 → TRI5 大

荒い銘柄（vol_std 大） → stability_factor 小 → TRI5 小

となるため、

「時間効率は良いがメンタル負荷が高い銘柄」を
自動的にスコアダウンさせる効果があります。
