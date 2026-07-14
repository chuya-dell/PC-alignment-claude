# PC-alignment-claude — 引き継ぎ / プロジェクト状況

暗視野DIC散乱像(SAM形成 pre/post)のバッチ位置合わせと精度評価のためのツール群。
新しいセッションはこのファイルを読めば現状・実行方法・次アクションが分かる。

- リポジトリ: `chuya-dell/PC-alignment-claude`
- 作業ブランチ: `claude/batch-image-registration-qe4cp7`(**mainではなくこのブランチで作業**)
- 依存: `pip install -r requirements.txt`(numpy, opencv-python, scipy, pandas, matplotlib, tifffile)
- 元となった単発版: `register_images.py`(このリポジトリのmainにある初期版)

## リポジトリ運用の約束
- **Claudeの成果物はこの -claude リポジトリに置く。** 開発・pushはすべてここ。
- **PC-alignment-anti は触らない**(Antigravityが別途開発中)。
- 実データTIFFは各8.3MB×多数でサンドボックスに取り込めない(Driveダウンロードはbase64インライン返却でコンテキスト超過)。
  **実データを読む処理はユーザーのPCでのみ実行**。開発・合成データ検証はサンドボックスで完結する。

## 実データの物理パラメータ(診断で確定済み)
- 入力フォルダ: `F:\GoogleDrive_local\1.実験データ_gdrive\4.生データ\4.生データ D\260704 sam 位置合わせ test\df`
- ファイル名規則: `条件-セット-連番`。連番0=pre、連番1=post。画像は 2048×2044, 16bit。
- **実ピラーピッチ ≈ 6.29px**(私のFFTとAntigravityの2D FFTが一致)。
- **pixel pitch ≈ 32 nm/px**(当初申告の60は約1.9倍過大)。設計200nm=6.29px と整合。
  → **顕微鏡キャリブレーションでの最終確認が未フォロー**(要ユーザー)。
- ピラーは非解像・ほぼ正弦波状(ボケMTFで高調波消失)。
  → NCCは1周期(≈6.3px)内のズレ検出に有効。**一致率(|Δ|≤5)と高域強調フィルタは今回データでは非力**。
- **傷(デザインナイフ)は暗視野で「明るい線」**として写る。位置は視野ごとに違う(位置合わせテストのため)。

## ファイル
- `batch_register_images.py` — 本体。pre/postグルーピング→傷ランドマーク検出→位置合わせ(ECC or 傷十字ベース)→
  共通領域クロップ→傷残差/細帯NCC&ピラー領域(自動マルチパッチ)の指標→CSV。CLI。
- `utils/diagnose_pillar_periodicity.py` — FFT周期/ボケσ/間隔/nm-px/指標適性の診断。
- `utils/diagnose_scratch.py` — 傷検出の可視化診断。**本体 detect_scratch_landmark を直接呼ぶ**ので
  本体と定義上ズレない。`--image2` で pre/post を比較し傷位置差(=dxy_coarse)を出力。
- `utils/validate_ncc_blur_sensitivity.py` — ボケ×ズレ に対するNCC/一致率感度の合成検証。

## 傷検出の設計(実データで判明した4つの事実に対応)
1. 暗視野で傷は明線 → **極性オート**(`--scratch-polarity auto|bright|dark`)。
2. 傷が波打つ → 帯域平均でなく**各行/列の極値位置を追跡**。
3. 傷十字が視野ごとに別位置 → **全フレーム探索**(既定で範囲=画像全体)+**1次元モード投票**で支配線を同定。
   信頼度=モード投票率(0..1)。ノイズ床≈2*tol/幅≈0.03。`--scratch-trace-tol`(既定20, 波打ち量に合わせる)。
4. pre/postの傷が大きくずれる視野がある → **pre/post各々の傷位置を中心に切り出して**ECC。

## 位置合わせと品質指標の設計(2つの弱点を修正済み)
実データ64ペアで判明した弱点と対策:
- **弱点A: 位置合わせの暴走** — ECCが周期ピラー(≈6.3px正弦波)の別解に収束し dx/dy が飛ぶ
  (旧: 4-4 dy=-215px 等)。旧フォールバックのガード閾値が250pxと緩すぎて捕まらなかった。
- **弱点B: 指標の交絡** — 旧 `scratch_ncc` は傷±250pxの500px箱NCC。中身の大半が周期ピラーで、
  数pxの残差で位相反転し負にすらなる。**合っていても低スコア**になり判定に使えなかった。

### 対策(マジックナンバー撤廃・自己検証式)
- **傷残差ベースの自己検証**: 傷十字ベース並進(post傷−pre傷)候補と ECC 候補の**両方を実際に適用**し、
  位置合わせ後に傷を再検出して pre傷との距離 `scratch_residual_px` を実測。ECCが残差許容
  (`--scratch-residual-tol` 既定2px)内なら回転精密化の効くECCを採用、超えたら傷十字ベース並進へ
  **自動フォールバック**(`registration_mode` = `ecc` / `scratch_translation`)。250px閾値は撤廃。
  ※残差再検出は pre傷±window で極性をpre固定(warpの0埋め縁を誤検出しないため)。
- **主品質指標 = `scratch_residual_px`(≤1〜2pxで合格)** と **`scratch_line_ncc`(傷線±8pxの細帯NCC、
  周期ピラーを含めない)**。旧 `scratch_ncc`(500px箱)は参考値として残すが交絡のため主判定から外す。
- **傷線の傾き角**を各行/列のインライアから推定(`angle_v_rad`/`angle_h_rad`)。pre/postの角度差から
  回転を推定し、ECC非収束時の傷十字ベース並進にも回転を付与(縦横の角度が一致し小さい時のみ採用)。
- **診断列**: `dxy_coarse_x/y`, `scratch_residual_px`, `residual_ecc_px`, `residual_scratch_px` を出力。
  ECCがどれだけ飛んだか(residual_ecc)を視野ごとに確認できる。

## 実行方法(ユーザーPC)
```
cd C:\Users\chuya\pc-alignment-claude
git pull
python batch_register_images.py --input-dir "F:\GoogleDrive_local\1.実験データ_gdrive\4.生データ\4.生データ D\260704 sam 位置合わせ test\df" --output-dir analysis_batch --save-patch-overlay
```
主な調整オプション: `--scratch-min-confidence`(既定0.35)、`--scratch-trace-tol`(既定20)、
`--scratch-min-contrast`(既定2.0)、`--scratch-crop-margin`(既定250)、
`--scratch-residual-tol`(既定2.0, ECC採否の残差許容px)、`--patch-grid`/`--patch-size`。

出力(`analysis_batch/`): 位置合わせ済みpre/post(bit深度保持)、`registration_summary.csv`、
`pillar_patch_metrics.csv`、パッチ重畳PNG、`batch_register.log`。

### 品質チェック(scratch_residual_px で見る ← scratch_ncc ではない)
```
python -c "import pandas as pd; d=pd.read_csv('analysis_batch/registration_summary.csv'); ok=d[d.status=='ok']; print('ok',len(ok),'/',len(d)); print('scratch_residual_px  median',round(ok.scratch_residual_px.median(),2),'max',round(ok.scratch_residual_px.max(),2)); print('mode:',ok.registration_mode.value_counts().to_dict()); bad=ok[ok.scratch_residual_px>2]; print('残差>2pxの視野:',len(bad)); print(bad[['condition','set','registration_mode','scratch_residual_px','residual_ecc_px','dx_px','dy_px','dxy_coarse_x','dxy_coarse_y','scratch_confidence_pre']].to_string())"
```
**成功=statusがok『かつ scratch_residual_px ≤ 1〜2px』で本当に合っている**と判断する。
`scratch_ncc`(500px箱)は周期ピラーで交絡するので主指標にしない(参考値)。

## 進捗と経緯(検出成功率の推移)
0/62 →(極性オート)→ 1/64 →(線追跡+並進フォールバック)→ 9/64 →(全フレーム探索)→ 47/64
→(閾値緩め tol35/min-conf0.2)→ 64/64検出だが一部 scratch_ncc 低(~14視野)
→(pre/post各自の傷中心で切り出す修正: commit 098de57)→ **実データ再実行で判明**: scratch_ncc median 0.649,
  min -0.29。位置合わせ暴走(4-4 dy=-215等)とscratch_ncc指標の交絡が原因と判明。
→(残差ベース自己検証+信頼できる指標に置換: commit dc7aa64)→ **実データ再実行で判明**:
  scratch_residual_px median 21px, max 277px。高信頼度視野でも残差100px超が多発、
  dxy_coarseがほぼ0の視野ですら残差244px等、物理的ズレでは説明不能な挙動。
  原因は _scratch_residual が縦傷の極性を横傷検出にも流用するバグ(実データは縦横で
  極性が異なる視野が多い)→ 誤極性で無関係な特徴を掴み見かけ上の残差が跳ね上がっていた。
→(polarity_v/polarity_h 分離で修正: commit 2a1f2c4)→ **実データ再実行で判明**:
  scratch_residual_px median 21px→6.4pxに改善したが、4-1(confidence=0.925という
  高信頼度)が residual=140.000000px のまま(修正前後で完全に同一)残った。
  diagnose_scratch.py --image2 の可視化(ユーザーPC)で調査 → 傷十字は画像左上隅付近
  (x≈110,y≈148)にあり、pre→postの並進dy=-20.5は上方向。_scratch_residual内の
  apply_warpがBORDER_CONSTANT(0埋め)だったため、位置合わせ後の画像上端約20行が
  無効領域として真っ黒に埋まり、探索窓内でこの人工的な黒縁を傷の影より暗い候補として
  誤検出(黒縁までの距離≈140px=実測残差と一致)。
→(_scratch_residual内のwarpをBORDER_REPLICATEに変更して修正: commit 390660e)
  → **実データ再実行で判明**: scratch_residual_px median 6.4px→5.0px、max 153px→26.48px
  に改善(4-1は解消)。最悪ケース3-8を diagnose_scratch.py --image2 で調査 → 横傷が
  pre='dark'(conf0.45)/post='bright'(conf0.37)と、信号が弱い(コントラストmin閾値
  ギリギリ)溝で pre/postが独立にauto極性判定した結果、溝の暗い側/明るい側という
  別々の物理特徴を追跡してしまい系統的なズレが発生。
→(postの検出にpreの極性を強制して同一特徴追跡に統一: commit 2cb9076)→ **実データ再実行で判明**:
  scratch_residual_px median 6.4px→5.0px、max 153px→26.48px に改善。残る36視野の
  残差値が 2.236(=√5), 3.605(=√13), 5.830(=√34) 等**整数の平方和の平方根**になって
  おり、位置検出が整数ピクセル単位に量子化されていることが判明(confidence高い視野
  3-1/3-2/3-3でも同程度の残差)。放物線補間によるサブピクセル推定 `_subpixel_min` が
  定義済みなのに `_trace_line` に未使用と判明(元の単発版由来、移植漏れ)。
→(_trace_line にサブピクセル補間を追加: commit a2de478)→ **実データ再実行で判明**:
  合成データでは全テストの残差が1px未満に改善したが、**実データでは効果なし**
  (median 3.61px→3.89pxとわずかに悪化、max 22.47px→22.59px)。原因を再考: 「残差が
  整数の平方和の平方根」という前回の観察は量子化の証拠として誤読していた可能性が高い
  (2点間ユークリッド距離はdx/dyが整数に近いだけで sqrt(整数²+整数²) になり得る)。
  実データの傷信号は粒状ノイズを伴う弱いコントラストで、3点放物線補間は滑らかな合成
  データには効くが実ノイズには効かない(むしろ増幅し得る)ミスマッチと判断。
  サブピクセル化は害がほぼないため維持。
→(最悪ケース2-3を診断: 一旦「新バグなし」と結論しかけたが、ユーザーから**傷の縦横の腕は
  常に直交(90度)しているとは限らない**との指摘。2-3のv傾き+2.62度を実画像高さ(2044px)に
  当てはめると上端から下端まで2044*tan(2.62)≈94pxも列位置が動く計算になるが、モード投票の
  窓幅は2*tol=40pxしかなく、傾いた線の一部だけを拾ってpre/postで異なる部分が選ばれ系統的な
  ズレが生じていた(40/94≈0.43の理論値が実測confidence0.37と整合)。「ノイズフロア」ではなく
  傾き非対応のモード投票アルゴリズムの限界だった。
→(**`_trace_line`のモード投票を2段階(暫定傾き推定→デトレンド→再クラスタ)化して修正:
  本コミット cbd846b**)← 最新。実データ2-3を模した合成シーン(縦傷2.62度傾き、画像高さ
  2044px)で、修正前confidence=0.47(実データ0.37と近似)→修正後confidence=1.00・
  角度推定2.63度(真値2.62度)に改善することを検証。既存の全回帰テストPASS。
  **実データでの再検証がまだ済んでいない**(要ユーザー再実行)。

## 到達点(2026-07-14時点、更新中)
5段階の実データ検証サイクルで scratch_residual_px median を 21.21→6.4→5.0→3.61→3.89px
まで改善し、4つの実バグ(極性軸混同/境界黒縁誤検出/pre-post極性不一致/傾き非対応モード投票)
を発見・修正・合成テストで再発防止済み。**4つ目(傾き非対応)はまだ実データ未検証**
(commit cbd846b、次セッションでの再実行待ち)。「ノイズフロアに到達した」という前回の
結論は時期尚早だった(2-3の低confidenceは信号ノイズでなく傾き非対応アルゴリズムが原因と
判明)ため、撤回する。バグ探索フェーズは継続中。

## 次アクション(NEXT)
1. **[ユーザー] 最新版(commit cbd846b)で再実行し scratch_residual_px を確認。**
   狙い: 傾いた縦横の傷を持つ視野(2-3等、confidence0.35〜0.6帯に多かった)でconfidenceが
   上がり残差が下がっているはず。上の品質チェックコマンドで確認し、残差>2pxが残る視野が
   あれば `diagnose_scratch.py --image2` でさらに可視化診断する。
2. **[ユーザー] pixel pitch ≈ 32 nm/px を顕微鏡キャリブレーションで確認。**
3. **[ユーザー] -anti の残ブランチ `claude/batch-image-registration-qe4cp7` を削除**
   (私はegressポリシー403で削除不可。GitHub UIのBranchesから)。
4. **[次の主要ゴール] 差分画像生成 → Grid_Heatmapマクロへの受け渡し形式。**
   バッチ位置合わせがノイズフロアに到達し運用値も定まったため、着手可能な段階に入った。
   ユーザーから具体的な指示があり次第、差分画像フォーマット・Grid_Heatmapとの受け渡し
   仕様を確認して設計する。

## 開発時の注意
- サンドボックスでは実TIFFを読めないので、`batch_register_images.py`/検出ロジックの検証は
  **合成データ**(明/暗傷、波打ち、十字位置バリエーション、大オフセット)で行う。過去の検証は全て合成で実施。
- 変更後は `python -m pyflakes batch_register_images.py` と合成データでのスモークを通してから push。
- コミットメッセージ末尾に付与:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` は不要(通常のコミットでよい)。
