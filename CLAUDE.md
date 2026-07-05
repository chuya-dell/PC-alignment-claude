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
- 入力フォルダ: `F:\GoogleDrive_local\1.実験データ_gdrive\5.生データ D\260704 sam 位置合わせ test\df`
- ファイル名規則: `条件-セット-連番`。連番0=pre、連番1=post。画像は 2048×2044, 16bit。
- **実ピラーピッチ ≈ 6.29px**(私のFFTとAntigravityの2D FFTが一致)。
- **pixel pitch ≈ 32 nm/px**(当初申告の60は約1.9倍過大)。設計200nm=6.29px と整合。
  → **顕微鏡キャリブレーションでの最終確認が未フォロー**(要ユーザー)。
- ピラーは非解像・ほぼ正弦波状(ボケMTFで高調波消失)。
  → NCCは1周期(≈6.3px)内のズレ検出に有効。**一致率(|Δ|≤5)と高域強調フィルタは今回データでは非力**。
- **傷(デザインナイフ)は暗視野で「明るい線」**として写る。位置は視野ごとに違う(位置合わせテストのため)。

## ファイル
- `batch_register_images.py` — 本体。pre/postグルーピング→傷ランドマーク検出→ECC位置合わせ→共通領域クロップ→
  傷付近&ピラー領域(自動マルチパッチ)のNCC/一致率→CSV。CLI。
- `utils/diagnose_pillar_periodicity.py` — FFT周期/ボケσ/間隔/nm-px/指標適性の診断。
- `utils/diagnose_scratch.py` — 傷検出の可視化診断(帯域・追跡軌跡・inlier率・波打ち)。
- `utils/validate_ncc_blur_sensitivity.py` — ボケ×ズレ に対するNCC/一致率感度の合成検証。

## 傷検出の設計(実データで判明した4つの事実に対応)
1. 暗視野で傷は明線 → **極性オート**(`--scratch-polarity auto|bright|dark`)。
2. 傷が波打つ → 帯域平均でなく**各行/列の極値位置を追跡**。
3. 傷十字が視野ごとに別位置 → **全フレーム探索**(既定で範囲=画像全体)+**1次元モード投票**で支配線を同定。
   信頼度=モード投票率(0..1)。ノイズ床≈2*tol/幅≈0.03。`--scratch-trace-tol`(既定20, 波打ち量に合わせる)。
4. pre/postの傷が大きくずれる視野がある → **pre/post各々の傷位置を中心に切り出して**ECC。
   ECC結果が傷位置ベース並進と大きく乖離したら**並進フォールバック**(`registration_mode=translation_fallback`)。

## 実行方法(ユーザーPC)
```
cd C:\Users\chuya\pc-alignment-claude
git pull
python batch_register_images.py --input-dir "F:\GoogleDrive_local\1.実験データ_gdrive\5.生データ D\260704 sam 位置合わせ test\df" --output-dir analysis_batch --save-patch-overlay
```
主な調整オプション: `--scratch-min-confidence`(既定0.35)、`--scratch-trace-tol`(既定20)、
`--scratch-min-contrast`(既定2.0)、`--scratch-crop-margin`(既定250)、`--patch-grid`/`--patch-size`。

出力(`analysis_batch/`): 位置合わせ済みpre/post(bit深度保持)、`registration_summary.csv`、
`pillar_patch_metrics.csv`、パッチ重畳PNG、`batch_register.log`。

### 品質チェック(必ず scratch_ncc で見る)
```
python -c "import pandas as pd; d=pd.read_csv('analysis_batch/registration_summary.csv'); ok=d[d.status=='ok']; print('ok',len(ok),'/',len(d)); print('scratch_ncc median',round(ok.scratch_ncc.median(),3),'min',round(ok.scratch_ncc.min(),3)); print(ok[ok.scratch_ncc<0.85][['condition','set','scratch_confidence_pre','scratch_ncc','registration_mode','dx_px','dy_px']].to_string())"
```
**成功=statusがokだけでは不十分。scratch_ncc≥0.9 で本当に合っている**と判断する。

## 進捗と経緯(検出成功率の推移)
0/62 →(極性オート)→ 1/64 →(線追跡+並進フォールバック)→ 9/64 →(全フレーム探索)→ 47/64
→(閾値緩め tol35/min-conf0.2)→ 64/64検出だが一部 scratch_ncc 低(~14視野)
→(**pre/post各自の傷中心で切り出す修正: commit 098de57**)← 最新。大オフセット視野の誤位置合わせを解消するはず。

## 次アクション(NEXT)
1. **[ユーザー] 最新版で再実行し scratch_ncc を確認**(上の品質チェックコマンド)。
   commit 098de57(各自の傷中心クロップ)で、前回 scratch_ncc が低かった視野(例 2-1=-0.37, 3-1=0.45,
   7-6=0.31 など)が改善しているはず。改善を確認したら:
   - まだ低品質な視野が残る → その視野を `diagnose_scratch.py` で可視化し原因特定(傷が視野外/極端に短い等)。
   - 低品質が消えた → 既定パラメータ(min-confidence等)を確定し、運用値としてこのファイルに記録。
2. **[ユーザー] pixel pitch ≈ 32 nm/px を顕微鏡キャリブレーションで確認。**
3. **[ユーザー] -anti の残ブランチ `claude/batch-image-registration-qe4cp7` を削除**
   (私はegressポリシー403で削除不可。GitHub UIのBranchesから)。
4. **[保留・未着手] 最終ゴール**: 差分画像生成 → Grid_Heatmapマクロへの受け渡し形式。
   元命令で「解析手順を再確認してから指示」とされ保留中。バッチ位置合わせ確定後に着手。

## 開発時の注意
- サンドボックスでは実TIFFを読めないので、`batch_register_images.py`/検出ロジックの検証は
  **合成データ**(明/暗傷、波打ち、十字位置バリエーション、大オフセット)で行う。過去の検証は全て合成で実施。
- 変更後は `python -m pyflakes batch_register_images.py` と合成データでのスモークを通してから push。
- コミットメッセージ末尾に付与:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` は不要(通常のコミットでよい)。
