# クラウド側 一方向バックアップ（GAS版・PC電源に依存しない）

`scripts/sync_digitalcount.bat` + `run_hidden.vbs` はremotefdtd PC上で動く
ローカル常駐プロセスだが、PCの電源が落ちている間は同期が進まない。

このGoogle Apps Script (GAS) は、Google Drive上（クラウド側）で直接
「I:アカウントの データ移動 フォルダ」→「G:アカウント(chuya2816)の
1.実験データ_gdrive/5.生データ D フォルダ」への一方向コピーを行う。
Googleのサーバー上で動くトリガーなので、remotefdtd PCの電源状態に
関係なく同期が進行する。

ローカルのrobocopy常駐と併用して構わない（どちらも上書きのみ・削除なし）。

## 前提（設定済み・確認済み）

- 移動先: chuya2816@gmail.com が所有する
  `1.実験データ_gdrive/5.生データ D`（フォルダID: `19XM79gcw2sV-7R74wDco3Pkgu-IPig0v`）
- 移動元: `remotefdtd@gmail.com` 所有の `データ移動` フォルダ
  （フォルダID: `1lJyOa1TciJbSdjz6DWCjUPSkzRWdX3rm`。chuya2816@gmail.comに
  共有済み・`Code.gs`にも設定済み）

## セットアップ手順（残りはこれだけ）

1. [script.google.com](https://script.google.com) に **chuya2816@gmail.com** で
   ログインし、新しいプロジェクトを作成
2. `Code.gs` の内容をそのままエディタに貼り付ける（IDは設定済みなので
   書き換え不要）
3. 左メニューの「サービス」（+ボタン）から **Drive API**
   （Advanced Drive Service）を追加する
4. 関数選択で `createTimeTrigger` を選び、一度だけ手動実行する
   - 初回は権限承認のダイアログが出るので許可する
   - これで5分おきの時間主導トリガーが設置され、以降は自動実行される

## 動作仕様

- `SYNC_INTERVAL_MINUTES`（既定5分）ごとに `syncOneWay` が自動実行される
- 移動元にあって移動先に無いファイル・フォルダ → コピーする
- 移動元の方が新しい（`getLastUpdated()`が新しい）同名ファイル →
  中身だけ上書き（`Drive.Files.update`、ファイルIDは維持）。**trash/削除は行わない**
- 移動先にのみ存在するファイル → 一切触らない（削除しない）
- 実行結果は移動先フォルダ直下の `sync_log_gas.txt` に追記される

## 確認方法

1. `createTimeTrigger` 実行後、Apps Scriptエディタの左メニュー「トリガー」で
   `syncOneWay` が5分間隔で登録されているか確認する
2. I:側（データ移動フォルダ）にテストファイルを置き、共有経由でchuya2816側の
   Driveからも見えることを確認したうえで、5〜10分待って
   `5.生データ D` に反映されるか確認する
3. `sync_log_gas.txt` に実行ログ（`copied=1` など）が追記されているか確認する
4. 確認できたらテストファイルを削除してよいか確認を取ってから削除する

## 制限事項

- Apps Scriptの1回の実行時間上限（無料アカウントで最大6分）があるため、
  一度に大量の新規ファイルがある場合は複数回のトリガー実行に分けて
  処理される（次回実行時に続きが処理される設計ではなく、単純に
  次回トリガーでも同じフォルダを再走査するだけなので、いずれ追いつく）
- Google Drive for Desktopのアップロード中（アップロードが完了しGoogle
  Drive上に反映されるまで）はこのスクリプトからは見えない。ローカルの
  robocopy側にある「撮影中ファイルを掴まない」ための5分待機と役割が近いが、
  念のためアップロード完了後さらに数分の余裕を見て確認すること
