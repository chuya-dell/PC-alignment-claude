/**
 * I:ドライブ（remotefdtd側アカウントの「データ移動」フォルダ）から
 * G:ドライブ（chuya2816の「1.実験データ_gdrive/5.生データ D」）への
 * 一方向バックアップ（クラウド側コピー）。
 *
 * remotefdtd PCの電源が落ちていても、Googleのサーバー上でトリガーが
 * 動く限り同期が進む。Google Drive for Desktopが各PC側にファイルを
 * 同期するのはそれぞれのアカウントの役目なので、このスクリプトは
 * あくまでcrowd側（Drive上）でのI:アカウント→G:アカウントのコピーを担う。
 *
 * 事前準備:
 *  1. remotefdtd側アカウントで「データ移動」フォルダをchuya2816@gmail.com
 *     に共有（閲覧者権限でOK）
 *  2. このプロジェクトをchuya2816@gmail.comでscript.google.comに作成し、
 *     このファイルを貼り付ける
 *  3. 左メニュー「サービス」から「Drive API」（Advanced Drive Service）を追加
 *  4. 下の SOURCE_FOLDER_ID / DEST_FOLDER_ID を実際のフォルダIDに書き換える
 *  5. createTimeTrigger を一度手動実行し、権限を許可する
 */

// ==== 設定 ====
const SOURCE_FOLDER_ID = '1lJyOa1TciJbSdjz6DWCjUPSkzRWdX3rm'; // remotefdtd@gmail.com の「データ移動」フォルダ（chuya2816に共有済み）
const DEST_FOLDER_ID = '19XM79gcw2sV-7R74wDco3Pkgu-IPig0v'; // chuya2816「1.実験データ_gdrive/5.生データ D」
const SYNC_INTERVAL_MINUTES = 5; // 変更検知〜同期のインターバル
const LOG_FILE_NAME = 'sync_log_gas.txt';
// ===============

function syncOneWay() {
  const srcRoot = DriveApp.getFolderById(SOURCE_FOLDER_ID);
  const destRoot = DriveApp.getFolderById(DEST_FOLDER_ID);
  const stats = { copied: 0, updated: 0, skipped: 0, foldersCreated: 0, errors: 0 };

  try {
    mirrorFolder(srcRoot, destRoot, stats);
  } catch (e) {
    stats.errors++;
    Logger.log('syncOneWay error: ' + e);
  }

  logResult(stats);
}

function mirrorFolder(srcFolder, destFolder, stats) {
  const files = srcFolder.getFiles();
  while (files.hasNext()) {
    copyOrUpdateFile(files.next(), destFolder, stats);
  }

  const subFolders = srcFolder.getFolders();
  while (subFolders.hasNext()) {
    const sub = subFolders.next();
    const destSub = getOrCreateSubFolder(destFolder, sub.getName(), stats);
    mirrorFolder(sub, destSub, stats);
  }
}

function getOrCreateSubFolder(parent, name, stats) {
  const existing = parent.getFoldersByName(name);
  if (existing.hasNext()) return existing.next();
  stats.foldersCreated++;
  return parent.createFolder(name);
}

// 上書きはするが、削除は一切行わない（destのみに存在するファイル/フォルダには触れない）
function copyOrUpdateFile(srcFile, destFolder, stats) {
  const name = srcFile.getName();
  const existing = destFolder.getFilesByName(name);

  if (!existing.hasNext()) {
    srcFile.makeCopy(name, destFolder);
    stats.copied++;
    return;
  }

  const destFile = existing.next();
  if (srcFile.getLastUpdated().getTime() > destFile.getLastUpdated().getTime()) {
    // 中身だけ上書き（ファイルIDは維持、trash/削除は発生しない）
    Drive.Files.update({}, destFile.getId(), srcFile.getBlob());
    stats.updated++;
  } else {
    stats.skipped++;
  }
}

function logResult(stats) {
  const line = `[${new Date().toISOString()}] copied=${stats.copied} updated=${stats.updated} `
    + `skipped=${stats.skipped} foldersCreated=${stats.foldersCreated} errors=${stats.errors}`;
  Logger.log(line);

  const destParent = DriveApp.getFolderById(DEST_FOLDER_ID);
  const existingLog = destParent.getFilesByName(LOG_FILE_NAME);
  if (existingLog.hasNext()) {
    const logFile = existingLog.next();
    const updated = logFile.getBlob().getDataAsString() + line + '\n';
    Drive.Files.update({}, logFile.getId(), Utilities.newBlob(updated, 'text/plain', LOG_FILE_NAME));
  } else {
    destParent.createFile(LOG_FILE_NAME, line + '\n', MimeType.PLAIN_TEXT);
  }
}

// 初回に一度だけ手動実行するセットアップ関数
function createTimeTrigger() {
  ScriptApp.getProjectTriggers().forEach(t => {
    if (t.getHandlerFunction() === 'syncOneWay') ScriptApp.deleteTrigger(t);
  });
  ScriptApp.newTrigger('syncOneWay')
    .timeBased()
    .everyMinutes(SYNC_INTERVAL_MINUTES)
    .create();
  Logger.log(`Trigger installed: syncOneWay every ${SYNC_INTERVAL_MINUTES} min`);
}
