"""
Obsidian Vault 整理スクリプト
- 空/中身のないMarkdownノートを検出
- 内容が重複しているファイル（ハッシュ一致）を検出
- デフォルトはdry-run（一覧表示のみ）。--execute で "_to_review" フォルダへ移動（完全削除はしない）
"""

import argparse
import hashlib
import shutil
from datetime import datetime
from pathlib import Path

# ===== 設定 =====
VAULT_DIR = Path(r"/Users/shiraishichuuya/Library/CloudStorage/GoogleDrive-chuya2816@gmail.com/マイドライブ/Obsidian Vault")
REVIEW_DIR_NAME = "_to_review"
EXCLUDE_DIRS = {".obsidian", ".trash", REVIEW_DIR_NAME}
# ================


def iter_markdown_files(vault_dir: Path):
    for path in vault_dir.rglob("*.md"):
        if any(part in EXCLUDE_DIRS for part in path.relative_to(vault_dir).parts):
            continue
        yield path


def find_empty_notes(vault_dir: Path):
    """本文が空白のみのノートを検出"""
    empty = []
    for path in iter_markdown_files(vault_dir):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if text.strip() == "":
            empty.append(path)
    return empty


def find_duplicate_notes(vault_dir: Path):
    """内容ハッシュが一致するノートをグループ化（各グループの最古のファイルを残す）"""
    hash_to_paths = {}
    for path in iter_markdown_files(vault_dir):
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        hash_to_paths.setdefault(digest, []).append(path)

    duplicates = []
    for paths in hash_to_paths.values():
        if len(paths) < 2:
            continue
        paths.sort(key=lambda p: p.stat().st_mtime)
        keep, *rest = paths
        for dup in rest:
            duplicates.append((dup, keep))
    return duplicates


def move_to_review(path: Path, vault_dir: Path, review_root: Path):
    rel = path.relative_to(vault_dir)
    dest = review_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(dest))
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=VAULT_DIR, help="Vaultのパス")
    parser.add_argument("--execute", action="store_true",
                         help="指定しない場合は一覧表示のみ（dry-run）。指定すると_to_reviewフォルダへ移動する")
    args = parser.parse_args()

    vault_dir = args.vault
    if not vault_dir.exists():
        print(f"Vaultが見つかりません: {vault_dir}")
        return

    empty_notes = find_empty_notes(vault_dir)
    duplicates = find_duplicate_notes(vault_dir)

    print(f"=== 空ノート ({len(empty_notes)}件) ===")
    for path in empty_notes:
        print(f"  {path.relative_to(vault_dir)}")

    print(f"\n=== 重複ノート ({len(duplicates)}件、括弧内は残すファイル) ===")
    for dup, keep in duplicates:
        print(f"  {dup.relative_to(vault_dir)}  (keep: {keep.relative_to(vault_dir)})")

    candidates = empty_notes + [dup for dup, _ in duplicates]

    if not candidates:
        print("\n削除候補はありません。")
        return

    if not args.execute:
        print(f"\n合計 {len(candidates)} 件が削除候補です。実行するには --execute を付けてください。")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    review_root = vault_dir / REVIEW_DIR_NAME / timestamp
    print(f"\n{len(candidates)} 件を {review_root} へ移動します（完全削除ではありません）。")
    for path in candidates:
        dest = move_to_review(path, vault_dir, review_root)
        print(f"  移動: {path.relative_to(vault_dir)} → {dest.relative_to(vault_dir)}")

    print(f"\n完了。内容を確認後、不要であれば {review_root} を手動で削除してください。")


if __name__ == "__main__":
    main()
