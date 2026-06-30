"""
基板位置合わせスクリプト
Step1: 縦線の傾きから回転補正
Step2: 輝度最小位置から平行移動補正
"""

import tifffile
import numpy as np
import cv2
from pathlib import Path

# ===== 設定 =====
INPUT_DIR = Path(r"G:\マイドライブ\1.実験データ_gdrive\5.生データ D\260630 dna sam\位置合わせ")
OUTPUT_DIR = INPUT_DIR / "registered"
REFERENCE = "1-0.tif"
TARGETS = ["1-1.tif", "1-2.tif", "1-3.tif"]
SCRATCH_COL_RANGE = (30, 250)  # 縦線スクラッチのcol範囲
# ================

OUTPUT_DIR.mkdir(exist_ok=True)

def detect_scratch_angle(img):
    """縦線の傾き（度）を各行の最暗点の線形回帰で検出"""
    c0, c1 = SCRATCH_COL_RANGE
    strip = img[:, c0:c1].astype(np.float32)
    dark_cols = np.argmin(strip, axis=1) + c0
    rows = np.arange(len(dark_cols))
    coeffs = np.polyfit(rows, dark_cols, 1)
    return np.degrees(np.arctan(coeffs[0]))  # 傾き（度）

def detect_scratch_col(img, col_search=(30, 300)):
    """縦線colを検出（端の暗い領域を除いた範囲で最小値）"""
    arr = img.astype(np.float32)
    c0, c1 = col_search
    col_means = arr[:, c0:c1].mean(axis=0)
    return int(np.argmin(col_means)) + c0

def detect_scratch_row(img, row_search=(1800, 2044), col_search=(200, 2048)):
    """横線rowを検出（縦線から離れた列の平均で検出）"""
    arr = img.astype(np.float32)
    r0, r1 = row_search
    c0, c1 = col_search
    row_means = arr[r0:r1, c0:c1].mean(axis=1)
    return int(np.argmin(row_means)) + r0

def detect_scratch_position(img, col_search=(30, 300), row_search=(1800, 2044)):
    col = detect_scratch_col(img, col_search)
    row = detect_scratch_row(img, row_search)
    return col, row

def rotate_image(img, angle_deg, center=None):
    """画像を中心周りに回転（背景はNaN→後でclip）"""
    h, w = img.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=0)
    return rotated

def translate_image(img, dx, dy):
    """画像を平行移動"""
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0)

ref = tifffile.imread(INPUT_DIR / REFERENCE).astype(np.float32)
ref_angle = detect_scratch_angle(ref)
ref_col, ref_row = detect_scratch_position(ref)
print(f"基準画像: {REFERENCE}")
print(f"  縦線角度: {ref_angle:.4f}°  縦線col: {ref_col}  横線row: {ref_row}\n")

results = []

for name in TARGETS:
    path = INPUT_DIR / name
    if not path.exists():
        print(f"スキップ: {name}")
        continue

    img = tifffile.imread(path).astype(np.float32)
    tgt_angle = detect_scratch_angle(img)
    tgt_col, tgt_row = detect_scratch_position(img)

    # Step1: 回転補正（基準との角度差を打ち消す）
    rot_angle = ref_angle - tgt_angle
    img_rot = rotate_image(img, rot_angle)

    # Step2: 回転後の画像で平行移動補正（元のスクラッチ位置±150pxの範囲で検索）
    col_margin = 150
    col_lo = max(30, tgt_col - col_margin)
    col_hi = min(2040, tgt_col + col_margin)
    rot_col, rot_row = detect_scratch_position(img_rot,
                                               col_search=(col_lo, col_hi))
    dx = ref_col - rot_col
    dy = ref_row - rot_row
    img_final = translate_image(img_rot, dx, dy)

    total = np.hypot(dx, dy)
    print(f"{name}:")
    print(f"  縦線角度: {tgt_angle:.4f}°  回転補正: {rot_angle:+.4f}°")
    print(f"  平行移動: dx={dx:+d}px  dy={dy:+d}px  |shift|={total:.1f}px")

    out_img = np.clip(img_final, 0, 255).astype(np.uint8)
    out_path = OUTPUT_DIR / f"{path.stem}_registered.tif"
    tifffile.imwrite(out_path, out_img)
    print(f"  → 保存: {out_path.name}\n")

    results.append({"file": name, "rot": rot_angle, "dx": dx, "dy": dy, "total": total})

print("=== サマリー ===")
print(f"{'ファイル':<12} {'回転(deg)':>10} {'dx(px)':>8} {'dy(px)':>8} {'|ズレ|(px)':>12}")
for r in results:
    print(f"{r['file']:<12} {r['rot']:>+10.4f} {r['dx']:>+8d} {r['dy']:>+8d} {r['total']:>12.1f}")
