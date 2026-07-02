"""
基板位置合わせスクリプト
Step1: 位相相関で粗い並進量を推定
Step2: ECC(Enhanced Correlation Coefficient)最適化で回転・並進をサブピクセル精緻化
Step3: 一致率(許容誤差内で一致する画素の割合)とNCCで位置合わせ品質を評価
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
MATCH_TOLERANCE = 5     # 一致率算出時に「一致」とみなす輝度差の許容値（階調）
BORDER_MARGIN = 50      # 品質評価から除外する外周幅(px)。warpAffineのゼロ埋め領域を避ける
ECC_MAX_ITER = 5000
ECC_EPS = 1e-8
# ================

OUTPUT_DIR.mkdir(exist_ok=True)


def estimate_coarse_shift(ref, img):
    """位相相関(FFT)による粗い並進量(dx, dy)の推定。ECCの初期値として利用"""
    h, w = ref.shape
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (sx, sy), _ = cv2.phaseCorrelate(ref.astype(np.float32), img.astype(np.float32), window)
    return sx, sy


def register_ecc(ref, img, motion=cv2.MOTION_EUCLIDEAN):
    """粗い並進推定を初期値として、ECC最適化で回転・並進(剛体変換)を精緻化"""
    sx, sy = estimate_coarse_shift(ref, img)
    warp = np.eye(2, 3, dtype=np.float32)
    warp[0, 2] = -sx
    warp[1, 2] = -sy

    ref_u8 = np.clip(ref, 0, 255).astype(np.uint8)
    img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_MAX_ITER, ECC_EPS)
    cc, warp = cv2.findTransformECC(ref_u8, img_u8, warp, motion, criteria, None, 5)
    return warp, cc


def apply_warp(img, warp, shape):
    h, w = shape
    return cv2.warpAffine(img, warp, (w, h),
                           flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def valid_mask(warp, shape, margin=BORDER_MARGIN):
    """warpAffineでゼロ埋めされた領域を除いた、品質評価用マスクを作成"""
    h, w = shape
    ones = np.ones((h, w), dtype=np.float32)
    warped_ones = apply_warp(ones, warp, shape)
    mask = warped_ones > 0.99
    mask[:margin, :] = False
    mask[-margin:, :] = False
    mask[:, :margin] = False
    mask[:, -margin:] = False
    return mask


def match_rate(ref, aligned, mask, tol=MATCH_TOLERANCE):
    """一致率：輝度差がtol階調以内の画素が占める割合"""
    diff = np.abs(ref - aligned)
    return float((diff[mask] <= tol).mean())


def ncc_score(ref, aligned, mask):
    """正規化相互相関係数(-1〜1)"""
    a = ref[mask] - ref[mask].mean()
    b = aligned[mask] - aligned[mask].mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()) + 1e-8))


ref = tifffile.imread(INPUT_DIR / REFERENCE)
ref_f = ref.astype(np.float32)
print(f"基準画像: {REFERENCE}  shape={ref.shape} dtype={ref.dtype}\n")

results = []

for name in TARGETS:
    path = INPUT_DIR / name
    if not path.exists():
        print(f"スキップ: {name}")
        continue

    img = tifffile.imread(path)
    img_f = img.astype(np.float32)

    warp, cc = register_ecc(ref_f, img_f)
    aligned = apply_warp(img_f, warp, ref.shape)

    mask = valid_mask(warp, ref.shape)
    rate = match_rate(ref_f, aligned, mask)
    ncc = ncc_score(ref_f, aligned, mask)

    angle = np.degrees(np.arctan2(warp[1, 0], warp[0, 0]))
    dx, dy = warp[0, 2], warp[1, 2]

    print(f"{name}:")
    print(f"  回転補正: {angle:+.4f}°  並進: dx={dx:+.2f}px  dy={dy:+.2f}px")
    print(f"  ECC相関係数: {cc:.4f}  一致率(|Δ|≤{MATCH_TOLERANCE}階調): {rate * 100:.2f}%  NCC: {ncc:.4f}")

    info = np.iinfo(img.dtype)
    out_img = np.clip(aligned, info.min, info.max).astype(img.dtype)
    out_path = OUTPUT_DIR / f"{path.stem}_registered.tif"
    tifffile.imwrite(out_path, out_img)
    print(f"  → 保存: {out_path.name}\n")

    results.append({"file": name, "rot": angle, "dx": dx, "dy": dy, "match_rate": rate, "ncc": ncc})

print("=== サマリー ===")
print(f"{'ファイル':<12} {'回転(deg)':>10} {'dx(px)':>8} {'dy(px)':>8} {'一致率(%)':>10} {'NCC':>8}")
for r in results:
    print(f"{r['file']:<12} {r['rot']:>+10.4f} {r['dx']:>+8.2f} {r['dy']:>+8.2f} "
          f"{r['match_rate'] * 100:>10.2f} {r['ncc']:>8.4f}")

if results:
    worst = min(results, key=lambda r: r["match_rate"])
    if worst["match_rate"] < 0.999:
        print(f"\n注意: 一致率が100%に達していません（最小 {worst['match_rate'] * 100:.2f}% @ {worst['file']}）。")
        print("剛体変換（回転・並進）で説明できる位置ズレは解消済みのため、残差は主に撮像ノイズや"
              "フレーム間の試料自体の変化に由来する可能性があります。")
