"""
基板位置合わせスクリプト v2（一致率向上版）

v1からの改善点:
  1. スクラッチ位置の検出をサブピクセル精度化（放物線フィッティング）
     → 平行移動の量子化誤差（±0.5px）を解消
  2. 角度検出にシグマクリッピング付きロバスト回帰を採用
     → ゴミ・ムラ等の外れ値行に引っ張られない
  3. 回転と平行移動を1つのアフィン変換に合成して1回だけワープ
     → 補間による画像劣化が1回分で済む
  4. 粗い位置合わせ後にECC（相関係数最大化）による精密位置合わせを追加
     → キズ（スクラッチ）周辺の帯領域だけで相関を評価し、サブピクセルで
       一致率を直接最大化。基板上のキズが位置合わせの基準であり、
       画像内部の試料は画像間で変化しうるため、内部の変化に
       引っ張られないようマスクで除外する
  5. 一致率（正規化相互相関 NCC）をキズ部/画像全体それぞれ表示
     ※キズ部NCC = 位置合わせ精度の指標
       全体NCC = 試料の変化も含んだ画像の一致度
  6. 入力画像のビット深度（uint16等）を保持して保存
"""

import tifffile
import numpy as np
import cv2
from pathlib import Path

# ===== 設定 =====
INPUT_DIR = Path(r"G:\マイドライブ\1.実験データ_gdrive\5.生データ D\260630 dna sam\位置合わせ")
OUTPUT_DIR = INPUT_DIR / "registered_v2"
REFERENCE = "1-0.tif"
TARGETS = ["1-1.tif", "1-2.tif", "1-3.tif"]
SCRATCH_COL_RANGE = (30, 250)   # 縦線スクラッチのcol範囲
ROW_SEARCH = (1800, 2044)       # 横線スクラッチのrow範囲
USE_ECC = True                  # ECC精密位置合わせを使う
ECC_ITERATIONS = 200
ECC_EPS = 1e-7
ECC_SCRATCH_BAND = 40           # キズ中心から±何pxの帯をECC/一致率評価に使うか
NCC_MARGIN = 180                # 全体NCC計算時に除外する外周幅(px)
# ================


def subpixel_argmin(profile):
    """1次元プロファイルの最小位置をサブピクセル精度で返す（放物線補間）"""
    profile = np.asarray(profile, dtype=np.float64)
    i = int(np.argmin(profile))
    if 0 < i < len(profile) - 1:
        y0, y1, y2 = profile[i - 1], profile[i], profile[i + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom > 1e-12:
            delta = 0.5 * (y0 - y2) / denom
            return i + float(np.clip(delta, -0.5, 0.5))
    return float(i)


def detect_scratch_angle(img, col_range=SCRATCH_COL_RANGE):
    """縦線の傾き（度）を各行の最暗点のロバスト線形回帰で検出

    各行の最暗点はサブピクセル化し、シグマクリッピングで
    外れ値行（ゴミ・輝度ムラ等）を除外して回帰する。
    """
    c0, c1 = col_range
    strip = img[:, c0:c1].astype(np.float32)
    n_rows = strip.shape[0]

    # 各行の最暗点（サブピクセル、ベクトル化した放物線補間）
    idx = np.argmin(strip, axis=1)
    idx_c = np.clip(idx, 1, strip.shape[1] - 2)
    rows_i = np.arange(n_rows)
    y0 = strip[rows_i, idx_c - 1]
    y1 = strip[rows_i, idx_c]
    y2 = strip[rows_i, idx_c + 1]
    denom = y0 - 2.0 * y1 + y2
    delta = np.where(denom > 1e-12, 0.5 * (y0 - y2) / np.maximum(denom, 1e-12), 0.0)
    dark_cols = idx + np.clip(delta, -0.5, 0.5) + c0

    # シグマクリッピング付き線形回帰
    rows = rows_i.astype(np.float64)
    mask = np.ones(n_rows, dtype=bool)
    coeffs = np.polyfit(rows, dark_cols, 1)
    for _ in range(3):
        resid = dark_cols - np.polyval(coeffs, rows)
        sigma = np.std(resid[mask])
        if sigma < 1e-9:
            break
        new_mask = np.abs(resid) < 2.5 * sigma
        if new_mask.sum() < n_rows * 0.2 or np.array_equal(new_mask, mask):
            break
        mask = new_mask
        coeffs = np.polyfit(rows[mask], dark_cols[mask], 1)
    return np.degrees(np.arctan(coeffs[0]))


def detect_scratch_col(img, col_search=(30, 300)):
    """縦線colをサブピクセル精度で検出"""
    arr = img.astype(np.float32)
    c0, c1 = col_search
    col_means = arr[:, c0:c1].mean(axis=0)
    return subpixel_argmin(col_means) + c0


def detect_scratch_row(img, row_search=ROW_SEARCH, col_search=(200, 2048)):
    """横線rowをサブピクセル精度で検出（縦線から離れた列の平均で検出）"""
    arr = img.astype(np.float32)
    r0, r1 = row_search
    c0, c1 = col_search
    row_means = arr[r0:r1, c0:c1].mean(axis=1)
    return subpixel_argmin(row_means) + r0


def detect_scratch_position(img, col_search=(30, 300), row_search=ROW_SEARCH):
    col = detect_scratch_col(img, col_search)
    row = detect_scratch_row(img, row_search)
    return col, row


def compose_rot_trans(angle_deg, dx, dy, shape):
    """中心回転 + 平行移動を1つの2x3アフィン行列（target→ref の順変換）に合成"""
    h, w = shape[:2]
    R = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    R3 = np.vstack([R, [0, 0, 1]])
    T3 = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)
    return (T3 @ R3)[:2]


def warp_affine(img, M, border=cv2.BORDER_CONSTANT):
    """順変換行列Mで1回だけワープ（バイキュービック補間）

    検出用途では border=cv2.BORDER_REPLICATE を指定して
    人工的な黒縁がスクラッチ検出を狂わせるのを防ぐ。
    """
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M.astype(np.float32), (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=border,
                          borderValue=0)


def valid_mask(shape, M, erode_px=8):
    """ワープ後に元画像の実データが存在する領域のマスク（uint8）"""
    h, w = shape[:2]
    ones = np.full((h, w), 255, dtype=np.uint8)
    mask = cv2.warpAffine(ones, M.astype(np.float32), (w, h),
                          flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    kernel = np.ones((erode_px, erode_px), np.uint8)
    return cv2.erode(mask, kernel)


def ncc(ref, img, margin=NCC_MARGIN):
    """外周marginを除いた領域の正規化相互相関（画像全体の一致率）"""
    a = ref[margin:-margin, margin:-margin].astype(np.float64).ravel()
    b = img[margin:-margin, margin:-margin].astype(np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def masked_ncc(ref, img, mask):
    """マスク領域内の正規化相互相関（キズ部の一致率評価に使用）"""
    sel = mask > 0
    a = ref[sel].astype(np.float64)
    b = img[sel].astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def scratch_mask(shape, col, row, band=ECC_SCRATCH_BAND):
    """縦キズ・横キズ周辺の帯領域マスク（uint8）

    位置合わせの基準は基板端のキズなので、精密位置合わせ(ECC)と
    キズ部一致率の評価はこの領域に限定する。
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    c = int(round(col))
    r = int(round(row))
    mask[:, max(0, c - band):min(w, c + band + 1)] = 255   # 縦キズの帯
    mask[max(0, r - band):min(h, r + band + 1), :] = 255   # 横キズの帯
    return mask


def refine_ecc(ref, moving, mask=None):
    """ECCで残差変換（moving→ref の順変換）を推定。失敗時はNoneを返す

    maskにはワープで実データが存在する領域（valid_mask）を渡し、
    黒縁が相関計算に混入しないようにする。
    """
    def norm8(x):
        lo, hi = np.percentile(x, (1, 99))
        return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)

    ref_n = norm8(ref)
    mov_n = norm8(moving)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                ECC_ITERATIONS, ECC_EPS)
    try:
        cc, warp = cv2.findTransformECC(ref_n, mov_n, warp,
                                        cv2.MOTION_EUCLIDEAN, criteria,
                                        mask, 5)
    except cv2.error:
        return None, None
    # findTransformECCのwarpは ref座標→moving座標 の写像なので逆変換が順変換
    fwd = cv2.invertAffineTransform(warp)
    return fwd.astype(np.float64), cc


def register_to_reference(img, ref, ref_angle, ref_col, ref_row, verbose=True):
    """1枚の画像を基準に位置合わせし、(補正後画像, 情報dict) を返す"""
    tgt_angle = detect_scratch_angle(img)
    tgt_col, tgt_row = detect_scratch_position(img)

    # Step1: 回転補正量（基準との角度差を打ち消す）
    rot_angle = ref_angle - tgt_angle
    M_rot = compose_rot_trans(rot_angle, 0, 0, img.shape)

    # Step2: 回転後座標でのスクラッチ位置から平行移動量を求める
    #        （検出専用の一時回転。黒縁で検出が狂わないようreplicate境界）
    img_rot = warp_affine(img, M_rot, border=cv2.BORDER_REPLICATE)
    col_margin = 150
    col_lo = int(max(30, tgt_col - col_margin))
    col_hi = int(min(2040, tgt_col + col_margin))
    rot_col, rot_row = detect_scratch_position(img_rot, col_search=(col_lo, col_hi))
    dx = ref_col - rot_col
    dy = ref_row - rot_row

    # 回転+平行移動を合成し、元画像から1回のワープで粗補正
    M_coarse = compose_rot_trans(rot_angle, dx, dy, img.shape)
    img_coarse = warp_affine(img, M_coarse)

    # 一致率評価用マスク: 基準画像のキズ周辺の帯
    s_mask = scratch_mask(ref.shape, ref_col, ref_row)

    ncc_before = ncc(ref, img)
    ncc_coarse = ncc(ref, img_coarse)
    sncc_before = masked_ncc(ref, img, s_mask)
    sncc_coarse = masked_ncc(ref, img_coarse, s_mask)

    # Step3: ECCによる精密位置合わせ（粗補正への残差をサブピクセル推定）
    # 位置合わせの基準はキズなので、相関評価はキズ周辺の帯に限定し、
    # 画像内部の試料の変化に引っ張られないようにする
    M_final = M_coarse
    img_final = img_coarse
    ncc_final = ncc_coarse
    sncc_final = sncc_coarse
    ecc_used = False
    if USE_ECC:
        mask = cv2.bitwise_and(valid_mask(img.shape, M_coarse), s_mask)
        fwd_res, _ = refine_ecc(ref, img_coarse, mask)
        if fwd_res is not None:
            M3_res = np.vstack([fwd_res, [0, 0, 1]])
            M3_coarse = np.vstack([M_coarse, [0, 0, 1]])
            M_refined = (M3_res @ M3_coarse)[:2]
            img_refined = warp_affine(img, M_refined)
            sncc_refined = masked_ncc(ref, img_refined, s_mask)
            # キズ部の一致率が改善する場合のみ採用
            if sncc_refined >= sncc_coarse:
                M_final, img_final = M_refined, img_refined
                sncc_final = sncc_refined
                ncc_final = ncc(ref, img_refined)
                ecc_used = True

    # 最終変換から回転・平行移動を読み取る（レポート用）
    final_rot = np.degrees(np.arctan2(M_final[1, 0], M_final[0, 0]))
    final_dx = M_final[0, 2]
    final_dy = M_final[1, 2]

    info = {
        "rot": final_rot, "dx": dx, "dy": dy,
        "ncc_before": ncc_before, "ncc_coarse": ncc_coarse,
        "ncc_final": ncc_final,
        "sncc_before": sncc_before, "sncc_coarse": sncc_coarse,
        "sncc_final": sncc_final,
        "ecc_used": ecc_used,
        "M": M_final,
    }
    if verbose:
        print(f"  縦線角度: {tgt_angle:.4f}°  回転補正: {rot_angle:+.4f}°")
        print(f"  平行移動: dx={dx:+.2f}px  dy={dy:+.2f}px")
        print(f"  キズ部一致率: 補正前 {sncc_before:.5f} → 粗補正 {sncc_coarse:.5f}"
              f" → 精密補正 {sncc_final:.5f}"
              f"{'' if ecc_used else '（ECC不採用/失敗のため粗補正を使用）'}")
        print(f"  全体一致率:   補正前 {ncc_before:.5f} → 補正後 {ncc_final:.5f}")
    return img_final, info


def save_like(img_float, like_dtype, path):
    """入力と同じビット深度で保存（uint8/uint16の範囲にクリップ）"""
    if np.issubdtype(like_dtype, np.integer):
        info = np.iinfo(like_dtype)
        out = np.clip(np.rint(img_float), info.min, info.max).astype(like_dtype)
    else:
        out = img_float.astype(np.float32)
    tifffile.imwrite(path, out)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    ref_raw = tifffile.imread(INPUT_DIR / REFERENCE)
    ref_dtype = ref_raw.dtype
    ref = ref_raw.astype(np.float32)
    ref_angle = detect_scratch_angle(ref)
    ref_col, ref_row = detect_scratch_position(ref)
    print(f"基準画像: {REFERENCE}")
    print(f"  縦線角度: {ref_angle:.4f}°  縦線col: {ref_col:.2f}  横線row: {ref_row:.2f}\n")

    results = []
    for name in TARGETS:
        path = INPUT_DIR / name
        if not path.exists():
            print(f"スキップ: {name}")
            continue

        img = tifffile.imread(path).astype(np.float32)
        print(f"{name}:")
        img_final, info = register_to_reference(img, ref, ref_angle, ref_col, ref_row)

        out_path = OUTPUT_DIR / f"{path.stem}_registered.tif"
        save_like(img_final, ref_dtype, out_path)
        print(f"  → 保存: {out_path.name}\n")

        info["file"] = name
        results.append(info)

    print("=== サマリー ===")
    print(f"{'ファイル':<12} {'回転(deg)':>10} {'dx(px)':>9} {'dy(px)':>9} "
          f"{'キズ部NCC':>10} {'全体NCC':>10}")
    for r in results:
        print(f"{r['file']:<12} {r['rot']:>+10.4f} {r['dx']:>+9.2f} {r['dy']:>+9.2f} "
              f"{r['sncc_final']:>10.5f} {r['ncc_final']:>10.5f}")


if __name__ == "__main__":
    main()
