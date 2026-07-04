"""
傷(スクラッチ)検出の診断ツール。

batch_register_images.py の傷ランドマーク検出が実データで低プロミネンス/失敗する
原因(波打ち・にじみ・帯域ズレ・極性)を目視で確定するために、1枚の画像に対して:
  - コントラスト強調した縮小プレビューに、探索帯域と検出結果を重畳
  - 縦傷帯: 各行で最強ピーク列を追跡した軌跡(波打ちが分かる)+ 帯域平均法の検出列
  - 横傷帯: 各列で最強ピーク行を追跡した軌跡 + 帯域平均法の検出行
  - 帯域平均プロミネンス(明/暗)、行(列)追跡のインライア率・位置ばらつき(波打ち量)
を1枚の図と数値で出力する。

実行例:
  python utils/diagnose_scratch.py --image "F:/.../df/1-1-0.tif" --output-dir scratch_diag
"""
import argparse
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# batch_register_images.py と同じ既定帯域
V_COL_RANGE = (0, 300)
V_ROW_BAND = (500, 1500)
H_ROW_RANGE = (0, 300)
H_COL_BAND = (1000, 1900)
SMOOTH = 15
EDGE = 30


def load_gray(path):
    try:
        import tifffile
        a = tifffile.imread(str(path))
    except Exception:
        a = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if a.ndim == 3:
        a = a.mean(-1)
    return a.astype(np.float64)


def smooth1d(v, k=SMOOTH):
    return np.convolve(v, np.ones(k) / k, mode="same")


def band_average_peak(band, axis, edge=EDGE):
    """現行 batch と同じ: 帯域平均→平滑→端除外で明/暗プロミネンスと位置を返す。"""
    prof = smooth1d(np.mean(band, axis=axis))
    lo, hi = edge, len(prof) - edge
    sub = prof[lo:hi]
    med = np.median(sub)
    mad = np.median(np.abs(sub - med)) * 1.4826 + 1e-6
    mn = int(np.argmin(sub)); dark = (med - sub[mn]) / mad
    mx = int(np.argmax(sub)); bright = (sub[mx] - med) / mad
    if bright >= dark:
        return lo + mx, bright, "bright", prof, lo
    return lo + mn, dark, "dark", prof, lo


def per_line_trace(band, line_axis, polarity, edge=EDGE, tol=6.0):
    """
    line_axis 方向に走る線を、各スライスの極値位置で追跡する。
    band を (走査N × 探索M) に整え、各行でピーク位置(列)を取る。
    戻り値: positions(各スライスのピーク位置), inlier_frac, pos_at_center, waviness_std
    """
    m = band if line_axis == 0 else band.T  # 常に行方向に線が走るよう転置
    N, M = m.shape
    ms = np.apply_along_axis(smooth1d, 1, m)
    lo, hi = edge, M - edge
    sub = ms[:, lo:hi]
    row_med = np.median(sub, axis=1)
    if polarity == "bright":
        pos = lo + np.argmax(sub, axis=1)
        contrast = np.max(sub, axis=1) - row_med
    else:
        pos = lo + np.argmin(sub, axis=1)
        contrast = row_med - np.min(sub, axis=1)

    idx = np.arange(N)
    good = contrast > np.median(contrast)  # コントラストのある行のみで傾き推定
    if good.sum() < 5:
        return pos, 0.0, float(np.median(pos)), float(np.std(pos))
    coeffs = np.polyfit(idx[good], pos[good], 2)  # 波打ち/傾きを2次で吸収
    fit = np.polyval(coeffs, idx)
    resid = pos - fit
    inlier = np.abs(resid) <= tol
    frac = float(np.mean(inlier))
    center = float(np.polyval(coeffs, N / 2))
    wav = float(np.std(fit))  # フィット自体の振れ幅 ≒ 波打ち量
    return pos, frac, center, wav


def main():
    ap = argparse.ArgumentParser(description="傷検出の診断(波打ち・にじみ・極性の可視化)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--output-dir", default="scratch_diag")
    ap.add_argument("--v-col-range", type=lambda s: tuple(int(x) for x in s.split(",")), default=V_COL_RANGE)
    ap.add_argument("--v-row-band", type=lambda s: tuple(int(x) for x in s.split(",")), default=V_ROW_BAND)
    ap.add_argument("--h-row-range", type=lambda s: tuple(int(x) for x in s.split(",")), default=H_ROW_RANGE)
    ap.add_argument("--h-col-band", type=lambda s: tuple(int(x) for x in s.split(",")), default=H_COL_BAND)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    img = load_gray(args.image)
    h, w = img.shape
    name = Path(args.image).name

    # 縦傷帯
    vr0, vr1 = args.v_row_band; vc0, vc1 = args.v_col_range
    v_band = img[vr0:vr1, vc0:vc1]
    v_pos, v_prom, v_pol, v_prof, v_lo = band_average_peak(v_band, axis=0)
    scratch_x = vc0 + v_pos
    v_trace, v_frac, v_center, v_wav = per_line_trace(v_band, line_axis=0, polarity=v_pol)
    scratch_x_trace = vc0 + v_center

    # 横傷帯
    hr0, hr1 = args.h_row_range; hc0, hc1 = args.h_col_band
    h_band = img[hr0:hr1, hc0:hc1]
    h_pos, h_prom, h_pol, h_prof, h_lo = band_average_peak(h_band, axis=1)
    scratch_y = hr0 + h_pos
    h_trace, h_frac, h_center, h_wav = per_line_trace(h_band, line_axis=1, polarity=h_pol)
    scratch_y_trace = hr0 + h_center

    print(f"=== {name} ({w}x{h}) ===")
    print(f"[縦傷] 帯域平均: x={scratch_x:.1f}, prominence={v_prom:.1f} ({v_pol})")
    print(f"       行追跡:   x={scratch_x_trace:.1f}, inlier率={v_frac:.2f}, 波打ちstd={v_wav:.1f}px")
    print(f"[横傷] 帯域平均: y={scratch_y:.1f}, prominence={h_prom:.1f} ({h_pol})")
    print(f"       列追跡:   y={scratch_y_trace:.1f}, inlier率={h_frac:.2f}, 波打ちstd={h_wav:.1f}px")
    print("解釈のヒント: inlier率が高く(>0.6)波打ちstdが大きいなら『線は明瞭だが波打ちで帯域平均が鈍る』"
          "→ 行/列追跡ベース検出に切替が有効。inlier率も低いなら線自体が弱い/帯域外。")

    # 可視化
    lo_p, hi_p = np.percentile(img, 1), np.percentile(img, 99.5)
    disp = np.clip((img - lo_p) / (hi_p - lo_p) * 255, 0, 255).astype(np.uint8)
    scale = 900 / max(h, w)
    small = cv2.resize(disp, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    canvas = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)

    def sx(x): return int(x * scale)
    def sy(y): return int(y * scale)
    # 探索帯域(黄)
    cv2.rectangle(canvas, (sx(vc0), sy(vr0)), (sx(vc1), sy(vr1)), (0, 220, 220), 1)
    cv2.rectangle(canvas, (sx(hc0), sy(hr0)), (sx(hc1), sy(hr1)), (0, 220, 220), 1)
    # 帯域平均の検出線(赤)
    cv2.line(canvas, (sx(scratch_x), 0), (sx(scratch_x), small.shape[0]), (0, 0, 255), 1)
    cv2.line(canvas, (0, sy(scratch_y)), (small.shape[1], sy(scratch_y)), (0, 0, 255), 1)
    # 行/列追跡の軌跡(緑)
    for r, c in zip(range(vr0, vr1), v_trace):
        cv2.circle(canvas, (sx(vc0 + c), sy(r)), 1, (0, 255, 0), -1)
    for c, r in zip(range(hc0, hc1), h_trace):
        cv2.circle(canvas, (sx(c), sy(hr0 + r)), 1, (0, 255, 0), -1)
    overlay_path = out / f"{Path(args.image).stem}_scratch_overlay.png"
    cv2.imwrite(str(overlay_path), canvas)

    # プロファイル図
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(np.arange(vc0, vc0 + len(v_prof)), v_prof)
    ax[0].axvline(scratch_x, color="r", ls="--", label=f"band-avg x={scratch_x:.0f} ({v_pol})")
    ax[0].axvline(scratch_x_trace, color="g", ls=":", label=f"trace x={scratch_x_trace:.0f}")
    ax[0].set_title(f"V-band column profile (prom={v_prom:.1f})"); ax[0].legend(); ax[0].set_xlabel("col")
    ax[1].plot(np.arange(hr0, hr0 + len(h_prof)), h_prof)
    ax[1].axvline(scratch_y, color="r", ls="--", label=f"band-avg y={scratch_y:.0f} ({h_pol})")
    ax[1].axvline(scratch_y_trace, color="g", ls=":", label=f"trace y={scratch_y_trace:.0f}")
    ax[1].set_title(f"H-band row profile (prom={h_prom:.1f})"); ax[1].legend(); ax[1].set_xlabel("row")
    fig.tight_layout()
    prof_path = out / f"{Path(args.image).stem}_scratch_profiles.png"
    fig.savefig(str(prof_path), dpi=110); plt.close(fig)

    print(f"出力: {overlay_path}")
    print(f"      {prof_path}")


if __name__ == "__main__":
    main()
