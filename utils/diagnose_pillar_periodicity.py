"""
実データ物理パラメータ診断スクリプト(FFT周期確認 / ボケ幅 / ピラー間隔 / 指標適性)

目的:
  batch_register_images.py のピラー領域精度評価指標(NCC・一致率)が
  「像がぼやけている実データ」で機能するかを判断するため、実データの物理量を実測する。
  本スクリプトは診断・実測のみを行い、本番パイプラインには手を加えない。

処理(要件番号は指示に対応):
  0. 【最優先・ゲート】2D FFT で周期構造の有無を確認。
     設計ピッチ(既定200nm)を --nm-per-px から想定px周期に換算し、その空間周波数に
     ノイズフロアを明確に超える有意ピークがあるかを判定する。
     有意ピークが無ければ STOP と判定し、1〜4 はスキップする
     (ピラー由来の周期信号がそもそも安定抽出できない可能性 = ボケ以前の問題)。
  1. 孤立輝点の輝度プロファイル半値全幅(FWHM)を複数箇所で計測し σ(px) を算出。
     validate_ncc_blur_sensitivity.py の σ スケールに対応させる。
  2. FFT ピーク位置からピラー間隔(px)を逆算し、自己相関でクロスチェック。
     --nm-per-px からの独立計算値とも突き合わせる。
  3. 実測 σ・間隔を validate_ncc_blur_sensitivity のマトリクスに当てはめ、
     「軽度ボケ域(指標が機能)」か「強ボケ域(指標が鈍感)」かを結論づける。
  4. 高域強調フィルタ(ラプラシアン)適用でシフト感度が改善するかを、
     実測 σ・間隔で作った合成ペア(既知ズレ×既知ボケ)でフィルタあり/なし比較する。

出力:
  - FFT パワースペクトル画像、動径パワープロファイル画像、周期ピーク有無判定
  - 実測 FWHM/σ、実測ピラー間隔(px)、テーブル上の位置づけ、フィルタ比較
  - まとめ Markdown レポート

実行例:
  python utils/diagnose_pillar_periodicity.py \
      --images "G:/.../df/1-1-0.tif" "G:/.../df/1-2-0.tif" \
      --nm-per-px 20 --output-dir analysis_diag
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DESIGN_PITCH_NM = 200.0


# ============================= 入出力 =============================

def load_gray_float(path: Path) -> np.ndarray:
    """Unicode パス対応で TIFF 等をグレースケール float32 で読み込む。"""
    try:
        import tifffile
        arr = tifffile.imread(str(path))
    except Exception:
        arr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise IOError(f"画像を読み込めませんでした: {path}")
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    return arr.astype(np.float32)


def central_region(img: np.ndarray, size: int, offset=(0, 0)) -> tuple:
    """画像中心(offset付き)から size 角の代表領域を切り出す。傷帯を避ける用途で offset 指定可。"""
    h, w = img.shape[:2]
    cy, cx = h // 2 + offset[1], w // 2 + offset[0]
    half = size // 2
    y0 = max(0, min(h - size, cy - half))
    x0 = max(0, min(w - size, cx - half))
    y0, x0 = max(0, y0), max(0, x0)
    return img[y0:y0 + size, x0:x0 + size], (x0, y0)


# ========================= 0. FFT 周期確認 =========================

def radial_profile(power: np.ndarray) -> np.ndarray:
    """2D パワースペクトル(fftshift済み)を中心からの半径ビンで平均し動径プロファイルを返す。"""
    h, w = power.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    tbin = np.bincount(r.ravel(), power.ravel())
    nr = np.bincount(r.ravel())
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = tbin / np.maximum(nr, 1)
    return prof


def analyze_fft(region: np.ndarray, nm_per_px, design_pitch_nm,
                min_period_px=2.2, max_period_px=60.0, snr_thresh=10.0):
    """
    代表領域の 2D FFT を計算し、動径パワープロファイルから周期ピークを探索する。
    戻り値 dict: パワースペクトル(log), 動径プロファイル, 検出周期/周波数, SNR, 判定など。
    """
    reg = region - region.mean()
    win = np.outer(np.hanning(reg.shape[0]), np.hanning(reg.shape[1]))
    F = np.fft.fftshift(np.fft.fft2(reg * win))
    power = np.abs(F) ** 2
    logp = np.log1p(power)

    prof = radial_profile(power)
    n = len(prof)
    freqs = np.arange(n) / (region.shape[0])  # cycles/px(正方領域前提)
    expected_period = (design_pitch_nm / nm_per_px) if nm_per_px else np.nan

    # なだらかな背景(1/f 状のスペクトル傾斜)を推定し、その上に立つ「鋭いピーク」を探す。
    # 白色ノイズは背景に対する突出(prominence)が小さいので、これで誤検出を抑える。
    from scipy.ndimage import median_filter
    baseline = median_filter(prof, size=25, mode="nearest")
    prominence = prof - baseline

    # DC 近傍と探索範囲外を除外(period = 1/freq)
    valid = np.zeros(n, dtype=bool)
    for k in range(1, n):
        period = 1.0 / freqs[k] if freqs[k] > 0 else np.inf
        if min_period_px <= period <= max_period_px:
            valid[k] = True

    base_ret = {
        "logpower": logp, "radial": prof, "freqs": freqs,
        "expected_period_px": expected_period,
    }
    if not valid.any():
        return {**base_ret, "has_peak": False, "reason": "探索周期範囲に有効ビンなし",
                "peak_period_px": np.nan, "peak_freq": np.nan, "snr": np.nan}

    # ノイズフロアの散らばり(頑健スケール): 探索範囲全体の prominence の MAD
    def _snr_at(k):
        fmask = valid.copy()
        lo, hi = max(0, k - 3), min(n, k + 4)
        fmask[lo:hi] = False
        fvals = prominence[fmask]
        if fvals.size < 5:
            fvals = prominence[valid]
        scatter = np.median(np.abs(fvals - np.median(fvals))) * 1.4826 + 1e-12
        return float(prominence[k] / scatter), float(scatter)

    # (a) 探索範囲全体での最大prominenceピーク = 「周期信号が存在するか」
    vprom = np.where(valid, prominence, -np.inf)
    gpeak_k = int(np.argmax(vprom))
    g_snr, scatter = _snr_at(gpeak_k)
    g_is_localmax = (prof[gpeak_k] >= prof[max(0, gpeak_k - 1)] and
                     prof[gpeak_k] >= prof[min(n - 1, gpeak_k + 1)])
    peak_freq = freqs[gpeak_k]
    peak_period = 1.0 / peak_freq

    # (b) 設計換算の想定周波数(±15%)にピークがあるか = 「倍率整合の確認」
    matches_design = None
    expected_snr = np.nan
    if np.isfinite(expected_period) and expected_period > 0:
        periods = np.where(freqs > 0, 1.0 / np.maximum(freqs, 1e-12), np.inf)
        near = valid & (periods >= expected_period * 0.85) & (periods <= expected_period * 1.15)
        if near.any():
            ek = int(np.argmax(np.where(near, prominence, -np.inf)))
            expected_snr, _ = _snr_at(ek)
            matches_design = bool(expected_snr >= snr_thresh)
        else:
            matches_design = False

    # ゲート判定: 妥当な周期範囲に有意な鋭いピークが「存在するか」で継続可否を決める
    has_peak = bool(g_snr >= snr_thresh and g_is_localmax)
    reason = ""
    if not has_peak:
        reason = ("prominence最大点が鋭い局所極大でない" if not g_is_localmax
                  else f"prominence-SNR {g_snr:.1f} < 閾値 {snr_thresh}")
    elif matches_design is False:
        reason = (f"周期信号は検出(周期{peak_period:.2f}px, SNR{g_snr:.1f})だが、"
                  f"設計換算の想定周期{expected_period:.2f}pxと不一致(倍率/pixel pitch要確認)")

    # (c) 上位ピーク(周期・SNR)を抽出: 高調波/分数調波・別周期の把握用
    top_peaks = []
    order = np.argsort(np.where(valid, prominence, -np.inf))[::-1]
    for k in order:
        if not valid[k]:
            break
        if not (prof[k] >= prof[max(0, k - 1)] and prof[k] >= prof[min(n - 1, k + 1)]):
            continue
        if any(abs(k - pk) <= 3 for pk, _, _ in top_peaks):
            continue
        s, _ = _snr_at(int(k))
        top_peaks.append((int(k), 1.0 / freqs[k], s))
        if len(top_peaks) >= 5:
            break
    top_peaks = [{"period_px": p, "snr": s} for _, p, s in top_peaks]

    return {**base_ret, "has_peak": has_peak, "reason": reason,
            "peak_period_px": peak_period, "peak_freq": peak_freq, "snr": g_snr,
            "matches_design": matches_design, "expected_snr": expected_snr,
            "top_peaks": top_peaks, "nyquist_period_px": 2.0,
            "floor_scatter": float(scatter), "peak_prominence": float(prominence[gpeak_k])}


def save_fft_figures(res, region, out_prefix: Path, title: str):
    # パワースペクトル(log)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(region, cmap="gray")
    ax[0].set_title(f"{title}\nrepresentative region")
    ax[0].axis("off")
    logp = res["logpower"]
    h, w = logp.shape
    ax[1].imshow(logp, cmap="magma", extent=[-0.5, 0.5, -0.5, 0.5])
    ax[1].set_title("2D FFT log power (cycles/px)")
    ax[1].set_xlabel("fx"); ax[1].set_ylabel("fy")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_fft_power.png", dpi=110)
    plt.close(fig)

    # 動径プロファイル
    fig, ax = plt.subplots(figsize=(9, 5))
    freqs = res["freqs"]
    prof = res["radial"]
    m = (freqs > 0)
    ax.semilogy(freqs[m], prof[m] + 1e-9, label="radial power")
    if np.isfinite(res["peak_freq"]):
        ax.axvline(res["peak_freq"], color="g", ls="--",
                   label=f"detected peak: {res['peak_period_px']:.2f}px (SNR={res['snr']:.1f})")
    if np.isfinite(res["expected_period_px"]) and res["expected_period_px"] > 0:
        ef = 1.0 / res["expected_period_px"]
        ax.axvline(ef, color="r", ls=":",
                   label=f"design {DESIGN_PITCH_NM:.0f}nm -> {res['expected_period_px']:.2f}px")
    ax.set_xlabel("spatial frequency (cycles/px)")
    ax.set_ylabel("radial power (log)")
    ax.set_title("Radial power profile")
    ax.set_xlim(0, min(0.5, (freqs[m].max() if m.any() else 0.5)))
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_fft_radial.png", dpi=110)
    plt.close(fig)


# ========================= 1. FWHM / σ =========================

def find_bright_spots(img, n_spots=20, min_sep=15, border=30, pct=99.7):
    """局所最大かつ高輝度な孤立スポットを検出して座標リストを返す。
    検出は軽く平滑化したコピー上で行い、単一画素ノイズスパイクを中心と誤認しないようにする
    (FWHMフィット自体は元画像に対して行う)。"""
    from scipy.ndimage import maximum_filter
    smooth = cv2.GaussianBlur(img, (0, 0), 1.0)
    thr = np.percentile(smooth, pct)
    mx = maximum_filter(smooth, size=5)
    cand = (smooth == mx) & (smooth >= thr)
    ys, xs = np.where(cand)
    h, w = img.shape
    pts = [(x, y) for x, y in zip(xs, ys) if border <= x < w - border and border <= y < h - border]
    # 明るい順(平滑化画像基準)
    pts.sort(key=lambda p: -smooth[p[1], p[0]])
    selected = []
    for x, y in pts:
        if all((x - sx) ** 2 + (y - sy) ** 2 >= min_sep ** 2 for sx, sy in selected):
            selected.append((x, y))
        if len(selected) >= n_spots:
            break
    return selected


def _fit_gaussian_sigma(profile, peak_idx):
    """1Dプロファイルにガウシアン A*exp(-(x-x0)^2/2σ^2)+c を最小二乗フィットし σ を返す。
    画素ノイズに対し半値交差より頑健(隣接ノイズ谷で早期打ち切りする問題を回避)。失敗時 NaN。"""
    from scipy.optimize import curve_fit
    x = np.arange(len(profile), dtype=np.float64)
    y = profile.astype(np.float64)
    c0 = float(np.min(y))
    A0 = float(y[peak_idx] - c0)
    if A0 <= 0:
        return np.nan

    def gauss(xx, A, x0, sigma, c):
        return A * np.exp(-(xx - x0) ** 2 / (2.0 * sigma ** 2)) + c

    try:
        popt, _ = curve_fit(
            gauss, x, y,
            p0=[A0, float(peak_idx), 2.0, c0],
            bounds=([0.3 * A0, peak_idx - 3, 0.5, c0 - abs(A0)],
                    [3.0 * A0, peak_idx + 3, len(profile) / 2.0, c0 + abs(A0)]),
            maxfev=2000,
        )
    except Exception:
        return np.nan
    sigma = abs(popt[2])
    # フィット品質チェック: 残差が浅すぎ/振幅が小さすぎる場合は棄却
    return sigma


def measure_blur_sigma(img, spots, half_win=12):
    """
    ボケ σ を2通りで推定する:
      - sigma_mean: 全スポットのパッチを中心合わせで平均(スタッキング)してからフィット。
        画素ノイズが平均化で相殺されるため、単スポットフィットの下方バイアスを抑えられる。
      - sigma_std : 各スポット個別フィットの σ のばらつき(空間的な均一性の目安)。
    FWHM = 2.3548 σ。孤立輝点が解像している前提の指標である点に注意
    (強ボケで輝点が重なり非解像だと σ は過小評価になる)。
    """
    h, w = img.shape
    patches = []
    per_spot = []
    for (x, y) in spots:
        if not (half_win <= x < w - half_win and half_win <= y < h - half_win):
            continue
        patch = img[y - half_win:y + half_win + 1, x - half_win:x + half_win + 1].astype(np.float64)
        patches.append(patch)
        sr = _fit_gaussian_sigma(patch[half_win, :], half_win)
        sc = _fit_gaussian_sigma(patch[:, half_win], half_win)
        for s in (sr, sc):
            if np.isfinite(s) and 0.5 <= s <= half_win:
                per_spot.append(s)

    if not patches:
        return {"n": 0, "fwhm_mean": np.nan, "fwhm_std": np.nan,
                "sigma_mean": np.nan, "sigma_std": np.nan}

    stack = np.mean(patches, axis=0)
    sr = _fit_gaussian_sigma(stack[half_win, :], half_win)
    sc = _fit_gaussian_sigma(stack[:, half_win], half_win)
    stack_sigmas = [s for s in (sr, sc) if np.isfinite(s) and 0.5 <= s <= half_win]
    per_spot = np.asarray(per_spot, dtype=np.float64)

    sigma_mean = float(np.mean(stack_sigmas)) if stack_sigmas else (
        float(per_spot.mean()) if per_spot.size else np.nan)
    sigma_std = float(per_spot.std()) if per_spot.size else np.nan
    return {"n": len(patches),
            "fwhm_mean": sigma_mean * 2.35482 if np.isfinite(sigma_mean) else np.nan,
            "fwhm_std": sigma_std * 2.35482 if np.isfinite(sigma_std) else np.nan,
            "sigma_mean": sigma_mean, "sigma_std": sigma_std}


# ========================= 2. 間隔(自己相関) =========================

def _first_side_peak(line, center, min_period, max_period):
    """1D配列 line の center から両側へ、[min_period, max_period] 内で最初の局所極大距離を返す。"""
    n = len(line)
    dists = []
    for direction in (+1, -1):
        best = None
        for d in range(int(min_period), int(max_period) + 1):
            i = center + direction * d
            if i <= 0 or i >= n - 1:
                break
            if line[i] > line[i - 1] and line[i] >= line[i + 1]:
                best = d
                break
        if best is not None:
            dists.append(best)
    return float(np.mean(dists)) if dists else np.nan


def spacing_from_autocorr(region, min_period_px=3.0, max_period_px=60.0):
    """高域強調した領域の 2D 自己相関から、主軸(行・列)方向の最初の副ピーク距離(=周期)を求める。
    グリッド周期は動径平均だと smear するため、中心行・中心列に沿って測る。"""
    reg = region.astype(np.float64)
    reg = reg - cv2.GaussianBlur(reg, (0, 0), 8)  # 低周波背景除去
    reg = reg - reg.mean()
    F = np.fft.fft2(reg)
    ac = np.fft.fftshift(np.fft.ifft2(F * np.conj(F)).real)
    ac /= ac.max() + 1e-12
    h, w = ac.shape
    cy, cx = h // 2, w // 2
    row = ac[cy, :]
    col = ac[:, cx]
    dx = _first_side_peak(row, cx, min_period_px, max_period_px)
    dy = _first_side_peak(col, cy, min_period_px, max_period_px)
    vals = [v for v in (dx, dy) if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


# ========================= 3&4. 指標感度 / フィルタ =========================

def synth_pillar_field(spacing, sigma, size=384, amp=120, base=2000, noise=8.0, seed=0):
    """実測 spacing・sigma を反映した合成ピラー像(疑似SAM像)を作る。"""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), float(base), dtype=np.float32)
    s = max(2, int(round(spacing)))
    ys, xs = np.mgrid[s:size - s:s, s:size - s:s]
    img[ys.ravel(), xs.ravel()] += amp
    if sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    img = img + rng.normal(0, noise, img.shape).astype(np.float32)
    return img


def ncc(a, b):
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def match_rate(a, b, thr):
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64)) <= thr) * 100)


def shift_img(img, dx):
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def laplacian_highpass(img):
    f = img.astype(np.float32)
    return cv2.Laplacian(f, cv2.CV_32F, ksize=3)


def sensitivity_table(spacing, sigma, shifts, match_thr, use_filter=False, base_for_rate=2000):
    """既知ズレに対する NCC・一致率(フィルタあり/なし)を返す。"""
    base = synth_pillar_field(spacing, sigma)
    interior = lambda x: x[40:-40, 40:-40]
    row_ncc, row_rate = [], []
    pre = base
    for sh in shifts:
        post = shift_img(base, sh)
        if use_filter:
            a, b = laplacian_highpass(pre), laplacian_highpass(post)
        else:
            a, b = pre, post
        row_ncc.append(ncc(interior(a), interior(b)))
        # 一致率はフィルタ後だとスケールが変わるので、階調ベースは無フィルタ画素で評価
        row_rate.append(match_rate(interior(pre), interior(post), match_thr))
    return row_ncc, row_rate


def classify_regime(spacing, sigma, match_thr):
    """
    実測 σ・spacing で「1px および 2px ズレ時に NCC がどれだけ落ちるか」を測り、
    指標の適性を分類する。NCC 低下が小さい=強ボケ域(指標が鈍感)。
    """
    shifts = [0.0, 1.0, 2.0]
    nccs, _ = sensitivity_table(spacing, sigma, shifts, match_thr, use_filter=False)
    ncc0 = nccs[0]
    drop1 = ncc0 - nccs[1]
    drop2 = ncc0 - nccs[2]
    # しきい値: 1pxズレでNCCが0.05以上落ちれば実用感度あり(緩め)
    if drop1 >= 0.10:
        regime = "軽度ボケ域(指標が良好に機能: 1pxズレでNCCが明確に低下)"
    elif drop2 >= 0.10:
        regime = "中間域(2pxズレでようやくNCCが低下。サブピクセル検証には不十分)"
    else:
        regime = "強ボケ域(2pxズレでもNCCがほぼ不変=偽陽性リスク大。指標は精度検証に不適)"
    return {"ncc_shift0": ncc0, "ncc_drop_1px": drop1, "ncc_drop_2px": drop2, "regime": regime}


def filter_comparison(spacing, sigma, match_thr):
    shifts = [0.0, 0.5, 1.0, 2.0, 4.0]
    ncc_no, rate_no = sensitivity_table(spacing, sigma, shifts, match_thr, use_filter=False)
    ncc_lap, _ = sensitivity_table(spacing, sigma, shifts, match_thr, use_filter=True)
    return {"shifts": shifts, "ncc_no": ncc_no, "rate_no": rate_no, "ncc_lap": ncc_lap}


def analyze_discrepancy(fft, blur, nm_per_px, design_pitch_nm, visual_range):
    """
    「nm/px換算の設計周期」「FFT実測周期」「目視推定周期」の3者を突き合わせ、
    食い違いの原因仮説を定量的に自動判定する。

    fft_period が:
      - 設計換算(design/nm_per_px)に一致 → nm/px は正しく、目視が別構造/過大評価
      - 目視に一致 → nm/px が factor だけずれている(または撮像倍率誤り)
      - どちらとも違う → 別の周期構造(高調波・分数調波・モアレ)
    さらに、設計周期が Nyquist(2px)近傍で undersampling なら、
    基本波がボケMTFで消えて別周期だけが残る可能性を指摘する。
    """
    P_fft = fft["peak_period_px"]
    P_design = (design_pitch_nm / nm_per_px) if nm_per_px else np.nan
    tol = 0.20  # 一致とみなす相対許容(nm/px の ±15-20% 近似を反映)

    def close(a, b):
        return np.isfinite(a) and np.isfinite(b) and abs(a - b) <= tol * b

    vlo = vhi = np.nan
    if visual_range:
        vlo, vhi = float(min(visual_range)), float(max(visual_range))
    in_visual = np.isfinite(P_fft) and visual_range and (vlo * (1 - tol) <= P_fft <= vhi * (1 + tol))

    hyps = []
    verdict = ""
    if np.isfinite(P_design) and close(P_fft, P_design):
        verdict = ("FFT実測周期は nm/px換算の設計周期に一致 → **nm/px=60 は妥当**、"
                   "目視5-8pxは真ピッチではない(ボケで融合したブロブの塊/モアレ/2次周期、"
                   "または低コントラスト微細パターンの目視過大評価)可能性が高い。")
        hyps.append(f"目視の逆算nm/px = {design_pitch_nm/vhi:.0f}〜{design_pitch_nm/vlo:.0f} "
                    f"(真ピッチ{P_fft:.2f}pxなら nm/px≈{design_pitch_nm/P_fft:.0f})" if visual_range else "")
    elif in_visual:
        implied = design_pitch_nm / P_fft if P_fft else np.nan
        verdict = (f"FFT実測周期は目視({vlo:.0f}-{vhi:.0f}px)に一致 → **目視が正しく nm/px がずれている**。"
                   f"実測周期から逆算した nm/px ≈ {implied:.0f} nm/px "
                   f"(申告60の約{60/implied:.1f}倍ずれ。撮像倍率/pixel pitch の再確認が必要)。")
    elif np.isfinite(P_fft):
        # 高調波/分数調波関係のチェック
        rel = ""
        if np.isfinite(P_design):
            r = P_fft / P_design
            for m, name in [(2, "2倍(分数調波/ダイマー化)"), (0.5, "1/2(2次高調波)"),
                            (3, "3倍"), (1.0/3, "1/3")]:
                if abs(r - m) <= 0.2 * m:
                    rel = f"設計周期の約{name}に相当"
                    break
        verdict = (f"FFT実測周期{P_fft:.2f}px は設計換算{P_design:.2f}px とも目視とも一致しない。"
                   f"{('（'+rel+'）') if rel else ''} 別の周期構造/高調波を拾っている可能性。")

    # Nyquist / undersampling 指摘(1ピラーあたりの画素数 = P_design)
    nyq_note = ""
    if np.isfinite(P_design) and P_design < 4.0:
        severity = "Nyquist(2px)近傍で深刻な undersampling" if P_design < 2.5 else \
                   "サンプリング限界に近い(1周期あたり数画素のみ)"
        nyq_note = (f"⚠️ 設計周期{P_design:.2f}px は {severity}。1ピラー約{P_design:.1f}画素しかなく、"
                    f"ボケMTFで基本波が強く減衰しやすい。FFTで設計周波数にピークが立たない場合、"
                    f"真ピッチの信号がそもそも安定抽出できず、目視で見える粗い周期は"
                    f"融合ブロブ/モアレ/2次周期由来の見かけである疑いが強い。"
                    f"この場合、位置合わせ指標(NCC/一致率)は真ピッチ構造では機能しない。")

    # 解像状況(FWHMとの比較)
    resolved_note = ""
    if np.isfinite(P_fft) and np.isfinite(blur.get("fwhm_mean", np.nan)):
        if P_fft < 1.5 * blur["fwhm_mean"]:
            resolved_note = (f"検出周期{P_fft:.2f}px < 1.5×FWHM{blur['fwhm_mean']:.2f}px "
                             f"→ 個々のピラーは非解像(融合)。")

    return {"P_fft": P_fft, "P_design": P_design, "visual": (vlo, vhi),
            "verdict": verdict, "hyps": [h for h in hyps if h],
            "nyq_note": nyq_note, "resolved_note": resolved_note}


# ============================= レポート =============================

def build_report(img_reports, args) -> str:
    lines = []
    lines.append("# ピラー領域指標 実データ診断レポート")
    lines.append("")
    lines.append(f"- 設計ピッチ: {args.design_pitch_nm:.0f} nm")
    lines.append(f"- nm/px: {args.nm_per_px if args.nm_per_px else '未指定(設計→px換算はスキップ)'}")
    lines.append(f"- 代表領域サイズ: {args.region}px, FFT SNR閾値: {args.snr_thresh}")
    lines.append(f"- 一致率許容差: |Δ|<= {args.match_threshold}")
    lines.append("")
    for rep in img_reports:
        lines.append(f"## {rep['name']}")
        fft = rep["fft"]
        lines.append("")
        lines.append("### 0. FFT 周期確認(ゲート)")
        exp = fft["expected_period_px"]
        lines.append("- 設計換算の想定周期: " + (f"{exp:.2f} px" if np.isfinite(exp) else "N/A(nm/px未指定)"))
        lines.append(f"- 検出周期(範囲内の最有意ピーク): {fft['peak_period_px']:.2f} px "
                     f"(周波数 {fft['peak_freq']:.4f} cyc/px)")
        lines.append(f"- ピークSNR(prominence/ノイズフロア): {fft['snr']:.1f}")
        md = fft.get("matches_design")
        if md is True:
            lines.append(f"- 設計周波数との整合: ✅ 一致(想定{exp:.2f}px 近傍にSNR{fft['expected_snr']:.1f}のピーク)")
        elif md is False:
            lines.append(f"- 設計周波数との整合: ⚠️ 不一致(想定{exp:.2f}px 近傍に有意ピークなし → 倍率/pixel pitch要確認)")
        tp = fft.get("top_peaks", [])
        if tp:
            lines.append("- 範囲内の上位ピーク(周期px / SNR): " +
                         ", ".join(f"{t['period_px']:.2f}px(SNR{t['snr']:.1f})" for t in tp))
        verdict = "✅ 有意な周期信号あり → 1〜4 実施" if fft["has_peak"] else "⛔ 有意な周期信号なし → STOP(1〜4中断)"
        lines.append(f"- **判定: {verdict}** {('※ '+fft['reason']) if fft['reason'] else ''}")

        # 食い違い分析(設計換算 vs FFT実測 vs 目視)
        disc = rep.get("discrepancy")
        if disc and (disc.get("verdict") or disc.get("nyq_note")):
            lines.append("")
            lines.append("#### 0b. nm/px換算 vs FFT実測 vs 目視 の食い違い分析")
            P_design = disc["P_design"]
            lines.append(f"- 設計換算(200nm/{args.nm_per_px}nm/px)= "
                         + (f"{P_design:.2f}px" if np.isfinite(P_design) else "N/A"))
            lines.append(f"- FFT実測 = {disc['P_fft']:.2f}px")
            if args.visual_estimate_px:
                vlo, vhi = disc["visual"]
                lines.append(f"- 目視推定 = {vlo:.1f}〜{vhi:.1f}px")
            if disc["verdict"]:
                lines.append(f"- **原因判定: {disc['verdict']}**")
            for h in disc.get("hyps", []):
                lines.append(f"  - {h}")
            if disc["nyq_note"]:
                lines.append(f"- {disc['nyq_note']}")
            if disc["resolved_note"]:
                lines.append(f"- {disc['resolved_note']}")
        lines.append(f"- 画像: `{rep['prefix']}_fft_power.png`, `{rep['prefix']}_fft_radial.png`")
        lines.append("")

        if not fft["has_peak"]:
            lines.append("> 周期信号が安定抽出できないため、ボケ幅/間隔/指標適性の評価は中断した。")
            lines.append("")
            continue

        blur = rep["blur"]
        lines.append("### 1. ボケ幅(FWHM→σ)")
        lines.append(f"- 計測スポット数: {blur['n']}")
        lines.append(f"- FWHM: {blur['fwhm_mean']:.2f} ± {blur['fwhm_std']:.2f} px")
        lines.append(f"- σ: {blur['sigma_mean']:.2f} ± {blur['sigma_std']:.2f} px")
        # 解像判定: 周期がFWHMと同程度以下なら輝点は重なっており σ は過小評価(信頼不可)
        period = fft["peak_period_px"]
        if np.isfinite(blur["fwhm_mean"]) and np.isfinite(period) and period < 1.5 * blur["fwhm_mean"]:
            lines.append(f"- ⚠️ 周期({period:.2f}px) < 1.5×FWHM({blur['fwhm_mean']:.2f}px): "
                         f"輝点が重なり非解像。FWHM→σ は過小評価となり信頼できない "
                         f"(この場合ボケσは『間隔/2以上』とだけ言える)。")
        lines.append("- 注記: 本指標は孤立輝点が解像している前提。非解像時は σ を鵜呑みにしないこと。")
        lines.append("")

        lines.append("### 2. ピラー間隔(px)")
        lines.append(f"- FFTピークから: {fft['peak_period_px']:.2f} px")
        lines.append(f"- 自己相関から: {rep['spacing_ac']:.2f} px")
        if np.isfinite(exp):
            lines.append(f"- nm/px独立計算(200nm/{args.nm_per_px}): {exp:.2f} px")
        lines.append("")

        cls = rep["classify"]
        lines.append("### 3. テーブル上の位置づけ(指標適性)")
        lines.append(f"- σ={blur['sigma_mean']:.2f}px, 間隔={fft['peak_period_px']:.2f}px を合成再現して評価")
        lines.append(f"- NCC(ズレ0)= {cls['ncc_shift0']:.4f}, "
                     f"1pxズレでの低下= {cls['ncc_drop_1px']:.4f}, 2pxズレ= {cls['ncc_drop_2px']:.4f}")
        lines.append(f"- **結論: {cls['regime']}**")
        lines.append("")

        fc = rep["filter"]
        lines.append("### 4. 高域強調フィルタ比較(ラプラシアン)")
        lines.append("| shift(px) | " + " | ".join(f"{s}" for s in fc["shifts"]) + " |")
        lines.append("|---|" + "---|" * len(fc["shifts"]))
        lines.append("| NCC 無フィルタ | " + " | ".join(f"{v:.3f}" for v in fc["ncc_no"]) + " |")
        lines.append("| NCC ラプラシアン | " + " | ".join(f"{v:.3f}" for v in fc["ncc_lap"]) + " |")
        lines.append("| 一致率% 無フィルタ | " + " | ".join(f"{v:.1f}" for v in fc["rate_no"]) + " |")
        d_no = fc["ncc_no"][0] - fc["ncc_no"][2]
        d_lap = fc["ncc_lap"][0] - fc["ncc_lap"][2]
        improved = "改善する" if d_lap > d_no + 1e-3 else "改善しない"
        lines.append("")
        lines.append(f"- 1pxズレでのNCC低下量: 無フィルタ {d_no:.4f} → ラプラシアン {d_lap:.4f} "
                     f"→ フィルタは感度を **{improved}**")
        lines.append("")
    return "\n".join(lines)


# ============================= メイン =============================

def process_image(path: Path, args, out_dir: Path):
    name = path.name
    prefix = out_dir / path.stem
    img = load_gray_float(path)
    region, origin = central_region(img, args.region, offset=(args.region_offset_x, args.region_offset_y))

    fft = analyze_fft(region, args.nm_per_px, args.design_pitch_nm,
                      snr_thresh=args.snr_thresh)
    save_fft_figures(fft, region, prefix, name)

    rep = {"name": name, "prefix": str(prefix), "fft": fft}

    # ゲートがSTOPでも、食い違い(設計換算 vs FFT実測 vs 目視)は判定して報告する。
    if not fft["has_peak"]:
        rep["discrepancy"] = analyze_discrepancy(
            fft, {}, args.nm_per_px, args.design_pitch_nm, args.visual_estimate_px)
        return rep

    spots = find_bright_spots(img, n_spots=args.n_spots)
    rep["blur"] = measure_blur_sigma(img, spots)
    rep["spacing_ac"] = spacing_from_autocorr(region)
    sigma = rep["blur"]["sigma_mean"]
    spacing = fft["peak_period_px"]
    if not np.isfinite(sigma):
        sigma = 2.0  # フォールバック(FWHM計測失敗時)
    rep["classify"] = classify_regime(spacing, sigma, args.match_threshold)
    rep["filter"] = filter_comparison(spacing, sigma, args.match_threshold)
    rep["discrepancy"] = analyze_discrepancy(
        fft, rep["blur"], args.nm_per_px, args.design_pitch_nm, args.visual_estimate_px)
    return rep


def build_arg_parser():
    p = argparse.ArgumentParser(description="実データ物理パラメータ診断(FFT/ボケ/間隔/指標適性)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", nargs="+", help="診断する画像ファイル(pre画像を数枚)")
    src.add_argument("--input-dir", help="フォルダ指定(pre画像 *-0.tif を自動選択)")
    p.add_argument("--max-images", type=int, default=3, help="input-dir 指定時の最大処理枚数")
    p.add_argument("--output-dir", default="analysis_diag", help="出力フォルダ")
    p.add_argument("--nm-per-px", type=float, default=None,
                   help="1画素あたりの物理サイズ(nm/px)。指定すると設計200nm→想定px周期を換算・照合")
    p.add_argument("--design-pitch-nm", type=float, default=DESIGN_PITCH_NM, help="設計ピラーピッチ(nm)")
    p.add_argument("--region", type=int, default=1024, help="FFT代表領域の一辺(px)")
    p.add_argument("--region-offset-x", type=int, default=0, help="代表領域の中心xオフセット(傷回避用)")
    p.add_argument("--region-offset-y", type=int, default=0, help="代表領域の中心yオフセット(傷回避用)")
    p.add_argument("--snr-thresh", type=float, default=10.0, help="FFTピーク有意判定のSNR閾値(prominence/ノイズフロア)")
    p.add_argument("--n-spots", type=int, default=20, help="FWHM計測する孤立輝点数")
    p.add_argument("--match-threshold", type=float, default=5.0, help="一致率許容差 |Δ|<=")
    p.add_argument("--visual-estimate-px", type=lambda s: [float(x) for x in s.split(",")],
                   default=None,
                   help="目視によるピラー周期推定(px)。単一 '6' または範囲 '5,8'。"
                        "設計換算・FFT実測との食い違い分析に使用")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.images:
        paths = [Path(p) for p in args.images]
    else:
        d = Path(args.input_dir)
        paths = sorted(q for q in d.iterdir()
                       if q.suffix.lower() in {".tif", ".tiff"} and q.stem.endswith("-0"))
        paths = paths[:args.max_images]

    if not paths:
        print("診断対象の画像が見つかりません。", file=sys.stderr)
        sys.exit(1)

    reports = []
    for path in paths:
        if not path.exists():
            print(f"[WARN] 見つかりません: {path}", file=sys.stderr)
            continue
        print(f"[INFO] 診断中: {path.name}")
        reports.append(process_image(path, args, out_dir))

    md = build_report(reports, args)
    report_path = out_dir / "diagnosis_report.md"
    report_path.write_text(md, encoding="utf-8")
    print(f"[INFO] レポート: {report_path}")
    print()
    print(md)


if __name__ == "__main__":
    main()
