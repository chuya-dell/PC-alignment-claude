"""
バッチ画像位置合わせスクリプト (df データセット汎用版)

フォルダ内の全画像を「条件-セット-連番」規則で pre(連番0)/post(連番1以降)に
自動グルーピングし、デザインナイフ傷(スクラッチ)をランドマークとして
phaseCorrelate初期値 -> findTransformECC (MOTION_EUCLIDEAN) で位置合わせを行う。

傷検出には PC-alignment-anti/registration.py の detect_grooves
(プロファイル最小値検出、移動平均窓で平滑化)の考え方を流用する。

位置合わせ後、pre/post間で共通の有効領域のみをクロップして保存する。
精度検証は2系統:
  (1) 傷付近の NCC・一致率
  (2) ピラー領域の自動マルチパッチ評価
      傷検出帯を除いた画像全体をグリッド状に分割し、各セルからパッチを切り出して
      パッチごとに NCC・一致率(|Δ|<=5階調)を算出。集計値(平均・最小・標準偏差)
      をサマリーに記録し、パッチ明細は別CSVに出力する。パッチ間ばらつき(std)が
      大きい場合、場所によって位置合わせがズレている可能性を示す。

留意点: 像がぼやけている(ピラーが1本ずつ解像していない)場合、NCC/一致率は
画素値相関ベースであるため「ズレていても相関が高く出る」偽陽性リスクがある
(合成検証で確認済み。ボケσが大きいとズレ量に対し指標がほぼ変化しなくなる)。
また周期構造のためズレ量がピラー1周期に達すると相関が復帰するエイリアシングにも注意。
"""

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import tifffile

logger = logging.getLogger("batch_register")

IMAGE_SUFFIXES = {".tif", ".tiff"}

# 傷(スクラッチ)ランドマーク検出のデフォルト探索帯域。
# registration.py の detect_grooves を参考にした値であり、CLI引数で上書き可能。
DEFAULT_V_COL_RANGE = (0, 300)
DEFAULT_V_ROW_BAND = (500, 1500)
DEFAULT_H_ROW_RANGE = (0, 300)
DEFAULT_H_COL_BAND = (1000, 1900)


# ===================== ファイルグルーピング =====================

@dataclass
class GroupKey:
    condition: str
    set_id: str

    @property
    def label(self) -> str:
        return f"{self.condition}-{self.set_id}"


def parse_filename(stem: str) -> Optional[tuple]:
    """'条件-セット-連番' 形式のファイル名を分解する。連番は末尾の整数。"""
    parts = stem.rsplit("-", 2)
    if len(parts) != 3:
        return None
    condition, set_id, seq_str = parts
    if not re.fullmatch(r"\d+", seq_str):
        return None
    return condition, set_id, int(seq_str)


def group_files(input_dir: Path) -> dict:
    """入力フォルダ内のファイルを (条件, セット) -> {連番: Path} にグルーピング。"""
    groups: dict = {}
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parsed = parse_filename(path.stem)
        if parsed is None:
            logger.warning("ファイル名規則に一致しないためスキップ: %s", path.name)
            continue
        condition, set_id, seq = parsed
        key = GroupKey(condition, set_id)
        groups.setdefault(key.label, {"key": key, "files": {}})
        groups[key.label]["files"][seq] = path
    return groups


# ===================== 傷(スクラッチ)ランドマーク検出 =====================

def _subpixel_min(profile: np.ndarray, idx: int) -> float:
    if 0 < idx < len(profile) - 1:
        y0, y1, y2 = profile[idx - 1], profile[idx], profile[idx + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            return idx + 0.5 * (y0 - y2) / denom
    return float(idx)


def _smooth_rows(m: np.ndarray, k: int) -> np.ndarray:
    kern = np.ones(k) / k
    return np.apply_along_axis(lambda v: np.convolve(v, kern, mode="same"), 1, m)


def _trace_line(band: np.ndarray, line_axis: int, polarity: str,
                edge_margin: int, smooth_window: int, tol: float) -> dict:
    """
    帯域を「線が走る方向 × 探索方向」に整え、各スライスで極値位置を追跡して
    傷ランドマーク線を頑健に同定する(帯域平均法と違い、波打ち・傾きに強い)。

    各スライスの極値位置に2次多項式をフィットして波打ち/傾きを吸収し、
    フィットからの残差が tol 以内に収まるスライスの割合(inlier率)を信頼度とする。
    ランダムなノイズやピラー模様は極値位置が散らばるため inlier率が低くなり、
    真の連続した傷だけが高い inlier率を示す。

    戻り値 dict:
      position   : 帯域中央での線位置(探索方向, サブ帯域オフセット込みの相対値)
      confidence : inlier率(0..1)
      contrast_snr: 線コントラストのノイズ比(帯域外の弱い勾配で誤検出しない担保)
      waviness   : フィット位置の振れ幅(px, 波打ち量の目安)
      polarity   : 採用した極性
    """
    m = band if line_axis == 0 else band.T  # 常に「行方向に線が走る」向きへ
    N, M = m.shape
    ms = _smooth_rows(m, smooth_window)
    margin = min(edge_margin, (M - 1) // 2)
    lo, hi = margin, M - margin
    sub = ms[:, lo:hi]
    row_med = np.median(sub, axis=1)

    if polarity == "bright":
        pos = lo + np.argmax(sub, axis=1).astype(np.float64)
        contrast = np.max(sub, axis=1) - row_med
    else:
        pos = lo + np.argmin(sub, axis=1).astype(np.float64)
        contrast = row_med - np.min(sub, axis=1)

    idx = np.arange(N)
    good = contrast > np.median(contrast)  # コントラストのある行だけで線形状を推定
    result = {"position": float(np.median(pos)), "confidence": 0.0,
              "contrast_snr": 0.0, "waviness": float(np.std(pos)), "polarity": polarity}
    if good.sum() < max(5, int(N * 0.1)):
        return result

    coeffs = np.polyfit(idx[good], pos[good], 2)
    fit = np.polyval(coeffs, idx)
    inlier = np.abs(pos - fit) <= tol
    noise = np.median(np.abs(sub - row_med[:, None])) * 1.4826 + 1e-6
    med_contrast = np.median(contrast[inlier]) if inlier.any() else 0.0

    result["confidence"] = float(np.mean(inlier))
    result["contrast_snr"] = float(med_contrast / noise)
    result["waviness"] = float(np.std(fit))
    result["position"] = float(np.polyval(coeffs, N / 2.0))
    return result


def _detect_line(band: np.ndarray, line_axis: int, polarity: str,
                 edge_margin: int, smooth_window: int, tol: float) -> dict:
    """polarity=auto なら明/暗を両方追跡し inlier率(同点はコントラスト)の高い方を採用。"""
    if polarity in ("bright", "dark"):
        return _trace_line(band, line_axis, polarity, edge_margin, smooth_window, tol)
    rb = _trace_line(band, line_axis, "bright", edge_margin, smooth_window, tol)
    rd = _trace_line(band, line_axis, "dark", edge_margin, smooth_window, tol)
    kb = (rb["confidence"], rb["contrast_snr"])
    kd = (rd["confidence"], rd["contrast_snr"])
    return rb if kb >= kd else rd


def detect_scratch_landmark(
    img: np.ndarray,
    v_col_range=DEFAULT_V_COL_RANGE,
    v_row_band=DEFAULT_V_ROW_BAND,
    h_row_range=DEFAULT_H_ROW_RANGE,
    h_col_band=DEFAULT_H_COL_BAND,
    smooth_window: int = 15,
    edge_margin: int = 30,
    polarity: str = "auto",
    trace_tol: float = 6.0,
) -> dict:
    """
    デザインナイフ傷(縦線・横線)のランドマーク座標を検出する。
    detect_grooves (registration.py) の考え方を踏襲しつつ、帯域平均ではなく
    各スライスの極値位置を追跡する頑健な方式(波打ち・傾きに強い)を用いる。
    polarity で暗い傷(明視野)/明るい傷(暗視野散乱像)/自動判別を切り替える。
    信頼度は inlier率(0..1)。
    """
    h, w = img.shape[:2]
    r0, r1 = max(0, v_row_band[0]), min(h, v_row_band[1])
    c0, c1 = max(0, v_col_range[0]), min(w, v_col_range[1])
    v_band = img[r0:r1, c0:c1].astype(np.float64)
    v = _detect_line(v_band, line_axis=0, polarity=polarity,
                     edge_margin=edge_margin, smooth_window=smooth_window, tol=trace_tol)
    scratch_x = c0 + v["position"]

    rr0, rr1 = max(0, h_row_range[0]), min(h, h_row_range[1])
    cc0, cc1 = max(0, h_col_band[0]), min(w, h_col_band[1])
    h_band = img[rr0:rr1, cc0:cc1].astype(np.float64)
    hln = _detect_line(h_band, line_axis=1, polarity=polarity,
                       edge_margin=edge_margin, smooth_window=smooth_window, tol=trace_tol)
    scratch_y = rr0 + hln["position"]

    return {"x": scratch_x, "y": scratch_y,
            "confidence_x": v["confidence"], "confidence_y": hln["confidence"],
            "contrast_x": v["contrast_snr"], "contrast_y": hln["contrast_snr"],
            "polarity_x": v["polarity"], "polarity_y": hln["polarity"],
            "waviness_x": v["waviness"], "waviness_y": hln["waviness"]}


# ===================== 位置合わせ (phaseCorrelate -> ECC) =====================

@dataclass
class RegistrationResult:
    status: str  # "ok" / "scratch_not_detected" / "scratch_crop_too_small" / ...
    scratch_detected_pre: bool = False
    scratch_detected_post: bool = False
    dx: float = float("nan")
    dy: float = float("nan")
    rotation_deg: float = float("nan")
    warp_matrix: Optional[np.ndarray] = None
    scratch_xy_pre: Optional[tuple] = None
    scratch_xy_post: Optional[tuple] = None
    confidence_pre: float = float("nan")   # min(confidence_x, confidence_y) of pre
    confidence_post: float = float("nan")
    polarity: str = ""
    mode: str = ""  # "ecc" / "translation_fallback"


def register_pair(
    pre_img: np.ndarray,
    post_img: np.ndarray,
    scratch_kwargs: dict,
    min_confidence: float,
    crop_margin: int,
    ecc_eps: float,
    ecc_iterations: int,
    min_contrast: float = 2.0,
) -> RegistrationResult:
    h, w = pre_img.shape[:2]

    pre_lm = detect_scratch_landmark(pre_img, **scratch_kwargs)
    post_lm = detect_scratch_landmark(post_img, **scratch_kwargs)

    def _ok(lm):
        return (lm["confidence_x"] >= min_confidence and lm["confidence_y"] >= min_confidence
                and lm["contrast_x"] >= min_contrast and lm["contrast_y"] >= min_contrast)

    pre_ok, post_ok = _ok(pre_lm), _ok(post_lm)

    result = RegistrationResult(
        status="ok",
        scratch_detected_pre=pre_ok,
        scratch_detected_post=post_ok,
        scratch_xy_pre=(pre_lm["x"], pre_lm["y"]),
        scratch_xy_post=(post_lm["x"], post_lm["y"]),
        confidence_pre=min(pre_lm["confidence_x"], pre_lm["confidence_y"]),
        confidence_post=min(post_lm["confidence_x"], post_lm["confidence_y"]),
        polarity=f"{pre_lm['polarity_x']}/{pre_lm['polarity_y']}",
    )

    if not (pre_ok and post_ok):
        result.status = "scratch_not_detected"
        return result

    # 傷位置から直接得られる粗い並進(ECC失敗時のフォールバックにも使う)。
    # M(pre->post)の並進成分 = post傷位置 - pre傷位置。
    dxy_coarse = np.array([post_lm["x"] - pre_lm["x"], post_lm["y"] - pre_lm["y"]], dtype=np.float64)

    # 傷ランドマーク周辺の同一座標範囲(pre基準)を pre/post 双方から切り出す。
    px, py = pre_lm["x"], pre_lm["y"]
    x0 = int(max(0, px - crop_margin))
    x1 = int(min(w, px + crop_margin))
    y0 = int(max(0, py - crop_margin))
    y1 = int(min(h, py + crop_margin))

    if (x1 - x0) < 50 or (y1 - y0) < 50:
        result.status = "scratch_crop_too_small"
        return result

    pre_crop = pre_img[y0:y1, x0:x1].astype(np.float32)
    post_crop = post_img[y0:y1, x0:x1].astype(np.float32)

    def _finalize(R, t_global, mode):
        M_global = np.zeros((2, 3), dtype=np.float64)
        M_global[:, :2] = R
        M_global[:, 2] = t_global
        result.warp_matrix = M_global
        result.dx = float(t_global[0])
        result.dy = float(t_global[1])
        result.rotation_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        result.mode = mode
        return result

    def _translation_fallback():
        # 傷位置から得た並進のみで位置合わせ(回転なし)。ECCが使えない場合の保険。
        return _finalize(np.eye(2), dxy_coarse, "translation_fallback")

    # 1. phaseCorrelate で並進初期値(傷位置ベースの粗並進を初期値に足す)
    hann = cv2.createHanningWindow((pre_crop.shape[1], pre_crop.shape[0]), cv2.CV_32F)
    try:
        (shift_x, shift_y), _response = cv2.phaseCorrelate(pre_crop, post_crop, hann)
    except cv2.error as exc:
        logger.warning("phaseCorrelate 失敗(傷位置ベースの並進にフォールバック): %s", exc)
        return _translation_fallback()

    warp_matrix = np.array([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32)

    # 2. findTransformECC (MOTION_EUCLIDEAN) で回転+並進を精密化
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ecc_iterations, ecc_eps)
    try:
        _cc, warp_matrix = cv2.findTransformECC(
            pre_crop, post_crop, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria
        )
    except cv2.error:
        # ECC非収束: 傷位置ベースの並進のみで確定(スキップせず処理継続)
        logger.info("findTransformECC 非収束のため傷位置ベースの並進で位置合わせします。")
        return _translation_fallback()

    # pre_crop / post_crop は同一オフセット(x0, y0)で切り出しているため、
    # クロップ座標系の変換行列をそのまま画像全体の座標系へ変換できる。
    R = warp_matrix[:, :2].astype(np.float64)
    t_local = warp_matrix[:, 2].astype(np.float64)
    offset = np.array([x0, y0], dtype=np.float64)
    t_global = t_local + offset - R @ offset
    return _finalize(R, t_global, "ecc")


def apply_warp(post_img: np.ndarray, warp_matrix: np.ndarray) -> tuple:
    """M_global (dst->src, WARP_INVERSE_MAP) を用いて post 画像を pre 座標系へ変換する。
    戻り値: (aligned_post_float, valid_mask)"""
    h, w = post_img.shape[:2]
    M = warp_matrix.astype(np.float32)

    aligned = cv2.warpAffine(
        post_img.astype(np.float32), M, (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    mask = cv2.warpAffine(
        np.ones((h, w), dtype=np.uint8), M, (w, h),
        flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return aligned, mask


def largest_valid_rect(mask: np.ndarray) -> tuple:
    """全画素が有効(mask==1)な軸並行矩形を、境界を内側へ縮めながら求める(貪欲法)。
    戻り値: (top, bottom, left, right)"""
    h, w = mask.shape
    top, bottom, left, right = 0, h, 0, w
    m = mask.astype(bool)
    while top < bottom and left < right:
        window = m[top:bottom, left:right]
        row_all = window.all(axis=1)
        col_all = window.all(axis=0)
        changed = False
        if not row_all[0]:
            top += 1
            changed = True
        if top < bottom and not row_all[-1]:
            bottom -= 1
            changed = True
        if not col_all[0]:
            left += 1
            changed = True
        if left < right and not col_all[-1]:
            right -= 1
            changed = True
        if not changed:
            break
    return top, bottom, left, right


# ===================== 精度評価 (NCC・一致率) =====================

def compute_ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    if denom <= 0:
        return float("nan")
    return float((a * b).sum() / denom)


def compute_match_rate(a: np.ndarray, b: np.ndarray, threshold: float) -> float:
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return float(np.mean(diff <= threshold) * 100.0)


def region_metrics(pre_img, aligned_post, box, match_threshold) -> dict:
    """box=(x0,y0,x1,y1) の共通領域内で NCC・一致率を計算する。範囲外なら NaN。"""
    x0, y0, x1, y1 = box
    h, w = pre_img.shape[:2]
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return {"ncc": float("nan"), "match_rate": float("nan")}
    a = pre_img[y0:y1, x0:x1]
    b = aligned_post[y0:y1, x0:x1]
    return {"ncc": compute_ncc(a, b), "match_rate": compute_match_rate(a, b, match_threshold)}


def _intervals_overlap(a0, a1, b0, b1) -> bool:
    return a0 < b1 and b0 < a1


def sample_grid_patches(crop_w, crop_h, grid_n, patch_size, scratch_bands):
    """
    共通クロップ領域を grid_n x grid_n に分割し、各セル中心から patch_size 角の
    パッチを切り出す座標を返す。傷帯(scratch_bands)と重なるパッチは除外する。

    傷は十字状(縦傷=全高に延びる列帯 / 横傷=全幅に延びる行帯)であるため、
    scratch_bands = {"v": (xlo, xhi), "h": (ylo, yhi)} (いずれもクロップ座標系、
    無効な帯は None) で表し、
      - 縦傷帯: パッチのx範囲が (xlo, xhi) と重なるものを除外(全行)
      - 横傷帯: パッチのy範囲が (ylo, yhi) と重なるものを除外(全列)
    とする。

    戻り値: [{"patch_id", "cx", "cy", "x0", "y0", "x1", "y1"}] のリスト。
    """
    patches = []
    half = patch_size // 2
    cell_w = crop_w / grid_n
    cell_h = crop_h / grid_n
    v_band = scratch_bands.get("v")
    h_band = scratch_bands.get("h")

    pid = 0
    for r in range(grid_n):
        for c in range(grid_n):
            cx = int((c + 0.5) * cell_w)
            cy = int((r + 0.5) * cell_h)
            x0 = cx - half
            y0 = cy - half
            x1 = x0 + patch_size
            y1 = y0 + patch_size

            # クロップ境界からはみ出す場合は内側へ寄せる
            if x0 < 0:
                x0, x1 = 0, patch_size
            if y0 < 0:
                y0, y1 = 0, patch_size
            if x1 > crop_w:
                x1, x0 = crop_w, crop_w - patch_size
            if y1 > crop_h:
                y1, y0 = crop_h, crop_h - patch_size
            if x0 < 0 or y0 < 0:
                # クロップ領域がパッチより小さい -> このパッチは作れない
                continue

            # 傷帯との重なり判定
            if v_band is not None and _intervals_overlap(x0, x1, v_band[0], v_band[1]):
                continue
            if h_band is not None and _intervals_overlap(y0, y1, h_band[0], h_band[1]):
                continue

            patches.append({
                "patch_id": pid,
                "cx": (x0 + x1) // 2,
                "cy": (y0 + y1) // 2,
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            })
            pid += 1
    return patches


def evaluate_patches(pre_crop, post_crop, patches, match_threshold):
    """各パッチで NCC・一致率を算出し、パッチ明細と集計統計を返す。"""
    details = []
    nccs = []
    rates = []
    for p in patches:
        a = pre_crop[p["y0"]:p["y1"], p["x0"]:p["x1"]]
        b = post_crop[p["y0"]:p["y1"], p["x0"]:p["x1"]]
        ncc = compute_ncc(a, b)
        rate = compute_match_rate(a, b, match_threshold)
        details.append({**p, "ncc": ncc, "match_rate": rate})
        if not np.isnan(ncc):
            nccs.append(ncc)
        if not np.isnan(rate):
            rates.append(rate)

    def _agg(vals):
        if not vals:
            return {"mean": float("nan"), "min": float("nan"), "std": float("nan")}
        arr = np.asarray(vals, dtype=np.float64)
        return {"mean": float(arr.mean()), "min": float(arr.min()), "std": float(arr.std())}

    agg = {
        "n_patches": len(details),
        "ncc": _agg(nccs),
        "match_rate": _agg(rates),
    }
    return details, agg


def draw_patch_overlay(pre_crop, patches, scratch_bands, out_path):
    """パッチ位置と傷帯を pre クロップ画像に重畳した参照PNGを保存する(目視確認用)。"""
    img = pre_crop.astype(np.float64)
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    if hi <= lo:
        hi = lo + 1
    disp = np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    canvas = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
    h, w = disp.shape

    # 傷帯を薄く塗る(赤)
    v_band = scratch_bands.get("v")
    h_band = scratch_bands.get("h")
    overlay = canvas.copy()
    if v_band is not None:
        x0 = max(0, int(v_band[0])); x1 = min(w, int(v_band[1]))
        if x1 > x0:
            overlay[:, x0:x1] = (0, 0, 200)
    if h_band is not None:
        y0 = max(0, int(h_band[0])); y1 = min(h, int(h_band[1]))
        if y1 > y0:
            overlay[y0:y1, :] = (0, 0, 200)
    canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)

    # パッチ矩形とID(緑)
    for p in patches:
        cv2.rectangle(canvas, (p["x0"], p["y0"]), (p["x1"] - 1, p["y1"] - 1), (0, 255, 0), 2)
        cv2.putText(canvas, str(p["patch_id"]), (p["x0"] + 4, p["y0"] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), canvas)


# ===================== メイン処理 =====================

def cast_to_dtype(img_float: np.ndarray, dtype: np.dtype) -> np.ndarray:
    info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else None
    if info is not None:
        clipped = np.clip(np.round(img_float), info.min, info.max)
    else:
        clipped = img_float
    return clipped.astype(dtype)


def load_gray(path: Path) -> np.ndarray:
    img = tifffile.imread(str(path))
    if img.ndim == 3:
        logger.warning("%s はグレースケールではありません。平均してグレースケール化します。", path.name)
        img = img.mean(axis=-1)
    return img


def process_group(label, key: GroupKey, files: dict, args, output_dir: Path):
    """戻り値: (summary_rows, patch_rows)。patch_rows はパッチ明細(long形式)。"""
    rows = []
    patch_rows = []
    if 0 not in files:
        logger.warning("[%s] 連番0(pre)が見つからないためスキップします。", label)
        return rows, patch_rows

    pre_path = files[0]
    pre_img = load_gray(pre_path)
    orig_dtype = pre_img.dtype

    scratch_kwargs = dict(
        v_col_range=args.v_col_range,
        v_row_band=args.v_row_band,
        h_row_range=args.h_row_range,
        h_col_band=args.h_col_band,
        polarity=args.scratch_polarity,
    )

    other_seqs = sorted(k for k in files if k != 0)
    for seq in other_seqs:
        if seq != 1:
            logger.warning(
                "[%s] 連番%d はpre(0)/post(1)以外のためスキップします(命名規則: 連番0=pre, 連番1=post): %s",
                label, seq, files[seq].name,
            )
            continue
        post_path = files[seq]
        row = {
            "condition": key.condition,
            "set": key.set_id,
            "pre_file": pre_path.name,
            "post_file": post_path.name,
            "status": "ok",
            "scratch_detected_pre": False,
            "scratch_detected_post": False,
            "dx_px": float("nan"),
            "dy_px": float("nan"),
            "rotation_deg": float("nan"),
            "registration_mode": "",
            "scratch_confidence_pre": float("nan"),
            "scratch_confidence_post": float("nan"),
            "scratch_polarity": "",
            "scratch_ncc": float("nan"),
            "scratch_match_rate_pct": float("nan"),
            "n_pillar_patches": 0,
            "pillar_ncc_mean": float("nan"),
            "pillar_ncc_min": float("nan"),
            "pillar_ncc_std": float("nan"),
            "pillar_match_rate_mean_pct": float("nan"),
            "pillar_match_rate_min_pct": float("nan"),
            "pillar_match_rate_std_pct": float("nan"),
            "crop_width": float("nan"),
            "crop_height": float("nan"),
        }

        logger.info("[%s] 処理中: %s -> %s", label, pre_path.name, post_path.name)
        post_img = load_gray(post_path)

        if post_img.shape != pre_img.shape:
            logger.warning("[%s] pre/postの画像サイズが一致しません: %s", label, post_path.name)
            row["status"] = "shape_mismatch"
            rows.append(row)
            continue

        reg = register_pair(
            pre_img, post_img, scratch_kwargs,
            min_confidence=args.scratch_min_confidence,
            crop_margin=args.scratch_crop_margin,
            ecc_eps=args.ecc_eps,
            ecc_iterations=args.ecc_iterations,
            min_contrast=args.scratch_min_contrast,
        )

        row["scratch_detected_pre"] = reg.scratch_detected_pre
        row["scratch_detected_post"] = reg.scratch_detected_post
        row["scratch_confidence_pre"] = reg.confidence_pre
        row["scratch_confidence_post"] = reg.confidence_post
        row["scratch_polarity"] = reg.polarity
        row["registration_mode"] = reg.mode

        if reg.status != "ok":
            logger.warning(
                "[%s] 位置合わせ失敗 (%s): %s。傷検出信頼度(inlier率) pre=%.2f post=%.2f "
                "(閾値%.2f, 極性%s)。スキップして次へ進みます。",
                label, reg.status, post_path.name,
                reg.confidence_pre, reg.confidence_post, args.scratch_min_confidence, reg.polarity or "?",
            )
            row["status"] = reg.status
            rows.append(row)
            continue

        if reg.mode == "translation_fallback":
            logger.info("[%s] %s は傷位置ベースの並進のみで位置合わせ(ECC非収束)。", label, post_path.name)

        row["dx_px"] = reg.dx
        row["dy_px"] = reg.dy
        row["rotation_deg"] = reg.rotation_deg

        aligned_post, mask = apply_warp(post_img, reg.warp_matrix)
        top, bottom, left, right = largest_valid_rect(mask)

        if bottom - top < 50 or right - left < 50:
            logger.warning("[%s] 共通有効領域が小さすぎるためスキップします: %s", label, post_path.name)
            row["status"] = "common_region_too_small"
            rows.append(row)
            continue

        pre_cropped_f = pre_img[top:bottom, left:right].astype(np.float32)
        post_cropped_f = aligned_post[top:bottom, left:right]

        row["crop_width"] = right - left
        row["crop_height"] = bottom - top

        # 傷付近の一致率(共通クロップ座標系に変換)
        sx, sy = reg.scratch_xy_pre
        scratch_box_global = (
            int(sx - args.scratch_crop_margin), int(sy - args.scratch_crop_margin),
            int(sx + args.scratch_crop_margin), int(sy + args.scratch_crop_margin),
        )
        scratch_box_local = (
            scratch_box_global[0] - left, scratch_box_global[1] - top,
            scratch_box_global[2] - left, scratch_box_global[3] - top,
        )
        scratch_metrics = region_metrics(pre_cropped_f, post_cropped_f, scratch_box_local, args.match_threshold)
        row["scratch_ncc"] = scratch_metrics["ncc"]
        row["scratch_match_rate_pct"] = scratch_metrics["match_rate"]

        # ピラー領域評価: 傷帯を除いた画像全体からグリッド状に自動サンプリングした
        # 複数パッチで NCC・一致率を個別に算出し、集計値(平均・最小・標準偏差)を記録する。
        # 傷は十字状(縦傷=全高の列帯 / 横傷=全幅の行帯)なので、傷位置±マージンを
        # クロップ座標系の除外帯として扱う。
        sxl = sx - left
        syl = sy - top
        m = args.scratch_exclude_margin
        scratch_bands = {
            "v": (sxl - m, sxl + m),
            "h": (syl - m, syl + m),
        }
        crop_h_px, crop_w_px = pre_cropped_f.shape[:2]
        patches = sample_grid_patches(
            crop_w_px, crop_h_px, args.patch_grid, args.patch_size, scratch_bands
        )
        if not patches:
            logger.warning(
                "[%s] 傷帯を除いた有効パッチが得られませんでした(patch_grid/patch_size を調整してください): %s",
                label, post_path.name,
            )
        else:
            details, agg = evaluate_patches(pre_cropped_f, post_cropped_f, patches, args.match_threshold)
            row["n_pillar_patches"] = agg["n_patches"]
            row["pillar_ncc_mean"] = agg["ncc"]["mean"]
            row["pillar_ncc_min"] = agg["ncc"]["min"]
            row["pillar_ncc_std"] = agg["ncc"]["std"]
            row["pillar_match_rate_mean_pct"] = agg["match_rate"]["mean"]
            row["pillar_match_rate_min_pct"] = agg["match_rate"]["min"]
            row["pillar_match_rate_std_pct"] = agg["match_rate"]["std"]

            for d in details:
                patch_rows.append({
                    "condition": key.condition,
                    "set": key.set_id,
                    "post_file": post_path.name,
                    "patch_id": d["patch_id"],
                    "center_x": d["cx"],
                    "center_y": d["cy"],
                    "x0": d["x0"], "y0": d["y0"], "x1": d["x1"], "y1": d["y1"],
                    "ncc": d["ncc"],
                    "match_rate_pct": d["match_rate"],
                })

            if args.save_patch_overlay:
                overlay_path = output_dir / f"{key.condition}-{key.set_id}_patch_overlay.png"
                draw_patch_overlay(pre_cropped_f, patches, scratch_bands, overlay_path)
                logger.info("[%s] パッチ配置図を保存しました: %s", label, overlay_path.name)

        # 出力画像の保存(元のbit深度を保持)
        pre_out = cast_to_dtype(pre_cropped_f, orig_dtype)
        post_out = cast_to_dtype(post_cropped_f, orig_dtype)

        output_dir.mkdir(parents=True, exist_ok=True)
        pre_out_path = output_dir / f"{key.condition}-{key.set_id}-0.tif"
        post_out_path = output_dir / f"{key.condition}-{key.set_id}-{seq}.tif"
        tifffile.imwrite(pre_out_path, pre_out)
        tifffile.imwrite(post_out_path, post_out)
        logger.info("[%s] 保存しました: %s, %s", label, pre_out_path.name, post_out_path.name)

        rows.append(row)

    return rows, patch_rows


def parse_int_tuple(s: str, n: int) -> tuple:
    parts = [int(x) for x in s.split(",")]
    if len(parts) != n:
        raise argparse.ArgumentTypeError(f"{n}個のカンマ区切り整数が必要です: {s}")
    return tuple(parts)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="df データセット全体を対象としたバッチ画像位置合わせスクリプト",
    )
    parser.add_argument("--input-dir", type=str, required=True, help="入力フォルダ(条件-セット-連番.tif が格納されたフォルダ)")
    parser.add_argument("--output-dir", type=str, default=None, help="出力フォルダ(省略時は <input-dir>/analysis)")

    # --- ピラー領域評価: 自動マルチパッチサンプリング ---
    parser.add_argument("--patch-grid", type=int, default=5,
                         help="共通クロップ領域を分割するグリッド数 N (NxN, 既定5)")
    parser.add_argument("--patch-size", type=int, default=128,
                         help="各パッチの一辺サイズ(px, 既定128)")
    parser.add_argument("--scratch-exclude-margin", type=int, default=120,
                         help="傷位置±このマージン(px)を傷帯として扱いパッチ除外する(既定120)")
    parser.add_argument("--save-patch-overlay", action="store_true",
                         help="パッチ配置と傷帯を重畳した参照PNGを条件-セット毎に出力する(目視確認用)")

    parser.add_argument("--scratch-crop-margin", type=int, default=250,
                         help="傷ランドマーク周辺の位置合わせ用切り出し半幅(px, 既定250)")
    parser.add_argument("--scratch-min-confidence", type=float, default=0.5,
                         help="傷検出とみなす最小信頼度=inlier率(0..1, 既定0.5)。"
                              "各スライスの極値位置が線状にそろっている割合")
    parser.add_argument("--scratch-min-contrast", type=float, default=2.0,
                         help="傷とみなす最小コントラストSNR(既定2.0)。弱い勾配での誤検出防止")
    parser.add_argument("--scratch-polarity", type=str, default="auto",
                         choices=["auto", "bright", "dark"],
                         help="傷の極性: bright=明るい線(暗視野散乱像) / dark=暗い線(明視野) / "
                              "auto=強い方を自動判別(既定)")
    parser.add_argument("--match-threshold", type=float, default=5.0,
                         help="一致率判定の許容階調差 |Δ|<=threshold (既定5)")
    parser.add_argument("--ecc-iterations", type=int, default=200, help="findTransformECC 最大反復回数")
    parser.add_argument("--ecc-eps", type=float, default=1e-6, help="findTransformECC 収束閾値")

    parser.add_argument("--v-col-range", type=lambda s: parse_int_tuple(s, 2), default=DEFAULT_V_COL_RANGE,
                         help="縦傷検出の列範囲 'c0,c1'")
    parser.add_argument("--v-row-band", type=lambda s: parse_int_tuple(s, 2), default=DEFAULT_V_ROW_BAND,
                         help="縦傷検出の平均化行帯域 'r0,r1'")
    parser.add_argument("--h-row-range", type=lambda s: parse_int_tuple(s, 2), default=DEFAULT_H_ROW_RANGE,
                         help="横傷検出の行範囲 'r0,r1'")
    parser.add_argument("--h-col-band", type=lambda s: parse_int_tuple(s, 2), default=DEFAULT_H_COL_BAND,
                         help="横傷検出の平均化列帯域 'c0,c1'")

    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"エラー: 入力フォルダが存在しません: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "batch_register.log", encoding="utf-8"),
        ],
    )

    logger.info("入力フォルダ: %s", input_dir)
    logger.info("出力フォルダ: %s", output_dir)

    groups = group_files(input_dir)
    if not groups:
        logger.warning("グルーピング可能なファイルが見つかりませんでした。")
        return

    all_rows = []
    all_patch_rows = []
    for label, group in sorted(groups.items()):
        rows, patch_rows = process_group(label, group["key"], group["files"], args, output_dir)
        all_rows.extend(rows)
        all_patch_rows.extend(patch_rows)

    summary_df = pd.DataFrame(all_rows)
    summary_path = output_dir / "registration_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    logger.info("サマリーCSVを保存しました: %s", summary_path)

    # パッチ明細(long形式): 条件-セット x パッチ ごとの NCC・一致率
    patch_df = pd.DataFrame(all_patch_rows)
    patch_path = output_dir / "pillar_patch_metrics.csv"
    patch_df.to_csv(patch_path, index=False, encoding="utf-8-sig")
    logger.info("パッチ明細CSVを保存しました: %s", patch_path)

    if not summary_df.empty:
        ok_count = (summary_df["status"] == "ok").sum()
        logger.info("処理完了: %d/%d ペアが正常に位置合わせされました。", ok_count, len(summary_df))


if __name__ == "__main__":
    main()
