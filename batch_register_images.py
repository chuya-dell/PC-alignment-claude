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

# 傷(スクラッチ)ランドマーク検出の探索範囲。
# 位置合わせテストでは同じ傷十字が視野ごとに異なる位置に写るため、既定は
# 画像全体(None=フルフレーム)を探索する。CLI引数で範囲を絞ることも可能。
#   V*: 縦傷 → 列(v_col_range)を探索し、行(v_row_band)方向に追跡
#   H*: 横傷 → 行(h_row_range)を探索し、列(h_col_band)方向に追跡
DEFAULT_V_COL_RANGE = None
DEFAULT_V_ROW_BAND = None
DEFAULT_H_ROW_RANGE = None
DEFAULT_H_COL_BAND = None


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


def _subpixel_offset(sub: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """_subpixel_min のベクトル化版。各行 sub[i,:] の整数極値位置 idx[i] を、隣接3点
    (idx-1,idx,idx+1) の放物線頂点でサブピクセル補正するオフセットを返す(行数と同じ長さ)。
    端(idx=0 or idx=w-1)や放物線が定義できない(分母0)行は補正なし(0)。"""
    n, w = sub.shape
    offset = np.zeros(n, dtype=np.float64)
    valid = (idx > 0) & (idx < w - 1)
    if not np.any(valid):
        return offset
    rows = np.nonzero(valid)[0]
    ic = idx[valid]
    y0 = sub[rows, ic - 1]
    y1 = sub[rows, ic]
    y2 = sub[rows, ic + 1]
    denom = y0 - 2 * y1 + y2
    nz = denom != 0
    off = np.zeros(len(rows), dtype=np.float64)
    off[nz] = 0.5 * (y0[nz] - y2[nz]) / denom[nz]
    offset[rows] = np.clip(off, -1.0, 1.0)
    return offset


def _smooth_rows(m: np.ndarray, k: int) -> np.ndarray:
    kern = np.ones(k) / k
    return np.apply_along_axis(lambda v: np.convolve(v, kern, mode="same"), 1, m)


def _trace_line(band: np.ndarray, line_axis: int, polarity: str,
                edge_margin: int, smooth_window: int, tol: float) -> dict:
    """
    フルフレーム探索で傷ランドマーク線を頑健に同定する。
    帯域を「線が走る方向(行) × 探索方向(列)」に整え、各行で極値位置を取り、
    その位置の1次元モード(幅2*tolの窓に最も多く入る位置=支配的な線)を求める。

    傷は視野ごとに任意の位置に写る(位置合わせテスト)ため、固定帯域ではなく
    全幅から線を探す。線がフレームの一部しか横切らなくても、支配的なクラスタの
    投票数/全行 で信頼度が下がるだけで位置は正しく取れる。ランダム/ピラー模様は
    極値位置が散らばるためモードの投票率が低い(≈2*tol/幅)。十字のもう一方の腕は
    少数行にしか現れないため主たる線が勝つ。

    戻り値 dict:
      position   : 線位置(探索方向, band内オフセット込み相対値)
      confidence : モード投票率(0..1) = 支配的な線上に極値が乗った行の割合
      contrast_snr: 線コントラストのノイズ比(弱い勾配での誤検出防止)
      waviness   : インライア位置の標準偏差(px, 波打ち量)
      polarity   : 採用極性
    """
    m = band if line_axis == 0 else band.T  # 常に「行方向に線が走る」向きへ
    N, M = m.shape
    ms = _smooth_rows(m, smooth_window)
    margin = min(edge_margin, (M - 1) // 2)
    lo, hi = margin, M - margin
    sub = ms[:, lo:hi]
    row_med = np.median(sub, axis=1)

    if polarity == "bright":
        idx = np.argmax(sub, axis=1)
        pos = (lo + idx).astype(np.float64) + _subpixel_offset(sub, idx)
        contrast = np.max(sub, axis=1) - row_med
    else:
        idx = np.argmin(sub, axis=1)
        pos = (lo + idx).astype(np.float64) + _subpixel_offset(sub, idx)
        contrast = row_med - np.min(sub, axis=1)

    noise = np.median(np.abs(sub - row_med[:, None])) * 1.4826 + 1e-6
    result = {"position": float(np.median(pos)), "confidence": 0.0,
              "contrast_snr": 0.0, "waviness": float(np.std(pos)), "polarity": polarity,
              "angle_rad": 0.0}
    if N < 10:
        return result

    def _windowed_mode(values: np.ndarray) -> np.ndarray:
        """幅 2*tol の窓に最も多くの行の位置が入るクラスタを inlier マスクで返す。"""
        order = np.argsort(values)
        sv = values[order]
        right = np.searchsorted(sv, sv + 2.0 * tol, side="right")
        counts = right - np.arange(len(sv))
        bi = int(np.argmax(counts))
        win_lo, win_hi = sv[bi], sv[bi] + 2.0 * tol
        return (values >= win_lo) & (values <= win_hi)

    idx_rows = np.arange(N, dtype=np.float64)

    # 1次元モード(1st pass): 生の位置でクラスタリング。線がほぼ傾いていなければこれで十分。
    inlier = _windowed_mode(pos)

    # 2nd pass: 線に実傾き(数度でも画像全高では数十px動き得る)があると、1st passの窓
    # (幅2*tol)は全長のごく一部しか拾えず信頼度が低くなる。1st passの結果で暫定的に
    # 傾きを推定し、傾きを除去(デトレンド)した位置で再クラスタリングする。デトレンド後は
    # 傾きの影響が消えて波打ちのみが残るため、傾いた線でも正しく支配的クラスタを拾える。
    if inlier.sum() >= 10:
        rows0 = idx_rows[inlier]
        if rows0.max() - rows0.min() >= 0.25 * N:
            slope0 = np.polyfit(rows0, pos[inlier], 1)[0]
            detrended = pos - slope0 * idx_rows
            inlier2 = _windowed_mode(detrended)
            if inlier2.sum() > inlier.sum():
                inlier = inlier2

    count = int(inlier.sum())
    result["confidence"] = float(count) / N
    result["waviness"] = float(np.std(pos[inlier])) if inlier.sum() > 1 else 0.0
    med_contrast = np.median(contrast[inlier]) if inlier.any() else 0.0
    result["contrast_snr"] = float(med_contrast / noise)

    # インライア行に対する position の1次傾き(=線の傾き, m座標系: d(position)/d(row))。
    # 回転推定に使う。インライアが少ない/短い場合は 0(傾き無し)とする。
    rows = idx_rows[inlier]
    if rows.size >= 10 and (rows.max() - rows.min()) >= 0.25 * N:
        slope, intercept = np.polyfit(rows, pos[inlier], 1)
        # position は「傾いた線をband中心行(N/2)で評価した値」とする(pre/postで一貫した
        # 基準行を使うことで、傾きにより支配的クラスタの行範囲がpre/postで微妙にずれても
        # 位置の再現性を保つ。インライアの中央値だと傾いた線では基準行がpre/postでずれ得る)。
        result["position"] = float(slope * (N / 2.0) + intercept)
    else:
        slope = 0.0
        result["position"] = float(np.median(pos[inlier])) if inlier.any() else result["position"]
    result["angle_rad"] = float(np.arctan(slope))
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
    edge_margin: int = 8,
    polarity: str = "auto",
    trace_tol: float = 20.0,
    polarity_v: Optional[str] = None,
    polarity_h: Optional[str] = None,
) -> dict:
    """
    デザインナイフ傷(縦線・横線)のランドマーク座標を検出する。
    位置合わせテストでは傷十字が視野ごとに任意位置に写るため、既定では
    画像全体(範囲=None)から支配的な縦/横の明線(または暗線)を探す。
    各範囲を明示指定すればその中に限定して探索できる。
    polarity で暗い傷/明るい傷/自動判別を切替。信頼度はモード投票率(0..1)。
    polarity_v/polarity_h を指定すると縦傷/横傷を個別の極性で固定できる
    (auto では縦横で異なる極性が採用されることが実データで多く、両方に同じ
    polarity を強制すると片方が誤検出する。残差再検出で使用)。
    """
    h, w = img.shape[:2]
    vc0, vc1 = v_col_range if v_col_range else (0, w)
    vr0, vr1 = v_row_band if v_row_band else (0, h)
    vc0, vc1 = max(0, vc0), min(w, vc1)
    vr0, vr1 = max(0, vr0), min(h, vr1)
    v_band = img[vr0:vr1, vc0:vc1].astype(np.float64)
    v = _detect_line(v_band, line_axis=0, polarity=polarity_v or polarity,
                     edge_margin=edge_margin, smooth_window=smooth_window, tol=trace_tol)
    scratch_x = vc0 + v["position"]

    hr0, hr1 = h_row_range if h_row_range else (0, h)
    hc0, hc1 = h_col_band if h_col_band else (0, w)
    hr0, hr1 = max(0, hr0), min(h, hr1)
    hc0, hc1 = max(0, hc0), min(w, hc1)
    h_band = img[hr0:hr1, hc0:hc1].astype(np.float64)
    hln = _detect_line(h_band, line_axis=1, polarity=polarity_h or polarity,
                       edge_margin=edge_margin, smooth_window=smooth_window, tol=trace_tol)
    scratch_y = hr0 + hln["position"]

    return {"x": scratch_x, "y": scratch_y,
            "confidence_x": v["confidence"], "confidence_y": hln["confidence"],
            "contrast_x": v["contrast_snr"], "contrast_y": hln["contrast_snr"],
            "polarity_x": v["polarity"], "polarity_y": hln["polarity"],
            "waviness_x": v["waviness"], "waviness_y": hln["waviness"],
            # 縦傷: d(col)/d(row) の傾き角、横傷: d(row)/d(col) の傾き角(いずれも rad)
            "angle_v_rad": v["angle_rad"], "angle_h_rad": hln["angle_rad"]}


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
    mode: str = ""  # "ecc" / "scratch_translation"
    # 傷位置ベースの粗い並進(post傷 - pre傷)。診断用。
    dxy_coarse: Optional[tuple] = None
    # 位置合わせ後に傷を再検出して測った残差(px)。採用warpのもの=主品質指標。
    scratch_residual_px: float = float("nan")
    # ECC候補単体の傷残差(px)。診断用: 大きいほどECCが別解へ飛んでいる。
    residual_ecc_px: float = float("nan")
    # 傷十字ベース並進候補の傷残差(px)。診断用(構造上ほぼ0のはず)。
    residual_scratch_px: float = float("nan")


def _euclidean_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.zeros((2, 3), dtype=np.float64)
    M[:, :2] = R
    M[:, 2] = t
    return M


def _scratch_rotation_deg(pre_lm: dict, post_lm: dict,
                          max_deg: float = 5.0, agree_tol_deg: float = 1.0) -> float:
    """pre→post の回転角(度)を傷十字の線傾きから推定する。
    縦傷と横傷から独立に推定し、両者が近い(agree_tol_deg以内)かつ小さい(max_deg以内)場合のみ
    採用する。そうでなければ 0(=並進のみ)を返す(不確かな回転で悪化させない)。"""
    tv = np.degrees(pre_lm["angle_v_rad"] - post_lm["angle_v_rad"])
    th = np.degrees(post_lm["angle_h_rad"] - pre_lm["angle_h_rad"])
    if abs(tv - th) > agree_tol_deg:
        return 0.0
    theta = 0.5 * (tv + th)
    if abs(theta) > max_deg:
        return 0.0
    return float(theta)


def _euclidean_from_scratch(pre_lm: dict, post_lm: dict, rot_deg: float) -> np.ndarray:
    """傷十字の交点(pre)→(post)を一致させる pre→post Euclidean 変換 M。
    p_post = R p_pre + (c_post - R c_pre)。rot_deg=0 なら純並進(t=post傷-pre傷)。"""
    th = np.radians(rot_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float64)
    c_pre = np.array([pre_lm["x"], pre_lm["y"]], dtype=np.float64)
    c_post = np.array([post_lm["x"], post_lm["y"]], dtype=np.float64)
    return _euclidean_matrix(R, c_post - R @ c_pre)


def _scratch_residual(pre_lm: dict, post_img: np.ndarray,
                      M_global: np.ndarray, scratch_kwargs: dict, window: int) -> float:
    """候補 warp を post に適用して pre 座標系へ戻し、傷を再検出して pre 傷位置との距離(px)。
    位置合わせが正しければ ≈0。ECCが別解へ飛べば大きくなる。周期ピラーに交絡されない。

    ロバスト化のポイント:
      - 探索を pre 傷位置±window に限定(ECC暴走はwindow内で検出できる)。
      - 極性を pre で採用したものに**縦横別々に**固定(auto だと誤検出する。縦傷と横傷で極性が
        異なる視野が実データで多く、片方の極性をもう片方に流用すると誤った極性で探索し
        無関係な特徴を掴んで見かけ上の残差が跳ね上がるバグがあったため)。
      - warp は BORDER_REPLICATE で行う(BORDER_CONSTANT=0埋めだと、傷が画像端に近く並進が
        その方向を向く視野で、無効領域の人工的な黒縁が背景よりずっと暗いため『暗い傷』の
        探索窓内でこの黒縁を誤って掴んでしまい、見かけ上の残差が跳ね上がるバグがあったため)。"""
    aligned, _ = apply_warp(post_img, M_global, border_mode=cv2.BORDER_REPLICATE)
    px, py = pre_lm["x"], pre_lm["y"]
    kw = dict(scratch_kwargs)
    kw.pop("polarity", None)
    kw["polarity_v"] = pre_lm["polarity_x"]
    kw["polarity_h"] = pre_lm["polarity_y"]
    kw["v_col_range"] = (int(px - window), int(px + window))
    kw["h_row_range"] = (int(py - window), int(py + window))
    lm = detect_scratch_landmark(aligned, **kw)
    return float(np.hypot(lm["x"] - px, lm["y"] - py))


def register_pair(
    pre_img: np.ndarray,
    post_img: np.ndarray,
    scratch_kwargs: dict,
    min_confidence: float,
    crop_margin: int,
    ecc_eps: float,
    ecc_iterations: int,
    min_contrast: float = 2.0,
    residual_tol: float = 2.0,
) -> RegistrationResult:
    h, w = pre_img.shape[:2]

    pre_lm = detect_scratch_landmark(pre_img, **scratch_kwargs)
    # post は pre で採用された極性を縦横それぞれ強制する(polarity="auto" のまま独立に
    # 選ばせると、信号が弱い視野で pre/post が溝の暗い側/明るい側を別々に掴んでしまい、
    # 見かけ上のズレが生じるため。同じ物理特徴を追跡させて dxy_coarse を意味のあるものにする)。
    post_kwargs = dict(scratch_kwargs)
    post_kwargs.pop("polarity", None)
    post_kwargs["polarity_v"] = pre_lm["polarity_x"]
    post_kwargs["polarity_h"] = pre_lm["polarity_y"]
    post_lm = detect_scratch_landmark(post_img, **post_kwargs)

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

    # 傷位置から直接得られる粗い並進(post傷 - pre傷)。診断・フォールバック用。
    dxy_coarse = np.array([post_lm["x"] - pre_lm["x"], post_lm["y"] - pre_lm["y"]], dtype=np.float64)
    result.dxy_coarse = (float(dxy_coarse[0]), float(dxy_coarse[1]))

    # --- 候補A: 傷十字ベースの Euclidean(並進 + 傷角度から推定した回転)---
    rot_scratch = _scratch_rotation_deg(pre_lm, post_lm)
    M_scratch = _euclidean_from_scratch(pre_lm, post_lm, rot_scratch)

    # pre/post それぞれ「自分の傷位置」を中心に同サイズで切り出す(ECC入力)。
    # 位置合わせテストでは pre/post の傷が大きくずれる視野があり、pre基準の同一座標で
    # post を切ると post 側の窓に傷が入らず ECC が別解(ピラー模様)に収束してしまう。
    size = int(min(2 * crop_margin, h, w))
    if size < 50:
        result.status = "scratch_crop_too_small"
        return result

    def _crop_around(img, cx, cy):
        x0 = int(round(cx - size / 2)); y0 = int(round(cy - size / 2))
        x0 = min(max(0, x0), w - size); y0 = min(max(0, y0), h - size)
        return img[y0:y0 + size, x0:x0 + size].astype(np.float32), x0, y0

    pre_crop, x0p, y0p = _crop_around(pre_img, pre_lm["x"], pre_lm["y"])
    post_crop, x0q, y0q = _crop_around(post_img, post_lm["x"], post_lm["y"])

    # --- 候補B: phaseCorrelate 初期化 -> findTransformECC(回転+残差並進を精密化)---
    M_ecc = None
    hann = cv2.createHanningWindow((size, size), cv2.CV_32F)
    try:
        (shift_x, shift_y), _response = cv2.phaseCorrelate(pre_crop, post_crop, hann)
        warp_matrix = np.array([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ecc_iterations, ecc_eps)
        _cc, warp_matrix = cv2.findTransformECC(
            pre_crop, post_crop, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria
        )
        # クロップ原点が pre/post で異なるため一般式で全体座標系へ合成する:
        #   p_post = R p_pre + [ (x0q,y0q) - R (x0p,y0p) + t_local ]
        R = warp_matrix[:, :2].astype(np.float64)
        t_local = warp_matrix[:, 2].astype(np.float64)
        op = np.array([x0p, y0p], dtype=np.float64)
        oq = np.array([x0q, y0q], dtype=np.float64)
        M_ecc = _euclidean_matrix(R, oq - R @ op + t_local)
    except cv2.error:
        logger.info("phaseCorrelate/findTransformECC 非収束。傷十字ベース並進を採用します。")

    # --- 自己検証: 各候補を実際に適用して傷を再検出し残差(px)を実測 ---
    # 傷十字ベースは構造上 pre 傷へ戻るので残差≈0。ECCは別解へ飛ぶと残差が大きくなる。
    # よって 250px 等のマジックナンバーではなく「位置合わせ後に傷がどれだけ合ったか」で選ぶ。
    res_window = max(int(crop_margin), 96)
    res_scratch = _scratch_residual(pre_lm, post_img, M_scratch, scratch_kwargs, res_window)
    result.residual_scratch_px = res_scratch
    if M_ecc is not None:
        res_ecc = _scratch_residual(pre_lm, post_img, M_ecc, scratch_kwargs, res_window)
    else:
        res_ecc = float("inf")
    result.residual_ecc_px = res_ecc

    def _finalize(M_global, residual, mode):
        R = M_global[:, :2]
        result.warp_matrix = M_global
        result.dx = float(M_global[0, 2])
        result.dy = float(M_global[1, 2])
        result.rotation_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        result.scratch_residual_px = float(residual)
        result.mode = mode
        return result

    # ECC が傷を残差許容内(residual_tol)に保っていれば、回転を精密化できる ECC を優先。
    # そうでなければ傷十字ベース並進(=暴走ECCの自動棄却)。
    if M_ecc is not None and res_ecc <= max(residual_tol, res_scratch):
        return _finalize(M_ecc, res_ecc, "ecc")
    if res_scratch <= res_ecc:
        return _finalize(M_scratch, res_scratch, "scratch_translation")
    return _finalize(M_ecc, res_ecc, "ecc")


def apply_warp(post_img: np.ndarray, warp_matrix: np.ndarray,
               border_mode: int = cv2.BORDER_CONSTANT) -> tuple:
    """M_global (dst->src, WARP_INVERSE_MAP) を用いて post 画像を pre 座標系へ変換する。
    戻り値: (aligned_post_float, valid_mask)

    border_mode: 既定の BORDER_CONSTANT(0埋め) は最終出力用(mask で無効域を除いてクロップする
    前提)。傷の再検出目的では 0埋めが人工的な極端に暗い縁を作り誤検出の原因になるため、
    BORDER_REPLICATE(端の画素を引き伸ばす)を指定して回避する(_scratch_residual で使用)。"""
    h, w = post_img.shape[:2]
    M = warp_matrix.astype(np.float32)

    aligned = cv2.warpAffine(
        post_img.astype(np.float32), M, (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=border_mode, borderValue=0.0,
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


def compute_scratch_line_ncc(pre_crop, post_crop, sxl, syl, half=8) -> float:
    """傷線に沿った細帯(縦傷=幅2*half列の全高帯 / 横傷=高さ2*half行の全幅帯)だけで NCC を取る。
    500px箱と違い周期ピラーをほとんど含まないため、非周期の傷そのものが合っているかを測れる。
    縦横の有効な帯の平均を返す。"""
    h, w = pre_crop.shape[:2]
    vals = []
    x0, x1 = int(round(sxl - half)), int(round(sxl + half + 1))
    if x0 >= 0 and x1 <= w and x1 - x0 >= 2:
        vals.append(compute_ncc(pre_crop[:, x0:x1], post_crop[:, x0:x1]))
    y0, y1 = int(round(syl - half)), int(round(syl + half + 1))
    if y0 >= 0 and y1 <= h and y1 - y0 >= 2:
        vals.append(compute_ncc(pre_crop[y0:y1, :], post_crop[y0:y1, :]))
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


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
        trace_tol=args.scratch_trace_tol,
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
            "dxy_coarse_x": float("nan"),
            "dxy_coarse_y": float("nan"),
            "scratch_residual_px": float("nan"),
            "residual_ecc_px": float("nan"),
            "residual_scratch_px": float("nan"),
            "scratch_confidence_pre": float("nan"),
            "scratch_confidence_post": float("nan"),
            "scratch_polarity": "",
            "scratch_line_ncc": float("nan"),
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
            residual_tol=args.scratch_residual_tol,
        )

        row["scratch_detected_pre"] = reg.scratch_detected_pre
        row["scratch_detected_post"] = reg.scratch_detected_post
        row["scratch_confidence_pre"] = reg.confidence_pre
        row["scratch_confidence_post"] = reg.confidence_post
        row["scratch_polarity"] = reg.polarity
        row["registration_mode"] = reg.mode
        row["scratch_residual_px"] = reg.scratch_residual_px
        row["residual_ecc_px"] = reg.residual_ecc_px
        row["residual_scratch_px"] = reg.residual_scratch_px
        if reg.dxy_coarse is not None:
            row["dxy_coarse_x"] = reg.dxy_coarse[0]
            row["dxy_coarse_y"] = reg.dxy_coarse[1]

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

        if reg.mode == "scratch_translation":
            logger.info(
                "[%s] %s は傷十字ベースの並進で位置合わせ(ECC残差 %.1fpx > 許容 %.1fpx のため棄却)。",
                label, post_path.name, reg.residual_ecc_px, args.scratch_residual_tol,
            )

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
        # 傷線に沿った細帯NCC(周期ピラーを含めない、非周期の傷そのものの一致度)
        row["scratch_line_ncc"] = compute_scratch_line_ncc(
            pre_cropped_f, post_cropped_f, sx - left, sy - top
        )

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
    parser.add_argument("--scratch-min-confidence", type=float, default=0.35,
                         help="傷検出とみなす最小信頼度=モード投票率(0..1, 既定0.35)。"
                              "支配的な線上に極値が乗った行/列の割合")
    parser.add_argument("--scratch-min-contrast", type=float, default=2.0,
                         help="傷とみなす最小コントラストSNR(既定2.0)。弱い勾配での誤検出防止")
    parser.add_argument("--scratch-trace-tol", type=float, default=20.0,
                         help="傷線追跡のモード窓半幅(px, 既定20)。波打ち量に合わせる")
    parser.add_argument("--scratch-residual-tol", type=float, default=2.0,
                         help="位置合わせ後の傷残差の許容(px, 既定2)。ECC結果をこの残差内に"
                              "保てればECCを採用、超えたら傷十字ベース並進に自動フォールバック")
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
