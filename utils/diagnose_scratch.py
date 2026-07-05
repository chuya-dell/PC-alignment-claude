"""傷(スクラッチ)検出の診断ツール。

batch_register_images.py の傷ランドマーク検出を**そのまま呼び出して**可視化する。
（以前は固定帯域を持つ独自実装で本体の全フレーム探索を再現できず誤診の元だった。
  現在は本体 detect_scratch_landmark を使うため定義上ズレない。）

用途:
  - 1枚 (--image):        本体が検出する傷十字の位置・信頼度・極性・波打ちを可視化。
  - 2枚 (--image --image2): pre/post を並べて各々の傷位置と差分(=傷位置ベース並進 dxy_coarse)を出力。
    大オフセット視野で pre/post の傷が別位置に取れていないか目視できる。

探索範囲は既定で画像全体(本体既定と同じ)。必要なら --v-col-range 等で絞れる。

実行例:
  python utils/diagnose_scratch.py --image "F:/.../df/4-4-0.tif" --image2 "F:/.../df/4-4-1.tif" --output-dir scratch_diag
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import cv2

# 本体の検出器を再利用(これにより診断と本体の挙動が定義上一致する)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batch_register_images import detect_scratch_landmark  # noqa: E402


def load_gray(path):
    try:
        import tifffile
        a = tifffile.imread(str(path))
    except Exception:
        a = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if a.ndim == 3:
        a = a.mean(-1)
    return a.astype(np.float64)


def parse_range(s):
    return tuple(int(x) for x in s.split(","))


def detect(img, args):
    return detect_scratch_landmark(
        img,
        v_col_range=args.v_col_range, v_row_band=args.v_row_band,
        h_row_range=args.h_row_range, h_col_band=args.h_col_band,
        polarity=args.polarity, trace_tol=args.trace_tol,
    )


def describe(name, lm):
    print(f"=== {name} ===")
    print(f"  縦傷: x={lm['x']:.1f}  conf={lm['confidence_x']:.2f}  contrast={lm['contrast_x']:.1f}  "
          f"pol={lm['polarity_x']}  波打ちstd={lm['waviness_x']:.1f}px  傾き={np.degrees(lm['angle_v_rad']):+.2f}deg")
    print(f"  横傷: y={lm['y']:.1f}  conf={lm['confidence_y']:.2f}  contrast={lm['contrast_y']:.1f}  "
          f"pol={lm['polarity_y']}  波打ちstd={lm['waviness_y']:.1f}px  傾き={np.degrees(lm['angle_h_rad']):+.2f}deg")


def overlay(img, lm, scale):
    lo_p, hi_p = np.percentile(img, 1), np.percentile(img, 99.5)
    disp = np.clip((img - lo_p) / max(hi_p - lo_p, 1) * 255, 0, 255).astype(np.uint8)
    h, w = img.shape
    small = cv2.resize(disp, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    canvas = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    x = int(lm["x"] * scale); y = int(lm["y"] * scale)
    cv2.line(canvas, (x, 0), (x, small.shape[0]), (0, 0, 255), 1)
    cv2.line(canvas, (0, y), (small.shape[1], y), (0, 0, 255), 1)
    return canvas


def main():
    ap = argparse.ArgumentParser(description="傷検出の診断(本体 detect_scratch_landmark を可視化)")
    ap.add_argument("--image", required=True, help="pre 画像")
    ap.add_argument("--image2", default=None, help="post 画像(指定すると pre/post 比較)")
    ap.add_argument("--output-dir", default="scratch_diag")
    ap.add_argument("--v-col-range", type=parse_range, default=None, help="縦傷の列探索範囲 'c0,c1'(既定=全幅)")
    ap.add_argument("--v-row-band", type=parse_range, default=None, help="縦傷の行帯域 'r0,r1'(既定=全高)")
    ap.add_argument("--h-row-range", type=parse_range, default=None, help="横傷の行探索範囲 'r0,r1'(既定=全高)")
    ap.add_argument("--h-col-band", type=parse_range, default=None, help="横傷の列帯域 'c0,c1'(既定=全幅)")
    ap.add_argument("--polarity", default="auto", choices=["auto", "bright", "dark"])
    ap.add_argument("--trace-tol", type=float, default=20.0)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    pre = load_gray(args.image)
    h, w = pre.shape
    scale = 900 / max(h, w)
    pre_lm = detect(pre, args)
    describe(Path(args.image).name, pre_lm)
    cv2.imwrite(str(out / f"{Path(args.image).stem}_scratch_overlay.png"), overlay(pre, pre_lm, scale))

    if args.image2:
        post = load_gray(args.image2)
        post_lm = detect(post, args)
        describe(Path(args.image2).name, post_lm)
        cv2.imwrite(str(out / f"{Path(args.image2).stem}_scratch_overlay.png"), overlay(post, post_lm, scale))
        dx = post_lm["x"] - pre_lm["x"]
        dy = post_lm["y"] - pre_lm["y"]
        print(f"--- pre→post 傷位置差 (dxy_coarse) = ({dx:+.1f}, {dy:+.1f}) px ---")
        print("  この差が数十px超なら、pre/postで別の線を掴んでいる/傷が大きくずれている可能性。")
        print("  batch の dxy_coarse_x/y 列と一致するはず。scratch_residual_px が大きい視野をここで確認する。")

    print(f"出力: {out}/")


if __name__ == "__main__":
    main()
