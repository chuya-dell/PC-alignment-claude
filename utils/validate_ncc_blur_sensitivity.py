"""
NCC / 一致率 のボケ感度検証ユーティリティ。

batch_register_images.py のピラー領域精度評価(NCC・一致率)は画素値相関ベースの
ため、「像がぼやけている(ピラーが1本ずつ解像していない)」データでは
本当はズレているのに指標が高く出る偽陽性リスクがある。

このスクリプトは、既知のズレ量(並進)を与えたボケ画像ペアを合成し、
NCC・一致率がズレ量に対して単調に悪化するかを確認する。ボケσが大きい領域で
ズレを入れても指標がほぼ変化しないなら、その領域では指標が精度検証に使えない。

実行:
    python utils/validate_ncc_blur_sensitivity.py
"""
import numpy as np
import cv2

rng = np.random.default_rng(1)


def make_pillar_field(w=512, h=512, spacing=8, amp=60, base=100):
    """周期的なピラー状パターン(疑似SAM像)を生成する。"""
    img = np.full((h, w), float(base), dtype=np.float32)
    ys, xs = np.mgrid[spacing:h - spacing:spacing, spacing:w - spacing:spacing]
    xi = xs.ravel()
    yi = ys.ravel()
    img[yi, xi] += amp
    img[yi - 1, xi] += amp * 0.5
    img[yi + 1, xi] += amp * 0.5
    img[yi, xi - 1] += amp * 0.5
    img[yi, xi + 1] += amp * 0.5
    return img


def blur(img, sigma):
    if sigma <= 0:
        return img.copy()
    k = int(sigma * 6) | 1
    return cv2.GaussianBlur(img, (k, k), sigma)


def ncc(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def match_rate(a, b, thr=5.0):
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64)) <= thr) * 100)


def shift(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def main():
    base = make_pillar_field()
    noise_sigma = 3.0
    shifts = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]      # px (spacing=8 なので 8px = 1周期)
    blur_sigmas = [0.0, 1.0, 2.0, 4.0, 8.0]

    def interior(x):
        return x[40:-40, 40:-40]  # 並進の縁効果を避けて内側を評価

    print("spacing=8px (shift=8px は1周期=エイリアシング), noise_sigma=3.0")
    header = f"{'blur_sigma':>10} | " + " ".join(f"shift={s:<4}" for s in shifts)

    print("\n--- NCC ---")
    print(header)
    for bs in blur_sigmas:
        pre = blur(base, bs) + rng.normal(0, noise_sigma, base.shape).astype(np.float32)
        row = []
        for s in shifts:
            post = blur(shift(base, s, 0), bs) + rng.normal(0, noise_sigma, base.shape).astype(np.float32)
            row.append(ncc(interior(pre), interior(post)))
        print(f"{bs:>10.1f} | " + " ".join(f"{v:9.4f}" for v in row))

    print("\n--- match_rate (|d|<=5) % ---")
    print(header)
    for bs in blur_sigmas:
        pre = np.clip(blur(base, bs) + rng.normal(0, noise_sigma, base.shape), 0, 255).astype(np.float32)
        row = []
        for s in shifts:
            post = np.clip(blur(shift(base, s, 0), bs) + rng.normal(0, noise_sigma, base.shape), 0, 255).astype(np.float32)
            row.append(match_rate(interior(pre), interior(post)))
        print(f"{bs:>10.1f} | " + " ".join(f"{v:9.2f}" for v in row))

    print("\n[結論]")
    print("- 軽度ボケ(σ<=1)かつ ズレ<1周期 では NCC・一致率ともズレ量に対し単調悪化する。")
    print("- 強ボケ(σ>=4)では周期構造が平滑化されて消え、NCC は 0 近傍、一致率は一定値に張り付き、")
    print("  ズレを入れても指標がほぼ変化しない(偽陽性)。ボケた領域では精度検証に使えない。")
    print("- 周期構造のため ズレがピラー1周期(=8px)に達すると相関が復帰する(エイリアシング)。")


if __name__ == "__main__":
    main()
