import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

try:
    import cv2
except Exception:
    cv2 = None

try:
    import pywt  # needed by skimage wavelet denoise
    from skimage.restoration import denoise_wavelet
except Exception:
    denoise_wavelet = None


BATCHES = [(f"WM_{i}", (i - 1) * 25 + 1, i * 25) for i in range(1, 9)]

EXPECTED_SIZES = {
    **{f"{i}.png": (256, 256) for i in range(1, 101)},
    **{f"{i}.png": (128, 128) for i in range(101, 126)},
    **{f"{i}.png": (256, 256) for i in range(126, 151)},
    **{f"{i}.png": (512, 512) for i in range(151, 201)},
}


def numeric_key(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return np.asarray(img.filter(ImageFilter.GaussianBlur(radius=sigma)), dtype=np.float32)


def residual_gauss(arr: np.ndarray, sigma: float) -> np.ndarray:
    r = arr - gaussian_blur(arr, sigma)
    r -= r.mean(axis=(0, 1), keepdims=True)
    return r


def residual_nlm(arr: np.ndarray, h: float) -> np.ndarray:
    if cv2 is None:
        return residual_gauss(arr, sigma=1.2)

    u8 = np.clip(arr, 0, 255).astype(np.uint8)
    den = cv2.fastNlMeansDenoisingColored(u8, None, h, h, 7, 21).astype(np.float32)
    r = arr - den
    r -= r.mean(axis=(0, 1), keepdims=True)
    return r


def residual_wavelet(arr: np.ndarray) -> np.ndarray:
    if denoise_wavelet is None:
        return residual_gauss(arr, sigma=1.0)

    x = np.clip(arr / 255.0, 0, 1)
    den = denoise_wavelet(
        x,
        channel_axis=-1,
        convert2ycbcr=True,
        method="BayesShrink",
        mode="soft",
        rescale_sigma=True,
    )
    r = arr - np.clip(den * 255.0, 0, 255)
    r -= r.mean(axis=(0, 1), keepdims=True)
    return r.astype(np.float32)


def residual_fft_band(arr: np.ndarray, low: float = 0.08, high: float = 0.55) -> np.ndarray:
    h, w, c = arr.shape

    yy, xx = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rr = rr / (min(h, w) / 2.0)

    mask = ((rr >= low) & (rr <= high)).astype(np.float32)

    out = np.zeros_like(arr, dtype=np.float32)
    for ch in range(c):
        f = np.fft.fftshift(np.fft.fft2(arr[:, :, ch]))
        b = np.fft.ifft2(np.fft.ifftshift(f * mask)).real
        out[:, :, ch] = b

    out -= out.mean(axis=(0, 1), keepdims=True)
    return out


def robust_std(x: np.ndarray) -> float:
    return float(np.median(np.abs(x - np.median(x))) * 1.4826 + 1e-6)


def normalize_pattern(p: np.ndarray) -> np.ndarray:
    p = p - p.mean(axis=(0, 1), keepdims=True)
    return p / robust_std(p)


def extract_residual(arr: np.ndarray, method: str, sigma: float, nlm_h: float) -> np.ndarray:
    if method == "gauss":
        return residual_gauss(arr, sigma)

    if method == "band":
        return residual_fft_band(arr)

    if method == "nlm":
        return residual_nlm(arr, nlm_h)

    if method == "wavelet":
        return residual_wavelet(arr)

    if method == "hybrid":
        parts = [
            1.0 * normalize_pattern(residual_gauss(arr, sigma)),
            1.0 * normalize_pattern(residual_fft_band(arr)),
            1.0 * normalize_pattern(residual_nlm(arr, nlm_h)),
        ]

        if denoise_wavelet is not None:
            parts.append(0.8 * normalize_pattern(residual_wavelet(arr)))

        return sum(parts) / len(parts)

    raise ValueError(f"Unknown method: {method}")


def aggregate_residuals(stack: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mean":
        return stack.mean(axis=0)

    if mode == "median":
        return np.median(stack, axis=0)

    if mode == "trim":
        sorted_stack = np.sort(stack, axis=0)
        k = max(1, int(round(0.2 * sorted_stack.shape[0])))
        return sorted_stack[k:-k].mean(axis=0)

    if mode == "sign":
        # Keeps only signs that are consistent across many source images.
        # Random image edges cancel more than the shared watermark direction.
        m = np.mean(np.sign(stack), axis=0)
        return np.sign(m) * (np.abs(m) ** 1.5)

    raise ValueError(f"Unknown aggregate mode: {mode}")


def build_pattern(dataset_dir: Path, wm_name: str, method: str, aggregate: str, sigma: float, nlm_h: float) -> np.ndarray:
    source_dir = dataset_dir / "watermarked_sources" / wm_name
    source_paths = sorted(source_dir.glob("*.png"), key=numeric_key)

    if len(source_paths) != 25:
        raise RuntimeError(f"Expected 25 source images in {source_dir}, found {len(source_paths)}")

    residuals = []
    for p in source_paths:
        arr = load_rgb(p)
        residual = extract_residual(arr, method=method, sigma=sigma, nlm_h=nlm_h)

        # Important: normalize each source residual before averaging.
        # This prevents one image's strong edges from dominating the pattern.
        residuals.append(normalize_pattern(residual))

    stack = np.stack(residuals, axis=0)
    pattern = aggregate_residuals(stack, mode=aggregate)
    pattern = normalize_pattern(pattern)
    return pattern.astype(np.float32)


def texture_mask(target: np.ndarray, strength: float, sigma: float = 1.2) -> np.ndarray:
    if strength <= 0:
        return np.ones(target.shape[:2] + (1,), dtype=np.float32)

    hp = np.abs(residual_gauss(target, sigma=sigma)).mean(axis=2)
    p95 = np.percentile(hp, 95) + 1e-6
    tex = np.clip(hp / p95, 0, 1)
    tex = tex - tex.mean()

    mask = np.clip(1.0 + strength * tex, 1.0 - strength, 1.0 + strength)
    return mask[:, :, None].astype(np.float32)


def forge_image(
    target: np.ndarray,
    pattern: np.ndarray,
    alpha: float,
    clip_delta: float,
    texture_strength: float,
    sat_aware: bool,
) -> np.ndarray:
    if target.shape != pattern.shape:
        raise ValueError(f"Shape mismatch: target={target.shape}, pattern={pattern.shape}")

    delta = alpha * pattern
    delta *= texture_mask(target, strength=texture_strength)

    if sat_aware:
        headroom = np.minimum(target, 255.0 - target)
        sat_mask = np.clip(headroom / max(clip_delta, 1.0), 0.35, 1.0)
        delta *= sat_mask

    delta = np.clip(delta, -clip_delta, clip_delta)
    forged = np.clip(target + delta, 0, 255).astype(np.uint8)
    return forged


def write_zip(folder: Path, out_zip: Path) -> None:
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx in range(1, 201):
            img_path = folder / f"{idx}.png"
            if not img_path.exists():
                raise FileNotFoundError(img_path)
            zf.write(img_path, arcname=img_path.name)


def validate_zip(zip_path: Path) -> None:
    expected_names = [f"{i}.png" for i in range(1, 201)]

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

        sorted_names = sorted(names, key=lambda x: int(Path(x).stem) if Path(x).stem.isdigit() else -1)
        if sorted_names != expected_names:
            missing = sorted(set(expected_names) - set(names), key=lambda x: int(Path(x).stem))
            extra = sorted(set(names) - set(expected_names))
            raise RuntimeError(f"Bad zip contents. Missing={missing[:5]}, Extra={extra[:5]}")

        for name in expected_names:
            if "/" in name or "\\" in name:
                raise RuntimeError(f"Subfolder detected: {name}")

            with zf.open(name) as fp:
                img = Image.open(fp)
                img.load()

            if img.mode != "RGB":
                raise RuntimeError(f"{name} mode is {img.mode}, expected RGB")

            if img.size != EXPECTED_SIZES[name]:
                raise RuntimeError(f"{name} size is {img.size}, expected {EXPECTED_SIZES[name]}")

    print(f"OK: {zip_path} is valid: 200 flat RGB PNG files.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("Dataset"))
    parser.add_argument("--work-dir", type=Path, default=Path("submission_v3_temp"))
    parser.add_argument("--out", type=Path, default=Path("submission_v3.zip"))

    parser.add_argument("--method", choices=["gauss", "band", "nlm", "wavelet", "hybrid"], default="band")
    parser.add_argument("--aggregate", choices=["mean", "median", "trim", "sign"], default="trim")

    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--sigma", type=float, default=1.6)
    parser.add_argument("--nlm-h", type=float, default=7.0)
    parser.add_argument("--clip", type=float, default=16.0)
    parser.add_argument("--texture", type=float, default=0.25)
    parser.add_argument("--sat-aware", action="store_true")

    args = parser.parse_args()

    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    args.work_dir.mkdir(parents=True)

    for wm_name, start, end in BATCHES:
        pattern = build_pattern(
            dataset_dir=args.dataset,
            wm_name=wm_name,
            method=args.method,
            aggregate=args.aggregate,
            sigma=args.sigma,
            nlm_h=args.nlm_h,
        )

        print(
            f"{wm_name}: {start}-{end}, "
            f"method={args.method}, aggregate={args.aggregate}, "
            f"pattern_std={pattern.std():.3f}"
        )

        for idx in range(start, end + 1):
            target_path = args.dataset / "clean_targets" / f"{idx}.png"
            target = load_rgb(target_path)

            forged = forge_image(
                target=target,
                pattern=pattern,
                alpha=args.alpha,
                clip_delta=args.clip,
                texture_strength=args.texture,
                sat_aware=args.sat_aware,
            )

            Image.fromarray(forged).save(args.work_dir / f"{idx}.png")

    write_zip(args.work_dir, args.out)
    validate_zip(args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()