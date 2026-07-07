import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from forge_v3 import (
    BATCHES,
    numeric_key,
    load_rgb,
    build_pattern,
    texture_mask,
    validate_zip,
)


def save_rgb(arr, path):
    arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def build_avg_image(dataset_dir: Path, wm_name: str, mode: str):
    source_dir = dataset_dir / "watermarked_sources" / wm_name
    paths = sorted(source_dir.glob("*.png"), key=numeric_key)

    imgs = np.stack([load_rgb(p) for p in paths], axis=0)

    if mode == "mean":
        return imgs.mean(axis=0).astype(np.float32)

    if mode == "median":
        return np.median(imgs, axis=0).astype(np.float32)

    if mode == "trim":
        imgs = np.sort(imgs, axis=0)
        k = max(1, int(round(0.2 * imgs.shape[0])))
        return imgs[k:-k].mean(axis=0).astype(np.float32)

    raise ValueError(mode)


def forge_one(
    target,
    avg_img,
    nlm_pattern,
    lambda_avg,
    residual_alpha,
    low_clip,
    total_clip,
    texture_strength,
):
    # Low/mid-frequency simple averaging component
    low_delta = lambda_avg * (avg_img - target)
    low_delta = np.clip(low_delta, -low_clip, low_clip)

    # Best-known high-frequency NLM component
    high_delta = residual_alpha * nlm_pattern
    high_delta = high_delta * texture_mask(target, strength=texture_strength)

    delta = low_delta + high_delta
    delta = np.clip(delta, -total_clip, total_clip)

    return np.clip(target + delta, 0, 255)


def write_zip(folder: Path, out_zip: Path):
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(1, 201):
            p = folder / f"{i}.png"
            zf.write(p, arcname=p.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("Dataset"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)

    parser.add_argument("--avg-mode", choices=["mean", "median", "trim"], default="mean")
    parser.add_argument("--lambda-avg", type=float, default=0.08)
    parser.add_argument("--residual-alpha", type=float, default=4.0)
    parser.add_argument("--low-clip", type=float, default=18.0)
    parser.add_argument("--total-clip", type=float, default=24.0)
    parser.add_argument("--texture", type=float, default=0.25)

    args = parser.parse_args()

    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    args.work_dir.mkdir(parents=True)

    for wm_name, start, end in BATCHES:
        print(
            f"{wm_name}: avg={args.avg_mode}, "
            f"lambda={args.lambda_avg}, residual_alpha={args.residual_alpha}"
        )

        avg_img = build_avg_image(args.dataset, wm_name, args.avg_mode)

        nlm_pattern = build_pattern(
            dataset_dir=args.dataset,
            wm_name=wm_name,
            method="nlm",
            aggregate="trim",
            sigma=1.6,
            nlm_h=7.0,
        )

        for idx in range(start, end + 1):
            target = load_rgb(args.dataset / "clean_targets" / f"{idx}.png")

            forged = forge_one(
                target=target,
                avg_img=avg_img,
                nlm_pattern=nlm_pattern,
                lambda_avg=args.lambda_avg,
                residual_alpha=args.residual_alpha,
                low_clip=args.low_clip,
                total_clip=args.total_clip,
                texture_strength=args.texture,
            )

            save_rgb(forged, args.work_dir / f"{idx}.png")

    write_zip(args.work_dir, args.out)
    validate_zip(args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()