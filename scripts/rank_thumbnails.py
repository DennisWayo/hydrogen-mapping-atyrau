#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def score_thumbnail(path: Path) -> tuple[float, float, float, float, float, float]:
    im = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    mx = np.max(im, axis=2)
    mn = np.min(im, axis=2)
    sat = np.where(mx > 0, (mx - mn) / (mx + 1e-6), 0.0)
    bright = im.mean(axis=2)

    cloud = float(((bright > 0.82) & (sat < 0.18)).mean())
    haze = float(((bright > 0.70) & (sat < 0.25)).mean())
    gx = np.diff(bright, axis=1, prepend=bright[:, :1])
    gy = np.diff(bright, axis=0, prepend=bright[:1, :])
    texture = float(np.mean(np.hypot(gx, gy)))
    contrast = float(np.std(bright))
    color_std = float(np.mean(np.std(im, axis=(0, 1))))
    score = 2.4 * texture + 1.2 * contrast + 0.8 * color_std - 1.8 * cloud - 0.8 * haze
    return score, cloud, haze, texture, contrast, color_std


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank Sentinel-2 thumbnails by visual publication quality.")
    parser.add_argument("thumbnail_dir", type=Path)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--bottom", type=int, default=10)
    parser.add_argument("--ref-name", default="")
    parser.add_argument("--name-contains", default="")
    args = parser.parse_args()

    rows: list[tuple[float, float, float, float, float, float, str]] = []
    for p in sorted(args.thumbnail_dir.glob("*.jpg")):
        if args.name_contains and args.name_contains not in p.name:
            continue
        score, cloud, haze, texture, contrast, color_std = score_thumbnail(p)
        rows.append((score, cloud, haze, texture, contrast, color_std, p.name))
    if not rows:
        raise SystemExit("No thumbnails matched the filter.")
    rows.sort(reverse=True)

    print("TOP")
    for r in rows[: max(1, args.top)]:
        print(
            f"{r[6]} | score={r[0]:.4f} cloud={r[1]:.3f} haze={r[2]:.3f} "
            f"tex={r[3]:.4f} ctr={r[4]:.4f} color={r[5]:.4f}"
        )

    print("\nBOTTOM")
    for r in rows[-max(1, args.bottom) :]:
        print(
            f"{r[6]} | score={r[0]:.4f} cloud={r[1]:.3f} haze={r[2]:.3f} "
            f"tex={r[3]:.4f} ctr={r[4]:.4f} color={r[5]:.4f}"
        )

    if args.ref_name:
        for i, r in enumerate(rows, 1):
            if r[6] == args.ref_name:
                print(
                    "\nREF "
                    f"{args.ref_name} rank={i}/{len(rows)} | score={r[0]:.4f} "
                    f"cloud={r[1]:.3f} haze={r[2]:.3f} tex={r[3]:.4f} ctr={r[4]:.4f}"
                )
                break


if __name__ == "__main__":
    main()
