#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio as rio
from PIL import Image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export before/after GeoTIFF site windows to PNG."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("paper_runs/ex_01/runs/ex_01/scenes/before_after"),
        help="Root folder containing before_dem, after_prob, after_mask GeoTIFF folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("paper_runs/ex_01/runs/ex_01/scenes/before_after_png"),
        help="Output root folder for PNG files.",
    )
    return parser.parse_args()


def _stretch_uint8(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not np.any(valid):
        return np.zeros(arr.shape, dtype=np.uint8)
    vals = arr[valid]
    lo = float(np.quantile(vals, 0.02))
    hi = float(np.quantile(vals, 0.98))
    if hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
    if hi <= lo:
        out = np.zeros(arr.shape, dtype=np.uint8)
        out[valid] = 128
        return out
    scaled = (arr - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    out = (scaled * 255.0).astype(np.uint8)
    out[~valid] = 0
    return out


def _to_png_array(arr: np.ndarray, valid: np.ndarray, layer_name: str) -> np.ndarray:
    if layer_name == "after_mask":
        out = np.zeros(arr.shape, dtype=np.uint8)
        out[valid] = (arr[valid] > 0).astype(np.uint8) * 255
        return out
    if layer_name == "after_prob":
        out = np.zeros(arr.shape, dtype=np.uint8)
        prob = np.clip(arr, 0.0, 1.0)
        out[valid] = (prob[valid] * 255.0).astype(np.uint8)
        return out
    return _stretch_uint8(arr, valid)


def _convert_one(src_path: Path, dst_path: Path, layer_name: str) -> None:
    with rio.open(src_path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= arr != nodata

    out = _to_png_array(arr=arr, valid=valid, layer_name=layer_name)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, mode="L").save(dst_path)


def main() -> int:
    args = _parse_args()
    input_root = args.input_root
    output_root = args.output_root

    layer_dirs = ["before_dem", "after_prob", "after_mask"]
    total = 0

    for layer in layer_dirs:
        in_dir = input_root / layer
        if not in_dir.exists():
            continue
        for tif_path in sorted(in_dir.glob("*.tif")):
            out_dir = output_root / layer
            png_name = tif_path.with_suffix(".png").name
            out_path = out_dir / png_name
            _convert_one(src_path=tif_path, dst_path=out_path, layer_name=layer)
            total += 1
            print(f"[png] {tif_path} -> {out_path}")

    print(f"[png] wrote {total} png file(s) to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
