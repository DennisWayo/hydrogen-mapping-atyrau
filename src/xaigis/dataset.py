from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.windows import Window

from .utils import ensure_parent, load_json


def sample_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    paths = cfg["paths"]
    dcfg = cfg["dataset"]
    tile_size = int(dcfg.get("tile_size", 256))
    max_per_tile = int(dcfg.get("max_per_tile", 3000))
    seed = int(dcfg.get("random_seed", 42))
    exclude_nodata = bool(dcfg.get("exclude_nodata", True))
    rng = np.random.default_rng(seed)

    stack_tif = paths["feature_stack_tif"]
    label_tif = paths["label_tif"]
    dataset_npz = ensure_parent(paths["dataset_npz"])
    dataset_csv = ensure_parent(paths["dataset_csv"])

    if not stack_tif.exists():
        raise FileNotFoundError(f"Feature stack not found: {stack_tif}")
    if not label_tif.exists():
        raise FileNotFoundError(f"Label raster not found: {label_tif}")

    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    sampled_tiles = 0
    skipped_tiles = 0
    finite_candidate_pixels = 0
    strict_candidate_pixels = 0
    finite_nodata_pixels = 0

    with rio.open(stack_tif) as src_x, rio.open(label_tif) as src_y:
        if src_x.width != src_y.width or src_x.height != src_y.height:
            raise ValueError("Feature stack and label raster dimensions do not match.")

        channels, height, width = src_x.count, src_x.height, src_x.width
        print(f"[dataset] scanning raster tiles (C={channels}, H={height}, W={width})")
        print(f"[dataset] exclude_nodata={exclude_nodata}, nodata={src_x.nodata}")

        for row in range(0, height, tile_size):
            h = min(tile_size, height - row)
            for col in range(0, width, tile_size):
                w = min(tile_size, width - col)
                win = Window(col_off=col, row_off=row, width=w, height=h)
                x_patch = src_x.read(window=win).astype(np.float32)  # (C,h,w)
                y_patch = src_y.read(1, window=win).astype(np.uint8)  # (h,w)

                finite_valid = np.isfinite(x_patch).all(axis=0)
                nodata_any = _nodata_any(x_patch, src_x.nodatavals)
                strict_valid = finite_valid & ~nodata_any
                valid = strict_valid if exclude_nodata else finite_valid
                finite_candidate_pixels += int(finite_valid.sum())
                strict_candidate_pixels += int(strict_valid.sum())
                finite_nodata_pixels += int((finite_valid & nodata_any).sum())
                if not np.any(valid):
                    skipped_tiles += 1
                    continue

                rr, cc = np.where(valid)
                n = rr.size
                if n == 0:
                    skipped_tiles += 1
                    continue

                if max_per_tile > 0 and n > max_per_tile:
                    keep = rng.choice(n, size=max_per_tile, replace=False)
                    rr, cc = rr[keep], cc[keep]

                x_sel = x_patch[:, rr, cc].T  # (n_keep, C)
                y_sel = y_patch[rr, cc]       # (n_keep,)
                x_chunks.append(x_sel)
                y_chunks.append(y_sel)
                sampled_tiles += 1

    if not x_chunks:
        mode = "finite and non-NoData" if exclude_nodata else "finite"
        raise RuntimeError(
            f"No valid samples extracted from stack/label rasters using {mode} validity. "
            "Run the coverage audit if this is unexpected."
        )

    x = np.vstack(x_chunks).astype(np.float32)
    y = np.concatenate(y_chunks).astype(np.uint8)

    np.savez_compressed(dataset_npz, X=x, y=y)
    print(f"[dataset] saved NPZ: {dataset_npz} -> X{x.shape}, y{y.shape}")

    feature_names = _load_feature_names(paths["feature_names_json"], x.shape[1])
    csv_cap = 200_000
    if x.shape[0] <= csv_cap:
        df = pd.DataFrame(x, columns=feature_names)
        df["label"] = y
        df.to_csv(dataset_csv, index=False)
        print(f"[dataset] saved CSV: {dataset_csv}")
    else:
        sample_idx = np.random.default_rng(seed).choice(x.shape[0], size=csv_cap, replace=False)
        df = pd.DataFrame(x[sample_idx], columns=feature_names)
        df["label"] = y[sample_idx]
        df.to_csv(dataset_csv, index=False)
        print(f"[dataset] saved sampled CSV ({csv_cap:,} rows): {dataset_csv}")

    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    print(f"[dataset] positives={positives:,}, negatives={negatives:,}")
    print(f"[dataset] sampled_tiles={sampled_tiles:,}, skipped_tiles={skipped_tiles:,}")
    print(f"[dataset] finite_candidate_pixels={finite_candidate_pixels:,}")
    print(f"[dataset] strict_candidate_pixels={strict_candidate_pixels:,}")
    print(f"[dataset] finite_pixels_with_nodata={finite_nodata_pixels:,}")
    return {
        "dataset_npz": str(dataset_npz),
        "dataset_csv": str(dataset_csv),
        "samples": int(x.shape[0]),
        "features": int(x.shape[1]),
        "positives": positives,
        "negatives": negatives,
        "exclude_nodata": exclude_nodata,
        "finite_candidate_pixels": finite_candidate_pixels,
        "strict_candidate_pixels": strict_candidate_pixels,
        "finite_pixels_with_nodata": finite_nodata_pixels,
    }


def _load_feature_names(path, n_features: int) -> list[str]:
    if path.exists():
        data = load_json(path)
        names = data.get("feature_names", [])
        if len(names) == n_features:
            return names
    return [f"f{i:02d}" for i in range(n_features)]


def _nodata_any(x_patch: np.ndarray, nodatavals: tuple[Any, ...] | list[Any]) -> np.ndarray:
    nodata_any = np.zeros(x_patch.shape[1:], dtype=bool)
    for band_idx, nodata in enumerate(nodatavals):
        if nodata is None:
            continue
        try:
            nodata_float = float(nodata)
        except (TypeError, ValueError):
            continue
        if np.isnan(nodata_float):
            continue
        nodata_any |= x_patch[band_idx] == nodata_float
    return nodata_any
