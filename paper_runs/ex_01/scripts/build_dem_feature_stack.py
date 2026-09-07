#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio as rio
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter

# Make local src importable when running from repository checkout.
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xaigis.config import load_config
from xaigis.utils import ensure_parent


FEATURE_NAMES = [
    "DEM_ELEV",
    "DEM_SLOPE_DEG",
    "DEM_ASPECT_DEG",
    "DEM_CURVATURE",
    "DEM_RELIEF",
    "DEM_TPI",
    "DEM_ROUGHNESS",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a DEM-derived feature stack for ex_01 baseline modeling."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "paper_runs/ex_01/configs/ex_01_paper_run.json",
        help="Path to run config JSON.",
    )
    return parser.parse_args()


def _save_feature_names(path: Path, names: list[str]) -> None:
    out = ensure_parent(path)
    with out.open("w", encoding="utf-8") as f:
        json.dump({"feature_names": names}, f, indent=2)


def _metric_resolution_m(src: rio.io.DatasetReader) -> tuple[float, float]:
    transform = src.transform
    x_res = abs(float(transform.a))
    y_res = abs(float(transform.e))

    if src.crs is not None and src.crs.is_geographic:
        center_lat = (src.bounds.top + src.bounds.bottom) * 0.5
        lat_rad = math.radians(center_lat)
        x_res = x_res * 111320.0 * max(math.cos(lat_rad), 1e-6)
        y_res = y_res * 110574.0

    if x_res <= 0 or y_res <= 0:
        raise ValueError(f"Invalid pixel resolution: x_res={x_res}, y_res={y_res}")
    return x_res, y_res


def _window_size(value: Any, default: int) -> int:
    v = int(value if value is not None else default)
    return max(3, v if v % 2 == 1 else v + 1)


def build_dem_feature_stack(cfg: dict[str, Any]) -> dict[str, Any]:
    paths = cfg["paths"]
    dem_cfg = cfg.get("dem_features", {})

    dem_tif = paths.get("dem_tif")
    feature_stack_tif = ensure_parent(paths["feature_stack_tif"])
    feature_names_json = ensure_parent(paths["feature_names_json"])
    work_dir = ensure_parent(paths["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)

    if dem_tif is None:
        raise ValueError("paths.dem_tif is required for DEM baseline workflow.")
    if not Path(dem_tif).exists():
        raise FileNotFoundError(f"DEM raster not found: {dem_tif}")

    relief_window = _window_size(dem_cfg.get("relief_window_px"), default=15)
    tpi_window = _window_size(dem_cfg.get("tpi_window_px"), default=15)

    with rio.open(dem_tif) as src:
        dem = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata_in = src.nodata
        x_res_m, y_res_m = _metric_resolution_m(src)

        valid = np.isfinite(dem)
        if nodata_in is not None:
            valid &= dem != nodata_in
        if not np.any(valid):
            raise ValueError(f"No valid DEM pixels found in {dem_tif}")

        fill_value = float(np.nanmedian(dem[valid]))
        dem_filled = dem.copy()
        dem_filled[~valid] = fill_value

        gy, gx = np.gradient(dem_filled, y_res_m, x_res_m)
        slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
        aspect = (np.degrees(np.arctan2(-gx, gy)) + 360.0) % 360.0
        aspect = aspect.astype(np.float32)

        d2y = np.gradient(gy, y_res_m, axis=0)
        d2x = np.gradient(gx, x_res_m, axis=1)
        curvature = (d2x + d2y).astype(np.float32)

        local_max = maximum_filter(dem_filled, size=relief_window, mode="nearest")
        local_min = minimum_filter(dem_filled, size=relief_window, mode="nearest")
        relief = (local_max - local_min).astype(np.float32)

        local_mean = uniform_filter(dem_filled, size=tpi_window, mode="nearest")
        tpi = (dem_filled - local_mean).astype(np.float32)

        roughness = np.hypot(gx, gy).astype(np.float32)

        out_nodata = -9999.0
        bands = [dem_filled, slope, aspect, curvature, relief, tpi, roughness]
        for band in bands:
            band[~valid] = out_nodata

        blockx = min(256, src.width)
        blocky = min(256, src.height)
        blockx = max(16, (blockx // 16) * 16)
        blocky = max(16, (blocky // 16) * 16)

        profile.update(
            driver="GTiff",
            dtype="float32",
            count=len(FEATURE_NAMES),
            nodata=out_nodata,
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=blockx,
            blockysize=blocky,
            BIGTIFF="YES",
        )

        with rio.open(feature_stack_tif, "w", **profile) as dst:
            for i, (name, band) in enumerate(zip(FEATURE_NAMES, bands), start=1):
                dst.write(band.astype(np.float32), i)
                dst.set_band_description(i, name)

    _save_feature_names(Path(feature_names_json), FEATURE_NAMES)

    print(f"[dem] input DEM: {dem_tif}")
    print(f"[dem] feature stack: {feature_stack_tif}")
    print(f"[dem] features: {FEATURE_NAMES}")
    print(f"[dem] relief_window_px={relief_window}, tpi_window_px={tpi_window}")

    return {
        "dem_tif": str(dem_tif),
        "feature_stack_tif": str(feature_stack_tif),
        "feature_names_json": str(feature_names_json),
        "feature_count": len(FEATURE_NAMES),
    }


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config)
    build_dem_feature_stack(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
