#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import rasterio as rio
from shapely.geometry import box


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create placeholder label polygons inside a feature stack extent."
    )
    parser.add_argument("--stack-tif", required=True, help="Feature stack GeoTIFF path")
    parser.add_argument("--out-geojson", required=True, help="Output label GeoJSON path")
    parser.add_argument(
        "--seed-size-frac",
        type=float,
        default=0.08,
        help="Relative polygon size (fraction of extent width/height)",
    )
    args = parser.parse_args()

    stack_path = Path(args.stack_tif).resolve()
    out_path = Path(args.out_geojson).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rio.open(stack_path) as src:
        bounds = src.bounds
        crs = src.crs

    xmin, ymin, xmax, ymax = bounds.left, bounds.bottom, bounds.right, bounds.top
    w = xmax - xmin
    h = ymax - ymin
    sx = max(w * args.seed_size_frac, 1.0)
    sy = max(h * args.seed_size_frac, 1.0)

    centers = [
        (xmin + 0.35 * w, ymin + 0.35 * h),
        (xmin + 0.65 * w, ymin + 0.65 * h),
    ]

    geoms = []
    props = []
    for i, (cx, cy) in enumerate(centers, start=1):
        geom = box(cx - sx / 2.0, cy - sy / 2.0, cx + sx / 2.0, cy + sy / 2.0)
        geoms.append(geom)
        props.append({"target": f"bootstrap_seed_{i}", "placeholder": True})

    gdf = gpd.GeoDataFrame(props, geometry=geoms, crs=crs)
    gdf.to_file(out_path, driver="GeoJSON")
    print(f"Saved bootstrap labels: {out_path}")
    print(f"CRS: {crs}")
    print(f"Bounds: {bounds}")


if __name__ == "__main__":
    main()
