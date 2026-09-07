#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import rasterio as rio
from rasterio.mask import mask
from rasterio.warp import transform_geom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop one raster window per point using a radius in km."
    )
    parser.add_argument("--raster", required=True, type=Path, help="Input raster path")
    parser.add_argument("--points", required=True, type=Path, help="Point GeoJSON path")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output folder")
    parser.add_argument(
        "--radius-km",
        type=float,
        default=10.0,
        help="Window radius around each point in kilometers",
    )
    parser.add_argument(
        "--label-field",
        type=str,
        default="site_code",
        help="Point property used for output naming",
    )
    parser.add_argument(
        "--all-touched",
        action="store_true",
        help="Use all touched pixels during mask",
    )
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "point"


def _load_point_features(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    features = data.get("features", [])
    out: list[dict] = []
    for feature in features:
        geom = feature.get("geometry")
        if not isinstance(geom, dict):
            continue
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates")
        if not isinstance(coords, list) or len(coords) < 2:
            continue
        out.append(feature)
    return out


def _square_polygon_wgs84(lon: float, lat: float, radius_km: float) -> dict:
    lat_delta = radius_km / 110.574
    cos_lat = math.cos(math.radians(lat))
    lon_scale = 111.320 * max(cos_lat, 1e-6)
    lon_delta = radius_km / lon_scale

    lon_min = lon - lon_delta
    lon_max = lon + lon_delta
    lat_min = lat - lat_delta
    lat_max = lat + lat_delta

    return {
        "type": "Polygon",
        "coordinates": [
            [
                [lon_min, lat_min],
                [lon_max, lat_min],
                [lon_max, lat_max],
                [lon_min, lat_max],
                [lon_min, lat_min],
            ]
        ],
    }


def main() -> int:
    args = parse_args()

    if not args.raster.exists():
        raise FileNotFoundError(f"Raster not found: {args.raster}")
    if not args.points.exists():
        raise FileNotFoundError(f"Points file not found: {args.points}")
    if args.radius_km <= 0:
        raise ValueError("--radius-km must be > 0")

    features = _load_point_features(args.points)
    if not features:
        raise ValueError(f"Points file has no valid point features: {args.points}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with rio.open(args.raster) as src:
        if src.crs is None:
            raise ValueError("Input raster has no CRS; cannot align point windows.")

        radius_label = str(args.radius_km).rstrip("0").rstrip(".").replace(".", "p")
        for idx, feature in enumerate(features, start=1):
            properties = feature.get("properties") or {}
            coords = feature["geometry"]["coordinates"]
            lon = float(coords[0])
            lat = float(coords[1])

            polygon = _square_polygon_wgs84(lon=lon, lat=lat, radius_km=float(args.radius_km))
            if str(src.crs) != "EPSG:4326":
                polygon = transform_geom("EPSG:4326", str(src.crs), polygon)

            raw_label = str(properties.get(args.label_field, f"pt_{idx}"))
            label = sanitize_name(raw_label)
            out_path = args.output_dir / f"{label}_r{radius_label}km.tif"

            cropped, transform = mask(
                src,
                [polygon],
                crop=True,
                all_touched=args.all_touched,
                nodata=src.nodata,
            )

            profile = src.profile.copy()
            profile.update(
                height=cropped.shape[1],
                width=cropped.shape[2],
                transform=transform,
                count=cropped.shape[0],
                compress="deflate",
                BIGTIFF="YES",
            )

            with rio.open(out_path, "w", **profile) as dst:
                dst.write(cropped)

            written += 1
            print(f"[crop-points] {raw_label} -> {out_path}")

    print(f"[crop-points] wrote {written} raster window(s) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
