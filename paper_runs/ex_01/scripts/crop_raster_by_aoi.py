#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import rasterio as rio
from rasterio.mask import mask
from rasterio.warp import transform_geom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop a raster using an AOI polygon GeoJSON."
    )
    parser.add_argument(
        "--raster",
        required=True,
        type=str,
        help="Input raster path or URL (http/https).",
    )
    parser.add_argument("--aoi", required=True, type=Path, help="AOI GeoJSON path")
    parser.add_argument("--out", required=True, type=Path, help="Output cropped raster path")
    parser.add_argument(
        "--all-touched",
        action="store_true",
        help="Use all touched pixels during mask",
    )
    return parser.parse_args()


def _infer_geojson_crs(geojson_data: dict) -> str:
    crs_obj = geojson_data.get("crs")
    if not isinstance(crs_obj, dict):
        return "EPSG:4326"
    props = crs_obj.get("properties")
    if not isinstance(props, dict):
        return "EPSG:4326"
    name = str(props.get("name", "")).strip()
    if not name:
        return "EPSG:4326"
    if "CRS84" in name.upper():
        return "EPSG:4326"
    return name


def _load_geometries(aoi_path: Path) -> tuple[list[dict], str]:
    with aoi_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    src_crs = _infer_geojson_crs(data)
    features = data.get("features", [])
    geometries: list[dict] = []
    for feature in features:
        geom = feature.get("geometry")
        if isinstance(geom, dict) and geom.get("type") and geom.get("coordinates"):
            geometries.append(geom)
    return geometries, src_crs


def main() -> int:
    args = parse_args()

    raster_arg = args.raster
    raster_is_url = raster_arg.startswith("http://") or raster_arg.startswith("https://")
    if not raster_is_url and not Path(raster_arg).exists():
        raise FileNotFoundError(f"Raster not found: {raster_arg}")
    if not args.aoi.exists():
        raise FileNotFoundError(f"AOI not found: {args.aoi}")

    geometries, aoi_crs = _load_geometries(args.aoi)
    if not geometries:
        raise ValueError(f"AOI contains no valid geometries: {args.aoi}")

    with rio.open(raster_arg) as src:
        if src.crs is None:
            raise ValueError("Input raster has no CRS; cannot align AOI.")
        if str(src.crs) != aoi_crs:
            geometries = [transform_geom(aoi_crs, str(src.crs), geom) for geom in geometries]

        cropped, transform = mask(
            src,
            geometries,
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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with rio.open(args.out, "w", **profile) as dst:
        dst.write(cropped)

    print(f"[crop-aoi] input:  {raster_arg}")
    print(f"[crop-aoi] aoi:    {args.aoi}")
    print(f"[crop-aoi] output: {args.out}")
    print(f"[crop-aoi] size:   {cropped.shape[2]} x {cropped.shape[1]} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
