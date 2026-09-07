#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlparse
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_geom
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform
from shapely.geometry import GeometryCollection, shape

# Local imports from repo.
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xaigis.config import load_config


STAC_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
EARTH_SEARCH_ITEM_URL = "https://earth-search.aws.element84.com/v1/collections/{collection}/items/{item_id}"
STAC_USER_AGENT = "XaiGis-ex01-multisource/1.0"
PC_STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PC_STAC_ITEM_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/{collection}/items/{item_id}"
PC_TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}"
NODATA_OUT = -9999.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and align Sentinel/Landsat stacks for ex_01 from Earth Search STAC."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("paper_runs/ex_01/configs/ex_01_paper_run.json"),
        help="Path to ex_01 config.",
    )
    parser.add_argument(
        "--datetime",
        type=str,
        default="2025-01-01T00:00:00Z/2025-12-31T23:59:59Z",
        help="STAC datetime range.",
    )
    parser.add_argument(
        "--cloud-max",
        type=float,
        default=20.0,
        help="Maximum cloud cover for STAC query.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=120,
        help="Maximum scenes returned per collection query.",
    )
    parser.add_argument(
        "--max-scenes-per-collection",
        type=int,
        default=20,
        help="Maximum scenes to mosaic for each collection.",
    )
    parser.add_argument(
        "--target-coverage",
        type=float,
        default=0.99,
        help="Greedy AOI footprint coverage target before scene selection stops.",
    )
    parser.add_argument(
        "--min-added-coverage",
        type=float,
        default=0.001,
        help="Minimum added AOI coverage fraction required after the first selected scene.",
    )
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Query/select scenes and write selection metadata without downloading/reprojecting assets.",
    )
    parser.add_argument(
        "--use-selected-scenes",
        action="store_true",
        help="Reuse paths.multisource_selection_json scene IDs instead of querying/selecting a new scene set.",
    )
    parser.add_argument(
        "--collection",
        choices=["all", "sentinel", "landsat"],
        default="all",
        help="Generate only one source stack or both.",
    )
    parser.add_argument(
        "--sentinel-provider",
        choices=["earth-search", "planetary-computer"],
        default="earth-search",
        help="STAC/provider endpoint for Sentinel-2 assets.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["full-aoi", "site-buffer"],
        default="full-aoi",
        help="Read remote rasters over the full scene footprint or only around labeled site buffers.",
    )
    parser.add_argument(
        "--site-buffer-km",
        type=float,
        default=25.0,
        help="Metric buffer around each label site when --target-mode=site-buffer.",
    )
    parser.add_argument(
        "--site-column",
        default="site_name",
        help="Site column in paths.geology_geojson when --target-mode=site-buffer.",
    )
    parser.add_argument(
        "--target-tile-size",
        type=int,
        default=768,
        help="Split target windows into tiles of this size. Use 0 to disable splitting.",
    )
    parser.add_argument(
        "--cache-assets-dir",
        type=Path,
        default=None,
        help="Optional local cache directory for remote COG assets before reprojection.",
    )
    return parser.parse_args()


def _load_aoi_geometry(aoi_path: Path) -> dict[str, Any]:
    with aoi_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    feats = data.get("features", [])
    if not feats:
        raise ValueError(f"AOI GeoJSON has no features: {aoi_path}")
    geom = feats[0].get("geometry")
    if not isinstance(geom, dict):
        raise ValueError(f"AOI geometry missing in {aoi_path}")
    return geom


def _post_json(url: str, payload: dict[str, Any], retries: int = 3) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": STAC_USER_AGENT,
        },
    )
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                break
            wait_s = 2 * attempt
            print(f"[fetch] warning: STAC search failed attempt {attempt}/{retries}: {exc}; retrying in {wait_s}s")
            time.sleep(wait_s)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"STAC search failed unexpectedly: {url}")


def _get_json(url: str, retries: int = 3) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": STAC_USER_AGENT})
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                break
            wait_s = 2 * attempt
            print(f"[fetch] warning: metadata fetch failed attempt {attempt}/{retries}: {exc}; retrying in {wait_s}s")
            time.sleep(wait_s)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Metadata fetch failed unexpectedly: {url}")


def _query_items(
    collection: str,
    bbox: list[float],
    datetime_range: str,
    cloud_max: float,
    limit: int,
    search_url: str = STAC_SEARCH_URL,
) -> list[dict[str, Any]]:
    payload = {
        "collections": [collection],
        "bbox": bbox,
        "datetime": datetime_range,
        "limit": limit,
        "query": {"eo:cloud_cover": {"lt": cloud_max}},
    }
    result = _post_json(search_url, payload)
    return list(result.get("features", []))


def _query_items_by_ids(
    collection: str,
    item_ids: list[str],
    search_url: str = STAC_SEARCH_URL,
) -> list[dict[str, Any]]:
    if not item_ids:
        return []
    payload = {
        "collections": [collection],
        "ids": item_ids,
        "limit": len(item_ids),
    }
    result = _post_json(search_url, payload)
    items = list(result.get("features", []))
    by_id = {str(item.get("id")): item for item in items}
    missing = [item_id for item_id in item_ids if item_id not in by_id]
    if missing:
        raise RuntimeError(f"STAC search did not return selected items for {collection}: {missing}")
    return [by_id[item_id] for item_id in item_ids]


def _fetch_earth_search_item(collection: str, item_id: str) -> dict[str, Any]:
    url = EARTH_SEARCH_ITEM_URL.format(collection=collection, item_id=item_id)
    return _get_json(url)


def _load_selected_scene_ids(selection_json: Path) -> tuple[list[str], list[str], float, float]:
    with selection_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    sentinel = payload.get("sentinel") or {}
    landsat = payload.get("landsat") or {}
    sentinel_ids = [str(row["id"]) for row in sentinel.get("items", []) if row.get("id")]
    landsat_ids = [str(row["id"]) for row in landsat.get("items", []) if row.get("id")]
    if not sentinel_ids and not landsat_ids:
        raise ValueError(f"No selected scene IDs found in {selection_json}")
    return (
        sentinel_ids,
        landsat_ids,
        float(sentinel.get("footprint_coverage_fraction", 0.0)),
        float(landsat.get("footprint_coverage_fraction", 0.0)),
    )


def _select_covering_items(
    items: list[dict[str, Any]],
    required_assets: list[str],
    aoi_geom: dict[str, Any],
    max_scenes: int,
    target_coverage: float,
    min_added_coverage: float,
) -> tuple[list[dict[str, Any]], float]:
    aoi = shape(aoi_geom)
    candidates: list[dict[str, Any]] = []
    for item in items:
        assets = item.get("assets", {})
        if not all(k in assets for k in required_assets):
            continue
        geom = item.get("geometry")
        if not isinstance(geom, dict):
            continue
        footprint = shape(geom)
        inter = footprint.intersection(aoi)
        if inter.is_empty:
            continue
        overlap_area = float(inter.area)
        overlap = float(overlap_area / max(aoi.area, 1e-12))
        cloud = float(item.get("properties", {}).get("eo:cloud_cover", 100.0))
        candidates.append(
            {
                "item": item,
                "footprint": inter,
                "overlap_area": overlap_area,
                "overlap_fraction": overlap,
                "cloud": cloud,
            }
        )

    if not candidates:
        raise RuntimeError("No usable STAC items found matching required assets and AOI overlap.")

    selected: list[dict[str, Any]] = []
    covered = GeometryCollection()
    aoi_area = max(float(aoi.area), 1e-12)
    target_area = aoi_area * max(0.0, min(float(target_coverage), 1.0))
    remaining = list(candidates)

    while remaining and len(selected) < max(1, int(max_scenes)):
        scored: list[tuple[float, float, float, int, dict[str, Any]]] = []
        for idx, candidate in enumerate(remaining):
            added_geom = candidate["footprint"].difference(covered)
            added_area = float(added_geom.area)
            scored.append(
                (
                    added_area,
                    -float(candidate["cloud"]),
                    float(candidate["overlap_area"]),
                    -idx,
                    candidate,
                )
            )
        scored.sort(reverse=True)
        added_area, _, _, neg_idx, best = scored[0]
        added_fraction = float(added_area / aoi_area)
        if selected and added_fraction < min_added_coverage:
            break

        selected.append(best["item"])
        covered = covered.union(best["footprint"])
        remaining.pop(-neg_idx)
        coverage = float(covered.area / aoi_area)
        print(
            "[fetch] selected",
            best["item"].get("collection"),
            best["item"].get("id"),
            f"added_coverage={added_fraction:.4f}",
            f"total_coverage={coverage:.4f}",
            f"cloud={best['cloud']}",
        )
        if float(covered.area) >= target_area:
            break

    return selected, float(covered.area / aoi_area)


def _asset_scale_offset(asset: dict[str, Any]) -> tuple[float, float]:
    bands = asset.get("raster:bands")
    if isinstance(bands, list) and bands:
        b0 = bands[0] if isinstance(bands[0], dict) else {}
        scale = float(b0.get("scale", 1.0))
        offset = float(b0.get("offset", 0.0))
        return scale, offset
    return 1.0, 0.0


def _asset_nodata(asset: dict[str, Any], src_nodata: float | int | None) -> float | int | None:
    if src_nodata is not None:
        return src_nodata
    bands = asset.get("raster:bands")
    if isinstance(bands, list) and bands:
        b0 = bands[0] if isinstance(bands[0], dict) else {}
        if "nodata" in b0:
            return b0["nodata"]
    return None


def _window_for_bounds(
    dst_transform: Any,
    width: int,
    height: int,
    bounds: tuple[float, float, float, float],
) -> Window:
    win = from_bounds(*bounds, transform=dst_transform)
    col0 = max(0, int(np.floor(win.col_off)))
    row0 = max(0, int(np.floor(win.row_off)))
    col1 = min(width, int(np.ceil(win.col_off + win.width)))
    row1 = min(height, int(np.ceil(win.row_off + win.height)))
    if col1 <= col0 or row1 <= row0:
        raise ValueError(f"Item footprint does not overlap destination grid: {bounds}")
    return Window(col_off=col0, row_off=row0, width=col1 - col0, height=row1 - row0)


def _intersect_windows(a: Window, b: Window) -> Window | None:
    col0 = max(int(a.col_off), int(b.col_off))
    row0 = max(int(a.row_off), int(b.row_off))
    col1 = min(int(a.col_off + a.width), int(b.col_off + b.width))
    row1 = min(int(a.row_off + a.height), int(b.row_off + b.height))
    if col1 <= col0 or row1 <= row0:
        return None
    return Window(col_off=col0, row_off=row0, width=col1 - col0, height=row1 - row0)


def _item_destination_window(
    item: dict[str, Any],
    dst_crs: Any,
    dst_transform: Any,
    width: int,
    height: int,
) -> Window:
    geom = item.get("geometry")
    if not isinstance(geom, dict):
        raise ValueError(f"Item geometry missing for {item.get('id')}")
    if dst_crs is not None and str(dst_crs).upper() != "EPSG:4326":
        geom = transform_geom("EPSG:4326", dst_crs, geom)
    bounds = shape(geom).bounds
    return _window_for_bounds(dst_transform=dst_transform, width=width, height=height, bounds=bounds)


def _union_geometry(gdf: gpd.GeoDataFrame) -> Any:
    if hasattr(gdf.geometry, "union_all"):
        return gdf.geometry.union_all()
    return gdf.geometry.unary_union


def _utm_crs_for_geometry(gdf: gpd.GeoDataFrame) -> str:
    wgs = gdf.to_crs("EPSG:4326")
    centroid = wgs.geometry.unary_union.centroid
    lon = float(centroid.x)
    lat = float(centroid.y)
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def _target_windows_from_sites(
    template_tif: Path,
    label_geojson: Path,
    site_column: str,
    buffer_km: float,
) -> list[tuple[str, Window]]:
    label_gdf = gpd.read_file(label_geojson)
    if label_gdf.empty:
        raise ValueError(f"No label polygons found: {label_geojson}")
    if site_column not in label_gdf.columns:
        raise ValueError(f"Missing site column '{site_column}' in {label_geojson}")

    windows: list[tuple[str, Window]] = []
    with rio.open(template_tif) as tpl:
        if label_gdf.crs is None:
            label_gdf = label_gdf.set_crs(tpl.crs)
        elif label_gdf.crs != tpl.crs:
            label_gdf = label_gdf.to_crs(tpl.crs)

        for site_name in sorted(str(site) for site in label_gdf[site_column].dropna().unique()):
            site_gdf = label_gdf[label_gdf[site_column].astype(str) == site_name]
            metric_crs = _utm_crs_for_geometry(site_gdf)
            buffered = _union_geometry(site_gdf.to_crs(metric_crs)).buffer(buffer_km * 1000.0)
            buffered_gdf = gpd.GeoDataFrame({"geometry": [buffered]}, crs=metric_crs)
            geom = buffered_gdf.to_crs(tpl.crs).geometry.iloc[0]
            win = _window_for_bounds(
                dst_transform=tpl.transform,
                width=tpl.width,
                height=tpl.height,
                bounds=geom.bounds,
            )
            windows.append((site_name, win))
            print(f"[fetch] target window {site_name}: {int(win.width)}x{int(win.height)}")
    return windows


def _split_window(name: str, win: Window, tile_size: int) -> list[tuple[str, Window]]:
    if tile_size <= 0:
        return [(name, win)]
    out: list[tuple[str, Window]] = []
    row_start = int(win.row_off)
    col_start = int(win.col_off)
    row_end = int(win.row_off + win.height)
    col_end = int(win.col_off + win.width)
    for row in range(row_start, row_end, tile_size):
        h = min(tile_size, row_end - row)
        for col in range(col_start, col_end, tile_size):
            w = min(tile_size, col_end - col)
            tile_name = f"{name}_r{row - row_start}_c{col - col_start}"
            out.append((tile_name, Window(col_off=col, row_off=row, width=w, height=h)))
    return out


def _split_target_windows(windows: list[tuple[str, Window]], tile_size: int) -> list[tuple[str, Window]]:
    out: list[tuple[str, Window]] = []
    for name, win in windows:
        out.extend(_split_window(name, win, tile_size))
    print(f"[fetch] target tiles: {len(out)}")
    return out


def _cache_asset(href: str, item: dict[str, Any], key: str, cache_dir: Path | None) -> str:
    if cache_dir is None:
        return href

    item_id = str(item.get("id") or "unknown_item")
    collection = str(item.get("collection") or "unknown_collection")
    basename = Path(urlparse(href).path).name or f"{key}.tif"
    out = cache_dir / collection / item_id / f"{key}_{basename}"
    done = out.with_suffix(out.suffix + ".done")
    out.parent.mkdir(parents=True, exist_ok=True)

    if done.exists() and out.exists() and out.stat().st_size > 0:
        print(f"[fetch] cache hit: {out}")
        return str(out)

    cmd = [
        "curl",
        "--fail",
        "--location",
        "--retry",
        "6",
        "--retry-delay",
        "3",
        "--connect-timeout",
        "30",
        "--speed-limit",
        "50000",
        "--speed-time",
        "60",
        "--silent",
        "--show-error",
    ]
    if out.exists() and out.stat().st_size > 0:
        cmd.extend(["--continue-at", "-"])
    cmd.extend(["--output", str(out), href])

    print(f"[fetch] caching asset: {item_id} {key} -> {out}")
    sys.stdout.flush()
    subprocess.run(cmd, check=True)
    done.write_text(str(out.stat().st_size), encoding="utf-8")
    print(f"[fetch] cached asset bytes={out.stat().st_size:,}: {out}")
    sys.stdout.flush()
    return str(out)


def _write_aligned_stack(
    template_tif: Path,
    items: list[dict[str, Any]],
    out_tif: Path,
    band_keys: list[str],
    band_names: list[str],
    target_windows: list[tuple[str, Window]] | None = None,
    cache_assets_dir: Path | None = None,
) -> None:
    if not items:
        raise ValueError("At least one item is required to write an aligned stack.")
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    tmp_tif = out_tif.with_name(f"{out_tif.stem}.tmp{out_tif.suffix}")
    if tmp_tif.exists():
        tmp_tif.unlink()

    with rio.open(template_tif) as tpl:
        profile = tpl.profile.copy()
        dst_crs = tpl.crs
        dst_transform = tpl.transform
        height, width = tpl.height, tpl.width

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=len(band_keys),
        nodata=NODATA_OUT,
        compress="deflate",
        predictor=3,
        tiled=True,
        BIGTIFF="YES",
    )

    with rio.open(tmp_tif, "w", **profile) as dst:
        for i, (key, band_name) in enumerate(zip(band_keys, band_names), start=1):
            mosaic = np.full((height, width), NODATA_OUT, dtype=np.float32)
            filled = np.zeros((height, width), dtype=bool)

            for scene_idx, item in enumerate(items, start=1):
                asset = item["assets"][key]
                href = _cache_asset(asset["href"], item=item, key=key, cache_dir=cache_assets_dir)
                try:
                    scene_win = _item_destination_window(
                        item=item,
                        dst_crs=dst_crs,
                        dst_transform=dst_transform,
                        width=width,
                        height=height,
                    )
                except ValueError as exc:
                    print(f"[fetch] {band_name}: scene {scene_idx}/{len(items)} skipped: {exc}")
                    continue

                if target_windows is None:
                    work_windows = [(item.get("id", f"scene_{scene_idx}"), scene_win)]
                else:
                    work_windows = []
                    for target_name, target_win in target_windows:
                        dst_win = _intersect_windows(scene_win, target_win)
                        if dst_win is not None:
                            work_windows.append((target_name, dst_win))
                if not work_windows:
                    print(f"[fetch] {band_name}: scene {scene_idx}/{len(items)} has no target-window overlap")
                    continue

                src = None
                with rio.Env(
                    AWS_NO_SIGN_REQUEST="YES",
                    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                    GDAL_HTTP_CONNECTTIMEOUT="30",
                    GDAL_HTTP_TIMEOUT="120",
                    GDAL_HTTP_MULTIRANGE="YES",
                    VSI_CACHE="TRUE",
                    VSI_CACHE_SIZE=50_000_000,
                ):
                    for open_attempt in range(1, 4):
                        try:
                            src = rio.open(href)
                            break
                        except Exception as exc:
                            if open_attempt >= 3:
                                print(
                                    f"[fetch] warning: skipped {band_name} scene={item.get('id')} "
                                    f"after raster open failure: {exc}"
                                )
                                sys.stdout.flush()
                                src = None
                                break
                            wait_s = 2 * open_attempt
                            print(
                                f"[fetch] warning: raster open failed for {band_name} scene={item.get('id')} "
                                f"attempt {open_attempt}/3: {exc}; retrying in {wait_s}s"
                            )
                            sys.stdout.flush()
                            time.sleep(wait_s)
                if src is None:
                    continue
                with src:
                    src_nodata = _asset_nodata(asset=asset, src_nodata=src.nodata)
                    for target_name, dst_win in work_windows:
                        win_h = int(dst_win.height)
                        win_w = int(dst_win.width)
                        row0 = int(dst_win.row_off)
                        col0 = int(dst_win.col_off)
                        row1 = row0 + win_h
                        col1 = col0 + win_w
                        filled_win = filled[row0:row1, col0:col1]
                        if filled_win.all():
                            continue

                        arr = np.full((win_h, win_w), NODATA_OUT, dtype=np.float32)
                        dst_win_transform = window_transform(dst_win, dst_transform)
                        print(
                            f"[fetch] {band_name}: scene {scene_idx}/{len(items)} "
                            f"{item.get('id')} target={target_name} window={win_w}x{win_h}"
                        )
                        sys.stdout.flush()

                        try:
                            reproject(
                                source=rio.band(src, 1),
                                destination=arr,
                                src_transform=src.transform,
                                src_crs=src.crs,
                                src_nodata=src_nodata,
                                dst_transform=dst_win_transform,
                                dst_crs=dst_crs,
                                dst_nodata=NODATA_OUT,
                                resampling=Resampling.bilinear,
                                num_threads=2,
                            )
                        except Exception as exc:
                            print(
                                f"[fetch] warning: skipped {band_name} scene={item.get('id')} "
                                f"target={target_name} after remote read/reproject failure: {exc}"
                            )
                            sys.stdout.flush()
                            continue

                        valid = np.isfinite(arr) & (arr != NODATA_OUT)
                        if src_nodata is not None:
                            valid &= arr != src_nodata

                        scale, offset = _asset_scale_offset(asset)
                        if scale != 1.0 or offset != 0.0:
                            arr[valid] = arr[valid] * scale + offset

                        mosaic_win = mosaic[row0:row1, col0:col1]
                        fill = valid & ~filled_win
                        mosaic_win[fill] = arr[fill]
                        filled_win[fill] = True
                        print(
                            f"[fetch] {band_name}: scene {scene_idx}/{len(items)} "
                            f"{item.get('id')} target={target_name} filled {int(fill.sum()):,} new pixels"
                        )
                        sys.stdout.flush()

            dst.write(mosaic, i)
            dst.set_band_description(i, band_name)
            print(f"[fetch] wrote band {band_name}; strict coverage={float(filled.mean()):.4f}")

    tmp_tif.replace(out_tif)
    print(f"[fetch] stack written: {out_tif}")


def _save_selection_metadata(
    out_json: Path,
    sentinel_items: list[dict[str, Any]],
    landsat_items: list[dict[str, Any]],
    sentinel_coverage: float,
    landsat_coverage: float,
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)

    def one(item: dict[str, Any]) -> dict[str, Any]:
        props = item.get("properties", {})
        return {
            "id": item.get("id"),
            "datetime": props.get("datetime"),
            "eo_cloud_cover": props.get("eo:cloud_cover"),
            "collection": item.get("collection"),
        }

    payload = {
        "sentinel": {
            "footprint_coverage_fraction": sentinel_coverage,
            "items": [one(item) for item in sentinel_items],
        },
        "landsat": {
            "footprint_coverage_fraction": landsat_coverage,
            "items": [one(item) for item in landsat_items],
        },
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[fetch] selection metadata: {out_json}")


def _fetch_pc_landsat_item(item_id: str) -> dict[str, Any]:
    url = PC_STAC_ITEM_URL.format(collection="landsat-c2-l2", item_id=item_id)
    req = urllib.request.Request(url, headers={"User-Agent": STAC_USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_pc_token(collection: str) -> str:
    url = PC_TOKEN_URL.format(collection=collection)
    req = urllib.request.Request(url, headers={"User-Agent": STAC_USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    token = payload.get("token")
    if not token:
        raise RuntimeError(f"Planetary Computer token response missing token for {collection}.")
    return str(token)


def _signed_pc_item_assets(item: dict[str, Any], collection: str) -> dict[str, Any]:
    token = _fetch_pc_token(collection)
    out = dict(item)
    out_assets: dict[str, Any] = {}
    for key, asset in (item.get("assets") or {}).items():
        a = dict(asset)
        href = a.get("href")
        if isinstance(href, str) and href.startswith("https://"):
            sep = "&" if "?" in href else "?"
            a["href"] = f"{href}{sep}{token}"
        out_assets[key] = a
    out["assets"] = out_assets
    return out


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config)
    paths = cfg["paths"]
    region = cfg["region"]
    cfg_base = Path(cfg["__config_path__"]).parent

    dem_tif = Path(paths["dem_tif"])
    sentinel_out = Path(paths["sentinel_stack_tif"])
    landsat_out = Path(paths["landsat_stack_tif"])
    selection_json = Path(paths["multisource_selection_json"])
    label_geojson = Path(paths["geology_geojson"])

    if not dem_tif.exists():
        raise FileNotFoundError(f"DEM template not found: {dem_tif}")
    if not label_geojson.is_absolute():
        label_geojson = (cfg_base / label_geojson).resolve()

    aoi_path = Path(region["aoi_geojson"])
    if not aoi_path.is_absolute():
        aoi_path = (cfg_base / aoi_path).resolve()
    aoi_geom = _load_aoi_geometry(aoi_path)
    xs = [p[0] for p in aoi_geom["coordinates"][0]]
    ys = [p[1] for p in aoi_geom["coordinates"][0]]
    bbox = [min(xs), min(ys), max(xs), max(ys)]

    if args.use_selected_scenes:
        if not selection_json.exists():
            raise FileNotFoundError(f"Selected-scenes metadata not found: {selection_json}")
        sentinel_ids, landsat_ids, sentinel_coverage, landsat_coverage = _load_selected_scene_ids(selection_json)
        print(f"[fetch] loading selected scene IDs from {selection_json}")
        sentinel_items = []
        landsat_items = []
        if args.collection in {"all", "sentinel"}:
            sentinel_search_url = PC_STAC_SEARCH_URL if args.sentinel_provider == "planetary-computer" else STAC_SEARCH_URL
            sentinel_items = _query_items_by_ids("sentinel-2-l2a", sentinel_ids, search_url=sentinel_search_url)
        if args.collection in {"all", "landsat"}:
            landsat_items = _query_items_by_ids("landsat-c2-l2", landsat_ids)
    else:
        sentinel_items = []
        landsat_items = []
        sentinel_coverage = 0.0
        landsat_coverage = 0.0
        if args.collection in {"all", "sentinel"}:
            sentinel_search_url = PC_STAC_SEARCH_URL if args.sentinel_provider == "planetary-computer" else STAC_SEARCH_URL
            sentinel_required_assets = (
                ["B02", "B03", "B04", "B08"]
                if args.sentinel_provider == "planetary-computer"
                else ["blue", "green", "red", "nir"]
            )
            sentinel_query_items = _query_items(
                collection="sentinel-2-l2a",
                bbox=bbox,
                datetime_range=args.datetime,
                cloud_max=args.cloud_max,
                limit=args.limit,
                search_url=sentinel_search_url,
            )
            sentinel_items, sentinel_coverage = _select_covering_items(
                items=sentinel_query_items,
                required_assets=sentinel_required_assets,
                aoi_geom=aoi_geom,
                max_scenes=args.max_scenes_per_collection,
                target_coverage=args.target_coverage,
                min_added_coverage=args.min_added_coverage,
            )
        if args.collection in {"all", "landsat"}:
            landsat_query_items = _query_items(
                collection="landsat-c2-l2",
                bbox=bbox,
                datetime_range=args.datetime,
                cloud_max=args.cloud_max,
                limit=args.limit,
            )
            landsat_items, landsat_coverage = _select_covering_items(
                items=landsat_query_items,
                required_assets=["blue", "green", "red", "nir08"],
                aoi_geom=aoi_geom,
                max_scenes=args.max_scenes_per_collection,
                target_coverage=args.target_coverage,
                min_added_coverage=args.min_added_coverage,
            )

    print(f"[fetch] selected {len(sentinel_items)} Sentinel scenes; footprint coverage={sentinel_coverage:.4f}")
    print(f"[fetch] selected {len(landsat_items)} Landsat scenes; footprint coverage={landsat_coverage:.4f}")
    if args.selection_only:
        if args.collection == "all":
            _save_selection_metadata(
                out_json=selection_json,
                sentinel_items=sentinel_items,
                landsat_items=landsat_items,
                sentinel_coverage=sentinel_coverage,
                landsat_coverage=landsat_coverage,
            )
        else:
            print("[fetch] collection-specific selection-only mode; left selection metadata unchanged.")
        print("[fetch] selection-only mode; skipped raster download/reprojection.")
        return 0

    target_windows = None
    if args.target_mode == "site-buffer":
        target_windows = _target_windows_from_sites(
            template_tif=dem_tif,
            label_geojson=label_geojson,
            site_column=args.site_column,
            buffer_km=float(args.site_buffer_km),
        )
        target_windows = _split_target_windows(target_windows, int(args.target_tile_size))

    if args.collection in {"all", "sentinel"}:
        if args.sentinel_provider == "planetary-computer":
            sentinel_items = [
                _signed_pc_item_assets(item, collection="sentinel-2-l2a")
                for item in sentinel_items
            ]
            sentinel_band_keys = ["B02", "B03", "B04", "B08"]
        else:
            sentinel_band_keys = ["blue", "green", "red", "nir"]
        _write_aligned_stack(
            template_tif=dem_tif,
            items=sentinel_items,
            out_tif=sentinel_out,
            band_keys=sentinel_band_keys,
            band_names=["S2_BLUE", "S2_GREEN", "S2_RED", "S2_NIR"],
            target_windows=target_windows,
            cache_assets_dir=args.cache_assets_dir,
        )

    if args.collection in {"all", "landsat"}:
        landsat_items_pc = []
        for item in landsat_items:
            item_pc = _fetch_pc_landsat_item(str(item.get("id")))
            item_pc = _signed_pc_item_assets(item_pc, collection="landsat-c2-l2")
            landsat_items_pc.append(item_pc)
        _write_aligned_stack(
            template_tif=dem_tif,
            items=landsat_items_pc,
            out_tif=landsat_out,
            band_keys=["blue", "green", "red", "nir08"],
            band_names=["L8_BLUE", "L8_GREEN", "L8_RED", "L8_NIR"],
            target_windows=target_windows,
            cache_assets_dir=args.cache_assets_dir,
        )
    if args.collection == "all":
        _save_selection_metadata(
            out_json=selection_json,
            sentinel_items=sentinel_items,
            landsat_items=landsat_items,
            sentinel_coverage=sentinel_coverage,
            landsat_coverage=landsat_coverage,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
