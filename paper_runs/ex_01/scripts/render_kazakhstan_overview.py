#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


COUNTRIES_GEOJSON_URL = (
    "https://raw.githubusercontent.com/datasets/geo-boundaries-world-110m/master/countries.geojson"
)
OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
OSM_USER_AGENT = "XaiGis-ex01-overview/1.0"
TILE_SIZE = 256
MAX_LAT = 85.05112878


@dataclass
class SiteRow:
    site_code: str
    site_name: str
    latitude: float
    longitude: float
    rank: int
    prospect_score: float
    p95_prob: float
    high_prob_area_km2: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Kazakhstan overview map with OSM city basemap and site callouts."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("paper_runs/ex_01/runs/ex_01/scenes/annotated/prospect_summary.csv"),
        help="Prospect summary CSV from annotated scene workflow.",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("paper_runs/ex_01/runs/ex_01/scenes/annotated/kazakhstan_overview_annotated.png"),
        help="Output overview PNG path.",
    )
    parser.add_argument(
        "--out-geojson",
        type=Path,
        default=Path("paper_runs/ex_01/runs/ex_01/scenes/annotated/kazakhstan_outline.geojson"),
        help="Local cache file for Kazakhstan country geometry.",
    )
    parser.add_argument(
        "--tile-cache-dir",
        type=Path,
        default=Path("paper_runs/ex_01/runs/ex_01/scenes/annotated/osm_tiles_cache"),
        help="Folder to cache downloaded OSM tiles.",
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=0,
        help="Fixed OSM zoom level. Use 0 for automatic zoom selection.",
    )
    return parser.parse_args()


def _load_sites(path: Path) -> list[SiteRow]:
    rows: list[SiteRow] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                SiteRow(
                    site_code=str(row["site_code"]).strip(),
                    site_name=str(row["site_name"]).strip(),
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    rank=int(row["rank"]),
                    prospect_score=float(row["prospect_score"]),
                    p95_prob=float(row["p95_prob"]),
                    high_prob_area_km2=float(row["high_prob_area_km2"]),
                )
            )
    if not rows:
        raise ValueError(f"No rows found in summary CSV: {path}")
    return rows


def _fetch_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": OSM_USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))


def _extract_kazakhstan_geometry(world_geojson: dict[str, Any]) -> dict[str, Any]:
    feats = world_geojson.get("features", [])
    for feat in feats:
        props = feat.get("properties") or {}
        name = str(props.get("name", "")).strip().lower()
        if name == "kazakhstan":
            geom = feat.get("geometry")
            if not isinstance(geom, dict):
                raise ValueError("Kazakhstan feature found but geometry missing.")
            return geom
    raise ValueError("Kazakhstan geometry not found in world boundaries dataset.")


def _save_feature_collection(geometry: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Kazakhstan"},
                "geometry": geometry,
            }
        ],
    }
    out_path.write_text(json.dumps(fc, indent=2), encoding="utf-8")


def _iter_outer_rings(geometry: dict[str, Any]) -> list[list[tuple[float, float]]]:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    rings: list[list[tuple[float, float]]] = []
    if gtype == "Polygon":
        if coords:
            outer = coords[0]
            rings.append([(float(x), float(y)) for x, y in outer])
    elif gtype == "MultiPolygon":
        for poly in coords or []:
            if not poly:
                continue
            outer = poly[0]
            rings.append([(float(x), float(y)) for x, y in outer])
    else:
        raise ValueError(f"Unsupported geometry type: {gtype}")
    if not rings:
        raise ValueError("No polygon rings available for Kazakhstan geometry.")
    return rings


def _bbox_from_rings(rings: list[list[tuple[float, float]]]) -> tuple[float, float, float, float]:
    xs = [x for ring in rings for x, _ in ring]
    ys = [y for ring in rings for _, y in ring]
    return min(xs), min(ys), max(xs), max(ys)


def _clamp_lat(lat: float) -> float:
    return max(-MAX_LAT, min(MAX_LAT, lat))


def _lonlat_to_world_px(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    lat = _clamp_lat(lat)
    n = float((1 << zoom) * TILE_SIZE)
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) * 0.5 * n
    return x, y


def _choose_zoom(
    bbox: tuple[float, float, float, float],
    map_w: int,
    map_h: int,
    min_zoom: int = 4,
    max_zoom: int = 7,
) -> int:
    min_lon, min_lat, max_lon, max_lat = bbox
    for z in range(max_zoom, min_zoom - 1, -1):
        left, top = _lonlat_to_world_px(min_lon, max_lat, z)
        right, bottom = _lonlat_to_world_px(max_lon, min_lat, z)
        w = right - left
        h = bottom - top
        if w >= map_w and h >= map_h:
            return z
    return min_zoom


def _fetch_osm_tile(z: int, x: int, y: int, cache_dir: Path) -> Image.Image:
    cache_path = cache_dir / str(z) / str(x) / f"{y}.png"
    if cache_path.exists():
        return Image.open(cache_path).convert("RGB")

    url = OSM_TILE_URL.format(z=z, x=x, y=y)
    req = urllib.request.Request(url, headers={"User-Agent": OSM_USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    return Image.open(cache_path).convert("RGB")


def _build_osm_basemap(
    bbox: tuple[float, float, float, float],
    map_w: int,
    map_h: int,
    cache_dir: Path,
    forced_zoom: int,
) -> tuple[Image.Image, dict[str, float | int]]:
    min_lon, min_lat, max_lon, max_lat = bbox
    zoom = forced_zoom if forced_zoom > 0 else _choose_zoom(bbox=bbox, map_w=map_w, map_h=map_h)

    left, top = _lonlat_to_world_px(min_lon, max_lat, zoom)
    right, bottom = _lonlat_to_world_px(max_lon, min_lat, zoom)

    tile_x0 = int(math.floor(left / TILE_SIZE))
    tile_x1 = int(math.floor((right - 1.0) / TILE_SIZE))
    tile_y0 = int(math.floor(top / TILE_SIZE))
    tile_y1 = int(math.floor((bottom - 1.0) / TILE_SIZE))

    tiles_w = tile_x1 - tile_x0 + 1
    tiles_h = tile_y1 - tile_y0 + 1
    stitched = Image.new("RGB", (tiles_w * TILE_SIZE, tiles_h * TILE_SIZE), (245, 245, 245))

    for tx in range(tile_x0, tile_x1 + 1):
        for ty in range(tile_y0, tile_y1 + 1):
            tile = _fetch_osm_tile(zoom, tx, ty, cache_dir=cache_dir)
            stitched.paste(tile, ((tx - tile_x0) * TILE_SIZE, (ty - tile_y0) * TILE_SIZE))

    crop_left = left - tile_x0 * TILE_SIZE
    crop_top = top - tile_y0 * TILE_SIZE
    crop_right = crop_left + (right - left)
    crop_bottom = crop_top + (bottom - top)

    cropped = stitched.crop(
        (
            int(math.floor(crop_left)),
            int(math.floor(crop_top)),
            int(math.ceil(crop_right)),
            int(math.ceil(crop_bottom)),
        )
    )

    src_w, src_h = cropped.size
    scale = min(map_w / max(src_w, 1), map_h / max(src_h, 1))
    draw_w = max(1, int(round(src_w * scale)))
    draw_h = max(1, int(round(src_h * scale)))
    resized = cropped.resize((draw_w, draw_h), resample=Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (map_w, map_h), (235, 235, 235))
    pad_x = (map_w - draw_w) // 2
    pad_y = (map_h - draw_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    transform = {
        "zoom": zoom,
        "world_left": left,
        "world_top": top,
        "scale": scale,
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
        "map_w": float(map_w),
        "map_h": float(map_h),
    }
    return canvas, transform


def _world_to_map_xy(world_x: float, world_y: float, transform: dict[str, float | int]) -> tuple[int, int]:
    scale = float(transform["scale"])
    x = float(transform["pad_x"]) + (world_x - float(transform["world_left"])) * scale
    y = float(transform["pad_y"]) + (world_y - float(transform["world_top"])) * scale
    return int(round(x)), int(round(y))


def _lonlat_to_map_xy(lon: float, lat: float, transform: dict[str, float | int]) -> tuple[int, int]:
    wx, wy = _lonlat_to_world_px(lon, lat, int(transform["zoom"]))
    return _world_to_map_xy(wx, wy, transform)


def _rank_color(rank: int) -> tuple[int, int, int]:
    palette = {
        1: (230, 60, 60),
        2: (245, 120, 40),
        3: (245, 180, 40),
        4: (120, 200, 70),
        5: (70, 170, 220),
        6: (140, 120, 220),
    }
    return palette.get(rank, (220, 220, 220))


def _callout_offsets() -> dict[str, tuple[int, int]]:
    return {
        "saumankol": (-320, -190),
        "aksu": (130, -140),
        "chistopoloe": (-320, 40),
        "atbasar": (-280, 170),
        "shchuchinsk": (120, -15),
        "suvorovka": (130, 140),
    }


def _draw_north_arrow(
    draw: ImageDraw.ImageDraw,
    map_x0: int,
    map_y0: int,
    map_w: int,
    font: ImageFont.ImageFont,
) -> None:
    x = map_x0 + map_w - 58
    y0 = map_y0 + 24
    y1 = y0 + 56
    draw.line([(x, y1), (x, y0)], fill=(255, 255, 255), width=3)
    draw.polygon([(x, y0 - 10), (x - 8, y0 + 4), (x + 8, y0 + 4)], fill=(255, 255, 255))
    draw.text((x - 7, y0 - 30), "N", fill=(255, 255, 255), font=font)


def _draw_scale_bar(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[float, float, float, float],
    transform: dict[str, float | int],
    map_x0: int,
    map_y0: int,
    map_h: int,
    font: ImageFont.ImageFont,
) -> None:
    min_lon, min_lat, _, max_lat = bbox
    center_lat = 0.5 * (min_lat + max_lat)
    meters_per_deg_lon = 111320.0 * max(math.cos(math.radians(center_lat)), 1e-6)
    candidates_km = [100, 200, 300, 500]

    x0_map, y_map = _lonlat_to_map_xy(min_lon + 0.8, min_lat + 0.5, transform)
    chosen_km = 200
    chosen_x1 = x0_map + 100
    for km in candidates_km:
        dlon = (km * 1000.0) / meters_per_deg_lon
        x1_map, _ = _lonlat_to_map_xy(min_lon + 0.8 + dlon, min_lat + 0.5, transform)
        px = x1_map - x0_map
        if 80 <= px <= 360:
            chosen_km = km
            chosen_x1 = x1_map
            break

    x0 = map_x0 + x0_map
    x1 = map_x0 + chosen_x1
    y = map_y0 + map_h - 26
    draw.line([(x0, y), (x1, y)], fill=(255, 255, 255), width=4)
    draw.line([(x0, y - 6), (x0, y + 6)], fill=(255, 255, 255), width=2)
    draw.line([(x1, y - 6), (x1, y + 6)], fill=(255, 255, 255), width=2)
    draw.text((x0, y - 26), f"{chosen_km} km", fill=(255, 255, 255), font=font)


def _draw_legend_inside(
    draw: ImageDraw.ImageDraw,
    map_x0: int,
    map_y0: int,
    map_w: int,
    font: ImageFont.ImageFont,
    small: ImageFont.ImageFont,
) -> None:
    box_w = 260
    box_h = 214
    x0 = map_x0 + map_w - box_w - 16
    y0 = map_y0 + 16
    draw.rounded_rectangle(
        [(x0, y0), (x0 + box_w, y0 + box_h)],
        radius=8,
        fill=(22, 22, 28, 190),
        outline=(190, 190, 200, 255),
        width=1,
    )
    draw.text((x0 + 10, y0 + 10), "Legend", fill=(255, 255, 255), font=font)
    draw.text((x0 + 10, y0 + 36), "Site marker color by rank", fill=(210, 210, 220), font=small)
    y = y0 + 60
    for r in range(1, 7):
        c = _rank_color(r)
        draw.ellipse([(x0 + 12, y - 1), (x0 + 24, y + 11)], fill=c, outline=(255, 255, 255))
        draw.text((x0 + 32, y - 3), f"Rank {r}", fill=(235, 235, 245), font=small)
        y += 22
    draw.text((x0 + 10, y0 + 190), "Arrow -> site location", fill=(190, 190, 200), font=small)


def _draw_title(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, small: ImageFont.ImageFont) -> None:
    draw.text((16, 12), "Kazakhstan Overview: DEM-Based Hydrogen Prospect Screening", fill=(255, 255, 255), font=font)
    draw.text((16, 38), "Real basemap with city labels (OpenStreetMap), ranked site callouts", fill=(210, 210, 220), font=small)


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def main() -> int:
    args = _parse_args()
    sites = _load_sites(args.summary_csv)

    world = _fetch_json(COUNTRIES_GEOJSON_URL)
    kaz_geom = _extract_kazakhstan_geometry(world)
    _save_feature_collection(kaz_geom, args.out_geojson)

    rings = _iter_outer_rings(kaz_geom)
    minx, miny, maxx, maxy = _bbox_from_rings(rings)
    bbox = (minx - 1.0, miny - 1.0, maxx + 1.0, maxy + 1.0)

    width, height = 2200, 1400
    map_x0, map_y0 = 40, 84
    map_w, map_h = 2120, 1270

    basemap, transform = _build_osm_basemap(
        bbox=bbox,
        map_w=map_w,
        map_h=map_h,
        cache_dir=args.tile_cache_dir,
        forced_zoom=int(args.zoom),
    )

    img = Image.new("RGB", (width, height), (12, 14, 18))
    img.paste(basemap, (map_x0, map_y0))
    draw = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.load_default(size=22)
    small = ImageFont.load_default(size=16)

    _draw_title(draw=draw, font=font, small=small)

    # Overlay country boundary outline.
    for ring in rings:
        points = []
        for lon, lat in ring:
            mx, my = _lonlat_to_map_xy(lon, lat, transform)
            points.append((map_x0 + mx, map_y0 + my))
        if len(points) >= 2:
            draw.line(points, fill=(130, 180, 240, 220), width=2)

    offsets = _callout_offsets()
    panel_x1 = map_x0 + map_w - 10
    panel_y1 = map_y0 + map_h - 10
    for site in sites:
        sx_map, sy_map = _lonlat_to_map_xy(site.longitude, site.latitude, transform)
        sx = map_x0 + sx_map
        sy = map_y0 + sy_map
        color = _rank_color(site.rank)

        draw.ellipse([(sx - 7, sy - 7), (sx + 7, sy + 7)], fill=color, outline=(255, 255, 255), width=2)

        ox, oy = offsets.get(site.site_code, (120, -100))
        tx = sx + ox
        ty = sy + oy

        lines = [
            f"#{site.rank} {site.site_name} ({site.site_code})",
            f"{site.latitude:.5f}N, {site.longitude:.5f}E",
            f"score={site.prospect_score:.3f} | p95={site.p95_prob:.3f}",
            f"high-area={site.high_prob_area_km2:.1f} km2",
        ]
        label = "\n".join(lines)
        bbox_text = draw.multiline_textbbox((0, 0), label, font=small, spacing=2)
        tw = (bbox_text[2] - bbox_text[0]) + 10
        th = (bbox_text[3] - bbox_text[1]) + 8

        bx0 = tx + 8
        by0 = ty - 8
        bx0 = _clamp(bx0, map_x0 + 10, panel_x1 - tw - 4)
        by0 = _clamp(by0, map_y0 + 10, panel_y1 - th - 4)
        bx1, by1 = bx0 + tw, by0 + th

        # Leader endpoint near label.
        ex = _clamp(tx, bx0, bx1)
        ey = _clamp(ty, by0, by1)
        draw.line([(sx, sy), (ex, ey)], fill=color, width=2)
        draw.ellipse([(ex - 3, ey - 3), (ex + 3, ey + 3)], fill=color)

        draw.rounded_rectangle([(bx0, by0), (bx1, by1)], radius=6, fill=(0, 0, 0, 175), outline=color, width=2)
        draw.multiline_text((bx0 + 5, by0 + 4), label, fill=(245, 245, 250), font=small, spacing=2)

    _draw_scale_bar(draw=draw, bbox=bbox, transform=transform, map_x0=map_x0, map_y0=map_y0, map_h=map_h, font=small)
    _draw_north_arrow(draw=draw, map_x0=map_x0, map_y0=map_y0, map_w=map_w, font=font)
    _draw_legend_inside(draw=draw, map_x0=map_x0, map_y0=map_y0, map_w=map_w, font=font, small=small)

    # OSM attribution.
    attrib = "© OpenStreetMap contributors"
    draw.text((map_x0 + 8, map_y0 + map_h - 18), attrib, fill=(255, 255, 255, 220), font=small)

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out_png)

    print(f"[overview] saved map: {args.out_png}")
    print(f"[overview] saved outline cache: {args.out_geojson}")
    print(f"[overview] OSM tile zoom: {transform['zoom']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
