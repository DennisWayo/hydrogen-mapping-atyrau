#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio as rio
from PIL import Image, ImageDraw, ImageFont
from rasterio.transform import rowcol


@dataclass
class PointMeta:
    site_name: str
    site_code: str
    latitude: float
    longitude: float
    lat_dms: str
    lon_dms: str
    indicator_raw: str
    target_hint: str


@dataclass
class RasterLayer:
    arr: np.ndarray
    valid: np.ndarray
    transform: Any
    crs: Any
    width: int
    height: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render annotated before/after scenes with keys, coordinates, scale, and north arrow."
    )
    parser.add_argument(
        "--points-csv",
        type=Path,
        default=Path("paper_runs/ex_01/regions/ex_01_ground_points.csv"),
        help="Input ground points CSV.",
    )
    parser.add_argument(
        "--before-dem-dir",
        type=Path,
        default=Path("paper_runs/ex_01/runs/ex_01/scenes/before_after/before_dem"),
        help="Folder with before DEM GeoTIFF windows.",
    )
    parser.add_argument(
        "--after-prob-dir",
        type=Path,
        default=Path("paper_runs/ex_01/runs/ex_01/scenes/before_after/after_prob"),
        help="Folder with after probability GeoTIFF windows.",
    )
    parser.add_argument(
        "--after-mask-dir",
        type=Path,
        default=Path("paper_runs/ex_01/runs/ex_01/scenes/before_after/after_mask"),
        help="Folder with after mask GeoTIFF windows.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("paper_runs/ex_01/runs/ex_01/scenes/annotated"),
        help="Output root for annotated PNG scenes.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("paper_runs/ex_01/runs/ex_01/scenes/annotated/prospect_summary.csv"),
        help="Output CSV ranking AOIs by prospect signal.",
    )
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.8,
        help="Probability threshold used for high-prospect area stats.",
    )
    return parser.parse_args()


def _load_points(points_csv: Path) -> dict[str, PointMeta]:
    out: dict[str, PointMeta] = {}
    with points_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = str(row["site_code"]).strip()
            out[code] = PointMeta(
                site_name=str(row["site_name"]).strip(),
                site_code=code,
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                lat_dms=str(row.get("lat_dms", "")).strip(),
                lon_dms=str(row.get("lon_dms", "")).strip(),
                indicator_raw=str(row.get("indicator_raw", "")).strip(),
                target_hint=str(row.get("target_hint", "")).strip(),
            )
    return out


def _read_layer(path: Path) -> RasterLayer:
    with rio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
        width, height = src.width, src.height
    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= arr != nodata
    return RasterLayer(
        arr=arr,
        valid=valid,
        transform=transform,
        crs=crs,
        width=width,
        height=height,
    )


def _stretch_uint8(arr: np.ndarray, valid: np.ndarray, q_lo: float = 0.02, q_hi: float = 0.98) -> np.ndarray:
    out = np.zeros(arr.shape, dtype=np.uint8)
    if not np.any(valid):
        return out
    vals = arr[valid]
    lo = float(np.quantile(vals, q_lo))
    hi = float(np.quantile(vals, q_hi))
    if hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
    if hi <= lo:
        out[valid] = 128
        return out
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    out[valid] = (scaled[valid] * 255.0).astype(np.uint8)
    return out


def _prob_to_rgb(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    # Custom colormap: deep blue -> cyan -> yellow -> red.
    stops = np.array([0.0, 0.35, 0.7, 1.0], dtype=np.float32)
    colors = np.array(
        [
            [20, 40, 120],
            [40, 180, 230],
            [250, 220, 70],
            [220, 40, 40],
        ],
        dtype=np.float32,
    )
    v = np.clip(arr, 0.0, 1.0)
    flat = v.reshape(-1)
    r = np.interp(flat, stops, colors[:, 0]).reshape(v.shape)
    g = np.interp(flat, stops, colors[:, 1]).reshape(v.shape)
    b = np.interp(flat, stops, colors[:, 2]).reshape(v.shape)
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def _mask_to_rgb(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    rgb[:] = (20, 20, 20)
    on = valid & (arr > 0)
    rgb[on] = (30, 220, 80)
    rgb[~valid] = (0, 0, 0)
    return rgb


def _meters_per_pixel_x(transform: Any, crs: Any, center_lat: float) -> float:
    px = abs(float(transform.a))
    if crs is not None and crs.is_geographic:
        return px * 111320.0 * max(math.cos(math.radians(center_lat)), 1e-6)
    return px


def _pixel_area_km2(transform: Any, crs: Any, center_lat: float) -> float:
    xres = abs(float(transform.a))
    yres = abs(float(transform.e))
    if crs is not None and crs.is_geographic:
        dx = xres * 111320.0 * max(math.cos(math.radians(center_lat)), 1e-6)
        dy = yres * 110574.0
    else:
        dx = xres
        dy = yres
    return (dx * dy) / 1_000_000.0


def _draw_north_arrow(draw: ImageDraw.ImageDraw, w: int, h: int, font: ImageFont.ImageFont) -> None:
    x = w - 38
    y0 = 26
    y1 = 76
    draw.line([(x, y1), (x, y0)], fill=(255, 255, 255), width=3)
    draw.polygon([(x, y0 - 10), (x - 7, y0 + 4), (x + 7, y0 + 4)], fill=(255, 255, 255))
    draw.text((x - 6, y0 - 28), "N", fill=(255, 255, 255), font=font)


def _draw_scale_bar(
    draw: ImageDraw.ImageDraw,
    transform: Any,
    crs: Any,
    center_lat: float,
    w: int,
    h: int,
    font: ImageFont.ImageFont,
) -> None:
    mpp = _meters_per_pixel_x(transform=transform, crs=crs, center_lat=center_lat)
    candidates_km = [1, 2, 5, 10, 20]
    target_px = max(80, int(w * 0.18))
    chosen_km = 1
    chosen_px = int((1000.0 * chosen_km) / max(mpp, 1e-6))
    for km in candidates_km:
        px = int((1000.0 * km) / max(mpp, 1e-6))
        if 40 <= px <= int(w * 0.35):
            chosen_km, chosen_px = km, px
        if px >= target_px:
            chosen_km, chosen_px = km, px
            break

    x0 = 24
    y = h - 30
    x1 = x0 + max(20, chosen_px)
    draw.line([(x0, y), (x1, y)], fill=(255, 255, 255), width=4)
    draw.line([(x0, y - 5), (x0, y + 5)], fill=(255, 255, 255), width=2)
    draw.line([(x1, y - 5), (x1, y + 5)], fill=(255, 255, 255), width=2)
    draw.text((x0, y - 22), f"{chosen_km} km", fill=(255, 255, 255), font=font)


def _draw_coord_marker(
    draw: ImageDraw.ImageDraw,
    transform: Any,
    width: int,
    height: int,
    lon: float,
    lat: float,
) -> tuple[int, int] | None:
    row, col = rowcol(transform, lon, lat)
    if row < 0 or col < 0 or row >= height or col >= width:
        return None
    x = int(col)
    y = int(row)
    c = (255, 255, 255)
    draw.line([(x - 10, y), (x + 10, y)], fill=c, width=2)
    draw.line([(x, y - 10), (x, y + 10)], fill=c, width=2)
    draw.ellipse([(x - 6, y - 6), (x + 6, y + 6)], outline=c, width=2)
    return x, y


def _draw_info_box(
    draw: ImageDraw.ImageDraw,
    site: PointMeta,
    layer_label: str,
    stats_lines: list[str],
    font: ImageFont.ImageFont,
) -> None:
    lines = [
        f"{site.site_name} ({site.site_code})",
        f"{site.latitude:.6f}N, {site.longitude:.6f}E",
        f"{site.lat_dms}, {site.lon_dms}",
        f"Layer: {layer_label}",
    ] + stats_lines
    text = "\n".join(lines)
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=2)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x0, y0 = 12, 10
    x1, y1 = x0 + w + 10, y0 + h + 8
    draw.rounded_rectangle([(x0, y0), (x1, y1)], radius=6, fill=(0, 0, 0, 155), outline=(255, 255, 255, 180), width=1)
    draw.multiline_text((x0 + 5, y0 + 4), text, fill=(255, 255, 255), font=font, spacing=2)


def _draw_legend(draw: ImageDraw.ImageDraw, layer: str, w: int, h: int, font: ImageFont.ImageFont) -> None:
    box_w = 220
    box_h = 64
    x0 = w - box_w - 10
    y0 = h - box_h - 10
    draw.rounded_rectangle([(x0, y0), (x0 + box_w, y0 + box_h)], radius=6, fill=(0, 0, 0, 155), outline=(255, 255, 255, 180), width=1)

    if layer == "before_dem":
        grad_w = 150
        grad_h = 10
        gx0, gy0 = x0 + 12, y0 + 30
        for i in range(grad_w):
            v = int((i / max(grad_w - 1, 1)) * 255)
            draw.line([(gx0 + i, gy0), (gx0 + i, gy0 + grad_h)], fill=(v, v, v), width=1)
        draw.text((x0 + 12, y0 + 10), "Key: DEM elevation", fill=(255, 255, 255), font=font)
        draw.text((gx0, gy0 + 14), "low", fill=(255, 255, 255), font=font)
        draw.text((gx0 + grad_w - 22, gy0 + 14), "high", fill=(255, 255, 255), font=font)
    elif layer == "after_prob":
        grad_w = 160
        grad_h = 10
        gx0, gy0 = x0 + 12, y0 + 30
        for i in range(grad_w):
            t = i / max(grad_w - 1, 1)
            if t <= 0.35:
                t2 = t / 0.35
                c0 = np.array([20, 40, 120], dtype=np.float32)
                c1 = np.array([40, 180, 230], dtype=np.float32)
            elif t <= 0.7:
                t2 = (t - 0.35) / 0.35
                c0 = np.array([40, 180, 230], dtype=np.float32)
                c1 = np.array([250, 220, 70], dtype=np.float32)
            else:
                t2 = (t - 0.7) / 0.3
                c0 = np.array([250, 220, 70], dtype=np.float32)
                c1 = np.array([220, 40, 40], dtype=np.float32)
            c = (c0 * (1.0 - t2) + c1 * t2).astype(np.uint8)
            draw.line([(gx0 + i, gy0), (gx0 + i, gy0 + grad_h)], fill=tuple(int(x) for x in c), width=1)
        draw.text((x0 + 12, y0 + 10), "Key: H2 prospect probability", fill=(255, 255, 255), font=font)
        draw.text((gx0, gy0 + 14), "0.0", fill=(255, 255, 255), font=font)
        draw.text((gx0 + grad_w - 20, gy0 + 14), "1.0", fill=(255, 255, 255), font=font)
    else:
        draw.text((x0 + 12, y0 + 10), "Key: High-prospect mask", fill=(255, 255, 255), font=font)
        draw.rectangle([(x0 + 12, y0 + 30), (x0 + 26, y0 + 44)], fill=(30, 220, 80), outline=(255, 255, 255))
        draw.text((x0 + 34, y0 + 28), ">= threshold", fill=(255, 255, 255), font=font)
        draw.rectangle([(x0 + 130, y0 + 30), (x0 + 144, y0 + 44)], fill=(20, 20, 20), outline=(255, 255, 255))
        draw.text((x0 + 150, y0 + 28), "< threshold", fill=(255, 255, 255), font=font)


def _annotate_image(
    img: Image.Image,
    site: PointMeta,
    layer_name: str,
    layer: RasterLayer,
    stats_lines: list[str],
) -> Image.Image:
    out = img.convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    font = ImageFont.load_default(size=15)

    center_lat = float(site.latitude)
    _draw_north_arrow(draw=draw, w=layer.width, h=layer.height, font=font)
    _draw_scale_bar(
        draw=draw,
        transform=layer.transform,
        crs=layer.crs,
        center_lat=center_lat,
        w=layer.width,
        h=layer.height,
        font=font,
    )
    _draw_coord_marker(
        draw=draw,
        transform=layer.transform,
        width=layer.width,
        height=layer.height,
        lon=site.longitude,
        lat=site.latitude,
    )
    _draw_info_box(draw=draw, site=site, layer_label=layer_name, stats_lines=stats_lines, font=font)
    _draw_legend(draw=draw, layer=layer_name, w=layer.width, h=layer.height, font=font)
    return out.convert("RGB")


def _combine_triptych(before: Image.Image, prob: Image.Image, mask: Image.Image, site: PointMeta) -> Image.Image:
    gap = 10
    panel_w, panel_h = before.size
    head_h = 44
    out_w = panel_w * 3 + gap * 4
    out_h = panel_h + head_h + gap * 2
    out = Image.new("RGB", (out_w, out_h), (18, 18, 22))
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=14)

    title = f"{site.site_name} ({site.site_code})  |  {site.latitude:.6f}N, {site.longitude:.6f}E"
    draw.text((gap, 10), title, fill=(255, 255, 255), font=font)
    draw.text((gap, 26), "Before DEM | After Probability | After Threshold Mask", fill=(220, 220, 220), font=small)

    y = head_h + gap
    x0 = gap
    x1 = x0 + panel_w + gap
    x2 = x1 + panel_w + gap
    out.paste(before, (x0, y))
    out.paste(prob, (x1, y))
    out.paste(mask, (x2, y))
    return out


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "site_code",
        "site_name",
        "latitude",
        "longitude",
        "mean_prob",
        "p95_prob",
        "max_prob",
        "high_prob_fraction",
        "high_prob_area_km2",
        "mask_high_area_km2",
        "prospect_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    args = _parse_args()

    points = _load_points(args.points_csv)
    out_root = args.out_root
    out_before = out_root / "before_dem"
    out_prob = out_root / "after_prob"
    out_mask = out_root / "after_mask"
    out_combined = out_root / "combined"
    for p in [out_before, out_prob, out_mask, out_combined]:
        p.mkdir(parents=True, exist_ok=True)

    site_codes = sorted(points.keys())
    summary_rows: list[dict[str, Any]] = []

    for code in site_codes:
        site = points[code]
        before_tif = args.before_dem_dir / f"{code}_r12km.tif"
        prob_tif = args.after_prob_dir / f"{code}_r12km.tif"
        mask_tif = args.after_mask_dir / f"{code}_r12km.tif"
        if not (before_tif.exists() and prob_tif.exists() and mask_tif.exists()):
            print(f"[annotate] skip {code}: missing one or more input windows")
            continue

        before_layer = _read_layer(before_tif)
        prob_layer = _read_layer(prob_tif)
        mask_layer = _read_layer(mask_tif)

        dem_u8 = _stretch_uint8(before_layer.arr, before_layer.valid)
        dem_rgb = np.stack([dem_u8, dem_u8, dem_u8], axis=-1)
        prob_rgb = _prob_to_rgb(prob_layer.arr, prob_layer.valid)
        mask_rgb = _mask_to_rgb(mask_layer.arr, mask_layer.valid)

        before_img = Image.fromarray(dem_rgb, mode="RGB")
        prob_img = Image.fromarray(prob_rgb, mode="RGB")
        mask_img = Image.fromarray(mask_rgb, mode="RGB")

        valid_prob = prob_layer.valid
        prob_vals = np.clip(prob_layer.arr[valid_prob], 0.0, 1.0) if np.any(valid_prob) else np.array([], dtype=np.float32)
        if prob_vals.size == 0:
            mean_prob = p95_prob = max_prob = high_frac = 0.0
        else:
            mean_prob = float(np.mean(prob_vals))
            p95_prob = float(np.quantile(prob_vals, 0.95))
            max_prob = float(np.max(prob_vals))
            high_frac = float(np.mean(prob_vals >= args.prob_threshold))

        px_area_km2 = _pixel_area_km2(
            transform=prob_layer.transform,
            crs=prob_layer.crs,
            center_lat=site.latitude,
        )
        high_prob_area = float(np.sum((prob_layer.arr >= args.prob_threshold) & valid_prob) * px_area_km2)
        mask_high_area = float(np.sum((mask_layer.arr > 0) & mask_layer.valid) * px_area_km2)
        prospect_score = 0.6 * p95_prob + 0.4 * mean_prob

        before_stats = [
            f"Indicator: {site.indicator_raw}",
            f"Target hint: {site.target_hint}",
            f"DEM min/max: {float(np.min(before_layer.arr[before_layer.valid])):.1f}/{float(np.max(before_layer.arr[before_layer.valid])):.1f}",
        ]
        prob_stats = [
            f"Mean P: {mean_prob:.3f}",
            f"P95 P: {p95_prob:.3f}",
            f"Max P: {max_prob:.3f}",
            f"High area >= {args.prob_threshold:.2f}: {high_prob_area:.2f} km2",
        ]
        mask_stats = [
            f"Threshold: {args.prob_threshold:.2f}",
            f"Mask high area: {mask_high_area:.2f} km2",
        ]

        before_anno = _annotate_image(before_img, site, "before_dem", before_layer, before_stats)
        prob_anno = _annotate_image(prob_img, site, "after_prob", prob_layer, prob_stats)
        mask_anno = _annotate_image(mask_img, site, "after_mask", mask_layer, mask_stats)
        combined = _combine_triptych(before_anno, prob_anno, mask_anno, site)

        before_out = out_before / f"{code}_before_dem_annotated.png"
        prob_out = out_prob / f"{code}_after_prob_annotated.png"
        mask_out = out_mask / f"{code}_after_mask_annotated.png"
        combined_out = out_combined / f"{code}_before_after_annotated.png"
        before_anno.save(before_out)
        prob_anno.save(prob_out)
        mask_anno.save(mask_out)
        combined.save(combined_out)

        print(f"[annotate] saved {before_out}")
        print(f"[annotate] saved {prob_out}")
        print(f"[annotate] saved {mask_out}")
        print(f"[annotate] saved {combined_out}")

        summary_rows.append(
            {
                "site_code": code,
                "site_name": site.site_name,
                "latitude": f"{site.latitude:.8f}",
                "longitude": f"{site.longitude:.8f}",
                "mean_prob": f"{mean_prob:.6f}",
                "p95_prob": f"{p95_prob:.6f}",
                "max_prob": f"{max_prob:.6f}",
                "high_prob_fraction": f"{high_frac:.6f}",
                "high_prob_area_km2": f"{high_prob_area:.6f}",
                "mask_high_area_km2": f"{mask_high_area:.6f}",
                "prospect_score": f"{prospect_score:.6f}",
            }
        )

    summary_rows_sorted = sorted(summary_rows, key=lambda r: float(r["prospect_score"]), reverse=True)
    for i, row in enumerate(summary_rows_sorted, start=1):
        row["rank"] = i

    _write_summary(args.summary_csv, summary_rows_sorted)
    print(f"[annotate] prospect summary: {args.summary_csv}")
    if summary_rows_sorted:
        top = summary_rows_sorted[0]
        print(
            "[annotate] top AOI by prospect score: "
            f"{top['site_name']} ({top['site_code']}) "
            f"score={top['prospect_score']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
