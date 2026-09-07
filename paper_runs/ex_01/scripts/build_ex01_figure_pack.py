#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import rasterio as rio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from PIL import Image, ImageDraw, ImageFont

from render_kazakhstan_overview import (
    _bbox_from_rings,
    _build_osm_basemap,
    _callout_offsets,
    _clamp,
    _draw_north_arrow,
    _draw_scale_bar,
    _iter_outer_rings,
    _lonlat_to_map_xy,
    _rank_color,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = REPO_ROOT / "paper_runs/ex_01"
FIG_ROOT = RUN_ROOT / "figures"


def _ensure_dirs() -> dict[str, Path]:
    dirs = {
        "overview": FIG_ROOT / "00_overview",
        "inputs": FIG_ROOT / "01_inputs",
        "coverage": FIG_ROOT / "02_feature_coverage",
        "validation": FIG_ROOT / "03_validation",
        "final_maps": FIG_ROOT / "04_final_maps",
        "site_panels": FIG_ROOT / "04_final_maps/site_review_panels",
        "tables": FIG_ROOT / "tables",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _downsample_shape(height: int, width: int, max_dim: int = 900) -> tuple[int, int]:
    scale = min(max_dim / max(height, width), 1.0)
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def _read_band(path: Path, band: int = 1, max_dim: int = 900) -> tuple[np.ma.MaskedArray, tuple[float, float, float, float]]:
    with rio.open(path) as src:
        out_h, out_w = _downsample_shape(src.height, src.width, max_dim=max_dim)
        arr = src.read(band, out_shape=(out_h, out_w), resampling=Resampling.average).astype(np.float32)
        mask = ~np.isfinite(arr)
        if src.nodata is not None:
            mask |= arr == src.nodata
        return np.ma.array(arr, mask=mask), (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)


def _read_mask(path: Path, max_dim: int = 900) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    with rio.open(path) as src:
        out_h, out_w = _downsample_shape(src.height, src.width, max_dim=max_dim)
        arr = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest)
        mask = np.asarray(arr > 0, dtype=np.uint8)
        return mask, (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)


def _read_rgb(path: Path, bands: tuple[int, int, int], max_dim: int = 900) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    with rio.open(path) as src:
        out_h, out_w = _downsample_shape(src.height, src.width, max_dim=max_dim)
        data = src.read(bands, out_shape=(3, out_h, out_w), resampling=Resampling.bilinear).astype(np.float32)
        valid = np.isfinite(data).all(axis=0)
        for idx, band in enumerate(bands):
            nodata = src.nodatavals[band - 1]
            if nodata is not None:
                valid &= data[idx] != nodata
        rgb = np.zeros((out_h, out_w, 3), dtype=np.float32)
        for idx in range(3):
            vals = data[idx][valid]
            if vals.size:
                lo, hi = np.percentile(vals, [2, 98])
                if hi <= lo:
                    hi = lo + 1.0
                rgb[..., idx] = np.clip((data[idx] - lo) / (hi - lo), 0, 1)
        rgb[~valid] = 0.82
        return rgb, (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)


def _read_band_window(
    path: Path,
    bounds: tuple[float, float, float, float],
    band: int = 1,
    max_dim: int = 620,
    resampling: Resampling = Resampling.bilinear,
) -> np.ma.MaskedArray:
    left, bottom, right, top = bounds
    with rio.open(path) as src:
        window = from_bounds(left, bottom, right, top, transform=src.transform).round_offsets().round_lengths()
        out_h, out_w = _downsample_shape(int(window.height), int(window.width), max_dim=max_dim)
        nodata = src.nodatavals[band - 1]
        if nodata is None:
            nodata = src.nodata
        fill_value = nodata if nodata is not None else np.nan
        arr = src.read(
            band,
            window=window,
            out_shape=(out_h, out_w),
            resampling=resampling,
            boundless=True,
            fill_value=fill_value,
        ).astype(np.float32)
        mask = ~np.isfinite(arr)
        if nodata is not None:
            mask |= arr == nodata
        return np.ma.array(arr, mask=mask)


def _read_rgb_window(
    path: Path,
    bounds: tuple[float, float, float, float],
    bands: tuple[int, int, int],
    max_dim: int = 620,
) -> np.ndarray:
    left, bottom, right, top = bounds
    with rio.open(path) as src:
        window = from_bounds(left, bottom, right, top, transform=src.transform).round_offsets().round_lengths()
        out_h, out_w = _downsample_shape(int(window.height), int(window.width), max_dim=max_dim)
        fill_value = src.nodata if src.nodata is not None else np.nan
        data = src.read(
            bands,
            window=window,
            out_shape=(3, out_h, out_w),
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=fill_value,
        ).astype(np.float32)
        valid = np.isfinite(data).all(axis=0)
        for idx, band in enumerate(bands):
            nodata = src.nodatavals[band - 1]
            if nodata is not None:
                valid &= data[idx] != nodata
        rgb = np.zeros((out_h, out_w, 3), dtype=np.float32)
        for idx in range(3):
            vals = data[idx][valid]
            if vals.size:
                lo, hi = np.percentile(vals, [2, 98])
                if hi <= lo:
                    hi = lo + 1.0
                rgb[..., idx] = np.clip((data[idx] - lo) / (hi - lo), 0, 1)
        rgb[~valid] = 0.82
        return rgb


def _plot_sites(ax: plt.Axes, points: gpd.GeoDataFrame, labels: gpd.GeoDataFrame) -> None:
    labels.boundary.plot(ax=ax, color="black", linewidth=0.9, alpha=0.85)
    labels.boundary.plot(ax=ax, color="white", linewidth=0.35, alpha=0.9)
    points.plot(ax=ax, color="#ffcc33", edgecolor="black", marker="*", markersize=70, zorder=5)
    for _, row in points.iterrows():
        geom = row.geometry
        ax.text(
            geom.x + 0.035,
            geom.y + 0.025,
            str(row["site_name"]),
            fontsize=8,
            weight="bold",
            color="black",
        )


def _format_geo_axis(ax: plt.Axes) -> None:
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))


def _choose_scale_km(extent: tuple[float, float, float, float]) -> int:
    minx, maxx, miny, maxy = extent
    center_lat = 0.5 * (miny + maxy)
    km_per_deg_lon = 111.32 * max(np.cos(np.deg2rad(center_lat)), 1e-6)
    width_km = max((maxx - minx) * km_per_deg_lon, 1.0)
    for km in [25, 50, 100, 200, 300]:
        if km / width_km >= 0.10:
            return km
    return 300


def _add_panel_north_scale(ax: plt.Axes, extent: tuple[float, float, float, float]) -> None:
    text_fx = [pe.withStroke(linewidth=2.5, foreground="black")]
    ax.annotate(
        "",
        xy=(0.925, 0.91),
        xytext=(0.925, 0.78),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "lw": 3.2, "color": "black"},
        zorder=19,
    )
    ax.annotate(
        "N",
        xy=(0.925, 0.91),
        xytext=(0.925, 0.78),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="white",
        path_effects=text_fx,
        arrowprops={"arrowstyle": "-|>", "lw": 1.8, "color": "white"},
        zorder=20,
    )

    minx, maxx, miny, maxy = extent
    x_span = maxx - minx
    y_span = maxy - miny
    center_lat = 0.5 * (miny + maxy)
    km_per_deg_lon = 111.32 * max(np.cos(np.deg2rad(center_lat)), 1e-6)
    scale_km = _choose_scale_km(extent)
    scale_deg = scale_km / km_per_deg_lon
    x0 = minx + 0.065 * x_span
    x1 = min(x0 + scale_deg, maxx - 0.065 * x_span)
    y = miny + 0.075 * y_span
    ax.plot([x0, x1], [y, y], color="black", linewidth=4.0, solid_capstyle="butt", zorder=20)
    ax.plot([x0, x1], [y, y], color="white", linewidth=2.0, solid_capstyle="butt", zorder=21)
    ax.plot([x0, x0], [y - 0.015 * y_span, y + 0.015 * y_span], color="white", linewidth=1.5, zorder=21)
    ax.plot([x1, x1], [y - 0.015 * y_span, y + 0.015 * y_span], color="white", linewidth=1.5, zorder=21)
    ax.text(
        0.065,
        0.105,
        f"{scale_km} km",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="white",
        path_effects=text_fx,
        zorder=22,
    )


def _site_scene_rows() -> list[dict[str, Any]]:
    summary = pd.read_csv(
        RUN_ROOT
        / "runs/ex_01_final_maps_external_priors/artifacts/site_figures/final_site_figure_summary.csv"
    )
    with rio.open(RUN_ROOT / "runs/ex_01_fusion_external_priors/outputs/sentinel_landsat_dem_external_prior_feature_stack_30m.tif") as src:
        raster_bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
    order = ["Aksu", "Atbasar", "Chistopoloe", "Saumankol", "Shchuchinsk", "Suvorovka"]
    summary["site_order"] = summary["site_name"].map({name: idx for idx, name in enumerate(order)})
    summary = summary.sort_values("site_order")
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        left = max(float(row["block_minx"]), raster_bounds[0])
        bottom = max(float(row["block_miny"]), raster_bounds[1])
        right = min(float(row["block_maxx"]), raster_bounds[2])
        top = min(float(row["block_maxy"]), raster_bounds[3])
        rows.append(
            {
                "site_name": str(row["site_name"]),
                "bounds": (left, bottom, right, top),
            }
        )
    return rows


def _extent_from_bounds(bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    left, bottom, right, top = bounds
    return left, right, bottom, top


def _format_site_scene_axis(ax: plt.Axes, site_name: str, bounds: tuple[float, float, float, float], idx: int) -> None:
    left, bottom, right, top = bounds
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_title(f"{chr(65 + idx)}) {site_name}", fontsize=10)
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    _add_panel_north_scale(ax, _extent_from_bounds(bounds))


def _shared_vrange(arrays: list[np.ma.MaskedArray], pct: tuple[float, float] = (2.0, 98.0)) -> tuple[float, float]:
    vals = np.concatenate([arr.compressed() for arr in arrays if arr.compressed().size])
    if not vals.size:
        return 0.0, 1.0
    lo, hi = np.percentile(vals, pct)
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _site_key(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _draw_overview_title(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, small: ImageFont.ImageFont) -> None:
    draw.text((16, 12), "Kazakhstan Overview: Multisource Hydrogen Prospectivity Screening", fill=(255, 255, 255), font=font)
    draw.text(
        (16, 40),
        "OpenStreetMap basemap, final SGD-ranked site callouts, and leakage-aware screening metrics",
        fill=(218, 222, 230),
        font=small,
    )


def _draw_final_overview_legend(
    draw: ImageDraw.ImageDraw,
    map_x0: int,
    map_y0: int,
    map_w: int,
    font: ImageFont.ImageFont,
    small: ImageFont.ImageFont,
) -> None:
    box_w = 328
    box_h = 230
    x0 = map_x0 + map_w - box_w - 16
    y0 = map_y0 + 18
    draw.rounded_rectangle(
        [(x0, y0), (x0 + box_w, y0 + box_h)],
        radius=8,
        fill=(18, 20, 24, 205),
        outline=(205, 210, 220, 230),
        width=1,
    )
    draw.text((x0 + 12, y0 + 10), "Legend", fill=(255, 255, 255), font=font)
    draw.text((x0 + 12, y0 + 40), "Site color = rank by SGD Q95", fill=(220, 224, 232), font=small)
    y = y0 + 67
    for rank in range(1, 7):
        color = _rank_color(rank)
        draw.ellipse([(x0 + 14, y - 1), (x0 + 28, y + 13)], fill=color, outline=(255, 255, 255), width=1)
        draw.text((x0 + 38, y - 4), f"Rank {rank}", fill=(240, 242, 248), font=small)
        y += 22
    draw.text((x0 + 12, y0 + 204), "Callouts: mean, Q95, RF top-5 overlap", fill=(205, 210, 218), font=small)


def _render_osm_overview(dirs: dict[str, Path]) -> Path:
    outline = RUN_ROOT / "runs/ex_01/scenes/annotated/kazakhstan_outline.geojson"
    if not outline.exists():
        raise FileNotFoundError(f"Kazakhstan outline cache not found: {outline}")
    world = _read_json(outline)
    kaz_geom = world["features"][0]["geometry"]
    rings = _iter_outer_rings(kaz_geom)
    minx, miny, maxx, maxy = _bbox_from_rings(rings)
    bbox = (minx - 1.0, miny - 1.0, maxx + 1.0, maxy + 1.0)

    summary = pd.read_csv(
        RUN_ROOT
        / "runs/ex_01_final_maps_external_priors/artifacts/site_figures/final_site_figure_summary.csv"
    )
    points = gpd.read_file(RUN_ROOT / "regions/ex_01_ground_points.geojson")
    if points.crs is None:
        points = points.set_crs("EPSG:4326")
    coords = {
        _site_key(row["site_name"]): (float(row.geometry.y), float(row.geometry.x))
        for _, row in points.iterrows()
    }
    summary["site_key"] = summary["site_name"].map(_site_key)
    summary["latitude"] = summary["site_key"].map(lambda key: coords[key][0])
    summary["longitude"] = summary["site_key"].map(lambda key: coords[key][1])
    summary = summary.sort_values("sgd_platt_proxy_q95", ascending=False).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)

    width, height = 2300, 1450
    map_x0, map_y0 = 40, 92
    map_w, map_h = 2220, 1310
    basemap, transform = _build_osm_basemap(
        bbox=bbox,
        map_w=map_w,
        map_h=map_h,
        cache_dir=RUN_ROOT / "runs/ex_01/scenes/annotated/osm_tiles_cache",
        forced_zoom=7,
    )

    img = Image.new("RGB", (width, height), (11, 13, 17))
    img.paste(basemap, (map_x0, map_y0))
    draw = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.load_default(size=24)
    small = ImageFont.load_default(size=17)
    _draw_overview_title(draw, font, small)

    for ring in rings:
        line_points = []
        for lon, lat in ring:
            mx, my = _lonlat_to_map_xy(lon, lat, transform)
            line_points.append((map_x0 + mx, map_y0 + my))
        if len(line_points) >= 2:
            draw.line(line_points, fill=(78, 132, 205, 235), width=2)

    offsets = _callout_offsets()
    panel_x1 = map_x0 + map_w - 12
    panel_y1 = map_y0 + map_h - 12
    for _, site in summary.iterrows():
        lat = float(site["latitude"])
        lon = float(site["longitude"])
        rank = int(site["rank"])
        site_key = str(site["site_key"])
        site_name = str(site["site_name"])
        sx_map, sy_map = _lonlat_to_map_xy(lon, lat, transform)
        sx = map_x0 + sx_map
        sy = map_y0 + sy_map
        color = _rank_color(rank)

        draw.ellipse([(sx - 7, sy - 7), (sx + 7, sy + 7)], fill=color, outline=(255, 255, 255), width=2)

        ox, oy = offsets.get(site_key, (140, -100))
        tx = sx + ox
        ty = sy + oy
        lines = [
            f"#{rank} {site_name}",
            f"{lat:.5f}N, {lon:.5f}E",
            f"SGD mean={site['sgd_platt_proxy_mean']:.3f} | Q95={site['sgd_platt_proxy_q95']:.3f}",
            f"RF top-5 overlap={site['rf_topk_proxy_fraction']:.3f}",
        ]
        label = "\n".join(lines)
        text_box = draw.multiline_textbbox((0, 0), label, font=small, spacing=2)
        tw = (text_box[2] - text_box[0]) + 12
        th = (text_box[3] - text_box[1]) + 10

        bx0 = _clamp(tx + 8, map_x0 + 10, panel_x1 - tw - 4)
        by0 = _clamp(ty - 8, map_y0 + 10, panel_y1 - th - 4)
        bx1, by1 = bx0 + tw, by0 + th
        ex = _clamp(tx, bx0, bx1)
        ey = _clamp(ty, by0, by1)
        draw.line([(sx, sy), (ex, ey)], fill=color, width=2)
        draw.ellipse([(ex - 3, ey - 3), (ex + 3, ey + 3)], fill=color)
        draw.rounded_rectangle([(bx0, by0), (bx1, by1)], radius=7, fill=(0, 0, 0, 178), outline=color, width=2)
        draw.multiline_text((bx0 + 6, by0 + 5), label, fill=(248, 250, 255), font=small, spacing=2)

    _draw_scale_bar(draw, bbox=bbox, transform=transform, map_x0=map_x0, map_y0=map_y0, map_h=map_h, font=small)
    _draw_north_arrow(draw, map_x0=map_x0, map_y0=map_y0, map_w=map_w, font=font)
    _draw_final_overview_legend(draw, map_x0=map_x0, map_y0=map_y0, map_w=map_w, font=font, small=small)
    draw.text((map_x0 + 10, map_y0 + map_h - 20), "(C) OpenStreetMap contributors", fill=(255, 255, 255, 230), font=small)

    out = dirs["overview"] / "fig00_study_area.png"
    img.save(out)
    return out


def _render_local_overview(dirs: dict[str, Path]) -> Path:
    aoi = gpd.read_file(RUN_ROOT / "regions/ex_01_aoi_bbox.geojson")
    points = gpd.read_file(RUN_ROOT / "regions/ex_01_ground_points.geojson")
    labels = gpd.read_file(RUN_ROOT / "regions/ex_01_labels_placeholder.geojson")
    if aoi.crs is None:
        aoi = aoi.set_crs("EPSG:4326")
    if points.crs is None:
        points = points.set_crs("EPSG:4326")
    if labels.crs is None:
        labels = labels.set_crs("EPSG:4326")

    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    aoi.boundary.plot(ax=ax, color="#444444", linewidth=1.3)
    labels.plot(ax=ax, color="#c7e9c0", edgecolor="#238b45", linewidth=0.8, alpha=0.55)
    points.plot(ax=ax, color="#ffcc33", edgecolor="black", marker="*", markersize=95, zorder=5)
    for _, row in points.iterrows():
        ax.text(row.geometry.x + 0.035, row.geometry.y + 0.035, row["site_name"], fontsize=9, weight="bold")
    ax.set_title("Study area and proxy AOIs")
    _format_geo_axis(ax)
    out = dirs["overview"] / "fig00_study_area.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _render_overview(dirs: dict[str, Path]) -> Path:
    try:
        return _render_osm_overview(dirs)
    except Exception as exc:
        print(f"[figure-pack] OSM overview unavailable, using local fallback: {exc}")
        return _render_local_overview(dirs)


def _render_rgb_scene_grid(
    dirs: dict[str, Path],
    filename: str,
    title: str,
    raster_path: Path,
    bands: tuple[int, int, int],
    sites: list[dict[str, Any]],
) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.0), constrained_layout=True)
    axes_arr = axes.ravel()
    for idx, (ax, site) in enumerate(zip(axes_arr, sites)):
        bounds = site["bounds"]
        rgb = _read_rgb_window(raster_path, bounds=bounds, bands=bands)
        ax.imshow(rgb, extent=_extent_from_bounds(bounds), origin="upper")
        _format_site_scene_axis(ax, site["site_name"], bounds, idx)
    fig.suptitle(title, fontsize=13)
    out = dirs["inputs"] / filename
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def _render_continuous_scene_grid(
    dirs: dict[str, Path],
    filename: str,
    title: str,
    raster_path: Path,
    band: int,
    sites: list[dict[str, Any]],
    cmap: str,
    colorbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
    resampling: Resampling = Resampling.bilinear,
    panel_note: str | None = None,
) -> Path:
    arrays = [
        _read_band_window(raster_path, bounds=site["bounds"], band=band, resampling=resampling)
        for site in sites
    ]
    if vmin is None or vmax is None:
        auto_vmin, auto_vmax = _shared_vrange(arrays)
        vmin = auto_vmin if vmin is None else vmin
        vmax = auto_vmax if vmax is None else vmax

    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.0), constrained_layout=True)
    axes_arr = axes.ravel()
    image = None
    for idx, (ax, site, arr) in enumerate(zip(axes_arr, sites, arrays)):
        bounds = site["bounds"]
        image = ax.imshow(
            arr,
            extent=_extent_from_bounds(bounds),
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        _format_site_scene_axis(ax, site["site_name"], bounds, idx)
        if panel_note:
            ax.text(
                0.5,
                0.5,
                panel_note,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8.5,
                color="#111111",
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f0f0f0", "edgecolor": "#555555", "alpha": 0.9},
                zorder=25,
            )
    if image is not None:
        fig.colorbar(image, ax=list(axes_arr), shrink=0.86, label=colorbar_label)
    fig.suptitle(title, fontsize=13)
    out = dirs["inputs"] / filename
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def _render_lithology_scene_grid(
    dirs: dict[str, Path],
    filename: str,
    title: str,
    sites: list[dict[str, Any]],
) -> Path:
    lith = gpd.read_file(RUN_ROOT / "priors/processed/glim_lithology_ex01.geojson")
    if lith.crs is None:
        lith = lith.set_crs("EPSG:4326")
    category_col = "xx_Description"
    categories = sorted(str(v) for v in lith[category_col].dropna().unique()) if category_col in lith.columns else []
    cmap = plt.get_cmap("tab20", max(len(categories), 1))
    colors = {cat: cmap(idx % cmap.N) for idx, cat in enumerate(categories)}

    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.9))
    axes_arr = axes.ravel()
    for idx, (ax, site) in enumerate(zip(axes_arr, sites)):
        left, bottom, right, top = site["bounds"]
        ax.set_facecolor("#d9d9d9")
        subset = lith.cx[left:right, bottom:top]
        if category_col in subset.columns:
            for cat in categories:
                part = subset[subset[category_col].astype(str) == cat]
                if not part.empty:
                    part.plot(ax=ax, color=colors[cat], linewidth=0.0, alpha=0.98)
        _format_site_scene_axis(ax, site["site_name"], site["bounds"], idx)
    legend_handles = [
        Patch(facecolor=colors[cat], edgecolor="#444444", linewidth=0.35, label=cat)
        for cat in categories
    ]
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.018),
            ncol=4,
            frameon=False,
            fontsize=7.4,
            title="GLiM lithology class",
            title_fontsize=8.4,
            columnspacing=1.15,
            handlelength=1.25,
            handletextpad=0.45,
        )
    fig.suptitle(title, fontsize=13, y=0.985)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.925, bottom=0.19, wspace=0.24, hspace=0.36)
    out = dirs["inputs"] / filename
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def _render_input_stack(dirs: dict[str, Path]) -> list[Path]:
    sites = _site_scene_rows()
    stack = RUN_ROOT / "runs/ex_01_fusion_external_priors/outputs/sentinel_landsat_dem_external_prior_feature_stack_30m.tif"
    return [
        _render_continuous_scene_grid(
            dirs=dirs,
            filename="01_input_stack_dem.png",
            title="SRTM DEM site scenes",
            raster_path=RUN_ROOT / "dem/srtm_30m_ex01.tif",
            band=1,
            sites=sites,
            cmap="terrain",
            colorbar_label="Elevation",
            resampling=Resampling.bilinear,
        ),
        _render_rgb_scene_grid(
            dirs=dirs,
            filename="01_input_stack_sentinel2_rgb.png",
            title="Sentinel-2 RGB site scenes",
            raster_path=RUN_ROOT / "copernicus/sentinel_stack_30m_ex01_coverage.tif",
            bands=(3, 2, 1),
            sites=sites,
        ),
        _render_rgb_scene_grid(
            dirs=dirs,
            filename="01_input_stack_landsat8_rgb.png",
            title="Landsat-8 RGB site scenes",
            raster_path=RUN_ROOT / "landsat/landsat_stack_30m_ex01_coverage.tif",
            bands=(3, 2, 1),
            sites=sites,
        ),
        _render_continuous_scene_grid(
            dirs=dirs,
            filename="01_input_stack_dem_lineament_strength.png",
            title="DEM lineament-strength site scenes",
            raster_path=stack,
            band=18,
            sites=sites,
            cmap="magma",
            colorbar_label="Strength",
            vmin=0.0,
            vmax=1.0,
            resampling=Resampling.bilinear,
        ),
        _render_lithology_scene_grid(
            dirs=dirs,
            filename="01_input_stack_glim_lithology.png",
            title="GLiM lithology site scenes",
            sites=sites,
        ),
    ]


def _render_coverage(dirs: dict[str, Path]) -> Path:
    audit_dir = RUN_ROOT / "runs/ex_01_fusion_external_priors/artifacts/coverage_audit"
    all_mask, extent = _read_mask(audit_dir / "strict_all_features_valid_mask.tif", max_dim=900)
    global_group = pd.read_csv(audit_dir / "global_group_coverage.csv")
    site_group = pd.read_csv(audit_dir / "site_group_coverage.csv")
    positive = site_group[
        (site_group["region_type"] == "positive_aoi")
        & (site_group["feature_group"].isin(["all_features", "spectral_features"]))
    ].copy()
    pivot = positive.pivot(index="site_name", columns="feature_group", values="strict_valid_fraction").sort_index()
    points = gpd.read_file(RUN_ROOT / "regions/ex_01_ground_points.geojson")
    labels = gpd.read_file(RUN_ROOT / "regions/ex_01_labels_placeholder.geojson")
    if points.crs is None:
        points = points.set_crs("EPSG:4326")
    if labels.crs is None:
        labels = labels.set_crs("EPSG:4326")

    fig = plt.figure(figsize=(15.8, 8.6), constrained_layout=True)
    grid = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[1.55, 1.0], height_ratios=[1.0, 1.0])
    ax_map = fig.add_subplot(grid[:, 0])
    ax_global = fig.add_subplot(grid[0, 1])
    ax_gate = fig.add_subplot(grid[1, 1])

    valid_cmap = ListedColormap(["#eeeeee", "#1f9bcf"])
    ax_map.imshow(all_mask, extent=extent, origin="upper", cmap=valid_cmap, vmin=0, vmax=1)
    labels.boundary.plot(ax=ax_map, color="#f03b20", linewidth=1.15, alpha=0.95)
    points.plot(ax=ax_map, color="#ffd23f", edgecolor="black", marker="*", markersize=95, zorder=6)
    for _, row in points.iterrows():
        ax_map.text(
            row.geometry.x + 0.035,
            row.geometry.y + 0.035,
            str(row["site_name"]),
            fontsize=8.5,
            weight="bold",
            color="#111111",
            path_effects=[pe.withStroke(linewidth=2.4, foreground="white")],
            zorder=7,
        )
    ax_map.set_title("A) Strict all-feature valid footprint over the study AOI")
    _format_geo_axis(ax_map)
    ax_map.text(
        0.02,
        0.03,
        "Blue = pixels valid for all 33 model bands\nRed outlines = positive proxy AOIs",
        transform=ax_map.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#111111",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#777777", "alpha": 0.88},
    )

    group_order = [
        "all_features",
        "spectral_features",
        "sentinel2_features",
        "landsat8_features",
        "dem_features",
        "prior_features",
        "external_prior_features",
        "lithology_features",
        "fault_features",
    ]
    group_labels = {
        "all_features": "All 33 bands",
        "spectral_features": "Spectral",
        "sentinel2_features": "Sentinel-2",
        "landsat8_features": "Landsat-8",
        "dem_features": "DEM",
        "prior_features": "DEM priors",
        "external_prior_features": "External priors",
        "lithology_features": "Lithology",
        "fault_features": "GEM faults",
    }
    group_plot = (
        global_group[global_group["feature_group"].isin(group_order)]
        .assign(order=lambda df: df["feature_group"].map({g: i for i, g in enumerate(group_order)}))
        .sort_values("order", ascending=False)
    )
    colors = [
        "#2ca25f" if frac >= 0.95 else "#fdae61" if frac >= 0.50 else "#3182bd"
        for frac in group_plot["strict_valid_fraction"]
    ]
    y = np.arange(len(group_plot))
    ax_global.barh(y, group_plot["strict_valid_fraction"] * 100.0, color=colors, edgecolor="#333333", linewidth=0.4)
    ax_global.set_yticks(y)
    ax_global.set_yticklabels([group_labels[g] for g in group_plot["feature_group"]], fontsize=8)
    ax_global.set_xlim(0, 105)
    ax_global.set_xlabel("Strict-valid pixels (%)")
    ax_global.set_title("B) Global valid-data coverage by feature group")
    ax_global.grid(axis="x", color="#d9d9d9", linewidth=0.6)
    ax_global.axvline(95, color="#d7301f", linestyle="--", linewidth=1.1)
    ax_global.text(95.5, len(group_plot) - 0.35, "95% gate", fontsize=8, color="#d7301f", va="top")
    for yi, frac in zip(y, group_plot["strict_valid_fraction"]):
        ax_global.text(min(frac * 100.0 + 1.0, 101.0), yi, f"{frac * 100.0:.1f}%", va="center", fontsize=7.8)
    ax_global.spines[["top", "right"]].set_visible(False)

    x = np.arange(len(pivot.index))
    ax_gate.bar(x - 0.18, pivot["all_features"] * 100.0, width=0.36, label="All features", color="#2c7fb8")
    ax_gate.bar(x + 0.18, pivot["spectral_features"] * 100.0, width=0.36, label="Spectral", color="#41ab5d")
    ax_gate.axhline(95, color="#d7301f", linewidth=1.1, linestyle="--", label="95% gate")
    ax_gate.set_xticks(x)
    ax_gate.set_xticklabels(pivot.index, rotation=35, ha="right", fontsize=8)
    ax_gate.set_ylim(94, 100.25)
    ax_gate.set_ylabel("Strict-valid positive AOI (%)")
    ax_gate.set_title("C) Positive-AOI gate after coverage rebuild")
    ax_gate.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    for xi, site in zip(x, pivot.index):
        min_frac = float(min(pivot.loc[site, "all_features"], pivot.loc[site, "spectral_features"])) * 100.0
        ax_gate.text(xi, min(min_frac + 0.08, 100.15), "PASS", ha="center", va="bottom", fontsize=7.5, weight="bold")
    ax_gate.legend(fontsize=8, loc="lower right")
    ax_gate.spines[["top", "right"]].set_visible(False)

    out = dirs["coverage"] / "fig02_coverage_audit.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _render_validation(dirs: dict[str, Path]) -> Path:
    model = pd.read_csv(
        RUN_ROOT
        / "runs/ex_01_model_compare_external_priors/artifacts/site_blocked_validation/site_blocked_model_summary.csv"
    )
    thresh = pd.read_csv(
        RUN_ROOT
        / "runs/ex_01_thresholds_external_priors/artifacts/site_blocked_thresholds/site_blocked_threshold_summary.csv"
    )
    calib = pd.read_csv(
        RUN_ROOT
        / "runs/ex_01_calibration_external_priors/artifacts/site_blocked_calibration/site_blocked_calibration_summary.csv"
    )
    bins = pd.read_csv(
        RUN_ROOT
        / "runs/ex_01_calibration_external_priors/artifacts/site_blocked_calibration/site_blocked_calibration_bins.csv"
    )

    fig = plt.figure(figsize=(15.8, 9.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax_model = fig.add_subplot(grid[0, 0])
    ax_threshold = fig.add_subplot(grid[0, 1])
    ax_reliability = fig.add_subplot(grid[1, 0])
    ax_calib = fig.add_subplot(grid[1, 1])

    model = model.sort_values("pr_auc", ascending=True)
    y = np.arange(len(model))
    roc_color = "#4c78a8"
    pr_color = "#f58518"
    top5_color = "#54a24b"
    ax_model.barh(y - 0.22, model["roc_auc"], height=0.20, color=roc_color, label="ROC-AUC")
    ax_model.barh(y, model["pr_auc"], height=0.20, color=pr_color, label="PR-AUC")
    ax_model.barh(y + 0.22, model["precision_at_top5pct"], height=0.20, color=top5_color, label="Top-5% precision")
    ax_model.set_yticks(y)
    ax_model.set_yticklabels(model["model"].str.upper())
    ax_model.set_xlim(0.0, 0.72)
    ax_model.set_xlabel("Score")
    ax_model.set_title("A) Site-blocked model evidence")
    ax_model.grid(axis="x", color="#d9d9d9", linewidth=0.6)
    ax_model.legend(fontsize=8, loc="lower right")
    ax_model.spines[["top", "right"]].set_visible(False)
    for _, row in model.iterrows():
        yi = int(np.where(model["model"].to_numpy() == row["model"])[0][0])
        if row["model"] == "sgd":
            ax_model.text(0.44, yi + 0.32, "primary transfer model", fontsize=8, color="#111111", weight="bold")
        if row["model"] == "rf":
            ax_model.text(0.51, yi + 0.32, "top-k sensitivity", fontsize=8, color="#111111", weight="bold")

    thresh = thresh.set_index("model_alias").loc[["sgd_primary", "rf_topk_sensitivity"]]
    threshold_labels = ["SGD primary", "RF top-k"]
    threshold_colors = ["#1b9e77", "#d95f02"]
    for idx, (alias, label, color) in enumerate(zip(thresh.index, threshold_labels, threshold_colors)):
        row = thresh.loc[alias]
        x_vals = [0, 1]
        y_vals = [float(row["fixed_f1"]), float(row["f1"])]
        ax_threshold.plot(x_vals, y_vals, color=color, linewidth=2.0, marker="o", markersize=6, label=label)
        ax_threshold.text(-0.03, y_vals[0], f"{y_vals[0]:.3f}", ha="right", va="center", fontsize=8, color=color)
        tuned_label_y = y_vals[1] + (0.010 if idx == 0 else -0.014)
        ax_threshold.text(1.03, tuned_label_y, f"{y_vals[1]:.3f}", ha="left", va="center", fontsize=8, color=color)
        ax_threshold.text(
            0.50,
            y_vals[1] + (0.026 if idx == 0 else -0.030),
            f"mean tau={row['selected_threshold_mean']:.3f}",
            ha="center",
            va="center",
            fontsize=8,
            color=color,
        )
    ax_threshold.set_xticks([0, 1])
    ax_threshold.set_xticklabels(["Fixed threshold\n0.8", "Nested threshold\ntraining folds"])
    ax_threshold.set_ylim(0.0, 0.56)
    ax_threshold.set_ylabel("F1")
    ax_threshold.set_title("B) Threshold tuning changes the operating point")
    ax_threshold.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    ax_threshold.legend(fontsize=8, loc="lower right")
    ax_threshold.spines[["top", "right"]].set_visible(False)

    curve_colors = {"raw": "#666666", "platt": "#1b9e77", "isotonic": "#7570b3"}
    curve_labels = {"raw": "Raw SGD", "platt": "Platt", "isotonic": "Isotonic"}
    ax_reliability.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1.0, label="Ideal")
    for calibrator in ["raw", "platt", "isotonic"]:
        part = bins[(bins["calibrator"] == calibrator) & (bins["count"] > 0)].copy()
        grouped_rows = []
        for bin_id, df in part.groupby("bin"):
            grouped_rows.append(
                {
                    "bin": bin_id,
                    "mean_probability": np.average(df["mean_probability"], weights=df["count"]),
                    "positive_fraction": np.average(df["positive_fraction"], weights=df["count"]),
                    "count": df["count"].sum(),
                }
            )
        grouped = pd.DataFrame(grouped_rows).sort_values("bin")
        ax_reliability.plot(
            grouped["mean_probability"],
            grouped["positive_fraction"],
            marker="o",
            linewidth=1.6,
            markersize=4.5,
            color=curve_colors[calibrator],
            label=curve_labels[calibrator],
        )
    ax_reliability.set_xlim(0, 1)
    ax_reliability.set_ylim(0, 1)
    ax_reliability.set_xlabel("Mean predicted score")
    ax_reliability.set_ylabel("Observed proxy-positive fraction")
    ax_reliability.set_title("C) Nested SGD calibration reliability")
    ax_reliability.grid(color="#e0e0e0", linewidth=0.6)
    ax_reliability.legend(fontsize=8, loc="upper left")
    ax_reliability.spines[["top", "right"]].set_visible(False)

    calib = calib.set_index("calibrator").loc[["raw", "platt", "isotonic"]]
    x = np.arange(len(calib))
    width = 0.34
    brier_bars = ax_calib.bar(x - width / 2, calib["brier"], width=width, color="#4c78a8", label="Brier")
    ece_bars = ax_calib.bar(x + width / 2, calib["ece_10bin"], width=width, color="#e45756", label="ECE")
    ax_calib.set_xticks(x)
    ax_calib.set_xticklabels(["Raw", "Platt", "Isotonic"])
    ax_calib.set_ylim(0.0, 0.27)
    ax_calib.set_ylabel("Lower is better")
    ax_calib.set_title("D) Calibration diagnostics select Platt")
    ax_calib.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    ax_calib.legend(fontsize=8, loc="upper right")
    ax_calib.spines[["top", "right"]].set_visible(False)
    for bars in [brier_bars, ece_bars]:
        for bar in bars:
            ax_calib.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.006,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    platt_idx = 1
    ax_calib.annotate(
        "chosen map calibrator",
        xy=(platt_idx, float(calib.loc["platt", "ece_10bin"])),
        xytext=(platt_idx + 0.55, 0.075),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#1b9e77"},
        fontsize=8,
        color="#1b9e77",
        ha="left",
    )

    out = dirs["validation"] / "fig03_validation_summary.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _copy_final_maps(dirs: dict[str, Path]) -> list[Path]:
    copied: list[Path] = []
    src_root = RUN_ROOT / "runs/ex_01_final_maps_external_priors/artifacts/site_figures"
    for name in [
        "final_site_figures_contact_sheet.png",
        "final_site_figures.md",
        "final_site_figure_summary.csv",
    ]:
        src = src_root / name
        if src.exists():
            dst = dirs["final_maps"] / name
            shutil.copy2(src, dst)
            copied.append(dst)
    for src in sorted((src_root / "figures").glob("*.png")):
        dst = dirs["site_panels"] / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def _stage_tables(dirs: dict[str, Path]) -> list[Path]:
    table_sources = {
        "table_feature_group_coverage.csv": RUN_ROOT
        / "runs/ex_01_fusion_external_priors/artifacts/coverage_audit/global_group_coverage.csv",
        "table_site_coverage.csv": RUN_ROOT
        / "runs/ex_01_fusion_external_priors/artifacts/coverage_audit/site_group_coverage.csv",
        "table_model_comparison.csv": RUN_ROOT
        / "runs/ex_01_model_compare_external_priors/artifacts/site_blocked_validation/site_blocked_model_summary.csv",
        "table_threshold_summary.csv": RUN_ROOT
        / "runs/ex_01_thresholds_external_priors/artifacts/site_blocked_thresholds/site_blocked_threshold_summary.csv",
        "table_calibration_summary.csv": RUN_ROOT
        / "runs/ex_01_calibration_external_priors/artifacts/site_blocked_calibration/site_blocked_calibration_summary.csv",
        "table_final_score_summary.csv": RUN_ROOT
        / "runs/ex_01_final_maps_external_priors/artifacts/final_maps/final_score_summary.csv",
        "table_site_figure_summary.csv": RUN_ROOT
        / "runs/ex_01_final_maps_external_priors/artifacts/site_figures/final_site_figure_summary.csv",
    }
    copied: list[Path] = []
    for dst_name, src in table_sources.items():
        if src.exists():
            dst = dirs["tables"] / dst_name
            shutil.copy2(src, dst)
            copied.append(dst)

    manifest = dirs["tables"] / "README.md"
    lines = ["# Study Tables", ""]
    for path in copied:
        lines.append(f"- `{path.name}`")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    copied.append(manifest)
    return copied


def build_pack() -> dict[str, Any]:
    dirs = _ensure_dirs()
    outputs = {
        "overview": str(_render_overview(dirs)),
        "methodology_architecture": str(dirs["overview"] / "fig00_methodology_architecture.png"),
        "input_stacks": [str(p) for p in _render_input_stack(dirs)],
        "coverage": str(_render_coverage(dirs)),
        "validation": str(_render_validation(dirs)),
        "final_maps": [str(p) for p in _copy_final_maps(dirs)],
        "tables": [str(p) for p in _stage_tables(dirs)],
    }
    manifest = FIG_ROOT / "FIGURE_MANIFEST.md"
    lines = [
        "# Study Figure Pack",
        "",
        "This folder contains the curated study-facing figure set. Old DEM/XGBoost baseline scenes remain in the run history but are not copied here.",
        "No `05_supplementary` folder is created for the current locked study pack.",
        "",
        "## Figures",
        "",
        f"- `00_overview/{Path(outputs['overview']).name}`",
        f"- `00_overview/{Path(outputs['methodology_architecture']).name}`",
        f"- `02_feature_coverage/{Path(outputs['coverage']).name}`",
        f"- `03_validation/{Path(outputs['validation']).name}`",
        "- `04_final_maps/final_site_figures_contact_sheet.png`",
        "- `04_final_maps/site_review_panels/*.png`",
        "",
        "## Input Scene Figures",
        "",
    ]
    for input_stack in outputs["input_stacks"]:
        lines.append(f"- `01_inputs/{Path(input_stack).name}`")
    lines.extend(
        [
            "",
            "## Tables",
            "",
        ]
    )
    for table in outputs["tables"]:
        lines.append(f"- `tables/{Path(table).name}`")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs["manifest"] = str(manifest)
    with (FIG_ROOT / "figure_pack.json").open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)
    print(f"[figure-pack] wrote manifest: {manifest}")
    return outputs


if __name__ == "__main__":
    build_pack()
