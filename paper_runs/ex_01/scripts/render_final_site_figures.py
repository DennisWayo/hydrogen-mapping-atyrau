#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from matplotlib.colors import ListedColormap
from PIL import Image, ImageDraw, ImageFont
from rasterio import features
from rasterio.transform import array_bounds
from rasterio.windows import Window, from_bounds

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for import_path in [SRC_ROOT, SCRIPT_ROOT]:
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from xaigis.utils import ensure_dir

from validate_site_blocked import _buffer_site_geometry, _union_geometry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render site-level final prospectivity review figures from final ex_01 rasters."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "paper_runs/ex_01/configs/ex_01_final_site_figures.json",
        help="Site-figure rendering config.",
    )
    return parser.parse_args()


def _load_config(path: Path) -> tuple[dict[str, Any], Path, Path]:
    cfg_path = path.expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    final_maps_json = Path(cfg["final_maps_json"]).expanduser()
    if not final_maps_json.is_absolute():
        final_maps_json = (cfg_path.parent / final_maps_json).resolve()

    output_dir = Path(cfg["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = (cfg_path.parent / output_dir).resolve()

    return cfg, final_maps_json, output_dir


def _slug(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in text.strip())
    return "_".join(part for part in out.split("_") if part)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _window_for_bounds(src: rio.io.DatasetReader, bounds: tuple[float, float, float, float]) -> Window:
    win = from_bounds(*bounds, transform=src.transform)
    col0 = max(0, int(math.floor(win.col_off)))
    row0 = max(0, int(math.floor(win.row_off)))
    col1 = min(src.width, int(math.ceil(win.col_off + win.width)))
    row1 = min(src.height, int(math.ceil(win.row_off + win.height)))
    if col1 <= col0 or row1 <= row0:
        raise ValueError(f"Bounds do not overlap raster: {bounds}")
    return Window(col_off=col0, row_off=row0, width=col1 - col0, height=row1 - row0)


def _read_masked(src: rio.io.DatasetReader, window: Window) -> np.ma.MaskedArray:
    arr = src.read(1, window=window)
    mask = ~np.isfinite(arr)
    if src.nodata is not None:
        mask |= arr == src.nodata
    return np.ma.array(arr, mask=mask)


def _window_extent(src: rio.io.DatasetReader, window: Window) -> tuple[float, float, float, float]:
    transform = src.window_transform(window)
    west, south, east, north = array_bounds(int(window.height), int(window.width), transform)
    return (west, east, south, north)


def _geometry_mask_for_window(
    geom: Any,
    src: rio.io.DatasetReader,
    window: Window,
) -> np.ndarray:
    return features.geometry_mask(
        [geom.__geo_interface__],
        out_shape=(int(window.height), int(window.width)),
        transform=src.window_transform(window),
        invert=True,
    )


def _plot_boundaries(ax: plt.Axes, site_gdf: gpd.GeoDataFrame, point_gdf: gpd.GeoDataFrame) -> None:
    site_gdf.boundary.plot(ax=ax, color="white", linewidth=1.8, alpha=0.95)
    site_gdf.boundary.plot(ax=ax, color="black", linewidth=0.7, alpha=0.85)
    if not point_gdf.empty:
        point_gdf.plot(
            ax=ax,
            color="#ffdd55",
            edgecolor="black",
            linewidth=0.8,
            markersize=32,
            marker="*",
            zorder=5,
        )


def _format_axis(ax: plt.Axes, extent: tuple[float, float, float, float]) -> None:
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("Longitude", fontsize=7)
    ax.set_ylabel("Latitude", fontsize=7)
    ax.tick_params(labelsize=6)


def _site_stats(
    sgd: np.ma.MaskedArray,
    rf_score: np.ma.MaskedArray,
    rf_zone: np.ma.MaskedArray,
    valid: np.ma.MaskedArray,
    site_mask: np.ndarray,
) -> dict[str, Any]:
    valid_data = np.asarray(valid.filled(0) > 0, dtype=bool)
    region = site_mask & valid_data
    valid_fraction = float(region.sum() / max(int(site_mask.sum()), 1))
    sgd_vals = np.ma.array(sgd, copy=False)[region].compressed().astype(np.float32)
    rf_score_vals = np.ma.array(rf_score, copy=False)[region].compressed().astype(np.float32)
    rf_vals = np.asarray(rf_zone.filled(0)[region] > 0, dtype=bool)
    if sgd_vals.size:
        sgd_mean = float(sgd_vals.mean())
        sgd_q95 = float(np.quantile(sgd_vals, 0.95))
    else:
        sgd_mean = 0.0
        sgd_q95 = 0.0
    if rf_score_vals.size:
        rf_score_mean = float(rf_score_vals.mean())
        rf_score_q95 = float(np.quantile(rf_score_vals, 0.95))
    else:
        rf_score_mean = 0.0
        rf_score_q95 = 0.0
    rf_pixels = int(rf_vals.sum()) if rf_vals.size else 0
    rf_fraction = float(rf_pixels / max(int(region.sum()), 1))
    return {
        "proxy_pixels": int(site_mask.sum()),
        "strict_valid_proxy_pixels": int(region.sum()),
        "strict_valid_proxy_fraction": valid_fraction,
        "sgd_platt_proxy_mean": sgd_mean,
        "sgd_platt_proxy_q95": sgd_q95,
        "rf_score_proxy_mean": rf_score_mean,
        "rf_score_proxy_q95": rf_score_q95,
        "rf_topk_proxy_pixels": rf_pixels,
        "rf_topk_proxy_fraction": rf_fraction,
    }


def _render_site_figure(
    site_name: str,
    site_gdf: gpd.GeoDataFrame,
    point_gdf: gpd.GeoDataFrame,
    sgd_src: rio.io.DatasetReader,
    rf_score_src: rio.io.DatasetReader,
    rf_zone_src: rio.io.DatasetReader,
    valid_src: rio.io.DatasetReader,
    out_dir: Path,
    buffer_km: float,
    rf_zone_label: str,
    dpi: int,
) -> dict[str, Any]:
    site_geom = _union_geometry(site_gdf)
    block_geom = _buffer_site_geometry(site_gdf, sgd_src.crs, buffer_km)
    win = _window_for_bounds(sgd_src, tuple(float(v) for v in block_geom.bounds))
    extent = _window_extent(sgd_src, win)

    sgd = _read_masked(sgd_src, win)
    rf_score = _read_masked(rf_score_src, win)
    rf_zone = _read_masked(rf_zone_src, win)
    valid = _read_masked(valid_src, win)
    site_mask = _geometry_mask_for_window(site_geom, sgd_src, win)
    stats = _site_stats(sgd, rf_score, rf_zone, valid, site_mask)

    sgd_cmap = plt.get_cmap("viridis").copy()
    sgd_cmap.set_bad("#d9d9d9")
    rf_score_cmap = plt.get_cmap("plasma").copy()
    rf_score_cmap.set_bad("#d9d9d9")
    valid_cmap = ListedColormap(["#d9d9d9", "#2c7fb8"])
    zone_cmap = ListedColormap(["#eeeeee", "#d7191c"])
    valid_cmap.set_bad("#d9d9d9")
    zone_cmap.set_bad("#d9d9d9")

    fig, axes = plt.subplots(1, 4, figsize=(15.4, 3.25), constrained_layout=True)
    fig.suptitle(
        f"{site_name} final map review | "
        f"proxy valid={stats['strict_valid_proxy_fraction']:.3f} | "
        f"SGD mean={stats['sgd_platt_proxy_mean']:.3f}, q95={stats['sgd_platt_proxy_q95']:.3f} | "
        f"RF mean={stats['rf_score_proxy_mean']:.3f}, q95={stats['rf_score_proxy_q95']:.3f} | "
        f"RF {rf_zone_label} overlap={stats['rf_topk_proxy_fraction']:.3f}",
        fontsize=9,
    )

    im0 = axes[0].imshow(sgd, extent=extent, origin="upper", cmap=sgd_cmap, vmin=0.17, vmax=0.59)
    _plot_boundaries(axes[0], site_gdf, point_gdf)
    axes[0].set_title("SGD Platt prospectivity", fontsize=8)
    cb0 = fig.colorbar(im0, ax=axes[0], shrink=0.74)
    cb0.set_label("Score")
    _format_axis(axes[0], extent)

    im1 = axes[1].imshow(rf_score, extent=extent, origin="upper", cmap=rf_score_cmap, vmin=0.0, vmax=1.0)
    _plot_boundaries(axes[1], site_gdf, point_gdf)
    axes[1].set_title("RF ranking score", fontsize=8)
    cb1 = fig.colorbar(im1, ax=axes[1], shrink=0.74)
    cb1.set_label("Score")
    _format_axis(axes[1], extent)

    rf_arr = np.ma.array(np.asarray(rf_zone.filled(0) > 0, dtype=np.uint8), mask=sgd.mask)
    im2 = axes[2].imshow(rf_arr, extent=extent, origin="upper", cmap=zone_cmap, vmin=0, vmax=1)
    _plot_boundaries(axes[2], site_gdf, point_gdf)
    axes[2].set_title(f"RF {rf_zone_label} zone", fontsize=8)
    cb2 = fig.colorbar(im2, ax=axes[2], shrink=0.74, ticks=[0, 1])
    cb2.ax.set_yticklabels(["outside", "inside"])
    _format_axis(axes[2], extent)

    valid_arr = np.ma.array(np.asarray(valid.filled(0) > 0, dtype=np.uint8), mask=np.zeros(valid.shape, dtype=bool))
    im3 = axes[3].imshow(valid_arr, extent=extent, origin="upper", cmap=valid_cmap, vmin=0, vmax=1)
    _plot_boundaries(axes[3], site_gdf, point_gdf)
    axes[3].set_title("Strict valid-data coverage", fontsize=8)
    cb3 = fig.colorbar(im3, ax=axes[3], shrink=0.74, ticks=[0, 1])
    cb3.ax.set_yticklabels(["NoData", "valid"])
    _format_axis(axes[3], extent)

    out_path = out_dir / f"{_slug(site_name)}_final_map_review.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "site_name": site_name,
        "figure_png": str(out_path),
        "block_minx": float(block_geom.bounds[0]),
        "block_miny": float(block_geom.bounds[1]),
        "block_maxx": float(block_geom.bounds[2]),
        "block_maxy": float(block_geom.bounds[3]),
        **stats,
    }


def _write_contact_sheet(rows: list[dict[str, Any]], out_path: Path, dpi: int) -> None:
    n = len(rows)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    thumb_w = int(6.9 * dpi)
    pad = int(0.12 * dpi)
    label_h = int(0.18 * dpi)
    font = ImageFont.load_default(size=max(11, int(0.09 * dpi)))
    thumbs: list[tuple[dict[str, Any], Image.Image]] = []
    for row in rows:
        img = Image.open(row["figure_png"]).convert("RGB")
        scale = thumb_w / img.width
        thumb_h = max(1, int(round(img.height * scale)))
        img = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        thumbs.append((row, img))

    row_heights = []
    for row_idx in range(nrows):
        imgs = [img for _, img in thumbs[row_idx * ncols : (row_idx + 1) * ncols]]
        row_heights.append(max((img.height for img in imgs), default=0))

    sheet_w = ncols * thumb_w + (ncols + 1) * pad
    sheet_h = sum(label_h + h for h in row_heights) + (nrows + 1) * pad
    canvas = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (row, img) in enumerate(thumbs):
        row_idx = idx // ncols
        col_idx = idx % ncols
        x = pad + col_idx * (thumb_w + pad)
        y = pad + sum(label_h + h + pad for h in row_heights[:row_idx])
        draw.text((x, y), str(row["site_name"]), fill=(15, 15, 15), font=font)
        canvas.paste(img, (x, y + label_h))
    canvas.save(out_path)


def _write_markdown(path: Path, rows: list[dict[str, Any]], final_maps_json: Path, rf_zone: str) -> None:
    lines = [
        "# Final site-level map figures",
        "",
        f"- Final map metadata: `{final_maps_json}`",
        f"- RF sensitivity zone shown: `{rf_zone}`",
        "",
        "## Site Summary",
        "",
        "| Site | Proxy valid fraction | SGD mean | SGD Q95 | RF mean | RF Q95 | RF overlap fraction | Figure |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['site_name']} | {row['strict_valid_proxy_fraction']:.3f} | "
            f"{row['sgd_platt_proxy_mean']:.3f} | {row['sgd_platt_proxy_q95']:.3f} | "
            f"{row['rf_score_proxy_mean']:.3f} | {row['rf_score_proxy_q95']:.3f} | "
            f"{row['rf_topk_proxy_fraction']:.3f} | `{row['figure_png']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Note",
            "",
            "These figures are for site-level review of final rasters. The SGD panel is a calibrated prospectivity score for the sampled proxy-label distribution. The RF score panel and RF top-k zone are ranking-sensitivity diagnostics, not calibrated probability products.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def render_figures(config_path: Path) -> dict[str, Any]:
    cfg, final_maps_json, output_dir = _load_config(config_path)
    output_dir = ensure_dir(output_dir)
    figures_dir = ensure_dir(output_dir / "figures")
    final_maps = _read_json(final_maps_json)
    inputs = final_maps["inputs"]
    outputs = final_maps["outputs"]
    rf_zone = str(cfg.get("rf_zone", "top05pct"))
    rf_zone_rows = {row["zone"]: row for row in final_maps["rf_topk_zones"]}
    if rf_zone not in rf_zone_rows:
        raise ValueError(f"RF zone '{rf_zone}' not found. Available zones: {sorted(rf_zone_rows)}")

    label_gdf = gpd.read_file(inputs["label_geojson"])
    point_gdf = gpd.read_file(REPO_ROOT / "paper_runs/ex_01/regions/ex_01_ground_points.geojson")
    site_column = str(cfg.get("site_column", "site_name"))
    dpi = int(cfg.get("figure_dpi", 180))
    rows: list[dict[str, Any]] = []
    with rio.open(outputs["sgd_platt_prospectivity_score_tif"]) as sgd_src, rio.open(
        outputs["rf_topk_sensitivity_score_tif"]
    ) as rf_score_src, rio.open(rf_zone_rows[rf_zone]["mask_tif"]) as rf_zone_src, rio.open(
        outputs["strict_valid_data_mask_tif"]
    ) as valid_src:
        if label_gdf.crs is None:
            label_gdf = label_gdf.set_crs(sgd_src.crs)
        elif label_gdf.crs != sgd_src.crs:
            label_gdf = label_gdf.to_crs(sgd_src.crs)
        if point_gdf.crs is None:
            point_gdf = point_gdf.set_crs(sgd_src.crs)
        elif point_gdf.crs != sgd_src.crs:
            point_gdf = point_gdf.to_crs(sgd_src.crs)

        for site_name in sorted(str(site) for site in label_gdf[site_column].dropna().unique()):
            site_gdf = label_gdf[label_gdf[site_column].astype(str) == site_name]
            site_points = point_gdf[point_gdf[site_column].astype(str) == site_name]
            row = _render_site_figure(
                site_name=site_name,
                site_gdf=site_gdf,
                point_gdf=site_points,
                sgd_src=sgd_src,
                rf_score_src=rf_score_src,
                rf_zone_src=rf_zone_src,
                valid_src=valid_src,
                out_dir=figures_dir,
                buffer_km=float(cfg.get("block_buffer_km", 20.0)),
                rf_zone_label=rf_zone,
                dpi=dpi,
            )
            rows.append(row)
            print(f"[site-fig] rendered {site_name}: {row['figure_png']}", flush=True)

    contact_sheet = output_dir / "final_site_figures_contact_sheet.png"
    _write_contact_sheet(rows, contact_sheet, dpi=dpi)
    pd.DataFrame(rows).to_csv(output_dir / "final_site_figure_summary.csv", index=False)
    _write_markdown(output_dir / "final_site_figures.md", rows, final_maps_json=final_maps_json, rf_zone=rf_zone)
    result = {
        "config": str(Path(config_path).expanduser().resolve()),
        "final_maps_json": str(final_maps_json),
        "rf_zone": rf_zone,
        "figures_dir": str(figures_dir),
        "contact_sheet_png": str(contact_sheet),
        "summary_csv": str(output_dir / "final_site_figure_summary.csv"),
        "report_md": str(output_dir / "final_site_figures.md"),
        "sites": rows,
    }
    with (output_dir / "final_site_figures.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[site-fig] saved report: {output_dir / 'final_site_figures.md'}", flush=True)
    return result


def main() -> int:
    args = _parse_args()
    render_figures(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
