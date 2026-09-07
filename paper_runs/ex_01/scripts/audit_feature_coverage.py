#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio import features
from rasterio.windows import Window, from_bounds

# Local imports from repo.
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xaigis.config import load_config
from xaigis.utils import ensure_dir, load_json, save_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit strict feature coverage for ex_01 feature stacks. Strict-valid pixels "
            "are finite and do not equal the raster NoData value."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "paper_runs/ex_01/configs/ex_01_fusion_noleak_run.json",
        help="Run config containing feature stack, feature names, labels, and artifacts path.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <artifacts_dir>/coverage_audit.",
    )
    parser.add_argument(
        "--site-column",
        default="site_name",
        help="Column in the label polygon GeoJSON identifying each AOI/site.",
    )
    parser.add_argument(
        "--block-buffer-km",
        type=float,
        default=20.0,
        help="Metric buffer around each site polygon used to define local background.",
    )
    parser.add_argument(
        "--min-site-valid-frac",
        type=float,
        default=0.95,
        help="Minimum strict-valid fraction expected for each site/group.",
    )
    parser.add_argument(
        "--write-rasters",
        action="store_true",
        help=(
            "Also write aggregate coverage rasters: strict all-feature valid mask, "
            "strict spectral valid mask, and strict valid-band count."
        ),
    )
    return parser.parse_args()


def _load_feature_names(path: Path, n_features: int) -> list[str]:
    if path.exists():
        data = load_json(path)
        names = data.get("feature_names", [])
        if len(names) == n_features:
            return [str(name) for name in names]
    return [f"f{i:02d}" for i in range(n_features)]


def _feature_groups(feature_names: list[str]) -> dict[str, list[int]]:
    groups = {
        "all_features": list(range(len(feature_names))),
        "dem_features": [i for i, name in enumerate(feature_names) if name.startswith("DEM_")],
        "sentinel2_features": [i for i, name in enumerate(feature_names) if name.startswith("S2_")],
        "landsat8_features": [i for i, name in enumerate(feature_names) if name.startswith("L8_")],
        "prior_features": [i for i, name in enumerate(feature_names) if name.startswith("PRIOR_")],
        "lithology_features": [i for i, name in enumerate(feature_names) if name.startswith("LITHO_")],
        "geology_age_features": [i for i, name in enumerate(feature_names) if name.startswith("GEOAGE_")],
        "fault_features": [i for i, name in enumerate(feature_names) if name.startswith("FAULT_")],
    }
    spectral = sorted(set(groups["sentinel2_features"] + groups["landsat8_features"]))
    groups["spectral_features"] = spectral
    external = sorted(
        set(groups["lithology_features"] + groups["geology_age_features"] + groups["fault_features"])
    )
    groups["external_prior_features"] = external
    return {name: idxs for name, idxs in groups.items() if idxs}


def _fraction(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records"))


def _valid_arrays(arr: np.ndarray, nodata_value: float | int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(arr)
    nodata = np.zeros(arr.shape, dtype=bool)
    if nodata_value is not None:
        nodata = arr == nodata_value
    strict = finite & ~nodata
    return finite, nodata, strict


def _init_global_group_stats(groups: dict[str, list[int]]) -> dict[str, dict[str, int]]:
    return {
        group_name: {
            "pixels": 0,
            "finite_valid_pixels": 0,
            "strict_valid_pixels": 0,
            "pixels_with_nodata": 0,
        }
        for group_name in groups
    }


def _update_global_group_stats(
    stats: dict[str, dict[str, int]],
    groups: dict[str, list[int]],
    finite: np.ndarray,
    nodata: np.ndarray,
    strict: np.ndarray,
) -> None:
    pixels = int(finite.shape[1] * finite.shape[2])
    for group_name, idxs in groups.items():
        finite_all = finite[idxs].all(axis=0)
        nodata_any = nodata[idxs].any(axis=0)
        strict_all = strict[idxs].all(axis=0)
        stats[group_name]["pixels"] += pixels
        stats[group_name]["finite_valid_pixels"] += int(finite_all.sum())
        stats[group_name]["strict_valid_pixels"] += int(strict_all.sum())
        stats[group_name]["pixels_with_nodata"] += int(nodata_any.sum())


def _group_rows_for_mask(
    site_name: str,
    region_type: str,
    mask: np.ndarray,
    groups: dict[str, list[int]],
    finite: np.ndarray,
    nodata: np.ndarray,
    strict: np.ndarray,
    min_valid_frac: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pixels = int(mask.sum())
    for group_name, idxs in groups.items():
        finite_all = finite[idxs].all(axis=0)
        nodata_any = nodata[idxs].any(axis=0)
        strict_all = strict[idxs].all(axis=0)
        finite_valid = int((finite_all & mask).sum())
        strict_valid = int((strict_all & mask).sum())
        with_nodata = int((nodata_any & mask).sum())
        strict_frac = _fraction(strict_valid, pixels)
        rows.append(
            {
                "site_name": site_name,
                "region_type": region_type,
                "feature_group": group_name,
                "pixels": pixels,
                "finite_valid_pixels": finite_valid,
                "finite_valid_fraction": _fraction(finite_valid, pixels),
                "strict_valid_pixels": strict_valid,
                "strict_valid_fraction": strict_frac,
                "pixels_with_nodata": with_nodata,
                "nodata_fraction": _fraction(with_nodata, pixels),
                "coverage_gate_passed": bool(strict_frac >= min_valid_frac),
            }
        )
    return rows


def _band_rows_for_mask(
    site_name: str,
    region_type: str,
    mask: np.ndarray,
    feature_names: list[str],
    finite: np.ndarray,
    nodata: np.ndarray,
    strict: np.ndarray,
    min_valid_frac: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pixels = int(mask.sum())
    for band_idx, feature_name in enumerate(feature_names):
        finite_valid = int((finite[band_idx] & mask).sum())
        strict_valid = int((strict[band_idx] & mask).sum())
        with_nodata = int((nodata[band_idx] & mask).sum())
        strict_frac = _fraction(strict_valid, pixels)
        rows.append(
            {
                "site_name": site_name,
                "region_type": region_type,
                "band": band_idx + 1,
                "feature": feature_name,
                "pixels": pixels,
                "finite_valid_pixels": finite_valid,
                "finite_valid_fraction": _fraction(finite_valid, pixels),
                "strict_valid_pixels": strict_valid,
                "strict_valid_fraction": strict_frac,
                "pixels_with_nodata": with_nodata,
                "nodata_fraction": _fraction(with_nodata, pixels),
                "coverage_gate_passed": bool(strict_frac >= min_valid_frac),
            }
        )
    return rows


def _window_for_bounds(src: rio.io.DatasetReader, bounds: tuple[float, float, float, float]) -> Window:
    win = from_bounds(*bounds, transform=src.transform)
    col0 = max(0, int(math.floor(win.col_off)))
    row0 = max(0, int(math.floor(win.row_off)))
    col1 = min(src.width, int(math.ceil(win.col_off + win.width)))
    row1 = min(src.height, int(math.ceil(win.row_off + win.height)))
    if col1 <= col0 or row1 <= row0:
        raise ValueError(f"Geometry bounds do not overlap raster bounds: {bounds}")
    return Window(col_off=col0, row_off=row0, width=col1 - col0, height=row1 - row0)


def _utm_crs_for_geometry(gdf: gpd.GeoDataFrame) -> str:
    wgs = gdf.to_crs("EPSG:4326")
    centroid = wgs.geometry.unary_union.centroid
    lon = float(centroid.x)
    lat = float(centroid.y)
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def _union_geometry(gdf: gpd.GeoDataFrame) -> Any:
    if hasattr(gdf.geometry, "union_all"):
        return gdf.geometry.union_all()
    return gdf.geometry.unary_union


def _buffer_site_geometry(site_gdf: gpd.GeoDataFrame, raster_crs: Any, buffer_km: float) -> Any:
    site_gdf = site_gdf.to_crs(raster_crs)
    site_geom = _union_geometry(site_gdf)
    if buffer_km <= 0:
        return site_geom

    metric_crs = _utm_crs_for_geometry(site_gdf)
    metric = site_gdf.to_crs(metric_crs)
    buffered = _union_geometry(metric).buffer(buffer_km * 1000.0)
    buffered_gdf = gpd.GeoDataFrame({"geometry": [buffered]}, crs=metric_crs)
    return buffered_gdf.to_crs(raster_crs).geometry.iloc[0]


def _geometry_mask(geometries: list[Any], out_shape: tuple[int, int], transform: Any) -> np.ndarray:
    clean = [geom.__geo_interface__ for geom in geometries if geom is not None and not geom.is_empty]
    if not clean:
        return np.zeros(out_shape, dtype=bool)
    return features.geometry_mask(clean, out_shape=out_shape, transform=transform, invert=True)


def _global_audit(
    src: rio.io.DatasetReader,
    feature_names: list[str],
    groups: dict[str, list[int]],
    out_dir: Path,
    write_rasters: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    n_features = src.count
    total_by_band = np.zeros(n_features, dtype=np.int64)
    finite_by_band = np.zeros(n_features, dtype=np.int64)
    nodata_by_band = np.zeros(n_features, dtype=np.int64)
    strict_by_band = np.zeros(n_features, dtype=np.int64)
    group_stats = _init_global_group_stats(groups)
    written_rasters: list[Path] = []

    mask_writers: dict[str, rio.io.DatasetWriter] = {}
    if write_rasters:
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="uint8",
            count=1,
            nodata=255,
            compress="deflate",
            tiled=True,
            BIGTIFF="YES",
        )
        raster_paths = {
            "strict_all_features_valid_mask": out_dir / "strict_all_features_valid_mask.tif",
            "strict_spectral_features_valid_mask": out_dir / "strict_spectral_features_valid_mask.tif",
            "strict_valid_band_count": out_dir / "strict_valid_band_count.tif",
        }
        for key, path in raster_paths.items():
            mask_writers[key] = rio.open(path, "w", **profile)
            mask_writers[key].set_band_description(1, key)
            written_rasters.append(path)

    try:
        for block_idx, (_, win) in enumerate(src.block_windows(1), start=1):
            arr = src.read(window=win).astype(np.float32)
            finite, nodata, strict = _valid_arrays(arr, src.nodata)
            pixels = int(win.width * win.height)
            total_by_band += pixels
            finite_by_band += finite.reshape(n_features, -1).sum(axis=1)
            nodata_by_band += nodata.reshape(n_features, -1).sum(axis=1)
            strict_by_band += strict.reshape(n_features, -1).sum(axis=1)
            _update_global_group_stats(group_stats, groups, finite, nodata, strict)

            if mask_writers:
                all_valid = strict[groups["all_features"]].all(axis=0).astype(np.uint8)
                if "spectral_features" in groups:
                    spectral_valid = strict[groups["spectral_features"]].all(axis=0).astype(np.uint8)
                else:
                    spectral_valid = np.zeros(all_valid.shape, dtype=np.uint8)
                valid_band_count = strict.sum(axis=0).astype(np.uint8)
                mask_writers["strict_all_features_valid_mask"].write(all_valid, 1, window=win)
                mask_writers["strict_spectral_features_valid_mask"].write(spectral_valid, 1, window=win)
                mask_writers["strict_valid_band_count"].write(valid_band_count, 1, window=win)
            if block_idx % 200 == 0:
                print(f"[coverage] global blocks processed={block_idx:,}")
    finally:
        for writer in mask_writers.values():
            writer.close()

    band_rows = []
    for i, feature_name in enumerate(feature_names):
        total = int(total_by_band[i])
        band_rows.append(
            {
                "band": i + 1,
                "feature": feature_name,
                "pixels": total,
                "finite_valid_pixels": int(finite_by_band[i]),
                "finite_valid_fraction": _fraction(int(finite_by_band[i]), total),
                "strict_valid_pixels": int(strict_by_band[i]),
                "strict_valid_fraction": _fraction(int(strict_by_band[i]), total),
                "pixels_with_nodata": int(nodata_by_band[i]),
                "nodata_fraction": _fraction(int(nodata_by_band[i]), total),
            }
        )

    group_rows = []
    for group_name, stats in group_stats.items():
        pixels = int(stats["pixels"])
        group_rows.append(
            {
                "feature_group": group_name,
                "features": len(groups[group_name]),
                "pixels": pixels,
                "finite_valid_pixels": int(stats["finite_valid_pixels"]),
                "finite_valid_fraction": _fraction(int(stats["finite_valid_pixels"]), pixels),
                "strict_valid_pixels": int(stats["strict_valid_pixels"]),
                "strict_valid_fraction": _fraction(int(stats["strict_valid_pixels"]), pixels),
                "pixels_with_nodata": int(stats["pixels_with_nodata"]),
                "nodata_fraction": _fraction(int(stats["pixels_with_nodata"]), pixels),
            }
        )

    return pd.DataFrame(band_rows), pd.DataFrame(group_rows), written_rasters


def _site_audit(
    src: rio.io.DatasetReader,
    label_src: rio.io.DatasetReader,
    label_gdf: gpd.GeoDataFrame,
    feature_names: list[str],
    groups: dict[str, list[int]],
    site_column: str,
    block_buffer_km: float,
    min_valid_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if label_gdf.crs is None:
        label_gdf = label_gdf.set_crs(src.crs)
    elif label_gdf.crs != src.crs:
        label_gdf = label_gdf.to_crs(src.crs)

    site_names = sorted(str(site) for site in label_gdf[site_column].dropna().unique())
    site_group_rows: list[dict[str, Any]] = []
    site_band_rows: list[dict[str, Any]] = []

    for site_name in site_names:
        print(f"[coverage] site={site_name}")
        site_gdf = label_gdf[label_gdf[site_column].astype(str) == site_name]
        site_geom = _union_geometry(site_gdf)
        block_geom = _buffer_site_geometry(site_gdf, src.crs, block_buffer_km)
        win = _window_for_bounds(src, block_geom.bounds)
        h, w = int(win.height), int(win.width)
        transform = src.window_transform(win)

        arr = src.read(window=win).astype(np.float32)
        labels = label_src.read(1, window=win).astype(np.uint8)
        finite, nodata, strict = _valid_arrays(arr, src.nodata)

        site_mask = _geometry_mask([site_geom], out_shape=(h, w), transform=transform)
        block_mask = _geometry_mask([block_geom], out_shape=(h, w), transform=transform)
        other_sites = label_gdf[label_gdf[site_column].astype(str) != site_name]
        other_mask = _geometry_mask(list(other_sites.geometry), out_shape=(h, w), transform=transform)

        positive_mask = site_mask & (labels == 1)
        background_mask = block_mask & (labels == 0) & ~site_mask & ~other_mask

        for region_type, mask in [
            ("positive_aoi", positive_mask),
            ("local_background", background_mask),
        ]:
            site_group_rows.extend(
                _group_rows_for_mask(
                    site_name=site_name,
                    region_type=region_type,
                    mask=mask,
                    groups=groups,
                    finite=finite,
                    nodata=nodata,
                    strict=strict,
                    min_valid_frac=min_valid_frac,
                )
            )
            site_band_rows.extend(
                _band_rows_for_mask(
                    site_name=site_name,
                    region_type=region_type,
                    mask=mask,
                    feature_names=feature_names,
                    finite=finite,
                    nodata=nodata,
                    strict=strict,
                    min_valid_frac=min_valid_frac,
                )
            )

    return pd.DataFrame(site_group_rows), pd.DataFrame(site_band_rows)


def _write_markdown_report(
    path: Path,
    cfg_path: Path,
    stack_tif: Path,
    nodata_value: Any,
    shape: tuple[int, int, int],
    global_group_df: pd.DataFrame,
    site_group_df: pd.DataFrame,
    site_band_df: pd.DataFrame,
    written_rasters: list[Path],
    min_valid_frac: float,
) -> None:
    positive_groups = site_group_df[site_group_df["region_type"] == "positive_aoi"].copy()
    positive_all = positive_groups[positive_groups["feature_group"] == "all_features"].sort_values(
        "strict_valid_fraction"
    )
    positive_spectral = positive_groups[positive_groups["feature_group"] == "spectral_features"].sort_values(
        "strict_valid_fraction"
    )
    poor_positive_bands = (
        site_band_df[site_band_df["region_type"] == "positive_aoi"]
        .groupby("feature", as_index=False)["strict_valid_fraction"]
        .min()
        .sort_values("strict_valid_fraction")
        .head(12)
    )
    gate_failures = positive_groups[positive_groups["strict_valid_fraction"] < min_valid_frac]

    lines = [
        "# ex_01 Feature Coverage Audit",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Config: `{cfg_path}`",
        f"- Stack: `{stack_tif}`",
        f"- Shape: `{shape[0]} bands x {shape[1]} rows x {shape[2]} columns`",
        f"- Raster NoData value: `{nodata_value}`",
        f"- Coverage gate: strict-valid fraction >= `{min_valid_frac:.2f}`",
        "",
        "Strict-valid means a pixel is finite and does not equal the raster NoData value. "
        "This matters because the current training sampler excludes non-finite values but does not exclude finite NoData codes such as `-9999`.",
        "",
        "## Global Coverage by Feature Group",
        "",
        "| Feature group | Features | Strict valid fraction | NoData fraction | Finite-valid fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in global_group_df.sort_values("feature_group").to_dict("records"):
        lines.append(
            f"| {row['feature_group']} | {int(row['features'])} | "
            f"{row['strict_valid_fraction']:.4f} | {row['nodata_fraction']:.4f} | "
            f"{row['finite_valid_fraction']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Positive AOI Coverage",
            "",
            "| Site | All-feature strict valid | Spectral strict valid | Gate note |",
            "|---|---:|---:|---|",
        ]
    )
    spectral_by_site = {
        row["site_name"]: row
        for row in positive_spectral.to_dict("records")
    }
    for row in positive_all.to_dict("records"):
        site = row["site_name"]
        spectral = spectral_by_site.get(site, {})
        all_frac = float(row["strict_valid_fraction"])
        spectral_frac = float(spectral.get("strict_valid_fraction", 0.0))
        gate_note = "pass" if all_frac >= min_valid_frac else "fail"
        lines.append(f"| {site} | {all_frac:.4f} | {spectral_frac:.4f} | {gate_note} |")

    lines.extend(
        [
            "",
            "## Worst Positive-AOI Band Coverage",
            "",
            "| Feature | Minimum strict valid fraction across sites |",
            "|---|---:|",
        ]
    )
    for row in poor_positive_bands.to_dict("records"):
        lines.append(f"| {row['feature']} | {row['strict_valid_fraction']:.4f} |")

    lines.extend(
        [
            "",
            "## Risk Flags",
            "",
        ]
    )
    if gate_failures.empty:
        lines.append("- No positive-AOI feature-group coverage failures were found.")
    else:
        failed_sites = sorted(set(str(site) for site in gate_failures["site_name"]))
        lines.append(
            "- Coverage gate failed for positive AOIs at: "
            + ", ".join(failed_sites)
            + "."
        )
    lines.extend(
        [
            "- Existing random-split and site-blocked results should be interpreted cautiously until strict NoData exclusion is used in sampling and validation.",
            "- Rebuilding spectral mosaics should be prioritized before any new performance claims.",
            "",
            "## Output Files",
            "",
            "- `global_band_coverage.csv`",
            "- `global_group_coverage.csv`",
            "- `site_band_coverage.csv`",
            "- `site_group_coverage.csv`",
            "- `coverage_audit.json`",
        ]
    )
    for raster_path in written_rasters:
        lines.append(f"- `{raster_path.name}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_coverage_audit(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    paths = cfg["paths"]
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path(paths["artifacts_dir"]) / "coverage_audit"
    out_dir = ensure_dir(out_dir)

    stack_tif = Path(paths["feature_stack_tif"])
    feature_names_json = Path(paths["feature_names_json"])
    label_tif = Path(paths["label_tif"])
    label_geojson = Path(paths["geology_geojson"])

    if not stack_tif.exists():
        raise FileNotFoundError(f"Feature stack not found: {stack_tif}")
    if not label_tif.exists():
        raise FileNotFoundError(f"Label raster not found: {label_tif}")
    if not label_geojson.exists():
        raise FileNotFoundError(f"Label polygon GeoJSON not found: {label_geojson}")

    label_gdf = gpd.read_file(label_geojson)
    if label_gdf.empty:
        raise ValueError(f"No label polygons found: {label_geojson}")
    if args.site_column not in label_gdf.columns:
        raise ValueError(f"Missing site column '{args.site_column}' in {label_geojson}")

    with rio.open(stack_tif) as src, rio.open(label_tif) as label_src:
        if src.width != label_src.width or src.height != label_src.height:
            raise ValueError("Feature stack and label raster dimensions do not match.")
        feature_names = _load_feature_names(feature_names_json, src.count)
        groups = _feature_groups(feature_names)
        shape = (src.count, src.height, src.width)

        print(f"[coverage] stack: {stack_tif}")
        print(f"[coverage] shape: C={src.count}, H={src.height}, W={src.width}, nodata={src.nodata}")
        print("[coverage] auditing global feature coverage")
        global_band_df, global_group_df, written_rasters = _global_audit(
            src=src,
            feature_names=feature_names,
            groups=groups,
            out_dir=out_dir,
            write_rasters=bool(args.write_rasters),
        )

        print("[coverage] auditing site and local-background coverage")
        site_group_df, site_band_df = _site_audit(
            src=src,
            label_src=label_src,
            label_gdf=label_gdf,
            feature_names=feature_names,
            groups=groups,
            site_column=args.site_column,
            block_buffer_km=float(args.block_buffer_km),
            min_valid_frac=float(args.min_site_valid_frac),
        )

        metadata = {
            "generated": datetime.now(timezone.utc).isoformat(),
            "config": str(Path(args.config).resolve()),
            "feature_stack_tif": str(stack_tif),
            "feature_names_json": str(feature_names_json),
            "label_tif": str(label_tif),
            "label_geojson": str(label_geojson),
            "shape": {
                "bands": int(src.count),
                "height": int(src.height),
                "width": int(src.width),
            },
            "crs": str(src.crs),
            "nodata": src.nodata,
            "block_buffer_km": float(args.block_buffer_km),
            "min_site_valid_frac": float(args.min_site_valid_frac),
            "feature_names": feature_names,
            "feature_groups": {key: [feature_names[i] for i in idxs] for key, idxs in groups.items()},
            "written_rasters": [str(path) for path in written_rasters],
        }

    global_band_csv = out_dir / "global_band_coverage.csv"
    global_group_csv = out_dir / "global_group_coverage.csv"
    site_group_csv = out_dir / "site_group_coverage.csv"
    site_band_csv = out_dir / "site_band_coverage.csv"
    report_md = out_dir / "coverage_audit.md"
    audit_json = out_dir / "coverage_audit.json"

    global_band_df.to_csv(global_band_csv, index=False)
    global_group_df.to_csv(global_group_csv, index=False)
    site_group_df.to_csv(site_group_csv, index=False)
    site_band_df.to_csv(site_band_csv, index=False)

    result = {
        "metadata": metadata,
        "global_band_coverage": _records(global_band_df),
        "global_group_coverage": _records(global_group_df),
        "site_group_coverage": _records(site_group_df),
        "site_band_coverage": _records(site_band_df),
    }
    save_json(audit_json, result)
    _write_markdown_report(
        path=report_md,
        cfg_path=Path(args.config).resolve(),
        stack_tif=stack_tif,
        nodata_value=metadata["nodata"],
        shape=(int(metadata["shape"]["bands"]), int(metadata["shape"]["height"]), int(metadata["shape"]["width"])),
        global_group_df=global_group_df,
        site_group_df=site_group_df,
        site_band_df=site_band_df,
        written_rasters=written_rasters,
        min_valid_frac=float(args.min_site_valid_frac),
    )

    failures = site_group_df[
        (site_group_df["region_type"] == "positive_aoi")
        & (site_group_df["strict_valid_fraction"] < float(args.min_site_valid_frac))
    ]
    print(f"[coverage] saved: {report_md}")
    print(f"[coverage] coverage gate failures: {len(failures)} site/group rows")
    return result


def main() -> int:
    args = _parse_args()
    run_coverage_audit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
