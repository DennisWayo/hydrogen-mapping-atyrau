#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio import features
from scipy.ndimage import distance_transform_edt, uniform_filter

# Local imports from repo.
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dem_feature_stack import FEATURE_NAMES as DEM_FEATURE_NAMES
from build_dem_feature_stack import build_dem_feature_stack
from xaigis.config import load_config
from xaigis.utils import ensure_parent


NODATA_OUT = -9999.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fused DEM + Sentinel + Landsat + geology + geochem feature stack for ex_01."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "paper_runs/ex_01/configs/ex_01_fusion_run.json",
        help="Path to fusion config JSON.",
    )
    return parser.parse_args()


def _save_feature_names(path: Path, names: list[str]) -> None:
    out = ensure_parent(path)
    with out.open("w", encoding="utf-8") as f:
        json.dump({"feature_names": names}, f, indent=2)


def _load_points(points_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with points_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"No point rows found in {points_csv}")
    return rows


def _severity_from_hint(hint: str) -> float:
    h = hint.strip().lower()
    if h == "very_high_anomaly":
        return 3.0
    if h == "high_anomaly":
        return 2.0
    if h == "moderate_anomaly":
        return 1.0
    return 0.5


def _make_dem_only_cfg(cfg: dict[str, Any], dem_stack_tif: Path, dem_names_json: Path) -> dict[str, Any]:
    dem_cfg = dict(cfg)
    dem_cfg["paths"] = dict(cfg["paths"])
    dem_cfg["paths"]["feature_stack_tif"] = dem_stack_tif
    dem_cfg["paths"]["feature_names_json"] = dem_names_json
    return dem_cfg


def _append_geology_mask(
    dst: rio.io.DatasetWriter,
    band_idx: int,
    geology_geojson: Path,
    ref_crs: Any,
    ref_transform: Any,
    width: int,
    height: int,
) -> str:
    gdf = gpd.read_file(geology_geojson)
    if gdf.empty:
        mask = np.zeros((height, width), dtype=np.float32)
    else:
        if gdf.crs is None:
            gdf = gdf.set_crs(ref_crs)
        elif gdf.crs != ref_crs:
            gdf = gdf.to_crs(ref_crs)
        geoms = [(geom, 1.0) for geom in gdf.geometry if geom is not None and not geom.is_empty]
        mask = features.rasterize(
            geoms,
            out_shape=(height, width),
            transform=ref_transform,
            fill=0.0,
            dtype=np.float32,
        )
    dst.write(mask.astype(np.float32), band_idx)
    name = "GEOLOGY_MASK"
    dst.set_band_description(band_idx, name)
    return name


def _append_geochem_features(
    dst: rio.io.DatasetWriter,
    start_idx: int,
    points_csv: Path,
    transform: Any,
    width: int,
    height: int,
    dem_src: rio.io.DatasetReader,
) -> tuple[int, list[str]]:
    points = _load_points(points_csv)
    lons = np.array([float(r["longitude"]) for r in points], dtype=np.float64)
    lats = np.array([float(r["latitude"]) for r in points], dtype=np.float64)
    severities = np.array(
        [_severity_from_hint(str(r.get("target_hint", ""))) for r in points],
        dtype=np.float64,
    )

    col_coords = transform.c + (np.arange(width, dtype=np.float64) + 0.5) * transform.a
    row_coords = transform.f + (np.arange(height, dtype=np.float64) + 0.5) * transform.e

    block = 256
    eps = 1e-6
    dem_nodata = dem_src.nodata
    for r0 in range(0, height, block):
        r1 = min(height, r0 + block)
        lat_vec = row_coords[r0:r1]
        lat_grid = np.repeat(lat_vec[:, None], width, axis=1).astype(np.float64)
        lon_grid = np.repeat(col_coords[None, :], r1 - r0, axis=0).astype(np.float64)
        km_per_lon = np.maximum(111.320 * np.cos(np.deg2rad(lat_grid)), 1e-6)

        min_dist = np.full((r1 - r0, width), np.inf, dtype=np.float32)
        min_idx = np.zeros((r1 - r0, width), dtype=np.int32)
        wsum = np.zeros((r1 - r0, width), dtype=np.float32)
        vsum = np.zeros((r1 - r0, width), dtype=np.float32)

        for i in range(lons.size):
            dx = (lon_grid - lons[i]) * km_per_lon
            dy = (lat_grid - lats[i]) * 110.574
            dist = np.hypot(dx, dy).astype(np.float32)

            closer = dist < min_dist
            min_idx[closer] = i
            min_dist[closer] = dist[closer]

            w = 1.0 / ((dist + eps) ** 2)
            wsum += w.astype(np.float32)
            vsum += (w * severities[i]).astype(np.float32)

        nearest_sev = severities[min_idx].astype(np.float32)
        nearest_dist_km = min_dist.astype(np.float32)
        idw_sev = (vsum / np.maximum(wsum, eps)).astype(np.float32)

        win = rio.windows.Window(col_off=0, row_off=r0, width=width, height=r1 - r0)
        dem_patch = dem_src.read(1, window=win).astype(np.float32)
        invalid = ~np.isfinite(dem_patch)
        if dem_nodata is not None:
            invalid |= dem_patch == dem_nodata
        else:
            invalid |= dem_patch == NODATA_OUT

        nearest_sev[invalid] = NODATA_OUT
        nearest_dist_km[invalid] = NODATA_OUT
        idw_sev[invalid] = NODATA_OUT

        dst.write(nearest_sev, start_idx, window=win)
        dst.write(nearest_dist_km, start_idx + 1, window=win)
        dst.write(idw_sev, start_idx + 2, window=win)

    names = ["GEOCHEM_NEAREST_SEVERITY", "GEOCHEM_NEAREST_DIST_KM", "GEOCHEM_IDW_SEVERITY"]
    for i, name in enumerate(names, start=start_idx):
        dst.set_band_description(i, name)

    return start_idx + 3, names


def _append_ndvi(
    dst: rio.io.DatasetWriter,
    band_idx: int,
    red_band_idx: int,
    nir_band_idx: int,
    out_name: str,
) -> str:
    for _, win in dst.block_windows(red_band_idx):
        red = dst.read(red_band_idx, window=win).astype(np.float32)
        nir = dst.read(nir_band_idx, window=win).astype(np.float32)
        valid = np.isfinite(red) & np.isfinite(nir) & (red != NODATA_OUT) & (nir != NODATA_OUT)
        ndvi = np.full(red.shape, NODATA_OUT, dtype=np.float32)
        ndvi[valid] = (nir[valid] - red[valid]) / (nir[valid] + red[valid] + 1e-6)
        dst.write(ndvi, band_idx, window=win)
    dst.set_band_description(band_idx, out_name)
    return out_name


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


def _odd_window(value: Any, default: int) -> int:
    window = int(value if value is not None else default)
    window = max(3, window)
    return window if window % 2 == 1 else window + 1


def _feature_band_index(src: rio.io.DatasetReader, name: str) -> int:
    descriptions = [desc or f"B{i:02d}" for i, desc in enumerate(src.descriptions, start=1)]
    if name not in descriptions:
        raise ValueError(f"DEM feature stack is missing required band description: {name}")
    return descriptions.index(name) + 1


def _robust_unit_scale(values: np.ndarray, valid: np.ndarray, low_q: float = 2.0, high_q: float = 98.0) -> np.ndarray:
    out = np.zeros(values.shape, dtype=np.float32)
    valid_values = values[valid]
    if valid_values.size == 0:
        return out
    lo, hi = np.nanpercentile(valid_values, [low_q, high_q])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return out
    scaled = (values - float(lo)) / float(hi - lo)
    out[valid] = np.clip(scaled[valid], 0.0, 1.0).astype(np.float32)
    return out


def _write_full_array_band(
    dst: rio.io.DatasetWriter,
    band_idx: int,
    array: np.ndarray,
    name: str,
) -> None:
    for _, win in dst.block_windows(band_idx):
        row0 = int(win.row_off)
        row1 = row0 + int(win.height)
        col0 = int(win.col_off)
        col1 = col0 + int(win.width)
        dst.write(array[row0:row1, col0:col1].astype(np.float32), band_idx, window=win)
    dst.set_band_description(band_idx, name)


def _sanitize_feature_token(value: str, max_len: int = 32) -> str:
    token = "".join(ch if ch.isalnum() else "_" for ch in value.upper())
    while "__" in token:
        token = token.replace("__", "_")
    token = token.strip("_")
    return token[:max_len] or "UNKNOWN"


def _load_vector(path: Path, ref_crs: Any, assume_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        gdf = gdf.set_crs(assume_crs)
    if ref_crs is not None and gdf.crs != ref_crs:
        gdf = gdf.to_crs(ref_crs)
    return gdf


def _vector_classes(
    path: Path,
    field: str,
    exclude_values: list[str] | None = None,
    max_classes: int = 24,
) -> list[str]:
    gdf = gpd.read_file(path)
    if gdf.empty or field not in gdf.columns:
        return []
    excluded = {v.strip().lower() for v in (exclude_values or [])}
    values = [
        str(value).strip()
        for value in gdf[field].dropna().tolist()
        if str(value).strip() and str(value).strip().lower() not in excluded
    ]
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    ordered = sorted(counts, key=lambda value: (-counts[value], value))
    return ordered[: max(1, int(max_classes))]


def _append_polygon_one_hot_features(
    dst: rio.io.DatasetWriter,
    start_idx: int,
    vector_path: Path,
    field: str,
    classes: list[str],
    prefix: str,
    ref_crs: Any,
    ref_transform: Any,
    width: int,
    height: int,
) -> tuple[int, list[str]]:
    gdf = _load_vector(vector_path, ref_crs=ref_crs)
    names: list[str] = []
    b = start_idx
    if gdf.empty or field not in gdf.columns or not classes:
        for cls in classes:
            name = f"{prefix}_{_sanitize_feature_token(cls)}"
            arr = np.zeros((height, width), dtype=np.float32)
            _write_full_array_band(dst, b, arr, name)
            names.append(name)
            b += 1
        return b, names

    for cls in classes:
        subset = gdf[gdf[field].astype(str) == cls]
        geoms = [
            (geom, 1.0)
            for geom in subset.geometry
            if geom is not None and not geom.is_empty
        ]
        if geoms:
            arr = features.rasterize(
                geoms,
                out_shape=(height, width),
                transform=ref_transform,
                fill=0.0,
                all_touched=True,
                dtype=np.float32,
            )
        else:
            arr = np.zeros((height, width), dtype=np.float32)
        name = f"{prefix}_{_sanitize_feature_token(cls)}"
        _write_full_array_band(dst, b, arr, name)
        names.append(name)
        b += 1
    return b, names


def _valid_reference_mask(src: rio.io.DatasetReader) -> np.ndarray:
    arr = src.read(1).astype(np.float32)
    valid = np.isfinite(arr)
    if src.nodata is not None:
        valid &= arr != src.nodata
    valid &= arr != NODATA_OUT
    return valid


def _append_external_fault_priors(
    dst: rio.io.DatasetWriter,
    start_idx: int,
    faults_geojson: Path,
    dem_src: rio.io.DatasetReader,
    prior_cfg: dict[str, Any],
) -> tuple[int, list[str]]:
    gdf = _load_vector(faults_geojson, ref_crs=dem_src.crs)
    valid = _valid_reference_mask(dem_src)
    height, width = dem_src.height, dem_src.width
    if gdf.empty:
        fault_mask = np.zeros((height, width), dtype=bool)
    else:
        geoms = [
            (geom, 1)
            for geom in gdf.geometry
            if geom is not None and not geom.is_empty
        ]
        fault_mask = features.rasterize(
            geoms,
            out_shape=(height, width),
            transform=dem_src.transform,
            fill=0,
            all_touched=bool(prior_cfg.get("fault_all_touched", True)),
            dtype=np.uint8,
        ).astype(bool)

    density_window = _odd_window(prior_cfg.get("fault_density_window_px"), default=51)
    density = uniform_filter(fault_mask.astype(np.float32), size=density_window, mode="nearest").astype(np.float32)
    density[~valid] = NODATA_OUT

    x_res_m, y_res_m = _metric_resolution_m(dem_src)
    cap_km = max(float(prior_cfg.get("fault_distance_cap_km", 75.0)), 0.001)
    if np.any(fault_mask):
        distance = distance_transform_edt(~fault_mask, sampling=(y_res_m / 1000.0, x_res_m / 1000.0)).astype(np.float32)
        distance = np.minimum(distance, cap_km).astype(np.float32)
    else:
        distance = np.full((height, width), cap_km, dtype=np.float32)
    distance[~valid] = NODATA_OUT

    names = ["FAULT_ACTIVE_DIST_KM", "FAULT_ACTIVE_DENSITY"]
    _write_full_array_band(dst, start_idx, distance, names[0])
    _write_full_array_band(dst, start_idx + 1, density, names[1])
    print(
        "[fusion] external fault priors: "
        f"features={len(gdf)}, density_window_px={density_window}, distance_cap_km={cap_km:g}"
    )
    return start_idx + len(names), names


def _append_dem_lineament_priors(
    dst: rio.io.DatasetWriter,
    start_idx: int,
    dem_src: rio.io.DatasetReader,
    prior_cfg: dict[str, Any],
) -> tuple[int, list[str]]:
    """Append target-independent structural priors derived only from the DEM feature stack."""
    curvature_idx = _feature_band_index(dem_src, "DEM_CURVATURE")
    relief_idx = _feature_band_index(dem_src, "DEM_RELIEF")
    roughness_idx = _feature_band_index(dem_src, "DEM_ROUGHNESS")

    curvature = dem_src.read(curvature_idx).astype(np.float32)
    relief = dem_src.read(relief_idx).astype(np.float32)
    roughness = dem_src.read(roughness_idx).astype(np.float32)

    valid = np.isfinite(curvature) & np.isfinite(relief) & np.isfinite(roughness)
    if dem_src.nodata is not None:
        valid &= (curvature != dem_src.nodata) & (relief != dem_src.nodata) & (roughness != dem_src.nodata)
    valid &= (curvature != NODATA_OUT) & (relief != NODATA_OUT) & (roughness != NODATA_OUT)
    if not np.any(valid):
        raise ValueError("Cannot build DEM lineament priors because no valid DEM feature pixels were found.")

    curvature_norm = _robust_unit_scale(np.abs(curvature), valid)
    relief_norm = _robust_unit_scale(relief, valid)
    roughness_norm = _robust_unit_scale(roughness, valid)
    strength = (0.55 * curvature_norm + 0.30 * roughness_norm + 0.15 * relief_norm).astype(np.float32)
    strength[~valid] = NODATA_OUT

    q = float(prior_cfg.get("lineament_quantile", 0.92))
    q = min(max(q, 0.50), 0.995)
    threshold = float(np.nanquantile(strength[valid], q))
    core = valid & (strength >= threshold)
    if not np.any(core):
        threshold = float(np.nanquantile(strength[valid], 0.90))
        core = valid & (strength >= threshold)

    density_window = _odd_window(prior_cfg.get("lineament_density_window_px"), default=31)
    density = uniform_filter(core.astype(np.float32), size=density_window, mode="nearest").astype(np.float32)
    density[~valid] = NODATA_OUT

    x_res_m, y_res_m = _metric_resolution_m(dem_src)
    cap_km = max(float(prior_cfg.get("lineament_distance_cap_km", 25.0)), 0.001)
    distance = distance_transform_edt(~core, sampling=(y_res_m / 1000.0, x_res_m / 1000.0)).astype(np.float32)
    distance = np.minimum(distance, cap_km).astype(np.float32)
    distance[~valid] = NODATA_OUT

    names = [
        "PRIOR_LINEAMENT_STRENGTH",
        "PRIOR_LINEAMENT_DENSITY",
        "PRIOR_LINEAMENT_DIST_KM",
    ]
    _write_full_array_band(dst, start_idx, strength, names[0])
    _write_full_array_band(dst, start_idx + 1, density, names[1])
    _write_full_array_band(dst, start_idx + 2, distance, names[2])

    print(
        "[fusion] DEM lineament priors: "
        f"quantile={q:.3f}, threshold={threshold:.6g}, "
        f"density_window_px={density_window}, distance_cap_km={cap_km:g}"
    )
    return start_idx + len(names), names


def build_multisource_feature_stack(cfg: dict[str, Any]) -> dict[str, Any]:
    paths = cfg["paths"]
    feature_cfg = cfg.get("fusion_features", {})
    prior_cfg = cfg.get("prior_features", {})
    include_geology_mask = bool(feature_cfg.get("include_geology_mask", True))
    include_geochem_features = bool(feature_cfg.get("include_geochem_features", True))
    include_dem_lineament_priors = bool(prior_cfg.get("include_dem_lineament_priors", False))
    include_external_lithology = bool(prior_cfg.get("include_external_lithology", False))
    include_external_geology_age = bool(prior_cfg.get("include_external_geology_age", False))
    include_external_fault_priors = bool(prior_cfg.get("include_external_fault_priors", False))

    dem_stack_tif = Path(paths["dem_feature_stack_tif"])
    dem_names_json = Path(paths["dem_feature_names_json"])
    fused_stack_tif = ensure_parent(paths["feature_stack_tif"])
    fused_names_json = ensure_parent(paths["feature_names_json"])
    sentinel_stack = Path(paths["sentinel_stack_tif"])
    landsat_stack = Path(paths["landsat_stack_tif"])
    geology_geojson = Path(paths["geology_geojson"]) if include_geology_mask else None
    points_csv = Path(paths["geochem_points_csv"]) if include_geochem_features else None
    lithology_geojson = Path(paths["external_lithology_geojson"]) if include_external_lithology else None
    external_geology_geojson = Path(paths["external_geology_geojson"]) if include_external_geology_age else None
    faults_geojson = Path(paths["external_faults_geojson"]) if include_external_fault_priors else None

    dem_cfg = _make_dem_only_cfg(cfg=cfg, dem_stack_tif=dem_stack_tif, dem_names_json=dem_names_json)
    build_dem_feature_stack(dem_cfg)

    if not sentinel_stack.exists():
        raise FileNotFoundError(f"Sentinel stack missing: {sentinel_stack}")
    if not landsat_stack.exists():
        raise FileNotFoundError(f"Landsat stack missing: {landsat_stack}")
    if include_geology_mask and geology_geojson is not None and not geology_geojson.exists():
        raise FileNotFoundError(f"Geology labels missing: {geology_geojson}")
    if include_geochem_features and points_csv is not None and not points_csv.exists():
        raise FileNotFoundError(f"Geochem points CSV missing: {points_csv}")
    if include_external_lithology and (lithology_geojson is None or not lithology_geojson.exists()):
        raise FileNotFoundError(f"External lithology prior missing: {lithology_geojson}")
    if include_external_geology_age and (
        external_geology_geojson is None or not external_geology_geojson.exists()
    ):
        raise FileNotFoundError(f"External geology-age prior missing: {external_geology_geojson}")
    if include_external_fault_priors and (faults_geojson is None or not faults_geojson.exists()):
        raise FileNotFoundError(f"External fault prior missing: {faults_geojson}")

    with rio.open(dem_stack_tif) as dem_src, rio.open(sentinel_stack) as s2_src, rio.open(landsat_stack) as l8_src:
        if (dem_src.width, dem_src.height) != (s2_src.width, s2_src.height):
            raise ValueError("DEM and Sentinel stack dimensions do not match.")
        if (dem_src.width, dem_src.height) != (l8_src.width, l8_src.height):
            raise ValueError("DEM and Landsat stack dimensions do not match.")

        lithology_classes: list[str] = []
        geology_age_classes: list[str] = []
        if include_external_lithology:
            if lithology_geojson is None:
                raise ValueError("include_external_lithology=true but paths.external_lithology_geojson is missing.")
            lithology_classes = list(prior_cfg.get("lithology_classes") or [])
            if not lithology_classes:
                lithology_classes = _vector_classes(
                    lithology_geojson,
                    field=str(prior_cfg.get("lithology_field", "xx_Description")),
                    exclude_values=list(
                        prior_cfg.get(
                            "lithology_exclude_values",
                            ["No Data", "Water Bodies", "Ice and Glaciers"],
                        )
                    ),
                    max_classes=int(prior_cfg.get("lithology_max_classes", 24)),
                )
            if not lithology_classes:
                raise ValueError(f"No lithology classes found in {lithology_geojson}")

        if include_external_geology_age:
            if external_geology_geojson is None:
                raise ValueError(
                    "include_external_geology_age=true but paths.external_geology_geojson is missing."
                )
            geology_age_classes = list(prior_cfg.get("geology_age_classes") or [])
            if not geology_age_classes:
                geology_age_classes = _vector_classes(
                    external_geology_geojson,
                    field=str(prior_cfg.get("geology_age_field", "Glg")),
                    exclude_values=list(prior_cfg.get("geology_age_exclude_values", ["H2O", "Ice", "oth"])),
                    max_classes=int(prior_cfg.get("geology_age_max_classes", 24)),
                )
            if not geology_age_classes:
                raise ValueError(f"No geology age classes found in {external_geology_geojson}")

        profile = dem_src.profile.copy()
        total_bands = dem_src.count + s2_src.count + l8_src.count + 2
        if include_dem_lineament_priors:
            total_bands += 3
        if include_external_lithology:
            total_bands += len(lithology_classes)
        if include_external_geology_age:
            total_bands += len(geology_age_classes)
        if include_external_fault_priors:
            total_bands += 2
        if include_geology_mask:
            total_bands += 1
        if include_geochem_features:
            total_bands += 3
        profile.update(
            driver="GTiff",
            dtype="float32",
            count=total_bands,
            nodata=NODATA_OUT,
            compress="deflate",
            predictor=3,
            tiled=True,
            BIGTIFF="YES",
        )

        with rio.open(fused_stack_tif, "w+", **profile) as dst:
            feature_names: list[str] = []
            b = 1

            # DEM bands.
            for i in range(1, dem_src.count + 1):
                for _, win in dem_src.block_windows(i):
                    arr = dem_src.read(i, window=win).astype(np.float32)
                    dst.write(arr, b, window=win)
                name = dem_src.descriptions[i - 1] or f"DEM_B{i:02d}"
                dst.set_band_description(b, name)
                feature_names.append(name)
                b += 1

            # Sentinel bands.
            for i in range(1, s2_src.count + 1):
                for _, win in s2_src.block_windows(i):
                    arr = s2_src.read(i, window=win).astype(np.float32)
                    dst.write(arr, b, window=win)
                name = s2_src.descriptions[i - 1] or f"S2_B{i:02d}"
                dst.set_band_description(b, name)
                feature_names.append(name)
                b += 1
            s2_red_idx = feature_names.index("S2_RED") + 1 if "S2_RED" in feature_names else None
            s2_nir_idx = feature_names.index("S2_NIR") + 1 if "S2_NIR" in feature_names else None

            # Landsat bands.
            for i in range(1, l8_src.count + 1):
                for _, win in l8_src.block_windows(i):
                    arr = l8_src.read(i, window=win).astype(np.float32)
                    dst.write(arr, b, window=win)
                name = l8_src.descriptions[i - 1] or f"L8_B{i:02d}"
                dst.set_band_description(b, name)
                feature_names.append(name)
                b += 1
            l8_red_idx = feature_names.index("L8_RED") + 1 if "L8_RED" in feature_names else None
            l8_nir_idx = feature_names.index("L8_NIR") + 1 if "L8_NIR" in feature_names else None

            # Spectral indices from imported stacks.
            if s2_red_idx is not None and s2_nir_idx is not None:
                feature_names.append(_append_ndvi(dst, b, s2_red_idx, s2_nir_idx, "S2_NDVI"))
                b += 1
            else:
                for _, win in dem_src.block_windows(1):
                    zero = np.full((int(win.height), int(win.width)), NODATA_OUT, dtype=np.float32)
                    dst.write(zero, b, window=win)
                dst.set_band_description(b, "S2_NDVI")
                feature_names.append("S2_NDVI")
                b += 1

            if l8_red_idx is not None and l8_nir_idx is not None:
                feature_names.append(_append_ndvi(dst, b, l8_red_idx, l8_nir_idx, "L8_NDVI"))
                b += 1
            else:
                for _, win in dem_src.block_windows(1):
                    zero = np.full((int(win.height), int(win.width)), NODATA_OUT, dtype=np.float32)
                    dst.write(zero, b, window=win)
                dst.set_band_description(b, "L8_NDVI")
                feature_names.append("L8_NDVI")
                b += 1

            if include_dem_lineament_priors:
                b, prior_names = _append_dem_lineament_priors(
                    dst=dst,
                    start_idx=b,
                    dem_src=dem_src,
                    prior_cfg=prior_cfg,
                )
                feature_names.extend(prior_names)

            if include_external_lithology:
                if lithology_geojson is None:
                    raise ValueError("include_external_lithology=true but paths.external_lithology_geojson is missing.")
                b, litho_names = _append_polygon_one_hot_features(
                    dst=dst,
                    start_idx=b,
                    vector_path=lithology_geojson,
                    field=str(prior_cfg.get("lithology_field", "xx_Description")),
                    classes=lithology_classes,
                    prefix="LITHO",
                    ref_crs=dem_src.crs,
                    ref_transform=dem_src.transform,
                    width=dem_src.width,
                    height=dem_src.height,
                )
                feature_names.extend(litho_names)
                print(f"[fusion] external lithology priors: {len(litho_names)} classes")

            if include_external_geology_age:
                if external_geology_geojson is None:
                    raise ValueError(
                        "include_external_geology_age=true but paths.external_geology_geojson is missing."
                    )
                b, geology_names = _append_polygon_one_hot_features(
                    dst=dst,
                    start_idx=b,
                    vector_path=external_geology_geojson,
                    field=str(prior_cfg.get("geology_age_field", "Glg")),
                    classes=geology_age_classes,
                    prefix="GEOAGE",
                    ref_crs=dem_src.crs,
                    ref_transform=dem_src.transform,
                    width=dem_src.width,
                    height=dem_src.height,
                )
                feature_names.extend(geology_names)
                print(f"[fusion] external geology-age priors: {len(geology_names)} classes")

            if include_external_fault_priors:
                if faults_geojson is None:
                    raise ValueError("include_external_fault_priors=true but paths.external_faults_geojson is missing.")
                b, fault_names = _append_external_fault_priors(
                    dst=dst,
                    start_idx=b,
                    faults_geojson=faults_geojson,
                    dem_src=dem_src,
                    prior_cfg=prior_cfg,
                )
                feature_names.extend(fault_names)

            if include_geology_mask:
                if geology_geojson is None:
                    raise ValueError("include_geology_mask=true but paths.geology_geojson is missing.")
                feature_names.append(
                    _append_geology_mask(
                        dst=dst,
                        band_idx=b,
                        geology_geojson=geology_geojson,
                        ref_crs=dem_src.crs,
                        ref_transform=dem_src.transform,
                        width=dem_src.width,
                        height=dem_src.height,
                    )
                )
                b += 1

            if include_geochem_features:
                if points_csv is None:
                    raise ValueError("include_geochem_features=true but paths.geochem_points_csv is missing.")
                b, geochem_names = _append_geochem_features(
                    dst=dst,
                    start_idx=b,
                    points_csv=points_csv,
                    transform=dem_src.transform,
                    width=dem_src.width,
                    height=dem_src.height,
                    dem_src=dem_src,
                )
                feature_names.extend(geochem_names)

    _save_feature_names(Path(fused_names_json), feature_names)
    print(f"[fusion] fused stack: {fused_stack_tif}")
    print(f"[fusion] feature count: {len(feature_names)}")
    return {
        "feature_stack_tif": str(fused_stack_tif),
        "feature_names_json": str(fused_names_json),
        "feature_count": len(feature_names),
    }


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config)
    build_multisource_feature_stack(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
