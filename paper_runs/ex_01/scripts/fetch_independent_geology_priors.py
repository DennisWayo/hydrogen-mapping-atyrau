#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

# Local imports from repo.
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xaigis.config import load_config
from xaigis.utils import ensure_dir, save_json


GLIM_QUERY_URL = (
    "https://services8.arcgis.com/4KhTMTZ1x0f76DSg/ArcGIS/rest/services/"
    "GLiM_Niveau_I/FeatureServer/1/query"
)
GEM_FAULT_URLS = [
    "https://raw.githubusercontent.com/GEMScienceTools/gem-global-active-faults/master/geojson/gem_active_faults.geojson",
    "https://raw.githubusercontent.com/cossatot/gem-global-active-faults/master/geojson/gem_active_faults.geojson",
]
USGS_FSU_BASE = "https://pubs.usgs.gov/pubs/of/2001/ofr-01-104/fsucoal/views/shapes"
USGS_FSU_REQUIRED_FILES = ["fsu_geol.shp", "fsu_geol.dbf"]
USGS_FSU_OPTIONAL_FILES = ["fsu_geol.shx", "fsu_geol.prj"]
USGS_FSU_ASCII_BASE = "https://pubs.usgs.gov/pubs/of/2001/ofr-01-104/fsucoal/export/ascii"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch independent public geology, lithology, and active-fault priors for ex_01."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "paper_runs/ex_01/configs/ex_01_fusion_external_priors_run.json",
        help="Config containing region.aoi_geojson and optional prior paths.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "paper_runs/ex_01/priors",
        help="Directory for raw, processed, and provenance prior files.",
    )
    parser.add_argument(
        "--fault-buffer-km",
        type=float,
        default=75.0,
        help="Buffer around the AOI used when cropping global active faults.",
    )
    return parser.parse_args()


def _request_json(url: str, params: dict[str, Any] | None = None, retries: int = 3) -> dict[str, Any]:
    full_url = url if params is None else f"{url}?{urllib.parse.urlencode(params)}"
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(full_url, headers={"User-Agent": "XaiGis-ex01-priors/1.0"})
            with urllib.request.urlopen(req, timeout=90) as resp:
                payload = resp.read()
            return json.loads(payload.decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_exc = exc
            if attempt == retries:
                break
            time.sleep(2.0 * attempt)
    raise RuntimeError(f"Failed to request JSON from {url}: {last_exc}") from last_exc


def _download_file(url: str, out_path: Path, retries: int = 3) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "XaiGis-ex01-priors/1.0"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = resp.read()
            out_path.write_bytes(data)
            return out_path
        except (urllib.error.URLError, TimeoutError) as exc:
            last_exc = exc
            if attempt == retries:
                break
            time.sleep(2.0 * attempt)
    raise RuntimeError(f"Failed to download {url}: {last_exc}") from last_exc


def _write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    gdf.to_file(path, driver="GeoJSON")
    return path


def _union_geometry(gdf: gpd.GeoDataFrame) -> Any:
    if hasattr(gdf.geometry, "union_all"):
        return gdf.geometry.union_all()
    return gdf.geometry.unary_union


def _buffered_aoi(aoi: gpd.GeoDataFrame, buffer_km: float) -> gpd.GeoDataFrame:
    metric = aoi.to_crs(aoi.estimate_utm_crs())
    geom = _union_geometry(metric).buffer(buffer_km * 1000.0)
    return gpd.GeoDataFrame({"geometry": [geom]}, crs=metric.crs).to_crs("EPSG:4326")


def _fetch_glim_lithology(aoi: gpd.GeoDataFrame, out_path: Path) -> dict[str, Any]:
    aoi_3857 = aoi.to_crs("EPSG:3857")
    xmin, ymin, xmax, ymax = aoi_3857.total_bounds
    all_features: list[dict[str, Any]] = []
    offset = 0
    page_size = 2000

    while True:
        params = {
            "f": "geojson",
            "where": "1=1",
            "outFields": "OBJECTID,IDENTITY_,Litho,xx,yy,zz,xx_Description,yy_Description,zz_Description",
            "returnGeometry": "true",
            "geometry": f"{xmin},{ymin},{xmax},{ymax}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": 3857,
            "spatialRel": "esriSpatialRelIntersects",
            "outSR": 4326,
            "resultOffset": offset,
            "resultRecordCount": page_size,
        }
        data = _request_json(GLIM_QUERY_URL, params=params)
        features = data.get("features", [])
        all_features.extend(features)
        if len(features) < page_size:
            break
        offset += page_size

    collection = {"type": "FeatureCollection", "features": all_features}
    raw_path = out_path.with_name(out_path.stem + "_raw.geojson")
    raw_path.write_text(json.dumps(collection), encoding="utf-8")
    gdf = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:4326")
    if not gdf.empty:
        gdf = gpd.clip(gdf, aoi.to_crs(gdf.crs))
    _write_geojson(gdf, out_path)
    classes = sorted(str(v) for v in gdf.get("xx_Description", pd.Series(dtype=object)).dropna().unique())
    return {
        "source": "GLiM_Niveau_I ArcGIS FeatureServer",
        "url": GLIM_QUERY_URL,
        "raw_path": str(raw_path),
        "processed_path": str(out_path),
        "feature_count": int(len(gdf)),
        "classes": classes,
    }


def _fetch_gem_faults(aoi_buffer: gpd.GeoDataFrame, raw_path: Path, out_path: Path) -> dict[str, Any]:
    selected_url = None
    last_error = None
    for url in GEM_FAULT_URLS:
        try:
            _download_file(url, raw_path)
            selected_url = url
            break
        except Exception as exc:
            last_error = exc
    if selected_url is None:
        raise RuntimeError(f"Failed to download GEM active faults from all URLs: {last_error}") from last_error

    gdf = gpd.read_file(raw_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    clipped = gpd.clip(gdf, aoi_buffer.to_crs(gdf.crs))
    _write_geojson(clipped, out_path)
    return {
        "source": "GEM Global Active Faults Database",
        "url": selected_url,
        "raw_path": str(raw_path),
        "processed_path": str(out_path),
        "feature_count": int(len(clipped)),
        "fields": list(clipped.columns),
    }


def _fetch_usgs_fsu_geology(aoi: gpd.GeoDataFrame, raw_dir: Path, out_path: Path) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    missing_optional: list[str] = []
    try:
        for filename in USGS_FSU_REQUIRED_FILES:
            _download_file(f"{USGS_FSU_BASE}/{filename}", raw_dir / filename)
        for filename in USGS_FSU_OPTIONAL_FILES:
            try:
                _download_file(f"{USGS_FSU_BASE}/{filename}", raw_dir / filename)
            except Exception:
                missing_optional.append(filename)

        shp_path = raw_dir / "fsu_geol.shp"
        os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")
        gdf = gpd.read_file(shp_path)
        if gdf.crs is None:
            # Metadata states geographic coordinates on Pulkovo 1942.
            gdf = gdf.set_crs("EPSG:4284")
        route = "shapefile"
    except Exception as exc:
        missing_optional.append(f"shapefile_route_failed:{type(exc).__name__}")
        gen_path = _download_file(f"{USGS_FSU_ASCII_BASE}/fsu_geo.gen", raw_dir / "fsu_geo.gen")
        csv_path = _download_file(f"{USGS_FSU_ASCII_BASE}/fsu_geol.csv", raw_dir / "fsu_geol.csv")
        gdf = _read_usgs_ascii_geology(gen_path=gen_path, csv_path=csv_path)
        route = "ascii_gen_csv"

    gdf = gdf.to_crs("EPSG:4326")
    clipped = gpd.clip(gdf, aoi.to_crs(gdf.crs))
    _write_geojson(clipped, out_path)
    age_values = sorted(str(v) for v in clipped.get("Glg", pd.Series(dtype=object)).dropna().unique())
    return {
        "source": "USGS OFR 01-104 Surface Geology of the Former Soviet Union",
        "url": f"{USGS_FSU_BASE}/fsu_geol.shp",
        "raw_dir": str(raw_dir),
        "download_route": route,
        "missing_optional_files": missing_optional,
        "processed_path": str(out_path),
        "feature_count": int(len(clipped)),
        "geologic_age_codes": age_values,
    }


def _read_usgs_ascii_geology(gen_path: Path, csv_path: Path) -> gpd.GeoDataFrame:
    records: list[dict[str, Any]] = []
    current_id: str | None = None
    coords: list[tuple[float, float]] = []

    def flush() -> None:
        nonlocal current_id, coords
        if current_id is None or len(coords) < 3:
            current_id = None
            coords = []
            return
        ring = coords
        if ring[0] != ring[-1]:
            ring = ring + [ring[0]]
        geom = Polygon(ring)
        if not geom.is_valid:
            geom = geom.buffer(0)
        if not geom.is_empty:
            records.append({"_gen_id": str(current_id), "geometry": geom})
        current_id = None
        coords = []

    with gen_path.open("r", encoding="latin-1") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.upper() == "END":
                if current_id is None:
                    break
                flush()
                continue
            parts = line.replace(",", " ").split()
            if current_id is None:
                current_id = str(parts[0])
                if len(parts) >= 3:
                    try:
                        coords.append((float(parts[1]), float(parts[2])))
                    except ValueError:
                        pass
                continue
            if len(parts) >= 2:
                try:
                    coords.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
    flush()

    geom_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4284")
    attrs = pd.read_csv(csv_path)
    id_col = _select_column(attrs, ["Fsu_geol_id", "FSU_GEOL_ID", "fsu_geol_id", "ID", "id"])
    if id_col is None:
        id_col = str(attrs.columns[0])
    attrs["_gen_id"] = attrs[id_col].astype(str)
    out = geom_gdf.merge(attrs, on="_gen_id", how="left")
    age_col = _select_column(out, ["Glg", "GLG", "glg"])
    if age_col is not None and age_col != "Glg":
        out["Glg"] = out[age_col]
    return out


def _select_column(frame: pd.DataFrame | gpd.GeoDataFrame, candidates: list[str]) -> str | None:
    by_lower = {str(col).lower(): str(col) for col in frame.columns}
    for candidate in candidates:
        found = by_lower.get(candidate.lower())
        if found is not None:
            return found
    return None


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config)
    config_base = Path(cfg["__config_path__"]).parent
    aoi_path = Path(cfg["region"]["aoi_geojson"]).expanduser()
    if not aoi_path.is_absolute():
        aoi_path = (config_base / aoi_path).resolve()
    out_dir = ensure_dir(args.out_dir)
    raw_dir = ensure_dir(out_dir / "raw")
    processed_dir = ensure_dir(out_dir / "processed")

    aoi = gpd.read_file(aoi_path)
    if aoi.crs is None:
        aoi = aoi.set_crs("EPSG:4326")
    aoi = aoi.to_crs("EPSG:4326")
    aoi_buffer = _buffered_aoi(aoi, float(args.fault_buffer_km))

    print("[priors] fetching GLiM lithology")
    glim_info = _fetch_glim_lithology(
        aoi=aoi,
        out_path=processed_dir / "glim_lithology_ex01.geojson",
    )
    print(f"[priors] GLiM features: {glim_info['feature_count']}")

    print("[priors] fetching GEM active faults")
    fault_info = _fetch_gem_faults(
        aoi_buffer=aoi_buffer,
        raw_path=raw_dir / "gem_active_faults.geojson",
        out_path=processed_dir / "gem_active_faults_ex01.geojson",
    )
    print(f"[priors] GEM fault features in buffered AOI: {fault_info['feature_count']}")

    fetch_usgs = bool(cfg.get("prior_features", {}).get("include_external_geology_age", False))
    if fetch_usgs:
        print("[priors] fetching USGS FSU surface geology")
        try:
            usgs_info = _fetch_usgs_fsu_geology(
                aoi=aoi,
                raw_dir=raw_dir / "usgs_fsu_geol",
                out_path=processed_dir / "usgs_fsu_surface_geology_ex01.geojson",
            )
            print(f"[priors] USGS geology features: {usgs_info['feature_count']}")
        except Exception as exc:
            usgs_info = {
                "source": "USGS OFR 01-104 Surface Geology of the Former Soviet Union",
                "status": "unavailable",
                "error": str(exc),
                "attempted_shape_url": f"{USGS_FSU_BASE}/fsu_geol.shp",
                "attempted_ascii_url": f"{USGS_FSU_ASCII_BASE}/fsu_geo.gen",
                "feature_count": 0,
            }
            print(f"[priors] warning: USGS geology unavailable ({exc})")
    else:
        usgs_info = {
            "source": "USGS OFR 01-104 Surface Geology of the Former Soviet Union",
            "status": "skipped_by_config",
            "reason": "prior_features.include_external_geology_age is false",
            "feature_count": 0,
        }
        print("[priors] skipping USGS FSU surface geology by config")

    provenance = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "aoi_geojson": str(aoi_path),
        "fault_buffer_km": float(args.fault_buffer_km),
        "datasets": {
            "lithology_glim": glim_info,
            "faults_gem": fault_info,
            "surface_geology_usgs_fsu": usgs_info,
        },
        "scientific_guardrail": (
            "These priors are independent public map datasets. They do not use ex_01 target "
            "points, proxy labels, model predictions, or manually digitized site buffers."
        ),
    }
    save_json(out_dir / "provenance.json", provenance)

    lines = [
        "# ex_01 Independent Prior Provenance",
        "",
        f"- Generated: {provenance['generated']}",
        f"- AOI: `{aoi_path}`",
        f"- Fault crop buffer: `{float(args.fault_buffer_km):g} km`",
        "",
        "## Datasets",
        "",
        f"- GLiM lithology: {glim_info['feature_count']} clipped polygons from `{GLIM_QUERY_URL}`.",
        f"- GEM active faults: {fault_info['feature_count']} clipped line features from `{fault_info['url']}`.",
        f"- USGS FSU surface geology: {usgs_info['feature_count']} clipped polygons from `{USGS_FSU_BASE}`; status `{usgs_info.get('status', 'available')}`.",
        "",
        "## Guardrail",
        "",
        provenance["scientific_guardrail"],
        "",
        "The prior features should be interpreted as regional geologic context. They may disagree with DEM-derived lineaments because mapped geology, active faults, and surface geomorphology measure different parts of the system.",
    ]
    (out_dir / "provenance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[priors] provenance written: {out_dir / 'provenance.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
