from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio import features

from .utils import ensure_parent


def rasterize_labels(cfg: dict[str, Any]) -> dict[str, Any]:
    paths = cfg["paths"]
    stack_tif = paths["feature_stack_tif"]
    geojson = paths["geology_geojson"]
    label_tif = ensure_parent(paths["label_tif"])

    if not stack_tif.exists():
        raise FileNotFoundError(f"Feature stack not found: {stack_tif}")
    if not geojson.exists():
        raise FileNotFoundError(f"Geology/target polygons not found: {geojson}")

    with rio.open(stack_tif) as src:
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        shape = (src.height, src.width)

    gdf = gpd.read_file(geojson)
    if gdf.empty:
        raise ValueError(f"No geometries found in {geojson}")
    if gdf.crs is None:
        gdf = gdf.set_crs(crs)
    elif gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    geoms = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    label = features.rasterize(
        geoms,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    profile.update(
        count=1,
        dtype="uint8",
        nodata=0,
        compress="deflate",
        BIGTIFF="YES",
    )
    with rio.open(label_tif, "w", **profile) as dst:
        dst.write(label, 1)
        dst.set_band_description(1, "label")

    positives = int((label == 1).sum())
    total = int(label.size)
    print(f"[labels] saved: {label_tif}")
    print(f"[labels] positives: {positives:,} / {total:,} ({positives / max(total, 1):.4%})")
    return {"label_tif": str(label_tif), "positives": positives, "total_pixels": total}
