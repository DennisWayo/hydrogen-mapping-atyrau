#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from dataclasses import dataclass
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
from xaigis.modeling import _build_models, _calc_metrics, _predict_positive_probability
from xaigis.utils import ensure_dir, load_json, save_json


@dataclass
class SiteSamples:
    site_name: str
    x: np.ndarray
    y: np.ndarray
    positives_available: int
    negatives_available: int
    positive_pixels_with_nodata: int
    background_pixels_with_nodata: int
    positives_sampled: int
    negatives_sampled: int
    block_bounds: tuple[float, float, float, float]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run leave-one-site-out validation for ex_01. Each fold trains on all "
            "other labeled AOIs and tests on the held-out AOI plus local background."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "paper_runs/ex_01/configs/ex_01_fusion_noleak_run.json",
        help="Run config containing feature stack, label raster, and label polygons.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <artifacts_dir>/site_blocked_validation.",
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
        help="Metric buffer around each site polygon used to sample local background.",
    )
    parser.add_argument(
        "--max-positive-per-site",
        type=int,
        default=20_000,
        help="Maximum positive pixels sampled per site. Use 0 for no cap.",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=2.0,
        help="Sample up to this many background pixels per sampled positive pixel.",
    )
    parser.add_argument(
        "--max-negative-per-site",
        type=int,
        default=40_000,
        help="Hard cap on background pixels sampled per site. Use 0 for no cap.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Validation sampling seed. Defaults to training.random_seed from config.",
    )
    parser.add_argument(
        "--exclude-nodata",
        action="store_true",
        default=True,
        help=(
            "Exclude pixels where any feature band equals the raster NoData value. "
            "This is the default."
        ),
    )
    parser.add_argument(
        "--include-nodata",
        dest="exclude_nodata",
        action="store_false",
        help="Reproduce legacy behavior by allowing finite NoData codes such as -9999 as model inputs.",
    )
    return parser.parse_args()


def _load_feature_names(path: Path, n_features: int) -> list[str]:
    if path.exists():
        data = load_json(path)
        names = data.get("feature_names", [])
        if len(names) == n_features:
            return [str(name) for name in names]
    return [f"f{i:02d}" for i in range(n_features)]


def _window_for_bounds(src: rio.io.DatasetReader, bounds: tuple[float, float, float, float]) -> Window:
    win = from_bounds(*bounds, transform=src.transform)
    col0 = max(0, int(math.floor(win.col_off)))
    row0 = max(0, int(math.floor(win.row_off)))
    col1 = min(src.width, int(math.ceil(win.col_off + win.width)))
    row1 = min(src.height, int(math.ceil(win.row_off + win.height)))
    if col1 <= col0 or row1 <= row0:
        raise ValueError(f"Site block does not overlap raster bounds: {bounds}")
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


def _sample_mask(mask: np.ndarray, max_count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    rr, cc = np.where(mask)
    n = rr.size
    if max_count > 0 and n > max_count:
        keep = rng.choice(n, size=max_count, replace=False)
        rr = rr[keep]
        cc = cc[keep]
    return rr, cc


def _read_site_samples(
    src_x: rio.io.DatasetReader,
    src_y: rio.io.DatasetReader,
    site_name: str,
    site_gdf: gpd.GeoDataFrame,
    all_label_gdf: gpd.GeoDataFrame,
    site_column: str,
    block_buffer_km: float,
    max_positive_per_site: int,
    negative_ratio: float,
    max_negative_per_site: int,
    exclude_nodata: bool,
    rng: np.random.Generator,
) -> SiteSamples:
    site_geom = _union_geometry(site_gdf)
    block_geom = _buffer_site_geometry(site_gdf, src_x.crs, block_buffer_km)
    win = _window_for_bounds(src_x, block_geom.bounds)
    h, w = int(win.height), int(win.width)
    transform = src_x.window_transform(win)

    x_patch = src_x.read(window=win).astype(np.float32)
    y_patch = src_y.read(1, window=win).astype(np.uint8)
    valid = np.isfinite(x_patch).all(axis=0)
    nodata_any = np.zeros((h, w), dtype=bool)
    if src_x.nodata is not None:
        nodata_any = (x_patch == src_x.nodata).any(axis=0)
        if exclude_nodata:
            valid &= ~nodata_any

    site_mask = features.geometry_mask(
        [site_geom.__geo_interface__],
        out_shape=(h, w),
        transform=transform,
        invert=True,
    )
    block_mask = features.geometry_mask(
        [block_geom.__geo_interface__],
        out_shape=(h, w),
        transform=transform,
        invert=True,
    )
    other_sites = all_label_gdf[all_label_gdf[site_column].astype(str) != site_name]
    if other_sites.empty:
        other_mask = np.zeros((h, w), dtype=bool)
    else:
        other_mask = features.geometry_mask(
            [geom.__geo_interface__ for geom in other_sites.geometry if geom is not None and not geom.is_empty],
            out_shape=(h, w),
            transform=transform,
            invert=True,
        )

    positive_raw = site_mask & (y_patch == 1)
    background_raw = block_mask & (y_patch == 0) & ~site_mask & ~other_mask
    positive_mask = valid & positive_raw
    background_mask = valid & background_raw

    pos_available = int(positive_mask.sum())
    neg_available = int(background_mask.sum())
    pos_with_nodata = int((positive_raw & nodata_any).sum())
    bg_with_nodata = int((background_raw & nodata_any).sum())
    if pos_available == 0:
        mode = " after strict NoData exclusion" if exclude_nodata else ""
        raise ValueError(
            f"No positive pixels sampled for held-out site: {site_name}{mode}. "
            f"positive_pixels_raw={int(positive_raw.sum())}, "
            f"positive_pixels_with_nodata={pos_with_nodata}"
        )
    if neg_available == 0:
        mode = " after strict NoData exclusion" if exclude_nodata else ""
        raise ValueError(
            f"No background pixels sampled for held-out site: {site_name}{mode}. "
            f"background_pixels_raw={int(background_raw.sum())}, "
            f"background_pixels_with_nodata={bg_with_nodata}"
        )

    pos_rr, pos_cc = _sample_mask(positive_mask, max_positive_per_site, rng)
    neg_target = int(math.ceil(pos_rr.size * max(negative_ratio, 0.0)))
    if max_negative_per_site > 0:
        neg_target = min(neg_target, max_negative_per_site)
    neg_rr, neg_cc = _sample_mask(background_mask, neg_target, rng)

    rr = np.concatenate([pos_rr, neg_rr])
    cc = np.concatenate([pos_cc, neg_cc])
    y = np.concatenate(
        [
            np.ones(pos_rr.size, dtype=np.uint8),
            np.zeros(neg_rr.size, dtype=np.uint8),
        ]
    )
    x = x_patch[:, rr, cc].T.astype(np.float32)

    order = rng.permutation(y.size)
    return SiteSamples(
        site_name=site_name,
        x=x[order],
        y=y[order],
        positives_available=pos_available,
        negatives_available=neg_available,
        positive_pixels_with_nodata=pos_with_nodata,
        background_pixels_with_nodata=bg_with_nodata,
        positives_sampled=int(pos_rr.size),
        negatives_sampled=int(neg_rr.size),
        block_bounds=tuple(float(v) for v in block_geom.bounds),
    )


def _top_fraction_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    positives = int((y_true == 1).sum())
    base_rate = float(positives / max(y_true.size, 1))
    order = np.argsort(prob)[::-1]
    for label, fraction in [("top1pct", 0.01), ("top5pct", 0.05), ("top10pct", 0.10)]:
        n = max(1, int(math.ceil(y_true.size * fraction)))
        selected = y_true[order[:n]]
        precision = float(selected.mean()) if selected.size else 0.0
        recall = float(selected.sum() / max(positives, 1))
        lift = float(precision / base_rate) if base_rate > 0 else 0.0
        out[f"precision_at_{label}"] = precision
        out[f"recall_at_{label}"] = recall
        out[f"lift_at_{label}"] = lift
    return out


def _mean_std(records: list[dict[str, Any]], keys: list[str]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        vals = np.array([float(row[key]) for row in records if row.get(key) is not None], dtype=np.float64)
        if vals.size:
            summary[key] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            }
    return summary


def _build_configured_models(training_cfg: dict[str, Any], y_train: np.ndarray) -> dict[str, Any]:
    params = inspect.signature(_build_models).parameters
    if "n_jobs" in params:
        return _build_models(training_cfg, y_train, -1)
    return _build_models(training_cfg, y_train)


def _write_markdown_report(
    path: Path,
    run_name: str,
    feature_names: list[str],
    site_rows: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    exclude_nodata: bool,
) -> None:
    model_summaries = aggregate.get("models", {})
    primary_metrics = aggregate.get("pooled_metrics", {})
    lines = [
        f"# Site-blocked validation: {run_name}",
        "",
        "This validation holds out one labeled AOI/site at a time. Training uses the remaining sites; testing uses the held-out site's positive pixels and local background sampled from a buffered site block.",
        "",
        f"- Feature count: {len(feature_names)}",
        f"- Features: {', '.join(feature_names)}",
        f"- Pooled ROC-AUC: {primary_metrics.get('roc_auc', 0.0):.4f}",
        f"- Pooled PR-AUC: {primary_metrics.get('pr_auc', 0.0):.4f}",
        f"- Pooled F1: {primary_metrics.get('f1', 0.0):.4f}",
        f"- Exclude NoData: {exclude_nodata}",
        "",
    ]
    if model_summaries:
        lines.extend(
            [
                "## Model Summary",
                "",
                "| Model | Pooled ROC-AUC | Pooled PR-AUC | Precision | Recall | F1 | Top 5% precision |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for model_name, summary in sorted(
            model_summaries.items(),
            key=lambda item: float(item[1]["pooled_metrics"].get("pr_auc", 0.0)),
            reverse=True,
        ):
            metrics = summary["pooled_metrics"]
            lines.append(
                f"| {model_name} | {metrics['roc_auc']:.4f} | {metrics['pr_auc']:.4f} | "
                f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | "
                f"{metrics.get('precision_at_top5pct', 0.0):.4f} |"
            )
        lines.append("")

    lines.extend(
        [
        "## Site Samples",
        "",
        "| Site | Positives sampled | Background sampled | Positives available | Background available | Positive pixels with NoData | Background pixels with NoData |",
        "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in site_rows:
        lines.append(
            f"| {row['site_name']} | {row['positives_sampled']} | {row['negatives_sampled']} | "
            f"{row['positives_available']} | {row['negatives_available']} | "
            f"{row['positive_pixels_with_nodata']} | {row['background_pixels_with_nodata']} |"
        )

    lines.extend(
        [
            "",
            "## Leave-one-site-out folds",
            "",
            "| Model | Held-out site | ROC-AUC | PR-AUC | Precision | Recall | F1 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in fold_rows:
        lines.append(
            f"| {row['model']} | {row['heldout_site']} | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Note",
            "",
            "These scores test spatial transfer across the six proxy AOIs. They are stronger than random pixel splits, but they still depend on placeholder buffered labels rather than independent hydrogen measurements.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_site_blocked_validation(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    paths = cfg["paths"]
    training_cfg = cfg["training"]
    seed = int(args.random_seed if args.random_seed is not None else training_cfg.get("random_seed", 42))
    rng = np.random.default_rng(seed)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path(paths["artifacts_dir"]) / "site_blocked_validation"
    out_dir = ensure_dir(out_dir)

    stack_tif = Path(paths["feature_stack_tif"])
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

    with rio.open(stack_tif) as src_x, rio.open(label_tif) as src_y:
        if src_x.width != src_y.width or src_x.height != src_y.height:
            raise ValueError("Feature stack and label raster dimensions do not match.")
        if label_gdf.crs is None:
            label_gdf = label_gdf.set_crs(src_x.crs)
        elif label_gdf.crs != src_x.crs:
            label_gdf = label_gdf.to_crs(src_x.crs)

        feature_names = _load_feature_names(Path(paths["feature_names_json"]), src_x.count)
        site_names = sorted(str(site) for site in label_gdf[args.site_column].dropna().unique())
        if len(site_names) < 3:
            raise ValueError("At least three labeled sites are recommended for site-blocked validation.")

        samples: dict[str, SiteSamples] = {}
        site_rows: list[dict[str, Any]] = []
        print(f"[site-val] reading samples from {len(site_names)} sites")
        for site_name in site_names:
            site_gdf = label_gdf[label_gdf[args.site_column].astype(str) == site_name]
            sample = _read_site_samples(
                src_x=src_x,
                src_y=src_y,
                site_name=site_name,
                site_gdf=site_gdf,
                all_label_gdf=label_gdf,
                site_column=args.site_column,
                block_buffer_km=float(args.block_buffer_km),
                max_positive_per_site=int(args.max_positive_per_site),
                negative_ratio=float(args.negative_ratio),
                max_negative_per_site=int(args.max_negative_per_site),
                exclude_nodata=bool(args.exclude_nodata),
                rng=rng,
            )
            samples[site_name] = sample
            site_rows.append(
                {
                    "site_name": sample.site_name,
                    "positives_available": sample.positives_available,
                    "negatives_available": sample.negatives_available,
                    "positive_pixels_with_nodata": sample.positive_pixels_with_nodata,
                    "background_pixels_with_nodata": sample.background_pixels_with_nodata,
                    "positives_sampled": sample.positives_sampled,
                    "negatives_sampled": sample.negatives_sampled,
                    "samples": int(sample.y.size),
                    "block_minx": sample.block_bounds[0],
                    "block_miny": sample.block_bounds[1],
                    "block_maxx": sample.block_bounds[2],
                    "block_maxy": sample.block_bounds[3],
                }
            )
            print(
                "[site-val] "
                f"{site_name}: pos={sample.positives_sampled:,}/{sample.positives_available:,}, "
                f"bg={sample.negatives_sampled:,}/{sample.negatives_available:,}, "
                f"pos_nodata={sample.positive_pixels_with_nodata:,}"
            )

    threshold = float(training_cfg.get("threshold", 0.8))
    fold_rows: list[dict[str, Any]] = []
    pooled_by_model: dict[str, dict[str, list[np.ndarray]]] = {}

    for heldout_site in site_names:
        train_sites = [site for site in site_names if site != heldout_site]
        x_train = np.vstack([samples[site].x for site in train_sites]).astype(np.float32)
        y_train = np.concatenate([samples[site].y for site in train_sites]).astype(np.uint8)
        x_test = samples[heldout_site].x.astype(np.float32)
        y_test = samples[heldout_site].y.astype(np.uint8)

        models = _build_configured_models(training_cfg, y_train)
        if len(models) != 1:
            print(f"[site-val] configured models: {', '.join(models)}")
        for model_name, model in models.items():
            print(
                f"[site-val] fold heldout={heldout_site}, model={model_name}, "
                f"train={y_train.size:,}, test={y_test.size:,}"
            )
            model.fit(x_train, y_train)
            prob = _predict_positive_probability(model, x_test)
            pred = (prob >= threshold).astype(np.uint8)
            metrics = _calc_metrics(y_test, prob, pred)
            metrics.update(_top_fraction_metrics(y_test, prob))

            row: dict[str, Any] = {
                "heldout_site": heldout_site,
                "model": model_name,
                "train_samples": int(y_train.size),
                "train_positives": int((y_train == 1).sum()),
                "train_negatives": int((y_train == 0).sum()),
                "test_samples": int(y_test.size),
                "test_positives": int((y_test == 1).sum()),
                "test_negatives": int((y_test == 0).sum()),
                **metrics,
            }
            fold_rows.append(row)
            if model_name not in pooled_by_model:
                pooled_by_model[model_name] = {"y": [], "prob": [], "pred": []}
            pooled_by_model[model_name]["y"].append(y_test)
            pooled_by_model[model_name]["prob"].append(prob)
            pooled_by_model[model_name]["pred"].append(pred)
            print(
                f"[site-val] {heldout_site}: ROC-AUC={metrics['roc_auc']:.4f}, "
                f"PR-AUC={metrics['pr_auc']:.4f}, F1={metrics['f1']:.4f}"
            )

    metric_keys = [
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "precision_at_top1pct",
        "recall_at_top1pct",
        "lift_at_top1pct",
        "precision_at_top5pct",
        "recall_at_top5pct",
        "lift_at_top5pct",
        "precision_at_top10pct",
        "recall_at_top10pct",
        "lift_at_top10pct",
    ]
    model_aggregates: dict[str, Any] = {}
    model_summary_rows: list[dict[str, Any]] = []
    for model_name, pooled in pooled_by_model.items():
        y_all = np.concatenate(pooled["y"])
        prob_all = np.concatenate(pooled["prob"])
        pred_all = np.concatenate(pooled["pred"])
        pooled_metrics_for_model = _calc_metrics(y_all, prob_all, pred_all)
        pooled_metrics_for_model.update(_top_fraction_metrics(y_all, prob_all))
        model_fold_rows = [row for row in fold_rows if row["model"] == model_name]
        fold_macro_summary = _mean_std(model_fold_rows, metric_keys)
        model_aggregates[model_name] = {
            "pooled_metrics": pooled_metrics_for_model,
            "fold_macro_summary": fold_macro_summary,
        }
        model_summary_rows.append(
            {
                "model": model_name,
                **pooled_metrics_for_model,
                **{
                    f"macro_{metric}_{stat}": value
                    for metric, stats in fold_macro_summary.items()
                    for stat, value in stats.items()
                },
            }
        )

    if not model_aggregates:
        raise RuntimeError("No site-blocked model results were produced.")

    primary_model = sorted(
        model_aggregates,
        key=lambda name: float(model_aggregates[name]["pooled_metrics"].get("pr_auc", 0.0)),
        reverse=True,
    )[0]
    pooled_metrics = model_aggregates[primary_model]["pooled_metrics"]
    aggregate = {
        "primary_model": primary_model,
        "pooled_metrics": pooled_metrics,
        "fold_macro_summary": model_aggregates[primary_model]["fold_macro_summary"],
        "models": model_aggregates,
    }

    run_name = Path(args.config).stem
    result: dict[str, Any] = {
        "validation": {
            "method": "leave_one_site_out_local_background",
            "site_column": args.site_column,
            "block_buffer_km": float(args.block_buffer_km),
            "max_positive_per_site": int(args.max_positive_per_site),
            "negative_ratio": float(args.negative_ratio),
            "max_negative_per_site": int(args.max_negative_per_site),
            "exclude_nodata": bool(args.exclude_nodata),
            "random_seed": seed,
            "threshold": threshold,
        },
        "inputs": {
            "config": str(Path(args.config).resolve()),
            "feature_stack_tif": str(stack_tif),
            "label_tif": str(label_tif),
            "label_geojson": str(label_geojson),
            "feature_names": feature_names,
        },
        "site_samples": site_rows,
        "folds": fold_rows,
        "aggregate": aggregate,
    }

    save_json(out_dir / "site_blocked_validation.json", result)
    pd.DataFrame(site_rows).to_csv(out_dir / "site_samples.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out_dir / "site_blocked_folds.csv", index=False)
    pd.DataFrame(model_summary_rows).sort_values("pr_auc", ascending=False).to_csv(
        out_dir / "site_blocked_model_summary.csv",
        index=False,
    )
    _write_markdown_report(
        out_dir / "site_blocked_validation.md",
        run_name=run_name,
        feature_names=feature_names,
        site_rows=site_rows,
        fold_rows=fold_rows,
        aggregate=aggregate,
        exclude_nodata=bool(args.exclude_nodata),
    )
    print(f"[site-val] saved: {out_dir / 'site_blocked_validation.json'}")
    for row in sorted(model_summary_rows, key=lambda item: float(item["pr_auc"]), reverse=True):
        print(
            f"[site-val] model={row['model']} pooled ROC-AUC={row['roc_auc']:.4f}, "
            f"PR-AUC={row['pr_auc']:.4f}, F1={row['f1']:.4f}"
        )
    print(
        f"[site-val] primary_model={primary_model} "
        f"pooled ROC-AUC={pooled_metrics['roc_auc']:.4f}, PR-AUC={pooled_metrics['pr_auc']:.4f}"
    )
    return result


def main() -> int:
    args = _parse_args()
    run_site_blocked_validation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
