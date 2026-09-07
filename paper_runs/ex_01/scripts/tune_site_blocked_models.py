#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for import_path in [SRC_ROOT, SCRIPT_ROOT]:
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from xaigis.config import load_config
from xaigis.modeling import _build_models, _calc_metrics, _predict_positive_probability
from xaigis.utils import ensure_dir, save_json

from validate_site_blocked import (
    SiteSamples,
    _load_feature_names,
    _mean_std,
    _read_site_samples,
    _top_fraction_metrics,
)


METRIC_KEYS = [
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune classical models with strict leave-one-site-out validation. "
            "The script samples sites once, then evaluates each configured trial "
            "with the same held-out folds."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "paper_runs/ex_01/configs/ex_01_site_blocked_tuning_external_priors.json",
        help="Tuning config containing the base run config, output directory, and model grids.",
    )
    return parser.parse_args()


def _load_tuning_config(path: Path) -> tuple[dict[str, Any], Path, Path]:
    cfg_path = path.expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        tuning_cfg = json.load(f)

    base_config = Path(tuning_cfg["base_config"]).expanduser()
    if not base_config.is_absolute():
        base_config = (cfg_path.parent / base_config).resolve()

    output_dir = Path(tuning_cfg["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = (cfg_path.parent / output_dir).resolve()

    return tuning_cfg, base_config, output_dir


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _iter_trials(tuning_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for model_name, model_cfg in tuning_cfg.get("models", {}).items():
        if not model_cfg.get("enabled", True):
            continue
        grid = model_cfg.get("grid", {})
        if not grid:
            trials.append({"model": model_name, "params": {}})
            continue
        keys = list(grid)
        values = [_as_list(grid[key]) for key in keys]
        for combo in itertools.product(*values):
            params = {key: value for key, value in zip(keys, combo)}
            trials.append({"model": model_name, "params": params})
    return trials


def _set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor: dict[str, Any] = cfg
    for part in parts[:-1]:
        next_value = cursor.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise TypeError(f"Cannot set nested parameter '{dotted_key}'; '{part}' is not a mapping.")
        cursor = next_value
    cursor[parts[-1]] = value


def _trial_training_config(base_training: dict[str, Any], model_name: str, params: dict[str, Any]) -> dict[str, Any]:
    training_cfg = copy.deepcopy(base_training)
    all_models = set(training_cfg.get("models", {})) | {model_name}
    training_cfg["models"] = {name: name == model_name for name in sorted(all_models)}
    for key, value in params.items():
        if key.startswith("training."):
            _set_nested(training_cfg, key.removeprefix("training."), value)
        else:
            _set_nested(training_cfg, key, value)
    return training_cfg


def _read_samples(
    base_cfg: dict[str, Any],
    tuning_cfg: dict[str, Any],
) -> tuple[list[str], list[str], dict[str, SiteSamples], list[dict[str, Any]]]:
    paths = base_cfg["paths"]
    site_cfg = tuning_cfg.get("site_blocked", {})
    stack_tif = Path(paths["feature_stack_tif"])
    label_tif = Path(paths["label_tif"])
    label_geojson = Path(paths["geology_geojson"])

    if not stack_tif.exists():
        raise FileNotFoundError(f"Feature stack not found: {stack_tif}")
    if not label_tif.exists():
        raise FileNotFoundError(f"Label raster not found: {label_tif}")
    if not label_geojson.exists():
        raise FileNotFoundError(f"Label polygon GeoJSON not found: {label_geojson}")

    site_column = str(site_cfg.get("site_column", "site_name"))
    seed = int(site_cfg.get("random_seed", base_cfg["training"].get("random_seed", 42)))
    rng = np.random.default_rng(seed)

    label_gdf = gpd.read_file(label_geojson)
    if label_gdf.empty:
        raise ValueError(f"No label polygons found: {label_geojson}")
    if site_column not in label_gdf.columns:
        raise ValueError(f"Missing site column '{site_column}' in {label_geojson}")

    with rio.open(stack_tif) as src_x, rio.open(label_tif) as src_y:
        if src_x.width != src_y.width or src_x.height != src_y.height:
            raise ValueError("Feature stack and label raster dimensions do not match.")
        if label_gdf.crs is None:
            label_gdf = label_gdf.set_crs(src_x.crs)
        elif label_gdf.crs != src_x.crs:
            label_gdf = label_gdf.to_crs(src_x.crs)

        feature_names = _load_feature_names(Path(paths["feature_names_json"]), src_x.count)
        site_names = sorted(str(site) for site in label_gdf[site_column].dropna().unique())
        samples: dict[str, SiteSamples] = {}
        site_rows: list[dict[str, Any]] = []

        print(f"[tune] reading samples from {len(site_names)} sites", flush=True)
        for site_name in site_names:
            site_gdf = label_gdf[label_gdf[site_column].astype(str) == site_name]
            sample = _read_site_samples(
                src_x=src_x,
                src_y=src_y,
                site_name=site_name,
                site_gdf=site_gdf,
                all_label_gdf=label_gdf,
                site_column=site_column,
                block_buffer_km=float(site_cfg.get("block_buffer_km", 20.0)),
                max_positive_per_site=int(site_cfg.get("max_positive_per_site", 20000)),
                negative_ratio=float(site_cfg.get("negative_ratio", 2.0)),
                max_negative_per_site=int(site_cfg.get("max_negative_per_site", 40000)),
                exclude_nodata=bool(site_cfg.get("exclude_nodata", True)),
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
                "[tune] "
                f"{site_name}: pos={sample.positives_sampled:,}/{sample.positives_available:,}, "
                f"bg={sample.negatives_sampled:,}/{sample.negatives_available:,}, "
                f"pos_nodata={sample.positive_pixels_with_nodata:,}",
                flush=True,
            )

    return feature_names, site_names, samples, site_rows


def _run_trial(
    trial_id: str,
    model_name: str,
    params: dict[str, Any],
    base_training: dict[str, Any],
    site_names: list[str],
    samples: dict[str, SiteSamples],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    training_cfg = _trial_training_config(base_training, model_name, params)
    threshold = float(training_cfg.get("threshold", 0.8))
    fold_rows: list[dict[str, Any]] = []
    pooled_y: list[np.ndarray] = []
    pooled_prob: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []

    print(f"[tune] trial={trial_id} model={model_name} params={json.dumps(params, sort_keys=True)}", flush=True)
    for heldout_site in site_names:
        train_sites = [site for site in site_names if site != heldout_site]
        x_train = np.vstack([samples[site].x for site in train_sites]).astype(np.float32)
        y_train = np.concatenate([samples[site].y for site in train_sites]).astype(np.uint8)
        x_test = samples[heldout_site].x.astype(np.float32)
        y_test = samples[heldout_site].y.astype(np.uint8)

        models = _build_models(training_cfg, y_train)
        if model_name not in models:
            raise RuntimeError(f"Configured model '{model_name}' was not built. Built models: {sorted(models)}")
        model = models[model_name]
        model.fit(x_train, y_train)
        prob = _predict_positive_probability(model, x_test)
        pred = (prob >= threshold).astype(np.uint8)
        metrics = _calc_metrics(y_test, prob, pred)
        metrics.update(_top_fraction_metrics(y_test, prob))

        fold_rows.append(
            {
                "trial_id": trial_id,
                "model": model_name,
                "heldout_site": heldout_site,
                "threshold": threshold,
                "train_samples": int(y_train.size),
                "train_positives": int((y_train == 1).sum()),
                "train_negatives": int((y_train == 0).sum()),
                "test_samples": int(y_test.size),
                "test_positives": int((y_test == 1).sum()),
                "test_negatives": int((y_test == 0).sum()),
                **metrics,
            }
        )
        pooled_y.append(y_test)
        pooled_prob.append(prob)
        pooled_pred.append(pred)
        print(
            f"[tune] {trial_id} heldout={heldout_site} "
            f"ROC-AUC={metrics['roc_auc']:.4f} PR-AUC={metrics['pr_auc']:.4f} "
            f"F1={metrics['f1']:.4f}",
            flush=True,
        )

    y_all = np.concatenate(pooled_y)
    prob_all = np.concatenate(pooled_prob)
    pred_all = np.concatenate(pooled_pred)
    pooled_metrics = _calc_metrics(y_all, prob_all, pred_all)
    pooled_metrics.update(_top_fraction_metrics(y_all, prob_all))
    macro_summary = _mean_std(fold_rows, METRIC_KEYS)

    trial_row: dict[str, Any] = {
        "trial_id": trial_id,
        "model": model_name,
        "params_json": json.dumps(params, sort_keys=True),
        "threshold": threshold,
        **pooled_metrics,
        **{
            f"macro_{metric}_{stat}": value
            for metric, stats in macro_summary.items()
            for stat, value in stats.items()
        },
    }
    print(
        f"[tune] done trial={trial_id} pooled ROC-AUC={trial_row['roc_auc']:.4f} "
        f"PR-AUC={trial_row['pr_auc']:.4f} top5={trial_row['precision_at_top5pct']:.4f}",
        flush=True,
    )
    return trial_row, fold_rows


def _write_markdown(
    path: Path,
    tuning_cfg: dict[str, Any],
    base_config: Path,
    feature_names: list[str],
    site_rows: list[dict[str, Any]],
    trial_rows: list[dict[str, Any]],
    best_trial: dict[str, Any],
) -> None:
    selection_metric = str(tuning_cfg.get("selection_metric", "pr_auc"))
    lines = [
        "# Site-blocked hyperparameter tuning",
        "",
        f"- Base config: `{base_config}`",
        f"- Selection metric: `{selection_metric}`",
        f"- Feature count: {len(feature_names)}",
        f"- Features: {', '.join(feature_names)}",
        f"- Best trial: `{best_trial['trial_id']}`",
        f"- Best model: `{best_trial['model']}`",
        f"- Best pooled ROC-AUC: {best_trial['roc_auc']:.4f}",
        f"- Best pooled PR-AUC: {best_trial['pr_auc']:.4f}",
        f"- Best top-5% precision: {best_trial['precision_at_top5pct']:.4f}",
        "",
        "## Trial Ranking",
        "",
        "| Rank | Trial | Model | Pooled ROC-AUC | Pooled PR-AUC | F1 | Top 5% precision | Params |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    ranked = sorted(trial_rows, key=lambda row: float(row.get(selection_metric, 0.0)), reverse=True)
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | {row['trial_id']} | {row['model']} | {row['roc_auc']:.4f} | "
            f"{row['pr_auc']:.4f} | {row['f1']:.4f} | {row['precision_at_top5pct']:.4f} | "
            f"`{row['params_json']}` |"
        )

    lines.extend(
        [
            "",
            "## Site Samples",
            "",
            "| Site | Positives sampled | Background sampled | Positives available | Background available |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in site_rows:
        lines.append(
            f"| {row['site_name']} | {row['positives_sampled']} | {row['negatives_sampled']} | "
            f"{row['positives_available']} | {row['negatives_available']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Note",
            "",
            "Trials are ranked by strict leave-one-site-out pooled PR-AUC. Fixed-threshold precision, recall, and F1 remain threshold diagnostics; final threshold selection should be tuned inside training folds before probability or mask maps are finalized.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_tuning(config_path: Path) -> dict[str, Any]:
    tuning_cfg, base_config_path, output_dir = _load_tuning_config(config_path)
    base_cfg = load_config(base_config_path)
    output_dir = ensure_dir(output_dir)
    feature_names, site_names, samples, site_rows = _read_samples(base_cfg, tuning_cfg)
    trials = _iter_trials(tuning_cfg)
    if not trials:
        raise ValueError("No enabled tuning trials found.")

    selection_metric = str(tuning_cfg.get("selection_metric", "pr_auc"))
    trial_rows: list[dict[str, Any]] = []
    all_fold_rows: list[dict[str, Any]] = []

    print(f"[tune] running {len(trials)} trials; selection_metric={selection_metric}", flush=True)
    for idx, trial in enumerate(trials, start=1):
        model_name = str(trial["model"])
        trial_id = f"{model_name}_{idx:03d}"
        trial_row, fold_rows = _run_trial(
            trial_id=trial_id,
            model_name=model_name,
            params=trial["params"],
            base_training=base_cfg["training"],
            site_names=site_names,
            samples=samples,
        )
        trial_rows.append(trial_row)
        all_fold_rows.extend(fold_rows)

    best_trial = sorted(trial_rows, key=lambda row: float(row.get(selection_metric, 0.0)), reverse=True)[0]
    result: dict[str, Any] = {
        "tuning": {
            "config": str(Path(config_path).expanduser().resolve()),
            "base_config": str(base_config_path),
            "selection_metric": selection_metric,
            "site_blocked": tuning_cfg.get("site_blocked", {}),
            "trial_count": len(trial_rows),
        },
        "inputs": {
            "feature_stack_tif": str(base_cfg["paths"]["feature_stack_tif"]),
            "label_tif": str(base_cfg["paths"]["label_tif"]),
            "label_geojson": str(base_cfg["paths"]["geology_geojson"]),
            "feature_names": feature_names,
        },
        "site_samples": site_rows,
        "best_trial": best_trial,
        "trials": trial_rows,
        "folds": all_fold_rows,
    }

    save_json(output_dir / "site_blocked_tuning.json", result)
    pd.DataFrame(trial_rows).sort_values(selection_metric, ascending=False).to_csv(
        output_dir / "site_blocked_tuning_trials.csv",
        index=False,
    )
    pd.DataFrame(all_fold_rows).to_csv(output_dir / "site_blocked_tuning_folds.csv", index=False)
    pd.DataFrame(site_rows).to_csv(output_dir / "site_blocked_tuning_site_samples.csv", index=False)
    _write_markdown(
        output_dir / "site_blocked_tuning.md",
        tuning_cfg=tuning_cfg,
        base_config=base_config_path,
        feature_names=feature_names,
        site_rows=site_rows,
        trial_rows=trial_rows,
        best_trial=best_trial,
    )
    print(f"[tune] saved: {output_dir / 'site_blocked_tuning.json'}", flush=True)
    print(
        f"[tune] best trial={best_trial['trial_id']} model={best_trial['model']} "
        f"ROC-AUC={best_trial['roc_auc']:.4f} PR-AUC={best_trial['pr_auc']:.4f} "
        f"top5={best_trial['precision_at_top5pct']:.4f}",
        flush=True,
    )
    return result


def main() -> int:
    args = _parse_args()
    run_tuning(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
