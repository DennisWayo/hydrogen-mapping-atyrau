#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for import_path in [SRC_ROOT, SCRIPT_ROOT]:
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from xaigis.config import load_config
from xaigis.modeling import _build_models, _calc_metrics, _predict_positive_probability
from xaigis.utils import ensure_dir, save_json

from tune_site_blocked_models import METRIC_KEYS, _read_samples, _trial_training_config
from validate_site_blocked import SiteSamples, _mean_std, _top_fraction_metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune classification thresholds with nested leave-one-site-out validation. "
            "Each outer fold chooses its threshold using only the remaining training sites."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "paper_runs/ex_01/configs/ex_01_site_blocked_thresholds_external_priors.json",
        help="Threshold tuning config.",
    )
    return parser.parse_args()


def _load_threshold_config(path: Path) -> tuple[dict[str, Any], Path, Path]:
    cfg_path = path.expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        threshold_cfg = json.load(f)

    base_config = Path(threshold_cfg["base_config"]).expanduser()
    if not base_config.is_absolute():
        base_config = (cfg_path.parent / base_config).resolve()

    output_dir = Path(threshold_cfg["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = (cfg_path.parent / output_dir).resolve()

    return threshold_cfg, base_config, output_dir


def _threshold_grid(cfg: dict[str, Any]) -> np.ndarray:
    grid_cfg = cfg.get("threshold_grid", {})
    if "values" in grid_cfg:
        values = np.array([float(v) for v in grid_cfg["values"]], dtype=np.float64)
    else:
        start = float(grid_cfg.get("min", 0.01))
        stop = float(grid_cfg.get("max", 0.99))
        step = float(grid_cfg.get("step", 0.01))
        values = np.arange(start, stop + step / 2.0, step, dtype=np.float64)
    values = values[(values >= 0.0) & (values <= 1.0)]
    if values.size == 0:
        raise ValueError("Threshold grid is empty.")
    return np.unique(np.round(values, 6))


def _score_thresholds(
    y_true: np.ndarray,
    prob: np.ndarray,
    thresholds: np.ndarray,
    selection_metric: str,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        pred = (prob >= threshold).astype(np.uint8)
        metrics = _calc_metrics(y_true, prob, pred)
        rows.append({"threshold": float(threshold), **metrics})

    if selection_metric not in rows[0]:
        raise ValueError(f"Unsupported threshold selection metric: {selection_metric}")

    # Prefer the higher threshold on ties so the tuned mask remains conservative.
    best = sorted(
        rows,
        key=lambda row: (
            float(row.get(selection_metric, 0.0)),
            float(row.get("precision", 0.0)),
            float(row["threshold"]),
        ),
        reverse=True,
    )[0]
    return float(best["threshold"]), best, rows


def _fit_predict(
    model_name: str,
    training_cfg: dict[str, Any],
    train_sites: list[str],
    test_site: str,
    samples: dict[str, SiteSamples],
) -> tuple[np.ndarray, np.ndarray]:
    x_train = np.vstack([samples[site].x for site in train_sites]).astype(np.float32)
    y_train = np.concatenate([samples[site].y for site in train_sites]).astype(np.uint8)
    x_test = samples[test_site].x.astype(np.float32)
    models = _build_models(training_cfg, y_train)
    if model_name not in models:
        raise RuntimeError(f"Configured model '{model_name}' was not built. Built models: {sorted(models)}")
    model = models[model_name]
    model.fit(x_train, y_train)
    return samples[test_site].y.astype(np.uint8), _predict_positive_probability(model, x_test)


def _select_outer_threshold(
    model_name: str,
    training_cfg: dict[str, Any],
    outer_train_sites: list[str],
    samples: dict[str, SiteSamples],
    thresholds: np.ndarray,
    selection_metric: str,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    inner_y: list[np.ndarray] = []
    inner_prob: list[np.ndarray] = []
    inner_rows: list[dict[str, Any]] = []
    for inner_site in outer_train_sites:
        inner_train_sites = [site for site in outer_train_sites if site != inner_site]
        y_val, prob_val = _fit_predict(
            model_name=model_name,
            training_cfg=training_cfg,
            train_sites=inner_train_sites,
            test_site=inner_site,
            samples=samples,
        )
        inner_y.append(y_val)
        inner_prob.append(prob_val)

    y_all = np.concatenate(inner_y)
    prob_all = np.concatenate(inner_prob)
    best_threshold, best_row, threshold_rows = _score_thresholds(
        y_true=y_all,
        prob=prob_all,
        thresholds=thresholds,
        selection_metric=selection_metric,
    )
    for row in threshold_rows:
        inner_rows.append({"inner_scope": "outer_training_sites", **row})
    return best_threshold, best_row, inner_rows


def _evaluate_model(
    alias: str,
    model_name: str,
    params: dict[str, Any],
    base_training: dict[str, Any],
    site_names: list[str],
    samples: dict[str, SiteSamples],
    thresholds: np.ndarray,
    selection_metric: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    training_cfg = _trial_training_config(base_training, model_name, params)
    fixed_threshold = float(training_cfg.get("threshold", 0.8))
    fold_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    pooled_y: list[np.ndarray] = []
    pooled_prob: list[np.ndarray] = []
    pooled_tuned_pred: list[np.ndarray] = []
    pooled_fixed_pred: list[np.ndarray] = []

    print(f"[threshold] model={alias} base={model_name} params={json.dumps(params, sort_keys=True)}", flush=True)
    for heldout_site in site_names:
        outer_train_sites = [site for site in site_names if site != heldout_site]
        selected_threshold, inner_best, inner_threshold_rows = _select_outer_threshold(
            model_name=model_name,
            training_cfg=training_cfg,
            outer_train_sites=outer_train_sites,
            samples=samples,
            thresholds=thresholds,
            selection_metric=selection_metric,
        )
        for row in inner_threshold_rows:
            threshold_rows.append(
                {
                    "model_alias": alias,
                    "model": model_name,
                    "heldout_site": heldout_site,
                    **row,
                }
            )

        y_test, prob_test = _fit_predict(
            model_name=model_name,
            training_cfg=training_cfg,
            train_sites=outer_train_sites,
            test_site=heldout_site,
            samples=samples,
        )
        tuned_pred = (prob_test >= selected_threshold).astype(np.uint8)
        fixed_pred = (prob_test >= fixed_threshold).astype(np.uint8)
        tuned_metrics = _calc_metrics(y_test, prob_test, tuned_pred)
        tuned_metrics.update(_top_fraction_metrics(y_test, prob_test))
        fixed_metrics = _calc_metrics(y_test, prob_test, fixed_pred)

        row = {
            "model_alias": alias,
            "model": model_name,
            "heldout_site": heldout_site,
            "selected_threshold": selected_threshold,
            "fixed_threshold": fixed_threshold,
            "inner_selected_metric": float(inner_best[selection_metric]),
            "inner_precision": float(inner_best["precision"]),
            "inner_recall": float(inner_best["recall"]),
            "inner_f1": float(inner_best["f1"]),
            "test_samples": int(y_test.size),
            "test_positives": int((y_test == 1).sum()),
            "test_negatives": int((y_test == 0).sum()),
            **tuned_metrics,
            **{f"fixed_{key}": value for key, value in fixed_metrics.items() if key != "confusion_matrix"},
            "fixed_confusion_matrix": fixed_metrics["confusion_matrix"],
        }
        fold_rows.append(row)
        pooled_y.append(y_test)
        pooled_prob.append(prob_test)
        pooled_tuned_pred.append(tuned_pred)
        pooled_fixed_pred.append(fixed_pred)
        print(
            f"[threshold] {alias} heldout={heldout_site} selected={selected_threshold:.2f} "
            f"inner_{selection_metric}={inner_best[selection_metric]:.4f} "
            f"outer_F1={tuned_metrics['f1']:.4f} fixed_F1={fixed_metrics['f1']:.4f}",
            flush=True,
        )

    y_all = np.concatenate(pooled_y)
    prob_all = np.concatenate(pooled_prob)
    tuned_pred_all = np.concatenate(pooled_tuned_pred)
    fixed_pred_all = np.concatenate(pooled_fixed_pred)
    pooled_tuned = _calc_metrics(y_all, prob_all, tuned_pred_all)
    pooled_tuned.update(_top_fraction_metrics(y_all, prob_all))
    pooled_fixed = _calc_metrics(y_all, prob_all, fixed_pred_all)
    threshold_values = np.array([float(row["selected_threshold"]) for row in fold_rows], dtype=np.float64)
    macro_summary = _mean_std(fold_rows, METRIC_KEYS)

    summary = {
        "model_alias": alias,
        "model": model_name,
        "params_json": json.dumps(params, sort_keys=True),
        "selection_metric": selection_metric,
        "fixed_threshold": fixed_threshold,
        "selected_threshold_mean": float(threshold_values.mean()),
        "selected_threshold_std": float(threshold_values.std(ddof=1)) if threshold_values.size > 1 else 0.0,
        "selected_threshold_min": float(threshold_values.min()),
        "selected_threshold_max": float(threshold_values.max()),
        **pooled_tuned,
        **{f"fixed_{key}": value for key, value in pooled_fixed.items() if key != "confusion_matrix"},
        "fixed_confusion_matrix": pooled_fixed["confusion_matrix"],
        **{
            f"macro_{metric}_{stat}": value
            for metric, stats in macro_summary.items()
            for stat, value in stats.items()
        },
    }
    print(
        f"[threshold] done model={alias} pooled tuned F1={summary['f1']:.4f} "
        f"fixed F1={summary['fixed_f1']:.4f} PR-AUC={summary['pr_auc']:.4f}",
        flush=True,
    )
    return summary, fold_rows, threshold_rows


def _write_markdown(
    path: Path,
    threshold_cfg: dict[str, Any],
    base_config: Path,
    summary_rows: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Nested site-blocked threshold tuning",
        "",
        f"- Base config: `{base_config}`",
        f"- Threshold selection metric: `{threshold_cfg.get('selection_metric', 'f1')}`",
        "- Outer validation: leave one site out",
        "- Inner threshold selection: leave one training site out inside each outer fold",
        "",
        "## Model Summary",
        "",
        "| Model | Mean selected threshold | Threshold range | Pooled PR-AUC | Tuned precision | Tuned recall | Tuned F1 | Fixed-0.8 F1 | Top 5% precision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary_rows, key=lambda item: float(item["f1"]), reverse=True):
        lines.append(
            f"| {row['model_alias']} | {row['selected_threshold_mean']:.3f} | "
            f"{row['selected_threshold_min']:.2f}-{row['selected_threshold_max']:.2f} | "
            f"{row['pr_auc']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['fixed_f1']:.4f} | {row['precision_at_top5pct']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Outer Folds",
            "",
            "| Model | Held-out site | Selected threshold | Inner F1 | Outer precision | Outer recall | Outer F1 | Fixed-0.8 F1 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in fold_rows:
        lines.append(
            f"| {row['model_alias']} | {row['heldout_site']} | {row['selected_threshold']:.2f} | "
            f"{row['inner_f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['fixed_f1']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Note",
            "",
            "These thresholded metrics remain honest outer-fold estimates because each selected threshold is chosen only from the training sites available inside that fold. ROC-AUC, PR-AUC, and top-k precision are ranking metrics and do not depend on the selected threshold.",
            "",
            "Caution: the tuned F1 values reflect the controlled validation sample balance, not natural landscape prevalence. Low thresholds that improve sampled-fold recall should not be used as broad probability-map cutoffs without calibration and field interpretation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_threshold_tuning(config_path: Path) -> dict[str, Any]:
    threshold_cfg, base_config_path, output_dir = _load_threshold_config(config_path)
    base_cfg = load_config(base_config_path)
    output_dir = ensure_dir(output_dir)
    thresholds = _threshold_grid(threshold_cfg)
    feature_names, site_names, samples, site_rows = _read_samples(base_cfg, threshold_cfg)
    selection_metric = str(threshold_cfg.get("selection_metric", "f1"))

    summary_rows: list[dict[str, Any]] = []
    all_fold_rows: list[dict[str, Any]] = []
    all_threshold_rows: list[dict[str, Any]] = []

    for alias, model_cfg in threshold_cfg.get("models", {}).items():
        model_name = str(model_cfg.get("model", alias))
        params = dict(model_cfg.get("params", {}))
        summary, fold_rows, threshold_rows = _evaluate_model(
            alias=alias,
            model_name=model_name,
            params=params,
            base_training=base_cfg["training"],
            site_names=site_names,
            samples=samples,
            thresholds=thresholds,
            selection_metric=selection_metric,
        )
        summary_rows.append(summary)
        all_fold_rows.extend(fold_rows)
        all_threshold_rows.extend(threshold_rows)

    result = {
        "threshold_tuning": {
            "config": str(Path(config_path).expanduser().resolve()),
            "base_config": str(base_config_path),
            "selection_metric": selection_metric,
            "threshold_grid": [float(v) for v in thresholds],
            "site_blocked": threshold_cfg.get("site_blocked", {}),
        },
        "inputs": {
            "feature_stack_tif": str(base_cfg["paths"]["feature_stack_tif"]),
            "label_tif": str(base_cfg["paths"]["label_tif"]),
            "label_geojson": str(base_cfg["paths"]["geology_geojson"]),
            "feature_names": feature_names,
        },
        "site_samples": site_rows,
        "models": summary_rows,
        "folds": all_fold_rows,
    }

    save_json(output_dir / "site_blocked_thresholds.json", result)
    pd.DataFrame(summary_rows).sort_values("f1", ascending=False).to_csv(
        output_dir / "site_blocked_threshold_summary.csv",
        index=False,
    )
    pd.DataFrame(all_fold_rows).to_csv(output_dir / "site_blocked_threshold_folds.csv", index=False)
    pd.DataFrame(all_threshold_rows).to_csv(output_dir / "site_blocked_inner_threshold_grid.csv", index=False)
    pd.DataFrame(site_rows).to_csv(output_dir / "site_blocked_threshold_site_samples.csv", index=False)
    _write_markdown(
        output_dir / "site_blocked_thresholds.md",
        threshold_cfg=threshold_cfg,
        base_config=base_config_path,
        summary_rows=summary_rows,
        fold_rows=all_fold_rows,
    )
    print(f"[threshold] saved: {output_dir / 'site_blocked_thresholds.json'}", flush=True)
    return result


def main() -> int:
    args = _parse_args()
    run_threshold_tuning(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
