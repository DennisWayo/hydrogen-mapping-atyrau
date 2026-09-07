#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for import_path in [SRC_ROOT, SCRIPT_ROOT]:
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from xaigis.config import load_config
from xaigis.modeling import _build_models, _predict_positive_probability
from xaigis.utils import ensure_dir, save_json

from tune_site_blocked_models import _read_samples, _trial_training_config
from validate_site_blocked import SiteSamples


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate probability calibration with nested leave-one-site-out folds. "
            "Each outer fold fits calibrators only from inner out-of-fold predictions "
            "on the training sites."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "paper_runs/ex_01/configs/ex_01_site_blocked_calibration_external_priors.json",
        help="Calibration config.",
    )
    return parser.parse_args()


def _load_calibration_config(path: Path) -> tuple[dict[str, Any], Path, Path]:
    cfg_path = path.expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        calibration_cfg = json.load(f)

    base_config = Path(calibration_cfg["base_config"]).expanduser()
    if not base_config.is_absolute():
        base_config = (cfg_path.parent / base_config).resolve()

    output_dir = Path(calibration_cfg["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = (cfg_path.parent / output_dir).resolve()

    return calibration_cfg, base_config, output_dir


def _fit_predict_raw(
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


def _inner_oof_predictions(
    model_name: str,
    training_cfg: dict[str, Any],
    outer_train_sites: list[str],
    samples: dict[str, SiteSamples],
) -> tuple[np.ndarray, np.ndarray]:
    y_parts: list[np.ndarray] = []
    prob_parts: list[np.ndarray] = []
    for inner_site in outer_train_sites:
        inner_train_sites = [site for site in outer_train_sites if site != inner_site]
        y_val, prob_val = _fit_predict_raw(
            model_name=model_name,
            training_cfg=training_cfg,
            train_sites=inner_train_sites,
            test_site=inner_site,
            samples=samples,
        )
        y_parts.append(y_val)
        prob_parts.append(prob_val)
    return np.concatenate(y_parts), np.concatenate(prob_parts)


def _fit_calibrator(kind: str, y_cal: np.ndarray, prob_cal: np.ndarray) -> Any:
    if kind == "raw":
        return None
    if kind == "platt":
        model = LogisticRegression(solver="lbfgs", max_iter=1000)
        model.fit(prob_cal.reshape(-1, 1), y_cal)
        return model
    if kind == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(prob_cal, y_cal)
        return model
    raise ValueError(f"Unsupported calibrator: {kind}")


def _apply_calibrator(kind: str, calibrator: Any, prob: np.ndarray) -> np.ndarray:
    if kind == "raw":
        calibrated = prob
    elif kind == "platt":
        calibrated = calibrator.predict_proba(prob.reshape(-1, 1))[:, 1]
    elif kind == "isotonic":
        calibrated = calibrator.predict(prob)
    else:
        raise ValueError(f"Unsupported calibrator: {kind}")
    return np.clip(np.asarray(calibrated, dtype=np.float64), 1e-6, 1.0 - 1e-6)


def _expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict[str, Any]]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    ece = 0.0
    for idx in range(n_bins):
        left = bins[idx]
        right = bins[idx + 1]
        if idx == n_bins - 1:
            mask = (prob >= left) & (prob <= right)
        else:
            mask = (prob >= left) & (prob < right)
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin": idx,
                    "bin_min": float(left),
                    "bin_max": float(right),
                    "count": 0,
                    "mean_probability": None,
                    "positive_fraction": None,
                    "abs_error": None,
                }
            )
            continue
        mean_probability = float(prob[mask].mean())
        positive_fraction = float(y_true[mask].mean())
        abs_error = abs(mean_probability - positive_fraction)
        ece += (count / max(y_true.size, 1)) * abs_error
        rows.append(
            {
                "bin": idx,
                "bin_min": float(left),
                "bin_max": float(right),
                "count": count,
                "mean_probability": mean_probability,
                "positive_fraction": positive_fraction,
                "abs_error": float(abs_error),
            }
        )
    return float(ece), rows


def _calibration_metrics(y_true: np.ndarray, prob: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ece, bin_rows = _expected_calibration_error(y_true, prob)
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "pr_auc": float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "brier": float(brier_score_loss(y_true, prob)),
        "log_loss": float(log_loss(y_true, prob, labels=[0, 1])),
        "ece_10bin": ece,
        "mean_probability": float(prob.mean()),
        "positive_rate": float(y_true.mean()),
    }
    return metrics, bin_rows


def _evaluate_model(
    alias: str,
    model_name: str,
    params: dict[str, Any],
    calibrators: list[str],
    base_training: dict[str, Any],
    site_names: list[str],
    samples: dict[str, SiteSamples],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    training_cfg = _trial_training_config(base_training, model_name, params)
    fold_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    pooled: dict[str, dict[str, list[np.ndarray]]] = {
        kind: {"y": [], "prob": []} for kind in calibrators
    }

    print(f"[calibrate] model={alias} base={model_name} params={json.dumps(params, sort_keys=True)}", flush=True)
    for heldout_site in site_names:
        outer_train_sites = [site for site in site_names if site != heldout_site]
        y_cal, prob_cal = _inner_oof_predictions(
            model_name=model_name,
            training_cfg=training_cfg,
            outer_train_sites=outer_train_sites,
            samples=samples,
        )
        y_test, raw_prob_test = _fit_predict_raw(
            model_name=model_name,
            training_cfg=training_cfg,
            train_sites=outer_train_sites,
            test_site=heldout_site,
            samples=samples,
        )

        for kind in calibrators:
            calibrator = _fit_calibrator(kind, y_cal, prob_cal)
            prob_test = _apply_calibrator(kind, calibrator, raw_prob_test)
            metrics, fold_bin_rows = _calibration_metrics(y_test, prob_test)
            fold_rows.append(
                {
                    "model_alias": alias,
                    "model": model_name,
                    "calibrator": kind,
                    "heldout_site": heldout_site,
                    "calibration_samples": int(y_cal.size),
                    "calibration_positive_rate": float(y_cal.mean()),
                    "test_samples": int(y_test.size),
                    "test_positive_rate": float(y_test.mean()),
                    **metrics,
                }
            )
            for row in fold_bin_rows:
                bin_rows.append(
                    {
                        "model_alias": alias,
                        "model": model_name,
                        "calibrator": kind,
                        "heldout_site": heldout_site,
                        **row,
                    }
                )
            pooled[kind]["y"].append(y_test)
            pooled[kind]["prob"].append(prob_test)
            print(
                f"[calibrate] {alias} heldout={heldout_site} calibrator={kind} "
                f"brier={metrics['brier']:.4f} ece={metrics['ece_10bin']:.4f}",
                flush=True,
            )

    summary_rows: list[dict[str, Any]] = []
    for kind, parts in pooled.items():
        y_all = np.concatenate(parts["y"])
        prob_all = np.concatenate(parts["prob"])
        metrics, pooled_bin_rows = _calibration_metrics(y_all, prob_all)
        model_fold_rows = [
            row
            for row in fold_rows
            if row["model_alias"] == alias and row["calibrator"] == kind
        ]
        for metric_name in ["brier", "log_loss", "ece_10bin", "mean_probability"]:
            values = np.array([float(row[metric_name]) for row in model_fold_rows], dtype=np.float64)
            metrics[f"macro_{metric_name}_mean"] = float(values.mean())
            metrics[f"macro_{metric_name}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        summary_rows.append(
            {
                "model_alias": alias,
                "model": model_name,
                "calibrator": kind,
                "params_json": json.dumps(params, sort_keys=True),
                **metrics,
            }
        )
        for row in pooled_bin_rows:
            bin_rows.append(
                {
                    "model_alias": alias,
                    "model": model_name,
                    "calibrator": kind,
                    "heldout_site": "pooled",
                    **row,
                }
            )
    return summary_rows, fold_rows, bin_rows


def _write_markdown(path: Path, base_config: Path, summary_rows: list[dict[str, Any]], fold_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Nested site-blocked probability calibration",
        "",
        f"- Base config: `{base_config}`",
        "- Outer validation: leave one site out",
        "- Calibration data: inner out-of-fold predictions from training sites only",
        "- Calibration target: sampled proxy-label validation distribution",
        "",
        "## Calibrator Summary",
        "",
        "| Model | Calibrator | Brier | Log loss | ECE 10-bin | Mean probability | Positive rate | ROC-AUC | PR-AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary_rows, key=lambda item: float(item["brier"])):
        lines.append(
            f"| {row['model_alias']} | {row['calibrator']} | {row['brier']:.4f} | "
            f"{row['log_loss']:.4f} | {row['ece_10bin']:.4f} | {row['mean_probability']:.4f} | "
            f"{row['positive_rate']:.4f} | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Outer Folds",
            "",
            "| Model | Calibrator | Held-out site | Brier | Log loss | ECE 10-bin | Mean probability | Positive rate |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in fold_rows:
        lines.append(
            f"| {row['model_alias']} | {row['calibrator']} | {row['heldout_site']} | "
            f"{row['brier']:.4f} | {row['log_loss']:.4f} | {row['ece_10bin']:.4f} | "
            f"{row['mean_probability']:.4f} | {row['test_positive_rate']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Note",
            "",
            "Calibration is leakage-aware because each held-out site receives a calibrator fitted only on inner out-of-fold predictions from the training sites. These calibrated values should still be described as calibrated prospectivity scores rather than field-confirmed hydrogen probabilities because labels are proxy AOI buffers and the validation sample prevalence is controlled.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_calibration(config_path: Path) -> dict[str, Any]:
    calibration_cfg, base_config_path, output_dir = _load_calibration_config(config_path)
    base_cfg = load_config(base_config_path)
    output_dir = ensure_dir(output_dir)
    feature_names, site_names, samples, site_rows = _read_samples(base_cfg, calibration_cfg)

    summary_rows: list[dict[str, Any]] = []
    all_fold_rows: list[dict[str, Any]] = []
    all_bin_rows: list[dict[str, Any]] = []
    for alias, model_cfg in calibration_cfg.get("models", {}).items():
        model_name = str(model_cfg.get("model", alias))
        params = dict(model_cfg.get("params", {}))
        calibrators = [str(item) for item in model_cfg.get("calibrators", ["raw", "platt", "isotonic"])]
        model_summary, fold_rows, bin_rows = _evaluate_model(
            alias=alias,
            model_name=model_name,
            params=params,
            calibrators=calibrators,
            base_training=base_cfg["training"],
            site_names=site_names,
            samples=samples,
        )
        summary_rows.extend(model_summary)
        all_fold_rows.extend(fold_rows)
        all_bin_rows.extend(bin_rows)

    best = sorted(summary_rows, key=lambda row: float(row["brier"]))[0]
    result = {
        "calibration": {
            "config": str(Path(config_path).expanduser().resolve()),
            "base_config": str(base_config_path),
            "site_blocked": calibration_cfg.get("site_blocked", {}),
            "selection_metric": "brier",
        },
        "inputs": {
            "feature_stack_tif": str(base_cfg["paths"]["feature_stack_tif"]),
            "label_tif": str(base_cfg["paths"]["label_tif"]),
            "label_geojson": str(base_cfg["paths"]["geology_geojson"]),
            "feature_names": feature_names,
        },
        "site_samples": site_rows,
        "best_calibrator": best,
        "models": summary_rows,
        "folds": all_fold_rows,
    }

    save_json(output_dir / "site_blocked_calibration.json", result)
    pd.DataFrame(summary_rows).sort_values("brier", ascending=True).to_csv(
        output_dir / "site_blocked_calibration_summary.csv",
        index=False,
    )
    pd.DataFrame(all_fold_rows).to_csv(output_dir / "site_blocked_calibration_folds.csv", index=False)
    pd.DataFrame(all_bin_rows).to_csv(output_dir / "site_blocked_calibration_bins.csv", index=False)
    pd.DataFrame(site_rows).to_csv(output_dir / "site_blocked_calibration_site_samples.csv", index=False)
    _write_markdown(
        output_dir / "site_blocked_calibration.md",
        base_config=base_config_path,
        summary_rows=summary_rows,
        fold_rows=all_fold_rows,
    )
    print(
        f"[calibrate] best model={best['model_alias']} calibrator={best['calibrator']} "
        f"brier={best['brier']:.4f} ece={best['ece_10bin']:.4f}",
        flush=True,
    )
    print(f"[calibrate] saved: {output_dir / 'site_blocked_calibration.json'}", flush=True)
    return result


def main() -> int:
    args = _parse_args()
    run_calibration(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
