#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.windows import Window
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for import_path in [SRC_ROOT, SCRIPT_ROOT]:
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from xaigis.config import load_config
from xaigis.modeling import _build_models, _nodata_any, _predict_positive_probability
from xaigis.utils import ensure_dir, save_json

from tune_site_blocked_models import _read_samples, _trial_training_config
from validate_site_blocked import SiteSamples


NODATA_FLOAT = -9999.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train final ex_01 carried-forward models and write final prospectivity rasters. "
            "SGD receives Platt calibration fitted from site-wise out-of-fold predictions; "
            "RF is exported only as a top-k ranking sensitivity layer."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "paper_runs/ex_01/configs/ex_01_final_maps_external_priors.json",
        help="Final map generation config.",
    )
    return parser.parse_args()


def _load_map_config(path: Path) -> tuple[dict[str, Any], Path, Path]:
    cfg_path = path.expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        map_cfg = json.load(f)

    base_config = Path(map_cfg["base_config"]).expanduser()
    if not base_config.is_absolute():
        base_config = (cfg_path.parent / base_config).resolve()

    output_dir = Path(map_cfg["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = (cfg_path.parent / output_dir).resolve()

    return map_cfg, base_config, output_dir


def _fit_model(
    model_name: str,
    training_cfg: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> Any:
    models = _build_models(training_cfg, y_train)
    if model_name not in models:
        raise RuntimeError(f"Configured model '{model_name}' was not built. Built models: {sorted(models)}")
    model = models[model_name]
    model.fit(x_train, y_train)
    return model


def _samples_xy(site_names: list[str], samples: dict[str, SiteSamples]) -> tuple[np.ndarray, np.ndarray]:
    x = np.vstack([samples[site].x for site in site_names]).astype(np.float32)
    y = np.concatenate([samples[site].y for site in site_names]).astype(np.uint8)
    return x, y


def _fit_predict_oof(
    model_name: str,
    training_cfg: dict[str, Any],
    train_sites: list[str],
    heldout_site: str,
    samples: dict[str, SiteSamples],
) -> tuple[np.ndarray, np.ndarray]:
    x_train, y_train = _samples_xy(train_sites, samples)
    model = _fit_model(model_name, training_cfg, x_train, y_train)
    x_test = samples[heldout_site].x.astype(np.float32)
    y_test = samples[heldout_site].y.astype(np.uint8)
    prob = _predict_positive_probability(model, x_test)
    return y_test, prob


def _fit_platt_from_site_oof(
    model_name: str,
    training_cfg: dict[str, Any],
    site_names: list[str],
    samples: dict[str, SiteSamples],
) -> tuple[LogisticRegression, dict[str, Any]]:
    y_parts: list[np.ndarray] = []
    prob_parts: list[np.ndarray] = []
    fold_rows: list[dict[str, Any]] = []
    for heldout_site in site_names:
        train_sites = [site for site in site_names if site != heldout_site]
        y_test, prob = _fit_predict_oof(
            model_name=model_name,
            training_cfg=training_cfg,
            train_sites=train_sites,
            heldout_site=heldout_site,
            samples=samples,
        )
        y_parts.append(y_test)
        prob_parts.append(prob)
        fold_rows.append(
            {
                "heldout_site": heldout_site,
                "samples": int(y_test.size),
                "positive_rate": float(y_test.mean()),
                "raw_mean_probability": float(prob.mean()),
            }
        )
        print(
            f"[final-map] calibration OOF {model_name} heldout={heldout_site} "
            f"raw_mean={prob.mean():.4f}",
            flush=True,
        )

    y_oof = np.concatenate(y_parts)
    prob_oof = np.concatenate(prob_parts)
    calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
    calibrator.fit(prob_oof.reshape(-1, 1), y_oof)
    calibrated = _apply_platt(calibrator, prob_oof)
    summary = {
        "method": "site_oof_platt",
        "samples": int(y_oof.size),
        "positive_rate": float(y_oof.mean()),
        "raw": _probability_metrics(y_oof, prob_oof),
        "platt": _probability_metrics(y_oof, calibrated),
        "folds": fold_rows,
        "coef": float(calibrator.coef_[0, 0]),
        "intercept": float(calibrator.intercept_[0]),
    }
    return calibrator, summary


def _apply_platt(calibrator: LogisticRegression, prob: np.ndarray) -> np.ndarray:
    calibrated = calibrator.predict_proba(prob.reshape(-1, 1))[:, 1]
    return np.clip(np.asarray(calibrated, dtype=np.float32), 1e-6, 1.0 - 1e-6)


def _probability_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    prob = np.clip(np.asarray(prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "pr_auc": float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "brier": float(brier_score_loss(y_true, prob)),
        "log_loss": float(log_loss(y_true, prob, labels=[0, 1])),
        "mean_probability": float(prob.mean()),
        "positive_rate": float(y_true.mean()),
    }


def _valid_mask(patch: np.ndarray, nodatavals: tuple[Any, ...], exclude_nodata: bool) -> np.ndarray:
    valid = np.isfinite(patch).all(axis=0)
    if exclude_nodata:
        valid &= ~_nodata_any(patch, nodatavals)
    return valid


def _iter_windows(height: int, width: int, tile_size: int) -> list[Window]:
    windows: list[Window] = []
    for row in range(0, height, tile_size):
        h = min(tile_size, height - row)
        for col in range(0, width, tile_size):
            w = min(tile_size, width - col)
            windows.append(Window(col_off=col, row_off=row, width=w, height=h))
    return windows


def _write_prediction_rasters(
    stack_tif: Path,
    out_dir: Path,
    sgd_model: Any,
    platt: LogisticRegression,
    rf_model: Any,
    tile_size: int,
    exclude_nodata: bool,
) -> dict[str, Any]:
    sgd_raw_path = out_dir / "sgd_primary_raw_score.tif"
    sgd_platt_path = out_dir / "sgd_primary_platt_prospectivity_score.tif"
    rf_score_path = out_dir / "rf_topk_sensitivity_score.tif"
    valid_mask_path = out_dir / "strict_valid_data_mask.tif"
    score_chunks: dict[str, list[np.ndarray]] = {"sgd_platt": [], "rf": []}

    with rio.open(stack_tif) as src:
        score_profile = src.profile.copy()
        score_profile.update(
            count=1,
            dtype="float32",
            nodata=NODATA_FLOAT,
            compress="deflate",
            BIGTIFF="YES",
            predictor=3,
        )
        mask_profile = src.profile.copy()
        mask_profile.update(
            count=1,
            dtype="uint8",
            nodata=0,
            compress="deflate",
            BIGTIFF="YES",
        )
        windows = _iter_windows(src.height, src.width, tile_size)
        valid_pixels = 0
        total_pixels = int(src.width * src.height)
        with rio.open(sgd_raw_path, "w", **score_profile) as dst_sgd_raw, rio.open(
            sgd_platt_path, "w", **score_profile
        ) as dst_sgd_platt, rio.open(rf_score_path, "w", **score_profile) as dst_rf, rio.open(
            valid_mask_path, "w", **mask_profile
        ) as dst_valid:
            for idx, window in enumerate(windows, start=1):
                patch = src.read(window=window).astype(np.float32)
                h, w = int(window.height), int(window.width)
                valid = _valid_mask(patch, src.nodatavals, exclude_nodata=exclude_nodata)
                valid_pixels += int(valid.sum())

                sgd_raw_patch = np.full((h, w), NODATA_FLOAT, dtype=np.float32)
                sgd_platt_patch = np.full((h, w), NODATA_FLOAT, dtype=np.float32)
                rf_patch = np.full((h, w), NODATA_FLOAT, dtype=np.float32)
                valid_patch = valid.astype(np.uint8)
                if np.any(valid):
                    x_tile = patch[:, valid].T.astype(np.float32)
                    sgd_raw = _predict_positive_probability(sgd_model, x_tile).astype(np.float32)
                    sgd_platt = _apply_platt(platt, sgd_raw)
                    rf_score = _predict_positive_probability(rf_model, x_tile).astype(np.float32)
                    sgd_raw_patch[valid] = sgd_raw
                    sgd_platt_patch[valid] = sgd_platt
                    rf_patch[valid] = rf_score
                    score_chunks["sgd_platt"].append(sgd_platt.astype(np.float32, copy=False))
                    score_chunks["rf"].append(rf_score.astype(np.float32, copy=False))

                dst_sgd_raw.write(sgd_raw_patch, 1, window=window)
                dst_sgd_platt.write(sgd_platt_patch, 1, window=window)
                dst_rf.write(rf_patch, 1, window=window)
                dst_valid.write(valid_patch, 1, window=window)

                if idx == 1 or idx % 50 == 0 or idx == len(windows):
                    print(
                        f"[final-map] predicted window {idx:,}/{len(windows):,}; "
                        f"valid_pixels={valid_pixels:,}/{total_pixels:,}",
                        flush=True,
                    )

    return {
        "sgd_raw_score_tif": str(sgd_raw_path),
        "sgd_platt_prospectivity_score_tif": str(sgd_platt_path),
        "rf_topk_sensitivity_score_tif": str(rf_score_path),
        "strict_valid_data_mask_tif": str(valid_mask_path),
        "valid_pixels": valid_pixels,
        "score_chunks": score_chunks,
    }


def _score_quantiles(scores: np.ndarray, quantiles: list[float]) -> dict[str, float]:
    return {f"q{int(q * 100):02d}": float(np.quantile(scores, q)) for q in quantiles}


def _topk_thresholds(scores: np.ndarray, fractions: list[float]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for frac in fractions:
        frac = float(frac)
        if not 0.0 < frac < 1.0:
            raise ValueError(f"Top-k fraction must be between 0 and 1: {frac}")
        thresholds[f"top{int(frac * 100):02d}pct"] = float(np.quantile(scores, 1.0 - frac))
    return thresholds


def _write_topk_masks(score_tif: Path, thresholds: dict[str, float], out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with rio.open(score_tif) as src:
        mask_profile = src.profile.copy()
        mask_profile.update(count=1, dtype="uint8", nodata=0, compress="deflate", BIGTIFF="YES")
        windows = _iter_windows(src.height, src.width, 512)
        for label, threshold in thresholds.items():
            out_path = out_dir / f"rf_{label}_sensitivity_zone.tif"
            selected_pixels = 0
            with rio.open(out_path, "w", **mask_profile) as dst:
                for window in windows:
                    score = src.read(1, window=window).astype(np.float32)
                    valid = np.isfinite(score) & (score != src.nodata)
                    mask = valid & (score >= threshold)
                    selected_pixels += int(mask.sum())
                    dst.write(mask.astype(np.uint8), 1, window=window)
            rows.append(
                {
                    "model": "rf_topk_sensitivity",
                    "zone": label,
                    "threshold": float(threshold),
                    "selected_pixels": selected_pixels,
                    "mask_tif": str(out_path),
                }
            )
            print(
                f"[final-map] wrote {label} RF zone; threshold={threshold:.6f}; "
                f"selected_pixels={selected_pixels:,}",
                flush=True,
            )
    return rows


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    outputs = result["outputs"]
    score_summary = result["score_summary"]
    rf_zones = result["rf_topk_zones"]
    calibration = result["calibration"]["platt_oof_summary"]
    lines = [
        "# Final prospectivity maps",
        "",
        "## Inputs",
        "",
        f"- Base config: `{result['config']['base_config']}`",
        f"- Feature stack: `{result['inputs']['feature_stack_tif']}`",
        f"- Training samples: {result['training']['samples']:,}",
        f"- Positive rate in training samples: {result['training']['positive_rate']:.4f}",
        f"- Strict-valid raster pixels: {outputs['valid_pixels']:,}",
        "",
        "## Outputs",
        "",
        f"- SGD raw score: `{outputs['sgd_raw_score_tif']}`",
        f"- SGD Platt-calibrated prospectivity score: `{outputs['sgd_platt_prospectivity_score_tif']}`",
        f"- RF sensitivity score: `{outputs['rf_topk_sensitivity_score_tif']}`",
        f"- Strict valid-data mask: `{outputs['strict_valid_data_mask_tif']}`",
        "",
        "## SGD Calibration",
        "",
        "This table summarizes the final Platt calibrator fitted from all site-wise out-of-fold predictions for map production. Leakage-aware calibration validation is reported separately in `site_blocked_calibration.md`.",
        "",
        "| Score | ROC-AUC | PR-AUC | Brier | Log loss | Mean score | Positive rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| raw OOF | {calibration['raw']['roc_auc']:.4f} | {calibration['raw']['pr_auc']:.4f} | "
            f"{calibration['raw']['brier']:.4f} | {calibration['raw']['log_loss']:.4f} | "
            f"{calibration['raw']['mean_probability']:.4f} | {calibration['raw']['positive_rate']:.4f} |"
        ),
        (
            f"| Platt OOF | {calibration['platt']['roc_auc']:.4f} | {calibration['platt']['pr_auc']:.4f} | "
            f"{calibration['platt']['brier']:.4f} | {calibration['platt']['log_loss']:.4f} | "
            f"{calibration['platt']['mean_probability']:.4f} | {calibration['platt']['positive_rate']:.4f} |"
        ),
        "",
        "## Raster Score Summary",
        "",
        "| Score | Valid pixels | Mean | Std | Q50 | Q75 | Q90 | Q95 | Q99 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in score_summary:
        lines.append(
            f"| {row['score']} | {row['valid_pixels']} | {row['mean']:.4f} | {row['std']:.4f} | "
            f"{row.get('q50', 0.0):.4f} | {row.get('q75', 0.0):.4f} | {row.get('q90', 0.0):.4f} | "
            f"{row.get('q95', 0.0):.4f} | {row.get('q99', 0.0):.4f} |"
        )

    lines.extend(
        [
            "",
            "## RF Top-k Sensitivity Zones",
            "",
            "| Zone | Threshold | Selected pixels | Mask |",
            "|---|---:|---:|---|",
        ]
    )
    for row in rf_zones:
        lines.append(
            f"| {row['zone']} | {row['threshold']:.6f} | {row['selected_pixels']} | `{row['mask_tif']}` |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Note",
            "",
            "The SGD Platt raster is a calibrated prospectivity score for the sampled proxy-label distribution, not a field-confirmed hydrogen probability. The RF masks are ranking-sensitivity zones and should be interpreted only as top-ranked candidate areas under the RF sensitivity model. Top-k mask pixel counts may slightly exceed the nominal fraction where many pixels share the same RF score at the threshold.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_final_maps(config_path: Path) -> dict[str, Any]:
    map_cfg, base_config_path, output_dir = _load_map_config(config_path)
    base_cfg = load_config(base_config_path)
    output_dir = ensure_dir(output_dir)
    models_dir = ensure_dir(output_dir / "models")
    feature_names, site_names, samples, site_rows = _read_samples(base_cfg, map_cfg)

    sgd_cfg = map_cfg["models"]["sgd_primary"]
    rf_cfg = map_cfg["models"]["rf_topk_sensitivity"]
    sgd_training_cfg = _trial_training_config(base_cfg["training"], str(sgd_cfg["model"]), dict(sgd_cfg["params"]))
    rf_training_cfg = _trial_training_config(base_cfg["training"], str(rf_cfg["model"]), dict(rf_cfg["params"]))

    print("[final-map] fitting Platt calibrator from site-wise OOF SGD predictions", flush=True)
    platt, platt_summary = _fit_platt_from_site_oof(
        model_name=str(sgd_cfg["model"]),
        training_cfg=sgd_training_cfg,
        site_names=site_names,
        samples=samples,
    )

    x_all, y_all = _samples_xy(site_names, samples)
    print(f"[final-map] fitting final SGD on all samples: {x_all.shape}", flush=True)
    sgd_model = _fit_model(str(sgd_cfg["model"]), sgd_training_cfg, x_all, y_all)
    print(f"[final-map] fitting final RF on all samples: {x_all.shape}", flush=True)
    rf_model = _fit_model(str(rf_cfg["model"]), rf_training_cfg, x_all, y_all)

    joblib.dump(sgd_model, models_dir / "sgd_primary.joblib")
    joblib.dump(platt, models_dir / "sgd_primary_platt_calibrator.joblib")
    joblib.dump(rf_model, models_dir / "rf_topk_sensitivity.joblib")

    pred_cfg = map_cfg.get("prediction", {})
    outputs = _write_prediction_rasters(
        stack_tif=Path(base_cfg["paths"]["feature_stack_tif"]),
        out_dir=output_dir,
        sgd_model=sgd_model,
        platt=platt,
        rf_model=rf_model,
        tile_size=int(pred_cfg.get("tile_size", 512)),
        exclude_nodata=bool(pred_cfg.get("exclude_nodata", True)),
    )
    score_chunks = outputs.pop("score_chunks")
    score_summary: list[dict[str, Any]] = []
    quantiles = [float(q) for q in map_cfg.get("score_quantiles", [0.5, 0.75, 0.9, 0.95, 0.99])]
    score_arrays: dict[str, np.ndarray] = {}
    for score_name, chunks in score_chunks.items():
        scores = np.concatenate(chunks).astype(np.float32) if chunks else np.array([], dtype=np.float32)
        score_arrays[score_name] = scores
        row: dict[str, Any] = {
            "score": score_name,
            "valid_pixels": int(scores.size),
            "mean": float(scores.mean()) if scores.size else 0.0,
            "std": float(scores.std()) if scores.size else 0.0,
            "min": float(scores.min()) if scores.size else 0.0,
            "max": float(scores.max()) if scores.size else 0.0,
        }
        row.update(_score_quantiles(scores, quantiles) if scores.size else {})
        score_summary.append(row)

    topk_fractions = [float(v) for v in rf_cfg.get("topk_fractions", [0.01, 0.05, 0.1])]
    rf_thresholds = _topk_thresholds(score_arrays["rf"], topk_fractions)
    rf_zone_rows = _write_topk_masks(
        score_tif=Path(outputs["rf_topk_sensitivity_score_tif"]),
        thresholds=rf_thresholds,
        out_dir=output_dir,
    )

    result: dict[str, Any] = {
        "config": {
            "map_config": str(Path(config_path).expanduser().resolve()),
            "base_config": str(base_config_path),
        },
        "inputs": {
            "feature_stack_tif": str(base_cfg["paths"]["feature_stack_tif"]),
            "label_tif": str(base_cfg["paths"]["label_tif"]),
            "label_geojson": str(base_cfg["paths"]["geology_geojson"]),
            "feature_names": feature_names,
        },
        "site_samples": site_rows,
        "training": {
            "samples": int(y_all.size),
            "positives": int((y_all == 1).sum()),
            "negatives": int((y_all == 0).sum()),
            "positive_rate": float(y_all.mean()),
        },
        "models": {
            "sgd_primary": {
                "model": str(sgd_cfg["model"]),
                "params": dict(sgd_cfg["params"]),
                "model_joblib": str(models_dir / "sgd_primary.joblib"),
                "platt_calibrator_joblib": str(models_dir / "sgd_primary_platt_calibrator.joblib"),
            },
            "rf_topk_sensitivity": {
                "model": str(rf_cfg["model"]),
                "params": dict(rf_cfg["params"]),
                "model_joblib": str(models_dir / "rf_topk_sensitivity.joblib"),
            },
        },
        "calibration": {
            "platt_oof_summary": platt_summary,
        },
        "outputs": outputs,
        "score_summary": score_summary,
        "rf_topk_zones": rf_zone_rows,
    }
    save_json(output_dir / "final_prospectivity_maps.json", result)
    pd.DataFrame(score_summary).to_csv(output_dir / "final_score_summary.csv", index=False)
    pd.DataFrame(rf_zone_rows).to_csv(output_dir / "rf_topk_sensitivity_zones.csv", index=False)
    _write_markdown(output_dir / "final_prospectivity_maps.md", result)
    print(f"[final-map] saved: {output_dir / 'final_prospectivity_maps.json'}", flush=True)
    return result


def main() -> int:
    args = _parse_args()
    run_final_maps(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
