#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.enums import Resampling
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0.0 else 0.0


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0, out=np.empty_like(x, dtype=np.float32))


def _stretch_rgb(rgb: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rgb, dtype=np.float32)
    for i in range(rgb.shape[0]):
        band = rgb[i]
        finite = np.isfinite(band)
        if not np.any(finite):
            continue
        lo = np.percentile(band[finite], 2.0)
        hi = np.percentile(band[finite], 98.0)
        if hi <= lo:
            hi = lo + 1.0
        out[i] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return out


def _model_sort_key(name: str) -> tuple[int, str]:
    pri = {"rf": 0, "sgd": 1, "xgb": 2, "lgbm": 3}
    return (pri.get(name, 99), name)


def _save_fig(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_base.with_suffix(".png")), dpi=300, bbox_inches="tight")
    fig.savefig(str(out_base.with_suffix(".svg")), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reviewer-facing spatial publication figures (ROC/PR, confusion heatmaps, prospectivity maps, error maps)."
    )
    parser.add_argument(
        "--label-tif",
        default="paper_runs/runs/south_kazakhstan_region/outputs/h2_label_poly_10m.tif",
    )
    parser.add_argument(
        "--stack-tif",
        default="paper_runs/runs/south_kazakhstan_region/outputs/S2_feature_stack_10m.tif",
    )
    parser.add_argument(
        "--pred-dir",
        default="paper_runs/runs/south_kazakhstan_region/artifacts/predictions",
    )
    parser.add_argument(
        "--metrics-json",
        default="paper_runs/runs/south_kazakhstan_region/artifacts/metrics.json",
    )
    parser.add_argument(
        "--out-fig-dir",
        default="figures/paper_results",
    )
    parser.add_argument(
        "--out-analysis-dir",
        default="paper_runs/runs/south_kazakhstan_region/analysis_spatial",
    )
    parser.add_argument("--max-curve-samples", type=int, default=1_500_000)
    parser.add_argument("--map-size", type=int, default=2200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    label_tif = Path(args.label_tif).resolve()
    stack_tif = Path(args.stack_tif).resolve()
    pred_dir = Path(args.pred_dir).resolve()
    metrics_json = Path(args.metrics_json).resolve()
    out_fig_dir = Path(args.out_fig_dir).resolve()
    out_analysis_dir = Path(args.out_analysis_dir).resolve()
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    out_analysis_dir.mkdir(parents=True, exist_ok=True)

    metrics = _read_json(metrics_json) if metrics_json.exists() else {}
    thresholds = {
        str(k): float(v)
        for k, v in dict(metrics.get("model_thresholds", {})).items()
    }
    default_threshold = float(metrics.get("threshold", 0.8))

    prob_paths = {p.stem.replace("_prob", ""): p for p in pred_dir.glob("*_prob.tif")}
    if not prob_paths:
        raise FileNotFoundError(f"No *_prob.tif files found in {pred_dir}")
    models = sorted(prob_paths.keys(), key=_model_sort_key)

    rng = np.random.default_rng(int(args.seed))

    with rio.open(label_tif) as ds_label:
        label_full = ds_label.read(1).astype(np.uint8)
    valid_label_mask = np.isin(label_full, [0, 1])

    roc_curves: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    pr_curves: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    confusion_by_model: dict[str, np.ndarray] = {}
    spatial_rows: list[dict[str, Any]] = []

    for model in models:
        threshold = float(thresholds.get(model, default_threshold))
        prob_path = prob_paths[model]
        with rio.open(prob_path) as ds_prob:
            prob_full = ds_prob.read(1).astype(np.float32)
            nodata = ds_prob.nodata

        valid = valid_label_mask & np.isfinite(prob_full)
        if nodata is not None:
            valid &= prob_full != float(nodata)

        n_valid = int(valid.sum())
        if n_valid == 0:
            print(f"[spatial-fig] warning: no valid labeled pixels for {model}, skipping")
            continue

        y_all = label_full[valid].astype(np.uint8)
        p_all = _clip01(prob_full[valid].astype(np.float32))
        del prob_full

        if n_valid > int(args.max_curve_samples):
            idx = rng.choice(n_valid, size=int(args.max_curve_samples), replace=False)
            y_curve = y_all[idx]
            p_curve = p_all[idx]
        else:
            y_curve = y_all
            p_curve = p_all

        fpr, tpr, _ = roc_curve(y_curve, p_curve)
        precision, recall, _ = precision_recall_curve(y_curve, p_curve)
        roc_auc = float(auc(fpr, tpr))
        pr_auc = float(auc(recall, precision))
        roc_curves[model] = (fpr, tpr, roc_auc)
        pr_curves[model] = (recall, precision, pr_auc)

        pred_all = (p_all >= threshold).astype(np.uint8)
        tn = int(np.sum((y_all == 0) & (pred_all == 0)))
        fp = int(np.sum((y_all == 0) & (pred_all == 1)))
        fn = int(np.sum((y_all == 1) & (pred_all == 0)))
        tp = int(np.sum((y_all == 1) & (pred_all == 1)))
        confusion_by_model[model] = np.array([[tn, fp], [fn, tp]], dtype=np.int64)

        prec_at_thr = _safe_div(tp, tp + fp)
        rec_at_thr = _safe_div(tp, tp + fn)
        f1_at_thr = _safe_div(2.0 * prec_at_thr * rec_at_thr, prec_at_thr + rec_at_thr)
        bal_acc = 0.5 * (_safe_div(tp, tp + fn) + _safe_div(tn, tn + fp))

        spatial_rows.append(
            {
                "model": model,
                "threshold": threshold,
                "valid_pixels": n_valid,
                "curve_samples": int(y_curve.shape[0]),
                "roc_auc_sample": roc_auc,
                "pr_auc_sample": pr_auc,
                "precision_at_threshold": prec_at_thr,
                "recall_at_threshold": rec_at_thr,
                "f1_at_threshold": f1_at_thr,
                "balanced_accuracy_at_threshold": bal_acc,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )
        print(
            f"[spatial-fig] {model}: n_valid={n_valid}, n_curve={y_curve.shape[0]}, "
            f"ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, thr={threshold:.4f}"
        )

    if not spatial_rows:
        raise RuntimeError("No models produced spatial metrics.")

    # Figure 7: ROC + PR curves.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    ax_roc, ax_pr = axes
    for model in sorted(roc_curves.keys(), key=_model_sort_key):
        fpr, tpr, roc_auc = roc_curves[model]
        recall, precision, pr_auc = pr_curves[model]
        ax_roc.plot(fpr, tpr, lw=2.0, label=f"{model} (AUC={roc_auc:.3f})")
        ax_pr.plot(recall, precision, lw=2.0, label=f"{model} (AUC={pr_auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], "--", lw=1.2, color="#888888")
    ax_roc.set_title("A) ROC Curves")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.grid(alpha=0.25)
    ax_roc.legend(loc="lower right", fontsize=9)

    ax_pr.set_title("B) Precision-Recall Curves")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.grid(alpha=0.25)
    ax_pr.legend(loc="lower left", fontsize=9)
    _save_fig(fig, out_fig_dir / "fig07_roc_pr_curves")

    # Figure 8: confusion matrix heatmaps (row-normalized, count annotated).
    cm_models = sorted(confusion_by_model.keys(), key=_model_sort_key)
    n_cm = len(cm_models)
    ncols = 2 if n_cm > 1 else 1
    nrows = int(math.ceil(n_cm / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.2 * nrows), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()
    for i, model in enumerate(cm_models):
        ax = axes_arr[i]
        cm = confusion_by_model[model].astype(np.float64)
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum != 0.0)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(f"{model} (thr={thresholds.get(model, default_threshold):.3f})")
        ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], labels=["True 0", "True 1"])
        for r in range(2):
            for c in range(2):
                ax.text(
                    c,
                    r,
                    f"{int(cm[r, c]):,}\n{cm_norm[r, c]*100:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black",
                )
    for j in range(n_cm, len(axes_arr)):
        axes_arr[j].axis("off")
    cbar = fig.colorbar(im, ax=axes_arr.tolist(), shrink=0.88)
    cbar.set_label("Row-normalized rate")
    fig.suptitle("Figure 8: Pixel-Level Confusion Heatmaps", fontsize=14)
    _save_fig(fig, out_fig_dir / "fig08_confusion_heatmaps")

    # Choose map models (RF + XGB if present).
    map_models = [m for m in ["rf", "xgb"] if m in prob_paths]
    if not map_models:
        map_models = [sorted(prob_paths.keys(), key=_model_sort_key)[0]]
    if len(map_models) == 1 and len(prob_paths) > 1:
        map_models.append(sorted(prob_paths.keys(), key=_model_sort_key)[1])

    map_hw = int(args.map_size)
    with rio.open(stack_tif) as ds_stack:
        rgb = ds_stack.read(
            [4, 3, 2],  # B04/B03/B02
            out_shape=(3, map_hw, map_hw),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
    rgb = _stretch_rgb(rgb)

    with rio.open(label_tif) as ds_label:
        label_small = ds_label.read(
            1,
            out_shape=(map_hw, map_hw),
            resampling=Resampling.nearest,
        ).astype(np.uint8)
    label_small_valid = np.isin(label_small, [0, 1])
    if np.any(label_small_valid):
        rr, cc = np.where(label_small_valid)
        r0, r1 = int(rr.min()), int(rr.max()) + 1
        c0, c1 = int(cc.min()), int(cc.max()) + 1
    else:
        r0, r1, c0, c1 = 0, map_hw, 0, map_hw

    rgb = rgb[:, r0:r1, c0:c1]
    label_small = label_small[r0:r1, c0:c1]
    label_small_valid = label_small_valid[r0:r1, c0:c1]

    map_prob_small: dict[str, np.ndarray] = {}
    for model in map_models:
        with rio.open(prob_paths[model]) as ds_prob:
            p_small = ds_prob.read(
                1,
                out_shape=(map_hw, map_hw),
                resampling=Resampling.bilinear,
            ).astype(np.float32)
            nodata = ds_prob.nodata
        if nodata is not None:
            p_small[p_small == float(nodata)] = np.nan
        p_small = np.clip(p_small, 0.0, 1.0)
        map_prob_small[model] = p_small[r0:r1, c0:c1]

    prob_vals = []
    for model in map_models:
        cur = map_prob_small[model]
        valid = label_small_valid & np.isfinite(cur)
        if np.any(valid):
            prob_vals.append(cur[valid])
    if prob_vals:
        all_vals = np.concatenate(prob_vals)
        p_lo = float(np.percentile(all_vals, 2.0))
        p_hi = float(np.percentile(all_vals, 98.0))
        if p_hi - p_lo < 0.02:
            mid = float(np.median(all_vals))
            p_lo = max(0.0, mid - 0.015)
            p_hi = min(1.0, mid + 0.015)
    else:
        p_lo, p_hi = 0.0, 1.0

    # Figure 9: imagery + probability maps.
    n_panels = 1 + len(map_models)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 6.0), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()
    axes_arr[0].imshow(np.transpose(rgb, (1, 2, 0)))
    axes_arr[0].set_title("A) Sentinel-2 RGB (B04/B03/B02)")
    axes_arr[0].axis("off")

    for i, model in enumerate(map_models, start=1):
        p_small = map_prob_small[model]
        im = axes_arr[i].imshow(p_small, cmap="cividis", vmin=p_lo, vmax=p_hi)
        th = float(thresholds.get(model, default_threshold))
        finite = np.isfinite(p_small)
        if np.any(finite):
            try:
                axes_arr[i].contour(
                    np.where(finite, p_small, np.nan),
                    levels=[th],
                    colors=["white"],
                    linewidths=0.8,
                    alpha=0.9,
                )
            except Exception:
                pass
        axes_arr[i].set_title(f"{chr(ord('A')+i)}) {model.upper()} prospectivity")
        axes_arr[i].axis("off")
    cbar = fig.colorbar(im, ax=axes_arr.tolist(), shrink=0.88)
    cbar.set_label("Predicted hydrogen probability (contrast-scaled)")
    _save_fig(fig, out_fig_dir / "fig09_prospectivity_maps")

    # Figure 10: pixel-wise error maps.
    err_cmap = ListedColormap(
        [
            "#bdbdbd",  # unlabeled/invalid
            "#2166ac",  # TN
            "#b2182b",  # FP
            "#fdae61",  # FN
            "#1a9850",  # TP
        ]
    )
    err_legend = [
        Patch(facecolor="#bdbdbd", label="Unlabeled/invalid"),
        Patch(facecolor="#2166ac", label="TN"),
        Patch(facecolor="#b2182b", label="FP"),
        Patch(facecolor="#fdae61", label="FN"),
        Patch(facecolor="#1a9850", label="TP"),
    ]

    fig, axes = plt.subplots(1, len(map_models), figsize=(7.0 * len(map_models), 6.0), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()
    for i, model in enumerate(map_models):
        p_small = map_prob_small[model]
        valid = label_small_valid & np.isfinite(p_small)
        pred_small = p_small >= float(thresholds.get(model, default_threshold))

        cat = np.zeros(label_small.shape, dtype=np.uint8)  # 0 = invalid/unlabeled
        cat[valid & (label_small == 0) & (~pred_small)] = 1  # TN
        cat[valid & (label_small == 0) & pred_small] = 2   # FP
        cat[valid & (label_small == 1) & (~pred_small)] = 3  # FN
        cat[valid & (label_small == 1) & pred_small] = 4   # TP

        axes_arr[i].imshow(cat, cmap=err_cmap, vmin=0, vmax=4)
        axes_arr[i].set_title(f"{model.upper()} error map (thr={thresholds.get(model, default_threshold):.3f})")
        axes_arr[i].axis("off")
    fig.legend(handles=err_legend, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    _save_fig(fig, out_fig_dir / "fig10_pixel_error_maps")

    _write_csv(
        out_analysis_dir / "table_spatial_metrics.csv",
        rows=sorted(spatial_rows, key=lambda r: _model_sort_key(str(r["model"]))),
        fieldnames=[
            "model",
            "threshold",
            "valid_pixels",
            "curve_samples",
            "roc_auc_sample",
            "pr_auc_sample",
            "precision_at_threshold",
            "recall_at_threshold",
            "f1_at_threshold",
            "balanced_accuracy_at_threshold",
            "tn",
            "fp",
            "fn",
            "tp",
        ],
    )
    _write_json(
        out_analysis_dir / "spatial_metrics.json",
        {
            "label_tif": str(label_tif),
            "stack_tif": str(stack_tif),
            "pred_dir": str(pred_dir),
            "models": sorted([str(r["model"]) for r in spatial_rows], key=_model_sort_key),
            "map_models": map_models,
            "max_curve_samples": int(args.max_curve_samples),
            "map_size": int(args.map_size),
            "rows": sorted(spatial_rows, key=lambda r: _model_sort_key(str(r["model"]))),
        },
    )

    print(f"[spatial-fig] wrote analysis tables: {out_analysis_dir}")
    print(f"[spatial-fig] wrote figures: {out_fig_dir}")


if __name__ == "__main__":
    main()
