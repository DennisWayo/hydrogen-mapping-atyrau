#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return int(default)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if not xs:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    if q <= 0.0:
        return min(xs)
    if q >= 1.0:
        return max(xs)
    ys = sorted(xs)
    pos = q * (len(ys) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ys[lo]
    w = pos - lo
    return ys[lo] * (1.0 - w) + ys[hi] * w


def _bootstrap_ci(
    xs: list[float],
    reps: int,
    rng: random.Random,
    alpha: float = 0.05,
) -> tuple[float, float]:
    boot = _bootstrap_means(xs, reps=reps, rng=rng)
    if not boot:
        return 0.0, 0.0
    lo = _quantile(boot, alpha / 2.0)
    hi = _quantile(boot, 1.0 - alpha / 2.0)
    return lo, hi


def _bootstrap_means(xs: list[float], reps: int, rng: random.Random) -> list[float]:
    if not xs:
        return []
    n = len(xs)
    boot: list[float] = []
    for _ in range(max(1, reps)):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        boot.append(_mean(sample))
    return boot


def _paired_sign_test_pvalue(deltas: list[float]) -> float:
    positives = sum(1 for d in deltas if d > 0.0)
    negatives = sum(1 for d in deltas if d < 0.0)
    n = positives + negatives
    if n == 0:
        return 1.0
    k = min(positives, negatives)
    tail = 0.0
    for i in range(k + 1):
        tail += math.comb(n, i) * (0.5**n)
    return min(1.0, 2.0 * tail)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _group_rows_by_model(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[str(row["model"])].append(row)
    return dict(out)


def _ordered_models(groups: dict[str, list[dict[str, str]]]) -> list[str]:
    # Stable presentation order: classical first, then qml.
    priority = {
        "rf": 0,
        "sgd": 1,
        "xgb": 2,
        "qml_vqc": 3,
        "qml_qnn": 4,
        "qml_qkernel_svm": 5,
    }
    return sorted(groups.keys(), key=lambda m: (priority.get(m, 99), m))


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0.0 else 0.0


def _display_model_name(model: str) -> str:
    mapping = {
        "rf": "rf",
        "sgd": "sgd",
        "xgb": "xgb",
        "qml_vqc": "qml-vqc",
        "qml_qnn": "qml-qnn",
        "qml_qkernel_svm": "qml-qkernel-svm",
    }
    return mapping.get(model, model)


def _fold_rate(row: dict[str, str], *, num_a: str, num_b: str) -> float:
    a = _to_float(row.get(num_a))
    b = _to_float(row.get(num_b))
    denom = a + b
    return _safe_div(a, denom)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run additional publication-oriented analyses for paper results and emit CSV/JSON tables."
    )
    parser.add_argument(
        "--fold-csv",
        default="paper_runs/runs/south_kazakhstan_region/artifacts/paper_eval_folds.csv",
    )
    parser.add_argument(
        "--metrics-json",
        default="paper_runs/runs/south_kazakhstan_region/artifacts/paper_eval_metrics.json",
    )
    parser.add_argument(
        "--leak-json",
        default="paper_runs/runs/south_kazakhstan_region/artifacts/leak_fit_summary.json",
    )
    parser.add_argument(
        "--out-dir",
        default="paper_runs/runs/south_kazakhstan_region/analysis",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fold_csv = Path(args.fold_csv).resolve()
    metrics_json = Path(args.metrics_json).resolve()
    leak_json = Path(args.leak_json).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = _read_csv(fold_csv)
    metrics_summary = _read_json(metrics_json)
    leak_summary = _read_json(leak_json)
    groups = _group_rows_by_model(fold_rows)
    models = _ordered_models(groups)
    rng = random.Random(int(args.seed))

    core_metrics = [
        "roc_auc",
        "pr_auc",
        "f1",
        "balanced_accuracy",
        "brier",
        "ece_10bin",
        "fit_seconds",
        "predict_seconds",
        "threshold_used",
    ]

    # Table 1: fold statistics with bootstrap CIs.
    summary_rows: list[dict[str, Any]] = []
    for idx, model in enumerate(models, start=1):
        row: dict[str, Any] = {"model": model, "model_idx": idx, "folds": len(groups[model])}
        for metric in core_metrics:
            vals = [_to_float(r.get(metric)) for r in groups[model]]
            row[f"{metric}_mean"] = _mean(vals)
            row[f"{metric}_std"] = _std(vals)
            lo, hi = _bootstrap_ci(vals, reps=int(args.bootstrap_reps), rng=rng)
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi
        summary_rows.append(row)

    summary_fields = ["model", "model_idx", "folds"]
    for metric in core_metrics:
        summary_fields.extend(
            [
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_ci95_low",
                f"{metric}_ci95_high",
            ]
        )
    _write_csv(out_dir / "table_model_metrics_with_ci.csv", summary_rows, summary_fields)

    # Table 2: paired fold deltas vs RF (reviewer-friendly significance proxy).
    ref_model = "rf" if "rf" in groups else models[0]
    ref_by_fold = {int(r["fold"]): r for r in groups[ref_model]}
    pair_rows: list[dict[str, Any]] = []
    for model in models:
        if model == ref_model:
            continue
        cur_by_fold = {int(r["fold"]): r for r in groups[model]}
        common_folds = sorted(set(ref_by_fold.keys()) & set(cur_by_fold.keys()))
        for metric in ("roc_auc", "pr_auc", "balanced_accuracy", "f1"):
            deltas = [
                _to_float(cur_by_fold[f][metric]) - _to_float(ref_by_fold[f][metric]) for f in common_folds
            ]
            boot = _bootstrap_means(deltas, reps=int(args.bootstrap_reps), rng=rng)
            lo = _quantile(boot, 0.025)
            hi = _quantile(boot, 0.975)
            pair_rows.append(
                {
                    "reference_model": ref_model,
                    "model": model,
                    "metric": metric,
                    "n_common_folds": len(common_folds),
                    "delta_mean": _mean(deltas),
                    "delta_std": _std(deltas),
                    "delta_ci95_low": lo,
                    "delta_ci95_high": hi,
                    "delta_prob_positive_bootstrap": _safe_div(
                        sum(1 for x in boot if x > 0.0),
                        len(boot),
                    ),
                    "paired_sign_test_pvalue": _paired_sign_test_pvalue(deltas),
                    "beats_reference_ci_positive": int(lo > 0.0),
                }
            )
    _write_csv(
        out_dir / "table_pairwise_deltas_vs_rf.csv",
        pair_rows,
        [
            "reference_model",
            "model",
            "metric",
            "n_common_folds",
            "delta_mean",
            "delta_std",
            "delta_ci95_low",
            "delta_ci95_high",
            "delta_prob_positive_bootstrap",
            "paired_sign_test_pvalue",
            "beats_reference_ci_positive",
        ],
    )

    # Table 3: operating points (achieved at current threshold strategy).
    op_rows: list[dict[str, Any]] = []
    for idx, model in enumerate(models, start=1):
        sub = groups[model]
        tp = sum(_to_int(r.get("tp")) for r in sub)
        tn = sum(_to_int(r.get("tn")) for r in sub)
        fp = sum(_to_int(r.get("fp")) for r in sub)
        fn = sum(_to_int(r.get("fn")) for r in sub)
        n_eval = tp + tn + fp + fn
        tpr_fold_vals = [_fold_rate(r, num_a="tp", num_b="fn") for r in sub]
        tnr_fold_vals = [_fold_rate(r, num_a="tn", num_b="fp") for r in sub]
        pred_pos_fold_vals = [
            _safe_div(_to_float(r.get("tp")) + _to_float(r.get("fp")), _to_float(r.get("test_samples")))
            for r in sub
        ]
        prevalence_fold_vals = [
            _safe_div(_to_float(r.get("tp")) + _to_float(r.get("fn")), _to_float(r.get("test_samples")))
            for r in sub
        ]
        tpr_lo, tpr_hi = _bootstrap_ci(tpr_fold_vals, reps=int(args.bootstrap_reps), rng=rng)
        tnr_lo, tnr_hi = _bootstrap_ci(tnr_fold_vals, reps=int(args.bootstrap_reps), rng=rng)
        ppr_lo, ppr_hi = _bootstrap_ci(pred_pos_fold_vals, reps=int(args.bootstrap_reps), rng=rng)
        prev_lo, prev_hi = _bootstrap_ci(prevalence_fold_vals, reps=int(args.bootstrap_reps), rng=rng)
        tpr = _safe_div(tp, tp + fn)
        tnr = _safe_div(tn, tn + fp)
        fpr = 1.0 - tnr
        fnr = 1.0 - tpr
        prevalence = _safe_div(tp + fn, n_eval)
        pred_pos_rate = _safe_div(tp + fp, n_eval)
        pred_neg_rate = 1.0 - pred_pos_rate
        precision = _safe_div(tp, tp + fp)
        npv = _safe_div(tn, tn + fn)
        op_rows.append(
            {
                "model": model,
                "model_idx": idx,
                "eval_samples_total": n_eval,
                "tp_total": tp,
                "tn_total": tn,
                "fp_total": fp,
                "fn_total": fn,
                "prevalence": prevalence,
                "pred_pos_rate": pred_pos_rate,
                "pred_neg_rate": pred_neg_rate,
                "tpr_recall": tpr,
                "tnr_specificity": tnr,
                "fpr": fpr,
                "fnr": fnr,
                "precision": precision,
                "npv": npv,
                "specificity_collapse_flag": int(tnr < 0.05),
                "tpr_recall_fold_mean": _mean(tpr_fold_vals),
                "tpr_recall_ci95_low": tpr_lo,
                "tpr_recall_ci95_high": tpr_hi,
                "tnr_specificity_fold_mean": _mean(tnr_fold_vals),
                "tnr_specificity_ci95_low": tnr_lo,
                "tnr_specificity_ci95_high": tnr_hi,
                "pred_pos_rate_fold_mean": _mean(pred_pos_fold_vals),
                "pred_pos_rate_ci95_low": ppr_lo,
                "pred_pos_rate_ci95_high": ppr_hi,
                "prevalence_fold_mean": _mean(prevalence_fold_vals),
                "prevalence_ci95_low": prev_lo,
                "prevalence_ci95_high": prev_hi,
            }
        )
    _write_csv(
        out_dir / "table_operating_points.csv",
        op_rows,
        [
            "model",
            "model_idx",
            "eval_samples_total",
            "tp_total",
            "tn_total",
            "fp_total",
            "fn_total",
            "prevalence",
            "pred_pos_rate",
            "pred_neg_rate",
            "tpr_recall",
            "tnr_specificity",
            "fpr",
            "fnr",
            "precision",
            "npv",
            "specificity_collapse_flag",
            "tpr_recall_fold_mean",
            "tpr_recall_ci95_low",
            "tpr_recall_ci95_high",
            "tnr_specificity_fold_mean",
            "tnr_specificity_ci95_low",
            "tnr_specificity_ci95_high",
            "pred_pos_rate_fold_mean",
            "pred_pos_rate_ci95_low",
            "pred_pos_rate_ci95_high",
            "prevalence_fold_mean",
            "prevalence_ci95_low",
            "prevalence_ci95_high",
        ],
    )

    # Table 4: runtime and pareto flags (maximize ROC, minimize time).
    runtime_rows: list[dict[str, Any]] = []
    for idx, model in enumerate(models, start=1):
        sub = groups[model]
        fit_total = sum(_to_float(r.get("fit_seconds")) for r in sub)
        pred_total = sum(_to_float(r.get("predict_seconds")) for r in sub)
        total = fit_total + pred_total
        train_total = sum(_to_int(r.get("train_samples")) for r in sub)
        test_total = sum(_to_int(r.get("test_samples")) for r in sub)
        roc_vals = [_to_float(r.get("roc_auc")) for r in sub]
        pr_vals = [_to_float(r.get("pr_auc")) for r in sub]
        runtime_rows.append(
            {
                "model": model,
                "model_idx": idx,
                "folds": len(sub),
                "train_samples_total": train_total,
                "test_samples_total": test_total,
                "fit_seconds_total": fit_total,
                "predict_seconds_total": pred_total,
                "total_seconds": total,
                "fit_seconds_per_1k_train": _safe_div(fit_total, _safe_div(train_total, 1000.0)),
                "predict_seconds_per_1k_test": _safe_div(pred_total, _safe_div(test_total, 1000.0)),
                "roc_auc_mean": _mean(roc_vals),
                "pr_auc_mean": _mean(pr_vals),
            }
        )

    # Pareto frontier on (total_seconds minimize, roc_auc maximize).
    for row in runtime_rows:
        dominated = False
        for other in runtime_rows:
            if other["model"] == row["model"]:
                continue
            better_or_equal_time = other["total_seconds"] <= row["total_seconds"]
            better_or_equal_roc = other["roc_auc_mean"] >= row["roc_auc_mean"]
            strictly_better = (other["total_seconds"] < row["total_seconds"]) or (
                other["roc_auc_mean"] > row["roc_auc_mean"]
            )
            if better_or_equal_time and better_or_equal_roc and strictly_better:
                dominated = True
                break
        row["pareto_front_flag"] = int(not dominated)
    _write_csv(
        out_dir / "table_runtime_pareto.csv",
        runtime_rows,
        [
            "model",
            "model_idx",
            "folds",
            "train_samples_total",
            "test_samples_total",
            "fit_seconds_total",
            "predict_seconds_total",
            "total_seconds",
            "fit_seconds_per_1k_train",
            "predict_seconds_per_1k_test",
            "roc_auc_mean",
            "pr_auc_mean",
            "pareto_front_flag",
        ],
    )

    # Table 5: leakage detail for classical diagnostics.
    leak_rows: list[dict[str, Any]] = []
    leak_models = leak_summary.get("models", {})
    model_idx_lookup = {str(r["model"]): int(r["model_idx"]) for r in summary_rows}
    for model in sorted(leak_models.keys()):
        item = leak_models[model]
        leak_rows.append(
            {
                "model": model,
                "model_idx": int(model_idx_lookup.get(model, 0)),
                "test_roc_auc_mean": _to_float(item.get("test_roc_auc_mean")),
                "test_pr_auc_mean": _to_float(item.get("test_pr_auc_mean")),
                "gap_roc_auc_mean": _to_float(item.get("gap_roc_auc_mean")),
                "perm_test_roc_auc_mean": _to_float(item.get("perm_test_roc_auc_mean")),
                "sample_id_overlap_rate_mean": _to_float(item.get("sample_id_overlap_rate_mean")),
                "overfit_flag_rate": _to_float(item.get("overfit_flag_rate")),
                "underfit_flag_rate": _to_float(item.get("underfit_flag_rate")),
                "leakage_flag_rate": _to_float(item.get("leakage_flag_rate")),
                "overfit_suspected": int(bool(item.get("overfit_suspected"))),
                "underfit_suspected": int(bool(item.get("underfit_suspected"))),
                "leakage_suspected": int(bool(item.get("leakage_suspected"))),
            }
        )
    _write_csv(
        out_dir / "table_leakage_extended.csv",
        leak_rows,
        [
            "model",
            "model_idx",
            "test_roc_auc_mean",
            "test_pr_auc_mean",
            "gap_roc_auc_mean",
            "perm_test_roc_auc_mean",
            "sample_id_overlap_rate_mean",
            "overfit_flag_rate",
            "underfit_flag_rate",
            "leakage_flag_rate",
            "overfit_suspected",
            "underfit_suspected",
            "leakage_suspected",
        ],
    )

    _write_csv(
        out_dir / "fig_leakage.csv",
        [
            {
                "model": _display_model_name(str(r["model"])),
                "model_idx": r["model_idx"],
                "perm_test_roc_auc_mean": r["perm_test_roc_auc_mean"],
                "leakage_flag_rate": r["leakage_flag_rate"],
                "gap_roc_auc_mean": r["gap_roc_auc_mean"],
            }
            for r in leak_rows
        ],
        [
            "model",
            "model_idx",
            "perm_test_roc_auc_mean",
            "leakage_flag_rate",
            "gap_roc_auc_mean",
        ],
    )

    # Save small figure-friendly files.
    _write_csv(
        out_dir / "fig_performance_ci.csv",
        [
            {
                "model": _display_model_name(str(r["model"])),
                "model_idx": r["model_idx"],
                "roc_auc_mean": r["roc_auc_mean"],
                "roc_auc_ci95_low": r["roc_auc_ci95_low"],
                "roc_auc_ci95_high": r["roc_auc_ci95_high"],
                "pr_auc_mean": r["pr_auc_mean"],
                "pr_auc_ci95_low": r["pr_auc_ci95_low"],
                "pr_auc_ci95_high": r["pr_auc_ci95_high"],
                "balanced_accuracy_mean": r["balanced_accuracy_mean"],
                "balanced_accuracy_ci95_low": r["balanced_accuracy_ci95_low"],
                "balanced_accuracy_ci95_high": r["balanced_accuracy_ci95_high"],
                "brier_mean": r["brier_mean"],
                "brier_ci95_low": r["brier_ci95_low"],
                "brier_ci95_high": r["brier_ci95_high"],
                "ece_10bin_mean": r["ece_10bin_mean"],
                "ece_10bin_ci95_low": r["ece_10bin_ci95_low"],
                "ece_10bin_ci95_high": r["ece_10bin_ci95_high"],
            }
            for r in summary_rows
        ],
        [
            "model",
            "model_idx",
            "roc_auc_mean",
            "roc_auc_ci95_low",
            "roc_auc_ci95_high",
            "pr_auc_mean",
            "pr_auc_ci95_low",
            "pr_auc_ci95_high",
            "balanced_accuracy_mean",
            "balanced_accuracy_ci95_low",
            "balanced_accuracy_ci95_high",
            "brier_mean",
            "brier_ci95_low",
            "brier_ci95_high",
            "ece_10bin_mean",
            "ece_10bin_ci95_low",
            "ece_10bin_ci95_high",
        ],
    )
    _write_csv(
        out_dir / "fig_operating_points.csv",
        [
            {
                "model": _display_model_name(str(r["model"])),
                "model_idx": r["model_idx"],
                "tpr_recall": r["tpr_recall"],
                "tpr_recall_ci95_low": r["tpr_recall_ci95_low"],
                "tpr_recall_ci95_high": r["tpr_recall_ci95_high"],
                "fpr": r["fpr"],
                "tnr_specificity": r["tnr_specificity"],
                "tnr_specificity_ci95_low": r["tnr_specificity_ci95_low"],
                "tnr_specificity_ci95_high": r["tnr_specificity_ci95_high"],
                "pred_pos_rate": r["pred_pos_rate"],
                "pred_pos_rate_ci95_low": r["pred_pos_rate_ci95_low"],
                "pred_pos_rate_ci95_high": r["pred_pos_rate_ci95_high"],
                "prevalence": r["prevalence"],
                "prevalence_ci95_low": r["prevalence_ci95_low"],
                "prevalence_ci95_high": r["prevalence_ci95_high"],
                "specificity_collapse_flag": r["specificity_collapse_flag"],
            }
            for r in op_rows
        ],
        [
            "model",
            "model_idx",
            "tpr_recall",
            "tpr_recall_ci95_low",
            "tpr_recall_ci95_high",
            "fpr",
            "tnr_specificity",
            "tnr_specificity_ci95_low",
            "tnr_specificity_ci95_high",
            "pred_pos_rate",
            "pred_pos_rate_ci95_low",
            "pred_pos_rate_ci95_high",
            "prevalence",
            "prevalence_ci95_low",
            "prevalence_ci95_high",
            "specificity_collapse_flag",
        ],
    )
    _write_csv(
        out_dir / "fig_runtime_tradeoff.csv",
        [
            {
                "model": _display_model_name(str(r["model"])),
                "model_idx": r["model_idx"],
                "roc_auc_mean": r["roc_auc_mean"],
                "pr_auc_mean": r["pr_auc_mean"],
                "total_seconds": r["total_seconds"],
                "fit_seconds_total": r["fit_seconds_total"],
                "predict_seconds_total": r["predict_seconds_total"],
                "predict_seconds_per_1k_test": r["predict_seconds_per_1k_test"],
                "pareto_front_flag": r["pareto_front_flag"],
            }
            for r in runtime_rows
        ],
        [
            "model",
            "model_idx",
            "roc_auc_mean",
            "pr_auc_mean",
            "total_seconds",
            "fit_seconds_total",
            "predict_seconds_total",
            "predict_seconds_per_1k_test",
            "pareto_front_flag",
        ],
    )
    _write_csv(
        out_dir / "fig_pairwise_deltas.csv",
        [
            {
                "model": _display_model_name(str(r["model"])),
                "model_idx": model_idx_lookup.get(str(r["model"]), 0),
                "metric": str(r["metric"]),
                "delta_mean": r["delta_mean"],
                "delta_ci95_low": r["delta_ci95_low"],
                "delta_ci95_high": r["delta_ci95_high"],
                "delta_prob_positive_bootstrap": r["delta_prob_positive_bootstrap"],
                "paired_sign_test_pvalue": r["paired_sign_test_pvalue"],
            }
            for r in sorted(pair_rows, key=lambda x: (str(x["metric"]), str(x["model"])))
        ],
        [
            "model",
            "model_idx",
            "metric",
            "delta_mean",
            "delta_ci95_low",
            "delta_ci95_high",
            "delta_prob_positive_bootstrap",
            "paired_sign_test_pvalue",
        ],
    )

    # Human-readable synopsis for direct manuscript drafting.
    top_roc_model = max(summary_rows, key=lambda r: r["roc_auc_mean"])["model"] if summary_rows else None
    fastest_model = min(runtime_rows, key=lambda r: r["total_seconds"])["model"] if runtime_rows else None
    qml_status = str(metrics_summary.get("qml", {}).get("status", "unknown"))

    synopsis = {
        "inputs": {
            "fold_csv": str(fold_csv),
            "metrics_json": str(metrics_json),
            "leak_json": str(leak_json),
            "bootstrap_reps": int(args.bootstrap_reps),
            "seed": int(args.seed),
        },
        "high_level": {
            "models": models,
            "qml_status": qml_status,
            "top_roc_model": top_roc_model,
            "fastest_model_total_seconds": fastest_model,
            "global_leakage_risk": leak_summary.get("data_leakage_risk"),
            "global_sample_overlap_rate": leak_summary.get("global_signals", {}).get("sample_id_overlap_rate_mean"),
            "global_max_permutation_auc": leak_summary.get("global_signals", {}).get("max_permutation_auc_mean"),
        },
        "notes": [
            "Operating-point table reflects achieved thresholds from current run; threshold-sweep analysis requires per-sample probabilities.",
            "QML and classical branches used different sample caps in this run; matched-budget reruns are recommended for strict fairness.",
        ],
    }
    with (out_dir / "summary_additional_analysis.json").open("w", encoding="utf-8") as f:
        json.dump(synopsis, f, indent=2)

    print(f"[analysis] wrote outputs to: {out_dir}")
    print(f"[analysis] models: {models}")
    print(f"[analysis] qml_status: {qml_status}")
    print(f"[analysis] top_roc_model: {top_roc_model}")
    print(f"[analysis] fastest_model_total_seconds: {fastest_model}")


if __name__ == "__main__":
    main()
