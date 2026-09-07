from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .utils import configure_cpu_threads, configure_process_parallelism, ensure_parent, save_json, thread_limiter


def run_leak_fit_checks(cfg: dict[str, Any]) -> dict[str, Any]:
    paths = cfg["paths"]
    dcfg = cfg.get("leak_fit_checks", {})
    thread_cfg = configure_cpu_threads(cfg=cfg, section_key="leak_fit_checks", log_prefix="[diag]")
    seed = int(dcfg.get("random_seed", cfg.get("training", {}).get("random_seed", 42)))
    n_splits = int(dcfg.get("n_splits", 5))
    block_size = int(dcfg.get("spatial_block_size", 512))
    max_samples = int(dcfg.get("max_samples", 50_000))
    perm_repeats = int(dcfg.get("permutation_repeats", 1))

    overfit_gap_threshold = float(dcfg.get("overfit_gap_threshold", 0.08))
    underfit_auc_threshold = float(dcfg.get("underfit_auc_threshold", 0.56))
    leakage_auc_threshold = float(dcfg.get("leakage_auc_threshold", 0.55))
    sample_overlap_rate_threshold = float(dcfg.get("sample_overlap_rate_threshold", 0.0001))

    dataset_npz = paths["dataset_npz"]
    artifacts_dir = paths["artifacts_dir"]
    out_json = paths.get("leak_fit_json", artifacts_dir / "leak_fit_summary.json")
    out_csv = paths.get("leak_fit_csv", artifacts_dir / "leak_fit_folds.csv")

    if not dataset_npz.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_npz}")

    data = np.load(dataset_npz)
    x = data["X"].astype(np.float32)
    y = data["y"].astype(np.uint8)
    row = data["row"].astype(np.int32) if "row" in data else None
    col = data["col"].astype(np.int32) if "col" in data else None

    if max_samples > 0 and x.shape[0] > max_samples:
        x, y, row, col = _stratified_subsample(x, y, row, col, max_samples, seed)
        print(f"[diag] using stratified subset: X{x.shape}, y{y.shape}")
    else:
        print(f"[diag] using full dataset: X{x.shape}, y{y.shape}")

    folds, split_strategy = _build_folds(y=y, row=row, col=col, n_splits=n_splits, block_size=block_size, seed=seed)
    print(f"[diag] split strategy: {split_strategy}, folds={len(folds)}")

    requested_models = set(dcfg.get("models", ["sgd", "rf", "xgb"]))
    available_models = _available_classical_models()
    missing_models = sorted(requested_models - available_models)
    if missing_models:
        print(f"[diag] warning: requested models unavailable: {missing_models}")
    model_names = sorted(requested_models & available_models)
    if not model_names:
        raise RuntimeError("No available diagnostic models were selected.")

    sample_overlap_rates = _sample_overlap_rates(x=x, row=row, col=col, folds=folds)
    task_count = len(folds) * len(model_names)
    parallel_cfg = configure_process_parallelism(
        cfg=cfg,
        section_key="leak_fit_checks",
        max_tasks=task_count,
        log_prefix="[diag]",
    )
    worker_n_jobs, worker_max_threads = _resolve_worker_execution_limits(
        thread_cfg=thread_cfg,
        parallel_cfg=parallel_cfg,
    )

    tasks: list[tuple[int, np.ndarray, str]] = []
    for fold_idx, test_mask in enumerate(folds):
        for model_name in model_names:
            tasks.append((fold_idx, test_mask, model_name))

    if parallel_cfg["workers"] > 1 and len(tasks) > 1:
        print(
            f"[diag] running fold diagnostics in parallel: "
            f"workers={parallel_cfg['workers']}, tasks={len(tasks)}"
        )
        rows = Parallel(n_jobs=int(parallel_cfg["workers"]), backend="loky")(
            delayed(_run_diag_fold_model_task)(
                fold_idx=fold_idx,
                test_mask=test_mask,
                x=x,
                y=y,
                model_name=model_name,
                seed=seed,
                split_strategy=split_strategy,
                sample_overlap_rate=float(sample_overlap_rates[fold_idx]),
                perm_repeats=perm_repeats,
                overfit_gap_threshold=overfit_gap_threshold,
                underfit_auc_threshold=underfit_auc_threshold,
                leakage_auc_threshold=leakage_auc_threshold,
                model_n_jobs=worker_n_jobs,
                max_threads=worker_max_threads,
            )
            for (fold_idx, test_mask, model_name) in tasks
        )
    else:
        rows = [
            _run_diag_fold_model_task(
                fold_idx=fold_idx,
                test_mask=test_mask,
                x=x,
                y=y,
                model_name=model_name,
                seed=seed,
                split_strategy=split_strategy,
                sample_overlap_rate=float(sample_overlap_rates[fold_idx]),
                perm_repeats=perm_repeats,
                overfit_gap_threshold=overfit_gap_threshold,
                underfit_auc_threshold=underfit_auc_threshold,
                leakage_auc_threshold=leakage_auc_threshold,
                model_n_jobs=worker_n_jobs,
                max_threads=worker_max_threads,
            )
            for (fold_idx, test_mask, model_name) in tasks
        ]

    fold_df = pd.DataFrame(rows)
    ensure_parent(out_csv)
    fold_df.to_csv(out_csv, index=False)
    print(f"[diag] saved fold diagnostics: {out_csv}")

    summary = _summarize(
        fold_df=fold_df,
        split_strategy=split_strategy,
        n_splits=len(folds),
        dataset_samples=int(x.shape[0]),
        overfit_gap_threshold=overfit_gap_threshold,
        underfit_auc_threshold=underfit_auc_threshold,
        leakage_auc_threshold=leakage_auc_threshold,
        sample_overlap_rate_threshold=sample_overlap_rate_threshold,
        missing_models=missing_models,
    )
    save_json(out_json, summary)
    print(f"[diag] saved summary diagnostics: {out_json}")
    return summary


def _build_folds(
    y: np.ndarray,
    row: np.ndarray | None,
    col: np.ndarray | None,
    n_splits: int,
    block_size: int,
    seed: int,
) -> tuple[list[np.ndarray], str]:
    if len(np.unique(y)) < 2:
        return [_single_split_mask(y.shape[0], seed=seed, test_ratio=0.2)], "single_class_random"
    if _minority_count(y) < 2:
        return [_single_split_mask(y.shape[0], seed=seed, test_ratio=0.2)], "minority_too_small_random"

    if row is not None and col is not None and row.shape == y.shape and col.shape == y.shape:
        return _spatial_block_folds(y=y, row=row, col=col, n_splits=n_splits, block_size=block_size, seed=seed), "spatial_block"

    effective_splits = _safe_n_splits(y, requested=n_splits)
    skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=seed)
    masks: list[np.ndarray] = []
    for _, test_idx in skf.split(np.zeros_like(y), y):
        m = np.zeros(y.shape[0], dtype=bool)
        m[test_idx] = True
        masks.append(m)
    return masks, "stratified_random"


def _spatial_block_folds(
    y: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    n_splits: int,
    block_size: int,
    seed: int,
) -> list[np.ndarray]:
    block_r = row // max(block_size, 1)
    block_c = col // max(block_size, 1)
    block_id = block_r.astype(np.int64) * 10_000_000 + block_c.astype(np.int64)
    unique_blocks = np.unique(block_id)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_blocks)
    block_groups = np.array_split(unique_blocks, n_splits)

    masks: list[np.ndarray] = []
    for bg in block_groups:
        mask = np.isin(block_id, bg)
        if np.any(mask):
            masks.append(mask)
    if len(masks) < 2:
        effective_splits = _safe_n_splits(y, requested=n_splits)
        skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=seed)
        masks = []
        for _, test_idx in skf.split(np.zeros_like(y), y):
            m = np.zeros(y.shape[0], dtype=bool)
            m[test_idx] = True
            masks.append(m)
    return masks


def _available_classical_models() -> set[str]:
    models = {"sgd", "rf"}
    try:
        import xgboost as _  # noqa: F401

        models.add("xgb")
    except Exception as exc:
        print(f"[diag] warning: xgboost unavailable, skipping xgb ({exc})")
    return models


def _build_classical_model(model_name: str, seed: int, n_jobs: int) -> Any:
    if model_name == "sgd":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", SGDClassifier(loss="log_loss", random_state=seed, max_iter=2000, class_weight="balanced")),
            ]
        )
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=n_jobs,
            random_state=seed,
            class_weight="balanced_subsample",
        )
    if model_name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=n_jobs,
        )
    raise ValueError(f"Unsupported diagnostics model: {model_name}")


def _resolve_worker_execution_limits(
    thread_cfg: dict[str, int],
    parallel_cfg: dict[str, int],
) -> tuple[int, int]:
    workers = int(parallel_cfg.get("workers", 1))
    threads_per_worker = int(parallel_cfg.get("threads_per_worker", 0))
    if workers <= 1:
        return int(thread_cfg.get("n_jobs", -1)), int(thread_cfg.get("max_threads", 0))
    if threads_per_worker > 0:
        return threads_per_worker, threads_per_worker
    return 1, 1


def _run_diag_fold_model_task(
    fold_idx: int,
    test_mask: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    model_name: str,
    seed: int,
    split_strategy: str,
    sample_overlap_rate: float,
    perm_repeats: int,
    overfit_gap_threshold: float,
    underfit_auc_threshold: float,
    leakage_auc_threshold: float,
    model_n_jobs: int,
    max_threads: int,
) -> dict[str, Any]:
    train_mask = ~test_mask
    x_train, y_train = x[train_mask], y[train_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    model = _build_classical_model(model_name=model_name, seed=seed, n_jobs=model_n_jobs)
    with thread_limiter(max_threads):
        model.fit(x_train, y_train)
        train_prob = _predict_positive_probability(model, x_train)
        test_prob = _predict_positive_probability(model, x_test)

    train_roc = _safe_roc_auc(y_train, train_prob)
    test_roc = _safe_roc_auc(y_test, test_prob)
    train_pr = _safe_pr_auc(y_train, train_prob)
    test_pr = _safe_pr_auc(y_test, test_prob)
    train_brier = _safe_brier(y_train, train_prob)
    test_brier = _safe_brier(y_test, test_prob)
    gap_roc = float(train_roc - test_roc)
    gap_pr = float(train_pr - test_pr)

    rng = np.random.default_rng(seed + fold_idx * 100 + _model_seed_offset(model_name))
    perm_aucs: list[float] = []
    for _ in range(max(1, perm_repeats)):
        y_perm = rng.permutation(y_train)
        perm_model = _build_classical_model(model_name=model_name, seed=seed, n_jobs=model_n_jobs)
        with thread_limiter(max_threads):
            perm_model.fit(x_train, y_perm)
            perm_prob = _predict_positive_probability(perm_model, x_test)
        perm_aucs.append(_safe_roc_auc(y_test, perm_prob))
    perm_test_roc = float(np.mean(perm_aucs))

    return {
        "model": model_name,
        "fold": int(fold_idx),
        "split": split_strategy,
        "train_samples": int(x_train.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "sample_id_overlap_rate": float(sample_overlap_rate),
        "train_roc_auc": train_roc,
        "test_roc_auc": test_roc,
        "train_pr_auc": train_pr,
        "test_pr_auc": test_pr,
        "train_brier": train_brier,
        "test_brier": test_brier,
        "gap_roc_auc": gap_roc,
        "gap_pr_auc": gap_pr,
        "perm_test_roc_auc": perm_test_roc,
        "overfit_flag": bool(gap_roc > overfit_gap_threshold),
        "underfit_flag": bool((train_roc < underfit_auc_threshold) and (test_roc < underfit_auc_threshold)),
        "leakage_flag": bool(perm_test_roc > leakage_auc_threshold),
    }


def _model_seed_offset(model_name: str) -> int:
    if model_name == "sgd":
        return 11
    if model_name == "rf":
        return 29
    if model_name == "xgb":
        return 47
    return 97


def _predict_positive_probability(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(x)
        if prob.ndim == 2 and prob.shape[1] >= 2:
            return prob[:, 1].astype(np.float64)
        return prob.reshape(-1).astype(np.float64)
    if hasattr(model, "decision_function"):
        score = model.decision_function(x)
        return (1.0 / (1.0 + np.exp(-score))).astype(np.float64)
    pred = model.predict(x)
    return pred.astype(np.float64)


def _safe_roc_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    p = np.clip(prob.astype(np.float64), 1e-6, 1.0 - 1e-6)
    return float(roc_auc_score(y_true, p))


def _safe_pr_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    p = np.clip(prob.astype(np.float64), 1e-6, 1.0 - 1e-6)
    return float(average_precision_score(y_true, p))


def _safe_brier(y_true: np.ndarray, prob: np.ndarray) -> float:
    p = np.clip(prob.astype(np.float64), 1e-6, 1.0 - 1e-6)
    return float(brier_score_loss(y_true, p))


def _sample_overlap_rates(
    x: np.ndarray,
    row: np.ndarray | None,
    col: np.ndarray | None,
    folds: list[np.ndarray],
) -> list[float]:
    if row is not None and col is not None and row.shape[0] == x.shape[0] and col.shape[0] == x.shape[0]:
        sample_ids = row.astype(np.int64) * 10_000_000 + col.astype(np.int64)
    else:
        sample_ids = np.arange(x.shape[0], dtype=np.int64)

    rates: list[float] = []
    for test_mask in folds:
        train_mask = ~test_mask
        train_ids = sample_ids[train_mask]
        test_ids = sample_ids[test_mask]
        overlap = np.intersect1d(train_ids, test_ids).size
        rate = float(overlap / max(test_ids.size, 1))
        rates.append(rate)
    return rates


def _stratified_subsample(
    x: np.ndarray,
    y: np.ndarray,
    row: np.ndarray | None,
    col: np.ndarray | None,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    if x.shape[0] <= max_samples:
        return x, y, row, col
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    keep_pos = min(len(pos), max(1, int(max_samples * (len(pos) / len(y)))))
    keep_neg = max_samples - keep_pos
    idx = np.concatenate(
        [
            rng.choice(pos, size=keep_pos, replace=False) if keep_pos > 0 and len(pos) > 0 else np.array([], dtype=int),
            rng.choice(neg, size=min(len(neg), keep_neg), replace=False) if keep_neg > 0 and len(neg) > 0 else np.array([], dtype=int),
        ]
    )
    if idx.size == 0:
        idx = rng.choice(np.arange(x.shape[0]), size=max_samples, replace=False)
    rng.shuffle(idx)
    row_sub = row[idx] if row is not None else None
    col_sub = col[idx] if col is not None else None
    return x[idx], y[idx], row_sub, col_sub


def _summarize(
    fold_df: pd.DataFrame,
    split_strategy: str,
    n_splits: int,
    dataset_samples: int,
    overfit_gap_threshold: float,
    underfit_auc_threshold: float,
    leakage_auc_threshold: float,
    sample_overlap_rate_threshold: float,
    missing_models: list[str],
) -> dict[str, Any]:
    if fold_df.empty:
        return {
            "status": "no_rows",
            "split_strategy": split_strategy,
            "n_splits": n_splits,
            "dataset_samples": dataset_samples,
            "missing_models": missing_models,
        }

    metric_cols = [
        "train_roc_auc",
        "test_roc_auc",
        "train_pr_auc",
        "test_pr_auc",
        "train_brier",
        "test_brier",
        "gap_roc_auc",
        "gap_pr_auc",
        "perm_test_roc_auc",
        "sample_id_overlap_rate",
    ]
    by_model: dict[str, Any] = {}
    for model_name, grp in fold_df.groupby("model"):
        stats: dict[str, Any] = {"folds": int(grp.shape[0])}
        for c in metric_cols:
            if c in grp:
                stats[f"{c}_mean"] = float(grp[c].mean())
                stats[f"{c}_std"] = float(grp[c].std(ddof=0))
        stats["overfit_flag_rate"] = float(grp["overfit_flag"].mean())
        stats["underfit_flag_rate"] = float(grp["underfit_flag"].mean())
        stats["leakage_flag_rate"] = float(grp["leakage_flag"].mean())
        stats["overfit_suspected"] = bool(stats["gap_roc_auc_mean"] > overfit_gap_threshold)
        stats["underfit_suspected"] = bool(
            (stats["train_roc_auc_mean"] < underfit_auc_threshold) and (stats["test_roc_auc_mean"] < underfit_auc_threshold)
        )
        stats["leakage_suspected"] = bool(stats["perm_test_roc_auc_mean"] > leakage_auc_threshold)
        by_model[str(model_name)] = stats

    best_model = max(
        by_model.items(),
        key=lambda kv: (kv[1].get("test_roc_auc_mean", 0.0), kv[1].get("test_pr_auc_mean", 0.0)),
    )[0]
    best = by_model[best_model]

    max_perm = max(v.get("perm_test_roc_auc_mean", 0.0) for v in by_model.values())
    sample_overlap_mean = float(fold_df["sample_id_overlap_rate"].mean())
    high_perm = max(leakage_auc_threshold + 0.05, 0.60)
    high_overlap = max(sample_overlap_rate_threshold * 10.0, 0.001)
    if (max_perm >= high_perm) or (sample_overlap_mean >= high_overlap):
        leak_risk = "high"
    elif (max_perm >= leakage_auc_threshold) or (sample_overlap_mean >= sample_overlap_rate_threshold):
        leak_risk = "medium"
    else:
        leak_risk = "low"

    gap = float(best.get("gap_roc_auc_mean", 0.0))
    train_auc = float(best.get("train_roc_auc_mean", 0.0))
    test_auc = float(best.get("test_roc_auc_mean", 0.0))
    if (train_auc < underfit_auc_threshold) and (test_auc < underfit_auc_threshold):
        fit_regime = "underfitting"
    elif gap > overfit_gap_threshold:
        fit_regime = "overfitting"
    else:
        fit_regime = "balanced_or_slight_overfit"

    return {
        "status": "ok",
        "split_strategy": split_strategy,
        "n_splits": n_splits,
        "dataset_samples": dataset_samples,
        "best_model_by_test_roc_auc": best_model,
        "fit_regime": fit_regime,
        "data_leakage_risk": leak_risk,
        "thresholds": {
            "overfit_gap_threshold": overfit_gap_threshold,
            "underfit_auc_threshold": underfit_auc_threshold,
            "leakage_auc_threshold": leakage_auc_threshold,
            "sample_overlap_rate_threshold": sample_overlap_rate_threshold,
        },
        "global_signals": {
            "sample_id_overlap_rate_mean": sample_overlap_mean,
            "max_permutation_auc_mean": max_perm,
        },
        "missing_models": missing_models,
        "models": by_model,
    }


def _single_split_mask(n_samples: int, seed: int, test_ratio: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.arange(n_samples)
    rng.shuffle(idx)
    n_test = max(1, int(n_samples * test_ratio))
    mask = np.zeros(n_samples, dtype=bool)
    mask[idx[:n_test]] = True
    return mask


def _safe_n_splits(y: np.ndarray, requested: int) -> int:
    minority = _minority_count(y)
    return max(2, min(int(requested), max(2, minority)))


def _minority_count(y: np.ndarray) -> int:
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        return 0
    return int(counts.min())
