from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from .utils import (
    configure_cpu_threads,
    configure_process_parallelism,
    ensure_parent,
    load_json,
    save_json,
    thread_limiter,
)


def run_unified_paper_eval(cfg: dict[str, Any]) -> dict[str, Any]:
    paths = cfg["paths"]
    ecfg = cfg.get("paper_eval", {})
    thread_cfg = configure_cpu_threads(cfg=cfg, section_key="paper_eval", log_prefix="[paper-eval]")
    seed = int(ecfg.get("random_seed", cfg.get("training", {}).get("random_seed", 42)))
    n_splits = int(ecfg.get("n_splits", 5))
    threshold = float(ecfg.get("threshold", cfg.get("training", {}).get("threshold", 0.8)))
    threshold_mode = str(ecfg.get("threshold_mode", "fixed")).strip().lower()
    if threshold_mode not in {"fixed", "f1_opt"}:
        print(f"[paper-eval] warning: unknown threshold_mode='{threshold_mode}', using 'fixed'")
        threshold_mode = "fixed"
    block_size = int(ecfg.get("spatial_block_size", 512))
    max_samples = int(ecfg.get("max_samples", 50_000))
    label_mode = str(ecfg.get("label_mode", "supervised")).strip().lower()
    pu_cfg = ecfg.get("pu", {})
    ranking_k_fracs = _normalize_ranking_fracs(ecfg.get("ranking_k_fracs", [0.001, 0.005, 0.01]))

    dataset_npz = paths["dataset_npz"]
    artifacts_dir = paths["artifacts_dir"]
    eval_json = paths.get("paper_eval_json", artifacts_dir / "paper_eval_metrics.json")
    eval_csv = paths.get("paper_eval_csv", artifacts_dir / "paper_eval_folds.csv")

    if not dataset_npz.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_npz}")

    data = np.load(dataset_npz)
    x = data["X"].astype(np.float32)
    y = data["y"].astype(np.uint8)
    row = data["row"].astype(np.int32) if "row" in data else None
    col = data["col"].astype(np.int32) if "col" in data else None
    feature_names = _load_feature_names(paths["feature_names_json"], x.shape[1])

    if max_samples > 0 and x.shape[0] > max_samples:
        x, y, row, col = _stratified_subsample(x, y, row, col, max_samples, seed)
        print(f"[paper-eval] using stratified subset: X{x.shape}, y{y.shape}")
    else:
        print(f"[paper-eval] using full dataset: X{x.shape}, y{y.shape}")

    folds, split_strategy = _build_folds(y=y, row=row, col=col, n_splits=n_splits, block_size=block_size, seed=seed)
    print(
        f"[paper-eval] split strategy: {split_strategy}, folds={len(folds)}, "
        f"threshold_mode={threshold_mode}, fixed_threshold={threshold}, label_mode={label_mode}"
    )

    fold_rows: list[dict[str, Any]] = []
    available_classical = _available_classical_models()
    requested_classical = set(ecfg.get("classical_models", sorted(available_classical)))
    missing_classical = requested_classical - available_classical
    if missing_classical:
        print(f"[paper-eval] warning: requested classical models unavailable: {sorted(missing_classical)}")
    model_filter = sorted(requested_classical & available_classical)
    classical_task_count = len(folds) * len(model_filter)
    if classical_task_count > 0:
        parallel_cfg = configure_process_parallelism(
            cfg=cfg,
            section_key="paper_eval",
            max_tasks=classical_task_count,
            log_prefix="[paper-eval]",
        )
        worker_n_jobs, worker_max_threads = _resolve_worker_execution_limits(
            thread_cfg=thread_cfg,
            parallel_cfg=parallel_cfg,
        )
        tasks: list[tuple[int, np.ndarray, str]] = []
        for fold_idx, test_mask in enumerate(folds):
            for model_name in model_filter:
                tasks.append((fold_idx, test_mask, model_name))

        if parallel_cfg["workers"] > 1 and len(tasks) > 1:
            print(
                f"[paper-eval] running classical folds in parallel: "
                f"workers={parallel_cfg['workers']}, tasks={len(tasks)}"
            )
            classical_rows = Parallel(n_jobs=int(parallel_cfg["workers"]), backend="loky")(
                delayed(_run_classical_fold_model_task)(
                    fold_idx=fold_idx,
                    test_mask=test_mask,
                    x=x,
                    y=y,
                    model_name=model_name,
                    seed=seed,
                    label_mode=label_mode,
                    pu_cfg=pu_cfg,
                    threshold_mode=threshold_mode,
                    threshold=threshold,
                    ranking_k_fracs=ranking_k_fracs,
                    split_strategy=split_strategy,
                    model_n_jobs=worker_n_jobs,
                    max_threads=worker_max_threads,
                )
                for (fold_idx, test_mask, model_name) in tasks
            )
        else:
            classical_rows = [
                _run_classical_fold_model_task(
                    fold_idx=fold_idx,
                    test_mask=test_mask,
                    x=x,
                    y=y,
                    model_name=model_name,
                    seed=seed,
                    label_mode=label_mode,
                    pu_cfg=pu_cfg,
                    threshold_mode=threshold_mode,
                    threshold=threshold,
                    ranking_k_fracs=ranking_k_fracs,
                    split_strategy=split_strategy,
                    model_n_jobs=worker_n_jobs,
                    max_threads=worker_max_threads,
                )
                for (fold_idx, test_mask, model_name) in tasks
            ]
        fold_rows.extend(classical_rows)

    qml_cfg = ecfg.get("qml", {})
    if bool(qml_cfg.get("enabled", True)):
        qml_rows, qml_meta = _run_qml_eval(
            x=x,
            y=y,
            folds=folds,
            threshold=threshold,
            threshold_mode=threshold_mode,
            ranking_k_fracs=ranking_k_fracs,
            seed=seed,
            qml_cfg=qml_cfg,
            split_strategy=split_strategy,
        )
        fold_rows.extend(qml_rows)
    else:
        qml_meta = {"enabled": False, "status": "disabled_in_config"}

    fold_df = pd.DataFrame(fold_rows)
    ensure_parent(eval_csv)
    fold_df.to_csv(eval_csv, index=False)
    print(f"[paper-eval] saved fold metrics: {eval_csv}")

    summary = _summarize_fold_metrics(
        fold_df=fold_df,
        dataset_samples=int(x.shape[0]),
        feature_count=int(x.shape[1]),
        feature_names=feature_names,
        split_strategy=split_strategy,
        block_size=block_size,
        n_splits=len(folds),
        threshold_mode=threshold_mode,
        fixed_threshold=threshold,
        label_mode=label_mode,
        ranking_k_fracs=ranking_k_fracs,
        qml_meta=qml_meta,
    )
    save_json(eval_json, summary)
    print(f"[paper-eval] saved summary metrics: {eval_json}")
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
        print(f"[paper-eval] warning: xgboost unavailable, skipping xgb ({exc})")
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
    raise ValueError(f"Unsupported classical model: {model_name}")


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
    # With auto thread settings, force one thread per worker to avoid heavy oversubscription.
    return 1, 1


def _run_classical_fold_model_task(
    fold_idx: int,
    test_mask: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    model_name: str,
    seed: int,
    label_mode: str,
    pu_cfg: dict[str, Any],
    threshold_mode: str,
    threshold: float,
    ranking_k_fracs: list[float],
    split_strategy: str,
    model_n_jobs: int,
    max_threads: int,
) -> dict[str, Any]:
    train_mask = ~test_mask
    x_train, y_train = x[train_mask], y[train_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    model = _build_classical_model(model_name=model_name, seed=seed, n_jobs=model_n_jobs)
    with thread_limiter(max_threads):
        fit_start = time.perf_counter()
        sw = _build_sample_weight(y_train, label_mode=label_mode, pu_cfg=pu_cfg)
        _fit_model(model, x_train, y_train, sw)
        fit_seconds = time.perf_counter() - fit_start

        train_prob = _predict_positive_probability(model, x_train)
        fold_threshold = _resolve_eval_threshold(
            y_train=y_train,
            train_prob=train_prob,
            threshold_mode=threshold_mode,
            fixed_threshold=threshold,
        )

        pred_start = time.perf_counter()
        prob = _predict_positive_probability(model, x_test)
        predict_seconds = time.perf_counter() - pred_start

    metrics = _calc_metrics(
        y_true=y_test,
        prob=prob,
        threshold=fold_threshold,
        ranking_k_fracs=ranking_k_fracs,
    )
    return {
        "model": model_name,
        "fold": int(fold_idx),
        "split": split_strategy,
        "train_samples": int(x_train.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "threshold_used": float(fold_threshold),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        **metrics,
    }


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


def _fit_model(model: Any, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None) -> None:
    if sample_weight is None:
        model.fit(x, y)
        return
    try:
        if isinstance(model, Pipeline) and model.steps:
            last_step = model.steps[-1][0]
            model.fit(x, y, **{f"{last_step}__sample_weight": sample_weight})
            return
        model.fit(x, y, sample_weight=sample_weight)
    except Exception:
        model.fit(x, y)


def _build_sample_weight(y: np.ndarray, label_mode: str, pu_cfg: dict[str, Any]) -> np.ndarray | None:
    if label_mode not in {"pu", "pu_weighted"}:
        return None
    pos_w = float(pu_cfg.get("positive_weight", 1.0))
    unl_w = float(pu_cfg.get("unlabeled_weight", 0.2))
    return np.where(y == 1, pos_w, unl_w).astype(np.float32)


def _normalize_ranking_fracs(raw: Any) -> list[float]:
    if isinstance(raw, (int, float)):
        raw = [raw]
    if not isinstance(raw, list):
        return [0.001, 0.005, 0.01]
    out: list[float] = []
    for item in raw:
        try:
            v = float(item)
        except Exception:
            continue
        if 0.0 < v <= 1.0:
            out.append(v)
    return sorted(set(out)) or [0.001, 0.005, 0.01]


def _calc_metrics(
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    ranking_k_fracs: list[float] | None = None,
) -> dict[str, Any]:
    prob = np.clip(prob.astype(np.float64), 1e-6, 1.0 - 1e-6)
    pred = (prob >= threshold).astype(np.uint8)
    has_both = len(np.unique(y_true)) > 1
    roc_auc = float(roc_auc_score(y_true, prob)) if has_both else 0.0
    pr_auc = float(average_precision_score(y_true, prob)) if has_both else 0.0
    precision = float(precision_score(y_true, pred, zero_division=0))
    recall = float(recall_score(y_true, pred, zero_division=0))
    f1 = float(f1_score(y_true, pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, pred))
    brier = float(brier_score_loss(y_true, prob))
    ece = float(_expected_calibration_error(y_true, prob, n_bins=10))
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    out = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "brier": brier,
        "ece_10bin": ece,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
    out.update(_ranking_metrics(y_true=y_true, prob=prob, k_fracs=ranking_k_fracs or []))
    return out


def _ranking_metrics(y_true: np.ndarray, prob: np.ndarray, k_fracs: list[float]) -> dict[str, float]:
    out: dict[str, float] = {}
    n = int(y_true.shape[0])
    if n == 0:
        return out
    pos_total = int((y_true == 1).sum())
    prevalence = float(pos_total / n) if n > 0 else 0.0
    order = np.argsort(-prob.astype(np.float64))

    for frac in k_fracs:
        k = max(1, int(round(frac * n)))
        top_idx = order[:k]
        tp = int((y_true[top_idx] == 1).sum())
        precision_k = float(tp / k)
        recall_k = float(tp / max(pos_total, 1))
        lift_k = float(precision_k / prevalence) if prevalence > 0 else 0.0
        top_bp = int(round(frac * 10000))
        out[f"precision_at_top{top_bp}bp"] = precision_k
        out[f"recall_at_top{top_bp}bp"] = recall_k
        out[f"lift_at_top{top_bp}bp"] = lift_k
    return out


def _expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(prob, bins) - 1
    ece = 0.0
    n = max(prob.shape[0], 1)
    for i in range(n_bins):
        mask = bin_idx == i
        if not np.any(mask):
            continue
        acc = np.mean(y_true[mask])
        conf = np.mean(prob[mask])
        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def _run_qml_eval(
    x: np.ndarray,
    y: np.ndarray,
    folds: list[np.ndarray],
    threshold: float,
    threshold_mode: str,
    ranking_k_fracs: list[float],
    seed: int,
    qml_cfg: dict[str, Any],
    split_strategy: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import pennylane as qml
    except Exception as exc:
        print(f"[paper-eval] warning: PennyLane unavailable, skipping QML ({exc})")
        return [], {"enabled": True, "status": "skipped_no_pennylane", "reason": str(exc)}

    n_qubits = int(qml_cfg.get("n_qubits", 6))
    n_layers = int(qml_cfg.get("n_layers", 2))
    epochs = int(qml_cfg.get("epochs", 15))
    batch_size = int(qml_cfg.get("batch_size", 32))
    learning_rate = float(qml_cfg.get("learning_rate", 0.05))
    max_train_samples = int(qml_cfg.get("max_train_samples", 1200))
    max_test_samples = int(qml_cfg.get("max_test_samples", 4000))
    max_train_samples_kernel = int(qml_cfg.get("max_train_samples_kernel", min(max_train_samples, 300)))
    max_test_samples_kernel = int(qml_cfg.get("max_test_samples_kernel", min(max_test_samples, 1200)))
    qml_models = [str(m).strip().lower() for m in qml_cfg.get("models", ["qml_vqc"]) if str(m).strip()]
    if not qml_models:
        qml_models = ["qml_vqc"]
    supported_models = {"qml_vqc", "qml_qnn", "qml_qkernel_svm"}
    qml_models = [m for m in qml_models if m in supported_models]
    if not qml_models:
        print("[paper-eval] warning: qml.models contained no supported entries, using qml_vqc")
        qml_models = ["qml_vqc"]
    try:
        qml_device_name = _resolve_qml_device_name(qml=qml, requested=qml_cfg.get("device", "auto"))
    except Exception as exc:
        print(f"[paper-eval] warning: no usable PennyLane device, skipping QML ({exc})")
        return [], {"enabled": True, "status": "skipped_no_qml_device", "reason": str(exc)}
    qml_shots = _parse_optional_positive_int(qml_cfg.get("shots"))
    kernel_probability = bool(qml_cfg.get("kernel_probability", False))
    kernel_cache_mb = float(qml_cfg.get("kernel_cache_mb", 1024.0))
    qml_shots_label = "analytic" if qml_shots is None else str(qml_shots)
    print(f"[paper-eval] qml backend: {qml_device_name}, shots={qml_shots_label}")

    rows: list[dict[str, Any]] = []
    fold_errors: list[str] = []
    for fold_idx, test_mask in enumerate(folds):
        train_mask = ~test_mask
        x_train, y_train = x[train_mask], y[train_mask]
        x_test, y_test = x[test_mask], y[test_mask]

        if len(np.unique(y_train)) < 2:
            print(f"[paper-eval] warning: fold {fold_idx} has single-class train set, skipping qml_vqc")
            continue
        if len(np.unique(y_test)) < 2:
            print(f"[paper-eval] warning: fold {fold_idx} has single-class test set, skipping qml_vqc")
            continue

        x_train, y_train = _stratified_subsample_only_xy(x_train, y_train, max_train_samples, seed + fold_idx)
        x_test, y_test = _stratified_subsample_only_xy(x_test, y_test, max_test_samples, seed + 1000 + fold_idx)

        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train)
        x_test_s = scaler.transform(x_test)

        comp = min(n_qubits, x_train_s.shape[1], x_train_s.shape[0])
        comp = max(1, comp)
        pca = PCA(n_components=comp, random_state=seed)
        x_train_pca = pca.fit_transform(x_train_s)
        x_test_pca = pca.transform(x_test_s)

        amp = MinMaxScaler(feature_range=(-np.pi, np.pi))
        x_train_q = amp.fit_transform(x_train_pca).astype(np.float64)
        x_test_q = amp.transform(x_test_pca).astype(np.float64)

        for qml_model in qml_models:
            try:
                x_train_model = x_train_q
                y_train_model = y_train
                x_test_model = x_test_q
                y_test_model = y_test

                if qml_model == "qml_qkernel_svm":
                    x_train_model, y_train_model = _stratified_subsample_only_xy(
                        x_train_q, y_train, max_train_samples_kernel, seed + 2000 + fold_idx
                    )
                    x_test_model, y_test_model = _stratified_subsample_only_xy(
                        x_test_q, y_test, max_test_samples_kernel, seed + 3000 + fold_idx
                    )

                if len(np.unique(y_train_model)) < 2:
                    print(
                        f"[paper-eval] warning: fold {fold_idx}, model {qml_model} has single-class train set after subsampling, skipping"
                    )
                    continue
                if len(np.unique(y_test_model)) < 2:
                    print(
                        f"[paper-eval] warning: fold {fold_idx}, model {qml_model} has single-class test set after subsampling, skipping"
                    )
                    continue

                fit_start = time.perf_counter()
                if qml_model == "qml_qnn":
                    pred_fn = _train_qml_qnn(
                        x_train=x_train_model,
                        y_train=y_train_model.astype(np.float64),
                        n_qubits=comp,
                        n_layers=n_layers,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        seed=seed + fold_idx,
                        device_name=qml_device_name,
                        shots=qml_shots,
                    )
                elif qml_model == "qml_qkernel_svm":
                    pred_fn = _train_qml_qkernel_svm(
                        x_train=x_train_model,
                        y_train=y_train_model.astype(np.uint8),
                        n_qubits=comp,
                        n_layers=n_layers,
                        seed=seed + fold_idx,
                        device_name=qml_device_name,
                        shots=qml_shots,
                        use_probability=kernel_probability,
                        cache_size_mb=kernel_cache_mb,
                    )
                else:
                    pred_fn = _train_qml_vqc(
                        x_train=x_train_model,
                        y_train=y_train_model.astype(np.float64),
                        n_qubits=comp,
                        n_layers=n_layers,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        seed=seed + fold_idx,
                        device_name=qml_device_name,
                        shots=qml_shots,
                    )
                fit_seconds = time.perf_counter() - fit_start

                train_prob = pred_fn(x_train_model)
                fold_threshold = _resolve_eval_threshold(
                    y_train=y_train_model,
                    train_prob=train_prob,
                    threshold_mode=threshold_mode,
                    fixed_threshold=threshold,
                )

                pred_start = time.perf_counter()
                prob = pred_fn(x_test_model)
                predict_seconds = time.perf_counter() - pred_start

                metrics = _calc_metrics(
                    y_true=y_test_model,
                    prob=prob,
                    threshold=fold_threshold,
                    ranking_k_fracs=ranking_k_fracs,
                )
                rows.append(
                    {
                        "model": qml_model,
                        "fold": fold_idx,
                        "split": split_strategy,
                        "train_samples": int(x_train_model.shape[0]),
                        "test_samples": int(x_test_model.shape[0]),
                        "threshold_used": float(fold_threshold),
                        "fit_seconds": fit_seconds,
                        "predict_seconds": predict_seconds,
                        **metrics,
                    }
                )
            except Exception as exc:
                msg = f"fold={fold_idx}, model={qml_model}, error={exc}"
                print(f"[paper-eval] warning: {msg}")
                fold_errors.append(msg)

    status = "completed" if rows else "failed"
    if fold_errors and rows:
        status = "completed_with_errors"
    return rows, {"enabled": True, "status": status, "models": qml_models, "errors": fold_errors}


def _parse_optional_positive_int(raw_value: Any) -> int | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str) and raw_value.strip().lower() in {"", "none", "null", "analytic"}:
        return None
    try:
        out = int(raw_value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _resolve_qml_device_name(qml: Any, requested: Any) -> str:
    requested_name = str(requested).strip().lower() if requested is not None else "auto"
    if requested_name in {"", "auto"}:
        candidates = ["lightning.qubit", "default.qubit"]
    else:
        candidates = [requested_name, "lightning.qubit", "default.qubit"]

    seen: set[str] = set()
    for device_name in candidates:
        if device_name in seen:
            continue
        seen.add(device_name)
        if _probe_qml_device(qml=qml, device_name=device_name):
            return device_name
    raise RuntimeError("No usable PennyLane device found (tried requested, lightning.qubit, default.qubit).")


def _make_qml_device(qml: Any, device_name: str, n_qubits: int, shots: int | None):
    kwargs: dict[str, Any] = {"wires": n_qubits}
    if shots is not None:
        kwargs["shots"] = int(shots)
    return qml.device(device_name, **kwargs)


def _probe_qml_device(qml: Any, device_name: str) -> bool:
    try:
        dev = qml.device(device_name, wires=1)

        @qml.qnode(dev)
        def _probe():
            return qml.expval(qml.PauliZ(0))

        _probe()
        return True
    except Exception as exc:
        print(f"[paper-eval] warning: qml device '{device_name}' probe failed: {exc}")
        return False


def _resolve_eval_threshold(
    y_train: np.ndarray,
    train_prob: np.ndarray,
    threshold_mode: str,
    fixed_threshold: float,
) -> float:
    if threshold_mode == "fixed":
        return float(np.clip(fixed_threshold, 1e-6, 1.0 - 1e-6))
    if threshold_mode == "f1_opt":
        return _optimal_f1_threshold(y_train=y_train, prob=train_prob)
    return float(np.clip(fixed_threshold, 1e-6, 1.0 - 1e-6))


def _optimal_f1_threshold(y_train: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_train)) < 2:
        return 0.5
    prob = np.clip(prob.astype(np.float64), 1e-6, 1.0 - 1e-6)
    precision, recall, thresholds = precision_recall_curve(y_train, prob)
    if thresholds.size == 0:
        return 0.5
    f1 = (2.0 * precision[:-1] * recall[:-1]) / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    if f1.size == 0:
        return 0.5
    best_idx = int(np.nanargmax(f1))
    return float(np.clip(thresholds[best_idx], 1e-6, 1.0 - 1.0e-6))


def _train_qml_vqc(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_qubits: int,
    n_layers: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device_name: str,
    shots: int | None,
):
    import pennylane as qml
    from pennylane import numpy as pnp

    rng = np.random.default_rng(seed)
    dev = _make_qml_device(qml=qml, device_name=device_name, n_qubits=n_qubits, shots=shots)
    wires = list(range(n_qubits))

    @qml.qnode(dev, interface="autograd")
    def circuit(features, weights, bias):
        qml.AngleEmbedding(features, wires=wires, rotation="Y")
        qml.BasicEntanglerLayers(weights, wires=wires)
        qml.RY(bias, wires=0)
        return qml.expval(qml.PauliZ(0))

    def predict_prob(weights, bias, xb):
        raw = pnp.array([circuit(x, weights, bias) for x in xb])
        prob = (raw + 1.0) / 2.0
        return pnp.clip(prob, 1e-6, 1.0 - 1e-6)

    def loss(weights, bias, xb, yb):
        p = predict_prob(weights, bias, xb)
        return -pnp.mean(yb * pnp.log(p) + (1.0 - yb) * pnp.log(1.0 - p))

    weights = pnp.array(0.01 * rng.standard_normal((n_layers, n_qubits)), requires_grad=True)
    bias = pnp.array(0.0, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=learning_rate)
    n = x_train.shape[0]

    for _ in range(max(1, epochs)):
        order = rng.permutation(n)
        for start in range(0, n, max(1, batch_size)):
            idx = order[start : start + max(1, batch_size)]
            xb = x_train[idx]
            yb = y_train[idx]
            (weights, bias), _ = opt.step_and_cost(lambda w, b: loss(w, b, xb, yb), weights, bias)

    def predict_fn(x_test: np.ndarray) -> np.ndarray:
        prob = predict_prob(weights, bias, x_test)
        return np.asarray(prob, dtype=np.float64)

    return predict_fn


def _train_qml_qnn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_qubits: int,
    n_layers: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device_name: str,
    shots: int | None,
):
    import pennylane as qml
    from pennylane import numpy as pnp

    rng = np.random.default_rng(seed)
    dev = _make_qml_device(qml=qml, device_name=device_name, n_qubits=n_qubits, shots=shots)
    wires = list(range(n_qubits))

    @qml.qnode(dev, interface="autograd")
    def circuit(features, q_weights, q_bias):
        qml.AngleEmbedding(features, wires=wires, rotation="Y")
        qml.StronglyEntanglingLayers(q_weights, wires=wires)
        for i, w in enumerate(wires):
            qml.RY(q_bias[i], wires=w)
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    def batch_logits(q_weights, q_bias, head_w, head_b, xb):
        q_out = pnp.stack([pnp.array(circuit(x, q_weights, q_bias)) for x in xb], axis=0)
        return q_out @ head_w + head_b

    def predict_prob(q_weights, q_bias, head_w, head_b, xb):
        logits = batch_logits(q_weights, q_bias, head_w, head_b, xb)
        prob = 1.0 / (1.0 + pnp.exp(-logits))
        return pnp.clip(prob, 1e-6, 1.0 - 1e-6)

    def loss(q_weights, q_bias, head_w, head_b, xb, yb):
        p = predict_prob(q_weights, q_bias, head_w, head_b, xb)
        return -pnp.mean(yb * pnp.log(p) + (1.0 - yb) * pnp.log(1.0 - p))

    q_weights = pnp.array(0.01 * rng.standard_normal((n_layers, n_qubits, 3)), requires_grad=True)
    q_bias = pnp.array(0.01 * rng.standard_normal(n_qubits), requires_grad=True)
    head_w = pnp.array(0.01 * rng.standard_normal(n_qubits), requires_grad=True)
    head_b = pnp.array(0.0, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=learning_rate)
    n = x_train.shape[0]

    for _ in range(max(1, epochs)):
        order = rng.permutation(n)
        for start in range(0, n, max(1, batch_size)):
            idx = order[start : start + max(1, batch_size)]
            xb = x_train[idx]
            yb = y_train[idx]
            (q_weights, q_bias, head_w, head_b), _ = opt.step_and_cost(
                lambda qw, qb, hw, hb: loss(qw, qb, hw, hb, xb, yb),
                q_weights,
                q_bias,
                head_w,
                head_b,
            )

    def predict_fn(x_test: np.ndarray) -> np.ndarray:
        prob = predict_prob(q_weights, q_bias, head_w, head_b, x_test)
        return np.asarray(prob, dtype=np.float64)

    return predict_fn


def _train_qml_qkernel_svm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_qubits: int,
    n_layers: int,
    seed: int,
    device_name: str,
    shots: int | None,
    use_probability: bool,
    cache_size_mb: float,
):
    import pennylane as qml

    rng = np.random.default_rng(seed)
    dev = _make_qml_device(qml=qml, device_name=device_name, n_qubits=n_qubits, shots=shots)
    wires = list(range(n_qubits))
    kernel_weights = rng.normal(loc=0.0, scale=0.2, size=(n_layers, n_qubits)).astype(np.float64)

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        qml.AngleEmbedding(x1, wires=wires, rotation="Y")
        qml.BasicEntanglerLayers(kernel_weights, wires=wires)
        qml.adjoint(qml.BasicEntanglerLayers)(kernel_weights, wires=wires)
        qml.adjoint(qml.AngleEmbedding)(x2, wires=wires, rotation="Y")
        return qml.probs(wires=wires)

    def kernel_value(x1: np.ndarray, x2: np.ndarray) -> float:
        k = float(np.asarray(kernel_circuit(x1, x2), dtype=np.float64)[0])
        return float(np.clip(k, 0.0, 1.0))

    k_train = _build_square_kernel_matrix(qml=qml, x=x_train, kernel_value=kernel_value)

    clf = SVC(
        kernel="precomputed",
        probability=bool(use_probability),
        class_weight="balanced",
        random_state=seed,
        cache_size=float(max(cache_size_mb, 64.0)),
    )
    clf.fit(k_train, y_train.astype(np.uint8))

    def predict_fn(x_test: np.ndarray) -> np.ndarray:
        k_test = _build_rect_kernel_matrix(qml=qml, x1=x_test, x2=x_train, kernel_value=kernel_value)
        if bool(use_probability):
            prob = clf.predict_proba(k_test)
            if prob.ndim == 2 and prob.shape[1] >= 2:
                return prob[:, 1].astype(np.float64)
            return prob.reshape(-1).astype(np.float64)

        score = np.asarray(clf.decision_function(k_test), dtype=np.float64).reshape(-1)
        score = np.clip(score, -50.0, 50.0)
        prob = 1.0 / (1.0 + np.exp(-score))
        return np.clip(prob, 1e-6, 1.0 - 1e-6)

    return predict_fn


def _build_square_kernel_matrix(qml: Any, x: np.ndarray, kernel_value: Any) -> np.ndarray:
    kernels_mod = getattr(qml, "kernels", None)
    if kernels_mod is not None and hasattr(kernels_mod, "square_kernel_matrix"):
        try:
            k = np.asarray(kernels_mod.square_kernel_matrix(x, kernel_value), dtype=np.float64)
            return np.clip(k, 0.0, 1.0)
        except Exception:
            pass

    n = x.shape[0]
    k = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        k[i, i] = 1.0
        for j in range(i + 1, n):
            kval = kernel_value(x[i], x[j])
            k[i, j] = kval
            k[j, i] = kval
    return np.clip(k, 0.0, 1.0)


def _build_rect_kernel_matrix(qml: Any, x1: np.ndarray, x2: np.ndarray, kernel_value: Any) -> np.ndarray:
    kernels_mod = getattr(qml, "kernels", None)
    if kernels_mod is not None and hasattr(kernels_mod, "kernel_matrix"):
        try:
            k = np.asarray(kernels_mod.kernel_matrix(x1, x2, kernel_value), dtype=np.float64)
            return np.clip(k, 0.0, 1.0)
        except Exception:
            pass

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    k = np.empty((n1, n2), dtype=np.float64)
    for i in range(n1):
        for j in range(n2):
            k[i, j] = kernel_value(x1[i], x2[j])
    return np.clip(k, 0.0, 1.0)


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


def _stratified_subsample_only_xy(
    x: np.ndarray,
    y: np.ndarray,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_sub, y_sub, _, _ = _stratified_subsample(x, y, None, None, max_samples, seed)
    return x_sub, y_sub


def _summarize_fold_metrics(
    fold_df: pd.DataFrame,
    dataset_samples: int,
    feature_count: int,
    feature_names: list[str],
    split_strategy: str,
    block_size: int,
    n_splits: int,
    threshold_mode: str,
    fixed_threshold: float,
    label_mode: str,
    ranking_k_fracs: list[float],
    qml_meta: dict[str, Any],
) -> dict[str, Any]:
    group_cols = [
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "brier",
        "ece_10bin",
        "threshold_used",
        "fit_seconds",
        "predict_seconds",
    ]
    for frac in ranking_k_fracs:
        bp = int(round(float(frac) * 10000))
        group_cols.extend([f"precision_at_top{bp}bp", f"recall_at_top{bp}bp", f"lift_at_top{bp}bp"])
    model_summary: dict[str, Any] = {}
    if not fold_df.empty:
        for model_name, grp in fold_df.groupby("model"):
            stats = {}
            for c in group_cols:
                if c in grp:
                    stats[f"{c}_mean"] = float(grp[c].mean())
                    stats[f"{c}_std"] = float(grp[c].std(ddof=0))
            stats["folds"] = int(grp.shape[0])
            model_summary[str(model_name)] = stats

    return {
        "dataset_samples": dataset_samples,
        "feature_count": feature_count,
        "feature_names": feature_names,
        "split_strategy": split_strategy,
        "spatial_block_size": block_size,
        "n_splits": n_splits,
        "threshold_mode": threshold_mode,
        "fixed_threshold": fixed_threshold,
        "label_mode": label_mode,
        "ranking_k_fracs": ranking_k_fracs,
        "models": model_summary,
        "qml": qml_meta,
    }


def _load_feature_names(path: Path, n_features: int) -> list[str]:
    if path.exists():
        data = load_json(path)
        names = data.get("feature_names", [])
        if len(names) == n_features:
            return names
    return [f"f{i:02d}" for i in range(n_features)]


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
