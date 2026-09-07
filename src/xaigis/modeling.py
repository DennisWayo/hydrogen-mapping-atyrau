from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
import rasterio as rio
from rasterio.windows import Window
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .utils import ensure_dir, save_json


def train_models(cfg: dict[str, Any]) -> dict[str, Any]:
    paths = cfg["paths"]
    tcfg = cfg["training"]
    npz_path = paths["dataset_npz"]
    models_dir = ensure_dir(paths["models_dir"])
    threshold = float(tcfg.get("threshold", 0.8))

    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {npz_path}")

    data = np.load(npz_path)
    x = data["X"].astype(np.float32)
    y = data["y"].astype(np.uint8)
    print(f"[train] loaded dataset X{x.shape}, y{y.shape}")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=float(tcfg.get("test_size", 0.2)),
        random_state=int(tcfg.get("random_seed", 42)),
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    models = _build_models(tcfg, y_train)
    metrics: dict[str, Any] = {"threshold": threshold, "models": {}}

    for name, model in models.items():
        print(f"[train] fitting {name}")
        model.fit(x_train, y_train)
        prob = _predict_positive_probability(model, x_test)
        pred = (prob >= threshold).astype(np.uint8)

        model_metrics = _calc_metrics(y_test, prob, pred)
        metrics["models"][name] = model_metrics

        model_path = models_dir / f"{name}.joblib"
        joblib.dump(model, model_path)
        print(f"[train] saved model: {model_path}")
        print(
            f"[train] {name} ROC-AUC={model_metrics['roc_auc']:.4f}, "
            f"PR-AUC={model_metrics['pr_auc']:.4f}, "
            f"precision={model_metrics['precision']:.4f}, "
            f"recall={model_metrics['recall']:.4f}"
        )

    save_json(paths["metrics_json"], metrics)
    print(f"[train] saved metrics: {paths['metrics_json']}")
    return metrics


def predict_rasters(cfg: dict[str, Any]) -> dict[str, Any]:
    paths = cfg["paths"]
    tcfg = cfg["training"]
    pcfg = cfg["prediction"]
    threshold = float(tcfg.get("threshold", 0.8))
    tile_size = int(pcfg.get("tile_size", 512))
    exclude_nodata = bool(pcfg.get("exclude_nodata", True))

    stack_tif = paths["feature_stack_tif"]
    models_dir = paths["models_dir"]
    out_dir = ensure_dir(paths["predictions_dir"])

    if not stack_tif.exists():
        raise FileNotFoundError(f"Feature stack not found: {stack_tif}")
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    model_paths = sorted(models_dir.glob("*.joblib"))
    if not model_paths:
        raise FileNotFoundError(f"No model files found in {models_dir}")

    outputs: dict[str, dict[str, str]] = {}
    with rio.open(stack_tif) as src:
        channels, height, width = src.count, src.height, src.width
        prob_profile = src.profile.copy()
        prob_profile.update(
            count=1,
            dtype="float32",
            nodata=-9999.0,
            compress="deflate",
            BIGTIFF="YES",
        )
        mask_profile = src.profile.copy()
        mask_profile.update(
            count=1,
            dtype="uint8",
            nodata=0,
            compress="deflate",
            BIGTIFF="YES",
        )

        for mp in model_paths:
            name = mp.stem
            model = joblib.load(mp)
            prob_path = out_dir / f"{name}_prob.tif"
            mask_path = out_dir / f"pred_{name}_thresh{int(threshold * 100):02d}.tif"

            print(f"[predict] running {name} on raster (C={channels}, H={height}, W={width})")
            print(f"[predict] exclude_nodata={exclude_nodata}, nodata={src.nodata}")
            with rio.open(prob_path, "w", **prob_profile) as dst_prob, rio.open(
                mask_path, "w", **mask_profile
            ) as dst_mask:
                for row in range(0, height, tile_size):
                    h = min(tile_size, height - row)
                    for col in range(0, width, tile_size):
                        w = min(tile_size, width - col)
                        window = Window(col_off=col, row_off=row, width=w, height=h)
                        patch = src.read(window=window).astype(np.float32)  # (C,h,w)
                        finite_valid = np.isfinite(patch).all(axis=0)       # (h,w)
                        valid = finite_valid
                        if exclude_nodata:
                            valid &= ~_nodata_any(patch, src.nodatavals)

                        prob_patch = np.full((h, w), -9999.0, dtype=np.float32)
                        if np.any(valid):
                            x_tile = patch[:, valid].T
                            prob_valid = _predict_positive_probability(model, x_tile).astype(np.float32)
                            prob_patch[valid] = prob_valid

                        mask_patch = np.zeros((h, w), dtype=np.uint8)
                        mask_patch[valid] = (prob_patch[valid] >= threshold).astype(np.uint8)

                        dst_prob.write(prob_patch, 1, window=window)
                        dst_mask.write(mask_patch, 1, window=window)

            outputs[name] = {"probability_tif": str(prob_path), "mask_tif": str(mask_path)}
            print(f"[predict] saved: {prob_path}")
            print(f"[predict] saved: {mask_path}")

    return outputs


def _build_models(training_cfg: dict[str, Any], y_train: np.ndarray) -> dict[str, Any]:
    model_flags = training_cfg.get("models", {})
    seed = int(training_cfg.get("random_seed", 42))
    models: dict[str, Any] = {}

    if model_flags.get("sgd", True):
        sgd_cfg = training_cfg.get("sgd", {})
        loss = str(sgd_cfg.get("loss", "log_loss")).strip().lower()
        if loss == "log":
            loss = "log_loss"
        models["sgd"] = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SGDClassifier(
                        loss=loss,
                        alpha=float(sgd_cfg.get("alpha", 1e-4)),
                        max_iter=int(sgd_cfg.get("max_iter", 2000)),
                        random_state=seed,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

    if model_flags.get("rf", True):
        rf_cfg = training_cfg.get("rf", {})
        models["rf"] = RandomForestClassifier(
            n_estimators=int(rf_cfg.get("n_estimators", 200)),
            max_depth=rf_cfg.get("max_depth", None),
            n_jobs=int(rf_cfg.get("n_jobs", -1)),
            random_state=seed,
            class_weight="balanced_subsample",
        )

    if model_flags.get("xgb", True):
        try:
            from xgboost import XGBClassifier

            xgb_cfg = training_cfg.get("xgb", {})
            pos = max(int((y_train == 1).sum()), 1)
            neg = max(int((y_train == 0).sum()), 1)
            xgb_kwargs: dict[str, Any] = dict(
                n_estimators=int(xgb_cfg.get("n_estimators", 100)),
                max_depth=int(xgb_cfg.get("max_depth", 5)),
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=seed,
                n_jobs=-1,
                scale_pos_weight=float(neg / pos),
            )
            for key in ["learning_rate", "subsample", "colsample_bytree"]:
                if key in xgb_cfg:
                    xgb_kwargs[key] = float(xgb_cfg[key])
            models["xgb"] = XGBClassifier(**xgb_kwargs)
        except Exception as exc:
            print(f"[train] warning: xgboost unavailable, skipping xgb model ({exc})")

    if model_flags.get("lgbm", True):
        try:
            from lightgbm import LGBMClassifier

            lgb_cfg = training_cfg.get("lgbm", {})
            lgb_kwargs: dict[str, Any] = dict(
                random_state=seed,
                class_weight="balanced",
            )
            if "n_estimators" in lgb_cfg:
                lgb_kwargs["n_estimators"] = int(lgb_cfg["n_estimators"])
            if "learning_rate" in lgb_cfg:
                lgb_kwargs["learning_rate"] = float(lgb_cfg["learning_rate"])
            if "num_leaves" in lgb_cfg:
                lgb_kwargs["num_leaves"] = int(lgb_cfg["num_leaves"])
            models["lgbm"] = LGBMClassifier(**lgb_kwargs)
        except Exception as exc:
            print(f"[train] warning: lightgbm unavailable, skipping lgbm model ({exc})")

    if not models:
        raise RuntimeError("No models configured/enabled.")
    return models


def _predict_positive_probability(model: Any, x: np.ndarray) -> np.ndarray:
    # Some sklearn wrappers warn per tile when feature names are absent.
    # Tile inference uses NumPy arrays intentionally for speed.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"X does not have valid feature names, but .* was fitted with feature names",
        )
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(x)
            if prob.ndim == 2 and prob.shape[1] >= 2:
                return prob[:, 1]
            return prob.ravel()
        if hasattr(model, "decision_function"):
            score = model.decision_function(x)
            return 1.0 / (1.0 + np.exp(-score))
        pred = model.predict(x)
        return pred.astype(np.float32)


def _calc_metrics(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    roc_auc = float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else 0.0
    pr_auc = float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) > 1 else 0.0
    prec = float(precision_score(y_true, pred, zero_division=0))
    rec = float(recall_score(y_true, pred, zero_division=0))
    f1 = float(f1_score(y_true, pred, zero_division=0))
    cm = confusion_matrix(y_true, pred, labels=[0, 1]).tolist()
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }


def _nodata_any(x_patch: np.ndarray, nodatavals: tuple[Any, ...] | list[Any]) -> np.ndarray:
    nodata_any = np.zeros(x_patch.shape[1:], dtype=bool)
    for band_idx, nodata in enumerate(nodatavals):
        if nodata is None:
            continue
        try:
            nodata_float = float(nodata)
        except (TypeError, ValueError):
            continue
        if np.isnan(nodata_float):
            continue
        nodata_any |= x_patch[band_idx] == nodata_float
    return nodata_any
