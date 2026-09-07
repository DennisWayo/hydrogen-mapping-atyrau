#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from matplotlib import patheffects as pe
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.enums import Resampling
from scipy.ndimage import binary_dilation
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)

QML_MODEL_NAMES = {"qml_vqc", "qml_qnn", "qml_qkernel_svm"}
AUX_MODEL_NAMES = {"ml_anomaly"}


def _bootstrap_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _get_access_token(username: str, password: str) -> str:
    form = urllib.parse.urlencode(
        {
            "grant_type": "password",
            "client_id": "cdse-public",
            "username": username,
            "password": password,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        TOKEN_URL,
        data=form,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    token = str(payload.get("access_token", "")).strip()
    if not token:
        raise RuntimeError("CDSE token response missing access_token.")
    return token


def _stream_download(url: str, out_file: Path, token: str) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = out_file.with_suffix(out_file.suffix + ".part")
    if tmp_file.exists():
        tmp_file.unlink()
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            with tmp_file.open("wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        tmp_file.replace(out_file)
    except Exception:
        if tmp_file.exists():
            tmp_file.unlink()
        raise


def _download_with_retries(
    *,
    url: str,
    out_file: Path,
    token: str,
    username: str,
    password: str,
    max_attempts: int = 6,
    base_backoff_sec: int = 15,
    max_backoff_sec: int = 180,
) -> str:
    local_token = token
    if not local_token:
        local_token = _get_access_token(username, password)

    for attempt in range(1, max_attempts + 1):
        try:
            _stream_download(url, out_file, local_token)
            return local_token
        except urllib.error.HTTPError as exc:
            code = int(getattr(exc, "code", 0))
            if code == 401:
                if attempt >= max_attempts:
                    raise
                print(
                    f"[multi-scene] download unauthorized (401), refreshing token and retrying "
                    f"(attempt {attempt + 1}/{max_attempts})"
                )
                local_token = _get_access_token(username, password)
                continue
            if code in (429, 500, 502, 503, 504):
                if attempt >= max_attempts:
                    raise
                delay = min(base_backoff_sec * (2 ** (attempt - 1)), max_backoff_sec)
                print(
                    f"[multi-scene] transient download error (HTTP {code}), retrying in {delay}s "
                    f"(attempt {attempt + 1}/{max_attempts})"
                )
                time.sleep(delay)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt >= max_attempts:
                raise
            delay = min(base_backoff_sec * (2 ** (attempt - 1)), max_backoff_sec)
            print(
                f"[multi-scene] transient network error ({exc.__class__.__name__}), retrying in {delay}s "
                f"(attempt {attempt + 1}/{max_attempts})"
            )
            time.sleep(delay)

    return local_token


def _read_search_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {str(r.get("id", "")).strip(): r for r in rows if str(r.get("id", "")).strip()}


def _stretch_rgb(rgb: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rgb, dtype=np.float32)
    for i in range(3):
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


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0.0 else 0.0


def _resolve_map_shape(
    *,
    stack_tif: Path,
    map_size: int,
    full_scale: bool,
) -> tuple[int, int]:
    if bool(full_scale) or int(map_size) <= 0:
        with rio.open(stack_tif) as ds:
            return int(ds.height), int(ds.width)
    size = max(32, int(map_size))
    return size, size


def _save_fig(
    fig: plt.Figure,
    base: Path,
    *,
    png_dpi: int = 300,
    png_compress_level: int = 4,
    png_min_size_mb: float = 0.0,
    png_target_size_mb: float = 0.0,
    png_target_tolerance: float = 0.15,
) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    png_path = base.with_suffix(".png")
    dpi = max(72, int(png_dpi))
    compress_level = int(np.clip(int(png_compress_level), 0, 9))

    def _save_png(target_dpi: int) -> None:
        try:
            fig.savefig(
                str(png_path),
                dpi=int(target_dpi),
                bbox_inches="tight",
                pil_kwargs={"compress_level": compress_level},
            )
        except TypeError:
            fig.savefig(
                str(png_path),
                dpi=int(target_dpi),
                bbox_inches="tight",
            )

    _save_png(dpi)
    current_bytes = int(png_path.stat().st_size) if png_path.exists() else 0
    target_size_bytes = int(max(0.0, float(png_target_size_mb)) * 1024.0 * 1024.0)
    target_tol = float(np.clip(float(png_target_tolerance), 0.0, 0.95))
    if target_size_bytes > 0:
        lower = max(1, int(target_size_bytes * (1.0 - target_tol)))
        upper = max(lower, int(target_size_bytes * (1.0 + target_tol)))
        current_dpi = dpi
        for _ in range(8):
            if lower <= current_bytes <= upper:
                break
            if current_bytes <= 0:
                ratio = 1.5
            else:
                ratio = math.sqrt(float(target_size_bytes) / float(current_bytes))
            if current_bytes < lower:
                next_dpi = int(math.ceil(current_dpi * min(1.8, max(1.05, ratio * 1.02))))
                next_dpi = min(2400, max(current_dpi + 20, next_dpi))
            else:
                next_dpi = int(math.floor(current_dpi * max(0.55, min(0.98, ratio * 0.98))))
                next_dpi = max(72, min(current_dpi - 20, next_dpi))
            if next_dpi == current_dpi:
                break
            current_dpi = next_dpi
            _save_png(current_dpi)
            current_bytes = int(png_path.stat().st_size) if png_path.exists() else current_bytes
    else:
        min_size_bytes = int(max(0.0, float(png_min_size_mb)) * 1024.0 * 1024.0)
        if min_size_bytes > 0:
            current_dpi = dpi
            for _ in range(5):
                if current_bytes >= min_size_bytes:
                    break
                scale = math.sqrt(float(min_size_bytes) / float(max(current_bytes, 1)))
                next_dpi = int(math.ceil(current_dpi * min(1.8, scale * 1.05)))
                next_dpi = min(2400, max(current_dpi + 50, next_dpi))
                if next_dpi <= current_dpi:
                    break
                current_dpi = next_dpi
                _save_png(current_dpi)
                current_bytes = int(png_path.stat().st_size) if png_path.exists() else current_bytes

    fig.savefig(str(base.with_suffix(".svg")), bbox_inches="tight")
    plt.close(fig)


def _prepare_model_subset_dir(
    *,
    base_models_dir: Path,
    selected_models: list[str],
    out_root: Path,
) -> Path:
    subset_dir = out_root / "_model_subset"
    subset_dir.mkdir(parents=True, exist_ok=True)
    for stale in subset_dir.glob("*.joblib"):
        stale.unlink()

    for model in selected_models:
        src = base_models_dir / f"{model}.joblib"
        if not src.exists():
            raise FileNotFoundError(f"Model file missing for requested model '{model}': {src}")
        dst = subset_dir / src.name
        try:
            dst.symlink_to(src.resolve())
        except Exception:
            shutil.copy2(src, dst)
    return subset_dir


def _split_requested_models(models: list[str]) -> tuple[list[str], list[str], list[str]]:
    classical: list[str] = []
    qml: list[str] = []
    aux: list[str] = []
    for m in models:
        key = str(m).strip().lower()
        if not key:
            continue
        if key in QML_MODEL_NAMES:
            qml.append(key)
        elif key in AUX_MODEL_NAMES:
            aux.append(key)
        else:
            classical.append(key)
    classical = list(dict.fromkeys(classical))
    qml = list(dict.fromkeys(qml))
    aux = list(dict.fromkeys(aux))
    return classical, qml, aux


def _load_qml_thresholds(paper_eval_json: Path) -> dict[str, float]:
    if not paper_eval_json.exists() or not paper_eval_json.is_file():
        return {}
    try:
        payload = json.loads(paper_eval_json.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[multi-scene] warning: failed to parse paper-eval metrics JSON ({exc})")
        return {}

    out: dict[str, float] = {}
    for model_name, model_stats in dict(payload.get("models", {})).items():
        key = str(model_name).strip().lower()
        if key not in QML_MODEL_NAMES:
            continue
        try:
            thr = float(dict(model_stats).get("threshold_used_mean"))
        except Exception:
            continue
        if 0.0 < thr < 1.0:
            out[key] = thr
    return out


def _build_bbox_core_mask(positive: np.ndarray, target_pixels: int) -> np.ndarray:
    core = np.zeros_like(positive, dtype=bool)
    if target_pixels <= 0:
        return core
    rr, cc = np.where(positive)
    if rr.size == 0:
        return core

    r_min, r_max = int(rr.min()), int(rr.max()) + 1
    c_min, c_max = int(cc.min()), int(cc.max()) + 1
    box_h = max(1, r_max - r_min)
    box_w = max(1, c_max - c_min)
    pos_pixels = int(positive.sum())
    scale = float(np.sqrt(min(1.0, target_pixels / max(pos_pixels, 1))))

    for _ in range(5):
        h = max(1, int(round(box_h * scale)))
        w = max(1, int(round(box_w * scale)))
        r_mid = (r_min + r_max) // 2
        c_mid = (c_min + c_max) // 2

        r0 = max(0, r_mid - h // 2)
        c0 = max(0, c_mid - w // 2)
        r1 = min(positive.shape[0], r0 + h)
        c1 = min(positive.shape[1], c0 + w)
        r0 = max(0, r1 - h)
        c0 = max(0, c1 - w)

        core.fill(False)
        core[r0:r1, c0:c1] = True
        core &= positive
        core_pixels = int(core.sum())
        if core_pixels <= target_pixels:
            break
        scale *= float(np.sqrt(target_pixels / max(core_pixels, 1)))
    return core


def _repair_degenerate_scene_labels(
    *,
    label_tif: Path,
    uncertain_value: int,
    negative_exclusion_buffer_px: int,
    max_positive_fraction: float,
) -> bool:
    with rio.open(label_tif) as src:
        profile = src.profile.copy()
        label = src.read(1).astype(np.uint8)

    valid = label <= 1
    total_valid = int(valid.sum())
    if total_valid == 0:
        return False

    positive = label == 1
    pos_count = int(positive.sum())
    pos_frac = pos_count / max(total_valid, 1)
    if pos_frac <= float(max_positive_fraction):
        return False

    target_pos = max(1, int(round(float(max_positive_fraction) * total_valid)))
    core = _build_bbox_core_mask(positive=positive, target_pixels=target_pos)
    core_count = int(core.sum())
    if core_count == 0:
        print(
            f"[labels] warning: degenerate label repair skipped for {label_tif} "
            "(core mask empty)"
        )
        return False

    repaired = np.zeros_like(label, dtype=np.uint8)
    repaired[core] = 1
    if negative_exclusion_buffer_px > 0:
        dilated = binary_dilation(core, iterations=int(negative_exclusion_buffer_px), border_value=0)
        uncertain = dilated & (~core)
        repaired[uncertain] = np.uint8(uncertain_value)

    with rio.open(label_tif, "w", **profile) as dst:
        dst.write(repaired, 1)
        dst.set_band_description(1, "label")

    repaired_pos = int((repaired == 1).sum())
    repaired_neg = int((repaired == 0).sum())
    repaired_unc = int((repaired == uncertain_value).sum())
    print(
        f"[labels] repaired degenerate scene labels in {label_tif}: "
        f"positive_ratio {pos_frac:.4%} -> {repaired_pos / max(total_valid, 1):.4%} "
        f"(target <= {float(max_positive_fraction):.1%})"
    )
    print(
        f"[labels] repaired counts: positives={repaired_pos:,}, "
        f"negatives={repaired_neg:,}, uncertain={repaired_unc:,} (value={uncertain_value})"
    )
    return True


def _stratified_subsample_xy(
    *,
    x: np.ndarray,
    y: np.ndarray,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples <= 0 or x.shape[0] <= max_samples:
        return x, y
    if len(np.unique(y)) < 2:
        idx = np.random.default_rng(seed).choice(x.shape[0], size=max_samples, replace=False)
        return x[idx], y[idx]
    x_sub, _, y_sub, _ = train_test_split(
        x,
        y,
        train_size=max_samples,
        random_state=seed,
        stratify=y,
    )
    return x_sub, y_sub


def _train_qml_predictors(
    *,
    cfg: dict[str, Any],
    qml_models: list[str],
    thresholds: dict[str, float],
    qml_train_samples_override: int,
) -> dict[str, dict[str, Any]]:
    if not qml_models:
        return {}

    from xaigis.paper_eval import (
        _parse_optional_positive_int,
        _resolve_qml_device_name,
        _train_qml_qkernel_svm,
        _train_qml_qnn,
        _train_qml_vqc,
    )

    try:
        import pennylane as qml  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"PennyLane is required for QML scene maps ({exc})") from exc

    paths = cfg["paths"]
    dataset_npz: Path = paths["dataset_npz"]
    if not dataset_npz.exists():
        raise FileNotFoundError(f"Dataset NPZ not found for QML training: {dataset_npz}")
    data = np.load(dataset_npz)
    x_all = data["X"].astype(np.float32)
    y_all = data["y"].astype(np.uint8)
    if len(np.unique(y_all)) < 2:
        raise ValueError("QML training requires at least two classes in dataset labels.")

    ecfg = cfg.get("paper_eval", {})
    qml_cfg = dict(ecfg.get("qml", {}))
    seed = int(ecfg.get("random_seed", cfg.get("training", {}).get("random_seed", 42)))
    n_qubits_cfg = int(qml_cfg.get("n_qubits", 6))
    n_layers = int(qml_cfg.get("n_layers", 2))
    epochs = int(qml_cfg.get("epochs", 15))
    batch_size = int(qml_cfg.get("batch_size", 32))
    learning_rate = float(qml_cfg.get("learning_rate", 0.05))
    default_max_train = int(qml_cfg.get("max_train_samples", 1200))
    default_max_kernel = int(
        qml_cfg.get("max_train_samples_kernel", min(default_max_train, 300))
    )
    if qml_train_samples_override > 0:
        default_max_train = int(qml_train_samples_override)
        default_max_kernel = min(default_max_kernel, default_max_train)
    use_kernel_probability = bool(qml_cfg.get("kernel_probability", False))
    kernel_cache_mb = float(qml_cfg.get("kernel_cache_mb", 1024.0))
    device_name = _resolve_qml_device_name(qml=qml, requested=qml_cfg.get("device", "auto"))
    shots = _parse_optional_positive_int(qml_cfg.get("shots"))
    shots_label = "analytic" if shots is None else str(shots)
    print(f"[multi-scene] qml backend: {device_name}, shots={shots_label}")

    predictors: dict[str, dict[str, Any]] = {}
    for idx, model_name in enumerate(qml_models):
        max_samples = default_max_kernel if model_name == "qml_qkernel_svm" else default_max_train
        x_train, y_train = _stratified_subsample_xy(
            x=x_all,
            y=y_all,
            max_samples=max_samples,
            seed=seed + idx,
        )

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_train)
        n_qubits = max(1, min(n_qubits_cfg, x_scaled.shape[1], x_scaled.shape[0]))
        pca = PCA(n_components=n_qubits, random_state=seed)
        x_pca = pca.fit_transform(x_scaled)
        amp = MinMaxScaler(feature_range=(-np.pi, np.pi))
        x_q = amp.fit_transform(x_pca).astype(np.float64)

        print(
            f"[multi-scene] training {model_name} for scene-map inference "
            f"(samples={x_q.shape[0]}, qubits={n_qubits}, layers={n_layers})"
        )
        fit_start = time.perf_counter()
        if model_name == "qml_qnn":
            pred_fn = _train_qml_qnn(
                x_train=x_q,
                y_train=y_train.astype(np.float64),
                n_qubits=n_qubits,
                n_layers=n_layers,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed + idx,
                device_name=device_name,
                shots=shots,
            )
        elif model_name == "qml_qkernel_svm":
            pred_fn = _train_qml_qkernel_svm(
                x_train=x_q,
                y_train=y_train.astype(np.uint8),
                n_qubits=n_qubits,
                n_layers=n_layers,
                seed=seed + idx,
                device_name=device_name,
                shots=shots,
                use_probability=use_kernel_probability,
                cache_size_mb=kernel_cache_mb,
            )
        else:
            pred_fn = _train_qml_vqc(
                x_train=x_q,
                y_train=y_train.astype(np.float64),
                n_qubits=n_qubits,
                n_layers=n_layers,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed + idx,
                device_name=device_name,
                shots=shots,
            )
        fit_seconds = time.perf_counter() - fit_start
        thr = float(thresholds.get(model_name, 0.5))
        print(
            f"[multi-scene] trained {model_name} in {fit_seconds:.1f}s "
            f"(threshold={thr:.4f})"
        )
        predictors[model_name] = {
            "predict_fn": pred_fn,
            "scaler": scaler,
            "pca": pca,
            "amp": amp,
        }

    return predictors


def _predict_qml_prob_small(
    *,
    stack_tif: Path,
    predictor: dict[str, Any],
    map_shape: tuple[int, int],
    batch_size: int,
) -> np.ndarray:
    map_h, map_w = int(map_shape[0]), int(map_shape[1])
    with rio.open(stack_tif) as ds:
        stack_small = ds.read(
            out_shape=(ds.count, map_h, map_w),
            resampling=Resampling.bilinear,
        ).astype(np.float32)

    valid = np.isfinite(stack_small).all(axis=0)
    prob = np.full((map_h, map_w), np.nan, dtype=np.float32)
    if not np.any(valid):
        return prob

    x = stack_small[:, valid].T
    x_scaled = predictor["scaler"].transform(x)
    x_pca = predictor["pca"].transform(x_scaled)
    x_q = predictor["amp"].transform(x_pca).astype(np.float64)

    pred_fn = predictor["predict_fn"]
    n = x_q.shape[0]
    batch = max(1, int(batch_size))
    out = np.empty(n, dtype=np.float32)
    for start in range(0, n, batch):
        end = min(n, start + batch)
        out[start:end] = np.clip(
            np.asarray(pred_fn(x_q[start:end]), dtype=np.float32).reshape(-1),
            0.0,
            1.0,
        )
    prob[valid] = out
    return prob


def _rank_normalize_valid(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr.astype(np.float32)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=np.float32)
    if arr.size == 1:
        ranks[0] = 0.5
        return ranks
    ranks[order] = np.linspace(0.0, 1.0, arr.size, endpoint=True, dtype=np.float32)
    return ranks


def _maybe_rank_normalize_prob(
    *,
    prob: np.ndarray,
    valid_mask: np.ndarray,
    saturation_eps: float = 1.0e-3,
) -> tuple[np.ndarray, bool]:
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(prob)
    if not np.any(valid):
        return prob.astype(np.float32), False
    vals = prob[valid].astype(np.float64)
    if vals.size < 16:
        return prob.astype(np.float32), False
    lo = float(np.percentile(vals, 2.0))
    hi = float(np.percentile(vals, 98.0))
    if (hi - lo) >= float(saturation_eps):
        return prob.astype(np.float32), False

    norm = np.full(prob.shape, np.nan, dtype=np.float32)
    norm[valid] = _rank_normalize_valid(vals)
    return norm, True


def _predict_ml_anomaly_prob_small(
    *,
    stack_tif: Path,
    map_shape: tuple[int, int],
    seed: int,
) -> np.ndarray:
    map_h, map_w = int(map_shape[0]), int(map_shape[1])
    with rio.open(stack_tif) as ds:
        stack_small = ds.read(
            out_shape=(ds.count, map_h, map_w),
            resampling=Resampling.bilinear,
        ).astype(np.float32)

    valid = np.isfinite(stack_small).all(axis=0)
    prob = np.full((map_h, map_w), np.nan, dtype=np.float32)
    if not np.any(valid):
        return prob

    x = stack_small[:, valid].T.astype(np.float64)
    med = np.median(x, axis=0)
    q25 = np.percentile(x, 25.0, axis=0)
    q75 = np.percentile(x, 75.0, axis=0)
    iqr = np.clip(q75 - q25, 1.0e-6, None)
    x_scaled = (x - med) / iqr
    x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    n_samples, n_features = x_scaled.shape
    n_components = 1
    if n_samples > 2 and n_features > 1:
        n_components = int(max(2, min(8, n_features, n_samples - 1)))

    if n_components > 1:
        pca = PCA(n_components=n_components, random_state=int(seed))
        z = pca.fit_transform(x_scaled)
    else:
        z = x_scaled[:, :1]

    center = np.median(z, axis=0)
    diff = z - center
    cov = np.cov(diff, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]], dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    cov += np.eye(cov.shape[0], dtype=np.float64) * 1.0e-6
    inv_cov = np.linalg.pinv(cov)
    dist2 = np.einsum("ij,jk,ik->i", diff, inv_cov, diff, optimize=True)
    dist = np.sqrt(np.clip(dist2, 0.0, None))

    prob[valid] = _rank_normalize_valid(dist)
    return prob


def _scene_label(scene_id: str) -> str:
    parts = [p for p in str(scene_id).split("_") if p]
    tile = next(
        (p for p in parts if p.startswith("T") and len(p) == 6 and p[1:3].isdigit()),
        parts[-2] if len(parts) >= 2 else scene_id,
    )
    acq_date = ""
    for p in parts:
        if len(p) >= 8 and p[:8].isdigit():
            acq_date = f"{p[:4]}-{p[4:6]}-{p[6:8]}"
            break
    return f"{tile} | {acq_date}" if acq_date else tile


def _crop_width_meters(
    *,
    stack_tif: Path,
    map_height: int,
    map_width: int,
    c0: int,
    c1: int,
    r0: int,
    r1: int,
) -> float | None:
    with rio.open(stack_tif) as ds:
        bounds = ds.bounds
        crs = ds.crs

    frac_x = float(max(c1 - c0, 1)) / float(max(map_width, 1))
    width_units = abs(float(bounds.right) - float(bounds.left)) * frac_x

    if crs is not None and bool(getattr(crs, "is_geographic", False)):
        frac_y = float(max(r1 - r0, 1)) / float(max(map_height, 1))
        _ = frac_y  # kept for readability
        rel_mid = (float(r0 + r1) * 0.5) / float(max(map_height, 1))
        lat_mid = float(bounds.top) - rel_mid * (float(bounds.top) - float(bounds.bottom))
        meters_per_deg_lon = 111_320.0 * max(0.05, math.cos(math.radians(lat_mid)))
        return width_units * meters_per_deg_lon

    if crs is not None and bool(getattr(crs, "is_projected", False)):
        return width_units

    return None


def _pick_scalebar_length(width_m: float) -> float:
    candidates = [
        200.0,
        500.0,
        1_000.0,
        2_000.0,
        5_000.0,
        10_000.0,
        20_000.0,
        50_000.0,
        100_000.0,
        200_000.0,
    ]
    target = max(100.0, 0.2 * float(width_m))
    chosen = candidates[0]
    for c in candidates:
        if c <= target:
            chosen = c
    return min(chosen, 0.45 * float(width_m))


def _fmt_distance(meters: float) -> str:
    m = float(meters)
    if m >= 1000.0:
        km = m / 1000.0
        if abs(km - round(km)) < 0.05:
            return f"{int(round(km))} km"
        return f"{km:.1f} km"
    return f"{int(round(m))} m"


def _overlay_text(ax: Any, x: float, y: float, text: str, *, size: int = 9, ha: str = "left", va: str = "top") -> Any:
    t = ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        color="white",
        fontsize=size,
        fontweight="bold",
        ha=ha,
        va=va,
    )
    t.set_path_effects([pe.withStroke(linewidth=2.2, foreground="black")])
    return t


def _add_map_decorations(ax: Any, *, scene_text: str, width_m: float | None) -> None:
    _overlay_text(ax, 0.02, 0.98, scene_text, size=9, ha="left", va="top")
    _overlay_text(ax, 0.92, 0.95, "N", size=10, ha="center", va="bottom")
    ax.annotate(
        "",
        xy=(0.92, 0.92),
        xytext=(0.92, 0.76),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=2.0, color="white"),
    )

    if width_m is None or not np.isfinite(width_m) or float(width_m) <= 0.0:
        return
    bar_m = _pick_scalebar_length(float(width_m))
    frac = max(0.08, min(0.42, bar_m / float(width_m)))
    x0, y0 = 0.05, 0.07
    x1 = x0 + frac
    ax.plot([x0, x1], [y0, y0], transform=ax.transAxes, color="white", lw=3.0, solid_capstyle="butt")
    ax.plot([x0, x0], [y0 - 0.01, y0 + 0.01], transform=ax.transAxes, color="white", lw=2.0)
    ax.plot([x1, x1], [y0 - 0.01, y0 + 0.01], transform=ax.transAxes, color="white", lw=2.0)
    _overlay_text(ax, (x0 + x1) * 0.5, y0 + 0.018, _fmt_distance(bar_m), size=8, ha="center", va="bottom")


def _select_rgb_indices(stack_tif: Path) -> tuple[int, int, int]:
    with rio.open(stack_tif) as ds:
        desc_to_idx = {str(desc): i + 1 for i, desc in enumerate(ds.descriptions) if desc}
    candidates = [
        ("B04_MEDIAN", "B03_MEDIAN", "B02_MEDIAN"),
        ("B04", "B03", "B02"),
    ]
    for r, g, b in candidates:
        if r in desc_to_idx and g in desc_to_idx and b in desc_to_idx:
            return desc_to_idx[r], desc_to_idx[g], desc_to_idx[b]
    raise KeyError("Could not find RGB bands in feature stack descriptions.")


def _make_scene_maps(
    *,
    scene_id: str,
    stack_tif: Path,
    label_tif: Path,
    pred_dir: Path,
    thresholds: dict[str, float],
    models: list[str],
    map_shape: tuple[int, int],
    out_dir: Path,
    extra_prob_small_by_model: dict[str, np.ndarray] | None = None,
    error_map_mode: str = "absolute",
    labels_proxy_mode: bool = False,
    saturation_eps: float = 1.0e-3,
    png_dpi: int = 300,
    png_compress_level: int = 4,
    png_min_size_mb: float = 0.0,
    png_target_size_mb: float = 0.0,
    png_target_tolerance: float = 0.15,
    plot_lock: threading.Lock | None = None,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    map_h, map_w = int(map_shape[0]), int(map_shape[1])

    with rio.open(label_tif) as ds_lbl:
        label_full = ds_lbl.read(1).astype(np.uint8)
        label_small = ds_lbl.read(
            1, out_shape=(map_h, map_w), resampling=Resampling.nearest
        ).astype(np.uint8)
    valid_label_full = np.isin(label_full, [0, 1])
    valid_label_small = np.isin(label_small, [0, 1])

    r_idx, g_idx, b_idx = _select_rgb_indices(stack_tif)
    with rio.open(stack_tif) as ds_stack:
        rgb = ds_stack.read(
            [r_idx, g_idx, b_idx],
            out_shape=(3, map_h, map_w),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
    rgb = _stretch_rgb(rgb)

    prob_small_by_model: dict[str, np.ndarray] = {}
    metrics_rows: list[dict[str, Any]] = []
    extra_prob_small = extra_prob_small_by_model or {}

    for model in models:
        thr = float(thresholds.get(model, 0.8))
        if model in extra_prob_small:
            p_small = np.asarray(extra_prob_small[model], dtype=np.float32)
            if p_small.shape != label_small.shape:
                raise ValueError(
                    f"extra_prob_small shape mismatch for {model}: "
                    f"expected {label_small.shape}, got {p_small.shape}"
                )
            p_small = np.clip(p_small, 0.0, 1.0)
            prob_small_by_model[model] = p_small

            valid = valid_label_small & np.isfinite(p_small)
            y = label_small[valid]
            pred = (p_small[valid] >= thr).astype(np.uint8)
            tn = int(np.sum((y == 0) & (pred == 0)))
            fp = int(np.sum((y == 0) & (pred == 1)))
            fn = int(np.sum((y == 1) & (pred == 0)))
            tp = int(np.sum((y == 1) & (pred == 1)))
            precision = _safe_div(tp, tp + fp)
            recall = _safe_div(tp, tp + fn)
            bal_acc = 0.5 * (_safe_div(tp, tp + fn) + _safe_div(tn, tn + fp))
            metrics_rows.append(
                {
                    "scene_id": scene_id,
                    "model": model,
                    "threshold": thr,
                    "valid_pixels": int(valid.sum()),
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                    "precision_at_threshold": precision,
                    "recall_at_threshold": recall,
                    "balanced_accuracy_at_threshold": bal_acc,
                }
            )
            continue

        prob_path = pred_dir / f"{model}_prob.tif"
        if not prob_path.exists():
            print(f"[scene-map] warning: missing probability raster for {model} in {pred_dir}")
            continue
        with rio.open(prob_path) as ds_prob:
            p_full = ds_prob.read(1).astype(np.float32)
            nodata = ds_prob.nodata
            p_small = ds_prob.read(
                1, out_shape=(map_h, map_w), resampling=Resampling.bilinear
            ).astype(np.float32)
        if nodata is not None:
            p_full[p_full == float(nodata)] = np.nan
            p_small[p_small == float(nodata)] = np.nan
        p_full = np.clip(p_full, 0.0, 1.0)
        p_small = np.clip(p_small, 0.0, 1.0)
        prob_small_by_model[model] = p_small

        valid = valid_label_full & np.isfinite(p_full)
        y = label_full[valid]
        pred = (p_full[valid] >= thr).astype(np.uint8)
        tn = int(np.sum((y == 0) & (pred == 0)))
        fp = int(np.sum((y == 0) & (pred == 1)))
        fn = int(np.sum((y == 1) & (pred == 0)))
        tp = int(np.sum((y == 1) & (pred == 1)))
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        bal_acc = 0.5 * (_safe_div(tp, tp + fn) + _safe_div(tn, tn + fp))
        metrics_rows.append(
            {
                "scene_id": scene_id,
                "model": model,
                "threshold": thr,
                "valid_pixels": int(valid.sum()),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "precision_at_threshold": precision,
                "recall_at_threshold": recall,
                "balanced_accuracy_at_threshold": bal_acc,
            }
        )

    if not prob_small_by_model:
        return metrics_rows

    # Crop to valid footprint in downsampled label for cleaner figures.
    if np.any(valid_label_small):
        rr, cc = np.where(valid_label_small)
        r0, r1 = int(rr.min()), int(rr.max()) + 1
        c0, c1 = int(cc.min()), int(cc.max()) + 1
    else:
        r0, r1, c0, c1 = 0, map_h, 0, map_w
    rgb = rgb[:, r0:r1, c0:c1]
    label_small = label_small[r0:r1, c0:c1]
    valid_label_small = valid_label_small[r0:r1, c0:c1]
    for model in list(prob_small_by_model.keys()):
        prob_small_by_model[model] = prob_small_by_model[model][r0:r1, c0:c1]
    plot_prob_small_by_model: dict[str, np.ndarray] = {}
    saturated_models: list[str] = []
    for model, p_raw in prob_small_by_model.items():
        p_plot, was_saturated = _maybe_rank_normalize_prob(
            prob=p_raw,
            valid_mask=valid_label_small,
            saturation_eps=float(saturation_eps),
        )
        plot_prob_small_by_model[model] = p_plot
        if was_saturated:
            saturated_models.append(model)
    if saturated_models:
        print(
            f"[scene-map] scene={scene_id} saturated probabilities detected for "
            f"{','.join(saturated_models)}; using rank-normalized display."
        )
    scene_text = _scene_label(scene_id)
    crop_width_m = _crop_width_meters(
        stack_tif=stack_tif,
        map_height=map_h,
        map_width=map_w,
        c0=c0,
        c1=c1,
        r0=r0,
        r1=r1,
    )

    prob_vals = []
    for p in plot_prob_small_by_model.values():
        m = valid_label_small & np.isfinite(p)
        if np.any(m):
            prob_vals.append(p[m])
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

    mode = str(error_map_mode).strip().lower()
    use_signed_disagreement = bool(labels_proxy_mode) or mode == "signed_disagreement"
    plot_guard = plot_lock if plot_lock is not None else nullcontext()
    with plot_guard:
        # Prospectivity figure.
        plot_models = [m for m in models if m in prob_small_by_model]
        fig, axes = plt.subplots(
            1,
            1 + len(plot_models),
            figsize=(6.2 * (1 + len(plot_models)), 6.0),
            constrained_layout=True,
        )
        axes_arr = np.atleast_1d(axes).ravel()
        axes_arr[0].imshow(np.transpose(rgb, (1, 2, 0)))
        axes_arr[0].set_title("A) RGB")
        axes_arr[0].axis("off")
        _add_map_decorations(axes_arr[0], scene_text=scene_text, width_m=crop_width_m)
        last_im = None
        for i, model in enumerate(plot_models, start=1):
            p_raw = prob_small_by_model[model]
            p_plot = plot_prob_small_by_model[model]
            last_im = axes_arr[i].imshow(p_plot, cmap="cividis", vmin=p_lo, vmax=p_hi)
            thr = float(thresholds.get(model, 0.8))
            finite = np.isfinite(p_raw)
            if np.any(finite):
                try:
                    axes_arr[i].contour(
                        np.where(finite, p_raw, np.nan),
                        levels=[thr],
                        colors=["white"],
                        linewidths=0.8,
                        alpha=0.9,
                    )
                except Exception:
                    pass
            axes_arr[i].set_title(f"{chr(ord('A') + i)}) {model.upper()} prospectivity")
            axes_arr[i].axis("off")
            _add_map_decorations(axes_arr[i], scene_text=scene_text, width_m=crop_width_m)
        if last_im is not None:
            cbar = fig.colorbar(last_im, ax=axes_arr.tolist(), shrink=0.88)
            if saturated_models:
                cbar.set_label("Prospectivity score (rank-normalized for saturated models)")
            else:
                cbar.set_label("Predicted hydrogen probability (contrast-scaled)")
        _save_fig(
            fig,
            out_dir / f"{scene_id}_prospectivity",
            png_dpi=int(png_dpi),
            png_compress_level=int(png_compress_level),
            png_min_size_mb=float(png_min_size_mb),
            png_target_size_mb=float(png_target_size_mb),
            png_target_tolerance=float(png_target_tolerance),
        )

        # Error map figure.
        fig, axes = plt.subplots(
            1,
            len(plot_models),
            figsize=(7.0 * max(1, len(plot_models)), 6.0),
            constrained_layout=True,
        )
        axes_arr = np.atleast_1d(axes).ravel()
        if use_signed_disagreement:
            if labels_proxy_mode:
                print(
                    f"[scene-map] scene={scene_id} uses proxy/degenerate labels; "
                    "pixel-error switched to cross-model signed disagreement."
                )
            elif mode == "signed_disagreement":
                print(
                    f"[scene-map] scene={scene_id} using cross-model signed disagreement "
                    "(forced by --error-map-mode=signed_disagreement)."
                )
            last_err = None
            plot_stack = np.stack([plot_prob_small_by_model[m] for m in plot_models], axis=0)
            for i, model in enumerate(plot_models):
                p = plot_prob_small_by_model[model]
                err = np.full(label_small.shape, np.nan, dtype=np.float32)
                if len(plot_models) > 1:
                    if plot_stack.shape[0] > 2:
                        ref = np.nanmean(np.delete(plot_stack, i, axis=0), axis=0)
                    else:
                        ref = plot_stack[1 - i]
                    valid = np.isfinite(p) & np.isfinite(ref)
                    err[valid] = np.clip(p[valid] - ref[valid], -1.0, 1.0)
                    title = f"{model.upper()} bias vs peers"
                    last_err = axes_arr[i].imshow(err, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
                else:
                    valid = np.isfinite(p)
                    err[valid] = 4.0 * p[valid] * (1.0 - p[valid])
                    title = f"{model.upper()} uncertainty"
                    last_err = axes_arr[i].imshow(err, cmap="Reds", vmin=0.0, vmax=1.0)
                axes_arr[i].set_title(title)
                axes_arr[i].axis("off")
                _add_map_decorations(axes_arr[i], scene_text=scene_text, width_m=crop_width_m)
            if last_err is not None:
                cbar = fig.colorbar(last_err, ax=axes_arr.tolist(), shrink=0.88)
                if len(plot_models) > 1:
                    cbar.set_label("Signed disagreement p - p_ref (red=higher, blue=lower)")
                else:
                    cbar.set_label("Model uncertainty 4p(1-p)")
        elif mode == "categorical":
            err_cmap = ListedColormap(["#bdbdbd", "#2166ac", "#b2182b", "#fdae61", "#1a9850"])
            err_legend = [
                Patch(facecolor="#bdbdbd", label="Unlabeled/invalid"),
                Patch(facecolor="#2166ac", label="TN"),
                Patch(facecolor="#b2182b", label="FP"),
                Patch(facecolor="#fdae61", label="FN"),
                Patch(facecolor="#1a9850", label="TP"),
            ]
            for i, model in enumerate(plot_models):
                p = prob_small_by_model[model]
                valid = valid_label_small & np.isfinite(p)
                pred = p >= float(thresholds.get(model, 0.8))
                cat = np.zeros(label_small.shape, dtype=np.uint8)
                cat[valid & (label_small == 0) & (~pred)] = 1
                cat[valid & (label_small == 0) & pred] = 2
                cat[valid & (label_small == 1) & (~pred)] = 3
                cat[valid & (label_small == 1) & pred] = 4
                axes_arr[i].imshow(cat, cmap=err_cmap, vmin=0, vmax=4)
                axes_arr[i].set_title(f"{model.upper()} error map")
                axes_arr[i].axis("off")
                _add_map_decorations(axes_arr[i], scene_text=scene_text, width_m=crop_width_m)
            fig.legend(
                handles=err_legend,
                loc="lower center",
                ncol=5,
                frameon=False,
                bbox_to_anchor=(0.5, -0.02),
            )
        else:
            last_err = None
            for i, model in enumerate(plot_models):
                p = prob_small_by_model[model]
                valid = valid_label_small & np.isfinite(p)
                err = np.full(label_small.shape, np.nan, dtype=np.float32)
                err[valid] = np.abs(p[valid] - label_small[valid].astype(np.float32))
                last_err = axes_arr[i].imshow(err, cmap="Reds", vmin=0.0, vmax=1.0)
                axes_arr[i].set_title(f"{model.upper()} absolute error")
                axes_arr[i].axis("off")
                _add_map_decorations(axes_arr[i], scene_text=scene_text, width_m=crop_width_m)
            if last_err is not None:
                cbar = fig.colorbar(last_err, ax=axes_arr.tolist(), shrink=0.88)
                cbar.set_label("Absolute error |p - y|")
        _save_fig(
            fig,
            out_dir / f"{scene_id}_pixel_error",
            png_dpi=int(png_dpi),
            png_compress_level=int(png_compress_level),
            png_min_size_mb=float(png_min_size_mb),
            png_target_size_mb=float(png_target_size_mb),
            png_target_tolerance=float(png_target_tolerance),
        )
    return metrics_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-scene prospectivity and pixel-error maps for selected Sentinel-2 scenes."
    )
    parser.add_argument(
        "--config",
        default="paper_runs/configs/south_kazakhstan_region_paper_run.json",
    )
    parser.add_argument(
        "--search-csv",
        default="paper_runs/copernicus/south_kazakhstan_region/search_20260319T070227Z.csv",
    )
    parser.add_argument(
        "--scene-id",
        action="append",
        required=True,
        help="Scene ID to process (repeat flag for multiple scenes).",
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download missing scene ZIPs using CDSE credentials.",
    )
    parser.add_argument(
        "--username",
        default=os.getenv("CDSE_USERNAME", ""),
        help="CDSE username (or CDSE_USERNAME env var).",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("CDSE_PASSWORD", ""),
        help="CDSE password (or CDSE_PASSWORD env var).",
    )
    parser.add_argument(
        "--out-root",
        default="paper_runs/runs/south_kazakhstan_region/scenes",
    )
    parser.add_argument(
        "--out-fig-dir",
        default="figures/paper_results/scenes",
    )
    parser.add_argument(
        "--models",
        default="rf,xgb",
        help="Comma-separated models for map figures.",
    )
    parser.add_argument(
        "--map-size",
        type=int,
        default=1600,
        help="Downsample target for maps (square). Use <=0 with --full-scale for native resolution.",
    )
    parser.add_argument(
        "--full-scale",
        action="store_true",
        help="Render maps at native raster resolution instead of downsampled map-size.",
    )
    parser.add_argument(
        "--scene-workers",
        type=int,
        default=1,
        help="Threaded worker count for per-scene processing.",
    )
    parser.add_argument(
        "--ml-anomaly-threshold",
        type=float,
        default=0.95,
        help="Threshold used for ml_anomaly prospectivity (top-ranked fraction).",
    )
    parser.add_argument(
        "--saturation-eps",
        type=float,
        default=1.0e-3,
        help="If p98-p2 falls below this value, display rank-normalized prospectivity.",
    )
    parser.add_argument(
        "--paper-eval-json",
        default="",
        help="Optional paper-eval metrics JSON to source QML thresholds (defaults to config path).",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing scene stack/predictions when present (recommended for reruns).",
    )
    parser.add_argument(
        "--auto-fix-degenerate-labels",
        action="store_true",
        help="Auto-repair scene labels when positive coverage is nearly full-scene.",
    )
    parser.add_argument(
        "--max-positive-fraction",
        type=float,
        default=0.35,
        help="Upper bound for positive pixel fraction when auto-fixing degenerate labels.",
    )
    parser.add_argument(
        "--qml-train-samples",
        type=int,
        default=0,
        help="Override max train samples for QML scene-map model fitting (0 = use config).",
    )
    parser.add_argument(
        "--qml-batch-size",
        type=int,
        default=256,
        help="Batch size for QML inference over downsampled scene features.",
    )
    parser.add_argument(
        "--error-map-mode",
        default="absolute",
        choices=["absolute", "categorical", "signed_disagreement"],
        help="Error map style: absolute|categorical|signed_disagreement.",
    )
    parser.add_argument(
        "--png-dpi",
        type=int,
        default=400,
        help="PNG export DPI for scene figures.",
    )
    parser.add_argument(
        "--png-compress-level",
        type=int,
        default=0,
        help="PNG compression level (0-9). Lower values create larger files.",
    )
    parser.add_argument(
        "--png-min-size-mb",
        type=float,
        default=0.0,
        help="If >0, iteratively increase DPI until each PNG reaches at least this size in MB.",
    )
    parser.add_argument(
        "--png-target-size-mb",
        type=float,
        default=0.0,
        help="If >0, tune DPI so each PNG lands near this target size (in MB).",
    )
    parser.add_argument(
        "--png-target-tolerance",
        type=float,
        default=0.15,
        help="Allowed relative error for --png-target-size-mb (e.g., 0.15 = +/-15%).",
    )
    args = parser.parse_args()

    _bootstrap_path()
    from xaigis.config import load_config
    from xaigis.features import prepare_features
    from xaigis.labels import rasterize_labels
    from xaigis.modeling import predict_rasters

    cfg = load_config(args.config)
    base_paths = cfg["paths"]
    base_models_dir: Path = base_paths["models_dir"]
    base_metrics_json: Path = base_paths["metrics_json"]
    if not base_models_dir.exists():
        raise FileNotFoundError(f"Trained models directory missing: {base_models_dir}")
    if not base_metrics_json.exists():
        raise FileNotFoundError(f"Base metrics JSON missing: {base_metrics_json}")
    base_metrics = json.loads(base_metrics_json.read_text(encoding="utf-8"))
    thresholds = {
        str(k).strip().lower(): float(v)
        for k, v in dict(base_metrics.get("model_thresholds", {})).items()
    }
    paper_eval_json = (
        Path(args.paper_eval_json).resolve()
        if str(args.paper_eval_json).strip()
        else Path(base_paths.get("paper_eval_json", "")).resolve()
    )
    thresholds.update(_load_qml_thresholds(paper_eval_json))
    thresholds["ml_anomaly"] = float(np.clip(float(args.ml_anomaly_threshold), 1.0e-6, 1.0 - 1.0e-6))

    search_rows = _read_search_csv(Path(args.search_csv).resolve())
    requested = list(dict.fromkeys([str(s).strip() for s in args.scene_id if str(s).strip()]))
    if not requested:
        raise ValueError("No scene IDs provided.")

    scene_download_dir = (base_paths["safe_zip"].parent if base_paths.get("safe_zip") else Path(args.search_csv).resolve().parent / "downloads").resolve()
    scene_safe_dir = (scene_download_dir.parent / "SAFE").resolve()
    out_root = Path(args.out_root).resolve()
    out_fig_dir = Path(args.out_fig_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    plot_models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    if not plot_models:
        raise ValueError("No models selected. Provide --models, e.g. --models rf,qml_qnn")
    classical_models, qml_models, aux_models = _split_requested_models(plot_models)
    models_dir_for_prediction = base_models_dir
    if classical_models:
        models_dir_for_prediction = _prepare_model_subset_dir(
            base_models_dir=base_models_dir,
            selected_models=classical_models,
            out_root=out_root,
        )
    qml_predictors = _train_qml_predictors(
        cfg=cfg,
        qml_models=qml_models,
        thresholds=thresholds,
        qml_train_samples_override=int(args.qml_train_samples),
    )

    token = ""
    scene_paths: list[tuple[str, Path, Path]] = []
    for scene_id in requested:
        zip_path = scene_download_dir / f"{scene_id}.zip"
        safe_path = scene_safe_dir / f"{scene_id}.SAFE"
        if not zip_path.exists() and not safe_path.exists():
            if not args.download_missing:
                raise FileNotFoundError(
                    f"Scene data missing for {scene_id}. Expected {zip_path} or {safe_path}. "
                    "Re-run with --download-missing."
                )
            row = search_rows.get(scene_id)
            if not row:
                raise KeyError(f"Scene {scene_id} not found in search CSV: {args.search_csv}")
            product_href = str(row.get("product_href", "")).strip()
            if not product_href:
                raise ValueError(f"Scene {scene_id} has no product_href in search CSV.")
            if not token:
                if not args.username or not args.password:
                    raise RuntimeError(
                        "CDSE credentials are required to download missing scenes. "
                        "Set CDSE_USERNAME/CDSE_PASSWORD or pass --username/--password."
                    )
            print(f"[multi-scene] downloading missing scene: {scene_id}")
            token = _download_with_retries(
                url=product_href,
                out_file=zip_path,
                token=token,
                username=args.username,
                password=args.password,
            )
        scene_paths.append((scene_id, zip_path, safe_path))

    scene_workers = max(1, int(args.scene_workers))
    if scene_workers > 1:
        print(f"[multi-scene] scene workers: {scene_workers} (threaded)")
    plot_lock = threading.Lock() if scene_workers > 1 else None
    qml_predict_locks = {name: threading.Lock() for name in qml_models}

    def _process_scene(scene_id: str, zip_path: Path, safe_path: Path) -> list[dict[str, Any]]:
        scene_root = out_root / scene_id
        scene_outputs = scene_root / "outputs"
        scene_artifacts = scene_root / "artifacts"
        scene_predictions = scene_artifacts / "predictions"
        scene_outputs.mkdir(parents=True, exist_ok=True)
        scene_artifacts.mkdir(parents=True, exist_ok=True)

        scene_cfg = deepcopy(cfg)
        scene_cfg["paths"]["safe_zip"] = zip_path
        scene_cfg["paths"]["safe_dir"] = safe_path
        scene_cfg["paths"]["safe_zips"] = []
        scene_cfg["paths"]["safe_dirs"] = []
        scene_cfg["paths"]["work_dir"] = scene_outputs
        scene_cfg["paths"]["artifacts_dir"] = scene_artifacts
        scene_cfg["paths"]["feature_stack_tif"] = scene_outputs / "S2_feature_stack_10m.tif"
        scene_cfg["paths"]["feature_names_json"] = scene_outputs / "feature_names.json"
        scene_cfg["paths"]["label_tif"] = scene_outputs / "h2_label_poly_10m.tif"
        scene_cfg["paths"]["predictions_dir"] = scene_predictions
        # Reuse trained models/thresholds from base run.
        scene_cfg["paths"]["models_dir"] = models_dir_for_prediction
        scene_cfg["paths"]["metrics_json"] = base_metrics_json

        stack_tif = scene_cfg["paths"]["feature_stack_tif"]
        names_json = scene_cfg["paths"]["feature_names_json"]
        if bool(args.reuse_existing) and stack_tif.exists() and names_json.exists():
            print(f"[multi-scene] scene={scene_id} prepare_features (reuse existing)")
        else:
            print(f"[multi-scene] scene={scene_id} prepare_features")
            prepare_features(scene_cfg)
        map_shape = _resolve_map_shape(
            stack_tif=scene_cfg["paths"]["feature_stack_tif"],
            map_size=int(args.map_size),
            full_scale=bool(args.full_scale),
        )
        print(
            f"[multi-scene] scene={scene_id} map-shape={int(map_shape[1])}x{int(map_shape[0])}"
        )

        print(f"[multi-scene] scene={scene_id} rasterize_labels")
        label_stats = rasterize_labels(scene_cfg)
        raw_pos_frac = _safe_div(
            float(label_stats.get("positives", 0)),
            float(label_stats.get("total_pixels", 0)),
        )
        labels_proxy_mode = raw_pos_frac > float(args.max_positive_fraction)
        if labels_proxy_mode:
            print(
                f"[multi-scene] scene={scene_id} warning: raw positive coverage "
                f"{raw_pos_frac:.2%} exceeds max-positive-fraction={float(args.max_positive_fraction):.2%}; "
                "treating labels as proxy."
            )
        if bool(args.auto_fix_degenerate_labels):
            _repair_degenerate_scene_labels(
                label_tif=scene_cfg["paths"]["label_tif"],
                uncertain_value=int(cfg.get("labels", {}).get("uncertain_value", 255)),
                negative_exclusion_buffer_px=int(
                    cfg.get("labels", {}).get("negative_exclusion_buffer_px", 0)
                ),
                max_positive_fraction=float(args.max_positive_fraction),
            )

        if classical_models:
            missing_probs = [
                m
                for m in classical_models
                if not (scene_cfg["paths"]["predictions_dir"] / f"{m}_prob.tif").exists()
            ]
            if bool(args.reuse_existing) and not missing_probs:
                print(
                    f"[multi-scene] scene={scene_id} predict_rasters (reuse existing "
                    f"{','.join(classical_models)})"
                )
            else:
                if missing_probs:
                    print(
                        f"[multi-scene] scene={scene_id} missing predictions for: "
                        f"{','.join(missing_probs)}"
                    )
                print(f"[multi-scene] scene={scene_id} predict_rasters")
                predict_rasters(scene_cfg)

        extra_prob_small: dict[str, np.ndarray] = {}
        if "ml_anomaly" in aux_models:
            print(f"[multi-scene] scene={scene_id} ml_anomaly_predict_small")
            extra_prob_small["ml_anomaly"] = _predict_ml_anomaly_prob_small(
                stack_tif=scene_cfg["paths"]["feature_stack_tif"],
                map_shape=map_shape,
                seed=int(cfg.get("training", {}).get("random_seed", 42)),
            )
        for qml_model in qml_models:
            predictor = qml_predictors.get(qml_model)
            if predictor is None:
                print(f"[multi-scene] warning: qml predictor not available for {qml_model}")
                continue
            print(f"[multi-scene] scene={scene_id} qml_predict_small model={qml_model}")
            qml_lock = qml_predict_locks.get(qml_model)
            if qml_lock is None:
                qml_prob_small = _predict_qml_prob_small(
                    stack_tif=scene_cfg["paths"]["feature_stack_tif"],
                    predictor=predictor,
                    map_shape=map_shape,
                    batch_size=int(args.qml_batch_size),
                )
            else:
                with qml_lock:
                    qml_prob_small = _predict_qml_prob_small(
                        stack_tif=scene_cfg["paths"]["feature_stack_tif"],
                        predictor=predictor,
                        map_shape=map_shape,
                        batch_size=int(args.qml_batch_size),
                    )
            extra_prob_small[qml_model] = qml_prob_small

        rows = _make_scene_maps(
            scene_id=scene_id,
            stack_tif=scene_cfg["paths"]["feature_stack_tif"],
            label_tif=scene_cfg["paths"]["label_tif"],
            pred_dir=scene_cfg["paths"]["predictions_dir"],
            thresholds=thresholds,
            models=plot_models,
            map_shape=map_shape,
            out_dir=out_fig_dir,
            extra_prob_small_by_model=extra_prob_small,
            error_map_mode=str(args.error_map_mode),
            labels_proxy_mode=bool(labels_proxy_mode),
            saturation_eps=float(args.saturation_eps),
            png_dpi=int(args.png_dpi),
            png_compress_level=int(args.png_compress_level),
            png_min_size_mb=float(args.png_min_size_mb),
            png_target_size_mb=float(args.png_target_size_mb),
            png_target_tolerance=float(args.png_target_tolerance),
            plot_lock=plot_lock,
        )
        print(f"[multi-scene] scene={scene_id} figures -> {out_fig_dir}")
        return rows

    rows_by_scene: dict[str, list[dict[str, Any]]] = {}
    if scene_workers <= 1:
        for scene_id, zip_path, safe_path in scene_paths:
            rows_by_scene[scene_id] = _process_scene(scene_id, zip_path, safe_path)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=scene_workers) as pool:
            future_to_scene = {
                pool.submit(_process_scene, scene_id, zip_path, safe_path): scene_id
                for scene_id, zip_path, safe_path in scene_paths
            }
            for future in concurrent.futures.as_completed(future_to_scene):
                scene_id = future_to_scene[future]
                rows_by_scene[scene_id] = future.result()

    all_rows: list[dict[str, Any]] = []
    for scene_id in requested:
        all_rows.extend(rows_by_scene.get(scene_id, []))

    summary_csv = out_root / "selected_scene_map_metrics.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "scene_id",
            "model",
            "threshold",
            "valid_pixels",
            "tn",
            "fp",
            "fn",
            "tp",
            "precision_at_threshold",
            "recall_at_threshold",
            "balanced_accuracy_at_threshold",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    summary_json = out_root / "selected_scene_map_metrics.json"
    summary_json.write_text(
        json.dumps(
            {
                "scene_ids": requested,
                "models": plot_models,
                "out_fig_dir": str(out_fig_dir),
                "rows": all_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[multi-scene] wrote metrics: {summary_csv}")
    print(f"[multi-scene] wrote metrics: {summary_json}")


if __name__ == "__main__":
    main()
