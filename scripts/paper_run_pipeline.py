#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paper-focused ML + QML workflow with outputs isolated under paper_runs."
    )
    parser.add_argument("--config", default="paper_runs/configs/south_kazakhstan_region_paper_run.json")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip feature/label/dataset generation.")
    parser.add_argument("--skip-ml", action="store_true", help="Skip classical train/predict/explain/report.")
    parser.add_argument("--skip-qml", action="store_true", help="Skip unified ML/QML evaluation.")
    parser.add_argument("--skip-diagnostics", action="store_true", help="Skip leakage and fit diagnostics.")
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Allow placeholder paths/labels (for smoke tests only).",
    )
    args = parser.parse_args()

    _bootstrap_path()
    from xaigis.config import find_placeholder_issues, load_config
    from xaigis.dataset import sample_dataset
    from xaigis.explain import explain_models
    from xaigis.features import prepare_features
    from xaigis.labels import rasterize_labels
    from xaigis.modeling import predict_rasters, train_models
    from xaigis.paper_eval import run_unified_paper_eval
    from xaigis.diagnostics import run_leak_fit_checks
    from xaigis.report import build_report

    cfg = load_config(args.config)
    print(f"[paper-run] config: {cfg['__config_path__']}")
    print(f"[paper-run] outputs root: {cfg['paths']['work_dir'].parent}")
    placeholder_issues = find_placeholder_issues(cfg)
    if placeholder_issues:
        for issue in placeholder_issues:
            print(f"[paper-run] placeholder-audit: {issue}")
        if not args.allow_placeholders:
            raise SystemExit(
                "[paper-run] placeholder audit failed; provide real scene/labels "
                "or rerun with --allow-placeholders for non-research smoke testing."
            )

    if not args.skip_prepare:
        print("[paper-run] step: prepare_features")
        prepare_features(cfg)
        print("[paper-run] step: rasterize_labels")
        rasterize_labels(cfg)
        print("[paper-run] step: sample_dataset")
        sample_dataset(cfg)

    if not args.skip_ml:
        print("[paper-run] step: train_models")
        train_models(cfg)
        print("[paper-run] step: predict_rasters")
        predict_rasters(cfg)
        print("[paper-run] step: explain_models")
        explain_models(cfg)
        print("[paper-run] step: build_report")
        build_report(cfg)

    if not args.skip_qml:
        print("[paper-run] step: run_unified_paper_eval")
        run_unified_paper_eval(cfg)

    if not args.skip_diagnostics:
        print("[paper-run] step: run_leak_fit_checks")
        run_leak_fit_checks(cfg)

    print("[paper-run] completed")


if __name__ == "__main__":
    main()
