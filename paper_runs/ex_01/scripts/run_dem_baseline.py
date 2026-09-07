#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make local src importable when running from repository checkout.
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dem_feature_stack import build_dem_feature_stack
from xaigis.config import load_config
from xaigis.dataset import sample_dataset
from xaigis.explain import explain_models
from xaigis.labels import rasterize_labels
from xaigis.modeling import predict_rasters, train_models
from xaigis.report import build_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DEM-only baseline workflow for paper_runs/ex_01."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "paper_runs/ex_01/configs/ex_01_paper_run.json",
        help="Path to run config JSON.",
    )
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Skip raster prediction stage.",
    )
    parser.add_argument(
        "--skip-explain",
        action="store_true",
        help="Skip feature-importance stage.",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip report generation stage.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config)

    print("[baseline] step 1/7: build DEM feature stack")
    build_dem_feature_stack(cfg)

    print("[baseline] step 2/7: rasterize labels")
    rasterize_labels(cfg)

    print("[baseline] step 3/7: sample dataset")
    sample_dataset(cfg)

    print("[baseline] step 4/7: train models")
    train_models(cfg)

    if not args.skip_predict:
        print("[baseline] step 5/7: predict rasters")
        predict_rasters(cfg)
    else:
        print("[baseline] step 5/7: predict rasters (skipped)")

    if not args.skip_explain:
        print("[baseline] step 6/7: explain models")
        explain_models(cfg)
    else:
        print("[baseline] step 6/7: explain models (skipped)")

    if not args.skip_report:
        print("[baseline] step 7/7: build report")
        build_report(cfg)
    else:
        print("[baseline] step 7/7: build report (skipped)")

    print("[baseline] completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
