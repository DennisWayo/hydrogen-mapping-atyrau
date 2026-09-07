#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Local imports from repo.
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_multisource_feature_stack import build_multisource_feature_stack
from xaigis.config import load_config
from xaigis.dataset import sample_dataset
from xaigis.explain import explain_models
from xaigis.labels import rasterize_labels
from xaigis.modeling import predict_rasters, train_models
from xaigis.report import build_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full ex_01 multisource fusion workflow (DEM + Sentinel + Landsat + geology + geochem)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "paper_runs/ex_01/configs/ex_01_fusion_run.json",
        help="Path to fusion config JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config)

    print("[fusion-run] step 1/7: build fused feature stack")
    build_multisource_feature_stack(cfg)

    print("[fusion-run] step 2/7: rasterize labels")
    rasterize_labels(cfg)

    print("[fusion-run] step 3/7: sample dataset")
    sample_dataset(cfg)

    print("[fusion-run] step 4/7: train models")
    train_models(cfg)

    print("[fusion-run] step 5/7: predict rasters")
    predict_rasters(cfg)

    print("[fusion-run] step 6/7: explain models")
    explain_models(cfg)

    print("[fusion-run] step 7/7: build report")
    build_report(cfg)

    print("[fusion-run] completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
