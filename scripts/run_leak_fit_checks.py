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
    parser = argparse.ArgumentParser(description="Run leakage and fit diagnostics on prepared dataset.")
    parser.add_argument("--config", default="paper_runs/configs/south_kazakhstan_region_paper_run.json")
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Allow placeholder paths/labels (for smoke tests only).",
    )
    args = parser.parse_args()

    _bootstrap_path()
    from xaigis.config import find_placeholder_issues, load_config
    from xaigis.diagnostics import run_leak_fit_checks

    cfg = load_config(args.config)
    placeholder_issues = find_placeholder_issues(cfg)
    if placeholder_issues:
        for issue in placeholder_issues:
            print(f"[diag] placeholder-audit: {issue}")
        if not args.allow_placeholders:
            raise SystemExit(
                "[diag] placeholder audit failed; provide real scene/labels "
                "or rerun with --allow-placeholders for non-research smoke testing."
            )
    summary = run_leak_fit_checks(cfg)
    print("[diag] completed")
    print(f"[diag] best model: {summary.get('best_model_by_test_roc_auc')}")
    print(f"[diag] fit regime: {summary.get('fit_regime')}")
    print(f"[diag] leakage risk: {summary.get('data_leakage_risk')}")


if __name__ == "__main__":
    main()
