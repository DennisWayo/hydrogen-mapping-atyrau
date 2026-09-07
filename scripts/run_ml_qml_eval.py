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
    parser = argparse.ArgumentParser(description="Run unified ML/QML paper evaluation.")
    parser.add_argument("--config", default="paper_runs/configs/south_kazakhstan_region_paper_run.json")
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Allow placeholder paths/labels (for smoke tests only).",
    )
    args = parser.parse_args()

    _bootstrap_path()
    from xaigis.config import find_placeholder_issues, load_config
    from xaigis.paper_eval import run_unified_paper_eval

    cfg = load_config(args.config)
    placeholder_issues = find_placeholder_issues(cfg)
    if placeholder_issues:
        for issue in placeholder_issues:
            print(f"[paper-eval] placeholder-audit: {issue}")
        if not args.allow_placeholders:
            raise SystemExit(
                "[paper-eval] placeholder audit failed; provide real scene/labels "
                "or rerun with --allow-placeholders for non-research smoke testing."
            )
    summary = run_unified_paper_eval(cfg)
    print("[paper-eval] completed")
    print(f"[paper-eval] models: {list(summary.get('models', {}).keys())}")


if __name__ == "__main__":
    main()
