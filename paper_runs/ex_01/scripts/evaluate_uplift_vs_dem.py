#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fusion metrics against DEM baseline and assess measurable gain."
    )
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=Path("paper_runs/ex_01/runs/ex_01/artifacts/metrics.json"),
        help="Baseline DEM metrics JSON.",
    )
    parser.add_argument(
        "--fusion-metrics",
        type=Path,
        default=Path("paper_runs/ex_01/runs/ex_01_fusion/artifacts/metrics.json"),
        help="Fusion metrics JSON.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("paper_runs/ex_01/runs/ex_01_fusion/analysis/uplift_vs_dem.json"),
        help="Output uplift evaluation JSON.",
    )
    parser.add_argument("--model", type=str, default="xgb", help="Model key to compare.")
    parser.add_argument(
        "--fusion-importance-csv",
        type=Path,
        default=Path("paper_runs/ex_01/runs/ex_01_fusion/artifacts/feature_importance.csv"),
        help="Fusion feature importance CSV (for leakage checks).",
    )
    parser.add_argument(
        "--min-roc-auc-gain",
        type=float,
        default=0.02,
        help="Minimum ROC-AUC gain required for trust gate.",
    )
    parser.add_argument(
        "--min-pr-auc-gain",
        type=float,
        default=0.005,
        help="Minimum PR-AUC gain required for trust gate.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _metrics_for_model(payload: dict, model: str) -> dict:
    return (payload.get("models") or {}).get(model) or {}


def _load_importance(path: Path, model: str) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("model", "")).strip() == model:
                rows.append(row)
    return rows


def main() -> int:
    args = _parse_args()
    baseline = _load_json(args.baseline_metrics)
    fusion = _load_json(args.fusion_metrics)

    b = _metrics_for_model(baseline, args.model)
    f = _metrics_for_model(fusion, args.model)
    if not b or not f:
        raise ValueError(f"Model '{args.model}' missing from baseline or fusion metrics.")

    delta_roc = float(f.get("roc_auc", 0.0)) - float(b.get("roc_auc", 0.0))
    delta_pr = float(f.get("pr_auc", 0.0)) - float(b.get("pr_auc", 0.0))
    delta_f1 = float(f.get("f1", 0.0)) - float(b.get("f1", 0.0))

    raw_gain_pass = (delta_roc >= args.min_roc_auc_gain) and (delta_pr >= args.min_pr_auc_gain)

    importance_rows = _load_importance(args.fusion_importance_csv, model=args.model)
    leakage_flags: list[str] = []
    if importance_rows:
        top = sorted(
            importance_rows,
            key=lambda r: float(r.get("importance", 0.0)),
            reverse=True,
        )[0]
        top_name = str(top.get("feature", ""))
        top_imp = float(top.get("importance", 0.0))
        if top_name == "GEOLOGY_MASK" and top_imp >= 0.95:
            leakage_flags.append(
                "Top feature GEOLOGY_MASK dominates importance (>=0.95), indicating label leakage risk."
            )
        if top_name.startswith("GEOCHEM_") and top_imp >= 0.95:
            leakage_flags.append(
                "Top feature is direct geochem proxy (>=0.95), indicating potential target-leakage risk."
            )

    if float(f.get("roc_auc", 0.0)) >= 0.999 and float(f.get("pr_auc", 0.0)) >= 0.999:
        leakage_flags.append("Near-perfect ROC-AUC/PR-AUC detected; likely leakage or non-independent labels.")

    pass_gate = raw_gain_pass and (len(leakage_flags) == 0)

    out = {
        "model": args.model,
        "baseline": b,
        "fusion": f,
        "deltas": {
            "roc_auc": delta_roc,
            "pr_auc": delta_pr,
            "f1": delta_f1,
        },
        "raw_gain_gate_passed": raw_gain_pass,
        "gain_thresholds": {
            "min_roc_auc_gain": args.min_roc_auc_gain,
            "min_pr_auc_gain": args.min_pr_auc_gain,
        },
        "leakage_flags": leakage_flags,
        "trust_gate_passed": pass_gate,
        "decision": (
            "Fusion shows measurable gain over DEM baseline and passes leakage checks."
            if pass_gate
            else "Fusion does not pass trust gate (gain and/or leakage checks); keep as experimental."
        ),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as fjson:
        json.dump(out, fjson, indent=2)

    print(f"[uplift] baseline ROC-AUC={b.get('roc_auc', 0):.4f}, PR-AUC={b.get('pr_auc', 0):.4f}")
    print(f"[uplift] fusion   ROC-AUC={f.get('roc_auc', 0):.4f}, PR-AUC={f.get('pr_auc', 0):.4f}")
    print(f"[uplift] delta ROC-AUC={delta_roc:+.4f}, delta PR-AUC={delta_pr:+.4f}, delta F1={delta_f1:+.4f}")
    if leakage_flags:
        print("[uplift] leakage flags:")
        for flag in leakage_flags:
            print(f"  - {flag}")
    print(f"[uplift] trust gate passed: {pass_gate}")
    print(f"[uplift] saved: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
