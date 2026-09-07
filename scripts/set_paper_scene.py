#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update paper-run config with selected Sentinel-2 scene ID."
    )
    parser.add_argument("--scene-id", required=True, help="Scene ID, e.g. S2C_MSIL2A_...")
    parser.add_argument("--config", default="paper_runs/configs/south_kazakhstan_region_paper_run.json")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    scene = args.scene_id.strip()

    cfg["paths"]["safe_zip"] = f"../copernicus/south_kazakhstan_region/downloads/{scene}.zip"
    cfg["paths"]["safe_dir"] = f"../copernicus/south_kazakhstan_region/SAFE/{scene}.SAFE"

    config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Updated {config_path}")
    print(f"safe_zip -> {cfg['paths']['safe_zip']}")
    print(f"safe_dir -> {cfg['paths']['safe_dir']}")


if __name__ == "__main__":
    main()
