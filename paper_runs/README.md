# Paper Runs

This directory stores manuscript-specific paper-run bundles that are small
enough to keep with the repository.

## Included

- South Kazakhstan archived paper snapshot: configs, AOI/labels, Copernicus
  query exports, and generated outputs.
- `ex_01/` - Northern Kazakhstan leakage-aware natural-hydrogen
  prospectivity-screening study, small figure pack, tables, and
  reproducibility scripts.

## Excluded

Large runtime products are intentionally not tracked:

- raw Sentinel/Landsat/DEM rasters
- generated GeoTIFF feature stacks
- model binaries
- sampled datasets
- temporary run outputs

These products can be regenerated or transferred locally when needed, but they
should not be pushed to GitHub.

## What is reproducible here

This checkout supports exact artifact reproducibility for the saved paper outputs in `paper_runs/runs/south_kazakhstan_region/` using checksum verification.

This checkout does **not** include the paper orchestration scripts previously used to produce those outputs (for example `paper_run_pipeline.py`, `run_ml_qml_eval.py`, `copernicus_s2_pull.py`, `set_paper_scene.py`). Because of that, full raw-data reruns are not executable from this folder alone.

## Exact verification (byte-for-byte)

Run from repository root:

```bash
python3 scripts/verify_paper_runs_repro.py
```

This verifies fixed SHA-256 hashes for:

- `paper_runs/configs/*.json`
- `paper_runs/regions/*.geojson`
- `paper_runs/copernicus/south_kazakhstan_region/{search_*.json,search_*.csv,last_query_params.json}`
- Key result artifacts in `paper_runs/runs/south_kazakhstan_region/{artifacts,analysis,analysis_spatial,scenes}`

If hashes match, the paper snapshot is exact for the checked files.

## Strict runtime-input check

To also require runtime inputs referenced by config (`safe_zip`, `safe_dir`, `geology_geojson`) to exist on disk:

```bash
python3 scripts/verify_paper_runs_repro.py --strict-inputs
```

## Rerun prerequisites (when full private scripts are available)

Use `paper_runs/configs/south_kazakhstan_region_paper_run.json` and ensure:

- `paths.safe_zip` points to a downloaded Sentinel-2 L2A ZIP
- `paths.safe_dir` points to the unzipped `.SAFE` directory
- `paths.geology_geojson` points to finalized expert labels

Current config target scene:

- `S2B_MSIL2A_20251018T062759_N0511_R077_T42TWR_20251018T074753`

Note: the corresponding `downloads/` and `SAFE/` directories are not present in this checkout.

## CPU/QML settings in saved configs

The paper config files retain performance controls used during the archived runs:

- `performance.max_threads`
- `performance.parallel_workers`
- `{training,prediction,paper_eval,leak_fit_checks}.max_threads`
- `{paper_eval,leak_fit_checks}.parallel_workers`
- `paper_eval.qml.{device,shots,kernel_probability,kernel_cache_mb}`
