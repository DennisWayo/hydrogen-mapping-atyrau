#!/usr/bin/env python3
"""Search (and optionally download) Copernicus Sentinel-2 L2A scenes for an AOI."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAC_SEARCH_URL = "https://stac.dataspace.copernicus.eu/v1/search"
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)


def read_geometry(aoi_path: Path) -> dict[str, Any]:
    data = json.loads(aoi_path.read_text(encoding="utf-8"))
    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
        if not features:
            raise ValueError("AOI FeatureCollection has no features.")
        geometry = features[0].get("geometry")
    elif data.get("type") == "Feature":
        geometry = data.get("geometry")
    else:
        geometry = data
    if not geometry or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
        raise ValueError("AOI must contain Polygon or MultiPolygon geometry.")
    return geometry


def http_json_post(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
    all_headers = {"Content-Type": "application/json"}
    if headers:
        all_headers.update(headers)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=all_headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_access_token(username: str, password: str) -> str:
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
        data = json.loads(resp.read().decode("utf-8"))
    token = data.get("access_token")
    if not token:
        raise RuntimeError("CDSE token response did not include access_token.")
    return token


def scene_rows(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in features:
        props = item.get("properties", {})
        assets = item.get("assets", {})
        rows.append(
            {
                "id": item.get("id", ""),
                "datetime": props.get("datetime", ""),
                "cloud_cover": props.get("eo:cloud_cover", ""),
                "platform": props.get("platform", ""),
                "orbit_state": props.get("sat:orbit_state", ""),
                "mgrs_tile": props.get("s2:mgrs_tile", ""),
                "product_href": assets.get("Product", {}).get("href", ""),
                "thumbnail_href": assets.get("thumbnail", {}).get("href", ""),
                "safe_manifest_href": assets.get("safe_manifest", {}).get("href", ""),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "datetime",
        "cloud_cover",
        "platform",
        "orbit_state",
        "mgrs_tile",
        "product_href",
        "thumbnail_href",
        "safe_manifest_href",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stream_download(url: str, out_file: Path, token: str) -> None:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open("wb") as fh:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aoi",
        default="paper_runs/regions/south_kazakhstan_region.geojson",
        help="AOI GeoJSON path (FeatureCollection/Feature/geometry)",
    )
    parser.add_argument("--start-date", default="2025-04-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2025-10-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--max-cloud", type=float, default=20.0, help="Max cloud cover (percent)")
    parser.add_argument("--limit", type=int, default=80, help="Max scenes to request")
    parser.add_argument(
        "--out-dir",
        default="paper_runs/copernicus/south_kazakhstan_region",
        help="Directory for search outputs",
    )
    parser.add_argument(
        "--download-count",
        type=int,
        default=0,
        help="Number of scenes to download (0 means metadata only)",
    )
    parser.add_argument(
        "--download-sort",
        choices=["datetime_desc", "cloud_asc"],
        default="cloud_asc",
        help="Ordering of scenes to download when download-count > 0",
    )
    parser.add_argument(
        "--username",
        default=os.getenv("CDSE_USERNAME", ""),
        help="CDSE username (or use CDSE_USERNAME env var)",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("CDSE_PASSWORD", ""),
        help="CDSE password (or use CDSE_PASSWORD env var)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    geometry = read_geometry(Path(args.aoi).resolve())
    datetime_range = f"{args.start_date}T00:00:00Z/{args.end_date}T23:59:59Z"
    payload: dict[str, Any] = {
        "collections": ["sentinel-2-l2a"],
        "intersects": geometry,
        "datetime": datetime_range,
        "query": {"eo:cloud_cover": {"lt": args.max_cloud}},
        "limit": max(1, min(args.limit, 100)),
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
    }

    print(f"[search] AOI: {args.aoi}")
    print(f"[search] datetime: {datetime_range}")
    print(f"[search] cloud cover < {args.max_cloud}%")

    try:
        result = http_json_post(STAC_SEARCH_URL, payload)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        print(f"[error] STAC query failed: HTTP {exc.code}", file=sys.stderr)
        print(body[:800], file=sys.stderr)
        return 1

    features = result.get("features", [])
    rows = scene_rows(features)
    fetched_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    raw_json_path = out_dir / f"search_{fetched_at}.json"
    rows_csv_path = out_dir / f"search_{fetched_at}.csv"
    params_json_path = out_dir / "last_query_params.json"

    raw_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(rows_csv_path, rows)
    params_json_path.write_text(
        json.dumps(
            {
                "aoi": str(Path(args.aoi).resolve()),
                "start_date": args.start_date,
                "end_date": args.end_date,
                "max_cloud": args.max_cloud,
                "limit": args.limit,
                "datetime": datetime_range,
                "fetched_at_utc": fetched_at,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[search] scenes found: {len(rows)}")
    print(f"[search] wrote: {raw_json_path}")
    print(f"[search] wrote: {rows_csv_path}")
    print(f"[search] wrote: {params_json_path}")

    if args.download_count <= 0:
        return 0

    if not args.username or not args.password:
        print(
            "[download] skipped: set CDSE_USERNAME and CDSE_PASSWORD (or pass --username/--password).",
            file=sys.stderr,
        )
        return 2

    token = get_access_token(args.username, args.password)
    to_download = _ordered_download_rows(rows=rows, mode=args.download_sort)[: args.download_count]
    downloads_dir = out_dir / "downloads"
    for idx, row in enumerate(to_download, start=1):
        product_url = row.get("product_href", "")
        scene_id = row.get("id", f"scene_{idx}")
        if not product_url:
            print(f"[download] {scene_id}: missing Product URL, skipping")
            continue
        out_file = downloads_dir / f"{scene_id}.zip"
        if out_file.exists() and out_file.stat().st_size > 0:
            print(f"[download] {scene_id}: already exists, skipping")
            continue
        print(f"[download] {idx}/{len(to_download)}: {scene_id}")
        try:
            stream_download(product_url, out_file, token)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            print(f"[download] failed {scene_id}: HTTP {exc.code}", file=sys.stderr)
            print(body[:300], file=sys.stderr)
        except Exception as exc:
            print(f"[download] failed {scene_id}: {exc}", file=sys.stderr)
    return 0


def _ordered_download_rows(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode == "cloud_asc":
        def cloud_value(r: dict[str, Any]) -> float:
            try:
                return float(r.get("cloud_cover", 1000.0))
            except Exception:
                return 1000.0

        return sorted(rows, key=lambda r: (cloud_value(r), str(r.get("datetime", ""))), reverse=False)
    return sorted(rows, key=lambda r: str(r.get("datetime", "")), reverse=True)


if __name__ == "__main__":
    raise SystemExit(main())
