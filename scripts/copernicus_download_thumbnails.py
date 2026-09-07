#!/usr/bin/env python3
"""Download Copernicus scene thumbnail images from a search CSV."""

from __future__ import annotations

import argparse
import csv
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def safe_name(scene_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", scene_id).strip("_") or "scene"


def normalize_thumbnail_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    host = (parsed.hostname or "").lower()
    if host.startswith("catalogue-svc.prod-catalogue"):
        parsed = parsed._replace(netloc="datahub.creodias.eu")
        return urllib.parse.urlunsplit(parsed)
    return url


def download(url: str, out_path: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as fh:
            while True:
                chunk = resp.read(1024 * 512)
                if not chunk:
                    break
                fh.write(chunk)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default="paper_runs/copernicus/south_kazakhstan_region/search_20260319T070227Z.csv",
        help="Input scene CSV from copernicus_s2_pull.py",
    )
    parser.add_argument(
        "--out-dir",
        default="paper_runs/copernicus/south_kazakhstan_region/thumbnails",
        help="Output folder for thumbnail images",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of thumbnails to fetch (0 means all)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    if args.limit > 0:
        rows = rows[: args.limit]

    total = len(rows)
    ok = 0
    fail = 0

    manifest_path = out_dir / "download_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(mf, fieldnames=["id", "thumbnail_href", "file", "status", "error"])
        writer.writeheader()

        for idx, row in enumerate(rows, start=1):
            scene_id = row.get("id", f"scene_{idx}")
            raw_url = row.get("thumbnail_href", "")
            url = normalize_thumbnail_url(raw_url)
            file_name = f"{safe_name(scene_id)}.jpg"
            out_file = out_dir / file_name

            if not url:
                fail += 1
                writer.writerow(
                    {"id": scene_id, "thumbnail_href": "", "file": file_name, "status": "failed", "error": "missing_url"}
                )
                continue

            if out_file.exists() and out_file.stat().st_size > 0:
                ok += 1
                writer.writerow(
                    {"id": scene_id, "thumbnail_href": url, "file": file_name, "status": "exists", "error": ""}
                )
                continue

            print(f"[{idx}/{total}] {scene_id}")
            try:
                download(url, out_file)
                ok += 1
                writer.writerow(
                    {"id": scene_id, "thumbnail_href": url, "file": file_name, "status": "downloaded", "error": ""}
                )
            except urllib.error.HTTPError as exc:
                fail += 1
                writer.writerow(
                    {
                        "id": scene_id,
                        "thumbnail_href": url,
                        "file": file_name,
                        "status": "failed",
                        "error": f"http_{exc.code}",
                    }
                )
                print(f"  failed: HTTP {exc.code}", file=sys.stderr)
            except Exception as exc:
                fail += 1
                writer.writerow(
                    {
                        "id": scene_id,
                        "thumbnail_href": url,
                        "file": file_name,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                print(f"  failed: {exc}", file=sys.stderr)

    print(f"Done. success={ok}, failed={fail}, total={total}")
    print(f"Manifest: {manifest_path}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
