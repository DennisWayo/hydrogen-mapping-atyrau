#!/usr/bin/env python3
"""Convert a Google Earth KMZ/KML polygon placemark into GeoJSON."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}


def _read_kml_text(path: Path) -> str:
    if path.suffix.lower() == ".kmz":
        with zipfile.ZipFile(path, "r") as zf:
            names = [name for name in zf.namelist() if name.lower().endswith(".kml")]
            if not names:
                raise ValueError(f"No .kml file found in KMZ: {path}")
            with zf.open(names[0], "r") as fh:
                return fh.read().decode("utf-8")
    return path.read_text(encoding="utf-8")


def _parse_first_polygon(kml_text: str) -> tuple[str, list[list[float]]]:
    root = ET.fromstring(kml_text)
    placemark = root.find(".//kml:Placemark", KML_NS)
    if placemark is None:
        raise ValueError("No Placemark found in KML.")

    name_node = placemark.find("kml:name", KML_NS)
    name = name_node.text.strip() if name_node is not None and name_node.text else "AOI"

    coords_node = placemark.find(
        ".//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", KML_NS
    )
    if coords_node is None or not coords_node.text:
        raise ValueError("No polygon coordinates found in KML Placemark.")

    raw_parts = coords_node.text.replace("\n", " ").split()
    coords: list[list[float]] = []
    for part in raw_parts:
        values = part.split(",")
        if len(values) < 2:
            continue
        lon = float(values[0])
        lat = float(values[1])
        coords.append([lon, lat])

    if len(coords) < 4:
        raise ValueError("Polygon must contain at least 4 coordinate points.")

    if coords[0] != coords[-1]:
        coords.append(coords[0])

    return name, coords


def _bbox(coords: list[list[float]]) -> tuple[float, float, float, float]:
    lons = [xy[0] for xy in coords]
    lats = [xy[1] for xy in coords]
    return (min(lons), min(lats), max(lons), max(lats))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to .kmz or .kml file")
    parser.add_argument("--output", required=True, help="Output GeoJSON path")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kml_text = _read_kml_text(input_path)
    name, coords = _parse_first_polygon(kml_text)
    min_lon, min_lat, max_lon, max_lat = _bbox(coords)

    feature = {
        "type": "Feature",
        "properties": {
            "name": name,
            "source_file": input_path.name,
            "bbox": [min_lon, min_lat, max_lon, max_lat],
        },
        "geometry": {"type": "Polygon", "coordinates": [coords]},
    }
    fc = {"type": "FeatureCollection", "features": [feature]}

    output_path.write_text(json.dumps(fc, indent=2), encoding="utf-8")

    print(f"Saved: {output_path}")
    print(f"BBOX (lon/lat): {min_lon}, {min_lat}, {max_lon}, {max_lat}")


if __name__ == "__main__":
    main()
