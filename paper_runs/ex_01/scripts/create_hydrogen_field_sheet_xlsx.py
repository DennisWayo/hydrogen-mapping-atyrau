#!/usr/bin/env python3
"""Create a fillable Excel workbook for ex_01 hydrogen field validation."""

from __future__ import annotations

import datetime as dt
import html
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "hydrogen_field_data_sheet.xlsx"
OUTPUT_COPY = REPO_ROOT / "outputs" / "ex_01_field_sheet" / "hydrogen_field_data_sheet.xlsx"
MAX_ROWS = 202


def col_letter(idx: int) -> str:
    letters = ""
    while idx:
        idx, rem = divmod(idx - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def cell_xml(row: int, col: int, value: object = None, style: int | None = None, formula: str | None = None) -> str:
    ref = f"{col_letter(col)}{row}"
    style_attr = f' s="{style}"' if style is not None else ""
    if formula is not None:
        if value in (None, ""):
            return f'<c r="{ref}"{style_attr}><f>{esc(formula)}</f></c>'
        return f'<c r="{ref}"{style_attr}><f>{esc(formula)}</f><v>{esc(value)}</v></c>'
    if value is None:
        return f'<c r="{ref}"{style_attr}/>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}"{style_attr}><v>{value}</v></c>'
    return f'<c r="{ref}" t="inlineStr"{style_attr}><is><t>{esc(value)}</t></is></c>'


def sheet_xml(
    name: str,
    rows: list[list[object]],
    *,
    formulas: dict[tuple[int, int], str] | None = None,
    styles: dict[tuple[int, int], int] | None = None,
    col_widths: dict[int, float] | None = None,
    freeze_row: int | None = None,
    autofilter_ref: str | None = None,
    validations: list[tuple[str, str]] | None = None,
) -> str:
    formulas = formulas or {}
    styles = styles or {}
    col_widths = col_widths or {}
    validations = validations or []
    max_col = max((len(r) for r in rows), default=1)
    max_row = len(rows)

    cols_xml = ""
    if col_widths:
        col_parts = []
        for col, width in sorted(col_widths.items()):
            col_parts.append(f'<col min="{col}" max="{col}" width="{width}" customWidth="1"/>')
        cols_xml = f"<cols>{''.join(col_parts)}</cols>"

    pane_xml = ""
    if freeze_row:
        pane_xml = (
            f'<pane ySplit="{freeze_row}" topLeftCell="A{freeze_row + 1}" '
            'activePane="bottomLeft" state="frozen"/>'
            '<selection pane="bottomLeft"/>'
        )

    row_parts = []
    for r_idx, row in enumerate(rows, start=1):
        height = ' ht="28" customHeight="1"' if r_idx == 1 else ""
        cells = []
        for c_idx in range(1, max_col + 1):
            value = row[c_idx - 1] if c_idx <= len(row) else None
            formula = formulas.get((r_idx, c_idx))
            style = styles.get((r_idx, c_idx))
            if value is None and formula is None and style is None:
                continue
            cells.append(cell_xml(r_idx, c_idx, value=value, style=style, formula=formula))
        row_parts.append(f'<row r="{r_idx}"{height}>{"".join(cells)}</row>')

    autofilter_xml = f'<autoFilter ref="{autofilter_ref}"/>' if autofilter_ref else ""
    validations_xml = ""
    if validations:
        val_parts = []
        for sqref, formula1 in validations:
            val_parts.append(
                '<dataValidation type="list" allowBlank="1" showErrorMessage="1" '
                f'sqref="{sqref}"><formula1>{esc(formula1)}</formula1></dataValidation>'
            )
        validations_xml = f'<dataValidations count="{len(val_parts)}">{"".join(val_parts)}</dataValidations>'

    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheetViews><sheetView workbookViewId="0" showGridLines="0">{pane_xml}</sheetView></sheetViews>
  {cols_xml}
  <sheetData>{''.join(row_parts)}</sheetData>
  {autofilter_xml}
  {validations_xml}
</worksheet>'''


def workbook_xml(sheet_names: list[str]) -> str:
    sheets = "".join(
        f'<sheet name="{esc(name)}" sheetId="{i}" r:id="rId{i}"/>'
        for i, name in enumerate(sheet_names, start=1)
    )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>{sheets}</sheets>
  <calcPr calcId="191029" fullCalcOnLoad="1" forceFullCalc="1"/>
</workbook>'''


def workbook_rels(sheet_names: list[str]) -> str:
    rels = []
    for i, _ in enumerate(sheet_names, start=1):
        rels.append(
            f'<Relationship Id="rId{i}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{i}.xml"/>'
        )
    style_id = len(sheet_names) + 1
    rels.append(
        f'<Relationship Id="rId{style_id}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {''.join(rels)}
</Relationships>'''


def content_types(sheet_count: int) -> str:
    sheets = "".join(
        f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(1, sheet_count + 1)
    )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  {sheets}
</Types>'''


def root_rels() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''


def styles_xml() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="4">
    <font><sz val="10"/><name val="Aptos"/></font>
    <font><b/><sz val="10"/><color rgb="FFFFFFFF"/><name val="Aptos"/></font>
    <font><b/><sz val="12"/><color rgb="FF1F2937"/><name val="Aptos Display"/></font>
    <font><i/><sz val="10"/><color rgb="FF64748B"/><name val="Aptos"/></font>
  </fonts>
  <fills count="6">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1E3A5F"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFEAF2F8"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFFFF2CC"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFE2F0D9"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="3">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border><left style="thin"><color rgb="FFD9E2EC"/></left><right style="thin"><color rgb="FFD9E2EC"/></right><top style="thin"><color rgb="FFD9E2EC"/></top><bottom style="thin"><color rgb="FFD9E2EC"/></bottom><diagonal/></border>
    <border><bottom style="medium"><color rgb="FF1E3A5F"/></bottom></border>
  </borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="8">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="0" fillId="3" borderId="1" xfId="0" applyFill="1" applyBorder="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="2" fillId="0" borderId="2" xfId="0" applyFont="1" applyBorder="1"/>
    <xf numFmtId="0" fontId="0" fillId="4" borderId="1" xfId="0" applyFill="1" applyBorder="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="0" fillId="5" borderId="1" xfId="0" applyFill="1" applyBorder="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="3" fillId="0" borderId="0" xfId="0" applyFont="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''


def doc_props() -> tuple[str, str]:
    now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    core = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>ex_01 Hydrogen Field Data Sheet</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>'''
    app = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
</Properties>'''
    return core, app


def build_field_data() -> tuple[list[list[object]], dict[tuple[int, int], str], dict[tuple[int, int], int], list[tuple[str, str]]]:
    headers = [
        "record_id",
        "site_id",
        "aoi_name",
        "planned_point_id",
        "sampling_group",
        "visit_date_yyyy_mm_dd",
        "local_time_hhmm",
        "utc_time_hhmm",
        "team_lead",
        "observer_names",
        "instrument_operator",
        "latitude_dd",
        "longitude_dd",
        "elevation_m",
        "gps_accuracy_m",
        "coordinate_system",
        "weather_notes",
        "air_temperature_c",
        "wind_condition",
        "land_cover",
        "soil_type",
        "soil_moisture",
        "vegetation_stress",
        "surface_disturbance",
        "sample_type",
        "soil_gas_probe_depth_cm",
        "probe_seal_method",
        "purge_time_s",
        "flow_rate_ml_min",
        "instrument_make_model",
        "instrument_serial_no",
        "instrument_calibration_date",
        "h2_detection_limit_ppm",
        "instrument_reading_duration_s",
        "h2_ppm_reading_1",
        "h2_ppm_reading_2",
        "h2_ppm_reading_3",
        "h2_ppm_mean",
        "h2_ppm_std",
        "h2_background_ppm",
        "h2_anomaly_ppm",
        "h2_positive_flag_suggested",
        "ch4_ppm",
        "co2_percent",
        "o2_percent",
        "helium_ppm",
        "nitrogen_percent",
        "h2s_ppm",
        "lab_sample_id",
        "lab_confirmed_h2_ppm",
        "lab_result_date",
        "sample_container_id",
        "preservation_method",
        "chain_of_custody_id",
        "lithology_observed",
        "weathering_grade",
        "alteration_observed",
        "fracture_present",
        "fracture_orientation_deg",
        "fracture_aperture_mm",
        "fracture_density_m2",
        "distance_to_mapped_fault_m",
        "fault_lineament_notes",
        "geomorphic_setting",
        "dem_slope_deg",
        "drainage_condition",
        "circular_feature_present",
        "photo_north_id",
        "photo_east_id",
        "photo_closeup_id",
        "photo_instrument_id",
        "model_score",
        "model_rank_percentile",
        "model_predicted_class",
        "field_interpretation_manual",
        "validation_label_code",
        "qa_flag",
        "contamination_risk",
        "duplicate_of_record_id",
        "notes",
    ]
    example = [
        "EX01-FLD-0001",
        "Saumankol",
        "saumankol",
        "SAU-HIGH-001",
        "high_rank",
        "2026-08-15",
        "10:35",
        "04:35",
        "D. Wayo",
        "D. Wayo; M. Leila",
        "Field operator",
        53.29379,
        68.09880,
        215,
        3.2,
        "WGS84_EPSG4326",
        "dry, clear, light wind",
        24.5,
        "light",
        "sparse grassland",
        "sandy loam",
        "dry",
        "none",
        "none",
        "soil_gas",
        80,
        "bentonite_seal",
        60,
        500,
        "portable H2 analyzer",
        "SN-0000",
        "2026-08-01",
        1,
        120,
        38.2,
        40.1,
        39.4,
        None,
        None,
        3.5,
        None,
        None,
        1.2,
        0.04,
        20.8,
        6.5,
        78.0,
        0,
        "LAB-SAU-0001",
        "",
        "",
        "vial-0001",
        "gas-tight vial",
        "COC-0001",
        "sandstone over weathered cover",
        "moderate",
        "iron oxide staining",
        "yes",
        35,
        2,
        4,
        1250,
        "minor fractures visible in shallow exposure",
        "shallow depression",
        2.3,
        "well drained",
        "no",
        "SAU001_N.jpg",
        "SAU001_E.jpg",
        "SAU001_close.jpg",
        "SAU001_inst.jpg",
        0.762,
        98.5,
        "high",
        "measured_positive",
        None,
        "good",
        "low",
        "",
        "Example row only; replace with field values.",
    ]
    rows: list[list[object]] = [headers, example]
    for _ in range(3, MAX_ROWS + 1):
        rows.append([None] * len(headers))

    formula_cols = {
        "h2_ppm_mean": "IF(COUNTA(AI{r}:AK{r})=0,\"\",AVERAGE(AI{r}:AK{r}))",
        "h2_ppm_std": "IF(COUNTA(AI{r}:AK{r})<2,\"\",STDEV.S(AI{r}:AK{r}))",
        "h2_anomaly_ppm": "IF(OR(AL{r}=\"\",AN{r}=\"\"),\"\",AL{r}-AN{r})",
        "h2_positive_flag_suggested": "IF(OR(AO{r}=\"\",AG{r}=\"\"),\"\",IF(AO{r}>=MAX(AG{r},AM{r}*3),\"Yes\",IF(AO{r}<=AG{r},\"No\",\"Uncertain\")))",
        "validation_label_code": "IF(BW{r}=\"measured_positive\",1,IF(BW{r}=\"measured_background\",0,IF(BW{r}=\"not_sampled\",999,255)))",
    }
    formulas = {}
    for r in range(2, MAX_ROWS + 1):
        for header, template in formula_cols.items():
            c = headers.index(header) + 1
            formulas[(r, c)] = template.format(r=r)

    styles = {}
    for c in range(1, len(headers) + 1):
        styles[(1, c)] = 1
    for r in range(2, MAX_ROWS + 1):
        for c in range(1, len(headers) + 1):
            header = headers[c - 1]
            if r == 2:
                styles[(r, c)] = 2
            elif header in formula_cols:
                styles[(r, c)] = 6
            else:
                styles[(r, c)] = 3

    list_ranges = {
        "sampling_group": "'Lookup_Lists'!$A$2:$A$7",
        "coordinate_system": "'Lookup_Lists'!$B$2:$B$5",
        "soil_moisture": "'Lookup_Lists'!$C$2:$C$6",
        "vegetation_stress": "'Lookup_Lists'!$D$2:$D$6",
        "surface_disturbance": "'Lookup_Lists'!$E$2:$E$7",
        "sample_type": "'Lookup_Lists'!$F$2:$F$8",
        "probe_seal_method": "'Lookup_Lists'!$G$2:$G$7",
        "weathering_grade": "'Lookup_Lists'!$H$2:$H$7",
        "fracture_present": "'Lookup_Lists'!$I$2:$I$4",
        "geomorphic_setting": "'Lookup_Lists'!$J$2:$J$9",
        "circular_feature_present": "'Lookup_Lists'!$I$2:$I$4",
        "model_predicted_class": "'Lookup_Lists'!$K$2:$K$5",
        "field_interpretation_manual": "'Lookup_Lists'!$L$2:$L$6",
        "qa_flag": "'Lookup_Lists'!$M$2:$M$6",
        "contamination_risk": "'Lookup_Lists'!$N$2:$N$5",
    }
    validations = []
    for header, source in list_ranges.items():
        c = headers.index(header) + 1
        validations.append((f"{col_letter(c)}2:{col_letter(c)}{MAX_ROWS}", source))

    return rows, formulas, styles, validations


def build_lookup() -> list[list[object]]:
    headers = [
        "sampling_group",
        "coordinate_system",
        "soil_moisture",
        "vegetation_stress",
        "surface_disturbance",
        "sample_type",
        "probe_seal_method",
        "weathering_grade",
        "yes_no_uncertain",
        "geomorphic_setting",
        "model_predicted_class",
        "field_interpretation_manual",
        "qa_flag",
        "contamination_risk",
    ]
    columns = [
        ["high_rank", "medium_rank", "low_rank_background", "geology_control", "duplicate", "opportunistic"],
        ["WGS84_EPSG4326", "UTM_zone_42N", "UTM_zone_43N", "other"],
        ["dry", "slightly_moist", "moist", "wet", "unknown"],
        ["none", "low", "moderate", "high", "unknown"],
        ["none", "road", "agriculture", "construction", "livestock", "other"],
        ["soil_gas", "free_gas", "water_gas", "soil", "rock", "microbial", "alteration"],
        ["bentonite_seal", "rubber_gasket", "water_seal", "manual_compaction", "none", "other"],
        ["fresh", "slight", "moderate", "high", "saprolitic", "unknown"],
        ["yes", "no", "uncertain"],
        ["depression", "slope", "ridge", "drainage", "mound", "circular_feature", "wetland", "plain"],
        ["high", "medium", "low", "outside"],
        ["measured_positive", "measured_background", "uncertain", "invalid", "not_sampled"],
        ["good", "suspect", "contaminated", "duplicate", "invalid"],
        ["low", "medium", "high", "unknown"],
    ]
    max_len = max(len(c) for c in columns)
    rows = [headers]
    for i in range(max_len):
        rows.append([col[i] if i < len(col) else None for col in columns])
    return rows


def build_protocol() -> list[list[object]]:
    return [
        ["ex_01 Hydrogen Field Validation Sheet", "Use Field_Data as the raw, machine-readable record. Row 2 is an example only; start real observations from row 3 or overwrite the example after training the team."],
        ["Purpose", "Replace proxy AOI labels with measured hydrogen observations while preserving uncertainty, QA flags, and field context."],
        ["Minimum valid point", "A valid point needs site ID, coordinates, GPS accuracy, date/time, instrument metadata, three H2 readings, background H2, sample type, QA flag, and manual field interpretation."],
        ["Replicates", "Collect at least three H2 readings per point. Keep the same probe depth, purge time, and reading duration across replicates where field conditions allow."],
        ["Background", "Measure local background 50-200 m away from the target point, preferably in similar land cover and soil condition but outside the model target pixel."],
        ["Suggested positive flag", "The workbook suggests Yes when anomaly is at least the detection limit and at least three times the replicate standard deviation. Treat this as a QA guide, not final interpretation."],
        ["Validation labels", "measured_positive maps to 1; measured_background maps to 0; uncertain or invalid maps to 255; not_sampled maps to 999."],
        ["Scientific use", "Use measured_positive and measured_background only for independent validation. Keep uncertain and invalid rows out of supervised model fitting."],
        ["Sampling balance", "For each AOI, include high-rank, medium-rank, low-rank/background, geology-control, and duplicate points so the model is challenged rather than only confirmed."],
        ["Photos", "Capture north-facing, east-facing, close-up, and instrument-setup photos. Use the photo ID columns to link image files to the observation row."],
    ]


def build_schema(headers: list[object]) -> list[list[object]]:
    descriptions = {
        "record_id": "Unique field record identifier.",
        "site_id": "Study site or named AOI.",
        "aoi_name": "Machine-readable AOI name matching ex_01 outputs.",
        "planned_point_id": "Pre-field point ID from the sampling plan.",
        "sampling_group": "High, medium, low/background, geology-control, duplicate, or opportunistic sample type.",
        "latitude_dd": "Latitude in decimal degrees.",
        "longitude_dd": "Longitude in decimal degrees.",
        "h2_ppm_reading_1": "First hydrogen reading in ppm.",
        "h2_ppm_reading_2": "Second hydrogen reading in ppm.",
        "h2_ppm_reading_3": "Third hydrogen reading in ppm.",
        "h2_ppm_mean": "Formula-derived mean of the three H2 readings.",
        "h2_ppm_std": "Formula-derived standard deviation of H2 readings.",
        "h2_background_ppm": "Local background hydrogen reading in ppm.",
        "h2_anomaly_ppm": "Formula-derived mean H2 minus background H2.",
        "h2_positive_flag_suggested": "Formula-derived screening flag based on anomaly, detection limit, and replicate variability.",
        "field_interpretation_manual": "Conservative final field interpretation selected by the team.",
        "validation_label_code": "Formula-derived label for model validation: 1, 0, 255, or 999.",
    }
    rows = [["column_name", "description", "unit_or_allowed_values", "required_for_validation"]]
    required = {
        "record_id",
        "site_id",
        "aoi_name",
        "visit_date_yyyy_mm_dd",
        "latitude_dd",
        "longitude_dd",
        "gps_accuracy_m",
        "instrument_make_model",
        "instrument_serial_no",
        "instrument_calibration_date",
        "h2_detection_limit_ppm",
        "h2_ppm_reading_1",
        "h2_ppm_reading_2",
        "h2_ppm_reading_3",
        "h2_background_ppm",
        "sample_type",
        "field_interpretation_manual",
        "qa_flag",
    }
    units = {
        "latitude_dd": "decimal degrees",
        "longitude_dd": "decimal degrees",
        "elevation_m": "m",
        "gps_accuracy_m": "m",
        "air_temperature_c": "deg C",
        "soil_gas_probe_depth_cm": "cm",
        "purge_time_s": "s",
        "flow_rate_ml_min": "mL/min",
        "h2_detection_limit_ppm": "ppm",
        "instrument_reading_duration_s": "s",
        "h2_ppm_reading_1": "ppm",
        "h2_ppm_reading_2": "ppm",
        "h2_ppm_reading_3": "ppm",
        "h2_ppm_mean": "ppm",
        "h2_ppm_std": "ppm",
        "h2_background_ppm": "ppm",
        "h2_anomaly_ppm": "ppm",
        "ch4_ppm": "ppm",
        "co2_percent": "%",
        "o2_percent": "%",
        "helium_ppm": "ppm",
        "nitrogen_percent": "%",
        "h2s_ppm": "ppm",
        "fracture_orientation_deg": "degrees",
        "fracture_aperture_mm": "mm",
        "fracture_density_m2": "count/m2",
        "distance_to_mapped_fault_m": "m",
        "dem_slope_deg": "degrees",
        "model_score": "0-1",
        "model_rank_percentile": "0-100",
    }
    for h in headers:
        rows.append([h, descriptions.get(h, "Field observation or QA/context attribute."), units.get(h, "see protocol or lookup list"), "yes" if h in required else "no"])
    return rows


def write_xlsx(path: Path) -> None:
    field_rows, field_formulas, field_styles, field_validations = build_field_data()
    headers = field_rows[0]
    lookup_rows = build_lookup()
    protocol_rows = build_protocol()
    schema_rows = build_schema(headers)
    sheet_names = ["Field_Data", "Field_Protocol", "Lookup_Lists", "Import_Schema"]

    widths = {i: 16 for i in range(1, len(headers) + 1)}
    for idx, header in enumerate(headers, start=1):
        if header in {"notes", "weather_notes", "fault_lineament_notes", "observer_names", "alteration_observed"}:
            widths[idx] = 28
        elif header in {"latitude_dd", "longitude_dd", "model_rank_percentile"}:
            widths[idx] = 14
        elif header.startswith("h2_"):
            widths[idx] = 16
        elif len(header) > 22:
            widths[idx] = 22

    def header_styles(rows: list[list[object]]) -> dict[tuple[int, int], int]:
        out = {}
        for c in range(1, max(len(r) for r in rows) + 1):
            out[(1, c)] = 1 if len(rows) > 2 else 4
        for r in range(2, len(rows) + 1):
            for c in range(1, max(len(row) for row in rows) + 1):
                out[(r, c)] = 3
        return out

    sheets = [
        sheet_xml(
            "Field_Data",
            field_rows,
            formulas=field_formulas,
            styles=field_styles,
            col_widths=widths,
            freeze_row=1,
            autofilter_ref=f"A1:{col_letter(len(headers))}{MAX_ROWS}",
            validations=field_validations,
        ),
        sheet_xml("Field_Protocol", protocol_rows, styles=header_styles(protocol_rows), col_widths={1: 24, 2: 120}, freeze_row=1),
        sheet_xml("Lookup_Lists", lookup_rows, styles=header_styles(lookup_rows), col_widths={i: 22 for i in range(1, 15)}, freeze_row=1),
        sheet_xml("Import_Schema", schema_rows, styles=header_styles(schema_rows), col_widths={1: 30, 2: 76, 3: 24, 4: 20}, freeze_row=1, autofilter_ref=f"A1:D{len(schema_rows)}"),
    ]

    core, app = doc_props()
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types(len(sheet_names)))
        zf.writestr("_rels/.rels", root_rels())
        zf.writestr("docProps/core.xml", core)
        zf.writestr("docProps/app.xml", app)
        zf.writestr("xl/workbook.xml", workbook_xml(sheet_names))
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels(sheet_names))
        zf.writestr("xl/styles.xml", styles_xml())
        for idx, xml in enumerate(sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", xml)


def main() -> None:
    write_xlsx(OUTPUT)
    OUTPUT_COPY.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUTPUT, OUTPUT_COPY)
    print(OUTPUT)
    print(OUTPUT_COPY)


if __name__ == "__main__":
    main()
