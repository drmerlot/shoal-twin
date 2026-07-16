"""
data/coa_gis_check.py
---------------------
Step 2: Inspect City of Austin stormwater GIS data availability.

This script:
  1. Queries the COA ArcGIS REST API for stormwater infrastructure layers.
  2. Downloads a sample of pipe/conduit records and checks attribute fields
     (diameter, invert elevations, shape, material, connectivity).
  3. Reports whether attributes are sufficient to build a SWMM conduit network.
  4. Downloads the Shoal Creek watershed boundary from COA open data.
  5. Gives a frank assessment of SWMM model buildability.

Data sources:
  - COA Open Data: https://data.austintexas.gov
  - ArcGIS REST: https://services.arcgis.com/0L95CJ0VTaxqcmED/arcgis/rest/services/
  - Watershed Protection: COA Watershed Protection Dept shapefiles

Decision log:
  - Using ArcGIS REST API (JSON) rather than direct Socrata download because
    stormwater infrastructure layers are published as feature services, not CSV.
  - Bounding box for Shoal Creek watershed: approx 30.27–30.35°N, 97.78–97.73°W.
    This is a rough pre-clip; will be refined with the actual watershed polygon.

Usage:
    uv run python data/coa_gis_check.py
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys
import time
from typing import Any, Optional
from urllib.parse import urlencode

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Target URLs
# ---------------------------------------------------------------------------
# COA ArcGIS REST services base
COA_ARCGIS_BASE = (
    "https://services.arcgis.com/0L95CJ0VTaxqcmED/arcgis/rest/services"
)

# Known COA stormwater feature services (confirmed as of 2024)
# Layer IDs may shift if COA publishes new versions — script validates them.
LAYER_CANDIDATES = {
    "storm_pipes": [
        f"{COA_ARCGIS_BASE}/DRAINAGE_PIPES/FeatureServer/0",
        f"{COA_ARCGIS_BASE}/Stormwater_Mains/FeatureServer/0",
        f"{COA_ARCGIS_BASE}/STORM_DRAINS/FeatureServer/0",
    ],
    "manholes": [
        f"{COA_ARCGIS_BASE}/DRAINAGE_STRUCTURES/FeatureServer/0",
        f"{COA_ARCGIS_BASE}/Stormwater_Structures/FeatureServer/0",
    ],
    "watersheds": [
        f"{COA_ARCGIS_BASE}/WATERSHED_MASTER_PLAN_AREAS/FeatureServer/0",
        f"{COA_ARCGIS_BASE}/Watersheds/FeatureServer/0",
        "https://services.arcgis.com/0L95CJ0VTaxqcmED/arcgis/rest/services/WATERSHED_MASTERPLAN/FeatureServer/0",
    ],
}

# Shoal Creek watershed approximate bounding box (WGS84)
SHOAL_BBOX = {
    "xmin": -97.795,
    "ymin":  30.265,
    "xmax": -97.720,
    "ymax":  30.360,
}

# SWMM-critical conduit attributes (what we need to build the network)
SWMM_REQUIRED_FIELDS = {
    "diameter":    ["diameter", "diam", "pipe_size", "size", "height", "width"],
    "invert_up":   ["invert_up", "us_invert", "us_inv", "upinv", "upstream_invert"],
    "invert_dn":   ["invert_dn", "ds_invert", "ds_inv", "dninv", "downstream_invert"],
    "shape":       ["shape", "xshape", "pipeshape", "conduit_shape"],
    "material":    ["material", "pipe_mat", "mat"],
    "length":      ["length", "pipe_len", "conduit_length"],
    "from_node":   ["from_node", "us_node", "up_node", "frm_node", "upstream_node"],
    "to_node":     ["to_node", "ds_node", "dn_node", "downstream_node"],
}

DATA_DIR   = pathlib.Path(__file__).parent / "raw" / "gis"
REPORT_OUT = DATA_DIR / "coa_gis_report.json"

TIMEOUT = 30  # seconds per request


# ---------------------------------------------------------------------------
# ArcGIS REST helpers
# ---------------------------------------------------------------------------
def _get_json(url: str, params: dict | None = None, retries: int = 2) -> dict:
    """GET an ArcGIS REST endpoint, return parsed JSON."""
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt < retries:
                log.warning("Request failed (%s) — retrying in 3s…", exc)
                time.sleep(3)
            else:
                raise


def probe_layer(url: str) -> Optional[dict]:
    """
    Try to reach an ArcGIS FeatureServer layer.
    Returns layer metadata dict on success, None on failure.
    """
    try:
        meta = _get_json(url, params={"f": "json"})
        if "error" in meta:
            log.debug("Layer error at %s: %s", url, meta["error"])
            return None
        return meta
    except Exception as exc:
        log.debug("Layer probe failed at %s: %s", url, exc)
        return None


def fetch_sample(url: str, max_records: int = 50) -> Optional[dict]:
    """
    Fetch a small sample of features from a FeatureServer layer within
    the Shoal Creek bounding box. Returns the raw GeoJSON/ESRI JSON dict.
    """
    params = {
        "where":         "1=1",
        "geometry":      f"{SHOAL_BBOX['xmin']},{SHOAL_BBOX['ymin']},"
                         f"{SHOAL_BBOX['xmax']},{SHOAL_BBOX['ymax']}",
        "geometryType":  "esriGeometryEnvelope",
        "inSR":          "4326",
        "spatialRel":    "esriSpatialRelIntersects",
        "outFields":     "*",
        "resultRecordCount": max_records,
        "f":             "geojson",
    }
    query_url = url.rstrip("/") + "/query"
    try:
        data = _get_json(query_url, params=params)
        return data
    except Exception as exc:
        log.warning("Sample fetch failed at %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Field analysis
# ---------------------------------------------------------------------------
def analyse_fields(
    fields: list[str],
    sample_props: list[dict],
) -> dict:
    """
    Check which SWMM-required attribute categories are present.
    Returns a dict of {category: {found_field, populated_pct}}.
    """
    fields_lower = [f.lower() for f in fields]
    result = {}

    for category, candidates in SWMM_REQUIRED_FIELDS.items():
        matched = None
        for cand in candidates:
            if cand.lower() in fields_lower:
                # Get the original-case field name
                idx = fields_lower.index(cand.lower())
                matched = fields[idx]
                break

        if matched:
            # Check population rate in sample
            n_populated = sum(
                1 for p in sample_props
                if p.get(matched) not in (None, "", 0, -9999, -9998)
            )
            pop_pct = round(n_populated / max(len(sample_props), 1) * 100, 1)
            result[category] = {"field": matched, "populated_pct": pop_pct}
        else:
            result[category] = {"field": None, "populated_pct": 0.0}

    return result


# ---------------------------------------------------------------------------
# Layer discovery
# ---------------------------------------------------------------------------
def discover_layers() -> dict[str, dict]:
    """
    Try each candidate URL for each layer type.
    Return dict of {layer_type: {url, meta, sample, field_analysis}}.
    """
    found = {}

    for layer_type, candidates in LAYER_CANDIDATES.items():
        log.info("Probing %s (%d candidates)…", layer_type, len(candidates))

        for url in candidates:
            meta = probe_layer(url)
            if meta is None:
                continue

            log.info("  ✓ Found %s at: %s", layer_type, url)
            layer_name = meta.get("name", "?")
            fields_raw  = meta.get("fields", [])
            field_names = [f.get("name", "") for f in fields_raw]
            log.info("    Name: %s  |  Fields: %d", layer_name, len(field_names))

            # Fetch sample
            log.info("    Fetching sample (bbox=Shoal Creek, n≤50)…")
            sample = fetch_sample(url)

            sample_props: list[dict] = []
            n_features = 0
            if sample and "features" in sample:
                n_features   = len(sample["features"])
                sample_props = [
                    f.get("properties", f.get("attributes", {}))
                    for f in sample["features"]
                ]

            log.info("    Features in bbox: %d", n_features)

            # Field analysis
            field_analysis = analyse_fields(field_names, sample_props)

            found[layer_type] = {
                "url":            url,
                "layer_name":     layer_name,
                "n_fields":       len(field_names),
                "field_names":    field_names,
                "n_features_bbox": n_features,
                "field_analysis": field_analysis,
            }
            break  # Use first successful URL

        if layer_type not in found:
            log.warning("  ✗ %s — no working URL found", layer_type)
            found[layer_type] = {"url": None, "error": "Not found"}

    return found


# ---------------------------------------------------------------------------
# Assessment print
# ---------------------------------------------------------------------------
def print_gis_assessment(layers: dict[str, dict]) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("City of Austin Stormwater GIS — SWMM Buildability Assessment")
    print(sep)

    for ltype, info in layers.items():
        print(f"\n── {ltype.upper()} ──────────────────────────────────────────────")
        if info.get("url") is None:
            print(f"  ✗  NOT FOUND — tried all candidate URLs")
            print("     Action: manually browse data.austintexas.gov for layer name")
            continue

        print(f"  URL:        {info['url']}")
        print(f"  Layer name: {info.get('layer_name', '?')}")
        print(f"  Fields:     {info.get('n_fields', 0)}")
        print(f"  Features in Shoal bbox: {info.get('n_features_bbox', 0)}")

        fa = info.get("field_analysis", {})
        if fa:
            print(f"  SWMM field coverage:")
            all_present = True
            for cat, detail in fa.items():
                fld  = detail.get("field")
                pop  = detail.get("populated_pct", 0)
                if fld:
                    flag = "✓" if pop >= 70 else "~"
                    print(f"    {flag} {cat:<15} → '{fld}'  ({pop:.0f}% populated)")
                else:
                    print(f"    ✗ {cat:<15} → NOT FOUND")
                    all_present = False

            if all_present:
                print("  → All SWMM-required fields present")
            else:
                missing = [c for c, d in fa.items() if d["field"] is None]
                print(f"  → MISSING: {', '.join(missing)}")
                print("    May need to derive from geometry or use defaults")

    print(f"\n── OVERALL SWMM BUILDABILITY ────────────────────────────────────")

    has_pipes     = layers.get("storm_pipes", {}).get("url") is not None
    has_manholes  = layers.get("manholes", {}).get("url") is not None
    has_watershed = layers.get("watersheds", {}).get("url") is not None

    pipe_fa = layers.get("storm_pipes", {}).get("field_analysis", {})
    has_diam    = pipe_fa.get("diameter", {}).get("field") is not None
    has_inv     = (pipe_fa.get("invert_up", {}).get("field") is not None and
                   pipe_fa.get("invert_dn", {}).get("field") is not None)
    has_connect = (pipe_fa.get("from_node", {}).get("field") is not None or
                   pipe_fa.get("to_node", {}).get("field") is not None)

    n_pipe_features = layers.get("storm_pipes", {}).get("n_features_bbox", 0)

    print(f"  Pipe layer accessible:   {'YES' if has_pipes else 'NO'}")
    print(f"  Manhole layer:           {'YES' if has_manholes else 'NO'}")
    print(f"  Watershed boundary:      {'YES' if has_watershed else 'NO'}")
    print(f"  Diameter field:          {'YES' if has_diam else 'NO — must default'}")
    print(f"  Invert elevations:       {'YES' if has_inv else 'NO — must derive from DEM'}")
    print(f"  Node connectivity:       {'YES' if has_connect else 'NO — must derive from geometry'}")
    print(f"  Pipes in Shoal bbox:     {n_pipe_features}")

    if has_pipes and has_diam and n_pipe_features >= 10:
        print("\n  VERDICT: Sufficient GIS data to build SWMM skeleton.")
        print("  Invert elevations may need DEM-derived values if not in attributes.")
        print("  Proceed to Step 3 (SWMM model skeleton).")
    elif has_pipes and n_pipe_features >= 10:
        print("\n  VERDICT: Pipe geometry available but missing key attributes.")
        print("  Can build geometry-only skeleton; need DEM for elevations.")
        print("  Proceed with caution — more manual work in SWMM INP assembly.")
    elif has_pipes:
        print("\n  VERDICT: Layer found but few/no features in Shoal Creek bbox.")
        print("  Check bbox coordinates or try downloading full layer.")
    else:
        print("\n  VERDICT: Stormwater pipe layer not accessible via API.")
        print("  Fallback options:")
        print("    1. Download directly from data.austintexas.gov (manual)")
        print("    2. Use NHD flowlines for channel geometry only")
        print("    3. Build schematic network from aerial imagery + survey data")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("Step 2: COA stormwater GIS availability check")
    log.info("Shoal Creek bbox: %s", SHOAL_BBOX)

    layers = discover_layers()

    # Save report
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    report = {"timestamp": __import__("datetime").datetime.utcnow().isoformat(), "layers": layers}
    # Convert any non-serializable types
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if hasattr(obj, "item"):  # numpy scalar
            return obj.item()
        return obj
    REPORT_OUT.write_text(json.dumps(_clean(report), indent=2))
    log.info("Report saved → %s", REPORT_OUT)

    print_gis_assessment(layers)


if __name__ == "__main__":
    main()
