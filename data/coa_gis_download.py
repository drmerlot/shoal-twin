"""
Step 2: COA Stormwater GIS Download
====================================
Pulls Shoal Creek open channel and drainage pipe layers from the
City of Austin Socrata API, filtered to watershed='SHL' and status='ACTIVE'.

Outputs:
    data/raw/gis/coa_open_channel_shl.geojson   — main channel reaches
    data/raw/gis/coa_drainage_pipe_shl.geojson  — storm sewer pipes
    data/raw/gis/coa_gis_summary.json           — field coverage report

Usage:
    uv run python data/coa_gis_download.py
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUT_DIR = Path("data/raw/gis")

# Shoal Creek bbox (degrees) — used for sanity check
SHOAL_BBOX = dict(xmin=-97.795, ymin=30.265, xmax=-97.720, ymax=30.400)

# COA Socrata dataset IDs
DATASETS = {
    "open_channel": {
        "id": "5uu3-d6ij",
        "label": "Open Channel (main stem reaches)",
        "out": "coa_open_channel_shl.geojson",
        # Fields we care about for SWMM skeleton
        "key_fields": [
            "drainage_id", "up_node_id", "dn_node_id",
            "up_invert", "dn_invert", "slope",
            "bottom_width", "top_width", "depth",
            "ditch_shape", "material", "length",
            "status", "watershed",
        ],
    },
    "drainage_pipe": {
        "id": "ybys-rwgs",
        "label": "Drainage Pipe (storm sewers)",
        "out": "coa_drainage_pipe_shl.geojson",
        "key_fields": [
            "drainage_id", "up_node_id", "dn_node_id",
            "up_invert", "dn_invert", "slope",
            "diameter", "width", "height", "pipe_shape",
            "material", "length", "status", "watershed",
        ],
    },
}

SOCRATA_BASE = "https://data.austintexas.gov/resource"
PAGE_SIZE = 2000  # max records per request


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def fetch_geojson(dataset_id: str, where: str = "1=1", limit: int = PAGE_SIZE) -> dict:
    """Fetch GeoJSON from COA Socrata API with pagination."""
    try:
        import urllib.request
        import urllib.parse
    except ImportError:
        log.error("urllib not available")
        sys.exit(1)

    all_features = []
    offset = 0

    while True:
        params = urllib.parse.urlencode({
            "$where": where,
            "$limit": limit,
            "$offset": offset,
            "$order": ":id",
        })
        url = f"{SOCRATA_BASE}/{dataset_id}.geojson?{params}"
        log.debug("GET %s", url)

        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
        except Exception as exc:
            log.error("Request failed: %s — URL: %s", exc, url)
            return None

        features = data.get("features", [])
        all_features.extend(features)
        log.info("  fetched %d features (total so far: %d)", len(features), len(all_features))

        if len(features) < limit:
            break  # last page
        offset += limit

    return {
        "type": "FeatureCollection",
        "features": all_features,
        "crs": data.get("crs"),
    }


def bbox_check(features: list) -> dict:
    """Check that features fall within the expected Shoal Creek bbox."""
    lons, lats = [], []
    for f in features:
        geom = f.get("geometry", {})
        if not geom:
            continue
        coords = geom.get("coordinates", [])
        # Flatten coordinate list (handles MultiLineString, LineString, Point)
        def flatten(c):
            if isinstance(c[0], (int, float)):
                return [c]
            return [pt for sub in c for pt in flatten(sub)]
        for pt in flatten(coords):
            lons.append(pt[0])
            lats.append(pt[1])

    if not lons:
        return {"ok": False, "reason": "no coordinates found"}

    actual = dict(xmin=min(lons), ymin=min(lats), xmax=max(lons), ymax=max(lats))
    ok = (
        actual["xmin"] >= SHOAL_BBOX["xmin"] - 0.05
        and actual["xmax"] <= SHOAL_BBOX["xmax"] + 0.05
        and actual["ymin"] >= SHOAL_BBOX["ymin"] - 0.05
        and actual["ymax"] <= SHOAL_BBOX["ymax"] + 0.05
    )
    return {"ok": ok, "actual_bbox": actual, "expected_bbox": SHOAL_BBOX}


def field_coverage(features: list, fields: list) -> dict:
    """Report what % of features have non-null values for each key field."""
    total = len(features)
    if total == 0:
        return {}
    coverage = {}
    for field in fields:
        populated = sum(
            1 for f in features
            if f.get("properties", {}).get(field) not in (None, "", "NODATA", "NO_SOURCE")
        )
        coverage[field] = round(100 * populated / total, 1)
    return coverage


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("Step 2: COA Stormwater GIS Download")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}

    for name, cfg in DATASETS.items():
        log.info("")
        log.info("── %s ──", cfg["label"])

        geojson = fetch_geojson(
            dataset_id=cfg["id"],
            where="watershed='SHL' AND status='ACTIVE'",
        )

        if geojson is None:
            log.warning("  FAILED — skipping")
            summary[name] = {"status": "FAILED"}
            continue

        features = geojson["features"]
        n = len(features)
        log.info("  Total active SHL features: %d", n)

        if n == 0:
            log.warning("  No features returned — check watershed/status filter")
            summary[name] = {"status": "EMPTY"}
            continue

        # Bbox sanity check
        bbox_result = bbox_check(features)
        if bbox_result["ok"]:
            log.info("  Bbox check: ✓ within Shoal Creek bounds")
        else:
            log.warning("  Bbox check: ✗ %s", bbox_result)

        # Field coverage
        coverage = field_coverage(features, cfg["key_fields"])
        log.info("  Field coverage (non-null %%)")
        for field, pct in coverage.items():
            bar = "▓" * int(pct / 10) + "░" * (10 - int(pct / 10))
            log.info("    %-20s %s %5.1f%%", field, bar, pct)

        # Save GeoJSON
        out_path = OUT_DIR / cfg["out"]
        with open(out_path, "w") as f:
            json.dump(geojson, f)
        log.info("  Saved → %s  (%s features)", out_path, n)

        summary[name] = {
            "status": "OK",
            "feature_count": n,
            "bbox": bbox_result,
            "field_coverage_pct": coverage,
            "output_file": str(out_path),
        }

    # Save summary
    summary_path = OUT_DIR / "coa_gis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("")
    log.info("Summary saved → %s", summary_path)

    # Print SWMM buildability verdict
    print_verdict(summary)


def print_verdict(summary: dict):
    oc = summary.get("open_channel", {})
    dp = summary.get("drainage_pipe", {})

    oc_ok = oc.get("status") == "OK"
    dp_ok = dp.get("status") == "OK"

    oc_cov = oc.get("field_coverage_pct", {})
    invert_pct = oc_cov.get("up_invert", 0)
    connectivity_pct = oc_cov.get("up_node_id", 0)

    print("\n" + "=" * 70)
    print("COA GIS — SWMM Buildability Verdict")
    print("=" * 70)
    print(f"  Open channel layer:     {'✓' if oc_ok else '✗'}  ({oc.get('feature_count', 0)} features)")
    print(f"  Drainage pipe layer:    {'✓' if dp_ok else '✗'}  ({dp.get('feature_count', 0)} features)")
    print(f"  Node connectivity:      {'✓' if connectivity_pct > 80 else '~'}  ({connectivity_pct:.0f}% populated)")
    print(f"  Invert elevations:      {'✓' if invert_pct > 50 else '~' if invert_pct > 20 else '✗'}  ({invert_pct:.0f}% populated)")
    print()

    if oc_ok and connectivity_pct > 50:
        print("  VERDICT: SWMM skeleton buildable from GIS data.")
        if invert_pct < 30:
            print("  ACTION:  Derive missing inverts from TNRIS 1m LiDAR DEM.")
    elif oc_ok:
        print("  VERDICT: Partial — open channels available, connectivity sparse.")
        print("  ACTION:  Manual network tracing or DEM-based snapping needed.")
    else:
        print("  VERDICT: Fallback to schematic SWMM model (hand-placed junctions).")
    print("=" * 70)


if __name__ == "__main__":
    main()
