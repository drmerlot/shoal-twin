"""
Step 3a: Fetch Ground Elevations at Channel Nodes (Missing Inverts)
====================================================================
Uses the USGS 3DEP Elevation Point Query Service (EPQS) to get ground
elevation at each open channel node that is missing an invert elevation.

These DEM-derived elevations are used in build_swmm_inp.py to fill gaps.
Invert = DEM_ground_elev - bank_depth_estimate (conservatively 1 ft for
nodes where we have no depth field).

Output:
    data/raw/gis/node_elevations.json   — {node_id: {lon, lat, elev_ft, source}}

Usage:
    uv run python data/fetch_dem_elevations.py

Notes:
    - EPQS endpoint: https://epqs.nationalmap.gov/v1/json
    - Returns elevation in feet (NAVD88), ~1/3 arc-second resolution (~10m)
    - Rate limit: ~0.5 req/sec to be polite; ~650 nodes takes ~5 min
    - Results cached to disk — safe to re-run after interruption
"""

import json
import logging
import time
import urllib.parse
import urllib.request
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

GIS_DIR = Path("data/raw/gis")
OC_PATH = GIS_DIR / "coa_open_channel_shl.geojson"
OUT_PATH = GIS_DIR / "node_elevations.json"

EPQS_URL = "https://epqs.nationalmap.gov/v1/json"
REQUEST_DELAY_S = 0.5   # seconds between EPQS requests
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def flatten_coords(coords):
    """Flatten nested coordinate list to list of [lon, lat] points."""
    if not coords:
        return []
    if isinstance(coords[0], (int, float)):
        return [coords]
    return [pt for sub in coords for pt in flatten_coords(sub)]


def reach_endpoints(feature):
    """Return (up_point, dn_point) as [lon, lat] from a reach feature.

    Convention: up = first coordinate of first line, dn = last coordinate of last line.
    COA GIS stores reaches in flow direction (upstream to downstream).
    """
    geom = feature["geometry"]
    coords = geom["coordinates"]
    gtype = geom["type"]

    if gtype == "LineString":
        lines = [coords]
    elif gtype == "MultiLineString":
        lines = coords
    else:
        return None, None

    if not lines or not lines[0] or not lines[-1]:
        return None, None

    up_pt = lines[0][0]    # first point of first segment
    dn_pt = lines[-1][-1]  # last point of last segment
    return up_pt, dn_pt


# ---------------------------------------------------------------------------
# EPQS query
# ---------------------------------------------------------------------------

def query_epqs(lon: float, lat: float):
    """Query USGS 3DEP EPQS for ground elevation in feet at (lon, lat).
    Returns float or None.
    """
    params = urllib.parse.urlencode({
        "x": f"{lon:.7f}",
        "y": f"{lat:.7f}",
        "wkid": "4326",
        "units": "Feet",
        "includeDate": "false",
    })
    url = f"{EPQS_URL}?{params}"

    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read())
            val = data.get("value")
            if val is None:
                return None
            val = float(val)
            # EPQS returns -1000000 for ocean / no-data
            if val < -9999:
                return None
            return val
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                log.warning("  EPQS retry %d for (%.5f, %.5f): %s", attempt + 1, lon, lat, exc)
                time.sleep(2)
            else:
                log.error("  EPQS failed after %d retries: %s", MAX_RETRIES, exc)
                return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("Step 3a: Fetch DEM elevations at open channel nodes")

    # Load existing cache
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            cache = json.load(f)
        log.info("Loaded %d cached node elevations from %s", len(cache), OUT_PATH)
    else:
        cache = {}

    # Load open channel GeoJSON
    with open(OC_PATH) as f:
        oc = json.load(f)
    features = oc["features"]
    log.info("Loaded %d open channel features", len(features))

    # Build node -> {lon, lat, invert} mapping
    # For each reach: up_node sits at up_pt coords, dn_node at dn_pt coords
    node_info = {}

    for feat in features:
        props = feat["properties"]
        up_id = str(props["up_node_id"])
        dn_id = str(props["dn_node_id"])
        up_inv = props.get("up_invert")
        dn_inv = props.get("dn_invert")

        up_pt, dn_pt = reach_endpoints(feat)
        if up_pt is None:
            continue

        # Register nodes (first-seen coordinates win; update invert if we get one)
        if up_id not in node_info:
            node_info[up_id] = {
                "lon": up_pt[0], "lat": up_pt[1],
                "invert": float(up_inv) if up_inv not in (None, "") else None,
            }
        elif node_info[up_id]["invert"] is None and up_inv not in (None, ""):
            node_info[up_id]["invert"] = float(up_inv)

        if dn_id not in node_info:
            node_info[dn_id] = {
                "lon": dn_pt[0], "lat": dn_pt[1],
                "invert": float(dn_inv) if dn_inv not in (None, "") else None,
            }
        elif node_info[dn_id]["invert"] is None and dn_inv not in (None, ""):
            node_info[dn_id]["invert"] = float(dn_inv)

    total_nodes = len(node_info)
    missing = {nid: info for nid, info in node_info.items() if info["invert"] is None}
    already_known = {nid: info for nid, info in node_info.items() if info["invert"] is not None}

    log.info("Total unique nodes: %d", total_nodes)
    log.info("Nodes with GIS inverts: %d", len(already_known))
    log.info("Nodes needing DEM elevation: %d", len(missing))

    # Store GIS-sourced nodes in cache (source='gis_invert')
    for nid, info in already_known.items():
        if nid not in cache:
            cache[nid] = {
                "lon": info["lon"],
                "lat": info["lat"],
                "elev_ft": info["invert"],
                "source": "gis_invert",
            }

    # Fetch DEM for missing nodes not already cached
    to_fetch = {nid: info for nid, info in missing.items() if nid not in cache}
    log.info("Need to query EPQS for %d nodes (already cached: %d)",
             len(to_fetch), len(missing) - len(to_fetch))

    if not to_fetch:
        log.info("All nodes already resolved. Nothing to fetch.")
    else:
        log.info("Starting EPQS queries at %.1fs per request...", REQUEST_DELAY_S)
        log.info("Estimated time: %.0f minutes", len(to_fetch) * REQUEST_DELAY_S / 60)

        success = 0
        fail = 0
        for i, (nid, info) in enumerate(to_fetch.items()):
            elev = query_epqs(info["lon"], info["lat"])
            if elev is not None:
                cache[nid] = {
                    "lon": info["lon"],
                    "lat": info["lat"],
                    "elev_ft": elev,
                    "source": "usgs_3dep",
                }
                success += 1
            else:
                cache[nid] = {
                    "lon": info["lon"],
                    "lat": info["lat"],
                    "elev_ft": None,
                    "source": "failed",
                }
                fail += 1

            if (i + 1) % 50 == 0:
                # Save checkpoint every 50 queries
                with open(OUT_PATH, "w") as f:
                    json.dump(cache, f, indent=2)
                log.info("  [%d/%d] checkpoint saved  (ok=%d fail=%d)",
                         i + 1, len(to_fetch), success, fail)

            time.sleep(REQUEST_DELAY_S)

        log.info("EPQS complete: %d ok, %d failed", success, fail)

    # Final save
    with open(OUT_PATH, "w") as f:
        json.dump(cache, f, indent=2)
    log.info("Saved %d node elevations -> %s", len(cache), OUT_PATH)

    # Summary
    counts = {}
    for v in cache.values():
        s = v["source"]
        counts[s] = counts.get(s, 0) + 1
    total_resolved = counts.get("gis_invert", 0) + counts.get("usgs_3dep", 0)
    pct = 100 * total_resolved / total_nodes if total_nodes else 0

    print("\n" + "=" * 60)
    print("Node Elevation Summary")
    print("=" * 60)
    print(f"  Total nodes:          {total_nodes}")
    print(f"  From GIS inverts:     {counts.get('gis_invert', 0)}")
    print(f"  From USGS 3DEP DEM:   {counts.get('usgs_3dep', 0)}")
    print(f"  Failed / no data:     {counts.get('failed', 0)}")
    print(f"  Coverage:             {pct:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
