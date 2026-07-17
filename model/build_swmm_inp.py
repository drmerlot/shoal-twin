"""
Step 3b / 4a: Build SWMM .inp from COA GIS + DEM + QPE
=======================================================
Converts the COA open channel GeoJSON + node_elevations.json into a
valid SWMM 5 input file, optionally injecting real QPE rainfall from
data/raw/qpe/storm_rainfall.parquet.

Design choices:
  - Open channels only for v1 skeleton (819 reaches, main conveyance)
  - Missing inverts filled from USGS 3DEP DEM (ground elev - 1 ft bank)
  - Cross-sections: TRAPEZOIDAL/RECT_OPEN/TRIANGULAR based on ditch_shape
  - Routing: DYNWAVE (full Saint-Venant, appropriate for flash flood sim)
  - Subcatchment: single lumped placeholder (replaced during calibration)
  - Manning n = 0.035 (natural channel, to be calibrated)

Outputs:
    model/shoal_creek_skeleton.inp          (no --storm: synthetic placeholder rainfall)
    model/shoal_creek_<storm_id>.inp        (--storm <id>: real QPE from IEM ASOS ATT)

Usage:
    uv run python model/build_swmm_inp.py
    uv run python model/build_swmm_inp.py --storm halloween_2013
    uv run python model/build_swmm_inp.py --storm memorial_day_2015
    uv run python model/build_swmm_inp.py --storm halloween_2015
"""

import argparse
import json
import logging
import math
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

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
OUT_DIR = Path("model")
QPE_DIR = Path("data/raw/qpe")
OC_PATH = GIS_DIR / "coa_open_channel_shl.geojson"
ELEV_PATH = GIS_DIR / "node_elevations.json"
QPE_PATH = QPE_DIR / "storm_rainfall.parquet"
INP_PATH = OUT_DIR / "shoal_creek_skeleton.inp"

DEFAULT_ROUGHNESS_OC = 0.035
DEFAULT_MAX_DEPTH = 10.0
DEFAULT_BANK_DEPTH = 1.0
WATERSHED_AREA_SQ_MI = 12.3
GAUGE_LON = -97.7480
GAUGE_LAT = 30.2747

# ---------------------------------------------------------------------------
# Channel routing: 3-reach representative main stem for Shoal Creek
# Invert elevations from USGS 3DEP 1m DEM (fetched 2025-07-18), minus
# estimated bank depth (4-8 ft). Downstream invert (510.6 ft) from
# pyswmm junction reading at USGS gauge 08158600.
# ---------------------------------------------------------------------------
CHANNEL_N_DEFAULT = 0.040   # Manning's n (calibration parameter, range 0.025-0.080)
CHANNEL_BW_FT     = 80.0   # Bottom width (ft) — COA GIS median 96 ft, conservative
CHANNEL_SS        = 3.0    # Side slope H:V — typical Austin natural channel
CHANNEL_MAX_DEPTH = 15.0   # Max depth (ft)

# Intermediate junctions: (id, invert_ft, lon, lat)
# DEM ground elevations: J_HEAD=642.6, J_MID=559.2, J_LOWER≈532 (interpolated)
CHANNEL_JUNCTIONS = [
    ("J_HEAD",  635.0, -97.7480, 30.3280),  # upper-mid (~30.328N), bank_depth≈8ft
    ("J_MID",   552.0, -97.7490, 30.3040),  # mid-watershed (~30.304N), bank_depth≈7ft
    ("J_LOWER", 525.0, -97.7490, 30.2850),  # lower-mid (~30.285N), interpolated
]

# Channel reaches: (name, from_node, to_node_or_None, length_ft, up_invert, dn_invert)
# to_node=None → resolved to upstream_of_outfall at write time
CHANNEL_REACHES_DEF = [
    ("R_UPPER", "J_HEAD",  "J_MID",   9_000, 635.0, 552.0),
    ("R_MID",   "J_MID",   "J_LOWER", 7_000, 552.0, 525.0),
    ("R_LOWER", "J_LOWER", None,      3_500, 525.0, 510.6),
]

# Storm simulation windows (start, end)
STORM_WINDOWS = {
    "halloween_2013":    ("10/29/2013", "11/02/2013"),
    "memorial_day_2015": ("05/24/2015", "05/28/2015"),
    "halloween_2015":    ("10/29/2015", "11/02/2015"),
    "july_2025":         ("07/01/2025", "07/08/2025"),
}

# Preferred QPE station (Camp Mabry is closest to SHL watershed)
QPE_STATION = "ATT"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def reach_endpoints(feature):
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
    return lines[0][0], lines[-1][-1]


def haversine_ft(lon1, lat1, lon2, lat2):
    R = 20925524.9
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Cross-section mapper
# ---------------------------------------------------------------------------

def swmm_xsection(ditch_shape, bottom_width, top_width, depth):
    shape = (ditch_shape or "NODATA").upper().strip()

    def f(v, default=0.0):
        try:
            return float(v) if v not in (None, "", "NODATA") else default
        except (ValueError, TypeError):
            return default

    bw = f(bottom_width)
    tw = f(top_width)
    d = f(depth, DEFAULT_MAX_DEPTH)
    if d <= 0:
        d = DEFAULT_MAX_DEPTH

    if shape in ("TRAP", "TRAPEZOIDAL", "NODATA"):
        if bw > 0:
            side_slope = (tw - bw) / (2 * d) if (tw > bw and d > 0) else 2.0
            side_slope = max(0.5, min(side_slope, 10.0))
        else:
            bw, side_slope = 10.0, 2.0
        return ("TRAPEZOIDAL", f"{d:.2f}", f"{bw:.2f}", f"{side_slope:.2f}", f"{side_slope:.2f}")

    elif shape == "RECT":
        w = bw if bw > 0 else 10.0
        return ("RECT_OPEN", f"{d:.2f}", f"{w:.2f}", "0", "0")

    elif shape in ("TRI", "TRIANGULAR"):
        ss = (tw / (2 * d)) if (tw > 0 and d > 0) else 3.0
        ss = max(0.5, min(ss, 10.0))
        return ("TRIANGULAR", f"{d:.2f}", f"{ss:.2f}", "0", "0")

    elif shape in ("IRR", "IRREGULAR"):
        return ("TRAPEZOIDAL", "8.00", "20.00", "2.00", "2.00")

    else:
        return ("TRAPEZOIDAL", "5.00", "10.00", "2.00", "2.00")


# ---------------------------------------------------------------------------
# Network builder
# ---------------------------------------------------------------------------

def build_network(features, elevations):
    node_raw = {}

    for feat in features:
        props = feat["properties"]
        up_id = str(props["up_node_id"])
        dn_id = str(props["dn_node_id"])
        up_inv = props.get("up_invert")
        dn_inv = props.get("dn_invert")
        up_type = props.get("up_node_type", "JUNCTION")
        dn_type = props.get("dn_node_type", "JUNCTION")

        up_pt, dn_pt = reach_endpoints(feat)
        if up_pt is None:
            continue

        for nid, pt, inv, ntype, degree_key in [
            (up_id, up_pt, up_inv, up_type, "out"),
            (dn_id, dn_pt, dn_inv, dn_type, "in"),
        ]:
            if nid not in node_raw:
                node_raw[nid] = {
                    "lon": pt[0], "lat": pt[1],
                    "invert": float(inv) if inv not in (None, "") else None,
                    "node_type": ntype,
                    "out_degree": 0, "in_degree": 0,
                }
            else:
                if node_raw[nid]["invert"] is None and inv not in (None, ""):
                    node_raw[nid]["invert"] = float(inv)
            node_raw[nid][f"{degree_key}_degree"] += 1

    # Identify outfall: prefer true topological sinks (in_degree>0, out_degree=0)
    # over PROUTFALL-typed nodes (which may be upstream sources with in_degree=0)
    sinks = {nid: info for nid, info in node_raw.items()
             if info["in_degree"] > 0 and info["out_degree"] == 0}
    outfall_typed = {nid: info for nid, info in node_raw.items()
                     if info["node_type"] in ("OUTFALL", "PROUTFALL")
                     and info["in_degree"] > 0}     # typed outfalls that receive flow
    candidates = (sinks or outfall_typed)  # prefer sinks; fall back to typed

    if candidates:
        outfall_id = min(candidates, key=lambda nid: haversine_ft(
            node_raw[nid]["lon"], node_raw[nid]["lat"], GAUGE_LON, GAUGE_LAT
        ))
    else:
        with_elev = [(nid, info["invert"]) for nid, info in node_raw.items()
                     if info.get("invert") is not None]
        outfall_id = min(with_elev, key=lambda x: x[1])[0] if with_elev else None

    log.info("Outfall node: %s  (lat=%.4f lon=%.4f)",
             outfall_id,
             node_raw[outfall_id]["lat"] if outfall_id else 0,
             node_raw[outfall_id]["lon"] if outfall_id else 0)

    # Resolve elevations
    nodes = {}
    counts = {"gis_invert": 0, "dem_filled": 0, "fallback": 0}

    for nid, info in node_raw.items():
        if info["invert"] is not None:
            elev = info["invert"]
            src = "gis_invert"
        elif nid in elevations and elevations[nid].get("elev_ft") is not None:
            elev = elevations[nid]["elev_ft"] - DEFAULT_BANK_DEPTH
            src = "dem_filled"
        else:
            elev = 500.0
            src = "fallback"

        counts[src] += 1
        nodes[nid] = {
            "lon": info["lon"], "lat": info["lat"],
            "elev_ft": elev, "source": src,
            "max_depth": DEFAULT_MAX_DEPTH,
            "is_outfall": (nid == outfall_id),
        }

    log.info("Node elevation sources: gis=%d  dem=%d  fallback=%d",
             counts["gis_invert"], counts["dem_filled"], counts["fallback"])

    # Build conduits
    conduits = []
    skipped = 0
    for i, feat in enumerate(features):
        props = feat["properties"]
        up_id = str(props["up_node_id"])
        dn_id = str(props["dn_node_id"])

        if up_id not in nodes or dn_id not in nodes:
            skipped += 1
            continue

        length_raw = props.get("length")
        try:
            length = float(length_raw) if length_raw not in (None, "") else None
        except (ValueError, TypeError):
            length = None

        if length is None or length <= 0:
            n = nodes[up_id]
            m = nodes[dn_id]
            length = max(haversine_ft(n["lon"], n["lat"], m["lon"], m["lat"]), 1.0)

        conduits.append({
            "name": f"C{props.get('drainage_id', i)}",
            "from_node": up_id,
            "to_node": dn_id,
            "length": length,
            "roughness": DEFAULT_ROUGHNESS_OC,
            "xs": swmm_xsection(
                props.get("ditch_shape"),
                props.get("bottom_width"),
                props.get("top_width"),
                props.get("depth"),
            ),
            "drainage_id": str(props.get("drainage_id", "")),
        })

    log.info("Conduits: %d built, %d skipped", len(conduits), skipped)
    return nodes, conduits, outfall_id


# ---------------------------------------------------------------------------
# INP writer
# ---------------------------------------------------------------------------

def write_inp(nodes, conduits, outfall_id, out_path, sim_start=None, sim_end=None,
              qpe_meta=None, qpe_rows=None):
    junctions = {nid: n for nid, n in nodes.items() if not n["is_outfall"]}
    outfalls  = {nid: n for nid, n in nodes.items() if n["is_outfall"]}

    if not outfalls and junctions:
        lowest = min(junctions, key=lambda nid: junctions[nid]["elev_ft"])
        outfalls[lowest] = junctions.pop(lowest)

    lines = []
    A = lines.append

    A("[TITLE]")
    A(";;Project Title/Notes")
    A("Shoal Creek Flood Forecast Model -- Skeleton v1")
    A("Generated by model/build_swmm_inp.py from COA GIS + USGS 3DEP DEM")
    A("")

    s_start = sim_start or "10/31/2013"
    s_end   = sim_end   or "11/02/2013"

    A("[OPTIONS]")
    A(";;Option             Value")
    A("FLOW_UNITS           CFS")
    A("INFILTRATION         GREEN_AMPT")
    A("FLOW_ROUTING         DYNWAVE")
    A("LINK_OFFSETS         DEPTH")
    A("MIN_SLOPE            0.0001")
    A("ALLOW_PONDING        YES")
    A("SKIP_STEADY_STATE    NO")
    A("")
    A(f"START_DATE           {s_start}")
    A(f"START_TIME           00:00:00")
    A(f"REPORT_START_DATE    {s_start}")
    A(f"REPORT_START_TIME    00:00:00")
    A(f"END_DATE             {s_end}")
    A(f"END_TIME             00:00:00")
    A("SWEEP_START          01/01")
    A("SWEEP_END            12/31")
    A("DRY_DAYS             5")
    A("REPORT_STEP          00:05:00")
    A("WET_STEP             00:05:00")
    A("DRY_STEP             01:00:00")
    A("ROUTING_STEP         30")
    A("")
    A("INERTIAL_DAMPING     PARTIAL")
    A("NORMAL_FLOW_LIMITED  BOTH")
    A("FORCE_MAIN_EQUATION  H-W")
    A("VARIABLE_STEP        0.75")
    A("LENGTHENING_STEP     0")
    A("MIN_SURFAREA         12.566")
    A("MAX_TRIALS           8")
    A("HEAD_TOLERANCE       0.005")
    A("SYS_FLOW_TOL         5")
    A("LAT_FLOW_TOL         5")
    A("MINIMUM_STEP         0.5")
    A("THREADS              4")
    A("")

    A("[EVAPORATION]")
    A(";;Data Source    Parameters")
    A("CONSTANT         0.0")
    A("DRY_ONLY         NO")
    A("")

    A("[RAINGAGES]")
    A(";;Name           Format    Interval SCF      Source")
    A(";;-------------- --------- -------- -------- --------------------")
    ts_name = f"RG_{qpe_meta['storm_id']}" if qpe_meta else "RAINFALL_TS"
    # Derive interval string from QPE timestep (e.g. 5-min → "0:05", 60-min → "1:00")
    interval_str = qpe_meta.get("interval_str", "1:00") if qpe_meta else "1:00"
    A(f"RG1              INTENSITY {interval_str}     1.0      TIMESERIES {ts_name}")
    A("")

    out_id = list(outfalls.keys())[0] if outfalls else "OUTFALL1"
    area_acres = WATERSHED_AREA_SQ_MI * 640

    # Find the GIS junction immediately upstream of the outfall (if any).
    upstream_of_outfall = out_id
    for c in conduits:
        if c["to_node"] == out_id and c["from_node"] in junctions:
            upstream_of_outfall = c["from_node"]
            break
    log.info("Channel terminus: %s (upstream GIS junction of outfall %s)",
             upstream_of_outfall, out_id)

    # Resolve R_LOWER target: use the upstream_of_outfall GIS junction if available,
    # otherwise route directly to the outfall.
    r_lower_to = upstream_of_outfall

    A("[SUBCATCHMENTS]")
    A(";;Name           Rain Gage        Outlet           Area     %Imperv  Width    %Slope   CurbLen  SnowPack")
    A(";;-------------- ---------------- ---------------- -------- -------- -------- -------- -------- --------")
    # SC_UPPER: upper/mid watershed → J_HEAD (full 19,500ft channel routing)
    # SC_LOWER: lower watershed → J_LOWER (only 3,500ft routing to gauge)
    # Areas are placeholder; patch_inp() splits based on lower_frac calibration param.
    A(f"{'SC_UPPER':<16} {'RG1':<16} {'J_HEAD':<16} {area_acres:<8.0f} {'60':<8} {'5000':<8} {'1.5':<8} {'0'}")
    A(f"{'SC_LOWER':<16} {'RG1':<16} {'J_LOWER':<16} {'0':<8} {'60':<8} {'5000':<8} {'1.5':<8} {'0'}")
    A("")
    A("")

    A("[SUBAREAS]")
    A(";;Subcatchment   N-Imperv   N-Perv     S-Imperv   S-Perv     PctZero    RouteTo    PctRouted")
    A(";;-------------- ---------- ---------- ---------- ---------- ---------- ---------- ----------")
    A(f"{'SC_UPPER':<16} {'0.011':<10} {'0.10':<10} {'0.05':<10} {'0.10':<10} {'25':<10} {'OUTLET'}")
    A(f"{'SC_LOWER':<16} {'0.011':<10} {'0.10':<10} {'0.05':<10} {'0.10':<10} {'25':<10} {'OUTLET'}")
    A("")

    A("[INFILTRATION]")
    A(";;Subcatchment   Suction    Ksat       IMD")
    A(";;-------------- ---------- ---------- ----------")
    A(f"{'SC_UPPER':<16} {'3.50':<10} {'0.10':<10} {'0.30'}")
    A(f"{'SC_LOWER':<16} {'3.50':<10} {'0.10':<10} {'0.30'}")
    A("")


    A("[JUNCTIONS]")
    A(";;Name           Elevation  MaxDepth   InitDepth  SurDepth   Aponded")
    A(";;-------------- ---------- ---------- ---------- ---------- ----------")
    # Channel routing junctions (main stem Shoal Creek)
    for (jid, inv, jlon, jlat) in CHANNEL_JUNCTIONS:
        A(f"{jid:<16} {inv:<10.2f} {CHANNEL_MAX_DEPTH:<10.1f} {'0':<10} {'0':<10} {'0'}")
    # GIS-derived junctions (orphaned but included for model completeness)
    for nid, n in sorted(junctions.items()):
        A(f"{nid:<16} {n['elev_ft']:<10.2f} {DEFAULT_MAX_DEPTH:<10.1f} {'0':<10} {'0':<10} {'0'}")
    A("")

    A("[OUTFALLS]")
    A(";;Name           Elevation  Type       Stage Data       Gated    Route To")
    A(";;-------------- ---------- ---------- ---------------- -------- ----------------")
    for nid, n in sorted(outfalls.items()):
        A(f"{nid:<16} {n['elev_ft']:<10.2f} {'FREE':<10} {'':16} {'NO'}")
    A("")

    A("[CONDUITS]")
    A(";;Name           From Node        To Node          Length     Roughness  InOffset   OutOffset  InitFlow   MaxFlow")
    A(";;-------------- ---------------- ---------------- ---------- ---------- ---------- ---------- ---------- ----------")
    # Main stem channel reaches (R_UPPER → R_MID → R_LOWER → gauge)
    for (rname, rfrom, rto, rlen, r_up_inv, r_dn_inv) in CHANNEL_REACHES_DEF:
        to_node = r_lower_to if rto is None else rto
        A(f"{rname:<16} {rfrom:<16} {to_node:<16} "
          f"{rlen:<10.1f} {CHANNEL_N_DEFAULT:<10.4f} {'0':<10} {'0':<10} {'0':<10} {'0'}")
    # GIS-derived conduits (orphaned — included for completeness but carry no flow)
    for c in conduits:
        A(f"{c['name']:<16} {c['from_node']:<16} {c['to_node']:<16} "
          f"{c['length']:<10.1f} {c['roughness']:<10.4f} {'0':<10} {'0':<10} {'0':<10} {'0'}")
    A("")

    A("[XSECTIONS]")
    A(";;Link           Shape        Geom1            Geom2      Geom3      Geom4      Barrels    Culvert")
    A(";;-------------- ------------ ---------------- ---------- ---------- ---------- ---------- ----------")
    # Main stem channel cross-sections: TRAPEZOIDAL depth/bottom_width/left_ss/right_ss
    for (rname, *_) in CHANNEL_REACHES_DEF:
        A(f"{rname:<16} {'TRAPEZOIDAL':<12} {CHANNEL_MAX_DEPTH:<16.2f} "
          f"{CHANNEL_BW_FT:<10.2f} {CHANNEL_SS:<10.2f} {CHANNEL_SS:<10.2f} {'1'}")
    # GIS-derived conduit cross-sections
    for c in conduits:
        s, g1, g2, g3, g4 = c["xs"]
        A(f"{c['name']:<16} {s:<12} {g1:<16} {g2:<10} {g3:<10} {g4:<10} {'1'}")
    A("")

    A("[TIMESERIES]")
    A(";;Name           Date       Time       Value")
    A(";;-------------- ---------- ---------- ----------")
    if qpe_meta and qpe_rows:
        ts_name = f"RG_{qpe_meta['storm_id']}"
        station = qpe_meta["station"]
        A(f";; QPE source: IEM ASOS {station} ({qpe_meta['n_rows']} obs)"
          f"  storm: {qpe_meta['storm_id']}")
        A(f";; Total precip: {qpe_meta['total_in']:.2f} in"
          f"  peak: {qpe_meta['peak_in_hr']:.2f} in/hr at {qpe_meta['peak_time']}")
        for dt_str, val in qpe_rows:
            A(f"{ts_name:<16} {dt_str:<10} {val:.4f}")
    else:
        A(";; Placeholder -- run data/fetch_qpe.py then rebuild with --storm <id>")
        A("RAINFALL_TS      10/31/2013 00:00      0.00")
        A("RAINFALL_TS      10/31/2013 06:00      0.50")
        A("RAINFALL_TS      10/31/2013 12:00      1.20")
        A("RAINFALL_TS      10/31/2013 18:00      0.30")
        A("RAINFALL_TS      11/01/2013 00:00      0.00")
    A("")

    A("[REPORT]")
    A("INPUT      NO")
    A("CONTROLS   NO")
    A("SUBCATCHMENTS ALL")
    A("NODES ALL")
    A("LINKS ALL")
    A("")

    A("[TAGS]")
    A("")

    all_lons = [n["lon"] for n in nodes.values()]
    all_lats = [n["lat"] for n in nodes.values()]
    A("[MAP]")
    A(f"DIMENSIONS {min(all_lons):.6f} {min(all_lats):.6f} {max(all_lons):.6f} {max(all_lats):.6f}")
    A("Units      Degrees")
    A("")

    A("[COORDINATES]")
    A(";;Node           X-Coord            Y-Coord")
    A(";;-------------- ------------------ ------------------")
    # Channel routing junctions
    for (jid, inv, jlon, jlat) in CHANNEL_JUNCTIONS:
        A(f"{jid:<16} {jlon:<18.7f} {jlat:<18.7f}")
    for nid, n in sorted(nodes.items()):
        A(f"{nid:<16} {n['lon']:<18.7f} {n['lat']:<18.7f}")
    A("")


    A("[SYMBOLS]")
    A(";;Gage           X-Coord            Y-Coord")
    A(";;-------------- ------------------ ------------------")
    A(f"{'RG1':<16} {GAUGE_LON:<18.7f} {GAUGE_LAT:<18.7f}")
    A("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log.info("Wrote %d lines -> %s  (%.1f KB)", len(lines), out_path, out_path.stat().st_size / 1024)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_qpe(storm_id: str, station: str = QPE_STATION):  # noqa: E501
    """
    Load QPE timeseries for a storm from the merged parquet.
    Returns (qpe_meta dict, qpe_rows list of (date_str, value)) or (None, None).
    """
    if not HAS_PANDAS:
        log.warning("pandas not available -- cannot load QPE")
        return None, None
    if not QPE_PATH.exists():
        log.warning("storm_rainfall.parquet not found -- run data/fetch_qpe.py first")
        return None, None

    df = pd.read_parquet(QPE_PATH)
    mask = (df["storm_id"] == storm_id) & (df["station"] == station)
    sub = df[mask].copy()

    if sub.empty:
        log.warning("No QPE data for storm=%s station=%s", storm_id, station)
        return None, None

    sub["valid"] = pd.to_datetime(sub["valid"], utc=True)
    sub = sub.sort_values("valid").reset_index(drop=True)
    sub["precip_in_per_hr"] = sub["precip_in_per_hr"].fillna(0.0).clip(lower=0.0)

    # Build SWMM TIMESERIES rows: "MM/DD/YYYY  HH:MM  value"
    rows = []
    for _, row in sub.iterrows():
        dt = row["valid"].to_pydatetime()
        date_str = dt.strftime("%m/%d/%Y")
        time_str = dt.strftime("%H:%M")
        val = row["precip_in_per_hr"]
        rows.append((f"{date_str}  {time_str}", val))

    precip_col = sub["precip_in_per_hr"]
    total_in = precip_col.sum()
    peak_in_hr = precip_col.max()
    peak_idx = precip_col.idxmax()
    peak_time = sub.loc[peak_idx, "valid"].strftime("%m/%d %H:%MZ") if pd.notna(peak_idx) else "N/A"

    # Derive timestep interval from consecutive record spacing
    if len(sub) >= 2:
        dt_diff = (sub["valid"].iloc[1] - sub["valid"].iloc[0]).total_seconds() / 60
    else:
        dt_diff = 60  # default hourly
    interval_min = int(round(dt_diff))
    interval_str = f"{interval_min // 60}:{interval_min % 60:02d}"

    meta = {
        "storm_id":     storm_id,
        "station":      station,
        "n_rows":       len(rows),
        "total_in":     round(total_in, 3),
        "peak_in_hr":   round(peak_in_hr, 3),
        "peak_time":    peak_time,
        "interval_min": interval_min,
        "interval_str": interval_str,
    }
    log.info("QPE loaded: %s/%s  %d obs  total=%.2fin  peak=%.2fin/hr at %s",
             storm_id, station, len(rows), total_in, peak_in_hr, peak_time)
    return meta, rows


def main():
    parser = argparse.ArgumentParser(
        description="Build SWMM .inp from COA GIS + DEM + optional QPE"
    )
    parser.add_argument(
        "--storm",
        choices=list(STORM_WINDOWS.keys()),
        default=None,
        help="Inject real QPE for this storm window",
    )
    parser.add_argument(
        "--station",
        default=None,
        help="QPE station to use (default: ATT for hourly, KATT_5min for 5-min). "
             "Must match a station name in storm_rainfall.parquet.",
    )
    args = parser.parse_args()

    storm_id = args.storm
    log.info("Step 3b/4a: Build SWMM .inp  storm=%s", storm_id or "(skeleton, no QPE)")

    with open(OC_PATH) as f:
        oc = json.load(f)
    features = oc["features"]
    log.info("Loaded %d open channel features", len(features))

    if ELEV_PATH.exists():
        with open(ELEV_PATH) as f:
            elevations = json.load(f)
        log.info("Loaded %d node elevations from cache", len(elevations))
    else:
        log.warning("node_elevations.json not found -- run fetch_dem_elevations.py first")
        elevations = {}

    # QPE
    qpe_meta, qpe_rows = None, None
    station_arg = args.station or QPE_STATION  # fallback to ATT
    if storm_id:
        qpe_meta, qpe_rows = load_qpe(storm_id, station=station_arg)

    # Storm window
    sim_start, sim_end = None, None
    if storm_id and storm_id in STORM_WINDOWS:
        sim_start, sim_end = STORM_WINDOWS[storm_id]

    # Output file
    if storm_id:
        out_path = OUT_DIR / f"shoal_creek_{storm_id}.inp"
    else:
        out_path = INP_PATH

    nodes, conduits, outfall_id = build_network(features, elevations)
    write_inp(nodes, conduits, outfall_id, out_path,
              sim_start=sim_start, sim_end=sim_end,
              qpe_meta=qpe_meta, qpe_rows=qpe_rows)

    gis_n = sum(1 for n in nodes.values() if n["source"] == "gis_invert")
    dem_n = sum(1 for n in nodes.values() if n["source"] == "dem_filled")
    fb_n  = sum(1 for n in nodes.values() if n["source"] == "fallback")
    total = len(nodes)

    print("\n" + "=" * 65)
    print("SWMM Build Summary")
    print("=" * 65)
    print(f"  Storm window:                    {storm_id or '(skeleton — no QPE)'}")
    if sim_start:
        print(f"  Simulation:                      {sim_start} → {sim_end}")
    print(f"  Conduits (open channel reaches): {len(conduits)}")
    print(f"  Junctions:                       {sum(1 for n in nodes.values() if not n['is_outfall'])}")
    print(f"  Outfalls:                        {sum(1 for n in nodes.values() if n['is_outfall'])}")
    print(f"  Primary outfall ID:              {outfall_id}")
    print()
    print(f"  Node elevation resolution:")
    print(f"    GIS inverts:     {gis_n:4d}  ({100*gis_n/total:.0f}%)")
    print(f"    DEM filled:      {dem_n:4d}  ({100*dem_n/total:.0f}%)")
    print(f"    Fallback 500ft:  {fb_n:4d}  ({100*fb_n/total:.0f}%)")
    print()
    if qpe_meta:
        print(f"  QPE rainfall (IEM ASOS {qpe_meta['station']}):")
        print(f"    Total:           {qpe_meta['total_in']:.2f} in")
        print(f"    Peak intensity:  {qpe_meta['peak_in_hr']:.2f} in/hr  at {qpe_meta['peak_time']}")
        print(f"    Observations:    {qpe_meta['n_rows']} hourly")
        print()
    else:
        print(f"  Rainfall: synthetic placeholder (run with --storm to inject QPE)")
        print()
    print(f"  Output: {out_path}")
    print("=" * 65)
    print()
    if fb_n > 50:
        print("  WARNING: >50 fallback elevations -- run fetch_dem_elevations.py first")
    elif fb_n > 0:
        print(f"  NOTE: {fb_n} nodes used fallback elev -- minor impact")
    if storm_id:
        print(f"  Next: .venv/bin/python model/validate_inp.py --storm {storm_id}")
    else:
        print("  Next: uv run python model/build_swmm_inp.py --storm halloween_2013")


if __name__ == "__main__":
    main()
