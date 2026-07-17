"""
Step 4a: Fetch QPE (rain gauge) for SWMM calibration storms
============================================================
Pulls hourly ASOS precipitation from Iowa Environmental Mesonet (IEM)
for Camp Mabry (ATT) and ABIA Airport (AUS) for three calibration
storm windows.

Sources:
  - Primary:  IEM ASOS ATT (Camp Mabry, Austin) -- 3.5 mi from watershed centroid
  - Backup:   IEM ASOS AUS (ABIA Airport)       -- 11 mi, used as cross-check
  Both are free, no authentication, hourly p01i (precip past 1 hour, inches).

Outputs:
  data/raw/qpe/<storm_id>_<station>.csv   -- per-storm, per-station hourly precip
  data/raw/qpe/storm_rainfall.parquet    -- merged authoritative dataset

SWMM TIMESERIES format note:
  SWMM [TIMESERIES] uses cumulative depth (in) or intensity (in/hr).
  We store p01i (in/hr = in / 1hr interval), which maps directly to
  SWMM INTENSITY format with 1:00 interval.

Usage:
    uv run python data/fetch_qpe.py
"""

import io
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QPE_DIR = Path("data/raw/qpe")

# Three calibration storm windows (+1 day padding each side)
STORMS = {
    "halloween_2013":    ("2013-10-29", "2013-11-02"),
    "memorial_day_2015": ("2015-05-24", "2015-05-28"),
    "halloween_2015":    ("2015-10-29", "2015-11-02"),
}

# IEM ASOS station codes
IEM_STATIONS = {
    "ATT": "Camp Mabry Austin (primary, 3.5 mi from SHL centroid)",
    "AUS": "ABIA Airport (backup cross-check, 11 mi)",
}

IEM_BASE = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# Rate limiting: IEM asks for at least 1s between requests
REQUEST_DELAY_S = 2.5


# ---------------------------------------------------------------------------
# IEM fetch
# ---------------------------------------------------------------------------

def build_iem_url(station: str, start_date: str, end_date: str) -> str:
    """Build IEM ASOS CSV download URL for hourly precip (p01i)."""
    y1, m1, d1 = start_date.split("-")
    y2, m2, d2 = end_date.split("-")
    return (
        f"{IEM_BASE}?station={station}"
        f"&data=p01i"           # precip past 1 hour (inches)
        f"&year1={y1}&month1={m1}&day1={d1}"
        f"&year2={y2}&month2={m2}&day2={d2}"
        f"&tz=UTC&format=comma&latlon=no&missing=M&trace=T"
        f"&direct=no&report_type=1,3"  # type 1 = hourly, 3 = special
    )


def fetch_iem_asos(station: str, storm_id: str, start_date: str, end_date: str,
                   cache_path: Path) -> pd.DataFrame | None:
    """Fetch IEM ASOS hourly precip. Returns DataFrame or None on failure."""
    if cache_path.exists():
        log.info("  Loaded from cache: %s", cache_path.name)
        return pd.read_csv(cache_path, parse_dates=["valid"])

    url = build_iem_url(station, start_date, end_date)
    log.info("  Fetching %s/%s from IEM...", station, storm_id)
    log.debug("  URL: %s", url)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "shoal-twin/1.0 (research)"})
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as e:
        log.error("  IEM HTTP %s for %s/%s", e.code, station, storm_id)
        return None
    except Exception as e:
        log.error("  IEM fetch error for %s/%s: %s", station, storm_id, e)
        return None

    # Strip IEM comment lines (start with #)
    csv_lines = [l for l in raw.split("\n") if not l.startswith("#") and l.strip()]
    if len(csv_lines) < 2:
        log.warning("  No data returned for %s/%s", station, storm_id)
        return None

    csv_text = "\n".join(csv_lines)
    df = pd.read_csv(io.StringIO(csv_text), dtype=str)

    # Rename columns
    df.columns = [c.strip() for c in df.columns]
    if "valid" not in df.columns:
        log.warning("  Unexpected columns: %s", list(df.columns))
        return None

    df["valid"] = pd.to_datetime(df["valid"], utc=True, errors="coerce")
    df = df.dropna(subset=["valid"])
    df = df.sort_values("valid").reset_index(drop=True)

    # Parse p01i: 'M' = missing, 'T' = trace (set to 0.001), numeric = inches
    def parse_precip(v):
        v = str(v).strip()
        if v in ("M", "", "nan"):
            return None
        if v == "T":
            return 0.001
        try:
            return float(v)
        except ValueError:
            return None

    df["precip_in_per_hr"] = df["p01i"].apply(parse_precip)

    # Gap-fill: if ≤ 2 consecutive missing, interpolate; longer gaps stay NaN
    df["precip_in_per_hr"] = (
        df["precip_in_per_hr"]
        .interpolate(method="linear", limit=2, limit_direction="forward")
        .clip(lower=0.0)
    )

    # Add metadata columns
    df["station"] = station
    df["storm_id"] = storm_id

    df = df[["station", "storm_id", "valid", "precip_in_per_hr"]].copy()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    log.info("  Saved %d rows -> %s", len(df), cache_path.name)
    return df


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

def summarize_storm(df: pd.DataFrame, station: str, storm_id: str) -> dict:
    """Compute summary stats for a storm/station dataframe."""
    if df is None or df.empty:
        return {"storm_id": storm_id, "station": station, "status": "EMPTY"}

    precip = df["precip_in_per_hr"].dropna()
    total = precip.sum()
    peak = precip.max()
    n_hours = len(precip)
    n_missing = df["precip_in_per_hr"].isna().sum()
    n_wet = (precip > 0.005).sum()

    # Peak hour timestamp
    peak_idx = df["precip_in_per_hr"].idxmax()
    peak_time = df.loc[peak_idx, "valid"] if pd.notna(peak_idx) else None

    return {
        "storm_id": storm_id,
        "station": station,
        "total_in": round(total, 3),
        "peak_in_hr": round(peak, 3),
        "peak_time_utc": str(peak_time) if peak_time else "N/A",
        "n_obs": n_hours,
        "n_missing": n_missing,
        "n_wet_hours": n_wet,
        "status": "OK",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("Step 4a: Fetch QPE for SWMM calibration storms")
    QPE_DIR.mkdir(parents=True, exist_ok=True)

    all_frames = []
    summaries = []

    for storm_id, (start_date, end_date) in STORMS.items():
        log.info("")
        log.info("Storm: %s  (%s – %s)", storm_id, start_date, end_date)

        for station, desc in IEM_STATIONS.items():
            log.info("  Station: %s (%s)", station, desc.split(" (")[0])

            cache_path = QPE_DIR / f"{storm_id}_{station.lower()}.csv"
            df = fetch_iem_asos(station, storm_id, start_date, end_date, cache_path)

            summary = summarize_storm(df, station, storm_id)
            summaries.append(summary)

            if df is not None and not df.empty:
                all_frames.append(df)

            time.sleep(REQUEST_DELAY_S)

    # Merge into single parquet
    if not all_frames:
        log.error("No data fetched. Check network and IEM availability.")
        return

    merged = pd.concat(all_frames, ignore_index=True)
    merged["valid"] = pd.to_datetime(merged["valid"], utc=True)
    merged = merged.sort_values(["storm_id", "station", "valid"]).reset_index(drop=True)

    out_path = QPE_DIR / "storm_rainfall.parquet"
    merged.to_parquet(out_path, index=False)
    log.info("Saved merged parquet: %s  (%d rows)", out_path, len(merged))

    # Print summary table
    print("\n" + "=" * 72)
    print("QPE Fetch Summary")
    print("=" * 72)
    print(f"{'Storm':<22} {'Stn':<5} {'Total(in)':<12} {'Peak(in/hr)':<13} {'Peak UTC':<22} {'Status'}")
    print("-" * 72)
    for s in summaries:
        peak_t = s.get("peak_time_utc", "N/A")
        if peak_t != "N/A":
            peak_t = peak_t[5:16]  # MM-DD HH:MM
        print(f"{s['storm_id']:<22} {s['station']:<5} "
              f"{s.get('total_in', 0):<12.2f} {s.get('peak_in_hr', 0):<13.2f} "
              f"{peak_t:<22} {s['status']}")
    print("=" * 72)
    print(f"\nOutput: {out_path}  ({len(merged)} total obs)")

    print()
    print("Cross-check (ATT vs AUS total precip):")
    for storm_id in STORMS:
        att = next((s.get("total_in") for s in summaries
                    if s["storm_id"] == storm_id and s["station"] == "ATT"
                    and s["status"] == "OK"), None)
        aus = next((s.get("total_in") for s in summaries
                    if s["storm_id"] == storm_id and s["station"] == "AUS"
                    and s["status"] == "OK"), None)
        if att is not None and aus is not None and att > 0:
            pct_diff = abs(att - aus) / att * 100
            flag = "⚠️  HIGH DIVERGENCE" if pct_diff > 30 else "✓"
            print(f"  {storm_id:<22}  ATT={att:.2f}in  AUS={aus:.2f}in  "
                  f"diff={pct_diff:.0f}%  {flag}")
        elif att is None and aus is not None:
            print(f"  {storm_id:<22}  ATT=MISSING  AUS={aus:.2f}in  -- rerun to fill ATT")
        elif aus is None and att is not None:
            print(f"  {storm_id:<22}  ATT={att:.2f}in  AUS=MISSING  -- rerun to fill AUS")
        else:
            print(f"  {storm_id:<22}  both EMPTY")

    print()
    print("Next step:")
    print("  uv run python model/build_swmm_inp.py --storm halloween_2013")
    print("  .venv/bin/python model/validate_inp.py")


if __name__ == "__main__":
    main()
