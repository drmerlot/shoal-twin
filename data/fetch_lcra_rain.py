"""
data/fetch_lcra_rain.py
========================
Pull 15-min tipping-bucket rainfall data from LCRA COA HYDROMET for the
three in-watershed Shoal Creek gauges:

  2400 - Shoal Creek at W 45th Street       (30.319°N, 97.749°W)
  2510 - Shoal Creek at Shoal Creek Blvd    (30.346°N, 97.745°W)
  2520 - Shoal Creek at Foster Lane         (30.358°N, 97.739°W)

Data format:
  value1Type = "PC" = Precipitation Counter (cumulative tipping bucket inches,
  resets periodically). Each record is ~15 min apart. Rainfall per interval =
  diff(value1) when positive; counter reset handled by sign check.

API:
  https://hydromet.lcra.org/api/CoaHistoricalData/GetDataBySite/{site}/rain/{start}/{end}

Output: data/raw/qpe/{storm_id}_15min_lcra.csv
        Columns: valid_utc, precip_15min_in, intensity_in_hr, n_sites_avg

Usage:
    uv run python data/fetch_lcra_rain.py
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUT_DIR = Path("data/raw/qpe")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LCRA_BASE = "https://hydromet.lcra.org/api/CoaHistoricalData/GetDataBySite"

# Three in-watershed tipping-bucket gauges
SITES = {
    "2400": "Shoal Creek @ W 45th St",
    "2510": "Shoal Creek @ Shoal Creek Blvd",
    "2520": "Shoal Creek @ Foster Lane",
}

# Storms to fetch — use broad windows to capture full event
STORMS = [
    {
        "id":    "july_2025",
        "start": "2025-07-01",
        "end":   "2025-07-08",
        "note":  "July 2025 (USGS obs 9,340 CFS, event 312)",
    },
]


def fetch_site(site_id: str, start: str, end: str) -> list[dict]:
    """Fetch raw records from LCRA CoA historical API."""
    url = f"{LCRA_BASE}/{site_id}/rain/{start}/{end}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode())
        return data.get("records", [])
    except urllib.error.HTTPError as e:
        log.warning("HTTP %s fetching site %s: %s", e.code, site_id, url)
        return []
    except Exception as e:
        log.warning("Error fetching site %s: %s", site_id, e)
        return []


def pc_to_increments(records: list[dict]) -> pd.Series:
    """
    Convert PC counter records to 15-min rainfall increments.

    Records arrive in reverse-chronological order from the API.
    value1 = cumulative tipping bucket counter (inches).

    Strategy:
      1. Build a time-indexed Series of raw PC values (irregular spacing).
      2. Resample to regular 15-min grid using last() — captures the most
         recent reading within each window.
      3. Compute diff() on the regular grid.
      4. Negative diffs = counter reset → treat as 0.
    """
    if not records:
        return pd.Series(dtype=float)

    rows = []
    for r in records:
        dt = r.get("dateTime")
        v1 = r.get("value1")
        if dt is not None and v1 is not None:
            rows.append({"valid_utc": pd.Timestamp(dt, tz="UTC"), "pc": float(v1)})

    if not rows:
        return pd.Series(dtype=float)

    df = pd.DataFrame(rows).sort_values("valid_utc").drop_duplicates("valid_utc")
    df = df.set_index("valid_utc")

    # Resample to 15-min grid — use LAST value in each bin (latest PC reading)
    pc_15min = df["pc"].resample("15min").last().ffill()

    # Diff on the regular grid
    diff = pc_15min.diff()
    diff.iloc[0] = 0.0  # first diff is NaN → set to 0
    diff[diff < 0] = 0.0  # counter reset → 0
    diff = diff.fillna(0.0).clip(lower=0.0)

    return diff



def process_storm(storm: dict) -> None:
    log.info("Storm: %s  (%s)", storm["id"], storm["note"])
    log.info("  Window: %s → %s", storm["start"], storm["end"])

    site_series: dict[str, pd.Series] = {}

    for site_id, site_name in SITES.items():
        log.info("  Fetching site %s (%s) ...", site_id, site_name)
        records = fetch_site(site_id, storm["start"], storm["end"])
        log.info("    %d raw records", len(records))

        series = pc_to_increments(records)
        if series.empty:
            log.warning("    No usable data for site %s", site_id)
            continue

        total = series.sum()
        peak_rate = (series / (15 / 60)).max()   # in/hr
        log.info(
            "    total=%.2f in  peak_rate=%.2f in/hr  n=%d intervals",
            total, peak_rate, len(series)
        )
        site_series[site_id] = series
        time.sleep(0.5)  # be polite to the API

    if not site_series:
        log.error("No data retrieved for %s", storm["id"])
        return

    # Align all sites on common index, average
    combined = pd.DataFrame(site_series)
    n_sites = combined.notna().sum(axis=1).clip(lower=1)
    avg_15min = combined.mean(axis=1)   # NaN-safe mean across sites

    # Build output
    out = pd.DataFrame({
        "valid_utc":       avg_15min.index,
        "precip_15min_in": avg_15min.values,
        "intensity_in_hr": (avg_15min / (15 / 60)).values,
        "n_sites_avg":     n_sites.values,
    })
    out["intensity_in_hr"] = out["intensity_in_hr"].clip(lower=0.0)

    # Summary
    total_avg = out["precip_15min_in"].sum()
    peak_avg  = out["intensity_in_hr"].max()
    peak_t    = out.loc[out["intensity_in_hr"].idxmax(), "valid_utc"]
    log.info("")
    log.info("  === %s summary ===", storm["id"])
    log.info("  Total (watershed avg): %.2f in  (%.0f mm)", total_avg, total_avg * 25.4)
    log.info("  Peak 15-min intensity: %.2f in/hr  at %s", peak_avg, peak_t)
    log.info("  Active intervals (>0): %d", int((out["precip_15min_in"] > 0).sum()))

    # Save
    out_path = OUT_DIR / f"{storm['id']}_15min_lcra.csv"
    out.to_csv(out_path, index=False)
    log.info("  Saved → %s", out_path)

    # Also build storm_rainfall.parquet-compatible format
    qpe_df = pd.DataFrame({
        "station":          "LCRA_15min",
        "storm_id":         storm["id"],
        "valid":            out["valid_utc"],
        "precip_in_per_hr": out["intensity_in_hr"],
    })
    qpe_path = OUT_DIR / f"{storm['id']}_qpe_lcra.parquet"
    qpe_df.to_parquet(qpe_path, index=False)
    log.info("  Saved QPE parquet → %s", qpe_path)


def main():
    for storm in STORMS:
        log.info("\n" + "=" * 60)
        process_storm(storm)

    log.info("\nDone.")
    log.info("Next: merge into storm_rainfall.parquet and build INP:")
    log.info("  uv run python data/merge_lcra_qpe.py")
    log.info("  .venv/bin/python model/build_swmm_inp.py --storm july_2025 --station LCRA_15min")


if __name__ == "__main__":
    main()
