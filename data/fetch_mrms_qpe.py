#!/usr/bin/env python3
"""
fetch_mrms_qpe.py — Extract watershed-averaged MRMS hourly QPE for Shoal Creek.

Reads the GRIB2.gz files downloaded to data/raw/mrms/YYYYMMDD/,
extracts the 1km pixel at the Shoal Creek watershed centroid
(and optionally averages over the watershed bounding box),
then appends to data/raw/qpe/storm_rainfall.parquet as station='MRMS_1H'.

Usage:
    uv run python data/fetch_mrms_qpe.py --storm july_2025
    uv run python data/fetch_mrms_qpe.py --start 2025-07-01 --end 2025-07-08

Requirements:
    cfgrib, eccodes (via pip/uv): uv add cfgrib eccodes
    OR: xarray with cfgrib backend
"""

import argparse
import gzip
import logging
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shoal Creek watershed spatial extent (lat/lon bounding box)
# ---------------------------------------------------------------------------
SHOAL_LAT_MIN = 30.265   # USGS gauge
SHOAL_LAT_MAX = 30.385   # Upper watershed
SHOAL_LON_MIN = -97.780  # Western extent
SHOAL_LON_MAX = -97.720  # Eastern extent
SHOAL_LAT_CTR = 30.290   # Centroid (approximate)
SHOAL_LON_CTR = -97.748

# Storm windows matching calibrate.py STORM_WINDOWS
STORM_WINDOWS = {
    "halloween_2013":    ("2013-10-29", "2013-11-02"),
    "halloween_2015":    ("2015-10-29", "2015-11-02"),
    "july_2025":         ("2025-07-01", "2025-07-08"),
}

MRMS_DIR = Path("data/raw/mrms")
QPE_OUT   = Path("data/raw/qpe/storm_rainfall.parquet")
STATION   = "MRMS_1H"


def iter_hourly_files(start: datetime, end: datetime):
    """Yield (timestamp, Path) for each MRMS 1H file in [start, end)."""
    cur = start.replace(minute=0, second=0, microsecond=0)
    while cur < end:
        date_str = cur.strftime("%Y%m%d")
        time_str = cur.strftime("%Y%m%d-%H0000")
        fname = f"MRMS_MultiSensor_QPE_01H_Pass2_00.00_{time_str}.grib2.gz"
        fpath = MRMS_DIR / date_str / fname
        yield cur, fpath
        cur += timedelta(hours=1)


def extract_watershed_avg(gz_path: Path) -> float | None:
    """
    Read one GRIB2.gz file, extract the mean QPE (inches/hr) over the
    Shoal Creek watershed bounding box. Returns None if file is missing.
    """
    if not gz_path.exists():
        log.warning("Missing: %s", gz_path)
        return None

    try:
        import cfgrib
    except ImportError:
        log.error("cfgrib not installed. Run: uv add cfgrib eccodes")
        raise

    # Decompress to temp file (cfgrib needs seekable file)
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        with gzip.open(gz_path, "rb") as gz:
            shutil.copyfileobj(gz, tmp)

    try:
        ds = xr.open_dataset(str(tmp_path), engine="cfgrib",
                             backend_kwargs={"indexpath": None},
                             filter_by_keys={"typeOfLevel": "heightAboveSea"})
        # Variable name in MRMS QPE is 'unknown' (non-standard paramId)
        var = None
        for v in ds.data_vars:
            if ds[v].values.ndim == 2:
                var = v
                break
        if var is None:
            log.warning("No 2D variable found in %s (vars: %s)", gz_path.name, list(ds.data_vars))
            return None

        log.debug("  Using variable '%s' from %s", var, gz_path.name)
        da = ds[var]

        # MRMS CONUS grid: lat/lon in standard -180 to 180
        lat_dim = "latitude" if "latitude" in da.coords else "lat"
        lon_dim = "longitude" if "longitude" in da.coords else "lon"
        lats = da[lat_dim].values
        lons = da[lon_dim].values

        # MRMS lons may be in [0, 360] — convert if needed
        if lons.min() > 0:
            lons = lons - 360.0

        # Mask to Shoal Creek watershed bounding box
        lat_mask = (lats >= SHOAL_LAT_MIN) & (lats <= SHOAL_LAT_MAX)
        lon_mask = (lons >= SHOAL_LON_MIN) & (lons <= SHOAL_LON_MAX)

        vals = da.values
        # vals shape is (lat, lon)
        if vals.shape[0] == len(lats):
            subset = vals[np.ix_(lat_mask, lon_mask)]
        else:
            subset = vals[np.ix_(lon_mask, lat_mask)]

        # MRMS QPE is in mm; convert to inches/hr (1H accumulation → rate = value/1)
        mean_mm = float(np.nanmean(subset[subset >= 0]))  # exclude fill values (-999)
        if np.isnan(mean_mm):
            return 0.0
        mean_in = mean_mm / 25.4

        log.debug("  %s → %.3f in (%.1f mm, %d pixels)",
                  gz_path.name, mean_in, mean_mm, np.sum(lat_mask) * np.sum(lon_mask))
        return mean_in


    finally:
        tmp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Extract MRMS QPE for Shoal Creek")
    parser.add_argument("--storm", choices=list(STORM_WINDOWS.keys()),
                        help="Named storm window")
    parser.add_argument("--start", help="Start date YYYY-MM-DD (UTC)")
    parser.add_argument("--end",   help="End date YYYY-MM-DD (UTC)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files only, don't extract")
    args = parser.parse_args()

    if args.storm:
        s, e = STORM_WINDOWS[args.storm]
        storm_id = args.storm
    elif args.start and args.end:
        s, e = args.start, args.end
        storm_id = f"{s}_{e}"
    else:
        parser.error("Provide --storm or --start/--end")
        return

    start = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    end   = datetime.fromisoformat(e).replace(tzinfo=timezone.utc)

    log.info("Extracting MRMS QPE: %s → %s  storm=%s", start.date(), end.date(), storm_id)

    records = []
    for ts, fpath in iter_hourly_files(start, end):
        if args.dry_run:
            status = "✓" if fpath.exists() else "✗ MISSING"
            print(f"  {ts.isoformat()}  {status}  {fpath.name}")
            continue

        val = extract_watershed_avg(fpath)
        if val is not None:
            records.append({
                "station":          STATION,
                "storm_id":         storm_id,
                "valid":            ts,
                "precip_in_per_hr": val,
            })

    if args.dry_run:
        missing = sum(1 for _, fp in iter_hourly_files(start, end) if not fp.exists())
        total   = sum(1 for _ in iter_hourly_files(start, end))
        print(f"\n{total - missing}/{total} files present")
        return

    if not records:
        log.warning("No data extracted!")
        return

    df_new = pd.DataFrame(records)
    total_in = df_new["precip_in_per_hr"].sum()
    peak_in  = df_new["precip_in_per_hr"].max()
    peak_ts  = df_new.loc[df_new["precip_in_per_hr"].idxmax(), "valid"]
    log.info("Extracted %d hourly values", len(records))
    log.info("Total: %.2f in  |  Peak: %.2f in/hr at %s", total_in, peak_in, peak_ts)

    # Append to existing parquet
    if QPE_OUT.exists():
        df_existing = pd.read_parquet(QPE_OUT)
        # Remove any existing MRMS rows for this storm
        df_existing = df_existing[
            ~((df_existing["station"] == STATION) &
              (df_existing["storm_id"] == storm_id))
        ]
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new

    QPE_OUT.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(QPE_OUT, index=False)
    log.info("Saved → %s  (%d total rows)", QPE_OUT, len(df_out))

    print(f"\n  MRMS QPE ({storm_id}):")
    print(f"    Total:  {total_in:.2f} in")
    print(f"    Peak:   {peak_in:.2f} in/hr at {peak_ts.strftime('%m/%d %H:%MZ')}")
    print(f"    Pixels: Shoal Creek bbox ~{SHOAL_LAT_MIN}–{SHOAL_LAT_MAX}N, "
          f"{SHOAL_LON_MIN}–{SHOAL_LON_MAX}E")
    print(f"\n  To rebuild SWMM INP with MRMS rainfall:")
    print(f"    uv run python model/build_swmm_inp.py --storm {storm_id} --station MRMS_1H")


if __name__ == "__main__":
    main()
