"""
data/fetch_asos_1min.py
========================
Pull NOAA NCEI DSI-6406 ASOS 1-minute precipitation data for Austin-Mueller
Airport (KATT) for the three calibration/validation storm events:

  - Memorial Day 2015  (2015-05-25)
  - Halloween 2013     (2013-10-31)
  - Halloween 2015     (2015-10-31)

DSI-6406 format (fixed-width):
  cols  0- 4 : WBAN (13958)
  cols  5- 8 : Station (KATT)
  cols  9    : space
  cols 10-12 : Prefix (ATT)
  cols 13-20 : Date YYYYMMDD
  cols 21-22 : Hour (UTC)
  cols 23-24 : Minute (UTC)
  cols 25-28 : Sequence
  cols 29-31 : spaces
  cols 32-33 : Precip flag (NP=none, R=rain, R+=heavy, S=snow ...)
  cols 34-35 : spaces
  cols 36-39 : Tipping bucket tip counter
  cols 40-47 : Precip amount (inches, 1-min total)

Data URL:
  https://www.ncei.noaa.gov/pub/data/asos-onemin/6406-{YYYY}/64060KATT{YYYYMM}.dat

Output: data/raw/qpe/{storm_id}_1min_katt.csv
        Columns: valid_utc, precip_in, intensity_in_hr, flag

Usage:
    uv run python data/fetch_asos_1min.py
"""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUT_DIR = Path("data/raw/qpe")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.ncei.noaa.gov/pub/data/asos-onemin/6406-{year}/64060KATT{year}{month:02d}.dat"

# Storm windows — grab the storm day ± 1 day for context
STORMS = [
    {
        "id":     "memorial_day_2015",
        "months": [("2015", 5)],
        "dates":  ["20150525", "20150526"],  # UTC window
        "note":   "Memorial Day 2015 (USGS obs 11,600 CFS)",
    },
    {
        "id":     "halloween_2013",
        "months": [("2013", 10), ("2013", 11)],
        "dates":  ["20131031", "20131101", "20131102"],
        "note":   "Halloween 2013 (USGS obs 10,400 CFS)",
    },
    {
        "id":     "halloween_2015",
        "months": [("2015", 10), ("2015", 11)],
        "dates":  ["20151031", "20151101", "20151102"],
        "note":   "Halloween 2015 (USGS obs 8,590 CFS)",
    },
]


def fetch_month(year: str, month: int) -> str:
    """Download one KATT 6406 monthly file, return raw text."""
    url = BASE_URL.format(year=year, month=month)
    log.info("Fetching %s ...", url)
    try:
        with urlopen(url, timeout=60) as r:
            return r.read().decode("latin-1")
    except HTTPError as e:
        log.warning("HTTP %s for %s", e.code, url)
        return ""


def parse_6406(raw: str, keep_dates: list[str]) -> pd.DataFrame:
    """
    Parse DSI-6406 fixed-width text into a DataFrame.

    Returns columns: valid_utc (datetime), precip_in (float), flag (str).
    """
    rows = []
    for line in raw.splitlines():
        if len(line) < 48:
            continue
        date  = line[13:21]
        if date not in keep_dates:
            continue
        hr    = line[21:23]
        mn    = line[23:25]
        flag  = line[32:34].strip()
        raw_p = line[40:48].strip()
        try:
            amount = float(raw_p)
        except ValueError:
            continue

        dt = pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:8]} {hr}:{mn}:00", tz="UTC")
        rows.append({"valid_utc": dt, "precip_in": amount, "flag": flag})

    if not rows:
        return pd.DataFrame(columns=["valid_utc", "precip_in", "flag"])
    return pd.DataFrame(rows).sort_values("valid_utc").reset_index(drop=True)


def make_5min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 1-minute data to 5-minute intervals.
    Returns columns: valid_utc, precip_5min_in, intensity_in_hr.
    """
    if df.empty:
        return df
    df = df.set_index("valid_utc")
    agg = df["precip_in"].resample("5min").sum().reset_index()
    agg.columns = ["valid_utc", "precip_5min_in"]
    agg["intensity_in_hr"] = agg["precip_5min_in"] * 12   # 5-min → hr rate
    return agg


def storm_summary(df1: pd.DataFrame, df5: pd.DataFrame, storm: dict) -> None:
    """Print a brief storm summary."""
    log.info("")
    log.info("=== %s ===", storm["note"])
    total = df1["precip_in"].sum()
    log.info("  Total accumulation : %.2f in  (%.0f mm)", total, total * 25.4)

    if not df1.empty:
        peak_row = df1.loc[df1["precip_in"].idxmax()]
        log.info(
            "  Peak 1-min intensity: %.3f in/min  (%.1f in/hr)  at %s",
            peak_row["precip_in"],
            peak_row["precip_in"] * 60,
            peak_row["valid_utc"],
        )

    if not df5.empty:
        peak5 = df5["intensity_in_hr"].max()
        log.info("  Peak 5-min intensity: %.2f in/hr", peak5)

    rain_min = int((df1["precip_in"] > 0).sum()) if not df1.empty else 0
    log.info("  Active rain minutes : %d min", rain_min)

    # Compare to our existing hourly estimate
    hourly_equiv = df1.set_index("valid_utc")["precip_in"].resample("1h").sum().max() if not df1.empty else 0
    log.info("  Peak hourly equiv   : %.2f in/hr  (was %.2f in/hr with old data)",
             hourly_equiv, {
                 "memorial_day_2015": 2.30,
                 "halloween_2013":    1.67,
                 "halloween_2015":    1.55,
             }.get(storm["id"], 0))


def main():
    for storm in STORMS:
        log.info("\n" + "=" * 60)
        log.info("Storm: %s", storm["id"])

        # Download all needed months
        all_raw = ""
        for year, month in storm["months"]:
            all_raw += fetch_month(year, month)

        if not all_raw.strip():
            log.error("No data retrieved for %s", storm["id"])
            continue

        # Parse 1-minute data
        df1 = parse_6406(all_raw, storm["dates"])
        log.info("  Parsed %d 1-min records", len(df1))

        if df1.empty:
            log.warning("  No records found for dates %s", storm["dates"])
            continue

        # Aggregate to 5-min
        df5 = make_5min(df1)

        # Summary
        storm_summary(df1, df5, storm)

        # Save 1-minute CSV
        out1 = OUT_DIR / f"{storm['id']}_1min_katt.csv"
        df1.to_csv(out1, index=False)
        log.info("  Saved 1-min → %s", out1)

        # Save 5-minute CSV (SWMM-ready format)
        out5 = OUT_DIR / f"{storm['id']}_5min_katt.csv"
        df5.to_csv(out5, index=False)
        log.info("  Saved 5-min → %s", out5)

    log.info("\nDone. Next step: re-run calibration with 5-min QPE inputs.")


if __name__ == "__main__":
    main()
