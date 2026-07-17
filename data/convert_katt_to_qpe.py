"""
data/convert_katt_to_qpe.py
============================
Converts the NOAA NCEI DSI-6406 1-minute KATT output (5-min aggregated CSV)
into the storm_rainfall.parquet format consumed by model/build_swmm_inp.py.

Updates storm_rainfall.parquet replacing the old hourly ATT data for
memorial_day_2015 with new 5-minute KATT data.

Also prints a comparison summary showing the intensity improvement.

Usage:
    uv run python data/convert_katt_to_qpe.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

QPE_PATH     = Path("data/raw/qpe/storm_rainfall.parquet")
KATT_5MIN_DIR = Path("data/raw/qpe")

# Only update memorial_day_2015 — it's the one with reliable KATT coverage
STORM_MAP = {
    "memorial_day_2015": "memorial_day_2015_5min_katt.csv",
    # halloween_2015 has partial KATT data (0.96 in, 4.2 in/hr peak)
    # halloween_2013 has zero KATT — skip; no sub-hourly data available
}


def load_katt_5min(storm_id: str, csv_name: str) -> pd.DataFrame:
    """Load 5-min KATT CSV and convert to storm_rainfall.parquet schema."""
    path = KATT_5MIN_DIR / csv_name
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}. Run data/fetch_asos_1min.py first.")

    df = pd.read_csv(path, parse_dates=["valid_utc"])
    df["valid_utc"] = pd.to_datetime(df["valid_utc"], utc=True)

    # Convert to storm_rainfall schema
    out = pd.DataFrame({
        "station":          "KATT_5min",   # distinct name so we can filter later
        "storm_id":         storm_id,
        "valid":            df["valid_utc"],
        "precip_in_per_hr": df["intensity_in_hr"].clip(lower=0.0),
    })
    return out


def main():
    # ------------------------------------------------------------------
    # Load existing parquet
    # ------------------------------------------------------------------
    if QPE_PATH.exists():
        existing = pd.read_parquet(QPE_PATH)
        log.info("Loaded existing parquet: %d rows", len(existing))
    else:
        existing = pd.DataFrame(columns=["station", "storm_id", "valid", "precip_in_per_hr"])
        log.warning("No existing parquet found — creating fresh.")

    existing["valid"] = pd.to_datetime(existing["valid"], utc=True)

    # ------------------------------------------------------------------
    # Print comparison for each storm being updated
    # ------------------------------------------------------------------
    log.info("\n=== Intensity Comparison: Old Hourly ATT vs New 5-min KATT ===")
    for storm_id in STORM_MAP:
        old = existing[
            (existing["storm_id"] == storm_id) &
            (existing["station"] == "ATT")
        ]
        if not old.empty:
            log.info(
                "  %s  old ATT:  peak=%.2f in/hr  total_hrs=%.1f",
                storm_id,
                old["precip_in_per_hr"].max(),
                old["precip_in_per_hr"].sum(),  # each row = 1 hr
            )

    # ------------------------------------------------------------------
    # Build new 5-min records and merge
    # ------------------------------------------------------------------
    new_records = []
    for storm_id, csv_name in STORM_MAP.items():
        try:
            df = load_katt_5min(storm_id, csv_name)
            new_records.append(df)
            log.info(
                "  %s  new KATT: peak=%.2f in/hr  total_in=%.2f  n_rows=%d",
                storm_id,
                df["precip_in_per_hr"].max(),
                (df["precip_in_per_hr"] * (5 / 60)).sum(),   # in/hr × 5/60 hr = inches
                len(df),
            )
        except FileNotFoundError as e:
            log.error(str(e))

    if not new_records:
        log.error("No KATT data converted. Exiting.")
        return

    new_df = pd.concat(new_records, ignore_index=True)

    # Keep all existing rows EXCEPT for storm-station combos being replaced
    keep_mask = ~(
        (existing["storm_id"].isin(STORM_MAP.keys())) &
        (existing["station"] == "KATT_5min")
    )
    merged = pd.concat([existing[keep_mask], new_df], ignore_index=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    QPE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Back up old parquet
    backup = QPE_PATH.with_suffix(".parquet.bak")
    if QPE_PATH.exists():
        import shutil
        shutil.copy2(QPE_PATH, backup)
        log.info("Backed up original → %s", backup)

    merged.to_parquet(QPE_PATH, index=False)
    log.info("Saved updated parquet: %d rows → %s", len(merged), QPE_PATH)

    # ------------------------------------------------------------------
    # Print summary of what's now in the parquet
    # ------------------------------------------------------------------
    log.info("\n=== Storm rainfall.parquet contents ===")
    for (storm, station), grp in merged.groupby(["storm_id", "station"]):
        log.info(
            "  storm=%-25s  station=%-12s  rows=%4d  peak_in_hr=%.2f",
            storm, station, len(grp), grp["precip_in_per_hr"].max()
        )

    log.info(
        "\nNext step: rebuild INP with 5-min interval:\n"
        "  .venv/bin/python model/build_swmm_inp.py --storm memorial_day_2015 "
        "--station KATT_5min --interval 5\n"
        "Then re-run calibration:\n"
        "  nohup .venv/bin/python model/calibrate.py > logs/calibrate_katt.log 2>&1 &"
    )


if __name__ == "__main__":
    main()
