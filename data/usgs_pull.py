"""
data/usgs_pull.py
-----------------
Step 1: Pull USGS Shoal Creek gauge record (site 08158600) and identify
usable calibration events.

Gauge: USGS 08158600 — Shoal Creek at West 12th St, Austin TX
Parameter: 00060 — Discharge, instantaneous [CFS]

This script:
  1. Downloads the complete record (2000–present) via dataretrieval.
  2. Saves the raw parquet locally (skip download if cache exists).
  3. Identifies flood events by threshold exceedance + recession logic.
  4. Reports event statistics: peak, volume, duration, data gaps.
  5. Prints a frank assessment of calibration readiness.

Decision log:
  - Start date 2000-01-01: NWS flood stage at 12th St is ~12 ft (700 CFS).
    Pre-2000 data quality is lower and radar QPE is sparse.
  - Threshold for "flood event": 200 CFS minimum peak (action stage ~6 ft).
    Memorial Day 2015 peaked ~5000 CFS; Halloween 2013 ~2000 CFS.
  - Event separation gap: 24 hours below threshold = new event.
  - Minimum event duration: 3 hours (filters noise spikes).

Usage:
    uv run python data/usgs_pull.py [--force-download]
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SITE_ID      = "08158600"               # USGS Shoal Creek at W 12th St
PARAM_CD     = "00060"                  # Instantaneous discharge [CFS]
START_DATE   = "2000-01-01"
END_DATE     = datetime.today().strftime("%Y-%m-%d")

# NWS flood stages for 08158600 (approx — confirmed from NWS Austin)
# Action: ~6 ft / ~200 CFS
# Minor flood: ~12 ft / ~700 CFS
# Moderate flood: ~18 ft / ~2200 CFS
# Major flood: ~22 ft / ~4000 CFS
ACTION_STAGE_CFS   = 200.0
MINOR_FLOOD_CFS    = 700.0
MODERATE_FLOOD_CFS = 2200.0
MAJOR_FLOOD_CFS    = 4000.0

# Event identification
PEAK_THRESH_CFS   = 200.0   # Minimum peak flow to count as an event
GAP_HOURS         = 24      # Hours below threshold = new event boundary
MIN_DURATION_HRS  = 3.0     # Minimum event duration to keep

# Key calibration events (target dates for focused validation)
PRIORITY_EVENTS = {
    "Halloween_2013":    ("2013-10-30", "2013-11-02"),
    "MemorialDay_2015":  ("2015-05-24", "2015-05-27"),
    "Halloween_2015":    ("2015-10-30", "2015-11-02"),
}

# Output
DATA_DIR   = pathlib.Path(__file__).parent / "raw" / "usgs"
CACHE_FILE = DATA_DIR / f"usgs_{SITE_ID}_00060.parquet"
EVENTS_CSV = DATA_DIR / f"usgs_{SITE_ID}_events.csv"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_gauge_record(
    site_id: str = SITE_ID,
    start: str = START_DATE,
    end: str = END_DATE,
    force: bool = False,
) -> pd.DataFrame:
    """
    Fetch instantaneous discharge (15-min) from NWIS.
    Returns DataFrame with DatetimeIndex (UTC), column 'q_cfs'.
    Caches result to parquet to avoid repeat downloads.
    """
    if CACHE_FILE.exists() and not force:
        log.info("Loading cached gauge record: %s", CACHE_FILE)
        df = pd.read_parquet(CACHE_FILE)
        log.info("  Loaded %d records  %s → %s",
                 len(df), df.index.min(), df.index.max())
        return df

    log.info("Downloading USGS site %s  param %s  %s → %s",
             site_id, PARAM_CD, start, end)
    try:
        import dataretrieval.nwis as nwis
    except ImportError:
        log.error("dataretrieval not installed — run: uv add dataretrieval")
        sys.exit(1)

    try:
        raw, meta = nwis.get_iv(
            sites=site_id,
            parameterCd=PARAM_CD,
            start=start,
            end=end,
        )
    except Exception as exc:
        log.error("NWIS download failed: %s", exc)
        sys.exit(1)

    if raw is None or raw.empty:
        log.error("NWIS returned empty DataFrame — check site ID and param code.")
        sys.exit(1)

    log.info("Raw NWIS columns: %s", list(raw.columns))
    log.info("Raw NWIS shape: %s", raw.shape)
    log.info("Index type: %s", type(raw.index))

    # -----------------------------------------------------------------------
    # Normalise: find discharge column regardless of NWIS naming convention
    # NWIS returns columns like '00060', '00060_00000', or multi-level.
    # -----------------------------------------------------------------------
    df = _extract_discharge(raw, site_id)

    # Ensure UTC timezone
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Drop negative values (NWIS sentinel for ice/equipment)
    n_neg = (df["q_cfs"] < 0).sum()
    if n_neg > 0:
        log.warning("Dropping %d negative flow values (NWIS sentinels)", n_neg)
        df = df[df["q_cfs"] >= 0]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_FILE)
    log.info("Saved cache → %s  (%d records)", CACHE_FILE, len(df))
    return df


def _extract_discharge(raw: pd.DataFrame, site_id: str) -> pd.DataFrame:
    """
    Extract a single 'q_cfs' column from whatever shape NWIS returns.
    Handles multi-level columns and site-prefixed column names.
    """
    # Flatten multi-level column index if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = ["_".join(str(c) for c in col).strip("_")
                       for col in raw.columns]
        log.info("Flattened MultiIndex columns: %s", list(raw.columns)[:6])

    # Try candidate column names in priority order
    candidates = [
        PARAM_CD,
        f"{PARAM_CD}_00000",
        f"{site_id}_{PARAM_CD}",
        f"{site_id}_{PARAM_CD}_00000",
    ]
    for cand in candidates:
        if cand in raw.columns:
            log.info("Using discharge column: '%s'", cand)
            return raw[[cand]].rename(columns={cand: "q_cfs"}).copy()

    # Fallback: find any column containing '00060'
    discharge_cols = [c for c in raw.columns if PARAM_CD in str(c)]
    if discharge_cols:
        chosen = discharge_cols[0]
        log.warning("Candidate columns not found; using first match: '%s'", chosen)
        return raw[[chosen]].rename(columns={chosen: "q_cfs"}).copy()

    raise ValueError(
        f"Cannot locate discharge column (param {PARAM_CD}) in NWIS output.\n"
        f"Available columns: {list(raw.columns)}"
    )


# ---------------------------------------------------------------------------
# Event identification
# ---------------------------------------------------------------------------
def identify_events(
    df: pd.DataFrame,
    peak_thresh: float = PEAK_THRESH_CFS,
    gap_hours: int = GAP_HOURS,
    min_duration_hrs: float = MIN_DURATION_HRS,
) -> pd.DataFrame:
    """
    Segment the streamflow record into discrete flood events.

    Algorithm:
      1. Mark timesteps where q_cfs >= peak_thresh (exceedance mask).
      2. Group consecutive exceedance blocks, padding gaps < gap_hours.
      3. For each block, compute peak, start, end, duration.
      4. Drop events shorter than min_duration_hrs.

    Returns DataFrame with columns:
      event_id, start, end, peak_cfs, duration_hrs, volume_cfs_hr,
      data_gap_pct, stage_class
    """
    # Work in local Austin time for interpretability
    q = df["q_cfs"].copy()

    # Infer median timestep (handles 15-min and 5-min records)
    dt_minutes = float(
        pd.Series(q.index).diff().dropna().dt.total_seconds().median() / 60
    )
    log.info("Median timestep: %.0f min", dt_minutes)

    gap_steps = int(gap_hours * 60 / dt_minutes)

    above = (q >= peak_thresh).astype(int)
    # Dilate: if gap between two exceedance blocks < gap_steps, merge them
    # (rolling max over gap window)
    merged = above.copy()
    for shift in range(1, gap_steps + 1):
        merged = merged | above.shift(shift, fill_value=0).astype(int)
        merged = merged | above.shift(-shift, fill_value=0).astype(int)
    merged = merged.astype(bool)

    # Label contiguous blocks
    blocks = (merged != merged.shift()).cumsum()
    blocks[~merged] = 0

    events = []
    for blk_id in blocks[blocks > 0].unique():
        mask = blocks == blk_id
        q_ev = q[mask]

        start_ts = q_ev.index.min()
        end_ts   = q_ev.index.max()
        peak_cfs = float(q_ev.max())
        dur_hrs  = (end_ts - start_ts).total_seconds() / 3600

        if dur_hrs < min_duration_hrs:
            continue
        if peak_cfs < peak_thresh:
            continue

        # Data gap % within event window
        n_expected = int(dur_hrs * 60 / dt_minutes) + 1
        gap_pct = max(0.0, 1.0 - len(q_ev) / n_expected) * 100

        # Volume (trapezoidal, CFS·hr)
        q_vals = q_ev.values.astype(float)
        t_hr   = np.arange(len(q_vals)) * dt_minutes / 60
        volume = float(np.trapezoid(q_vals, t_hr))

        # NWS stage classification
        if peak_cfs >= MAJOR_FLOOD_CFS:
            stage_class = "major"
        elif peak_cfs >= MODERATE_FLOOD_CFS:
            stage_class = "moderate"
        elif peak_cfs >= MINOR_FLOOD_CFS:
            stage_class = "minor"
        else:
            stage_class = "action"

        events.append({
            "event_id":     len(events) + 1,
            "start":        start_ts.tz_convert("US/Central"),
            "end":          end_ts.tz_convert("US/Central"),
            "peak_cfs":     round(peak_cfs, 1),
            "duration_hrs": round(dur_hrs, 1),
            "volume_cfs_hr": round(volume, 0),
            "data_gap_pct": round(gap_pct, 1),
            "stage_class":  stage_class,
        })

    ev_df = pd.DataFrame(events)
    if ev_df.empty:
        log.warning("No events found above %.0f CFS threshold!", peak_thresh)
        return ev_df

    ev_df = ev_df.sort_values("start").reset_index(drop=True)
    ev_df["event_id"] = np.arange(1, len(ev_df) + 1)
    return ev_df


# ---------------------------------------------------------------------------
# Data quality report
# ---------------------------------------------------------------------------
def gauge_summary(df: pd.DataFrame) -> dict:
    """Return basic data quality statistics for the full record."""
    q = df["q_cfs"]
    n_total   = len(q)

    # Expected record count at 15-min intervals
    if n_total > 1:
        dt_med = float(
            pd.Series(q.index).diff().dropna().dt.total_seconds().median() / 60
        )
    else:
        dt_med = 15.0

    span_days  = (q.index.max() - q.index.min()).total_seconds() / 86400
    n_expected = int(span_days * 24 * 60 / dt_med)
    completeness_pct = min(100.0, n_total / max(n_expected, 1) * 100)

    return {
        "site":            SITE_ID,
        "param":           PARAM_CD,
        "start":           str(q.index.min()),
        "end":             str(q.index.max()),
        "span_years":      round(span_days / 365.25, 1),
        "n_records":       n_total,
        "dt_minutes":      round(dt_med, 0),
        "completeness_pct": round(completeness_pct, 1),
        "q_min_cfs":       round(float(q.min()), 1),
        "q_max_cfs":       round(float(q.max()), 1),
        "q_median_cfs":    round(float(q.median()), 1),
        "q_p99_cfs":       round(float(q.quantile(0.99)), 1),
    }


# ---------------------------------------------------------------------------
# Priority event check
# ---------------------------------------------------------------------------
def check_priority_events(
    df: pd.DataFrame, ev_df: pd.DataFrame
) -> dict[str, dict]:
    """
    For each named priority event, report observed peak, data completeness,
    and whether the event was detected by the algorithm.
    """
    results = {}
    q = df["q_cfs"]

    for name, (t0_str, t1_str) in PRIORITY_EVENTS.items():
        try:
            t0 = pd.Timestamp(t0_str, tz="US/Central")
            t1 = pd.Timestamp(t1_str, tz="US/Central")
            window = q[t0:t1]
        except Exception:
            # Try UTC if Central fails
            t0 = pd.Timestamp(t0_str, tz="UTC")
            t1 = pd.Timestamp(t1_str, tz="UTC")
            window = q[t0:t1]

        if window.empty:
            results[name] = {"status": "NO_DATA", "peak_cfs": None}
            continue

        peak  = float(window.max())
        n_obs = len(window)
        dt_med = float(
            pd.Series(window.index).diff().dropna().dt.total_seconds().median() / 60
        ) if n_obs > 1 else 15.0
        span_hrs  = (window.index.max() - window.index.min()).total_seconds() / 3600
        n_exp     = int(span_hrs * 60 / dt_med) + 1
        complete  = min(100.0, n_obs / max(n_exp, 1) * 100)

        results[name] = {
            "status":         "OK" if peak >= PEAK_THRESH_CFS else "BELOW_THRESH",
            "peak_cfs":       round(peak, 1),
            "completeness_pct": round(complete, 1),
            "n_obs":          n_obs,
            "in_event_list":  False,
        }

    # Check which were caught by the event algorithm
    if not ev_df.empty:
        for name, (t0_str, t1_str) in PRIORITY_EVENTS.items():
            t0 = pd.Timestamp(t0_str).tz_localize("US/Central")
            t1 = pd.Timestamp(t1_str).tz_localize("US/Central")
            overlaps = ev_df[
                (ev_df["start"] <= t1) & (ev_df["end"] >= t0)
            ]
            if name in results:
                results[name]["in_event_list"] = len(overlaps) > 0

    return results


# ---------------------------------------------------------------------------
# Print frank assessment
# ---------------------------------------------------------------------------
def print_assessment(
    summary: dict, ev_df: pd.DataFrame, priority: dict[str, dict]
) -> None:
    """Print a direct, no-hedging calibration readiness report."""

    sep = "=" * 70
    print(f"\n{sep}")
    print("USGS 08158600 — Shoal Creek at W 12th St, Austin TX")
    print("CALIBRATION READINESS ASSESSMENT")
    print(sep)

    print("\n── RECORD OVERVIEW ──────────────────────────────────────────────")
    print(f"  Period:        {summary['start'][:10]} → {summary['end'][:10]}")
    print(f"  Span:          {summary['span_years']} years")
    print(f"  Records:       {summary['n_records']:,}")
    print(f"  Timestep:      {summary['dt_minutes']:.0f} min")
    print(f"  Completeness:  {summary['completeness_pct']:.1f}%")
    print(f"  Peak on record:{summary['q_max_cfs']:,.0f} CFS")
    print(f"  Median flow:   {summary['q_median_cfs']:.1f} CFS")
    print(f"  P99 flow:      {summary['q_p99_cfs']:.1f} CFS")

    print("\n── FLOOD EVENTS (peak ≥ 200 CFS) ───────────────────────────────")
    if ev_df.empty:
        print("  !! NO EVENTS FOUND — check threshold or data quality !!")
    else:
        n_major    = (ev_df["stage_class"] == "major").sum()
        n_moderate = (ev_df["stage_class"] == "moderate").sum()
        n_minor    = (ev_df["stage_class"] == "minor").sum()
        n_action   = (ev_df["stage_class"] == "action").sum()

        print(f"  Total events:  {len(ev_df)}")
        print(f"  Major    (≥{MAJOR_FLOOD_CFS/1000:.0f}k CFS): {n_major}")
        print(f"  Moderate (≥{MODERATE_FLOOD_CFS/1000:.1f}k CFS): {n_moderate}")
        print(f"  Minor    (≥{MINOR_FLOOD_CFS:.0f} CFS):  {n_minor}")
        print(f"  Action   (≥{ACTION_STAGE_CFS:.0f} CFS):  {n_action}")

        print(f"\n  Top 10 events by peak flow:")
        top = ev_df.nlargest(10, "peak_cfs")[
            ["event_id", "start", "peak_cfs", "duration_hrs",
             "data_gap_pct", "stage_class"]
        ]
        for _, row in top.iterrows():
            gap_warn = " ← DATA GAP" if row["data_gap_pct"] > 5 else ""
            print(
                f"    #{int(row['event_id']):>3}  "
                f"{str(row['start'])[:16]}  "
                f"{row['peak_cfs']:>7,.0f} CFS  "
                f"{row['duration_hrs']:>5.1f} h  "
                f"gap={row['data_gap_pct']:.0f}%"
                f"{gap_warn}"
            )

    print("\n── PRIORITY CALIBRATION EVENTS ──────────────────────────────────")
    for name, info in priority.items():
        status = info["status"]
        peak   = info.get("peak_cfs")
        comp   = info.get("completeness_pct", "?")
        found  = info.get("in_event_list", False)

        if status == "NO_DATA":
            verdict = "✗  NO DATA IN WINDOW — cannot calibrate on this event"
        elif status == "BELOW_THRESH":
            verdict = f"~  Low peak ({peak:.0f} CFS) — possible data gap or wrong dates"
        elif not found:
            verdict = f"?  Peak={peak:,.0f} CFS complete={comp:.0f}% — not in event list (check gap logic)"
        else:
            verdict = f"✓  Peak={peak:,.0f} CFS  complete={comp:.0f}%  USABLE"

        print(f"  {name:<22}  {verdict}")

    print("\n── CALIBRATION VERDICT ──────────────────────────────────────────")
    usable = [k for k, v in priority.items()
              if v.get("status") == "OK" and v.get("in_event_list")]
    if len(usable) >= 2:
        print(f"  GOOD: {len(usable)}/3 priority events usable for calibration.")
        print("  Proceed to Step 2 (COA GIS data check).")
    elif len(usable) == 1:
        print(f"  MARGINAL: Only {len(usable)}/3 priority events look clean.")
        print("  Investigate data gaps before committing to calibration plan.")
        print("  May need to use additional events from the full event list.")
    else:
        print("  PROBLEM: Priority events not confirmed in gauge record.")
        print("  Check date ranges and NWIS data quality flags before proceeding.")

    if not ev_df.empty:
        good_events = ev_df[
            (ev_df["peak_cfs"] >= MINOR_FLOOD_CFS) &
            (ev_df["data_gap_pct"] < 5)
        ]
        print(f"\n  Backup pool: {len(good_events)} events ≥{MINOR_FLOOD_CFS:.0f} CFS with <5% data gaps")
        if len(good_events) >= 3:
            print("  → Enough backup events for train/val/holdout split.")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(force: bool = False) -> None:
    parser = argparse.ArgumentParser(
        description="Pull USGS Shoal Creek gauge record and identify calibration events."
    )
    parser.add_argument(
        "--force-download", action="store_true",
        help="Re-download even if cache exists."
    )
    args = parser.parse_args()
    force = args.force_download or force

    # 1. Download / load
    df = download_gauge_record(force=force)

    # 2. Summary statistics
    summary = gauge_summary(df)

    # 3. Event identification
    log.info("Identifying flood events (threshold=%.0f CFS, gap=%dh)…",
             PEAK_THRESH_CFS, GAP_HOURS)
    ev_df = identify_events(df)
    log.info("Found %d events", len(ev_df))

    # 4. Priority event check
    priority = check_priority_events(df, ev_df)

    # 5. Save events CSV
    if not ev_df.empty:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        ev_df.to_csv(EVENTS_CSV, index=False)
        log.info("Events saved → %s", EVENTS_CSV)

    # 6. Print frank assessment
    print_assessment(summary, ev_df, priority)

    # 7. Full event table
    if not ev_df.empty:
        print("Full event table (all events ≥ 200 CFS):")
        print(ev_df.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
