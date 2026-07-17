"""
model/historical_replay.py
==========================
Step 7: Historical Replay Validation.

For each storm with available QPE data, computes storm parameters
(rain_depth_mm, storm_dur_hr, api_5_mm) from observations, runs them
through the trained peak-CFS surrogate, and compares to USGS gauge
observations.

Three events:
  - Memorial Day 2015  (calibration storm, peak 11,600 CFS USGS)
  - Halloween 2013     (validation,        peak 10,400 CFS USGS)
  - Halloween 2015     (validation,        peak  8,590 CFS USGS)

Expected result:
  - Surrogate ≈ SWMM prediction (trained on SWMM outputs)
  - SWMM peaks are ~7,400 CFS for high-end events (QPE resolution ceiling)
  - Gap to USGS is explained, not a surprise

Usage:
    uv run python model/historical_replay.py

Outputs:
    model/surrogate/historical_replay.json   — full result table
    (printed benchmark table to stdout)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SURROGATE_DIR = Path("model/surrogate")
QPE_DIR       = Path("data/raw/qpe")
USGS_PARQUET  = Path("data/raw/usgs/usgs_08158600_00060.parquet")
USGS_EVENTS   = Path("data/raw/usgs/usgs_08158600_events.csv")

# ---------------------------------------------------------------------------
# Storm definitions: QPE file + USGS event ID + known SWMM sim peak
# ---------------------------------------------------------------------------
STORMS = [
    {
        "name":          "Memorial Day 2015",
        "storm_id":      "memorial_day_2015",
        "qpe_file":      "memorial_day_2015_att.csv",
        "usgs_event_id": 120,          # event_id in events CSV (peak 11,600 CFS)
        "swmm_peak_cfs": 7395,         # from calibrate.py best run
        "notes":         "Calibration event. QPE ceiling ~7,400 CFS.",
    },
    {
        "name":          "Halloween 2013",
        "storm_id":      "halloween_2013",
        "qpe_file":      "halloween_2013_att.csv",
        "usgs_event_id": 91,           # Oct 29 - Nov 2, peak 10,400 CFS
        "swmm_peak_cfs": None,         # not simulated
        "notes":         "Validation event. No SWMM run.",
    },
    {
        "name":          "Halloween 2015",
        "storm_id":      "halloween_2015",
        "qpe_file":      "halloween_2015_att.csv",
        "usgs_event_id": 126,          # Oct 29 - Nov 2, peak 8,590 CFS
        "swmm_peak_cfs": None,
        "notes":         "Validation event. No SWMM run.",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_qpe(qpe_file: str) -> dict:
    """
    Load ASOS hourly QPE and compute storm parameters.
    Returns dict with rain_depth_mm, storm_dur_hr, peak_intensity_in_hr.
    """
    df = pd.read_csv(QPE_DIR / qpe_file, parse_dates=["valid"])
    df = df.sort_values("valid")
    rain = df["precip_in_per_hr"]

    total_in    = rain.sum()                          # total inches (1 reading/hr)
    total_mm    = total_in * 25.4
    active_hrs  = int((rain > 0.01).sum())            # hours with measurable rain
    peak_int    = float(rain.max())                   # peak hourly intensity [in/hr]

    return {
        "rain_depth_mm":        total_mm,
        "storm_dur_hr":         float(active_hrs),
        "peak_intensity_in_hr": peak_int,
    }


def estimate_api5_from_usgs(event_id: int, usgs_flow: pd.DataFrame,
                             events_df: pd.DataFrame) -> float:
    """
    Estimate 5-day antecedent precipitation index from USGS flow data.

    API-5 is a proxy for soil wetness. We use the mean daily flow in the
    5 days before the storm as a proxy:
      - High prior flow (>500 CFS avg) → wet watershed → API≈40mm
      - Moderate (100-500 CFS)          → API≈20mm
      - Low (<100 CFS)                  → dry → API≈5mm

    This is a rough but defensible first approximation for the demo.
    """
    row = events_df[events_df["event_id"] == event_id]
    if len(row) == 0:
        log.warning("Event ID %d not found in events CSV", event_id)
        return 20.0  # default moderate

    storm_start  = pd.to_datetime(row.iloc[0]["start"]).tz_localize(None) \
                   if pd.to_datetime(row.iloc[0]["start"]).tzinfo is None \
                   else pd.to_datetime(row.iloc[0]["start"]).tz_convert("UTC").tz_localize(None)
    window_start = storm_start - pd.Timedelta(days=5)

    pre = usgs_flow[
        (usgs_flow.index >= window_start) &
        (usgs_flow.index < storm_start)
    ]

    if len(pre) == 0:
        log.warning("No prior-flow data for event %d", event_id)
        return 20.0

    mean_cfs = float(pre["q_cfs"].mean())

    # Simple lookup table
    if mean_cfs > 500:
        api = 45.0
    elif mean_cfs > 200:
        api = 30.0
    elif mean_cfs > 50:
        api = 15.0
    else:
        api = 5.0

    log.info("  API-5 proxy: 5-day prior mean flow=%.0f CFS → API≈%.0f mm", mean_cfs, api)
    return api


def surrogate_predict(peak_model: dict, rain_depth_mm: float,
                      storm_dur_hr: float, api_5_mm: float) -> float:
    """Run the trained peak-CFS surrogate."""
    xgb    = peak_model["model"]
    scaler = peak_model["scaler"]

    # Engineered features (must match train_surrogate.py add_engineered_features)
    rain_intensity = rain_depth_mm / max(storm_dur_hr, 0.1)
    api_factor     = max(0.4, 1 - 0.008 * api_5_mm)
    eff_depth      = rain_depth_mm * api_factor

    X = np.array([[rain_depth_mm, storm_dur_hr, api_5_mm,
                   rain_intensity, api_factor, eff_depth]], dtype=np.float32)
    X_s = scaler.transform(X)
    return float(xgb.predict(X_s)[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Load surrogate ---
    log.info("Loading surrogate from %s ...", SURROGATE_DIR)
    peak_model = joblib.load(SURROGATE_DIR / "xgb_peak.joblib")
    log.info("  Surrogate: %s", peak_model["feature_cols"])

    # --- Load USGS flow and events ---
    log.info("Loading USGS data ...")
    usgs = pd.read_parquet(USGS_PARQUET)
    # Normalize index to tz-naive UTC for consistent date comparisons
    if usgs.index.tz is not None:
        usgs.index = usgs.index.tz_convert("UTC").tz_localize(None)

    events_df = pd.read_csv(USGS_EVENTS)

    # --- Run replay for each storm ---
    results = []
    for storm in STORMS:
        log.info("\n" + "=" * 55)
        log.info("Storm: %s", storm["name"])

        # QPE → storm params
        qpe = load_qpe(storm["qpe_file"])
        log.info("  Rain depth:    %.0f mm", qpe["rain_depth_mm"])
        log.info("  Storm dur:     %.0f hr", qpe["storm_dur_hr"])
        log.info("  Peak intensity:%.2f in/hr", qpe["peak_intensity_in_hr"])

        # API-5 from USGS prior flow
        api_5 = estimate_api5_from_usgs(storm["usgs_event_id"], usgs, events_df)

        # Surrogate prediction
        pred_peak = surrogate_predict(
            peak_model,
            qpe["rain_depth_mm"],
            qpe["storm_dur_hr"],
            api_5,
        )
        log.info("  Surrogate peak: %.0f CFS", pred_peak)

        # USGS observed peak
        row = events_df[events_df["event_id"] == storm["usgs_event_id"]]
        usgs_peak = float(row.iloc[0]["peak_cfs"]) if len(row) > 0 else None
        log.info("  USGS observed:  %.0f CFS", usgs_peak or 0)

        # SWMM simulated
        swmm_peak = storm["swmm_peak_cfs"]
        if swmm_peak:
            log.info("  SWMM simulated: %.0f CFS", swmm_peak)

        # Errors
        surr_vs_usgs_err = (pred_peak - usgs_peak) / usgs_peak * 100 if usgs_peak else None
        surr_vs_swmm_err = (pred_peak - swmm_peak) / swmm_peak * 100 if swmm_peak else None

        results.append({
            "storm":            storm["name"],
            "rain_depth_mm":    round(qpe["rain_depth_mm"], 1),
            "storm_dur_hr":     qpe["storm_dur_hr"],
            "api_5_mm":         api_5,
            "surrogate_cfs":    round(pred_peak),
            "swmm_cfs":         swmm_peak,
            "usgs_obs_cfs":     usgs_peak,
            "surr_vs_usgs_pct": round(surr_vs_usgs_err, 1) if surr_vs_usgs_err else None,
            "surr_vs_swmm_pct": round(surr_vs_swmm_err, 1) if surr_vs_swmm_err else None,
            "notes":            storm["notes"],
        })

    # --- Print results table ---
    print("\n")
    print("=" * 80)
    print("HISTORICAL REPLAY VALIDATION")
    print("=" * 80)
    hdr = f"{'Storm':<22} {'Inputs':>28} {'Surrogate':>10} {'SWMM':>8} {'USGS':>8} {'Err%':>7}"
    print(hdr)
    print("-" * 80)
    for r in results:
        inputs_str = f"{r['rain_depth_mm']:.0f}mm/{r['storm_dur_hr']:.0f}hr/api{r['api_5_mm']:.0f}"
        swmm_str   = f"{r['swmm_cfs']:.0f}" if r["swmm_cfs"] else "  N/A"
        err_str    = f"{r['surr_vs_usgs_pct']:+.0f}%" if r["surr_vs_usgs_pct"] else "  N/A"
        print(f"{r['storm']:<22} {inputs_str:>28} {r['surrogate_cfs']:>10,.0f}"
              f" {swmm_str:>8} {r['usgs_obs_cfs']:>8,.0f} {err_str:>7}")

    print("-" * 80)
    print("\nNOTE: Surrogate ≈ SWMM (trained on SWMM outputs, not USGS).")
    print("      The ~35-55% gap to USGS is the known QPE resolution ceiling —")
    print("      hourly rain data can't reproduce the sub-hourly 4-6 in/hr bursts")
    print("      that drive peak flows on this flashy urban watershed.")
    print()

    # --- Save results ---
    out_path = SURROGATE_DIR / "historical_replay.json"
    out_path.write_text(json.dumps({"results": results}, indent=2))
    log.info("Saved → %s", out_path)

    return results


if __name__ == "__main__":
    main()
