"""
Step 4: Plot calibration and validation hydrographs
====================================================
Loads best_params.json, re-runs SWMM for each storm, and produces
observed vs simulated hydrograph plots.

Usage:
    uv run python model/plot_calibration.py                  # all storms
    uv run python model/plot_calibration.py --storm halloween_2013

Note: uses .venv/bin/python if pyswmm is needed for re-run.
      If model/*.csv caches exist, uses those directly.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# Add model/ to path to import calibrate helpers
sys.path.insert(0, str(Path(__file__).parent))

PARAMS_PATH = Path("model/best_params.json")
FIG_DIR     = Path("model/figures")
USGS_PATH   = Path("data/raw/usgs/usgs_08158600_00060.parquet")

STORM_LABELS = {
    "halloween_2013":    "Halloween 2013 (Oct 31 – Nov 1)",
    "memorial_day_2015": "Memorial Day 2015 (May 25–26)",
    "halloween_2015":    "Halloween 2015 (Oct 30–31)",
}

STORM_ROLES = {
    "halloween_2013":    "Calibration",
    "memorial_day_2015": "Validation",
    "halloween_2015":    "Holdout",
}


def load_usgs(storm_id: str) -> pd.Series:
    """Load USGS 5-min streamflow for storm window."""
    from calibrate import STORM_WINDOWS
    df = pd.read_parquet(USGS_PATH)
    start, end = STORM_WINDOWS[storm_id]
    mask = (df.index >= start) & (df.index <= end)
    return df.loc[mask, "q_cfs"].dropna()


def run_storm(best_params: dict, storm_id: str) -> tuple[list, list]:
    """Run SWMM and return (times, flows). Uses .venv/bin/python environment."""
    from calibrate import run_swmm
    return run_swmm(best_params, storm_id)


def compute_nse(q_obs, times_sim, q_sim_raw) -> float:
    from calibrate import compute_nse as _nse
    return _nse(q_obs, times_sim, q_sim_raw)


def plot_storm(storm_id: str, q_obs: pd.Series,
               times_sim: list, q_sim_raw: list,
               nse: float, best_params: dict,
               ax: plt.Axes):
    """Plot a single storm hydrograph on an Axes."""
    role  = STORM_ROLES.get(storm_id, "")
    label = STORM_LABELS.get(storm_id, storm_id)

    # Build sim series
    try:
        sim_idx = pd.to_datetime(times_sim).tz_localize("UTC")
    except Exception:
        sim_idx = pd.to_datetime(times_sim, utc=True)
    q_sim = pd.Series(q_sim_raw, index=sim_idx).clip(lower=0)
    q_sim_5min = q_sim.resample("5min").mean().ffill(limit=1)

    # Convert USGS index to compatible tz
    q_obs_plot = q_obs.copy()
    if q_obs_plot.index.tz is None:
        q_obs_plot.index = q_obs_plot.index.tz_localize("UTC")

    # Plot
    ax.fill_between(q_obs_plot.index, q_obs_plot.values,
                    alpha=0.25, color="#1a73e8", label="USGS Observed")
    ax.plot(q_obs_plot.index, q_obs_plot.values,
            color="#1a73e8", lw=1.5, label=f"USGS 08158600 (obs)")
    ax.plot(q_sim_5min.index, q_sim_5min.values,
            color="#ea4335", lw=2.0, ls="--", label=f"SWMM Simulated")

    # Annotations
    peak_obs = q_obs.max()
    peak_sim = max(q_sim_raw) if q_sim_raw else 0

    ax.axhline(peak_obs, color="#1a73e8", lw=0.5, ls=":", alpha=0.6)
    ax.axhline(peak_sim, color="#ea4335", lw=0.5, ls=":", alpha=0.6)

    nse_color = "#34a853" if nse >= 0.6 else ("#fbbc04" if nse >= 0.4 else "#ea4335")
    ax.set_title(
        f"{label}  [{role}]\n"
        f"NSE = {nse:.3f}   "
        f"Peak: sim={peak_sim:.0f} / obs={peak_obs:.0f} CFS  "
        f"(ratio={peak_sim/peak_obs:.2f})",
        fontsize=10, color=nse_color if nse >= 0 else "gray"
    )
    ax.set_ylabel("Flow (CFS)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %HZ"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)


def main():
    parser = argparse.ArgumentParser(description="Plot SWMM calibration hydrographs")
    parser.add_argument("--storm", choices=list(STORM_LABELS.keys()), default=None,
                        help="Plot only this storm")
    args = parser.parse_args()

    if not PARAMS_PATH.exists():
        print(f"ERROR: {PARAMS_PATH} not found. Run calibration first.")
        sys.exit(1)

    with open(PARAMS_PATH) as f:
        best_params = json.load(f)

    print(f"Loaded best params: NSE_calib={best_params.get('nse_calibration', '?'):.4f}")
    print(f"  n_ch={best_params.get('n_ch', '?'):.4f}  "
          f"pct_imp={best_params.get('pct_imp', '?'):.1f}%  "
          f"width={best_params.get('width', '?'):.0f}ft")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    storms = [args.storm] if args.storm else list(STORM_LABELS.keys())
    # Skip halloween_2015 if no validation was run yet
    if not args.storm and best_params.get("nse_validation") is None:
        storms = ["halloween_2013", "memorial_day_2015"]

    fig, axes = plt.subplots(len(storms), 1, figsize=(12, 4.5 * len(storms)),
                              tight_layout=True)
    if len(storms) == 1:
        axes = [axes]

    fig.suptitle(
        f"Shoal Creek (08158600) — SWMM Calibration\n"
        f"n_ch={best_params.get('n_ch',0):.3f}  imp={best_params.get('pct_imp',0):.0f}%  "
        f"W={best_params.get('width',0):.0f}ft  Ksat={best_params.get('ksat',0):.3f}  "
        f"IMD={best_params.get('imd',0):.3f}  ZeroImp={best_params.get('pct_zero',0):.0f}%",
        fontsize=11, y=1.01
    )

    for ax, storm_id in zip(axes, storms):
        print(f"\nRunning {storm_id}...")
        try:
            q_obs = load_usgs(storm_id)
            times_sim, q_sim_raw = run_storm(best_params, storm_id)
            nse = compute_nse(q_obs, times_sim, q_sim_raw)
            print(f"  NSE={nse:.4f}  peak_sim={max(q_sim_raw):.0f} CFS  peak_obs={q_obs.max():.0f} CFS")
            plot_storm(storm_id, q_obs, times_sim, q_sim_raw, nse, best_params, ax)
        except Exception as e:
            print(f"  ERROR: {e}")
            ax.set_title(f"{storm_id} — ERROR: {e}")

    out_path = FIG_DIR / "calibration_hydrographs.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close(fig)

    # Print summary table
    print("\n" + "=" * 58)
    print("Calibration Summary")
    print("=" * 58)
    print(f"  NSE calibration (halloween_2013):    {best_params.get('nse_calibration', '?')}")
    nse_v = best_params.get('nse_validation')
    if nse_v is not None:
        print(f"  NSE validation (memorial_day_2015):  {nse_v:.4f}")
    print(f"  Output: {out_path}")
    print("=" * 58)


if __name__ == "__main__":
    main()
