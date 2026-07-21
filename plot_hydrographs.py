import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd

def compute_nse(obs, sim):
    mean_obs = np.nanmean(obs)
    numerator = np.nansum((obs - sim)**2)
    denominator = np.nansum((obs - mean_obs)**2)
    return 1 - (numerator / denominator)

def plot_events():
    data_path = Path("data/processed/hist_hydrographs.json")
    with open(data_path, "r") as f:
        data = json.load(f)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    events = [
        ("july_2025", "Calibration: July 2025 (MRMS QPE)"),
        ("halloween_2015", "Validation: Halloween 2015")
    ]

    for ax, (event_id, title) in zip(axs, events):
        if event_id in data:
            swmm_t = [datetime.fromisoformat(t) for t in data[event_id]["times"]]
            swmm_q = np.array(data[event_id]["flows"])
            
            usgs_t = [datetime.fromisoformat(t) for t in data[event_id]["usgs_times"]]
            usgs_q = np.array(data[event_id]["usgs_flows"])
            
            # Create dataframes to align them for NSE calculation
            df_usgs = pd.DataFrame({"obs": usgs_q}, index=pd.DatetimeIndex(usgs_t).tz_localize(None))
            df_swmm = pd.DataFrame({"sim": swmm_q}, index=pd.DatetimeIndex(swmm_t).tz_localize(None))
            
            # resample to common 5 min grid, interpolate, inner join
            df_merged = pd.merge_asof(df_usgs.sort_index(), df_swmm.sort_index(), left_index=True, right_index=True, tolerance=pd.Timedelta("15min"))
            df_merged = df_merged.dropna()
            
            nse = compute_nse(df_merged["obs"].values, df_merged["sim"].values)

            # Center time around the peak of the observed
            peak_time = df_usgs["obs"].idxmax()
            
            swmm_hrs = (pd.DatetimeIndex(swmm_t).tz_localize(None) - peak_time).total_seconds() / 3600
            usgs_hrs = (pd.DatetimeIndex(usgs_t).tz_localize(None) - peak_time).total_seconds() / 3600
            
            ax.plot(usgs_hrs, usgs_q, label="Observed (USGS)", color="black", linewidth=2)
            ax.plot(swmm_hrs, swmm_q, label="Simulated (SWMM)", color="#3b82f6", linestyle="--", linewidth=2)
            
            ax.set_title(f"{title}\nNSE = {nse:.2f}")
            ax.set_ylabel("Flow (CFS)")
            ax.set_xlabel("Time relative to peak (Hours)")
            ax.set_xlim(-12, 12)
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig("docs/assets/hydrographs.png", dpi=300)
    print("Plot saved to docs/assets/hydrographs.png")

if __name__ == "__main__":
    plot_events()
