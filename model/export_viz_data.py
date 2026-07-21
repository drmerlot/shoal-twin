import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from model.calibrate import run_swmm, load_usgs, STORM_WINDOWS
from model.train_surrogate import add_engineered_features

PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 1. Generate Historical SWMM Hydrographs
with open(ROOT / "model" / "best_params.json") as f:
    best_params = json.load(f)
    
storms = ["memorial_day_2015", "halloween_2013", "halloween_2015", "july_2025"]
hist_hydrographs = {}

print("Running SWMM for historic events...")
for s in storms:
    try:
        times, flows = run_swmm(best_params, s)
        usgs_series = load_usgs(s)
        usgs_times = [str(t) for t in usgs_series.index]
        usgs_flows = [float(f) for f in usgs_series.values]
        hist_hydrographs[s] = {
            "times": [str(t) for t in times], 
            "flows": [float(f) for f in flows],
            "usgs_times": usgs_times,
            "usgs_flows": usgs_flows
        }
        print(f"  {s}: peak {max(flows):.0f} CFS")
    except Exception as e:
        print(f"Failed {s}: {e}")

with open(PROCESSED_DIR / "hist_hydrographs.json", "w") as f:
    json.dump(hist_hydrographs, f)

# 2. Generate Surrogate Scatter Data
print("Generating train/test scatter data...")
MC_DIR = ROOT / "data" / "raw" / "mc"
SURR_DIR = ROOT / "model" / "surrogate"

inp = pd.read_parquet(MC_DIR / "mc_inputs.parquet")
meta = pd.read_parquet(MC_DIR / "mc_metadata.parquet")
hyd = pd.read_parquet(MC_DIR / "mc_hydrographs.parquet")

run_ids = hyd.index.tolist()
meta = meta.set_index("run_id").loc[run_ids].reset_index()
inp = inp.set_index("run_id").loc[run_ids].reset_index()

valid = meta["peak_cfs"] >= 50
inp = inp[valid].reset_index(drop=True)
meta = meta[valid].reset_index(drop=True)
y = meta["peak_cfs"].values.astype(np.float32)
X = add_engineered_features(inp)

idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)

model_dict = joblib.load(SURR_DIR / "xgb_peak.joblib")
scaler = model_dict["scaler"]
xgb_model = model_dict["model"]

X_tr_scaled = scaler.transform(X[idx_tr])
X_te_scaled = scaler.transform(X[idx_te])
y_tr_pred = xgb_model.predict(X_tr_scaled)
y_te_pred = xgb_model.predict(X_te_scaled)

def float_list(np_arr):
    return [float(x) for x in np_arr]

scatter_data = {
    "train": {"true": float_list(y[idx_tr]), "pred": float_list(y_tr_pred)},
    "test": {"true": float_list(y[idx_te]), "pred": float_list(y_te_pred)}
}

with open(PROCESSED_DIR / "surrogate_scatter.json", "w") as f:
    json.dump(scatter_data, f)

print("Saved visualization data.")
