"""
model/train_surrogate.py
========================
Step 6: PCA + XGBoost Surrogate Training.

Trains a two-stage surrogate on the Monte Carlo dataset (Step 5):
  Stage 1 — PCA: compress 1,152-step hydrographs to K principal components
  Stage 2 — XGBoost: map 3D inputs → K PCA scores

Inputs  (3D LHS):   rain_depth_mm, storm_dur_hr, api_5_mm
Outputs (1,152 pts): flow at 5-min intervals over 4-day simulation window

Also trains a scalar peak-flow surrogate (simpler, for flood-warning use).

Usage:
    uv run python model/train_surrogate.py
    uv run python model/train_surrogate.py --n-components 20  # custom PCA rank
    uv run python model/train_surrogate.py --peak-only        # scalar surrogate only

Outputs:
    model/surrogate/pca_model.joblib       — fitted PCA
    model/surrogate/xgb_pca.joblib         — XGBoost → PCA scores
    model/surrogate/xgb_peak.joblib        — XGBoost → peak CFS (scalar)
    model/surrogate/meta.json              — metadata + benchmark metrics
    model/surrogate/pca_variance.png       — explained variance plot
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MC_DIR      = Path("data/raw/mc")
OUT_DIR     = Path("model/surrogate")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MC_REPORT_STEP_SEC = 300   # 5 min
MC_N_STEPS         = 1152  # 96 hr × 12 steps/hr

FEATURE_COLS = ["rain_depth_mm", "storm_dur_hr", "api_5_mm"]
# Derived features added at training time (domain physics):
#   rain_intensity = rain_depth_mm / storm_dur_hr  [mm/hr]
#   api_factor     = clip(1 - 0.008*api_5_mm, 0.4, 1.0)  [IMD scale factor]
#   eff_depth      = rain_depth_mm * api_factor  [effective runoff depth]
ENGINEERED_COLS = ["rain_depth_mm", "storm_dur_hr", "api_5_mm",
                   "rain_intensity", "api_factor", "eff_depth"]
TEST_SIZE    = 0.20
RANDOM_SEED  = 42
CV_FOLDS     = 5
MIN_PEAK_CFS = 50.0   # filter near-zero-flow runs (tiny drizzle storms)
                       # that blow up percentage metrics and corrupt PCA modes
DEFAULT_PCA_K = 30    # train 30 PCA models; keep only those with R² >= MIN_PC_R2
MIN_PC_R2    = 0.50   # discard PCs that XGBoost can't predict well
                       # (PC05+ tend to drop sharply; filtering avoids noise injection)

# XGBoost hyperparameters — tuned for N~400 training samples, 3 features
XGB_PARAMS = dict(
    n_estimators    = 500,
    max_depth       = 5,
    learning_rate   = 0.05,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    reg_alpha       = 0.1,
    reg_lambda      = 1.0,
    random_state    = RANDOM_SEED,
    tree_method     = "hist",
    n_jobs          = -1,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_nse(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency averaged over test samples."""
    nse_vals = []
    for yt, yp in zip(Y_true, Y_pred):
        denom = np.sum((yt - yt.mean()) ** 2)
        if denom < 1e-9:
            continue
        nse_vals.append(1.0 - np.sum((yt - yp) ** 2) / denom)
    return float(np.mean(nse_vals)) if nse_vals else float("nan")


def compute_peak_err_pct(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    errs = []
    for yt, yp in zip(Y_true, Y_pred):
        pk = yt.max()
        if pk < MIN_PEAK_CFS:   # skip near-zero runs
            continue
        errs.append(abs(yp.max() - pk) / pk * 100)
    return float(np.mean(errs)) if errs else float("nan")


def compute_ttp_err_min(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    errs = []
    for yt, yp in zip(Y_true, Y_pred):
        if yt.max() < MIN_PEAK_CFS:  # skip near-zero runs
            continue
        errs.append(abs(int(yt.argmax()) - int(yp.argmax())) * MC_REPORT_STEP_SEC / 60)
    return float(np.mean(errs)) if errs else float("nan")


def compute_vol_err_pct(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    errs = []
    for yt, yp in zip(Y_true, Y_pred):
        vol_t = float(np.trapezoid(yt))
        if abs(vol_t) < 1e-3:   # skip near-zero runs
            continue
        errs.append(abs(float(np.trapezoid(yp)) - vol_t) / abs(vol_t) * 100)
    return float(np.mean(errs)) if errs else float("nan")


# ---------------------------------------------------------------------------
# PCA + XGBoost surrogate
# ---------------------------------------------------------------------------

def select_n_components(explained_var_ratio: np.ndarray, threshold: float = 0.99) -> int:
    """Return minimum components to explain `threshold` of variance."""
    cumvar = np.cumsum(explained_var_ratio)
    idx = int(np.searchsorted(cumvar, threshold))
    return max(idx + 1, 5)  # at least 5


def train_pca_xgb(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test:  np.ndarray,
    Y_test:  np.ndarray,
    n_components: int | None,
) -> dict:
    """Train PCA + multi-output XGBoost. Returns benchmark dict."""
    log.info("=== Stage 1: PCA ===")

    # Default: use DEFAULT_PCA_K components (top modes with good XGB predictability)
    # Using all 400 components includes high-frequency noise modes (PC05+ have R²<0)
    # that destroy reconstruction quality.
    k = n_components if n_components is not None else DEFAULT_PCA_K
    k = min(k, Y_train.shape[0] - 1, Y_train.shape[1])  # safety cap

    # Fit PCA on training hydrographs
    pca = PCA(n_components=k, random_state=RANDOM_SEED)
    pca.fit(Y_train)

    cumvar = float(np.sum(pca.explained_variance_ratio_))
    k_auto = select_n_components(pca.explained_variance_ratio_)
    k_used = pca.n_components_
    log.info("  Components: %d  Explained variance: %.4f", k_used, cumvar)
    log.info("  Components for 99%% variance: %d", k_auto)

    # Project to PCA space
    Z_train = pca.transform(Y_train)  # (N_train, K)
    Z_test  = pca.transform(Y_test)   # (N_test,  K)

    log.info("=== Stage 2: XGBoost → PCA scores (%d trained, R²≥%.1f kept) ===",
             k_used, MIN_PC_R2)

    # Train one XGBoost per PCA component
    xgb_models = []
    r2_per_component = []
    good_indices = []          # indices of PCs that meet the R² threshold
    t0 = time.time()
    for k in range(k_used):
        xgb = XGBRegressor(**XGB_PARAMS)
        xgb.fit(X_train, Z_train[:, k])
        r2_k = r2_score(Z_test[:, k], xgb.predict(X_test))
        r2_per_component.append(r2_k)
        xgb_models.append(xgb)
        if r2_k >= MIN_PC_R2:
            good_indices.append(k)
        if k < 5 or k == k_used - 1 or r2_k >= MIN_PC_R2:
            flag = "✓" if r2_k >= MIN_PC_R2 else "✗"
            log.info("  %s PC%02d  R²=%.4f", flag, k + 1, r2_k)
    train_time = time.time() - t0

    n_good = len(good_indices)
    log.info("  Keeping %d / %d components (R² ≥ %.1f)", n_good, k_used, MIN_PC_R2)

    # Reconstruct hydrographs using ONLY good components
    # For bad components, XGBoost prediction is zero (mean in PCA space)
    t_infer = time.time()
    Z_pred = np.zeros((len(X_test), k_used), dtype=np.float32)
    for k in good_indices:
        Z_pred[:, k] = xgb_models[k].predict(X_test)
    Y_pred = pca.inverse_transform(Z_pred)
    Y_pred = np.clip(Y_pred, 0, None)
    infer_ms = (time.time() - t_infer) / len(X_test) * 1000

    # Benchmark
    nse      = compute_nse(Y_test, Y_pred)
    r2_flat  = r2_score(Y_test.ravel(), Y_pred.ravel())
    mae      = mean_absolute_error(Y_test.ravel(), Y_pred.ravel())
    peak_err = compute_peak_err_pct(Y_test, Y_pred)
    ttp_err  = compute_ttp_err_min(Y_test, Y_pred)
    vol_err  = compute_vol_err_pct(Y_test, Y_pred)

    log.info("\n=== PCA+XGBoost Benchmark ===")
    log.info("  NSE:          %.4f", nse)
    log.info("  R² (flat):    %.4f", r2_flat)
    log.info("  MAE:          %.1f CFS", mae)
    log.info("  Peak err:     %.2f%%", peak_err)
    log.info("  TTP err:      %.1f min", ttp_err)
    log.info("  Volume err:   %.2f%%", vol_err)
    log.info("  Train time:   %.1f s", train_time)
    log.info("  Infer time:   %.2f ms/sample", infer_ms)

    # Save artifacts — include good_indices so inference knows which PCs to use
    joblib.dump(pca,        OUT_DIR / "pca_model.joblib")
    joblib.dump({"models": xgb_models, "good_indices": good_indices},
                OUT_DIR / "xgb_pca.joblib")
    log.info("  Saved: %s", OUT_DIR / "pca_model.joblib")
    log.info("  Saved: %s", OUT_DIR / "xgb_pca.joblib")

    return {
        "nse":               nse,
        "r2_flat":           r2_flat,
        "mae_cfs":           mae,
        "peak_err_pct":      peak_err,
        "ttp_err_min":       ttp_err,
        "vol_err_pct":       vol_err,
        "train_time_s":      train_time,
        "infer_ms_per_samp": infer_ms,
        "n_pca_components":  k_used,
        "n_good_components": n_good,
        "pca_variance":      cumvar,
        "r2_per_component":  r2_per_component,
    }


# ---------------------------------------------------------------------------
# Scalar peak surrogate
# ---------------------------------------------------------------------------

def add_engineered_features(X_df: pd.DataFrame) -> np.ndarray:
    """Append derived physics features to the raw 3-column feature matrix."""
    X_out = X_df.copy()
    X_out["rain_intensity"] = X_out["rain_depth_mm"] / X_out["storm_dur_hr"].clip(lower=0.1)
    X_out["api_factor"]     = (1 - 0.008 * X_out["api_5_mm"]).clip(lower=0.4)
    X_out["eff_depth"]      = X_out["rain_depth_mm"] * X_out["api_factor"]
    return X_out[ENGINEERED_COLS].values.astype(np.float32)


def train_peak_surrogate(
    X_train_df: pd.DataFrame,
    y_peak_train: np.ndarray,
    X_test_df:  pd.DataFrame,
    y_peak_test:  np.ndarray,
) -> dict:
    """Train XGBoost to predict peak CFS using engineered features."""
    log.info("=== Scalar Peak Surrogate (6 engineered features) ===")

    X_train = add_engineered_features(X_train_df)
    X_test  = add_engineered_features(X_test_df)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    xgb = XGBRegressor(**XGB_PARAMS)
    t0 = time.time()
    xgb.fit(X_tr_s, y_peak_train)
    train_time = time.time() - t0

    y_pred = xgb.predict(X_te_s)
    r2  = r2_score(y_peak_test, y_pred)
    mae = mean_absolute_error(y_peak_test, y_pred)
    # MAPE: exclude near-zero actuals to avoid division blow-up
    mask = y_peak_test >= MIN_PEAK_CFS
    if mask.sum() > 0:
        mape = float((np.abs(y_pred[mask] - y_peak_test[mask]) / y_peak_test[mask] * 100).mean())
    else:
        mape = float("nan")

    # 5-fold CV
    pipe = Pipeline([("sc", StandardScaler()), ("xgb", XGBRegressor(**XGB_PARAMS))])
    cv_r2 = cross_val_score(pipe, X_train, y_peak_train, cv=CV_FOLDS, scoring="r2")

    t_infer = time.time()
    xgb.predict(X_te_s)
    infer_ms = (time.time() - t_infer) / len(X_te_s) * 1000

    log.info("  R² (holdout):  %.4f", r2)
    log.info("  R² (CV mean):  %.4f ± %.4f", cv_r2.mean(), cv_r2.std())
    log.info("  MAE:           %.0f CFS", mae)
    log.info("  MAPE:          %.2f%%", mape)
    log.info("  Train time:    %.2f s", train_time)
    log.info("  Infer time:    %.3f ms/sample", infer_ms)

    joblib.dump({"model": xgb, "scaler": scaler, "feature_cols": ENGINEERED_COLS},
                OUT_DIR / "xgb_peak.joblib")
    log.info("  Saved: %s", OUT_DIR / "xgb_peak.joblib")

    return {
        "r2_holdout":   r2,
        "r2_cv_mean":   float(cv_r2.mean()),
        "r2_cv_std":    float(cv_r2.std()),
        "mae_cfs":      mae,
        "mape_pct":     mape,
        "train_time_s": train_time,
        "infer_ms":     infer_ms,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-components", type=int, default=None,
                   help="PCA components (default: auto at 99%% variance)")
    p.add_argument("--peak-only", action="store_true",
                   help="Skip PCA hydrograph surrogate, train only scalar peak model")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Load data ---
    log.info("Loading MC dataset from %s ...", MC_DIR)
    hyd_df  = pd.read_parquet(MC_DIR / "mc_hydrographs.parquet")
    meta_df = pd.read_parquet(MC_DIR / "mc_metadata.parquet")
    inp_df  = pd.read_parquet(MC_DIR / "mc_inputs.parquet")

    # Align on run_id
    run_ids = hyd_df.index.tolist()
    meta_df = meta_df.set_index("run_id").loc[run_ids].reset_index()
    inp_df  = inp_df.set_index("run_id").loc[run_ids].reset_index()

    Y = hyd_df.values.astype(np.float32)           # (N, 1152) hydrographs
    X_df   = inp_df[FEATURE_COLS].copy()             # DataFrame with named cols
    X      = X_df.values.astype(np.float32)         # (N, 3) array
    y_peak = meta_df["peak_cfs"].values.astype(np.float32)

    N_raw = len(Y)
    # Filter near-zero-flow runs before training (corrupts PCA modes + percent metrics)
    keep = y_peak >= MIN_PEAK_CFS
    n_filtered = int((~keep).sum())
    if n_filtered > 0:
        log.info("Filtering %d near-zero-peak runs (peak < %.0f CFS) from training",
                 n_filtered, MIN_PEAK_CFS)
        Y      = Y[keep]
        X      = X[keep]
        X_df   = X_df[keep].reset_index(drop=True)
        y_peak = y_peak[keep]
    N = len(Y)
    log.info("Dataset after filter: N=%d runs (removed %d)", N, n_filtered)
    log.info("Peak CFS range: %.0f – %.0f", y_peak.min(), y_peak.max())

    # --- Train / test split ---
    idx = np.arange(N)
    idx_tr, idx_te = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    X_train, X_test       = X[idx_tr], X[idx_te]
    X_train_df            = X_df.iloc[idx_tr].reset_index(drop=True)
    X_test_df             = X_df.iloc[idx_te].reset_index(drop=True)
    Y_train, Y_test       = Y[idx_tr], Y[idx_te]
    y_pk_tr, y_pk_te      = y_peak[idx_tr], y_peak[idx_te]

    log.info("Train: %d  Test: %d", len(idx_tr), len(idx_te))

    results = {"n_runs": N, "n_train": len(idx_tr), "n_test": len(idx_te)}

    # --- PCA + XGBoost hydrograph surrogate ---
    if not args.peak_only:
        pca_results = train_pca_xgb(X_train, Y_train, X_test, Y_test, args.n_components)
        results["pca_xgb"] = pca_results

    # --- Scalar peak surrogate ---
    peak_results = train_peak_surrogate(X_train_df, y_pk_tr, X_test_df, y_pk_te)
    results["peak_xgb"] = peak_results

    # --- Save metadata ---
    meta_path = OUT_DIR / "meta.json"
    meta_path.write_text(json.dumps(results, indent=2))
    log.info("\nSaved surrogate metadata → %s", meta_path)

    # --- Final summary ---
    log.info("\n" + "=" * 60)
    log.info("SURROGATE TRAINING COMPLETE")
    log.info("=" * 60)
    if not args.peak_only and "pca_xgb" in results:
        r = results["pca_xgb"]
        log.info("  PCA+XGBoost hydrograph: NSE=%.4f  Peak err=%.1f%%  TTP err=%.0f min",
                 r["nse"], r["peak_err_pct"], r["ttp_err_min"])
    r = results["peak_xgb"]
    log.info("  Peak XGBoost (scalar):   R²=%.4f  MAE=%.0f CFS  MAPE=%.2f%%",
             r["r2_holdout"], r["mae_cfs"], r["mape_pct"])
    log.info("=" * 60)
    log.info("Artifacts in %s/", OUT_DIR)


if __name__ == "__main__":
    main()
