"""
model/generate_mc_dataset.py
============================
Step 5: Monte Carlo Dataset Generation for Surrogate Training.

Generates N SWMM runs across a Latin-Hypercube-sampled 3D input space:
  - rain_depth_mm   : total storm rainfall depth (mm)
  - storm_dur_hr    : storm duration (hours)
  - api_5_mm        : 5-day antecedent precip index (mm) — proxy for initial moisture

Each run uses the best-calibrated SWMM parameters (best_params.json) with
rainfall scaled to match the LHS sample, then outputs:
  - full hydrograph at 5-min steps
  - scalar summary (peak_cfs, vol_acft, TTP_hr)

Output:
  data/raw/mc/mc_hydrographs.parquet    — full (N, T) hydrograph matrix
  data/raw/mc/mc_metadata.parquet       — per-run scalar summaries
  data/raw/mc/mc_inputs.parquet         — LHS input parameters

Usage:
    .venv/bin/python model/generate_mc_dataset.py              # 500 runs
    .venv/bin/python model/generate_mc_dataset.py --n-runs 50  # quick test
    .venv/bin/python model/generate_mc_dataset.py --n-runs 50 --dry-run  # LHS only

IMPORTANT: Must use .venv/bin/python (not uv run python).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyswmm import Nodes, Simulation
from scipy.stats import qmc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DIR  = Path("model")
DATA_DIR   = Path("data/raw")
OUT_DIR    = DATA_DIR / "mc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARAMS_PATH   = MODEL_DIR / "best_params.json"
BASE_INP      = MODEL_DIR / "shoal_creek_memorial_day_2015.inp"  # template storm
OUTFALL_ID    = "373195"   # junction where weir delivers flow to gauge

# LHS input space — 3 dimensions
# Based on Shoal Creek watershed characteristics and historical storm catalog
INPUT_SPACE = {
    # name         : (low,   high,  unit)
    "rain_depth_mm":  (15.0,  300.0, "mm"),     # 0.6 in → 11.8 in total storm depth
    "storm_dur_hr":   (1.0,   24.0,  "hr"),     # 1-hour burst → 1-day frontal storm
    "api_5_mm":       (0.0,   75.0,  "mm"),     # dry → very wet antecedent conditions
}

DEFAULT_N_RUNS = 500


# ---------------------------------------------------------------------------
# Rainfall synthesis — triangular hyetograph
# ---------------------------------------------------------------------------

def build_triangular_rain_timeseries(
    depth_mm: float,
    dur_hr: float,
    api_5_mm: float,
    base_ts_name: str = "RG_synthetic",
    dt_min: int = 60,
) -> tuple[list[tuple[str, str, str, float]], dict]:
    """
    Build a synthetic triangular hyetograph timeseries for SWMM [TIMESERIES].

    Triangular distribution: peak at 40% of duration (Huff Type II approximation).
    Total depth = depth_mm (converted to inches for SWMM).
    Returns list of (name, date, time, value_in_hr) tuples.

    API effect: increases initial moisture (imd → imd * api_factor).
    """
    depth_in = depth_mm / 25.4

    # Number of 1-hour intervals
    n_hours = int(np.ceil(dur_hr))
    # Triangular distribution: peak at 40% of duration
    peak_hr = 0.40 * dur_hr

    # Compute hourly intensities from triangular distribution
    intensities = []
    for i in range(n_hours):
        t_mid = i + 0.5  # midpoint of hour
        if t_mid <= peak_hr:
            frac = t_mid / peak_hr
        else:
            frac = (dur_hr - t_mid) / (dur_hr - peak_hr) if dur_hr > peak_hr else 0.0
        frac = max(frac, 0.0)
        intensities.append(frac)

    # Normalize to total depth
    total_frac = sum(intensities)
    if total_frac > 0:
        intensities = [f / total_frac * depth_in for f in intensities]
    else:
        intensities = [depth_in / n_hours] * n_hours

    # Build SWMM timeseries rows — use a fixed reference date
    # (SWMM [TIMESERIES] rows: Name  Date  Time  Value)
    ref_date = "01/01/2000"
    rows = []
    for i, intensity in enumerate(intensities):
        hh, mm = divmod(i * 60, 60)
        time_str = f"{hh:02d}:{mm:02d}"
        rows.append((base_ts_name, ref_date, time_str, intensity))
    # Add explicit zero at end
    end_hh, end_mm = divmod(n_hours * 60, 60)
    rows.append((base_ts_name, ref_date, f"{end_hh:02d}:{end_mm:02d}", 0.0))

    # API factor: scale initial moisture deficit (IMD) based on antecedent precip
    # Higher API → more saturated → lower effective IMD
    # api_5_mm=0 → no effect; api_5_mm=75 → IMD reduced by 60%
    api_factor = max(0.4, 1.0 - 0.008 * api_5_mm)  # 1.0 at dry, 0.4 at API=75mm

    return rows, {"api_factor": api_factor, "n_hours": n_hours}


def patch_inp_for_mc(
    inp_text: str,
    best_params: dict,
    rain_rows: list,
    ts_name: str,
    api_factor: float,
    sim_dur_hr: float,
) -> str:
    """
    Patch an INP template for a Monte Carlo run:
    1. Replace [TIMESERIES] with synthetic rainfall
    2. Adjust [OPTIONS] simulation duration
    3. Apply best_params (using calibrate.patch_inp to handle architecture correctly)
    """
    import sys
    if "model" not in sys.path:
        sys.path.insert(0, "model")
    from calibrate import patch_inp
    
    # 1. Base patching using calibrate.py (handles new architecture)
    p_mc = best_params.copy()
    p_mc["imd"] = p_mc["imd"] * api_factor
    base_patched = patch_inp(inp_text, p_mc)
    
    # 2. MC-specific patching for rainfall & timing
    lines = base_patched.splitlines()
    out = []
    current_section = ""
    in_timeseries = False
    
    # Fixed 4-day simulation window (96 hours) for ALL runs
    FIXED_END_DATE = "01/05/2000"   
    FIXED_END_TIME = "00:00:00"
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            current_section = stripped.split("]")[0] + "]"
            in_timeseries = (current_section == "[TIMESERIES]")
            if in_timeseries:
                out.append("[TIMESERIES]")
                out.append(";;Name           Date       Time       Value")
                out.append(";;-------------- ---------- ---------- ----------")
                for ts_name_, date, t, val in rain_rows:
                    out.append(f"{ts_name_:<16} {date:<10} {t:<10} {val:.4f}")
                out.append("")
                continue
            else:
                in_timeseries = False
            out.append(line)
            continue
            
        if in_timeseries:
            continue
            
        fields = stripped.split()
        if current_section == "[OPTIONS]" and len(fields) >= 2:
            if fields[0] == "END_DATE":
                out.append(f"END_DATE             {FIXED_END_DATE}")
                continue
            if fields[0] == "END_TIME":
                out.append(f"END_TIME             {FIXED_END_TIME}")
                continue
            if fields[0] == "START_DATE":
                out.append(f"START_DATE           01/01/2000")
                continue
            if fields[0] == "START_TIME":
                out.append(f"START_TIME           00:00:00")
                continue
            if fields[0] == "REPORT_START_DATE":
                out.append(f"REPORT_START_DATE    01/01/2000")
                continue
            if fields[0] == "REPORT_START_TIME":
                out.append(f"REPORT_START_TIME    00:00:00")
                continue
            if fields[0] == "DRY_DAYS":
                out.append(f"DRY_DAYS             0")
                continue
                
        if (current_section == "[RAINGAGES]"
                and fields and not fields[0].startswith(";;")
                and len(fields) >= 6
                and fields[4] == "TIMESERIES"):
            f = fields[:]
            f[5] = ts_name
            out.append("  ".join(f"{v:<16}" if i == 0 else f"{v:<10}" for i, v in enumerate(f)).rstrip())
            continue
            
        out.append(line)
        
    return "\n".join(out)



# Fixed output grid: 4 days at 5-min steps = 1,152 points
MC_REPORT_STEP_SEC = 300   # 5 minutes
MC_SIM_DURATION_HR = 96   # 4 days
MC_N_STEPS = MC_SIM_DURATION_HR * 3600 // MC_REPORT_STEP_SEC  # = 1152


def run_swmm_mc(inp_text: str) -> tuple[np.ndarray, float]:
    """
    Run SWMM from INP text; resample to regular 5-min grid.
    Returns (flows_5min: np.ndarray of shape (1152,), peak_cfs: float).

    pyswmm with VARIABLE_STEP yields at every adaptive routing sub-step
    (~9s avg), so we collect raw (time_sec, flow) pairs then interpolate
    onto the fixed 5-min grid.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".inp", mode="w", delete=False, prefix="mc_shoal_"
    ) as f:
        f.write(inp_text)
        tmp_path = Path(f.name)

    raw_sec, raw_flow = [], []
    rpt = tmp_path.with_suffix(".rpt")
    out = tmp_path.with_suffix(".out")
    try:
        with Simulation(str(tmp_path)) as sim:
            t0 = sim.start_time
            nodes = Nodes(sim)
            sim.start()
            for step in sim:
                dt = (sim.current_time - t0).total_seconds()
                raw_sec.append(dt)
                raw_flow.append(nodes[OUTFALL_ID].total_inflow)
    except Exception as e:
        log.warning("SWMM run failed: %s", e)
        return np.zeros(MC_N_STEPS, dtype=np.float32), 0.0
    finally:
        for p in [tmp_path, rpt, out]:
            if p.exists():
                p.unlink(missing_ok=True)

    if not raw_flow:
        return np.zeros(MC_N_STEPS, dtype=np.float32), 0.0

    # Resample to regular 5-min grid via linear interpolation
    raw_sec_arr  = np.array(raw_sec,  dtype=np.float64)
    raw_flow_arr = np.array(raw_flow, dtype=np.float64)
    grid_sec = np.arange(0, MC_N_STEPS) * MC_REPORT_STEP_SEC  # 0, 300, 600, ...
    flows_5min = np.interp(grid_sec, raw_sec_arr, raw_flow_arr).astype(np.float32)

    peak_cfs = float(flows_5min.max())
    return flows_5min, peak_cfs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Monte Carlo dataset generation")
    p.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS)
    p.add_argument("--dry-run", action="store_true",
                   help="Generate LHS samples only, don't run SWMM")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    N = args.n_runs

    # Load best calibration params
    if not PARAMS_PATH.exists():
        log.error("best_params.json not found — run calibrate.py first")
        sys.exit(1)
    best_params = json.loads(PARAMS_PATH.read_text())
    log.info("Loaded best params: NSE=%.4f (calib storm: %s)",
             best_params.get("nse_calibration", -999),
             best_params.get("storm_calibration", "?"))

    # Load INP template
    if not BASE_INP.exists():
        log.error("Base INP not found: %s", BASE_INP)
        sys.exit(1)
    inp_template = BASE_INP.read_text()

    # Latin Hypercube Sampling — 3D input space
    log.info("Generating LHS samples: N=%d, seed=%d", N, args.seed)
    sampler = qmc.LatinHypercube(d=3, seed=args.seed)
    sample = sampler.random(n=N)

    input_names = list(INPUT_SPACE.keys())
    lows  = np.array([v[0] for v in INPUT_SPACE.values()])
    highs = np.array([v[1] for v in INPUT_SPACE.values()])
    scaled = qmc.scale(sample, lows, highs)

    inputs_df = pd.DataFrame(scaled, columns=input_names)
    inputs_df["run_id"] = [f"MC_{i:04d}" for i in range(N)]
    log.info("LHS sample stats:")
    for col in input_names:
        log.info("  %-20s  min=%.1f  max=%.1f  mean=%.1f",
                 col, inputs_df[col].min(), inputs_df[col].max(), inputs_df[col].mean())

    if args.dry_run:
        out_path = OUT_DIR / "mc_inputs.parquet"
        inputs_df.to_parquet(out_path, index=False)
        log.info("Dry run: saved LHS inputs → %s", out_path)
        return

    # Run SWMM for each sample
    meta_rows = []
    hydrograph_dict = {}
    t_start = time.time()

    for i, row in inputs_df.iterrows():
        run_id = row["run_id"]
        rain_mm   = row["rain_depth_mm"]
        dur_hr    = row["storm_dur_hr"]
        api_mm    = row["api_5_mm"]
        ts_name   = f"RG_{run_id}"

        rain_rows, rain_meta = build_triangular_rain_timeseries(
            depth_mm=rain_mm,
            dur_hr=dur_hr,
            api_5_mm=api_mm,
            base_ts_name=ts_name,
        )
        api_factor = rain_meta["api_factor"]

        patched_inp = patch_inp_for_mc(
            inp_text=inp_template,
            best_params=best_params,
            rain_rows=rain_rows,
            ts_name=ts_name,
            api_factor=api_factor,
            sim_dur_hr=dur_hr,
        )

        t0 = time.time()
        flows_5min, peak_cfs = run_swmm_mc(patched_inp)
        elapsed = time.time() - t0

        if peak_cfs == 0.0:
            log.warning("Run %s failed (zero peak)", run_id)
            continue

        # Metrics on the uniform 5-min grid
        vol_acft = float(flows_5min.sum() * MC_REPORT_STEP_SEC / 43560)
        ttp_hr   = float(np.argmax(flows_5min) * MC_REPORT_STEP_SEC / 3600)

        hydrograph_dict[run_id] = flows_5min  # shape (1152,) always

        meta_rows.append({
            "run_id":        run_id,
            "rain_depth_mm": rain_mm,
            "storm_dur_hr":  dur_hr,
            "api_5_mm":      api_mm,
            "peak_cfs":      peak_cfs,
            "vol_acft":      vol_acft,
            "ttp_hr":        ttp_hr,
            "n_steps":       MC_N_STEPS,
        })

        if (i + 1) % 25 == 0 or i == 0:
            elapsed_total = time.time() - t_start
            rate = elapsed_total / (i + 1)
            remaining = (N - i - 1) * rate / 60
            log.info(
                "Run %4d/%d  rain=%.0fmm  dur=%.1fhr  peak=%.0fCFS  "
                "t=%.0fs  ETA=%.0f min",
                i + 1, N, rain_mm, dur_hr, peak_cfs, elapsed, remaining,
            )

    # Save outputs
    if not hydrograph_dict:
        log.error("All SWMM runs failed — check INP patching. Aborting.")
        sys.exit(1)

    log.info("Saving outputs to %s ...", OUT_DIR)

    # Inputs
    inputs_df.to_parquet(OUT_DIR / "mc_inputs.parquet", index=False)

    # Metadata (scalar summaries)
    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_parquet(OUT_DIR / "mc_metadata.parquet", index=False)

    # Hydrographs — all runs already have uniform shape (MC_N_STEPS,)
    run_ids    = list(hydrograph_dict.keys())
    hyd_matrix = np.stack([hydrograph_dict[rid] for rid in run_ids])  # (N, 1152)
    hyd_df     = pd.DataFrame(hyd_matrix, index=run_ids)
    hyd_df.index.name = "run_id"
    hyd_df.to_parquet(OUT_DIR / "mc_hydrographs.parquet")

    total_time = time.time() - t_start
    log.info("Done. %d runs in %.0f min (%.0f s/run avg)",
             len(meta_rows), total_time / 60, total_time / len(meta_rows))
    log.info("Outputs:")
    log.info("  %s  (%d runs)", OUT_DIR / "mc_inputs.parquet", len(inputs_df))
    log.info("  %s  (%d runs × %d steps)", OUT_DIR / "mc_hydrographs.parquet",
             len(run_ids), MC_N_STEPS)
    log.info("  %s", OUT_DIR / "mc_metadata.parquet")

    # Summary stats
    log.info("\n=== MC Dataset Summary ===")
    log.info("  Peak CFS:  min=%.0f  mean=%.0f  max=%.0f",
             meta_df["peak_cfs"].min(), meta_df["peak_cfs"].mean(), meta_df["peak_cfs"].max())
    log.info("  Volume:    min=%.0f  mean=%.0f  max=%.0f ac-ft",
             meta_df["vol_acft"].min(), meta_df["vol_acft"].mean(), meta_df["vol_acft"].max())
    log.info("  TTP:       min=%.1f  mean=%.1f  max=%.1f hr",
             meta_df["ttp_hr"].min(), meta_df["ttp_hr"].mean(), meta_df["ttp_hr"].max())


if __name__ == "__main__":
    main()
