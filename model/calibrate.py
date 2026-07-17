"""
Step 4: SWMM Calibration — maximize NSE vs USGS 08158600
=========================================================
Optimizes 6 hydrologic parameters against Memorial Day 2015 observed
streamflow, validates on Halloween 2013.

Parameters tuned:
  pct_imp   % Impervious subcatchment (runoff volume)
  width     Subcatchment width (ft) — shapes peak timing
  ksat      Green-Ampt Ksat (in/hr)
  imd       Green-Ampt Initial Moisture Deficit
  pct_zero  % Impervious with zero depression storage
  n_ch      Manning's n for Shoal Creek main stem channel (R_UPPER, R_MID, R_LOWER)
            Range 0.025 (clean earthen) to 0.080 (dense vegetation/irregular)

Usage:
    .venv/bin/python model/calibrate.py                # full DE (~4 hrs)
    .venv/bin/python model/calibrate.py --quick        # 200 LHS samples (~10 min)
    .venv/bin/python model/calibrate.py --validate     # validate best_params.json

IMPORTANT: Must use .venv/bin/python — uv run python triggers macOS
           Gatekeeper SIGKILL on libswmm5.dylib.
"""

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
from scipy.optimize import differential_evolution
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

OUTFALL_ID         = "373195"           # junction at the USGS gauge (R_LOWER terminus)
                                         # 373196 is downstream via C373197 (25.7ft GIS artifact)
                                         # that conduit has limited capacity; measuring at 373195
                                         # captures the full weir outflow as the gauge cross-section
CALIBRATION_STORM  = "july_2025"
VALIDATION_STORM   = "halloween_2015"
NSE_TARGET         = 0.65          # raised from 0.60 — MRMS data warrants tighter fit

MODEL_DIR  = Path("model")
DATA_DIR   = Path("data/raw")
USGS_PATH  = DATA_DIR / "usgs" / "usgs_08158600_00060.parquet"
PARAMS_OUT = MODEL_DIR / "best_params.json"

# Storm INP templates (built by build_swmm_inp.py)
INP_TEMPLATES = {
    "halloween_2013":    MODEL_DIR / "shoal_creek_halloween_2013.inp",
    "memorial_day_2015": MODEL_DIR / "shoal_creek_memorial_day_2015.inp",
    "halloween_2015":    MODEL_DIR / "shoal_creek_halloween_2015.inp",
    "july_2025":         MODEL_DIR / "shoal_creek_july_2025.inp",
}

# Storm observation windows
STORM_WINDOWS = {
    "halloween_2013":    ("2013-10-29", "2013-11-02"),
    "memorial_day_2015": ("2015-05-24", "2015-05-28"),
    "halloween_2015":    ("2015-10-29", "2015-11-02"),
    "july_2025":         ("2025-07-01", "2025-07-08"),
}

# Calibration parameter bounds — MRMS QPE + physically restricted
# Architecture: split subcatchment + channel routing
#   SC_UPPER (1-lower_frac) → J_HEAD → R_UPPER → J_MID → R_MID → J_LOWER → R_LOWER → gauge
#   SC_LOWER (lower_frac)   → J_LOWER → R_LOWER → gauge  (only 3,500ft routing)
# Rainfall: MRMS_1H (basin-average 1km gridded QPE, 6.11 in total, 0.83 in/hr peak)
TOTAL_AREA_AC = 12.3 * 640   # 7,872 acres (full Shoal Creek watershed above USGS gauge)

PARAM_NAMES = ["pct_imp", "width", "ksat", "imd", "pct_zero", "n_ch", "lower_frac"]
PARAM_BOUNDS = [
    (80.0,  95.0),    # pct_imp:    % Impervious  [RESTRICTED: GIS shows >80% for Shoal Creek]
    (5000.0, 40000.0),# width:      Subcatchment width (ft)
    (0.010,  0.20),   # ksat:       Green-Ampt Ksat (in/hr)  [RESTRICTED: Austin Vertisol clay]
    (0.10,   0.40),   # imd:        Green-Ampt IMD  [RESTRICTED: realistic pre-storm moisture]
    (0.00,   50.0),   # pct_zero:   % Zero Imperv
    (0.030,  0.080),  # n_ch:       Manning's n  [RESTRICTED: Natural channel with heavy brush/weeds]
    (0.00,   0.30),   # lower_frac: fraction of watershed draining directly to J_LOWER
                      #             0.00 = all routes full 19,500ft; 0.30 = 30% bypasses to J_LOWER
]

# ---------------------------------------------------------------------------
# INP text patching
# ---------------------------------------------------------------------------

def patch_inp(template_text: str, params: dict) -> str:
    """
    Patch the SWMM .inp template with calibration parameters.
    Works section-by-section, modifying specific whitespace-delimited
    fields by index so column widths don't interfere.

    Channel routing architecture (no storage/weir):
      SC_SHOAL → J_HEAD → R_UPPER → J_MID → R_MID → J_LOWER → R_LOWER → gauge
    """
    lines = template_text.split("\n")
    out = []
    current_section = ""

    for line in lines:
        stripped = line.strip()

        # Track current section
        if stripped.startswith("["):
            current_section = stripped.split(";")[0].strip()

        # Skip comments and empty lines
        if not stripped or stripped.startswith(";"):
            out.append(line)
            continue

        fields = stripped.split()

        if fields[0] in ("SC_UPPER", "SC_LOWER", "SC_SHOAL"):
            if current_section == "[SUBCATCHMENTS]" and len(fields) >= 7:
                # Format: Name  RainGage  Outlet  Area  %Imperv  Width  %Slope  [CurbLen]
                # Indices:  0      1        2       3      4       5       6        7
                lower_frac = params.get("lower_frac", 0.0)
                if fields[0] == "SC_UPPER":
                    area = TOTAL_AREA_AC * (1.0 - lower_frac)
                elif fields[0] == "SC_LOWER":
                    area = TOTAL_AREA_AC * lower_frac
                else:  # SC_SHOAL (old templates)
                    area = TOTAL_AREA_AC
                fields[3] = f"{area:.1f}"
                fields[4] = f"{params['pct_imp']:.1f}"
                fields[5] = f"{params['width']:.0f}"
                out.append("  ".join(f"{f:<16}" if i < 3 else f"{f:<8}" for i, f in enumerate(fields)).rstrip())
                continue

            elif current_section == "[SUBAREAS]" and len(fields) >= 6:
                # Format: Name  N-Imperv  N-Perv  S-Imperv  S-Perv  PctZero  RouteTo  [PctRouted]
                fields[5] = f"{params['pct_zero']:.1f}"
                out.append("  ".join(f"{f:<16}" if i == 0 else f"{f:<10}" for i, f in enumerate(fields)).rstrip())
                continue

            elif current_section == "[INFILTRATION]" and len(fields) >= 4:
                # Format: Name  Suction  Ksat  IMD
                fields[2] = f"{params['ksat']:.4f}"
                fields[3] = f"{params['imd']:.4f}"
                out.append("  ".join(f"{f:<16}" if i == 0 else f"{f:<10}" for i, f in enumerate(fields)).rstrip())
                continue

        # Patch Manning's n into the 3 channel reaches (R_UPPER, R_MID, R_LOWER)
        # [CONDUITS] format: Name  FromNode  ToNode  Length  Roughness  InOff OutOff InitQ MaxQ
        # Index:              0       1         2       3        4
        if (current_section == "[CONDUITS]"
                and fields[0] in ("R_UPPER", "R_MID", "R_LOWER")
                and len(fields) >= 5):
            fields[4] = f"{params['n_ch']:.4f}"
            out.append("  ".join(f"{f:<16}" if i < 3 else f"{f:<10}" for i, f in enumerate(fields)).rstrip())
            continue

        out.append(line)

    return "\n".join(out)



# ---------------------------------------------------------------------------
# SWMM runner
# ---------------------------------------------------------------------------

def run_swmm(params: dict, storm_id: str) -> tuple[list, list]:
    """
    Patch the INP template, run SWMM via pyswmm, return (times, q_sim).
    Uses a temp file that is deleted after the run.
    """
    template_path = INP_TEMPLATES[storm_id]
    template_text = template_path.read_text()
    patched_text  = patch_inp(template_text, params)

    # Write to temp file (same dir as model so relative paths still work)
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".inp",
        prefix="swmm_trial_",
        dir=str(MODEL_DIR),
        delete=False,
    )
    tmp.write(patched_text)
    tmp.close()
    tmp_path = Path(tmp.name)

    times = []
    flows = []

    try:
        with Simulation(str(tmp_path)) as sim:
            nodes = Nodes(sim)
            sim.step_advance(300)   # 5-min output steps
            for _ in sim:
                times.append(sim.current_time)
                flows.append(nodes[OUTFALL_ID].total_inflow)
    finally:
        tmp_path.unlink(missing_ok=True)
        # Clean up SWMM's side-car files
        for ext in (".rpt", ".out"):
            Path(str(tmp_path).replace(".inp", ext)).unlink(missing_ok=True)

    return times, flows


# ---------------------------------------------------------------------------
# NSE computation
# ---------------------------------------------------------------------------

def load_usgs(storm_id: str) -> pd.Series:
    """Load USGS 5-min streamflow for a storm window."""
    df = pd.read_parquet(USGS_PATH)
    start, end = STORM_WINDOWS[storm_id]
    mask = (df.index >= start) & (df.index <= end)
    q_obs = df.loc[mask, "q_cfs"].dropna()
    return q_obs


def compute_nse(q_obs: pd.Series, times_sim: list, q_sim_raw: list) -> float:
    """
    Align simulated hydrograph to USGS 5-min UTC grid and compute NSE.
    Returns NSE ∈ (-∞, 1]. Returns -999 if alignment fails.
    """
    if not times_sim or not q_sim_raw:
        return -999.0

    # Build sim series (times from pyswmm are tz-naive Python datetimes)
    try:
        sim_idx = pd.to_datetime(times_sim).tz_localize("UTC")
    except Exception:
        sim_idx = pd.to_datetime(times_sim, utc=True)

    q_sim_series = pd.Series(q_sim_raw, index=sim_idx, dtype=float)
    q_sim_series = q_sim_series.clip(lower=0.0)

    # Resample to 5-min grid, forward-fill gaps ≤ 1 step
    q_sim_5min = q_sim_series.resample("5min").mean().ffill(limit=1)

    # Inner join on USGS timestamps
    aligned = pd.concat([q_obs, q_sim_5min], axis=1, join="inner")
    aligned.columns = ["obs", "sim"]
    aligned = aligned.dropna()

    if len(aligned) < 10:
        return -999.0

    obs = aligned["obs"].values
    sim = aligned["sim"].values
    obs_mean = obs.mean()
    ss_tot = np.sum((obs - obs_mean) ** 2)

    if ss_tot < 1e-6:
        return -999.0

    nse = 1.0 - np.sum((obs - sim) ** 2) / ss_tot
    return float(nse)


def compute_metrics(q_obs: pd.Series, times_sim: list, q_sim_raw: list) -> dict:
    """Full metric suite for reporting."""
    nse = compute_nse(q_obs, times_sim, q_sim_raw)

    if not times_sim:
        return {"nse": nse, "peak_sim": 0, "peak_obs": q_obs.max(),
                "peak_ratio": 0, "ttp_err_hr": float("nan")}

    peak_obs     = q_obs.max()
    peak_obs_t   = q_obs.idxmax()
    peak_sim     = max(q_sim_raw) if q_sim_raw else 0.0
    peak_sim_t   = times_sim[q_sim_raw.index(max(q_sim_raw))] if q_sim_raw else None

    try:
        peak_sim_t_utc = pd.Timestamp(peak_sim_t, tz="UTC")
        ttp_err = abs((peak_sim_t_utc - peak_obs_t).total_seconds()) / 3600.0
    except Exception:
        ttp_err = float("nan")

    return {
        "nse":         nse,
        "peak_sim":    peak_sim,
        "peak_obs":    peak_obs,
        "peak_ratio":  peak_sim / peak_obs if peak_obs > 0 else 0,
        "ttp_err_hr":  ttp_err,
    }


# ---------------------------------------------------------------------------
# Calibration loop
# ---------------------------------------------------------------------------

class CalibrationTracker:
    """Thread-safe callback for tracking best result during optimization."""
    def __init__(self, q_obs: pd.Series, storm_id: str):
        self.q_obs    = q_obs
        self.storm_id = storm_id
        self.best_nse = -999.0
        self.best_params: dict = {}
        self.n_eval    = 0
        self.t_start   = time.time()

    def evaluate(self, x: np.ndarray) -> float:
        """Objective function: returns 1 - NSE (minimize)."""
        params = dict(zip(PARAM_NAMES, x))
        self.n_eval += 1

        try:
            times, flows = run_swmm(params, self.storm_id)
            nse = compute_nse(self.q_obs, times, flows)
        except Exception as e:
            log.debug("Trial %d FAILED: %s", self.n_eval, e)
            return 2.0  # large penalty

        if nse > self.best_nse:
            self.best_nse    = nse
            self.best_params = params.copy()
            elapsed = time.time() - self.t_start
            log.info(
                "Trial %4d  NSE=%.4f *** NEW BEST ***  "
                "imp=%.0f%% w=%.0f ksat=%.3f imd=%.3f z=%.0f%% n_ch=%.4f lf=%.2f  "
                "elapsed %.0fs",
                self.n_eval, nse,
                params["pct_imp"], params["width"],
                params["ksat"], params["imd"], params["pct_zero"],
                params["n_ch"], params.get("lower_frac", 0.0),
                elapsed,
            )
        elif self.n_eval % 25 == 0:
            elapsed = time.time() - self.t_start
            log.info(
                "Trial %4d  NSE=%.4f  best=%.4f  elapsed %.0fs",
                self.n_eval, nse, self.best_nse, elapsed,
            )

        return max(0.0, 1.0 - nse)  # DE minimizes

    def de_callback(self, xk, convergence):
        """Called by DE after each generation."""
        log.info(
            "  [DE generation]  best NSE=%.4f  convergence=%.4f",
            self.best_nse, convergence,
        )
        return self.best_nse >= NSE_TARGET  # stop early if target met


# ---------------------------------------------------------------------------
# Quick mode: Latin Hypercube sampling
# ---------------------------------------------------------------------------

def run_quick(q_obs: pd.Series, n_samples: int = 200,
              storm_id: str = CALIBRATION_STORM) -> dict:
    """
    Random search via Latin Hypercube Sampling.
    Much faster than DE for an initial feasibility check.
    """
    log.info("Quick mode: %d LHS samples  storm=%s", n_samples, storm_id)
    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=42)
    sample  = sampler.random(n=n_samples)

    # Scale from [0,1] to parameter bounds
    lo = np.array([b[0] for b in PARAM_BOUNDS])
    hi = np.array([b[1] for b in PARAM_BOUNDS])
    scaled = qmc.scale(sample, lo, hi)

    tracker = CalibrationTracker(q_obs, storm_id)

    results = []
    for i, x in enumerate(scaled):
        val = tracker.evaluate(x)
        results.append((val, x.copy()))

    results.sort(key=lambda r: r[0])  # lowest (1-NSE) first
    best_val, best_x = results[0]
    best_params = dict(zip(PARAM_NAMES, best_x))
    best_nse    = tracker.best_nse

    log.info("Quick mode complete. Best NSE=%.4f", best_nse)
    return {**best_params, "nse_calibration": best_nse, "mode": "quick_lhs",
            "storm_calibration": storm_id}


# ---------------------------------------------------------------------------
# Full DE calibration
# ---------------------------------------------------------------------------

def run_de(q_obs: pd.Series, init_params: dict | None = None,
           storm_id: str = CALIBRATION_STORM) -> dict:
    """
    Differential Evolution global optimization.
    popsize=8, maxiter=150 → ~7,200 evaluations at ~3s each (~6 hrs).
    """
    tracker = CalibrationTracker(q_obs, storm_id)

    # Seed initial population from known good params if available
    init_pop = None
    if init_params:
        x0 = np.array([init_params.get(k, (lo + hi) / 2)
                        for k, (lo, hi) in zip(PARAM_NAMES, PARAM_BOUNDS)])
        # Create a population centered around x0 with small perturbation
        rng = np.random.default_rng(42)
        pop_size = 8 * len(PARAM_NAMES)
        lo = np.array([b[0] for b in PARAM_BOUNDS])
        hi = np.array([b[1] for b in PARAM_BOUNDS])
        perturbation = rng.uniform(-0.1, 0.1, size=(pop_size, len(PARAM_NAMES)))
        init_pop = np.clip(x0 + perturbation * (hi - lo), lo, hi)

    log.info("Starting Differential Evolution (workers=1, macOS-safe)")
    result = differential_evolution(
        tracker.evaluate,
        bounds=PARAM_BOUNDS,
        maxiter=40,       # 6 params × 3 pop × 40 iter ≈ 720 runs; at ~57s/run ≈ 11.4 hrs
        popsize=3,        # minimal viable population for 6D space (pop size = 3*6 = 18)
        tol=0.005,
        seed=42,
        workers=1,
        init=init_pop if init_pop is not None else "latinhypercube",
        callback=tracker.de_callback,
        disp=False,
        mutation=(0.5, 1.0),
        recombination=0.7,
    )

    best_nse = tracker.best_nse
    log.info(
        "DE complete. Evaluations: %d  Best NSE: %.4f  DE success: %s",
        tracker.n_eval, best_nse, result.success,
    )
    return {**tracker.best_params, "nse_calibration": best_nse, "mode": "differential_evolution",
            "storm_calibration": storm_id}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_validation(best_params: dict) -> dict:
    """Run best_params on the validation storm and compute metrics."""
    storm_id = VALIDATION_STORM
    log.info("Validating on %s...", storm_id)

    q_obs = load_usgs(storm_id)
    times, flows = run_swmm(best_params, storm_id)
    metrics = compute_metrics(q_obs, times, flows)

    log.info(
        "Validation %s: NSE=%.4f  peak_ratio=%.2f  TTP_err=%.1fhr",
        storm_id, metrics["nse"], metrics["peak_ratio"], metrics["ttp_err_hr"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_params(params: dict, calib_metrics: dict, valid_metrics: dict | None = None):
    """Save best parameters + metrics to JSON."""
    out = {
        **params,
        "nse_calibration": calib_metrics.get("nse", params.get("nse_calibration", -999)),
        "peak_sim_calib":  calib_metrics.get("peak_sim", 0),
        "peak_obs_calib":  calib_metrics.get("peak_obs", 0),
        "peak_ratio_calib": calib_metrics.get("peak_ratio", 0),
        "ttp_err_hr_calib": calib_metrics.get("ttp_err_hr", float("nan")),
        "storm_calibration": CALIBRATION_STORM,
    }
    if valid_metrics:
        out.update({
            "nse_validation":     valid_metrics.get("nse", -999),
            "peak_ratio_valid":   valid_metrics.get("peak_ratio", 0),
            "ttp_err_hr_valid":   valid_metrics.get("ttp_err_hr", float("nan")),
            "storm_validation":   VALIDATION_STORM,
        })

    PARAMS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(PARAMS_OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info("Saved best params -> %s", PARAMS_OUT)
    return out


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_summary(params: dict, calib_metrics: dict, valid_metrics: dict | None = None):
    print("\n" + "=" * 68)
    print("SWMM Calibration Results")
    print("=" * 68)
    print(f"  Mode:              {params.get('mode', 'unknown')}")
    print()
    print("  Best Parameters:")
    print(f"    % Impervious:          {params.get('pct_imp', 0):.1f}%")
    print(f"    Subcatchment Width:    {params.get('width', 0):.0f} ft")
    print(f"    Green-Ampt Ksat:       {params.get('ksat', 0):.4f} in/hr")
    print(f"    Green-Ampt IMD:        {params.get('imd', 0):.4f}")
    print(f"    % Zero Imperv:         {params.get('pct_zero', 0):.1f}%")
    print(f"    Manning's n (channel): {params.get('n_ch', 0):.4f}  "
          f"[bankfull Q ≈ {1.486/max(params.get('n_ch',0.04),0.001)*1875*10.72**(2/3)*0.00592**0.5:.0f} CFS]")
    print()
    print(f"  Calibration ({CALIBRATION_STORM}):")
    nse_c = calib_metrics.get("nse", params.get("nse_calibration", -999))
    print(f"    NSE:                   {nse_c:.4f}  {'✅ TARGET MET' if nse_c >= NSE_TARGET else '❌ below 0.60'}")
    print(f"    Peak sim/obs:          {calib_metrics.get('peak_ratio', 0):.2f}  "
          f"({calib_metrics.get('peak_sim', 0):.0f} / {calib_metrics.get('peak_obs', 0):.0f} CFS)")
    print(f"    Time-to-peak error:    {calib_metrics.get('ttp_err_hr', float('nan')):.1f} hr")
    if valid_metrics:
        print()
        print(f"  Validation ({VALIDATION_STORM}):")
        nse_v = valid_metrics.get("nse", -999)
        print(f"    NSE:                   {nse_v:.4f}  {'✅ PASS (≥0.50)' if nse_v >= 0.50 else '❌ below 0.50'}")
        print(f"    Peak sim/obs:          {valid_metrics.get('peak_ratio', 0):.2f}")
        print(f"    Time-to-peak error:    {valid_metrics.get('ttp_err_hr', float('nan')):.1f} hr")
    print()
    print(f"  Output: {PARAMS_OUT}")
    print("=" * 68)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SWMM Calibration — maximize NSE vs USGS 08158600"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: LHS random samples (~10 min)",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run validation using best_params.json",
    )
    parser.add_argument(
        "--cal-storm",
        choices=list(STORM_WINDOWS.keys()),
        default=CALIBRATION_STORM,
        help=f"Storm to calibrate against (default: {CALIBRATION_STORM})",
    )
    parser.add_argument(
        "--val-storm",
        choices=list(STORM_WINDOWS.keys()),
        default=VALIDATION_STORM,
        help=f"Storm to validate against (default: {VALIDATION_STORM})",
    )
    parser.add_argument(
        "--n-samples", type=int, default=200,
        help="Number of samples for --quick mode (default: 200)",
    )
    args = parser.parse_args()

    # Resolve storm IDs
    cal_storm = args.cal_storm
    val_storm = args.val_storm
    if cal_storm == val_storm:
        log.warning("cal-storm == val-storm (%s) — choosing different validation storm", cal_storm)
        val_storm = next(s for s in STORM_WINDOWS if s != cal_storm and s != "halloween_2015")

    log.info("Calibration storm: %s  Validation storm: %s", cal_storm, val_storm)

    # Check prerequisites
    for storm_id, inp_path in INP_TEMPLATES.items():
        if not inp_path.exists():
            log.error("Missing INP: %s", inp_path)
            log.error("Run: uv run python model/build_swmm_inp.py --storm %s", storm_id)
            sys.exit(1)
    if not USGS_PATH.exists():
        log.error("Missing USGS data: %s", USGS_PATH)
        sys.exit(1)

    # --- Validate-only mode ---
    if args.validate:
        if not PARAMS_OUT.exists():
            log.error("No best_params.json found. Run calibration first.")
            sys.exit(1)
        with open(PARAMS_OUT) as f:
            best_params = json.load(f)
        log.info("Loaded best params from %s", PARAMS_OUT)
        # Use stored storm IDs if present
        cal_storm = best_params.get("storm_calibration", cal_storm)
        val_storm = best_params.get("storm_validation", val_storm) if best_params.get("storm_validation") else val_storm

        q_obs_c = load_usgs(cal_storm)
        times_c, flows_c = run_swmm(best_params, cal_storm)
        calib_metrics = compute_metrics(q_obs_c, times_c, flows_c)

        valid_metrics = run_validation(best_params)
        out = save_params(best_params, calib_metrics, valid_metrics)
        print_summary(best_params, calib_metrics, valid_metrics)
        return

    # --- Calibration ---
    log.info("Step 4: SWMM Calibration  storm=%s  mode=%s",
             cal_storm, "quick" if args.quick else "full DE")
    q_obs = load_usgs(cal_storm)
    log.info(
        "USGS obs: %d records  peak=%.0f CFS  mean=%.0f CFS",
        len(q_obs), q_obs.max(), q_obs.mean(),
    )

    # Load existing best params to seed DE if available
    init_params = None
    if PARAMS_OUT.exists() and not args.quick:
        with open(PARAMS_OUT) as f:
            init_params = json.load(f)
        log.info("Seeding DE from existing best_params (NSE=%.4f)",
                 init_params.get("nse_calibration", -999))

    t0 = time.time()

    if args.quick:
        best_params = run_quick(q_obs, n_samples=args.n_samples, storm_id=cal_storm)
    else:
        best_params = run_de(q_obs, init_params=init_params, storm_id=cal_storm)

    elapsed = time.time() - t0
    log.info("Calibration complete in %.0f s  (%.1f min)", elapsed, elapsed / 60)

    # Compute full metrics on calibration storm
    log.info("Computing final calibration metrics...")
    times_c, flows_c = run_swmm(best_params, cal_storm)
    calib_metrics = compute_metrics(q_obs, times_c, flows_c)

    # Auto-validate on the other storm if NSE > 0.3
    valid_metrics = None
    if calib_metrics["nse"] > 0.3:
        log.info("NSE=%.4f > 0.3 — running validation on %s", calib_metrics["nse"], val_storm)
        q_obs_v = load_usgs(val_storm)
        times_v, flows_v = run_swmm(best_params, val_storm)
        valid_metrics = compute_metrics(q_obs_v, times_v, flows_v)
        valid_metrics["storm"] = val_storm
        log.info("Validation %s: NSE=%.4f  peak_ratio=%.2f  TTP=%.1fhr",
                 val_storm, valid_metrics["nse"], valid_metrics["peak_ratio"], valid_metrics["ttp_err_hr"])

    save_params(best_params, calib_metrics, valid_metrics)
    print_summary(best_params, calib_metrics, valid_metrics)

    if calib_metrics["nse"] >= NSE_TARGET:
        log.info("Step 4 COMPLETE — NSE target met (%.4f >= %.2f)", calib_metrics["nse"], NSE_TARGET)
        print("\n  Step 4 COMPLETE. Next: uv run python model/plot_calibration.py")
    elif calib_metrics["nse"] > 0.3:
        log.info("NSE=%.4f — promising. Run full DE overnight:", calib_metrics["nse"])
        print("\n  Promising result. Run full overnight DE:")
        print("    .venv/bin/python model/calibrate.py")
    else:
        log.warning("NSE=%.4f — poor fit. Check parameter bounds or QPE quality.", calib_metrics["nse"])


if __name__ == "__main__":
    main()
