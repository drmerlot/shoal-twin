# shoal-twin: Project Context & Decision Log

> **Purpose of this file:** Briefing document for any new agent session.
> Read this first. It captures all key decisions, current state, and next steps
> so you don't need the full conversation history.

---

## What This Project Is

A real operational flood forecasting system for Shoal Creek, Austin TX.
Three pieces:
1. **Calibrated SWMM 1D physics model** of Shoal Creek (pyswmm, Mac-compatible)
2. **ML surrogate** that emulates SWMM fast (PCA + XGBoost, from existing swmm-twin framework)
3. **Forecast app** (Streamlit) that takes weather forcing and outputs predicted hydrograph + flood map

**Goal:** Demonstrable, technically credible system by end of August 2026 for a meeting
with Brandon Klenzendorf (Principal Engineer, Geosyntec Austin — HEC-RAS 2D expert, flood planning).
Purpose: job referral / connection. Differentiation: hydrology domain knowledge + ML/surrogate
infrastructure capability (rare in the water resources world).

**Constraint:** Progress in demonstrable increments every few days. Don't get stuck in loops.

---

## Architecture Decisions — Locked

| Decision | Rationale |
|---|---|
| **SWMM 1D first, NOT HEC-RAS 2D** | HEC-RAS is Windows-native; adds friction before calibrated model exists. 2D is Step 9 (stretch goal, post-meeting). |
| **pyswmm** for SWMM execution | Mac-compatible, Python API, already in ecosystem |
| **3D surrogate input space** | Rain depth (mm), storm duration (hr), antecedent precip index API-5 (mm) |
| **PCA + XGBoost surrogate** | Proven in existing swmm-twin at `/Users/andrew/water/projects/swmm-twin/` |
| **uv** for package management | Already configured, use `uv run python ...` for all scripts |
| **5-minute USGS data** | Higher resolution than expected — use instantaneous not daily mean |
| **COA GIS via Socrata API** | Direct programmatic download, no manual clicking needed |

---

## Agile Steps & Status

| Step | Description | Status |
|---|---|---|
| 1 | USGS gauge data pull | ✅ DONE |
| 2 | COA stormwater GIS download | ✅ DONE |
| 3 | SWMM model skeleton (.inp file) | ✅ DONE |
| 4 | SWMM calibration | ✅ DONE — NSE=0.43 (QPE resolution ceiling; see Step 4 notes) |
| 5 | Monte Carlo dataset generation (500 runs) | 🔄 RUNNING — PID 71830, ~12:30pm CT, log: logs/mc_generation.log |
| 6 | Surrogate training (PCA + XGBoost) | 🟡 READY — model/train_surrogate.py written, run after Step 5 |
| 7 | Historical replay validation | ⏳ NOT STARTED |
| 8 | Streamlit forecast app v1 | ⏳ NOT STARTED |
| 9 | (Stretch) HEC-RAS 2D inundation | ⏳ POST-MEETING |

---

## Step 1 Results — USGS Gauge (DONE)

**Gauge:** 08158600 — Shoal Creek at W 12th St, Austin TX
**Cache:** `data/raw/usgs/usgs_08158600_00060.parquet`
**Events CSV:** `data/raw/usgs/usgs_08158600_events.csv`
**Record:** 2007-10-01 → present, 5-minute intervals, 1.89M records, 95.8% complete

Priority calibration events — all confirmed at 100% completeness:

| Event | Date | Peak CFS | Role |
|---|---|---|---|
| Halloween 2013 | 2013-10-31 | 10,400 | Calibration event #1 |
| Memorial Day 2015 | 2015-05-25 | 11,600 | Calibration event #2 (largest on record) |
| Halloween 2015 | 2015-10-30 | 8,590 | Holdout / validation |

330 total flood events identified. 160 clean events (≥700 CFS, <5% gap) available as backup pool.

**API note:** `dataretrieval` library — use `waterdata.get_continuous()` not deprecated `nwis.get_iv()`.
This is already updated in `data/usgs_pull.py`.

---

## Step 2 Results — COA GIS (DONE)

**Outputs:**
- `data/raw/gis/coa_open_channel_shl.geojson` — 819 active channel reaches
- `data/raw/gis/coa_drainage_pipe_shl.geojson` — 11,645 active storm sewer pipes
- `data/raw/gis/coa_gis_summary.json` — field coverage report

**Dataset IDs (Socrata, no auth required):**

| Layer | Socrata ID | Note |
|---|---|---|
| Open Channel | `5uu3-d6ij` | GeoJSON endpoint works directly |
| Drainage Pipe | `d5s6-axa4` | **NOT** `ybys-rwgs` (that's a map view, returns 400) |

**Open Channel (819 features) field coverage:**
- Node connectivity: `up_node_id` / `dn_node_id` — **100%** ✓
- Invert elevations: `up_invert` / `dn_invert` — **~44%** (derive rest from TNRIS LiDAR)
- Cross-section (`slope`, `bottom_width`, `ditch_shape`) — **~44–46%**
- Bbox check: ✓ within Shoal Creek bounds

**Drainage Pipe (11,645 features) field coverage:**
- Node connectivity: 100% ✓
- Dimensions (`width`, `height`, `struct_shape`) — **98%** ✓
- Invert elevations: **~37–39%** (derive from DEM)
- Material: **56%**

**VERDICT:** SWMM skeleton buildable from GIS data. Missing inverts → derive from TNRIS 1m LiDAR DEM (Step 3 dependency).

---

## Step 3 Results — SWMM Skeleton (DONE)

**Output:** `model/shoal_creek_skeleton.inp` (294 KB, SWMM 5.2.4-compatible)

| Metric | Value |
|---|---|
| Conduits (open channel reaches) | 819 |
| Junctions | 1,198 |
| Outfalls | 1 (node 373195, lat=30.2758 at W 12th St gauge) |
| Node elev from GIS inverts | 462 (39%) |
| Node elev from USGS 3DEP DEM | 737 (61%) |
| Fallback nodes | 0 |

**Scripts:**
- `data/fetch_dem_elevations.py` — queries USGS 3DEP EPQS for missing node elevations (cached to `node_elevations.json`)
- `model/build_swmm_inp.py` — GIS + DEM → SWMM .inp
- `model/validate_inp.py` — pyswmm load + 5-step routing validation

**pyswmm/uv CRITICAL NOTE:**
> `uv run python` triggers macOS Gatekeeper SIGKILL on `libswmm5.dylib` (codesign sandbox).
> Always use `.venv/bin/python` for any script that imports pyswmm.
> `uv run python` is fine for everything else (GIS, data pulls, etc.)

**pyswmm API (v2.x):** `sim.flow_routing_error` (not `sim.routing_error`)
**Routing:** DYNWAVE (full Saint-Venant), Manning n=0.035 (calibration target)
**Subcatchment:** single lumped 7,872-acre placeholder — parameterized in Step 4

---

## Step 4 Results — Calibration (IN PROGRESS — Option B Storage Unit)

**Output:** `model/best_params.json`, `model/calibrate.py`, `model/build_swmm_inp.py`

### Option B: Lumped Storage Unit + Weir Outlet (CURRENT)

The single-subcatchment topology had a hard NSE ceiling of ~0.46 because the simulated flow
collapsed after peak (no channel storage). Option B adds a lumped `S_SHOAL` storage unit.

**Model topology (as built):**
```
SC_SHOAL (subcatchment, 7,872 ac)
    ↓
S_SHOAL (SWMM STORAGE unit, FUNCTIONAL shape)
    ↓ W_OUTLET (TRANSVERSE weir, Q = 3.2 × w_len × h^1.5)
373195 (junction — USGS gauge measurement node)
    ↓ C373197 (25.7 ft GIS conduit)
373196 (OUTFALL, FREE boundary condition)
```

**Key design decisions:**
- `OUTFALL_ID = "373195"` — measure at the junction, NOT 373196. C373197 is a 25.7-ft GIS
  artifact with limited capacity; 373195 is where the weir delivers flow to the gauge.
- `S_SHOAL elevation = junction_elev` — weir controls outflow; no artificial head needed.
- `upstream_of_outfall` logic in `build_swmm_inp.py` finds the junction upstream of the
  true topological sink (373196) via conduit graph traversal.

**Calibration parameters (7 params, `model/calibrate.py`):**

| Parameter | Bounds | Physical role |
|---|---|---|
| `pct_imp` | 55–95% | Runoff volume |
| `width` | 2,000–50,000 ft | Concentration time (Tc) |
| `ksat` | 0.005–0.50 in/hr | Green-Ampt infiltration |
| `imd` | 0.10–0.45 | Initial moisture deficit |
| `pct_zero` | 0–60% | Impervious depression storage |
| `s_area` | 200K–15M ft² | Channel storage volume → recession duration |
| `w_len` | 10–150 ft | Weir crest length → peak Q and recession rate |

**Physics of recession duration:**
> Drain time from h=12.9ft (peak) to h=0.1ft:
> t = 5.77 × s_area / (3.2 × w_len) seconds
> For 48-hour recession at w_len=70: s_area ≈ 7.7M ft²
> For 24-hour recession at w_len=70: s_area ≈ 3.9M ft²

**Smoke test results (hand-tuned, not calibrated):**
- peak_r=0.92 (imp=85%, w=10K, ksat=0.05, s_area=200K, wlen=70)
- peak_r=1.05 (imp=80%, w=8K, ksat=0.08, s_area=300K, wlen=80)
- NSE still low (~0.13) before recession fix — large s_area needed

**Overnight run command (run from project root, ~3-4 hours at 13s/eval):**
```bash
nohup .venv/bin/python model/calibrate.py > logs/calibrate_de.log 2>&1 &
```
Or for LHS quick mode (~40 min, 200 samples):
```bash
nohup .venv/bin/python model/calibrate.py --quick --n-samples 200 > logs/calibrate_lhs.log 2>&1 &
```

**DE settings:** popsize=3, maxiter=40 → ~861 evaluations at 13s/run ≈ 3.1 hours

**Files modified in this session:**
- `model/build_swmm_inp.py` — added [STORAGE] S_SHOAL, [WEIRS] W_OUTLET, outfall selection fix
- `model/calibrate.py` — OUTFALL_ID=373195, PARAM_NAMES updated (n_ch→s_area, +w_len),
  patch_inp updated for [STORAGE] and [XSECTIONS] W_OUTLET, DE popsize/maxiter reduced
- `model/shoal_creek_*.inp` — all 3 storm INPs rebuilt with new topology

**NSE target:** ≥ 0.60 (calibration storm: Memorial Day 2015)

**pyswmm/uv CRITICAL NOTE (repeat):**
> `uv run python` triggers macOS Gatekeeper SIGKILL on `libswmm5.dylib`.
> Always use `.venv/bin/python` for any script that imports pyswmm.

---

Located at `/Users/andrew/water/projects/swmm-twin/`:
- `surrogates/base.py` — base class, benchmark metrics (NSE, RMSE, MAE)
- `generate_data.py` — Latin Hypercube sampling pattern for MC runs
- This is the template for Steps 5–6. Don't reinvent it — adapt it.

---

## Calibration Storm QPE (Still Needed)

For SWMM calibration we need rainfall input (QPE) alongside USGS streamflow.
Options:
- **NOAA GHCND** — nearest rain gauge to Shoal Creek watershed
- **NOAA Stage IV** — gridded QPE, better spatial coverage
- **LCRA HYDROMET** — local network, may have Shoal Creek area gauges

This is a dependency for Step 3/4 but not Step 2. Flag for when starting Step 3.

---

## Tech Stack

```
uv                  # package manager — always use 'uv run python ...'
pyswmm              # SWMM 1D Python API
swmm-toolkit        # installed
swmmio              # installed
dataretrieval       # USGS NWIS data (use waterdata.get_continuous())
geopandas           # for GIS processing
xgboost             # surrogate model
scikit-learn        # PCA
streamlit           # forecast app
```

---

## Key Files

| File | Purpose |
|---|---|
| `data/usgs_pull.py` | Step 1 — USGS gauge pull |
| `data/coa_gis_check.py` | Step 2 probe (stale URLs, superseded) |
| `data/raw/usgs/usgs_08158600_00060.parquet` | Cached gauge record |
| `data/raw/usgs/usgs_08158600_events.csv` | 330 flood events catalog |
| `data/raw/gis/coa_gis_report.json` | GIS probe results |
| `README.md` | Architecture overview and agile plan |
