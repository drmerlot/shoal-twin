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
| 2 | COA stormwater GIS download | 🔄 IN PROGRESS |
| 3 | SWMM model skeleton (.inp file) | ⏳ NOT STARTED |
| 4 | SWMM calibration (NSE > 0.6) | ⏳ NOT STARTED |
| 5 | Monte Carlo dataset generation (200+ runs) | ⏳ NOT STARTED |
| 6 | Surrogate training (PCA + XGBoost) | ⏳ NOT STARTED |
| 7 | Historical replay validation (Memorial Day 2015) | ⏳ NOT STARTED |
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

## Step 2 Status — COA GIS (IN PROGRESS)

**Original ArcGIS REST URLs were stale** — all failed. Found correct datasets via Socrata API.

### Confirmed Working Dataset IDs

| Layer | Socrata ID | Filter for Shoal Creek |
|---|---|---|
| **Open Channel** (main stem reaches) | `5uu3-d6ij` | `watershed='SHL'` |
| **Drainage Pipe** (storm sewers) | `ybys-rwgs` | `watershed='SHL'` |

### GeoJSON API endpoints (programmatic, no auth required):
```
https://data.austintexas.gov/resource/5uu3-d6ij.geojson?$where=watershed='SHL'&$limit=2000
https://data.austintexas.gov/resource/ybys-rwgs.geojson?$where=watershed='SHL'&$limit=2000
```

### Open Channel fields confirmed present (from live API sample):
- `up_node_id`, `dn_node_id` — **node connectivity is there**
- `up_invert`, `dn_invert` — survey elevations (some populated, some null → derive from DEM)
- `slope`, `bottom_width`, `top_width`, `depth` — cross-section geometry
- `ditch_shape` (TRAP, TRI, RECT, IRR) — maps to SWMM cross-section types
- `status` — filter to `ACTIVE` only
- `watershed` — `SHL` = Shoal Creek

### Next action for Step 2:
Write `data/coa_gis_download.py` to:
1. Pull both layers filtered to `watershed='SHL'` and `status='ACTIVE'`
2. Save as GeoJSON to `data/raw/gis/`
3. Print summary: feature count, % with invert elevations, bbox check

---

## Existing Surrogate Framework (Reference)

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
