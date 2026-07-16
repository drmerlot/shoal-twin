# shoal-twin

**Shoal Creek Flood Forecasting System**  
Austin, TX · USGS 08158600 · SWMM 1D physics + ML surrogate + forecast app

> Portfolio/demonstration project for the Geosyntec Austin meeting (Aug 2026).  
> Differentiation: hydrology domain knowledge + modern ML/surrogate infrastructure.

---

## Architecture

```
Weather forcing (rain_depth, duration, antecedent)
        │
        ▼
  SWMM 1D physics model  ←── calibrated to USGS 08158600
  (pyswmm, Shoal Creek)
        │
        ▼
  Monte Carlo ensemble (200+ runs, 3D LHS)
  Y: (N, nodes, T) — multi-node hydrographs
        │
        ▼
  ML Surrogate (PCAXGBoost, primary)
  NSE target > 0.65 on holdout
        │
        ▼
  Streamlit forecast app
  ├── Map: predicted flood depth on streets
  └── Gauge plot: forecast trace vs NWS flood stage thresholds
```

## Agile Steps

| # | What "done" means | Status |
|---|---|---|
| 1 | USGS gauge pull — identify 3+ calibration events | 🔄 |
| 2 | COA stormwater GIS — confirm buildable pipe network | ⬜ |
| 3 | SWMM model skeleton — runs without crashing | ⬜ |
| 4 | SWMM calibrated — NSE > 0.6 on 1 event | ⬜ |
| 5 | Monte Carlo dataset — 200+ runs, Y:(N,nodes,T) | ⬜ |
| 6 | Surrogate trained — PCAXGBoost, NSE reported | ⬜ |
| 7 | Historical replay — Memorial Day 2015 surrogate vs observed | ⬜ |
| 8 | App v1 — map + sliders → predicted hydrograph | ⬜ |
| 9 | (Stretch) 2D inundation via HEC-RAS | ⬜ |
| 10 | (Stretch) Live HRRR integration | ⬜ |

## Calibration Target

- **Gauge:** USGS 08158600 — Shoal Creek at W 12th St, Austin TX  
- **Metric:** NSE > 0.65 on ≥ 2 holdout storm events  
- **Priority events:**
  - Halloween 2013 (2013-10-30 to 2013-11-02)
  - Memorial Day 2015 (2015-05-24 to 2015-05-27) — ~5,000 CFS peak
  - Halloween 2015 (2015-10-30 to 2015-11-02)

## Setup

```bash
cd /Users/andrew/water/projects/shoal-twin

# Install dependencies
uv sync

# Step 1: USGS gauge data + event identification
uv run python data/usgs_pull.py

# Step 2: COA stormwater GIS availability
uv run python data/coa_gis_check.py
```

## Project Structure

```
shoal-twin/
├── data/
│   ├── usgs_pull.py          # Step 1: USGS gauge download + event ID
│   ├── coa_gis_check.py      # Step 2: COA ArcGIS REST probe
│   └── raw/
│       ├── usgs/             # Cached gauge parquet + events CSV
│       └── gis/              # GIS shapefiles / JSON reports
├── model/
│   └── shoal_creek.inp       # SWMM model (Steps 3-4)
├── surrogates/               # Surrogate base + implementations (from swmm-twin)
├── output/                   # Monte Carlo NPZ, trained models
├── app.py                    # Streamlit forecast app (Step 8)
├── generate_data.py          # Monte Carlo ensemble (Step 5)
├── train_surrogate.py        # Surrogate training (Step 6)
├── pyproject.toml
└── README.md
```

## Data Sources

| Source | Data | Access |
|---|---|---|
| USGS NWIS | Instantaneous discharge 08158600 | `dataretrieval` |
| City of Austin | Stormwater pipe network | ArcGIS REST / data.austintexas.gov |
| TNRIS | 1m LiDAR DEM | tnris.org (free) |
| NHD | Flowlines for channel burning | USGS NHD Plus |
| NOAA HRRR | QPF for real-time forecast | NOMADS (stretch) |

## Reference: swmm-twin

Surrogate framework patterns from `/Users/andrew/water/projects/swmm-twin`:
- `surrogates/base.py` — abstract `SurrogateModel`, `BenchmarkResult`, metric functions
- `generate_data.py` — Monte Carlo loop with LHS sampling
- `benchmark.py` — comparative benchmark across surrogate types

Key extensions in shoal-twin:
- 3D input space: `[rain_depth, duration, antecedent_API5]`
- Multi-node output: `Y: (N, nodes, T)` instead of single outfall
- Real calibration against observed gauge data (not toy example)
