"""
data/fetch_stage4_qpe.py
========================
Fetch NOAA Stage-IV hourly QPE for the Shoal Creek watershed from IEM's
JSON point-extraction API:
    https://mesonet.agron.iastate.edu/json/stage4.py

Samples 5 points across the watershed and returns area-weighted mean.
Watershed bbox: lat 30.25-30.40, lon -97.77--97.73 (Austin, TX)

Usage:
    uv run python data/fetch_stage4_qpe.py
"""

import json
import time
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUT_DIR = Path("data/raw/qpe")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUT_DIR / "stage4_cache"
CACHE_DIR.mkdir(exist_ok=True)

# 5 sample points over the Shoal Creek watershed (south to north)
WATERSHED_POINTS = {
    "gauge_12th": (30.2750, -97.7484),  # near USGS gauge, W 12th St
    "mid_south":  (30.2950, -97.7550),  # mid-watershed south
    "centroid":   (30.3200, -97.7500),  # geometric centroid
    "mid_north":  (30.3500, -97.7400),  # mid-watershed north
    "headwaters": (30.3800, -97.7300),  # headwaters NE of Mopac
}

# Area weights (triangular watershed, tapering toward headwaters)
AREA_WEIGHTS = {
    "gauge_12th": 0.10,
    "mid_south":  0.20,
    "centroid":   0.30,
    "mid_north":  0.25,
    "headwaters": 0.15,
}

STORMS = {
    "memorial_day_2015": {
        "dates": ["2015-05-24", "2015-05-25", "2015-05-26", "2015-05-27"],
    },
    "halloween_2013": {
        "dates": ["2013-10-29", "2013-10-30", "2013-10-31", "2013-11-01"],
    },
    "halloween_2015": {
        "dates": ["2015-10-29", "2015-10-30", "2015-10-31", "2015-11-01"],
    },
}

IEM_URL = "https://mesonet.agron.iastate.edu/json/stage4.py"


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def fetch_day(lat: float, lon: float, date: str) -> list:
    """Fetch hourly Stage-IV for one lat/lon/date."""
    cache_key = f"{date}_{lat:.4f}_{lon:.4f}.json"
    cache_path = CACHE_DIR / cache_key
    if cache_path.exists():
        return json.loads(cache_path.read_text())["data"]

    params = {"lat": lat, "lon": lon, "valid": date, "tz": "UTC"}
    for attempt in range(3):
        try:
            r = requests.get(IEM_URL, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            cache_path.write_text(json.dumps(data))
            return data["data"]
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"Stage-IV fetch failed {date} {lat},{lon}: {e}")


def fetch_storm_series(storm_id: str, dates: list) -> pd.DataFrame:
    """Fetch all points for a storm, return weighted mean hourly rainfall series."""
    all_rows = []
    for date in dates:
        point_data = {}
        for pt_name, (lat, lon) in WATERSHED_POINTS.items():
            rows = fetch_day(lat, lon, date)
            point_data[pt_name] = {r["end_valid"]: r["precip_in"] for r in rows}
            time.sleep(0.15)  # polite to IEM

        timestamps = sorted({t for rows in point_data.values() for t in rows.keys()})
        for ts in timestamps:
            weighted = sum(
                AREA_WEIGHTS[pt] * point_data[pt].get(ts, 0.0)
                for pt in WATERSHED_POINTS
            )
            all_rows.append({
                "storm_id": storm_id,
                "valid":    ts,
                "precip_in_per_hr": round(weighted, 4),
                "source":   "stage4_weighted",
            })

    df = pd.DataFrame(all_rows)
    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    return df.sort_values("valid").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_dfs = []
    for storm_id, cfg in STORMS.items():
        print(f"\n{'='*60}")
        print(f"Fetching Stage-IV QPE: {storm_id}")
        df = fetch_storm_series(storm_id, cfg["dates"])
        total = df["precip_in_per_hr"].sum()
        peak  = df["precip_in_per_hr"].max()
        peak_t = df.loc[df["precip_in_per_hr"].idxmax(), "valid"]
        print(f"  Total: {total:.2f} in   Peak: {peak:.2f} in/hr at {peak_t}")

        # Per-point stats for Memorial Day (most important)
        if storm_id == "memorial_day_2015":
            print("  Per-point peak intensities:")
            for pt_name, (lat, lon) in WATERSHED_POINTS.items():
                day_rows = []
                for date in cfg["dates"]:
                    day_rows.extend(fetch_day(lat, lon, date))
                pt_peak = max(r["precip_in"] for r in day_rows)
                print(f"    {pt_name:<12}: {pt_peak:.2f} in/hr")

        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = Path("data/raw/qpe/storm_rainfall_stage4.parquet")
    combined.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(combined)} records)")

    # Compare vs ASOS
    asos_path = Path("data/raw/qpe/storm_rainfall.parquet")
    if asos_path.exists():
        asos = pd.read_parquet(asos_path)
        print("\n=== QPE Comparison: ASOS(ATT airport) vs Stage-IV (watershed-avg) ===")
        for sid in STORMS:
            asos_att = asos[(asos["storm_id"] == sid) & (asos["station"] == "ATT")]
            s4_s     = combined[combined["storm_id"] == sid]
            print(f"\n  {sid}:")
            print(f"    ASOS(ATT)  total={asos_att['precip_in_per_hr'].sum():.2f}in  "
                  f"peak={asos_att['precip_in_per_hr'].max():.2f} in/hr")
            print(f"    Stage-IV   total={s4_s['precip_in_per_hr'].sum():.2f}in  "
                  f"peak={s4_s['precip_in_per_hr'].max():.2f} in/hr")


if __name__ == "__main__":
    main()
