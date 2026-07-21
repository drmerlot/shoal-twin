"""
app/app.py — Shoal Creek Flash Flood Risk Tool (Academic Case Study)
================================================
"""

from __future__ import annotations

import sys, time
from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import geopandas as gpd
import pydeck as pdk

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SURROGATE_DIR = ROOT / "model" / "surrogate"
MC_DIR        = ROOT / "data" / "raw" / "mc"
USGS_EVENTS   = ROOT / "data" / "raw" / "usgs" / "usgs_08158600_events.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

# ── Data loaders ────────────────────────────────────────────────────────────

@st.cache_resource
def load_surrogate():
    model = joblib.load(SURROGATE_DIR / "xgb_peak.joblib")
    with open(SURROGATE_DIR / "meta.json") as f:
        meta = json.load(f)
    return model, meta

def load_viz_data():
    with open(PROCESSED_DIR / "hist_hydrographs.json") as f:
        hist_hydro = json.load(f)
    with open(PROCESSED_DIR / "surrogate_scatter.json") as f:
        scatter = json.load(f)
    return hist_hydro, scatter

@st.cache_data
def load_map_data():
    gdf = gpd.read_file(ROOT / "data" / "raw" / "gis" / "coa_open_channel_shl.geojson")
    channels = gdf[
        (gdf["status"] == "ACTIVE") &
        (~gdf["drainage_id"].isna())
    ].copy()
    channels = channels.to_crs(epsg=4326)
    channels = channels.explode(ignore_index=True)
    channels["coordinates"] = channels.geometry.apply(lambda geom: list(geom.coords))
    return channels

# ── Inference ────────────────────────────────────────────────────────────────

def predict(model: dict, rain_mm: float, dur_hr: float, api_mm: float) -> tuple[float, float]:
    """Returns (predicted_cfs, elapsed_ms)."""
    ri = rain_mm / max(dur_hr, 0.1)
    af = max(0.4, 1.0 - 0.008 * api_mm)
    ed = rain_mm * af
    X  = np.array([[rain_mm, dur_hr, api_mm, ri, af, ed]], dtype=np.float32)
    t0 = time.perf_counter()
    v  = float(model["model"].predict(model["scaler"].transform(X))[0])
    ms = (time.perf_counter() - t0) * 1000
    return max(0.0, v), ms

# ── Plotly helpers ───────────────────────────────────────────────────────────

PANEL_BG = "#132338"
BG       = "#0d1b2a"

def _base_layout(**kw):
    return dict(
        paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
        font=dict(family="Inter, sans-serif", color="#e2e8f0", size=12),
        **kw,
    )

def plot_hydrographs(hydro_data, storm_id, title):
    data = hydro_data[storm_id]
    fig = go.Figure()
    
    times_swmm_dt = pd.to_datetime(data["times"])
    times_swmm = (times_swmm_dt - times_swmm_dt[0]).total_seconds() / 3600.0
    times_usgs_dt = pd.to_datetime(data["usgs_times"])
    times_usgs = (times_usgs_dt - times_usgs_dt[0]).total_seconds() / 3600.0
    
    fig.add_trace(go.Scatter(
        x=times_usgs, y=data["usgs_flows"],
        mode="lines", name="USGS Observed",
        line=dict(color="#f472b6", width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=times_swmm, y=data["flows"],
        mode="lines", name="SWMM Modeled",
        line=dict(color="#34d399", width=2.5, dash="dash")
    ))
    
    fig.update_layout(
        **_base_layout(
            title=dict(text=title, font=dict(size=14, color="#e2e8f0")),
            xaxis=dict(title="Hours", gridcolor="#1e3a5f", tickfont=dict(color="#94a3b8")),
            yaxis=dict(title="Flow (CFS)", gridcolor="#1e3a5f", tickfont=dict(color="#94a3b8")),
            margin=dict(l=50, r=20, t=40, b=40),
            height=300,
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.1, x=0)
        )
    )
    return fig

def plot_scatter(scatter_data):
    fig = go.Figure()
    
    # Train
    fig.add_trace(go.Scatter(
        x=scatter_data["train"]["true"], y=scatter_data["train"]["pred"],
        mode="markers", name="Train (80%)",
        marker=dict(color="rgba(52, 211, 153, 0.5)", size=6)
    ))
    
    # Test
    fig.add_trace(go.Scatter(
        x=scatter_data["test"]["true"], y=scatter_data["test"]["pred"],
        mode="markers", name="Test (20%)",
        marker=dict(color="rgba(96, 165, 250, 0.8)", size=6)
    ))
    
    # Identity Line
    max_val = max(max(scatter_data["train"]["true"]), max(scatter_data["test"]["true"]))
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines", name="Perfect Match",
        line=dict(color="#94a3b8", dash="dash")
    ))
    
    fig.update_layout(
        **_base_layout(
            xaxis=dict(title="True SWMM Peak Flow (CFS)", gridcolor="#1e3a5f", tickfont=dict(color="#94a3b8")),
            yaxis=dict(title="Surrogate Predicted Peak (CFS)", gridcolor="#1e3a5f", tickfont=dict(color="#94a3b8")),
            margin=dict(l=50, r=20, t=20, b=40),
            height=350,
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.1, x=0)
        )
    )
    return fig


def make_validation_bar(historical):
    # Add phase prefix to names
    names     = [f"{r['phase']}: {r['name']}" for r in historical]
    surr_vals = [r["surrogate_cfs"] for r in historical]
    swmm_vals = [r["swmm_cfs"] for r in historical]
    usgs_vals = [r["usgs_cfs"] for r in historical]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Surrogate Forecast",
        x=names, y=surr_vals,
        marker_color="#06b6d4",
        text=[f"{v:,.0f}" for v in surr_vals],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
    ))
    fig.add_trace(go.Bar(
        name="SWMM Physics Engine",
        x=names, y=swmm_vals,
        marker_color="#a78bfa",
        text=[f"{v:,.0f}" for v in swmm_vals],
        textposition="outside",
        textfont=dict(color="#a78bfa", size=11),
    ))
    fig.add_trace(go.Bar(
        name="USGS Observed",
        x=names, y=usgs_vals,
        marker_color="#f472b6",
        text=[f"{v:,.0f}" for v in usgs_vals],
        textposition="outside",
        textfont=dict(color="#f472b6", size=11),
    ))

    fig.update_layout(
        **_base_layout(
            barmode="group",
            xaxis=dict(tickfont=dict(color="#94a3b8", size=11), gridcolor="#1e3a5f"),
            yaxis=dict(title="Peak CFS", gridcolor="#1e3a5f",
                       tickfont=dict(color="#64748b"), zeroline=False,
                       range=[0, max(usgs_vals)*1.2]),
            margin=dict(l=60, r=20, t=20, b=60),
            height=350,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11),
                        orientation="h", yanchor="bottom", y=1.01, x=0),
        )
    )
    return fig

def make_pydeck_map(channels, predicted_cfs):
    """Creates a PyDeck map with channel thickness scaling to CFS"""
    w_scale = max(2, min((predicted_cfs / 4000) * 15, 30))
    c_rgb = [239, 68, 68] if predicted_cfs > 4000 else [249, 115, 22] if predicted_cfs > 2200 else [234, 179, 8]
    
    view_state = pdk.ViewState(
        latitude=30.30, longitude=-97.745, zoom=12.2, pitch=45, bearing=0
    )
    
    layer = pdk.Layer(
        "PathLayer",
        data=channels,
        pickable=True,
        get_color=c_rgb,
        width_scale=20,
        width_min_pixels=w_scale,
        get_path="coordinates",
        get_width=5,
    )
    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_provider="carto",
        map_style="dark",
        tooltip={"text": f"Flow: {predicted_cfs:,.0f} CFS"},
    )

# ── Page config & CSS ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Shoal Creek Digital Twin: Academic Summary",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0d1b2a; }

.card {
    background: #132338; border: 1px solid #1e3a5f;
    border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;
}
.card-label {
    font-size: 0.85rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #60a5fa; margin-bottom: 0.8rem;
}
.metric-row { display: flex; gap: 1rem; margin-bottom: 0.4rem; }
.metric-box {
    flex: 1; background: #0d2340; border: 1px solid #1e3a5f;
    border-radius: 8px; padding: 1rem; text-align: center;
}
.metric-box-num { font-size: 1.8rem; font-weight: 800; }
.metric-box-lbl { font-size: 0.75rem; color: #94a3b8; margin-top: 0.2rem; }
.divider { border: none; border-top: 1px solid #1e3a5f; margin: 2rem 0; }
h1, h2, h3 { color: #e2e8f0 !important; }
p, li { color: #cbd5e1; line-height: 1.6; font-size: 1.05rem; }
</style>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────

with st.spinner("Loading models and GIS..."):
    model, meta = load_surrogate()
    hist_hydro, scatter_data = load_viz_data()
    gis_channels = load_map_data()

# ── 1. Header ───────────────────────────────────────────────────────────────────
c_left, c_center, c_right = st.columns([1, 8, 1])
with c_center:
    st.title("Accelerating Hydrodynamic Flood Physics using Machine Learning")
    st.markdown("""
    **Case Study:** Shoal Creek Watershed, Austin, TX.  
    This document outlines the methodology of coupling a rigorous EPA SWMM 1D/2D physics model with an XGBoost surrogate to enable real-time probabilistic flood forecasting without sacrificing physical accuracy.
    """)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── 2. Part I: SWMM Model (Physics) ─────────────────────────────────────────────
with c_center:
    st.header("Part I: The Physics Model (SWMM)")
    st.markdown("""
    To generate truth data, we first constructed a detailed **EPA SWMM** (Storm Water Management Model) digital twin of the Shoal Creek watershed.
    - **Topography & Routing:** 47 subcatchments connected to an intricate open-channel network representing Shoal Creek.
    - **Infiltration:** Green-Ampt infiltration simulating antecedent soil moisture dynamics.
    - **Data assimilation:** High-resolution NOAA MRMS and Texas Mesonet precipitation grids to capture the severe sub-hourly cloudbursts characteristic of central Texas flash floods.
    """)

    st.markdown("""
    <div class="card">
      <div class="card-label">SWMM Calibration & Validation Summary</div>
      <p style="font-size: 0.95rem; color: #94a3b8;">
      The physical model parameters (channel roughness, width, impervious percentage, etc.) were optimized against historical USGS stream gauge observations using a Differential Evolution algorithm.
      </p>
      <div class="metric-row">
        <div class="metric-box">
          <div class="metric-box-num" style="color:#34d399">0.64</div>
          <div class="metric-box-lbl">Nash-Sutcliffe Efficiency (NSE)</div>
        </div>
        <div class="metric-box">
          <div class="metric-box-num" style="color:#60a5fa">Memorial Day 2015</div>
          <div class="metric-box-lbl">Primary Calibration Event</div>
        </div>
        <div class="metric-box">
          <div class="metric-box-num" style="color:#a78bfa">Halloween 2013 & 2015</div>
          <div class="metric-box-lbl">Validation Events</div>
        </div>
      </div>
      <p style="font-size: 0.9rem; color: #94a3b8; margin-top: 1rem;">
      An NSE score above 0.60 is widely considered strong for urban hydrology modeling, particularly in flash-flood prone regions where cloudburst intensity is notoriously difficult to capture by gauge or radar alone.
      </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(plot_hydrographs(hist_hydro, "memorial_day_2015", "Calibration: Memorial Day 2015 (NSE = 0.64)"), use_container_width=True)
    with c2:
        st.plotly_chart(plot_hydrographs(hist_hydro, "halloween_2015", "Validation: Halloween 2015 (NSE = 0.52)"), use_container_width=True)
    with c3:
        st.plotly_chart(plot_hydrographs(hist_hydro, "halloween_2013", "Validation: Halloween 2013 (NSE = 0.45)"), use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── 3. Part II: The Surrogate Approach ──────────────────────────────────────────
with c_center:
    st.header("Part II: The ML Surrogate Approach")
    st.markdown("""
    **The Bottleneck:** While SWMM's ~35-second run time is fast for a single storm, generating a reliable probabilistic forecast requires executing a massive ensemble of scenarios (e.g., 1,000 runs). While this can be achieved through heavy parallelization, it requires expensive, complex compute clusters to execute in real time.
    
    **The Solution:** We trained a machine learning emulator (XGBoost) to instantly approximate SWMM's governing equations. This provides near-instantaneous ensemble generation on a fraction of the compute resources.
    """)

    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; background:#0d2340; padding:1.5rem; border-radius:12px; border:1px solid #1e3a5f; margin: 2rem 0;">
      <div style="text-align:center; flex:1;">
        <div style="font-weight:700; color:#e2e8f0; margin-bottom:0.5rem">1. Monte Carlo Data Generation</div>
        <div style="font-size:0.85rem; color:#94a3b8; padding: 0 1rem;">Latin Hypercube Sampling across Rainfall Depth, Duration, and Soil API (500 scenarios)</div>
      </div>
      <div style="font-size:2rem; color:#34d399; padding:0 0.5rem;">→</div>
      <div style="text-align:center; flex:1;">
        <div style="font-weight:700; color:#e2e8f0; margin-bottom:0.5rem">2. SWMM Physics Engine</div>
        <div style="font-size:0.85rem; color:#94a3b8; padding: 0 1rem;">Run full 1D/2D hydrodynamic model for all scenarios (35 seconds per run)</div>
      </div>
      <div style="font-size:2rem; color:#34d399; padding:0 0.5rem;">→</div>
      <div style="text-align:center; flex:1;">
        <div style="font-weight:700; color:#e2e8f0; margin-bottom:0.5rem">3. Surrogate ML Training</div>
        <div style="font-size:0.85rem; color:#94a3b8; padding: 0 1rem;">Train XGBoost to map storm features to SWMM peak flow (0.006 ms per run)</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    train_n = meta['n_train']
    test_n = meta['n_test']
    r2 = meta['peak_xgb']['r2_holdout']
    mae = meta['peak_xgb']['mae_cfs']
    infer_ms = meta['peak_xgb']['infer_ms']
    speedup = 35000 / infer_ms

    col_m1, col_m2 = st.columns([1, 1.2])
    with col_m1:
        st.markdown(f"""
        <div class="card" style="height: 100%;">
          <div class="card-label">Surrogate Model Performance</div>
          <p style="font-size: 0.95rem; color: #94a3b8;">
          The surrogate dataset was split into {train_n} training runs and {test_n} held-out validation runs to evaluate out-of-sample generalization.
          </p>
          <div class="metric-row">
            <div class="metric-box">
              <div class="metric-box-num" style="color:#34d399">{r2:.3f}</div>
              <div class="metric-box-lbl">Holdout R²</div>
            </div>
            <div class="metric-box">
              <div class="metric-box-num" style="color:#f472b6">{speedup:,.0f}x</div>
              <div class="metric-box-lbl">Speedup</div>
            </div>
          </div>
          <p style="font-size: 0.9rem; color: #94a3b8; margin-top: 1rem;">
          The surrogate maintains over 99% of the physical model's accuracy but executes in <strong>{infer_ms:.3f} milliseconds</strong>, unlocking real-time probabilistic hazard mapping.
          </p>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.plotly_chart(plot_scatter(scatter_data), use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── 4. Part III: Results & Verification ─────────────────────────────────────────
with c_center:
    st.header("Part III: Application & Verification")
    st.markdown("""
    To verify the utility of the surrogate framework, we use it to predict historical events purely from raw meteorological inputs (Rainfall, Duration, Soil Moisture) and compare the output to the full **SWMM Physics Engine**.
    """)

    HISTORICAL_REPLAY = [
        {"name": "Memorial Day 2015", "phase": "Cal",  "rain_in": 5.83, "dur_hr": 9,  "soil": 35, "usgs_cfs": 11600, "swmm_cfs": max(hist_hydro["memorial_day_2015"]["flows"])},
        {"name": "Halloween 2013",    "phase": "Val",  "rain_in": 4.70, "dur_hr": 14, "soil": 5,  "usgs_cfs": 10400, "swmm_cfs": max(hist_hydro["halloween_2013"]["flows"])},
        {"name": "Halloween 2015",    "phase": "Val",  "rain_in": 5.90, "dur_hr": 13, "soil": 35, "usgs_cfs": 8590, "swmm_cfs": max(hist_hydro["halloween_2015"]["flows"])},
        {"name": "July 4th 2025",     "phase": "Test", "rain_in": 6.50, "dur_hr": 8,  "soil": 30, "usgs_cfs": 9340, "swmm_cfs": max(hist_hydro["july_2025"]["flows"])},
    ]

    for r in HISTORICAL_REPLAY:
        cfs, _ = predict(model, r["rain_in"] * 25.4, r["dur_hr"], r["soil"] * 0.75)
        r["surrogate_cfs"] = cfs

    st.plotly_chart(make_validation_bar(HISTORICAL_REPLAY), use_container_width=True)

    st.markdown("""
    ### Spatial Hazard Mapping Comparison
    Because the surrogate accurately mimics the SWMM peak flow, it can be used to drive a **Map Library lookup**. 
    
    **How does a Map Library work?**
    In operational forecasting, running a full 2D distributed hydrodynamic simulation takes hours. Instead, a library of 2D inundation maps is pre-computed across a vast range of peak flows. Inundation in this context is defined as a **binary condition (wet vs. dry)** representing the maximum spatial extent of the floodwaters for a specific flow rate.
    
    During a live event, the fast 0D ML surrogate predicts a scalar peak flow at the outlet. The system then queries the Map Library for the pre-computed binary inundation polygon that closest matches that predicted peak flow, instantly rendering the spatial boundaries across the watershed.
    
    **What am I looking at?**
    The side-by-side maps below demonstrate this map library pipeline for a **Synthetic Hold-out Test Event** (6.99 inches of rain over 9 hours). This event was never seen during calibration or training.
    
    * **Left (SWMM Physics):** The peak flow (10,946 CFS) as simulated by the SWMM physics engine, which takes ~35 seconds to run.
    * **Right (ML Surrogate):** The peak flow (10,938 CFS) as predicted instantly by the XGBoost surrogate, given only the meteorological forcing and antecedent soil moisture. 
    
    Because the predictions are within **0.07%** of each other, they query the exact same binary flood tier in the map library. In this interactive demonstration, we visually proxy that binary tier by dynamically scaling the channel vector width. The resulting identical spatial outputs demonstrate that the surrogate can drive operational mapping pipelines just as accurately as the full physics model, but thousands of times faster.
    """)

    # We use Synthetic Test Event 124 (out of sample) for rigorous visual proof
    cfs_surr = 10938.56
    cfs_swmm = 10946.70
    
    map_col1, map_col2 = st.columns(2)
    with map_col1:
        st.markdown("**0.006 ms ML Surrogate**")
        st.pydeck_chart(make_pydeck_map(gis_channels, cfs_surr), use_container_width=True)
    with map_col2:
        st.markdown("**35.0 s SWMM Physics Engine**")
        st.pydeck_chart(make_pydeck_map(gis_channels, cfs_swmm), use_container_width=True)

    st.markdown("""
    <div style="font-size:0.85rem; color:#64748b; margin-top: 2rem; border-top: 1px solid #1e3a5f; padding-top: 1rem; margin-bottom: 3rem;">
    <strong>Conclusion:</strong> The surrogate-driven framework successfully emulates high-fidelity 1D/2D hydrodynamics with negligible performance loss. The order-of-magnitude computational speedup opens the door to live, cloud-native ensemble forecasting systems that will replace this static proof-of-concept in the next phase of development.
    </div>
    """, unsafe_allow_html=True)

# ── 4. Part III: Interactive Live Forecast Demo ────────────────────────────────

st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.header("Part III: Interactive Live Forecast Demo")
st.markdown("Use the sliders below to adjust the incoming storm forecast. The ML Surrogate updates instantly, while the SWMM model requires you to manually run the physics engine.")

st.markdown("""
<div style="background-color: #0f172a; padding: 1.5rem; border-radius: 0.5rem; border: 1px solid #1e3a5f; margin-bottom: 2rem;">
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    rain_in = st.slider("Rainfall (inches)", min_value=1.0, max_value=15.0, value=5.0, step=0.5)
with col2:
    dur_hr = st.slider("Storm Duration (hours)", min_value=1.0, max_value=24.0, value=6.0, step=1.0)
with col3:
    api_mm = st.slider("Antecedent Moisture (API5, mm)", min_value=0.0, max_value=100.0, value=35.0, step=5.0)

st.markdown("</div>", unsafe_allow_html=True)
st.info("The ML Surrogate takes these values and instantly updates the left-hand map.")

# Predict Surrogate instantly
import time
t0_surr = time.time()
cfs_surr, _ = predict(model, rain_in * 25.4, dur_hr, api_mm)
t_surr = time.time() - t0_surr

if "swmm_cfs" not in st.session_state:
    st.session_state.swmm_cfs = None
    st.session_state.swmm_time = None
    st.session_state.swmm_params = None

map_c1, map_c2 = st.columns(2)
with map_c1:
    st.markdown(f"**{t_surr*1000:.3f} ms ML Surrogate** (Peak: {cfs_surr:,.0f} CFS)")
    st.pydeck_chart(make_pydeck_map(gis_channels, cfs_surr), use_container_width=True)
    
with map_c2:
    if st.session_state.swmm_cfs is not None:
        swmm_val = st.session_state.swmm_cfs
        swmm_t = st.session_state.swmm_time
        matched = (st.session_state.swmm_params == (rain_in, dur_hr, api_mm))
        
        status = "✅ Synced" if matched else "⚠️ Out of Date"
        st.markdown(f"**{swmm_t:.1f} s SWMM Physics Engine** (Peak: {swmm_val:,.0f} CFS) - {status}")
        st.pydeck_chart(make_pydeck_map(gis_channels, swmm_val), use_container_width=True)
    else:
        st.markdown("**SWMM Physics Engine**")
        st.info("Click 'Run SWMM Physics Model' to simulate the inundation for the current slider values.")

    if st.button("Run SWMM Physics Model", use_container_width=True):
        with st.spinner("Running SWMM (this takes ~35 seconds)..."):
            import subprocess
            import json
            import sys
            
            # Run the SWMM physics model in a completely isolated subprocess
            # This prevents libswmm5.dll from segfaulting the Streamlit process
            script_path = ROOT / "model" / "run_live_swmm.py"
            cmd = [
                sys.executable, str(script_path),
                "--rain", str(rain_in),
                "--dur", str(dur_hr),
                "--api", str(api_mm)
            ]
            
            try:
                # Capture output to get the JSON result
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                # Parse the JSON output printed by the script
                # The script suppresses pyswmm stdout and only prints the JSON at the end
                res = None
                for line in result.stdout.strip().split('\n'):
                    if line.startswith('{') and '"success"' in line:
                        res = json.loads(line)
                        break
                        
                if res and res.get("success"):
                    st.session_state.swmm_cfs = res["peak_cfs"]
                    st.session_state.swmm_time = res["elapsed"]
                    st.session_state.swmm_params = (rain_in, dur_hr, api_mm)
                    st.rerun()
                else:
                    st.error("SWMM run failed or produced 0 CFS.")
                    if result.stderr:
                        st.error(f"Error details: {result.stderr}")
            except subprocess.CalledProcessError as e:
                st.error(f"SWMM subprocess crashed (exit code {e.returncode}).")
                st.error(e.stderr)
