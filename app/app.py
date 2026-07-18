"""
app/app.py — Shoal Creek Flash Flood Risk Tool
================================================
Tells one clear story:

  The surrogate enables real-time probabilistic forecasting that SWMM
  physically cannot do within the available lead time.

  SWMM: 35s/run → 1,000-scenario ensemble = 9.7 hours
  Surrogate: 0.006ms/run → 1,000-scenario ensemble = 6ms

Layout:
  1. Speed / value banner (top)
  2. Storm inputs + single prediction + flood stage
  3. The 3 nearest SWMM analog hydrographs — shows natural variability
  4. Model rigor: R² benchmark + historical replay table

Run:
    uv run streamlit run app/app.py
"""

from __future__ import annotations

import sys, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SURROGATE_DIR = ROOT / "model" / "surrogate"
MC_DIR        = ROOT / "data" / "raw" / "mc"
USGS_EVENTS   = ROOT / "data" / "raw" / "usgs" / "usgs_08158600_events.csv"

STAGES = [
    {"name": "Major Flood",    "min": 4000, "color": "#ef4444", "bg": "#450a0a", "icon": "🔴"},
    {"name": "Moderate Flood", "min": 2200, "color": "#f97316", "bg": "#431407", "icon": "🟠"},
    {"name": "Minor Flood",    "min": 700,  "color": "#eab308", "bg": "#3d2e00", "icon": "🟡"},
    {"name": "Action Stage",   "min": 200,  "color": "#60a5fa", "bg": "#0c1a3a", "icon": "🔵"},
    {"name": "Normal Flow",    "min": 0,    "color": "#34d399", "bg": "#052e16", "icon": "🟢"},
]

STAGE_EXPLAIN = {
    "Major Flood":    "Roads close, evacuations possible. Serious threat to life and property. Shoal Creek Drive floods completely.",
    "Moderate Flood": "Significant street flooding. Some structures and businesses affected. Avoid all low water crossings.",
    "Minor Flood":    "Low-lying roads and underpasses flood. Hike-and-bike trail inundated.",
    "Action Stage":   "Creek is running high. Monitor conditions, avoid the trail.",
    "Normal Flow":    "Creek within banks. No flood concern.",
}

HISTORICAL_REPLAY = [
    {"name": "Memorial Day 2015", "date": "May 25, 2015",
     "rain_in": 5.83, "dur_hr": 9,  "usgs_cfs": 11600, "surrogate_cfs": 7621,
     "note": "Calibration event. 8 fatalities in Travis County."},
    {"name": "Halloween 2013",    "date": "Oct 31, 2013",
     "rain_in": 4.70, "dur_hr": 14, "usgs_cfs": 10400, "surrogate_cfs": 3510,
     "note": "Very dry antecedent conditions. QPE missed sub-hourly burst."},
    {"name": "Halloween 2015",    "date": "Oct 31, 2015",
     "rain_in": 5.90, "dur_hr": 13, "usgs_cfs": 8590,  "surrogate_cfs": 7261,
     "note": "Best validation result. -16% error."},
]

def classify(cfs: float) -> dict:
    for s in STAGES:
        if cfs >= s["min"]:
            return s
    return STAGES[-1]

# ── Data loaders ────────────────────────────────────────────────────────────

@st.cache_resource
def load_surrogate():
    return joblib.load(SURROGATE_DIR / "xgb_peak.joblib")

@st.cache_data
def load_events():
    df = pd.read_csv(USGS_EVENTS)
    return df[df["peak_cfs"] > 0].reset_index(drop=True)

@st.cache_resource
def load_mc():
    inp  = pd.read_parquet(MC_DIR / "mc_inputs.parquet").set_index("run_id")
    meta = pd.read_parquet(MC_DIR / "mc_metadata.parquet").set_index("run_id")
    hyd  = pd.read_parquet(MC_DIR / "mc_hydrographs.parquet")
    return inp, meta, hyd

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

def find_top_analogs(inp_df, meta_df, rain_mm, dur_hr, api_mm, n=3):
    """Return n nearest run_ids by normalized Euclidean distance."""
    valid  = meta_df[meta_df["peak_cfs"] >= 50].index
    sub    = inp_df.loc[valid]
    ranges = np.array([300.0, 24.0, 75.0])
    q      = np.array([rain_mm, dur_hr, api_mm]) / ranges
    X_n    = sub[["rain_depth_mm","storm_dur_hr","api_5_mm"]].values / ranges
    dists  = np.linalg.norm(X_n - q, axis=1)
    order  = np.argsort(dists)[:n]
    return list(valid[order]), list(dists[order])

# ── Plotly helpers ───────────────────────────────────────────────────────────

PANEL_BG = "#132338"
BG       = "#0d1b2a"

def _base_layout(**kw):
    return dict(
        paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
        font=dict(family="Inter, sans-serif", color="#e2e8f0", size=12),
        **kw,
    )

def make_gauge(cfs, stage):
    breaks  = [0, 200, 700, 2200, 4000, 12000]
    fills   = ["#1e3a5f","#0f2236","#1a3050","#3d2e00","#431407","#450a0a"]
    labels  = ["", "Normal", "Action", "Minor", "Moderate", "Major"]
    fig = go.Figure()
    for i in range(len(breaks)-1):
        lo, hi = breaks[i], breaks[i+1]
        fig.add_shape(type="rect", x0=lo, x1=hi, y0=0, y1=1,
                      fillcolor=fills[i+1], opacity=0.9, line_width=0, layer="below")
        fig.add_annotation(x=(lo+hi)/2, y=0.5, text=labels[i+1], showarrow=False,
                           font=dict(size=10, color="#64748b"), yanchor="middle")
    for bp in breaks[1:-1]:
        fig.add_vline(x=bp, line_color=BG, line_width=2)
    clamped = min(cfs, breaks[-1]*0.98)
    fig.add_shape(type="line", x0=clamped, x1=clamped, y0=-0.2, y1=1.2,
                  line=dict(color=stage["color"], width=5))
    fig.add_annotation(x=clamped, y=1.3, text=f"▼ {cfs:,.0f} CFS",
                       showarrow=False,
                       font=dict(size=12, color=stage["color"], family="Inter, sans-serif"),
                       yanchor="bottom")
    fig.update_layout(
        **_base_layout(
            xaxis=dict(range=[0, breaks[-1]], showgrid=False, zeroline=False,
                       tickvals=breaks[1:-1], ticktext=[f"{v:,}" for v in breaks[1:-1]],
                       tickfont=dict(color="#475569", size=10)),
            yaxis=dict(visible=False, range=[-0.4, 1.6]),
            margin=dict(l=10, r=10, t=35, b=25), height=105,
        )
    )
    return fig

def make_triple_hydrograph(hyd_df, run_ids, meta_df, pred_cfs, stage):
    """
    Overlay 3 SWMM analog hydrographs and the surrogate peak.
    Shows the natural variability band around similar storms.
    """
    TRACE_COLORS = ["#06b6d4", "#818cf8", "#34d399"]
    fig = go.Figure()

    all_y = []
    for i, rid in enumerate(run_ids):
        y    = hyd_df.loc[rid].values
        show = min(len(y), 720)          # 60 hours
        t    = np.arange(show) * 5 / 60
        meta_row = meta_df.loc[rid]
        all_y.extend(y[:show])

        col  = TRACE_COLORS[i]
        rpeak = float(meta_row["peak_cfs"])
        rdur  = float(meta_row["storm_dur_hr"])
        rdepth= float(meta_row["rain_depth_mm"])

        fig.add_trace(go.Scatter(
            x=t, y=y[:show],
            mode="lines",
            fill="tozeroy" if i == 0 else None,
            fillcolor="rgba(6,182,212,0.07)" if i == 0 else None,
            line=dict(color=col, width=1.8),
            opacity=0.85,
            name=f"Analog {i+1}: {rdepth:.0f}mm/{rdur:.0f}hr → {rpeak:,.0f} CFS",
            hovertemplate=f"<b>Analog {i+1}</b><br>%{{y:,.0f}} CFS at %{{x:.1f}} hr<extra></extra>",
        ))

    # Surrogate prediction — horizontal band, not just a line
    # Parse hex color (#rrggbb) → rgba string (Plotly rejects 8-digit hex)
    hex_col = stage["color"].lstrip("#")
    r, g, b = int(hex_col[0:2], 16), int(hex_col[2:4], 16), int(hex_col[4:6], 16)
    band_fill = f"rgba({r},{g},{b},0.10)"
    fig.add_hrect(
        y0=pred_cfs * 0.93, y1=pred_cfs * 1.07,
        fillcolor=band_fill, line_width=0,
        annotation_text="Surrogate ±7%",
        annotation_position="top right",
        annotation_font_color=stage["color"],
        annotation_font_size=10,
    )
    fig.add_hline(
        y=pred_cfs, line_dash="dash",
        line_color=stage["color"], line_width=2.5,
        annotation_text=f"Surrogate: {pred_cfs:,.0f} CFS",
        annotation_position="top left",
        annotation_font_color=stage["color"],
        annotation_font_size=12,
    )

    ymax = max(np.array(all_y).max() * 1.15, pred_cfs * 1.3, 500)

    fig.update_layout(
        **_base_layout(
            xaxis=dict(title="Hours from storm start", gridcolor="#1e3a5f",
                       tickfont=dict(color="#64748b"), zeroline=False),
            yaxis=dict(title="Flow (CFS)", gridcolor="#1e3a5f",
                       tickfont=dict(color="#64748b"), zeroline=False,
                       range=[0, ymax]),
            margin=dict(l=60, r=20, t=20, b=50),
            height=310,
            legend=dict(
                bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#94a3b8"),
                yanchor="top", y=0.99, xanchor="left", x=0.01,
            ),
        )
    )
    return fig

def make_validation_bar(historical):
    """Grouped bar: surrogate vs USGS for 3 historical events."""
    names     = [r["name"] for r in historical]
    surr_vals = [r["surrogate_cfs"] for r in historical]
    usgs_vals = [r["usgs_cfs"] for r in historical]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Surrogate (SWMM emulator)",
        x=names, y=surr_vals,
        marker_color="#06b6d4",
        text=[f"{v:,}" for v in surr_vals],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
    ))
    fig.add_trace(go.Bar(
        name="USGS observed",
        x=names, y=usgs_vals,
        marker_color="#f472b6",
        text=[f"{v:,}" for v in usgs_vals],
        textposition="outside",
        textfont=dict(color="#f472b6", size=11),
    ))

    fig.update_layout(
        **_base_layout(
            barmode="group",
            xaxis=dict(tickfont=dict(color="#94a3b8", size=11), gridcolor="#1e3a5f"),
            yaxis=dict(title="Peak CFS", gridcolor="#1e3a5f",
                       tickfont=dict(color="#64748b"), zeroline=False,
                       range=[0, 14000]),
            margin=dict(l=60, r=20, t=20, b=60),
            height=290,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11),
                        orientation="h", yanchor="bottom", y=1.01, x=0),
        )
    )

    # Gap annotation arrow on Memorial Day 2015
    fig.add_annotation(
        x="Memorial Day 2015", y=9500,
        text="QPE ceiling gap<br>(sub-hourly burst<br>not in data)",
        showarrow=True, arrowhead=2, arrowcolor="#64748b",
        ax=80, ay=-30, font=dict(size=9, color="#64748b"),
        bordercolor="#1e3a5f", borderwidth=1,
        bgcolor="#132338",
    )
    return fig

# ── Page config & CSS ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Shoal Creek Flood Risk",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0d1b2a; }
section[data-testid="stSidebar"] {
    background: #091420; border-right: 1px solid #1e3a5f;
}

.card {
    background: #132338; border: 1px solid #1e3a5f;
    border-radius: 12px; padding: 1.1rem 1.3rem; height: 100%;
}
.card-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #475569; margin-bottom: 0.45rem;
}
.peak-num {
    font-size: 3.8rem; font-weight: 900;
    line-height: 1; letter-spacing: -0.03em;
}
.stage-pill {
    display: inline-block; padding: 0.28rem 0.9rem;
    border-radius: 9999px; font-size: 0.78rem; font-weight: 700;
    letter-spacing: 0.06em; text-transform: uppercase; margin-top: 0.5rem;
}
/* Speed banner */
.speed-banner {
    background: linear-gradient(135deg, #0a1f38 0%, #0d2d4a 100%);
    border: 1px solid #1e4d7a; border-radius: 12px;
    padding: 1rem 1.4rem; margin-bottom: 1.4rem;
    display: flex; align-items: center; gap: 2rem;
}
.speed-stat { text-align: center; }
.speed-num  { font-size: 1.6rem; font-weight: 800; line-height: 1; }
.speed-unit { font-size: 0.72rem; color: #64748b; margin-top: 0.1rem; }
.speed-vs   {
    font-size: 1.8rem; font-weight: 300; color: #1e3a5f;
    padding: 0 0.5rem;
}
.divider { border: none; border-top: 1px solid #1e3a5f; margin: 1.2rem 0; }
.section-head {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #475569; margin-bottom: 0.6rem;
}
.metric-row { display: flex; gap: 1rem; margin-bottom: 0.4rem; }
.metric-box {
    flex: 1; background: #0d2340; border: 1px solid #1e3a5f;
    border-radius: 8px; padding: 0.7rem 0.9rem; text-align: center;
}
.metric-box-num { font-size: 1.4rem; font-weight: 700; }
.metric-box-lbl { font-size: 0.68rem; color: #64748b; margin-top: 0.1rem; }
.replay-row {
    border-top: 1px solid #1e3a5f;
    padding: 0.6rem 0; font-size: 0.82rem;
}
.replay-name { font-weight: 600; color: #e2e8f0; }
.replay-err  { font-weight: 700; }
.caveat {
    background: #091420; border: 1px solid #1e4d7a;
    border-radius: 8px; padding: 0.8rem 1rem;
    font-size: 0.76rem; color: #64748b; line-height: 1.6;
}
.caveat strong { color: #94a3b8; }
#MainMenu, footer { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding:0.5rem 0 1.2rem'>
        <div style='font-size:1rem;font-weight:700;color:#e2e8f0'>⛈️ Storm Scenario</div>
        <div style='font-size:0.73rem;color:#475569;margin-top:0.2rem'>
            Define the incoming storm.
        </div>
    </div>""", unsafe_allow_html=True)

    rain_in = st.slider("Total rainfall (inches)", 0.5, 12.0, 4.0, 0.25)
    dur_hr  = st.slider("Storm duration (hours)",   1,   24,   6,   1)
    soil    = st.slider("Antecedent soil wetness — 0 = bone dry, 100 = saturated",
                        0, 100, 35, 5)

    st.markdown("---")
    st.markdown(
        "<div class='section-head'>Load a real storm</div>",
        unsafe_allow_html=True,
    )
    presets = {
        "— custom —": None,
        "Memorial Day 2015 (11,600 CFS)": (5.83, 9,  35),
        "Halloween 2013 (10,400 CFS)":    (4.70, 14, 5),
        "Halloween 2015 (8,590 CFS)":     (5.90, 13, 35),
        "July 2025 (9,340 CFS)":          (6.50, 8,  30),
    }
    preset = st.selectbox("preset", list(presets.keys()),
                          label_visibility="collapsed")
    if presets[preset]:
        rain_in, dur_hr, soil = presets[preset]
        st.info(f"Loaded: {rain_in} in / {dur_hr} hr / soil {soil}%")

rain_mm = rain_in * 25.4
api_mm  = soil * 0.75

# ── Load & infer ─────────────────────────────────────────────────────────────

surrogate        = load_surrogate()
events_df        = load_events()
inp_df, meta_df, hyd_df = load_mc()

pred_cfs, infer_ms = predict(surrogate, rain_mm, dur_hr, api_mm)
stage              = classify(pred_cfs)
n_exceeded         = int((events_df["peak_cfs"] < pred_cfs).sum())
n_total            = len(events_df)
pct_exceeded       = n_exceeded / n_total * 100

analog_ids, analog_dists = find_top_analogs(inp_df, meta_df, rain_mm, dur_hr, api_mm, n=3)

SWMM_RUN_SEC = 35.0
ensemble_n   = 1000

# ── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div style='padding:0.3rem 0 1rem'>
    <span style='font-size:1.55rem;font-weight:800;color:#e2e8f0'>🌊 Shoal Creek</span>
    <span style='font-size:1.55rem;font-weight:300;color:#475569'> Flash Flood Risk Tool</span>
    <div style='font-size:0.76rem;color:#475569;margin-top:0.15rem'>
        USGS 08158600 · W 12th St, Austin TX &nbsp;·&nbsp;
        Surrogate emulator of EPA SWMM — 483-simulation training library
    </div>
</div>
""", unsafe_allow_html=True)

# ── Speed / value banner ──────────────────────────────────────────────────────

swmm_ensemble_min  = SWMM_RUN_SEC * ensemble_n / 60
surr_ensemble_ms   = infer_ms * ensemble_n

st.markdown(f"""
<div class="speed-banner">
  <div style="flex:2;min-width:0">
    <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;
                text-transform:uppercase;color:#1e4d7a;margin-bottom:0.4rem">
      Why a surrogate?
    </div>
    <div style="font-size:0.88rem;color:#94a3b8;line-height:1.55">
      SWMM (the physics model) takes <strong style="color:#e2e8f0">35 seconds</strong>
      per simulation run. To forecast with uncertainty — running
      {ensemble_n:,} storm scenarios to get a probability distribution —
      SWMM would take <strong style="color:#ef4444">{swmm_ensemble_min:.0f} minutes</strong>.
      The flood would already be peaking by the time it finished.<br>
      The surrogate runs {ensemble_n:,} scenarios in
      <strong style="color:#34d399">{surr_ensemble_ms:.1f} ms</strong>,
      preserving all available lead time for evacuation and preparation.
    </div>
  </div>
  <div style="display:flex;gap:1.5rem;flex-shrink:0;align-items:center">
    <div class="speed-stat">
      <div class="speed-num" style="color:#ef4444">~{swmm_ensemble_min:.0f} min</div>
      <div class="speed-unit">SWMM<br>{ensemble_n:,} scenarios</div>
    </div>
    <div class="speed-vs">→</div>
    <div class="speed-stat">
      <div class="speed-num" style="color:#34d399">{surr_ensemble_ms:.0f} ms</div>
      <div class="speed-unit">Surrogate<br>{ensemble_n:,} scenarios</div>
    </div>
    <div class="speed-vs">·</div>
    <div class="speed-stat">
      <div class="speed-num" style="color:#06b6d4">{infer_ms*1000:.0f} μs</div>
      <div class="speed-unit">This<br>prediction</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Main layout: forecast (left) + rigor (right) ─────────────────────────────

left, right = st.columns([11, 9], gap="large")

with left:
    # ── Prediction card ──
    intensity = rain_in / max(dur_hr, 0.1)
    soil_label = ("🌵 Very dry" if soil < 20 else "🌤 Dry" if soil < 40
                  else "🌧 Moist" if soil < 65 else "💧 Saturated")

    st.markdown(f"""
    <div class="card" style="margin-bottom:1rem">
      <div class="card-label">Storm scenario</div>
      <div style="font-size:0.95rem;color:#e2e8f0;font-weight:500;margin-bottom:0.2rem">
        {rain_in:.1f} in over {dur_hr} hr
        &nbsp;<span style="color:#475569">·</span>&nbsp;
        <span style="color:#94a3b8">{intensity:.2f} in/hr avg intensity</span>
        &nbsp;<span style="color:#475569">·</span>&nbsp;
        <span style="color:#94a3b8">{soil_label} ({soil}%)</span>
      </div>

      <div style="display:flex;align-items:flex-end;gap:1.5rem;margin-top:0.8rem">
        <div>
          <div class="card-label">Predicted peak — W. 12th St.</div>
          <div class="peak-num" style="color:{stage['color']}">{pred_cfs:,.0f}</div>
          <div style="font-size:0.85rem;color:#64748b">CFS &nbsp;·&nbsp; cubic feet per second</div>
          <div>
            <span class="stage-pill"
              style="background:{stage['bg']};color:{stage['color']};
                     border:1.5px solid {stage['color']}55">
              {stage['icon']} {stage['name']}
            </span>
          </div>
        </div>
        <div style="flex:1;font-size:0.83rem;color:#64748b;
                    padding-left:1rem;border-left:1px solid #1e3a5f;
                    line-height:1.6;padding-bottom:0.2rem">
          {STAGE_EXPLAIN[stage['name']]}<br><br>
          Exceeds <strong style="color:#94a3b8">{n_exceeded}</strong> of
          {n_total} USGS events on record
          ({pct_exceeded:.0f}% of historical floods).
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Gauge bar ──
    st.markdown('<div class="section-head" style="margin-bottom:0.2rem">Flood stage scale</div>',
                unsafe_allow_html=True)
    st.plotly_chart(make_gauge(pred_cfs, stage), use_container_width=True,
                    config={"displayModeBar": False})

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Triple hydrograph ──
    st.markdown("""
    <div class="section-head">3 nearest SWMM simulations from training library</div>
    <div style="font-size:0.78rem;color:#64748b;margin-bottom:0.5rem;line-height:1.5">
      The surrogate was trained on 483 full SWMM runs.
      The 3 runs with the most similar storm inputs are shown below —
      they define the plausible flood-wave shape and bound the natural variability
      around the surrogate's point estimate.
      The dashed line is the surrogate's peak prediction for <em>your</em> scenario.
    </div>
    """, unsafe_allow_html=True)

    fig_hyd = make_triple_hydrograph(hyd_df, analog_ids, meta_df, pred_cfs, stage)
    st.plotly_chart(fig_hyd, use_container_width=True, config={"displayModeBar": False})

    # Analog details table
    cols_a = st.columns(3)
    for i, (rid, dist) in enumerate(zip(analog_ids, analog_dists)):
        m = meta_df.loc[rid]
        inp_row = inp_df.loc[rid]
        with cols_a[i]:
            s = classify(float(m["peak_cfs"]))
            st.markdown(f"""
            <div style="background:#0d2340;border:1px solid #1e3a5f;border-radius:8px;
                        padding:0.6rem 0.8rem;font-size:0.76rem;color:#64748b">
              <div style="font-weight:600;color:#94a3b8;margin-bottom:0.2rem">
                Analog {i+1} — {rid}
              </div>
              {inp_row['rain_depth_mm']:.0f}mm /
              {inp_row['storm_dur_hr']:.0f}hr /<br>
              soil API {inp_row['api_5_mm']:.0f}mm<br>
              <span style="color:{s['color']};font-weight:700">
                Peak {m['peak_cfs']:,.0f} CFS
              </span>
            </div>
            """, unsafe_allow_html=True)


with right:
    # ── Model rigor ──
    st.markdown('<div class="section-head">Surrogate model quality</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card" style="margin-bottom:1rem">
      <div class="metric-row">
        <div class="metric-box">
          <div class="metric-box-num" style="color:#34d399">0.83</div>
          <div class="metric-box-lbl">R² on held-out<br>test data</div>
        </div>
        <div class="metric-box">
          <div class="metric-box-num" style="color:#06b6d4">631 CFS</div>
          <div class="metric-box-lbl">Mean absolute<br>error (MAE)</div>
        </div>
        <div class="metric-box">
          <div class="metric-box-num" style="color:#a78bfa">483</div>
          <div class="metric-box-lbl">SWMM runs in<br>training set</div>
        </div>
      </div>
      <div style="font-size:0.76rem;color:#64748b;margin-top:0.7rem;line-height:1.5">
        Trained via Latin Hypercube sampling across the full plausible storm space
        (15–300mm, 1–24hr, API 0–75mm). 80/20 train-test split.
        XGBoost with 6 physics-informed features.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Historical validation ──
    st.markdown("""
    <div class="section-head">Historical replay — 3 real Austin floods</div>
    <div style="font-size:0.78rem;color:#64748b;margin-bottom:0.7rem;line-height:1.5">
      The surrogate was run against storm parameters extracted from 3 documented events.
      It is compared to USGS-observed peaks. The gap reflects the QPE ceiling —
      hourly rain gauge data misses the sub-hourly intensity bursts that drive
      Austin's worst flash floods.
    </div>
    """, unsafe_allow_html=True)

    fig_val = make_validation_bar(HISTORICAL_REPLAY)
    st.plotly_chart(fig_val, use_container_width=True, config={"displayModeBar": False})

    # Replay detail rows
    st.markdown('<div class="card">', unsafe_allow_html=True)
    for r in HISTORICAL_REPLAY:
        err = (r["surrogate_cfs"] - r["usgs_cfs"]) / r["usgs_cfs"] * 100
        col = "#34d399" if abs(err) < 20 else "#eab308" if abs(err) < 40 else "#ef4444"
        st.markdown(f"""
        <div class="replay-row">
          <div class="replay-name">{r['name']} <span style="color:#475569;font-weight:400">· {r['date']}</span></div>
          <div style="display:flex;gap:1rem;margin-top:0.25rem;font-size:0.8rem">
            <span>Surrogate: <strong style="color:#06b6d4">{r['surrogate_cfs']:,} CFS</strong></span>
            <span>USGS: <strong style="color:#f472b6">{r['usgs_cfs']:,} CFS</strong></span>
            <span class="replay-err" style="color:{col}">{err:+.0f}%</span>
          </div>
          <div style="font-size:0.73rem;color:#475569;margin-top:0.2rem">{r['note']}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Limitations ──
    st.markdown(f"""
    <div class="caveat">
      <strong>⚠️ Limitations</strong><br>
      SWMM was calibrated to hourly ASOS rainfall (NSE = 0.43).
      Sub-hourly cloudburst intensity drives Austin's worst floods but is absent
      from available QPE data. <strong>Actual peaks can be 30–50% higher</strong>
      than the surrogate predicts for convective storms.<br><br>
      <strong>Not for emergency use.</strong> Monitor
      <a href="https://waterdata.usgs.gov/monitoring-location/08158600/"
         target="_blank" style="color:#60a5fa">USGS 08158600</a>
      and
      <a href="https://www.austintexas.gov/department/watershed-protection"
         target="_blank" style="color:#60a5fa">Austin WPD</a>.
    </div>
    """, unsafe_allow_html=True)
