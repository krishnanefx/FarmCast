"""
app.py — FarmCast: AI weather intelligence dashboard for UK farmers.
Run: streamlit run app.py --server.port 8501 --server.address 0.0.0.0
"""

import streamlit as st
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from datetime import datetime

from data_loader import get_forecast
from decision_logic import get_all_decisions, get_worst_verdict, GOOD, CAUTION, AVOID
from ollama_client import get_advisory, DEFAULT_MODEL

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OLLAMA_MODEL = "llama3.2:1b"  # Change here to switch model globally

REGIONS = [
    "East Midlands",
    "Yorkshire",
    "East Anglia",
    "Southwest England",
    "Wales",
    "Scotland",
    "Northern Ireland",
]

REGION_COORDS = {
    "East Midlands":     (52.80, -1.20),
    "Yorkshire":         (53.90, -1.50),
    "East Anglia":       (52.40,  0.90),
    "Southwest England": (50.70, -3.50),
    "Wales":             (52.10, -3.80),
    "Scotland":          (56.50, -4.00),
    "Northern Ireland":  (54.60, -6.70),
}

# Earth-tone palette
C_GREEN_DARK  = "#2d5a27"
C_GREEN_MID   = "#4a7c59"
C_BROWN_DARK  = "#5c4a1e"
C_BROWN_MID   = "#8b6914"
C_CREAM       = "#f5f0e8"
C_AVOID       = "#c0392b"
C_CAUTION     = "#e6a817"
C_GOOD        = "#2d6a2d"

VERDICT_COLOURS    = {GOOD: C_GOOD, CAUTION: C_CAUTION, AVOID: C_AVOID}
VERDICT_LABELS     = {GOOD: "✅ Good", CAUTION: "⚠️ Caution", AVOID: "🚫 Avoid"}
# Caution amber is light — use dark text for WCAG AA contrast (4.5:1)
VERDICT_TEXT_COLOURS = {GOOD: "white", CAUTION: "#1a1a1a", AVOID: "white"}
MARKER_COLOURS     = {GOOD: "green", CAUTION: "orange", AVOID: "red"}

# ---------------------------------------------------------------------------
# Page config & CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FarmCast",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  /* ------------------------------------------------------------------ */
  /* Accessibility                                                        */
  /* ------------------------------------------------------------------ */
  /* Skip-to-content: visible on keyboard focus only */
  .skip-link {{
      position: absolute; top: -48px; left: 0;
      background: {C_GREEN_DARK}; color: white;
      padding: 10px 18px; z-index: 9999;
      font-size: 16px; font-weight: 600;
      border-radius: 0 0 6px 0; text-decoration: none;
  }}
  .skip-link:focus {{ top: 0; outline: 3px solid #fff; }}

  /* ------------------------------------------------------------------ */
  /* Base — minimum 16px for WCAG AA                                     */
  /* ------------------------------------------------------------------ */
  html, body, [class*="css"] {{
      font-size: 16px;
      font-family: 'Segoe UI', Arial, sans-serif;
      background-color: {C_CREAM};
      color: #1a1a1a;
  }}

  /* ------------------------------------------------------------------ */
  /* Demo banner                                                          */
  /* ------------------------------------------------------------------ */
  .demo-banner {{
      background: {C_GREEN_DARK};
      color: white;
      text-align: center;
      padding: 0.45rem 1rem;
      font-size: 14px;
      font-weight: 500;
      border-radius: 6px;
      margin-bottom: 0.8rem;
      letter-spacing: 0.01em;
  }}

  /* ------------------------------------------------------------------ */
  /* Header strip                                                         */
  /* ------------------------------------------------------------------ */
  .farmcast-header {{
      background: linear-gradient(135deg, {C_GREEN_DARK}, {C_GREEN_MID});
      padding: 1.4rem 2rem;
      border-radius: 10px;
      color: white;
      margin-bottom: 1.2rem;
  }}
  .farmcast-header h1 {{
      margin: 0; font-size: 2.4rem; font-weight: 800; color: white;
  }}
  .farmcast-header p {{
      margin: 0.25rem 0 0; font-size: 1.05rem; opacity: 0.9;
  }}

  /* ------------------------------------------------------------------ */
  /* Section titles                                                       */
  /* ------------------------------------------------------------------ */
  .section-title {{
      font-size: 1.15rem; font-weight: 700;
      color: {C_GREEN_DARK}; margin: 1.2rem 0 0.6rem;
      border-left: 4px solid {C_GREEN_MID}; padding-left: 0.6rem;
  }}

  /* ------------------------------------------------------------------ */
  /* Decision cards — left coloured border                               */
  /* ------------------------------------------------------------------ */
  .decision-card {{
      background: white;
      border-radius: 10px;
      padding: 1.1rem 1.2rem;
      box-shadow: 0 2px 8px rgba(0,0,0,0.09);
      border-left: 5px solid #ccc;
      height: 100%;
  }}
  /* Pulsing animation on Avoid cards — colour + motion for accessibility */
  @keyframes pulse {{
      0%,  100% {{ opacity: 1;    }}
      50%        {{ opacity: 0.65; }}
  }}
  .avoid-card {{ animation: pulse 2s infinite; }}

  .card-icon     {{ font-size: 2rem; line-height: 1; margin-bottom: 0.3rem; }}
  .card-activity {{ font-size: 1rem; font-weight: 600; margin: 0.2rem 0 0.35rem; }}
  .card-reason   {{ font-size: 0.88rem; color: #444; margin: 0.3rem 0 0; line-height: 1.4; }}
  .card-metric   {{ font-size: 0.82rem; color: #777; margin: 0.25rem 0 0; }}

  /* Verdict badge — uses BOTH colour AND text+icon (never colour alone) */
  .verdict-badge {{
      display: inline-block;
      padding: 0.2rem 0.65rem;
      border-radius: 20px;
      font-size: 0.82rem; font-weight: 700;
      margin-bottom: 0.3rem;
  }}

  /* ------------------------------------------------------------------ */
  /* Frost banner                                                         */
  /* ------------------------------------------------------------------ */
  .frost-banner {{
      background: #fff3cd; border: 2px solid {C_BROWN_MID};
      border-radius: 8px; padding: 0.85rem 1.2rem;
      font-size: 1rem; font-weight: 600; color: {C_BROWN_DARK};
      margin-bottom: 0.8rem;
  }}

  /* ------------------------------------------------------------------ */
  /* Advisory box                                                         */
  /* ------------------------------------------------------------------ */
  .advisory-box {{
      background: #e8f5e9; border-left: 5px solid {C_GREEN_MID};
      border-radius: 8px; padding: 1rem 1.2rem;
      font-size: 1rem; line-height: 1.6; color: #1b3a1b;
  }}
  .advisory-title {{
      font-weight: 700; font-size: 1.05rem;
      color: {C_GREEN_DARK}; margin-bottom: 0.4rem;
  }}

  /* Misc */
  .updated-stamp {{ font-size: 0.82rem; color: #555; margin-top: 0.15rem; }}
</style>
""", unsafe_allow_html=True)

# Skip-to-content link (keyboard accessibility)
st.markdown('<a href="#main-content" class="skip-link">Skip to main content</a>',
            unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — controls & live status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ FarmCast Controls")

    # GPU status
    try:
        import onnxruntime as _ort
        _gpu = "CUDAExecutionProvider" in _ort.get_available_providers()
    except ImportError:
        _gpu = False

    if _gpu:
        st.markdown(
            '<p style="color:#2d6a2d; font-weight:700; font-size:1.05rem;">⚡ GPU Active</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p style="color:#d4870a; font-weight:700; font-size:1.05rem;">💻 CPU Mode</p>',
            unsafe_allow_html=True,
        )

    st.caption("Model: Pangu-Weather 1hr ONNX")
    st.divider()

    run_fresh = st.button("🔄 Run Fresh Forecast", use_container_width=True, type="primary")
    st.caption("Deletes cache and re-runs inference on the DGX GPU.")

# ---------------------------------------------------------------------------
# Region selection (query params take priority)
# ---------------------------------------------------------------------------
params = st.query_params
default_region = params.get("region", REGIONS[0])
if default_region not in REGIONS:
    default_region = REGIONS[0]

# ---------------------------------------------------------------------------
# Demo banner — P7
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="demo-banner">'
    "🏆 Built at the Lenovo × NVIDIA Hackathon — Pangu-Weather + Ollama running locally on a DGX"
    " &nbsp;|&nbsp; No cloud. No API. Pure edge AI."
    "</div>",
    unsafe_allow_html=True,
)

# Anchor for skip-to-content
st.markdown('<div id="main-content"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="farmcast-header" role="banner">
  <h1>FarmCast 🌾</h1>
  <p>Know your field before you leave the house</p>
</div>
""", unsafe_allow_html=True)

col_sel, col_stamp = st.columns([3, 2])
with col_sel:
    selected_region = st.selectbox(
        "Select region",
        REGIONS,
        index=REGIONS.index(default_region),
        label_visibility="collapsed",
    )

# Sync query params
st.query_params["region"] = selected_region

# ---------------------------------------------------------------------------
# Load forecast — fresh inference or from cache
# ---------------------------------------------------------------------------
if run_fresh:
    with st.spinner("Running Pangu-Weather inference on DGX GPU..."):
        forecast = get_forecast(selected_region, force_refresh=True)
    st.success(f"✅ Inference complete — data timestamp: {forecast['forecast_time']}")
else:
    forecast = get_forecast(selected_region)

decisions    = get_all_decisions(forecast)
worst_verdict = get_worst_verdict(decisions)
worst_card   = next(d for d in decisions if d["verdict"] == worst_verdict)

with col_stamp:
    st.markdown(
        f'<p class="updated-stamp">Last updated: {datetime.now().strftime("%d %b %Y, %H:%M")}</p>',
        unsafe_allow_html=True,
    )
    st.caption(f"Forecast target: {forecast['forecast_time']}")

# ---------------------------------------------------------------------------
# Section 1 — Decision Cards (2 × 2 grid)  — P3
# ---------------------------------------------------------------------------
st.markdown('<p class="section-title">Today\'s Activity Forecast</p>', unsafe_allow_html=True)

CARD_METRICS = [
    f"Wind: {forecast['wind_speed_kmh']:.0f} km/h",
    f"Precipitation: {forecast['precipitation_mm']:.1f} mm",
    f"Temp: {forecast['temperature_c']:.1f}°C",
    f"Temp: {forecast['temperature_c']:.1f}°C",
]

card_cols = st.columns(2)
for idx, card in enumerate(decisions):
    col = card_cols[idx % 2]
    verdict      = card["verdict"]
    badge_colour = VERDICT_COLOURS[verdict]
    text_colour  = VERDICT_TEXT_COLOURS[verdict]
    avoid_class  = "avoid-card" if verdict == AVOID else ""

    with col:
        st.markdown(f"""
        <div class="decision-card {avoid_class}"
             style="border-left-color:{badge_colour};"
             role="region"
             aria-label="{card['activity']} forecast: {verdict}">
          <div class="card-icon" aria-hidden="true">{card['icon']}</div>
          <p class="card-activity">{card['activity']}</p>
          <span class="verdict-badge"
                style="background:{badge_colour}; color:{text_colour};"
                role="status"
                aria-label="Verdict: {verdict}">{VERDICT_LABELS[verdict]}</span>
          <p class="card-reason">{card['reason']}</p>
          <p class="card-metric">{CARD_METRICS[idx]}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom:0.7rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 2 — Frost Risk Alert
# ---------------------------------------------------------------------------
if forecast["frost_risk"]:
    st.markdown(
        '<div class="frost-banner" role="alert">'
        "⚠️ Frost Risk Tonight — Protect vulnerable crops and bring in livestock before dusk."
        "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Section 3 — Ollama LLM Advisory  — P1
# ---------------------------------------------------------------------------
st.markdown('<p class="section-title">AI Evening Advisory</p>', unsafe_allow_html=True)

with st.spinner("Generating advisory…"):
    advisory_text = get_advisory(
        forecast,
        worst_verdict=worst_verdict,
        worst_reason=worst_card["reason"],
        model=OLLAMA_MODEL,
    )

st.markdown(f"""
<div class="advisory-box" role="complementary" aria-label="AI evening advisory">
  <p class="advisory-title">📋 Your Evening Advisory</p>
  <p>{advisory_text}</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 4 — Hourly Chart  — P4
# ---------------------------------------------------------------------------
st.markdown(
    f'<p class="section-title">Today\'s Conditions — {selected_region}</p>',
    unsafe_allow_html=True,
)

hourly      = forecast["hourly"]
hours       = [h["hour"]        for h in hourly]
temps       = [h["temp_c"]      for h in hourly]
winds       = [h["wind_kmh"]    for h in hourly]
precip_prob = [h["precip_prob"] for h in hourly]


def _precip_bar_color(prob: float) -> str:
    """Light blue (low) → dark blue (high) gradient."""
    r = int(173 + (31  - 173) * prob)
    g = int(216 + (119 - 216) * prob)
    b = int(230 + (180 - 230) * prob)
    a = round(0.35 + 0.45 * prob, 2)
    return f"rgba({r},{g},{b},{a})"


fig = go.Figure()

# Precipitation — gradient bars (behind the lines)
fig.add_trace(go.Bar(
    x=hours,
    y=[p * 100 for p in precip_prob],
    name="Precip Probability (%)",
    marker_color=[_precip_bar_color(p) for p in precip_prob],
    yaxis="y3",
    hovertemplate="%{y:.0f}%<extra>Precip Prob</extra>",
))

# Temperature — left axis
fig.add_trace(go.Scatter(
    x=hours, y=temps,
    name="Temperature (°C)",
    line=dict(color=C_GREEN_MID, width=2.5),
    mode="lines+markers",
    marker=dict(size=5),
    yaxis="y1",
    hovertemplate="%{y:.1f}°C<extra>Temperature</extra>",
))

# Wind speed — right axis
fig.add_trace(go.Scatter(
    x=hours, y=winds,
    name="Wind Speed (km/h)",
    line=dict(color=C_BROWN_MID, width=2.5, dash="dash"),
    mode="lines+markers",
    marker=dict(size=5),
    yaxis="y2",
    hovertemplate="%{y:.1f} km/h<extra>Wind Speed</extra>",
))

# --- Shapes & annotations ---
shapes      = []
annotations = []

# Vertical "Now" line
current_hour_str = f"{datetime.now().hour:02d}:00"
if current_hour_str in hours:
    shapes.append(dict(
        type="line",
        x0=current_hour_str, x1=current_hour_str,
        y0=0, y1=1, xref="x", yref="paper",
        line=dict(color="#666", width=1.5, dash="dot"),
    ))
    annotations.append(dict(
        x=current_hour_str, y=1.03, xref="x", yref="paper",
        text="Now", showarrow=False,
        font=dict(size=11, color="#555"),
        yanchor="bottom",
    ))

# Frost threshold — 0°C on temperature axis
shapes.append(dict(
    type="line",
    x0=hours[0], x1=hours[-1], y0=0, y1=0,
    xref="x", yref="y",
    line=dict(color=C_AVOID, width=1.5, dash="dash"),
))
annotations.append(dict(
    x=hours[-1], y=0, xref="x", yref="y",
    text="Frost threshold", showarrow=False,
    font=dict(size=10, color=C_AVOID),
    xanchor="right", yanchor="bottom",
))

# Spray threshold — 15 km/h on wind axis
shapes.append(dict(
    type="line",
    x0=hours[0], x1=hours[-1], y0=15, y1=15,
    xref="x", yref="y2",
    line=dict(color=C_AVOID, width=1.5, dash="dash"),
))
annotations.append(dict(
    x=hours[0], y=15, xref="x", yref="y2",
    text="Spray threshold", showarrow=False,
    font=dict(size=10, color=C_AVOID),
    xanchor="left", yanchor="bottom",
))

fig.update_layout(
    paper_bgcolor=C_CREAM,
    plot_bgcolor="#faf6ee",
    font=dict(color="#1a1a1a", size=13),
    title=dict(
        text=f"Today's Conditions — {selected_region}",
        font=dict(size=16, color=C_GREEN_DARK),
    ),
    height=400,
    xaxis=dict(
        title="Hour",
        gridcolor="#e0dbd0",
        linecolor="#b0a898",
        tickfont=dict(size=12),
    ),
    yaxis=dict(
        title=dict(text="Temperature (°C)", font=dict(color=C_GREEN_MID)),
        tickfont=dict(color=C_GREEN_MID, size=12),
        gridcolor="#e0dbd0",
        zeroline=False,
    ),
    yaxis2=dict(
        title=dict(text="Wind Speed (km/h)", font=dict(color=C_BROWN_MID)),
        tickfont=dict(color=C_BROWN_MID, size=12),
        overlaying="y",
        side="right",
        gridcolor="rgba(0,0,0,0)",
        zeroline=False,
    ),
    yaxis3=dict(
        title=dict(text="Precip Prob (%)", font=dict(color="rgba(31,119,180,0.7)")),
        tickfont=dict(color="rgba(31,119,180,0.7)", size=11),
        overlaying="y",
        side="right",
        position=1.0,
        range=[0, 200],
        gridcolor="rgba(0,0,0,0)",
        zeroline=False,
        showgrid=False,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right",  x=1,
        bgcolor="rgba(245,240,232,0.8)",
        bordercolor="#c0b898",
        borderwidth=1,
    ),
    margin=dict(l=60, r=80, t=60, b=50),
    hovermode="x unified",
    shapes=shapes,
    annotations=annotations,
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 5 — UK Map
# ---------------------------------------------------------------------------
st.markdown('<p class="section-title">Regional Overview Map</p>', unsafe_allow_html=True)
st.caption("Click a region marker to switch the dashboard to that region.")


@st.cache_data(ttl=300)
def _all_region_verdicts() -> dict:
    result = {}
    for reg in REGIONS:
        fc  = get_forecast(reg)
        dec = get_all_decisions(fc)
        result[reg] = get_worst_verdict(dec)
    return result


region_verdicts = _all_region_verdicts()

uk_map = folium.Map(location=[54.5, -3.5], zoom_start=6, tiles="CartoDB positron")

for reg, coords in REGION_COORDS.items():
    worst        = region_verdicts[reg]
    marker_colour = MARKER_COLOURS[worst]
    is_active    = reg == selected_region

    folium.Marker(
        location=coords,
        tooltip=reg,
        popup=folium.Popup(reg, max_width=150),
        icon=folium.Icon(
            color=marker_colour,
            icon="leaf" if is_active else "info-sign",
            prefix="glyphicon",
        ),
    ).add_to(uk_map)

map_result = st_folium(
    uk_map, width="100%", height=420,
    returned_objects=["last_object_clicked_tooltip"],
)

if map_result and map_result.get("last_object_clicked_tooltip"):
    clicked_region = map_result["last_object_clicked_tooltip"]
    if clicked_region in REGIONS and clicked_region != selected_region:
        st.query_params["region"] = clicked_region
        st.rerun()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("<hr style='border-color:#c8bfa8; margin-top:2rem;'>", unsafe_allow_html=True)
st.markdown(
    f"<p style='text-align:center; font-size:0.82rem; color:#888;'>"
    f"FarmCast — Lenovo × NVIDIA Hackathon &nbsp;|&nbsp; "
    f"Powered by Pangu-Weather &amp; Ollama ({OLLAMA_MODEL})"
    f"</p>",
    unsafe_allow_html=True,
)
