"""
app.py — FarmCast: AI weather intelligence dashboard for UK farmers.
Run: streamlit run app.py --server.port 8501 --server.address 0.0.0.0
"""

import json
import os
import folium
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from streamlit_folium import st_folium

from data_loader import get_forecast, get_weekly_forecast
from decision_logic import get_all_decisions, get_worst_verdict, GOOD, CAUTION, AVOID
from ollama_client import get_advisory, DEFAULT_MODEL
from dotenv import load_dotenv

# Load model paths from .env
load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OLLAMA_MODEL = "llama3.2:1b"  # Change here to switch model globally

REGIONS = [
    "East Midlands", "Yorkshire", "East Anglia",
    "Southwest England", "Wales", "Scotland", "Northern Ireland",
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

MARKER_COLOURS = {GOOD: "green", CAUTION: "orange", AVOID: "red"}

# Glass-card verdict colours (light, semi-transparent for dark backgrounds)
VERDICT_GLASS = {
    GOOD:    {"border": "#4ade80", "bg": "rgba(74,222,128,0.18)",  "text": "#4ade80", "label": "✓ Good"},
    CAUTION: {"border": "#fbbf24", "bg": "rgba(251,191,36,0.18)",  "text": "#fbbf24", "label": "⚠ Caution"},
    AVOID:   {"border": "#f87171", "bg": "rgba(248,113,113,0.18)", "text": "#f87171", "label": "✕ Avoid"},
}

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FarmCast",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — region selector + controls (needed before forecast load)
# ---------------------------------------------------------------------------
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="farmcast-ai-dashboard")

params = st.query_params
default_region = params.get("region", REGIONS[0])
if default_region not in REGIONS:
    default_region = REGIONS[0]

with st.sidebar:
    st.markdown(
        "<div style='font-size:18px; font-weight:600; color:white; margin-bottom:2px;'>🌾 FarmCast</div>"
        "<div style='font-size:11px; letter-spacing:2px; text-transform:uppercase; "
        "color:rgba(255,255,255,0.55); margin-bottom:18px;'>Weather Intelligence</div>",
        unsafe_allow_html=True,
    )
    
    st.markdown("<div style='font-size:13px; font-weight:600; color:white; margin-bottom:4px;'>Search Location</div>", unsafe_allow_html=True)
    search_loc = st.text_input("e.g. Tokyo, London, Paris", placeholder="Search globally...", label_visibility="collapsed")
    
    st.markdown("<div style='font-size:13px; font-weight:600; color:white; margin-top:8px; margin-bottom:4px;'>Or Select UK Region</div>", unsafe_allow_html=True)
    selected_region = st.selectbox(
        "Region", REGIONS,
        index=REGIONS.index(default_region),
        label_visibility="collapsed",
    )
    
    st.markdown("<div style='font-size:13px; font-weight:600; color:white; margin-top:8px; margin-bottom:4px;'>Forecast Window</div>", unsafe_allow_html=True)
    model_hours = int(st.selectbox(
        "Forecast Horizon", ["1", "3", "6", "24"],
        index=3, # Default 24h as per Yubo preference
        label_visibility="collapsed"
    ))
    
    st.divider()
    run_fresh = st.button("🔄 Refresh All Forecasts", use_container_width=True, type="primary")
    st.caption("Re-runs Pangu-Weather ONNX inference on the DGX GPU.")
    st.caption("📅 7-day forecast temporarily disabled.")

st.query_params["region"] = selected_region

# Geocode resolution if search term exists
custom_lat = None
custom_lon = None
resolved_location_name = selected_region

if search_loc.strip():
    with st.spinner(f"Geocoding {search_loc}..."):
        try:
            loc_data = geolocator.geocode(search_loc)
            if loc_data:
                custom_lat = loc_data.latitude
                custom_lon = loc_data.longitude
                resolved_location_name = loc_data.address.split(',')[0] # Grab just the city/main name
            else:
                st.sidebar.error("Location not found.")
        except Exception as e:
            st.sidebar.error("Geocoding failed.")

# ---------------------------------------------------------------------------
# Load forecast
# ---------------------------------------------------------------------------
if run_fresh:
    # Clear weekly session cache on refresh
    for _k in ["weekly_forecast", "weekly_region"]:
        st.session_state.pop(_k, None)
    with st.spinner("Running Pangu-Weather inference on DGX GPU..."):
        forecast = get_forecast(selected_region, lat=custom_lat, lon=custom_lon, model_hours=model_hours, force_refresh=True)
    
    # 7-day forecast paused per request
    # with st.spinner("Running 7-day iterative forecast chain..."):
    #     weekly = get_weekly_forecast(selected_region, force_refresh=True)
    # st.session_state["weekly_forecast"] = weekly
    
    st.session_state["weekly_region"]   = selected_region
    st.success(f"✅ Inference complete — {forecast.get('forecast_time', '')}")
else:
    forecast = get_forecast(selected_region, lat=custom_lat, lon=custom_lon, model_hours=model_hours)

# Load weekly forecast — cached in session_state per region
# if (
#     "weekly_forecast" not in st.session_state
#     or st.session_state.get("weekly_region") != selected_region
# ):
#     with st.spinner("Building 7-day forecast..."):
#         weekly = get_weekly_forecast(selected_region)
#     st.session_state["weekly_forecast"] = weekly
#     st.session_state["weekly_region"]   = selected_region
# else:
#     weekly = st.session_state["weekly_forecast"]

decisions     = get_all_decisions(forecast)
worst_verdict = get_worst_verdict(decisions)
worst_card    = next(d for d in decisions if d["verdict"] == worst_verdict)

# ---------------------------------------------------------------------------
# Sidebar — status (rendered after forecast loaded)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.divider()
    try:
        import onnxruntime as _ort
        _providers = _ort.get_available_providers()
        _gpu = "CUDAExecutionProvider" in _providers
    except ImportError:
        _gpu = False

    gpu_colour = "#4ade80" if _gpu else "#fbbf24"
    gpu_label  = "⚡ AI Accelerator Active" if _gpu else "💻 CPU Inference Mode"
    
    _meta_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "input_data", "era5_meta.json")
    if os.path.exists(_meta_path):
        with open(_meta_path) as _f:
            _meta = json.load(_f)
        _src = f"ERA5 {_meta['date']} {_meta['time']} UTC"
    else:
        _src = "Mock Data Fallback"

    # Weekly cache status
    try:
        from pangu_inference import count_cached_steps
        _cached = count_cached_steps()
        _total  = 13
        _weekly_src = f"{_cached}/{_total} cached" if _cached > 0 else "0/13 steps cached"
    except Exception:
        _weekly_src = "Mock Data"

    st.markdown(f"""
<div style="background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.15); border-radius:12px; padding:16px; margin-top:20px;">
  <div style="font-size:11px; letter-spacing:2px; text-transform:uppercase; color:rgba(255,255,255,0.5); font-weight:600; margin-bottom:12px;">System Status</div>
  <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.08); padding-bottom:8px; margin-bottom:8px;">
    <span style="font-size:12px; color:rgba(255,255,255,0.7);">Compute Engine</span>
    <span style="font-size:12px; font-weight:700; color:{gpu_colour};">{gpu_label}</span>
  </div>
  <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.08); padding-bottom:8px; margin-bottom:8px;">
    <span style="font-size:12px; color:rgba(255,255,255,0.7);">Loaded Model</span>
    <span style="font-size:12px; color:white; font-weight:500;">Pangu-Weather {model_hours}h</span>
  </div>
  <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.08); padding-bottom:8px; margin-bottom:8px;">
    <span style="font-size:12px; color:rgba(255,255,255,0.7);">7-Day Cache</span>
    <span style="font-size:12px; color:white; font-weight:500;">{_weekly_src}</span>
  </div>
  <div style="display:flex; justify-content:space-between; padding-bottom:4px;">
    <span style="font-size:12px; color:rgba(255,255,255,0.7);">Data Source</span>
    <span style="font-size:12px; color:white; font-weight:500;">{_src}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Theme — computed from forecast conditions
# ---------------------------------------------------------------------------
if forecast["frost_risk"]:
    _t = dict(
        bg_top="#0f2027", bg_mid="#203a43", bg_bottom="#2c5364",
        card_bg="rgba(200,230,255,0.12)", card_border="rgba(200,230,255,0.25)",
        text_primary="#e8f4fd", text_secondary="rgba(200,230,255,0.7)", accent="#90caf9",
    )
    condition_label = "Freezing"
elif forecast["precipitation_probability"] > 0.5:
    _t = dict(
        bg_top="#2c3e50", bg_mid="#3d5166", bg_bottom="#546e7a",
        card_bg="rgba(255,255,255,0.10)", card_border="rgba(255,255,255,0.15)",
        text_primary="#ffffff", text_secondary="rgba(255,255,255,0.7)", accent="#b0bec5",
    )
    condition_label = "Heavy Rain Likely" if forecast["precipitation_probability"] > 0.6 else "Showers Possible"
else:
    _t = dict(
        bg_top="#1a3a5c", bg_mid="#2d6a8f", bg_bottom="#4a9abe",
        card_bg="rgba(255,255,255,0.12)", card_border="rgba(255,255,255,0.2)",
        text_primary="#ffffff", text_secondary="rgba(255,255,255,0.75)", accent="#7dd3fc",
    )
    condition_label = "Windy" if forecast["wind_speed_kmh"] > 30 else "Clear Conditions"

# ---------------------------------------------------------------------------
# CSS — full injection with theme variables + glassmorphism
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Serif+Display&display=swap');

:root {{
  --bg-top:         {_t['bg_top']};
  --bg-mid:         {_t['bg_mid']};
  --bg-bottom:      {_t['bg_bottom']};
  --card-bg:        {_t['card_bg']};
  --card-border:    {_t['card_border']};
  --text-primary:   {_t['text_primary']};
  --text-secondary: {_t['text_secondary']};
  --accent:         {_t['accent']};
}}

/* Font */
*, *::before, *::after {{ font-family: 'DM Sans', sans-serif !important; }}

/* Hide Streamlit chrome */
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{
  background: transparent !important;
  height: 0 !important;
}}

/* Full-page atmospheric gradient */
.stApp {{
  background: linear-gradient(
    170deg,
    var(--bg-top)    0%,
    var(--bg-mid)   50%,
    var(--bg-bottom) 100%
  ) !important;
  min-height: 100vh;
  background-attachment: fixed !important; 
}}

/* Transparent scaffolding */
.block-container {{
  background: transparent !important;
  padding-top: 1.2rem !important;
  max-width: 1200px;
}}
[data-testid="stVerticalBlock"],
.element-container,
[data-testid="stHorizontalBlock"] {{
  background: transparent !important;
}}
div[data-testid="stHorizontalBlock"] {{ gap: 12px; }}

/* Sidebar — frosted glass */
[data-testid="stSidebar"] {{
  background: rgba(0,0,0,0.38) !important;
  backdrop-filter: blur(24px) !important;
  -webkit-backdrop-filter: blur(24px) !important;
  border-right: 1px solid rgba(255,255,255,0.1) !important;
}}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] .stCaption {{ color: rgba(255,255,255,0.8) !important; }}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {{
  background: rgba(255,255,255,0.08) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  color: white !important;
  border-radius: 12px !important;
  transition: border-color 0.2s;
}}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div:hover {{
  border-color: rgba(255,255,255,0.3) !important;
}}
[data-testid="stSidebar"] .stButton > button {{
  border-radius: 12px !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em !important;
  border: 1px solid rgba(255,255,255,0.3) !important;
  color: white !important;
  background: rgba(255,255,255,0.12) !important;
}}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
  background: rgba(125,211,252,0.22) !important;
  border-color: rgba(125,211,252,0.45) !important;
}}
[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.12) !important; }}

/* Frosted glass card */
.glass-card {{
  background: var(--card-bg);
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  border: 1px inset var(--card-border);
  border-radius: 20px;
  padding: 24px;
  margin-bottom: 20px;
  color: var(--text-primary);
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.15), inset 0 1px 0 rgba(255,255,255,0.1);
  transition: transform 0.3s cubic-bezier(0.2, 0.8, 0.2, 1), box-shadow 0.3s ease;
}}

.glass-card:hover {{
  transform: translateY(-4px);
  box-shadow: 0 16px 40px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255,255,255,0.15);
}}

/* Horizontal hourly strip — hide scrollbar */
.hourly-strip {{
  overflow-x: auto;
  display: flex;
  gap: 0;
  padding: 0 12px 8px;
  scrollbar-width: none;
  -ms-overflow-style: none;
}}
.hourly-strip::-webkit-scrollbar {{ display: none; }}

/* Skip link */
.skip-link {{
  position: absolute; top: -48px; left: 0;
  background: rgba(0,0,0,0.85); color: white;
  padding: 10px 18px; z-index: 9999;
  font-size: 16px; border-radius: 0 0 6px 0;
  text-decoration: none;
}}
.skip-link:focus {{ top: 0; outline: 3px solid white; }}

/* Avoid card pulse */
@keyframes pulse {{
  0%, 100% {{ opacity: 1;   }}
  50%       {{ opacity: 0.6; }}
}}
.avoid-pulse {{ animation: pulse 2s infinite; }}

/* Floating particles */
@keyframes float {{
  0%, 100% {{ transform: translateY(0px) rotate(0deg);   opacity: 0.25; }}
  50%       {{ transform: translateY(-20px) rotate(180deg); opacity: 0.55; }}
}}
.particle {{
  position: fixed;
  border-radius: 50%;
  background: rgba(255,255,255,0.35);
  animation: float linear infinite;
  pointer-events: none;
  z-index: 0;
}}

/* Success/spinner text colour fix */
[data-testid="stAlert"] {{ background: rgba(255,255,255,0.1) !important; border-radius: 12px !important; }}
</style>
""", unsafe_allow_html=True)

# Floating particles
st.markdown("""
<div aria-hidden="true">
  <div class="particle" style="width:4px;height:4px;left:8%;top:15%;animation-duration:12s;animation-delay:0s;"></div>
  <div class="particle" style="width:6px;height:6px;left:20%;top:60%;animation-duration:18s;animation-delay:2s;"></div>
  <div class="particle" style="width:3px;height:3px;left:35%;top:25%;animation-duration:10s;animation-delay:4s;"></div>
  <div class="particle" style="width:5px;height:5px;left:55%;top:72%;animation-duration:15s;animation-delay:1s;"></div>
  <div class="particle" style="width:4px;height:4px;left:68%;top:10%;animation-duration:20s;animation-delay:3s;"></div>
  <div class="particle" style="width:6px;height:6px;left:78%;top:48%;animation-duration:9s;animation-delay:5s;"></div>
  <div class="particle" style="width:3px;height:3px;left:88%;top:70%;animation-duration:14s;animation-delay:2s;"></div>
  <div class="particle" style="width:5px;height:5px;left:45%;top:88%;animation-duration:16s;animation-delay:6s;"></div>
</div>
""", unsafe_allow_html=True)

# Accessibility skip link
st.markdown('<a href="#main-content" class="skip-link">Skip to main content</a>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Demo banner
# ---------------------------------------------------------------------------
st.markdown(
    "<div style='text-align:center; padding:6px 20px; margin-bottom:4px; "
    "background:rgba(0,0,0,0.22); border:1px solid rgba(255,255,255,0.12); "
    "border-radius:30px; font-size:13px; color:rgba(255,255,255,0.75); "
    "letter-spacing:0.03em;'>"
    "🏆 Lenovo × NVIDIA Hackathon &nbsp;·&nbsp; "
    "Pangu-Weather + Ollama on DGX &nbsp;·&nbsp; No cloud. No API. Pure edge AI."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown('<div id="main-content"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero section
# ---------------------------------------------------------------------------
temp_c     = forecast["temperature_c"]
wind_kmh   = forecast["wind_speed_kmh"]
precip_pct = forecast["precipitation_probability"] * 100
frost_tonight = "❄️ Frost Risk Tonight" if forecast["frost_risk"] else "✓ No Frost Risk"

st.markdown(f"""
<div style="text-align:center; padding:36px 20px 20px;" role="banner">
  <div style="font-size:13px; font-weight:500; letter-spacing:3px;
              text-transform:uppercase; color:var(--text-secondary);
              margin-bottom:10px;">
    🌾 FARMCAST &nbsp;·&nbsp; {resolved_location_name.upper()}
  </div>
  <div style="font-family:'DM Serif Display', Georgia, serif;
              font-size:96px; font-weight:400; color:var(--text-primary);
              line-height:1; margin-bottom:8px;
              text-shadow: 0 4px 40px rgba(0,0,0,0.3);">
    {temp_c:.0f}°
  </div>
  <div style="font-size:22px; font-weight:300; color:var(--text-secondary);
              margin-bottom:10px; letter-spacing:0.02em;">
    {condition_label}
  </div>
  <div style="font-size:15px; color:var(--text-secondary); letter-spacing:0.01em;">
    Wind {wind_kmh:.0f} km/h &nbsp;·&nbsp;
    Rain {precip_pct:.0f}% &nbsp;·&nbsp;
    {frost_tonight}
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hourly Forecast Strip
# ---------------------------------------------------------------------------
hourly           = forecast["hourly"]
current_hour_str = f"{datetime.now().hour:02d}:00"


def _hour_icon(h: dict) -> str:
    if h["temp_c"] < 2:
        return "🌨️"
    if h["precip_prob"] > 0.6:
        return "🌧️"
    if h["precip_prob"] > 0.3:
        return "🌦️"
    if h["wind_kmh"] > 30:
        return "💨"
    hour_num = int(h["hour"].split(":")[0])
    return "🌤️" if 12 <= hour_num <= 17 else "☀️"


hour_cells = ""
for h in hourly:
    is_now = h["hour"] == current_hour_str
    hl_bg  = "rgba(255,255,255,0.16)" if is_now else "transparent"
    fw     = "700" if is_now else "400"
    
    # NO INDENTATION ALLOWED: Streamlit will markdown-parse indented blocks as <pre><code>!
    cell = (
        f'<div style="min-width:66px; display:flex; flex-direction:column; align-items:center; '
        f'padding:10px 6px; border-radius:16px; background:{hl_bg}; flex-shrink:0;">'
        f'<div style="font-size:12px; color:var(--text-secondary); font-weight:{fw}; margin-bottom:4px;">{h["hour"]}</div>'
        f'<div style="font-size:26px; line-height:1; margin-bottom:6px;" aria-hidden="true">{_hour_icon(h)}</div>'
        f'<div style="font-size:16px; font-weight:600; color:var(--text-primary);">{h["temp_c"]:.0f}°</div>'
        f'<div style="font-size:11px; color:var(--accent); margin-top:3px;">{h["wind_kmh"]:.0f} km/h</div>'
        f'</div>'
    )
    hour_cells += cell
    
st.markdown(f"""
<div class="glass-card" style="padding:16px 0 10px;" role="region" aria-label="Hourly forecast">
<div style="padding:0 20px 10px; font-size:11px; letter-spacing:2px; text-transform:uppercase; color:var(--text-secondary); font-weight:500;">
Hourly Forecast
</div>
<div class="hourly-strip">
{hour_cells}
</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 7-Day Forecast Strip + Chart (TEMPORARILY DISABLED)
# ---------------------------------------------------------------------------

# def _day_precip_icon(day: dict) -> str:
#     p = day["precipitation_probability"]
#     t = day["temp_min_c"]
#     if t < 2 and p > 0.4:
#         return "🌨️"
#     if p > 0.65:
#         return "🌧️"
#     if p > 0.35:
#         return "🌦️"
#     if day["wind_speed_kmh"] > 35:
#         return "💨"
#     return "☀️"


# # Build day strip cards
# _day_cells = ""
# for _day in weekly:
#     _frost_star = " ❄" if _day["frost_risk"] else ""
#     _label      = _day["date"]
#     _icon       = _day_precip_icon(_day)
#     _tmax       = _day["temp_max_c"]
#     _tmin       = _day["temp_min_c"]
#     _src_dot    = (
#         "rgba(125,211,252,0.7)" if _day.get("source") == "pangu"
#         else "rgba(251,191,36,0.55)"
#     )
#     _bar_pct    = int(_day["precipitation_probability"] * 100)
#     _day_cells += f"""
#     <div style="flex:1; min-width:80px; display:flex; flex-direction:column;
#                 align-items:center; padding:10px 6px; border-radius:16px;
#                 background:rgba(255,255,255,0.07);">
#       <div style="font-size:11px; font-weight:500; color:var(--text-secondary);
#                   margin-bottom:5px; white-space:nowrap;">{_label}</div>
#       <div style="font-size:28px; line-height:1; margin-bottom:6px;">{_icon}</div>
#       <div style="font-size:14px; font-weight:700; color:var(--text-primary);">{_tmax:.0f}°</div>
#       <div style="font-size:12px; color:var(--text-secondary);">{_tmin:.0f}°{_frost_star}</div>
#       <div style="width:32px; height:4px; border-radius:2px; margin-top:6px;
#                   background:rgba(255,255,255,0.15);">
#         <div style="width:{_bar_pct}%; height:4px; border-radius:2px;
#                     background:rgba(125,211,252,0.65);"></div>
#       </div>
#       <div style="width:6px; height:6px; border-radius:50%; margin-top:6px;
#                   background:{_src_dot};" title="{'Pangu-Weather' if _day.get('source')=='pangu' else 'Mock'}"></div>
#     </div>"""

# st.markdown(f"""
# <div class="glass-card" style="padding:16px 12px 14px;" role="region" aria-label="7-day forecast">
#   <div style="padding:0 8px 12px; font-size:11px; letter-spacing:2px;
#               text-transform:uppercase; color:var(--text-secondary); font-weight:500;
#               display:flex; justify-content:space-between; align-items:center;">
#     <span>7-Day Forecast</span>
#     <span style="font-size:10px; letter-spacing:1px; color:var(--text-secondary); opacity:0.7;">
#       🔵 Pangu-Weather &nbsp; 🟡 Mock
#     </span>
#   </div>
#   <div style="display:flex; gap:6px; overflow-x:auto; scrollbar-width:none;">
#     {_day_cells}
#   </div>
# </div>
# """, unsafe_allow_html=True)

# # Plotly 7-day chart
# _dates    = [d["date"]                       for d in weekly]
# _tmax     = [d["temp_max_c"]                 for d in weekly]
# _tmin     = [d["temp_min_c"]                 for d in weekly]
# _precip   = [d["precipitation_probability"]  for d in weekly]
# _frost_x  = [d["date"] for d in weekly if d["frost_risk"]]
# _frost_y  = [d["temp_min_c"] for d in weekly if d["frost_risk"]]

# _fig = go.Figure()

# # Temperature band fill
# _fig.add_trace(go.Scatter(
#     x=_dates + _dates[::-1],
#     y=_tmax + _tmin[::-1],
#     fill="toself",
#     fillcolor="rgba(125,211,252,0.15)",
#     line=dict(color="rgba(0,0,0,0)"),
#     hoverinfo="skip",
#     showlegend=False,
#     name="Temp band",
# ))

# # Max temperature line
# _fig.add_trace(go.Scatter(
#     x=_dates, y=_tmax,
#     mode="lines+markers",
#     name="Max °C",
#     line=dict(color="#7dd3fc", width=2),
#     marker=dict(size=7, color="#7dd3fc"),
#     yaxis="y1",
# ))

# # Min temperature line
# _fig.add_trace(go.Scatter(
#     x=_dates, y=_tmin,
#     mode="lines+markers",
#     name="Min °C",
#     line=dict(color="#93c5fd", width=2, dash="dot"),
#     marker=dict(size=6, color="#93c5fd"),
#     yaxis="y1",
# ))

# # Frost star markers
# if _frost_x:
#     _fig.add_trace(go.Scatter(
#         x=_frost_x, y=_frost_y,
#         mode="markers",
#         name="Frost Risk",
#         marker=dict(symbol="star", size=14, color="#bfdbfe", line=dict(color="white", width=1)),
#         yaxis="y1",
#     ))

# # 0°C reference line
# _fig.add_hline(
#     y=0, line=dict(color="rgba(255,255,255,0.3)", dash="dash", width=1),
#     annotation_text="0°C", annotation_font_color="rgba(255,255,255,0.5)",
#     annotation_font_size=10,
# )

# # Precipitation probability bars on secondary axis
# _fig.add_trace(go.Bar(
#     x=_dates, y=_precip,
#     name="Rain prob.",
#     marker_color="rgba(125,211,252,0.35)",
#     marker_line=dict(color="rgba(125,211,252,0.5)", width=1),
#     yaxis="y2",
#     opacity=0.8,
# ))

# _fig.update_layout(
#     paper_bgcolor="rgba(0,0,0,0)",
#     plot_bgcolor="rgba(0,0,0,0)",
#     font=dict(color="rgba(255,255,255,0.75)", family="DM Sans, sans-serif", size=12),
#     legend=dict(
#         orientation="h", x=0, y=1.12,
#         font=dict(size=11),
#         bgcolor="rgba(0,0,0,0)",
#     ),
#     margin=dict(l=0, r=0, t=32, b=0),
#     xaxis=dict(
#         showgrid=False,
#         tickfont=dict(color="rgba(255,255,255,0.65)", size=11),
#         linecolor="rgba(255,255,255,0.1)",
#     ),
#     yaxis=dict(
#         title=dict(text="Temperature (°C)", font=dict(size=11)),
#         tickfont=dict(color="rgba(255,255,255,0.65)", size=10),
#         gridcolor="rgba(255,255,255,0.07)",
#         zerolinecolor="rgba(255,255,255,0.15)",
#     ),
#     yaxis2=dict(
#         title="Rain probability",
#         overlaying="y", side="right",
#         range=[0, 1],
#         tickformat=".0%",
#         tickfont=dict(color="rgba(125,211,252,0.7)", size=10),
#         showgrid=False,
#     ),
#     hovermode="x unified",
# )

# st.plotly_chart(_fig, use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Decision Cards — 2 × 2 glass grid
# ---------------------------------------------------------------------------
CARD_METRICS = [
    f"Wind: {forecast['wind_speed_kmh']:.0f} km/h",
    f"Precipitation: {forecast['precipitation_mm']:.1f} mm",
    f"Temp: {forecast['temperature_c']:.1f}°C",
    f"Temp: {forecast['temperature_c']:.1f}°C",
]

card_cols = st.columns(2)
for idx, card in enumerate(decisions):
    col         = card_cols[idx % 2]
    verdict     = card["verdict"]
    vg          = VERDICT_GLASS[verdict]
    extra_class = "avoid-pulse" if verdict == AVOID else ""

    with col:
        st.markdown(f"""
        <div class="glass-card {extra_class}"
             style="border-left:3px solid {vg['border']};
                    display:flex; justify-content:space-between;
                    align-items:center; padding:18px 20px;"
             role="region"
             aria-label="{card['activity']} — {verdict}">
          <div style="flex:1; min-width:0; padding-right:12px;">
            <div style="font-size:11px; letter-spacing:2px; text-transform:uppercase;
                        color:var(--text-secondary); margin-bottom:5px;">
              {card['icon']} &nbsp;{card['activity']}
            </div>
            <div style="font-size:14px; color:var(--text-primary); line-height:1.45;">
              {card['reason']}
            </div>
            <div style="font-size:12px; color:var(--text-secondary); margin-top:5px;">
              {CARD_METRICS[idx]}
            </div>
          </div>
          <div style="flex-shrink:0;">
            <span style="font-size:13px; font-weight:600; padding:6px 14px;
                         border-radius:20px; background:{vg['bg']};
                         color:{vg['text']}; white-space:nowrap;
                         border:1px solid {vg['border']}44;"
                  role="status" aria-label="Verdict: {verdict}">
              {vg['label']}
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Frost Alert
# ---------------------------------------------------------------------------
if forecast["frost_risk"]:
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(147,197,253,0.18),rgba(96,165,250,0.08));
                border:1px solid rgba(147,197,253,0.38);
                border-radius:16px; padding:16px 20px;
                display:flex; align-items:center; gap:14px;
                margin-bottom:16px;" role="alert">
      <div style="font-size:32px; flex-shrink:0;" aria-hidden="true">🌨️</div>
      <div>
        <div style="font-size:14px; font-weight:600; color:#93c5fd; margin-bottom:3px;">
          Frost Risk Tonight
        </div>
        <div style="font-size:13px; color:rgba(255,255,255,0.72); line-height:1.5;">
          Protect vulnerable crops and bring in livestock before dusk.
          Temperature dropping to {forecast['temperature_c']:.1f}°C.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Ollama Advisory
# ---------------------------------------------------------------------------
with st.spinner("Generating advisory…"):
    advisory_text = get_advisory(
        forecast,
        worst_verdict=worst_verdict,
        worst_reason=worst_card["reason"],
        model=OLLAMA_MODEL,
    )

st.markdown(f"""
<div class="glass-card" role="complementary" aria-label="AI evening advisory">
  <div style="font-size:11px; letter-spacing:2px; text-transform:uppercase;
              color:var(--text-secondary); margin-bottom:14px; font-weight:500;">
    🤖 AI Evening Advisory
  </div>
  <div style="font-size:16px; line-height:1.75; color:var(--text-primary);
              font-style:italic;">
    "{advisory_text}"
  </div>
  <div style="font-size:11px; color:var(--text-secondary); margin-top:12px; text-align:right;">
    Powered by Ollama &nbsp;·&nbsp; {OLLAMA_MODEL}
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# UK Map — dark_matter tiles
# ---------------------------------------------------------------------------
st.markdown(
    "<div style='font-size:11px; letter-spacing:2px; text-transform:uppercase; "
    "color:var(--text-secondary); font-weight:500; margin-bottom:6px;'>"
    "📍 Regional Overview</div>",
    unsafe_allow_html=True,
)
st.caption("Click a region marker to switch the dashboard.")


@st.cache_data(ttl=300)
def _all_region_verdicts() -> dict:
    result = {}
    for reg in REGIONS:
        fc  = get_forecast(reg)
        dec = get_all_decisions(fc)
        result[reg] = get_worst_verdict(dec)
    return result


region_verdicts = _all_region_verdicts()

uk_map = folium.Map(location=[54.5, -3.5], zoom_start=6, tiles="CartoDB dark_matter")

for reg, coords in REGION_COORDS.items():
    worst         = region_verdicts[reg]
    marker_colour = MARKER_COLOURS[worst]
    is_active     = reg == selected_region

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
st.markdown(
    "<div style='text-align:center; padding:24px 0 16px; "
    "font-size:12px; color:var(--text-secondary); letter-spacing:0.03em;'>"
    f"FarmCast &nbsp;·&nbsp; Lenovo × NVIDIA Hackathon &nbsp;·&nbsp; "
    f"Pangu-Weather + Ollama ({OLLAMA_MODEL})"
    "</div>",
    unsafe_allow_html=True,
)
