"""
app.py — FarmCast: AI weather intelligence dashboard for UK farmers.
Run: streamlit run app.py --server.port 8501 --server.address 0.0.0.0
"""

import json
import os
import time
import folium
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from branca.element import Element
from datetime import datetime, timedelta, timezone
from streamlit_folium import st_folium
from zoneinfo import ZoneInfo

try:
    from geopy.geocoders import Nominatim as _Nominatim
    _geolocator = _Nominatim(user_agent="farmcast-ai-dashboard")
except Exception:
    _geolocator = None

try:
    from timezonefinder import TimezoneFinder as _TZFinder
    _tz_finder = _TZFinder()
except Exception:
    _tz_finder = None

from data_loader import get_forecast, get_weekly_forecast
from decision_logic import get_all_decisions, get_worst_verdict, GOOD, CAUTION, AVOID
from ollama_client import get_advisory, DEFAULT_MODEL

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

COLOR_VISION_PROFILES = {
    "Standard (Default)": {
        "verdict_glass": {
            GOOD: {"border": "#4ade80", "bg": "rgba(20,83,45,0.32)", "text": "#bbf7d0", "icon": "✓", "label": "Good"},
            CAUTION: {"border": "#fcd34d", "bg": "rgba(146,64,14,0.32)", "text": "#fde68a", "icon": "⚠", "label": "Caution"},
            AVOID: {"border": "#fca5a5", "bg": "rgba(127,29,29,0.34)", "text": "#fecaca", "icon": "✕", "label": "Avoid"},
        },
        "marker_colours": {GOOD: "green", CAUTION: "orange", AVOID: "red"},
        "rain_colours": {"low": "#4ade80", "mid": "#fbbf24", "high": "#f87171"},
    },
    "Red-Green Blind (Protanopia)": {
        "verdict_glass": {
            GOOD: {"border": "#93c5fd", "bg": "rgba(30,64,175,0.30)", "text": "#dbeafe", "icon": "◆", "label": "Good"},
            CAUTION: {"border": "#d1d5db", "bg": "rgba(75,85,99,0.32)", "text": "#f3f4f6", "icon": "▲", "label": "Caution"},
            AVOID: {"border": "#fdba74", "bg": "rgba(154,52,18,0.34)", "text": "#ffedd5", "icon": "■", "label": "Avoid"},
        },
        "marker_colours": {GOOD: "blue", CAUTION: "lightgray", AVOID: "orange"},
        "rain_colours": {"low": "#60a5fa", "mid": "#d1d5db", "high": "#fb923c"},
    },
    "Blue-Yellow Blind (Tritanopia)": {
        "verdict_glass": {
            GOOD: {"border": "#67e8f9", "bg": "rgba(8,145,178,0.30)", "text": "#cffafe", "icon": "●", "label": "Good"},
            CAUTION: {"border": "#c4b5fd", "bg": "rgba(109,40,217,0.30)", "text": "#ede9fe", "icon": "◆", "label": "Caution"},
            AVOID: {"border": "#fca5a5", "bg": "rgba(153,27,27,0.35)", "text": "#fee2e2", "icon": "✕", "label": "Avoid"},
        },
        "marker_colours": {GOOD: "cadetblue", CAUTION: "purple", AVOID: "darkred"},
        "rain_colours": {"low": "#06b6d4", "mid": "#8b5cf6", "high": "#ef4444"},
    },
    "Monochrome (High Contrast)": {
        "verdict_glass": {
            GOOD: {"border": "#e5e7eb", "bg": "rgba(31,41,55,0.40)", "text": "#f9fafb", "icon": "✅", "label": "Good"},
            CAUTION: {"border": "#9ca3af", "bg": "rgba(55,65,81,0.42)", "text": "#f3f4f6", "icon": "⚠", "label": "Caution"},
            AVOID: {"border": "#ffffff", "bg": "rgba(17,24,39,0.55)", "text": "#ffffff", "icon": "❌", "label": "Avoid"},
        },
        "marker_colours": {GOOD: "white", CAUTION: "gray", AVOID: "black"},
        "rain_colours": {"low": "#d1d5db", "mid": "#9ca3af", "high": "#111827"},
    },
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
params = st.query_params
default_region = params.get("region", REGIONS[0])
if default_region not in REGIONS:
    default_region = REGIONS[0]

custom_lat = None
custom_lon = None
custom_location_name = None

with st.sidebar:
    st.markdown(
        "<div style='font-size:18px; font-weight:700; color:white; margin-bottom:2px;'>🌾 FarmCast</div>"
        "<div style='font-size:11px; letter-spacing:2px; text-transform:uppercase; "
        "color:rgba(255,255,255,0.55); margin-bottom:16px;'>Weather Intelligence</div>",
        unsafe_allow_html=True,
    )

    selected_region = st.selectbox("Region", REGIONS, index=REGIONS.index(default_region))

    if _geolocator is not None:
        search_loc = st.text_input(
            "Search location",
            placeholder="e.g. Edinburgh, UK",
            help="Overrides the region selector above.",
        )
        if search_loc.strip():
            with st.spinner("Geocoding…"):
                try:
                    _loc = _geolocator.geocode(search_loc)
                    if _loc:
                        custom_lat = _loc.latitude
                        custom_lon = _loc.longitude
                        custom_location_name = _loc.address.split(",")[0].strip()
                        st.caption(f"Found: {custom_location_name}")
                    else:
                        st.caption("Location not found — using region centre.")
                except Exception:
                    st.caption("Geocoding unavailable.")

    _col_w, _col_u = st.columns(2)
    with _col_w:
        model_hours = int(st.selectbox(
            "Window (h)", ["1", "6", "24"],
            index=0,
            help="Pangu-Weather model lead time.",
        ))
    with _col_u:
        unit_system = st.selectbox("Units", ["Metric", "Imperial"], index=0)

    st.markdown("<div class='sb-section'>Accessibility</div>", unsafe_allow_html=True)
    font_size_px = st.slider("Text size (px)", min_value=14, max_value=24, value=16, step=1)
    color_vision_mode = st.selectbox(
        "Color vision mode",
        list(COLOR_VISION_PROFILES.keys()),
        index=0,
        help="Applies colorblind-safe palettes and distinct symbols for status.",
    )
    high_contrast = st.checkbox("High contrast", value=False)
    reduce_motion = st.checkbox("Reduce motion", value=False)
    enable_tts = st.checkbox("Show text-to-speech controls", value=True)

    st.divider()
    run_fresh = st.button("🔄 Refresh All Forecasts", use_container_width=True, type="primary")
    st.caption("Re-runs Pangu-Weather ONNX inference.\n📅 7-day: Days 1–2 via 6h model, Days 3–7 via 24h model.")

st.query_params["region"] = selected_region
resolved_location_name = custom_location_name if custom_location_name else selected_region

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

_refresh_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
if run_fresh:
    _refresh_status_message = (
        f"Forecast refresh complete for {resolved_location_name}. "
        f"Lead time {model_hours} hours. Updated {_refresh_timestamp}."
    )
else:
    _refresh_status_message = (
        f"Forecast loaded for {resolved_location_name}. "
        f"Lead time {model_hours} hours. Updated {_refresh_timestamp}."
    )

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
# Unit formatting (display only; model logic remains metric)
# ---------------------------------------------------------------------------
is_imperial = unit_system == "Imperial"


def _temp_to_display(temp_c: float) -> float:
    return (temp_c * 9.0 / 5.0 + 32.0) if is_imperial else temp_c


def _wind_to_display(wind_kmh: float) -> float:
    return (wind_kmh * 0.621371) if is_imperial else wind_kmh


def _precip_to_display(precip_mm: float) -> float:
    return (precip_mm / 25.4) if is_imperial else precip_mm


def _format_temp(temp_c: float, decimals: int = 1) -> str:
    unit = "°F" if is_imperial else "°C"
    return f"{_temp_to_display(temp_c):.{decimals}f}{unit}"


def _format_wind(wind_kmh: float, decimals: int = 0) -> str:
    unit = "mph" if is_imperial else "km/h"
    return f"{_wind_to_display(wind_kmh):.{decimals}f} {unit}"


def _format_precip(precip_mm: float, decimals: int = 1) -> str:
    if is_imperial:
        return f"{_precip_to_display(precip_mm):.{max(2, decimals)}f} in"
    return f"{precip_mm:.{decimals}f} mm"


def _wind_dir_compass(deg: float) -> str:
    points = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
              "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = int(((deg % 360.0) + 11.25) // 22.5) % 16
    return points[idx]


def _format_wind_dir(deg: float) -> str:
    return f"🧭 {_wind_dir_compass(deg)} ({deg:.0f}°)"


def _parse_hour(hour_label: str) -> int:
    try:
        return int(str(hour_label).split(":")[0])
    except Exception:
        return -1


def _resolve_location_now(lat: float, lon: float) -> tuple:
    tz_name = None
    local_now = None

    if _tz_finder is not None:
        try:
            tz_name = _tz_finder.timezone_at(lat=lat, lng=lon)
        except Exception:
            tz_name = None

    if tz_name:
        try:
            local_now = datetime.now(ZoneInfo(tz_name))
        except Exception:
            local_now = None

    if local_now is None:
        approx_offset = int(round(lon / 15.0))
        local_now = datetime.now(timezone.utc) + timedelta(hours=approx_offset)
        if not tz_name:
            tz_name = f"UTC{approx_offset:+d}"

    return local_now, tz_name


_loc_lat = custom_lat if custom_lat is not None else REGION_COORDS.get(selected_region, (52.8, -1.2))[0]
_loc_lon = custom_lon if custom_lon is not None else REGION_COORDS.get(selected_region, (52.8, -1.2))[1]
_local_now, _local_tz_label = _resolve_location_now(_loc_lat, _loc_lon)
current_hour = _local_now.hour
current_hour_str = f"{current_hour:02d}:00"

_hourly_raw = forecast.get("hourly", [])
_hourly_by_hour = {}
for _h in _hourly_raw:
    _hour_num = _parse_hour(_h.get("hour", ""))
    if 0 <= _hour_num <= 23:
        _hourly_by_hour[_hour_num] = _h

# Rolling 24h window from local current hour.
rolling_hourly = []
for _offset in range(24):
    _target_hour = (current_hour + _offset) % 24
    _entry = _hourly_by_hour.get(_target_hour)
    if _entry:
        rolling_hourly.append({
            **_entry,
            "hour": f"{_target_hour:02d}:00",
            "next_day": (_offset + current_hour) >= 24,
        })

visible_hourly = rolling_hourly if rolling_hourly else _hourly_raw
if not visible_hourly:
    visible_hourly = []

# ---------------------------------------------------------------------------
# Sidebar — status (rendered after forecast loaded)
# ---------------------------------------------------------------------------
with st.sidebar:
    try:
        import onnxruntime as _ort
        _gpu = "CUDAExecutionProvider" in _ort.get_available_providers()
    except ImportError:
        _gpu = False

    gpu_colour = "#4ade80" if _gpu else "#fbbf24"
    gpu_label  = "⚡ GPU Active" if _gpu else "💻 CPU Mode"

    _meta_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "input_data", "era5_meta.json")
    if os.path.exists(_meta_path):
        with open(_meta_path) as _f:
            _meta = json.load(_f)
        _src = f"ERA5 {_meta['date']} {_meta['time']} UTC"
    else:
        _src = "Mock data"

    try:
        from pangu_inference import count_cached_steps
        _cached = count_cached_steps()
        _weekly_src = f"{_cached}/13 steps cached" if _cached else "No cache"
    except Exception:
        _weekly_src = "Mock data"

    st.markdown(
        f'<div style="font-weight:700; color:{gpu_colour}; font-size:13px; margin-bottom:2px;">{gpu_label}</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Model: Pangu-Weather {model_hours}h ONNX")
    st.caption(f"Source: {_src}")
    st.caption(f"Weekly: {_weekly_src}")
    st.caption(f"Updated: {datetime.now().strftime('%d %b %Y, %H:%M')}")

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

if high_contrast:
    _t["text_primary"] = "#ffffff"
    _t["text_secondary"] = "rgba(255,255,255,0.96)"
    _t["card_border"] = "rgba(255,255,255,0.5)"
    _t["accent"] = "#a5f3fc"

card_hover_transform = "none" if reduce_motion else "translateY(-4px)"
pulse_animation = "none" if reduce_motion else "pulse 2s infinite"
particle_animation = "none" if reduce_motion else "float linear infinite"
particle_display = "none" if reduce_motion else "block"

_vision_profile = COLOR_VISION_PROFILES.get(color_vision_mode, COLOR_VISION_PROFILES["Standard (Default)"])
active_verdict_glass = _vision_profile["verdict_glass"]
active_marker_colours = _vision_profile["marker_colours"]
active_rain_colours = _vision_profile["rain_colours"]

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

/* Font — applied broadly but NOT with !important so icon fonts can override */
*, *::before, *::after {{ font-family: 'DM Sans', sans-serif; }}
/* Re-apply with !important only to text-bearing elements */
body, p, h1, h2, h3, h4, h5, h6, label, button, input, select, textarea,
[data-testid="stMarkdownContainer"], [data-testid="stSidebar"],
.block-container {{ font-family: 'DM Sans', sans-serif !important; }}
html, body, [class*="st-"] {{ font-size: {font_size_px}px !important; }}

/* Sidebar dropdowns — selected value text always visible and correctly sized */
[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] [data-baseweb="select"] input {{
  color: white !important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] span:not([class*="material"]) {{
  color: white !important;
  font-size: 13px !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}}

/* Restore Material icon fonts — span-qualified selector beats the broad rules above */
span.material-icons,
span.material-icons-round,
span.material-icons-outlined,
span.material-symbols-outlined,
span.material-symbols-rounded,
i.material-icons,
i.material-icons-round {{
  font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
  font-style: normal !important;
  font-weight: normal !important;
  line-height: 1 !important;
  letter-spacing: normal !important;
  text-transform: none !important;
  display: inline-block !important;
  white-space: nowrap !important;
  direction: ltr !important;
}}

/* Screen-reader only utility */
.sr-only {{
  position: absolute !important;
  width: 1px !important;
  height: 1px !important;
  padding: 0 !important;
  margin: -1px !important;
  overflow: hidden !important;
  clip: rect(0, 0, 0, 0) !important;
  white-space: nowrap !important;
  border: 0 !important;
}}

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
  border: 1px solid rgba(255,255,255,0.18) !important;
  border-radius: 12px !important;
  transition: border-color 0.2s;
}}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div:hover {{
  border-color: rgba(255,255,255,0.4) !important;
}}
/* Force ALL text inside every selectbox control to white */
[data-testid="stSidebar"] [data-baseweb="select"] * {{
  color: white !important;
}}
/* Restore icon arrow to slightly dimmed white (not pure) */
[data-testid="stSidebar"] [data-baseweb="select"] span.material-symbols-rounded {{
  color: rgba(255,255,255,0.6) !important;
  font-size: 20px !important;
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
[data-testid="stSidebar"] .sb-section {{
  font-size: 11px;
  letter-spacing: 1.8px;
  text-transform: uppercase;
  color: rgba(255,255,255,0.62);
  font-weight: 700;
  margin: 8px 0 4px;
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
  margin-bottom: 0.25rem !important;
}}
[data-testid="stSidebar"] .stTextInput,
[data-testid="stSidebar"] .stSelectbox,
[data-testid="stSidebar"] .stCheckbox {{
  margin-bottom: 0.65rem !important;
}}
[data-testid="stSidebar"] .stSlider {{
  margin-bottom: 1.4rem !important;
  padding-bottom: 0.5rem !important;
}}
[data-testid="stSidebar"] details {{
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 12px !important;
  padding: 8px 12px 14px !important;
  margin-bottom: 0.65rem !important;
}}
[data-testid="stSidebar"] details summary {{
  color: rgba(255,255,255,0.9) !important;
  font-weight: 600 !important;
  display: flex !important;
  align-items: center !important;
  gap: 8px !important;
  padding-bottom: 4px !important;
}}

/* Minimum 44px touch targets in sidebar */
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] [data-testid="stTextInput"] input,
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div,
[data-testid="stSidebar"] [data-baseweb="select"] > div {{
  min-height: 44px !important;
}}
/* Round slider thumb */
[data-testid="stSidebar"] [role="slider"] {{
  width: 20px !important;
  height: 20px !important;
  min-height: unset !important;
  border-radius: 50% !important;
  box-sizing: border-box !important;
}}
[data-testid="stSidebar"] [data-testid="stTextInput"] input {{
  padding-top: 10px !important;
  padding-bottom: 10px !important;
}}

/* Strong keyboard focus rings */
[data-testid="stSidebar"] button:focus-visible,
[data-testid="stSidebar"] input:focus-visible,
[data-testid="stSidebar"] [data-baseweb="select"] *:focus-visible,
[data-testid="stSidebar"] [role="slider"]:focus-visible,
.leaflet-control-layers-selector:focus-visible,
.leaflet-control-layers-toggle:focus-visible,
.leaflet-control-zoom a:focus-visible {{
  outline: 3px solid #facc15 !important;
  outline-offset: 2px !important;
  box-shadow: 0 0 0 4px rgba(250,204,21,0.35) !important;
}}

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
  transform: {card_hover_transform};
  box-shadow: 0 16px 40px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255,255,255,0.15);
}}

/* Equal-height decision cards */
.decision-card {{
  height: 250px !important;
  overflow: hidden !important;
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

/* Avoid card pulse */
@keyframes pulse {{
  0%, 100% {{ opacity: 1;   }}
  50%       {{ opacity: 0.6; }}
}}
.avoid-pulse {{ animation: {pulse_animation}; }}

/* Floating particles */
@keyframes float {{
  0%, 100% {{ transform: translateY(0px) rotate(0deg);   opacity: 0.25; }}
  50%       {{ transform: translateY(-20px) rotate(180deg); opacity: 0.55; }}
}}
.particle {{
  position: fixed;
  border-radius: 50%;
  background: rgba(255,255,255,0.35);
  animation: {particle_animation};
  pointer-events: none;
  z-index: 0;
  display: {particle_display};
}}

/* Skip link */
.skip-link {{
  position: absolute;
  left: -9999px;
  top: 0;
  z-index: 2000;
  padding: 10px 14px;
  border-radius: 10px;
  background: rgba(10, 25, 41, 0.95);
  color: #7dd3fc;
  border: 1px solid rgba(125, 211, 252, 0.5);
  text-decoration: none;
  font-weight: 600;
}}
.skip-link:focus {{
  left: 16px;
  top: 12px;
}}

/* Responsive layout */
@media (max-width: 1200px) {{
  .block-container {{
    max-width: 1000px !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
  }}
}}

@media (max-width: 900px) {{
  .block-container {{
    padding-top: 0.75rem !important;
    padding-left: 0.75rem !important;
    padding-right: 0.75rem !important;
  }}
  div[data-testid="stHorizontalBlock"] {{
    flex-direction: column !important;
    gap: 10px !important;
  }}
  div[data-testid="stHorizontalBlock"] > div {{
    width: 100% !important;
    min-width: 100% !important;
    flex: 1 1 100% !important;
  }}
  .glass-card {{
    padding: 16px !important;
    border-radius: 16px !important;
    margin-bottom: 12px !important;
  }}
  .hero-temp {{
    font-size: 68px !important;
  }}
  .hero-kicker {{
    font-size: 12px !important;
    letter-spacing: 2px !important;
  }}
  .hero-condition {{
    font-size: 19px !important;
  }}
}}

@media (max-width: 560px) {{
  .glass-card {{
    padding: 14px !important;
    border-radius: 14px !important;
  }}
  .hourly-strip {{
    padding: 0 8px 6px !important;
  }}
  .hour-cell {{
    min-width: 56px !important;
    padding: 8px 4px !important;
  }}
  .hero-temp {{
    font-size: 56px !important;
  }}
  .hero-kicker {{
    font-size: 11px !important;
  }}
  .hero-condition {{
    font-size: 17px !important;
  }}
  .hero-stats {{
    font-size: 13px !important;
  }}
  .decision-card {{
    height: auto !important;
    min-height: 210px !important;
  }}
}}

/* Success/spinner text colour fix */
[data-testid="stAlert"] {{ background: rgba(255,255,255,0.1) !important; border-radius: 12px !important; }}

/* Respect OS-level reduced motion preference */
@media (prefers-reduced-motion: reduce) {{
  * {{
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }}
  .glass-card:hover {{
    transform: none !important;
  }}
  .avoid-pulse,
  .particle {{
    animation: none !important;
  }}
  .particle {{
    display: none !important;
  }}
}}
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

st.markdown('<a href="#main-content" class="skip-link">Skip to main content</a>', unsafe_allow_html=True)
st.markdown('<div id="main-content" tabindex="-1"></div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sr-only" role="status" aria-live="polite" aria-atomic="true">{_refresh_status_message}</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero section
# ---------------------------------------------------------------------------
temp_c     = forecast["temperature_c"]
wind_kmh   = forecast["wind_speed_kmh"]
precip_pct = forecast["precipitation_probability"] * 100
humidity_pct = float(forecast.get("humidity_percent", max(20.0, min(100.0, 45.0 + precip_pct * 0.45))))
frost_tonight = "❄️ Frost Risk Tonight" if forecast["frost_risk"] else "✓ No Frost Risk"
display_temp_major = _temp_to_display(temp_c)
display_temp_unit = "F" if is_imperial else "C"
display_wind = _format_wind(wind_kmh, 0)
_window_temps = [
    float(h.get("temp_c"))
    for h in visible_hourly
    if isinstance(h, dict) and h.get("temp_c") is not None
]
_window_high_c = max(_window_temps) if _window_temps else temp_c
_window_low_c = min(_window_temps) if _window_temps else temp_c
_window_high = _format_temp(_window_high_c, 0)
_window_low = _format_temp(_window_low_c, 0)

st.markdown(f"""
<div class="hero-wrap" style="text-align:center; padding:36px 20px 20px;" role="banner">
  <div class="hero-kicker" style="font-size:13px; font-weight:500; letter-spacing:3px;
              text-transform:uppercase; color:var(--text-secondary);
              margin-bottom:10px;">
    🌾 FARMCAST &nbsp;·&nbsp; {resolved_location_name.upper()}
  </div>
  <div class="hero-temp" style="font-family:'DM Serif Display', Georgia, serif;
              font-size:96px; font-weight:400; color:var(--text-primary);
              line-height:1; margin-bottom:8px;
              text-shadow: 0 4px 40px rgba(0,0,0,0.3);">
    {display_temp_major:.0f}°{display_temp_unit}
  </div>
  <div class="hero-condition" style="font-size:22px; font-weight:300; color:var(--text-secondary);
              margin-bottom:10px; letter-spacing:0.02em;">
    {condition_label}
  </div>
  <div style="font-size:15px; color:var(--text-primary); font-weight:600; letter-spacing:0.01em; margin-bottom:4px;">
    Next 24h: H {_window_high} &nbsp;·&nbsp; L {_window_low}
  </div>
  <div class="hero-stats" style="font-size:15px; color:var(--text-secondary); letter-spacing:0.01em;">
    Wind {display_wind} &nbsp;·&nbsp;
    Humidity {humidity_pct:.0f}% &nbsp;·&nbsp;
    Rain {precip_pct:.0f}% &nbsp;·&nbsp;
    {frost_tonight}
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hourly Forecast Strip
# ---------------------------------------------------------------------------
# `visible_hourly`, `current_hour_str`, `_local_now`, and `_local_tz_label`
# are prepared above so this strip always represents a rolling next-24h view.


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
for h in visible_hourly:
    is_now = h["hour"] == current_hour_str
    hl_bg  = "rgba(255,255,255,0.16)" if is_now else "transparent"
    fw     = "700" if is_now else "400"
    day_suffix = " +1d" if h.get("next_day") else ""
    h_temp = _temp_to_display(h["temp_c"])
    h_wind = _wind_to_display(h["wind_kmh"])
    h_temp_unit = "F" if is_imperial else "C"
    h_wind_unit = "mph" if is_imperial else "km/h"
    
    # NO INDENTATION ALLOWED: Streamlit will markdown-parse indented blocks as <pre><code>!
    cell = (
        f'<div class="hour-cell" style="min-width:66px; display:flex; flex-direction:column; align-items:center; '
        f'padding:10px 6px; border-radius:16px; background:{hl_bg}; flex-shrink:0;">'
        f'<div style="font-size:12px; color:var(--text-secondary); font-weight:{fw}; margin-bottom:4px;">{h["hour"]}{day_suffix}</div>'
        f'<div style="font-size:26px; line-height:1; margin-bottom:6px;" aria-hidden="true">{_hour_icon(h)}</div>'
        f'<div style="font-size:16px; font-weight:600; color:var(--text-primary);">{h_temp:.0f}°{h_temp_unit}</div>'
        f'<div style="font-size:11px; color:var(--accent); margin-top:3px;">{h_wind:.0f} {h_wind_unit}</div>'
        f'</div>'
    )
    hour_cells += cell
    
st.markdown(f"""
<div class="glass-card" style="padding:16px 0 10px;" role="region" aria-label="Hourly forecast">
<div style="padding:0 20px 10px; font-size:11px; letter-spacing:2px; text-transform:uppercase; color:var(--text-secondary); font-weight:500;">
Upcoming Forecast
</div>
<div style="padding:0 20px 8px; font-size:11px; color:var(--text-secondary);">
Local time {_local_now.strftime('%H:%M')} ({_local_tz_label})
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
def _compact_reason(reason: str, max_words: int = 14) -> str:
    text = reason
    replacements = {
        " before committing machinery.": " before machinery use.",
        " is too high.": ".",
        " is available.": " is available.",
        " with no frost — suitable for planting.": " with no frost.",
        " and manageable wind — outdoor grazing is fine today.": " and manageable wind.",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(".,;:") + "…"


_wind_dir_label = _format_wind_dir(float(forecast["wind_direction_deg"]))
CARD_METRICS = [
    f"Wind {_format_wind(forecast['wind_speed_kmh'], 0)} · {_wind_dir_label}",
    f"Rain {_format_precip(forecast['precipitation_mm'], 1)} · Humidity {humidity_pct:.0f}%",
    f"{'Frost risk' if forecast['frost_risk'] else 'No frost'} · Humidity {humidity_pct:.0f}%",
    f"Temp {_format_temp(forecast['temperature_c'], 1)} · Humidity {humidity_pct:.0f}%",
]

CARD_HIGHLIGHTS = [
    f"{_format_wind(forecast['wind_speed_kmh'], 0)} · {humidity_pct:.0f}% RH",
    f"{_format_precip(forecast['precipitation_mm'], 1)}",
    f"{'Frost risk' if forecast['frost_risk'] else 'No frost'}",
    f"{_format_temp(forecast['temperature_c'], 1)}",
]

card_cols = st.columns(2)
for idx, card in enumerate(decisions):
    col         = card_cols[idx % 2]
    verdict     = card["verdict"]
    vg          = active_verdict_glass[verdict]
    extra_class = "avoid-pulse" if verdict == AVOID else ""

    with col:
        st.markdown(f"""
        <div class="glass-card decision-card {extra_class}"
             style="border-left:3px solid {vg['border']};
                    min-height:182px;
                    display:flex; justify-content:space-between;
                    align-items:stretch; padding:18px 20px;"
             role="region"
             aria-label="{card['activity']} — {verdict}">
          <div style="flex:1; min-width:0; padding-right:12px;
                      display:flex; flex-direction:column; gap:6px;">
            <div style="font-size:11px; letter-spacing:2px; text-transform:uppercase;
                        color:var(--text-secondary);">
              {card['icon']} &nbsp;{card['activity']}
            </div>
            <div style="font-size:30px; font-weight:700; color:var(--text-primary);
                        line-height:1.05; letter-spacing:-0.01em;
                        min-height:2.2em; display:flex; align-items:flex-end;">
              {CARD_HIGHLIGHTS[idx]}
            </div>
            <div style="font-size:13px; color:var(--text-primary); line-height:1.35; opacity:0.92;
                        display:-webkit-box; -webkit-line-clamp:2;
                        -webkit-box-orient:vertical; overflow:hidden;">
              {_compact_reason(card['reason'])}
            </div>
            <div style="font-size:16px; color:var(--text-secondary); margin-top:auto; font-weight:600;">
              {CARD_METRICS[idx]}
            </div>
            <div style="font-size:12px; color:var(--text-primary);">
              Status: {vg['icon']} {vg['label']}
            </div>
          </div>
          <div style="flex-shrink:0; display:flex; align-items:center;">
            <span style="font-size:13px; font-weight:600; padding:6px 14px;
                         border-radius:20px; background:{vg['bg']};
                         color:{vg['text']}; white-space:nowrap;
                         border:1px solid {vg['border']}44;"
                  role="status" aria-label="Verdict: {verdict}">
              {vg['icon']} {vg['label']}
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
          Temperature dropping to {_format_temp(forecast['temperature_c'], 1)}.
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
    {advisory_text}
  </div>
  <div style="font-size:11px; color:var(--text-secondary); margin-top:12px; text-align:right;">
    Powered by Ollama &nbsp;·&nbsp; {OLLAMA_MODEL}
  </div>
</div>
""", unsafe_allow_html=True)

if enable_tts:
    tts_report = (
        f"FarmCast update for {resolved_location_name}. "
        f"Temperature {temp_c:.1f} degrees Celsius. "
        f"Wind {wind_kmh:.0f} kilometres per hour. "
        f"Rain probability {precip_pct:.0f} percent. "
        f"Frost risk is {'present' if forecast['frost_risk'] else 'not present'}. "
        f"Advisory: {advisory_text}"
    )
    safe_tts_report = (
        tts_report
        .replace("\\", " ")
        .replace("'", "")
        .replace('"', "")
        .replace("`", "")
        .replace("\n", " ")
        .strip()
    )
    components.html(
        f"""
        <button
          onclick="
            window.speechSynthesis.cancel();
            const msg = new SpeechSynthesisUtterance('{safe_tts_report}');
            msg.lang = 'en-GB';
            msg.rate = 0.95;
            window.speechSynthesis.speak(msg);
          "
          aria-label="Read forecast aloud"
          style="
            width:100%;
            border:none;
            border-radius:12px;
            padding:12px 14px;
            background:rgba(125,211,252,0.24);
            color:white;
            font-weight:600;
            cursor:pointer;
          ">
          🔊 Listen to forecast
        </button>
        """,
        height=62,
    )

# ---------------------------------------------------------------------------
# Interactive Farm Controls
# ---------------------------------------------------------------------------
st.markdown(
    "<div style='font-size:11px; letter-spacing:2px; text-transform:uppercase; "
    "color:var(--text-secondary); font-weight:500; margin-bottom:6px;'>"
    "🎮 Interactive Farm Controls</div>",
    unsafe_allow_html=True,
)
st.caption("Test field actions against the current forecast conditions.")


def _show_control_gif(path: str, caption: str) -> None:
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing asset: {path}")


if "tractor_status" not in st.session_state:
    st.session_state["tractor_status"] = "Idle"
if "sprayer_status" not in st.session_state:
    st.session_state["sprayer_status"] = "Idle"
if "livestock_status" not in st.session_state:
    st.session_state["livestock_status"] = "Idle"

ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)

with ctrl_col1:
    st.markdown("#### 🚜 Tractor")
    if st.button("Dispatch Tractor", key="tractor_dispatch_btn", use_container_width=True):
        with st.spinner("Checking soil conditions..."):
            time.sleep(0.8)
            if forecast["precipitation_mm"] >= 2.0:
                st.session_state["tractor_status"] = "Blocked: wet soil"
                st.error(f"Blocked. Rain is {_format_precip(forecast['precipitation_mm'], 1)}.")
                _show_control_gif("tractor_fail.gif", "Too wet for safe field work.")
            else:
                st.session_state["tractor_status"] = "Running: field work active"
                st.success("Tractor deployed successfully.")
                _show_control_gif("tractor_success.gif", "Ground conditions are suitable.")
    st.info(f"Status: {st.session_state['tractor_status']}")

with ctrl_col2:
    st.markdown("#### 🚁 Sprayer Drone")
    if st.button("Deploy Drone", key="drone_deploy_btn", use_container_width=True):
        with st.spinner("Checking drift and wash-off risk..."):
            time.sleep(0.8)
            if forecast["wind_speed_kmh"] >= 15.0:
                st.session_state["sprayer_status"] = "Blocked: high wind"
                st.warning(f"Blocked. Wind is {_format_wind(forecast['wind_speed_kmh'], 0)}.")
                _show_control_gif("drone_fail.gif", "Wind is too high for precision spraying.")
            elif forecast["precipitation_mm"] > 0.0:
                st.session_state["sprayer_status"] = "Blocked: rain risk"
                st.warning("Blocked. Rain may wash off spray.")
                _show_control_gif("drone_fail.gif", "Rain risk is too high for spraying.")
            else:
                st.session_state["sprayer_status"] = "Running: spraying active"
                st.success("Drone deployed successfully.")
                _show_control_gif("drone_success.gif", "Conditions are suitable for spraying.")
    st.info(f"Status: {st.session_state['sprayer_status']}")

with ctrl_col3:
    st.markdown("#### 🐄 Livestock")
    if st.button("Check Pasture", key="pasture_check_btn", use_container_width=True):
        with st.spinner("Checking overnight exposure risk..."):
            time.sleep(0.8)
            if forecast["frost_risk"]:
                st.session_state["livestock_status"] = "Alert: frost risk"
                st.error("Move vulnerable livestock indoors.")
                _show_control_gif("cow_fail.gif", "Frost risk detected for overnight grazing.")
            else:
                st.session_state["livestock_status"] = "Safe: outdoor grazing"
                st.success("Pasture conditions are safe.")
                _show_control_gif("cow_success.gif", "Livestock can remain outdoors.")
    st.info(f"Status: {st.session_state['livestock_status']}")

# ---------------------------------------------------------------------------
# Global Map — layered atmospheric view
# ---------------------------------------------------------------------------
st.markdown(
    "<div style='font-size:11px; letter-spacing:2px; text-transform:uppercase; "
    "color:var(--text-secondary); font-weight:500; margin-bottom:6px;'>"
    "🌍 Global Weather Layers</div>",
    unsafe_allow_html=True,
)
st.caption("Toggle layers for active forecast, UK risk, wind, rain, and lat/lon grid.")


@st.cache_data(ttl=300)
def _regional_layer_data(_model_hours: int) -> dict:
    result = {}
    for reg in REGIONS:
        fc = get_forecast(reg, model_hours=_model_hours)
        dec = get_all_decisions(fc)
        result[reg] = {"forecast": fc, "verdict": get_worst_verdict(dec)}
    return result


def _build_global_grid(layer_obj) -> None:
    # Latitude lines
    for lat_line in range(-60, 61, 30):
        folium.PolyLine(
            locations=[(lat_line, -180), (lat_line, 180)],
            color="rgba(255,255,255,0.28)",
            weight=1,
            opacity=0.45,
            tooltip=f"Lat {lat_line}°",
        ).add_to(layer_obj)
    # Longitude lines
    for lon_line in range(-150, 181, 30):
        folium.PolyLine(
            locations=[(-80, lon_line), (80, lon_line)],
            color="rgba(255,255,255,0.24)",
            weight=1,
            opacity=0.4,
            tooltip=f"Lon {lon_line}°",
        ).add_to(layer_obj)


region_layers = _regional_layer_data(model_hours)

active_lat = custom_lat if custom_lat is not None else REGION_COORDS.get(selected_region, (52.8, -1.2))[0]
active_lon = custom_lon if custom_lon is not None else REGION_COORDS.get(selected_region, (52.8, -1.2))[1]
active_label = resolved_location_name if custom_lat is not None else selected_region

global_map = folium.Map(
    location=[54.5, -3.0],
    zoom_start=5,
    min_zoom=2,
    tiles=None,
    control_scale=True,
    prefer_canvas=True,
)

# Base maps
folium.TileLayer("CartoDB dark_matter", name="Dark", control=True, show=True).add_to(global_map)
folium.TileLayer("CartoDB positron", name="Light", control=True, show=False).add_to(global_map)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles © Esri",
    name="Satellite",
    control=True,
    show=False,
).add_to(global_map)

# Active location layer
active_layer = folium.FeatureGroup(name="Active Forecast", show=True)
folium.Marker(
    location=(active_lat, active_lon),
    tooltip=f"{active_label} (active)",
    popup=folium.Popup(
        f"<b>{active_label}</b><br>"
        f"Temp: {_format_temp(forecast['temperature_c'], 1)}<br>"
        f"Wind: {_format_wind(forecast['wind_speed_kmh'], 1)}<br>"
        f"Humidity: {humidity_pct:.0f}%<br>"
        f"Rain prob: {forecast['precipitation_probability']*100:.0f}%",
        max_width=220,
    ),
    icon=folium.Icon(color="blue", icon="cloud", prefix="fa"),
).add_to(active_layer)
folium.Circle(
    location=(active_lat, active_lon),
    radius=120000 + int(forecast["precipitation_probability"] * 160000),
    color="#7dd3fc",
    weight=2,
    fill=True,
    fill_opacity=0.12,
).add_to(active_layer)
active_layer.add_to(global_map)

# UK verdict layer
risk_layer = folium.FeatureGroup(name="UK Regional Risk", show=True)
for reg, coords in REGION_COORDS.items():
    reg_verdict = region_layers[reg]["verdict"]
    marker_colour = active_marker_colours[reg_verdict]
    is_active = reg == selected_region and custom_lat is None
    folium.Marker(
        location=coords,
        tooltip=reg,
        popup=folium.Popup(
            f"<b>{reg}</b><br>Verdict: {reg_verdict.title()}",
            max_width=180,
        ),
        icon=folium.Icon(
            color=marker_colour,
            icon="leaf" if is_active else "info-sign",
            prefix="glyphicon",
        ),
    ).add_to(risk_layer)
risk_layer.add_to(global_map)

# UK wind layer
wind_layer = folium.FeatureGroup(name="UK Wind Layer", show=False)
for reg, coords in REGION_COORDS.items():
    reg_fc = region_layers[reg]["forecast"]
    wind_kmh = reg_fc["wind_speed_kmh"]
    folium.CircleMarker(
        location=coords,
        radius=max(5, min(16, 4 + int(wind_kmh / 6))),
        color="#7dd3fc",
        weight=1.5,
        fill=True,
        fill_color="#38bdf8",
        fill_opacity=0.5,
        tooltip=f"{reg}: Wind {_format_wind(wind_kmh, 0)}",
    ).add_to(wind_layer)
wind_layer.add_to(global_map)

# UK rain layer
rain_layer = folium.FeatureGroup(name="UK Rain Probability", show=True)
for reg, coords in REGION_COORDS.items():
    reg_fc = region_layers[reg]["forecast"]
    rain_prob = reg_fc["precipitation_probability"]
    if rain_prob < 0.33:
        rain_color = active_rain_colours["low"]
    elif rain_prob < 0.66:
        rain_color = active_rain_colours["mid"]
    else:
        rain_color = active_rain_colours["high"]
    folium.CircleMarker(
        location=coords,
        radius=max(4, min(15, 4 + int(rain_prob * 14))),
        color=rain_color,
        weight=1.5,
        fill=True,
        fill_color=rain_color,
        fill_opacity=0.45,
        tooltip=f"{reg}: Rain {rain_prob*100:.0f}%",
    ).add_to(rain_layer)
rain_layer.add_to(global_map)

# Lat/Lon grid layer
grid_layer = folium.FeatureGroup(name="Lat/Lon Grid", show=True)
_build_global_grid(grid_layer)
grid_layer.add_to(global_map)

folium.LayerControl(collapsed=True).add_to(global_map)
global_map.get_root().header.add_child(Element("""
<style>
.leaflet-control-layers-selector:focus-visible,
.leaflet-control-layers-toggle:focus-visible,
.leaflet-control-zoom a:focus-visible {
  outline: 3px solid #facc15 !important;
  outline-offset: 2px !important;
  box-shadow: 0 0 0 4px rgba(250,204,21,0.35) !important;
}
</style>
"""))

map_result = st_folium(
    global_map,
    width="100%",
    height=500,
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
    f"FarmCast &nbsp;·&nbsp; Pangu-Weather + Ollama ({OLLAMA_MODEL})"
    "</div>",
    unsafe_allow_html=True,
)
