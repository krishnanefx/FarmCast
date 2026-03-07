"""
data_loader.py — FarmCast weather data abstraction layer.

Real Pangu-Weather inference via the 1-hour ONNX model.
Falls back to regional mock data if the model or input files are not found,
so the Streamlit app never crashes during development or demo.

Expected file layout (relative to the repo root):
  ../pangu_weather_1.onnx
  ../input_data/input_surface.npy   shape (4, 721, 1440) float32
  ../input_data/input_upper.npy     shape (5, 13, 721, 1440) float32
  ../output_data/                   created automatically for cache
"""

import math
import os
import random
import warnings
from datetime import datetime, timedelta

import numpy as np

# ---------------------------------------------------------------------------
# Region grid coordinates
# Latitude index  = int((90 - lat) / 0.25)
# Longitude index = int(lon_positive / 0.25)  where lon_positive = lon % 360
# ---------------------------------------------------------------------------
_REGION_GRID = {
    "East Midlands":     {"lat": 52.8, "lon":   1.2},
    "Yorkshire":         {"lat": 53.8, "lon": 358.5},   # -1.5
    "East Anglia":       {"lat": 52.2, "lon":   0.9},
    "Southwest England": {"lat": 50.7, "lon": 356.5},   # -3.5
    "Wales":             {"lat": 52.3, "lon": 356.0},   # -4.0
    "Scotland":          {"lat": 56.8, "lon": 356.0},   # -4.0
    "Northern Ireland":  {"lat": 54.7, "lon": 353.5},   # -6.5
}


def _grid_index(lat: float, lon: float) -> tuple[int, int]:
    """Return (lat_idx, lon_idx) for the 0.25-degree global grid."""
    lon_pos = lon % 360.0
    lat_idx = int((90.0 - lat) / 0.25)
    lon_idx = int(lon_pos / 0.25)
    # Clamp to valid range
    lat_idx = max(0, min(lat_idx, 720))
    lon_idx = max(0, min(lon_idx, 1439))
    return lat_idx, lon_idx


# ---------------------------------------------------------------------------
# Pangu-Weather inference (cached to output_data/pangu_output.npz)
# ---------------------------------------------------------------------------

def _run_inference(model_path: str, input_dir: str, output_dir: str) -> tuple:
    """
    Load input arrays, run the Pangu-Weather 1-hour ONNX model, and return
    (output_upper, output_surface) as numpy arrays.

    Uses CUDAExecutionProvider when available, falls back to CPU.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is not installed. Run: pip install onnxruntime-gpu"
        ) from exc

    surface_path = os.path.join(input_dir, "input_surface.npy")
    upper_path   = os.path.join(input_dir, "input_upper.npy")

    input_surface = np.load(surface_path).astype(np.float32)   # (4, 721, 1440)
    input_upper   = np.load(upper_path).astype(np.float32)     # (5, 13, 721, 1440)

    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif "CoreMLExecutionProvider" in available_providers:
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    print(f"[FarmCast] Running Pangu-Weather inference with: {providers[0]}")

    sess = ort.InferenceSession(model_path, providers=providers)

    outputs = sess.run(
        ["output", "output_surface"],
        {"input": input_upper, "input_surface": input_surface},
    )

    output_upper   = outputs[0]   # (5, 13, 721, 1440)
    output_surface = outputs[1]   # (4, 721, 1440)

    return output_upper, output_surface


def _load_or_run(model_path: str, input_dir: str, output_dir: str,
                 force_refresh: bool) -> tuple:
    """
    Return (output_upper, output_surface), loading from cache when possible.
    Saves inference results to output_dir/pangu_output.npz for reuse.
    """
    os.makedirs(output_dir, exist_ok=True)
    cache_path = os.path.join(output_dir, "pangu_output.npz")

    if not force_refresh and os.path.exists(cache_path):
        print("[FarmCast] Loading Pangu-Weather output from cache.")
        data = np.load(cache_path)
        return data["output_upper"], data["output_surface"]

    print("[FarmCast] Cache not found or force_refresh=True — running inference.")
    output_upper, output_surface = _run_inference(model_path, input_dir, output_dir)
    np.savez_compressed(cache_path, output_upper=output_upper,
                        output_surface=output_surface)
    print(f"[FarmCast] Inference complete. Results cached to {cache_path}")
    return output_upper, output_surface


# ---------------------------------------------------------------------------
# Diurnal hourly builder (shared by both real and mock paths)
# ---------------------------------------------------------------------------

def _generate_hourly(base_temp: float, base_wind: float,
                     precip_base: float) -> list:
    """
    Build 18 hourly entries (06:00–23:00) with realistic diurnal variation.

    base_temp  — the 18:00 forecast temperature (°C); used as the reference.
    base_wind  — base wind speed (km/h); varies ±20% sinusoidally.
    precip_base — base precipitation probability; varies ±0.1.

    Temperature minimum at 06:00, maximum at 14:00.
    The 18:00 slot sits on the declining side of the diurnal curve.
    """
    hours = [f"{h:02d}:00" for h in range(6, 24)]   # 18 entries
    n = len(hours)
    result = []

    # Map each hour index to a fraction of the diurnal cycle.
    # Hour 6 = 0 h past 06:00, hour 23 = 17 h past 06:00.
    # Peak (14:00) is at index 8 (8 hours after 06:00).
    PEAK_IDX = 8           # index for 14:00
    TEMP_AMPLITUDE = 3.0   # ±3 °C

    # Compute the phase offset so that index 12 (18:00) maps to base_temp.
    # We use: temp(i) = base_temp_shifted + A * sin(pi * i / (n-1))
    # But we want the curve anchored at 18:00 = index 12.
    # Simpler: compute the diurnal factor at index 12, then shift accordingly.

    def _diurnal_factor(i: int) -> float:
        """Normalised [-1, +1] diurnal factor; +1 at PEAK_IDX, -1 at 06:00/23:00."""
        return math.sin(math.pi * i / (2 * PEAK_IDX)) if i <= PEAK_IDX else \
               math.cos(math.pi * (i - PEAK_IDX) / (2 * (n - 1 - PEAK_IDX)))

    factor_at_18 = _diurnal_factor(12)   # index 12 → 18:00
    # Shift base_temp so that temp(18:00) == base_temp
    shifted_base = base_temp - TEMP_AMPLITUDE * factor_at_18

    for i, hour in enumerate(hours):
        factor = _diurnal_factor(i)
        temp_c = shifted_base + TEMP_AMPLITUDE * factor

        # Wind: ±20% sinusoidal, peaking midday
        wind_factor = 1.0 + 0.2 * math.sin(math.pi * i / (n - 1))
        wind_kmh = base_wind * wind_factor

        # Precipitation probability: ±0.1, rising gently into the afternoon
        t = i / (n - 1)
        precip_prob = precip_base + 0.1 * math.sin(math.pi * t)

        result.append({
            "hour": hour,
            "temp_c":      round(max(-20.0, min(45.0, temp_c)), 1),
            "wind_kmh":    round(max(0.0, wind_kmh), 1),
            "precip_prob": round(max(0.0, min(1.0, precip_prob)), 2),
        })

    return result


# ---------------------------------------------------------------------------
# Mock data fallback (original regional profiles)
# ---------------------------------------------------------------------------
_MOCK_PROFILES = {
    "East Midlands":     {"temperature_c":  6.2, "wind_speed_kmh": 22.0, "wind_direction_deg": 247, "precipitation_mm": 1.4, "precipitation_probability": 0.65, "frost_risk": True},
    "Yorkshire":         {"temperature_c":  4.1, "wind_speed_kmh": 31.0, "wind_direction_deg": 268, "precipitation_mm": 2.8, "precipitation_probability": 0.70, "frost_risk": True},
    "East Anglia":       {"temperature_c":  5.8, "wind_speed_kmh": 26.0, "wind_direction_deg": 232, "precipitation_mm": 0.9, "precipitation_probability": 0.42, "frost_risk": True},
    "Southwest England": {"temperature_c": 10.3, "wind_speed_kmh": 19.0, "wind_direction_deg": 218, "precipitation_mm": 3.6, "precipitation_probability": 0.75, "frost_risk": False},
    "Wales":             {"temperature_c":  7.5, "wind_speed_kmh": 34.0, "wind_direction_deg": 225, "precipitation_mm": 4.2, "precipitation_probability": 0.80, "frost_risk": False},
    "Scotland":          {"temperature_c":  2.4, "wind_speed_kmh": 38.0, "wind_direction_deg": 285, "precipitation_mm": 5.1, "precipitation_probability": 0.85, "frost_risk": True},
    "Northern Ireland":  {"temperature_c":  8.7, "wind_speed_kmh": 24.0, "wind_direction_deg": 212, "precipitation_mm": 2.3, "precipitation_probability": 0.68, "frost_risk": False},
}


def _mock_forecast(region: str) -> dict:
    profile = _MOCK_PROFILES.get(region, _MOCK_PROFILES["East Midlands"])
    hourly = _generate_hourly(
        base_temp=profile["temperature_c"],
        base_wind=profile["wind_speed_kmh"],
        precip_base=profile["precipitation_probability"],
    )
    return {
        "region": region,
        "forecast_time": (datetime.now() + timedelta(hours=1)).strftime("%Y-%m-%dT%H:00:00"),
        **profile,
        "hourly": hourly,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_forecast(
    region: str,
    lat: float = None,
    lon: float = None,
    model_hours: int = 1,
    model_path: str = None,
    input_dir:  str = "../input_data",
    output_dir: str = "../output_data",
    force_refresh: bool = False,
) -> dict:
    """
    Return a standardised forecast dict for the given UK region.

    Attempts real Pangu-Weather ONNX inference; falls back to mock data if
    the model or input files are absent.

    Parameters
    ----------
    region        : One of the seven supported UK regions.
    model_path    : Path to pangu_weather_1.onnx (relative to repo root or absolute).
    input_dir     : Directory containing input_surface.npy and input_upper.npy.
    output_dir    : Directory for caching pangu_output.npz.
    force_refresh : Skip cache and re-run inference even if cache exists.

    Returns
    -------
    dict with keys: region, forecast_time, temperature_c, wind_speed_kmh,
    wind_direction_deg, precipitation_mm, precipitation_probability,
    frost_risk, hourly.
    """
    if lat is None or lon is None:
        coords = REGION_COORDS.get(region, REGION_COORDS["East Midlands"])
        lat, lon = coords
        
    if model_path is None:
        model_path = f"/Users/krishnanadaikkappan/Downloads/pangu_weather_{model_hours}.onnx"

    # Resolve paths relative to this file's directory so the app works regardless
    # of the process working directory.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, model_path) if not os.path.isabs(model_path) else model_path
    input_dir  = os.path.join(base_dir, input_dir)  if not os.path.isabs(input_dir)  else input_dir
    output_dir = os.path.join(base_dir, output_dir) if not os.path.isabs(output_dir) else output_dir

    # Check that model and inputs exist before attempting inference
    surface_path = os.path.join(input_dir, "input_surface.npy")
    upper_path   = os.path.join(input_dir, "input_upper.npy")

    if not os.path.exists(model_path) or not os.path.exists(surface_path) or not os.path.exists(upper_path):
        missing = [p for p in [model_path, surface_path, upper_path] if not os.path.exists(p)]
        warnings.warn(
            f"\u26a0\ufe0f Pangu-Weather model not found \u2014 using mock data.\n"
            f"  Missing: {missing}",
            stacklevel=2,
        )
        print(f"[FarmCast] \u26a0\ufe0f  Pangu-Weather model not found \u2014 using mock data. Missing: {missing}")
        return _mock_forecast(region)

    try:
        _, output_surface = _load_or_run(model_path, input_dir, output_dir, force_refresh)
    except (FileNotFoundError, RuntimeError) as exc:
        warnings.warn(
            f"\u26a0\ufe0f Pangu-Weather inference failed ({exc}) \u2014 using mock data.",
            stacklevel=2,
        )
        print(f"[FarmCast] \u26a0\ufe0f  Inference error: {exc} \u2014 falling back to mock data.")
        return _mock_forecast(region)

    # ------------------------------------------------------------------
    # Extract grid-point values for the requested region
    # output_surface shape: (4, 721, 1440)
    # Variables: [0] MSLP (Pa), [1] U10 (m/s), [2] V10 (m/s), [3] T2M (K)
    # ------------------------------------------------------------------
    coords = _REGION_GRID.get(region, _REGION_GRID["East Midlands"])
    lat_idx, lon_idx = _grid_index(coords["lat"], coords["lon"])

    mslp = float(output_surface[0, lat_idx, lon_idx])   # Pa
    u10  = float(output_surface[1, lat_idx, lon_idx])   # m/s
    v10  = float(output_surface[2, lat_idx, lon_idx])   # m/s
    t2m  = float(output_surface[3, lat_idx, lon_idx])   # K

    temperature_c     = round(t2m - 273.15, 2)
    wind_speed_ms     = math.sqrt(u10 ** 2 + v10 ** 2)
    wind_speed_kmh    = round(wind_speed_ms * 3.6, 2)
    wind_direction_deg = round((math.degrees(math.atan2(u10, v10)) + 180.0) % 360.0, 1)

    # Multi-factor precipitation heuristic
    # Low pressure increases rain probability
    mslp_factor = float(np.clip((101325.0 - mslp) / 3000.0, 0.0, 1.0))
    # Westerly winds (225-270°) bring Atlantic moisture to the UK
    wind_dir_rad = math.radians(wind_direction_deg)
    westerly_factor = float(np.clip((math.cos(wind_dir_rad - math.radians(247)) + 1.0) / 2.0, 0.0, 1.0))
    wind_mag_factor = float(np.clip(wind_speed_kmh / 60.0, 0.0, 1.0))
    precip_probability = round(float(np.clip(
        0.5 * mslp_factor + 0.3 * westerly_factor * wind_mag_factor + 0.1,
        0.0, 1.0,
    )), 3)
    precipitation_mm   = round(precip_probability * 3.0, 2)

    frost_risk = temperature_c < 2.0

    hourly = _generate_hourly(
        base_temp=temperature_c,
        base_wind=wind_speed_kmh,
        precip_base=precip_probability,
    )

    forecast_time = (datetime.now() + timedelta(hours=1)).strftime("%Y-%m-%dT%H:00:00")

    return {
        "region":                   region,
        "forecast_time":            forecast_time,
        "temperature_c":            temperature_c,
        "wind_speed_kmh":           wind_speed_kmh,
        "wind_direction_deg":       wind_direction_deg,
        "precipitation_mm":         precipitation_mm,
        "precipitation_probability": precip_probability,
        "frost_risk":               frost_risk,
        "hourly":                   hourly,
    }


# ---------------------------------------------------------------------------
# Weekly forecast helpers
# ---------------------------------------------------------------------------

# Raw lat/lon for each region (used for grid extraction in weekly path)
REGION_COORDS = {
    "East Midlands":     (52.80,  -1.20),
    "Yorkshire":         (53.90,  -1.50),
    "East Anglia":       (52.40,   0.90),
    "Southwest England": (50.70,  -3.50),
    "Wales":             (52.10,  -3.80),
    "Scotland":          (56.50,  -4.00),
    "Northern Ireland":  (54.60,  -6.70),
}


def _latlon_to_idx(lat: float, lon: float) -> tuple:
    """Convert (lat, lon) to (lat_idx, lon_idx) on the 0.25° global grid."""
    lon_pos = lon % 360.0
    lat_idx = int((90.0 - lat) / 0.25)
    lon_idx = int(lon_pos / 0.25)
    lat_idx = max(0, min(lat_idx, 720))
    lon_idx = max(0, min(lon_idx, 1439))
    return lat_idx, lon_idx


def _extract_surface_vars(output_surface: "np.ndarray", lat: float, lon: float) -> dict:
    """
    Extract T2M, U10, V10, MSLP from a Pangu output_surface array at a grid point
    and return a standardised weather dict with precip heuristic.
    """
    lat_idx, lon_idx = _latlon_to_idx(lat, lon)

    mslp = float(output_surface[0, lat_idx, lon_idx])
    u10  = float(output_surface[1, lat_idx, lon_idx])
    v10  = float(output_surface[2, lat_idx, lon_idx])
    t2m  = float(output_surface[3, lat_idx, lon_idx])

    temperature_c      = round(t2m - 273.15, 2)
    wind_speed_ms      = math.sqrt(u10 ** 2 + v10 ** 2)
    wind_speed_kmh     = round(wind_speed_ms * 3.6, 2)
    wind_direction_deg = round((math.degrees(math.atan2(u10, v10)) + 180.0) % 360.0, 1)

    mslp_factor      = float(np.clip((101325.0 - mslp) / 3000.0, 0.0, 1.0))
    wind_dir_rad     = math.radians(wind_direction_deg)
    westerly_factor  = float(np.clip((math.cos(wind_dir_rad - math.radians(247)) + 1.0) / 2.0, 0.0, 1.0))
    wind_mag_factor  = float(np.clip(wind_speed_kmh / 60.0, 0.0, 1.0))
    precip_prob      = round(float(np.clip(
        0.5 * mslp_factor + 0.3 * westerly_factor * wind_mag_factor + 0.1, 0.0, 1.0,
    )), 3)
    precip_mm        = round(precip_prob * 3.0, 2)

    return {
        "temperature_c":             temperature_c,
        "wind_speed_kmh":            wind_speed_kmh,
        "wind_direction_deg":        wind_direction_deg,
        "precipitation_mm":          precip_mm,
        "precipitation_probability": precip_prob,
        "frost_risk":                temperature_c < 2.0,
    }


# Day groupings: (day_offset_from_today, [step_indices])
_DAY_GROUPS = [
    (0, [0, 1, 2, 3]),   # Day 1: 6h,12h,18h,24h
    (1, [4, 5, 6, 7]),   # Day 2: 30h,36h,42h,48h
    (2, [8]),            # Day 3: 72h
    (3, [9]),            # Day 4: 96h
    (4, [10]),           # Day 5: 120h
    (5, [11]),           # Day 6: 144h
    (6, [12]),           # Day 7: 168h
]


def _aggregate_day(step_vars: list) -> dict:
    """Average multiple step dicts into a single daily summary."""
    keys = ["temperature_c", "wind_speed_kmh", "precipitation_mm", "precipitation_probability"]
    out  = {}
    for k in keys:
        out[k] = round(sum(s[k] for s in step_vars) / len(step_vars), 2)
    out["temp_min_c"]  = round(min(s["temperature_c"] for s in step_vars), 2)
    out["temp_max_c"]  = round(max(s["temperature_c"] for s in step_vars), 2)
    out["frost_risk"]  = any(s["frost_risk"] for s in step_vars)
    out["wind_direction_deg"] = step_vars[-1]["wind_direction_deg"]
    return out


def _mock_weekly_forecast(region: str) -> list:
    """
    Generate 7 days of plausible mock daily summaries seeded by region name
    so results are deterministic per region.
    """
    rng    = random.Random(hash(region) & 0xFFFFFFFF)
    profile = _MOCK_PROFILES.get(region, _MOCK_PROFILES["East Midlands"])
    base_t  = profile["temperature_c"]
    base_w  = profile["wind_speed_kmh"]
    base_p  = profile["precipitation_probability"]

    days = []
    today = datetime.now()
    for d in range(7):
        t_offset = rng.uniform(-3.0, 3.0)
        temp_mid  = round(base_t + t_offset, 1)
        temp_min  = round(temp_mid - rng.uniform(1.5, 4.0), 1)
        temp_max  = round(temp_mid + rng.uniform(1.5, 4.0), 1)
        wind      = round(base_w * rng.uniform(0.7, 1.4), 1)
        precip_p  = round(min(1.0, max(0.0, base_p + rng.uniform(-0.2, 0.2))), 2)
        precip_mm = round(precip_p * 3.0, 2)

        days.append({
            "day_offset":                d,
            "date":                      (today + timedelta(days=d)).strftime("%a %d %b"),
            "temperature_c":             temp_mid,
            "temp_min_c":                temp_min,
            "temp_max_c":                temp_max,
            "wind_speed_kmh":            wind,
            "wind_direction_deg":        round(rng.uniform(180, 320), 1),
            "precipitation_mm":          precip_mm,
            "precipitation_probability": precip_p,
            "frost_risk":                temp_min < 2.0,
            "source":                    "mock",
        })
    return days


def get_weekly_forecast(
    region: str,
    model_6h_path:  str = "/Users/krishnanadaikkappan/Downloads/pangu_weather_6.onnx",
    model_24h_path: str = "/Users/krishnanadaikkappan/Downloads/pangu_weather_24.onnx",
    input_dir:      str = "../input_data",
    output_dir:     str = "../output_data",
    force_refresh:  bool = False,
) -> list:
    """
    Return 7 daily weather summaries for the given region using the
    iterative Pangu-Weather chain (8×6h + 5×24h steps).

    Falls back to deterministic mock data if models or input files are absent.

    Each entry in the returned list contains:
        day_offset, date, temperature_c, temp_min_c, temp_max_c,
        wind_speed_kmh, wind_direction_deg, precipitation_mm,
        precipitation_probability, frost_risk, source ("pangu" or "mock")
    """
    try:
        from pangu_inference import run_weekly_inference
    except ImportError as exc:
        warnings.warn(f"pangu_inference not available ({exc}) — using mock weekly data.", stacklevel=2)
        return _mock_weekly_forecast(region)

    try:
        steps = run_weekly_inference(force_refresh=force_refresh)
    except (FileNotFoundError, RuntimeError) as exc:
        warnings.warn(f"Weekly inference unavailable ({exc}) — using mock weekly data.", stacklevel=2)
        print(f"[FarmCast] Weekly inference unavailable: {exc} — falling back to mock data.")
        return _mock_weekly_forecast(region)

    coords = REGION_COORDS.get(region, REGION_COORDS["East Midlands"])
    lat, lon = coords

    # Map step_index → extracted vars dict
    step_map: dict = {}
    for s in steps:
        step_map[s["step"]] = _extract_surface_vars(s["output_surface"], lat, lon)

    today = datetime.now()
    days  = []
    for day_offset, step_indices in _DAY_GROUPS:
        step_vars = [step_map[i] for i in step_indices if i in step_map]
        if not step_vars:
            continue
        agg = _aggregate_day(step_vars)
        days.append({
            "day_offset":                day_offset,
            "date":                      (today + timedelta(days=day_offset)).strftime("%a %d %b"),
            "temperature_c":             agg["temperature_c"],
            "temp_min_c":                agg["temp_min_c"],
            "temp_max_c":                agg["temp_max_c"],
            "wind_speed_kmh":            agg["wind_speed_kmh"],
            "wind_direction_deg":        agg["wind_direction_deg"],
            "precipitation_mm":          agg["precipitation_mm"],
            "precipitation_probability": agg["precipitation_probability"],
            "frost_risk":                agg["frost_risk"],
            "source":                    "pangu",
        })
    return days
