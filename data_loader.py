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

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
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
    model_path: str = "../pangu_weather_1.onnx",
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
    except Exception as exc:  # noqa: BLE001
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

    # Precipitation heuristic: low pressure → higher rain probability
    precip_probability = round(min(1.0, max(0.0, (mslp - 99000.0) / -3000.0)), 3)
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
