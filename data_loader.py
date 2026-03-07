"""
data_loader.py — FarmCast weather data abstraction layer.

All Pangu-Weather integration lives exclusively in this file.
To swap mock data for real Pangu-Weather output, replace the body of
get_forecast() only — the returned dict schema must remain unchanged.
"""

import math
from datetime import datetime

# ---------------------------------------------------------------------------
# Regional base profiles (mock values — replace with Pangu-Weather output)
# ---------------------------------------------------------------------------
_REGION_PROFILES = {
    "East Midlands": {
        "temperature_c": 6.2,
        "wind_speed_kmh": 22.0,
        "wind_direction_deg": 247,
        "precipitation_mm": 1.4,
        "precipitation_probability": 0.65,
        "frost_risk": True,
        "_temp_amplitude": 3.8,
        "_wind_amplitude": 6.0,
        "_precip_base": 0.55,
    },
    "Yorkshire": {
        "temperature_c": 4.1,
        "wind_speed_kmh": 31.0,
        "wind_direction_deg": 268,
        "precipitation_mm": 2.8,
        "precipitation_probability": 0.70,
        "frost_risk": True,
        "_temp_amplitude": 4.5,
        "_wind_amplitude": 9.0,
        "_precip_base": 0.60,
    },
    "East Anglia": {
        "temperature_c": 5.8,
        "wind_speed_kmh": 26.0,
        "wind_direction_deg": 232,
        "precipitation_mm": 0.9,
        "precipitation_probability": 0.42,
        "frost_risk": True,
        "_temp_amplitude": 4.2,
        "_wind_amplitude": 7.5,
        "_precip_base": 0.35,
    },
    "Southwest England": {
        "temperature_c": 10.3,
        "wind_speed_kmh": 19.0,
        "wind_direction_deg": 218,
        "precipitation_mm": 3.6,
        "precipitation_probability": 0.75,
        "frost_risk": False,
        "_temp_amplitude": 3.0,
        "_wind_amplitude": 5.0,
        "_precip_base": 0.65,
    },
    "Wales": {
        "temperature_c": 7.5,
        "wind_speed_kmh": 34.0,
        "wind_direction_deg": 225,
        "precipitation_mm": 4.2,
        "precipitation_probability": 0.80,
        "frost_risk": False,
        "_temp_amplitude": 3.5,
        "_wind_amplitude": 10.0,
        "_precip_base": 0.72,
    },
    "Scotland": {
        "temperature_c": 2.4,
        "wind_speed_kmh": 38.0,
        "wind_direction_deg": 285,
        "precipitation_mm": 5.1,
        "precipitation_probability": 0.85,
        "frost_risk": True,
        "_temp_amplitude": 5.0,
        "_wind_amplitude": 12.0,
        "_precip_base": 0.78,
    },
    "Northern Ireland": {
        "temperature_c": 8.7,
        "wind_speed_kmh": 24.0,
        "wind_direction_deg": 212,
        "precipitation_mm": 2.3,
        "precipitation_probability": 0.68,
        "frost_risk": False,
        "_temp_amplitude": 3.2,
        "_wind_amplitude": 6.5,
        "_precip_base": 0.60,
    },
}


def _generate_hourly(base_temp: float, base_wind: float, temp_amplitude: float,
                     wind_amplitude: float, precip_base: float) -> list:
    """
    Generate realistic hourly data from 06:00 to 23:00.
    Temperature follows a diurnal curve (coldest at dawn, peaking ~14:00).
    Wind varies with a midday peak. Precipitation probability rises into evening.
    """
    hours = [f"{h:02d}:00" for h in range(6, 24)]  # 06:00 … 23:00 = 18 points
    n = len(hours)
    result = []

    for i, hour in enumerate(hours):
        t = i / (n - 1)  # 0.0 (06:00) → 1.0 (23:00)

        # Diurnal temperature: bell peaking around 55% through the day (~14:00)
        temp_factor = math.sin(math.pi * (t / 0.55)) if t <= 0.55 else math.sin(
            math.pi * (1.0 - (t - 0.55) / 0.90))
        temp_c = base_temp - temp_amplitude * 0.5 + temp_amplitude * max(0.0, temp_factor)

        # Wind: slightly higher midday, settling in evening
        wind_factor = 0.7 + 0.3 * math.sin(math.pi * t * 0.9 + 0.3)
        wind_kmh = base_wind + wind_amplitude * (wind_factor - 0.5)

        # Precipitation probability: rises gently through the afternoon
        precip_prob = precip_base * (0.5 + 0.6 * t)

        result.append({
            "hour": hour,
            "temp_c": round(max(-10.0, min(35.0, temp_c)), 1),
            "wind_kmh": round(max(0.0, wind_kmh), 1),
            "precip_prob": round(max(0.0, min(1.0, precip_prob)), 2),
        })

    return result


def get_forecast(region: str) -> dict:
    """
    Return a standardised forecast dict for the given UK region.

    When Pangu-Weather output is available, replace the body of this function
    with real model inference. The returned schema must not change.

    Parameters
    ----------
    region : str
        One of the seven supported UK regions.

    Returns
    -------
    dict with keys:
        region, forecast_time, temperature_c, wind_speed_kmh,
        wind_direction_deg, precipitation_mm, precipitation_probability,
        frost_risk, hourly (list of dicts with hour/temp_c/wind_kmh/precip_prob)
    """
    profile = _REGION_PROFILES.get(region, _REGION_PROFILES["East Midlands"])

    hourly = _generate_hourly(
        base_temp=profile["temperature_c"],
        base_wind=profile["wind_speed_kmh"],
        temp_amplitude=profile["_temp_amplitude"],
        wind_amplitude=profile["_wind_amplitude"],
        precip_base=profile["_precip_base"],
    )

    return {
        "region": region,
        "forecast_time": datetime.now().strftime("%Y-%m-%dT18:00:00"),
        "temperature_c": profile["temperature_c"],
        "wind_speed_kmh": profile["wind_speed_kmh"],
        "wind_direction_deg": profile["wind_direction_deg"],
        "precipitation_mm": profile["precipitation_mm"],
        "precipitation_probability": profile["precipitation_probability"],
        "frost_risk": profile["frost_risk"],
        "hourly": hourly,
    }
