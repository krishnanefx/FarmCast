"""
decision_logic.py — Pure decision functions for FarmCast.

No Streamlit imports. No side effects. Input → output only.
"""

# Verdict constants
GOOD = "good"
CAUTION = "caution"
AVOID = "avoid"


def get_pesticide_verdict(wind_speed_kmh: float, precip_probability: float) -> dict:
    """
    Pesticide Spraying suitability.
    Avoid:   wind > 15 km/h  OR  precip_prob > 0.4
    Caution: wind 10-15 km/h OR  precip_prob 0.2-0.4
    Good:    otherwise
    """
    if wind_speed_kmh > 15 or precip_probability > 0.4:
        if wind_speed_kmh > 15 and precip_probability > 0.4:
            reason = f"Wind at {wind_speed_kmh:.0f} km/h and {precip_probability:.0%} rain chance — spray will drift and wash off."
        elif wind_speed_kmh > 15:
            reason = f"Wind at {wind_speed_kmh:.0f} km/h — spray drift risk is too high."
        else:
            reason = f"Rain probability at {precip_probability:.0%} — product will wash off before uptake."
        return {"verdict": AVOID, "reason": reason}

    if wind_speed_kmh >= 10 or precip_probability >= 0.2:
        if wind_speed_kmh >= 10:
            reason = f"Wind at {wind_speed_kmh:.0f} km/h — monitor conditions and spray early in the day."
        else:
            reason = f"Moderate rain chance ({precip_probability:.0%}) — choose a rain-fast formulation."
        return {"verdict": CAUTION, "reason": reason}

    return {
        "verdict": GOOD,
        "reason": f"Light wind ({wind_speed_kmh:.0f} km/h) and low rain risk — ideal spraying window.",
    }


def get_field_work_verdict(precipitation_mm: float) -> dict:
    """
    Field Work / Tilling suitability.
    Avoid:   precip_mm > 2
    Caution: precip_mm 0.5-2
    Good:    otherwise
    """
    if precipitation_mm > 2:
        return {
            "verdict": AVOID,
            "reason": f"{precipitation_mm:.1f} mm expected — ground too wet, risk of compaction and rutting.",
        }
    if precipitation_mm >= 0.5:
        return {
            "verdict": CAUTION,
            "reason": f"{precipitation_mm:.1f} mm expected — check soil moisture before committing machinery.",
        }
    return {
        "verdict": GOOD,
        "reason": f"Only {precipitation_mm:.1f} mm expected — ground conditions should support fieldwork.",
    }


def get_livestock_verdict(frost_risk: bool, wind_speed_kmh: float) -> dict:
    """
    Livestock Outdoors suitability.
    Avoid:   frost_risk AND wind > 20 km/h
    Caution: frost_risk OR wind > 25 km/h
    Good:    otherwise
    """
    if frost_risk and wind_speed_kmh > 20:
        return {
            "verdict": AVOID,
            "reason": f"Frost plus {wind_speed_kmh:.0f} km/h wind — dangerous wind-chill for livestock.",
        }
    if frost_risk or wind_speed_kmh > 25:
        if frost_risk:
            reason = "Overnight frost risk — bring vulnerable animals in before dusk."
        else:
            reason = f"Wind at {wind_speed_kmh:.0f} km/h — ensure adequate shelter is available."
        return {"verdict": CAUTION, "reason": reason}

    return {
        "verdict": GOOD,
        "reason": "No frost risk and manageable wind — outdoor grazing is fine today.",
    }


def get_planting_verdict(frost_risk: bool, temperature_c: float) -> dict:
    """
    Planting suitability.
    Avoid:   frost_risk OR temp_c < 3
    Caution: temp_c 3-7
    Good:    otherwise
    """
    if frost_risk or temperature_c < 3:
        if frost_risk:
            reason = "Frost risk present — soil temperature will be too cold for germination."
        else:
            reason = f"Temperature {temperature_c:.1f}°C — too cold for safe planting."
        return {"verdict": AVOID, "reason": reason}

    if temperature_c <= 7:
        return {
            "verdict": CAUTION,
            "reason": f"Temperature {temperature_c:.1f}°C — borderline for planting; cold-hardy crops only.",
        }

    return {
        "verdict": GOOD,
        "reason": f"Temperature {temperature_c:.1f}°C with no frost — suitable for planting.",
    }


def get_all_decisions(forecast: dict) -> list:
    """
    Evaluate all four decision cards for a forecast dict.

    Returns a list of dicts, each with keys:
        icon, activity, verdict, reason
    """
    wind = forecast["wind_speed_kmh"]
    precip_prob = forecast["precipitation_probability"]
    precip_mm = forecast["precipitation_mm"]
    frost = forecast["frost_risk"]
    temp = forecast["temperature_c"]

    pesticide = get_pesticide_verdict(wind, precip_prob)
    field = get_field_work_verdict(precip_mm)
    livestock = get_livestock_verdict(frost, wind)
    planting = get_planting_verdict(frost, temp)

    return [
        {"icon": "🌿", "activity": "Pesticide Spraying", **pesticide},
        {"icon": "🚜", "activity": "Field Work / Tilling", **field},
        {"icon": "🐄", "activity": "Livestock Outdoors", **livestock},
        {"icon": "🌱", "activity": "Planting", **planting},
    ]


def get_worst_verdict(decisions: list) -> str:
    """
    Return the most severe verdict across all decision cards.
    Priority: avoid > caution > good
    """
    for verdict in (AVOID, CAUTION, GOOD):
        if any(d["verdict"] == verdict for d in decisions):
            return verdict
    return GOOD
