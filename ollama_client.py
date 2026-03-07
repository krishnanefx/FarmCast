"""
ollama_client.py — FarmCast Ollama LLM integration.

Posts forecast data to a locally running Ollama instance and returns
a plain-English evening advisory for UK farmers.
"""

import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2"

_SYSTEM_PROMPT = (
    "You are an experienced UK agricultural advisor with 30 years working with British farmers. "
    "You give short, direct, no-nonsense advice. "
    "You understand that farmers are busy and outdoors — they need actionable guidance, not weather commentary. "
    "Never use bullet points. Write exactly 3 sentences. "
    "Always end with a single concrete recommendation for this evening."
)


def get_advisory(
    forecast: dict,
    worst_verdict: str = "",
    worst_reason: str = "",
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Request an evening advisory from Ollama.

    Parameters
    ----------
    forecast : dict
        Standardised forecast dict from data_loader.get_forecast().
    worst_verdict : str
        The most critical decision verdict (avoid/caution/good).
    worst_reason : str
        The reason text for the worst verdict, to anchor the advisory.
    model : str
        Ollama model name. Must already be pulled on the host machine.

    Returns
    -------
    str
        Advisory text, or a fallback message if Ollama is unreachable.
    """
    precip_pct = round(forecast["precipitation_probability"] * 100)
    user_prompt = (
        f"It is {forecast['forecast_time']} in {forecast['region']}. "
        f"Temperature is {forecast['temperature_c']}°C, "
        f"wind is {forecast['wind_speed_kmh']} km/h from {forecast['wind_direction_deg']}°, "
        f"precipitation probability is {precip_pct}%, "
        f"frost risk is {forecast['frost_risk']}. "
        f"The worst farming verdict today is {worst_verdict} because: {worst_reason} "
        f"What should the farmer prioritise this evening?"
    )

    payload = {
        "model": model,
        "system": _SYSTEM_PROMPT,
        "prompt": user_prompt,
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return (
            "Advisory unavailable — Ollama is not running. "
            "Start it with `ollama serve` and ensure the model is pulled."
        )
    except requests.exceptions.Timeout:
        return "Advisory unavailable — Ollama took too long to respond. Try again in a moment."
    except requests.exceptions.RequestException as exc:
        return f"Advisory unavailable — unexpected network error: {exc}"
