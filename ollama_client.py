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
    "You are a practical, no-nonsense agricultural weather advisor for UK farmers. "
    "Write in plain English. Be concise — 3 sentences max. "
    "Focus on what the farmer should actually do today."
)


def get_advisory(forecast: dict, model: str = DEFAULT_MODEL) -> str:
    """
    Request an evening advisory from Ollama.

    Parameters
    ----------
    forecast : dict
        Standardised forecast dict from data_loader.get_forecast().
    model : str
        Ollama model name. Must already be pulled on the host machine.

    Returns
    -------
    str
        Advisory text, or a fallback message if Ollama is unreachable.
    """
    user_prompt = (
        f"Here is today's forecast data for {forecast['region']}:\n"
        f"{json.dumps(forecast, indent=2)}\n\n"
        "Based on this data, give me a concise evening advisory — what should the farmer do or watch out for today?"
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
    except Exception as exc:  # noqa: BLE001
        return f"Advisory unavailable — unexpected error: {exc}"
