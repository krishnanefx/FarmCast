# FarmCast 🌾

**AI weather intelligence for UK farmers — built at the Lenovo × NVIDIA Hackathon**

FarmCast is a Streamlit dashboard that runs Pangu-Weather ONNX inference locally and feeds the output into a local Ollama LLM to give UK farmers a clear, plain-English picture of conditions in their region. At a glance, farmers can see whether it is safe to spray pesticides, run machinery, put livestock out, or plant crops — plus an AI-generated evening advisory tailored to the day's conditions.

If the ONNX model or input files are not present, the app falls back to realistic regional mock data automatically — it never crashes.

---

## Features

- **Pangu-Weather inference** — 1-hour ONNX model via `onnxruntime-gpu` (CPU fallback); results cached to `output_data/pangu_output.npz`
- **Decision cards** — four activity verdicts (Good / Caution / Avoid) for Pesticide Spraying, Field Work, Livestock Outdoors, and Planting
- **Frost risk alert** — amber banner shown automatically when `temperature_c < 2°C`
- **AI evening advisory** — concise plain-English guidance from a local Ollama LLM (graceful fallback if Ollama is offline)
- **Hourly conditions chart** — dual-axis Plotly chart: temperature and wind speed lines over a precipitation probability bar background, 06:00–23:00
- **Interactive UK map** — Folium map with green/orange/red region markers; click any marker to switch region

---

## File layout

The ONNX model and input data live **one directory above the repo root**:

```
pangu_weather_1.onnx
input_data/
    input_surface.npy       # (4, 721, 1440) float32 — MSLP, U10, V10, T2M
    input_upper.npy         # (5, 13, 721, 1440) float32 — Z, Q, T, U, V
output_data/                # created automatically; cache stored here
LenovoNvidia/               # repo root
    app.py
    data_loader.py
    decision_logic.py
    ollama_client.py
    requirements.txt
    README.md
```

To use different paths, pass them explicitly to `get_forecast()` or update the defaults in `data_loader.py`.

---

## Setup

### DGX (production)

```bash
git clone https://github.com/HiPeople21/LenovoNvidia
cd LenovoNvidia
pip install -r requirements.txt

# Pull the Ollama model (run once)
ollama pull llama3.2

streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Open `http://<dgx-ip>:8501` in your browser.

`onnxruntime-gpu` is listed in `requirements.txt`. If CUDA is unavailable, install the CPU build instead:
```bash
pip install onnxruntime
```

### Local / development (macOS or Linux without GPU)

```bash
git clone https://github.com/HiPeople21/LenovoNvidia
cd LenovoNvidia
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install onnxruntime          # CPU build — replaces onnxruntime-gpu locally

# Install and start Ollama (macOS)
brew install ollama
brew services start ollama
ollama pull llama3.2:1b          # lighter model for local testing

streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

The app will use mock data if `pangu_weather_1.onnx` / input files are absent.

---

## Ollama model

The active model is set by a single constant at the top of `app.py`:

```python
OLLAMA_MODEL = "llama3.2:1b"   # change to "llama3.2" on the DGX
```

| Environment | Recommended model | Pull command |
|---|---|---|
| DGX (GPU) | `llama3.2` (3B) | `ollama pull llama3.2` |
| Local / CPU | `llama3.2:1b` (1B) | `ollama pull llama3.2:1b` |

---

## Pangu-Weather inference

`data_loader.py` handles the full inference pipeline:

1. Loads `input_surface.npy` and `input_upper.npy` from `input_dir`
2. Runs the 1-hour ONNX model — uses `CUDAExecutionProvider` if available, otherwise `CPUExecutionProvider`
3. Extracts T2M, U10, V10, MSLP at the 0.25° grid point nearest each UK region
4. Derives temperature, wind speed/direction, and a pressure-based precipitation heuristic
5. Saves output to `output_dir/pangu_output.npz` — subsequent calls load from cache (instant)

To force a fresh inference run (e.g. after updating input files):

```python
from data_loader import get_forecast
fc = get_forecast("East Midlands", force_refresh=True)
```

### Output schema

```python
{
    "region":                    str,    # e.g. "East Midlands"
    "forecast_time":             str,    # ISO-8601, current time + 1 hour
    "temperature_c":             float,  # converted from Kelvin
    "wind_speed_kmh":            float,  # sqrt(U10² + V10²) * 3.6
    "wind_direction_deg":        float,  # meteorological convention
    "precipitation_mm":          float,  # heuristic: precip_prob * 3.0
    "precipitation_probability": float,  # 0.0–1.0, derived from MSLP
    "frost_risk":                bool,   # True if temperature_c < 2.0
    "hourly": [                          # 18 entries, 06:00–23:00
        {"hour": str, "temp_c": float, "wind_kmh": float, "precip_prob": float},
    ]
}
```

This schema is the contract between `data_loader.py` and the rest of the app. **Only `data_loader.py` ever needs to change** when swapping or updating the weather model.

---

## Hackathon context

Built at the **Lenovo × NVIDIA Hackathon** on a Lenovo DGX machine.
The app runs on port 8501 and is accessed via browser over the local network.

Goal: demonstrate how edge-hosted AI — Pangu-Weather for forecasting, Ollama for narrative — can deliver actionable, low-latency agricultural intelligence without any cloud dependency.
