import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from geopy.geocoders import Nominatim
from pydantic import BaseModel

from backend.era5_fetcher import ensure_era5_inputs

app = FastAPI()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(REPO_ROOT, "frontend")
HOME_DIR = os.path.expanduser("~")

INPUT_SEARCH_DIRS = [
    os.path.join(REPO_ROOT, "input_data"),
    os.path.join(REPO_ROOT, "..", "input_data"),
    os.path.join(HOME_DIR, "Downloads", "input_data"),
]
MODEL_SEARCH_DIRS = [
    REPO_ROOT,
    os.path.join(HOME_DIR, "Downloads"),
]
VALID_MODELS = {"1", "3", "6", "24"}

os.makedirs(FRONTEND_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

geolocator = Nominatim(user_agent="pangu-weather-app")


class WeatherRequest(BaseModel):
    location: str
    model: str


@dataclass
class CachedSession:
    model_path: str
    session: object


_SESSION_CACHE: Dict[str, CachedSession] = {}
_INPUT_CACHE: Dict[str, object] = {
    "state": None,
    "surface": None,
    "upper": None,
    "surface_path": None,
    "upper_path": None,
    "meta_path": None,
}
_SURFACE_CACHE: Dict[Tuple[str, tuple], np.ndarray] = {}


@app.get("/")
async def read_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse("Frontend not built yet.", status_code=404)
    return FileResponse(index_path)


def get_grid_idx(lat: float, lon: float) -> Tuple[int, int]:
    lat = max(-90.0, min(90.0, lat))
    lon_360 = lon % 360.0
    lat_idx = int(round((90.0 - lat) / 0.25))
    lon_idx = int(round(lon_360 / 0.25))
    lat_idx = max(0, min(lat_idx, 720))
    lon_idx = max(0, min(lon_idx, 1439))
    return lat_idx, lon_idx


def _resolve_model_path(model_hours: str) -> str:
    if model_hours not in VALID_MODELS:
        raise ValueError(f"Invalid model hours: {model_hours}")

    candidates = [os.path.join(base, f"pangu_weather_{model_hours}.onnx") for base in MODEL_SEARCH_DIRS]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Model pangu_weather_{model_hours}.onnx not found. Checked: {candidates}"
    )


def _find_input_files() -> Optional[Tuple[str, str, Optional[str]]]:
    for base_dir in INPUT_SEARCH_DIRS:
        surface_path = os.path.join(base_dir, "input_surface.npy")
        upper_path = os.path.join(base_dir, "input_upper.npy")
        if os.path.exists(surface_path) and os.path.exists(upper_path):
            meta_json = os.path.join(base_dir, "era5_meta.json")
            meta_txt = os.path.join(base_dir, "era5_meta.txt")
            meta_path = meta_json if os.path.exists(meta_json) else (meta_txt if os.path.exists(meta_txt) else None)
            return surface_path, upper_path, meta_path
    return None


def _ensure_input_files(force_refresh: bool = False) -> Tuple[str, str, Optional[str]]:
    found = _find_input_files()
    target_dir = os.path.join(REPO_ROOT, "input_data")
    try:
        ensure_era5_inputs(target_dir, force_refresh=force_refresh)
    except Exception:
        # If local refresh fails but usable inputs already exist elsewhere, keep serving.
        if not found:
            raise

    found = _find_input_files()
    if not found:
        checked = [os.path.join(d, "input_surface.npy") for d in INPUT_SEARCH_DIRS]
        raise FileNotFoundError(
            "Input fields are missing after ERA5 fetch attempt. "
            f"Checked: {checked}"
        )
    return found


def _input_state(surface_path: str, upper_path: str) -> tuple:
    return (
        surface_path,
        os.path.getmtime(surface_path),
        os.path.getsize(surface_path),
        upper_path,
        os.path.getmtime(upper_path),
        os.path.getsize(upper_path),
    )


def _load_inputs(force_refresh: bool = False) -> Tuple[np.ndarray, np.ndarray, tuple, Optional[str]]:
    surface_path, upper_path, meta_path = _ensure_input_files(force_refresh=force_refresh)
    state = _input_state(surface_path, upper_path)

    if _INPUT_CACHE["state"] == state and _INPUT_CACHE["surface"] is not None and _INPUT_CACHE["upper"] is not None:
        return _INPUT_CACHE["surface"], _INPUT_CACHE["upper"], state, _INPUT_CACHE["meta_path"]

    surface = np.load(surface_path).astype(np.float32)
    upper = np.load(upper_path).astype(np.float32)

    if surface.shape != (4, 721, 1440):
        raise ValueError(f"Unexpected input_surface shape {surface.shape}; expected (4, 721, 1440)")
    if upper.shape != (5, 13, 721, 1440):
        raise ValueError(f"Unexpected input_upper shape {upper.shape}; expected (5, 13, 721, 1440)")

    _INPUT_CACHE["state"] = state
    _INPUT_CACHE["surface"] = surface
    _INPUT_CACHE["upper"] = upper
    _INPUT_CACHE["surface_path"] = surface_path
    _INPUT_CACHE["upper_path"] = upper_path
    _INPUT_CACHE["meta_path"] = meta_path
    return surface, upper, state, meta_path


def _load_model_session(model_hours: str):
    model_path = _resolve_model_path(model_hours)
    cached = _SESSION_CACHE.get(model_hours)
    if cached and cached.model_path == model_path:
        return cached.session, model_path

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for inference") from exc

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(model_path, providers=providers)
    _SESSION_CACHE[model_hours] = CachedSession(model_path=model_path, session=session)
    return session, model_path


def _run_inference(model_hours: str) -> Tuple[np.ndarray, str, Optional[str], tuple]:
    surface, upper, state, meta_path = _load_inputs(force_refresh=False)
    cache_key = (model_hours, state)
    if cache_key in _SURFACE_CACHE:
        return _SURFACE_CACHE[cache_key], _resolve_model_path(model_hours), meta_path, state

    session, model_path = _load_model_session(model_hours)

    # Model variants differ: some use [C,H,W], others [B,C,H,W].
    try:
        outputs = session.run(
            ["output", "output_surface"],
            {"input": upper, "input_surface": surface},
        )
    except Exception:
        outputs = session.run(
            ["output", "output_surface"],
            {"input": upper[np.newaxis, ...], "input_surface": surface[np.newaxis, ...]},
        )

    output_surface = outputs[1]
    while output_surface.ndim > 3:
        output_surface = output_surface[0]

    if output_surface.shape != (4, 721, 1440):
        raise ValueError(
            f"Unexpected output_surface shape {output_surface.shape}; expected (4, 721, 1440)"
        )

    _SURFACE_CACHE[cache_key] = output_surface
    return output_surface, model_path, meta_path, state


def _load_source_meta(meta_path: Optional[str]) -> dict:
    if not meta_path or not os.path.exists(meta_path):
        return {"present": False, "warning": "ERA5 metadata not found"}

    try:
        if meta_path.endswith(".json"):
            with open(meta_path) as f:
                payload = json.load(f)
            out = {
                "present": True,
                "path": meta_path,
                "date": payload.get("date"),
                "time": payload.get("time"),
                "fetched_at": payload.get("fetched_at"),
                "source": payload.get("source"),
            }
        else:
            payload = {}
            with open(meta_path) as f:
                for line in f:
                    key, _, value = line.strip().partition("=")
                    if key:
                        payload[key] = value
            out = {
                "present": True,
                "path": meta_path,
                "analysis_time": payload.get("analysis_time"),
                "download_time": payload.get("download_time"),
                "source": "ERA5 reanalysis via CDS API",
            }
    except Exception as exc:
        return {"present": False, "warning": f"Failed to parse metadata: {exc}"}

    fetched_at = out.get("fetched_at") or out.get("download_time")
    if isinstance(fetched_at, str):
        try:
            dt = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
            age_hours = (time.time() - dt.timestamp()) / 3600.0
            out["age_hours"] = round(age_hours, 2)
        except Exception:
            pass
    return out


@app.post("/api/weather")
async def get_weather(req: WeatherRequest):
    if req.model not in VALID_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Expected one of {', '.join(sorted(VALID_MODELS))}.",
        )
    if not req.location.strip():
        raise HTTPException(status_code=400, detail="Location cannot be empty.")

    try:
        location = geolocator.geocode(req.location, timeout=10)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Geocoder error: {exc}") from exc
    if not location:
        raise HTTPException(status_code=404, detail="Location not found.")

    lat, lon = location.latitude, location.longitude
    lat_idx, lon_idx = get_grid_idx(lat, lon)

    try:
        output_surface, model_path, meta_path, _ = _run_inference(req.model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    mslp_pa = float(output_surface[0, lat_idx, lon_idx])
    u10 = float(output_surface[1, lat_idx, lon_idx])
    v10 = float(output_surface[2, lat_idx, lon_idx])
    t2m_k = float(output_surface[3, lat_idx, lon_idx])

    temp_c = t2m_k - 273.15
    wind_speed_ms = math.sqrt(u10 ** 2 + v10 ** 2)
    wind_speed_kmh = wind_speed_ms * 3.6

    source_meta = _load_source_meta(meta_path)

    return {
        "location": location.address,
        "coordinates": {"lat": round(lat, 4), "lon": round(lon, 4)},
        "grid_idx": {"lat_idx": lat_idx, "lon_idx": lon_idx},
        "forecast_hours": int(req.model),
        "using_real_data": True,
        "metrics": {
            # Retained for frontend compatibility:
            # temperature_2m in K, pressure in Pa, wind components in m/s.
            "temperature_2m": t2m_k,
            "u_wind_10m": u10,
            "v_wind_10m": v10,
            "mean_sea_level_pressure": mslp_pa,
        },
        "surface": {
            "temperature_c": round(temp_c, 2),
            "wind_speed_ms": round(wind_speed_ms, 2),
            "wind_speed_kmh": round(wind_speed_kmh, 2),
            "pressure_hpa": round(mslp_pa / 100.0, 2),
        },
        "model": {
            "hours": req.model,
            "file": model_path,
            "backend": "fastapi",
        },
        "source": source_meta,
        "forecast_time": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/api/data-status")
async def data_status():
    found = _find_input_files()
    if not found:
        return {
            "has_data": False,
            "source": None,
            "cache_age_hours": None,
        }

    surface_path, _, meta_path = found
    cache_age_hours = round((time.time() - os.path.getmtime(surface_path)) / 3600.0, 2)
    return {
        "has_data": True,
        "source": _load_source_meta(meta_path),
        "cache_age_hours": cache_age_hours,
    }
